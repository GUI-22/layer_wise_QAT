# -*- coding: utf-8 -*-
"""LLM activation quantization calibration module."""

import gc
import logging
import typing as tp
import logging

import torch
import torch.nn as nn
import torch.utils.hooks
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.optim import Adam
import torch.nn.functional as F
import torch.cuda.amp as amp

from datasets import load_dataset
import random
import functools
import json
import psutil
import os

from transformers.cache_utils import DynamicCache

from lmquant.dataset import IOActivationsCache
from lmquant.quant.calib.config import QuantTensorType
from lmquant.quant.quantizer.activation import ActivationQuantizer
from lmquant.quant.quantizer.weight import WeightQuantizer
from lmquant.utils import tools


from .dataset import LlmCalibConfig, LlmCalibrationCache
from .eval import LlmEvalConfig
from .nn import LlmDecoderLayerStruct, LlmModelStruct
from .utils import get_needs_inputs_fn
from .quant.config import LlmModuleKey, LlmQuantConfig
from .quant.weight import quantize_llm_decoder_layer_weights_with_grad
from .quant.activation import quantize_llm_decoder_layer_activations
from . import tasks, models

__all__ = ["quantize_llm_activations"]


from fairseq_cli.train import cli_main



def qat_llm(
    model: LlmModelStruct,
    quant_config: LlmQuantConfig,
    tokenizer: nn.Module | None = None,
    calib_config: LlmCalibConfig | None = None,
    eval_config: LlmEvalConfig | None = None,
    seq_length: int = 1024,
    num_train_rows: int = 128,
    num_train_tokens: int = 128 * 1024,
    batch_size: int = 64,
    cache_save_path: str | None = "/data/gyy/lmquant-main/lmquant/llm/kd_data/",
    lr: float = 1e-3,
    num_epochs: int = 5,
    weight_dacay: float = 1e-2,
    kl_temperature: float = 2.0,
    alpha: float = 0.5,         
    loss_scale: float = 1e-2,
    eps: float = 1e-6,
    fairseq_args: str = "./fairseq_args.json",
    gen_teacher_opts: bool = False,
    orig_model_path: bool = "/data/gyy/TinyLlama",
    orig_model: LlmModelStruct | None = None
) -> tuple[
    dict[str, dict[str, torch.Tensor | float | None]],
    dict[str, WeightQuantizer],
    dict[str, torch.Tensor | float | None],
]:
    logger = logging.getLogger(__name__)



    # region : forward and backward and update, wrapped in fairseq
    kwargs = {
        "quant_config": quant_config, 
        "calib_config": calib_config, 
        "eval_config": eval_config,
        "gen_teacher_opts": gen_teacher_opts,
        "orig_model_path": orig_model_path,
        "orig_model": orig_model
    }
    cli_main(
        args_path=fairseq_args,
        model=model,
        tokenizer=tokenizer,
        **kwargs
    )
    # endregion


    if isinstance(model, LlmModelStruct):
        model_struct = model
        model = model_struct.module
    else:
        model_struct = LlmModelStruct.build(model)
    backbone_struct = model_struct.backbone_struct
    layer_structs = backbone_struct.layer_structs

    args_cache: dict[str, tuple[torch.Tensor]] = {}
    kwargs_cache: dict[str, dict[str, tp.Any]] = {}
    outputs_cache: dict[str, torch.Tensor] = {}

    logger.info("*collecting labels")
    # args_cache, kwargs_cache, outputs_cache = \
    # _iter_layer_for_labels(  # noqa: C901
    #     model=model,
    #     layer_structs=layer_structs,
    #     tokenizer=tokenizer,
    #     seq_length=seq_length,
    #     num_train_rows=num_train_rows,
    #     num_train_tokens=num_train_tokens,
    #     batch_size=64,
    #     cache_save_path=cache_save_path
    # )
    # torch.save(kwargs_cache, cache_save_path + "kwargs_cache.pt")
    kwargs_cache = torch.load("/data/gyy/lmquant-main/lmquant/llm/kd_data/kwargs_cache.pt")

    # layer_args = args_cache[layer_structs[0].full_name]
    layer_args = load_tensors('args', layer_structs[0].full_name, cache_save_path)
    num_args = len(layer_args)

    for layer_idx, layer_struct in enumerate(layer_structs):

        logger.info(f"in layer {layer_struct.full_name}")

        # set optimizer and loss function
        optimizer = torch.optim.Adam(layer_struct.module.parameters(), lr=lr, eps=eps)
        mse_loss_function = torch.nn.MSELoss(reduction="sum")
        kldiv_loss_function = torch.nn.KLDivLoss(reduction="sum")

        # get layer input and label
        layer_labels = load_tensors('outputs', layer_struct.full_name, cache_save_path)
        layer = layer_struct.module
        layer_kwargs = kwargs_cache[layer_struct.full_name]

        device = next(layer.parameters()).device
        total_num = layer_args[0].shape[0]

        # region training this layer
        for epoch in range(0, num_epochs):
            logger.info(f"in layer {layer_struct.full_name}, epoch {epoch+1}, total_epoch {num_epochs}")
            for i in tqdm(
                range(0, total_num, batch_size), desc=f"in epoch {epoch}", leave=False
            ):
                
                # region quant weights in this layer
                orig_module_weights: dict[str, nn.Parameter] = {}
                dequantized_weights: dict[str, torch.Tensor]
                logger.info(f"quantizing weights for layer {layer_struct.full_name}")
                weight_layer_cache = \
                LlmCalibrationCache(calib_config).get_layer_activations(
                    layer_struct=layer_struct,
                    layer_args=layer_args,
                    layer_kwargs=kwargs_cache[layer_struct.full_name],
                    needs_inputs_fn=get_needs_inputs_fn(config=quant_config),
                    batch_size=batch_size
                )

                orig_module_weights, dequantized_weights = quantize_llm_decoder_layer_weights_with_grad(
                    layer=layer_struct,
                    config=quant_config,
                    quant_cache={},
                    layer_cache=weight_layer_cache,
                    layer_kwargs=kwargs_cache[layer_struct.full_name],
                )

                del weight_layer_cache
                gc.collect()
                torch.cuda.empty_cache()
                # endregion

                # region quant activations in this layer 
                activation_hooks: dict[str, list[torch.utils.hooks.RemovableHandle]] = {}

                logger.info(f"quantizing activations for layer {layer_struct.full_name}")
                activation_layer_caches = \
                LlmCalibrationCache(calib_config).get_layer_activations(
                    layer_struct=layer_struct,
                    layer_args=layer_args,
                    layer_kwargs=kwargs_cache[layer_struct.full_name],
                    needs_inputs_fn=get_needs_inputs_fn(config=quant_config),
                    needs_outputs_fn=quant_config.needs_quant_outputs,
                    batch_size=batch_size
                )

                activation_quantizers, activation_hooks = quantize_llm_decoder_layer_activations(
                    layer=layer_struct,
                    config=quant_config,
                    quant_cache={},
                    layer_cache=activation_layer_caches,
                    layer_kwargs=kwargs_cache[layer_struct.full_name],
                    orig_state_dict=None
                )

                gc.collect()
                torch.cuda.empty_cache()
                # endregion




                









                batch_inputs = tuple(arg[i : min(total_num, i + batch_size)] for arg in layer_args)
                batch_inputs = [arg.to(device=device) for arg in batch_inputs]

                # forward
                logger.info("forward this layer")
                batch_outputs = layer(*batch_inputs, **layer_kwargs)
                
                for _, hook_list in activation_hooks.items():
                    for hook in hook_list:
                        hook.remove()
                

                if not isinstance(batch_outputs, (list, tuple)):
                    batch_outputs = (batch_outputs,)
                assert num_args <= len(batch_outputs)
                batch_outputs = batch_outputs[:num_args]
                batch_outputs = [output.to(dtype=torch.float32) for output in batch_outputs]

                batch_labels = tuple(layer_label[i : min(total_num, i + batch_size)] for layer_label in layer_labels)
                batch_labels = [label.to(device=device).to(dtype=torch.float32) for label in batch_labels]

                # compute loss and backward
                logger.info(f"computing loss and backwarding")
                mse_loss = sum(mse_loss_function(output, label) for output, label in zip(batch_outputs, batch_labels))
                kl_loss = 0
                for output, label in zip(batch_outputs, batch_labels):
                    soft_output = F.log_softmax(output / kl_temperature, dim=-1)
                    soft_label = F.softmax(label / kl_temperature, dim=-1)
                    kl_loss += kldiv_loss_function(soft_output, soft_label)
                loss = alpha * mse_loss + (1 - alpha) * kl_loss
                loss *= loss_scale

                del batch_labels, batch_outputs, batch_inputs
                gc.collect()
                torch.cuda.empty_cache()

                optimizer.zero_grad()  # Clear gradients


                loss.backward()       # Backpropagate the loss

                # region replace back original weights
                module_name_list = [
                    layer_struct.proj_q_full_name, 
                    layer_struct.proj_k_full_name, 
                    layer_struct.proj_v_full_name, 
                    layer_struct.proj_o_full_name
                ]
                module_list = [
                    layer_struct.proj_q, 
                    layer_struct.proj_k, 
                    layer_struct.proj_v, 
                    layer_struct.proj_o
                ]
                if layer_struct.router is not None:
                    module_name_list.append(layer_struct.router_full_name)
                    module_list.append(layer_struct.router)
                num_experts = layer_struct.config.num_experts
                for expert_idx in range(num_experts):
                    for module_name, module in zip(
                        layer_struct.proj_1st_full_names[expert_idx::num_experts], layer_struct.proj_1st[expert_idx::num_experts]
                    ):
                        module_name_list.append(module_name)
                        module_list.append(module)
                    module_name_list.append(layer_struct.proj_2nd_full_names[expert_idx])
                    module_list.append(layer_struct.proj_2nd[expert_idx])

                for module_name, module in zip(module_name_list, module_list):
                    orig_weight = orig_module_weights[module_name]
                    orig_weight.grad = torch.autograd.grad(outputs=dequantized_weights[module_name], inputs=orig_weight, grad_outputs=module.weight.grad)[0]
                    module.weight = orig_weight
                # endregion
                del dequantized_weights
                # del module_name_list, module_list
                # endregion
                
                logger.info(f"in epoch {epoch+1}, in rows[{i}: {i+batch_size}], loss {loss.item()}")
                # torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=10.0)

                for module_name, module in zip(module_name_list, module_list):
                    logger.debug(f"In layer {layer_struct.full_name}, gradient of {module_name} has nan: {torch.isnan(module.weight.grad).any()}")
                    logger.debug(f"In layer {layer_struct.full_name}, gradient of {module_name} has inf: {torch.isinf(module.weight.grad).any()}")

                    grad = module.weight.grad
                    inf_mask = torch.isinf(grad)
                    inf_indices = torch.nonzero(inf_mask, as_tuple=False)
                    if len(inf_indices) > 0:
                        first_inf_position = inf_indices[0]  
                        # first_inf_value = grad[first_inf_position]  # 获取该位置的值
                        logger.debug(f"inf: first position {first_inf_position}")

                    grad = module.weight.grad
                    nan_mask = torch.isnan(grad)
                    nan_indices = torch.nonzero(nan_mask, as_tuple=False)
                    if len(nan_indices) > 0:
                        first_nan_position = nan_indices[0]  
                        # first_nan_value = grad[first_nan_position]  # 获取该位置的值
                        logger.debug(f"nan: first position {first_nan_position}")

                optimizer.step()
        # endregion


        # region run this layer(after quantizing wgts and acts) and get layer outputs
        outputs: list[list[torch.Tensor]] = [[] for _ in num_args]
        for i in tqdm(
            range(0, total_num, batch_size), desc=f"forwarding {layer_struct.full_name} after quantizing weights and activations", leave=False
        ):
            batch_inputs = tuple(arg[i : min(total_num, i + batch_size)] for arg in layer_args)
            batch_inputs = [arg.to(device=device) for arg in batch_inputs]

            batch_outputs = layer(*batch_inputs, **layer_kwargs)

            if not isinstance(batch_outputs, (list, tuple)):
                batch_outputs = (batch_outputs,)
            assert num_args <= len(batch_outputs)
            batch_outputs = batch_outputs[:num_args]

            batch_labels = tuple(layer_label[i : min(total_num, i + batch_size)] for layer_label in layer_labels)

            for i, output in enumerate(batch_outputs):
                outputs[i].append(output.detach().cpu())

        outputs = tuple(torch.cat(output_list, dim=0) for output_list in outputs)
        layer_args = outputs
        gc.collect()
        torch.cuda.empty_cache()
        # endregion

    


def _iter_layer_for_labels(  # noqa: C901
    model: nn.Module,
    layer_structs: list[LlmDecoderLayerStruct],
    tokenizer: nn.Module | None = None,
    seq_length: int = 1024,
    num_train_rows: int = 1024,
    num_train_tokens: int = 1024 * 1024,
    batch_size: int = 64,
    cache_save_path: str | None = "/data/gyy/lmquant-main/lmquant/llm/kd_data/"
) -> tuple[
    dict[str, tuple[torch.Tensor]],
    dict[str, dict[str, tp.Any]],
    dict[str, torch.Tensor]
]:
    labels_hooks: list[torch.utils.hooks.RemovableHandle] = []

    args_cache: dict[str, tuple[torch.Tensor]] = {}
    kwargs_cache: dict[str, dict[str, tp.Any]] = {}
    outputs_cache: dict[str, torch.Tensor] = {}
    
    for layer_struct in layer_structs:
        layer = layer_struct.module
        if layer_struct.idx == 0:
            labels_hooks.append(
                layer.register_forward_hook(
                    functools.partial(
                        _args_kwargs_outputs_hook,
                        layer_name=layer_struct.full_name,
                        args_cache=args_cache,
                        kwargs_cache=kwargs_cache,
                        outputs_cache=outputs_cache,
                        cache_save_path=cache_save_path,
                        save_args=True
                    ),
                    with_kwargs=True
                )
            )
        else:
            labels_hooks.append(
                layer.register_forward_hook(
                    functools.partial(
                        _args_kwargs_outputs_hook,
                        layer_name=layer_struct.full_name,
                        args_cache=args_cache,
                        kwargs_cache=kwargs_cache,
                        outputs_cache=outputs_cache,
                        cache_save_path=cache_save_path,
                        save_args=False
                    ),
                    with_kwargs=True
                )
            )
            

    with logging_redirect_tqdm():
        # region we first collect cache information by running the model with all samples
        with torch.inference_mode():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for train_batch in tqdm(
                iter_train_rows(
                    tokenizer=tokenizer,
                    seq_length=seq_length,
                    num_train_rows=num_train_rows,
                    num_train_tokens=num_train_tokens,
                    batch_size=batch_size
                ),
                desc="collecting labels generated by original model",
                leave=False,
                total=num_train_rows // batch_size,
            ):
                model(train_batch.to(device=device))
                if psutil.virtual_memory().percent > 90:
                    raise RuntimeError("memory usage > 90%%, aborting")
                
        for hook in labels_hooks:
            hook.remove()
        del labels_hooks
        # endregion

        gc.collect()
        torch.cuda.empty_cache()

        return args_cache, kwargs_cache, outputs_cache
        



def _hook_get_args_kwargs_outputs(
    m: nn.Module,
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, tp.Any],
    outputs: torch.Tensor,
    layer_name: str,
    args_cache: dict[str, list[tuple[torch.Tensor]]],
    kwargs_cache: dict[str, dict[str, tp.Any]],
    outputs_cache: dict[str, list[tuple[torch.Tensor]]],
    cache_save_path: str | None = "./kd_data/",
    save_args: bool = True
) -> None:
    
    # region cache args
    if save_args is True:
        assert all(isinstance(x, torch.Tensor) for x in args)
        # if layer_name not in args_cache:
        #     args_cache[layer_name] = [tuple(x.detach().cpu() for x in args)]
        # else:
        #     args_cache[layer_name].append(tuple(x.detach().cpu() for x in args))

        # with open(cache_save_path + f'{layer_name}.args.jsonl', 'a') as file:
        #     file.write(json.dumps(list(x.detach().cpu().tolist() for x in args)) + '\n')

        save_tensors(args, "args", layer_name, cache_save_path)
    # endregion

    # region cache kwargs
    if layer_name not in kwargs_cache:
        kwargs_cache[layer_name] = {}
    layer_kwargs_cache = kwargs_cache[layer_name]
    if layer_kwargs_cache:
        assert len(layer_kwargs_cache) == len(kwargs), "kwargs_cache should have the same length as kwargs"
        for k, v in kwargs.items():
            assert k in layer_kwargs_cache, f"kwargs_cache should have the same keys as kwargs, but missing {k}"
            cached = layer_kwargs_cache[k]
            if isinstance(v, DynamicCache):
                assert cached is None, f"kwargs_cache[{k}] should be None"
            elif isinstance(v, torch.Tensor):
                assert v.allclose(cached), f"kwargs_cache[{k}] should be the same as kwargs[{k}]"
            else:
                assert v == cached, f"kwargs_cache[{k}] should be the same as kwargs[{k}]"
    else:
        for k, v in kwargs.items():
            if isinstance(v, DynamicCache):
                layer_kwargs_cache[k] = None
            else:
                layer_kwargs_cache[k] = v
    # endregion

    # region cache output
    if isinstance(outputs, list):
        outputs = tuple(outputs)
    elif isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    assert len(outputs) >= len(args)
    outputs = outputs[:len(args)]

    # if layer_name not in outputs_cache:
    #     outputs_cache[layer_name] = [tuple(output.detach().cpu() for output in outputs)]
    # else:
    #     outputs_cache[layer_name].append(tuple(output.detach().cpu() for output in outputs))

    # with open(cache_save_path + f'{layer_name}.outputs.jsonl', 'a') as file:
    #     file.write(json.dumps(list(output.detach().cpu().tolist() for output in outputs)) + '\n')
    save_tensors(outputs, "outputs", layer_name, cache_save_path)
    # endregion



def save_tensors(
        tensors: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor], 
        tensors_name: str, 
        layer_name: str, 
        cache_save_path: str
) -> None:
    file_path = cache_save_path + f'{layer_name}.{tensors_name}.pt'
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    if os.path.exists(file_path):
        existing_tensors = torch.load(file_path)
        assert len(tensors) == len(existing_tensors)
        for existing_tensor, tensor in zip(existing_tensors, tensors):
            existing_tensor = torch.cat((existing_tensor, tensor), dim=0)
    else:
        existing_tensors = tensors
    torch.save(existing_tensors, file_path)


def load_tensors(
        tensors_name: str, 
        layer_name: str, 
        cache_save_path: str
) -> tuple[torch.Tensor, ...] | list[torch.Tensor]:
    
    file_path = cache_save_path + f'{layer_name}.{tensors_name}.pt'
    assert os.path.exists(file_path), f"when loading {file_path}, the file path doesn't exist"
    return torch.load(file_path)




def iter_train_rows(
    tokenizer: nn.Module | None = None,
    seq_length: int = 1024,
    num_train_rows: int = 1024,
    num_train_tokens: int = 1024 * 1024,
    batch_size: int = 64
) -> tp.Generator[torch.Tensor, None, None]:
    '''
    return:
        train_rows[i : i + batch_size, :] : a 2-dim tensor, shape[0]==batch_size, shape[1]=seq_length, every element is an index (not a fp32)
    '''
    # region raw training data
    dataset = load_dataset('mit-han-lab/pile-val-backup', split="validation")
    dataset = dataset.shuffle(seed=42)
    rng = random.Random(42)
    train_rows, num_tokens = [], 0
    for _data in dataset:
        line = _data["text"]
        line = line.strip()
        # line_encoded is a list of token ids
        line_encoded = tokenizer.encode(line)
        line_seq_length = len(line_encoded)
        if line_seq_length == 0:
            continue
        # sample is a tensor of shape (1, seq_length)
        train_row = torch.tensor([line_encoded])
        if line_seq_length > seq_length:
            tok = rng.randint(0, line_seq_length - seq_length)
            train_row = train_row[:, tok : tok + 1024]
        train_rows.append(train_row)
        num_tokens += train_row.shape[1]
        if len(train_rows) >= num_train_rows and num_tokens >= num_train_tokens:
            break
    # now concatenate all train_rows and split according to seq_length
    train_rows = torch.cat(train_rows, dim=1).split(seq_length, dim=1)
    if num_tokens > num_train_tokens:
        train_rows = train_rows[:-1]
    train_rows = train_rows[: num_train_rows]
    train_rows = torch.stack(train_rows).squeeze(1)
    # endregion
    for i in range(0, num_train_rows, batch_size):
        if i + batch_size > train_rows.shape[0]:
            break
        yield train_rows[i : i + batch_size, :]