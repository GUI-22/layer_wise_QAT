import os
from typing import Optional
import json
from argparse import Namespace
import torch
import typing as tp
import functools
import gc
import logging
import copy


from fairseq.tasks import register_task, FairseqDataclass, FairseqTask
from dataclasses import dataclass, field
from omegaconf import II
from torch import nn

from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaModel

from .data.lm_loader import LMLoader, LMLoader_TextInput, LMLoader_Args_and_Outputs
from .data.tiktoken_tokenizer import TiktokenTokenizer
from .data.llama_tokenizer import LLaMATokenizer

from lmquant.llm.dataset import LlmCalibConfig, LlmCalibrationCache
from lmquant.llm.nn import LlmDecoderLayerStruct, LlmModelStruct
from lmquant.llm.utils import get_needs_inputs_fn
from lmquant.llm.quant.config import LlmModuleKey, LlmQuantConfig
from lmquant.llm.eval import LlmEvalConfig
from lmquant.llm.quant.weight import quantize_llm_decoder_layer_weights_with_grad, quantize_llm_decoder_layer_weights
from lmquant.llm.quant.activation import quantize_llm_decoder_layer_activations
from lmquant.llm.quant import quantize_llm_activations, quantize_llm_weights
from lmquant.llm.models.llama_model_full import LlamaDecoderLayersInFairseq, LlamaModelFull




@dataclass
class KDLanguageModelingConfig(FairseqDataclass):
    path_to_labels: Optional[str] = field(
        default=None, metadata={"help": "path to outputs of teacher layers"}
    )
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    batch_size_in_quant: int = field(
        default=8,
        metadata={"help": "the batch size when quantizing weights and activations"},
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    llama_model: Optional[str] = field(
        default=None,
        metadata={"help": "path to load tokenizer and config"},
    )
    quant_acts_when_training: bool = field(
        default=False,
        metadata={"help": "if you quant activations when training, set it True"}
    )
    tiktoken_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "tiktoken model to tokenize the data"
        },
    )
    batch_read_ahead: int = field(
        default=10000,
        metadata={"help": "batch read ahead size for infinibatch"},
    )
    pad_to_max_len: bool = field(
        default=False,
        metadata={"help": "pad each sentence to max length"},
    )
    absolute_path: bool = field(
        default=False,
        metadata={"help": "use absolute path in data config"},
    )
    tokenizer_pad_to_multiple: int = field(
        default=8,
        metadata={"help": "pad to multiple of this value"},
    )
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")


@register_task('kd', dataclass=KDLanguageModelingConfig)
class KDTask(FairseqTask):
    def __init__(self, args):
        super().__init__(args)
        self.cfg = args
        self.tokenizer: nn.Module | None = None

        # self.model_in_llama_class is the trained model (trained part: quanted; untrained part : fp16), LlmModelStruct
        self.model_in_llama_class: nn.Module | None = None

        # layer (class: LlamaModelDecoderLayer)
        self.model: nn.Module | None = None

        # params are trained (copy from self.model_in_llama_class), then use lmquant to quant it
        self.model_quanted_in_llama_class: nn.Module | None = None

        # share params with self.model_quanted_in_llama_class, but self.model_quanted in class (LlamaModelFull)
        self.model_quanted: nn.Module | None = None

        # self.model_orig is the teacher
        self.model_orig = None 

        self.layer_struct: LlmDecoderLayerStruct | None = None
        self.current_layer_idx: int | None = None
        self.total_layer_kwargs: dict | None = None
        self.quant_config: LlmQuantConfig | None = None
        self.calib_config: LlmCalibConfig | None = None
        self.eval_config: LlmEvalConfig | None = None
        self.orig_model_path: str | None = None
        self.activation_quant_hooks_in_valid: dict[str, list[torch.utils.hooks.RemovableHandle]] = {} # del them after valid
        self.layers_original_weights_in_valid: dict[int, dict[str, nn.Parameter]] = {}


        self.lmquant_ppl_result_wikitext_in_train_no_quant: float | None = None
        self.lmquant_ppl_result_val_in_train_no_quant: float | None = None
        self.lmquant_ppl_result_wikitext_in_train_with_quant: float | None = None
        self.lmquant_ppl_result_val_in_train_with_quant: float | None = None

        self.activation_quant_hooks_for_final_quant: dict[str, list[torch.utils.hooks.RemovableHandle]] = {} # keep them, do not delete


        self.curr_data_name: str | None = None
        self.logger = logging.getLogger("KDTask logger")
    
    
    @classmethod
    def setup_task(cls, cfg, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        return cls(cfg)
    

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = {
            'data': json.load(open(f'{self.cfg.data}/json/{split}.json')),
            'data_dir': self.cfg.data,
            'shuffle': True if split == 'train' else False,
        }
        self.datasets[split] = Namespace(**self.datasets[split])
    
    def dataset(self, split):
        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        
        return self.datasets[split]
    
    
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False
    ):
        return LMLoader(
                self.cfg,
                dataset,
                self.tokenizer,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                epoch=epoch,
                num_shards=num_shards,
                shard_id=shard_id,
        )
    

    def get_teacher_input_text_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False
    ):
        return LMLoader_TextInput(
                self.cfg,
                dataset,
                self.tokenizer,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                epoch=epoch,
                num_shards=num_shards,
                shard_id=shard_id,
        )
    


    def get_teacher_args_and_outputs_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_sentences_training=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        return LMLoader_Args_and_Outputs(
                self.cfg,
                dataset,
                layer_idx=self.current_layer_idx,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_sentences_training=max_sentences_training,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                epoch=epoch,
                num_shards=num_shards,
                shard_id=shard_id,
        )
    

    def get_layer_kwargs(
        self,
        dataset,
        layer_idx
    ):
        kwargs_list = []
        for data in dataset.data:
            # load source from single file, format: self.data_dir/json/{name}.json
            file_path = os.path.join(dataset.data_dir, 'shard', f"{data['name']}_input_kwargs_layer_{str(layer_idx)}.pt")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"file {file_path} not exists")
            kwargs = torch.load(file_path)
            kwargs_list.append(kwargs)
        return kwargs_list
    
        

    def process_after_train(self, model, tokenizer, layer_idx):
        # the param model is the "whole_model"
        layer_args_for_final_quant: tuple[torch.Tensor] | None = None
        layer_kwargs_for_final_quant: dict[str, tp.Any] | None = None
        layer_args_for_final_quant, layer_kwargs_for_final_quant = LlmCalibrationCache(self.calib_config).get_layer_args_kwargs_for_final_quant(
            model,
            layer_idx,
            tokenizer,
            needs_inputs_fn=get_needs_inputs_fn(config=self.quant_config),
            needs_samples_caching=False,
        )

        weight_layer_cache = \
        LlmCalibrationCache(self.calib_config).get_layer_activations(
            layer_struct=self.layer_struct,
            layer_args=layer_args_for_final_quant,
            layer_kwargs=layer_kwargs_for_final_quant,
            needs_inputs_fn=get_needs_inputs_fn(config=self.quant_config),
            batch_size=self.cfg.batch_size_in_quant
        )

        # quant with NO grad
        quantize_llm_decoder_layer_weights(  # noqa: C901
            layer=self.layer_struct,
            config=self.quant_config,
            quant_cache={},
            layer_cache=weight_layer_cache,
            layer_kwargs=layer_kwargs_for_final_quant,
            return_with_quantizers=False,
            return_with_scale_state_dict=False
        )

        del weight_layer_cache
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg.quant_acts_when_training:
            activation_hooks: dict[str, list[torch.utils.hooks.RemovableHandle]] = {}

            self.logger.info(f"after training, quantize activations for layer {self.layer_struct.full_name}")

            activation_layer_caches = \
            LlmCalibrationCache(self.calib_config).get_layer_activations(
                layer_struct=self.layer_struct,
                layer_args=layer_args_for_final_quant,
                layer_kwargs=layer_kwargs_for_final_quant,
                needs_inputs_fn=get_needs_inputs_fn(config=self.quant_config),
                needs_outputs_fn=self.quant_config.needs_quant_outputs,
                batch_size=self.cfg.batch_size_in_quant
            )

            activation_quantizers, activation_hooks = quantize_llm_decoder_layer_activations(
                layer=self.layer_struct,
                config=self.quant_config,
                quant_cache={},
                layer_cache=activation_layer_caches,
                layer_kwargs=layer_kwargs_for_final_quant,
                orig_state_dict=None
            )

            self.activation_quant_hooks_for_final_quant.update(activation_hooks)

            del activation_quantizers, activation_layer_caches
            gc.collect()
            torch.cuda.empty_cache()

        

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        
        model.set_num_updates(update_num)
        logger = self.logger
        logger.info(f"in layer {self.layer_struct.full_name}")

        layer_kwargs = self.total_layer_kwargs[self.layer_struct.full_name]
    
        # region quant weights in this layer
        orig_module_weights: dict[str, nn.Parameter] = {}
        dequantized_weights: dict[str, torch.Tensor] = {}
        logger.info(f"quantizing weights for layer {self.layer_struct.full_name}")
        weight_layer_cache = \
        LlmCalibrationCache(self.calib_config).get_layer_activations(
            layer_struct=self.layer_struct,
            layer_args=sample["args"],
            layer_kwargs=layer_kwargs,
            needs_inputs_fn=get_needs_inputs_fn(config=self.quant_config),
            batch_size=self.cfg.batch_size_in_quant
        )

        orig_module_weights, dequantized_weights = quantize_llm_decoder_layer_weights_with_grad(
            layer=self.layer_struct,
            config=self.quant_config,
            quant_cache={},
            layer_cache=weight_layer_cache,
            layer_kwargs=layer_kwargs,
        )

        del weight_layer_cache
        gc.collect()
        torch.cuda.empty_cache()
        # endregion

        # region quant activations in this layer 
        if self.cfg.quant_acts_when_training:
            activation_hooks: dict[str, list[torch.utils.hooks.RemovableHandle]] = {}

            logger.info(f"quantizing activations for layer {self.layer_struct.full_name}")
            activation_layer_caches = \
            LlmCalibrationCache(self.calib_config).get_layer_activations(
                layer_struct=self.layer_struct,
                layer_args=sample["args"],
                layer_kwargs=layer_kwargs,
                needs_inputs_fn=get_needs_inputs_fn(config=self.quant_config),
                needs_outputs_fn=self.quant_config.needs_quant_outputs,
                batch_size=self.cfg.batch_size_in_quant
            )

            activation_quantizers, activation_hooks = quantize_llm_decoder_layer_activations(
                layer=self.layer_struct,
                config=self.quant_config,
                quant_cache={},
                layer_cache=activation_layer_caches,
                layer_kwargs=layer_kwargs,
                orig_state_dict=None
            )

            del activation_quantizers, activation_layer_caches
            gc.collect()
            torch.cuda.empty_cache()
        # endregion

        model.train()

        # forward
        with torch.autograd.profiler.record_function("forward"):
            logger.info("forward this layer")
            logger.info(f"input_file: {sample['input_file']}")
            logger.info(f"output_file: {sample['output_file']}")
            sample["kwargs"] = layer_kwargs
            loss, sample_size, logging_output = criterion(model, sample)

        if ignore_grad:
            logger.info("Attention!! loss*=0, GRAD IS IGNORED!")
            loss *= 0

        # backward
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        module_name_list, module_list = self.gen_module_and_name_list(self.layer_struct)

        for module_name, module in zip(module_name_list, module_list):
            orig_weight = orig_module_weights[module_name]
            orig_weight.grad = torch.autograd.grad(outputs=dequantized_weights[module_name], inputs=orig_weight, grad_outputs=module.weight.grad)[0]
            module.weight = orig_weight

        del dequantized_weights

        if self.cfg.quant_acts_when_training:
            for _, hook_list in activation_hooks.items():
                for hook in hook_list:
                    hook.remove()
        # DEBUG?
        gc.collect()
        torch.cuda.empty_cache()
        # END DEBUG
        # endregion

        for module_name, module in zip(module_name_list, module_list):
            logger.info(f"In layer {self.layer_struct.full_name}, gradient of {module_name} has nan: {torch.isnan(module.weight.grad).any()}")
            logger.info(f"In layer {self.layer_struct.full_name}, gradient of {module_name} has inf: {torch.isinf(module.weight.grad).any()}")

            grad = module.weight.grad
            inf_mask = torch.isinf(grad)
            inf_indices = torch.nonzero(inf_mask, as_tuple=False)
            if len(inf_indices) > 0:
                first_inf_position = inf_indices[0]  
                # first_inf_value = grad[first_inf_position]  # 获取该位置的值
                logger.info(f"inf: first position {first_inf_position}")

            grad = module.weight.grad
            nan_mask = torch.isnan(grad)
            nan_indices = torch.nonzero(nan_mask, as_tuple=False)
            if len(nan_indices) > 0:
                first_nan_position = nan_indices[0]  
                # first_nan_value = grad[first_nan_position]  # 获取该位置的值
                logger.info(f"nan: first position {first_nan_position}")
                
        return loss, sample_size, logging_output
    

    def gen_module_and_name_list(self, layer_struct: LlmDecoderLayerStruct):

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
        
        return module_name_list, module_list

    
    def process_before_valid(self):
        logger = self.logger
        results_for_partly_quanted = None
        results_for_all_quanted = None

        # DEBUG: now results_for_all_quanted only record the ppl for "model without quant"!
        results_for_all_quanted = self.eval_config.evaluate(self.model_in_llama_class.module, self.tokenizer, "Tiny-llama-1.1b without quant", **{"data_path": "/data/gyy/lmquant-main/lmquant/data/data_without_preprocess_llama_1.1b/shard/less_data/0.jsonl"})

        # quant current layer
        layer_args_for_quant_in_valid: tuple[torch.Tensor] | None = None
        layer_kwargs_for_quant_in_valid: dict[str, tp.Any] | None = None
        logger.info(f"in valid, quantize current layer weights")
        layer_args_for_quant_in_valid, layer_kwargs_for_quant_in_valid = LlmCalibrationCache(self.calib_config).get_layer_args_kwargs_for_final_quant(
            self.model_in_llama_class,
            self.current_layer_idx,
            self.tokenizer,
            needs_inputs_fn=get_needs_inputs_fn(config=self.quant_config),
            needs_samples_caching=False,
        )
        weight_layer_cache = \
        LlmCalibrationCache(self.calib_config).get_layer_activations(
            layer_struct=self.layer_struct,
            layer_args=layer_args_for_quant_in_valid,
            layer_kwargs=layer_kwargs_for_quant_in_valid,
            needs_inputs_fn=get_needs_inputs_fn(config=self.quant_config),
            batch_size=self.cfg.batch_size_in_quant
        )
        quantizers, scale_state_dict, original_weights_current_layer = quantize_llm_decoder_layer_weights(  # noqa: C901
            layer=self.layer_struct,
            config=self.quant_config,
            quant_cache={},
            layer_cache=weight_layer_cache,
            layer_kwargs=layer_kwargs_for_quant_in_valid,
            return_with_quantizers=False,
            return_with_scale_state_dict=False
        )
        self.layers_original_weights_in_valid[self.current_layer_idx] = original_weights_current_layer
        del weight_layer_cache, quantizers, scale_state_dict
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg.quant_acts_when_training:
            logger.info(f"in valid, quantize current layer acts")
            activation_layer_caches = \
            LlmCalibrationCache(self.calib_config).get_layer_activations(
                layer_struct=self.layer_struct,
                layer_args=layer_args_for_quant_in_valid,
                layer_kwargs=layer_kwargs_for_quant_in_valid,
                needs_inputs_fn=get_needs_inputs_fn(config=self.quant_config),
                needs_outputs_fn=self.quant_config.needs_quant_outputs,
                batch_size=self.cfg.batch_size_in_quant
            )
            activation_quantizers, activation_hooks_current_layer = quantize_llm_decoder_layer_activations(
                layer=self.layer_struct,
                config=self.quant_config,
                quant_cache={},
                layer_cache=activation_layer_caches,
                layer_kwargs=layer_kwargs_for_quant_in_valid,
                orig_state_dict=None
            )
            self.activation_quant_hooks_in_valid.update(activation_hooks_current_layer)
            del activation_quantizers, activation_layer_caches
            gc.collect()
            torch.cuda.empty_cache()


        # eval partly quanted model(current layer and previous layers are quanted)
        logger.info("use gptq_eval(lmquant) to calculate ppl, partly quanted")
        results_for_partly_quanted = self.eval_config.evaluate(self.model_in_llama_class.module, self.tokenizer, "Tiny-llama-1.1b partly quanted", **{"data_path": "/data/gyy/lmquant-main/lmquant/data/data_without_preprocess_llama_1.1b/shard/less_data/0.jsonl"})

        # quant following layers
        # logger.info("In vilid_step, quantizing weights for following layers")
        # quant_cache, quantizers, scale_state_dict, layers_original_weights_following_layers = quantize_llm_weights(
        #     self.model_in_llama_class,
        #     self.quant_config,
        #     tokenizer=self.tokenizer,
        #     calib_config=self.calib_config,
        #     return_with_quantizers=False,
        #     return_with_scale_state_dict=False,
        #     begin_layer_idx=self.current_layer_idx+1
        # )
        # del quant_cache, quantizers, scale_state_dict
        # self.layers_original_weights_in_valid.update(layers_original_weights_following_layers)
        # gc.collect()
        # torch.cuda.empty_cache()


        # logger.info("In vilid_step, quantizing activations for following layers")
        # quant_cache, activation_quantizers, activation_hooks_following_layers = quantize_llm_activations(
        #     self.model_in_llama_class,
        #     self.quant_config,
        #     tokenizer=self.tokenizer,
        #     calib_config=self.calib_config,
        #     orig_state_dict=None,
        #     return_with_quantizers=False,
        #     begin_layer_idx=self.current_layer_idx+1
        # )
        # self.activation_quant_hooks_in_valid.update(activation_hooks_following_layers)
        # del quant_cache, activation_quantizers
        # gc.collect()
        # torch.cuda.empty_cache()

        # # eval fully quanted model
        # logger.info("use gptq_eval(lmquant) to calculate ppl, all quanted")
        # results_for_all_quanted = self.eval_config.evaluate(self.model_in_llama_class.module, self.tokenizer, "Tiny-llama-1.1b fully quanted", **{"data_path": "/data/gyy/lmquant-main/lmquant/data/data_without_preprocess_llama_1.1b/shard/less_data/0.jsonl"})

        return results_for_partly_quanted, results_for_all_quanted




    def valid_step(self, sample, model, criterion):
        model.eval()
        self.model_quanted.eval()
        with torch.inference_mode():
            # compute valid_loss & ppl
            loss, sample_size, logging_output = criterion(self.model_quanted, sample)

        return loss, sample_size, logging_output



    def process_after_valid(self):
        # region recover the following layers from quanted state
        if self.activation_quant_hooks_in_valid:
            for _, hook_list in self.activation_quant_hooks_in_valid.items():
                for handle in hook_list:
                    handle.remove()
            self.activation_quant_hooks_in_valid = {}

        layer_structs = self.model_in_llama_class.backbone_struct.layer_structs
        for layer_idx, layer_struct in enumerate(layer_structs):
            if not (layer_idx in self.layers_original_weights_in_valid and self.layers_original_weights_in_valid[layer_idx]):
                continue
            module_name_list, module_list = self.gen_module_and_name_list(layer_struct)
            layer_original_weights = self.layers_original_weights_in_valid[layer_idx]
            for module_name, module in zip(module_name_list, module_list):
                module.weight = layer_original_weights[module_name]
        self.layers_original_weights_in_valid = {}
        # endregion

        if self.model_quanted is not None:
            del self.model_quanted
        gc.collect()
        torch.cuda.empty_cache()


    @property
    def target_dictionary(self):
        padding_idx = self.tokenizer.pad_id
        class Dict:
            def pad(self):
               return padding_idx
        dictionary = Dict()
        return dictionary


    # gen teacher output only for this layer
    def gen_teacher_outputs(self, progress, idx_layer_to_train, data_name):

        if idx_layer_to_train == 0:
            if self.layer_args_exist(idx_layer_to_train, data_name) and self.layer_kwargs_exist(idx_layer_to_train, data_name) and self.layer_outputs_exist(idx_layer_to_train, data_name):
                return

        elif self.layer_outputs_exist(idx_layer_to_train, data_name):
            return

        self.curr_data_name = data_name
        model_struct = self.model_orig
        model = model_struct.module
        layer_structs = model_struct.backbone_struct.layer_structs

        model.eval()
        # region : add hooks 
        teacher_output_hooks: list[torch.utils.hooks.RemovableHandle] = []

        kwargs_cache: dict[str, dict[str, tp.Any]] = {}

        for layer_struct in layer_structs:
            layer = layer_struct.module
            if layer_struct.idx == idx_layer_to_train:
                if layer_struct.idx == 0:
                    teacher_output_hooks.append(
                        layer.register_forward_hook(
                            functools.partial(
                                self._hook_get_args_kwargs_outputs,
                                layer_name=layer_struct.full_name,
                                kwargs_cache=kwargs_cache,
                                data_root_path=self.cfg.data,
                                data_name=data_name,
                                layer_idx=layer_struct.idx,
                                save_args=True,
                                save_kwargs=True,
                                save_outputs=True
                            ),
                            with_kwargs=True
                        )
                    )
                











                # debug
                # elif layer_struct.idx == 1:
                #     teacher_output_hooks.append(
                #         layer.register_forward_hook(
                #             functools.partial(
                #                 self._hook_get_args_kwargs_outputs,
                #                 layer_name=layer_struct.full_name,
                #                 kwargs_cache=kwargs_cache,
                #                 data_root_path=self.cfg.data,
                #                 data_name=data_name,
                #                 layer_idx=layer_struct.idx,
                #                 save_args=True,
                #                 save_kwargs=True,
                #                 save_outputs=True
                #             ),
                #             with_kwargs=True
                #         )
                #     )


















                else:
                    teacher_output_hooks.append(
                        layer.register_forward_hook(
                            functools.partial(
                                self._hook_get_args_kwargs_outputs,
                                layer_name=layer_struct.full_name,
                                kwargs_cache=kwargs_cache,
                                data_root_path=self.cfg.data,
                                data_name=data_name,
                                layer_idx=layer_struct.idx,
                                save_args=False,
                                save_kwargs=False,
                                save_outputs=True
                            ),
                            with_kwargs=True
                        )
                    )
        # endregion

        
        with torch.no_grad():
            for sample in progress:
                if sample['net_input'].device != model.device:
                    sample['net_input'] = sample['net_input'].to(model.device)
                model(sample['net_input'])
        torch.cuda.empty_cache()
        gc.collect()

        if idx_layer_to_train == 0:
            kwargs_save_path = os.path.join(self.cfg.data, "shard", f"{data_name}_input_kwargs_layer_{str(idx_layer_to_train)}.pt")
            torch.save(kwargs_cache, kwargs_save_path)










        # debug
        # if idx_layer_to_train == 1:
        #     kwargs_save_path = os.path.join(self.cfg.data, "shard", f"{data_name}_input_kwargs_layer_{str(idx_layer_to_train)}.pt")
        #     torch.save(kwargs_cache, kwargs_save_path)












        for hook in teacher_output_hooks:
            hook.remove()
        del teacher_output_hooks

        torch.cuda.empty_cache()
        gc.collect()

        return None






    def gen_student_args_and_kwargs(self, progress, idx_layer_to_train, data_name):

        if idx_layer_to_train == 0:
            return   
        
        if self.layer_args_exist(idx_layer_to_train, data_name) and self.layer_kwargs_exist(idx_layer_to_train, data_name):
            return

        self.curr_data_name = data_name

        model = self.model_in_llama_class.module
        model_struct = self.model_in_llama_class
        layer_structs = model_struct.backbone_struct.layer_structs
        model.eval()

        # region : add hooks 
        teacher_output_hooks: list[torch.utils.hooks.RemovableHandle] = []
        kwargs_cache: dict[str, dict[str, tp.Any]] = {}
        
        for layer_struct in layer_structs:
            layer = layer_struct.module
            if layer_struct.idx == idx_layer_to_train and layer_struct.idx != 0:
                teacher_output_hooks.append(
                    layer.register_forward_hook(
                        functools.partial(
                            self._hook_get_args_kwargs_outputs,
                            layer_name=layer_struct.full_name,
                            kwargs_cache=kwargs_cache,
                            data_root_path=self.cfg.data,
                            data_name=data_name,
                            layer_idx=layer_struct.idx,
                            save_args=True,
                            save_kwargs=True,
                            save_outputs=False
                        ),
                        with_kwargs=True
                    )
                )
        # endregion

        with torch.no_grad():
            for sample in progress:
                if sample['net_input'].device != model.device:
                    sample['net_input'] = sample['net_input'].to(model.device)
                model(sample['net_input'])

        if idx_layer_to_train != 0:
            kwargs_save_path = os.path.join(self.cfg.data, "shard", f"{data_name}_input_kwargs_layer_{str(idx_layer_to_train)}.pt")
            torch.save(kwargs_cache, kwargs_save_path)

        for hook in teacher_output_hooks:
            hook.remove()
        del teacher_output_hooks
        torch.cuda.empty_cache()
        gc.collect()

        return None
    


    def layer_args_exist(self, layer_idx, data_name):
        data_root_path=self.cfg.data
        json_path = os.path.join(data_root_path, "json", f"{data_name}_input_args_layer_{str(layer_idx)}.json")
        pt_dir = os.path.join(data_root_path, "shard", f"{data_name}_input_args_layer_{str(layer_idx)}")
        return os.path.exists(json_path) and os.path.exists(pt_dir)
    

    def layer_kwargs_exist(self, layer_idx, data_name):
        data_root_path=self.cfg.data
        kwargs_save_path = os.path.join(data_root_path, "shard", f"{data_name}_input_kwargs_layer_{str(layer_idx)}.pt")
        return os.path.exists(kwargs_save_path)
    

    def layer_outputs_exist(self, layer_idx, data_name):
        data_root_path=self.cfg.data
        json_path = os.path.join(data_root_path, "json", f"{data_name}_teacher_output_layer_{str(layer_idx)}.json")
        pt_dir = os.path.join(data_root_path, "shard", f"{data_name}_teacher_output_layer_{str(layer_idx)}")
        return os.path.exists(json_path) and os.path.exists(pt_dir)


    def del_args_and_kwargs_and_outputs(
        self, 
        layer_idx,
        data_name: str | None = None
    ):
        if data_name is None:
            data_name = self.curr_data_name
        import shutil
        data_root_path=self.cfg.data

        # del args
        json_path = os.path.join(data_root_path, "json", f"{data_name}_input_args_layer_{str(layer_idx)}.json")
        try:
            os.remove(json_path)
            self.logger.info(f"{json_path} is deleted successfully")
        except OSError as e:
            self.logger.info(f"error: {e} when deleting {json_path}")

        pt_dir = os.path.join(data_root_path, "shard", f"{data_name}_input_args_layer_{str(layer_idx)}")
        try:
            shutil.rmtree(pt_dir)
            self.logger.info(f"the folder {pt_dir} is completely deleted")
        except OSError as e:
            self.logger.info(f"error: {e} when deleting {pt_dir}")


        # del kwargs
        kwargs_save_path = os.path.join(self.cfg.data, "shard", f"{data_name}_input_kwargs_layer_{str(layer_idx)}.pt")
        try:
            os.remove(kwargs_save_path)
            self.logger.info(f"{kwargs_save_path} is deleted successfully")
        except OSError as e:
            self.logger.info(f"error: {e} when deleting {kwargs_save_path}")


        # del outputs
        json_path = os.path.join(data_root_path, "json", f"{data_name}_teacher_output_layer_{str(layer_idx)}.json")
        try:
            os.remove(json_path)
            self.logger.info(f"{json_path} is deleted successfully")
        except OSError as e:
            self.logger.info(f"error: {e} when deleting {json_path}")

        pt_dir = os.path.join(data_root_path, "shard", f"{data_name}_teacher_output_layer_{str(layer_idx)}")
        try:
            shutil.rmtree(pt_dir)
            self.logger.info(f"the folder {pt_dir} is completely deleted")
        except OSError as e:
            self.logger.info(f"error: {e} when deleting {pt_dir}")


    # def collect_teacher_outputs(
    #     self, progress, model: LlamaModelFull, layer_structs
    # ):
    #     model.eval()
    #     # region : add hooks 
    #     teacher_output_hooks: list[torch.utils.hooks.RemovableHandle] = []

    #     kwargs_cache: dict[str, dict[str, tp.Any]] = {}

    #     for sample in progress:
    #         data_name = sample['data_name']
    #         break
        

    #     for layer_struct in layer_structs:
    #         layer = layer_struct.module
    #         if layer_struct.idx == 0:
    #             teacher_output_hooks.append(
    #                 layer.register_forward_hook(
    #                     functools.partial(
    #                         self._hook_get_args_kwargs_outputs,
    #                         layer_name=layer_struct.full_name,
    #                         kwargs_cache=kwargs_cache,
    #                         data_root_path=self.cfg.data,
    #                         data_name=data_name,
    #                         layer_idx=layer_struct.idx,
    #                         save_args=True
    #                     ),
    #                     with_kwargs=True
    #                 )
    #             )
    #         else:
    #             teacher_output_hooks.append(
    #                 layer.register_forward_hook(
    #                     functools.partial(
    #                         self._hook_get_args_kwargs_outputs,
    #                         layer_name=layer_struct.full_name,
    #                         kwargs_cache=kwargs_cache,
    #                         data_root_path=self.cfg.data,
    #                         data_name=data_name,
    #                         layer_idx=layer_struct.idx,
    #                         save_args=False
    #                     ),
    #                     with_kwargs=True
    #                 )
    #             )
            
    #     # endregion

        
    #     with torch.no_grad():
    #         for sample in progress:
    #             if sample['net_input'].device != model.llama_model_full.device:
    #                 sample['net_input'] = sample['net_input'].to(model.llama_model_full.device)
    #             model(sample['net_input'])

    #     kwargs_save_path = os.path.join(self.cfg.data, "shard", f"{data_name}_teacher_input_kwargs.pt")
    #     torch.save(kwargs_cache, kwargs_save_path)

    #     for hook in teacher_output_hooks:
    #         hook.remove()
    #     del teacher_output_hooks

    #     torch.cuda.empty_cache()
    #     gc.collect()

    #     return None
    

    def _hook_get_args_kwargs_outputs(
        _self,
        m: nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, tp.Any],
        outputs: tuple,
        *,
        layer_name: str,
        kwargs_cache: dict[str, dict[str, tp.Any]],
        data_root_path: str,
        data_name: str,
        layer_idx: int,
        save_args: bool = True,
        save_kwargs: bool = True,
        save_outputs: bool = True
    ) -> None:
        
        # region save args (output of embedding layer)
        if save_args is True:
            assert all(isinstance(x, torch.Tensor) for x in args)
            json_path = os.path.join(data_root_path, "json", f"{data_name}_input_args_layer_{str(layer_idx)}.json")
            pt_dir = os.path.join(data_root_path, "shard", f"{data_name}_input_args_layer_{str(layer_idx)}")
            if not os.path.exists(pt_dir):
                os.makedirs(pt_dir)
                
            # region count existed file num
            files_and_dirs = os.listdir(pt_dir)
            files = [f for f in files_and_dirs if os.path.isfile(os.path.join(pt_dir, f))]
            file_count_input = len(files)
            # endregion

            pt_path = os.path.join(pt_dir, f"{str(file_count_input).zfill(2)}.pt")

            if not os.path.exists(json_path):
                with open(json_path, "w") as file:
                    file.write("[]")

            with open(json_path, "r") as file:
                items = json.load(file)
            if items is None:
                items = [pt_path]
            else:
                items.append(pt_path)
            with open(json_path, "w") as file:
                json.dump(items, file, indent=4)

            torch.save(args, pt_path)
        # endregion

        # region cache kwargs
        if save_kwargs:
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
                    elif isinstance(v, list) or isinstance(v, tuple):
                        for v_item, cached_item in zip(v, cached):
                            if isinstance(v_item, DynamicCache):
                                assert cached_item is None, f"kwargs_cache[{k}] should be None"
                            elif isinstance(v_item, torch.Tensor):
                                assert v_item.allclose(cached_item), f"kwargs_cache[{k}] should be the same as kwargs[{k}]"
                            else:
                                assert v_item == cached_item, f"kwargs_cache[{k}] should be the same as kwargs[{k}]"
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
        if save_outputs:
            # if isinstance(outputs, list):
            #     outputs = tuple(outputs)
            # elif isinstance(outputs, torch.Tensor):
            #     outputs = (outputs,)
            assert len(outputs) >= len(args)
            outputs = outputs[:len(args)]

            json_path = os.path.join(data_root_path, "json", f"{data_name}_teacher_output_layer_{str(layer_idx)}.json")
            pt_dir = os.path.join(data_root_path, "shard", f"{data_name}_teacher_output_layer_{str(layer_idx)}")
            if not os.path.exists(pt_dir):
                os.makedirs(pt_dir)
                
            # region count existed file num
            files_and_dirs = os.listdir(pt_dir)
            files = [f for f in files_and_dirs if os.path.isfile(os.path.join(pt_dir, f))]
            file_count_output = len(files)
            # endregion

            pt_path = os.path.join(pt_dir, f"{str(file_count_output).zfill(2)}.pt")

            if not os.path.exists(json_path):
                with open(json_path, "w") as file:
                    file.write("[]")

            with open(json_path, "r") as file:
                items = json.load(file)
            if items is None:
                items = [pt_path]
            else:
                items.append(pt_path)
            with open(json_path, "w") as file:
                json.dump(items, file, indent=4)

            torch.save(outputs, pt_path)
        # endregion



    # def delete_useless_teacher_opts(self):
        
    #     base_dir = self.cfg.data
    #     import os

    #     file_path = os.path.join(base_dir, 'json/val_teacher_input_layer_0.json')
    #     try:
    #         os.remove(file_path)
    #         self.logger.info(f"文件 {file_path} 已被删除")
    #     except OSError as e:
    #         self.logger.info(f"删除文件时出错: {e}")

        
    #     for i in range(0, 22):
    #         file_path = os.path.join(base_dir, f'json/val_teacher_output_layer_{str(i)}.json')
    #         try:
    #             os.remove(file_path)
    #             self.logger.info(f"文件 {file_path} 已被删除")
    #         except OSError as e:
    #             self.logger.info(f"删除文件时出错: {e}")
    #     import shutil


    #     file_path = os.path.join(base_dir, 'shard/val_teacher_input_kwargs.pt')
    #     try:
    #         os.remove(file_path)
    #         self.logger.info(f"文件 {file_path} 已被删除")
    #     except OSError as e:
    #         self.logger.info(f"删除文件时出错: {e}")


    #     # 指定要删除的文件夹路径
    #     folder_path = os.path.join(base_dir, 'shard/val_teacher_input_layer_0')

    #     # 删除文件夹及其所有内容
    #     try:
    #         shutil.rmtree(folder_path)
    #         self.logger.info(f"文件夹 {folder_path} 及其所有内容已被删除")
    #     except OSError as e:
    #         self.logger.info(f"删除文件夹时出错: {e}")

    #     for i in range(0, 22):
    #         # 指定要删除的文件夹路径
    #         folder_path = os.path.join(base_dir, f'shard/val_teacher_output_layer_{str(i)}')

    #         # 删除文件夹及其所有内容
    #         try:
    #             shutil.rmtree(folder_path)
    #             self.logger.info(f"文件夹 {folder_path} 及其所有内容已被删除")
    #         except OSError as e:
    #             self.logger.info(f"删除文件夹时出错: {e}")



