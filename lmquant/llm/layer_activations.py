
import os
import random
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn as nn
from datasets import load_dataset
from omniconfig import configclass
from transformers.cache_utils import DynamicCache
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from lmquant.dataset.cache.action import AverageCache, CacheAction, ConcatCache
from lmquant.dataset.cache.activation import ActivationCache, IOActivationsCache
from lmquant.dataset.cache.calibration import CalibrationCache
from lmquant.dataset.config import BaseCalibDatasetConfig
from lmquant.dataset.transform import LinearTransformFn

from .nn import LlmDecoderLayerStruct, LlmModelStruct, RotaryEmbedding




def iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module | LlmModelStruct,
        *args,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool],
        action: CacheAction | None = None,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | None = None,
        needs_samples_caching: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                LlmDecoderLayerStruct,
                dict[str, IOActivationsCache],
                dict[str, tp.Any],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model activations for each layer.

        Args:
            model (nn.Module | LlmModelStruct): Model.
            action (CacheAction): Action for caching activations. If ``None``, ``LlmConcatCache("cpu")`` is used.
                Defaults to ``None``.
            needs_inputs (Callable[[str, nn.Module], bool]): Function for determining whether to cache inputs
                for a module given its name and itself.
            needs_outputs (Callable[[str, nn.Module], bool], optional): Function for determining whether to
                cache outputs for a module given its name and itself. Defaults to ``None``. If ``None``,
                ``False`` is always returned.
            needs_samples_caching (bool, optional): Whether to cache input samples. Defaults to ``True``.
            *args: Arguments for ``_iter_samples``.
            **kwargs: Keyword arguments for ``_iter_samples``.

        Yields:
            Generator[tuple[str, tuple[LlmDecoderLayerStruct, dict[str, IOActivationsCache], dict[str, Any]]],
                    None, None]: Generator of tuple of
                - layer name
                - a tuple of
                    - layer struct,
                    - input and output caches for each module in the layer,
                    - layer input keyword arguments.
        """
        if isinstance(model, LlmModelStruct):
            model_struct = model
            model = model_struct.module
        else:
            model_struct = LlmModelStruct.build(model)
        backbone_struct = model_struct.backbone_struct
        layer_structs = backbone_struct.layer_structs
        action = LlmConcatCache("cpu") if action is None else action
        for layer_idx, (layer_name, (layer, layer_cache, layer_kwargs)) in enumerate(
            self._iter_layer_activations(
                model,
                *args,
                action=action,
                layers=backbone_struct.layers,
                needs_inputs_fn=needs_inputs_fn,
                needs_outputs_fn=needs_outputs_fn,
                needs_samples_caching=needs_samples_caching,
                **kwargs,
            )
        ):
            layer_struct = layer_structs[layer_idx]
            assert layer_idx == layer_struct.idx
            assert layer_name == layer_struct.full_name
            assert layer is layer_struct.module
            if layer_struct.proj_v_full_name in layer_cache:
                cache = layer_cache[layer_struct.proj_v_full_name]
                layer_cache[layer_struct.proj_q_full_name] = cache
                layer_cache[layer_struct.proj_k_full_name] = cache
            if layer_struct.proj_1st_full_names[0] in layer_cache:
                for expert_idx in range(layer_struct.config.num_experts):
                    cache = layer_cache[layer_struct.proj_1st_full_names[expert_idx]]
                    for name in layer_struct.proj_1st_full_names[expert_idx :: layer_struct.config.num_experts]:
                        layer_cache[name] = cache
                if layer_struct.config.num_experts == 1 and layer_struct.ffn_block_full_name not in layer_cache:
                    layer_cache[layer_struct.ffn_block_full_name] = layer_cache[layer_struct.proj_1st_full_names[0]]
            if layer_struct.config.num_experts > 1 and layer_struct.ffn_block_full_name in layer_cache:
                layer_cache[layer_struct.router_full_name] = layer_cache[layer_struct.ffn_block_full_name]
            yield layer_name, (layer_struct, layer_cache, layer_kwargs)
