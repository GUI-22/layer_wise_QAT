import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from fairseq import distributed_utils, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

from omegaconf import II

from fairseq.model_parallel.megatron.mpu import (
    initialize_model_parallel,
    model_parallel_is_initialized
)
from torch import nn

DEFAULT_MAX_TARGET_POSITIONS = 4096
logger = logging.getLogger(__name__)


@dataclass
class LanguageConfig(FairseqDataclass):
    load_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "path to load checkpoint from"},
    )
    batch_size: int = field(
        default=1,
    )
    share_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    sliding_window: Optional[bool] = field(
        default=None,
    )
    rope_theta: Optional[float] = field(
        default=10000.0,
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    tokens_per_sample: int = II("task.tokens_per_sample")
    model_parallel_size: int = II("common.model_parallel_size")
    

@register_model("llama_model_full", dataclass=LanguageConfig)
class LlamaModelFull(FairseqLanguageModel):
    def __init__(self, args, llama_model_full, tokenizer):
        super().__init__(llama_model_full.model.layers)
        self.args = args
        self.llama_model_full: nn.Module
        self.llama_model_full = llama_model_full
        self.tokenizer = tokenizer

    @classmethod
    def build_model(cls, args, task):
        assert isinstance(task.model, nn.Module)
        if not model_parallel_is_initialized():
            initialize_model_parallel(args.model_parallel_size)
            
        task.model.model.layers = LlamaDecoderLayersInFairseq(task.model.model.layers)
        return cls(args, task.model, task.tokenizer)
    
    def forward(self, *input, **kwargs):
        return self.llama_model_full(*input, **kwargs)
    

class LlamaDecoderLayersInFairseq(FairseqDecoder):
    def __init__(self, model):
        super().__init__(None)
        self.model = model

    def forward(self, *input, **kwargs):
        for layer in self.model:
            layer_outputs = layer(*input, **kwargs)
            (input, kwargs) = layer_outputs
        return layer_outputs

    def __iter__(self):
        for layer in self.model:
            yield layer


    

@register_model_architecture("llama_model_full", "llama_for_layer_wise_qat")
def llama_model_full(args):
    pass

