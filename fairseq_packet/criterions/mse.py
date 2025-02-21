# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class MSECriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("mse", dataclass=MSECriterionConfig)
class MSECriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(*sample["args"], **sample["kwargs"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        ntokens = sample["teacher_outputs"][0].size(0) * sample["teacher_outputs"][0].size(1)
        nsentences = sample["teacher_outputs"][0].size(0)
        sample_size = nsentences if self.sentence_avg else ntokens
        logging_output = {
            "loss": loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "hidden_dim": sample["teacher_outputs"][0].size(2)
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        net_output = net_output[0]
        net_output = net_output.view(-1, net_output.size(-1)).to(torch.float32)

        target = sample["teacher_outputs"][0]
        target = target.view(-1, target.size(-1)).to(torch.float32)

        #F.mse_loss有在每个token，对token对应的所有hiddens做平均吗？？
        loss = F.mse_loss(
            net_output,
            target,
            reduction="sum" if reduce else "none"
        ) 
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        hidden_dim = logging_outputs[0].get("hidden_dim", 0)

#下面几个metrics.log_scalar要改
        metrics.log_scalar(
            "loss", loss_sum / sample_size / hidden_dim , sample_size * hidden_dim, round=10
        )
        metrics.log_scalar(
            "loss_per_token", loss_sum / sample_size , sample_size, round=10
        )
        metrics.log_scalar(
            "loss_sum", loss_sum, 1, round=10
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens, ntokens, round=10
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
