# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class KLDivergenceCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    # kl_temperature: bool = II("optimization.kl_temperature")


@register_criterion("kl_divergence", dataclass=KLDivergenceCriterionConfig)
class KLDivergenceCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, temperature: float | None = 1.0):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.temperature = temperature

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
            "sample_size": sample_size
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        net_output = (net_output[0] / self.temperature,) + net_output[1:]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1)) 

        target = (sample["teacher_outputs"][0] / self.temperature,) + sample["teacher_outputs"][1:]
        target = model.get_normalized_probs(target, log_probs=True)
        target = target.view(-1, target.size(-1))

        loss = F.kl_div(
            lprobs,
            target,
            reduction="sum" if reduce else "none",
            log_target=True
        ) ** self.temperature
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=10
        )
        metrics.log_scalar(
            "loss_sum", loss_sum, sample_size, round=10
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=10
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
