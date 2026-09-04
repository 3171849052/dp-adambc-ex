"""BERT sequence-classification model construction for QNLI."""

from __future__ import annotations

import torch
from torch import nn
from transformers import BertForSequenceClassification, PreTrainedTokenizerBase

from .config import Config


def _official_train_from_scratch_init(module: nn.Module) -> None:
    """Match DP-AdamBC's train_from_scratch initialization for trainable blocks."""

    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=1.0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def prepare_bert_for_qnli(
    model: BertForSequenceClassification,
) -> BertForSequenceClassification:
    """Freeze pretrained BERT except layer 11, pooler, and classifier.

    The three trainable blocks are reinitialized exactly as in the official
    PyTorch experiment's train_from_scratch branch. All frozen blocks retain
    their pretrained parameters.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_blocks = (
        model.bert.encoder.layer[-1],
        model.bert.pooler,
        model.classifier,
    )
    for block in trainable_blocks:
        block.apply(_official_train_from_scratch_init)
        for parameter in block.parameters():
            parameter.requires_grad = True
    return model


def build_model(
    config: Config,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> BertForSequenceClassification:
    del tokenizer
    model = BertForSequenceClassification.from_pretrained(
        config.model.name,
        num_labels=config.model.num_labels,
    )
    return prepare_bert_for_qnli(model)
