"""Hugging Face model construction."""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification

from .config import Config


def build_model(config: Config):
    return AutoModelForSequenceClassification.from_pretrained(
        config.model.name,
        num_labels=config.model.num_labels,
    )
