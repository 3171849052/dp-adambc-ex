from pathlib import Path

import pytest

from dp_adam_iid.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_contains_requested_experiment_defaults():
    config = load_config(PROJECT_ROOT / "config/qnli_roberta_base.yaml")
    assert config.algorithm == "dpadam"
    assert config.model.name == "FacebookAI/roberta-base"
    assert config.data.logical_batch_size == 1024
    assert config.data.max_physical_batch_size == 8
    assert config.data.max_length == 128
    assert config.training.epochs == 3
    assert config.training.learning_rate == 1.0e-4
    assert config.privacy.epsilon == 3.0
    assert config.privacy.delta == 1.0e-5
    assert config.privacy.max_grad_norm == 1.0
    assert config.privacy.accountant == "gdp"
    assert config.privacy.grad_sample_mode == "ghost"
    assert config.output.root == "outputs"


def test_config_rejects_non_adam_or_non_gdp_choices():
    config = load_config(PROJECT_ROOT / "config/qnli_roberta_base.yaml")
    config.training.optimizer = "adamw"
    with pytest.raises(ValueError, match="Adam"):
        config.validate()

    config.training.optimizer = "adam"
    config.privacy.accountant = "rdp"
    with pytest.raises(ValueError, match="gdp"):
        config.validate()
