from pathlib import Path

import pytest

import yaml

from dp_adam_iid.config import Config, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_contains_requested_experiment_defaults():
    config = load_config(PROJECT_ROOT / "config/qnli_roberta_base.yaml")
    assert config.algorithm == "dpadam"
    assert config.model.name == "FacebookAI/roberta-base"
    assert config.data.logical_batch_size == 512
    assert config.data.max_physical_batch_size == 32
    assert config.data.max_length == 256
    assert config.data.first_sent_limit == 200
    assert config.data.other_sent_limit == 200
    assert config.data.truncate_head is True
    assert config.training.epochs == 5
    assert config.training.learning_rate == 1.0e-4
    assert config.privacy.epsilon == 3.0
    assert config.privacy.delta == 1.0e-5
    assert config.privacy.max_grad_norm == 1.0
    assert config.privacy.accountant == "gdp"
    assert config.privacy.grad_sample_mode == "ghost"
    assert config.output.root == "outputs"
    assert config.runtime.gpu == 0


def test_config_rejects_non_adam_or_non_gdp_choices():
    config = load_config(PROJECT_ROOT / "config/qnli_roberta_base.yaml")
    config.training.optimizer = "adamw"
    with pytest.raises(ValueError, match="Adam"):
        config.validate()

    config.training.optimizer = "adam"
    config.privacy.accountant = "rdp"
    with pytest.raises(ValueError, match="gdp"):
        config.validate()


@pytest.mark.parametrize("field", ["max_length", "first_sent_limit", "other_sent_limit"])
def test_config_rejects_nonpositive_prompt_lengths(field):
    raw = yaml.safe_load(
        (PROJECT_ROOT / "config/qnli_roberta_base.yaml").read_text(encoding="utf-8")
    )
    raw["data"][field] = 0

    with pytest.raises(ValueError, match=field):
        Config.from_dict(raw)


@pytest.mark.parametrize("gpu", [-1, 1.5, "1", True])
def test_config_rejects_invalid_gpu_values(gpu):
    raw = yaml.safe_load(
        (PROJECT_ROOT / "config/qnli_roberta_base.yaml").read_text(encoding="utf-8")
    )
    raw["runtime"]["gpu"] = gpu
    with pytest.raises(ValueError, match="runtime.gpu"):
        Config.from_dict(raw)
