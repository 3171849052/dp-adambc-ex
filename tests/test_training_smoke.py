import csv
from pathlib import Path
from datetime import datetime

import torch
import pytest
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig, RobertaForMaskedLM

from roberta_qnli.config import Config
from roberta_qnli.model import RobertaPromptForQNLI
from opacus import PrivacyEngine
from roberta_qnli.run_logging import (
    FPCDiagnosticsCSVWriter,
    MetricsCSVWriter,
    create_run_directory,
    tee_output,
    write_run_metadata,
)
from roberta_qnli.trainer import train_model


class TinyDataset(Dataset):
    def __init__(self, size=8):
        self.input_ids = torch.tensor(
            [[0, 7 + index, 4, 15 + index, 2] for index in range(size)]
        )
        self.attention_mask = torch.ones_like(self.input_ids)
        self.mask_pos = torch.full((size,), 2)
        self.labels = torch.arange(size) % 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "mask_pos": self.mask_pos[index],
            "labels": self.labels[index],
        }


def _tiny_prompt_model():
    masked_lm = RobertaForMaskedLM(
        RobertaConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
        )
    )
    return RobertaPromptForQNLI(masked_lm, yes_token_id=5, no_token_id=6)


def test_prompt_dp_smoke_writes_one_epoch_row_without_step_metrics(
    tmp_path: Path, monkeypatch
):
    raw = {
        "algorithm": "dpadam",
        "seed": 0,
        "model": {"name": "test", "num_labels": 2},
        "data": {
            "dataset_name": "glue",
            "dataset_config": "qnli",
            "train_split": "train",
            "eval_split": "validation",
            "max_length": 128,
            "logical_batch_size": 4,
            "max_physical_batch_size": 2,
            "eval_batch_size": 4,
        },
        "training": {
            "epochs": 1,
            "max_steps": None,
            "learning_rate": 1.0e-4,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "scheduler": "none",
        },
        "privacy": {
            "epsilon": 3.0,
            "delta": 1.0e-5,
            "accountant": "gdp",
            "noise_type": "iid_gaussian",
            "max_grad_norm": 1.0,
            "clipping": "flat",
            "grad_sample_mode": "ghost",
            "poisson_sampling": True,
            "loss_reduction": "mean",
            "wrap_model": False,
        },
        "runtime": {"device": "cpu", "num_workers": 0, "pin_memory": False},
        "output": {
            "root": str(tmp_path),
        },
        "logging": {"log_every_steps": 1},
    }
    config = Config.from_dict(raw)
    paths = create_run_directory(
        config, now=datetime(2026, 9, 2, 18, 5, 0)
    )
    source_yaml = "algorithm: dpadam\n"
    write_run_metadata(
        paths, source_yaml=source_yaml, resolved_config=config.to_dict()
    )
    metrics_writer = MetricsCSVWriter(paths.metrics)
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
    model = _tiny_prompt_model()
    epsilon_calls = 0
    original_get_epsilon = PrivacyEngine.get_epsilon

    def counted_get_epsilon(self, delta):
        nonlocal epsilon_calls
        epsilon_calls += 1
        return original_get_epsilon(self, delta)

    monkeypatch.setattr(PrivacyEngine, "get_epsilon", counted_get_epsilon)
    with tee_output(paths.train_log):
        result = train_model(
            config,
            model,
            loader,
            loader,
            torch.device("cpu"),
            metrics_writer=metrics_writer,
        )
    write_run_metadata(
        paths,
        source_yaml=source_yaml,
        resolved_config=config.to_dict(),
        summary={"global_step": result.global_step},
    )

    assert result.global_step == len(loader)
    assert result.expected_batch_size is None
    assert result.phi is None
    assert epsilon_calls == 1
    assert 0.0 <= result.final_metrics["accuracy"] <= 1.0
    assert {
        "config.yaml",
        "resolved_config.yaml",
        "metrics.csv",
        "summary.json",
        "train.log",
    } == {path.name for path in paths.directory.iterdir()}
    assert not list(paths.directory.glob("*.pt"))
    log = paths.train_log.read_text()
    assert "\"phase\"" not in log
    assert log.count("\"global_step\"") == 1
    assert "initializing Opacus private training..." in log
    assert "Opacus private training ready: noise_multiplier=" in log
    assert "Epoch 1/1" in log
    assert "Evaluating" in log

    with paths.metrics.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert list(rows[0]) == [
        "epoch",
        "global_step",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "epsilon",
        "noise_multiplier",
        "predictor_x_gap_mse",
    ]
    assert int(rows[0]["global_step"]) == result.global_step


def test_fpc_training_writes_one_diagnostic_per_logical_step(tmp_path: Path):
    config = Config.from_dict(
        {
            "algorithm": "fpcdpadam",
            "seed": 0,
            "model": {"name": "test", "num_labels": 2},
            "data": {
                "logical_batch_size": 4,
                "max_physical_batch_size": 2,
                "eval_batch_size": 4,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 1.0e-4,
                "optimizer": "fpcdpadam",
                "gamma_prime": 1.0e-4,
                "fpc_lambda": 0.5,
                "fpc_mode": "current",
                "fpc_delay_q0": 1.0,
            },
            "privacy": {
                "epsilon": 3.0,
                "delta": 1.0e-5,
                "accountant": "gdp",
                "max_grad_norm": 1.0,
                "clipping": "flat",
                "grad_sample_mode": "ghost",
                "poisson_sampling": True,
                "loss_reduction": "mean",
                "wrap_model": False,
            },
            "runtime": {"device": "cpu", "pin_memory": False},
            "output": {"root": str(tmp_path)},
        }
    )
    paths = create_run_directory(config)
    metrics_writer = MetricsCSVWriter(paths.metrics)
    diagnostics_writer = FPCDiagnosticsCSVWriter(paths.fpc_diagnostics)
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)

    result = train_model(
        config,
        _tiny_prompt_model(),
        loader,
        loader,
        torch.device("cpu"),
        metrics_writer=metrics_writer,
        fpc_diagnostics_writer=diagnostics_writer,
    )

    with paths.fpc_diagnostics.open(newline="", encoding="utf-8") as stream:
        diagnostic_rows = list(csv.DictReader(stream))
    assert len(diagnostic_rows) == result.global_step
    assert [int(row["global_step"]) for row in diagnostic_rows] == list(
        range(1, result.global_step + 1)
    )
    assert all(float(row["predictor_x_gap_mse"]) >= 0 for row in diagnostic_rows)

    with paths.metrics.open(newline="", encoding="utf-8") as stream:
        metrics_rows = list(csv.DictReader(stream))
    expected_mean = sum(
        float(row["predictor_x_gap_mse"]) for row in diagnostic_rows
    ) / len(diagnostic_rows)
    assert float(metrics_rows[0]["predictor_x_gap_mse"]) == pytest.approx(expected_mean)
