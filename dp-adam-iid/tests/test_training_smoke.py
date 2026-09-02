from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dp_adam_iid.config import Config
from dp_adam_iid.run_logging import (
    MetricsCSVWriter,
    create_run_directory,
    tee_output,
    write_run_metadata,
)
from dp_adam_iid.trainer import train_model


class TinyDataset(Dataset):
    def __init__(self, size=8):
        self.input_ids = torch.arange(1, size * 3 + 1).view(size, 3) % 15
        self.labels = torch.arange(size) % 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "labels": self.labels[index]}


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.classifier(self.embedding(input_ids).mean(dim=1)))


def test_training_smoke_writes_metrics_without_checkpoints(tmp_path: Path):
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
            "max_steps": 1,
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
    model = TinyClassifier()
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

    assert result.global_step == 1
    assert 0.0 <= result.final_metrics["accuracy"] <= 1.0
    assert {
        "config.yaml",
        "resolved_config.yaml",
        "metrics.csv",
        "summary.json",
        "train.log",
    } == {path.name for path in paths.directory.iterdir()}
    assert not list(paths.directory.glob("*.pt"))
    assert "\"phase\": \"train\"" in paths.train_log.read_text()
    assert "\"phase\": \"validation\"" in paths.train_log.read_text()
