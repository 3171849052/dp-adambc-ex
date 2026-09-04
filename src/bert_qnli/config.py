"""Configuration loading and validation for the BERT QNLI experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str
    num_labels: int = 2


@dataclass
class DataConfig:
    dataset_name: str = "glue"
    dataset_config: str = "qnli"
    train_split: str = "train"
    eval_split: str = "validation"
    max_length: int = 256
    logical_batch_size: int = 1024
    max_physical_batch_size: int = 8
    eval_batch_size: int = 32
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


@dataclass
class TrainingConfig:
    epochs: int = 3
    max_steps: int | None = None
    learning_rate: float = 1.0e-4
    optimizer: str = "adam"
    gamma_prime: float = 1.0e-8
    fpc_lambda: float = 0.5
    fpc_mode: str = "current"
    fpc_delay_q0: float = 1.0
    weight_decay: float = 0.0
    warmup_steps: int = 0
    scheduler: str = "none"


@dataclass
class PrivacyConfig:
    epsilon: float = 3.0
    delta: float = 1.0e-5
    accountant: str = "gdp"
    noise_type: str = "iid_gaussian"
    max_grad_norm: float = 1.0
    clipping: str = "flat"
    grad_sample_mode: str = "ghost"
    poisson_sampling: bool = True
    loss_reduction: str = "mean"
    wrap_model: bool = False


@dataclass
class RuntimeConfig:
    device: str = "auto"
    gpu: int = 0
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class OutputConfig:
    root: str = "outputs"


@dataclass
class LoggingConfig:
    log_every_steps: int = 1


@dataclass
class Config:
    algorithm: str
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    privacy: PrivacyConfig
    runtime: RuntimeConfig
    output: OutputConfig
    logging: LoggingConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        algorithm = raw.get("algorithm")
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise ValueError("algorithm is required and must be a non-empty string")
        config = cls(
            algorithm=algorithm,
            seed=int(raw.get("seed", 0)),
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            training=TrainingConfig(**raw.get("training", {})),
            privacy=PrivacyConfig(**raw.get("privacy", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            output=OutputConfig(**raw.get("output", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model.num_labels != 2:
            raise ValueError("QNLI is a binary classification task: num_labels must be 2")
        if self.data.dataset_name != "glue" or self.data.dataset_config != "qnli":
            raise ValueError("This project supports only glue/qnli")
        if self.data.train_split != "train" or self.data.eval_split != "validation":
            raise ValueError("QNLI must use train and validation splits")
        if self.data.max_length <= 0:
            raise ValueError("max_length must be positive")
        if (
            self.data.logical_batch_size <= 0
            or self.data.max_physical_batch_size <= 0
            or self.data.eval_batch_size <= 0
        ):
            raise ValueError("batch sizes must be positive")
        if self.data.max_train_samples is not None and self.data.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive or null")
        if self.data.max_eval_samples is not None and self.data.max_eval_samples <= 0:
            raise ValueError("max_eval_samples must be positive or null")

        if self.training.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.training.max_steps is not None and self.training.max_steps <= 0:
            raise ValueError("max_steps must be positive or null")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.training.optimizer.lower() not in {
            "adam",
            "dpadambc",
            "fpcdpadam",
        }:
            raise ValueError("Only Adam, DPAdamBC, and FPCDPAdam optimizers are supported")
        expected_optimizer = {
            "dpadam": "adam",
            "dpadambc": "dpadambc",
            "fpcdpadam": "fpcdpadam",
        }.get(self.algorithm.lower())
        if expected_optimizer is None:
            raise ValueError(
                "algorithm must be 'dpadam', 'dpadambc', or 'fpcdpadam'"
            )
        if self.training.optimizer.lower() != expected_optimizer:
            raise ValueError(
                f"algorithm '{self.algorithm}' requires training.optimizer "
                f"to be '{expected_optimizer}'"
            )
        if self.training.gamma_prime <= 0:
            raise ValueError("gamma_prime must be positive")
        if expected_optimizer == "fpcdpadam":
            if not 0 <= self.training.fpc_lambda <= 1:
                raise ValueError("fpc_lambda must be between 0 and 1")
            if self.training.fpc_mode not in {"current", "delay"}:
                raise ValueError("fpc_mode must be 'current' or 'delay'")
            if self.training.fpc_delay_q0 <= 0:
                raise ValueError("fpc_delay_q0 must be positive")
        if self.training.weight_decay != 0:
            raise ValueError("weight_decay is intentionally disabled")
        if self.training.warmup_steps != 0:
            raise ValueError("warmup is intentionally disabled")
        if self.training.scheduler.lower() != "none":
            raise ValueError("scheduler/decay is intentionally disabled")

        if self.privacy.epsilon <= 0 or self.privacy.delta <= 0:
            raise ValueError("epsilon and delta must be positive")
        if self.privacy.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.privacy.accountant.lower() != "gdp":
            raise ValueError("This project requires PrivacyEngine(accountant='gdp')")
        if self.privacy.noise_type.lower() != "iid_gaussian":
            raise ValueError("Only IID Gaussian noise is supported")
        if self.privacy.clipping.lower() != "flat":
            raise ValueError("Only global flat clipping is supported")
        if self.privacy.grad_sample_mode.lower() != "ghost":
            raise ValueError("Only Opacus Ghost Clipping is supported")
        if not self.privacy.poisson_sampling:
            raise ValueError("Poisson sampling is required")
        if self.privacy.loss_reduction.lower() != "mean":
            raise ValueError("The training loop uses mean loss reduction")

        if self.runtime.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if isinstance(self.runtime.gpu, bool) or not isinstance(self.runtime.gpu, int):
            raise ValueError("runtime.gpu must be a non-negative integer")
        if self.runtime.gpu < 0:
            raise ValueError("runtime.gpu must be a non-negative integer")
        if not isinstance(self.output.root, str) or not self.output.root.strip():
            raise ValueError("output.root must be a non-empty path")
        if self.logging.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return Config.from_dict(raw)
