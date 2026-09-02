"""A small explicit PyTorch training loop for DP QNLI fine-tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .privacy import PrivateTraining, make_private_training
from .run_logging import MetricsCSVWriter


@dataclass
class TrainingResult:
    global_step: int
    best_accuracy: float
    final_metrics: dict[str, float]
    noise_multiplier: float
    epsilon: float
    sample_rate: float


def _log(record: dict[str, Any], metrics_writer: MetricsCSVWriter | None = None) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)
    if metrics_writer is not None:
        metrics_writer.append(
            {
                "phase": record["phase"],
                "epoch": record["epoch"],
                "step": record["step"],
                "loss": record["loss"],
                "accuracy": record["accuracy"],
                "epsilon": record["epsilon"],
                "noise_multiplier": record["noise_multiplier"],
            }
        )


def _extract_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate accuracy and mean cross-entropy on the QNLI validation split."""

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in data_loader:
        if not batch["labels"].numel():
            continue
        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        logits = _extract_logits(model(**batch))
        total_loss += float(criterion(logits, labels).item())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total_examples += int(labels.shape[0])
    if total_examples == 0:
        raise RuntimeError("Evaluation loader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_model(
    config: Config,
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    metrics_writer: MetricsCSVWriter | None = None,
) -> TrainingResult:
    """Train a model using one private Adam update per logical batch."""

    model.to(device)
    # Opacus validates that the model is in training mode before installing
    # Ghost Clipping hooks. ``from_pretrained`` returns Transformers models in
    # eval mode, so set this explicitly before make_private().
    model.train()
    private: PrivateTraining = make_private_training(model, train_loader, config)
    private_model = private.model
    private_optimizer = private.optimizer
    private_model.train()

    best_accuracy = float("-inf")
    global_step = 0
    final_metrics: dict[str, float] = {"loss": float("nan"), "accuracy": float("nan")}
    stop_training = False
    sample_rate = float(
        getattr(
            private.data_loader,
            "sample_rate",
            config.data.logical_batch_size / max(len(train_loader.dataset), 1),
        )
    )

    try:
        for epoch in range(1, config.training.epochs + 1):
            private_model.train()
            running_loss = 0.0
            running_examples = 0
            with BatchMemoryManager(
                data_loader=private.data_loader,
                max_physical_batch_size=config.data.max_physical_batch_size,
                optimizer=private_optimizer,
            ) as memory_safe_loader:
                for batch in memory_safe_loader:
                    batch_size = int(batch["labels"].shape[0])
                    if batch_size == 0:
                        # Poisson sampling can produce an empty batch. Consume
                        # the corresponding optimizer signal without updating.
                        private_optimizer.zero_grad()
                        private_optimizer.step()
                        private_optimizer.zero_grad()
                        continue

                    batch = _move_batch(batch, device)
                    labels = batch.pop("labels")
                    private_optimizer.zero_grad()
                    outputs = private_model(**batch)
                    loss = private.criterion(_extract_logits(outputs), labels)
                    loss_value = float(loss.item())
                    loss.backward()
                    private_optimizer.step()
                    private_optimizer.zero_grad()

                    running_loss += loss_value * batch_size
                    running_examples += batch_size

                    # BatchMemoryManager marks all non-final physical batches
                    # as skipped. This field is part of Opacus' DPOptimizer
                    # state and is the direct indicator of a real optimizer step.
                    if not bool(getattr(private_optimizer, "_is_last_step_skipped", False)):
                        global_step += 1
                        if global_step % config.logging.log_every_steps == 0:
                            epsilon = private.privacy_engine.get_epsilon(config.privacy.delta)
                            _log(
                                {
                                    "epoch": epoch,
                                    "step": global_step,
                                    "loss": running_loss / max(running_examples, 1),
                                    "accuracy": None,
                                    "epsilon": epsilon,
                                    "noise_multiplier": private.noise_multiplier,
                                    "phase": "train",
                                },
                                metrics_writer,
                            )
                        running_loss = 0.0
                        running_examples = 0

                        if (
                            config.training.max_steps is not None
                            and global_step >= config.training.max_steps
                        ):
                            stop_training = True
                            break

            final_metrics = evaluate_model(private_model, eval_loader, device)
            epsilon = private.privacy_engine.get_epsilon(config.privacy.delta)
            _log(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": final_metrics["loss"],
                    "accuracy": final_metrics["accuracy"],
                    "epsilon": epsilon,
                    "noise_multiplier": private.noise_multiplier,
                    "phase": "validation",
                },
                metrics_writer,
            )
            best_accuracy = max(best_accuracy, final_metrics["accuracy"])
            if stop_training:
                break

        final_epsilon = private.privacy_engine.get_epsilon(config.privacy.delta)
    finally:
        if private.hooks is not None:
            private.hooks.cleanup()

    return TrainingResult(
        global_step=global_step,
        best_accuracy=best_accuracy,
        final_metrics=final_metrics,
        noise_multiplier=private.noise_multiplier,
        epsilon=final_epsilon,
        sample_rate=sample_rate,
    )
