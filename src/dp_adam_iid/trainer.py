"""A small explicit PyTorch training loop for DP QNLI fine-tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    expected_batch_size: int | None
    phi: float | None


def _log_epoch(
    record: dict[str, Any], metrics_writer: MetricsCSVWriter | None = None
) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)
    if metrics_writer is not None:
        metrics_writer.append(record)


def _extract_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def _progress_bar(iterable=None, *, total=None, desc: str, unit: str):
    """Use concise, rate-limited bars that remain readable through ``tee``."""

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=False,
        dynamic_ncols=True,
        leave=False,
        mininterval=1.0,
    )


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
    with _progress_bar(
        data_loader, total=len(data_loader), desc="Evaluating", unit="batch"
    ) as progress:
        for batch in progress:
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
    print("initializing Opacus private training...", flush=True)
    private: PrivateTraining = make_private_training(model, train_loader, config)
    print(
        f"Opacus private training ready: "
        f"noise_multiplier={private.noise_multiplier}",
        flush=True,
    )
    private_model = private.model
    private_optimizer = private.optimizer
    private_model.train()

    best_accuracy = float("-inf")
    global_step = 0
    final_metrics: dict[str, float] = {"loss": float("nan"), "accuracy": float("nan")}
    final_epsilon = float("nan")
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
            logical_loss_sum = 0.0
            logical_examples = 0
            epoch_loss_sum = 0.0
            epoch_examples = 0
            with _progress_bar(
                total=len(private.data_loader),
                desc=f"Epoch {epoch}/{config.training.epochs}",
                unit="logical step",
            ) as progress:
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

                        logical_loss_sum += loss_value * batch_size
                        logical_examples += batch_size
                        epoch_loss_sum += loss_value * batch_size
                        epoch_examples += batch_size

                        # BatchMemoryManager marks all non-final physical batches
                        # as skipped. This field is part of Opacus' DPOptimizer
                        # state and is the direct indicator of a real optimizer step.
                        if not bool(getattr(private_optimizer, "_is_last_step_skipped", False)):
                            global_step += 1
                            logical_loss = logical_loss_sum / max(logical_examples, 1)
                            progress.update(1)
                            progress.set_postfix(loss=f"{logical_loss:.2f}")
                            logical_loss_sum = 0.0
                            logical_examples = 0

                            if (
                                config.training.max_steps is not None
                                and global_step >= config.training.max_steps
                            ):
                                stop_training = True
                                break

            epoch_train_loss = epoch_loss_sum / max(epoch_examples, 1)
            final_metrics = evaluate_model(private_model, eval_loader, device)
            epsilon = private.privacy_engine.get_epsilon(config.privacy.delta)
            final_epsilon = epsilon
            _log_epoch(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": epoch_train_loss,
                    "val_loss": final_metrics["loss"],
                    "val_accuracy": final_metrics["accuracy"],
                    "epsilon": epsilon,
                    "noise_multiplier": private.noise_multiplier,
                },
                metrics_writer,
            )
            best_accuracy = max(best_accuracy, final_metrics["accuracy"])
            if stop_training:
                break

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
        expected_batch_size=private.expected_batch_size,
        phi=private.phi,
    )
