#!/usr/bin/env python
"""Run the standalone BERT/QNLI DP-AdamBC clamp diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# These imports are intentionally limited to Hugging Face/data/model code and
# the local optimizer wrapper. Opacus, SciPy, and bert_qnli.privacy are loaded
# inside run(), after QNLI has been loaded and BERT has been constructed.
from bert_qnli.config import load_config  # noqa: E402
from bert_qnli.data import load_qnli  # noqa: E402
from bert_qnli.model import build_model  # noqa: E402
from bert_qnli.utils import resolve_device, set_seed  # noqa: E402

from exp2.csv_writer import DiagnosticsCSVWriter, ValidationCSVWriter  # noqa: E402
from exp2.diagnostic_optimizer import DiagnosticDPAdamBC  # noqa: E402


DEFAULT_CONFIG = ROOT / "exp2/configs/bc_g3e-9_lr3e-3.yaml"
DEFAULT_EVAL_STEPS = (410, 820, 1230)


def is_real_logical_step(dp_optimizer: Any) -> bool:
    """Return whether the preceding DPOptimizer call made a real update."""

    return not bool(getattr(dp_optimizer, "_is_last_step_skipped", False))


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def _extract_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Compute mean validation loss and accuracy without touching optimizer state."""

    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    examples = 0
    for batch in data_loader:
        if not batch["labels"].numel():
            continue
        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        logits = _extract_logits(model(**batch))
        loss_sum += float(criterion(logits, labels).item())
        correct += int((logits.argmax(dim=-1) == labels).sum().item())
        examples += int(labels.shape[0])
    if examples == 0:
        raise RuntimeError("validation loader produced no examples")
    if was_training:
        model.train()
    return {
        "loss": loss_sum / examples,
        "accuracy": correct / examples,
    }


def _resolve_output_dir(config, output_dir: Path | None) -> Path:
    selected = Path(config.output.root) if output_dir is None else output_dir
    return selected if selected.is_absolute() else ROOT / selected


def _parse_eval_steps(value: str | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_EVAL_STEPS
    try:
        steps = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise ValueError("--eval-steps must be a comma-separated list of integers") from error
    if any(step <= 0 for step in steps):
        raise ValueError("evaluation steps must be positive")
    return steps


def _write_initial_metadata(
    output_dir: Path,
    *,
    source_yaml: str,
    config,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(source_yaml, encoding="utf-8")
    resolved = config.to_dict()
    resolved["runtime"]["actual_device"] = str(device)
    resolved["run"] = {"directory": str(output_dir.resolve())}
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )


def run(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    max_steps: int | None = None,
    device_name: str | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    eval_steps: tuple[int, ...] = DEFAULT_EVAL_STEPS,
) -> int:
    """Run QNLI training and write only scalar diagnostics and validation rows."""

    source_yaml = config_path.read_text(encoding="utf-8")
    config = load_config(config_path)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        config.training.max_steps = max_steps
    if device_name is not None:
        config.runtime.device = device_name
    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive")
        config.data.max_train_samples = max_train_samples
    if max_eval_samples is not None:
        if max_eval_samples <= 0:
            raise ValueError("max_eval_samples must be positive")
        config.data.max_eval_samples = max_eval_samples
    if config.algorithm.lower() != "dpadambc":
        raise ValueError("Experiment 2 requires algorithm=dpadambc")
    if config.model.name != "bert-base-cased" or config.data.dataset_config != "qnli":
        raise ValueError("Experiment 2 requires bert-base-cased + QNLI")

    set_seed(config.seed)
    device = resolve_device(config.runtime.device)
    destination = _resolve_output_dir(config, output_dir)
    _write_initial_metadata(
        destination, source_yaml=source_yaml, config=config, device=device
    )
    diagnostics_writer = DiagnosticsCSVWriter(destination / "bc_diagnostics.csv")
    validation_writer = ValidationCSVWriter(destination / "validation_metrics.csv")

    print("loading QNLI data...", flush=True)
    data = load_qnli(config)
    print(
        f"QNLI data ready: train_examples={data.train_size} eval_examples={data.eval_size}",
        flush=True,
    )
    print("loading BERT classifier model...", flush=True)
    model = build_model(config, data.tokenizer)
    model.to(device)
    # from_pretrained() returns eval mode; Ghost Clipping requires training mode.
    model.train()
    print("BERT classifier model ready", flush=True)

    # Keep the import order used by exp1: data loading and model construction
    # must precede Opacus, SciPy, and bert_qnli.privacy imports.
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    from bert_qnli import privacy as privacy_module
    from bert_qnli.privacy import cleanup_private_hooks, make_private_training

    original_optimizer_class = privacy_module.DPAdamBC
    privacy_module.DPAdamBC = DiagnosticDPAdamBC
    try:
        print("initializing Opacus private training...", flush=True)
        private = make_private_training(model, data.train_loader, config)
    finally:
        privacy_module.DPAdamBC = original_optimizer_class
    print(
        f"Opacus private training ready: noise_multiplier={private.noise_multiplier}",
        flush=True,
    )

    private_model = private.model
    private_optimizer = private.optimizer
    underlying = private_optimizer.original_optimizer
    if not isinstance(underlying, DiagnosticDPAdamBC):
        raise RuntimeError("Opacus did not construct DiagnosticDPAdamBC")
    private_model.train()

    global_step = 0
    stop_training = False
    validation_rows = 0
    try:
        for epoch in range(1, config.training.epochs + 1):
            private_model.train()
            with BatchMemoryManager(
                data_loader=private.data_loader,
                max_physical_batch_size=config.data.max_physical_batch_size,
                optimizer=private_optimizer,
            ) as memory_safe_loader:
                for batch in memory_safe_loader:
                    if not batch["labels"].numel():
                        # Empty Poisson batches consume no optimizer diagnostic row.
                        private_optimizer.zero_grad()
                        private_optimizer.step()
                        private_optimizer.zero_grad()
                        continue

                    batch = _move_batch(batch, device)
                    labels = batch.pop("labels")
                    private_optimizer.zero_grad()
                    outputs = private_model(**batch)
                    loss = private.criterion(_extract_logits(outputs), labels)
                    loss.backward()
                    private_optimizer.step()
                    private_optimizer.zero_grad()

                    if not is_real_logical_step(private_optimizer):
                        continue
                    global_step += 1
                    record = underlying.last_diagnostics
                    if record is None:
                        raise RuntimeError(
                            "no diagnostic row was produced for a non-empty logical step"
                        )
                    row = dict(record)
                    row["global_step"] = global_step
                    diagnostics_writer.append(row)

                    if global_step in eval_steps:
                        metrics = evaluate_model(private_model, data.eval_loader, device)
                        validation_writer.append(
                            {
                                "global_step": global_step,
                                "val_loss": metrics["loss"],
                                "val_accuracy": metrics["accuracy"],
                            }
                        )
                        validation_rows += 1
                        print(
                            json.dumps(
                                {
                                    "global_step": global_step,
                                    "val_loss": metrics["loss"],
                                    "val_accuracy": metrics["accuracy"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

                    if (
                        config.training.max_steps is not None
                        and global_step >= config.training.max_steps
                    ):
                        stop_training = True
                        break
            if stop_training:
                break
    finally:
        cleanup_private_hooks(private.hooks)

    summary: dict[str, Any] = {
        "algorithm": config.algorithm,
        "config": config.to_dict(),
        "device": str(device),
        "train_size": data.train_size,
        "eval_size": data.eval_size,
        "global_step": global_step,
        "validation_rows": validation_rows,
        "noise_multiplier": private.noise_multiplier,
        "expected_batch_size": private.expected_batch_size,
        "phi": private.phi,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None, help="optional device override, e.g. cpu")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--eval-steps",
        default=None,
        help="comma-separated logical steps; default: 410,820,1230",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    return run(
        config_path.resolve(),
        output_dir=output_dir,
        max_steps=args.max_steps,
        device_name=args.device,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        eval_steps=_parse_eval_steps(args.eval_steps),
    )


if __name__ == "__main__":
    raise SystemExit(main())
