#!/usr/bin/env python
"""Run Experiment 1 without changing the repository's training modules."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bert_qnli import privacy as privacy_module  # noqa: E402
from bert_qnli.config import load_config  # noqa: E402
from bert_qnli.data import load_qnli  # noqa: E402
from bert_qnli.model import build_model  # noqa: E402
from bert_qnli.privacy import cleanup_private_hooks, make_private_training  # noqa: E402
from bert_qnli.utils import resolve_device, set_seed  # noqa: E402

from exp1.csv_writer import DiagnosticsCSVWriter  # noqa: E402
from exp1.diagnostic_optimizer import DiagnosticFPCDPAdam  # noqa: E402
from exp1.diagnostic_optimizer import DIAGNOSTIC_FIELDS  # noqa: E402


def _move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def is_real_logical_step(dp_optimizer) -> bool:
    """Return whether the preceding DPOptimizer call updated the base Adam.

    BatchMemoryManager sets ``_is_last_step_skipped`` on intermediate physical
    batches. The flag is checked only after ``step()`` and is the single gate
    for advancing the CSV/global-step counter.
    """

    return not bool(getattr(dp_optimizer, "_is_last_step_skipped", False))


def run_synthetic_smoke(output: Path, steps: int = 3) -> int:
    """Exercise the optimizer and logical-step CSV contract without downloads."""

    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    optimizer = DiagnosticFPCDPAdam(
        [parameter],
        lr=0.01,
        betas=(0.9, 0.999),
        gamma_prime=3.0e-9,
        fpc_lambda=0.5,
        fpc_mode="current",
        noise_multiplier=1.0,
        max_grad_norm=0.1,
        expected_batch_size=256,
    )
    writer = DiagnosticsCSVWriter(output)
    for global_step in range(1, steps + 1):
        parameter.grad = torch.tensor(
            [0.01 * global_step, -0.02, 0.03, -0.01], dtype=torch.float64
        )
        parameter.summed_grad = torch.tensor(
            [0.5, -0.25, 0.75, -0.5], dtype=torch.float64
        )
        optimizer.step()
        record = dict(optimizer.last_diagnostics or {})
        record["global_step"] = global_step
        if tuple(record) != DIAGNOSTIC_FIELDS:
            raise RuntimeError("synthetic smoke produced an invalid diagnostic row")
        writer.append(record)
    return 0


def run(
    config_path: Path,
    *,
    output: Path,
    max_steps: int | None = None,
    device_name: str | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> int:
    config = load_config(config_path)
    if max_steps is not None:
        config.training.max_steps = max_steps
    if device_name is not None:
        config.runtime.device = device_name
    if max_train_samples is not None:
        config.data.max_train_samples = max_train_samples
    if max_eval_samples is not None:
        config.data.max_eval_samples = max_eval_samples
    if config.model.name != "bert-base-cased" or config.data.dataset_config != "qnli":
        raise ValueError("Experiment 1 requires bert-base-cased + QNLI")
    set_seed(config.seed)
    device = resolve_device(config.runtime.device)
    print("loading QNLI data...", flush=True)
    data = load_qnli(config)
    print(
        f"QNLI data ready: train_examples={data.train_size} "
        f"eval_examples={data.eval_size}",
        flush=True,
    )
    print("loading BERT model...", flush=True)
    model = build_model(config, data.tokenizer)
    model.to(device)
    print("initializing Opacus private training...", flush=True)
    # make_private_training resolves FPCDPAdam from this module global. The
    # subclass is API-compatible and leaves the actual FPC update unchanged.
    original_optimizer = privacy_module.FPCDPAdam
    privacy_module.FPCDPAdam = DiagnosticFPCDPAdam
    try:
        private = make_private_training(model, data.train_loader, config)
    finally:
        privacy_module.FPCDPAdam = original_optimizer
    print(
        f"Opacus private training ready: noise_multiplier={private.noise_multiplier}",
        flush=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = DiagnosticsCSVWriter(output)
    private_model = private.model
    private_model.train()
    underlying = private.optimizer.original_optimizer
    try:
        global_step = 0
        for _epoch in range(1, config.training.epochs + 1):
            with BatchMemoryManager(
                data_loader=private.data_loader,
                max_physical_batch_size=config.data.max_physical_batch_size,
                optimizer=private.optimizer,
            ) as memory_safe_loader:
                for batch in memory_safe_loader:
                    if not batch["labels"].numel():
                        private.optimizer.zero_grad()
                        private.optimizer.step()
                        private.optimizer.zero_grad()
                        continue
                    batch = _move(batch, device)
                    labels = batch.pop("labels")
                    private.optimizer.zero_grad()
                    outputs = private_model(**batch)
                    loss = private.criterion(outputs.logits, labels)
                    loss.backward()
                    private.optimizer.step()
                    private.optimizer.zero_grad()
                    if is_real_logical_step(private.optimizer):
                        global_step += 1
                        record = underlying.last_diagnostics
                        if record is None:
                            raise RuntimeError("diagnostics unavailable at logical step")
                        record = dict(record)
                        record["global_step"] = global_step
                        writer.append(record)
                        if config.training.max_steps is not None and global_step >= config.training.max_steps:
                            return 0
    finally:
        cleanup_private_hooks(private.hooks)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "exp1/config_p0.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "exp1/results/p0_diagnostics.csv")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None, help="optional device override, e.g. cpu")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return run_synthetic_smoke(args.output.resolve(), steps=args.max_steps or 3)
    return run(
        args.config.resolve(),
        output=args.output.resolve(),
        max_steps=args.max_steps,
        device_name=args.device,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    raise SystemExit(main())
