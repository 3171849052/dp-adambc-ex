#!/usr/bin/env python
"""Train RoBERTa on QNLI with Opacus GDP + Ghost Clipping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dp_adam_iid.config import load_config  # noqa: E402
from dp_adam_iid.data import load_qnli  # noqa: E402
from dp_adam_iid.model import build_model  # noqa: E402
from dp_adam_iid.trainer import train_model  # noqa: E402
from dp_adam_iid.utils import resolve_device, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    device = resolve_device(config.runtime.device)
    data = load_qnli(config)
    model = build_model(config)
    print(
        f"device={device} train_examples={data.train_size} "
        f"eval_examples={data.eval_size} logical_batch_size={config.data.logical_batch_size} "
        f"max_physical_batch_size={config.data.max_physical_batch_size}",
        flush=True,
    )
    result = train_model(config, model, data.train_loader, data.eval_loader, device)
    print(
        f"saved best={result.best_checkpoint} final={result.final_checkpoint} "
        f"step={result.global_step} epsilon={result.epsilon:.6f} "
        f"noise_multiplier={result.noise_multiplier:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
