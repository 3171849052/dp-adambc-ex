#!/usr/bin/env python
"""Evaluate a saved QNLI checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dp_adam_iid.config import load_config  # noqa: E402
from dp_adam_iid.data import load_qnli  # noqa: E402
from dp_adam_iid.model import build_model  # noqa: E402
from dp_adam_iid.trainer import evaluate_model  # noqa: E402
from dp_adam_iid.utils import resolve_device, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    device = resolve_device(config.runtime.device)
    checkpoint_path = args.checkpoint or Path(config.output.checkpoint_dir) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    data = load_qnli(config)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(model.to(device), data.eval_loader, device)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "epoch": checkpoint.get("epoch"),
                "step": checkpoint.get("step"),
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
