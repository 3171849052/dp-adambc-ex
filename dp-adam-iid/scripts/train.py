#!/usr/bin/env python
"""Create a run directory and train RoBERTa on QNLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(os.environ.get("DP_ADAM_IID_REPOSITORY_ROOT", PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from dp_adam_iid.config import Config, load_config  # noqa: E402
from dp_adam_iid.data import load_qnli  # noqa: E402
from dp_adam_iid.model import build_model  # noqa: E402
from dp_adam_iid.run_logging import (  # noqa: E402
    MetricsCSVWriter,
    RunPaths,
    create_run_directory,
    tee_output,
    write_run_metadata,
)
from dp_adam_iid.trainer import TrainingResult, train_model  # noqa: E402
from dp_adam_iid.utils import resolve_device, set_seed  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "config/qnli_roberta_base.yaml"


def _config_path(value: str | None) -> Path:
    path = DEFAULT_CONFIG if value is None else Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def _output_root(config: Config) -> Path:
    root = Path(config.output.root)
    return root if root.is_absolute() else REPOSITORY_ROOT / root


def _resolved_config(
    config: Config,
    paths: RunPaths,
    device: torch.device,
    *,
    train_size: int | None = None,
    eval_size: int | None = None,
    result: TrainingResult | None = None,
) -> dict[str, Any]:
    resolved = config.to_dict()
    resolved["runtime"]["actual_device"] = str(device)
    resolved["data"]["train_size"] = train_size
    resolved["data"]["eval_size"] = eval_size
    resolved["privacy"]["noise_multiplier"] = (
        None if result is None else result.noise_multiplier
    )
    resolved["privacy"]["epsilon_spent"] = None if result is None else result.epsilon
    resolved["privacy"]["sample_rate"] = None if result is None else result.sample_rate
    resolved["run"] = {"directory": str(paths.directory.resolve())}
    return resolved


def _summary(
    config: Config,
    paths: RunPaths,
    device: torch.device,
    train_size: int,
    eval_size: int,
    result: TrainingResult,
) -> dict[str, Any]:
    return {
        "algorithm": config.algorithm,
        "run_directory": str(paths.directory.resolve()),
        "seed": config.seed,
        "device": str(device),
        "train_size": train_size,
        "eval_size": eval_size,
        "global_step": result.global_step,
        "best_accuracy": result.best_accuracy,
        "final_metrics": result.final_metrics,
        "epsilon": result.epsilon,
        "noise_multiplier": result.noise_multiplier,
        "sample_rate": result.sample_rate,
        "config": config.to_dict(),
    }


def run_experiment(config_file: str | Path) -> int:
    config_path = Path(config_file)
    source_yaml = config_path.read_text(encoding="utf-8")
    config = load_config(config_path)
    device = resolve_device(config.runtime.device)
    paths = create_run_directory(config, root=_output_root(config))
    write_run_metadata(
        paths,
        source_yaml=source_yaml,
        resolved_config=_resolved_config(config, paths, device),
    )
    metrics_writer = MetricsCSVWriter(paths.metrics)

    with tee_output(paths.train_log):
        try:
            print(f"run_directory={paths.directory.resolve()}", flush=True)
            set_seed(config.seed)
            data = load_qnli(config)
            model = build_model(config)
            print(
                f"device={device} train_examples={data.train_size} "
                f"eval_examples={data.eval_size} "
                f"logical_batch_size={config.data.logical_batch_size} "
                f"max_physical_batch_size={config.data.max_physical_batch_size}",
                flush=True,
            )
            write_run_metadata(
                paths,
                source_yaml=source_yaml,
                resolved_config=_resolved_config(
                    config,
                    paths,
                    device,
                    train_size=data.train_size,
                    eval_size=data.eval_size,
                ),
            )
            result = train_model(
                config,
                model,
                data.train_loader,
                data.eval_loader,
                device,
                metrics_writer=metrics_writer,
            )
            resolved = _resolved_config(
                config,
                paths,
                device,
                train_size=data.train_size,
                eval_size=data.eval_size,
                result=result,
            )
            write_run_metadata(
                paths,
                source_yaml=source_yaml,
                resolved_config=resolved,
                summary=_summary(
                    config, paths, device, data.train_size, data.eval_size, result
                ),
            )
            print(
                json.dumps(
                    {
                        "global_step": result.global_step,
                        "final_metrics": result.final_metrics,
                        "epsilon": result.epsilon,
                        "noise_multiplier": result.noise_multiplier,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        except Exception:
            traceback.print_exc()
            return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--config", dest="config_option", type=Path)
    args = parser.parse_args()
    if args.config is not None and args.config_option is not None:
        parser.error("provide the config path either positionally or with --config")
    return run_experiment(_config_path(args.config_option or args.config))


if __name__ == "__main__":
    raise SystemExit(main())
