"""Generic per-run directory, metadata, logging, and metrics helpers."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, TextIO

import yaml

from .config import Config


METRICS_FIELDS = (
    "phase",
    "epoch",
    "step",
    "loss",
    "accuracy",
    "epsilon",
    "noise_multiplier",
)


@dataclass(frozen=True)
class RunPaths:
    """Files belonging to one experiment run."""

    directory: Path
    config: Path
    resolved_config: Path
    metrics: Path
    summary: Path
    train_log: Path


def _format_number(value: float | int) -> str:
    """Format numeric configuration values for stable, readable path names."""

    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("run name values must be finite")
    if number == number.to_integral_value():
        return str(number.quantize(Decimal("1")))

    # Scientific notation makes the small privacy and learning-rate values
    # unambiguous while keeping ordinary decimal values readable.
    if abs(number) < Decimal("0.001"):
        text = format(number.normalize(), "E").replace("E", "e")
        text = re.sub(r"e([+-])0+(\d+)$", r"e\1\2", text)
        text = text.replace("e+", "e")
        return text
    return format(number.normalize(), "f")


def format_run_name(config: Config, timestamp: datetime | None = None) -> str:
    """Return the documented run name for a parsed configuration."""

    stamp = timestamp or datetime.now()
    return (
        f"{stamp:%Y%m%d-%H%M%S}_{config.algorithm}"
        f"_eps{_format_number(config.privacy.epsilon)}"
        f"_d{_format_number(config.privacy.delta)}"
        f"_ep{_format_number(config.training.epochs)}"
        f"_lb{_format_number(config.data.logical_batch_size)}"
        f"_lr{_format_number(config.training.learning_rate)}"
        f"_C{_format_number(config.privacy.max_grad_norm)}"
        f"_s{_format_number(config.seed)}"
    )


def _run_paths(directory: Path) -> RunPaths:
    return RunPaths(
        directory=directory,
        config=directory / "config.yaml",
        resolved_config=directory / "resolved_config.yaml",
        metrics=directory / "metrics.csv",
        summary=directory / "summary.json",
        train_log=directory / "train.log",
    )


def create_run_directory(
    config: Config,
    *,
    root: str | Path | None = None,
    now: datetime | None = None,
) -> RunPaths:
    """Create a collision-free run directory using second precision timestamps."""

    output_root = Path(root if root is not None else config.output.root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = now or datetime.now()
    for collision in range(10_000):
        candidate_time = timestamp + timedelta(seconds=collision)
        directory = output_root / format_run_name(config, candidate_time)
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        paths = _run_paths(directory)
        paths.train_log.touch()
        return paths
    raise RuntimeError("could not allocate a unique run directory")


def write_run_metadata(
    paths: RunPaths,
    *,
    source_yaml: str,
    resolved_config: Mapping[str, Any],
    summary: Mapping[str, Any] | None = None,
) -> None:
    """Write the source snapshot, resolved settings, and optional final summary."""

    paths.directory.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(source_yaml, encoding="utf-8")
    paths.resolved_config.write_text(
        yaml.safe_dump(dict(resolved_config), sort_keys=True), encoding="utf-8"
    )
    paths.train_log.touch(exist_ok=True)
    if summary is not None:
        paths.summary.write_text(
            json.dumps(dict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class MetricsCSVWriter:
    """Append flushed training and validation metric records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or not self.path.stat().st_size:
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=METRICS_FIELDS).writeheader()
                stream.flush()
                os.fsync(stream.fileno())

    def append(self, record: Mapping[str, Any]) -> None:
        if set(record) != set(METRICS_FIELDS):
            raise ValueError("metrics record must contain exactly the CSV fields")
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=METRICS_FIELDS).writerow(record)
            stream.flush()
            os.fsync(stream.fileno())


class _TeeStream:
    def __init__(self, terminal: TextIO, log: TextIO):
        self._terminal = terminal
        self._log = log

    def write(self, text: str) -> int:
        self._terminal.write(text)
        self._log.write(text)
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._terminal, name)


@contextmanager
def tee_output(path: str | Path) -> Iterator[None]:
    """Mirror stdout and stderr to the terminal and the current run log."""

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_stdout, log)  # type: ignore[assignment]
        sys.stderr = _TeeStream(old_stderr, log)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr


__all__ = [
    "METRICS_FIELDS",
    "MetricsCSVWriter",
    "RunPaths",
    "create_run_directory",
    "format_run_name",
    "tee_output",
    "write_run_metadata",
]
