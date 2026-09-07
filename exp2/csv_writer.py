"""CSV writers for Experiment 2's flushed scalar artifacts."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .diagnostic_optimizer import DIAGNOSTIC_FIELDS, QUANTILE_FIELDS


VALIDATION_FIELDS = (
    "epoch",
    "global_step",
    "val_loss",
    "val_accuracy",
    "epsilon_spent",
)


class _CSVWriter:
    fields: tuple[str, ...]
    description: str

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(
                stream, fieldnames=self.fields, lineterminator="\n"
            ).writeheader()
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, record: Mapping[str, Any]) -> None:
        if tuple(record) != self.fields or set(record) != set(self.fields):
            raise ValueError(f"{self.description} record fields do not match schema")
        try:
            finite = all(math.isfinite(float(record[field])) for field in self.fields)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{self.description} record is not numeric") from error
        if not finite:
            raise ValueError(f"{self.description} record contains NaN or Inf")
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(
                stream, fieldnames=self.fields, lineterminator="\n"
            ).writerow(record)
            stream.flush()
            os.fsync(stream.fileno())


class DiagnosticsCSVWriter(_CSVWriter):
    fields = DIAGNOSTIC_FIELDS
    description = "BC diagnostic"


class ValidationCSVWriter(_CSVWriter):
    fields = VALIDATION_FIELDS
    description = "validation"


class QuantileCSVWriter(_CSVWriter):
    fields = QUANTILE_FIELDS
    description = "quantile"


__all__ = [
    "DIAGNOSTIC_FIELDS",
    "DiagnosticsCSVWriter",
    "QUANTILE_FIELDS",
    "QuantileCSVWriter",
    "VALIDATION_FIELDS",
    "ValidationCSVWriter",
]
