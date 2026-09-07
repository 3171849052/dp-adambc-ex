"""CSV output for Experiment 1 diagnostics."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Mapping, Any

from .diagnostic_optimizer import DIAGNOSTIC_FIELDS


class DiagnosticsCSVWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=DIAGNOSTIC_FIELDS).writeheader()
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, record: Mapping[str, Any]) -> None:
        if set(record) != set(DIAGNOSTIC_FIELDS):
            raise ValueError("diagnostic record fields do not match the P0 schema")
        if not all(math.isfinite(float(record[field])) for field in DIAGNOSTIC_FIELDS):
            raise ValueError("diagnostic record contains NaN or Inf")
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=DIAGNOSTIC_FIELDS).writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
