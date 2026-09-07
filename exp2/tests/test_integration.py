from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from exp2.csv_writer import DIAGNOSTIC_FIELDS, QUANTILE_FIELDS, VALIDATION_FIELDS
from exp2.run_exp2 import run


@pytest.mark.integration
def test_real_qnli_bert_bmm_split_writes_one_row_per_logical_step(tmp_path: Path):
    """Exercise real QNLI/BERT/Ghost/BMM with 64-example physical batches."""

    output = tmp_path / "bmm_split"
    physical_counts: list[int] = []
    config = Path(__file__).resolve().parents[1] / "configs/bc_g3e-9_lr3e-3.yaml"
    status = run(
        config,
        output_dir=output,
        device_name="cpu",
        max_steps=3,
        max_train_samples=1024,
        max_eval_samples=256,
        max_physical_batch_size=64,
        quantile_every_steps=20,
        physical_batch_counter=physical_counts,
    )
    assert status == 0
    assert physical_counts and physical_counts[0] > 3

    with (output / "bc_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        diagnostic_rows = list(csv.DictReader(stream))
    assert len(diagnostic_rows) == 3
    assert [int(row["global_step"]) for row in diagnostic_rows] == [1, 2, 3]
    assert all(tuple(row) == DIAGNOSTIC_FIELDS for row in diagnostic_rows)
    assert all(
        math.isfinite(float(value))
        for row in diagnostic_rows
        for value in row.values()
    )
    assert all(
        float(row["active_fraction"]) + float(row["clamp_fraction"]) == 1.0
        for row in diagnostic_rows
    )

    with (output / "q_quantiles.csv").open(newline="", encoding="utf-8") as stream:
        quantile_rows = list(csv.DictReader(stream))
    assert len(quantile_rows) == 1
    assert tuple(quantile_rows[0]) == QUANTILE_FIELDS
    assert int(quantile_rows[0]["global_step"]) == 3
    assert all(math.isfinite(float(value)) for value in quantile_rows[0].values())

    with (output / "validation_metrics.csv").open(newline="", encoding="utf-8") as stream:
        validation_rows = list(csv.DictReader(stream))
    assert len(validation_rows) == 1
    assert tuple(validation_rows[0]) == VALIDATION_FIELDS
    assert int(validation_rows[0]["global_step"]) == 3
    assert float(validation_rows[0]["epsilon_spent"]) < 7.0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["global_step"] == 3
    assert summary["epochs_completed"] == 0
    assert summary["epsilon_spent"] == pytest.approx(
        float(validation_rows[0]["epsilon_spent"])
    )
    assert summary["noise_multiplier"] > 0
    assert summary["expected_batch_size"] == 256
    assert summary["phi"] > 0
