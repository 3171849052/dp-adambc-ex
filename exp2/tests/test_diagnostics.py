from __future__ import annotations

import ast
import csv
from pathlib import Path

import pytest
import torch

from bert_qnli.config import load_config
from bert_qnli.optim import DPAdamBC
from exp2.csv_writer import (
    DIAGNOSTIC_FIELDS,
    DiagnosticsCSVWriter,
    QUANTILE_FIELDS,
    QuantileCSVWriter,
    VALIDATION_FIELDS,
    ValidationCSVWriter,
)
from exp2.diagnostic_optimizer import DiagnosticDPAdamBC
from exp2.run_exp2 import append_diagnostic_row, is_real_logical_step


def _options(**overrides):
    options = {
        "lr": 0.03,
        "betas": (0.5, 0.25),
        "gamma_prime": 0.04,
        "noise_multiplier": 1.5,
        "max_grad_norm": 2.0,
        "expected_batch_size": 2,
    }
    options.update(overrides)
    return options


def test_diagnostic_and_original_updates_match_every_step():
    diagnostic_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    original_parameter = torch.nn.Parameter(diagnostic_parameter.detach().clone())
    diagnostic = DiagnosticDPAdamBC([diagnostic_parameter], **_options())
    original = DPAdamBC([original_parameter], **_options())

    for step in range(1, 4):
        gradient = torch.tensor([0.2 * step, -0.7], dtype=torch.float64)
        diagnostic_parameter.grad = gradient.clone()
        diagnostic_parameter.summed_grad = torch.tensor([0.3, -0.2], dtype=torch.float64)
        original_parameter.grad = gradient.clone()
        diagnostic.step()
        original.step()
        torch.testing.assert_close(diagnostic_parameter, original_parameter, rtol=0, atol=0)


def test_clean_state_is_research_only_and_does_not_change_real_update():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    reference = torch.nn.Parameter(parameter.detach().clone())
    diagnostic = DiagnosticDPAdamBC([parameter], **_options())
    original = DPAdamBC([reference], **_options())

    for step in range(1, 3):
        gradient = torch.tensor([0.4 * step, -0.7], dtype=torch.float64)
        parameter.grad = gradient.clone()
        parameter.summed_grad = torch.tensor([100.0, -200.0], dtype=torch.float64)
        reference.grad = gradient.clone()
        diagnostic.step()
        original.step()
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    assert parameter in diagnostic._clean_v
    assert torch.isfinite(diagnostic._clean_v[parameter]).all()


def test_r_uses_unclamped_dp_formula_and_nrmse_is_before_clamp():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = DiagnosticDPAdamBC(
        [parameter],
        lr=0.1,
        betas=(0.0, 0.0),
        gamma_prime=10.0,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=2,
    )
    parameter.grad = torch.tensor([2.0, 4.0], dtype=torch.float64)
    parameter.summed_grad = torch.tensor([2.0, 6.0], dtype=torch.float64)
    optimizer.step()
    row = optimizer.last_diagnostics
    assert row is not None
    # v_hat = [4, 16], phi = 0.25, so r is [3.75, 15.75]. Using q before
    # subtracting phi or q after clamp would produce a different NRMSE.
    assert row["bc_clean_nrmse"] == pytest.approx((53.125 / 82.0) ** 0.5)
    assert row["clamp_fraction"] == pytest.approx(0.5)
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg_sq"],
        torch.tensor([4.0, 16.0], dtype=torch.float64),
    )


def test_fractions_and_complement_match_manual_tensor():
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    optimizer = DiagnosticDPAdamBC(
        [parameter],
        lr=0.1,
        betas=(0.0, 0.0),
        gamma_prime=1.0,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=2,
    )
    parameter.grad = torch.tensor([0.1, 0.5, 2.0, 3.0], dtype=torch.float64)
    parameter.summed_grad = torch.ones(4, dtype=torch.float64)
    optimizer.step()
    row = optimizer.last_diagnostics
    assert row is not None
    # phi=0.25, so r=[-.24, 0, 3.75, 8.75].
    assert row["negative_fraction"] == pytest.approx(2 / 4)
    assert row["clamp_fraction"] == pytest.approx(2 / 4)
    assert row["active_fraction"] == pytest.approx(2 / 4)
    assert row["active_fraction"] + row["clamp_fraction"] == pytest.approx(1.0)


def test_floor_update_metrics_and_active_energy_match_manual_tensor():
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = DiagnosticDPAdamBC(
        [parameter],
        lr=0.1,
        betas=(0.0, 0.0),
        gamma_prime=4.0,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=1,
    )
    parameter.grad = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    parameter.summed_grad = torch.ones(3, dtype=torch.float64)
    optimizer.step()
    row = optimizer.last_diagnostics
    assert row is not None
    u_bc = torch.tensor([0.5, 1.0, 1.0], dtype=torch.float64)
    u_floor = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    expected_cosine = torch.dot(u_bc, u_floor) / (u_bc.norm() * u_floor.norm())
    expected_ratio = u_bc.norm() / u_floor.norm()
    expected_energy = 1.0 / float(u_bc.square().sum())
    assert row["update_cosine_to_floor"] == pytest.approx(float(expected_cosine))
    assert row["update_norm_ratio_to_floor"] == pytest.approx(float(expected_ratio))
    assert row["active_update_energy_fraction"] == pytest.approx(expected_energy)


def test_global_quantiles_are_taken_over_all_parameter_elements():
    first = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    second = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = DiagnosticDPAdamBC(
        [first, second],
        lr=0.1,
        betas=(0.0, 0.0),
        gamma_prime=1.0,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=1,
    )
    first.grad = torch.tensor([1.0], dtype=torch.float64)
    first.summed_grad = torch.ones(1, dtype=torch.float64)
    second.grad = torch.tensor([2.0, 3.0], dtype=torch.float64)
    second.summed_grad = torch.ones(2, dtype=torch.float64)
    optimizer.quantile_requested = True
    optimizer.step()
    row = optimizer.last_diagnostics
    quantile_row = optimizer.last_quantile_diagnostics
    assert row is not None
    assert quantile_row is not None
    # q/gamma is the global [1, 4, 9], not a mean of per-tensor quantiles.
    assert tuple(quantile_row) == QUANTILE_FIELDS
    assert quantile_row["q_over_gamma_mean"] == pytest.approx(14 / 3)
    assert quantile_row["q_over_gamma_p50"] == pytest.approx(4.0)
    assert quantile_row["q_over_gamma_p90"] == pytest.approx(8.0)
    assert quantile_row["q_over_gamma_p99"] == pytest.approx(8.9)


def test_quantile_interval_is_sparse_but_final_step_can_be_requested():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = DiagnosticDPAdamBC(
        [parameter],
        betas=(0.0, 0.0),
        gamma_prime=1.0,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=1,
    )
    for step in range(1, 4):
        parameter.grad = torch.tensor([float(step), 2.0], dtype=torch.float64)
        parameter.summed_grad = torch.ones(2, dtype=torch.float64)
        optimizer.quantile_requested = step % 2 == 0
        optimizer.step()
        if step == 2:
            assert optimizer.last_quantile_diagnostics is not None
            assert optimizer.last_quantile_diagnostics["global_step"] == 2
        else:
            assert optimizer.last_quantile_diagnostics is None
    final = optimizer.current_quantile_diagnostics(global_step=3)
    assert final["global_step"] == 3


def test_empty_optimizer_call_has_no_diagnostic_row():
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = DiagnosticDPAdamBC([parameter], **_options())
    optimizer.step()
    assert optimizer.last_diagnostics is None
    assert optimizer.logical_diagnostic_steps == 0
    torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_csv_schemas_finite_values_and_validation_rows(tmp_path: Path):
    diagnostic_path = tmp_path / "bc_diagnostics.csv"
    diagnostics = DiagnosticsCSVWriter(diagnostic_path)
    diagnostics.append({field: 1.0 for field in DIAGNOSTIC_FIELDS})
    with diagnostic_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert tuple(rows[0]) == DIAGNOSTIC_FIELDS
    with pytest.raises(ValueError, match="NaN or Inf"):
        diagnostics.append({**{field: 1.0 for field in DIAGNOSTIC_FIELDS}, "phi": float("nan")})

    validation_path = tmp_path / "validation_metrics.csv"
    validation = ValidationCSVWriter(validation_path)
    validation.append(
        {
            "epoch": 1,
            "global_step": 410,
            "val_loss": 0.5,
            "val_accuracy": 0.75,
            "epsilon_spent": 2.0,
        }
    )
    with validation_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == VALIDATION_FIELDS

    quantile_path = tmp_path / "q_quantiles.csv"
    quantiles = QuantileCSVWriter(quantile_path)
    quantiles.append({field: 1.0 for field in QUANTILE_FIELDS})
    with quantile_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == QUANTILE_FIELDS


def test_csv_gate_writes_only_real_logical_steps(tmp_path: Path):
    class OptimizerSignal:
        def __init__(self, skipped: bool):
            self._is_last_step_skipped = skipped

    path = tmp_path / "logical_steps.csv"
    writer = DiagnosticsCSVWriter(path)
    template = {field: 1.0 for field in DIAGNOSTIC_FIELDS}
    written_steps = 0
    for skipped in (True, True, False, True, False):
        if is_real_logical_step(OptimizerSignal(skipped)):
            written_steps += 1
            row = dict(template)
            row["global_step"] = written_steps
            writer.append(row)

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert [int(float(row["global_step"])) for row in rows] == [1, 2]


def test_runner_rejects_optimizer_runner_step_mismatch(tmp_path: Path):
    writer = DiagnosticsCSVWriter(tmp_path / "diagnostics.csv")
    row = {field: 1.0 for field in DIAGNOSTIC_FIELDS}
    row["global_step"] = 1
    append_diagnostic_row(writer, row, global_step=1)
    with pytest.raises(RuntimeError, match=r"diagnostic step mismatch: 1 != 2"):
        append_diagnostic_row(writer, row, global_step=2)


def test_runner_keeps_heavy_imports_after_data_and_model_setup():
    runner = Path(__file__).resolve().parents[1] / "run_exp2.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    top_level_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    def modules(node):
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        return [node.module or ""]

    forbidden = ("opacus", "scipy", "bert_qnli.privacy")
    assert not any(
        any(module == item or module.startswith(item + ".") for item in forbidden)
        for node in top_level_imports
        for module in modules(node)
    )
    run_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    load_line = next(
        node.lineno for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_qnli"
    )
    build_line = next(
        node.lineno for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_model"
    )
    heavy_import_lines = [
        node.lineno for node in ast.walk(run_function)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(
            module == item or module.startswith(item + ".")
            for module in modules(node)
            for item in forbidden
        )
    ]
    assert heavy_import_lines
    assert min(heavy_import_lines) > max(load_line, build_line)


def test_every_optimizer_diagnostic_scalar_is_finite():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = DiagnosticDPAdamBC(
        [parameter],
        betas=(0.0, 0.0),
        gamma_prime=1.0,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=1,
    )
    parameter.grad = torch.tensor([1.0, -2.0], dtype=torch.float64)
    parameter.summed_grad = torch.tensor([0.5, -0.5], dtype=torch.float64)
    optimizer.step()
    assert optimizer.last_diagnostics is not None
    assert all(torch.isfinite(torch.tensor(value, dtype=torch.float64)) for value in optimizer.last_diagnostics.values())


def test_formal_configs_run_for_all_ten_epochs_without_step_cap():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    configs = sorted(config_dir.glob("bc_*.yaml"))
    assert len(configs) == 5
    for path in configs:
        config = load_config(path)
        assert config.training.epochs == 10
        assert config.training.max_steps is None
        assert config.seed == 0
        assert config.privacy.epsilon == pytest.approx(7.0)
