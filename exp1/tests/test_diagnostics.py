from __future__ import annotations

import ast
import csv
from pathlib import Path

import pytest
import torch

from bert_qnli.optim import DPAdamBC, FPCDPAdam
from exp1.csv_writer import DiagnosticsCSVWriter
from exp1.diagnostic_optimizer import DIAGNOSTIC_FIELDS, DiagnosticFPCDPAdam
from exp1.run_exp1 import is_real_logical_step


def _options(**overrides):
    result = {
        "lr": 0.03,
        "betas": (0.5, 0.25),
        "gamma_prime": 0.04,
        "noise_multiplier": 1.5,
        "max_grad_norm": 2.0,
        "expected_batch_size": 2,
    }
    result.update(overrides)
    return result


def test_lambda_one_observation_matches_dp_adambc_correction():
    p1 = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    p2 = torch.nn.Parameter(p1.detach().clone())
    fpc = DiagnosticFPCDPAdam([p1], fpc_lambda=1.0, **_options())
    bc = DPAdamBC([p2], **_options())
    for gradient in (torch.tensor([1.0, 0.5], dtype=torch.float64), torch.tensor([-0.5, 1.5], dtype=torch.float64)):
        p1.grad = gradient.clone()
        p2.grad = gradient.clone()
        fpc.step()
        bc.step()
        torch.testing.assert_close(p1, p2)
        expected_corrected = bc.state[p2]["exp_avg_sq"] - fpc.phi * (1.0 - 0.25 ** fpc.state[p1]["step"])
        torch.testing.assert_close(fpc.state[p1]["exp_avg_sq"], expected_corrected)


def test_shadow_and_clean_diagnostics_do_not_modify_parameters():
    p_diag = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    p_plain = torch.nn.Parameter(p_diag.detach().clone())
    opt_diag = DiagnosticFPCDPAdam([p_diag], fpc_lambda=0.5, **_options())
    opt_plain = FPCDPAdam([p_plain], fpc_lambda=0.5, **_options())
    for step in range(1, 3):
        gradient = torch.tensor([0.4 * step, -0.7], dtype=torch.float64)
        summed = torch.tensor([0.3, -0.2], dtype=torch.float64)
        p_diag.grad = gradient.clone()
        p_diag.summed_grad = summed.clone()
        p_plain.grad = gradient.clone()
        opt_diag.step()
        opt_plain.step()
        torch.testing.assert_close(p_diag, p_plain)
    assert torch.isfinite(opt_diag._clean_v[p_diag]).all()
    assert torch.isfinite(opt_diag._shadow_v[p_diag]).all()


def test_empty_optimizer_call_has_no_diagnostic_row():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    opt = DiagnosticFPCDPAdam([p], fpc_lambda=0.5, **_options())
    opt.step()
    assert opt.last_diagnostics is None
    assert opt.logical_diagnostic_steps == 0
    torch.testing.assert_close(p, torch.ones_like(p))


def test_diagnostics_match_manual_global_reductions():
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = DiagnosticFPCDPAdam(
        [p], lr=0.1, betas=(0.0, 0.0), gamma_prime=0.25,
        fpc_lambda=0.5, noise_multiplier=0.0, max_grad_norm=1.0,
        expected_batch_size=2,
    )
    p.grad = torch.tensor([2.0, 4.0], dtype=torch.float64)
    p.summed_grad = torch.tensor([2.0, 6.0], dtype=torch.float64)
    opt.step()
    row = opt.last_diagnostics
    assert row is not None
    # p_1=0, x=[1,3], y=[2,4], phi=0, and beta2=0.
    assert row["predictor_x_gap_mse"] == pytest.approx(5.0)
    assert row["x2_mean"] == pytest.approx(5.0)
    assert row["predictor_gap_over_x2"] == pytest.approx(1.0)
    assert row["predictor_gap_over_phi"] == pytest.approx(5.0 / 1.0e-30)
    assert row["fpc_expected_bias_ratio"] == pytest.approx(-0.5)
    assert row["v_clean_nrmse_fpc"] == pytest.approx((2.0**0.5) / (82.0**0.5))
    assert row["v_clean_nrmse_bc"] == pytest.approx((58.0**0.5) / (82.0**0.5))
    assert row["v_nrmse_ratio_fpc_over_bc"] == pytest.approx((2.0 / 58.0) ** 0.5)
    assert row["clamp_fraction_fpc"] == pytest.approx(0.0)
    assert row["clamp_fraction_bc"] == pytest.approx(0.0)
    assert row["update_cosine_fpc_bc"] == pytest.approx(1.0)
    assert row["update_norm_ratio_fpc_over_bc"] == pytest.approx(2.0**0.5)
    assert all(torch.isfinite(torch.tensor(value)) for value in row.values())


def test_global_reductions_span_parameter_tensors_and_clamp_before_updates():
    p1 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    p2 = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = DiagnosticFPCDPAdam(
        [p1, p2], lr=0.1, betas=(0.0, 0.0), gamma_prime=10.0,
        fpc_lambda=0.5, noise_multiplier=0.0, max_grad_norm=1.0,
        expected_batch_size=1,
    )
    p1.grad = torch.tensor([2.0], dtype=torch.float64)
    p2.grad = torch.tensor([4.0, 6.0], dtype=torch.float64)
    p1.summed_grad = torch.tensor([1.0], dtype=torch.float64)
    p2.summed_grad = torch.tensor([3.0, 5.0], dtype=torch.float64)
    opt.step()
    row = opt.last_diagnostics
    assert row is not None

    # The three elements are reduced together: gap^2=x^2=1+9+25=35.
    assert row["predictor_x_gap_mse"] == pytest.approx(35.0 / 3.0)
    assert row["x2_mean"] == pytest.approx(35.0 / 3.0)
    assert row["predictor_gap_over_x2"] == pytest.approx(1.0)
    assert row["fpc_expected_bias_ratio"] == pytest.approx(-0.5)
    # Before clamp, v_fpc=[2,8,18] and v_bc=[4,16,36].
    assert row["v_clean_nrmse_fpc"] == pytest.approx((51.0 / 707.0) ** 0.5)
    assert row["v_clean_nrmse_bc"] == pytest.approx((179.0 / 707.0) ** 0.5)
    assert row["clamp_fraction_fpc"] == pytest.approx(2.0 / 3.0)
    assert row["clamp_fraction_bc"] == pytest.approx(1.0 / 3.0)
    assert row["clamp_fraction_delta"] == pytest.approx(1.0 / 3.0)
    expected_dot = 0.4 + 4.0 / (10.0**0.5) + 6.0 / (18.0**0.5)
    assert row["update_cosine_fpc_bc"] == pytest.approx(
        expected_dot / (4.0 * 2.4) ** 0.5
    )
    assert row["update_norm_ratio_fpc_over_bc"] == pytest.approx((5.0 / 3.0) ** 0.5)


def test_csv_writer_schema_and_one_row_per_real_step(tmp_path: Path):
    path = tmp_path / "p0.csv"
    writer = DiagnosticsCSVWriter(path)
    row = {field: 1.0 for field in DIAGNOSTIC_FIELDS}
    writer.append(row)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert tuple(rows[0]) == DIAGNOSTIC_FIELDS

    with pytest.raises(ValueError, match="NaN or Inf"):
        writer.append({**row, "phi": float("nan")})


def test_csv_gate_writes_only_real_logical_steps(tmp_path: Path):
    class OptimizerSignal:
        def __init__(self, skipped: bool):
            self._is_last_step_skipped = skipped

    path = tmp_path / "logical_steps.csv"
    writer = DiagnosticsCSVWriter(path)
    row = {field: 1.0 for field in DIAGNOSTIC_FIELDS}
    written_steps = 0
    for skipped in (True, True, False, True, False):
        signal = OptimizerSignal(skipped)
        if is_real_logical_step(signal):
            written_steps += 1
            row["global_step"] = written_steps
            writer.append(row)

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert [int(float(item["global_step"])) for item in rows] == [1, 2]


def test_runner_defers_opacus_privacy_imports_until_after_data_and_model():
    runner = Path(__file__).resolve().parents[1] / "run_exp1.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    def imported_module(node):
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        modules = [node.module or ""]
        if node.module == "bert_qnli":
            modules.extend(
                f"bert_qnli.{alias.name}"
                for alias in node.names
                if alias.name == "privacy"
            )
        return modules

    forbidden = ("opacus", "scipy", "bert_qnli.privacy")
    assert not any(
        any(module == item or module.startswith(item + ".") for item in forbidden)
        for node in top_level_imports
        for module in imported_module(node)
    )

    run_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    load_line = next(
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_qnli"
    )
    build_line = next(
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_model"
    )
    heavy_import_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(
            module == item or module.startswith(item + ".")
            for module in imported_module(node)
            for item in forbidden
        )
    ]
    assert heavy_import_lines
    assert min(heavy_import_lines) > max(load_line, build_line)
