import os
from pathlib import Path
from types import SimpleNamespace
import runpy
import subprocess
import sys

import yaml

import torch

from roberta_qnli.config import load_config
from roberta_qnli.trainer import TrainingResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prepare_run_creates_metadata_and_no_checkpoint(tmp_path: Path):
    source = yaml.safe_load(
        (PROJECT_ROOT / "config/qnli_roberta_base_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["runtime"]["gpu"] = 2
    source["output"]["root"] = str(tmp_path / "outputs")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train.py"),
            "--config",
            str(config_path),
            "--prepare-run",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    run_directory = Path(result.stdout.strip())
    assert run_directory.parent == tmp_path / "outputs"
    assert {
        "config.yaml",
        "resolved_config.yaml",
        "metrics.csv",
        "train.log",
    } == {path.name for path in run_directory.iterdir()}
    assert not list(run_directory.rglob("*.pt"))
    assert not list(run_directory.rglob("*.pth"))
    resolved = yaml.safe_load(
        (run_directory / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert resolved["runtime"]["gpu"] == 2
    assert resolved["runtime"]["physical_gpu_index"] == 2
    assert "cuda_visible_devices" in resolved["runtime"]
    assert "actual_device" in resolved["runtime"]
    assert "gpu_name" in resolved["runtime"]

    print_result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train.py"),
            "--config",
            str(config_path),
            "--print-gpu",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert print_result.stdout == "2\n"


def test_prepare_fpc_run_creates_diagnostics_file(tmp_path: Path):
    source = yaml.safe_load(
        (PROJECT_ROOT / "config/qnli_roberta_base_fpcdpadam.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["runtime"]["device"] = "cpu"
    source["output"]["root"] = str(tmp_path / "outputs")
    config_path = tmp_path / "fpc.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train.py"),
            "--config",
            str(config_path),
            "--prepare-run",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    diagnostics = Path(result.stdout.strip()) / "fpc_diagnostics.csv"
    assert diagnostics.read_text(encoding="utf-8") == (
        "global_step,predictor_x_gap_mse\n"
    )


def test_launcher_uses_python_gpu_selection_and_visible_device_mapping():
    launcher = (PROJECT_ROOT / "run.sh").read_text(encoding="utf-8")
    assert "--print-gpu" in launcher
    assert 'CUDA_VISIBLE_DEVICES="$GPU" python -u' in launcher


def test_launcher_exports_stable_tokenizer_environment_in_tmux_command():
    launcher = (PROJECT_ROOT / "run.sh").read_text(encoding="utf-8")

    assert "export TOKENIZERS_PARALLELISM=false" in launcher
    assert "export RAYON_NUM_THREADS=1" in launcher
    assert "export PYTHONFAULTHANDLER=1" in launcher
    assert "conda activate adamex && export TOKENIZERS_PARALLELISM=false" in launcher
    assert "export OMP_NUM_THREADS=1" not in launcher
    assert "export MKL_NUM_THREADS=1" not in launcher


def test_runner_emits_startup_phase_logs_in_order():
    runner = (PROJECT_ROOT / "scripts/train.py").read_text(encoding="utf-8")

    markers = [
        "loading QNLI data...",
        "QNLI data ready:",
        "loading RoBERTa MLM model...",
        "RoBERTa MLM model ready",
        "starting private training initialization...",
    ]
    positions = [runner.index(marker) for marker in markers]
    assert positions == sorted(positions)

    trainer = (PROJECT_ROOT / "src/roberta_qnli/trainer.py").read_text(
        encoding="utf-8"
    )
    assert 'print("initializing Opacus private training...", flush=True)' in trainer
    assert "Opacus private training ready: " in trainer


def test_resolved_config_and_summary_include_dpadambc_noise_metadata(tmp_path):
    config = load_config(PROJECT_ROOT / "config/qnli_roberta_base_dpadambc.yaml")
    paths = SimpleNamespace(directory=tmp_path)
    train_script = runpy.run_path(str(PROJECT_ROOT / "scripts/train.py"))
    result = TrainingResult(
        global_step=1,
        best_accuracy=0.5,
        final_metrics={"loss": 1.0, "accuracy": 0.5},
        noise_multiplier=1.2,
        epsilon=3.0,
        sample_rate=0.25,
        expected_batch_size=17,
        phi=0.0123,
    )

    resolved = train_script["_resolved_config"](
        config,
        paths,
        torch.device("cpu"),
        train_size=68,
        eval_size=8,
        result=result,
    )
    summary = train_script["_summary"](
        config, paths, torch.device("cpu"), 68, 8, result
    )

    assert resolved["privacy"]["expected_batch_size"] == 17
    assert resolved["privacy"]["phi"] == 0.0123
    assert summary["expected_batch_size"] == 17
    assert summary["phi"] == 0.0123
