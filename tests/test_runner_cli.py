import os
from pathlib import Path
import subprocess
import sys

import yaml


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


def test_launcher_uses_python_gpu_selection_and_visible_device_mapping():
    launcher = (PROJECT_ROOT / "run.sh").read_text(encoding="utf-8")
    assert "--print-gpu" in launcher
    assert 'CUDA_VISIBLE_DEVICES="$GPU" python -u' in launcher
