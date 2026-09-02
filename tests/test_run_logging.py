from datetime import datetime

from dp_adam_iid.config import Config
from dp_adam_iid.run_logging import (
    create_run_directory,
    format_run_name,
    format_tmux_session_name,
)


def _config(tmp_path, **overrides):
    raw = {
        "algorithm": "dpadam",
        "seed": 0,
        "model": {"name": "test", "num_labels": 2},
        "data": {
            "logical_batch_size": 1024,
            "max_physical_batch_size": 8,
            "eval_batch_size": 32,
        },
        "training": {"epochs": 3, "learning_rate": 1.0e-4},
        "privacy": {"epsilon": 3.0, "delta": 1.0e-5, "max_grad_norm": 1.0},
        "output": {"root": str(tmp_path)},
    }
    for section, values in overrides.items():
        raw[section].update(values)
    return Config.from_dict(raw)


def test_run_name_has_exact_required_format(tmp_path):
    config = _config(tmp_path)
    assert format_run_name(config, datetime(2026, 9, 2, 18, 5, 0)) == (
        "20260902-180500_dpadam_eps3_d1e-5_ep3_lb1024_lr1e-4_C1_s0"
    )


def test_run_name_changes_with_algorithm_and_parameters(tmp_path):
    config = _config(tmp_path)
    config.algorithm = "another_algorithm"
    config.privacy.epsilon = 2.0
    config.privacy.delta = 1.0e-6
    config.training.epochs = 7
    config.data.logical_batch_size = 512
    config.training.learning_rate = 2.5e-4
    config.privacy.max_grad_norm = 0.5
    config.seed = 9
    assert format_run_name(config, datetime(2026, 9, 2, 18, 5, 0)) == (
        "20260902-180500_another_algorithm_eps2_d1e-6_ep7_lb512_lr2.5e-4_C0.5_s9"
    )


def test_same_second_collision_advances_timestamp_without_suffix(tmp_path):
    config = _config(tmp_path)
    now = datetime(2026, 9, 2, 18, 5, 0)
    first = create_run_directory(config, now=now)
    second = create_run_directory(config, now=now)

    assert first.directory.name.endswith("_s0")
    assert second.directory.name.startswith(
        "20260902-180501_dpadam_eps3_d1e-5_ep3_lb1024_lr1e-4_C1_s0"
    )
    assert first.directory != second.directory
    assert "_1" not in second.directory.name
    assert not any("hash" in path.name for path in tmp_path.iterdir())


def test_tmux_session_name_is_deterministic_and_tmux_safe():
    assert format_tmux_session_name(
        "/tmp/20260902-180500_dpadam_eps3_d1e-5_ep3_lb1024_lr1e-4_C1_s0"
    ) == "dp_adam_iid_20260902-180500_dpadam_eps3_d1e-5_ep3_lb1024_lr1e-4_C1_s0"
    assert format_tmux_session_name("/tmp/run.name:with spaces") == (
        "dp_adam_iid_run_name_with_spaces"
    )
