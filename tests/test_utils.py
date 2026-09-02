import pytest
import torch

from dp_adam_iid.utils import resolve_device, validate_gpu_selection


def test_cuda_resolution_always_uses_process_local_cuda_zero(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda:0")
    assert resolve_device("cuda") == torch.device("cuda:0")
    assert resolve_device("cuda:0") == torch.device("cuda:0")
    with pytest.raises(RuntimeError, match="cuda:0"):
        resolve_device("cuda:2")


def test_gpu_validation_rejects_unavailable_or_out_of_range_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        validate_gpu_selection(0)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="only 2 physical CUDA GPU"):
        validate_gpu_selection(2)
