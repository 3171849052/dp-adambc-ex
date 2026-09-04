from types import SimpleNamespace

import pytest
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from dp_adam_iid.config import Config
from dp_adam_iid.optim import DPAdamBC, FPCDPAdam
from dp_adam_iid.privacy import make_private_training


class TinyDataset(Dataset):
    def __init__(self):
        self.input_ids = torch.tensor(
            [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10]]
        )
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "labels": self.labels[index]}


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.classifier(self.embedding(input_ids).mean(dim=1)))


def test_opacus_gdp_ghost_uses_adam_and_one_logical_step():
    torch.manual_seed(0)
    model = TinyClassifier()
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
    original_optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    criterion = nn.CrossEntropyLoss()
    engine = PrivacyEngine(accountant="gdp")
    hooks, optimizer, criterion, private_loader = engine.make_private_with_epsilon(
        module=model,
        optimizer=original_optimizer,
        criterion=criterion,
        data_loader=loader,
        target_epsilon=3.0,
        target_delta=1.0e-5,
        epochs=1,
        max_grad_norm=1.0,
        clipping="flat",
        grad_sample_mode="ghost",
        poisson_sampling=True,
        wrap_model=False,
    )
    assert isinstance(optimizer.original_optimizer, torch.optim.Adam)
    assert optimizer.noise_multiplier > 0

    logical_steps = 0
    with BatchMemoryManager(
        data_loader=private_loader,
        max_physical_batch_size=2,
        optimizer=optimizer,
    ) as memory_loader:
        for batch in memory_loader:
            if not batch["labels"].numel():
                optimizer.zero_grad()
                optimizer.step()
                optimizer.zero_grad()
                continue
            outputs = model(input_ids=batch["input_ids"])
            loss = criterion(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if not optimizer._is_last_step_skipped:
                logical_steps += 1

    assert logical_steps == len(private_loader)
    assert engine.get_epsilon(1.0e-5) > 0
    hooks.cleanup()


def test_unsupported_optimizer_does_not_fall_back_to_fpcdpadam():
    config = Config.from_dict(
        {
            "algorithm": "dpadam",
            "model": {"name": "test"},
            "data": {"logical_batch_size": 4, "max_physical_batch_size": 2},
            "training": {"optimizer": "adam"},
        }
    )
    config.training.optimizer = "not-an-optimizer"

    with pytest.raises(ValueError, match="Unsupported optimizer: not-an-optimizer"):
        make_private_training(
            TinyClassifier(), DataLoader(TinyDataset(), batch_size=4), config
        )


def test_dpadambc_opacus_ghost_and_batch_memory_manager_smoke():
    torch.manual_seed(0)
    config = Config.from_dict(
        {
            "algorithm": "dpadambc",
            "model": {"name": "test"},
            "data": {
                "logical_batch_size": 4,
                "max_physical_batch_size": 2,
            },
            "training": {
                "epochs": 1,
                "optimizer": "dpadambc",
                "gamma_prime": 1.0e-4,
            },
            "privacy": {
                "epsilon": 3.0,
                "delta": 1.0e-5,
                "accountant": "gdp",
                "max_grad_norm": 1.0,
                "clipping": "flat",
                "grad_sample_mode": "ghost",
                "poisson_sampling": True,
                "loss_reduction": "mean",
                "wrap_model": False,
            },
        }
    )
    model = TinyClassifier()
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
    model.train()
    private = make_private_training(model, loader, config)
    optimizer = private.optimizer
    underlying = optimizer.original_optimizer

    assert isinstance(underlying, DPAdamBC)
    assert private.expected_batch_size == optimizer.expected_batch_size
    assert private.phi == underlying.phi
    assert underlying.expected_batch_size == optimizer.expected_batch_size
    assert underlying.phi == pytest.approx(
        (
            optimizer.noise_multiplier
            * optimizer.max_grad_norm
            / optimizer.expected_batch_size
        )
        ** 2
    )

    logical_steps = 0
    try:
        with BatchMemoryManager(
            data_loader=private.data_loader,
            max_physical_batch_size=config.data.max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_loader:
            for batch in memory_loader:
                if not batch["labels"].numel():
                    optimizer.zero_grad()
                    optimizer.step()
                    optimizer.zero_grad()
                    continue
                outputs = model(input_ids=batch["input_ids"])
                loss = private.criterion(outputs.logits, batch["labels"])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if not optimizer._is_last_step_skipped:
                    logical_steps += 1
    finally:
        private.hooks.cleanup()

    assert logical_steps >= 1
    assert any(state["step"] >= 1 for state in underlying.state.values())


def test_privacy_wires_opacus_parameters_into_fpcdpadam():
    config = Config.from_dict(
        {
            "algorithm": "fpcdpadam",
            "model": {"name": "test"},
            "data": {"logical_batch_size": 4, "max_physical_batch_size": 2},
            "training": {
                "epochs": 1,
                "optimizer": "fpcdpadam",
                "gamma_prime": 1.0e-4,
                "fpc_lambda": 0.25,
                "fpc_mode": "delay",
                "fpc_delay_q0": 2.0,
            },
            "privacy": {
                "epsilon": 3.0,
                "delta": 1.0e-5,
                "accountant": "gdp",
                "max_grad_norm": 1.0,
                "clipping": "flat",
                "grad_sample_mode": "ghost",
                "poisson_sampling": True,
                "loss_reduction": "mean",
                "wrap_model": False,
            },
        }
    )
    model = TinyClassifier()
    model.train()
    private = make_private_training(
        model, DataLoader(TinyDataset(), batch_size=4, shuffle=False), config
    )
    underlying = private.optimizer.original_optimizer

    try:
        assert isinstance(underlying, FPCDPAdam)
        assert underlying.param_groups[0]["fpc_lambda"] == pytest.approx(0.25)
        assert underlying.param_groups[0]["fpc_mode"] == "delay"
        assert underlying.param_groups[0]["fpc_delay_q0"] == pytest.approx(2.0)
        assert underlying.noise_multiplier == private.optimizer.noise_multiplier
        assert underlying.max_grad_norm == private.optimizer.max_grad_norm
        assert underlying.expected_batch_size == private.optimizer.expected_batch_size
        assert private.phi == pytest.approx(underlying.phi)
    finally:
        private.hooks.cleanup()


def _manual_clipped_gradient_sum(model, examples, *, max_grad_norm):
    criterion = nn.CrossEntropyLoss()
    sums = [torch.zeros_like(parameter) for parameter in model.parameters()]
    for example in examples:
        model.zero_grad()
        logits = model(input_ids=example["input_ids"].unsqueeze(0)).logits
        loss = criterion(logits, example["labels"].unsqueeze(0))
        loss.backward()
        gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
        global_norm = torch.stack([gradient.norm() for gradient in gradients]).norm()
        coefficient = min(1.0, max_grad_norm / (float(global_norm) + 1.0e-6))
        for total, gradient in zip(sums, gradients):
            total.add_(gradient, alpha=coefficient)
    model.zero_grad()
    return sums


def test_fpc_ghost_extracts_clipped_pre_noise_scaled_logical_gradient():
    torch.manual_seed(7)
    dataset = TinyDataset()
    model = TinyClassifier()
    reference_model = TinyClassifier()
    reference_model.load_state_dict(model.state_dict())
    max_grad_norm = 0.2
    expected_batch_size = 4
    expected_sums = _manual_clipped_gradient_sum(
        reference_model,
        [dataset[index] for index in range(expected_batch_size)],
        max_grad_norm=max_grad_norm,
    )

    underlying = FPCDPAdam(
        model.parameters(),
        lr=1.0e-4,
        gamma_prime=1.0e-5,
        fpc_lambda=0.5,
        noise_multiplier=0.8,
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
    )
    engine = PrivacyEngine(accountant="gdp")
    model.train()
    hooks, optimizer, criterion, private_loader = engine.make_private(
        module=model,
        optimizer=underlying,
        criterion=nn.CrossEntropyLoss(),
        data_loader=DataLoader(
            Subset(dataset, range(expected_batch_size)),
            batch_size=expected_batch_size,
            shuffle=False,
        ),
        noise_multiplier=0.8,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="ghost",
        loss_reduction="mean",
        wrap_model=False,
    )

    physical_steps = 0
    logical_steps = 0
    try:
        with BatchMemoryManager(
            data_loader=private_loader,
            max_physical_batch_size=2,
            optimizer=optimizer,
        ) as memory_loader:
            for batch in memory_loader:
                physical_steps += 1
                optimizer.zero_grad()
                loss = criterion(
                    model(input_ids=batch["input_ids"]).logits, batch["labels"]
                )
                loss.backward()
                optimizer.step()

                if optimizer._is_last_step_skipped:
                    assert underlying.diagnostic_step == 0
                else:
                    logical_steps += 1
                    extracted_x = [
                        parameter.summed_grad / expected_batch_size
                        for parameter in model.parameters()
                    ]
                    for actual, expected_sum in zip(extracted_x, expected_sums):
                        torch.testing.assert_close(
                            actual,
                            expected_sum / expected_batch_size,
                            rtol=2.0e-5,
                            atol=2.0e-6,
                        )

                    expected_gap = sum(
                        float(value.double().square().sum()) for value in extracted_x
                    ) / sum(value.numel() for value in extracted_x)
                    assert underlying.last_predictor_x_gap_mse == pytest.approx(
                        expected_gap
                    )
                    # The gradient consumed by FPC includes Gaussian noise, while
                    # summed_grad (and x above) remains pre-noise.
                    assert any(
                        not torch.allclose(parameter.grad, clean)
                        for parameter, clean in zip(model.parameters(), extracted_x)
                    )
                optimizer.zero_grad()
    finally:
        hooks.cleanup()

    assert physical_steps == 2
    assert logical_steps == 1
    assert underlying.diagnostic_step == 1
