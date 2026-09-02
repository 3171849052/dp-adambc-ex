"""Opacus setup: GDP accountant, Ghost Clipping and IID Gaussian DP Adam."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any
import warnings

import torch
from opacus import PrivacyEngine
from opacus.accountants.analysis import gdp as gdp_analysis
from scipy import optimize
from torch import nn
from torch.utils.data import DataLoader

from .config import Config


@dataclass
class PrivateTraining:
    """Objects returned by Opacus for one private training run."""

    model: nn.Module
    optimizer: Any
    criterion: Any
    data_loader: DataLoader
    privacy_engine: PrivacyEngine
    hooks: Any | None

    @property
    def noise_multiplier(self) -> float:
        return float(self.optimizer.noise_multiplier)


def _safe_eps_from_mu(*, mu: float, delta: float) -> float:
    """Use Opacus' GDP formula with a wider numerical root bracket.

    Opacus 1.6.0 fixes the upper root bracket at 500. For the full QNLI
    configuration, an intermediate binary-search sigma can have a GDP-to-DP
    root slightly above 500 even though the final calibrated sigma is valid.
    This compatibility helper keeps Opacus' ``delta_eps_mu`` formula and only
    widens that numerical bracket; it does not implement a new accountant or
    Gaussian mechanism.
    """

    def objective(epsilon: float) -> float:
        value = gdp_analysis.delta_eps_mu(eps=epsilon, mu=mu)
        return float(value - delta)

    upper = 500.0
    upper_value = objective(upper)
    while (not math.isfinite(upper_value) or upper_value > 0) and upper < 700.0:
        upper += 25.0
        upper_value = objective(upper)
    if not math.isfinite(upper_value) or upper_value > 0:
        raise ValueError("Unable to bracket the GDP epsilon root")
    return float(optimize.brentq(objective, 0.0, upper))


@contextmanager
def _wider_gdp_root_bracket():
    """Temporarily fix only Opacus 1.6.0's too-small GDP root bracket."""

    original = gdp_analysis.eps_from_mu
    gdp_analysis.eps_from_mu = _safe_eps_from_mu
    try:
        yield
    finally:
        gdp_analysis.eps_from_mu = original


def _make_private_with_epsilon(
    privacy_engine: PrivacyEngine,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    config: Config,
):
    """Call Opacus calibration, with a narrow compatibility fallback."""

    kwargs = dict(
        module=model,
        optimizer=optimizer,
        criterion=criterion,
        data_loader=train_loader,
        target_epsilon=config.privacy.epsilon,
        target_delta=config.privacy.delta,
        epochs=config.training.epochs,
        max_grad_norm=config.privacy.max_grad_norm,
        poisson_sampling=config.privacy.poisson_sampling,
        clipping=config.privacy.clipping,
        loss_reduction=config.privacy.loss_reduction,
        grad_sample_mode=config.privacy.grad_sample_mode,
        wrap_model=config.privacy.wrap_model,
    )
    try:
        return privacy_engine.make_private_with_epsilon(**kwargs)
    except ValueError as exc:
        if "f(a) and f(b) must have different signs" not in str(exc):
            raise
        warnings.warn(
            "Using a local numerical compatibility bracket for Opacus 1.6.0 GDP calibration.",
            RuntimeWarning,
            stacklevel=2,
        )
        with _wider_gdp_root_bracket():
            return privacy_engine.make_private_with_epsilon(**kwargs)


def make_private_training(
    model: nn.Module,
    train_loader: DataLoader,
    config: Config,
) -> PrivateTraining:
    """Wrap a pure Adam optimizer with Opacus' official DP machinery.

    With ``grad_sample_mode='ghost'``, Opacus returns a criterion wrapper. Its
    ``loss.backward()`` performs the two backward passes required by Ghost
    Clipping. The optimizer then clips, aggregates, adds IID Gaussian noise once
    per logical batch, and calls the underlying Adam exactly once.
    """

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(reduction=config.privacy.loss_reduction)
    privacy_engine = PrivacyEngine(accountant=config.privacy.accountant)

    private_handle, private_optimizer, private_criterion, private_loader = (
        _make_private_with_epsilon(
            privacy_engine,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            config=config,
        )
    )

    if config.privacy.wrap_model:
        training_model = private_handle
        hooks = None
    else:
        # In non-wrapping mode Opacus returns hook management separately and
        # the original Transformers model remains the model used for forward.
        training_model = model
        hooks = private_handle

    return PrivateTraining(
        model=training_model,
        optimizer=private_optimizer,
        criterion=private_criterion,
        data_loader=private_loader,
        privacy_engine=privacy_engine,
        hooks=hooks,
    )
