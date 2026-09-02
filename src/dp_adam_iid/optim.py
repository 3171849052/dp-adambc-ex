"""Optimizers used by the private training pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class DPAdamBC(Optimizer):
    """Adam with the DP second-moment bias correction from DP-AdamBC.

    Opacus supplies this optimizer with the mean clipped, noised gradient.
    Privacy accounting, clipping, and noise generation remain the
    responsibility of Opacus' ``DPOptimizer`` wrapper.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        gamma_prime: float = 1.0e-8,
        noise_multiplier: float | None = None,
        max_grad_norm: float | None = None,
        expected_batch_size: int | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if gamma_prime <= 0.0:
            raise ValueError(f"gamma_prime must be positive: {gamma_prime}")
        if weight_decay != 0.0:
            raise ValueError("DPAdamBC does not support weight decay")

        super().__init__(
            params,
            defaults={
                "lr": lr,
                "betas": betas,
                "gamma_prime": gamma_prime,
                "weight_decay": weight_decay,
            },
        )
        self.noise_multiplier: float | None = None
        self.max_grad_norm: float | None = None
        self.expected_batch_size: int | None = None
        if any(
            value is not None
            for value in (noise_multiplier, max_grad_norm, expected_batch_size)
        ):
            if None in (noise_multiplier, max_grad_norm, expected_batch_size):
                raise ValueError(
                    "noise_multiplier, max_grad_norm, and expected_batch_size "
                    "must be configured together"
                )
            self.configure_dp(
                noise_multiplier=float(noise_multiplier),
                max_grad_norm=float(max_grad_norm),
                expected_batch_size=int(expected_batch_size),
            )

    def configure_dp(
        self,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int,
    ) -> None:
        """Set the Opacus mechanism values that determine the gradient noise."""

        if noise_multiplier < 0.0:
            raise ValueError("noise_multiplier must be non-negative")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if expected_batch_size <= 0:
            raise ValueError("expected_batch_size must be positive")
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)
        self.expected_batch_size = int(expected_batch_size)

    @property
    def phi(self) -> float:
        """Variance of the mean Gaussian noise received from Opacus."""

        if (
            self.noise_multiplier is None
            or self.max_grad_norm is None
            or self.expected_batch_size is None
        ):
            raise RuntimeError("DPAdamBC must be configured with Opacus DP parameters")
        return (
            self.noise_multiplier
            * self.max_grad_norm
            / self.expected_batch_size
        ) ** 2

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one DP-AdamBC parameter update."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        phi = self.phi
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError(
                        "DPAdamBC does not support sparse gradients; use a dense gradient"
                    )

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2
                )

                m_hat = exp_avg / (1.0 - beta1**step)
                v_hat = exp_avg_sq / (1.0 - beta2**step)
                v_corr = torch.clamp(
                    v_hat - phi, min=group["gamma_prime"]
                )
                parameter.addcdiv_(m_hat, v_corr.sqrt(), value=-group["lr"])

        return loss
