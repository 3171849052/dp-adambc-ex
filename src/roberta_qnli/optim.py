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
            raise ValueError(f"{type(self).__name__} does not support weight decay")

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
            raise RuntimeError(
                f"{type(self).__name__} must be configured with Opacus DP parameters"
            )
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


class FPCDPAdam(DPAdamBC):
    """DP Adam with the free-prediction-correction second moment.

    Opacus still owns clipping, accumulation, noise generation, and privacy
    accounting. At a real logical step its ``summed_grad`` tensors contain the
    clipped, pre-noise logical-batch gradient sums, which are also used here for
    the predictor-gap research diagnostic.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        gamma_prime: float = 1.0e-8,
        fpc_lambda: float = 0.5,
        fpc_mode: str = "current",
        fpc_delay_q0: float = 1.0,
        noise_multiplier: float | None = None,
        max_grad_norm: float | None = None,
        expected_batch_size: int | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        if not 0.0 <= fpc_lambda <= 1.0:
            raise ValueError(f"fpc_lambda must be between 0 and 1: {fpc_lambda}")
        if fpc_mode not in {"current", "delay"}:
            raise ValueError("fpc_mode must be 'current' or 'delay'")
        if fpc_delay_q0 <= 0.0:
            raise ValueError(f"fpc_delay_q0 must be positive: {fpc_delay_q0}")

        super().__init__(
            params,
            lr=lr,
            betas=betas,
            gamma_prime=gamma_prime,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            weight_decay=weight_decay,
        )
        fpc_defaults = {
            "fpc_lambda": float(fpc_lambda),
            "fpc_mode": fpc_mode,
            "fpc_delay_q0": float(fpc_delay_q0),
        }
        self.defaults.update(fpc_defaults)
        for group in self.param_groups:
            for key, value in fpc_defaults.items():
                group.setdefault(key, value)

        # Scalars only: the diagnostic never retains predictors, clean gradients,
        # or any other full tensors.
        self.last_predictor_x_gap_mse: float | None = None
        self.diagnostic_step = 0

    @staticmethod
    def _predictor(
        exp_avg: Tensor, *, previous_step: int, beta1: float
    ) -> Tensor:
        """Return p_t before incorporating the current noised gradient."""

        if previous_step == 0:
            return torch.zeros_like(exp_avg, memory_format=torch.preserve_format)
        return exp_avg / (1.0 - beta1**previous_step)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one current- or delayed-FPC logical parameter update."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        phi = self.phi
        gap_squared_sum = 0.0
        gap_elements = 0
        gap_available = True

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            fpc_lambda = group["fpc_lambda"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError(
                        "FPCDPAdam does not support sparse gradients; use a dense gradient"
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

                previous_step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # p_t must be constructed before exp_avg sees tilde g_t.
                predictor = self._predictor(
                    exp_avg, previous_step=previous_step, beta1=beta1
                )

                summed_grad = getattr(parameter, "summed_grad", None)
                if summed_grad is None or self.expected_batch_size is None:
                    gap_available = False
                else:
                    x = summed_grad / self.expected_batch_size
                    difference = predictor - x
                    gap_squared_sum += float(difference.double().square().sum().item())
                    gap_elements += parameter.numel()

                state["step"] = previous_step + 1
                step = state["step"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)

                prediction_error = gradient - predictor
                observation = 2.0 * predictor * gradient - predictor.square()
                observation.add_(
                    prediction_error.square() - phi,
                    alpha=fpc_lambda,
                )
                exp_avg_sq.mul_(beta2).add_(observation, alpha=1.0 - beta2)

                m_hat = exp_avg / (1.0 - beta1**step)
                v_hat = exp_avg_sq / (1.0 - beta2**step)
                q = torch.clamp(v_hat, min=group["gamma_prime"])

                if group["fpc_mode"] == "delay":
                    denominator_q = state.get("q")
                    if denominator_q is None:
                        denominator_q = torch.full_like(
                            parameter, group["fpc_delay_q0"]
                        )
                else:
                    denominator_q = q

                parameter.addcdiv_(m_hat, denominator_q.sqrt(), value=-group["lr"])
                state["q"] = q

        if gap_available and gap_elements:
            self.last_predictor_x_gap_mse = gap_squared_sum / gap_elements
            self.diagnostic_step += 1
        else:
            self.last_predictor_x_gap_mse = None

        return loss
