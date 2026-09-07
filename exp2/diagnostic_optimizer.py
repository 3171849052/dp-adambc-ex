"""DP-AdamBC diagnostics that leave the real optimizer update untouched."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch
from torch import Tensor

from bert_qnli.optim import DPAdamBC


DIAGNOSTIC_FIELDS = (
    "global_step",
    "phi",
    "x2_mean",
    "bc_clean_nrmse",
    "negative_fraction",
    "clamp_fraction",
    "active_fraction",
    "q_over_gamma_mean",
    "q_over_gamma_p50",
    "q_over_gamma_p90",
    "q_over_gamma_p99",
    "update_cosine_to_floor",
    "update_norm_ratio_to_floor",
    "active_update_energy_fraction",
)

_EPS_NUM = 1.0e-30


def _finite(value: float, *, name: str) -> float:
    """Validate a scalar before it is exposed to a CSV writer."""

    if not math.isfinite(float(value)):
        raise FloatingPointError(f"non-finite DP-AdamBC diagnostic {name}: {value!r}")
    return float(value)


def _new_accumulator() -> dict[str, float]:
    """Create float64 scalar reductions for the complete trainable model."""

    return {
        "x2_sum": 0.0,
        "clean_hat_sq_sum": 0.0,
        "bc_clean_diff_sq_sum": 0.0,
        "negative_count": 0.0,
        "clamp_count": 0.0,
        "active_count": 0.0,
        "elements": 0.0,
        "u_dot_floor": 0.0,
        "u_bc_sq_sum": 0.0,
        "u_floor_sq_sum": 0.0,
        "active_u_bc_sq_sum": 0.0,
    }


def _update_accumulator(
    accumulator: dict[str, float],
    *,
    x: Tensor,
    clean_hat: Tensor,
    r: Tensor,
    m_hat: Tensor,
    gamma_prime: float,
    q_values: list[Tensor],
) -> None:
    """Accumulate one parameter tensor using float64 reductions.

    ``q_values`` receives CPU float64 chunks for exact global quantiles. The
    chunks are temporary and are never serialized or retained between steps.
    """

    x_d = x.detach().double()
    clean_d = clean_hat.detach().double()
    r_d = r.detach().double()
    m_d = m_hat.detach().double()
    q_d = torch.clamp(r_d, min=gamma_prime)

    u_bc = m_d / q_d.sqrt()
    u_floor = m_d / math.sqrt(gamma_prime)
    active = r_d > gamma_prime

    accumulator["x2_sum"] += float(x_d.square().sum().item())
    accumulator["clean_hat_sq_sum"] += float(clean_d.square().sum().item())
    # This is deliberately computed from unclamped r, before q is formed.
    accumulator["bc_clean_diff_sq_sum"] += float(
        (r_d - clean_d).square().sum().item()
    )
    accumulator["negative_count"] += float((r_d <= 0.0).sum().item())
    accumulator["clamp_count"] += float((r_d <= gamma_prime).sum().item())
    accumulator["active_count"] += float(active.sum().item())
    accumulator["elements"] += float(r_d.numel())
    accumulator["u_dot_floor"] += float((u_bc * u_floor).sum().item())
    accumulator["u_bc_sq_sum"] += float(u_bc.square().sum().item())
    accumulator["u_floor_sq_sum"] += float(u_floor.square().sum().item())
    accumulator["active_u_bc_sq_sum"] += float(
        u_bc.square().masked_select(active).sum().item()
    )
    # Flatten each parameter tensor so tensors of different ranks can be
    # concatenated into one element-wise global distribution.
    q_values.append(q_d.reshape(-1).cpu())


def _make_diagnostics(
    accumulator: dict[str, float],
    *,
    phi: float,
    gamma_prime: float,
    q_values: list[Tensor],
    global_step: int,
) -> dict[str, int | float]:
    """Finish element-weighted reductions and calculate global quantiles."""

    elements = accumulator["elements"]
    if elements <= 0.0:
        raise ValueError("cannot make diagnostics for an empty parameter set")

    q_over_gamma = torch.cat(q_values).double() / gamma_prime
    clean_norm = math.sqrt(max(accumulator["clean_hat_sq_sum"], 0.0))
    bc_clean_nrmse = math.sqrt(
        max(accumulator["bc_clean_diff_sq_sum"], 0.0)
    ) / (clean_norm + _EPS_NUM)
    q_percentiles = torch.quantile(
        q_over_gamma,
        torch.tensor([0.50, 0.90, 0.99], dtype=torch.float64),
    )
    u_bc_norm = math.sqrt(max(accumulator["u_bc_sq_sum"], 0.0))
    u_floor_norm = math.sqrt(max(accumulator["u_floor_sq_sum"], 0.0))

    values: dict[str, int | float] = {
        "global_step": int(global_step),
        "phi": phi,
        "x2_mean": accumulator["x2_sum"] / elements,
        "bc_clean_nrmse": bc_clean_nrmse,
        "negative_fraction": accumulator["negative_count"] / elements,
        "clamp_fraction": accumulator["clamp_count"] / elements,
        "active_fraction": accumulator["active_count"] / elements,
        "q_over_gamma_mean": float(q_over_gamma.mean().item()),
        "q_over_gamma_p50": float(q_percentiles[0].item()),
        "q_over_gamma_p90": float(q_percentiles[1].item()),
        "q_over_gamma_p99": float(q_percentiles[2].item()),
        "update_cosine_to_floor": accumulator["u_dot_floor"]
        / (u_bc_norm * u_floor_norm + _EPS_NUM),
        "update_norm_ratio_to_floor": u_bc_norm / (u_floor_norm + _EPS_NUM),
        "active_update_energy_fraction": accumulator["active_u_bc_sq_sum"]
        / (accumulator["u_bc_sq_sum"] + _EPS_NUM),
    }
    result: dict[str, int | float] = {}
    for key, value in values.items():
        result[key] = value if isinstance(value, int) else _finite(value, name=key)
    return result


class DiagnosticDPAdamBC(DPAdamBC):
    """DPAdamBC with in-memory clamp/adaptivity diagnostics.

    The call to ``super().step()`` is the only parameter-update path. Clean
    second moments, reductions, and percentile buffers are research-only state
    and are created or updated after the parent has completed its real update.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        super().__init__(params, **kwargs)
        self._clean_v: dict[Tensor, Tensor] = {}
        self.last_diagnostics: dict[str, int | float] | None = None
        self.logical_diagnostic_steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform the unchanged DP-AdamBC update, then collect one row."""

        self.last_diagnostics = None
        parent_loss = super().step(closure=closure)

        # DPOptimizer can call the underlying optimizer for an empty Poisson
        # batch. No parameter has a gradient in that case, so no row is valid.
        accumulator = _new_accumulator()
        q_values: list[Tensor] = []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            gamma_prime = float(group["gamma_prime"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                summed_grad = getattr(parameter, "summed_grad", None)
                if summed_grad is None or self.expected_batch_size is None:
                    # This is expected only for direct non-Opacus use. The
                    # real update has already happened and remains untouched.
                    return parent_loss

                state = self.state[parameter]
                step = int(state["step"])
                clean_v = self._clean_v.get(parameter)
                if clean_v is None:
                    clean_v = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    self._clean_v[parameter] = clean_v

                x = summed_grad / self.expected_batch_size
                clean_v.mul_(beta2).addcmul_(x, x, value=1.0 - beta2)
                bias2 = 1.0 - beta2**step
                clean_hat = clean_v / bias2

                # ``exp_avg_sq`` is the parent's actual DP-AdamBC state. The
                # subtraction is intentionally made before clamp, as required.
                v_hat = state["exp_avg_sq"] / bias2
                r = v_hat - self.phi
                m_hat = state["exp_avg"] / (1.0 - beta1**step)
                _update_accumulator(
                    accumulator,
                    x=x,
                    clean_hat=clean_hat,
                    r=r,
                    m_hat=m_hat,
                    gamma_prime=gamma_prime,
                    q_values=q_values,
                )

        if accumulator["elements"] == 0.0:
            return parent_loss

        self.logical_diagnostic_steps += 1
        gamma_values = {
            float(group["gamma_prime"])
            for group in self.param_groups
        }
        if len(gamma_values) != 1:
            raise ValueError("Experiment 2 requires one gamma_prime across groups")
        gamma_prime = gamma_values.pop()
        self.last_diagnostics = _make_diagnostics(
            accumulator,
            phi=self.phi,
            gamma_prime=gamma_prime,
            q_values=q_values,
            global_step=self.logical_diagnostic_steps,
        )
        self.diagnostic_step = self.logical_diagnostic_steps
        return parent_loss


__all__ = ["DIAGNOSTIC_FIELDS", "DiagnosticDPAdamBC"]
