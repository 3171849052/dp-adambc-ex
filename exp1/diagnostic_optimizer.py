"""In-memory P0 diagnostics for the existing FPC-DPAdam optimizer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from bert_qnli.optim import FPCDPAdam


DIAGNOSTIC_FIELDS = (
    "global_step",
    "predictor_x_gap_mse",
    "x2_mean",
    "phi",
    "predictor_gap_over_x2",
    "predictor_gap_over_phi",
    "fpc_expected_bias_ratio",
    "v_clean_nrmse_fpc",
    "v_clean_nrmse_bc",
    "v_nrmse_ratio_fpc_over_bc",
    "clamp_fraction_fpc",
    "clamp_fraction_bc",
    "clamp_fraction_delta",
    "update_cosine_fpc_bc",
    "update_norm_ratio_fpc_over_bc",
)

_EPS_NUM = 1.0e-30


def _finite(value: float) -> float:
    """Return a scalar diagnostic or fail before it can reach the CSV."""

    if not torch.isfinite(torch.tensor(value, dtype=torch.float64)):
        raise FloatingPointError(f"non-finite P0 diagnostic: {value!r}")
    return float(value)


class DiagnosticFPCDPAdam(FPCDPAdam):
    """FPCDPAdam with scalar P0 diagnostics and no alternate update path.

    ``super().step()`` performs the repository's existing FPC update. The
    diagnostic reads its state only after that update and maintains clean and
    shadow second moments in separate, in-memory tensors. Neither diagnostic
    state is passed to ``addcdiv_`` or otherwise used for the real update.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        super().__init__(params, **kwargs)
        self._clean_v: dict[Tensor, Tensor] = {}
        self._shadow_v: dict[Tensor, Tensor] = {}
        self.last_diagnostics: dict[str, int | float] | None = None
        self.logical_diagnostic_steps = 0

    @staticmethod
    def _new_accumulator() -> dict[str, float]:
        """Create Python-float (IEEE float64) scalar accumulators."""

        return {
            "clean_hat_sq": 0.0,
            "x_sq": 0.0,
            "predictor_gap_sq": 0.0,
            "expected_bias_numerator": 0.0,
            "fpc_clean_diff_sq": 0.0,
            "bc_diff_sq": 0.0,
            "fpc_clamped": 0.0,
            "bc_clamped": 0.0,
            "elements": 0.0,
            "u_dot": 0.0,
            "u_fpc_sq": 0.0,
            "u_bc_sq": 0.0,
        }

    @staticmethod
    def _accumulate_predictor_gap(
        parameter: Tensor,
        *,
        predictor: Tensor | None,
        fpc_lambda: float,
        expected_batch_size: int,
        accum: dict[str, float],
    ) -> None:
        """Accumulate x/predictor statistics in float64 before FPC updates y."""

        summed_grad = getattr(parameter, "summed_grad", None)
        if summed_grad is None:
            raise RuntimeError("summed_grad is required for P0 diagnostics")
        x = summed_grad / expected_batch_size
        x_d = x.double()
        gap_d = -x_d if predictor is None else predictor.double() - x_d
        x_sq = float(x_d.square().sum().item())
        gap_sq = float(gap_d.square().sum().item())
        accum["x_sq"] += x_sq
        accum["predictor_gap_sq"] += gap_sq
        accum["expected_bias_numerator"] += (1.0 - fpc_lambda) * gap_sq

    @staticmethod
    def _accumulate_update_metrics(
        clean_hat: Tensor,
        fpc_hat: Tensor,
        bc_hat_corrected: Tensor,
        m_hat: Tensor,
        gamma_prime: float,
        accum: dict[str, float],
    ) -> None:
        """Accumulate all second-moment and update-vector global reductions."""

        # Convert before every reduction. Each sum is therefore float64 and
        # the scalar additions below are also Python/IEEE float64 additions.
        clean_d = clean_hat.double()
        fpc_d = fpc_hat.double()
        bc_d = bc_hat_corrected.double()
        m_d = m_hat.double()

        q_fpc = torch.clamp(fpc_d, min=gamma_prime)
        q_bc = torch.clamp(bc_d, min=gamma_prime)
        u_fpc = m_d / q_fpc.sqrt()
        u_bc = m_d / q_bc.sqrt()

        accum["clean_hat_sq"] += float(clean_d.square().sum().item())
        accum["fpc_clean_diff_sq"] += float(
            (fpc_d - clean_d).square().sum().item()
        )
        accum["bc_diff_sq"] += float((bc_d - clean_d).square().sum().item())
        accum["fpc_clamped"] += float((fpc_d <= gamma_prime).sum().item())
        accum["bc_clamped"] += float((bc_d <= gamma_prime).sum().item())
        accum["elements"] += float(clean_d.numel())
        accum["u_dot"] += float((u_fpc * u_bc).sum().item())
        accum["u_fpc_sq"] += float(u_fpc.square().sum().item())
        accum["u_bc_sq"] += float(u_bc.square().sum().item())

    @torch.no_grad()
    def step(self, closure=None):
        """Run the unchanged FPC update and then collect one diagnostic row."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        phi = self.phi
        accum = self._new_accumulator()
        diagnostic_available = self.expected_batch_size is not None

        # Predictor statistics must be captured before the parent sees y_t.
        # No predictor is retained after each parameter's reduction.
        if diagnostic_available:
            assert self.expected_batch_size is not None
            for group in self.param_groups:
                beta1 = group["betas"][0]
                fpc_lambda = group["fpc_lambda"]
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    previous_step = state.get("step", 0)
                    predictor = None
                    if state and "exp_avg" in state:
                        predictor = self._predictor(
                            state["exp_avg"],
                            previous_step=previous_step,
                            beta1=beta1,
                        )
                    if getattr(parameter, "summed_grad", None) is None:
                        diagnostic_available = False
                        continue
                    self._accumulate_predictor_gap(
                        parameter,
                        predictor=predictor,
                        fpc_lambda=fpc_lambda,
                        expected_batch_size=self.expected_batch_size,
                        accum=accum,
                    )

        # The parent implementation owns every real FPC state mutation and
        # parameter update. In particular, its state exp_avg_sq is the source
        # of the required *unclamped* v_fpc_hat below.
        parent_loss = super().step()
        if loss is None:
            loss = parent_loss

        # No diagnostic row is possible unless every updated parameter exposed
        # the logical-batch summed_grad used by the predictor and clean state.
        if not diagnostic_available:
            self.last_diagnostics = None
            self.last_predictor_x_gap_mse = None
            return loss

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                summed_grad = getattr(parameter, "summed_grad", None)
                if summed_grad is None:
                    # This branch cannot normally be reached because the
                    # pre-update validation above would have disabled the row.
                    self.last_diagnostics = None
                    self.last_predictor_x_gap_mse = None
                    return loss

                state = self.state[parameter]
                step = state["step"]
                clean_v = self._clean_v.get(parameter)
                shadow_v = self._shadow_v.get(parameter)
                if clean_v is None:
                    clean_v = torch.zeros_like(parameter)
                    shadow_v = torch.zeros_like(parameter)
                    self._clean_v[parameter] = clean_v
                    self._shadow_v[parameter] = shadow_v

                x = summed_grad / self.expected_batch_size
                y = parameter.grad
                clean_v.mul_(beta2).addcmul_(x, x, value=1.0 - beta2)
                shadow_v.mul_(beta2).addcmul_(y, y, value=1.0 - beta2)

                bias2 = 1.0 - beta2**step
                clean_hat = clean_v / bias2
                bc_hat_corrected = shadow_v / bias2 - phi
                m_hat = state["exp_avg"] / (1.0 - beta1**step)
                v_fpc_hat = state["exp_avg_sq"] / bias2
                self._accumulate_update_metrics(
                    clean_hat,
                    v_fpc_hat,
                    bc_hat_corrected,
                    m_hat,
                    group["gamma_prime"],
                    accum,
                )

        if accum["elements"] == 0:
            # An empty logical batch can reach the wrapped optimizer without
            # any parameter gradient. It is not a real parameter update and
            # must not produce a divide-by-zero diagnostic row.
            self.last_diagnostics = None
            self.last_predictor_x_gap_mse = None
            return loss

        d = accum["elements"]
        x2_sum = accum["x_sq"]
        gap_mse = accum["predictor_gap_sq"] / d
        clean_norm = max(accum["clean_hat_sq"], 0.0) ** 0.5
        fpc_nrmse = (accum["fpc_clean_diff_sq"] ** 0.5) / (
            clean_norm + _EPS_NUM
        )
        bc_nrmse = (accum["bc_diff_sq"] ** 0.5) / (clean_norm + _EPS_NUM)
        u_fpc_norm = accum["u_fpc_sq"] ** 0.5
        u_bc_norm = accum["u_bc_sq"] ** 0.5

        self.logical_diagnostic_steps += 1
        diagnostics: dict[str, int | float] = {
            "global_step": self.logical_diagnostic_steps,
            "predictor_x_gap_mse": gap_mse,
            "x2_mean": accum["x_sq"] / d,
            "phi": float(phi),
            "predictor_gap_over_x2": accum["predictor_gap_sq"]
            / (x2_sum + _EPS_NUM),
            "predictor_gap_over_phi": gap_mse / max(phi, _EPS_NUM),
            "fpc_expected_bias_ratio": -accum["expected_bias_numerator"]
            / (x2_sum + _EPS_NUM),
            "v_clean_nrmse_fpc": fpc_nrmse,
            "v_clean_nrmse_bc": bc_nrmse,
            "v_nrmse_ratio_fpc_over_bc": fpc_nrmse / (bc_nrmse + _EPS_NUM),
            "clamp_fraction_fpc": accum["fpc_clamped"] / d,
            "clamp_fraction_bc": accum["bc_clamped"] / d,
            "clamp_fraction_delta": (accum["fpc_clamped"] - accum["bc_clamped"])
            / d,
            "update_cosine_fpc_bc": accum["u_dot"]
            / (u_fpc_norm * u_bc_norm + _EPS_NUM),
            "update_norm_ratio_fpc_over_bc": u_fpc_norm / (u_bc_norm + _EPS_NUM),
        }
        self.last_diagnostics = {
            key: value if isinstance(value, int) else _finite(value)
            for key, value in diagnostics.items()
        }
        self.last_predictor_x_gap_mse = _finite(gap_mse)
        self.diagnostic_step = self.logical_diagnostic_steps
        return loss
