import pytest
import torch

from dp_adam_iid.optim import DPAdamBC


def test_dpadambc_matches_hand_computed_moments_and_updates_for_two_steps():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    learning_rate = 0.1
    beta1, beta2 = 0.5, 0.25
    gamma_prime = 0.04
    optimizer = DPAdamBC(
        [parameter],
        lr=learning_rate,
        betas=(beta1, beta2),
        gamma_prime=gamma_prime,
        noise_multiplier=2.0,
        max_grad_norm=3.0,
        expected_batch_size=10,
    )
    phi = (2.0 * 3.0 / 10.0) ** 2
    expected_parameter = parameter.detach().clone()
    expected_m = torch.zeros_like(parameter)
    expected_v = torch.zeros_like(parameter)

    for step, gradient in enumerate(
        (
            torch.tensor([1.0, 0.5], dtype=torch.float64),
            torch.tensor([-0.5, 1.5], dtype=torch.float64),
        ),
        start=1,
    ):
        expected_m = beta1 * expected_m + (1.0 - beta1) * gradient
        expected_v = beta2 * expected_v + (1.0 - beta2) * gradient.square()
        expected_m_hat = expected_m / (1.0 - beta1**step)
        expected_v_hat = expected_v / (1.0 - beta2**step)
        expected_v_corr = torch.clamp(
            expected_v_hat - phi, min=gamma_prime
        )
        expected_parameter = expected_parameter - (
            learning_rate * expected_m_hat / expected_v_corr.sqrt()
        )

        parameter.grad = gradient.clone()
        optimizer.step()

        state = optimizer.state[parameter]
        actual_m_hat = state["exp_avg"] / (1.0 - beta1**step)
        actual_v_hat = state["exp_avg_sq"] / (1.0 - beta2**step)
        actual_v_corr = torch.clamp(actual_v_hat - optimizer.phi, min=gamma_prime)
        torch.testing.assert_close(actual_m_hat, expected_m_hat)
        torch.testing.assert_close(actual_v_hat, expected_v_hat)
        torch.testing.assert_close(actual_v_corr, expected_v_corr)
        torch.testing.assert_close(parameter, expected_parameter)


def test_dpadambc_phi_uses_expected_batch_size_not_observed_batch_size():
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    observed_batch_size = 3
    optimizer = DPAdamBC(
        [parameter],
        noise_multiplier=1.5,
        max_grad_norm=2.0,
        expected_batch_size=12,
    )

    assert optimizer.phi == pytest.approx((1.5 * 2.0 / 12.0) ** 2)
    assert optimizer.phi != pytest.approx((1.5 * 2.0 / observed_batch_size) ** 2)
