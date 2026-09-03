import pytest
import torch

from dp_adam_iid.optim import DPAdamBC, FPCDPAdam


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


def _fpc(parameter, **overrides):
    options = {
        "lr": 0.1,
        "betas": (0.5, 0.25),
        "gamma_prime": 1.0e-6,
        "fpc_lambda": 0.5,
        "fpc_mode": "current",
        "fpc_delay_q0": 1.0,
        "noise_multiplier": 0.0,
        "max_grad_norm": 1.0,
        "expected_batch_size": 2,
    }
    options.update(overrides)
    return FPCDPAdam([parameter], **options)


def test_fpc_predictor_is_zero_then_previous_bias_corrected_first_moment():
    parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    optimizer = _fpc(
        parameter,
        betas=(0.5, 0.0),
        fpc_lambda=0.0,
        gamma_prime=1.0e-8,
    )

    parameter.grad = torch.tensor([2.0], dtype=torch.float64)
    optimizer.step()
    # p_1=0 makes z_1=0 when lambda=0.
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg_sq"], torch.tensor([0.0], dtype=torch.float64)
    )

    parameter.grad = torch.tensor([5.0], dtype=torch.float64)
    optimizer.step()
    # m_hat_1=2, so z_2=2*p_2*g_2-p_2^2=16. A predictor containing
    # the current gradient would not produce this observation.
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg_sq"], torch.tensor([16.0], dtype=torch.float64)
    )


def test_fpc_current_lambda_one_matches_dpadambc():
    parameter_bc = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    parameter_fpc = torch.nn.Parameter(parameter_bc.detach().clone())
    common = {
        "lr": 0.03,
        "betas": (0.6, 0.7),
        "gamma_prime": 0.04,
        "noise_multiplier": 1.5,
        "max_grad_norm": 2.0,
        "expected_batch_size": 10,
    }
    optimizer_bc = DPAdamBC([parameter_bc], **common)
    optimizer_fpc = FPCDPAdam(
        [parameter_fpc], fpc_lambda=1.0, fpc_mode="current", **common
    )

    for gradient in (
        torch.tensor([1.0, 0.5], dtype=torch.float64),
        torch.tensor([-0.5, 1.5], dtype=torch.float64),
        torch.tensor([0.25, -2.0], dtype=torch.float64),
    ):
        parameter_bc.grad = gradient.clone()
        parameter_fpc.grad = gradient.clone()
        optimizer_bc.step()
        optimizer_fpc.step()
        torch.testing.assert_close(parameter_fpc, parameter_bc)


def test_fpc_current_uses_q_t_while_delay_uses_q_t_minus_one():
    current_parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    delay_parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    options = {
        "lr": 0.1,
        "betas": (0.0, 0.0),
        "gamma_prime": 1.0e-8,
        "fpc_lambda": 1.0,
        "fpc_delay_q0": 1.0,
        "noise_multiplier": 0.0,
        "max_grad_norm": 1.0,
        "expected_batch_size": 1,
    }
    current = FPCDPAdam([current_parameter], fpc_mode="current", **options)
    delay = FPCDPAdam([delay_parameter], fpc_mode="delay", **options)

    for parameter, optimizer in (
        (current_parameter, current),
        (delay_parameter, delay),
    ):
        parameter.grad = torch.tensor([2.0], dtype=torch.float64)
        optimizer.step()

    # q_1=4: current divides by 2, while delay's first step uses sqrt(q_0)=1.
    torch.testing.assert_close(current_parameter, torch.tensor([-0.1], dtype=torch.float64))
    torch.testing.assert_close(delay_parameter, torch.tensor([-0.2], dtype=torch.float64))

    delay_parameter.grad = torch.tensor([3.0], dtype=torch.float64)
    delay.step()
    # The second delayed update divides m_hat_2=3 by sqrt(q_1)=2.
    torch.testing.assert_close(delay_parameter, torch.tensor([-0.35], dtype=torch.float64))


def test_fpc_negative_observation_is_clamped_before_sqrt():
    parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    optimizer = _fpc(
        parameter,
        betas=(0.0, 0.0),
        gamma_prime=0.25,
        fpc_lambda=1.0,
        noise_multiplier=2.0,
        max_grad_norm=1.0,
        expected_batch_size=1,
    )
    parameter.grad = torch.tensor([1.0], dtype=torch.float64)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["exp_avg_sq"].item() == pytest.approx(-3.0)
    torch.testing.assert_close(state["q"], torch.tensor([0.25], dtype=torch.float64))
    torch.testing.assert_close(parameter, torch.tensor([-0.2], dtype=torch.float64))
    assert torch.isfinite(parameter).all()


def test_fpc_predictor_x_gap_mse_is_exact_for_small_tensors():
    first = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    optimizer = FPCDPAdam(
        [first, second],
        betas=(0.0, 0.0),
        gamma_prime=1.0e-8,
        fpc_lambda=1.0,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=2,
    )
    first.grad = torch.tensor([9.0, 9.0], dtype=torch.float64)
    second.grad = torch.tensor([9.0], dtype=torch.float64)
    first.summed_grad = torch.tensor([2.0, 4.0], dtype=torch.float64)
    second.summed_grad = torch.tensor([-2.0], dtype=torch.float64)

    optimizer.step()

    # p_1=0 and x=[1, 2, -1], hence ||p_1-x||^2 / 3 = 2.
    assert optimizer.last_predictor_x_gap_mse == pytest.approx(2.0)
    assert optimizer.diagnostic_step == 1


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("fpc_lambda", -0.1, "fpc_lambda"),
        ("fpc_lambda", 1.1, "fpc_lambda"),
        ("fpc_mode", "future", "fpc_mode"),
        ("fpc_delay_q0", 0.0, "fpc_delay_q0"),
    ],
)
def test_fpc_optimizer_rejects_invalid_options(option, value, message):
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    with pytest.raises(ValueError, match=message):
        _fpc(parameter, **{option: value})
