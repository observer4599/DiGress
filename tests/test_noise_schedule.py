"""Tests for src/diffusion/noise_schedule.py.

Captures load-bearing behavioral contracts for the noise schedule and
transition matrix classes — shapes, stochastic-matrix invariants, and
boundary conditions that must survive refactoring.
"""

import torch
import pytest

from src.diffusion.noise_schedule import (
    PredefinedNoiseScheduleDiscrete,
    DiscreteUniformTransition,
    MarginalUniformTransition,
    AbsorbingStateTransition,
)


def test_predefined_noise_schedule_discrete_alphas_bar_decreasing():
    """alphas_bar is monotonically non-increasing — signal retention ᾱ_t decays over time."""
    sched = PredefinedNoiseScheduleDiscrete(noise_schedule='cosine', timesteps=100)
    diffs = sched.alphas_bar[1:] - sched.alphas_bar[:-1]
    assert (diffs <= 0).all()


def test_predefined_noise_schedule_discrete_forward_returns_beta():
    """Forward pass with integer timesteps returns the corresponding β_t values."""
    sched = PredefinedNoiseScheduleDiscrete(noise_schedule='cosine', timesteps=50)
    result = sched(t_int=torch.tensor([0, 10, 49]))
    expected = sched.betas[[0, 10, 49]]
    assert torch.allclose(result, expected)


def test_discrete_uniform_transition_qt_shape_and_rows_sum():
    """get_Qt returns stochastic matrices of shape (bs, K, K) with rows summing to 1.

    NOTE: bs > 1 triggers a shape-broadcast bug in the source — ``beta_t.unsqueeze(1)``
    produces shape (bs, 1) instead of the needed (bs, 1, 1) for broadcasting against
    (1, K, K). Tested with bs=1 to capture the working behavior.
    """
    transition = DiscreteUniformTransition(x_classes=4, e_classes=3, y_classes=2)
    beta_t = torch.tensor([0.3])  # bs=1
    result = transition.get_Qt(beta_t, device='cpu')
    assert result.X.shape == (1, 4, 4)
    assert result.E.shape == (1, 3, 3)
    assert torch.allclose(result.X.sum(dim=-1), torch.ones(1, 4), atol=1e-6)
    assert torch.allclose(result.E.sum(dim=-1), torch.ones(1, 3), atol=1e-6)


def test_discrete_uniform_transition_qt_bar_identity_at_alpha1():
    """get_Qt_bar at ᾱ_t=1 returns the identity matrix — no diffusion has occurred.

    bs=1 due to shape-broadcast limitation described in test_discrete_uniform_transition_qt_shape_and_rows_sum.
    """
    transition = DiscreteUniformTransition(x_classes=4, e_classes=3, y_classes=2)
    alpha_bar = torch.ones(1)  # bs=1, alpha_bar=1 → no diffusion
    result = transition.get_Qt_bar(alpha_bar, device='cpu')
    assert torch.allclose(result.X, torch.eye(4).unsqueeze(0), atol=1e-6)
    assert torch.allclose(result.E, torch.eye(3).unsqueeze(0), atol=1e-6)


def test_marginal_uniform_transition_qt_bar_rows_sum():
    """get_Qt_bar rows sum to 1 for any ᾱ_t — valid stochastic matrices are produced.

    bs=1 due to shape-broadcast limitation described in DiscreteUniformTransition tests.
    """
    x_marginals = torch.tensor([0.2, 0.5, 0.3])
    e_marginals = torch.tensor([0.6, 0.4])
    transition = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals, y_classes=2)
    alpha_bar = torch.tensor([0.4])  # bs=1
    result = transition.get_Qt_bar(alpha_bar, device='cpu')
    assert torch.allclose(result.X.sum(dim=-1), torch.ones(1, 3), atol=1e-6)
    assert torch.allclose(result.E.sum(dim=-1), torch.ones(1, 2), atol=1e-6)


def test_absorbing_state_transition_qt_absorbs_to_correct_column():
    """get_Qt routes probability mass to the abs_state column at the expected rate.

    For off-diagonal rows the absorbing-column entry equals β_t (0.5 here); for
    the absorbing state's own row the entry equals 1.0 — once absorbed, a feature
    stays absorbed.
    """
    transition = AbsorbingStateTransition(abs_state=0, x_classes=3, e_classes=3, y_classes=2)
    beta_t = torch.tensor([0.5])
    q_x, q_e, _ = transition.get_Qt(beta_t)
    assert q_x[0, 1, 0].item() == pytest.approx(0.5)
    assert q_x[0, 2, 0].item() == pytest.approx(0.5)
    assert q_x[0, 0, 0].item() == pytest.approx(1.0)
