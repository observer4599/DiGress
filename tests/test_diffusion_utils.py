"""Tests for src/diffusion/diffusion_utils.py.

Captures load-bearing behavioral contracts for the diffusion utility functions —
primarily shapes, mathematical invariants, and symmetry properties that must
survive refactoring.
"""

import torch
import numpy as np
import pytest

from src.diffusion.diffusion_utils import (
    sum_except_batch,
    cosine_beta_schedule_discrete,
    reverse_tensor,
    sample_discrete_features,
    compute_posterior_distribution,
    mask_distributions,
    posterior_distributions,
)
from src.utils import PlaceHolder


# --- sum_except_batch ---

def test_sum_except_batch_returns_batch_shape() -> None:
    """All non-batch dims are reduced; result shape is always (batch_size,)."""
    x = torch.ones(3, 4, 5)
    result = sum_except_batch(x)
    assert result.shape == (3,)


def test_sum_except_batch_values() -> None:
    """Each batch element sums all inner values; ones(2, 3, 4) → 3*4=12 per element."""
    x = torch.ones(2, 3, 4)
    result = sum_except_batch(x)
    assert result.tolist() == pytest.approx([12.0, 12.0])


# --- reverse_tensor ---

def test_reverse_tensor_reverses_batch_dim() -> None:
    """Elements appear in reverse order along dim 0; inner dims are unchanged."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected = torch.tensor([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
    assert torch.equal(reverse_tensor(x), expected)


# --- cosine_beta_schedule_discrete ---

def test_cosine_beta_schedule_discrete_length() -> None:
    """Returns timesteps + 1 beta values, derived from timesteps + 2 alpha cumprods."""
    betas = cosine_beta_schedule_discrete(timesteps=100)
    assert betas.shape == (101,)


def test_cosine_beta_schedule_discrete_values_in_range() -> None:
    """All discrete betas are valid probabilities in [0, 1]."""
    betas = cosine_beta_schedule_discrete(timesteps=50)
    assert np.all(betas >= 0) and np.all(betas <= 1)


# --- sample_discrete_features ---

def test_sample_discrete_features_edge_symmetry() -> None:
    """Sampled discrete edge indices are symmetric, matching an undirected graph."""
    torch.manual_seed(7)
    bs, n, dx, de = 2, 4, 3, 2
    probX = torch.ones(bs, n, dx) / dx
    probE = torch.ones(bs, n, n, de) / de
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    result = sample_discrete_features(probX, probE, node_mask)
    assert (result.E == result.E.transpose(1, 2)).all()


# --- compute_posterior_distribution ---

def test_compute_posterior_distribution_output_shape() -> None:
    """Posterior q(xₛ | xₜ, x₀) has shape (bs, N, d), matching the input feature dim."""
    bs, n, d = 2, 3, 4
    M = torch.softmax(torch.randn(bs, n, d), dim=-1)
    M_t = torch.softmax(torch.randn(bs, n, d), dim=-1)
    Qt_M = torch.softmax(torch.randn(bs, d, d), dim=-1)
    Qsb_M = torch.softmax(torch.randn(bs, d, d), dim=-1)
    Qtb_M = torch.softmax(torch.randn(bs, d, d), dim=-1)
    result = compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M)
    assert result.shape == (bs, n, d)


# --- mask_distributions ---

def test_mask_distributions_outputs_sum_to_one() -> None:
    """After masking, every node's X distribution sums to 1, including padded rows."""
    bs, n, dx, de = 2, 4, 3, 2
    true_X = torch.ones(bs, n, dx) / dx
    true_E = torch.ones(bs, n, n, de) / de
    pred_X = torch.ones(bs, n, dx) / dx
    pred_E = torch.ones(bs, n, n, de) / de
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    true_X, true_E, pred_X, pred_E = mask_distributions(true_X, true_E, pred_X, pred_E, node_mask)
    assert torch.allclose(true_X.sum(dim=-1), torch.ones(bs, n), atol=1e-6)
    assert torch.allclose(pred_X.sum(dim=-1), torch.ones(bs, n), atol=1e-6)


def test_mask_distributions_masked_nodes_set_to_row_X() -> None:
    """Padded node rows are replaced with the canonical one-hot (1, 0, …) distribution."""
    true_X = torch.ones(1, 3, 4) / 4
    true_E = torch.ones(1, 3, 3, 2) / 2
    pred_X = torch.ones(1, 3, 4) / 4
    pred_E = torch.ones(1, 3, 3, 2) / 2
    node_mask = torch.tensor([[True, True, False]])
    true_X_out, _, pred_X_out, _ = mask_distributions(true_X, true_E, pred_X, pred_E, node_mask)
    assert true_X_out[0, 2, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert pred_X_out[0, 2, 0].item() == pytest.approx(1.0, abs=1e-5)
