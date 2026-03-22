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
    inflate_batch_array,
    sigma,
    alpha,
    sigma_and_alpha_t_given_s,
    cosine_beta_schedule,
    cosine_beta_schedule_discrete,
    clip_noise_schedule,
    sample_gaussian_with_mask,
    sample_feature_noise,
    sample_discrete_features,
    compute_posterior_distribution,
    mask_distributions,
    posterior_distributions,
    gaussian_KL,
    cdf_std_gaussian,
    SNR,
    reverse_tensor,
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


# --- inflate_batch_array ---

def test_inflate_batch_array_shape() -> None:
    """Batch array is reshaped to (batch, 1, 1, ...) matching the target rank."""
    arr = torch.tensor([1.0, 2.0, 3.0])
    target = torch.zeros(3, 5, 7)
    result = inflate_batch_array(arr, target)
    assert result.shape == (3, 1, 1)


# --- sigma and alpha ---

def test_sigma_squared_plus_alpha_squared_equals_one() -> None:
    """σ²(γ) + α²(γ) = 1 for all γ, following from sigmoid(γ) + sigmoid(−γ) = 1."""
    gamma = torch.tensor([0.0, 1.0, -2.0, 3.5])
    target = torch.zeros(4, 8, 8)
    sig = sigma(gamma, target)
    alp = alpha(gamma, target)
    total = sig ** 2 + alp ** 2
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


# --- SNR ---

def test_SNR_equals_exp_neg_gamma() -> None:
    """SNR is defined as exp(−γ), the ratio of signal power to noise power."""
    gamma = torch.tensor([0.0, 1.0, -1.5])
    assert torch.allclose(SNR(gamma), torch.exp(-gamma), atol=1e-6)


# --- cdf_std_gaussian ---

def test_cdf_std_gaussian_at_zero_is_half() -> None:
    """CDF of the standard normal at 0 is exactly 0.5 by symmetry."""
    result = cdf_std_gaussian(torch.tensor(0.0))
    assert result.item() == pytest.approx(0.5, abs=1e-6)


# --- reverse_tensor ---

def test_reverse_tensor_reverses_batch_dim() -> None:
    """Elements appear in reverse order along dim 0; inner dims are unchanged."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected = torch.tensor([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
    assert torch.equal(reverse_tensor(x), expected)


# --- sigma_and_alpha_t_given_s ---

def test_sigma_and_alpha_t_given_s_return_shapes() -> None:
    """Returns three tensors inflated to match the rank of target_size.

    target_size is a torch.Size, not a tensor — trailing singleton dims are
    inserted so the results broadcast against a feature tensor of that shape.
    """
    gamma_t = torch.tensor([0.5, 1.0])
    gamma_s = torch.tensor([0.1, 0.3])
    target_size = torch.Size([2, 5, 4])
    sig2, sig, alp = sigma_and_alpha_t_given_s(gamma_t, gamma_s, target_size)
    assert sig2.shape == (2, 1, 1)
    assert sig.shape == (2, 1, 1)
    assert alp.shape == (2, 1, 1)


# --- cosine_beta_schedule_discrete ---

def test_cosine_beta_schedule_discrete_length() -> None:
    """Returns timesteps + 1 beta values, derived from timesteps + 2 alpha cumprods."""
    betas = cosine_beta_schedule_discrete(timesteps=100)
    assert betas.shape == (101,)


def test_cosine_beta_schedule_discrete_values_in_range() -> None:
    """All discrete betas are valid probabilities in [0, 1]."""
    betas = cosine_beta_schedule_discrete(timesteps=50)
    assert np.all(betas >= 0) and np.all(betas <= 1)


# --- cosine_beta_schedule (continuous) ---

def test_cosine_beta_schedule_monotone_decreasing() -> None:
    """Alpha cumprods decrease monotonically — more noise is added at each step."""
    alphas = cosine_beta_schedule(timesteps=100)
    assert np.all(np.diff(alphas) <= 0)


# --- clip_noise_schedule ---

def test_clip_noise_schedule_clips_step_ratios() -> None:
    """After clipping, consecutive alpha ratios αₜ/αₜ₋₁ are all ≥ clip_value.

    Clipping prevents the noise schedule from decaying too rapidly at the tail,
    which can destabilize training when the SNR becomes very small.
    """
    alphas2 = np.array([1.0, 0.9, 0.5, 0.01, 0.0001])
    clipped = clip_noise_schedule(alphas2, clip_value=0.01)
    full = np.concatenate([[1.0], clipped])
    ratios = full[1:] / full[:-1]
    assert np.all(ratios >= 0.01 - 1e-10)


# --- sample_gaussian_with_mask ---

def test_sample_gaussian_with_mask_zeros_padded_nodes() -> None:
    """Positions where node_mask is 0 are zeroed out in the sampled noise."""
    torch.manual_seed(1)
    node_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    result = sample_gaussian_with_mask((2, 3), node_mask)
    assert result[0, 2].item() == pytest.approx(0.0)
    assert result[1, 1].item() == pytest.approx(0.0)
    assert result[1, 2].item() == pytest.approx(0.0)


# --- sample_feature_noise ---

def test_sample_feature_noise_edge_symmetry() -> None:
    """Edge noise E is symmetric: E[b,i,j] == E[b,j,i], preserving undirected structure."""
    torch.manual_seed(42)
    node_mask = torch.ones(2, 5, dtype=torch.bool)
    result = sample_feature_noise((2, 5, 3), (2, 5, 5, 4), (2, 6), node_mask)
    assert torch.allclose(result.E, result.E.transpose(1, 2))


def test_sample_feature_noise_shapes() -> None:
    """Output X, E, and y shapes match their requested size arguments."""
    torch.manual_seed(0)
    node_mask = torch.ones(2, 4, dtype=torch.bool)
    result = sample_feature_noise((2, 4, 3), (2, 4, 4, 2), (2, 5), node_mask)
    assert result.X.shape == (2, 4, 3)
    assert result.E.shape == (2, 4, 4, 2)
    assert result.y.shape == (2, 5)


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


# --- gaussian_KL ---

def test_gaussian_KL_standard_normal_is_zero() -> None:
    """KL(N(0,1) ‖ N(0,1)) = 0; identical distributions have zero divergence."""
    q_mu = torch.zeros(3, 4)
    q_sigma = torch.ones(3, 4)
    result = gaussian_KL(q_mu, q_sigma)
    assert torch.allclose(result, torch.zeros(3), atol=1e-6)
