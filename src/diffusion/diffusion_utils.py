"""Utility functions for diffusion processes in DiGress.

Covers two diffusion paradigms used in Vignac et al. (2023) 'DiGress:
Discrete Denoising Diffusion Models for Graph Generation' (ICLR 2023,
https://openreview.net/forum?id=UaAD-Nu86WX):

- **Continuous diffusion**: noise schedules parametrized via a scalar log-SNR
  γ, with signal level α = √sigmoid(−γ) and noise level σ = √sigmoid(γ),
  so that α² + σ² = 1 at every timestep.

- **Discrete diffusion** (main DiGress model): categorical features perturbed
  via Markov transition matrices Q_t (one-step), Q_{s|0} / Qsb (cumulative to
  s), Q_{t|0} / Qtb (cumulative to t). The posterior p(x_s | x_t, x_0) is
  computed in closed form from these matrices.

Graphs are represented as batched (X, E, y) triples:
  - X: node features,  shape (bs, n, dx)
  - E: edge features,  shape (bs, n, n, de), symmetric (undirected graphs)
  - y: global features, shape (bs, dy)

Variable-length graphs are padded to the same n and gated by a boolean
node_mask of shape (bs, n).
"""

import math
from collections.abc import Callable

import numpy as np
import torch
from torch.nn import functional as F

from src.utils import PlaceHolder


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """Sum all dimensions of a tensor except the batch dimension.

    Flattens x to (batch_size, -1) and reduces along the last axis, producing
    a scalar summary per sample. Used to aggregate per-element losses or KL
    terms into per-sample scalars before averaging over the batch.

    Args:
        x: Tensor of any shape (batch_size, *dims).

    Returns:
        Tensor of shape (batch_size,) with per-sample sums.
    """
    return x.reshape(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor) -> None:
    """Assert that all padded positions in a tensor are approximately zero.

    Raises an AssertionError if any masked-out entry has absolute value ≥ 1e-4.
    Called after attention and feature update layers to catch masking bugs early.

    Args:
        variable: Feature tensor of shape (bs, n, ...) to validate.
        node_mask: Boolean or integer mask of shape broadcastable to variable,
            where 1 marks valid nodes and 0 marks padding.

    Raises:
        AssertionError: If any masked position has |value| ≥ 1e-4.
    """
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def sample_gaussian(size: torch.Size | tuple[int, ...]) -> torch.Tensor:
    """Sample from a standard normal distribution.

    Args:
        size: Shape of the output tensor.

    Returns:
        Tensor of the given shape sampled from N(0, 1).
    """
    x = torch.randn(size)
    return x


def sample_gaussian_with_mask(
    size: torch.Size | tuple[int, ...], node_mask: torch.Tensor
) -> torch.Tensor:
    """Sample from a standard normal distribution and zero out padded positions.

    Args:
        size: Shape of the output tensor, typically (bs, n, d).
        node_mask: Boolean mask of shape (bs, n) or broadcastable; padded
            positions are zeroed after sampling.

    Returns:
        Masked Gaussian sample of the given size, cast to the same dtype as
        node_mask.
    """
    x = torch.randn(size)
    x = x.type_as(node_mask.float())
    x_masked = x * node_mask
    return x_masked


def clip_noise_schedule(
    alphas2: np.ndarray, clip_value: float = 0.001
) -> np.ndarray:
    """Clip step-wise alpha ratios in a noise schedule to improve sampling stability.

    For a cumulative-product schedule given by ᾱ²_t, the per-step ratio
    ᾱ²_t / ᾱ²_{t-1} can shrink very close to zero at high noise levels,
    causing numerical instability. This function clips each ratio to
    [clip_value, 1] and recomputes the cumulative product.

    Args:
        alphas2: Array of cumulative alpha-squared values ᾱ²_1 … ᾱ²_T,
            shape (T,).
        clip_value: Minimum allowed per-step ratio. Smaller values permit
            more aggressive noise schedules.

    Returns:
        Clipped and renormalized array of cumulative alpha-squared values,
        shape (T,).
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(
    timesteps: int, s: float = 0.008, raise_to_power: float = 1
) -> np.ndarray:
    """Compute a cosine cumulative-alpha² schedule for continuous diffusion.

    Proposed by Nichol & Dhariwal (2021) 'Improved Denoising Diffusion
    Probabilistic Models' (https://openreview.net/forum?id=-NEXDKk8gZ).
    The cumulative signal schedule is:

        ᾱ_t = cos²(π/2 · ((t/T + s) / (1 + s))) / ᾱ_0

    where s=0.008 prevents β from growing too large near t=0. Compared to
    linear schedules, the cosine schedule decays the signal more gradually
    and suits continuous data better.

    Args:
        timesteps: Number of diffusion steps T.
        s: Small offset to keep β_0 near zero.
        raise_to_power: If not 1, raises the final ᾱ values to this power,
            compressing or expanding the effective noise schedule.

    Returns:
        Array of shape (T,) containing cumulative alpha-squared values
        ᾱ²_1 … ᾱ²_T.
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(timesteps: int, s: float = 0.008) -> np.ndarray:
    """Compute per-step beta values for a discrete cosine noise schedule.

    Adapts the cosine schedule (Nichol & Dhariwal, 2021) to discrete diffusion
    by returning individual β_t = 1 − α_t / α_{t−1} values rather than
    cumulative products. Used to parametrize Markov transition matrices Q_t
    for categorical node and edge features.

    Args:
        timesteps: Number of diffusion steps T.
        s: Small offset to keep β_0 near zero.

    Returns:
        Array of shape (T,) with per-step beta values β_1 … β_T ∈ [0, 1].
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(
    timesteps: int, average_num_nodes: int = 50, s: float = 0.008
) -> np.ndarray:
    """Compute a graph-aware discrete beta schedule with a minimum early-step rate.

    Extends the cosine schedule by enforcing a minimum beta for all steps so
    that on average ~1.2 edges are updated per diffusion step. This prevents
    the schedule from being too small, ensuring meaningful transitions even
    for dense graphs.

    The minimum beta is derived from:

        β_min = updates_per_graph / (p · |edges|)

    where p = 4/5 (complement of the edge-class prior) and
    |edges| = n(n−1)/2 for a complete graph on average_num_nodes nodes.

    Args:
        timesteps: Number of diffusion steps T. Must be ≥ 100.
        average_num_nodes: Expected number of nodes per graph, used to
            estimate the number of edges and calibrate the minimum beta.
        s: Cosine schedule offset.

    Returns:
        Array of shape (T,) with per-step beta values.

    Raises:
        AssertionError: If timesteps < 100.
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def gaussian_KL(q_mu: torch.Tensor, q_sigma: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence KL(q ‖ p) where p = N(0, 1).

    Uses the closed-form KL between a Gaussian q = N(μ, σ²) and the standard
    normal p = N(0, 1):

        KL(q ‖ p) = log(1/σ) + (σ² + μ²)/2 − 1/2

    Summation is over all non-batch dimensions via `sum_except_batch`, giving
    a per-sample scalar. Used to compute the prior-matching loss L_T in the
    continuous diffusion ELBO, enforcing that q(x_T | x_0) ≈ N(0, I).

    Args:
        q_mu: Mean of the posterior distribution q, shape (bs, *dims).
        q_sigma: Standard deviation of q, shape (bs, *dims). Must be positive.

    Returns:
        Per-sample KL values of shape (bs,).
    """
    return sum_except_batch((torch.log(1 / q_sigma) + 0.5 * (q_sigma ** 2 + q_mu ** 2) - 0.5))


def cdf_std_gaussian(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the standard normal CDF Φ(x) = P(Z ≤ x) for Z ~ N(0, 1).

    Implemented via the error function: Φ(x) = (1 + erf(x / √2)) / 2.

    Used in the continuous-diffusion reconstruction loss to integrate the
    Gaussian likelihood over integer-valued bins [i − 0.5, i + 0.5] for
    discrete-valued features (e.g. atom types encoded as integers).

    Args:
        x: Input tensor of any shape.

    Returns:
        CDF values in (0, 1), same shape as x.
    """
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


def SNR(gamma: torch.Tensor) -> torch.Tensor:
    """Compute the signal-to-noise ratio (SNR) from the log-SNR γ.

    The continuous noise schedule is parametrized by the log-SNR γ, where
    SNR_t = α²_t / σ²_t = exp(−γ_t). Larger γ means more noise and a lower
    SNR.

    Args:
        gamma: Log-SNR values of any shape.

    Returns:
        SNR values (α²/σ²) of the same shape as gamma.
    """
    return torch.exp(-gamma)


def inflate_batch_array(
    array: torch.Tensor, target_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    """Reshape a batch-only tensor to broadcast against a multi-dimensional target.

    Given an array of shape (batch_size,) (or (batch_size, 1, …, 1)), returns
    a view of shape (batch_size, 1, 1, …, 1) matching the rank of target_shape.
    This allows per-sample scalars (e.g. σ_t, α_t) to be multiplied
    element-wise with feature tensors of higher rank.

    Args:
        array: Tensor of shape (batch_size,) or (batch_size, 1, …, 1).
        target_shape: Shape of the tensor to broadcast against; only its rank
            is used — the actual sizes are replaced with 1s except the first.

    Returns:
        A view of array with shape (batch_size, 1, …, 1) matching the rank of
        target_shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target_shape) - 1)
    return array.view(target_shape)


def sigma(
    gamma: torch.Tensor, target_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    """Compute the noise level σ_t from the log-SNR γ_t, broadcast to target shape.

    In the variance-preserving (VP) parametrization: σ²_t = sigmoid(γ_t),
    so σ_t = √sigmoid(γ_t). Larger γ gives higher noise. The result is
    inflated to match the rank of target_shape for direct multiplication with
    feature tensors.

    Together with `alpha`, satisfies α²_t + σ²_t = 1.

    Args:
        gamma: Per-sample log-SNR values of shape (batch_size,).
        target_shape: Shape of the feature tensor to broadcast against.

    Returns:
        Noise level tensor of shape (batch_size, 1, …, 1) matching the rank
        of target_shape.
    """
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_shape)


def alpha(
    gamma: torch.Tensor, target_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    """Compute the signal level α_t from the log-SNR γ_t, broadcast to target shape.

    In the variance-preserving parametrization: α²_t = sigmoid(−γ_t),
    so α_t = √sigmoid(−γ_t). Together with `sigma`, satisfies α²_t + σ²_t = 1.
    The result is inflated to match the rank of target_shape.

    Args:
        gamma: Per-sample log-SNR values of shape (batch_size,).
        target_shape: Shape of the feature tensor to broadcast against.

    Returns:
        Signal level tensor of shape (batch_size, 1, …, 1) matching the rank
        of target_shape.
    """
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_shape)


def check_mask_correct(
    variables: list[torch.Tensor], node_mask: torch.Tensor
) -> None:
    """Assert that all non-empty tensors in a list are correctly masked.

    Iterates over variables and calls `assert_correctly_masked` on each
    non-empty tensor. Empty tensors (e.g. global feature y with shape (bs, 0))
    are skipped.

    Args:
        variables: List of feature tensors to validate.
        node_mask: Boolean or integer mask where 0 marks padding positions.
    """
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def check_tensor_same_size(*args: torch.Tensor) -> None:
    """Assert that all tensors share the same shape.

    Args:
        *args: Two or more tensors to compare. The first tensor's shape is
            used as the reference.

    Raises:
        AssertionError: If any tensor's shape differs from the first.
    """
    for arg in args[1:]:
        assert args[0].size() == arg.size()


def sigma_and_alpha_t_given_s(
    gamma_t: torch.Tensor,
    gamma_s: torch.Tensor,
    target_size: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the conditional noise and signal parameters for the transition t → s.

    Given the log-SNR values at times t > s, computes the parameters of the
    reverse conditional q(x_s | x_t, x_0) in the continuous VP-SDE:

        α_{t|s} = α_t / α_s
        σ²_{t|s} = 1 − α²_{t|s}

    σ²_{t|s} is computed stably as −expm1(softplus(γ_s) − softplus(γ_t)) to
    avoid catastrophic cancellation when t and s are close. These parameters
    are used during the reverse diffusion loop to compute the posterior mean
    and variance at each denoising step.

    Args:
        gamma_t: Log-SNR at time t, shape (batch_size,). Requires t > s
            (higher γ = more noise).
        gamma_s: Log-SNR at time s, shape (batch_size,).
        target_size: Shape of the feature tensor; used to inflate outputs for
            broadcasting.

    Returns:
        A tuple (sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s) where:
          - sigma2_t_given_s: Conditional variance σ²_{t|s}, inflated to
              (batch_size, 1, …, 1).
          - sigma_t_given_s: Conditional std σ_{t|s}, same shape.
          - alpha_t_given_s: Conditional signal ratio α_{t|s} = α_t / α_s,
              same shape.
    """
    sigma2_t_given_s = inflate_batch_array(
        -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_size
    )

    # alpha_t_given_s = alpha_t / alpha_s
    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = inflate_batch_array(alpha_t_given_s, target_size)

    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


def reverse_tensor(x: torch.Tensor) -> torch.Tensor:
    """Reverse a tensor along the batch (first) dimension.

    Used to flip diffusion chain visualizations from the noising order
    (x_0 → x_T) to the generation order (x_T → x_0).

    Args:
        x: Tensor of shape (batch_size, *dims).

    Returns:
        Tensor of the same shape with the first dimension reversed.
    """
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_feature_noise(
    X_size: torch.Size,
    E_size: torch.Size,
    y_size: torch.Size,
    node_mask: torch.Tensor,
) -> PlaceHolder:
    """Sample masked symmetric Gaussian noise for graph features (X, E, y).

    Generates independent standard normal noise for node features X, edge
    features E, and global features y, then enforces two invariants required
    by the graph structure:

    1. **Symmetry**: Edge noise is made symmetric (E[i,j] = E[j,i]) by
       sampling only the upper-triangular part and mirroring it, so that
       undirected-graph consistency is preserved.
    2. **Masking**: All noise tensors are zeroed at padded positions via
       node_mask.

    Args:
        X_size: Shape (bs, n, dx) for node feature noise.
        E_size: Shape (bs, n, n, de) for edge feature noise.
        y_size: Shape (bs, dy) for global feature noise.
        node_mask: Boolean mask of shape (bs, n); False marks padding.

    Returns:
        PlaceHolder with fields X, E, y containing masked noise tensors cast
        to the same dtype as node_mask.
    """
    # TODO: How to change this for the multi-gpu case?
    epsX = sample_gaussian(X_size)
    epsE = sample_gaussian(E_size)
    epsy = sample_gaussian(y_size)

    float_mask = node_mask.float()
    epsX = epsX.type_as(float_mask)
    epsE = epsE.type_as(float_mask)
    epsy = epsy.type_as(float_mask)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(epsE)
    indices = torch.triu_indices(row=epsE.size(1), col=epsE.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    epsE = epsE * upper_triangular_mask
    epsE = (epsE + torch.transpose(epsE, 1, 2))

    assert (epsE == torch.transpose(epsE, 1, 2)).all()

    return PlaceHolder(X=epsX, E=epsE, y=epsy).mask(node_mask)


def sample_normal(
    mu_X: torch.Tensor,
    mu_E: torch.Tensor,
    mu_y: torch.Tensor,
    sigma: torch.Tensor,
    node_mask: torch.Tensor,
) -> PlaceHolder:
    """Sample from a normal distribution with the given means and scalar std.

    Computes z = μ + σ · ε where ε is masked symmetric graph noise from
    `sample_feature_noise`. The same per-sample scalar σ is applied to all
    features but broadcast differently for each tensor rank: σ for X (rank 3),
    σ.unsqueeze(1) for E (rank 4), and σ.squeeze(1) for y (rank 2).

    Args:
        mu_X: Node feature means, shape (bs, n, dx).
        mu_E: Edge feature means, shape (bs, n, n, de).
        mu_y: Global feature means, shape (bs, dy).
        sigma: Per-sample noise std, shape broadcastable to (bs, 1, …, 1).
        node_mask: Boolean mask of shape (bs, n).

    Returns:
        PlaceHolder with sampled (X, E, y) tensors.
    """
    # TODO: change for multi-gpu case
    eps = sample_feature_noise(mu_X.size(), mu_E.size(), mu_y.size(), node_mask).type_as(mu_X)
    X = mu_X + sigma * eps.X
    E = mu_E + sigma.unsqueeze(1) * eps.E
    y = mu_y + sigma.squeeze(1) * eps.y
    return PlaceHolder(X=X, E=E, y=y)


def check_issues_norm_values(
    gamma: Callable[[torch.Tensor], torch.Tensor],
    norm_val1: float,
    norm_val2: float,
    num_stdevs: int = 8,
) -> None:
    """Validate that normalization values are compatible with the noise schedule.

    Checks that 1 / max(norm_val1, norm_val2) > num_stdevs · σ_0, where σ_0
    is the noise level at t=0. If this condition fails, the input normalization
    compresses feature values so much that the initial noise already dominates
    the signal range, making reconstruction from x_0 very difficult.

    Args:
        gamma: Callable noise schedule mapping a (1, 1) tensor of normalized
            time to a (1, 1) log-SNR scalar.
        norm_val1: First normalization denominator (e.g. for node features).
        norm_val2: Second normalization denominator (e.g. for edge features).
        num_stdevs: Safety margin in standard deviations. Raises an error if
            σ_0 · num_stdevs exceeds 1 / max(norm_val1, norm_val2).

    Raises:
        ValueError: If max(norm_val1, norm_val2) is too large relative to σ_0.
    """
    zeros = torch.zeros((1, 1))
    gamma_0 = gamma(zeros)
    sigma_0 = sigma(gamma_0, target_shape=zeros.size()).item()
    max_norm_value = max(norm_val1, norm_val2)
    if sigma_0 * num_stdevs > 1. / max_norm_value:
        raise ValueError(
            f'Value for normalization value {max_norm_value} probably too '
            f'large with sigma_0 {sigma_0:.5f} and '
            f'1 / norm_value = {1. / max_norm_value}')


def sample_discrete_features(
    probX: torch.Tensor,
    probE: torch.Tensor,
    node_mask: torch.Tensor,
) -> PlaceHolder:
    """Sample discrete graph features from categorical distributions.

    Draws node and edge categories from multinomial distributions defined by
    probX and probE. Before sampling:

    - Masked node rows (where node_mask is False) are set to a uniform
      distribution so that multinomial sampling does not error on zero-sum rows.
    - Masked and diagonal edge positions are also set to uniform.

    After sampling, edge categories are made symmetric by keeping only the
    upper-triangular part and mirroring it, preserving the undirected-graph
    invariant.

    Note: probX and probE are modified in-place at masked positions.

    Args:
        probX: Node class probabilities, shape (bs, n, dx_out).
        probE: Edge class probabilities, shape (bs, n, n, de_out).
        node_mask: Boolean mask of shape (bs, n); False marks padding nodes.

    Returns:
        PlaceHolder with:
          - X: Sampled node class indices, shape (bs, n).
          - E: Sampled symmetric edge class indices, shape (bs, n, n).
          - y: Empty tensor of shape (bs, 0), cast to the same type as X.
    """
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)                                  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)     # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)    # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)   # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


def compute_posterior_distribution(
    M: torch.Tensor,
    M_t: torch.Tensor,
    Qt_M: torch.Tensor,
    Qsb_M: torch.Tensor,
    Qtb_M: torch.Tensor,
) -> torch.Tensor:
    """Compute the posterior distribution p(x_s | x_t, x_0) for one feature type.

    Implements the closed-form posterior from the discrete diffusion forward
    process (Austin et al., 2021; Vignac et al., 2023):

        p(x_s | x_t, x_0) ∝ [x_t · Q_t^T] ⊙ [x_0 · Q_{s|0}]
                             ─────────────────────────────────
                                  x_0 · Q_{t|0} · x_t^T

    where Q_t is the one-step transition matrix from s to t, Q_{s|0} (Qsb)
    is the cumulative transition from 0 to s, and Q_{t|0} (Qtb) is the
    cumulative transition from 0 to t.

    Args:
        M: Clean features x_0, shape (bs, n, d) or (bs, n, n, d). Flattened
            to (bs, N, d) internally.
        M_t: Noisy features x_t at time t, same shape as M.
        Qt_M: One-step transition matrix Q_t, shape (bs, d, d).
        Qsb_M: Cumulative transition Q_{s|0}, shape (bs, d, d).
        Qtb_M: Cumulative transition Q_{t|0}, shape (bs, d, d).

    Returns:
        Posterior probabilities of shape (bs, N, d), where N = n for node
        features or n·n for edge features.
    """
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)        # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)    # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)      # (bs, d, d)

    left_term = M_t @ Qt_M_T   # (bs, N, d)
    right_term = M @ Qsb_M     # (bs, N, d)
    product = left_term * right_term    # (bs, N, d)

    denom = M @ Qtb_M     # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)   # (bs, N, d) * (bs, N, d) + sum = (bs, N)

    prob = product / denom.unsqueeze(-1)    # (bs, N, d)

    return prob


def compute_batched_over0_posterior_distribution(
    X_t: torch.Tensor,
    Qt: torch.Tensor,
    Qsb: torch.Tensor,
    Qtb: torch.Tensor,
) -> torch.Tensor:
    """Compute p(x_s | x_t, x_0) for all possible x_0 values in parallel.

    Unlike `compute_posterior_distribution`, which takes a single x_0, this
    function vectorizes over all d_0 possible discrete values of x_0 at once.
    The numerator for each (position, x_0, x_{t-1}) triple is:

        numerator[n, x_0, x_{t-1}] = (x_t · Q_t^T)[x_{t-1}] · Q_{s|0}[x_0, x_{t-1}]

    and the denominator is Q_{t|0}[x_0, :] · x_t^T.

    Used during sampling when the model predicts a full distribution over x_0
    rather than a single clean state, allowing marginalization over all
    possible clean states.

    Args:
        X_t: Noisy features at time t, shape (bs, n, dt) or (bs, n, n, dt).
            Flattened to (bs, N, dt) internally.
        Qt: One-step transition matrix, shape (bs, d_{t-1}, dt).
        Qsb: Cumulative transition Q_{s|0}, shape (bs, d_0, d_{t-1}).
        Qtb: Cumulative transition Q_{t|0}, shape (bs, d_0, dt).

    Returns:
        Unnormalized posterior weights of shape (bs, N, d_0, d_{t-1}), giving
        relative probabilities over x_{t-1} for each (sample, position, x_0)
        triple.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out


def mask_distributions(
    true_X: torch.Tensor,
    true_E: torch.Tensor,
    pred_X: torch.Tensor,
    pred_E: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Set padded positions to a fixed distribution and renormalize.

    Sets masked nodes and edges to the one-hot distribution [1, 0, 0, …] in
    both the true and predicted distributions so that padding does not
    contribute to the KL loss. Diagonal edge entries (self-loops) are also
    excluded. Finally, adds ε = 1e-7 and renormalizes all distributions to
    sum to 1 along the feature axis, preventing log(0) in the KL computation.

    Args:
        true_X: True node distributions, shape (bs, n, dx_out).
        true_E: True edge distributions, shape (bs, n, n, de_out).
        pred_X: Predicted node distributions, shape (bs, n, dx_out).
        pred_E: Predicted edge distributions, shape (bs, n, n, de_out).
        node_mask: Boolean mask of shape (bs, n); False marks padding.

    Returns:
        Tuple (true_X, true_E, pred_X, pred_E) with masked and renormalized
        distributions; all shapes unchanged.
    """
    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1.
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X[~node_mask] = row_X
    pred_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    true_X = true_X + 1e-7
    pred_X = pred_X + 1e-7
    true_E = true_E + 1e-7
    pred_E = pred_E + 1e-7

    true_X = true_X / torch.sum(true_X, dim=-1, keepdim=True)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)
    true_E = true_E / torch.sum(true_E, dim=-1, keepdim=True)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    return true_X, true_E, pred_X, pred_E


def posterior_distributions(
    X: torch.Tensor,
    E: torch.Tensor,
    y: torch.Tensor,
    X_t: torch.Tensor,
    E_t: torch.Tensor,
    y_t: torch.Tensor,
    Qt: PlaceHolder,
    Qsb: PlaceHolder,
    Qtb: PlaceHolder,
) -> PlaceHolder:
    """Compute posterior distributions p(x_s | x_t, x_0) for all graph features.

    Applies `compute_posterior_distribution` independently to node features X
    and edge features E using their respective transition matrices. Global
    features y are passed through unchanged (no discrete diffusion on y).

    Args:
        X: Clean node features x_0, shape (bs, n, dx).
        E: Clean edge features x_0, shape (bs, n, n, de).
        y: Global features, shape (bs, dy). Passed through to output.
        X_t: Noisy node features x_t at time t, shape (bs, n, dx).
        E_t: Noisy edge features x_t at time t, shape (bs, n, n, de).
        y_t: Noisy global features at time t, shape (bs, dy).
        Qt: PlaceHolder holding one-step transition matrices Qt.X and Qt.E.
        Qsb: PlaceHolder holding cumulative transition matrices Q_{s|0}.X
            and Q_{s|0}.E.
        Qtb: PlaceHolder holding cumulative transition matrices Q_{t|0}.X
            and Q_{t|0}.E.

    Returns:
        PlaceHolder with:
          - X: Node posterior probabilities, shape (bs, n, dx).
          - E: Edge posterior probabilities, shape (bs, n·n, de).
          - y: Global features from y_t, passed through unchanged.
    """
    prob_X = compute_posterior_distribution(M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)   # (bs, n, dx)
    prob_E = compute_posterior_distribution(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)   # (bs, n * n, de)

    return PlaceHolder(X=prob_X, E=prob_E, y=y_t)


def sample_discrete_feature_noise(
    limit_dist: PlaceHolder,
    node_mask: torch.Tensor,
) -> PlaceHolder:
    """Sample the initial noisy graph x_T from the diffusion limit distribution.

    Draws node and edge categories from the stationary (limit) distribution of
    the discrete diffusion chain, used to initialize the reverse sampling loop.
    For a uniform schedule this is the uniform distribution; for marginal-based
    schedules it matches empirical class frequencies.

    Edge samples are made symmetric by taking only the upper-triangular part
    and mirroring it. All features are then masked via node_mask and one-hot
    encoded.

    Args:
        limit_dist: PlaceHolder containing 1-D marginal distributions:
            - X: Node class marginals, shape (dx,).
            - E: Edge class marginals, shape (de,).
            - y: Global marginals (unused; an empty tensor is returned for y).
        node_mask: Boolean mask of shape (bs, n_max); False marks padding.

    Returns:
        PlaceHolder with one-hot encoded features:
          - X: Node features, shape (bs, n_max, dx).
          - E: Symmetric edge features, shape (bs, n_max, n_max, de).
          - y: Empty tensor of shape (bs, 0).
    """
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    y_limit = limit_dist.y[None, :].expand(bs, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = (U_E + torch.transpose(U_E, 1, 2))

    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)
