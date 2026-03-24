"""Utility functions for discrete graph diffusion in DiGress.

Implements utilities for the discrete diffusion model from Vignac et al. (2023)
'DiGress: Discrete Denoising Diffusion Models for Graph Generation' (ICLR 2023,
https://openreview.net/forum?id=UaAD-Nu86WX).

Categorical graph features are perturbed via Markov transition matrices Q_t
(one-step), Q_{s|0} / Qsb (cumulative to s), Q_{t|0} / Qtb (cumulative to t).
The posterior p(x_s | x_t, x_0) is computed in closed form from these matrices.

Graphs are represented as batched (X, E, y) triples:
  - X: node features,  shape (bs, n, dx)
  - E: edge features,  shape (bs, n, n, de), symmetric (undirected graphs)
  - y: global features, shape (bs, dy)

Variable-length graphs are padded to the same n and gated by a boolean
node_mask of shape (bs, n).
"""

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
