"""Tests for DiscreteDenoisingDiffusion in src/model.py.

Captures key behavioral invariants for regression detection during refactoring:
limit_dist normalisation, apply_noise output contract, compute_extra_data
structure, and forward shape pass-through.
"""

import sys
import os

# diffusion_model_discrete.py uses bare imports (e.g. `from diffusion.noise_schedule import ...`)
# that resolve relative to src/, not the repo root — add src/ to the path before importing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from types import SimpleNamespace

from src.model import DiscreteDenoisingDiffusion
from src.diffusion.distributions import DistributionNodes
from src import utils


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BS, N = 2, 4        # batch size, number of nodes
XDIM, EDIM = 3, 2  # node / edge feature classes (also output dims)
T = 10              # diffusion steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ZeroFeatures:
    """Extra-features stub returning zero-width tensors — adds no features.

    Used in place of real extra_features and domain_features objects so that
    compute_extra_data's concatenation logic runs without contributing any
    additional dimensions to X, E, or y.
    """

    def __call__(self, noisy_data: dict) -> utils.PlaceHolder:
        """Return a PlaceHolder with zero-width feature tensors for every field.

        Args:
            noisy_data: Dict produced by ``apply_noise``, must contain ``X_t``
              so that batch size and node count can be inferred.

        Returns:
            PlaceHolder with X of shape ``(bs, n, 0)``, E of shape
            ``(bs, n, n, 0)``, and y of shape ``(bs, 0)``.
        """
        bs = noisy_data["X_t"].size(0)
        n = noisy_data["X_t"].size(1)
        return utils.PlaceHolder(
            X=torch.zeros(bs, n, 0),
            E=torch.zeros(bs, n, n, 0),
            y=torch.zeros(bs, 0),
        )


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a small batch of clean graph data with all-class-0 one-hot features.

    All nodes are assigned to class 0 and all edges to class 0.  The node mask
    marks every node as valid.  This gives a deterministic, easy-to-reason-about
    starting point for noise tests without requiring real dataset objects.

    Returns:
        Tuple ``(X, E, y, node_mask)`` where X has shape ``(BS, N, XDIM)``,
        E has shape ``(BS, N, N, EDIM)``, y has shape ``(BS, 0)``, and
        node_mask has shape ``(BS, N)`` with all entries ``True``.
    """
    X = torch.zeros(BS, N, XDIM)
    X[:, :, 0] = 1.0
    E = torch.zeros(BS, N, N, EDIM)
    E[:, :, :, 0] = 1.0
    y = torch.zeros(BS, 0)
    node_mask = torch.ones(BS, N, dtype=torch.bool)
    return X, E, y, node_mask


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> DiscreteDenoisingDiffusion:
    """Minimal DiscreteDenoisingDiffusion: uniform transition, 1-layer transformer."""
    torch.manual_seed(0)

    cfg = SimpleNamespace(
        general=SimpleNamespace(
            name="test",
            log_every_steps=1,
            number_chain_steps=5,
            sample_every_val=100,
            samples_to_generate=1,
            samples_to_save=1,
            chains_to_save=0,
            final_model_samples_to_generate=1,
            final_model_samples_to_save=1,
            final_model_chains_to_save=0,
        ),
        model=SimpleNamespace(
            diffusion_steps=T,
            diffusion_noise_schedule="cosine",
            transition="uniform",
            lambda_train=[1.0, 1.0],
            n_layers=1,
            hidden_mlp_dims={"X": 8, "E": 4, "y": 4},
            # dy=6 is the internal transformer y-dim, distinct from input/output dims
            hidden_dims={"dx": 8, "de": 4, "dy": 6, "n_head": 2, "dim_ffX": 8, "dim_ffE": 4},
        ),
        train=SimpleNamespace(lr=1e-4, weight_decay=0.0, batch_size=2),
    )

    nodes_dist = DistributionNodes({3: 5, 4: 3})
    dataset_infos = SimpleNamespace(
        # input_dims['y']=1: 0 raw graph labels + 1 timestep t appended by compute_extra_data
        input_dims={"X": XDIM, "E": EDIM, "y": 1},
        output_dims={"X": XDIM, "E": EDIM, "y": 1},
        nodes_dist=nodes_dist,
    )

    return DiscreteDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=None,
        sampling_metrics=None,
        visualization_tools=None,
        extra_features=_ZeroFeatures(),
        domain_features=_ZeroFeatures(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_uniform_limit_dist_is_normalized(model: DiscreteDenoisingDiffusion) -> None:
    """Under uniform transition, both limit distributions must sum to 1."""
    assert model.limit_dist.X.sum().item() == pytest.approx(1.0)
    assert model.limit_dist.E.sum().item() == pytest.approx(1.0)


def test_apply_noise_returns_expected_keys(model: DiscreteDenoisingDiffusion) -> None:
    """apply_noise must return a dict with exactly these nine keys for downstream use."""
    model.eval()
    X, E, y, node_mask = _make_inputs()
    noisy = model.apply_noise(X, E, y, node_mask)
    expected = {"t_int", "t", "beta_t", "alpha_s_bar", "alpha_t_bar", "X_t", "E_t", "y_t", "node_mask"}
    assert set(noisy.keys()) == expected


def test_apply_noise_X_t_shape_matches_input(model: DiscreteDenoisingDiffusion) -> None:
    """Noisy node features X_t must have the same shape as the clean input X."""
    model.eval()
    X, E, y, node_mask = _make_inputs()
    noisy = model.apply_noise(X, E, y, node_mask)
    assert noisy["X_t"].shape == X.shape


def test_apply_noise_E_t_is_symmetric(model: DiscreteDenoisingDiffusion) -> None:
    """Edge features must be symmetric: E_t[b, i, j] == E_t[b, j, i] (undirected graphs)."""
    model.eval()
    X, E, y, node_mask = _make_inputs()
    noisy = model.apply_noise(X, E, y, node_mask)
    assert torch.equal(noisy["E_t"], noisy["E_t"].transpose(1, 2))


def test_apply_noise_timestep_in_range(model: DiscreteDenoisingDiffusion) -> None:
    """During eval (lowest_t=1), t_int must be in [1, T] — never 0."""
    model.eval()
    X, E, y, node_mask = _make_inputs()
    noisy = model.apply_noise(X, E, y, node_mask)
    t_int = noisy["t_int"]
    assert (t_int >= 1).all() and (t_int <= T).all()


def test_compute_extra_data_t_is_appended(model: DiscreteDenoisingDiffusion) -> None:
    """compute_extra_data appends the normalised timestep t as the last y feature."""
    X, E, y, node_mask = _make_inputs()
    t_float = torch.full((BS, 1), 0.5)
    noisy_data = {
        "X_t": X, "E_t": E, "y_t": y,
        "t": t_float, "node_mask": node_mask,
    }
    extra = model.compute_extra_data(noisy_data)
    # extra.y = cat(extra_features.y=0-wide, domain_features.y=0-wide, t), so last col is t
    assert extra.y[:, -1:].allclose(t_float)


def test_forward_output_shapes(model: DiscreteDenoisingDiffusion) -> None:
    """forward must return a PlaceHolder whose X/E/y match the configured output dims."""
    model.eval()
    X, E, y, node_mask = _make_inputs()
    noisy_data = model.apply_noise(X, E, y, node_mask)
    extra_data = model.compute_extra_data(noisy_data)
    pred = model.forward(noisy_data, extra_data, node_mask)
    assert pred.X.shape == (BS, N, XDIM)
    assert pred.E.shape == (BS, N, N, EDIM)
    assert pred.y.shape == (BS, 1)


def test_uniform_limit_dist_values_are_equal(model: DiscreteDenoisingDiffusion) -> None:
    """Under uniform transition every class gets identical probability 1/num_classes."""
    assert model.limit_dist.X.allclose(torch.full((XDIM,), 1.0 / XDIM))
    assert model.limit_dist.E.allclose(torch.full((EDIM,), 1.0 / EDIM))


def test_apply_noise_X_t_is_one_hot(model: DiscreteDenoisingDiffusion) -> None:
    """X_t must be one-hot: each node row sums to 1 and contains only 0s and 1s."""
    model.eval()
    X, E, y, node_mask = _make_inputs()
    noisy = model.apply_noise(X, E, y, node_mask)
    X_t = noisy["X_t"].float()
    assert X_t.sum(dim=-1).allclose(torch.ones(BS, N))
    assert X_t.max().item() == pytest.approx(1.0)
    assert X_t.min().item() == pytest.approx(0.0)


def test_apply_noise_masked_nodes_zeroed(model: DiscreteDenoisingDiffusion) -> None:
    """Masked-out nodes must have a zero X_t row after apply_noise."""
    model.eval()
    X, E, y, node_mask = _make_inputs()
    node_mask[0, -1] = False  # mask the last node of the first graph
    noisy = model.apply_noise(X, E, y, node_mask)
    assert torch.equal(noisy["X_t"][0, -1], torch.zeros(XDIM))


def test_apply_noise_t_is_t_int_over_T(model: DiscreteDenoisingDiffusion) -> None:
    """Normalised time t must equal t_int / T exactly."""
    model.eval()
    X, E, y, node_mask = _make_inputs()
    noisy = model.apply_noise(X, E, y, node_mask)
    assert noisy["t"].allclose(noisy["t_int"] / T)


def test_apply_noise_training_mode_allows_t0(model: DiscreteDenoisingDiffusion) -> None:
    """During training (lowest_t=0), t_int=0 is reachable — confirmed over many draws."""
    model.train()
    X, E, y, node_mask = _make_inputs()
    seen_zero = False
    for _ in range(200):
        noisy = model.apply_noise(X, E, y, node_mask)
        if (noisy["t_int"] == 0).any():
            seen_zero = True
            break
    assert seen_zero, "t_int=0 should be reachable in training mode (lowest_t=0)"


def test_kl_prior_shape_and_sign(model: DiscreteDenoisingDiffusion) -> None:
    """kl_prior returns a per-sample scalar (shape ``(bs,)``) and KL divergence is always >= 0."""
    model.eval()
    X, E, y, node_mask = _make_inputs()
    kl = model.kl_prior(X, E, node_mask)
    assert kl.shape == (BS,)
    assert (kl >= 0).all()


def test_sample_p_zs_given_zt_E_symmetric_and_shapes(model: DiscreteDenoisingDiffusion) -> None:
    """sample_p_zs_given_zt must return E_s that is symmetric and shapes that match input."""
    model.eval()
    X, E, y, node_mask = _make_inputs()
    s = torch.full((BS, 1), 0.4)
    t = torch.full((BS, 1), 0.5)
    out_one_hot, _ = model.sample_p_zs_given_zt(s, t, X, E, torch.zeros(BS, 0), node_mask)
    assert out_one_hot.X.shape == X.shape
    assert out_one_hot.E.shape == E.shape
    assert torch.equal(out_one_hot.E, out_one_hot.E.transpose(1, 2))
