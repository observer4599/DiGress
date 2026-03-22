"""Tests for src/utils.py.

Covers the key behavioral contracts of normalize, unnormalize, encode_no_edge,
and PlaceHolder.mask — the functions most likely to silently break during
refactoring.

Test groups:

- ``normalize``: scaling, forced-zero diagonal, and roundtrip with unnormalize.
- ``encode_no_edge``: absent-edge marker and diagonal zeroing.
- ``PlaceHolder.mask``: padding zeroing for nodes, edges, and collapsed indices.
"""

import torch
from src.utils import normalize, unnormalize, encode_no_edge, PlaceHolder


# --- normalize ---

def test_normalize_scales_values() -> None:
    # (X - bias) / value is applied element-wise to X, E, and y
    X = torch.ones(2, 3, 4)
    E = torch.zeros(2, 3, 3, 5)  # zero diagonal avoids the forced-zero interference
    y = torch.ones(2, 6)
    node_mask = torch.ones(2, 3, dtype=torch.bool)
    result = normalize(X, E, y, norm_values=[2.0, 1.0, 4.0], norm_biases=[0.0, 0.0, 0.0], node_mask=node_mask)
    assert torch.allclose(result.X, torch.full((2, 3, 4), 0.5))
    assert torch.allclose(result.y, torch.full((2, 6), 0.25))


def test_normalize_zeros_E_diagonal() -> None:
    # diagonal entries of E are always forced to 0 after normalization
    E = torch.ones(2, 3, 3, 5)
    X = torch.ones(2, 3, 4)
    y = torch.ones(2, 6)
    node_mask = torch.ones(2, 3, dtype=torch.bool)
    result = normalize(X, E, y, norm_values=[1.0, 1.0, 1.0], norm_biases=[0.0, 0.0, 0.0], node_mask=node_mask)
    for i in range(3):
        assert result.E[:, i, i, :].sum() == 0


def test_normalize_unnormalize_roundtrip() -> None:
    # normalize followed by unnormalize recovers the original tensors
    X = torch.ones(2, 3, 4) * 3.0
    E = torch.ones(2, 3, 3, 5) * 2.0
    diag = torch.eye(3, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1)
    E[diag] = 0  # pre-zero diagonal so roundtrip holds for all elements
    y = torch.ones(2, 6) * 5.0
    node_mask = torch.ones(2, 3, dtype=torch.bool)
    norm_values = [2.0, 4.0, 3.0]
    norm_biases = [1.0, 0.0, 2.0]  # E bias must be 0: normalize zeros E diagonal,
                                    # but unnormalize adds bias back, so bias!=0 breaks roundtrip for diagonal
    ph_norm = normalize(X.clone(), E.clone(), y.clone(), norm_values, norm_biases, node_mask)
    ph_back = unnormalize(ph_norm.X, ph_norm.E, ph_norm.y, norm_values, norm_biases, node_mask)
    assert torch.allclose(ph_back.X, X, atol=1e-5)
    assert torch.allclose(ph_back.E, E, atol=1e-5)
    assert torch.allclose(ph_back.y, y, atol=1e-5)


# --- encode_no_edge ---

def test_encode_no_edge_marks_absent_edges() -> None:
    # edges whose feature vector is all-zero get a 1 in the first channel
    E = torch.zeros(1, 3, 3, 4)
    E[0, 0, 1, 2] = 1.0  # only edge (0,1) is present
    result = encode_no_edge(E)
    assert result[0, 0, 2, 0] == 1.0   # (0,2) is absent → first channel marked
    assert result[0, 0, 1, 0] == 0.0   # (0,1) is present → no marker


def test_encode_no_edge_zeros_diagonal() -> None:
    # diagonal entries are always zeroed regardless of their original values
    E = torch.ones(1, 3, 3, 4)
    result = encode_no_edge(E)
    for i in range(3):
        assert result[0, i, i, :].sum() == 0


# --- PlaceHolder.mask ---

def test_placeholder_mask_zeroes_padded_node_features() -> None:
    # node features for masked-out (padding) nodes are set to 0
    X = torch.ones(2, 3, 4)
    E = torch.zeros(2, 3, 3, 5)
    y = torch.zeros(2, 6)
    node_mask = torch.tensor([[True, True, True], [True, True, False]])
    ph = PlaceHolder(X=X, E=E, y=y).mask(node_mask)
    assert ph.X[1, 2].sum() == 0        # padded node is zeroed
    assert ph.X[1, 0].sum() > 0         # real node is untouched


def test_placeholder_mask_zeroes_edges_to_padded_nodes() -> None:
    # edges connecting to a padded node are zeroed in both directions
    X = torch.ones(2, 3, 4)
    E = torch.ones(2, 3, 3, 5)
    y = torch.zeros(2, 6)
    node_mask = torch.tensor([[True, True, True], [True, True, False]])
    ph = PlaceHolder(X=X, E=E, y=y).mask(node_mask)
    assert ph.E[1, 0, 2, :].sum() == 0  # edge from real to padded node
    assert ph.E[1, 2, 0, :].sum() == 0  # edge from padded to real node
    assert ph.E[1, 0, 1, :].sum() > 0   # edge between real nodes is kept


def test_placeholder_mask_collapse_takes_argmax() -> None:
    # collapse=True replaces one-hot X and E with argmax indices; padded → -1
    X = torch.zeros(1, 3, 4)
    X[0, 0, 2] = 1.0   # node 0 → class 2
    X[0, 1, 3] = 1.0   # node 1 → class 3
    E = torch.zeros(1, 3, 3, 5)
    y = torch.zeros(1, 6)
    node_mask = torch.tensor([[True, True, False]])
    ph = PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse=True)
    assert ph.X[0, 0].item() == 2       # argmax for node 0
    assert ph.X[0, 1].item() == 3       # argmax for node 1
    assert ph.X[0, 2].item() == -1      # padded node → -1
