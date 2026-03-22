"""Tests for src.models.transformer_model.

Captures the current behavior of NodeEdgeBlock, XEyTransformerLayer, and
GraphTransformer for regression detection during refactoring.
"""

import pytest
import torch
from src.models.transformer_model import NodeEdgeBlock, XEyTransformerLayer, GraphTransformer


# --- Shared dimensions ---

BS, N = 2, 4
DX, DE, DY, N_HEAD = 8, 4, 6, 2


@pytest.fixture
def node_edge_block():
    torch.manual_seed(0)
    return NodeEdgeBlock(dx=DX, de=DE, dy=DY, n_head=N_HEAD)


@pytest.fixture
def xey_layer():
    torch.manual_seed(0)
    return XEyTransformerLayer(dx=DX, de=DE, dy=DY, n_head=N_HEAD, dim_ffX=16, dim_ffE=8, dim_ffy=16)


@pytest.fixture
def graph_transformer():
    torch.manual_seed(0)
    return GraphTransformer(
        n_layers=2,
        input_dims={'X': 3, 'E': 2, 'y': 4},
        hidden_mlp_dims={'X': 16, 'E': 8, 'y': 8},
        hidden_dims={'dx': DX, 'de': DE, 'dy': DY, 'n_head': N_HEAD, 'dim_ffX': 16, 'dim_ffE': 8},
        output_dims={'X': 3, 'E': 2, 'y': 4},
        act_fn_in=torch.nn.ReLU(),
        act_fn_out=torch.nn.ReLU(),
    )


# --- NodeEdgeBlock ---

def test_node_edge_block_output_shapes(node_edge_block):
    # forward returns (newX, newE, new_y) with the expected shapes
    X = torch.randn(BS, N, DX)
    E = torch.randn(BS, N, N, DE)
    y = torch.randn(BS, DY)
    node_mask = torch.ones(BS, N)
    newX, newE, new_y = node_edge_block(X, E, y, node_mask)
    assert newX.shape == (BS, N, DX)
    assert newE.shape == (BS, N, N, DE)
    assert new_y.shape == (BS, DY)


def test_node_edge_block_masked_nodes_are_zero(node_edge_block):
    # nodes with mask=0 produce zero output in newX and zero rows/cols in newE
    X = torch.randn(BS, N, DX)
    E = torch.randn(BS, N, N, DE)
    y = torch.randn(BS, DY)
    node_mask = torch.ones(BS, N)
    node_mask[0, -1] = 0.0  # mask last node in first graph
    newX, newE, _ = node_edge_block(X, E, y, node_mask)
    assert torch.allclose(newX[0, -1], torch.zeros(DX), atol=1e-6)
    assert torch.allclose(newE[0, -1, :], torch.zeros(N, DE), atol=1e-6)
    assert torch.allclose(newE[0, :, -1], torch.zeros(N, DE), atol=1e-6)


# --- XEyTransformerLayer ---

def test_xey_layer_output_shapes(xey_layer):
    # forward returns X, E, y with the same shapes as the inputs
    X = torch.randn(BS, N, DX)
    E = torch.randn(BS, N, N, DE)
    y = torch.randn(BS, DY)
    node_mask = torch.ones(BS, N)
    outX, outE, out_y = xey_layer(X, E, y, node_mask)
    assert outX.shape == (BS, N, DX)
    assert outE.shape == (BS, N, N, DE)
    assert out_y.shape == (BS, DY)


# --- GraphTransformer ---

def test_graph_transformer_output_shapes(graph_transformer):
    # forward returns a PlaceHolder with X, E, y of the expected output dims
    X = torch.randn(BS, N, 3)
    E = torch.randn(BS, N, N, 2)
    y = torch.randn(BS, 4)
    node_mask = torch.ones(BS, N)
    out = graph_transformer(X, E, y, node_mask)
    assert out.X.shape == (BS, N, 3)
    assert out.E.shape == (BS, N, N, 2)
    assert out.y.shape == (BS, 4)


def test_graph_transformer_edge_symmetry(graph_transformer):
    # output E is symmetric: E[b, i, j] == E[b, j, i]
    torch.manual_seed(1)
    X = torch.randn(BS, N, 3)
    E = torch.randn(BS, N, N, 2)
    y = torch.randn(BS, 4)
    node_mask = torch.ones(BS, N)
    out = graph_transformer(X, E, y, node_mask)
    assert torch.allclose(out.E, out.E.transpose(1, 2), atol=1e-6)


def test_graph_transformer_diagonal_zeroed(graph_transformer):
    # diagonal entries of output E are zero (self-loops are masked out)
    torch.manual_seed(1)
    X = torch.randn(BS, N, 3)
    E = torch.randn(BS, N, N, 2)
    y = torch.randn(BS, 4)
    node_mask = torch.ones(BS, N)
    out = graph_transformer(X, E, y, node_mask)
    for b in range(BS):
        for i in range(N):
            assert torch.allclose(out.E[b, i, i], torch.zeros(2), atol=1e-6)


def test_graph_transformer_masked_nodes_zeroed(graph_transformer):
    # masked-out nodes produce zero vectors in output X
    X = torch.randn(BS, N, 3)
    E = torch.randn(BS, N, N, 2)
    y = torch.randn(BS, 4)
    node_mask = torch.ones(BS, N)
    node_mask[1, -1] = 0.0
    out = graph_transformer(X, E, y, node_mask)
    assert torch.allclose(out.X[1, -1], torch.zeros(3), atol=1e-6)
