import pytest
import torch
from src.models.layers import Xtoy, Etoy, masked_softmax


# --- Xtoy ---

def test_xtoy_output_shape():
    bs, n, dx, dy = 2, 5, 8, 16
    model = Xtoy(dx, dy)
    X = torch.randn(bs, n, dx)
    out = model(X)
    assert out.shape == (bs, dy)


def test_xtoy_single_node():
    """With n=2 (minimum for valid std), module should not crash."""
    model = Xtoy(4, 8)
    X = torch.randn(2, 2, 4)
    out = model(X)
    assert out.shape == (2, 8)


# --- Etoy ---

def test_etoy_output_shape():
    bs, n, de, dy = 2, 5, 4, 16
    model = Etoy(de, dy)
    E = torch.randn(bs, n, n, de)
    out = model(E)
    assert out.shape == (bs, dy)


# --- masked_softmax ---

def test_masked_softmax_zeros_masked_entries():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1, 1, 0]])
    out = masked_softmax(x, mask, dim=-1)
    assert out[0, 2].item() == pytest.approx(0.0, abs=1e-6)


def test_masked_softmax_all_zero_mask_returns_input():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.zeros(1, 3)
    out = masked_softmax(x, mask, dim=-1)
    assert torch.allclose(out, x)


def test_masked_softmax_full_mask_sums_to_one():
    x = torch.randn(3, 6)
    mask = torch.ones(3, 6)
    out = masked_softmax(x, mask, dim=-1)
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-6)
