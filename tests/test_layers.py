"""Tests for utility layers in src.transformer_model.

Covers the three components — Xtoy, Etoy, and masked_softmax —
verifying output shapes, numerical contracts, and edge-case handling
(e.g. all-zero masks, minimum-node graphs).
"""

import pytest
import torch
from src.transformer_model import Xtoy, Etoy, masked_softmax


# --- Xtoy ---

def test_xtoy_output_shape():
    """Xtoy produces one global vector per graph in the batch."""
    bs, n, dx, dy = 2, 5, 8, 16
    model = Xtoy(dx, dy)
    X = torch.randn(bs, n, dx)
    out = model(X)
    assert out.shape == (bs, dy)


def test_xtoy_single_node():
    """Xtoy does not crash when n=2, the minimum for a well-defined std."""
    model = Xtoy(4, 8)
    X = torch.randn(2, 2, 4)
    out = model(X)
    assert out.shape == (2, 8)


# --- Etoy ---

def test_etoy_output_shape():
    """Etoy produces one global vector per graph in the batch."""
    bs, n, de, dy = 2, 5, 4, 16
    model = Etoy(de, dy)
    E = torch.randn(bs, n, n, de)
    out = model(E)
    assert out.shape == (bs, dy)


# --- masked_softmax ---

def test_masked_softmax_zeros_masked_entries():
    """Masked positions receive exactly zero probability after softmax."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1, 1, 0]])
    out = masked_softmax(x, mask, dim=-1)
    assert out[0, 2].item() == pytest.approx(0.0, abs=1e-6)


def test_masked_softmax_all_zero_mask_returns_input():
    """An all-zero mask (no valid positions) returns x unchanged to avoid NaN."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.zeros(1, 3)
    out = masked_softmax(x, mask, dim=-1)
    assert torch.allclose(out, x)


def test_masked_softmax_full_mask_sums_to_one():
    """With all positions unmasked, outputs form a valid probability distribution."""
    x = torch.randn(3, 6)
    mask = torch.ones(3, 6)
    out = masked_softmax(x, mask, dim=-1)
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-6)
