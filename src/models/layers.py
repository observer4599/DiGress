"""Utility layers shared across graph transformer components.

Provides two aggregation modules — Xtoy and Etoy — that pool node and
edge features into a graph-level (global) feature vector, and a
masked softmax helper for attention over variable-length graphs.

These modules are used inside each GraphTransformerLayer to update the
global feature y from the local node and edge representations.
"""

import torch
import torch.nn as nn


class Xtoy(nn.Module):
    """Aggregates node features into a graph-level feature vector.

    Summarises the node feature matrix X into a single vector per graph
    by computing four statistics over the node dimension (mean, min, max,
    std), concatenating them, and projecting to the global feature space.
    Used in GraphTransformerLayer to compute the node contribution to y.

    Args:
        dx: Dimensionality of each node feature vector.
        dy: Dimensionality of the output global feature vector.
    """

    def __init__(self, dx: int, dy: int) -> None:
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Aggregate node features to a graph-level vector.

        Args:
            X: Node feature matrix of shape (bs, n, dx), where bs is
                batch size and n is the number of nodes. Padding nodes
                should be zeroed out before calling.

        Returns:
            Global feature vectors of shape (bs, dy), one per graph.
        """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    """Aggregates edge features into a graph-level feature vector.

    Summarises the edge feature tensor E into a single vector per graph
    by computing four statistics over all (i, j) edge pairs (mean, min,
    max, std), concatenating them, and projecting to the global feature
    space. Used in GraphTransformerLayer to compute the edge contribution
    to y.

    Args:
        d: Dimensionality of each edge feature vector.
        dy: Dimensionality of the output global feature vector.
    """

    def __init__(self, d: int, dy: int) -> None:
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """Aggregate edge features to a graph-level vector.

        Args:
            E: Edge feature tensor of shape (bs, n, n, de), where bs is
                batch size and n is the number of nodes. Self-loop and
                padding entries should be zeroed out before calling.

        Returns:
            Global feature vectors of shape (bs, dy), one per graph.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(
    x: torch.Tensor,
    mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Apply softmax over positions indicated by a boolean mask.

    Positions where mask is 0 are set to -inf before the softmax, so
    they receive zero attention weight. Used in GraphTransformerLayer
    to exclude padding nodes from attention score normalisation.

    If the mask is entirely zero (no valid positions), x is returned
    unchanged to avoid NaN from softmax over all-inf inputs.

    Args:
        x: Logit tensor of any shape. The softmax and masking are
            applied element-wise; dim must be supplied via kwargs.
        mask: Boolean or 0/1 tensor broadcastable to x. Positions with
            value 0 are masked out.
        **kwargs: Passed directly to torch.softmax (e.g. dim=2).

    Returns:
        Tensor of the same shape as x with softmax applied only over
        unmasked positions.
    """
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)
