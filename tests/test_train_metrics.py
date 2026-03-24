"""Tests for TrainLossDiscrete (discrete cross-entropy).

Covers the discrete training-loss module used by DiGress during diffusion
model training.  ``TrainLossDiscrete`` supervises x_0-prediction with weighted
cross-entropy, accumulating per-batch statistics and exposing
``log_epoch_metrics()`` for PyTorch Lightning's ``on_train_epoch_end`` hook.
"""

import pytest
import torch
from src.metrics.train_metrics import TrainLossDiscrete


# ---------------------------------------------------------------------------
# TrainLossDiscrete (discrete / cross-entropy)
# ---------------------------------------------------------------------------

def test_train_loss_discrete_forward_returns_scalar() -> None:
    """Return a 0-d scalar tensor regardless of batch or graph size."""
    torch.manual_seed(0)
    bs, n, dx, de = 2, 3, 4, 5
    pred_X = torch.randn(bs, n, dx)
    pred_E = torch.randn(bs, n, n, de)
    pred_y = torch.randn(bs, de)
    true_X = torch.zeros(bs, n, dx);     true_X[..., 0] = 1.0
    true_E = torch.zeros(bs, n, n, de);  true_E[..., 0] = 1.0
    true_y = torch.zeros(bs, de);        true_y[..., 0] = 1.0
    loss = TrainLossDiscrete(lambda_train=[1.0, 1.0])
    result, _ = loss(pred_X, pred_E, pred_y, true_X, true_E, true_y, log=False)
    assert result.shape == torch.Size([])


def test_train_loss_discrete_forward_lambda_scales_edge_loss() -> None:
    """Scale edge CE by lambda_train[0]; zero weight gives a lower total than weight 1.

    ``lambda_train = [λ_E, λ_y]``.  Setting ``λ_E = 1`` adds the edge
    cross-entropy to the total; ``λ_E = 0`` drops it, so the weighted total
    must be strictly smaller when edge predictions are imperfect.
    """
    torch.manual_seed(42)
    bs, n, dx, de = 2, 3, 4, 5
    pred_X = torch.randn(bs, n, dx)
    pred_E = torch.randn(bs, n, n, de)
    pred_y = torch.randn(bs, de)
    true_X = torch.zeros(bs, n, dx);     true_X[..., 0] = 1.0
    true_E = torch.zeros(bs, n, n, de);  true_E[..., 0] = 1.0
    true_y = torch.zeros(bs, de);        true_y[..., 0] = 1.0

    result_no_edge, _ = TrainLossDiscrete(lambda_train=[0.0, 0.0])(
        pred_X, pred_E, pred_y, true_X, true_E, true_y, log=False
    )
    result_with_edge, _ = TrainLossDiscrete(lambda_train=[1.0, 0.0])(
        pred_X, pred_E, pred_y, true_X, true_E, true_y, log=False
    )
    assert result_with_edge.item() > result_no_edge.item()


def test_train_loss_discrete_log_epoch_metrics_returns_expected_keys() -> None:
    """Return exactly the three expected CE metric keys after a forward pass."""
    torch.manual_seed(0)
    bs, n, dx, de = 2, 3, 4, 5
    pred_X = torch.randn(bs, n, dx)
    pred_E = torch.randn(bs, n, n, de)
    pred_y = torch.randn(bs, de)
    true_X = torch.zeros(bs, n, dx);     true_X[..., 0] = 1.0
    true_E = torch.zeros(bs, n, n, de);  true_E[..., 0] = 1.0
    true_y = torch.zeros(bs, de);        true_y[..., 0] = 1.0
    loss = TrainLossDiscrete(lambda_train=[1.0, 1.0])
    loss(pred_X, pred_E, pred_y, true_X, true_E, true_y, log=False)
    metrics = loss.log_epoch_metrics()
    assert set(metrics.keys()) == {
        "train_epoch/x_CE",
        "train_epoch/E_CE",
        "train_epoch/y_CE",
    }
