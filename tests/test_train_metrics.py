"""Tests for TrainLoss (continuous MSE) and TrainLossDiscrete (discrete cross-entropy).

Covers the two training-loss modules used by DiGress during diffusion model
training.  ``TrainLoss`` supervises ε-prediction with MSE; ``TrainLossDiscrete``
supervises x_0-prediction with weighted cross-entropy.  Both classes accumulate
per-batch statistics and expose ``log_epoch_metrics()`` for PyTorch Lightning's
``on_train_epoch_end`` hook.
"""

import pytest
import torch
from src.metrics.train_metrics import TrainLoss, TrainLossDiscrete


# ---------------------------------------------------------------------------
# TrainLoss (continuous / MSE)
# ---------------------------------------------------------------------------

def test_train_loss_forward_zero_when_preds_match() -> None:
    """Return zero loss when predicted noise equals true noise for all components."""
    loss = TrainLoss()
    X = torch.ones(2, 3, 4)
    E = torch.ones(2, 3, 3, 5)
    y = torch.ones(2, 6)
    result = loss(X, E, y, X, E, y, log=False)
    assert result.item() == pytest.approx(0.0)


def test_train_loss_forward_positive_mse_on_mismatch() -> None:
    """Return positive loss when node predictions differ from targets."""
    loss = TrainLoss()
    X_pred = torch.ones(2, 3, 4)
    X_true = torch.zeros(2, 3, 4)
    E = torch.ones(2, 3, 3, 5)
    y = torch.ones(2, 6)
    result = loss(X_pred, E, y, X_true, E, y, log=False)
    assert result.item() > 0.0


def test_train_loss_forward_skips_empty_node_tensor() -> None:
    """Set node MSE to 0.0 when X is empty; still compute edge MSE normally.

    When ``true_epsX.numel() == 0`` the node component is skipped entirely,
    so the total loss equals only the edge MSE (1.0 here) plus the y MSE (0.0).
    """
    loss = TrainLoss()
    X_empty = torch.empty(0)
    E_pred = torch.ones(2, 3, 3, 5)
    E_true = torch.zeros(2, 3, 3, 5)   # mismatch → edge MSE == 1.0
    y = torch.zeros(2, 6)              # pred == true → y MSE == 0.0
    result = loss(X_empty, E_pred, y, X_empty, E_true, y, log=False)
    assert result.item() == pytest.approx(1.0)


def test_train_loss_log_epoch_metrics_returns_expected_keys() -> None:
    """Return exactly the three expected metric keys after a forward pass."""
    loss = TrainLoss()
    X = torch.ones(2, 3, 4)
    E = torch.ones(2, 3, 3, 5)
    y = torch.ones(2, 6)
    loss(X, E, y, X, E, y, log=False)
    metrics = loss.log_epoch_metrics()
    assert set(metrics.keys()) == {
        "train_epoch/epoch_X_mse",
        "train_epoch/epoch_E_mse",
        "train_epoch/epoch_y_mse",
    }


def test_train_loss_log_epoch_metrics_minus_one_before_data() -> None:
    """Return -1 sentinel for every component when no forward pass has occurred.

    ``total == 0`` in the torchmetrics accumulator triggers the ``-1`` fallback,
    signalling that no samples have been seen yet for that component.
    """
    loss = TrainLoss()
    metrics = loss.log_epoch_metrics()
    assert metrics["train_epoch/epoch_X_mse"] == -1
    assert metrics["train_epoch/epoch_E_mse"] == -1
    assert metrics["train_epoch/epoch_y_mse"] == -1


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
    result = loss(pred_X, pred_E, pred_y, true_X, true_E, true_y, log=False)
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

    result_no_edge = TrainLossDiscrete(lambda_train=[0.0, 0.0])(
        pred_X, pred_E, pred_y, true_X, true_E, true_y, log=False
    )
    result_with_edge = TrainLossDiscrete(lambda_train=[1.0, 0.0])(
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
