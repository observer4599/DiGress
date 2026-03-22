"""Training loss modules for the DiGress continuous and discrete diffusion models.

This module provides two ``nn.Module`` loss classes used during training:

- :class:`TrainLoss` — MSE loss for continuous-time DiGress, where the model
  predicts the noise (ε) added to node features, edge features, and global
  graph properties.
- :class:`TrainLossDiscrete` — weighted cross-entropy loss for discrete-time
  DiGress, where the model predicts the original clean graph (x_0-prediction)
  for node types, edge types, and global features.

Both classes accumulate per-batch statistics and expose ``reset()`` and
``log_epoch_metrics()`` to fit the PyTorch Lightning training-loop interface
used in ``diffusion_model.py`` and ``diffusion_model_discrete.py``.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import MeanSquaredError
from src.metrics.abstract_metrics import CrossEntropyMetric


class NodeMSE(MeanSquaredError):
    """MeanSquaredError specialised for node-feature noise predictions.

    A thin type-tagged wrapper so that node and edge MSE metrics are registered
    as distinct ``nn.Module`` children (and therefore appear separately in
    ``state_dict`` and on the correct device).
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    """MeanSquaredError specialised for edge-feature noise predictions.

    A thin type-tagged wrapper so that node and edge MSE metrics are registered
    as distinct ``nn.Module`` children (and therefore appear separately in
    ``state_dict`` and on the correct device).
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)


class TrainLoss(nn.Module):
    """MSE training loss for continuous-time DiGress (ε-prediction).

    Computes the sum of per-component mean squared errors between the model's
    predicted noise and the true noise injected during the forward diffusion
    process.  Three components are tracked independently:

    - **X** — node feature noise, shape ``(bs, n, d_X)``
    - **E** — edge feature noise, shape ``(bs, n, n, d_E)``
    - **y** — global graph feature noise, shape ``(bs, d_y)``

    Each component uses a separate torchmetrics accumulator so that per-epoch
    averages can be reported independently via :meth:`log_epoch_metrics`.

    Attributes:
        train_node_mse: Accumulates MSE for node noise predictions.
        train_edge_mse: Accumulates MSE for edge noise predictions.
        train_y_mse: Accumulates MSE for global feature noise predictions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(
        self,
        masked_pred_epsX: Tensor,
        masked_pred_epsE: Tensor,
        pred_y: Tensor,
        true_epsX: Tensor,
        true_epsE: Tensor,
        true_y: Tensor,
        log: bool,
    ) -> Tensor:
        """Compute the total noise-prediction MSE for one training batch.

        Each non-empty component contributes equally to the returned loss.
        If a tensor has zero elements (e.g. no global features), its component
        is treated as ``0.0`` and excluded from the sum.

        Args:
            masked_pred_epsX: Predicted node noise with padding masked out,
                shape ``(bs * n_valid, d_X)``.
            masked_pred_epsE: Predicted edge noise with padding masked out,
                shape ``(bs * n_valid * n_valid, d_E)``.
            pred_y: Predicted global feature noise, shape ``(bs, d_y)``.
            true_epsX: Ground-truth node noise, same shape as
                ``masked_pred_epsX``.
            true_epsE: Ground-truth edge noise, same shape as
                ``masked_pred_epsE``.
            true_y: Ground-truth global feature noise, same shape as
                ``pred_y``.
            log: If ``True``, a ``to_log`` dict is built containing
                ``train_loss/batch_mse`` and per-component epoch-level MSEs.
                The dict is currently unused (logging is handled by the caller).

        Returns:
            Scalar loss equal to ``mse_X + mse_E + mse_y``.
        """
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}

        return mse

    def reset(self) -> None:
        """Reset all metric accumulators at the start of a new epoch."""
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self) -> dict[str, Tensor | float]:
        """Return per-epoch MSE averages for all three components.

        Called by the training loop at the end of each epoch (via
        ``on_train_epoch_end``).  Components with no accumulated samples
        return ``-1`` as a sentinel value.

        Returns:
            Dict with keys ``train_epoch/epoch_X_mse``,
            ``train_epoch/epoch_E_mse``, and ``train_epoch/epoch_y_mse``.
        """
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        return to_log


class TrainLossDiscrete(nn.Module):
    """Weighted cross-entropy training loss for discrete-time DiGress.

    Computes cross-entropy between the model's predicted logits and the
    one-hot ground-truth graph components, with configurable per-component
    weights (``lambda_train``).  Three components are tracked independently:

    - **X** — node type logits, shape ``(bs, n, d_X)``
    - **E** — edge type logits, shape ``(bs, n, n, d_E)``
    - **y** — global graph feature logits, shape ``(bs, d_y)``

    Padding (all-zero rows) is removed from X and E before computing the
    loss, so the metric is not diluted by masked positions.

    The total loss is ``loss_X + λ_E * loss_E + λ_y * loss_y``, where
    ``lambda_train = [λ_E, λ_y]``.  Setting ``λ_E = λ_y = 1`` gives equal
    weight to all components.

    Attributes:
        node_loss: Cross-entropy accumulator for node type predictions.
        edge_loss: Cross-entropy accumulator for edge type predictions.
        y_loss: Cross-entropy accumulator for global feature predictions.
        lambda_train: Two-element list ``[λ_E, λ_y]`` controlling the
            relative weight of edge and global feature losses.
    """

    def __init__(self, lambda_train: list[float]) -> None:
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(
        self,
        masked_pred_X: Tensor,
        masked_pred_E: Tensor,
        pred_y: Tensor,
        true_X: Tensor,
        true_E: Tensor,
        true_y: Tensor,
        log: bool,
    ) -> Tensor:
        """Compute the weighted cross-entropy loss for one training batch.

        Flattens X and E to 2-D, removes all-zero (padding) rows, then
        computes per-component cross-entropy.  Empty tensors are treated as
        ``0.0`` and excluded from the sum.

        Args:
            masked_pred_X: Predicted node-type logits, shape ``(bs, n, d_X)``.
            masked_pred_E: Predicted edge-type logits, shape ``(bs, n, n, d_E)``.
            pred_y: Predicted global feature logits, shape ``(bs, d_y)``.
            true_X: One-hot ground-truth node types, shape ``(bs, n, d_X)``.
            true_E: One-hot ground-truth edge types, shape ``(bs, n, n, d_E)``.
            true_y: One-hot ground-truth global features, shape ``(bs, d_y)``.
            log: If ``True``, a ``to_log`` dict is built containing
                ``train_loss/batch_CE`` and per-component CE values.
                The dict is currently unused (logging is handled by the caller).

        Returns:
            Scalar loss equal to ``loss_X + λ_E * loss_E + λ_y * loss_y``.
        """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def reset(self) -> None:
        """Reset all metric accumulators at the start of a new epoch."""
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self) -> dict[str, Tensor | float]:
        """Return per-epoch cross-entropy averages for all three components.

        Called by the training loop at the end of each epoch (via
        ``on_train_epoch_end``).  Components with no accumulated samples
        return ``-1`` as a sentinel value.

        Returns:
            Dict with keys ``train_epoch/x_CE``, ``train_epoch/E_CE``, and
            ``train_epoch/y_CE``.
        """
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        return to_log
