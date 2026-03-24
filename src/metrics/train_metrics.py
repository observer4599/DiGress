"""Weighted cross-entropy training loss for the DiGress discrete diffusion model."""

import torch.nn as nn
from torch import Tensor
from src.metrics.abstract_metrics import CrossEntropyMetric


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
    ) -> tuple[Tensor, dict[str, Tensor | float] | None]:
        """Compute the weighted cross-entropy loss for one training batch.

        X and E are flattened to 2-D and all-zero (padding) rows are removed
        before computing their cross-entropy losses.  ``y`` is a per-graph
        global feature with no padding, so it is passed directly.  A component
        whose rows are all padding contributes ``0.0`` to the total loss.

        Args:
            masked_pred_X: Predicted node-type logits, shape ``(bs, n, d_X)``.
            masked_pred_E: Predicted edge-type logits, shape ``(bs, n, n, d_E)``.
            pred_y: Predicted global feature logits, shape ``(bs, d_y)``.
            true_X: One-hot ground-truth node types, shape ``(bs, n, d_X)``.
            true_E: One-hot ground-truth edge types, shape ``(bs, n, n, d_E)``.
            true_y: One-hot ground-truth global features, shape ``(bs, d_y)``.
            log: If ``True``, ``to_log`` contains ``train_loss/batch_CE``,
                ``train_loss/X_CE``, ``train_loss/E_CE``, and
                ``train_loss/y_CE``; components with no valid rows are ``-1``.

        Returns:
            ``(loss, to_log)`` where ``loss`` is the scalar
            ``loss_X + λ_E * loss_E + λ_y * loss_y`` and ``to_log`` is a
            dict of batch metrics when ``log=True``, or ``None`` otherwise.
        """
        true_X = true_X.flatten(0, -2)          # (bs * n, dx)
        true_E = true_E.flatten(0, -2)          # (bs * n * n, de)
        masked_pred_X = masked_pred_X.flatten(0, -2)
        masked_pred_E = masked_pred_E.flatten(0, -2)

        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        loss_X = self.node_loss(masked_pred_X[mask_X], true_X[mask_X]) if mask_X.any() else 0.0
        loss_E = self.edge_loss(masked_pred_E[mask_E], true_E[mask_E]) if mask_E.any() else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        loss = loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y
        to_log = None
        if log:
            to_log = {
                "train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                "train_loss/X_CE": self.node_loss.compute() if mask_X.any() else -1.0,
                "train_loss/E_CE": self.edge_loss.compute() if mask_E.any() else -1.0,
                "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1.0,
            }
        return loss, to_log

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
            Dict with keys ``train_epoch/X_CE``, ``train_epoch/E_CE``, and
            ``train_epoch/y_CE``.
        """
        components = {"X": self.node_loss, "E": self.edge_loss, "y": self.y_loss}
        return {
            f"train_epoch/{k}_CE": m.compute() if m.total_samples > 0 else -1.0
            for k, m in components.items()
        }
