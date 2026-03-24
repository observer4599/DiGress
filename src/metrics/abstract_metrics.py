"""Abstract metric classes and batch-normalised torchmetrics for the DiGress diffusion model.

This module provides two kinds of building blocks:

1. **Abstract training-metric interfaces** (`TrainAbstractMetricsDiscrete`,
   `TrainAbstractMetrics`) — no-op base classes that define the interface
   expected by the diffusion training loop.  Both inherit from
   `_TrainAbstractMetricsBase`, which provides default no-op `reset` and
   `log_epoch_metrics` implementations.  Concrete subclasses (e.g.
   `TrainMolecularMetricsDiscrete`) override `forward`, `reset`, and
   `log_epoch_metrics` to track dataset-specific losses.

2. **Batch-normalised torchmetrics** — `Metric` subclasses that accumulate
   statistics across batches and return per-graph averages via `compute()`.
   All four concrete metrics (`SumExceptBatchMetric`, `SumExceptBatchMSE`,
   `SumExceptBatchKL`, `CrossEntropyMetric`) inherit from `_SumExceptBatchBase`,
   which holds the shared state and `compute` logic.  Each subclass only
   implements `update`.  All normalise by the number of *graphs* (first
   dimension) rather than the total number of elements, matching the loss
   formulation in the DiGress paper (Vignac et al., 2022).
"""

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric


class _TrainAbstractMetricsBase:
    """Shared no-op base for discrete and continuous training-metric interfaces.

    Provides default implementations of ``reset`` and ``log_epoch_metrics``
    that are identical across both variants.  Concrete subclasses override
    ``forward`` (and optionally the other methods) to add dataset-specific
    logic.
    """

    def __call__(self, *args, **kwargs) -> None:
        return self.forward(*args, **kwargs)

    def reset(self) -> None:
        """Reset all accumulated metric state at the start of a new epoch."""
        pass

    def log_epoch_metrics(self) -> tuple[None, None]:
        """Return per-epoch metric summaries after all batches have been seen.

        Returns:
            A ``(None, None)`` tuple; subclasses return ``(node_metric, edge_metric)``.
        """
        return None, None


class TrainAbstractMetricsDiscrete(_TrainAbstractMetricsBase):
    """No-op base class defining the training-metric interface for discrete diffusion.

    Concrete subclasses should override ``forward`` (and optionally ``reset``
    and ``log_epoch_metrics``) to track node- and edge-type prediction quality
    during discrete-time diffusion training.  The default implementation is
    used for non-molecular datasets where no additional per-epoch logging is
    required.
    """

    def forward(
        self,
        masked_pred_X: Tensor,
        masked_pred_E: Tensor,
        true_X: Tensor,
        true_E: Tensor,
        log: bool,
    ) -> None:
        """Accumulate per-batch training metrics.

        Args:
            masked_pred_X: Predicted node-type logits with padding masked out,
                shape ``(bs * n, d_X)``.
            masked_pred_E: Predicted edge-type logits with padding masked out,
                shape ``(bs * n * n, d_E)``.
            true_X: One-hot ground-truth node types, shape ``(bs * n, d_X)``.
            true_E: One-hot ground-truth edge types, shape ``(bs * n * n, d_E)``.
            log: Whether to log scalar values to the trainer logger this step.
        """
        pass


class TrainAbstractMetrics(_TrainAbstractMetricsBase):
    """No-op base class defining the training-metric interface for continuous diffusion.

    Concrete subclasses should override ``forward`` (and optionally ``reset``
    and ``log_epoch_metrics``) to track the quality of predicted noise
    (ε-prediction) for nodes, edges, and global graph features during
    continuous-time diffusion training.  The default implementation is used
    for non-molecular datasets.
    """

    def forward(
        self,
        masked_pred_epsX: Tensor,
        masked_pred_epsE: Tensor,
        pred_y: Tensor,
        true_epsX: Tensor,
        true_epsE: Tensor,
        true_y: Tensor,
        log: bool,
    ) -> None:
        """Accumulate per-batch training metrics.

        Args:
            masked_pred_epsX: Predicted noise for node features with padding
                masked out, shape ``(bs * n, d_X)``.
            masked_pred_epsE: Predicted noise for edge features with padding
                masked out, shape ``(bs * n * n, d_E)``.
            pred_y: Predicted global graph features, shape ``(bs, d_y)``.
            true_epsX: Ground-truth noise for node features, shape ``(bs * n, d_X)``.
            true_epsE: Ground-truth noise for edge features,
                shape ``(bs * n * n, d_E)``.
            true_y: Ground-truth global graph features, shape ``(bs, d_y)``.
            log: Whether to log scalar values to the trainer logger this step.
        """
        pass


class _SumExceptBatchBase(Metric):
    """Base class for metrics that sum over all elements but normalise by batch size.

    Subclasses implement ``update`` to accumulate ``total_value`` and
    ``total_samples``; ``compute`` returns their ratio (per-graph average).
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_state("total_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def compute(self) -> Tensor:
        """Return the accumulated sum divided by the total number of graphs.

        Returns:
            Per-graph average.
        """
        return self.total_value / self.total_samples


class SumExceptBatchMetric(_SumExceptBatchBase):
    """Accumulates a scalar sum over all tensor elements, normalised by batch size.

    Divides the total accumulated value by the number of graphs (first
    dimension of the input), not by the total number of elements.  This makes
    `compute()` return the *per-graph* average, which is consistent with the
    DiGress log-probability formulation.

    Used in the diffusion models to track validation and test log-probabilities
    for nodes (``val_X_logp``) and edges (``val_E_logp``).
    """

    def update(self, values: Tensor) -> None:
        """Accumulate the element-wise sum and the batch count.

        Args:
            values: Tensor of any shape whose first dimension is the batch
                size.  All elements are summed; only the first dimension is
                used for normalisation.
        """
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]


class SumExceptBatchMSE(_SumExceptBatchBase):
    """MSE metric that normalises by batch size rather than total element count.

    Standard MSE divides the squared-error sum by the total number of scalar
    elements.  This class instead divides by the number of rows (graphs /
    examples), so the result is a *per-graph* MSE.  This matches the
    noise-prediction loss used in continuous-time DiGress.

    Used in the diffusion model to track ``val_X_mse``, ``val_E_mse``,
    ``val_y_mse``, and their test counterparts.
    """

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate squared error sum and row count.

        Args:
            preds: Model predictions; must have the same shape as ``target``.
            target: Ground-truth values; must have the same shape as ``preds``.
        """
        assert preds.shape == target.shape
        self.total_value += torch.sum((preds - target) ** 2)
        self.total_samples += preds.shape[0]


class SumExceptBatchKL(_SumExceptBatchBase):
    """KL divergence metric normalised by batch size.

    Accumulates ``KL(p ‖ q)`` across batches and returns the per-graph
    average.  The KL is computed with ``reduction='sum'`` so that all
    node/edge positions within a graph are included, then the total is
    divided by the number of graphs.

    Used in the discrete diffusion model to track validation and test KL
    divergences for nodes (``val_X_kl``) and edges (``val_E_kl``).
    """

    def update(self, p: Tensor, q: Tensor) -> None:
        """Accumulate the KL divergence sum and the batch count.

        Computes ``KL(p ‖ q) = Σ p log(p/q)`` using PyTorch's ``F.kl_div``,
        which expects ``q`` in *log-probability* space.

        Args:
            p: Target probability distribution (not log-space),
                shape ``(bs, ..., d)``.
            q: Predicted log-probabilities (log-space),
                shape ``(bs, ..., d)``.
        """
        self.total_value += F.kl_div(q, p, reduction="sum")
        self.total_samples += p.size(0)


class CrossEntropyMetric(_SumExceptBatchBase):
    """Cross-entropy loss metric for one-hot encoded targets, normalised by batch size.

    Accepts *one-hot* target tensors (as produced by the diffusion model's
    graph representation), converts them to class indices via ``argmax``, and
    accumulates the cross-entropy sum.  ``compute()`` returns the per-sample
    average.

    Used in ``TrainLossDiscrete`` to track node loss, edge loss, and global
    feature loss (``node_loss``, ``edge_loss``, ``y_loss``).
    """

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate cross-entropy loss and sample count.

        Args:
            preds: Predicted logits of shape ``(bs * n, d)`` for nodes or
                ``(bs * n * n, d)`` for edges, where ``d`` is the number of
                classes.
            target: One-hot encoded ground-truth labels with the same shape as
                ``preds``.  Converted internally to class indices via
                ``argmax`` along the last dimension.
        """
        self.total_value += F.cross_entropy(preds, torch.argmax(target, dim=-1), reduction="sum")
        self.total_samples += preds.size(0)


