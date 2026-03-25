"""Training-metric interface and batch-normalised torchmetrics for DiGress.

``TrainAbstractMetricsDiscrete`` is the no-op base class for the discrete
diffusion training loop; concrete subclasses (e.g. ``TrainMolecularMetricsDiscrete``)
override ``forward``, ``reset``, and ``log_epoch_metrics``.

``SumExceptBatchMetric``, ``SumExceptBatchKL``, and ``CrossEntropyMetric`` are
torchmetrics ``Metric`` subclasses that accumulate statistics across batches and
return per-graph averages via ``compute()``.  All normalise by the number of
graphs (first dimension) rather than the total number of elements, matching the
DiGress loss formulation (Vignac et al., 2022).
"""

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric


class TrainAbstractMetricsDiscrete:
    """No-op base class defining the training-metric interface for discrete diffusion.

    Concrete subclasses (e.g. ``TrainMolecularMetricsDiscrete``) override
    ``forward``, ``reset``, and ``log_epoch_metrics`` to track node- and
    edge-type prediction quality.  The default implementation is used for
    non-molecular datasets where no per-epoch logging is required.
    """

    def __call__(self, *args, **kwargs) -> None:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        masked_pred_X: Tensor,
        masked_pred_E: Tensor,
        true_X: Tensor,
        true_E: Tensor,
        log: bool,
        writer=None,
        global_step: int = 0,
    ) -> None:
        pass

    def reset(self) -> None:
        pass

    def log_epoch_metrics(
        self,
        writer=None,
        global_step: int = 0,
    ) -> tuple[None, None]:
        return None, None


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


