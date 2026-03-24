"""Per-class cross-entropy metrics for discrete molecular graph diffusion.

This module defines TorchMetrics-based metrics that track binary cross-entropy
loss separately for each atom type and bond type. Tracking per-class CE helps
diagnose which atom or bond types are learned well versus poorly during training.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric, MetricCollection
import torch.nn as nn


class CEPerClass(Metric):
    """Accumulates binary cross-entropy for a single class index across batches.

    Used to measure how well the model predicts a specific atom type or bond
    type. The metric applies softmax to the raw predictions and computes BCE
    against the one-hot target column for `class_id`, averaging over all
    non-padding positions.

    Padding is detected by checking which rows of the one-hot target are all
    zeros; those positions are excluded from the loss.
    """

    full_state_update = False

    def __init__(self, class_id: int):
        """Args:
            class_id: Index of the class (atom type or bond type) to track.
        """
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate BCE for this class over one batch.

        Args:
            preds: Raw (pre-softmax) model logits, shape ``(bs, n, d)`` for
                atoms or ``(bs, n, n, d)`` for edges.
            target: One-hot ground-truth labels with the same shape as
                ``preds``. Rows that are all zeros are treated as padding and
                excluded from the loss.
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.).any(dim=-1)

        prob = F.softmax(preds, dim=-1)[..., self.class_id].flatten()[mask]
        target = target[:, self.class_id][mask]

        self.total_ce += F.binary_cross_entropy(prob, target, reduction='sum')
        self.total_samples += prob.numel()

    def compute(self) -> Tensor:
        """Returns the mean BCE for this class over all accumulated samples."""
        return self.total_ce / self.total_samples


_BOND_TYPES = ['NoBond', 'Single', 'Double', 'Triple', 'Aromatic']


class AtomMetricsCE(MetricCollection):
    """MetricCollection of per-atom-type BCE metrics for a given dataset.

    Automatically instantiates one ``CEPerClass`` metric for each atom type
    listed in ``dataset_infos.atom_decoder``. Only elements present in the
    dataset are tracked, keeping logs compact.
    """

    def __init__(self, dataset_infos):
        metrics = {
            f"{atom_type}CE": CEPerClass(i)
            for i, atom_type in enumerate(dataset_infos.atom_decoder)
        }
        super().__init__(metrics)


class BondMetricsCE(MetricCollection):
    """MetricCollection of per-bond-type BCE metrics.

    Tracks cross-entropy for the five bond categories used in the discrete
    molecular graph representation: no-bond, single, double, triple, and
    aromatic. Class indices match the column ordering in the edge one-hot
    encoding.
    """

    def __init__(self):
        metrics = {f"{bond}CE": CEPerClass(i) for i, bond in enumerate(_BOND_TYPES)}
        super().__init__(metrics)


class TrainMolecularMetricsDiscrete(nn.Module):
    """Aggregates atom and bond CE metrics for discrete molecular diffusion training.

    Wraps ``AtomMetricsCE`` and ``BondMetricsCE`` into a single module so that
    training code can update both collections in one call. Metrics accumulate
    across batches and can be reset at epoch boundaries.
    """

    def __init__(self, dataset_infos):
        super().__init__()
        self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        self.train_bond_metrics = BondMetricsCE()

    def forward(
        self,
        masked_pred_X: Tensor,
        masked_pred_E: Tensor,
        true_X: Tensor,
        true_E: Tensor,
        log: bool,
        writer: SummaryWriter | None = None,
        global_step: int = 0,
    ) -> None:
        """Update atom and bond metrics for one training batch.

        Args:
            masked_pred_X: Node logits with padding masked out, shape
                ``(bs, n, atom_types)``).
            masked_pred_E: Edge logits with padding masked out, shape
                ``(bs, n, n, bond_types)``).
            true_X: One-hot atom targets, same shape as ``masked_pred_X``.
            true_E: One-hot bond targets, same shape as ``masked_pred_E``.
            log: If ``True``, write per-class BCE scalars to TensorBoard.
            writer: TensorBoard ``SummaryWriter`` to log to. No-op if ``None``.
            global_step: Training step passed to ``writer.add_scalar``.
        """
        self.train_atom_metrics(masked_pred_X, true_X)
        self.train_bond_metrics(masked_pred_E, true_E)
        if log:
            self._log_metrics('train/', writer, global_step)

    def reset(self) -> None:
        """Reset accumulated state in both atom and bond metric collections."""
        self.train_atom_metrics.reset()
        self.train_bond_metrics.reset()

    def log_epoch_metrics(
        self,
        writer: SummaryWriter | None = None,
        global_step: int = 0,
    ) -> tuple[dict, dict]:
        """Compute epoch-level metrics, log to TensorBoard, and return as dicts.

        Computes the mean BCE for each atom type and bond type over the entire
        epoch. If a writer is supplied, each metric is written under the
        ``train_epoch/`` tag prefix. The caller is responsible for resetting
        the metrics afterwards via ``reset()``.

        Args:
            writer: TensorBoard ``SummaryWriter`` to log to. No-op if ``None``.
            global_step: Step (typically the epoch number) passed to
                ``writer.add_scalar``.

        Returns:
            A tuple ``(epoch_atom_metrics, epoch_bond_metrics)`` where each
            dict maps metric names to scalar float values.
        """
        return self._log_metrics('train_epoch/', writer, global_step)

    def _log_metrics(
        self,
        prefix: str,
        writer: SummaryWriter | None,
        global_step: int,
    ) -> tuple[dict, dict]:
        """Compute current metrics, optionally write them to TensorBoard, and return them.

        Args:
            prefix: Tag prefix passed to ``writer.add_scalar`` (e.g. ``'train/'``).
            writer: TensorBoard ``SummaryWriter``. Skipped if ``None``.
            global_step: Step value passed to ``writer.add_scalar``.

        Returns:
            A tuple ``(atom_metrics, bond_metrics)`` mapping metric names to
            scalar float values.
        """
        atom_metrics = {k: v.item() for k, v in self.train_atom_metrics.compute().items()}
        bond_metrics = {k: v.item() for k, v in self.train_bond_metrics.compute().items()}
        if writer is not None:
            for key, val in {**atom_metrics, **bond_metrics}.items():
                writer.add_scalar(prefix + key, val, global_step)
        return atom_metrics, bond_metrics
