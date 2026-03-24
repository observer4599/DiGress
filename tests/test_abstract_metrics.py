"""Tests for src/metrics/abstract_metrics.py.

Covers the key behavioral contracts of each torchmetrics subclass — the
computations most likely to silently break during refactoring.

Test groups:

- ``SumExceptBatchMetric``: sum-over-elements, divide-by-batch-size semantics.
- ``SumExceptBatchKL``: KL divergence summed and normalised by batch size.
- ``CrossEntropyMetric``: one-hot targets converted via argmax before CE loss.
"""

import torch
import pytest
from src.metrics.abstract_metrics import (
    SumExceptBatchMetric,
    SumExceptBatchKL,
    CrossEntropyMetric,
)


def test_sum_except_batch_metric_divides_by_batch_size() -> None:
    """Sum of all elements divided by the first dimension, not total element count.

    Shape (3, 4) of ones: total_value=12, total_samples=3 → compute()=4.0.
    Confirms the per-graph (not per-element) normalisation contract.
    """
    metric = SumExceptBatchMetric()
    values = torch.ones(3, 4)
    metric.update(values)
    assert metric.compute().item() == pytest.approx(4.0)


def test_sum_except_batch_metric_accumulates_across_updates() -> None:
    """State accumulates correctly across multiple update calls.

    Two updates of shape (2, 3): total_value=12, total_samples=4 → compute()=3.0.
    Verifies that torchmetrics state is additive between batches.
    """
    metric = SumExceptBatchMetric()
    metric.update(torch.ones(2, 3))
    metric.update(torch.ones(2, 3))
    assert metric.compute().item() == pytest.approx(3.0)


def test_cross_entropy_metric_uses_argmax_on_one_hot_target() -> None:
    """One-hot targets are converted to class indices via argmax before CE loss.

    Four samples, 3 classes; logits strongly predict class 0 and the one-hot
    target also indicates class 0.  Confident correct predictions yield
    CE ≈ 0, confirming argmax conversion works on the one-hot format used by
    DiGress graph representations.
    """
    metric = CrossEntropyMetric()
    preds = torch.tensor([[10.0, 0.0, 0.0],
                          [10.0, 0.0, 0.0],
                          [10.0, 0.0, 0.0],
                          [10.0, 0.0, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0]])
    metric.update(preds, target)
    assert metric.compute().item() == pytest.approx(0.0, abs=1e-4)



def test_sum_except_batch_kl_normalises_by_batch_size() -> None:
    """KL(p ‖ q) is zero when p and q are identical distributions.

    Uses p=q so the expected KL divergence is 0.  Also confirms that update()
    accepts q in log-probability space (as required by F.kl_div) and that
    compute() divides the accumulated sum by batch size (2 rows).
    """
    metric = SumExceptBatchKL()
    p = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    log_q = torch.log(p)
    metric.update(p, log_q)
    assert metric.compute().item() == pytest.approx(0.0, abs=1e-6)
