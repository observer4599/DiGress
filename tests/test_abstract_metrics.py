"""Tests for src/metrics/abstract_metrics.py.

Covers the key behavioral contracts of each torchmetrics subclass — the
computations most likely to silently break during refactoring.

Test groups:

- ``SumExceptBatchMetric``: sum-over-elements, divide-by-batch-size semantics.
- ``SumExceptBatchMSE``: squared error summed per element, normalised by batch rows.
- ``SumExceptBatchKL``: KL divergence summed and normalised by batch size.
- ``CrossEntropyMetric``: one-hot targets converted via argmax before CE loss.
- ``ProbabilityMetric``: marginal probability averaged over *all* elements.
- ``NLL``: per-sample NLL averaged over all values in the batch tensor.
"""

import torch
import pytest
from src.metrics.abstract_metrics import (
    SumExceptBatchMetric,
    SumExceptBatchMSE,
    SumExceptBatchKL,
    CrossEntropyMetric,
    ProbabilityMetric,
    NLL,
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


def test_sum_except_batch_mse_normalises_by_batch_rows() -> None:
    """MSE divides by number of rows (graphs), not total number of elements.

    Shape (2, 3) with preds=1, target=0 everywhere: sum_squared_error=6,
    n_obs=2 (rows only) → compute()=3.0.  A standard MeanSquaredError would
    return 1.0 by dividing by numel()=6 instead.
    """
    metric = SumExceptBatchMSE()
    preds = torch.ones(2, 3)
    target = torch.zeros(2, 3)
    metric.update(preds, target)
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


def test_probability_metric_averages_over_all_elements() -> None:
    """Mean is taken over every element, not just the batch dimension.

    A (3, 4) tensor of 0.5 contains 12 elements; compute() returns 0.5.
    Distinguishes ProbabilityMetric from metrics that only divide by batch size.
    """
    metric = ProbabilityMetric()
    metric.update(torch.full((3, 4), 0.5))
    assert metric.compute().item() == pytest.approx(0.5)


def test_nll_averages_over_all_values() -> None:
    """Mean NLL is computed over all elements in the accumulated batch_nll tensor.

    Four samples each with NLL=2.0: compute() returns 2.0.  Confirms that
    total_nll / total_samples gives the correct per-sample average.
    """
    metric = NLL()
    metric.update(torch.full((4,), 2.0))
    assert metric.compute().item() == pytest.approx(2.0)


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
