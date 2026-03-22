"""Tests for :class:`src.diffusion.distributions.DistributionNodes`.

Covers construction from both dict and tensor inputs, probability
normalization, sampling shape and support, and log-probability values.
"""

import pytest
import torch

from src.diffusion.distributions import DistributionNodes


@pytest.fixture
def dist_from_dict():
    histogram = {1: 1, 2: 3, 3: 2}
    return DistributionNodes(histogram)


def test_prob_sums_to_one(dist_from_dict):
    # normalization invariant: prob must always sum to 1
    assert dist_from_dict.prob.sum().item() == pytest.approx(1.0)


def test_dict_input_sets_correct_probabilities():
    # dict histogram is correctly mapped to probability vector
    dist = DistributionNodes({2: 1, 4: 3})
    assert dist.prob[2].item() == pytest.approx(0.25)
    assert dist.prob[4].item() == pytest.approx(0.75)


def test_tensor_input_is_normalized():
    # tensor input path: raw counts are normalized to probabilities
    raw = torch.tensor([0.0, 2.0, 2.0])
    dist = DistributionNodes(raw)
    assert dist.prob.sum().item() == pytest.approx(1.0)
    assert dist.prob[1].item() == pytest.approx(0.5)


def test_sample_n_shape(dist_from_dict):
    # sample_n returns a 1-D tensor of length n_samples
    samples = dist_from_dict.sample_n(7, device=torch.device("cpu"))
    assert samples.shape == (7,)


def test_sample_n_values_in_support(dist_from_dict):
    # sampled node counts are within the support of the histogram
    torch.manual_seed(0)
    samples = dist_from_dict.sample_n(50, device=torch.device("cpu"))
    assert samples.min().item() >= 0
    assert samples.max().item() <= 3


def test_log_prob_shape(dist_from_dict):
    # log_prob returns a 1-D tensor matching batch length
    nodes = torch.tensor([1, 2, 3])
    lp = dist_from_dict.log_prob(nodes)
    assert lp.shape == (3,)


def test_log_prob_values():
    # log_prob returns correct log probabilities for known distribution
    dist = DistributionNodes({1: 1, 2: 3})  # p(1)=0.25, p(2)=0.75
    nodes = torch.tensor([1, 2])
    lp = dist.log_prob(nodes)
    assert lp[0].item() == pytest.approx(torch.log(torch.tensor(0.25)).item(), abs=1e-5)
    assert lp[1].item() == pytest.approx(torch.log(torch.tensor(0.75)).item(), abs=1e-5)
