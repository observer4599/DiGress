"""Node-count distribution for graph generation.

Wraps a categorical distribution over the number of nodes in a graph,
built from dataset statistics. Used during sampling to draw graph sizes
that match the training distribution.
"""

import torch


class DistributionNodes:
    """Categorical distribution over the number of nodes in a graph.

    Built from dataset histogram counts and used in two ways: sampling
    graph sizes during generation, and computing the log-likelihood of
    a batch's node counts (e.g. for prior evaluation).

    Attributes:
        prob: Normalized probability vector of shape (max_n_nodes + 1,),
            where index i holds the probability that a graph has i nodes.
    """

    def __init__(self, histogram: dict[int, int] | torch.Tensor) -> None:
        """Build the node-count distribution from raw counts.

        Args:
            histogram: Either a dict mapping number-of-nodes to raw counts,
                or a 1-D tensor whose index i holds the raw count for
                graphs with i nodes. Counts are normalized internally so
                any positive scale is accepted.
        """
        if isinstance(histogram, dict):
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Sample node counts from the distribution.

        Args:
            n_samples: Number of independent samples to draw.
            device: Device on which to place the returned tensor.

        Returns:
            1-D integer tensor of shape (n_samples,) containing sampled
            node counts, each in [0, max_n_nodes].
        """
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes: torch.Tensor) -> torch.Tensor:
        """Compute the log-probability of each node count in a batch.

        A small epsilon (1e-30) is added before the log to avoid -inf for
        node counts that have zero probability in the training distribution.

        Args:
            batch_n_nodes: 1-D integer tensor of node counts, one per graph
                in the batch.

        Returns:
            1-D float tensor of the same length with the log-probability of
            each node count under the training distribution.
        """
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p
