"""Tests for src/datasets/abstract_dataset.py.

Captures key behavioral contracts of AbstractDatasetInfos and
AbstractDataModule that must be preserved after refactoring:

- complete_infos: sets num_classes, max_n_nodes, and nodes_dist type.
- node_counts: returns a normalized probability distribution.
- node_types: returns a distribution with correct length and normalization.
- edge_counts: distributes mass to the non-edge class (index 0).
- valency_count: returns a normalized probability distribution.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from types import SimpleNamespace
from torch_geometric.data import Data

from src.datasets.abstract_dataset import (
    AbstractDataModule,
    AbstractDatasetInfos,
    MolecularDataModule,
)
from src.diffusion.distributions import DistributionNodes


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _cfg(batch_size: int = 2) -> SimpleNamespace:
    """Build a minimal Hydra-style config accepted by AbstractDataModule.

    Args:
        batch_size: Number of graphs per batch in the train dataloader.

    Returns:
        A SimpleNamespace with ``train``, ``general``, and ``dataset``
        sub-namespaces containing just the fields AbstractDataModule reads.
    """
    return SimpleNamespace(
        train=SimpleNamespace(batch_size=batch_size, num_workers=0),
        general=SimpleNamespace(name="test"),
        dataset=SimpleNamespace(pin_memory=False),
    )


def _path_graph_3nodes() -> Data:
    """Path graph 0–1–2: 3 nodes (all class 0), 4 directed edges, 2 edge classes."""
    x = torch.zeros(3, 2)
    x[:, 0] = 1.0                                   # all nodes are class 0
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.zeros(4, 2)
    edge_attr[:, 1] = 1.0                           # all edges are type 1
    y = torch.zeros(1, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _two_class_graphs() -> list[Data]:
    """Two graphs where node classes differ, for testing node_types values."""
    # graph 1: node 0 is class 0, node 1 is class 1
    x1 = torch.eye(2)
    e1 = torch.tensor([[0, 1], [1, 0]])
    ea1 = torch.zeros(2, 2); ea1[:, 1] = 1.0
    g1 = Data(x=x1, edge_index=e1, edge_attr=ea1, y=torch.zeros(1, 1))
    # graph 2: both nodes are class 0
    x2 = torch.zeros(2, 2); x2[:, 0] = 1.0
    g2 = Data(x=x2, edge_index=e1, edge_attr=ea1, y=torch.zeros(1, 1))
    return [g1, g2]


def _make_module(
    graphs: list[Data],
    cls: type = AbstractDataModule,
    batch_size: int = 2,
) -> AbstractDataModule:
    """Instantiate a data module from a list of graphs, using all splits identically.

    Passes the same graph list as train, val, and test to avoid the need for
    separate splits in unit tests. The resulting module is self-contained and
    does not require a real dataset on disk.

    Args:
        graphs: List of PyG Data objects to use for all three splits.
        cls: Data module class to instantiate. Defaults to AbstractDataModule;
            pass MolecularDataModule to test valency-specific methods.
        batch_size: Number of graphs per batch.

    Returns:
        An initialised data module of type ``cls``.
    """
    cfg = _cfg(batch_size=batch_size)
    datasets = {"train": graphs, "val": graphs, "test": graphs}
    return cls(cfg, datasets)


# ---------------------------------------------------------------------------
# AbstractDatasetInfos.complete_infos
# ---------------------------------------------------------------------------

def test_complete_infos_num_classes() -> None:
    """complete_infos sets num_classes to the length of the node_types tensor."""
    infos = AbstractDatasetInfos()
    node_types = torch.tensor([0.2, 0.3, 0.5])
    infos.complete_infos(torch.tensor([0.0, 0.4, 0.6]), node_types)
    assert infos.num_classes == 3


def test_complete_infos_max_n_nodes() -> None:
    """complete_infos sets max_n_nodes to len(n_nodes) - 1 (index of last bin)."""
    infos = AbstractDatasetInfos()
    n_nodes = torch.tensor([0.0, 0.1, 0.3, 0.6])
    infos.complete_infos(n_nodes, torch.tensor([1.0]))
    assert infos.max_n_nodes == 3


def test_complete_infos_nodes_dist_type() -> None:
    """complete_infos stores nodes_dist as a DistributionNodes instance."""
    infos = AbstractDatasetInfos()
    infos.complete_infos(torch.tensor([0.0, 0.5, 0.5]), torch.tensor([1.0]))
    assert isinstance(infos.nodes_dist, DistributionNodes)


# ---------------------------------------------------------------------------
# AbstractDataModule statistical methods
# ---------------------------------------------------------------------------

def test_node_counts_sums_to_one() -> None:
    """node_counts returns a normalised probability distribution over graph sizes."""
    graphs = [_path_graph_3nodes(), _path_graph_3nodes()]
    module = _make_module(graphs, batch_size=1)
    counts = module.node_counts(max_nodes_possible=300)
    assert counts.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_node_types_distribution() -> None:
    """node_types returns a normalised distribution with correct per-class mass.

    The two-graph fixture has 3 class-0 nodes and 1 class-1 node, so the
    expected marginal is [0.75, 0.25].
    """
    graphs = _two_class_graphs()
    module = _make_module(graphs, batch_size=2)
    dist = module.node_types()
    assert dist.shape[0] == 2
    assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert dist[0].item() == pytest.approx(0.75, abs=1e-5)
    assert dist[1].item() == pytest.approx(0.25, abs=1e-5)


def test_edge_counts_non_edge_at_index_zero() -> None:
    """edge_counts accumulates absent node pairs at index 0.

    A 3-node path graph has 4 directed edges out of 6 possible ordered pairs,
    leaving 2 absent pairs. Index 0 must therefore carry positive mass.
    """
    graphs = [_path_graph_3nodes()]
    module = _make_module(graphs, batch_size=1)
    dist = module.edge_counts()
    assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert dist[0].item() > 0.0   # non-edge mass must be positive


def test_valency_count_sums_to_one() -> None:
    """valency_count returns a normalised probability distribution over atom valencies.

    edge_attr width must be 5 to match the bond-order multiplier
    [0, 1, 2, 3, 1.5] for [no bond, single, double, triple, aromatic].
    """
    x = torch.zeros(2, 4); x[:, 0] = 1.0
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.zeros(2, 5); edge_attr[:, 1] = 1.0  # single bond
    graphs = [Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.zeros(1, 1))]
    module = _make_module(graphs, cls=MolecularDataModule, batch_size=1)
    dist = module.valency_count(max_n_nodes=2)
    assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)
