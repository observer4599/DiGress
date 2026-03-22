"""Tests for SpectreGraphDataset.

Covers three areas:
- File-name properties (raw and processed caching paths).
- Split-size arithmetic (the 80/10/10 partition of 200 graphs).
- Data-object construction (node features, edge attributes, graph label).

File I/O is avoided throughout: properties are tested on a manually
constructed instance, and process() logic is replicated inline.
"""

import pytest
import torch
import torch_geometric.utils

from src.datasets.spectre_dataset import SpectreGraphDataset


# ── helpers ──────────────────────────────────────────────────────────────────

def _new_dataset(dataset_name: str = "sbm", split: str = "train") -> SpectreGraphDataset:
    """Create a SpectreGraphDataset instance with essential attributes set, bypassing __init__.

    Constructs the object without triggering any file I/O or network access,
    making it safe to call in tests that only exercise pure Python properties.

    Args:
        dataset_name: SPECTRE graph family name — one of ``"sbm"``,
            ``"planar"``, or ``"comm20"``.
        split: Dataset partition — ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        A partially initialised ``SpectreGraphDataset`` with ``dataset_name``,
        ``split``, and ``num_graphs`` set to 200.
    """
    ds = object.__new__(SpectreGraphDataset)
    ds.dataset_name = dataset_name
    ds.split = split
    ds.num_graphs = 200
    return ds


# ── properties ───────────────────────────────────────────────────────────────

def test_raw_file_names():
    """raw_file_names always lists all three split files regardless of the active split."""
    ds = _new_dataset()
    assert ds.raw_file_names == ["train.pt", "val.pt", "test.pt"]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_processed_file_names(split: str):
    """processed_file_names returns only the active split's file so each split caches independently."""
    ds = _new_dataset(split=split)
    assert ds.processed_file_names == [f"{split}.pt"]


# ── split arithmetic ─────────────────────────────────────────────────────────

def test_split_sizes_are_exact():
    """Split arithmetic produces the expected 128/32/40 (train/val/test) partition.

    The download() method carves off 20% for test first, then applies an 80/20
    split to the remainder: 200 → 40 test, 160 × 0.8 = 128 train, 32 val.
    """
    num_graphs = 200
    test_len = int(round(num_graphs * 0.2))
    train_len = int(round((num_graphs - test_len) * 0.8))
    val_len = num_graphs - train_len - test_len
    assert (train_len, val_len, test_len) == (128, 32, 40)


def test_split_sizes_sum_to_total():
    """Split sizes are exhaustive — every graph belongs to exactly one partition."""
    num_graphs = 200
    test_len = int(round(num_graphs * 0.2))
    train_len = int(round((num_graphs - test_len) * 0.8))
    val_len = num_graphs - train_len - test_len
    assert train_len + val_len + test_len == num_graphs


# ── data-object construction (process logic) ─────────────────────────────────

def _make_data_object(adj: torch.Tensor) -> torch_geometric.data.Data:
    """Reproduce the Data construction logic from SpectreGraphDataset.process().

    Mirrors the per-graph loop body in process() so tests can verify the
    output schema without running the full dataset pipeline.

    Args:
        adj: Square binary adjacency matrix of shape ``(n, n)``.

    Returns:
        A PyG Data object with uniform node features (all ones, shape ``(n, 1)``),
        one-hot edge attributes (shape ``(|E|, 2)``, column 1 set to 1), an
        empty graph label (shape ``(1, 0)``), and ``n_nodes`` set to ``n``.
    """
    n = adj.shape[-1]
    X = torch.ones(n, 1, dtype=torch.float)
    y = torch.zeros([1, 0]).float()
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
    edge_attr[:, 1] = 1
    num_nodes = n * torch.ones(1, dtype=torch.long)
    return torch_geometric.data.Data(
        x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
    )


def test_process_node_features_are_ones():
    """Node feature matrix x is all-ones with shape (n, 1).

    SPECTRE graphs have no node types; a constant feature vector is used as a
    uniform placeholder so all nodes are treated identically by the model.
    """
    adj = torch.zeros(5, 5)
    data = _make_data_object(adj)
    assert data.x.shape == (5, 1)
    assert data.x.eq(1).all()


def test_process_y_is_empty():
    """Graph-level label y is a zero-feature placeholder of shape (1, 0).

    SPECTRE graphs carry no graph-level target; the empty tensor satisfies
    DiGress's expectation that y exists while encoding zero information.
    """
    adj = torch.zeros(3, 3)
    data = _make_data_object(adj)
    assert data.y.shape == (1, 0)


def test_process_edge_attr_second_column_is_ones():
    """Edge attributes are two-class one-hot vectors with column 1 (edge-present) always set.

    SPECTRE adjacency matrices are binary, so every entry in edge_index
    corresponds to a real edge. The one-hot encoding [0, 1] signals
    "bond present" for compatibility with DiGress's edge-type marginals.
    """
    adj = torch.zeros(4, 4)
    adj[0, 1] = adj[1, 0] = 1  # one undirected edge → two directed edges
    data = _make_data_object(adj)
    assert data.edge_attr.shape == (2, 2)
    assert data.edge_attr[:, 1].eq(1).all()
    assert data.edge_attr[:, 0].eq(0).all()
