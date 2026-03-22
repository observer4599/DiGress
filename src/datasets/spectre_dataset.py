"""Datasets and data modules for the SPECTRE benchmark graphs.

Supports three synthetic graph families used in SPECTRE (Martinkus et al., 2022):
stochastic block model (SBM), planar graphs, and 20-community graphs. Each
dataset is loaded from a pre-built `.pt` file, split into train/val/test, and
converted into PyG ``Data`` objects with uniform node features and binary edge
attributes compatible with DiGress.

Typical usage example:

    datamodule = SpectreGraphDataModule(cfg)
    dataset_infos = SpectreDatasetInfos(datamodule, cfg.dataset)
"""

import os
import pathlib

import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class SpectreGraphDataset(InMemoryDataset):
    """Single split (train, val, or test) of a SPECTRE benchmark graph dataset.

    Wraps one of three synthetic graph families from the SPECTRE benchmark
    (Martinkus et al., 2022): stochastic block model (``sbm``), planar
    (``planar``), or 20-community (``comm20``) graphs. Each graph is stored as
    a PyG ``Data`` object with:

    - ``x``: all-ones node feature matrix of shape ``(n, 1)``
    - ``edge_index``: COO edge list derived from the raw adjacency matrix
    - ``edge_attr``: one-hot edge type of shape ``(|E|, 2)``, index 1 set to 1
      (i.e. all edges are of type "bond present")
    - ``y``: empty graph-level label of shape ``(1, 0)``
    - ``n_nodes``: node count tensor of shape ``(1,)``

    The raw dataset (200 graphs) is downloaded once and deterministically split
    into train/val/test using a fixed random seed (0) with an 80/10/10 ratio.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        """Initialize and load the dataset split from disk.

        Downloads and processes the raw data on first use. On subsequent calls
        the cached processed file is loaded directly.

        Args:
            dataset_name: One of ``"sbm"``, ``"planar"``, or ``"comm20"``.
            split: Which portion to load — ``"train"``, ``"val"``, or ``"test"``.
            root: Root directory where raw and processed files are cached.
            transform: Optional per-access transform applied to each graph.
            pre_transform: Optional transform applied once during processing.
            pre_filter: Optional predicate; graphs that return ``False`` are
                excluded during processing.
        """
        self.sbm_file = 'sbm_200.pt'
        self.planar_file = 'planar_64_200.pt'
        self.comm20_file = 'community_12_21_100.pt'
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 200
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        """Return the expected raw split file names used by PyG's caching logic."""
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self) -> list[str]:
        """Return the processed file name for this split."""
        return [self.split + '.pt']

    def download(self) -> None:
        """Download the raw dataset and write per-split adjacency lists to disk.

        Fetches the appropriate ``.pt`` file from the SPECTRE GitHub repository,
        then partitions all 200 graphs into train/val/test using a seeded
        random permutation (seed 0) with an approximate 80/10/10 split:

        - test:  20% of 200 = 40 graphs
        - train: 80% of the remaining 160 = 128 graphs
        - val:   remaining 32 graphs

        The three split files are saved to ``self.raw_paths[0..2]`` as lists of
        adjacency tensors, ready for ``process()`` to convert into PyG objects.

        Raises:
            ValueError: If ``dataset_name`` is not one of the three supported
                graph families.
        """
        if self.dataset_name == 'sbm':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt'
        elif self.dataset_name == 'planar':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt'
        elif self.dataset_name == 'comm20':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt'
        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')
        file_path = download_url(raw_url, self.raw_dir)

        adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self) -> None:
        """Convert raw adjacency matrices for this split into PyG Data objects.

        Reads the split's adjacency list from disk and builds a ``Data`` object
        per graph with uniform node features (all ones), two-class edge
        attributes (index 1 = edge present), and an empty graph label. The
        processed dataset is collated and saved to ``self.processed_paths[0]``.

        Note: ``pre_filter`` is checked before appending; ``pre_transform`` is
        applied after the check but the original (untransformed) graph is also
        appended, so each graph may appear twice when a pre-transform is set.
        """
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class SpectreGraphDataModule(AbstractDataModule):
    """Lightning data module that assembles train/val/test SPECTRE splits.

    Constructs three ``SpectreGraphDataset`` instances (one per split) and
    passes them to ``AbstractDataModule``, which wraps them in PyG
    ``DataLoader`` objects with the batch size and worker count from ``cfg``.

    The ``inner`` attribute points to the training dataset for direct indexing
    via ``__getitem__``.
    """

    def __init__(self, cfg, n_graphs: int = 200) -> None:
        """Build and register train, val, and test dataset splits.

        Args:
            cfg: Hydra config object. Must expose ``cfg.dataset.name``,
                ``cfg.dataset.datadir``, ``cfg.train.batch_size``, and
                ``cfg.train.num_workers``.
            n_graphs: Total number of graphs in the raw dataset. Currently
                unused (the value is hard-coded in ``SpectreGraphDataset``);
                reserved for future parameterisation.
        """
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path)}

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item: int):
        """Return the graph at position ``item`` from the training split."""
        return self.inner[item]


class SpectreDatasetInfos(AbstractDatasetInfos):
    """Dataset-level statistics for a SPECTRE graph family.

    Computes and stores node-count distribution and edge-type marginals over
    the train and validation splits. These statistics are consumed by DiGress
    to initialise the noise schedule marginals and the node-count prior.

    Since SPECTRE graphs are untyped (all nodes are equivalent), ``node_types``
    is fixed to ``[1]`` — a single node class with probability 1.

    Attributes:
        datamodule: The ``SpectreGraphDataModule`` used to compute statistics.
        name: Fixed identifier ``"nx_graphs"`` used elsewhere in the pipeline.
        n_nodes: Normalised histogram of node counts across train+val graphs,
            shape ``(max_n_nodes + 1,)``.
        node_types: Constant tensor ``[1]`` indicating one node type.
        edge_types: Normalised histogram over edge types (no-edge vs. edge),
            shape ``(2,)``.
    """

    def __init__(self, datamodule: SpectreGraphDataModule, dataset_config) -> None:
        """Compute node and edge statistics from the data module.

        Args:
            datamodule: Assembled SPECTRE data module providing train/val
                data loaders for statistics computation.
            dataset_config: Dataset config section from the Hydra config
                (``cfg.dataset``). Not used directly here but kept for
                interface consistency with other ``DatasetInfos`` classes.
        """
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
