"""Base classes for dataset modules and dataset info objects used by DiGress.

This module defines three base classes that all dataset-specific implementations
inherit from:

- ``AbstractDataModule``: a PyG ``LightningDataset`` wrapper that computes
  dataset-level graph statistics (node counts, node types, edge types) over
  the training split.
- ``MolecularDataModule``: extends ``AbstractDataModule`` with a valency
  distribution method for molecular graphs.
- ``AbstractDatasetInfos``: stores pre-computed statistics used to initialise
  the diffusion model's input/output dimensions and node-count prior.

Typical usage example:

    class MyDataModule(AbstractDataModule):
        def __init__(self, cfg):
            datasets = {'train': ..., 'val': ..., 'test': ...}
            super().__init__(cfg, datasets)

    class MyDatasetInfos(AbstractDatasetInfos):
        def __init__(self, datamodule, cfg):
            n_nodes = datamodule.node_counts()
            node_types = datamodule.node_types()
            super().complete_infos(n_nodes, node_types)
"""

import torch
from torch_geometric.data.lightning import LightningDataset

from src.diffusion.distributions import DistributionNodes
import src.utils as utils


class AbstractDataModule(LightningDataset):
    """PyG LightningDataset wrapper with graph-statistics helpers.

    Wraps train/val/test datasets in PyG ``DataLoader`` objects and exposes
    methods that scan the training split (and validation split for
    ``node_counts``) to compute normalised marginal distributions over node
    counts, node types, and edge types. These distributions are consumed by
    ``AbstractDatasetInfos`` to configure the diffusion model prior.

    Attributes:
        input_dims: Populated later by ``AbstractDatasetInfos.compute_input_output_dims``.
        output_dims: Populated later by ``AbstractDatasetInfos.compute_input_output_dims``.
    """

    def __init__(self, cfg, datasets):
        """Initialise dataloaders from pre-built dataset splits.

        Forces ``batch_size=2`` when the run name contains "debug" to allow
        fast local testing without changing config files.

        Args:
            cfg: Hydra config object. Reads ``cfg.train.batch_size``,
                ``cfg.train.num_workers``, ``cfg.dataset.pin_memory``, and
                ``cfg.general.name``.
            datasets: Mapping with keys ``'train'``, ``'val'``, and
                ``'test'``, each holding a PyG ``Dataset`` or list of
                ``Data`` objects.
        """
        super().__init__(train_dataset=datasets['train'], val_dataset=datasets['val'], test_dataset=datasets['test'],
                         batch_size=cfg.train.batch_size if 'debug' not in cfg.general.name else 2,
                         num_workers=cfg.train.num_workers,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def node_counts(self, max_nodes_possible: int = 300) -> torch.Tensor:
        """Compute the normalised node-count distribution over train and val splits.

        Counts how many graphs in the combined train+val set have exactly
        ``n`` nodes for each ``n``, then normalises to a probability vector.
        The returned tensor is trimmed to ``max_observed_count + 1`` entries
        so it can be used directly as a categorical prior over graph sizes.

        Args:
            max_nodes_possible: Upper bound on node count used to pre-allocate
                the count buffer. Graphs with more nodes than this will cause
                an index error.

        Returns:
            Normalised probability vector of shape ``(max_n + 1,)`` where
            index ``i`` holds the fraction of graphs that have exactly ``i``
            nodes.
        """
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                _, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self) -> torch.Tensor:
        """Compute the normalised node-type marginal over the training split.

        Node features are assumed to be one-hot encoded. Sums the one-hot
        vectors across all nodes in all training graphs to obtain raw type
        counts, then normalises.

        Returns:
            Probability vector of shape ``(num_node_types,)`` where index
            ``k`` is the fraction of training nodes with type ``k``.
        """
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for data in self.train_dataloader():
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self) -> torch.Tensor:
        """Compute the normalised edge-type marginal over the training split.

        Edge features are assumed to be one-hot encoded, with index 0
        representing *no edge*. For each graph in the batch the method counts
        all ``n*(n-1)`` ordered node pairs, subtracts the number of real edges
        to obtain the no-edge count, then accumulates type counts for each
        real edge type (indices 1 onward). The final vector is normalised over
        all (edge, non-edge) slots.

        This treatment of non-edges is essential for the DiGress diffusion
        prior: the model must know the marginal probability of the absence of
        an edge, not just the conditional distribution over edge types.

        Returns:
            Probability vector of shape ``(num_edge_types,)`` where index 0
            is the no-edge probability and indices 1+ are bond-type
            probabilities.
        """
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for data in self.train_dataloader():
            _, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    """``AbstractDataModule`` extended with a per-atom valency distribution.

    Subclassed by molecule-specific modules (QM9, MOSES, GuacaMol) that need
    a valency prior for molecular validity evaluation.
    """

    def valency_count(self, max_n_nodes: int) -> torch.Tensor:
        """Compute the normalised valency distribution over training atoms.

        For each atom in the training set, sums its bond orders using the
        multiplier ``[0, 1, 2, 3, 1.5]`` for bond types [no bond, single,
        double, triple, aromatic] and increments the bin at the resulting
        integer valency. The buffer is pre-sized to ``3 * max_n_nodes - 2``,
        which is the theoretical maximum valency when every node in a
        fully-connected graph has triple bonds.

        Args:
            max_n_nodes: Maximum number of nodes in any graph of the dataset.
                Used to size the valency histogram.

        Returns:
            Normalised probability vector of shape ``(3 * max_n_nodes - 2,)``
            where index ``v`` is the fraction of training atoms with valency
            ``v``.
        """
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for data in self.train_dataloader():
            n = data.x.shape[0]

            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    """Container for pre-computed dataset statistics consumed by the diffusion model.

    Subclasses (e.g. ``QM9infos``, ``SpectreDatasetInfos``) call
    ``complete_infos`` to populate the core statistics, then optionally call
    ``compute_input_output_dims`` to derive network input/output sizes.

    Attributes:
        num_classes: Number of distinct node types in the dataset.
        max_n_nodes: Maximum number of nodes across all graphs.
        nodes_dist: Categorical prior over graph sizes used during generation.
        input_dims: Dict ``{'X': int, 'E': int, 'y': int}`` with the feature
            dimensionality seen by the denoising network, including extra and
            domain features plus one time-conditioning scalar in ``'y'``.
        output_dims: Dict ``{'X': int, 'E': int, 'y': int}`` with the raw
            dataset feature dimensions predicted by the network (``y`` is
            always 0 — the model does not predict global graph features).
    """

    def complete_infos(self, n_nodes: torch.Tensor, node_types: torch.Tensor) -> None:
        """Populate core dataset statistics from pre-computed marginals.

        Must be called before ``compute_input_output_dims`` and before the
        info object is passed to the model. Resets ``input_dims`` and
        ``output_dims`` to ``None`` so they must be filled separately.

        Args:
            n_nodes: Normalised node-count distribution as returned by
                ``AbstractDataModule.node_counts()``. Index ``i`` holds the
                probability that a graph has ``i`` nodes; index 0 is unused
                (no graph has zero nodes), so ``max_n_nodes = len(n_nodes) - 1``.
            node_types: Normalised node-type marginal as returned by
                ``AbstractDataModule.node_types()``. Length determines
                ``num_classes``.
        """
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features) -> None:
        """Determine network input/output feature dimensions from a real batch.

        Runs one batch through the feature-extraction pipeline to count
        dimensions, rather than computing them analytically. This accounts for
        arbitrary combinations of extra graph features and domain-specific
        features without hard-coding any sizes.

        The ``'y'`` input dimension gets ``+1`` for the diffusion timestep
        ``t``, which is concatenated to the global conditioning vector before
        the forward pass.

        Populates ``self.input_dims`` and ``self.output_dims`` in-place.

        Args:
            datamodule: An ``AbstractDataModule`` instance whose
                ``train_dataloader()`` provides the example batch.
            extra_features: Callable that maps a dense example batch dict to
                an object with ``.X``, ``.E``, ``.y`` tensors containing
                extra graph features (e.g. cycle counts, eigenvalues).
            domain_features: Callable with the same signature as
                ``extra_features`` for domain-specific features (e.g. RDKit
                molecular descriptors). Pass ``DummyExtraFeatures()`` for
                non-molecular datasets.
        """
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch['x'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}
