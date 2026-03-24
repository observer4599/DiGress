"""Shared utilities for the DiGress graph diffusion model.

Contains the PlaceHolder container for batched graph data (node features X,
edge features E, global features y), dense-format conversion helpers, and
config-merging utilities.

This module is used throughout the pipeline — by dataset loaders, the
diffusion model, the noise schedule, and the transformer backbone — so
it sits at the root of the import graph.
"""

import os

import torch
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch


def create_folders(args) -> None:
    """Create output directories for generated graphs and diffusion chains.

    Creates ``graphs/`` and ``chains/`` at the working directory root, then
    creates run-specific subdirectories named after ``args.general.name``.
    Existing directories are silently ignored.

    Args:
        args: Hydra config object with a ``general.name`` string attribute
            that identifies the current run.
    """
    try:
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def to_dense(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
) -> tuple['PlaceHolder', torch.Tensor]:
    """Convert a sparse PyG batch to dense padded tensors.

    PyTorch Geometric stores graphs in sparse COO format (variable-length node
    lists and edge lists). This function pads a batch of such graphs to the
    maximum node count in the batch, producing fixed-shape tensors suitable for
    the transformer backbone.

    Self-loops are removed before converting edges so that the diagonal of E
    remains zero. Missing edges (no feature vector) are encoded by
    ``encode_no_edge``, which sets ``E[..., 0] = 1`` to distinguish them from
    present-but-zero-valued edges.

    Args:
        x: Node features from a PyG ``Data`` object, shape
            ``(total_nodes_in_batch, node_feature_dim)``.
        edge_index: Edge connectivity, shape ``(2, num_edges)``, as returned by
            PyG.
        edge_attr: Edge attributes, shape ``(num_edges, edge_feature_dim)``.
        batch: Node-to-graph assignment vector of length ``total_nodes_in_batch``,
            as returned by PyG DataLoader.

    Returns:
        A tuple ``(data, node_mask)`` where ``data`` is a PlaceHolder with:

        - ``X``: Padded node features of shape
          ``(batch_size, max_nodes, node_feature_dim)``.
        - ``E``: Dense edge tensor of shape
          ``(batch_size, max_nodes, max_nodes, edge_feature_dim)`` with the
          "no edge" class encoded in channel 0 and a zeroed diagonal.
        - ``y``: ``None`` (global features are not populated here).

        And ``node_mask`` is a boolean tensor of shape
        ``(batch_size, max_nodes)``; ``True`` for real nodes, ``False`` for
        padding.
    """
    X, node_mask = to_dense_batch(x=x, batch=batch)
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E: torch.Tensor) -> torch.Tensor:
    """Encode absent edges as a dedicated "no edge" class and zero the diagonal.

    In the dense representation, positions where no edge exists have an
    all-zero feature vector. This function disambiguates them from
    present-but-zero-valued edges by setting channel 0 of those positions to
    1, treating channel 0 as the "no edge" indicator class. The diagonal
    (self-loops) is then forced to zero regardless.

    This encoding is required before passing edge tensors to the transformer
    so it can distinguish "no edge" from "edge with all-zero features".

    Args:
        E: Edge feature tensor of shape
            ``(batch_size, num_nodes, num_nodes, edge_feature_dim)``.
            An all-zero feature vector at position ``[b, i, j]`` indicates no
            edge between nodes ``i`` and ``j`` in graph ``b``.

    Returns:
        Modified ``E`` with channel 0 set to 1 for absent edges and all
        diagonal entries zeroed out.
    """
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    """Merge a saved config into the current config, adding any missing keys.

    When resuming training, the saved config may contain keys that were added
    after the original run was started (e.g. new hyperparameters with defaults).
    This function back-fills those keys into ``cfg`` without overwriting values
    that already exist.

    Only the ``general``, ``train``, and ``model`` config sections are merged.

    Args:
        cfg: Current Hydra/OmegaConf DictConfig to update in place.
        saved_cfg: Previously saved DictConfig loaded from a checkpoint.

    Returns:
        The updated ``cfg`` with missing keys filled in from ``saved_cfg``.
    """
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    """Container for the three graph feature types used throughout DiGress.

    Holds node features (X), edge features (E), and global graph features (y)
    as a unit, making it easy to pass and transform all three together through
    the diffusion pipeline. Most operations in DiGress — noising, denoising,
    normalization, masking — consume and return PlaceHolder instances.

    In a batched setting, graphs are padded to the same node count; the
    ``mask`` method zeroes out padded positions so they do not influence
    downstream computations.
    """

    def __init__(self, X: torch.Tensor, E: torch.Tensor, y: torch.Tensor | None) -> None:
        """Initialize the placeholder with graph feature tensors.

        Args:
            X: Node features of shape ``(batch_size, num_nodes, node_feature_dim)``.
            E: Edge features of shape
                ``(batch_size, num_nodes, num_nodes, edge_feature_dim)``.
                Expected to be symmetric with a zeroed diagonal (undirected
                graph, no self-loops).
            y: Global graph-level features of shape
                ``(batch_size, global_feature_dim)``, or ``None`` when not used.
        """
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor) -> 'PlaceHolder':
        """Cast X, E, and y to the same device and dtype as ``x``.

        Args:
            x: Reference tensor whose device and dtype are used for casting.

        Returns:
            ``self``, with X, E, and y updated in place.
        """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask: torch.Tensor, collapse: bool = False) -> 'PlaceHolder':
        """Apply the node mask to zero out or collapse padded positions.

        Graphs in a batch are padded to the maximum node count. This method
        ensures padded node and edge positions do not carry signal.

        When ``collapse=False`` (used during training and intermediate sampling
        steps), padded positions in X and E are multiplied by zero.

        When ``collapse=True`` (used at the final sampling step to obtain
        discrete graphs), X and E are collapsed from one-hot/soft distributions
        to integer class indices via ``argmax``, and padded positions are set
        to ``-1`` to flag them as invalid.

        The edge mask is constructed as the outer product of the node mask with
        itself, so an edge ``(i, j)`` is masked out if either endpoint is a
        padding node.

        Args:
            node_mask: Boolean tensor of shape ``(batch_size, num_nodes)``
                where ``True`` marks real nodes.
            collapse: If ``True``, take ``argmax`` over the feature dimension
                and mark padded entries as ``-1``.

        Returns:
            ``self``, with X and E updated in place.
        """
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self
