"""Graph transformer architecture for joint node, edge, and global feature denoising.

Implements the denoising network used in DiGress (Vignac et al., 2022), a
graph diffusion model that jointly updates three feature types at each layer:
node features X, edge features E, and a graph-level feature vector y.

The main entry point is GraphTransformer, which stacks XEyTransformerLayers.
Each layer contains a NodeEdgeBlock (edge-biased multi-head self-attention
with FiLM conditioning from y) followed by per-feature-type feed-forward
networks and residual connections.
"""

import math

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, LayerNorm
from torch.nn import functional as F
from torch import Tensor

from src import utils
from src.diffusion import diffusion_utils
from src.models.layers import Xtoy, Etoy, masked_softmax


class XEyTransformerLayer(nn.Module):
    """Single transformer layer that jointly updates node, edge, and global features.

    Wraps a NodeEdgeBlock (the attention sub-layer) with residual connections,
    layer normalisation, and independent feed-forward networks for each of the
    three feature types: node features X, edge features E, and graph-level
    features y. The structure follows the Pre-LN transformer variant applied
    simultaneously to all three feature spaces.

    Args:
        dx: Dimensionality of node features.
        de: Dimensionality of edge features.
        dy: Dimensionality of graph-level (global) features.
        n_head: Number of attention heads. Must divide dx evenly.
        dim_ffX: Hidden width of the feed-forward network for X.
        dim_ffE: Hidden width of the feed-forward network for E.
        dim_ffy: Hidden width of the feed-forward network for y.
        dropout: Dropout probability applied after each sub-layer. 0 disables.
        layer_norm_eps: Epsilon for numerical stability in LayerNorm.
        device: Torch device for all parameters.
        dtype: Dtype for all parameters.
    """

    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply one transformer layer to node, edge, and global features.

        Runs NodeEdgeBlock then applies residual + LayerNorm + feed-forward
        independently for each of X, E, and y, matching the standard
        transformer post-attention pattern applied to all three feature spaces.

        Args:
            X: Node feature matrix of shape (bs, n, dx).
            E: Edge feature tensor of shape (bs, n, n, de). Expected to be
                symmetric (undirected graph) with zeroed self-loops.
            y: Graph-level feature vectors of shape (bs, dy).
            node_mask: Boolean mask of shape (bs, n) where True indicates a
                real node. Padding nodes are zeroed out throughout.

        Returns:
            Tuple (X, E, y) with the same shapes as the inputs, representing
            the updated node, edge, and global features.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """Edge-biased multi-head self-attention with FiLM conditioning from global features.

    Implements the attention sub-layer at the core of each XEyTransformerLayer.
    The mechanism has three distinct outputs:

    - **newX**: Updated node features. Attention scores are biased by edge
      features E via FiLM (multiplicative + additive), then FiLM-conditioned
      again by y before the final linear projection.
    - **newE**: Updated edge features. Derived from the raw (pre-softmax) QK
      attention scores for each (i, j) pair, then FiLM-conditioned by y.
    - **new_y**: Updated global features. Aggregated from the current X and E
      via Xtoy/Etoy pooling, combined with a linear projection of the current y.

    FiLM conditioning (Perez et al., 2018) modulates a representation h with a
    context vector c as: h' = (W_mul · c + 1) * h + W_add · c, keeping h
    unchanged when the weight matrices are initialised near zero.

    Args:
        dx: Dimensionality of node features.
        de: Dimensionality of edge features.
        dy: Dimensionality of graph-level features.
        n_head: Number of attention heads. Must divide dx evenly, so that each
            head operates on df = dx // n_head dimensions.
        **kwargs: Forwarded to all Linear layers (e.g. device, dtype).
    """

    def __init__(self, dx: int, de: int, dy: int, n_head: int, **kwargs) -> None:
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute edge-biased attention and update all three feature types.

        The computation proceeds in five stages:

        1. **QK scores**: Queries and keys are projected from X and multiplied
           per head, giving raw attention logits Y of shape (bs, n, n, n_head, df).
        2. **Edge-biased attention**: E is projected to FiLM parameters that
           multiplicatively and additively modulate Y:
               Y = Y * (E_mul + 1) + E_add
           where E_mul, E_add ∈ R^{n×n×n_head×df}. This injects edge structure
           directly into the attention scores.
        3. **newE**: Y is flattened to (bs, n, n, dx) and FiLM-conditioned by y
           to produce the updated edge representation, projected to de dims.
        4. **newX**: Softmax is applied to the (masked) Y logits, values V are
           computed from X, and the weighted sum is FiLM-conditioned by y.
        5. **new_y**: Updated from a linear projection of the current y plus
           graph-level pooling of X (via Xtoy) and E (via Etoy).

        Args:
            X: Node feature matrix of shape (bs, n, dx).
            E: Edge feature tensor of shape (bs, n, n, de). Padding edges
                (where either endpoint is masked) are zeroed out.
            y: Graph-level feature vectors of shape (bs, dy).
            node_mask: Boolean mask of shape (bs, n). False positions are
                padding nodes and receive zero output.

        Returns:
            Tuple (newX, newE, new_y) with shapes (bs, n, dx), (bs, n, n, de),
            and (bs, dy) respectively.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X and E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    """Full graph transformer denoising network for node, edge, and global features.

    Serves as the score/noise-prediction network in the DiGress diffusion model
    (Vignac et al., 2022). Given a noisy graph (X_t, E_t, y_t) at diffusion
    step t, the model predicts the original clean graph components.

    The architecture has three stages:

    1. **Input projection**: Three separate MLPs map X, E, and y from their
       input dimensionalities to a shared hidden space (hidden_dims).
    2. **Transformer layers**: A stack of n_layers XEyTransformerLayers, each
       jointly updating all three feature types via edge-biased attention.
    3. **Output projection**: Three separate MLPs map back to output dims.

    Skip connections from the original inputs are added before the final
    output. E is symmetrised (averaged with its transpose) after projection
    to enforce the undirected-graph invariant; self-loops are zeroed out.

    Args:
        n_layers: Number of XEyTransformerLayer blocks to stack.
        input_dims: Dict with keys 'X', 'E', 'y' giving the input feature
            dimensionality for each type.
        hidden_mlp_dims: Dict with keys 'X', 'E', 'y' giving the hidden
            width of the input/output MLP projections.
        hidden_dims: Dict with keys 'dx', 'de', 'dy', 'n_head', 'dim_ffX',
            'dim_ffE' specifying the internal transformer dimensions.
        output_dims: Dict with keys 'X', 'E', 'y' giving the output feature
            dimensionality for each type.
        act_fn_in: Activation module applied inside the input-projection MLPs.
        act_fn_out: Activation module applied inside the output-projection MLPs.
    """

    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for _ in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor) -> utils.PlaceHolder:
        """Denoise a batched graph by predicting clean node, edge, and global features.

        Applies input projection, all transformer layers, output projection,
        and skip connections. The final edge tensor is symmetrised and has its
        diagonal (self-loops) zeroed out to maintain the undirected-graph
        invariant expected by the diffusion model.

        Args:
            X: Noisy node features of shape (bs, n, input_dims['X']).
            E: Noisy edge features of shape (bs, n, n, input_dims['E']).
                Should already be symmetric with zeroed diagonal.
            y: Noisy graph-level features of shape (bs, input_dims['y']).
            node_mask: Boolean mask of shape (bs, n). False marks padding nodes
                that should not influence the output.

        Returns:
            A PlaceHolder with fields X (bs, n, out_dim_X), E (bs, n, n, out_dim_E),
            and y (bs, out_dim_y), masked so padding positions are zero.
        """
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = utils.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
