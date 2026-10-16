"""Type-aware, directed normalized propagation from the revised manuscript.

Each layer averages P^q X Phi for q=1..K. There is no zero-hop/residual term,
convolution bias, feature-derived hyperedge, or max-pooling branch.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool


def _inverse_sqrt(degree):
    # Avoid rsqrt at zero or a tiny value whose backward derivative overflows.
    positive = degree > 0
    safe = torch.where(positive, degree, torch.ones_like(degree))
    return torch.where(positive, safe.rsqrt(), torch.zeros_like(degree))


def _memberships(hyperedge_index, roles, directed):
    if directed:
        return hyperedge_index[:, roles == 0], hyperedge_index[:, roles == 1]
    # Union membership (a self-loop endpoint appears only once), retaining each
    # relation's distinct column even if another relation has the same members.
    union = torch.unique(hyperedge_index, dim=1)
    return union, union


def normalized_propagate(x, hyperedge_index, incidence_role, relation_weights,
                         directed=True):
    """Apply D_T^-1/2 H_T W D_E^-1 H_S^T D_S^-1/2 to x.

    In undirected mode H_S=H_T=union incidence, recovering normalized HGNN.
    The sparse gather/scatter implementation uses independent relation IDs and
    scales to graphs where constructing a dense n-by-n operator is unsuitable.
    """
    if hyperedge_index.ndim != 2 or hyperedge_index.shape[0] != 2:
        raise ValueError("hyperedge_index must have shape [2, memberships]")
    if incidence_role.numel() != hyperedge_index.shape[1]:
        raise ValueError("incidence_role must align with memberships")
    if incidence_role.numel() and not torch.all((incidence_role == 0) | (incidence_role == 1)):
        raise ValueError("Incidence roles must be 0 (source) or 1 (target)")
    output_dtype = x.dtype
    # Tiny positive softmax weights can give a finite normalized operator while
    # float32 rsqrt's intermediate backward derivative overflows. Compute the
    # normalization/aggregation in float64, then return the input feature dtype.
    # No positive degree is clamped, so the mathematical operator is unchanged.
    work_dtype = torch.float64 if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else x.dtype
    x = x.to(work_dtype)
    relation_weights = relation_weights.to(work_dtype)
    n, m = x.shape[0], relation_weights.numel()
    if hyperedge_index.numel():
        if hyperedge_index.min() < 0 or hyperedge_index[0].max() >= n or hyperedge_index[1].max() >= m:
            raise ValueError("Incidence index is outside node/relation bounds")
    if not torch.isfinite(relation_weights).all() or (relation_weights < 0).any():
        raise ValueError("Relation weights must be finite and nonnegative")
    if m:
        # A common positive scale cancels between W and the two weighted degree
        # factors. Removing it avoids needless tiny factors in homogeneous cases.
        scale = relation_weights.max()
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        relation_weights = relation_weights / safe_scale
    source, target = _memberships(hyperedge_index, incidence_role, directed)
    source_degree = x.new_zeros(n)
    target_degree = x.new_zeros(n)
    source_count = x.new_zeros(m)
    if source.numel():
        source_degree.index_add_(0, source[0], relation_weights[source[1]])
        source_count.index_add_(0, source[1], x.new_ones(source.shape[1]))
    if target.numel():
        target_degree.index_add_(0, target[0], relation_weights[target[1]])
    source_factor = _inverse_sqrt(source_degree)
    target_factor = _inverse_sqrt(target_degree)
    relation_features = x.new_zeros((m, x.shape[1]))
    if source.numel():
        relation_features.index_add_(0, source[1], x[source[0]] * source_factor[source[0], None])
    relation_features = relation_features * (
        relation_weights / source_count.clamp_min(1)
    )[:, None]
    out = x.new_zeros(x.shape)
    if target.numel():
        out.index_add_(0, target[0], relation_features[target[1]])
    return (out * target_factor[:, None]).to(output_dtype)


class TAFHGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, type_dim=64, hop_size=3,
                 use_types=True, directed=True):
        super().__init__()
        if hop_size < 1 or int(hop_size) != hop_size:
            raise ValueError("hop_size must be a positive integer")
        self.hop_size = int(hop_size)
        self.use_types = use_types
        self.directed = directed
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if use_types:
            # The relation embedding table is shared across layers by TAFHGCN;
            # each layer learns its own score MLP.
            self.type_score = nn.Sequential(nn.Linear(type_dim, type_dim), nn.ReLU(),
                                            nn.Linear(type_dim, 1))

    def kind_weights(self, relation_embeddings=None):
        """Softmax over all six kinds, including kinds absent from this graph."""
        if not self.use_types:
            return self.linear.weight.new_ones(6)
        if relation_embeddings is None or relation_embeddings.shape[0] != 6:
            raise ValueError("Expected the six relation-type embeddings")
        scores = self.type_score(relation_embeddings).squeeze(-1)
        # Preserve very small positive kind weights beyond float32 softmax range.
        softmax_dtype = torch.float64 if scores.dtype in (torch.float16, torch.bfloat16, torch.float32) else scores.dtype
        return torch.softmax(scores, dim=0, dtype=softmax_dtype)

    def forward(self, x, data, relation_embeddings=None):
        if data.relation_types.numel() and (
            data.relation_types.min() < 0 or data.relation_types.max() >= 6
        ):
            raise ValueError("Relation types must be in 0..5")
        weights = self.kind_weights(relation_embeddings)[data.relation_types]
        current = self.linear(x)
        total = torch.zeros_like(current)
        for _ in range(self.hop_size):
            current = normalized_propagate(current, data.hyperedge_index,
                                           data.incidence_role, weights, self.directed)
            total = total + current
        return F.relu(total / self.hop_size)


class TAFHGCN(nn.Module):
    architecture = "TAFHGCN"

    def __init__(self, dropout=0.4, in_channels=768, hidden_channels=512,
                 num_classes=2, num_layers=3, hop_size=3, type_dim=64,
                 use_types=True, directed=True):
        super().__init__()
        if num_layers < 1 or int(num_layers) != num_layers:
            raise ValueError("num_layers must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.use_types = bool(use_types)
        self.directed = bool(directed)
        self.dropout = float(dropout)
        self.config = dict(dropout=dropout, in_channels=in_channels,
                           hidden_channels=hidden_channels, num_classes=num_classes,
                           num_layers=num_layers, hop_size=hop_size, type_dim=type_dim,
                           use_types=use_types, directed=directed)
        if use_types:
            self.node_type_embedding = nn.Embedding(17, type_dim)
            self.relation_type_embedding = nn.Embedding(6, type_dim)
        widths = [in_channels + (type_dim if use_types else 0)] + [hidden_channels] * int(num_layers)
        self.layers = nn.ModuleList([
            TAFHGCNLayer(widths[i], widths[i+1], type_dim, hop_size, use_types, directed)
            for i in range(int(num_layers))
        ])
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def encode_nodes(self, data):
        x = data.x
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Expected a nonempty node feature matrix")
        if self.use_types:
            if data.node_types.numel() != x.shape[0] or data.node_types.min() < 0 or data.node_types.max() >= 17:
                raise ValueError("Node types must align with nodes and be in 0..16")
            x = torch.cat((x, self.node_type_embedding(data.node_types)), dim=1)
            relation_embeddings = self.relation_type_embedding.weight
        else:
            relation_embeddings = None
        for layer in self.layers:
            x = layer(x, data, relation_embeddings)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def encode_graph(self, data):
        x = self.encode_nodes(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)

    def forward(self, data):
        return self.classifier(self.encode_graph(data))
