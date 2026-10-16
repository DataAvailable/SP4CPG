"""Standard normalized undirected HGNN baseline on existing memberships.

Each layer is ReLU(D_v^-1/2 H D_e^-1 H^T D_v^-1/2 X Phi), with unit relation
weights and union memberships. There are no type embeddings, direction, or
additional feature-based hyperedges. The standard baseline uses one hop per
trainable layer; hop_size other than 1 is rejected to expose configuration errors.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool

from .taf_hgcn import normalized_propagate


class HGNN(nn.Module):
    architecture = "HGNN"

    def __init__(self, dropout=0.4, in_channels=768, hidden_channels=512,
                 num_classes=2, num_layers=3, hop_size=1, **kwargs):
        super().__init__()
        # Common factory arguments may specify the inapplicable type/direction
        # options, but these never alter this baseline's unit undirected operator.
        unknown = set(kwargs) - {"type_dim", "use_types", "directed"}
        if unknown:
            raise TypeError(f"Unexpected HGNN arguments: {sorted(unknown)}")
        if hop_size != 1:
            raise ValueError("Standard HGNN uses hop_size=1 per trainable layer")
        if num_layers < 1 or int(num_layers) != num_layers:
            raise ValueError("num_layers must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.dropout = float(dropout)
        self.config = dict(dropout=dropout, in_channels=in_channels,
                           hidden_channels=hidden_channels, num_classes=num_classes,
                           num_layers=num_layers, hop_size=1)
        widths = [in_channels] + [hidden_channels] * int(num_layers)
        self.layers = nn.ModuleList([
            nn.Linear(widths[i], widths[i+1], bias=False) for i in range(int(num_layers))
        ])
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def encode_nodes(self, data):
        x = data.x
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Expected a nonempty node feature matrix")
        weights = x.new_ones(data.relation_types.numel())
        for layer in self.layers:
            x = F.relu(normalized_propagate(layer(x), data.hyperedge_index,
                                            data.incidence_role, weights, directed=False))
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
