"""Type-aware source-to-relation-to-target convolution and graph classification."""

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


NODE_TYPE_COUNT = 17
RELATION_TYPE_COUNT = 6


def _indices(x, hyperedge_index, incidence_role, relation_types):
    if not torch.is_tensor(x) or x.ndim != 2 or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("x must contain finite floating-point node features [nodes, channels]")
    if not torch.is_tensor(hyperedge_index) or hyperedge_index.dtype != torch.long or hyperedge_index.ndim != 2 or hyperedge_index.size(0) != 2:
        raise ValueError("hyperedge_index must be a long tensor [2, memberships]")
    if not torch.is_tensor(incidence_role) or incidence_role.dtype != torch.long or incidence_role.ndim != 1 or incidence_role.numel() != hyperedge_index.size(1):
        raise ValueError("incidence_role must specify one long source/target role for each membership")
    if not torch.is_tensor(relation_types) or relation_types.dtype != torch.long or relation_types.ndim != 1:
        raise ValueError("relation_types must be a long vector of six-kind indices")
    if any(value.device != x.device for value in (hyperedge_index, incidence_role, relation_types)):
        raise ValueError("Node features and incidence tensors must be on the same device")
    if relation_types.numel() and ((relation_types < 0).any() or (relation_types >= RELATION_TYPE_COUNT).any()):
        raise ValueError("Relation kinds must be in [0, 5]")
    if incidence_role.numel() and ((incidence_role < 0).any() or (incidence_role > 1).any()):
        raise ValueError("Incidence roles must be 0 for source and 1 for target")
    n, m = x.size(0), relation_types.numel()
    if hyperedge_index.numel():
        if (hyperedge_index < 0).any() or (hyperedge_index[0] >= n).any() or (hyperedge_index[1] >= m).any():
            raise ValueError("Incidence node/relation index is outside its index space")
        keys = (hyperedge_index[1] * n + hyperedge_index[0]) * 2 + incidence_role
        if torch.unique(keys).numel() != keys.numel():
            raise ValueError("Duplicate memberships change binary incidence values")
    source = incidence_role == 0
    target = incidence_role == 1
    source_count = torch.bincount(hyperedge_index[1, source], minlength=m)
    target_count = torch.bincount(hyperedge_index[1, target], minlength=m)
    if m and ((source_count == 0).any() or (target_count == 0).any()):
        raise ValueError("Every relation needs at least one source and at least one target")
    if m and (((source_count != 1) | (target_count != 1)) & (relation_types < 4)).any():
        raise ValueError("Binary relation kinds require one source and one target")
    if m and hyperedge_index.numel():
        source_keys = hyperedge_index[1, source] * n + hyperedge_index[0, source]
        target_keys = hyperedge_index[1, target] * n + hyperedge_index[0, target]
        hyper_source = source_keys[relation_types[hyperedge_index[1, source]] >= 4]
        hyper_target = target_keys[relation_types[hyperedge_index[1, target]] >= 4]
        if hyper_source.numel() and hyper_target.numel() and torch.isin(hyper_source, hyper_target).any():
            raise ValueError("Control/data hyperedges must have disjoint source and target members")
    return source, target, source_count


def _incident_factors(node_ids, relation_ids, relation_types, type_scores, node_count):
    """Return sqrt(w(e)/d(v)) without forming a small degree's reciprocal."""
    counts = torch.zeros(node_count * RELATION_TYPE_COUNT, dtype=torch.float64, device=type_scores.device)
    if not node_ids.numel():
        return counts[:0]
    kind_ids = relation_types[relation_ids]
    counts.index_add_(0, node_ids * RELATION_TYPE_COUNT + kind_ids,
                      torch.ones(node_ids.numel(), dtype=torch.float64, device=counts.device))
    counts = counts.view(node_count, RELATION_TYPE_COUNT)
    active = counts.sum(dim=1) > 0
    active_counts = counts[active]
    scores = type_scores.to(torch.float64)
    present = active_counts > 0
    maximum = scores.expand_as(active_counts).masked_fill(~present, float("-inf")).max(dim=1).values
    shifted = scores.unsqueeze(0) - maximum.unsqueeze(1)
    log_counts = active_counts.masked_fill(~present, 1).log().masked_fill(~present, float("-inf"))
    log_denominator = torch.logsumexp(log_counts + shifted, dim=1)
    node_maximum = torch.zeros(node_count, dtype=torch.float64, device=counts.device)
    node_log_denominator = torch.zeros_like(node_maximum)
    node_maximum[active] = maximum
    node_log_denominator[active] = log_denominator
    # The common six-kind softmax denominator cancels from w(e)/d(v).
    log_ratio = scores[kind_ids] - node_maximum[node_ids] - node_log_denominator[node_ids]
    return torch.exp(0.5 * log_ratio)


def directed_propagation(x, hyperedge_index, incidence_role, relation_types, type_scores):
    """Apply D_T^-1/2 H_T W D_E^-1 H_S^T D_S^-1/2 to x.

    ``type_scores`` contains one scalar for each of the six relation kinds.
    The source and target incident factors are bounded by one. Their product
    equals w(e)/sqrt(d_S(source)*d_T(target)); zero-degree rows stay zero.
    """
    source, target, source_count = _indices(x, hyperedge_index, incidence_role, relation_types)
    if not torch.is_tensor(type_scores) or type_scores.shape != (RELATION_TYPE_COUNT,) or not type_scores.is_floating_point() or not torch.isfinite(type_scores).all():
        raise ValueError("type_scores must contain six finite floating-point values")
    if type_scores.device != x.device:
        raise ValueError("Type scores and node features must be on the same device")
    n, m = x.size(0), relation_types.numel()
    source_nodes, source_relations = hyperedge_index[:, source]
    target_nodes, target_relations = hyperedge_index[:, target]
    source_factor = _incident_factors(source_nodes, source_relations, relation_types, type_scores, n)
    target_factor = _incident_factors(target_nodes, target_relations, relation_types, type_scores, n)
    messages = torch.zeros((m, x.size(1)), dtype=torch.float64, device=x.device)
    messages.index_add_(0, source_relations, x.to(torch.float64)[source_nodes] * source_factor.unsqueeze(1))
    if m:
        messages = messages / source_count.to(torch.float64).unsqueeze(1)
    result = torch.zeros((n, x.size(1)), dtype=torch.float64, device=x.device)
    result.index_add_(0, target_nodes, messages[target_relations] * target_factor.unsqueeze(1))
    return result.to(x.dtype)


class FlowSensitiveLayer(nn.Module):
    def __init__(self, in_channels, out_channels, type_dim, hop_size):
        super().__init__()
        self.hop_size = hop_size
        self.transform = nn.Linear(in_channels, out_channels, bias=False)
        self.type_mlp = nn.Sequential(nn.Linear(type_dim, type_dim), nn.ReLU(), nn.Linear(type_dim, 1))

    def type_scores(self, type_embeddings):
        return self.type_mlp(type_embeddings).squeeze(1)

    def type_weights(self, type_embeddings):
        return torch.softmax(self.type_scores(type_embeddings).to(torch.float64), dim=0)

    def forward(self, x, hyperedge_index, incidence_role, relation_types, type_embeddings):
        scores = self.type_scores(type_embeddings)
        signal = self.transform(x)
        accumulated = torch.zeros_like(signal)
        for _ in range(self.hop_size):
            signal = directed_propagation(signal, hyperedge_index, incidence_role, relation_types, scores)
            accumulated = accumulated + signal
        return F.relu(accumulated / self.hop_size)


class TAFHGCN(nn.Module):
    def __init__(self, dropout=0.4, in_channels=768, hidden_channels=512, num_classes=2,
                 num_layers=3, hop_size=3, type_dim=64, *, use_residual=False,
                 use_attention=False, use_layer_norm=False):
        super().__init__()
        for name, value in (("in_channels", in_channels), ("hidden_channels", hidden_channels),
                            ("num_classes", num_classes), ("num_layers", num_layers),
                            ("hop_size", hop_size), ("type_dim", type_dim)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name, value in (("use_residual", use_residual), ("use_attention", use_attention), ("use_layer_norm", use_layer_norm)):
            if value is not False:
                raise ValueError(f"{name}=True is not part of this propagation operator")
        self.dropout = float(dropout)
        self.in_channels = int(in_channels)
        self.num_layers = int(num_layers)
        self.hop_size = int(hop_size)
        self.node_type_embedding = nn.Embedding(NODE_TYPE_COUNT, type_dim)
        self.relation_type_embedding = nn.Embedding(RELATION_TYPE_COUNT, type_dim)
        widths = [in_channels + type_dim] + [hidden_channels] * num_layers
        self.layers = nn.ModuleList(FlowSensitiveLayer(widths[i], widths[i + 1], type_dim, hop_size)
                                    for i in range(num_layers))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def _input(self, data):
        required = ("x", "hyperedge_index", "incidence_role", "node_types", "relation_types", "schema_version")
        if any(getattr(data, name, None) is None for name in required):
            raise ValueError("Directed incidence, node/relation types and schema_version=2 are required")
        x = data.x
        _indices(x, data.hyperedge_index, data.incidence_role, data.relation_types)
        if x.size(0) == 0 or x.size(1) != self.in_channels:
            raise ValueError("x must have nonempty nodes and match in_channels")
        if x.device != self.node_type_embedding.weight.device:
            raise ValueError("Model parameters and node features must be on the same device")
        node_types = data.node_types
        if not torch.is_tensor(node_types) or node_types.dtype != torch.long or node_types.shape != (x.size(0),) or node_types.device != x.device:
            raise ValueError("node_types must be one long kind index per node on the feature device")
        if (node_types < 0).any() or (node_types >= NODE_TYPE_COUNT).any():
            raise ValueError("Node kinds must be in [0, 16]")
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if not torch.is_tensor(batch) or batch.dtype != torch.long or batch.shape != (x.size(0),) or batch.device != x.device or (batch < 0).any():
            raise ValueError("batch must contain nonnegative long graph IDs for all nodes")
        graph_count = int(batch.max().item()) + 1
        if not torch.equal(torch.unique(batch), torch.arange(graph_count, device=x.device)):
            raise ValueError("batch graph IDs must be contiguous and every graph nonempty")
        version = data.schema_version
        if torch.is_tensor(version):
            if version.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) or version.numel() not in (1, graph_count) or not (version == 2).all():
                raise ValueError("Every graph must have schema_version=2")
        elif isinstance(version, bool) or not isinstance(version, Integral) or version != 2:
            raise ValueError("Every graph must have schema_version=2")
        m = data.relation_types.numel()
        count = getattr(data, "num_hyperedges", None)
        if count is None:
            count = getattr(data, "n_hyperedges", None)
        if torch.is_tensor(count):
            if count.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) or count.numel() not in (1, graph_count) or (count < 0).any() or int(count.sum().item()) != m:
                raise ValueError("num_hyperedges must count the relation index space")
        elif isinstance(count, bool) or not isinstance(count, Integral) or count != m:
            raise ValueError("num_hyperedges must count the relation index space")
        if m:
            owners = torch.full((m,), -1, dtype=torch.long, device=x.device)
            owners[data.hyperedge_index[1]] = batch[data.hyperedge_index[0]]
            if not torch.equal(owners[data.hyperedge_index[1]], batch[data.hyperedge_index[0]]):
                raise ValueError("A relation cannot contain nodes from different graphs")
            if torch.is_tensor(count) and count.numel() == graph_count and not torch.equal(
                    torch.bincount(owners, minlength=graph_count).to(count.device), count.reshape(-1).long()):
                raise ValueError("Per-graph relation counts disagree with incidence ownership")
        return x, node_types, batch, graph_count

    def _encode(self, data, x, node_types):
        x = torch.cat((x, self.node_type_embedding(node_types)), dim=1)
        for layer in self.layers:
            x = layer(x, data.hyperedge_index, data.incidence_role, data.relation_types,
                      self.relation_type_embedding.weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def encode_nodes(self, data):
        x, node_types, _, _ = self._input(data)
        return self._encode(data, x, node_types)

    def forward(self, data):
        x, node_types, batch, graph_count = self._input(data)
        encoded = self._encode(data, x, node_types)
        pooled = encoded.new_zeros((graph_count, encoded.size(1)))
        pooled.index_add_(0, batch, encoded)
        pooled = pooled / torch.bincount(batch, minlength=graph_count).to(encoded.dtype).unsqueeze(1)
        return self.classifier(pooled)


HGCN = TAFHGCN
