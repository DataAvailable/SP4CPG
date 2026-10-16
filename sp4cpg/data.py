"""Versioned PyG input data for physical graphs and directed hypergraphs.

Types are *indices*, not pretrained features. CodeBERT features stay separate so
the type embeddings and relation scores can be learned by the classifier.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch_geometric.data import Data


SCHEMA_VERSION = 2
NODE_KINDS = (
    "FILE", "CLASS", "METHOD", "BLOCK", "MODIFIER", "CONTROL_STRUCTURE",
    "LITERAL", "PARAM", "LOCAL", "IDENTIFIER", "METHOD_RETURN", "CALL",
    "RETURN", "JUMP_TARGET", "FIELD_IDENTIFIER", "TYPE", "TYPE_REF",
)
RELATION_KINDS = ("AST", "CFG", "CDG", "DDG", "CTRL", "DATA")
NODE_TYPE_MAP = {kind: i for i, kind in enumerate(NODE_KINDS)}
RELATION_TYPE_MAP = {kind: i for i, kind in enumerate(RELATION_KINDS)}
NODE_KIND_ALIASES = {
    "METHOD_PARAMETER_IN": "PARAM", "METHOD_PARAMETER_OUT": "PARAM",
    "PARAMETER": "PARAM", "TYPE_DECL": "CLASS",
}


def node_kind_index(kind: str) -> int:
    """Map the paper's 17 categories and explicit Joern spelling aliases.

    Operator roles are CALL nodes, rather than additional embedding categories.
    Unsupported kinds fail visibly instead of being mapped to an invented type.
    """
    if kind.startswith("<operator>."):
        kind = "CALL"
    kind = NODE_KIND_ALIASES.get(kind, kind)
    if kind not in NODE_TYPE_MAP:
        raise ValueError(f"Unsupported node kind {kind!r}; expected one of {NODE_KINDS}")
    return NODE_TYPE_MAP[kind]


class GraphData(Data):
    """A graph with independent node and relation index spaces.

    ``edge_index`` contains ordinary node-to-node physical witnesses. Only
    ``hyperedge_index`` uses a node row and a relation row, whose batch offsets
    must therefore differ. Standard PyG Data increments both rows by num_nodes.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "hyperedge_index":
            return torch.tensor([[self.num_nodes], [int(self.num_hyperedges)]],
                                dtype=torch.long, device=value.device)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "hyperedge_index":
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)


def _resolve_endpoint(original_id: str, dictionary: Mapping[str, Any],
                      memberships: set[str]) -> str:
    entry = dictionary.get(str(original_id))
    if not isinstance(entry, Mapping):
        raise ValueError(f"Missing dictionary entry for binding endpoint {original_id!r}")
    # A carried occurrence is represented by its actual content carrier.
    candidates = (entry.get("carrier"), entry.get("representative"))
    for value in candidates:
        if value is not None and str(value) in memberships:
            return str(value)
    raise ValueError(f"Binding endpoint {original_id!r} does not match active memberships")


def _check_dictionary(nodes, dictionary):
    """Check active entry ownership before converting logical endpoints to nodes."""
    if not isinstance(dictionary, Mapping) or not dictionary:
        raise ValueError("A complete original-entry dictionary is required")
    active_ids = {str(node["id"]) for node in nodes}
    ownership = {}
    for node in nodes:
        entries = [str(value) for value in node.get("entries", [])]
        if not entries or len(entries) != len(set(entries)):
            raise ValueError(f"Node {node['id']!r} needs distinct original entries")
        for original_id in entries:
            if original_id in ownership:
                raise ValueError(f"Original entry {original_id!r} belongs to multiple physical nodes")
            ownership[original_id] = str(node["id"])
            if original_id not in dictionary:
                raise ValueError(f"Missing dictionary entry {original_id!r}")
    for original_id, entry in dictionary.items():
        if str(entry.get("id", "")) != str(original_id):
            raise ValueError(f"Dictionary key/id mismatch for {original_id!r}")
        status = entry.get("status")
        representative, carrier = entry.get("representative"), entry.get("carrier")
        if status in {"retained", "merged"}:
            owner = None if representative is None else str(representative)
            if carrier is not None:
                raise ValueError(f"Retained/merged entry {original_id!r} cannot also have a carrier")
        elif status == "carried":
            owner = None if carrier is None else str(carrier)
            if representative is not None:
                raise ValueError(f"Carried entry {original_id!r} cannot also have a representative")
        elif status == "removed":
            if representative is not None or carrier is not None or str(original_id) in ownership:
                raise ValueError(f"Removed entry {original_id!r} still has an active owner")
            continue
        else:
            raise ValueError(f"Invalid dictionary status for {original_id!r}: {status!r}")
        if owner not in active_ids or ownership.get(str(original_id)) != owner:
            raise ValueError(f"Dictionary entry ownership mismatch for {original_id!r}")


def graph_to_data(graph: Mapping[str, Any], features, label=None) -> GraphData:
    """Convert canonical schema v2 without expanding logical associations.

    Physical binary-GNN edges are the witnessed endpoint pairs decoded from
    active relations' bindings, never an unverified Cartesian product. Every
    relation remains a distinct incidence column even for identical memberships.
    """
    if graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Expected graph schema_version=2; regenerate legacy graph inputs")
    nodes = graph.get("nodes", [])
    if not nodes:
        raise ValueError("Cannot classify an empty graph")
    sample_id = str(graph.get("graph_id", ""))
    if not sample_id:
        raise ValueError("graph_id must be a nonempty stable sample ID")
    node_ids = [str(node["id"]) for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Duplicate physical node IDs")
    node_index = {name: i for i, name in enumerate(node_ids)}
    x = torch.as_tensor(features, dtype=torch.float32).detach().cpu().clone()
    if x.ndim != 2 or x.shape[0] != len(nodes) or x.shape[1] < 1:
        raise ValueError(f"Expected features [num_nodes, width], got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("Node features contain NaN or infinity")

    incidences, roles, relation_types = [], [], []
    binary_edges = set()
    relation_ids = set()
    dictionary = graph.get("dictionary", {})
    _check_dictionary(nodes, dictionary)
    for relation_index, relation in enumerate(graph.get("relations", [])):
        relation_id = str(relation["id"])
        if relation_id in relation_ids:
            raise ValueError(f"Duplicate relation ID {relation_id!r}")
        relation_ids.add(relation_id)
        kind = str(relation["kind"]).upper()
        if kind not in RELATION_TYPE_MAP:
            raise ValueError(f"Unsupported relation kind {kind!r}")
        sources = [str(v) for v in relation.get("sources", [])]
        targets = [str(v) for v in relation.get("targets", [])]
        if not sources or not targets:
            raise ValueError(f"Relation {relation_id!r} needs source and target members")
        if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
            raise ValueError(f"Duplicate members in relation {relation_id!r}")
        if kind not in {"CTRL", "DATA"} and (len(sources) != 1 or len(targets) != 1):
            raise ValueError(f"Binary relation {relation_id!r} must have singleton source/target")
        for members, role in ((sources, 0), (targets, 1)):
            for name in members:
                if name not in node_index:
                    raise ValueError(f"Relation {relation_id!r} references missing node {name!r}")
                incidences.append((node_index[name], relation_index))
                roles.append(role)
        relation_types.append(RELATION_TYPE_MAP[kind])
        bindings = relation.get("bindings", [])
        if not bindings:
            raise ValueError(f"Relation {relation_id!r} has no endpoint witness bindings")
        witnessed_sources, witnessed_targets, witnessed_pairs = set(), set(), set()
        source_ports, target_ports = {}, {}
        binding_kind = {"CTRL": "CFG", "DATA": "DDG"}.get(kind, kind)
        for binding in bindings:
            if str(binding.get("kind", "")).upper() != binding_kind:
                raise ValueError(f"Binding kind does not match relation {relation_id!r} ({kind})")
            source = _resolve_endpoint(str(binding["source"]), dictionary, set(sources))
            target = _resolve_endpoint(str(binding["target"]), dictionary, set(targets))
            binary_edges.add((node_index[source], node_index[target]))
            witnessed_sources.add(source)
            witnessed_targets.add(target)
            witnessed_pairs.add((source, target))
            source_ports.setdefault(source, set()).add(str(binding["source"]))
            target_ports.setdefault(target, set()).add(str(binding["target"]))
        if witnessed_sources != set(sources) or witnessed_targets != set(targets):
            raise ValueError(f"Unwitnessed physical member in relation {relation_id!r}")
        if kind in {"CTRL", "DATA"}:
            if set(sources) & set(targets):
                raise ValueError(f"Hyperedge {relation_id!r} has overlapping source/target sets")
            expected_pairs = {(s, t) for s in sources for t in targets}
            if witnessed_pairs != expected_pairs:
                raise ValueError(f"Unwitnessed physical source-target pair in hyperedge {relation_id!r}")
            if any(len(entries) != 1 for entries in (*source_ports.values(), *target_ports.values())):
                raise ValueError(f"Hyperedge {relation_id!r} mixes different original endpoint ports")

    def index_tensor(pairs):
        return (torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs
                else torch.empty((2, 0), dtype=torch.long))

    metadata = graph.get("metadata", {})
    representation = str(metadata.get("representation", "HCPG")).upper()
    if representation not in {"CPG", "HCPG"}:
        raise ValueError(f"Unknown representation {representation!r}")
    original_count = int(metadata.get("original_node_count", len(dictionary) or len(nodes)))
    if original_count != len(dictionary):
        raise ValueError("original_node_count must equal the complete original dictionary size")
    data = GraphData(
        x=x, edge_index=index_tensor(sorted(binary_edges)),
        hyperedge_index=index_tensor(incidences),
        incidence_role=torch.tensor(roles, dtype=torch.long),
        relation_types=torch.tensor(relation_types, dtype=torch.long),
        node_types=torch.tensor([node_kind_index(str(n["kind"])) for n in nodes],
                                dtype=torch.long),
        num_nodes=len(nodes), num_hyperedges=len(relation_types), sample_id=sample_id,
        original_node_count=original_count, schema_version=SCHEMA_VERSION,
        representation=representation,
    )
    if label is not None:
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("label must be integer 0 or 1")
        data.y = torch.tensor([int(label)], dtype=torch.long)
    return data
