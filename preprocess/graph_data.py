"""PyG tensors with separate node and directed relation index spaces."""
from __future__ import annotations

import json
from numbers import Integral
from typing import Mapping

import torch
from torch_geometric.data import Data


SCHEMA_VERSION = 2
NODE_KINDS = (
    "FILE", "CLASS", "METHOD", "BLOCK", "MODIFIER", "CONTROL_STRUCTURE",
    "LITERAL", "PARAM", "LOCAL", "IDENTIFIER", "METHOD_RETURN", "CALL",
    "RETURN", "JUMP_TARGET", "FIELD_IDENTIFIER", "TYPE", "TYPE_REF",
)
RELATION_KINDS = ("AST", "CFG", "CDG", "DDG", "CTRL", "DATA")
NODE_TYPE_MAP = {name: index for index, name in enumerate(NODE_KINDS)}
EDGE_TYPE_MAP = {name: index for index, name in enumerate(RELATION_KINDS)}
RELATION_TYPE_MAP = EDGE_TYPE_MAP
NODE_KIND_ALIASES = {
    "METHOD_PARAMETER_IN": "PARAM", "METHOD_PARAMETER_OUT": "PARAM",
    "PARAMETER": "PARAM", "TYPE_DECL": "CLASS",
}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _record(entry):
    return {key: entry[key] for key in ("id", "kind", "code", "attributes")}


def _has_content(node, entry):
    records = node["attributes"].get("retained_contents", [])
    return isinstance(records, list) and any(_json(record) == _json(_record(entry)) for record in records)


class GraphData(Data):
    """Keep node and relation offsets independent during PyG batching."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "hyperedge_index":
            return torch.tensor([[self.num_nodes], [int(self.num_hyperedges)]],
                                dtype=torch.long, device=value.device)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "hyperedge_index":
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)


def node_kind_index(kind):
    kind = NODE_KIND_ALIASES.get(kind, kind)
    if kind not in NODE_TYPE_MAP:
        raise ValueError(f"Unsupported canonical node kind: {kind!r}")
    return NODE_TYPE_MAP[kind]


def _check_dictionary(nodes, dictionary):
    from preprocess.graph_core import _known
    if not isinstance(dictionary, Mapping) or not dictionary:
        raise ValueError("A complete original-entry dictionary is required")
    node_ids = {str(node["id"]) for node in nodes}
    by_id = {str(node["id"]): node for node in nodes}
    merged = {}
    ownership = {}
    for node in nodes:
        entries = [str(value) for value in node.get("entries", [])]
        if not entries or len(entries) != len(set(entries)):
            raise ValueError(f"Node {node['id']!r} needs distinct original entries")
        for entry_id in entries:
            if entry_id in ownership:
                raise ValueError(f"Original entry {entry_id!r} has multiple physical owners")
            if entry_id not in dictionary:
                raise ValueError(f"Missing dictionary entry: {entry_id!r}")
            ownership[entry_id] = str(node["id"])
    for original_id, entry in dictionary.items():
        if str(entry.get("id", "")) != str(original_id):
            raise ValueError(f"Dictionary key/id mismatch: {original_id!r}")
        status = entry.get("status")
        representative, carrier = entry.get("representative"), entry.get("carrier")
        if status in {"retained", "merged"}:
            owner = None if representative is None else str(representative)
            if carrier is not None:
                raise ValueError(f"Entry {original_id!r} has conflicting ownership fields")
            if status == "retained" and owner != str(original_id):
                raise ValueError(f"Retained entry {original_id!r} changed its original identity")
            if status == "merged" and owner in dictionary:
                raise ValueError(f"Merged entry {original_id!r} requires a fresh representative")
        elif status == "carried":
            owner = None if carrier is None else str(carrier)
            if representative is not None:
                raise ValueError(f"Carried entry {original_id!r} also has a representative")
        elif status == "removed":
            if representative is not None or carrier is not None or str(original_id) in ownership:
                raise ValueError(f"Removed entry {original_id!r} has an active owner")
            continue
        else:
            raise ValueError(f"Invalid dictionary status for {original_id!r}: {status!r}")
        if owner not in node_ids or ownership.get(str(original_id)) != owner:
            raise ValueError(f"Dictionary entry ownership mismatch: {original_id!r}")
        node = by_id[owner]
        if status == "retained":
            if node["kind"] != entry["kind"] or node["code"] != entry["code"]:
                raise ValueError(f"Retained entry content/kind mismatch: {original_id!r}")
            original_attributes = entry["attributes"]
            if any(key not in node["attributes"] for key in original_attributes) or _json({
                key: node["attributes"].get(key) for key in original_attributes
            }) != _json(original_attributes):
                raise ValueError(f"Retained entry attributes mismatch: {original_id!r}")
        elif status == "carried":
            association = entry.get("carrier_original")
            if association is None or str(association) == str(original_id):
                raise ValueError(f"Carried entry needs an original carrier association: {original_id!r}")
            _endpoint(str(association), dictionary, {owner})
            if not _has_content(node, entry):
                raise ValueError(f"Missing carried content record: {original_id!r}")
        else:
            merged.setdefault(owner, []).append(entry)
    declaration_kinds = {"LOCAL", "PARAM", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "PARAMETER"}
    for owner, entries in merged.items():
        node = by_id[owner]
        if len(entries) < 2:
            raise ValueError(f"Merged node {owner!r} needs multiple original entries")
        kind = node["kind"]
        if kind not in declaration_kinds | {"TYPE_REF", "METHOD_RETURN"}:
            raise ValueError(f"Node kind cannot share a representative: {kind!r}")
        if any(entry["kind"] != kind for entry in entries):
            raise ValueError(f"Merged entries must retain a common kind: {owner!r}")
        for field in ("declared_type", "function_context"):
            value = node["attributes"].get(field)
            if not _known(value):
                raise ValueError(f"Merged node {owner!r} requires known {field}")
            if any(field not in entry["attributes"] or _json(entry["attributes"][field]) != _json(value)
                   for entry in entries):
                raise ValueError(f"Merged entries disagree on {field}: {owner!r}")
        for entry in entries:
            required = ("name", "scope") if kind in declaration_kinds else ("occurrence",)
            if any(not _known(entry["attributes"].get(field)) for field in required):
                raise ValueError(f"Merged entry lacks declaration/occurrence identity: {entry['id']!r}")
            if not _has_content(node, entry):
                raise ValueError(f"Missing merged content record: {entry['id']!r}")


def _endpoint(original_id, dictionary, members):
    entry = dictionary.get(str(original_id))
    if not isinstance(entry, Mapping):
        raise ValueError(f"Missing binding dictionary entry: {original_id!r}")
    field = "carrier" if entry.get("status") == "carried" else "representative"
    physical = entry.get(field)
    if physical is None or str(physical) not in members:
        raise ValueError(f"Binding endpoint {original_id!r} does not match active memberships")
    return str(physical)


def _graph_layout(graph):
    from preprocess.graph_core import normalize_graph
    if not isinstance(graph, Mapping) or graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Expected graph schema_version=2")
    if not isinstance(graph.get("dictionary"), Mapping) or not graph["dictionary"]:
        raise ValueError("A complete original-entry dictionary is required")
    if any(not isinstance(relation.get("bindings"), list) or not relation["bindings"]
           for relation in graph.get("relations", [])):
        raise ValueError("Every active relation requires binding witnesses")
    input_ids = [str(node["id"]) for node in graph.get("nodes", [])]
    if not input_ids or len(set(input_ids)) != len(input_ids):
        raise ValueError("Graph needs nonempty distinct physical node IDs")
    canonical = normalize_graph(dict(graph))
    nodes = canonical["nodes"]
    sample_id = canonical.get("graph_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("graph_id must be a nonempty stable string")
    node_ids = [str(node["id"]) for node in nodes]
    if set(node_ids) != set(input_ids):
        raise ValueError("Normalization changed the physical node set")
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    dictionary = canonical["dictionary"]
    _check_dictionary(nodes, dictionary)
    node_types = [node_kind_index(node["kind"]) for node in nodes]
    incidences, roles, relation_types, binary_pairs = [], [], [], set()
    for relation_index, relation in enumerate(canonical["relations"]):
        kind, relation_id = relation["kind"], relation["id"]
        if kind not in EDGE_TYPE_MAP:
            raise ValueError(f"Unsupported relation kind: {kind!r}")
        sources, targets = set(relation["sources"]), set(relation["targets"])
        if not sources or not targets or not (sources | targets) <= node_index.keys():
            raise ValueError(f"Invalid memberships in relation {relation_id!r}")
        if kind not in {"CTRL", "DATA"} and (len(sources) != 1 or len(targets) != 1):
            raise ValueError(f"Binary relation {relation_id!r} requires singleton endpoints")
        if kind in {"CTRL", "DATA"} and sources & targets:
            raise ValueError(f"Hyperedge {relation_id!r} has overlapping source/target memberships")
        for names, role in ((relation["sources"], 0), (relation["targets"], 1)):
            for name in names:
                incidences.append((node_index[name], relation_index))
                roles.append(role)
        expected_kind = {"CTRL": "CFG", "DATA": "DDG"}.get(kind, kind)
        bindings = relation.get("bindings", [])
        if not bindings:
            raise ValueError(f"Relation {relation_id!r} has no binding witnesses")
        witnessed, source_ports, target_ports = set(), {}, {}
        for binding in bindings:
            if binding.get("kind") != expected_kind:
                raise ValueError(f"Binding kind mismatch in relation {relation_id!r}")
            source = _endpoint(binding["source"], dictionary, sources)
            target = _endpoint(binding["target"], dictionary, targets)
            witnessed.add((source, target))
            binary_pairs.add((node_index[source], node_index[target]))
            source_ports.setdefault(source, set()).add(str(binding["source"]))
            target_ports.setdefault(target, set()).add(str(binding["target"]))
        if witnessed != {(source, target) for source in sources for target in targets}:
            raise ValueError(f"Relation {relation_id!r} has an unwitnessed physical pair")
        if kind in {"CTRL", "DATA"} and any(
            len(entries) != 1 for entries in (*source_ports.values(), *target_ports.values())
        ):
            raise ValueError(f"Hyperedge {relation_id!r} mixes different logical endpoint ports")
        relation_types.append(EDGE_TYPE_MAP[kind])

    metadata = canonical["metadata"]
    original_count = metadata.get("original_node_count", len(dictionary))
    if type(original_count) is not int or original_count != len(dictionary):
        raise ValueError("original_node_count must equal the original dictionary size")
    representation = metadata.get("representation", "HCPG")
    if representation not in {"CPG", "HCPG"}:
        raise ValueError(f"Invalid representation: {representation!r}")
    graph_json = json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False)
    return dict(graph=canonical, input_ids=input_ids, node_ids=node_ids,
                graph_json=graph_json, sample_id=sample_id, original_node_count=original_count,
                representation=representation, node_types=node_types,
                incidences=incidences, roles=roles, relation_types=relation_types,
                binary_pairs=sorted(binary_pairs))


def validate_graph(graph):
    """Check the full data layout before any text encoding is performed."""
    return _graph_layout(graph)["graph"]


def graph_to_data(graph, features, label=None):
    """Convert a canonical graph; feature rows follow the input nodes' order.

    Every active relation retains its own incidence column. Ordinary binary-GNN
    edges contain witnessed physical endpoint pairs. The complete graph is kept
    as JSON so batching cannot rewrite or discard logical identity metadata.
    """
    layout = _graph_layout(graph)
    x = torch.as_tensor(features, dtype=torch.float32).detach().cpu().clone()
    if x.ndim != 2 or x.shape[0] != len(layout["input_ids"]) or x.shape[1] < 1:
        raise ValueError("Features must have shape [num_nodes, feature_width]")
    if not torch.isfinite(x).all():
        raise ValueError("Node features contain NaN or infinity")
    input_index = {node_id: index for index, node_id in enumerate(layout["input_ids"])}
    x = x[[input_index[node_id] for node_id in layout["node_ids"]]]
    def index_tensor(pairs):
        return (torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs
                else torch.empty((2, 0), dtype=torch.long))
    count = len(layout["relation_types"])
    data = GraphData(
        x=x, edge_index=index_tensor(layout["binary_pairs"]),
        hyperedge_index=index_tensor(layout["incidences"]),
        incidence_role=torch.tensor(layout["roles"], dtype=torch.long),
        node_types=torch.tensor(layout["node_types"], dtype=torch.long),
        relation_types=torch.tensor(layout["relation_types"], dtype=torch.long),
        num_nodes=len(layout["node_ids"]), num_hyperedges=count, n_hyperedges=count,
        sample_id=layout["sample_id"], original_node_count=layout["original_node_count"],
        schema_version=SCHEMA_VERSION, representation=layout["representation"],
        graph_json=layout["graph_json"], canonical_json=layout["graph_json"],
    )
    if label is not None:
        if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
            raise ValueError("label must be integer 0 or 1")
        data.y = torch.tensor([int(label)], dtype=torch.long)
    return data
