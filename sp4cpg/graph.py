"""Lossless, deterministic HCPG construction.

The JSON v2 dictionary is the canonical representation. DOT is an interchange
format: new DOT files embed that entire dictionary in ``sp4cpg_json``. Legacy
binary DOT edges are importable; legacy comma-separated hyperedges without
bindings are explicitly marked incomplete rather than inventing a Cartesian
product of dependencies.

Pruning uses graph evidence and explicit semantic facts, never target labels.
Ordinary Joern DOT does not contain enough call semantics to justify Rule 2,
nor enough declaration identity to justify every Rule 4 merge. Such nodes
remain. Optional facts live in node.attributes.pruning.rule1/rule2; their exact
requirements are implemented by rule1_eligible/rule2_eligible below. These facts
must come from a trusted analysis of the input program, not vulnerability labels.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any

SCHEMA_VERSION = 2
DEPENDENCY_KINDS = {"CFG", "DDG", "CDG"}
RELATION_KINDS = ("AST", "CFG", "CDG", "DDG", "CTRL", "DATA")
STRUCTURAL_NODE_KINDS = {"FILE", "CLASS", "METHOD", "BLOCK", "MODIFIER", "CONTROL_STRUCTURE", "LITERAL", "PARAM", "LOCAL",
                         "IDENTIFIER", "METHOD_RETURN", "CALL", "RETURN", "JUMP_TARGET", "FIELD_IDENTIFIER", "TYPE", "TYPE_REF",
                         "COMMENT", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "PARAMETER", "TYPE_DECL", "UNKNOWN"}


def canonical_text(value: Any) -> str:
    """Keep logical identity/context values comparable without lossy str(dict)."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _id_key(value: Any):
    # Original numeric IDs sort numerically; compound provenance IDs sort stably.
    return tuple((0, int(x)) if x.isdigit() else (1, x) for x in re.split(r"(\d+)", str(value)) if x)


def _fresh(prefix: str, value: Any) -> str:
    return prefix + ":" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()[:24]


def _node_record(node: dict) -> dict:
    return {"id": node["id"], "kind": node["kind"], "code": node["code"],
            "attributes": deepcopy(node.get("attributes", {}))}


def normalize_graph(graph: dict) -> dict:
    """Return a checked independent v2 copy, preserving all supplied attributes.

    Missing binding records are synthesized only for an ordinary binary edge.
    Hyperedge memberships alone cannot establish original dependencies.
    """
    if not isinstance(graph, dict):
        raise ValueError("Graph must be a JSON object")
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("relations"), list):
        raise ValueError("Graph requires nodes and relations arrays")
    version = graph.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported graph schema {version!r}; expected 2")
    out = deepcopy(graph)
    out["schema_version"] = SCHEMA_VERSION
    out["graph_id"] = str(out.get("graph_id", ""))
    out.setdefault("nodes", [])
    out.setdefault("relations", [])
    out.setdefault("metadata", {})
    out.setdefault("dictionary", {})
    if not isinstance(out["metadata"], dict) or not isinstance(out["dictionary"], dict):
        raise ValueError("metadata and dictionary must be objects")
    nodes = {}
    for node in out["nodes"]:
        node["id"] = str(node["id"])
        if node["id"] in nodes:
            raise ValueError(f"Duplicate physical node ID: {node['id']}")
        node["kind"] = str(node.get("kind", "UNKNOWN"))
        node["code"] = str(node.get("code", ""))
        node.setdefault("attributes", {})
        if not isinstance(node["attributes"], dict):
            raise ValueError("Node attributes must be an object")
        node["entries"] = sorted({str(x) for x in node.get("entries", [node["id"]])}, key=_id_key)
        nodes[node["id"]] = node
        if not out["dictionary"] or node["id"] not in out["dictionary"] and node["entries"] == [node["id"]]:
            out["dictionary"].setdefault(node["id"], {**_node_record(node), "representative": node["id"],
                                                       "carrier": None, "status": "retained", "rule": None})
    dictionary = {}
    for key, entry in out["dictionary"].items():
        key = str(key)
        entry["id"] = str(entry.get("id", key))
        if entry["id"] != key:
            raise ValueError("Dictionary key and original ID disagree")
        entry.setdefault("kind", "UNKNOWN")
        entry.setdefault("code", "")
        entry.setdefault("attributes", {})
        entry.setdefault("representative", key if key in nodes else None)
        entry.setdefault("carrier", None)
        entry.setdefault("status", "retained")
        entry.setdefault("rule", None)
        for field in ("representative", "carrier"):
            if entry[field] is not None:
                entry[field] = str(entry[field])
        dictionary[key] = entry
    out["dictionary"] = dictionary
    relation_ids = set()
    # Assign content-derived IDs to anonymous edges, including parallel copies.
    anonymous_counts = defaultdict(int)
    for relation in sorted(out["relations"], key=_json):
        relation["kind"] = str(relation.get("kind", "AST")).upper()
        relation["sources"] = sorted({str(x) for x in relation.get("sources", [])}, key=_id_key)
        relation["targets"] = sorted({str(x) for x in relation.get("targets", [])}, key=_id_key)
        if not relation["sources"] or not relation["targets"]:
            raise ValueError("Every relation must have source and target members")
        if not set(relation["sources"] + relation["targets"]) <= nodes.keys():
            raise ValueError("Relation refers to a missing physical node")
        relation.setdefault("attributes", {})
        if not isinstance(relation["attributes"], dict):
            raise ValueError("Relation attributes must be an object")
        if "bindings" in relation:
            if not isinstance(relation["bindings"], list):
                raise ValueError("Relation bindings must be an array")
            relation["bindings"].sort(key=_json)
        if "id" not in relation:
            signature = _json(relation)
            count = anonymous_counts[signature]
            anonymous_counts[signature] += 1
            relation["id"] = _fresh("edge", [signature, count])
        relation["id"] = str(relation["id"])
        if relation["id"] in relation_ids:
            raise ValueError(f"Duplicate relation ID: {relation['id']}")
        relation_ids.add(relation["id"])
        if "bindings" not in relation:
            if len(relation["sources"]) == len(relation["targets"]) == 1:
                relation["bindings"] = [{"source": relation["sources"][0], "target": relation["targets"][0],
                                         "kind": relation["kind"], "label": relation.get("label", ""),
                                         "entity": relation.get("entity", ""), "context": relation.get("context", ""),
                                         "attributes": deepcopy(relation["attributes"]), "path": []}]
            else:
                relation["bindings"] = []
                relation["attributes"]["legacy_unbound"] = True
        for binding in relation["bindings"]:
            binding["source"] = str(binding["source"])
            binding["target"] = str(binding["target"])
            binding.setdefault("kind", {"CTRL": "CFG", "DATA": "DDG"}.get(relation["kind"], relation["kind"]))
            binding["kind"] = str(binding["kind"]).upper()
            for field in ("label", "entity", "context"):
                binding[field] = canonical_text(binding.get(field, ""))
            binding.setdefault("attributes", {})
            binding["path"] = [str(x) for x in binding.get("path", [])]
            binding.setdefault("relation_id", relation["id"])
        relation["bindings"].sort(key=_json)
    out["nodes"].sort(key=lambda n: _id_key(n["id"]))
    out["relations"].sort(key=lambda r: _id_key(r["id"]))
    out["dictionary"] = dict(sorted(dictionary.items(), key=lambda kv: _id_key(kv[0])))
    out["metadata"].setdefault("representation", "CPG")
    out["metadata"].setdefault("original_node_count", len(dictionary))
    out["metadata"].setdefault("provenance_complete", all(r["bindings"] for r in out["relations"]))
    return out


def _dot_value(value: Any) -> str:
    if value is None:
        return ""
    value = str(value)
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value[1:-1].replace(r'\"', '"')
    return value


def from_pydot(dot, graph_id: str = "") -> dict:
    """Import pydot objects, retaining raw labels, parallel edges and attributes."""
    embedded = dot.get_attributes().get("sp4cpg_json")
    if embedded is not None:
        return normalize_graph(json.loads(_dot_value(embedded)))
    node_map = {}
    raw_edges = []

    def visit(subgraph):
        for node in subgraph.get_nodes():
            node_id = _dot_value(node.get_name()).strip()
            if node_id in {"", "node", "edge", "graph", "\\n", "\n"}:
                continue
            attrs = {str(k): _dot_value(v) for k, v in node.get_attributes().items()}
            label = attrs.get("label", "")
            if label.startswith("<") and label.endswith(">"):
                label_text = label[1:-1]
            else:
                label_text = label
            parts = re.split(r"<BR\s*/?>", label_text, flags=re.I)
            inferred_kind = html.unescape(parts[0].split(",", 1)[0].strip().strip("()")) if parts[0] else "UNKNOWN"
            kind = attrs.get("kind", inferred_kind)
            code = attrs.get("code", html.unescape(parts[1]) if len(parts) > 1 else "")
            if len(parts) > 2:
                attrs["legacy_occurrence_context"] = [html.unescape(p) for p in parts[2:]]
            # Joern's HTML DOT exporter writes a CALL's callee name in the
            # first field (e.g. <alloc, 5<BR/>alloc(...)>). This is an explicit
            # exporter convention, not a safety classification of the callee.
            joern_named_call = ("kind" not in attrs and label.startswith("<") and len(parts) > 1
                                and re.search(r",\s*\d+\)?\s*$", parts[0]) is not None
                                and kind not in STRUCTURAL_NODE_KINDS and not kind.startswith("<operator>."))
            if joern_named_call:
                attrs["legacy_call_name"] = inferred_kind
                attrs.setdefault("name", inferred_kind)
                kind = "CALL"
            if node_id in node_map:
                node_map[node_id]["attributes"].update(attrs)
            else:
                node_map[node_id] = {"id": node_id, "kind": kind, "code": code, "attributes": attrs}
        raw_edges.extend(subgraph.get_edges())
        for child in subgraph.get_subgraphs():
            visit(child)

    visit(dot)
    relations = []
    for edge in raw_edges:
        attrs = {str(k): _dot_value(v) for k, v in edge.get_attributes().items()}
        sources = [_dot_value(x).strip() for x in _dot_value(edge.get_source()).split(",") if x.strip()]
        targets = [_dot_value(x).strip() for x in _dot_value(edge.get_destination()).split(",") if x.strip()]
        for node_id in sources + targets:
            node_map.setdefault(node_id, {"id": node_id, "kind": "UNKNOWN", "code": "", "attributes": {"implicit": True}})
        label = attrs.get("label", "AST")
        kind = attrs.get("kind", label.split(":", 1)[0].strip()).upper()
        binary = len(sources) == len(targets) == 1
        bindings = [{"source": sources[0], "target": targets[0], "kind": kind, "label": label,
                     "entity": attrs.get("entity", ""), "context": attrs.get("context", ""),
                     "attributes": deepcopy(attrs), "path": []}] if binary else []
        if not binary:
            attrs["legacy_unbound"] = True
        relations.append({"kind": {"CFG": "CTRL", "DDG": "DATA"}.get(kind, kind) if not binary else kind,
                          "sources": sources, "targets": targets, "bindings": bindings, "attributes": attrs})
    return normalize_graph({"schema_version": 2, "graph_id": graph_id or _dot_value(dot.get_name()),
                            "nodes": list(node_map.values()), "relations": relations,
                            "metadata": {"representation": "HCPG" if any(not r["bindings"] for r in relations) else "CPG",
                                         "source_format": "legacy_dot",
                                         "provenance_complete": all(r["bindings"] for r in relations)}})


def load_graph(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        graph = normalize_graph(json.loads(path.read_text(encoding="utf-8-sig")))
        if not graph["graph_id"]:
            graph["graph_id"] = path.stem
        return graph
    if path.suffix.lower() != ".dot":
        raise ValueError(f"Expected .json or .dot, got {path.name}")
    import pydot
    graphs = pydot.graph_from_dot_data(path.read_text(encoding="utf-8-sig"))
    if not graphs or len(graphs) != 1:
        raise ValueError("Expected exactly one DOT graph")
    return from_pydot(graphs[0], graph_id=path.stem)


def dump_graph(graph: dict, path: str | Path) -> None:
    graph = normalize_graph(graph)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(graph, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    if path.suffix.lower() != ".dot":
        raise ValueError("Graph output must have .json or .dot extension")
    import pydot
    dot = pydot.Dot(graph_name=json.dumps(graph["graph_id"] or "graph"), graph_type="digraph")
    dot.set("sp4cpg_json", json.dumps(_json(graph), ensure_ascii=False))
    for node in graph["nodes"]:
        dot.add_node(pydot.Node(json.dumps(node["id"]), label=json.dumps(node["kind"] + "\n" + node["code"], ensure_ascii=False)))
    for relation in graph["relations"]:
        # The visible DOT is a convenience view. Embedded v2 data is authoritative.
        dot.add_edge(pydot.Edge(json.dumps(",".join(relation["sources"])), json.dumps(",".join(relation["targets"])),
                                label=json.dumps(relation["kind"]), id=json.dumps(relation["id"])))
    path.write_text(dot.to_string(), encoding="utf-8")


def _semantic_attrs(node: dict) -> dict:
    attrs = node.get("attributes", {})
    pruning = attrs.get("pruning", attrs.get("sp4cpg_pruning", {}))
    if isinstance(pruning, str):
        try:
            pruning = json.loads(pruning)
        except ValueError:
            return {}
    return pruning if isinstance(pruning, dict) else {}


def _incident(graph: dict, node_id: str) -> list[dict]:
    return [r for r in graph["relations"] if node_id in r["sources"] or node_id in r["targets"]]


def _content_present(node: dict, carrier: dict) -> bool:
    code = node["code"]
    if not code or not carrier["code"]:
        return False
    if node["kind"] == "IDENTIFIER":
        return re.search(r"(?<![\w$])" + re.escape(code) + r"(?![\w$])", carrier["code"]) is not None
    return code in carrier["code"]


def rule1_eligible(node: dict, original: dict) -> tuple[bool, str | None]:
    """Explicit redundant structural assignment, with an adjacent content carrier."""
    proof = _semantic_attrs(node).get("rule1", {})
    if node["kind"] not in {"<operator>.assignment", "operator.assignment", "ASSIGNMENT"} or not isinstance(proof, dict) or proof.get("redundant") is not True:
        return False, None
    incident = _incident(original, node["id"])
    if not incident or any(r["kind"] != "AST" or len(r["sources"]) != 1 or len(r["targets"]) != 1 for r in incident):
        return False, None
    nodes = {n["id"]: n for n in original["nodes"]}
    carrier_id = str(proof.get("carrier", ""))
    neighbors = {x for r in incident for x in r["sources"] + r["targets"]} - {node["id"]}
    incoming = [r for r in incident if r["targets"] == [node["id"]]]
    outgoing = [r for r in incident if r["sources"] == [node["id"]]]
    if len(incoming) != 1 or carrier_id not in neighbors or carrier_id not in nodes or not outgoing:
        return False, None
    parent = nodes[incoming[0]["sources"][0]]
    allocation_children = [nodes[r["targets"][0]] for r in outgoing
                           if nodes[r["targets"][0]]["kind"] in {"<operator>.alloc", "operator.alloc", "ALLOC"}]
    if parent["kind"] != "BLOCK" or not _content_present(node, parent) or not any(_content_present(node, child) for child in allocation_children):
        return False, None
    if carrier_id not in {parent["id"], *(child["id"] for child in allocation_children)}:
        return False, None
    return (_content_present(node, nodes[carrier_id]), carrier_id)


def rule2_eligible(node: dict, original: dict) -> bool:
    """Require explicit resolved call semantics plus a checkable linear CFG bypass.

    Fields under attributes.pruning.rule2: call_name, format, constant_output,
    result_unused, output_only, arguments_valid, control_context_preserved,
    vulnerability_relevant. A format may contain literal %% but no conversion.
    A nonempty identical recorded control context is required on CFG path edges.
    """
    if node["kind"] != "CALL":
        return False
    proof = _semantic_attrs(node).get("rule2", {})
    if not isinstance(proof, dict):
        return False
    name = proof.get("call_name")
    if name not in {"printf", "fprintf", "puts", "putchar"}:
        return False
    known_names = [node.get("attributes", {}).get(k) for k in ("name", "full_name", "method_full_name")]
    for known in known_names:
        if known and str(known).split("(", 1)[0].split("::")[-1] != name:
            return False
    required = ("constant_output", "result_unused", "output_only", "arguments_valid", "control_context_preserved")
    if any(proof.get(k) is not True for k in required) or proof.get("vulnerability_relevant") is not False:
        return False
    fmt = proof.get("format")
    if not isinstance(fmt, str) or "%" in fmt.replace("%%", ""):
        return False
    incident = _incident(original, node["id"])
    if any(r["kind"] not in {"AST", "CFG"} or len(r["sources"]) != 1 or len(r["targets"]) != 1 or len(r["bindings"]) != 1 for r in incident):
        return False
    cfg = [r for r in incident if r["kind"] == "CFG"]
    if not cfg:
        return True
    incoming = [r for r in cfg if r["targets"] == [node["id"]]]
    outgoing = [r for r in cfg if r["sources"] == [node["id"]]]
    if len(incoming) != 1 or len(outgoing) != 1 or incoming[0]["sources"] == outgoing[0]["targets"]:
        return False
    context = incoming[0]["bindings"][0]["context"]
    return bool(context) and context == outgoing[0]["bindings"][0]["context"]


def _carry(graph: dict, node_id: str, carrier_id: str, rule: int, carrier_original: str) -> None:
    nodes = {n["id"]: n for n in graph["nodes"]}
    node, carrier = nodes[node_id], nodes[carrier_id]
    carrier["entries"] = sorted(set(carrier["entries"] + node["entries"]), key=_id_key)
    records = carrier["attributes"].setdefault("retained_contents", [])
    for entry_id in node["entries"]:
        entry = graph["dictionary"][entry_id]
        records.append(_node_record(entry))
        entry.update({"representative": None, "carrier": carrier_id, "carrier_original": carrier_original,
                      "status": "carried", "rule": rule})
    records.sort(key=lambda r: _id_key(r["id"]))
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] != node_id]


def _path_parts(binding: dict) -> dict:
    if binding.get("path"):
        return {k: deepcopy(binding["attributes"][k]) for k in ("path_labels", "path_contexts", "path_entities", "path_attributes", "path_relation_ids")}
    return {"path_labels": [binding["label"]], "path_contexts": [binding["context"]],
            "path_entities": [binding["entity"]], "path_attributes": [deepcopy(binding["attributes"])],
            "path_relation_ids": [binding["relation_id"]]}


def _remove_with_reconnect(graph: dict, node_id: str, rule: int, carrier: str | None = None) -> bool:
    incident = _incident(graph, node_id)
    incoming_cfg = [r for r in incident if r["kind"] == "CFG" and r["targets"] == [node_id]]
    outgoing_cfg = [r for r in incident if r["kind"] == "CFG" and r["sources"] == [node_id]]
    additions = []
    if incoming_cfg or outgoing_cfg:
        if len(incoming_cfg) != 1 or len(outgoing_cfg) != 1:
            return False
        left, right = incoming_cfg[0], outgoing_cfg[0]
        if left["sources"] == right["targets"]:
            return False
        a, b = left["bindings"][0], right["bindings"][0]
        if not a["context"] or a["context"] != b["context"]:
            return False
        ap, bp = _path_parts(a), _path_parts(b)
        path = a["path"] + [node_id] + b["path"]
        attributes = {k: ap[k] + bp[k] for k in ap}
        rid = _fresh("bypass", [attributes["path_relation_ids"], path])
        binding = {"source": a["source"], "target": b["target"], "kind": "CFG", "label": a["label"],
                   "entity": a["entity"], "context": a["context"], "attributes": attributes,
                   "path": path, "relation_id": rid}
        additions.append({"id": rid, "kind": "CFG", "sources": left["sources"], "targets": right["targets"],
                          "bindings": [binding], "attributes": {"rule": rule}})
    incoming_ast = [r for r in incident if r["kind"] == "AST" and r["targets"] == [node_id]]
    outgoing_ast = [r for r in incident if r["kind"] == "AST" and r["sources"] == [node_id]]
    # Structural reconnection is independently checkable against the original
    # AST: active bindings carry complete original path witnesses, rather than
    # relying on an opaque construction log or an overwritten intermediate ID.
    if len(incoming_ast) > 1:
        return False
    if incoming_ast:
        parent = incoming_ast[0]["sources"][0]
        for relation in outgoing_ast:
            if parent in relation["targets"]:
                return False
            rid = _fresh("ast-reconnect", [incoming_ast[0]["id"], relation["id"], node_id])
            new = {"id": rid, "kind": "AST", "sources": [parent], "targets": deepcopy(relation["targets"]),
                   "bindings": [], "attributes": {"rule": rule}}
            for left_binding in incoming_ast[0]["bindings"]:
                for right_binding in relation["bindings"]:
                    ap, bp = _path_parts(left_binding), _path_parts(right_binding)
                    attributes = {k: ap[k] + bp[k] for k in ap}
                    new["bindings"].append({"source": left_binding["source"], "target": right_binding["target"],
                                            "kind": "AST", "label": left_binding["label"],
                                            "entity": left_binding["entity"], "context": left_binding["context"],
                                            "attributes": attributes,
                                            "path": left_binding["path"] + [node_id] + right_binding["path"],
                                            "relation_id": rid})
            additions.append(new)
    graph["relations"] = [r for r in graph["relations"] if r not in incident] + additions
    if carrier is not None:
        _carry(graph, node_id, carrier, rule, carrier)
    else:
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] != node_id]
        graph["dictionary"][node_id].update({"representative": None, "carrier": None, "status": "removed", "rule": rule})
    return True


def _merge_declarations(graph: dict) -> None:
    groups = defaultdict(list)
    for node in graph["nodes"]:
        attrs = node["attributes"]
        kind = node["kind"]
        if kind not in {"LOCAL", "PARAM", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "TYPE_REF", "METHOD_RETURN"}:
            continue
        declared_type, context = attrs.get("declared_type"), attrs.get("function_context")
        if not declared_type or not context or str(declared_type).strip().upper() in {"ANY", "UNKNOWN", "<UNKNOWN>", "?"}:
            continue
        if kind in {"LOCAL", "PARAM", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT"}:
            if not attrs.get("name") or "scope" not in attrs or attrs["scope"] in (None, ""):
                continue
        elif not attrs.get("occurrence"):
            continue
        # Staying inside one known function context is a conservative refinement.
        groups[(kind, canonical_text(declared_type), canonical_text(context))].append(node)
    for key, members in sorted(groups.items(), key=lambda item: _json(item[0])):
        if len(members) < 2:
            continue
        member_ids = {n["id"] for n in members}
        incident = [r for r in graph["relations"] if member_ids.intersection(r["sources"] + r["targets"])]
        if any(not r["bindings"] for r in incident):
            continue
        entries = sorted({e for n in members for e in n["entries"]}, key=_id_key)
        representative = _fresh("merge", [key, entries])
        # Preserve every content carrier as well as the original declaration text.
        contents = {_json(_node_record(graph["dictionary"][e])): _node_record(graph["dictionary"][e]) for e in entries}
        new_node = {"id": representative, "kind": key[0], "code": "\n".join(n["code"] for n in sorted(members, key=lambda n: _id_key(n["id"]))),
                    "entries": entries, "attributes": {"declared_type": deepcopy(members[0]["attributes"]["declared_type"]),
                                                          "function_context": deepcopy(members[0]["attributes"]["function_context"]),
                                                          "retained_contents": sorted(contents.values(), key=lambda n: _id_key(n["id"]))}}
        for e in entries:
            entry = graph["dictionary"][e]
            if entry["status"] == "carried":
                entry["carrier"] = representative
            else:
                entry.update({"representative": representative, "carrier": None, "status": "merged", "rule": 4})
        for relation in incident:
            for field in ("sources", "targets"):
                relation[field] = sorted({representative if e in member_ids else e for e in relation[field]}, key=_id_key)
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] not in member_ids] + [new_node]


def _physical(graph: dict, original_id: str) -> str | None:
    entry = graph["dictionary"].get(original_id, {})
    return entry.get("representative") or entry.get("carrier")


def _binding_compatible(graph: dict, edges: list[dict]) -> bool:
    sources = {s for r in edges for s in r["sources"]}
    targets = {t for r in edges for t in r["targets"]}
    if sources & targets:
        return False
    witnesses = set()
    originals = defaultdict(set)
    for relation in edges:
        if len(relation["sources"]) != 1 or len(relation["targets"]) != 1 or not relation["bindings"]:
            return False
        s, t = relation["sources"][0], relation["targets"][0]
        for binding in relation["bindings"]:
            if _physical(graph, binding["source"]) != s or _physical(graph, binding["target"]) != t:
                return False
            originals[(s, "source")].add(binding["source"])
            originals[(t, "target")].add(binding["target"])
        witnesses.add((s, t))
    # A shared representative cannot turn distinct original entries into a star.
    return all(len(ids) == 1 for ids in originals.values()) and witnesses == {(s, t) for s in sources for t in targets}


def build_hyperedges(graph: dict, kind: str) -> None:
    """Consume CFG/DDG binary witnesses in MM, SM, MS order, retaining bindings.

    CFG pools are partitioned by exact recorded control context. Unknown context
    is retained binary. MM column signatures omit target occurrence IDs but keep
    source identity, labels, entities, attributes and path witnesses.
    """
    pool = [r for r in graph["relations"] if r["kind"] == kind and len(r["sources"]) == len(r["targets"]) == 1 and r["bindings"]]
    pool.sort(key=lambda r: (_id_key(r["sources"][0]), _id_key(r["targets"][0]), _id_key(r["id"])))
    partitions = defaultdict(list)
    for edge in pool:
        contexts = {b["context"] for b in edge["bindings"]}
        if kind == "CFG" and (len(contexts) != 1 or not next(iter(contexts))):
            continue
        partitions[next(iter(contexts)) if kind == "CFG" else ""].append(edge)
    accepted = []
    consumed = set()

    def add(group, mode):
        sources = sorted({s for r in group for s in r["sources"]}, key=_id_key)
        targets = sorted({t for r in group for t in r["targets"]}, key=_id_key)
        thresholds = {"MM": len(sources) >= 2 and len(targets) >= 2,
                      "SM": len(sources) == 1 and len(targets) >= 2,
                      "MS": len(sources) >= 2 and len(targets) == 1}
        if len(group) < 2 or not thresholds[mode] or not _binding_compatible(graph, group):
            return False
        edge_ids = sorted((r["id"] for r in group), key=_id_key)
        attributes = {"mode": mode, "relation_kind": kind, "witness_relation_ids": edge_ids}
        if kind == "DDG":
            attributes["labels"] = sorted({b["label"] for r in group for b in r["bindings"]})
            annotations = [{"source": b["source"], "entity": b["entity"], "constant": deepcopy(b["source_constant"])}
                           for r in group for b in r["bindings"] if "source_constant" in b]
            attributes["source_annotations"] = [json.loads(item) for item in sorted({_json(a) for a in annotations})]
        accepted.append({"id": _fresh("hyper", [kind, mode, edge_ids]), "kind": "CTRL" if kind == "CFG" else "DATA",
                         "sources": sources, "targets": targets,
                         "bindings": sorted([deepcopy(b) for r in group for b in r["bindings"]], key=_json),
                         "attributes": attributes})
        consumed.update(edge_ids)
        return True

    def signature(column):
        records = []
        for relation in column:
            for b in relation["bindings"]:
                records.append([relation["sources"][0], b["source"], b["kind"], b["label"], b["entity"], b["context"], b["attributes"], b["path"]])
        return _json(sorted(records, key=_json))

    for _, partition in sorted(partitions.items()):
        # Each MM seed starts with a complete current incoming target column.
        for target in sorted({r["targets"][0] for r in partition}, key=_id_key):
            remaining = [r for r in partition if r["id"] not in consumed]
            column = [r for r in remaining if r["targets"] == [target]]
            if not column or len({r["sources"][0] for r in column}) < 2:
                continue
            group = list(column)
            for other in sorted({r["targets"][0] for r in remaining} - {target}, key=_id_key):
                candidate = [r for r in remaining if r["targets"] == [other]]
                if signature(candidate) == signature(column) and _binding_compatible(graph, group + candidate):
                    group += candidate
            add(group, "MM")
        # Stars are seeded and grown in provenance order, recomputing unconsumed
        # edges after every accepted candidate. Whole-group compatibility matters.
        for mode, fixed_field, binding_field in (("SM", "sources", "source"), ("MS", "targets", "target")):
            for seed in partition:
                if seed["id"] in consumed:
                    continue
                original_entries = {b[binding_field] for b in seed["bindings"]}
                if len(original_entries) != 1:
                    continue
                group = [seed]
                for edge in partition:
                    if edge["id"] in consumed or edge is seed or edge[fixed_field] != seed[fixed_field]:
                        continue
                    if {b[binding_field] for b in edge["bindings"]} != original_entries:
                        continue
                    if _binding_compatible(graph, group + [edge]):
                        group.append(edge)
                add(group, mode)
    graph["relations"] = [r for r in graph["relations"] if r["id"] not in consumed] + accepted


def transform_graph(original: dict) -> dict:
    """Apply rule-conditioned pruning and deterministic directed edge packing.

    Input is never mutated. Original dictionary attributes and every dependency
    witness survive. Incomplete legacy hyperedges are rejected before rewriting.
    """
    original = normalize_graph(original)
    if not original["metadata"].get("provenance_complete", True) or any(not r["bindings"] for r in original["relations"]):
        raise ValueError("Cannot transform legacy hyperedges without endpoint bindings; regenerate from binary CPG or import v2 DOT")
    if original["metadata"].get("representation") == "HCPG":
        raise ValueError("Expected an original CPG; refusing to prune an already transformed HCPG")
    original_nodes = {n["id"]: n for n in original["nodes"]}
    if set(original["dictionary"]) != set(original_nodes):
        raise ValueError("Original CPG dictionary must cover exactly its original nodes")
    for node_id, node in original_nodes.items():
        entry = original["dictionary"][node_id]
        if node["entries"] != [node_id] or entry["status"] != "retained" or entry["representative"] != node_id or entry["carrier"] is not None:
            raise ValueError("Expected an unmodified original CPG identity dictionary")
        if any(entry.get(k) != node.get(k) for k in ("id", "kind", "code", "attributes")):
            raise ValueError("Original CPG dictionary differs from its node records")
    for relation in original["relations"]:
        if relation["kind"] in {"CTRL", "DATA"} or len(relation["sources"]) != 1 or len(relation["targets"]) != 1:
            raise ValueError("Expected original binary CPG relations")
        for binding in relation["bindings"]:
            if ([binding["source"]] != relation["sources"] or [binding["target"]] != relation["targets"]
                    or binding["kind"] != relation["kind"] or binding["path"]):
                raise ValueError("Original CPG binding disagrees with its physical relation")
    graph = deepcopy(original)
    audit = []
    for node in original["nodes"]:
        if node["id"] not in {n["id"] for n in graph["nodes"]}:
            continue
        eligible, carrier = rule1_eligible(node, original)
        if eligible and carrier in {n["id"] for n in graph["nodes"]}:
            if _remove_with_reconnect(graph, node["id"], 1, carrier):
                audit.append({"id": node["id"], "rule": 1, "status": "carried"})
                continue
        if rule2_eligible(node, original) and _remove_with_reconnect(graph, node["id"], 2):
            audit.append({"id": node["id"], "rule": 2, "status": "removed"})
        elif node["kind"] == "COMMENT":
            incident = _incident(original, node["id"])
            if all(r["kind"] == "AST" and r["targets"] == [node["id"]] for r in incident) and len(incident) <= 1:
                if _remove_with_reconnect(graph, node["id"], 2):
                    audit.append({"id": node["id"], "rule": 2, "status": "removed"})
    # Dependency/association eligibility always uses the independent input graph.
    originals = {n["id"]: n for n in original["nodes"]}
    for node in original["nodes"]:
        node_id = node["id"]
        if node["kind"] not in {"LITERAL", "IDENTIFIER"} or node_id not in {n["id"] for n in graph["nodes"]}:
            continue
        incident = _incident(original, node_id)
        if len(incident) != 1 or incident[0]["kind"] != "AST" or incident[0]["targets"] != [node_id] or len(incident[0]["sources"]) != 1:
            continue
        # Extra identity/binding metadata is independent information, even when
        # there is no CFG/DDG edge. Such leaves are deliberately retained.
        if any(k in node["attributes"] for k in ("entity", "binding", "bindings", "symbol_id", "ref", "declaration_id")):
            continue
        parent = incident[0]["sources"][0]
        carrier = _physical(graph, parent)
        if carrier == node_id or carrier not in {n["id"] for n in graph["nodes"]} or not _content_present(node, originals[parent]):
            continue
        rule = 3
        graph["relations"] = [r for r in graph["relations"] if node_id not in r["sources"] + r["targets"]]
        _carry(graph, node_id, carrier, rule, parent)
        audit.append({"id": node_id, "rule": rule, "status": "carried"})
    _merge_declarations(graph)
    for relation in graph["relations"]:
        if relation["kind"] == "DDG":
            for binding in relation["bindings"]:
                attrs = original["dictionary"].get(binding["source"], {}).get("attributes", {})
                if "explicit_constant" in attrs:
                    binding["source_constant"] = deepcopy(attrs["explicit_constant"])
    build_hyperedges(graph, "CFG")
    build_hyperedges(graph, "DDG")
    graph["metadata"].update({"representation": "HCPG", "original_node_count": len(original["dictionary"]),
                               "construction": "rule-conditioned-mm-sm-ms-v2", "pruning_audit": audit})
    return normalize_graph(graph)
