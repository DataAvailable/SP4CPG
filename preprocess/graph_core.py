"""Deterministic dependency-preserving graph pruning and hyperedge packing.

JSON dictionaries store physical nodes separately from original logical roles.
Every active relation owns its endpoint bindings. DOT files carry the complete
dictionary in a graph attribute in addition to their visible graph structure.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import re


SCHEMA_VERSION = 2
RELATION_KINDS = ("AST", "CFG", "CDG", "DDG", "CTRL", "DATA")
NODE_KINDS = {"FILE", "CLASS", "METHOD", "BLOCK", "MODIFIER", "CONTROL_STRUCTURE",
              "LITERAL", "PARAM", "LOCAL", "IDENTIFIER", "METHOD_RETURN", "CALL",
              "RETURN", "JUMP_TARGET", "FIELD_IDENTIFIER", "TYPE", "TYPE_REF", "COMMENT",
              "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "PARAMETER", "TYPE_DECL"}
PATH_FIELDS = ("path_labels", "path_contexts", "path_entities", "path_attributes", "path_relation_ids")
CODE_TOKENS = re.compile(
    r'//[^\r\n]*|/\*[\s\S]*?\*/|(?:u8|u|U|L)?R"(?P<delimiter>[^\s()\\]{0,16})\([\s\S]*?\)(?P=delimiter)"'
    r'''|(?:u8|u|U|L)?(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')'''
    r"|(?:0[xX][0-9A-Fa-f']+(?:\.[0-9A-Fa-f']*)?(?:[pP][+-]?[0-9']+)?"
    r"|0[bB][01']+|(?:[0-9][0-9']*(?:\.[0-9']*)?|\.[0-9][0-9']*)(?:[eE][+-]?[0-9']+)?)"
    r"(?:[A-Za-z_][A-Za-z_0-9]*)?|(?:[^\W\d]|[$])[\w$]*")


def _encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _text(value):
    return value if isinstance(value, str) else "" if value is None else _encode(value)


def _order(value):
    return tuple((0, int(piece)) if piece.isdigit() else (1, piece)
                 for piece in re.split(r"(\d+)", str(value)) if piece)


def _identity(prefix, payload, occupied=()):
    token = prefix + ":" + hashlib.sha256(_encode(payload).encode("utf-8")).hexdigest()[:24]
    number = 0
    candidate = token
    while candidate in occupied:
        number += 1
        candidate = token + ":" + str(number)
    return candidate


def _record(node):
    return {field: deepcopy(node[field]) for field in ("id", "kind", "code", "attributes")}


def _role(node):
    if node["kind"] != "CALL":
        return ""
    attrs = node["attributes"]
    return next((str(attrs[key]) for key in ("operation", "name", "method_full_name", "full_name")
                 if attrs.get(key)), "")


def _physical(graph, original_id):
    entry = graph["dictionary"].get(original_id, {})
    return entry.get("carrier") if entry.get("status") == "carried" else entry.get("representative")


def _canonical_kind(record):
    kind = record["kind"]
    if kind.startswith("<operator>."):
        record["attributes"].setdefault("operation", kind)
        record["kind"] = "CALL"


def _known(value):
    return value is not None and value not in ("", {}, []) and _text(value).strip().upper() not in {
        "", "ANY", "UNKNOWN", "<UNKNOWN>", "?", "NULL", "NONE", "{}", "[]"}


def _source_checksum(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("source_code_sha256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def normalize_graph(graph):
    """Copy and check the graph schema without filling unknown hyperedge bindings."""
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("relations"), list):
        raise ValueError("A graph requires nodes and relations arrays")
    if graph.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("Only graph schema_version=2 is supported")
    out = deepcopy(graph)
    out["schema_version"] = SCHEMA_VERSION
    out["graph_id"] = str(out.get("graph_id", ""))
    out.setdefault("metadata", {})
    if not isinstance(out["metadata"], dict):
        raise ValueError("metadata must be an object")
    if "source_code_sha256" in out["metadata"]:
        out["metadata"]["source_code_sha256"] = _source_checksum(out["metadata"]["source_code_sha256"])
    dictionary_supplied = "dictionary" in out
    dictionary = out.setdefault("dictionary", {})
    if not isinstance(dictionary, dict):
        raise ValueError("dictionary must be an object")
    node_ids = set()
    for node in out["nodes"]:
        node["id"] = str(node["id"])
        if node["id"] in node_ids:
            raise ValueError("Duplicate node ID: " + node["id"])
        node_ids.add(node["id"])
        node["kind"] = str(node.get("kind", "UNKNOWN"))
        node["code"] = str(node.get("code", ""))
        node.setdefault("attributes", {})
        if not isinstance(node["attributes"], dict):
            raise ValueError("Node attributes must be an object")
        _canonical_kind(node)
        node["entries"] = sorted({str(x) for x in node.get("entries", [node["id"]])}, key=_order)
        if not dictionary_supplied:
            dictionary[node["id"]] = {**_record(node), "representative": node["id"], "carrier": None,
                                       "status": "retained", "rule": None}
    for oid, entry in dictionary.items():
        if not isinstance(oid, str) or not isinstance(entry, dict) or entry.get("id") != oid:
            raise ValueError("Dictionary keys must equal their original string identities")
        if any(field not in entry for field in ("kind", "code", "attributes", "representative", "carrier", "status", "rule")):
            raise ValueError("Incomplete dictionary entry: " + oid)
        if entry["status"] not in {"retained", "merged", "carried", "removed"}:
            raise ValueError("Unknown dictionary status")
        if not isinstance(entry["attributes"], dict):
            raise ValueError("Dictionary attributes must be objects")
        _canonical_kind(entry)
    counts = defaultdict(int)
    relation_ids = set()
    for relation in sorted(out["relations"], key=_encode):
        relation["kind"] = str(relation["kind"]).upper()
        if relation["kind"] not in RELATION_KINDS:
            raise ValueError("Unknown relation kind: " + relation["kind"])
        for side in ("sources", "targets"):
            relation[side] = sorted({str(x) for x in relation.get(side, [])}, key=_order)
            if not relation[side] or not set(relation[side]) <= node_ids:
                raise ValueError("Relation has missing physical members")
        binary = len(relation["sources"]) == len(relation["targets"]) == 1
        if relation["kind"] not in {"CTRL", "DATA"} and not binary:
            raise ValueError("Binary relation kinds require singleton endpoints")
        relation.setdefault("attributes", {})
        if not isinstance(relation["attributes"], dict):
            raise ValueError("Relation attributes must be an object")
        if "id" not in relation:
            signature = _encode(relation)
            relation["id"] = _identity("edge", [signature, counts[signature]])
            counts[signature] += 1
        relation["id"] = str(relation["id"])
        if relation["id"] in relation_ids:
            raise ValueError("Duplicate relation ID: " + relation["id"])
        relation_ids.add(relation["id"])
        if "bindings" not in relation:
            relation["bindings"] = ([{"source": relation["sources"][0], "target": relation["targets"][0],
                                      "kind": relation["kind"], "label": relation.get("label", ""),
                                      "entity": relation.get("entity", ""), "context": relation.get("context", ""),
                                      "attributes": deepcopy(relation["attributes"]), "path": []}]
                                     if binary and relation["kind"] not in {"CTRL", "DATA"} else [])
        if not isinstance(relation["bindings"], list):
            raise ValueError("Relation bindings must be an array")
        for binding in relation["bindings"]:
            binding["source"], binding["target"] = str(binding["source"]), str(binding["target"])
            binding["kind"] = str(binding.get("kind", {"CTRL": "CFG", "DATA": "DDG"}.get(relation["kind"], relation["kind"]))).upper()
            for field in ("label", "entity", "context"):
                binding[field] = _text(binding.get(field, ""))
            binding.setdefault("attributes", {})
            if not isinstance(binding["attributes"], dict):
                raise ValueError("Binding attributes must be an object")
            binding["path"] = [str(x) for x in binding.get("path", [])]
            binding["relation_id"] = str(binding.get("relation_id", relation["id"]))
        relation["bindings"].sort(key=_encode)
    out["nodes"].sort(key=lambda node: (_order(min(node["entries"], key=_order)) if node["entries"] else _order(node["id"]), _order(node["id"])))
    out["relations"].sort(key=lambda relation: _order(relation["id"]))
    out["dictionary"] = dict(sorted(dictionary.items(), key=lambda item: _order(item[0])))
    out["metadata"].setdefault("representation", "CPG")
    out["metadata"].setdefault("original_node_count", len(dictionary))
    out["metadata"]["binding_complete"] = all(relation["bindings"] for relation in out["relations"])
    return out


def _unquote(value):
    value = str(value) if value is not None else ""
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except ValueError:
            return value[1:-1].replace(r'\"', '"')
    return value


def from_pydot(dot, graph_id=""):
    """Import ordinary DOT edges or a complete dictionary stored in DOT."""
    graph_attributes = dot.get_attributes()
    source_checksum = (_source_checksum(_unquote(graph_attributes["source_code_sha256"]))
                       if "source_code_sha256" in graph_attributes else None)
    stored = graph_attributes.get("sp4cpg_json")
    if stored is not None:
        graph = normalize_graph(json.loads(_unquote(stored)))
        if source_checksum is not None:
            existing = graph["metadata"].get("source_code_sha256")
            if existing is not None and existing != source_checksum:
                raise ValueError("DOT and embedded graph source_code_sha256 values differ")
            graph["metadata"]["source_code_sha256"] = source_checksum
        return graph
    nodes, edges = {}, []

    def visit(container):
        for obj in container.get_nodes():
            oid = _unquote(obj.get_name()).strip()
            if oid in {"", "node", "edge", "graph", "\\n", "\n"}:
                continue
            attrs = {str(key): _unquote(value) for key, value in obj.get_attributes().items()}
            raw = attrs.get("label", "")
            content = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
            parts = re.split(r"<BR\s*/?>", content, flags=re.I)
            token = html.unescape(parts[0].split(",", 1)[0].strip().strip("()")) if parts[0] else "UNKNOWN"
            kind = attrs.get("kind", token)
            code = attrs.get("code", html.unescape(parts[1]) if len(parts) > 1 else "")
            if len(parts) > 2:
                attrs["occurrence_context"] = [html.unescape(piece) for piece in parts[2:]]
            if kind.startswith("<operator>."):
                attrs.setdefault("operation", kind)
                kind = "CALL"
            elif "kind" not in attrs and kind not in NODE_KINDS and raw.startswith("<") and len(parts) > 1 and re.search(r",\s*\d+\)?\s*$", parts[0]):
                attrs.setdefault("name", kind)
                kind = "CALL"
            if oid in nodes:
                nodes[oid]["attributes"].update(attrs)
            else:
                nodes[oid] = {"id": oid, "kind": kind, "code": code, "attributes": attrs}
        edges.extend(container.get_edges())
        for child in container.get_subgraphs():
            visit(child)

    visit(dot)
    relations = []
    for obj in edges:
        attrs = {str(key): _unquote(value) for key, value in obj.get_attributes().items()}
        sources = [piece.strip() for piece in _unquote(obj.get_source()).split(",") if piece.strip()]
        targets = [piece.strip() for piece in _unquote(obj.get_destination()).split(",") if piece.strip()]
        for oid in sources + targets:
            nodes.setdefault(oid, {"id": oid, "kind": "UNKNOWN", "code": "", "attributes": {"implicit": True}})
        raw_label = attrs.get("label", "AST")
        kind = attrs.get("kind", raw_label.split(":", 1)[0].strip()).upper()
        binary = len(sources) == len(targets) == 1
        bindings = ([{"source": sources[0], "target": targets[0], "kind": kind, "label": raw_label,
                      "entity": attrs.get("entity", ""), "context": attrs.get("context", ""),
                      "attributes": deepcopy(attrs), "path": []}] if binary else [])
        if not binary:
            kind = {"CFG": "CTRL", "DDG": "DATA"}.get(kind, kind)
            attrs["unbound_memberships"] = True
        relations.append({"kind": kind, "sources": sources, "targets": targets, "bindings": bindings, "attributes": attrs})
    metadata = {"source_format": "dot"}
    if source_checksum is not None:
        metadata["source_code_sha256"] = source_checksum
    graph = normalize_graph({"graph_id": graph_id or _unquote(dot.get_name()), "nodes": list(nodes.values()),
                             "relations": relations, "metadata": metadata})
    if any(not relation["bindings"] for relation in graph["relations"]):
        graph["metadata"]["representation"] = "HCPG"
    return graph


def load_graph(path):
    path = Path(path)
    match = re.fullmatch(r"(.+)-(?:cpg|hcpg)-([01])", path.stem, flags=re.I)
    file_id = match.group(1) if match else path.stem
    if path.suffix.lower() == ".json":
        graph = normalize_graph(json.loads(path.read_text(encoding="utf-8-sig")))
        if not graph["graph_id"]:
            graph["graph_id"] = file_id
        if match:
            graph["metadata"].setdefault("label", int(match.group(2)))
        return graph
    if path.suffix.lower() != ".dot":
        raise ValueError("Graph files must use .dot or .json")
    import pydot
    parsed = pydot.graph_from_dot_data(path.read_text(encoding="utf-8-sig"))
    if not parsed or len(parsed) != 1:
        raise ValueError("Expected exactly one DOT graph")
    graph = from_pydot(parsed[0], graph_id=file_id)
    if match:
        graph["metadata"].setdefault("label", int(match.group(2)))
    return graph


def dump_graph(graph, path):
    graph, path = normalize_graph(graph), Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(graph, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return
    if path.suffix.lower() != ".dot":
        raise ValueError("Graph files must use .dot or .json")
    import pydot
    dot = pydot.Dot(graph_name=json.dumps(graph["graph_id"] or "graph"), graph_type="digraph")
    dot.set("sp4cpg_json", json.dumps(_encode(graph), ensure_ascii=False))
    for node in graph["nodes"]:
        label = node["kind"] + "\n" + node["code"]
        dot.add_node(pydot.Node(json.dumps(node["id"]), label=json.dumps(label, ensure_ascii=False)))
    for relation in graph["relations"]:
        dot.add_edge(pydot.Edge(json.dumps(",".join(relation["sources"])), json.dumps(",".join(relation["targets"])),
                                label=json.dumps(relation["kind"]), id=json.dumps(relation["id"])))
    path.write_text(dot.to_string(), encoding="utf-8")


def _facts(node, rule):
    value = node["attributes"].get("pruning", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    value = value.get("rule" + str(rule), {}) if isinstance(value, dict) else {}
    return value if isinstance(value, dict) else {}


def _incidents(graph, oid):
    return [edge for edge in graph["relations"] if oid in edge["sources"] + edge["targets"]]


def _contains(node, carrier):
    content = node["code"].strip()
    if not content:
        return False
    if node["kind"] in {"IDENTIFIER", "LITERAL"}:
        if any(token.group() == content for token in CODE_TOKENS.finditer(carrier["code"])):
            return True
        if node["kind"] == "LITERAL" and re.fullmatch(r"[+-][0-9].*", content):
            # A signed token must occur in a unary-expression position.
            for match in re.finditer(re.escape(content), carrier["code"]):
                before = carrier["code"][:match.start()].rstrip()
                after = carrier["code"][match.end():]
                if (not before or before[-1] in "([=,:?{!~+-*/%&|^<>") and (not after or not re.match(r"[\w$'.]", after)):
                    return True
        return False
    return content in carrier["code"]


def _path(binding):
    if binding["path"]:
        if any(not isinstance(binding["attributes"].get(field), list) for field in PATH_FIELDS):
            raise ValueError("Projected bindings require complete path annotations")
        return {field: deepcopy(binding["attributes"][field]) for field in PATH_FIELDS}
    return {"path_labels": [binding["label"]], "path_contexts": [binding["context"]],
            "path_entities": [binding["entity"]], "path_attributes": [deepcopy(binding["attributes"])],
            "path_relation_ids": [binding["relation_id"]]}


def _bypass(left, right, oid, kind, rid):
    records = []
    for first in left["bindings"]:
        for last in right["bindings"]:
            if first["target"] != oid or last["source"] != oid:
                return None
            if kind == "CFG" and (not _known(first["context"]) or first["context"] != last["context"]):
                return None
            a, b = _path(first), _path(last)
            records.append({"source": first["source"], "target": last["target"], "kind": kind,
                            "label": first["label"], "entity": first["entity"], "context": first["context"],
                            "attributes": {field: a[field] + b[field] for field in PATH_FIELDS},
                            "path": first["path"] + [oid] + last["path"], "relation_id": rid})
    return {"id": rid, "kind": kind, "sources": deepcopy(left["sources"]), "targets": deepcopy(right["targets"]),
            "bindings": records, "attributes": {}}


def _carry(graph, oid, carrier, rule, original_carrier):
    node = next(node for node in graph["nodes"] if node["id"] == oid)
    target = next(node for node in graph["nodes"] if node["id"] == carrier)
    target["entries"] = sorted(set(target["entries"] + node["entries"]), key=_order)
    retained = target["attributes"].setdefault("retained_contents", [])
    for entry_id in node["entries"]:
        entry = graph["dictionary"][entry_id]
        retained.append(_record(entry))
        if entry["status"] != "carried":
            entry.update(status="carried", representative=None, carrier=carrier,
                         carrier_original=original_carrier, rule=rule)
        else:
            entry["carrier"] = carrier
    retained.sort(key=lambda entry: _order(entry["id"]))
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] != oid]


def _rule1_carrier(graph, node):
    if _role(node) not in {"<operator>.assignment", "assignment"}:
        return None
    edges = _incidents(graph, node["id"])
    incoming = [edge for edge in edges if edge["kind"] == "AST" and edge["targets"] == [node["id"]]]
    outgoing = [edge for edge in edges if edge["kind"] == "AST" and edge["sources"] == [node["id"]]]
    nodes = {node["id"]: node for node in graph["nodes"]}
    if len(incoming) != 1 or not outgoing:
        return None
    parent = nodes[incoming[0]["sources"][0]]
    if parent["kind"] != "BLOCK" or not _contains(node, parent):
        return None
    carriers = [nodes[edge["targets"][0]] for edge in outgoing
                if _role(nodes[edge["targets"][0]]) in {"<operator>.alloc", "alloc"}
                and _contains(node, nodes[edge["targets"][0]])]
    if len(carriers) != 1 or carriers[0]["id"] == node["id"]:
        return None
    return carriers[0]["id"]


def _rule2_allowed(original, node):
    edges = _incidents(original, node["id"])
    if node["kind"] == "COMMENT":
        return len(edges) <= 1 and all(edge["kind"] == "AST" and edge["targets"] == [node["id"]] for edge in edges)
    if node["kind"] != "CALL" or _role(node).startswith("<operator>."):
        return False
    proof = _facts(node, 2)
    name = proof.get("call_name")
    if name not in {"printf", "fprintf", "puts", "putchar"}:
        return False
    for key in ("name", "method_full_name", "full_name", "operation"):
        recorded = node["attributes"].get(key)
        if recorded and str(recorded).split("(", 1)[0].split("::")[-1] != name:
            return False
    required = ("constant_output", "result_unused", "output_only", "arguments_valid", "control_context_preserved")
    if any(proof.get(key) is not True for key in required) or proof.get("vulnerability_relevant") is not False:
        return False
    output = proof.get("format")
    if not isinstance(output, str) or "%" in output.replace("%%", ""):
        return False
    if any(edge["kind"] not in {"AST", "CFG"} or len(edge["bindings"]) != 1 for edge in edges):
        return False
    cfg_in = [edge for edge in edges if edge["kind"] == "CFG" and edge["targets"] == [node["id"]]]
    cfg_out = [edge for edge in edges if edge["kind"] == "CFG" and edge["sources"] == [node["id"]]]
    if not cfg_in and not cfg_out:
        return True
    if len(cfg_in) != 1 or len(cfg_out) != 1 or cfg_in[0]["sources"] == cfg_out[0]["targets"]:
        return False
    context = cfg_in[0]["bindings"][0]["context"]
    return _known(context) and context == cfg_out[0]["bindings"][0]["context"]


def _remove_intermediate(graph, oid, rule, carrier=None):
    affected = _incidents(graph, oid)
    occupied = {edge["id"] for edge in graph["relations"]}
    ast_in = [edge for edge in affected if edge["kind"] == "AST" and edge["targets"] == [oid]]
    ast_out = [edge for edge in affected if edge["kind"] == "AST" and edge["sources"] == [oid]]
    if len(ast_in) > 1:
        return False
    additions = []
    if ast_in:
        for child_edge in ast_out:
            if set(ast_in[0]["sources"]) & set(child_edge["targets"]):
                return False
            rid = _identity("ast", [ast_in[0]["id"], oid, child_edge["id"]], occupied)
            occupied.add(rid)
            edge = _bypass(ast_in[0], child_edge, oid, "AST", rid)
            if edge is None:
                return False
            additions.append(edge)
    if rule == 1:
        for edge in affected:
            if edge["kind"] == "AST":
                continue
            migrated = deepcopy(edge)
            migrated["sources"] = [carrier if endpoint == oid else endpoint for endpoint in migrated["sources"]]
            migrated["targets"] = [carrier if endpoint == oid else endpoint for endpoint in migrated["targets"]]
            additions.append(migrated)
    else:
        cfg_in = [edge for edge in affected if edge["kind"] == "CFG" and edge["targets"] == [oid]]
        cfg_out = [edge for edge in affected if edge["kind"] == "CFG" and edge["sources"] == [oid]]
        if cfg_in or cfg_out:
            if len(cfg_in) != 1 or len(cfg_out) != 1 or cfg_in[0]["sources"] == cfg_out[0]["targets"]:
                return False
            rid = _identity("cfg", [cfg_in[0]["id"], oid, cfg_out[0]["id"]], occupied)
            edge = _bypass(cfg_in[0], cfg_out[0], oid, "CFG", rid)
            if edge is None:
                return False
            additions.append(edge)
    graph["relations"] = [edge for edge in graph["relations"] if edge not in affected] + additions
    if rule == 1:
        _carry(graph, oid, carrier, rule, carrier)
    else:
        graph["nodes"] = [node for node in graph["nodes"] if node["id"] != oid]
        graph["dictionary"][oid].update(status="removed", representative=None, carrier=None, rule=2)
    return True


def _merge(graph):
    candidates = defaultdict(list)
    unknown_types = {"", "ANY", "UNKNOWN", "<UNKNOWN>", "?"}
    declaration_kinds = {"LOCAL", "PARAM", "PARAMETER", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT"}
    for node in graph["nodes"]:
        kind, attrs = node["kind"], node["attributes"]
        if kind not in declaration_kinds | {"TYPE_REF", "METHOD_RETURN"}:
            continue
        declared, context = attrs.get("declared_type"), attrs.get("function_context")
        if not declared or _text(declared).upper().strip() in unknown_types or not _known(context):
            continue
        if kind in declaration_kinds and (not _known(attrs.get("name")) or not _known(attrs.get("scope"))):
            continue
        if kind not in declaration_kinds and not _known(attrs.get("occurrence")):
            continue
        candidates[(kind, _text(declared), _text(context))].append(node)
    for key, members in sorted(candidates.items()):
        if len(members) < 2:
            continue
        ids = {node["id"] for node in members}
        incident = [edge for edge in graph["relations"] if ids.intersection(edge["sources"] + edge["targets"])]
        if any(not edge["bindings"] for edge in incident):
            continue
        entries = sorted({entry for node in members for entry in node["entries"]}, key=_order)
        representative = _identity("node", [key, entries], set(graph["dictionary"]) | {node["id"] for node in graph["nodes"]})
        records = [_record(graph["dictionary"][entry]) for entry in entries]
        new = {"id": representative, "kind": key[0], "code": "\n".join(node["code"] for node in sorted(members, key=lambda node: _order(node["id"]))),
               "entries": entries, "attributes": {"declared_type": deepcopy(members[0]["attributes"]["declared_type"]),
                                                   "function_context": deepcopy(members[0]["attributes"]["function_context"]),
                                                   "retained_contents": records}}
        for entry_id in entries:
            entry = graph["dictionary"][entry_id]
            if entry["status"] == "carried":
                entry["carrier"] = representative
            else:
                entry.update(status="merged", rule=4, representative=representative, carrier=None)
        for edge in incident:
            for side in ("sources", "targets"):
                edge[side] = sorted({representative if endpoint in ids else endpoint for endpoint in edge[side]}, key=_order)
        graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in ids] + [new]


def _compatible(graph, edges):
    ports, witnesses = defaultdict(set), set()
    sources = {edge["sources"][0] for edge in edges}
    targets = {edge["targets"][0] for edge in edges}
    if sources & targets:
        return False
    for edge in edges:
        for binding in edge["bindings"]:
            source, target = _physical(graph, binding["source"]), _physical(graph, binding["target"])
            if [source] != edge["sources"] or [target] != edge["targets"]:
                return False
            ports[(source, 0)].add(binding["source"])
            ports[(target, 1)].add(binding["target"])
            witnesses.add((source, target))
    return all(len(values) == 1 for values in ports.values()) and witnesses == {(s, t) for s in sources for t in targets}


def _control_guard(graph, edge):
    contexts = {binding["context"] for binding in edge["bindings"]}
    if len(contexts) == 1 and _known(next(iter(contexts))):
        return {"context": next(iter(contexts))}, []
    if contexts != {""}:
        return None, []
    guards, witnesses = [], []
    for source in {binding["source"] for binding in edge["bindings"]}:
        records = [binding for relation in graph["relations"] if relation["kind"] == "CDG"
                   for binding in relation["bindings"] if binding["target"] == source]
        if not records:
            return None, []
        conditions = []
        for binding in records:
            attrs = binding["attributes"]
            condition = next((attrs[key] for key in ("branch", "condition", "control_condition")
                              if key in attrs and attrs[key] is not None and attrs[key] != ""), None)
            if condition is None and binding["context"]:
                condition = binding["context"]
            if condition is None:
                label = binding["label"].split(":", 1)
                if len(label) == 2 and label[1].strip().lower() in {"true", "false"}:
                    condition = label[1].strip().lower()
            if not _known(condition):
                return None, []
            conditions.append({"controller": binding["source"], "condition": deepcopy(condition)})
            witnesses.append(binding["relation_id"])
        guards.append(sorted({_encode(condition) for condition in conditions}))
    if any(guard != guards[0] for guard in guards[1:]):
        return None, []
    return {"cdg": [json.loads(condition) for condition in guards[0]]}, sorted(set(witnesses), key=_order)


def _pack(graph, kind):
    nodes = {node["id"]: node for node in graph["nodes"]}

    def vertex(oid):
        return _order(min(nodes[oid]["entries"], key=_order))

    def edge_order(edge):
        origin_ids = [oid for binding in edge["bindings"] for oid in
                      (binding["attributes"].get("path_relation_ids", []) if binding["path"] else [binding["relation_id"]])]
        return vertex(edge["sources"][0]), vertex(edge["targets"][0]), tuple(_order(oid) for oid in sorted(origin_ids, key=_order)), _order(edge["id"])

    pools = defaultdict(list)
    guards, guard_witnesses = {}, {}
    for edge in graph["relations"]:
        if edge["kind"] != kind:
            continue
        if kind == "CFG":
            guard, evidence = _control_guard(graph, edge)
            if guard is None:
                continue
            guards[edge["id"]], guard_witnesses[edge["id"]] = guard, evidence
            pools[_encode(guard)].append(edge)
        else:
            pools[""].append(edge)
    consumed, packed = set(), []
    occupied = {edge["id"] for edge in graph["relations"]}

    def accept(group, mode):
        source_count = len({edge["sources"][0] for edge in group})
        target_count = len({edge["targets"][0] for edge in group})
        limits = {"MM": source_count >= 2 and target_count >= 2,
                  "SM": source_count == 1 and target_count >= 2,
                  "MS": source_count >= 2 and target_count == 1}
        if len(group) < 2 or not limits[mode] or not _compatible(graph, group):
            return False
        ids = sorted((edge["id"] for edge in group), key=_order)
        rid = _identity("hyperedge", [kind, mode, ids], occupied)
        occupied.add(rid)
        bindings = sorted([deepcopy(binding) for edge in group for binding in edge["bindings"]], key=_encode)
        attrs = {"mode": mode, "witness_relation_ids": ids}
        if kind == "CFG":
            attrs["control_guard"] = deepcopy(guards[ids[0]])
            attrs["guard_relation_ids"] = sorted({oid for edge_id in ids for oid in guard_witnesses[edge_id]}, key=_order)
        if kind == "DDG":
            attrs["labels"] = sorted({binding["label"] for binding in bindings})
            annotations = [{"source": binding["source"], "entity": binding["entity"], "constant": binding["source_constant"]}
                           for binding in bindings if "source_constant" in binding]
            attrs["source_annotations"] = [json.loads(value) for value in sorted({_encode(item) for item in annotations})]
        packed.append({"id": rid, "kind": "CTRL" if kind == "CFG" else "DATA",
                       "sources": sorted({edge["sources"][0] for edge in group}, key=vertex),
                       "targets": sorted({edge["targets"][0] for edge in group}, key=vertex),
                       "bindings": bindings, "attributes": attrs})
        consumed.update(ids)
        return True

    def pattern(column):
        records = []
        for edge in column:
            for binding in edge["bindings"]:
                attrs = deepcopy(binding["attributes"])
                if binding["path"]:
                    attrs.pop("path_relation_ids", None)
                records.append([edge["sources"][0], binding["source"], binding["kind"], binding["label"],
                                binding["entity"], binding["context"], attrs,
                                {"constant": binding["source_constant"]} if "source_constant" in binding else {}])
        return _encode(sorted(records, key=_encode))

    for pool in (sorted(pool, key=edge_order) for _, pool in sorted(pools.items())):
        for mode in ("MM", "SM", "MS"):
            while True:
                remaining = [edge for edge in pool if edge["id"] not in consumed]
                found = False
                if mode == "MM":
                    columns = defaultdict(list)
                    for edge in remaining:
                        logical_targets = {binding["target"] for binding in edge["bindings"]}
                        if len(logical_targets) == 1:
                            columns[(edge["targets"][0], next(iter(logical_targets)))].append(edge)
                    ordered = sorted(columns, key=lambda key: (vertex(key[0]), _order(key[1])))
                    for seed in ordered:
                        group = list(columns[seed])
                        signature = pattern(group)
                        for other in ordered:
                            if other != seed and pattern(columns[other]) == signature and _compatible(graph, group + columns[other]):
                                group += columns[other]
                        if accept(group, mode):
                            found = True
                            break
                else:
                    side, port = ("sources", "source") if mode == "SM" else ("targets", "target")
                    for seed in remaining:
                        originals = {binding[port] for binding in seed["bindings"]}
                        if len(originals) != 1:
                            continue
                        group = [seed]
                        for edge in remaining:
                            if edge is seed or edge[side] != seed[side] or {binding[port] for binding in edge["bindings"]} != originals:
                                continue
                            if _compatible(graph, group + [edge]):
                                group.append(edge)
                        if accept(group, mode):
                            found = True
                            break
                if not found:
                    break
    graph["relations"] = [edge for edge in graph["relations"] if edge["id"] not in consumed] + packed


def transform_graph(original):
    """Construct an HCPG from an immutable original CPG and known conditions.

    Rule 2 facts are provided in attributes.pruning.rule2: call_name, format,
    constant_output, result_unused, output_only, arguments_valid,
    control_context_preserved and vulnerability_relevant. Declaration sharing
    requires declared_type/function_context and name/scope or occurrence fields.
    Unknown conditions preserve the original node and its relations.
    """
    original = normalize_graph(original)
    if original["metadata"].get("representation") == "HCPG" or any(edge["kind"] in {"CTRL", "DATA"} for edge in original["relations"]):
        raise ValueError("An original binary CPG is required; hyperedge memberships cannot establish missing bindings")
    nodes = {node["id"]: node for node in original["nodes"]}
    if set(nodes) != set(original["dictionary"]):
        raise ValueError("Original CPG dictionary must cover its original nodes exactly")
    for oid, node in nodes.items():
        entry = original["dictionary"][oid]
        if node["entries"] != [oid] or entry["status"] != "retained" or entry["representative"] != oid or entry["carrier"] is not None or _encode(_record(entry)) != _encode(_record(node)):
            raise ValueError("Original CPG identities must be intact")
    for edge in original["relations"]:
        if not edge["bindings"]:
            raise ValueError("Every original relation requires endpoint bindings")
        for binding in edge["bindings"]:
            if [binding["source"]] != edge["sources"] or [binding["target"]] != edge["targets"] or binding["kind"] != edge["kind"] or binding["path"]:
                raise ValueError("Original relation bindings disagree with their endpoints or kind")
    graph = deepcopy(original)
    for original_node in original["nodes"]:
        active = {node["id"]: node for node in graph["nodes"]}
        oid = original_node["id"]
        if oid not in active:
            continue
        carrier = _rule1_carrier(graph, active[oid])
        if carrier is not None:
            _remove_intermediate(graph, oid, 1, carrier)
        elif _rule2_allowed(original, original_node):
            _remove_intermediate(graph, oid, 2)
    for node in original["nodes"]:
        oid = node["id"]
        if node["kind"] not in {"LITERAL", "IDENTIFIER"} or oid not in {current["id"] for current in graph["nodes"]}:
            continue
        edges = _incidents(original, oid)
        if len(edges) != 1 or edges[0]["kind"] != "AST" or edges[0]["targets"] != [oid]:
            continue
        if any(field in node["attributes"] for field in ("entity", "binding", "bindings", "symbol_id", "declaration_id", "ref")):
            continue
        parent = edges[0]["sources"][0]
        carrier = _physical(graph, parent)
        if carrier == oid or carrier not in {current["id"] for current in graph["nodes"]} or not _contains(node, nodes[parent]):
            continue
        # Current bindings can carry a dependency role migrated from another
        # occurrence; pruning that carrier would discard an independent role.
        if any(edge["kind"] != "AST" for edge in _incidents(graph, oid)):
            continue
        graph["relations"] = [edge for edge in graph["relations"] if oid not in edge["sources"] + edge["targets"]]
        _carry(graph, oid, carrier, 3, parent)
    _merge(graph)
    for edge in graph["relations"]:
        if edge["kind"] == "DDG":
            for binding in edge["bindings"]:
                attrs = original["dictionary"][binding["source"]]["attributes"]
                if "explicit_constant" in attrs:
                    binding["source_constant"] = deepcopy(attrs["explicit_constant"])
    _pack(graph, "CFG")
    _pack(graph, "DDG")
    graph["metadata"].update(representation="HCPG", original_node_count=len(original["dictionary"]))
    return normalize_graph(graph)
