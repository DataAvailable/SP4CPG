"""Independent, provenance-aware validation of a supplied CPG/HCPG pair.

Reference CFG/DDG relations are projected from the immutable original graph.
Only active output relations are decoded; a dictionary or construction log
never supplies an absent output edge.  Relation keys retain the complete raw
label, entity, control context and attributes. AST projection and unchanged
CDG edges are also checked as structural integrity obligations. This checks dependencies
encoded by the input CPG, not full program-behaviour equivalence.
"""

import json
import re
from collections import defaultdict, deque


KINDS = ("CFG", "DDG")
RELATION_KINDS = {"AST", "CFG", "CDG", "DDG", "CTRL", "DATA"}
BASE_FIELDS = ("id", "kind", "code", "attributes")
BINDING_FIELDS = ("source", "target", "kind", "label", "entity", "context", "attributes", "path")


def _freeze(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _key(binding):
    return (binding["source"], binding["target"], binding["label"],
            binding["entity"], binding["context"], _freeze(binding["attributes"]))


def _same_original_fields(record, original):
    return isinstance(record, dict) and all(field in record and _freeze(record[field]) == _freeze(original.get(field)) for field in BASE_FIELDS)


def _content_in(node, carrier):
    content = str(node.get("code", "")).strip()
    code = str(carrier.get("code", ""))
    if not content:
        return False
    if node.get("kind") == "IDENTIFIER":
        return bool(re.search(r"(?<![\w$])" + re.escape(content) + r"(?![\w$])", code))
    return content in code


def _declared_type(node):
    value = node.get("attributes", {}).get("declared_type")
    return value if value not in (None, "") and str(value).strip().upper() not in {"ANY", "UNKNOWN", "<UNKNOWN>", "?"} else None


def _pruning_facts(node):
    """Read explicit facts from JSON/DOT attributes without importing the builder."""
    attributes = node.get("attributes", {})
    value = attributes.get("pruning", attributes.get("sp4cpg_pruning", {}))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _valid_call_proof(node):
    """Check explicit semantic-provider evidence, never vulnerability labels."""
    attributes = node.get("attributes", {})
    proof = _pruning_facts(node).get("rule2", {})
    if not isinstance(proof, dict):
        return False
    names = [str(proof.get("call_name", "")), str(attributes.get("name", "")),
             str(attributes.get("full_name", "")), str(attributes.get("method_full_name", ""))]
    if any(re.search(r"\b(?:sprintf|snprintf|vsnprintf)\b", name) for name in names):
        return False
    if proof.get("call_name") not in {"printf", "fprintf", "puts", "putchar"}:
        return False
    if any(name and name.split("(", 1)[0].split("::")[-1] != proof["call_name"] for name in names[1:]):
        return False
    output = proof.get("format")
    if not isinstance(output, str) or "%" in output.replace("%%", ""):
        return False
    flags = ("constant_output", "result_unused", "output_only", "arguments_valid", "control_context_preserved")
    return all(proof.get(flag) is True for flag in flags) and proof.get("vulnerability_relevant") is False


def _inspect_graph(graph, label, errors):
    if not isinstance(graph, dict):
        errors["schema"].append(f"{label}: graph must be an object")
        return {}, [], {}
    if graph.get("schema_version") != 2:
        errors["schema"].append(f"{label}: schema_version must be 2")
    if not isinstance(graph.get("graph_id"), str) or not graph["graph_id"]:
        errors["schema"].append(f"{label}: stable string graph_id is required")
    nodes = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            errors["schema"].append(f"{label}: every node requires a string ID")
            continue
        if node["id"] in nodes:
            errors["identity"].append(f"{label}: duplicate physical node {node['id']}")
        nodes[node["id"]] = node
        if any(field not in node for field in BASE_FIELDS) or not isinstance(node.get("attributes"), dict):
            errors["schema"].append(f"{label}: node {node['id']} lacks canonical attributes")
        if not isinstance(node.get("entries"), list) or any(not isinstance(value, str) for value in node.get("entries", [])):
            errors["schema"].append(f"{label}: node {node['id']} requires original identity entries")
    relations = []
    relation_ids = set()
    for relation in graph.get("relations", []):
        if not isinstance(relation, dict) or not isinstance(relation.get("id"), str):
            errors["schema"].append(f"{label}: every relation requires a string ID")
            continue
        rid = relation["id"]
        if rid in relation_ids:
            errors["identity"].append(f"{label}: duplicate relation ID {rid}")
        relation_ids.add(rid)
        if relation.get("kind") not in RELATION_KINDS:
            errors["schema"].append(f"{label}: {rid} has unknown relation kind")
        if not isinstance(relation.get("attributes"), dict):
            errors["schema"].append(f"{label}: {rid} requires relation attributes")
        members_valid = True
        for side in ("sources", "targets"):
            members = relation.get(side)
            if not isinstance(members, list) or not members or any(not isinstance(v, str) for v in members):
                errors["schema"].append(f"{label}: {rid} requires nonempty string {side}")
                members_valid = False
                continue
            if len(set(members)) != len(members) or any(v not in nodes for v in members):
                errors["binding"].append(f"{label}: {rid} has duplicate or absent physical {side}")
        if members_valid:
            if relation.get("kind") in {"CTRL", "DATA"}:
                if set(relation["sources"]) & set(relation["targets"]):
                    errors["binding"].append(f"{label}: hyperedge {rid} source/target sets intersect")
            elif len(relation["sources"]) != 1 or len(relation["targets"]) != 1:
                errors["binding"].append(f"{label}: binary relation {rid} needs singleton endpoints")
        if not members_valid:
            continue
        bindings = relation.get("bindings")
        if not isinstance(bindings, list) or not bindings:
            errors["binding"].append(f"{label}: active relation {rid} has no binding records")
            continue
        valid_bindings = []
        for binding in bindings:
            if not isinstance(binding, dict) or any(field not in binding for field in BINDING_FIELDS):
                errors["binding"].append(f"{label}: {rid} has incomplete binding attributes")
                continue
            if any(not isinstance(binding[field], str) for field in ("source", "target", "kind", "label", "entity", "context")):
                errors["binding"].append(f"{label}: {rid} binding identities/labels must be strings")
                continue
            if not isinstance(binding["attributes"], dict) or not isinstance(binding["path"], list) or any(not isinstance(value, str) for value in binding["path"]):
                errors["binding"].append(f"{label}: {rid} binding attributes/path have invalid types")
                continue
            expected = {"CTRL": "CFG", "DATA": "DDG"}.get(relation.get("kind"), relation.get("kind"))
            if binding["kind"] != expected:
                errors["binding"].append(f"{label}: {rid} changes original relation kind")
                continue
            valid_bindings.append(binding)
        relations.append((relation, valid_bindings))
    dictionary = graph.get("dictionary", {})
    if not isinstance(dictionary, dict):
        errors["schema"].append(f"{label}: dictionary must be an object")
        dictionary = {}
    return nodes, relations, dictionary


def _original_edges(relations, originals, errors):
    edges = []
    for relation, bindings in relations:
        if relation.get("kind") in {"CTRL", "DATA"}:
            errors["schema"].append("original: expected binary CPG input, not packed hyperedges")
        for binding in bindings:
            if binding["source"] not in originals or binding["target"] not in originals:
                errors["identity"].append(f"original: {relation['id']} has unknown logical endpoints")
                continue
            if binding["source"] not in relation.get("sources", []) or binding["target"] not in relation.get("targets", []):
                errors["binding"].append(f"original: {relation['id']} bindings disagree with physical endpoints")
                continue
            if binding["path"]:
                errors["schema"].append(f"original: {relation['id']} is already projected")
            if binding.get("relation_id", relation["id"]) != relation["id"]:
                errors["identity"].append(f"original: {relation['id']} binding changes its original relation identity")
            edge = dict(binding)
            edge["relation_id"] = relation["id"]
            edges.append(edge)
    return edges


def _check_dictionary(originals, original_dictionary, nodes, dictionary, edges, errors):
    if set(original_dictionary) != set(originals):
        errors["identity"].append("original: dictionary must cover exactly the original CPG node identities")
    if set(dictionary) != set(originals):
        errors["identity"].append("output: dictionary must cover exactly the immutable original identity domain")
    incidence = defaultdict(list)
    for edge in edges:
        incidence[edge["source"]].append(edge)
        if edge["target"] != edge["source"]:
            incidence[edge["target"]].append(edge)
    physical_for = {}
    removed = set()
    for oid, original in originals.items():
        baseline_entry = original_dictionary.get(oid, {})
        entry = dictionary.get(oid)
        if not _same_original_fields(baseline_entry, original):
            errors["identity"].append(f"original: dictionary entry {oid} differs from original node attributes")
        if not isinstance(baseline_entry, dict) or baseline_entry.get("status") != "retained" or baseline_entry.get("representative") != oid or baseline_entry.get("carrier") is not None or original.get("entries") != [oid]:
            errors["identity"].append(f"original: {oid} must be an untransformed self-bound CPG identity")
        if not isinstance(entry, dict):
            errors["identity"].append(f"output: missing original role {oid}")
            continue
        if not _same_original_fields(entry, original):
            errors["content"].append(f"output: dictionary entry {oid} changes original kind/code/attributes/identity")
        status = entry.get("status")
        physical = entry.get("carrier") if status == "carried" else entry.get("representative")
        if status == "removed":
            if entry.get("representative") is not None or entry.get("carrier") is not None:
                errors["identity"].append(f"output: completely removed role {oid} still has a physical mapping")
            incident = incidence[oid]
            allowed = entry.get("rule") == 2 and len(incident) <= 1 and all(edge["kind"] == "AST" and edge["target"] == oid for edge in incident)
            if original.get("kind") == "CALL" and entry.get("rule") == 2 and _valid_call_proof(original):
                non_ast = [edge for edge in incident if edge["kind"] != "AST"]
                incoming = [edge for edge in non_ast if edge["kind"] == "CFG" and edge["target"] == oid]
                outgoing = [edge for edge in non_ast if edge["kind"] == "CFG" and edge["source"] == oid]
                allowed = not non_ast or (
                    len(non_ast) == 2 and len(incoming) == len(outgoing) == 1
                    and incoming[0]["source"] != outgoing[0]["target"]
                    and bool(incoming[0]["context"])
                    and incoming[0]["context"] == outgoing[0]["context"])
            elif original.get("kind") != "COMMENT":
                allowed = False
            if not allowed:
                errors["identity"].append(f"output: original role {oid} cannot be independently confirmed as completely removable")
            else:
                removed.add(oid)
            continue
        if status not in {"retained", "merged", "carried"}:
            errors["identity"].append(f"output: {oid} has unknown retention status")
            continue
        if status == "retained" and (entry.get("representative") != oid or entry.get("carrier") is not None):
            errors["identity"].append(f"output: retained role {oid} must keep its own physical representative")
        if status == "merged" and (entry.get("carrier") is not None or physical in originals):
            errors["identity"].append(f"output: merged role {oid} requires a fresh Rule 4 representative")
        if status == "carried" and entry.get("representative") is not None:
            errors["identity"].append(f"output: carried role {oid} cannot retain a second physical representative")
        if physical not in nodes:
            errors["identity"].append(f"output: original role {oid} has no active representative/content carrier")
            continue
        physical_for[oid] = physical
        node = nodes[physical]
        if oid not in node.get("entries", []):
            errors["identity"].append(f"output: {oid} is absent from its physical node entries")
        if status != "carried" and node.get("kind") != original.get("kind"):
            errors["identity"].append(f"output: {oid} representative changes node kind")
        if status == "retained":
            if node.get("code") != original.get("code") or any(k not in node.get("attributes", {}) or _freeze(node["attributes"][k]) != _freeze(v) for k, v in original.get("attributes", {}).items()):
                errors["content"].append(f"output: retained role {oid} loses original code/attributes")
        if status == "carried":
            carrier_original = entry.get("carrier_original")
            carrier_entry = dictionary.get(carrier_original, {})
            carrier_mapping = carrier_entry.get("carrier") if carrier_entry.get("status") == "carried" else carrier_entry.get("representative")
            incident = incidence[oid]
            carrier_node = originals.get(carrier_original, {})
            ast_connected = any(edge["kind"] == "AST" and {edge["source"], edge["target"]} == {oid, carrier_original} for edge in incident)
            if carrier_mapping != physical or not ast_connected or not _content_in(original, carrier_node):
                errors["content"].append(f"output: carried role {oid} lacks its original expression/content association")
            if entry.get("rule") == 3:
                if original.get("kind") not in {"IDENTIFIER", "LITERAL"} or len(incident) != 1 or incident[0]["kind"] != "AST" or incident[0]["target"] != oid:
                    errors["identity"].append(f"output: Rule 3 role {oid} was not an original AST-only leaf")
                if any(key in original.get("attributes", {}) for key in ("entity", "binding", "bindings", "symbol_id", "ref", "declaration_id")):
                    errors["identity"].append(f"output: Rule 3 role {oid} had independent original binding attributes")
            elif entry.get("rule") == 1:
                proof = _pruning_facts(original).get("rule1", {})
                incoming = [edge for edge in incident if edge["target"] == oid]
                parent = originals.get(incoming[0]["source"], {}) if len(incoming) == 1 else {}
                children = [originals.get(edge["target"], {}) for edge in incident if edge["source"] == oid]
                pattern = parent.get("kind") == "BLOCK" and _content_in(original, parent) and any(child.get("kind") in {"<operator>.alloc", "operator.alloc", "ALLOC"} and _content_in(original, child) for child in children)
                if original.get("kind") not in {"<operator>.assignment", "operator.assignment", "ASSIGNMENT"} or not pattern or len(incoming) != 1 or not isinstance(proof, dict) or proof.get("redundant") is not True or proof.get("carrier") != carrier_original or any(edge["kind"] != "AST" for edge in incident):
                    errors["identity"].append(f"output: Rule 1 role {oid} lacks independently checkable redundancy evidence")
            else:
                errors["identity"].append(f"output: unsupported content-carrying rule for {oid}")
            records = node.get("attributes", {}).get("retained_contents", [])
            if not any(_same_original_fields(record, original) for record in records):
                errors["content"].append(f"output: carried role {oid} missing actual retained content record")
    for pid, node in nodes.items():
        entries = node.get("entries", [])
        if not isinstance(entries, list) or any(not isinstance(value, str) for value in entries):
            continue  # Already reported as a schema error; never coerce identities.
        if not entries or len(entries) != len(set(entries)):
            errors["identity"].append(f"output: {pid} has empty or duplicate original entries")
        for oid in entries:
            if physical_for.get(oid) != pid:
                errors["identity"].append(f"output: physical node {pid} has an unrelated/unmapped original entry {oid}")
        merged = [oid for oid in entries if dictionary.get(oid, {}).get("status") == "merged"]
        if merged:
            known = [originals[oid] for oid in merged if oid in originals]
            if len(known) < 2 or len({node.get("kind") for node in known}) != 1 or len({_freeze(_declared_type(node)) for node in known}) != 1 or _declared_type(known[0]) is None:
                errors["identity"].append(f"output: merge {pid} does not preserve a common known kind/type")
            if any(node.get("kind") not in {"LOCAL", "PARAM", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "TYPE_REF", "METHOD_RETURN"} for node in known):
                errors["identity"].append(f"output: merge {pid} has an ineligible node kind")
            contexts = {_freeze(item.get("attributes", {}).get("function_context")) for item in known}
            if len(contexts) != 1 or any(not item.get("attributes", {}).get("function_context") for item in known):
                errors["identity"].append(f"output: merge {pid} lacks a common known function context")
            records = node.get("attributes", {}).get("retained_contents", [])
            for item in known:
                attrs = item.get("attributes", {})
                declaration = item.get("kind") in {"LOCAL", "PARAM", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT"}
                if not attrs.get("declared_type") or (declaration and (not attrs.get("name") or attrs.get("scope") in (None, ""))) or (not declaration and not attrs.get("occurrence")):
                    errors["identity"].append(f"output: merge {pid} lacks declaration/occurrence identity for {item['id']}")
                if dictionary[item["id"]].get("rule") != 4:
                    errors["identity"].append(f"output: merge {pid} lacks Rule 4 provenance for {item['id']}")
                if not any(_same_original_fields(record, item) for record in records):
                    errors["content"].append(f"output: merge {pid} drops original declaration content {item['id']}")
    return physical_for, removed


def _project_paths(edges, internal_nodes, kind, dropped_targets=()):
    """Project original typed paths, stopping at every surviving logical anchor."""
    outgoing = defaultdict(list)
    for edge in edges:
        if edge["kind"] == kind:
            outgoing[edge["source"]].append(edge)
    reference = set()
    paths = defaultdict(set)
    for source, initial_edges in outgoing.items():
        if source in internal_nodes or source in dropped_targets:
            continue
        for first in initial_edges:
            pending = [(first["target"], [first], [])]
            while pending:
                target, path_edges, internal = pending.pop()
                if target in dropped_targets:
                    continue
                if target not in internal_nodes:
                    binding = dict(first, target=target)
                    if internal:
                        binding["attributes"] = {
                            "path_labels": [edge["label"] for edge in path_edges],
                            "path_contexts": [edge["context"] for edge in path_edges],
                            "path_entities": [edge["entity"] for edge in path_edges],
                            "path_attributes": [edge["attributes"] for edge in path_edges],
                            "path_relation_ids": [edge["relation_id"] for edge in path_edges],
                        }
                    key = _key(binding)
                    reference.add(key)
                    paths[key].add(tuple(internal))
                    continue
                if target in internal:
                    continue
                for following in outgoing[target]:
                    if kind != "CFG" or (following["context"] == first["context"] and bool(first["context"])):
                        pending.append((following["target"], path_edges + [following], internal + [target]))
    return reference, paths


def _closure(keys, anchors):
    adjacency = {anchor: set() for anchor in anchors}
    for key in keys:
        adjacency.setdefault(key[0], set()).add(key[1])
    reach = set()
    for source in sorted(anchors):
        # Starting with neighbours, not the source, detects positive cycles.
        pending = deque(adjacency.get(source, set()))
        seen = set()
        while pending:
            target = pending.popleft()
            if target in seen:
                continue
            seen.add(target)
            pending.extend(adjacency.get(target, set()) - seen)
        reach.update((source, target) for target in seen)
    return reach


def validate_pair(original, transformed):
    """Validate one independently supplied original/transformed graph pair.

    Returned precision/recall are fractions in [0, 1], or ``None`` (N/A)
    when their denominator is zero.  Failures remain explicit in the report.
    """
    if not isinstance(original, dict) or not isinstance(transformed, dict):
        raise ValueError("Original and transformed graphs must be JSON objects")
    errors = {"schema": [], "identity": [], "content": [], "binding": []}
    originals, input_relations, original_dictionary = _inspect_graph(original, "original", errors)
    nodes, output_relations, dictionary = _inspect_graph(transformed, "output", errors)
    if original.get("graph_id") != transformed.get("graph_id"):
        errors["identity"].append("Original/output graph_id mismatch; graph-local identities cannot be aligned across samples")
    edges = _original_edges(input_relations, originals, errors)
    physical_for, removed = _check_dictionary(originals, original_dictionary, nodes, dictionary, edges, errors)
    cfg, cfg_paths = _project_paths(edges, removed, "CFG")
    ast_internal = removed | {oid for oid, entry in dictionary.items() if isinstance(entry, dict) and entry.get("status") == "carried" and entry.get("rule") == 1}
    ast_leaves = {oid for oid, entry in dictionary.items() if isinstance(entry, dict) and entry.get("status") == "carried" and entry.get("rule") == 3}
    ast, ast_paths = _project_paths(edges, ast_internal, "AST", dropped_targets=ast_leaves)
    reference = {"CFG": cfg, "AST": ast,
                 **{kind: {_key(edge) for edge in edges if edge["kind"] == kind} for kind in ("DDG", "CDG")}}
    projected_paths = {"CFG": cfg_paths, "AST": ast_paths}
    decoded = {kind: set() for kind in reference}
    original_id_keys = {kind: defaultdict(set) for kind in ("CFG", "DDG", "CDG")}
    covered_ids = {kind: set() for kind in original_id_keys}
    for edge in edges:
        if edge["kind"] in original_id_keys:
            original_id_keys[edge["kind"]][edge["relation_id"]].add(_key(edge))
    for relation, bindings in output_relations:
        witnesses = set()
        physical_logical_ports = defaultdict(set)
        dependency_kind = {"CTRL": "CFG", "DATA": "DDG"}.get(relation.get("kind"), relation.get("kind"))
        expected_annotations = set()
        if relation.get("kind") == "CTRL" and (not bindings or len({b["context"] for b in bindings}) != 1 or not bindings[0]["context"]):
            errors["binding"].append(f"output: {relation['id']} groups incompatible or unknown control contexts")
        for binding in bindings:
            source, target = binding["source"], binding["target"]
            source_physical, target_physical = physical_for.get(source), physical_for.get(target)
            actual_member = source_physical in relation.get("sources", []) and target_physical in relation.get("targets", [])
            if source not in originals or target not in originals or not actual_member:
                errors["binding"].append(f"output: {relation['id']} binding has absent roles or disagrees with active memberships")
                continue
            physical_logical_ports[(source_physical, "source")].add(source)
            physical_logical_ports[(target_physical, "target")].add(target)
            key = _key(binding)
            valid_witness = True
            witnessed_original_ids = set()
            if dependency_kind in reference:
                decoded[dependency_kind].add(key)
                if key not in reference[dependency_kind]:
                    errors["binding"].append(f"output: {relation['id']} has no independently projected {dependency_kind} witness")
                    valid_witness = False
                if dependency_kind in {"DDG", "CDG"} and binding["path"]:
                    errors["binding"].append(f"output: {relation['id']} replaces a direct {dependency_kind} relation by a path")
                    valid_witness = False
                if dependency_kind == "DDG":
                    source_attrs = originals[source].get("attributes", {})
                    explicit = "explicit_constant" in source_attrs
                    should_annotate = explicit and (relation.get("kind") == "DATA" or transformed.get("metadata", {}).get("representation") == "HCPG")
                    if "source_constant" in binding:
                        if not explicit or _freeze(binding["source_constant"]) != _freeze(source_attrs["explicit_constant"]):
                            errors["binding"].append(f"output: {relation['id']} changes or invents an explicit source constant")
                            valid_witness = False
                    elif should_annotate:
                        errors["binding"].append(f"output: {relation['id']} loses an available explicit source constant")
                        valid_witness = False
                    if explicit:
                        expected_annotations.add(_freeze({"source": source, "entity": binding["entity"], "constant": source_attrs["explicit_constant"]}))
                if dependency_kind in projected_paths and tuple(binding["path"]) not in projected_paths[dependency_kind].get(key, set()):
                    errors["binding"].append(f"output: {relation['id']} has inconsistent {dependency_kind} bypass provenance")
                    valid_witness = False
                if dependency_kind in covered_ids:
                    if dependency_kind == "CFG" and binding["path"]:
                        # The complete annotation arrays above are independently
                        # matched against an original path before these IDs count.
                        if valid_witness:
                            witnessed_original_ids = set(binding["attributes"]["path_relation_ids"])
                    else:
                        original_relation_id = binding.get("relation_id", relation["id"])
                        if not isinstance(original_relation_id, str) or key not in original_id_keys[dependency_kind].get(original_relation_id, set()):
                            errors["identity"].append(f"output: {relation['id']} binding does not preserve its original {dependency_kind} relation identity")
                            valid_witness = False
                        else:
                            witnessed_original_ids.add(original_relation_id)
            if valid_witness:
                witnesses.add((source_physical, target_physical))
                if dependency_kind in covered_ids:
                    covered_ids[dependency_kind].update(witnessed_original_ids)
        if relation.get("kind") in {"CTRL", "DATA"}:
            if len(bindings) < 2 or len(relation["sources"]) * len(relation["targets"]) < 2:
                errors["binding"].append(f"output: {relation['id']} is not a multi-edge hyperedge grouping")
            if any(len(ports) != 1 for ports in physical_logical_ports.values()):
                errors["binding"].append(f"output: {relation['id']} mixes unrelated original entries through a shared physical member")
        if relation.get("kind") == "DATA":
            attributes = relation.get("attributes", {})
            labels = attributes.get("labels")
            if not isinstance(labels, list) or any(not isinstance(value, str) for value in labels) or set(labels) != {b["label"] for b in bindings}:
                errors["binding"].append(f"output: {relation['id']} label set disagrees with its raw bound DDG labels")
            annotations = attributes.get("source_annotations", [])
            if not isinstance(annotations, list) or {_freeze(item) for item in annotations} != expected_annotations:
                errors["binding"].append(f"output: {relation['id']} explicit source annotations disagree with original source entries")
        for source in relation.get("sources", []):
            for target in relation.get("targets", []):
                if (source, target) not in witnesses:
                    errors["binding"].append(f"output: {relation['id']} physical pair {source}->{target} has no valid bound witness")
    anchors = set(originals) - removed
    relations_report = {}
    for kind in KINDS:
        expected, actual = reference[kind], decoded[kind]
        tp, fp, fn = len(expected & actual), len(actual - expected), len(expected - actual)
        equal = expected == actual
        differences = set() if equal else _closure(expected, anchors) ^ _closure(actual, anchors)
        relations_report[kind] = {
            "tp": tp, "fp": fp, "fn": fn,
            "reference_count": len(expected), "decoded_count": len(actual),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "reachability_equal": not differences,
            "reachability_method": "edge_equality" if equal else "exhaustive_closure",
            "reachability_difference_count": len(differences),
            "reachability_difference_examples": [list(pair) for pair in sorted(differences)[:20]],
        }
    structural_report = {}
    for kind in ("AST", "CDG"):
        expected, actual = reference[kind], decoded[kind]
        tp, fp, fn = len(expected & actual), len(actual - expected), len(expected - actual)
        structural_report[kind] = {"tp": tp, "fp": fp, "fn": fn,
                                   "reference_count": len(expected), "decoded_count": len(actual)}
        if fp or fn:
            errors["binding"].append(f"output: {kind} structural relations differ from independently projected input (FP={fp}, FN={fn})")
    identity_coverage = {}
    for kind in covered_ids:
        expected_ids = set(original_id_keys[kind])
        missing_ids = expected_ids - covered_ids[kind]
        identity_coverage[kind] = {"original_relation_count": len(expected_ids),
                                   "covered_original_relation_count": len(covered_ids[kind]),
                                   "missing_original_relation_ids": sorted(missing_ids)}
        if missing_ids:
            errors["identity"].append(f"output: missing active witness coverage for original {kind} relation IDs: {', '.join(sorted(missing_ids))}")
    return {
        "graph_id": original.get("graph_id"),
        "exact_match": not any(errors.values()) and all(r["fp"] == r["fn"] == 0 and r["reachability_equal"] for r in relations_report.values()),
        "relations": relations_report, "structural_relations": structural_report,
        "relation_identity_coverage": identity_coverage, "errors": errors,
        "logical_anchor_count": len(anchors), "completely_removed_count": len(removed),
        "scope": "Dependencies encoded in the supplied original CPG; not full program behaviour equivalence",
    }


def aggregate_reports(reports):
    """Sum per-graph counts before micro aggregation; never merge graph IDs."""
    reports = list(reports)
    result = {"graph_count": len(reports), "exact_match_count": sum(r.get("exact_match", False) for r in reports), "relations": {}}
    for kind in KINDS:
        counts = {name: sum(r["relations"][kind][name] for r in reports) for name in ("tp", "fp", "fn")}
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        result["relations"][kind] = {
            **counts, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "reachability_difference_graph_count": sum(not r["relations"][kind]["reachability_equal"] for r in reports),
        }
    result["all_exact_match"] = bool(reports) and result["exact_match_count"] == len(reports)
    result["undefined_metric_value"] = None
    result["aggregation"] = "micro: sum independent per-graph TP/FP/FN before ratios; null means N/A"
    return result
