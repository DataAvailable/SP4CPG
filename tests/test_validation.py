"""Independent hand-constructed fixtures and destructive-output mutations."""

from copy import deepcopy

from sp4cpg.validation import aggregate_reports, validate_pair


def node(node_id, kind="CALL", code=None, attributes=None):
    return {"id": node_id, "kind": kind, "code": node_id if code is None else code,
            "attributes": deepcopy(attributes or {}), "entries": [node_id]}


def relation(relation_id, source, target, kind="DDG", label="x", entity="entity:x", context="branch:1", attributes=None):
    binding = {"source": source, "target": target, "kind": kind, "label": label,
               "entity": entity, "context": context, "attributes": deepcopy(attributes or {}),
               "path": [], "relation_id": relation_id}
    return {"id": relation_id, "kind": kind, "sources": [source], "targets": [target],
            "bindings": [binding], "attributes": {}}


def graph(nodes, relations=(), graph_id="sample"):
    dictionary = {}
    for item in nodes:
        dictionary[item["id"]] = {
            **{field: deepcopy(item[field]) for field in ("id", "kind", "code", "attributes")},
            "representative": item["id"], "carrier": None, "status": "retained", "rule": None}
    return {"schema_version": 2, "graph_id": graph_id, "nodes": deepcopy(nodes),
            "relations": deepcopy(list(relations)), "dictionary": dictionary, "metadata": {}}


def test_identity_and_empty_dependency_sets_are_not_given_100_percent():
    original = graph([node("a")])
    report = validate_pair(original, deepcopy(original))
    assert report["exact_match"]
    assert report["relations"]["CFG"]["precision"] is None
    assert report["relations"]["DDG"]["recall"] is None
    assert aggregate_reports([report])["relations"]["DDG"]["precision"] is None


def test_dropped_active_edge_is_not_restored_from_dictionary_or_history():
    original = graph([node("a"), node("b")], [relation("d", "a", "b")])
    output = deepcopy(original)
    output["relations"] = []
    output["metadata"]["construction_history"] = deepcopy(original["relations"])
    output["dictionary"]["a"]["historical_relations"] = deepcopy(original["relations"])
    report = validate_pair(original, output)
    assert not report["exact_match"]
    assert report["relations"]["DDG"]["fn"] == 1
    assert not report["relations"]["DDG"]["reachability_equal"]


def test_entity_context_label_and_original_attributes_cannot_shrink_comparison_key():
    original = graph([node("a"), node("b")], [relation("d", "a", "b", attributes={"scope": "local"})])
    for field in ("entity", "context", "label", "attributes"):
        output = deepcopy(original)
        output["relations"][0]["bindings"][0][field] = {} if field == "attributes" else ""
        report = validate_pair(original, output)
        assert not report["exact_match"], field
        assert report["relations"]["DDG"]["fn"] == 1, field
        assert report["relations"]["DDG"]["fp"] == 1, field
    output = deepcopy(original)
    del output["relations"][0]["bindings"][0]["context"]
    assert not validate_pair(original, output)["exact_match"]


def test_missing_original_role_cannot_remove_itself_from_domain():
    original = graph([node("a"), node("b")], [relation("d", "a", "b")])
    output = deepcopy(original)
    output["nodes"] = [output["nodes"][0]]
    output["relations"] = []
    del output["dictionary"]["b"]
    report = validate_pair(original, output)
    assert not report["exact_match"]
    assert report["relations"]["DDG"]["fn"] == 1
    assert report["logical_anchor_count"] == 2
    assert report["errors"]["identity"]


def test_invalid_complete_removal_does_not_hide_a_dependency_role():
    original = graph([node("a"), node("b")], [relation("d", "a", "b")])
    output = deepcopy(original)
    output["nodes"].pop()
    output["relations"] = []
    output["dictionary"]["b"].update(status="removed", rule=2, representative=None)
    report = validate_pair(original, output)
    assert report["relations"]["DDG"]["fn"] == 1
    assert report["logical_anchor_count"] == 2
    assert not report["exact_match"]


def test_original_attributes_and_actual_content_carriers_are_checked():
    original = graph([node("p", code="char buf[2]"), node("l", "LITERAL", "2", {"value": 2})],
                     [relation("ast", "p", "l", "AST", label="", entity="", context="")])
    output = deepcopy(original)
    output["nodes"].pop()
    output["nodes"][0]["entries"].append("l")
    output["nodes"][0]["attributes"]["retained_contents"] = [
        {key: deepcopy(original["nodes"][1][key]) for key in ("id", "kind", "code", "attributes")}]
    output["dictionary"]["l"].update(status="carried", rule=3, representative=None, carrier="p", carrier_original="p")
    output["relations"] = []
    assert validate_pair(original, output)["exact_match"]
    missing = deepcopy(output)
    missing["nodes"][0]["attributes"]["retained_contents"] = []
    assert validate_pair(original, missing)["errors"]["content"]
    changed = deepcopy(output)
    changed["dictionary"]["l"]["attributes"]["value"] = 20
    assert not validate_pair(original, changed)["exact_match"]
    changed = deepcopy(output)
    changed["nodes"][0]["code"] = "char buf[20]"
    assert not validate_pair(original, changed)["exact_match"]


def test_hyperedge_decoding_keeps_per_edge_labels_without_cartesian_product():
    original = graph([node(key) for key in ("a", "b", "x", "y", "z")], [
        relation("ax", "a", "x", label="a", entity="a"),
        relation("ay", "a", "y", label="a", entity="a"),
        relation("bx", "b", "x", label="b", entity="b"),
        relation("by", "b", "y", label="b", entity="b")])
    output = deepcopy(original)
    output["relations"] = [{"id": "hyper", "kind": "DATA", "sources": ["a", "b"], "targets": ["x", "y"],
                            "bindings": [deepcopy(r["bindings"][0]) for r in original["relations"]], "attributes": {"labels": ["a", "b"]}}]
    report = validate_pair(original, output)
    assert report["exact_match"]
    assert report["relations"]["DDG"]["decoded_count"] == 4
    output["relations"][0]["targets"].append("z")
    report = validate_pair(original, output)
    assert not report["exact_match"]
    assert any("no valid bound witness" in message for message in report["errors"]["binding"])


def test_binding_to_unrelated_physical_member_is_rejected():
    original = graph([node("a"), node("b"), node("c")], [relation("d", "a", "b")])
    output = deepcopy(original)
    output["relations"][0]["targets"] = ["c"]
    report = validate_pair(original, output)
    assert not report["exact_match"]
    assert report["relations"]["DDG"]["decoded_count"] == 0
    assert report["relations"]["DDG"]["fn"] == 1


def safe_call_graph():
    proof = {"call_name": "printf", "format": "hello", "constant_output": True,
             "result_unused": True, "output_only": True, "arguments_valid": True,
             "control_context_preserved": True, "vulnerability_relevant": False}
    return graph([node("a"), node("m", code='printf("hello")', attributes={"pruning": {"rule2": proof}}), node("b")], [
        relation("am", "a", "m", "CFG", label="enter", entity="e1", attributes={"raw": 1}),
        relation("mb", "m", "b", "CFG", label="leave", entity="e2", attributes={"raw": 2})])


def bypass_output(original):
    output = deepcopy(original)
    output["nodes"] = [n for n in output["nodes"] if n["id"] != "m"]
    output["dictionary"]["m"].update(status="removed", rule=2, representative=None)
    original_bindings = [r["bindings"][0] for r in original["relations"]]
    binding = deepcopy(original_bindings[0])
    binding.update(target="b", path=["m"], relation_id="bypass")
    binding["attributes"] = {"path_labels": [b["label"] for b in original_bindings],
                             "path_contexts": [b["context"] for b in original_bindings],
                             "path_entities": [b["entity"] for b in original_bindings],
                             "path_attributes": [b["attributes"] for b in original_bindings],
                             "path_relation_ids": [b["relation_id"] for b in original_bindings]}
    output["relations"] = [{"id": "bypass", "kind": "CFG", "sources": ["a"], "targets": ["b"], "bindings": [binding], "attributes": {}}]
    return output


def test_cfg_only_contracts_confirmed_completely_removed_pass_through_nodes():
    original = safe_call_graph()
    output = bypass_output(original)
    assert validate_pair(original, output)["exact_match"]
    changed = deepcopy(output)
    changed["relations"][0]["bindings"][0]["attributes"]["path_labels"][1] = "wrong"
    report = validate_pair(original, changed)
    assert not report["exact_match"]
    assert report["relations"]["CFG"]["fp"] == report["relations"]["CFG"]["fn"] == 1
    changed = deepcopy(original)
    changed["nodes"][1]["attributes"]["pruning"]["rule2"]["output_only"] = False
    changed["dictionary"]["m"]["attributes"] = deepcopy(changed["nodes"][1]["attributes"])
    assert not validate_pair(changed, bypass_output(changed))["exact_match"]


def test_ddg_direct_relations_cannot_be_replaced_by_transitive_chain_shortcut():
    original = graph([node(key) for key in ("a", "b", "c")], [relation("ab", "a", "b"), relation("bc", "b", "c")])
    output = deepcopy(original)
    output["relations"] = [relation("ac", "a", "c")]
    report = validate_pair(original, output)
    assert report["relations"]["DDG"]["fp"] == 1
    assert report["relations"]["DDG"]["fn"] == 2
    assert not report["exact_match"]


def test_positive_length_reachability_includes_cycles_and_self_reachability():
    original = graph([node("a"), node("b")], [relation("ab", "a", "b", "CFG"), relation("ba", "b", "a", "CFG")])
    output = deepcopy(original)
    assert validate_pair(original, output)["relations"]["CFG"]["reachability_method"] == "edge_equality"
    output["relations"].pop(0)
    report = validate_pair(original, output)
    assert report["relations"]["CFG"]["reachability_method"] == "exhaustive_closure"
    assert report["relations"]["CFG"]["reachability_difference_count"] == 3
    assert ["a", "a"] in report["relations"]["CFG"]["reachability_difference_examples"]
    assert ["b", "b"] in report["relations"]["CFG"]["reachability_difference_examples"]


def test_graph_local_ids_are_never_pooled_before_micro_counts():
    first = graph([node("a"), node("b")], [relation("d", "a", "b")], graph_id="first")
    second = deepcopy(first)
    second["graph_id"] = "second"
    missing = deepcopy(second)
    missing["relations"] = []
    summary = aggregate_reports([validate_pair(first, first), validate_pair(second, missing)])
    assert summary["relations"]["DDG"]["tp"] == 1
    assert summary["relations"]["DDG"]["fn"] == 1
    assert summary["relations"]["DDG"]["recall"] == 0.5
    assert not validate_pair(first, second)["exact_match"]


def test_explicit_source_constants_are_checked_separately_from_raw_labels():
    original = graph([node("a", attributes={"explicit_constant": 0}), node("b"), node("c")],
                     [relation("ab", "a", "b"), relation("ac", "a", "c")])
    output = deepcopy(original)
    output["metadata"]["representation"] = "HCPG"
    bindings = [deepcopy(item["bindings"][0]) for item in original["relations"]]
    for binding in bindings:
        binding["source_constant"] = 0
    output["relations"] = [{"id": "h", "kind": "DATA", "sources": ["a"], "targets": ["b", "c"],
                            "bindings": bindings, "attributes": {"labels": ["x"], "source_annotations": [
                                {"source": "a", "entity": "entity:x", "constant": 0}]}}]
    assert validate_pair(original, output)["exact_match"]
    changed = deepcopy(output)
    changed["relations"][0]["bindings"][0]["source_constant"] = False
    assert not validate_pair(original, changed)["exact_match"]  # JSON false is not JSON 0
    changed = deepcopy(output)
    del changed["relations"][0]["bindings"][0]["source_constant"]
    assert not validate_pair(original, changed)["exact_match"]
    changed = deepcopy(output)
    changed["relations"][0]["attributes"]["source_annotations"] = []
    assert not validate_pair(original, changed)["exact_match"]
    changed = deepcopy(output)
    changed["relations"][0]["attributes"]["labels"] = ["x:0"]
    assert not validate_pair(original, changed)["exact_match"]


def test_invalid_merge_and_shared_logical_ports_cannot_create_a_hyperedge_star():
    attrs = {"declared_type": "int", "function_context": "f", "scope": "s"}
    original = graph([node("a", "LOCAL", "int a", {**attrs, "name": "a"}),
                      node("b", "LOCAL", "int b", {**attrs, "name": "b"}), node("x"), node("y")],
                     [relation("ax", "a", "x"), relation("by", "b", "y")])
    output = deepcopy(original)
    declarations = original["nodes"][:2]
    records = [{key: deepcopy(item[key]) for key in ("id", "kind", "code", "attributes")} for item in declarations]
    output["nodes"] = [{"id": "merged", "kind": "LOCAL", "code": "int a\nint b", "entries": ["a", "b"],
                        "attributes": {"declared_type": "int", "function_context": "f", "retained_contents": records}}, *output["nodes"][2:]]
    for oid in ("a", "b"):
        output["dictionary"][oid].update(status="merged", rule=4, representative="merged")
    output["relations"][0]["sources"] = ["merged"]
    output["relations"][1]["sources"] = ["merged"]
    assert validate_pair(original, output)["exact_match"]
    unsafe = deepcopy(output)
    unsafe["relations"] = [{"id": "h", "kind": "DATA", "sources": ["merged"], "targets": ["x", "y"],
                            "bindings": [deepcopy(r["bindings"][0]) for r in output["relations"]], "attributes": {"labels": ["x"]}}]
    report = validate_pair(original, unsafe)
    assert not report["exact_match"]
    assert any("mixes unrelated original entries" in msg for msg in report["errors"]["binding"])
    missing_content = deepcopy(output)
    missing_content["nodes"][0]["attributes"]["retained_contents"].pop()
    assert not validate_pair(original, missing_content)["exact_match"]


def test_retained_status_cannot_bypass_rule4_merge_restrictions():
    original = graph([node("a", code="same_call()"), node("b", code="same_call()"), node("d")],
                     [relation("bd", "b", "d")])
    output = deepcopy(original)
    output["nodes"] = [output["nodes"][0], output["nodes"][2]]
    output["nodes"][0]["entries"].append("b")
    output["dictionary"]["b"]["representative"] = "a"
    output["relations"][0]["sources"] = ["a"]
    report = validate_pair(original, output)
    assert not report["exact_match"]
    assert any("retained role b must keep" in message for message in report["errors"]["identity"])


def test_ast_and_cdg_reject_deletions_and_unsupported_replacements():
    for kind in ("AST", "CDG"):
        original = graph([node("a"), node("b")], [relation("edge", "a", "b", kind)])
        output = deepcopy(original)
        output["relations"] = []
        report = validate_pair(original, output)
        assert not report["exact_match"], kind
        assert report["structural_relations"][kind]["fn"] == 1
        assert report["errors"]["binding"]
        no_edge_original = graph([node("a"), node("b")])
        output = deepcopy(no_edge_original)
        output["relations"] = [relation("invented", "a", "b", kind)]
        report = validate_pair(no_edge_original, output)
        assert not report["exact_match"], kind
        assert report["structural_relations"][kind]["fp"] == 1
        assert any("no independently projected" in msg for msg in report["errors"]["binding"])


def test_ast_projection_preserves_every_original_path_annotation():
    original = safe_call_graph()
    for edge in original["relations"]:
        edge["kind"] = edge["bindings"][0]["kind"] = "AST"
    output = bypass_output(original)
    output["relations"][0]["kind"] = output["relations"][0]["bindings"][0]["kind"] = "AST"
    report = validate_pair(original, output)
    assert report["exact_match"], report["errors"]
    assert report["structural_relations"]["AST"]["reference_count"] == 1
    changed = deepcopy(output)
    changed["relations"][0]["bindings"][0]["attributes"]["path_entities"][1] = "wrong"
    assert not validate_pair(original, changed)["exact_match"]
    changed = deepcopy(output)
    changed["relations"][0]["bindings"][0]["path"] = []
    assert not validate_pair(original, changed)["exact_match"]


def test_ast_path_transform_integration_keeps_independent_projection():
    from sp4cpg.graph import transform_graph
    original = safe_call_graph()
    for edge in original["relations"]:
        edge["kind"] = edge["bindings"][0]["kind"] = "AST"
    transformed = transform_graph(original)
    report = validate_pair(original, transformed)
    assert report["exact_match"], report["errors"]


def test_explicit_pruning_facts_in_dot_json_strings_and_alias_are_supported():
    import json
    from sp4cpg.graph import transform_graph

    for field in ("pruning", "sp4cpg_pruning"):
        original = safe_call_graph()
        facts = original["nodes"][1]["attributes"].pop("pruning")
        original["nodes"][1]["attributes"][field] = json.dumps(facts)
        original["dictionary"]["m"]["attributes"] = deepcopy(original["nodes"][1]["attributes"])
        report = validate_pair(original, transform_graph(original))
        assert report["exact_match"], (field, report["errors"])


def test_parallel_relation_identity_loss_fails_without_changing_logical_micro_counts():
    for kind in ("CFG", "DDG", "CDG"):
        original = graph([node("a"), node("b")], [relation("r1", "a", "b", kind), relation("r2", "a", "b", kind)])
        output = deepcopy(original)
        assert validate_pair(original, output)["exact_match"]
        output["relations"].pop()
        output["metadata"]["history"] = deepcopy(original["relations"])
        report = validate_pair(original, output)
        assert not report["exact_match"], kind
        counts = report["relations" if kind in ("CFG", "DDG") else "structural_relations"][kind]
        assert (counts["tp"], counts["fp"], counts["fn"]) == (1, 0, 0)
        assert report["relation_identity_coverage"][kind]["missing_original_relation_ids"] == ["r2"]
        # Raw logical tuples agree, but a different relation's ID cannot cover r2.
        wrong_id = deepcopy(original)
        wrong_id["relations"][1]["bindings"][0]["relation_id"] = "r1"
        assert not validate_pair(original, wrong_id)["exact_match"]


def test_cfg_original_relation_id_coverage_comes_from_active_validated_path_only():
    original = safe_call_graph()
    output = bypass_output(original)
    report = validate_pair(original, output)
    assert report["exact_match"]
    assert report["relation_identity_coverage"]["CFG"]["covered_original_relation_count"] == 2
    output["relations"] = []
    output["metadata"]["history"] = [bypass_output(original)["relations"][0]]
    report = validate_pair(original, output)
    assert report["relation_identity_coverage"]["CFG"]["missing_original_relation_ids"] == ["am", "mb"]
    assert not report["exact_match"]


def test_cli_reports_only_computed_results_and_rejects_unmatched_graph_ids(tmp_path):
    import json
    from scripts.validate_dependencies import main
    from scripts.graph_statistics import main as statistics_main

    original = graph([node("a"), node("b")], [relation("d", "a", "b")])
    original_path, output_path, report_path = (tmp_path / name for name in ("cpg.json", "hcpg.json", "report.json"))
    original_path.write_text(json.dumps(original), encoding="utf-8")
    output_path.write_text(json.dumps(original), encoding="utf-8")
    assert main(["--original", str(original_path), "--transformed", str(output_path), "--output", str(report_path)]) == 0
    result = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["provenance"] == "computed_from_supplied_original_and_active_output_graphs"
    assert result["aggregate"]["exact_match_count"] == 1
    assert statistics_main(["--input", str(original_path), "--output", str(tmp_path / "statistics.json")]) == 0
    original["graph_id"] = "unrelated"
    output_path.write_text(json.dumps(original), encoding="utf-8")
    assert main(["--original", str(original_path), "--transformed", str(output_path), "--output", str(report_path)]) == 1
    result = json.loads(report_path.read_text(encoding="utf-8"))
    assert not result["aggregate"]["all_exact_match"]
    assert len(result["input_errors"]) == 2


def test_directory_clis_ignore_only_known_construction_reports(tmp_path):
    import json
    import pytest
    from scripts.validate_dependencies import load_collection, main
    from scripts.graph_statistics import main as statistics_main

    cpg_dir, hcpg_dir = tmp_path / "cpg", tmp_path / "hcpg"
    cpg_dir.mkdir()
    hcpg_dir.mkdir()
    original = graph([node("a"), node("b")], [relation("d", "a", "b")])
    for directory in (cpg_dir, hcpg_dir):
        (directory / "sample.json").write_text(json.dumps(original), encoding="utf-8")
        (directory / "construction_report.json").write_text('{"success": 1}', encoding="utf-8")
        (directory / "sample.construction_report.json").write_text("This is a sidecar, not a graph", encoding="utf-8")
    report_path = tmp_path / "validation.json"
    assert main(["--original", str(cpg_dir), "--transformed", str(hcpg_dir), "--output", str(report_path)]) == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["aggregate"]["graph_count"] == 1
    assert statistics_main(["--input", str(hcpg_dir), "--output", str(tmp_path / "stats.json")]) == 0
    # Corrupt graph assets must fail instead of silently reducing the sample set.
    (hcpg_dir / "broken.json").write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError):
        load_collection(hcpg_dir)
    assert main(["--original", str(cpg_dir), "--transformed", str(hcpg_dir), "--output", str(report_path)]) == 1
    with pytest.raises(SystemExit) as failure:
        statistics_main(["--input", str(hcpg_dir), "--output", str(tmp_path / "stats.json")])
    assert failure.value.code == 1
    (hcpg_dir / "broken.json").unlink()
    (hcpg_dir / "sample.json").unlink()
    with pytest.raises(ValueError, match="No JSON/DOT graphs"):
        load_collection(hcpg_dir)
    with pytest.raises(ValueError, match="Construction report is not a graph"):
        load_collection(hcpg_dir / "construction_report.json")
