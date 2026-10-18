from copy import deepcopy
import json
from pathlib import Path

import pytest

from preprocess.graph_core import dump_graph, from_pydot, load_graph, normalize_graph, transform_graph


def n(oid, code="work()", kind="CALL", **attrs):
    return {"id": str(oid), "kind": kind, "code": code, "attributes": attrs}


def e(oid, source, target, kind="DDG", label=None, entity="", context="", **attrs):
    return {"id": str(oid), "kind": kind, "sources": [str(source)], "targets": [str(target)], "attributes": {},
            "bindings": [{"source": str(source), "target": str(target), "kind": kind,
                          "label": kind + ": " if label is None else label, "entity": entity, "context": context,
                          "attributes": attrs, "path": []}]}


def g(nodes, edges):
    return normalize_graph({"graph_id": "sample", "nodes": nodes, "relations": edges})


def constant_call(oid):
    return n(oid, 'printf("ready")', name="printf", pruning={"rule2": {
        "call_name": "printf", "format": "ready", "constant_output": True, "result_unused": True,
        "output_only": True, "arguments_valid": True, "control_context_preserved": True,
        "vulnerability_relevant": False}})


def allocation_graph(extra=()):
    return g([n(1, "char buf[2]", "BLOCK"), n(2, "buf[2]", operation="<operator>.assignment"),
              n(3, "buf[2]", operation="<operator>.alloc"), n(4, "use(buf)"), n(5, "before()")],
             [e("a", 1, 2, "AST", label="AST: declaration"), e("b", 2, 3, "AST", label="AST: allocation"), *extra])


def all_bindings(graph):
    return [binding for relation in graph["relations"] for binding in relation["bindings"]]


def test_rule1_carries_dependency_roles_and_only_bypasses_ast():
    original = allocation_graph([e("cfg-in", 5, 2, "CFG", context="ctx"), e("cfg-out", 2, 4, "CFG", context="ctx"),
                                 e("ddg-out", 2, 4, label="DDG: buf", entity="buf", role="definition"),
                                 e("ddg-in", 5, 2, label="DDG: size", entity="size"), e("cdg", 5, 2, "CDG", context="true")])
    before = deepcopy(original)
    result = transform_graph(original)
    assert original == before
    entry = result["dictionary"]["2"]
    assert entry["status"] == "carried" and entry["carrier"] == "3" and entry["rule"] == 1
    assert entry["kind"] == "CALL" and entry["attributes"]["operation"] == "<operator>.assignment"
    for kind in ("CFG", "DDG", "CDG"):
        expected = [b for b in all_bindings(original) if b["kind"] == kind]
        actual = [b for b in all_bindings(result) if b["kind"] == kind]
        assert sorted(actual, key=lambda b: b["relation_id"]) == sorted(expected, key=lambda b: b["relation_id"])
    outgoing = next(r for r in result["relations"] if r["id"] == "ddg-out")
    assert outgoing["sources"] == ["3"] and outgoing["bindings"][0]["source"] == "2"
    assert not any(r["kind"] == "CFG" and r["sources"] == ["5"] and r["targets"] == ["4"] for r in result["relations"])
    ast = next(r for r in result["relations"] if r["kind"] == "AST")
    assert ast["sources"] == ["1"] and ast["targets"] == ["3"]
    assert ast["bindings"][0]["path"] == ["2"]
    assert ast["bindings"][0]["attributes"]["path_labels"] == ["AST: declaration", "AST: allocation"]


def test_rule1_keeps_a_logical_dependency_between_roles_sharing_a_carrier():
    result = transform_graph(allocation_graph([e("data", 2, 3, label="DDG: buf")]))
    relation = next(r for r in result["relations"] if r["id"] == "data")
    assert relation["sources"] == relation["targets"] == ["3"]
    assert (relation["bindings"][0]["source"], relation["bindings"][0]["target"]) == ("2", "3")


@pytest.mark.parametrize("part,code", [("1", "char buf[8]"), ("3", "buf[3]")])
def test_rule1_requires_the_same_content_in_both_related_expressions(part, code):
    original = allocation_graph()
    for record in (next(node for node in original["nodes"] if node["id"] == part), original["dictionary"][part]):
        record["code"] = code
    assert transform_graph(original)["dictionary"]["2"]["status"] == "retained"


def test_rule1_ambiguous_allocation_carriers_are_retained():
    original = allocation_graph()
    original["nodes"].append(n(6, "buf[2]", operation="<operator>.alloc"))
    original["relations"].append(e("c", 2, 6, "AST"))
    del original["dictionary"]
    assert transform_graph(original)["dictionary"]["2"]["status"] == "retained"


def test_unknown_call_conditions_are_retained_regardless_of_target():
    for name in ("printf", "fprintf", "sprintf", "snprintf", "vsnprintf"):
        original = g([n(1, name + '("ready")', name=name)], [])
        original["metadata"]["label"] = 0
        assert transform_graph(original)["dictionary"]["1"]["status"] == "retained"


@pytest.mark.parametrize("changes", [{"format": "%n"}, {"format": "%s"}, {"call_name": "snprintf"},
                                      {"result_unused": False}, {"output_only": None},
                                      {"arguments_valid": False}, {"vulnerability_relevant": True},
                                      {"control_context_preserved": None}])
def test_rule2_requires_all_call_conditions(changes):
    call = constant_call(2)
    call["attributes"]["pruning"]["rule2"].update(changes)
    original = g([n(1), call, n(3)], [e("a", 1, 2, "CFG", context="ctx"), e("b", 2, 3, "CFG", context="ctx")])
    assert transform_graph(original)["dictionary"]["2"]["status"] == "retained"


def test_rule2_rejects_a_conflicting_recorded_call_name():
    call = constant_call(1)
    call["attributes"]["name"] = "sprintf"
    assert transform_graph(g([call], []))["dictionary"]["1"]["status"] == "retained"


@pytest.mark.parametrize("kind", ["DDG", "CDG"])
def test_rule2_independent_dependencies_prevent_removal(kind):
    original = g([constant_call(1), n(2)], [e("dependency", 1, 2, kind)])
    assert transform_graph(original)["dictionary"]["1"]["status"] == "retained"


def test_rule2_cfg_paths_keep_every_annotation_and_flatten_multiple_removals():
    original = g([n(1), constant_call(2), constant_call(3), n(4)], [
        e("a", 1, 2, "CFG", label="CFG: before", entity="a", context="ctx", branch="enter"),
        e("b", 2, 3, "CFG", label="CFG: between", entity="b", context="ctx", branch="stay"),
        e("c", 3, 4, "CFG", label="CFG: after", entity="c", context="ctx", branch="leave")])
    result = transform_graph(original)
    binding = result["relations"][0]["bindings"][0]
    assert (binding["source"], binding["target"], binding["path"]) == ("1", "4", ["2", "3"])
    assert binding["label"] == "CFG: before" and binding["entity"] == "a"
    assert binding["attributes"]["path_relation_ids"] == ["a", "b", "c"]
    assert binding["attributes"]["path_labels"] == ["CFG: before", "CFG: between", "CFG: after"]
    assert binding["attributes"]["path_attributes"] == [{"branch": "enter"}, {"branch": "stay"}, {"branch": "leave"}]
    assert result["dictionary"]["2"]["status"] == result["dictionary"]["3"]["status"] == "removed"


def test_ast_paths_flatten_independently_of_cfg_context():
    original = g([n(1, "block", "BLOCK"), constant_call(2), constant_call(3), n(4, "ready", "LITERAL")], [
        e("a", 1, 2, "AST", label="AST: first", context="one", position=0),
        e("b", 2, 3, "AST", label="AST: second", context="two", position=1),
        e("c", 3, 4, "AST", label="AST: third", context="three", position=2)])
    result = transform_graph(original)
    binding = result["relations"][0]["bindings"][0]
    assert binding["kind"] == "AST" and binding["path"] == ["2", "3"]
    assert binding["attributes"]["path_contexts"] == ["one", "two", "three"]
    assert binding["attributes"]["path_attributes"] == [{"position": 0}, {"position": 1}, {"position": 2}]


@pytest.mark.parametrize("contexts", [("", ""), ("a", "b"), ("UNKNOWN", "UNKNOWN"),
                                     ("<UNKNOWN>", "<UNKNOWN>"), ("?", "?"), ("{}", "{}")])
def test_rule2_unknown_or_incompatible_cfg_context_is_retained(contexts):
    original = g([n(1), constant_call(2), n(3)], [e("a", 1, 2, "CFG", context=contexts[0]), e("b", 2, 3, "CFG", context=contexts[1])])
    assert transform_graph(original)["dictionary"]["2"]["status"] == "retained"


def test_comment_removal_requires_no_independent_role():
    original = g([n(1, "block", "BLOCK"), n(2, "comment", "COMMENT"), n(3)], [e("ast", 1, 2, "AST")])
    assert transform_graph(original)["dictionary"]["2"]["status"] == "removed"
    original["relations"].append(e("data", 2, 3))
    assert transform_graph(original)["dictionary"]["2"]["status"] == "retained"


def test_rule3_checks_original_dependencies_and_actual_parent():
    original = g([n(1, "buf[2]"), n(2, "2", "LITERAL"), n(3)], [e("ast", 1, 2, "AST"), e("data", 2, 3)])
    assert transform_graph(original)["dictionary"]["2"]["status"] == "retained"
    original = g([n(1, "buf[2]"), n(2, "2", "LITERAL")], [e("ast", 1, 2, "AST")])
    result = transform_graph(original)
    assert result["dictionary"]["2"]["carrier"] == "1"
    assert result["dictionary"]["2"]["carrier_original"] == "1"
    assert result["nodes"][0]["attributes"]["retained_contents"][0]["code"] == "2"
    original = g([n(1, "other()"), n(2, "2", "LITERAL"), n(3, "buf[2]")], [e("ast", 1, 2, "AST")])
    assert transform_graph(original)["dictionary"]["2"]["status"] == "retained"


def test_rule3_does_not_discard_identifier_bindings():
    original = g([n(1, "use(x)"), n(2, "x", "IDENTIFIER", symbol_id="declaration")], [e("ast", 1, 2, "AST")])
    assert transform_graph(original)["dictionary"]["2"]["status"] == "retained"


@pytest.mark.parametrize("carrier", ['print("x")', "use(foo_x)", "f(/* x */ y)", 'f(R"tag(inner "x" text)tag")'])
def test_identifier_leaf_requires_an_actual_identifier_token(carrier):
    original = g([n(1, carrier), n(2, "x", "IDENTIFIER")], [e("ast", 1, 2, "AST")])
    assert transform_graph(original)["dictionary"]["2"]["status"] == "retained"


def test_identifier_leaf_in_a_retained_expression_is_carried():
    original = g([n(1, "use(x)"), n(2, "x", "IDENTIFIER")], [e("ast", 1, 2, "AST")])
    assert transform_graph(original)["dictionary"]["2"]["status"] == "carried"


@pytest.mark.parametrize("carrier", ["buf[20]", "x=12", "x=0x2", "x=2.0", "x=2U", "x=a2", "x=开2", 's="2"', "x=2e3", "x=.2"])
def test_numeric_leaf_requires_an_entire_literal_token(carrier):
    original = g([n(1, carrier), n(2, "2", "LITERAL")], [e("ast", 1, 2, "AST")])
    assert transform_graph(original)["dictionary"]["2"]["status"] == "retained"


@pytest.mark.parametrize("literal,carrier", [("2", "buf[2]"), ("2", "a+2"), ("2", "-2"), ("2", "f(2)"),
                                           ("-2", "f(-2)"), ('"ready"', 'puts("ready")'), ("'2'", "f('2')")])
def test_leaf_values_survive_in_real_expression_tokens(literal, carrier):
    original = g([n(1, carrier), n(2, literal, "LITERAL")], [e("ast", 1, 2, "AST")])
    assert transform_graph(original)["dictionary"]["2"]["status"] == "carried"


def test_rule4_updates_an_existing_leaf_carrier_and_preserves_entries():
    attrs = {"declared_type": "int[2]", "function_context": "f", "scope": "s"}
    original = g([n(1, "int a[2]", "LOCAL", name="a", **attrs), n(2, "2", "LITERAL"),
                  n(3, "int b[2]", "LOCAL", name="b", **attrs)], [e("ast", 1, 2, "AST")])
    result = transform_graph(original)
    representative = result["dictionary"]["1"]["representative"]
    assert representative == result["dictionary"]["3"]["representative"]
    assert representative not in original["dictionary"]
    assert result["dictionary"]["2"]["carrier"] == representative
    assert result["dictionary"]["2"]["carrier_original"] == "1"
    assert {record["id"] for record in result["nodes"][0]["attributes"]["retained_contents"]} == {"1", "2", "3"}


def test_rule4_unknown_scope_type_or_function_is_retained():
    for attrs in ({"declared_type": "int"}, {"declared_type": "UNKNOWN", "function_context": "f", "scope": "s"},
                  {"declared_type": "int", "function_context": "f"}):
        original = g([n(1, "int a", "LOCAL", name="a", **attrs), n(2, "int b", "LOCAL", name="b", **attrs)], [])
        assert len(transform_graph(original)["nodes"]) == 2


def test_rule4_does_not_cross_kinds_or_functions():
    attrs = {"declared_type": "int", "scope": "s"}
    original = g([n(1, "int a", "LOCAL", name="a", function_context="f", **attrs),
                  n(2, "int b", "PARAM", name="b", function_context="f", **attrs),
                  n(3, "int c", "LOCAL", name="c", function_context="g", **attrs)], [])
    assert len(transform_graph(original)["nodes"]) == 3


@pytest.mark.parametrize("field", ["declared_type", "function_context", "name", "scope"])
@pytest.mark.parametrize("value", ["UNKNOWN", "<UNKNOWN>", "?", "   "])
def test_rule4_unresolved_identity_fields_prevent_sharing(field, value):
    nodes = [n(1, "int a", "LOCAL", declared_type="int", function_context="f", name="a", scope="s"),
             n(2, "int b", "LOCAL", declared_type="int", function_context="f", name="b", scope="s")]
    for node in nodes:
        node["attributes"][field] = value
    assert len(transform_graph(g(nodes, []))["nodes"]) == 2


def test_rule4_numeric_zero_scope_is_known_and_occurrence_unknown_is_not():
    attrs = {"declared_type": "int", "function_context": "f", "scope": 0}
    original = g([n(1, "int a", "LOCAL", name="a", **attrs), n(2, "int b", "LOCAL", name="b", **attrs)], [])
    assert len(transform_graph(original)["nodes"]) == 1
    original = g([n(1, "int", "TYPE_REF", declared_type="int", function_context="f", occurrence="UNKNOWN"),
                  n(2, "int", "TYPE_REF", declared_type="int", function_context="f", occurrence="UNKNOWN")], [])
    assert len(transform_graph(original)["nodes"]) == 2


def test_merging_never_creates_a_star_between_different_logical_ports():
    attrs = {"declared_type": "int", "function_context": "f", "scope": "s"}
    original = g([n(1, "int a", "LOCAL", name="a", **attrs), n(2, "int b", "LOCAL", name="b", **attrs), n(3), n(4)],
                 [e("a", 1, 3, label="DDG: a"), e("b", 2, 4, label="DDG: b")])
    result = transform_graph(original)
    assert result["dictionary"]["1"]["representative"] == result["dictionary"]["2"]["representative"]
    assert {relation["kind"] for relation in result["relations"]} == {"DDG"}
    assert {(b["source"], b["target"]) for b in all_bindings(result)} == {("1", "3"), ("2", "4")}


def test_mm_source_patterns_keep_constants_separate_from_raw_labels():
    original = g([n(1, explicit_constant=0), n(2, explicit_constant=False), n(3), n(4)], [
        e(str(source) + str(target), source, target, label="DDG: v" + str(source), entity="v" + str(source))
        for source in (1, 2) for target in (3, 4)])
    result = transform_graph(original)
    assert len(result["relations"]) == 1
    relation = result["relations"][0]
    assert relation["kind"] == "DATA" and relation["attributes"]["mode"] == "MM"
    assert len(relation["bindings"]) == 4
    assert relation["attributes"]["labels"] == ["DDG: v1", "DDG: v2"]
    constants = {b["source"]: json.dumps(b["source_constant"]) for b in relation["bindings"]}
    assert constants == {"1": "0", "2": "false"}


def test_sm_and_ms_follow_mm_without_inventing_a_missing_pair():
    original = g([n(i) for i in range(1, 7)], [e("a", 1, 2), e("b", 1, 3), e("c", 4, 6), e("d", 5, 6)])
    assert {r["attributes"]["mode"] for r in transform_graph(original)["relations"]} == {"SM", "MS"}
    original = g([n(i) for i in range(1, 5)], [e("a", 1, 3), e("b", 2, 3), e("c", 1, 4)])
    result = transform_graph(original)
    assert not any(r["attributes"].get("mode") == "MM" for r in result["relations"])
    assert {(b["source"], b["target"]) for b in all_bindings(result)} == {("1", "3"), ("2", "3"), ("1", "4")}


def test_equal_memberships_can_have_distinct_bound_hyperedge_identities():
    attrs = {"declared_type": "int", "function_context": "f"}
    original = g([n(1), n(2), n(3, "int", "TYPE_REF", occurrence="o3", **attrs), n(4, "int", "TYPE_REF", occurrence="o4", **attrs)],
                 [e(str(source) + str(target), source, target, label="DDG: x") for source in (1, 2) for target in (3, 4)])
    result = transform_graph(original)
    relations = result["relations"]
    assert len(relations) == 2 and {r["kind"] for r in relations} == {"DATA"}
    assert relations[0]["sources"] == relations[1]["sources"] and relations[0]["targets"] == relations[1]["targets"]
    assert relations[0]["id"] != relations[1]["id"]
    assert {frozenset(b["target"] for b in r["bindings"]) for r in relations} == {frozenset({"3"}), frozenset({"4"})}


def test_parallel_edges_keep_distinct_relation_ids_and_labels():
    original = g([n(1), n(2)], [e("a", 1, 2, label="DDG: x"), e("b", 1, 2, label="DDG: x")])
    result = transform_graph(original)
    assert {r["id"] for r in result["relations"]} == {"a", "b"}
    assert {b["relation_id"] for b in all_bindings(result)} == {"a", "b"}
    original = g([n(1), n(2), n(3)], [e("a", 1, 2, label="DDG: x"), e("b", 1, 2, label="DDG: y"), e("c", 1, 3, label="DDG: z")])
    assert {b["label"] for b in all_bindings(transform_graph(original))} == {"DDG: x", "DDG: y", "DDG: z"}


def test_cfg_groups_require_known_compatible_control_guards():
    original = g([n(i) for i in range(1, 5)], [e("a", 1, 3, "CFG", context="a"), e("b", 2, 3, "CFG", context="a"),
                                               e("c", 1, 4, "CFG", context="b"), e("d", 2, 4, "CFG", context="b")])
    result = transform_graph(original)
    assert len(result["relations"]) == 2
    assert {r["attributes"]["control_guard"]["context"] for r in result["relations"]} == {"a", "b"}
    unknown = g([n(i) for i in range(1, 4)], [e("a", 1, 3, "CFG"), e("b", 2, 3, "CFG")])
    assert {r["kind"] for r in transform_graph(unknown)["relations"]} == {"CFG"}


@pytest.mark.parametrize("context", ["UNKNOWN", "<UNKNOWN>", "unknown", "?", "{}", "[]", "None"])
def test_unresolved_control_contexts_cannot_form_hyperedges(context):
    original = g([n(i) for i in range(1, 4)], [e("a", 1, 3, "CFG", context=context), e("b", 2, 3, "CFG", context=context)])
    assert {r["kind"] for r in transform_graph(original)["relations"]} == {"CFG"}


def test_explicit_cdg_guards_group_cfg_without_changing_raw_context():
    original = g([n(i) for i in range(1, 5)], [e("cfg1", 1, 3, "CFG"), e("cfg2", 2, 3, "CFG"),
        e("cdg1", 4, 1, "CDG", branch=True), e("cdg2", 4, 2, "CDG", branch=True)])
    result = transform_graph(original)
    relation = next(r for r in result["relations"] if r["kind"] == "CTRL")
    assert relation["attributes"]["control_guard"] == {"cdg": [{"controller": "4", "condition": True}]}
    assert relation["attributes"]["guard_relation_ids"] == ["cdg1", "cdg2"]
    assert all(b["context"] == "" for b in relation["bindings"])
    original["relations"][1]["bindings"][0]["attributes"]["branch"] = False
    assert not any(r["kind"] == "CTRL" for r in transform_graph(original)["relations"])


def test_construction_is_deterministic_under_input_permutation():
    original = g([n(i) for i in range(1, 5)], [e(str(s) + str(t), s, t) for s in (1, 2) for t in (3, 4)])
    reordered = deepcopy(original)
    reordered["nodes"].reverse()
    reordered["relations"].reverse()
    reordered["dictionary"] = dict(reversed(list(reordered["dictionary"].items())))
    assert transform_graph(original) == transform_graph(reordered)


def test_json_and_dot_roundtrips_preserve_all_dictionary_and_binding_fields(tmp_path):
    result = transform_graph(allocation_graph([e("dep", 2, 4, label='DDG: {x:"a,b"}', entity="buf", annotation=[1, False])]))
    for suffix in ("dot", "json"):
        path = tmp_path / ("sample." + suffix)
        dump_graph(result, path)
        assert load_graph(path) == result


def test_legacy_dot_imports_call_roles_occurrences_and_parallel_labels(tmp_path):
    source = tmp_path / "7-cpg-1.dot"
    source.write_text('digraph g { 1 [label=<&lt;operator&gt;.assignment, 2<BR/>x=1>]; '
                      '2 [label=<printf, 3<BR/>printf("x")>]; 3 [label=<IDENTIFIER, 4<BR/>x<BR/>use(x)>]; '
                      '1 -> 2 [label="DDG: x"]; 1 -> 2 [label="DDG: y"]; }', encoding="utf-8")
    result = load_graph(source)
    assert result["graph_id"] == "7" and result["metadata"]["label"] == 1
    assert result["nodes"][0]["kind"] == "CALL" and result["nodes"][0]["attributes"]["operation"] == "<operator>.assignment"
    assert result["nodes"][1]["kind"] == "CALL" and result["nodes"][1]["attributes"]["name"] == "printf"
    assert result["nodes"][2]["code"] == "x" and result["nodes"][2]["attributes"]["occurrence_context"] == ["use(x)"]
    assert {b["label"] for b in all_bindings(result)} == {"DDG: x", "DDG: y"}


def test_legacy_unbound_hyperedges_are_not_expanded():
    import pydot
    source = pydot.graph_from_dot_data('digraph g { 1 [label="CALL"]; 2 [label="CALL"]; 3 [label="CALL"]; "1,2" -> 3 [label="DDG: x"]; }')[0]
    graph = from_pydot(source)
    assert not graph["metadata"]["binding_complete"] and graph["relations"][0]["bindings"] == []
    with pytest.raises(ValueError, match="memberships"):
        transform_graph(graph)


def test_legacy_parameter_alias_keeps_its_original_node_kind():
    import pydot
    source = pydot.graph_from_dot_data('digraph g { 1 [label=<PARAMETER, 1<BR/>int value>]; }')[0]
    graph = from_pydot(source)
    assert graph["nodes"][0]["kind"] == "PARAMETER"
    assert graph["dictionary"]["1"]["kind"] == "PARAMETER"


def test_original_binding_dictionary_and_types_cannot_be_replaced():
    original = g([n(1, flag=0), n(2)], [e("a", 1, 2)])
    changed = deepcopy(original)
    changed["dictionary"]["1"]["attributes"]["flag"] = False
    with pytest.raises(ValueError, match="identities"):
        transform_graph(changed)
    changed = deepcopy(original)
    changed["relations"][0]["bindings"][0]["source"] = "2"
    with pytest.raises(ValueError, match="bindings"):
        transform_graph(changed)


def test_serial_and_parallel_use_the_same_core_with_ordered_failures(tmp_path):
    from preprocess.hcpg_generate import run
    source = tmp_path / "input"
    source.mkdir()
    dump_graph(g([n(1), n(2)], [e("a", 1, 2)]), source / "b.json")
    (source / "a.json").write_text("invalid", encoding="utf-8")
    serial = run(source, tmp_path / "serial")
    parallel = run(source, tmp_path / "parallel", workers=2, executor="thread")
    assert (serial["succeeded"], serial["failed"]) == (1, 1)
    assert [Path(row["input"]).name for row in serial["files"]] == ["a.json", "b.json"]
    assert [row["status"] for row in serial["files"]] == [row["status"] for row in parallel["files"]]
    assert load_graph(tmp_path / "serial" / "b.dot") == load_graph(tmp_path / "parallel" / "b.dot")


def test_default_dot_filename_is_compatible_with_dataset_labels(tmp_path):
    from preprocess.hcpg_generate import run
    source = tmp_path / "input"
    source.mkdir()
    dump_graph(g([n(1), n(2)], [e("a", 1, 2)]), source / "7-cpg-1.dot")
    result = run(source, tmp_path / "output")
    assert result["succeeded"] == 1 and (tmp_path / "output" / "7-hcpg-1.dot").exists()
