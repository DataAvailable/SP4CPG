from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from sp4cpg.graph import dump_graph, from_pydot, load_graph, normalize_graph, transform_graph


def node(node_id, kind="CALL", code="work()", **attributes):
    return {"id": str(node_id), "kind": kind, "code": code, "attributes": attributes}


def edge(relation_id, source, target, kind="DDG", label="DDG: x", context="", entity="", **attributes):
    return {"id": str(relation_id), "kind": kind, "sources": [str(source)], "targets": [str(target)],
            "attributes": {}, "bindings": [{"source": str(source), "target": str(target), "kind": kind,
                                              "label": label, "context": context, "entity": entity,
                                              "attributes": attributes, "path": []}]}


def graph(nodes, relations):
    return normalize_graph({"graph_id": "sample", "nodes": nodes, "relations": relations})


def safe_call(node_id):
    return node(node_id, code='printf("hello")', name="printf", pruning={"rule2": {
        "call_name": "printf", "format": "hello", "constant_output": True,
        "result_unused": True, "output_only": True, "arguments_valid": True,
        "vulnerability_relevant": False, "control_context_preserved": True}})


class GraphTests(unittest.TestCase):
    def test_parallel_relations_and_labels_are_not_overwritten(self):
        original = graph([node(i) for i in (1, 2, 3)], [
            edge("x", 1, 2, label="DDG: x", entity="x"),
            edge("y", 1, 2, label="DDG: y", entity="y"),
            edge("z", 1, 3, label="DDG: z", entity="z")])
        before = deepcopy(original)
        result = transform_graph(original)
        self.assertEqual(original, before)
        bindings = [b for r in result["relations"] for b in r["bindings"]]
        self.assertEqual({b["label"] for b in bindings}, {"DDG: x", "DDG: y", "DDG: z"})
        self.assertEqual(len(bindings), 3)
        self.assertEqual(result["relations"][0]["attributes"]["mode"], "SM")

    def test_mm_precedes_stars_and_keeps_complete_union(self):
        original = graph([node(i) for i in range(1, 5)], [
            edge(f"e{s}{t}", s, t, label=f"DDG: v{s}", entity=f"v{s}")
            for s in (1, 2) for t in (3, 4)])
        result = transform_graph(original)
        self.assertEqual(len(result["relations"]), 1)
        relation = result["relations"][0]
        self.assertEqual(relation["attributes"]["mode"], "MM")
        self.assertEqual(relation["sources"], ["1", "2"])
        self.assertEqual(relation["targets"], ["3", "4"])
        self.assertEqual(len(relation["bindings"]), 4)

    def test_sm_ms_cardinalities_and_partial_biclique(self):
        original = graph([node(i) for i in range(1, 7)], [
            edge("a", 1, 2), edge("b", 1, 3), edge("c", 4, 6), edge("d", 5, 6)])
        result = transform_graph(original)
        self.assertEqual({r["attributes"]["mode"] for r in result["relations"]}, {"SM", "MS"})
        # A missing 2->4 witness must never be filled in by MM construction.
        incomplete = graph([node(i) for i in range(1, 5)], [edge("a", 1, 3), edge("b", 2, 3), edge("c", 1, 4)])
        result = transform_graph(incomplete)
        self.assertFalse(any(r["attributes"].get("mode") == "MM" for r in result["relations"]))
        self.assertEqual({(b["source"], b["target"]) for r in result["relations"] for b in r["bindings"]},
                         {("1", "3"), ("2", "3"), ("1", "4")})

    def test_deterministic_under_permutation_and_parallel_anonymous_edges(self):
        original = graph([node(i) for i in range(1, 5)], [edge(f"e{s}{t}", s, t) for s in (1, 2) for t in (3, 4)])
        permuted = deepcopy(original)
        permuted["nodes"].reverse()
        permuted["relations"].reverse()
        self.assertEqual(transform_graph(original), transform_graph(permuted))
        anonymous = {"nodes": [node(1), node(2)], "relations": [
            {"kind": "DDG", "sources": ["1"], "targets": ["2"], "label": "x"},
            {"kind": "DDG", "sources": ["1"], "targets": ["2"], "label": "x"}]}
        normalized = normalize_graph(anonymous)
        self.assertEqual(len({r["id"] for r in normalized["relations"]}), 2)

    def test_cfg_requires_recorded_compatible_context(self):
        unknown = graph([node(i) for i in (1, 2, 3)], [edge("a", 1, 3, "CFG"), edge("b", 2, 3, "CFG")])
        self.assertEqual([r["kind"] for r in transform_graph(unknown)["relations"]], ["CFG", "CFG"])
        known = graph([node(i) for i in (1, 2, 3)], [edge("a", 1, 3, "CFG", context="branch-A"), edge("b", 2, 3, "CFG", context="branch-A")])
        self.assertEqual(transform_graph(known)["relations"][0]["kind"], "CTRL")

    def test_leaf_uses_original_dependencies_and_actual_parent_content(self):
        original = graph([node(1, code="buf[2]"), node(2, "LITERAL", "2"), node(3)], [
            edge("ast", 1, 2, "AST", label="AST: "), edge("ddg", 2, 3)])
        self.assertIn("2", {n["id"] for n in transform_graph(original)["nodes"]})
        clean = graph([node(1, code="buf[2]"), node(2, "LITERAL", "2")], [edge("ast", 1, 2, "AST")])
        result = transform_graph(clean)
        self.assertEqual(result["dictionary"]["2"]["status"], "carried")
        self.assertEqual(result["dictionary"]["2"]["carrier"], "1")
        self.assertEqual(result["nodes"][0]["attributes"]["retained_contents"][0]["code"], "2")
        unrelated = graph([node(1, code="other"), node(2, "LITERAL", "2"), node(3, code="buf[2]")], [edge("ast", 1, 2, "AST")])
        self.assertIn("2", {n["id"] for n in transform_graph(unrelated)["nodes"]})

    def test_independent_identifier_binding_prevents_leaf_removal(self):
        original = graph([node(1, code="use(x)"), node(2, "IDENTIFIER", "x", symbol_id="decl-x")], [edge("ast", 1, 2, "AST")])
        self.assertEqual(len(transform_graph(original)["nodes"]), 2)

    def test_rule2_never_uses_call_name_or_target_label_as_proof(self):
        for name in ("printf", "sprintf", "snprintf", "vsnprintf"):
            original = graph([node(1, code=f'{name}("constant")', name=name)], [])
            original["metadata"]["target"] = 0
            self.assertEqual(transform_graph(original)["dictionary"]["1"]["status"], "retained")

    def test_rule2_boundary_conditions(self):
        for change in ({"call_name": "sprintf"}, {"format": "%n"}, {"format": "%s"},
                       {"arguments_valid": False}, {"vulnerability_relevant": True},
                       {"output_only": False}, {"result_unused": None}):
            call = safe_call(2)
            call["attributes"]["pruning"]["rule2"].update(change)
            original = graph([node(1), call, node(3)], [edge("a", 1, 2, "CFG", context="ctx"), edge("b", 2, 3, "CFG", context="ctx")])
            self.assertEqual(transform_graph(original)["dictionary"]["2"]["status"], "retained", change)
        mismatch = safe_call(2)
        mismatch["attributes"]["name"] = "snprintf"
        self.assertEqual(transform_graph(graph([mismatch], []))["dictionary"]["2"]["status"], "retained")

    def test_safe_cfg_bypass_preserves_raw_path_annotations(self):
        original = graph([node(1), safe_call(2), node(3)], [
            edge("a", 1, 2, "CFG", label="CFG: before", context="ctx", entity="first", branch="true"),
            edge("b", 2, 3, "CFG", label="CFG: after", context="ctx", entity="second", branch="exit")])
        result = transform_graph(original)
        self.assertEqual(result["dictionary"]["2"]["status"], "removed")
        binding = result["relations"][0]["bindings"][0]
        self.assertEqual((binding["source"], binding["target"], binding["path"]), ("1", "3", ["2"]))
        self.assertEqual(binding["label"], "CFG: before")
        self.assertEqual(binding["attributes"]["path_labels"], ["CFG: before", "CFG: after"])
        self.assertEqual(binding["attributes"]["path_attributes"], [{"branch": "true"}, {"branch": "exit"}])
        changed = deepcopy(original)
        changed["relations"][1]["bindings"][0]["context"] = "other"
        self.assertEqual(transform_graph(changed)["dictionary"]["2"]["status"], "retained")

    def test_cfg_chain_flattens_witness_records(self):
        original = graph([node(1), safe_call(2), safe_call(3), node(4)], [
            edge("a", 1, 2, "CFG", context="ctx"), edge("b", 2, 3, "CFG", context="ctx"), edge("c", 3, 4, "CFG", context="ctx")])
        binding = transform_graph(original)["relations"][0]["bindings"][0]
        self.assertEqual(binding["path"], ["2", "3"])
        self.assertEqual(binding["attributes"]["path_relation_ids"], ["a", "b", "c"])
        self.assertEqual(binding["attributes"]["path_attributes"], [{}, {}, {}])

    def test_ast_reconnection_flattens_complete_original_paths(self):
        original = graph([node(1, "BLOCK", "block"), safe_call(2), safe_call(3), node(4, "LITERAL", "hello")], [
            edge("a", 1, 2, "AST", label="AST: first", context="one", position=0),
            edge("b", 2, 3, "AST", label="AST: second", context="two", position=1),
            edge("c", 3, 4, "AST", label="AST: third", context="three", position=2)])
        result = transform_graph(original)
        self.assertEqual([r["kind"] for r in result["relations"]], ["AST"])
        binding = result["relations"][0]["bindings"][0]
        self.assertEqual((binding["source"], binding["target"], binding["path"]), ("1", "4", ["2", "3"]))
        self.assertEqual(binding["label"], "AST: first")
        self.assertEqual(binding["attributes"]["path_relation_ids"], ["a", "b", "c"])
        self.assertEqual(binding["attributes"]["path_labels"], ["AST: first", "AST: second", "AST: third"])
        self.assertEqual(binding["attributes"]["path_contexts"], ["one", "two", "three"])
        self.assertEqual(binding["attributes"]["path_attributes"], [{"position": 0}, {"position": 1}, {"position": 2}])

    def test_rule4_keeps_distinct_logical_entries_and_edges(self):
        attrs = {"declared_type": "char*", "function_context": "f", "scope": "block-1"}
        original = graph([node(1, "LOCAL", "char *a", name="a", **attrs),
                          node(2, "LOCAL", "char *b", name="b", **attrs), node(3), node(4)],
                         [edge("a", 1, 3), edge("b", 2, 4)])
        result = transform_graph(original)
        representative = result["dictionary"]["1"]["representative"]
        self.assertEqual(representative, result["dictionary"]["2"]["representative"])
        self.assertNotEqual(representative, "1")
        self.assertEqual(result["dictionary"]["1"]["attributes"], original["dictionary"]["1"]["attributes"])
        self.assertEqual({r["kind"] for r in result["relations"]}, {"DDG"})
        self.assertEqual({(b["source"], b["target"]) for r in result["relations"] for b in r["bindings"]}, {("1", "3"), ("2", "4")})

    def test_rule4_unknown_identity_or_other_context_remains_separate(self):
        for nodes in ([node(1, "LOCAL", "int a", declared_type="int"), node(2, "LOCAL", "int b", declared_type="int")],
                      [node(1, "LOCAL", "int a", declared_type="int", function_context="f", name="a", scope="s"),
                       node(2, "LOCAL", "int b", declared_type="int", function_context="g", name="b", scope="s")]):
            self.assertEqual(len(transform_graph(graph(nodes, []))["nodes"]), 2)

    def test_rule1_requires_explicit_redundancy_and_content_carrier(self):
        original = graph([node(1, "BLOCK", "x=1"), node(2, "<operator>.assignment", "x=1", pruning={"rule1": {"redundant": True, "carrier": "1"}}),
                          node(3, "<operator>.alloc", "x=1")], [edge("a", 1, 2, "AST"), edge("b", 2, 3, "AST")])
        result = transform_graph(original)
        self.assertEqual(result["dictionary"]["2"]["status"], "carried")
        self.assertEqual(result["dictionary"]["2"]["carrier"], "1")
        unknown = deepcopy(original)
        unknown["nodes"][1]["attributes"] = {}
        unknown["dictionary"]["2"]["attributes"] = {}
        self.assertIn("2", {n["id"] for n in transform_graph(unknown)["nodes"]})

    def test_explicit_constants_remain_separate_from_raw_ddg_labels(self):
        original = graph([node(1, code="b=5", explicit_constant=5), node(2), node(3)], [
            edge("a", 1, 2, label="DDG: b", entity="b"), edge("b", 1, 3, label="DDG: b", entity="b")])
        relation = transform_graph(original)["relations"][0]
        self.assertEqual(relation["attributes"]["labels"], ["DDG: b"])
        self.assertEqual(relation["attributes"]["source_annotations"], [{"source": "1", "entity": "b", "constant": 5}])
        self.assertTrue(all(b["label"] == "DDG: b" and b["source_constant"] == 5 for b in relation["bindings"]))
        inferred = graph([node(1, code="b=5"), node(2), node(3)], [edge("a", 1, 2), edge("b", 1, 3)])
        self.assertTrue(all("source_constant" not in b for r in transform_graph(inferred)["relations"] for b in r["bindings"]))

    def test_inconsistent_input_binding_or_dictionary_fails(self):
        original = graph([node(1), node(2)], [edge("a", 1, 2)])
        original["relations"][0]["bindings"][0]["source"] = "2"
        with self.assertRaisesRegex(ValueError, "binding disagrees"):
            transform_graph(original)
        with self.assertRaisesRegex(ValueError, "requires nodes and relations"):
            normalize_graph({"aggregate": {"exact_match": True}})

    def test_json_and_dot_roundtrip_is_lossless(self):
        original = graph([node(1, code='f("a,b")'), node(2), node(3)], [edge("a", 1, 2, label='DDG: {x:"a,b"}'), edge("b", 1, 3)])
        result = transform_graph(original)
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in ("json", "dot"):
                path = Path(tmp) / ("sample." + suffix)
                dump_graph(result, path)
                self.assertEqual(load_graph(path), result)

    def test_legacy_binary_dot_keeps_parallel_edges_and_unbound_hyperedges_fail(self):
        import pydot
        dot = pydot.graph_from_dot_data('digraph g { 1 [label=<CALL, 1<BR/>f()>]; 2 [label=<CALL, 2<BR/>g()>]; 1 -> 2 [label="DDG: x"]; 1 -> 2 [label="DDG: y"]; }')[0]
        loaded = from_pydot(dot, "g")
        self.assertEqual({b["label"] for r in loaded["relations"] for b in r["bindings"]}, {"DDG: x", "DDG: y"})
        legacy = from_pydot(pydot.graph_from_dot_data('digraph g { 1 [label="CALL"]; 2 [label="CALL"]; 3 [label="CALL"]; "1,2" -> 3 [label="DDG: x"]; }')[0])
        self.assertFalse(legacy["metadata"]["provenance_complete"])
        with self.assertRaisesRegex(ValueError, "without endpoint bindings"):
            transform_graph(legacy)

    def test_joern_named_calls_and_occurrence_context_are_explicitly_imported(self):
        import pydot
        source = 'digraph g { 1 [label=<printf, 2<BR/>printf("hello")>]; 2 [label=<IDENTIFIER, 3<BR/>x<BR/>use(x)>]; }'
        loaded = from_pydot(pydot.graph_from_dot_data(source)[0], "g")
        self.assertEqual(loaded["nodes"][0]["kind"], "CALL")
        self.assertEqual(loaded["nodes"][0]["attributes"]["name"], "printf")
        self.assertEqual(loaded["nodes"][1]["code"], "x")
        self.assertEqual(loaded["nodes"][1]["attributes"]["legacy_occurrence_context"], ["use(x)"])
        # Knowing a name is insufficient evidence to remove a call.
        self.assertEqual(transform_graph(loaded)["dictionary"]["1"]["status"], "retained")

    def test_directory_cli_ordered_failures_and_serial_parallel_equivalence(self):
        from preprocess.hcpg_generate import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "original"
            source.mkdir()
            original = graph([node(1), node(2)], [edge("a", 1, 2)])
            dump_graph(original, source / "b.json")
            (source / "a.json").write_text("not JSON", encoding="utf-8")
            serial = run(source, root / "serial")
            parallel = run(source, root / "parallel", workers=2, executor="thread")
            self.assertEqual((serial["success"], serial["failed"]), (1, 1))
            self.assertEqual([Path(r["input"]).name for r in serial["reports"]], ["a.json", "b.json"])
            self.assertEqual(load_graph(root / "serial" / "b.json"), load_graph(root / "parallel" / "b.json"))
            self.assertEqual([r["status"] for r in serial["reports"]], [r["status"] for r in parallel["reports"]])


if __name__ == "__main__":
    unittest.main()
