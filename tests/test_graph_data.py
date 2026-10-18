import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from preprocess.graph_data import (
    EDGE_TYPE_MAP, NODE_KINDS, NODE_TYPE_MAP, GraphData, graph_to_data,
)
from preprocess.parse_dot import CodeGraphEncoder, parse_dot_file_enhanced, parse_dot_to_data
from preprocess.dot_embedding import HCPGDataset, label_for, main as embedding_main


def graph_fixture(sample_id="entry-7"):
    nodes = [{"id": str(i), "kind": "CALL", "code": f"f{i}()", "entries": [str(i)],
              "attributes": {"position": i}} for i in range(3)]
    return {
        "schema_version": 2, "graph_id": sample_id, "nodes": nodes,
        "relations": [{"id": "data-0", "kind": "DATA", "sources": ["0", "1"],
                       "targets": ["2"], "attributes": {"control_context": "loop"},
                       "bindings": [{"source": source, "target": "2", "kind": "DDG",
                                     "label": f"DDG: v{source}", "entity": f"var:{source}",
                                     "context": "loop", "attributes": {"constant": "2"},
                                     "path": [], "relation_id": f"input-{source}"}
                                    for source in ("0", "1")]}],
        "dictionary": {node["id"]: {"id": node["id"], "kind": node["kind"],
                                    "code": node["code"], "attributes": node["attributes"].copy(),
                                    "representative": node["id"], "carrier": None,
                                    "status": "retained", "rule": None} for node in nodes},
        "metadata": {"representation": "HCPG", "original_node_count": 3},
    }


class FixtureTokenizer:
    def __call__(self, texts, **kwargs):
        self.options = kwargs
        return {"input_ids": torch.tensor([[len(text)] for text in texts])}


class FixtureCodeBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(hidden_size=2)
        self.dropout = nn.Dropout(.9)

    def forward(self, input_ids):
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, 2) * self.weight
        return SimpleNamespace(last_hidden_state=self.dropout(x))


def encoder_fixture():
    return CodeGraphEncoder(tokenizer=FixtureTokenizer(), model=FixtureCodeBERT(), batch_size=2)


def test_binary_witness_and_incidence_use_different_index_spaces():
    graph = graph_fixture()
    data = graph_to_data(graph, torch.ones(3, 2), 1)
    assert isinstance(data, GraphData)
    assert data.__class__.__module__ == "preprocess.graph_data"
    assert data.schema_version == 2
    assert data.sample_id == "entry-7"
    assert data.edge_index.tolist() == [[0, 1], [2, 2]]
    assert data.hyperedge_index.tolist() == [[0, 1, 2], [0, 0, 0]]
    assert data.incidence_role.tolist() == [0, 0, 1]
    assert data.relation_types.tolist() == [5]
    assert data.y.tolist() == [1]
    assert data.original_node_count == 3
    assert data.num_hyperedges == data.n_hyperedges == 1


def test_batch_offsets_follow_node_and_relation_counts_and_restore_metadata():
    first = graph_to_data(graph_fixture("first"), torch.ones(3, 2), 1)
    graph = graph_fixture("second")
    extra = copy.deepcopy(graph["relations"][0])
    extra["id"] = "data-1"
    graph["relations"].append(extra)
    second = graph_to_data(graph, torch.ones(3, 2), 0)
    batch = Batch.from_data_list([first, second])
    assert batch.hyperedge_index.tolist() == [[0, 1, 2, 3, 4, 5, 3, 4, 5],
                                              [0, 0, 0, 1, 1, 1, 2, 2, 2]]
    assert batch.edge_index.tolist() == [[0, 1, 3, 4], [2, 2, 5, 5]]
    assert batch.graph_json == [first.graph_json, second.graph_json]
    restored = batch.to_data_list()
    assert restored[1].sample_id == "second"
    assert restored[1].graph_json == second.graph_json
    torch.testing.assert_close(restored[1].hyperedge_index, second.hyperedge_index)
    assert second.num_hyperedges == 2


def test_all_logical_identity_content_and_binding_metadata_survives():
    from preprocess.graph_core import normalize_graph
    graph = graph_fixture()
    graph["dictionary"]["0"]["attributes"]["lexical_scope"] = "function/block"
    graph["nodes"][0]["attributes"]["lexical_scope"] = "function/block"
    data = graph_to_data(graph, torch.ones(3, 2))
    assert json.loads(data.graph_json) == normalize_graph(graph)
    assert data.canonical_json == data.graph_json
    binding = json.loads(data.graph_json)["relations"][0]["bindings"][0]
    assert binding["context"] == "loop"
    assert binding["attributes"]["constant"] == "2"
    assert binding["entity"].startswith("var:")


def test_feature_rows_remain_attached_to_their_nodes_after_normalization():
    graph = graph_fixture()
    graph["nodes"] = list(reversed(graph["nodes"]))
    features = torch.tensor([[20.], [10.], [0.]])
    data = graph_to_data(graph, features)
    encoded_ids = [node["id"] for node in json.loads(data.graph_json)["nodes"]]
    assert dict(zip(encoded_ids, data.x[:, 0].tolist())) == {"0": 0., "1": 10., "2": 20.}


def test_carried_binding_uses_active_carrier_and_retains_original_identity():
    graph = graph_fixture()
    graph["nodes"][0]["code"] = graph["dictionary"]["0"]["code"] = "value + 1"
    graph["nodes"][0]["entries"].append("old")
    graph["dictionary"]["old"] = {"id": "old", "kind": "IDENTIFIER", "code": "value",
                                  "attributes": {"expression": "f0()"}, "representative": None,
                                  "carrier": "0", "carrier_original": "0", "status": "carried", "rule": 1}
    graph["nodes"][0]["attributes"]["retained_contents"] = [
        {key: graph["dictionary"]["old"][key] for key in ("id", "kind", "code", "attributes")}]
    graph["relations"][0]["bindings"][0]["source"] = "old"
    graph["metadata"]["original_node_count"] = 4
    data = graph_to_data(graph, torch.ones(3, 2))
    assert data.edge_index.tolist() == [[0, 1], [2, 2]]
    assert data.original_node_count == 4
    assert json.loads(data.graph_json)["dictionary"]["old"]["carrier"] == "0"


def test_canonical_type_counts_and_all_six_relation_kinds_reach_tensors():
    assert len(NODE_KINDS) == len(NODE_TYPE_MAP) == 17
    assert "<operator>.assignment" not in NODE_TYPE_MAP
    graph = graph_fixture()
    graph["relations"] = []
    for kind in EDGE_TYPE_MAP:
        binary_kind = {"CTRL": "CFG", "DATA": "DDG"}.get(kind, kind)
        sources = ["0"] if kind != "DATA" else ["0", "1"]
        targets = ["2"] if kind != "CTRL" else ["1", "2"]
        graph["relations"].append({"id": kind, "kind": kind, "sources": sources,
                                   "targets": targets, "attributes": {}, "bindings": [
                                       {"source": source, "target": target, "kind": binary_kind,
                                        "label": kind, "entity": "x", "context": "block",
                                        "attributes": {}, "path": [], "relation_id": kind}
                                       for source in sources for target in targets]})
    data = graph_to_data(graph, torch.ones(3, 2))
    assert sorted(data.relation_types.tolist()) == list(range(6))
    assert data.num_hyperedges == 6


@pytest.mark.parametrize("change,match", [
    (lambda g: g.update(schema_version=1), "schema_version"),
    (lambda g: g["nodes"][0].update(kind="UNKNOWN"), "content/kind mismatch"),
    (lambda g: g["relations"][0].update(kind="DDG"), "singleton"),
    (lambda g: g["relations"][0].update(bindings=[]), "binding witnesses"),
    (lambda g: g["relations"][0]["bindings"][0].update(kind="CFG"), "Binding kind"),
    (lambda g: g["dictionary"]["0"].update(representative="1"), "Retained entry.*identity"),
    (lambda g: g["metadata"].update(original_node_count=999), "dictionary size"),
])
def test_invalid_graphs_raise_instead_of_returning_an_empty_sample(change, match):
    graph = graph_fixture()
    change(graph)
    with pytest.raises(ValueError, match=match):
        graph_to_data(graph, torch.ones(3, 2))


def test_unwitnessed_pairs_and_mixed_logical_ports_are_rejected():
    graph = graph_fixture()
    graph["relations"][0]["bindings"].pop()
    with pytest.raises(ValueError, match="unwitnessed physical pair"):
        graph_to_data(graph, torch.ones(3, 2))
    graph = graph_fixture()
    graph["nodes"][0].update(id="merged-node", kind="LOCAL", code="int a;\nint b;", entries=["0", "old"],
                               attributes={"declared_type": "int", "function_context": "f"})
    graph["dictionary"]["0"].update(status="merged", representative="merged-node", kind="LOCAL", code="int a;",
                                    attributes={"declared_type": "int", "function_context": "f", "name": "a", "scope": "f"})
    graph["dictionary"]["old"] = dict(graph["dictionary"]["0"], id="old", code="int b;",
                                       attributes={"declared_type": "int", "function_context": "f", "name": "b", "scope": "f"})
    graph["nodes"][0]["attributes"]["retained_contents"] = [
        {key: graph["dictionary"][entry_id][key] for key in ("id", "kind", "code", "attributes")}
        for entry_id in ("0", "old")]
    graph["relations"][0]["sources"] = ["merged-node", "1"]
    graph["metadata"]["original_node_count"] = 4
    other = dict(graph["relations"][0]["bindings"][0], source="old")
    graph["relations"][0]["bindings"].append(other)
    with pytest.raises(ValueError, match="logical endpoint ports"):
        graph_to_data(graph, torch.ones(3, 2))


def test_empty_and_nonfinite_feature_inputs_are_explicit_errors():
    graph = graph_fixture()
    with pytest.raises(ValueError, match="Features"):
        graph_to_data(graph, torch.ones(2, 2))
    with pytest.raises(ValueError, match="NaN"):
        graph_to_data(graph, torch.full((3, 2), float("nan")))
    graph["nodes"] = []
    with pytest.raises(ValueError, match="nonempty"):
        graph_to_data(graph, torch.empty(0, 2))


def test_codebert_features_are_text_only_eval_frozen_and_reproducible():
    encoder = encoder_fixture()
    encoder.train()
    first = encoder.encode_code_text(["abc", "x", "hello"])
    second = encoder.encode_code_text(["abc", "x", "hello"])
    torch.testing.assert_close(first, torch.tensor([[3., 3.], [1., 1.], [5., 5.]]))
    torch.testing.assert_close(first, second)
    assert not encoder.codebert.training
    assert not first.requires_grad
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert not any("embedding" in name for name, _ in encoder.named_parameters())
    assert encoder.encode_code_text([]).shape == (0, 2)
    assert encoder.tokenizer.options["max_length"] == 128


def test_parser_entrypoints_accept_canonical_graphs_and_propagate_errors():
    encoder = encoder_fixture()
    graph = graph_fixture()
    first = parse_dot_file_enhanced(graph, 1, encoder)
    second = parse_dot_to_data(graph, 1, encoder)
    torch.testing.assert_close(first.x, second.x)
    assert first.graph_json == second.graph_json
    class FixtureUnavailableEncoder:
        def encode_code_text(self, _):
            raise RuntimeError("encoder unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        parse_dot_file_enhanced(graph, 1, FixtureUnavailableEncoder())


def test_parser_accepts_path_and_pydot_and_preserves_graph_id(tmp_path):
    import pydot
    from preprocess.graph_core import dump_graph
    graph = graph_fixture("stable-id")
    path = tmp_path / "labelled-hcpg-1.dot"
    dump_graph(graph, path)
    first = parse_dot_file_enhanced(path, 1, encoder_fixture())
    dot = pydot.graph_from_dot_data(path.read_text(encoding="utf-8"))[0]
    second = parse_dot_file_enhanced(dot, 1, encoder_fixture())
    assert first.sample_id == second.sample_id == "stable-id"


def test_labels_follow_stable_ids_and_malformed_inputs_do_not_shift_them(tmp_path):
    from preprocess.graph_core import dump_graph
    dump_graph(graph_fixture("first"), tmp_path / "a.json")
    dump_graph(graph_fixture("last"), tmp_path / "z.json")
    dataset = HCPGDataset(dot_dir=tmp_path, encoder=encoder_fixture(), labels={"first": 1, "last": 0})
    samples = dataset.encode_dot_to_feature()
    assert [(sample.sample_id, sample.y.item()) for sample in samples] == [("first", 1), ("last", 0)]
    (tmp_path / "broken.dot").write_text("not DOT", encoding="utf-8")
    dataset = HCPGDataset(dot_dir=tmp_path, encoder=encoder_fixture(), labels={"first": 1, "last": 0})
    with pytest.raises(ValueError, match="broken.dot"):
        dataset.encode_dot_to_feature()


def test_conflicting_or_missing_labels_raise():
    graph = graph_fixture()
    assert label_for(graph, "7-hcpg-1.dot", {}) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        label_for(graph, "7-hcpg-1.dot", {graph["graph_id"]: 0})
    with pytest.raises(ValueError, match="No binary label"):
        label_for(graph, "a.dot", {})


def test_cli_saves_a_list_and_refuses_existing_output(tmp_path, monkeypatch):
    from preprocess.graph_core import dump_graph
    from preprocess import dot_embedding
    graph = graph_fixture()
    graph["metadata"]["label"] = 1
    path = tmp_path / "input.json"
    dump_graph(graph, path)
    output = tmp_path / "hcpg_dataset.pkl"
    monkeypatch.setattr(dot_embedding, "CodeGraphEncoder", lambda *args, **kwargs: encoder_fixture())
    assert embedding_main(["--input", str(path), "--output", str(output)]) == 0
    saved = torch.load(output)
    assert isinstance(saved, list) and len(saved) == 1
    assert isinstance(saved[0], GraphData) and saved[0].schema_version == 2
    assert saved[0].sample_id == graph["graph_id"]
    before = output.read_bytes()
    with pytest.raises(SystemExit) as failure:
        embedding_main(["--input", str(path), "--output", str(output)])
    assert failure.value.code == 1
    assert output.read_bytes() == before


def test_graphdata_metadata_is_distinct_from_unversioned_data():
    current = graph_to_data(graph_fixture(), torch.ones(3, 2))
    ordinary = Data(x=torch.ones(3, 2), edge_index=torch.tensor([[0], [1]]))
    assert current.schema_version == 2
    assert not hasattr(ordinary, "schema_version")
    assert hasattr(current, "hyperedge_index")


def test_absent_dictionary_or_bindings_cannot_be_filled_during_data_conversion():
    graph = graph_fixture()
    del graph["dictionary"]
    with pytest.raises(ValueError, match="dictionary is required"):
        graph_to_data(graph, torch.ones(3, 2))
    graph = graph_fixture()
    del graph["relations"][0]["bindings"]
    with pytest.raises(ValueError, match="binding witnesses"):
        graph_to_data(graph, torch.ones(3, 2))


def test_label_manifest_inside_graph_directory_is_not_encoded_as_a_graph(tmp_path):
    from preprocess.graph_core import dump_graph
    dump_graph(graph_fixture(), tmp_path / "input.json")
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"entry-7": 1}), encoding="utf-8")
    dataset = HCPGDataset(dot_dir=tmp_path, encoder=encoder_fixture(), labels=labels_path)
    samples = dataset.encode_dot_to_feature()
    assert len(samples) == 1
    assert samples[0].y.item() == 1


def test_dictionary_cannot_hide_a_merge_as_a_retained_identity():
    graph = graph_fixture()
    graph["dictionary"]["1"]["representative"] = "0"
    graph["nodes"][0]["entries"].append("1")
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] != "1"]
    graph["relations"][0]["sources"] = ["0"]
    with pytest.raises(ValueError, match="Retained entry.*identity"):
        graph_to_data(graph, torch.ones(2, 2))
    graph["dictionary"]["1"]["status"] = "merged"
    with pytest.raises(ValueError, match="fresh representative"):
        graph_to_data(graph, torch.ones(2, 2))


def test_invalid_graph_format_is_checked_before_encoder_construction(tmp_path, monkeypatch):
    from preprocess import dot_embedding
    from preprocess.graph_core import dump_graph
    graph = graph_fixture()
    graph["nodes"][0]["kind"] = "UNKNOWN"
    graph["dictionary"]["0"]["kind"] = "UNKNOWN"
    graph["metadata"]["label"] = 1
    dump_graph(graph, tmp_path / "input.json")
    def unavailable(*args, **kwargs):
        raise AssertionError("Encoder must not be constructed")
    monkeypatch.setattr(dot_embedding, "CodeGraphEncoder", unavailable)
    dataset = HCPGDataset(dot_dir=tmp_path)
    with pytest.raises(ValueError, match="Unsupported canonical node kind"):
        dataset.encode_dot_to_feature()


def test_joern_type_aliases_share_categories_without_changing_graph_metadata():
    for kind, category in (("METHOD_PARAMETER_IN", "PARAM"), ("METHOD_PARAMETER_OUT", "PARAM"),
                           ("PARAMETER", "PARAM"), ("TYPE_DECL", "CLASS")):
        graph = graph_fixture()
        graph["nodes"][0]["kind"] = kind
        graph["dictionary"]["0"]["kind"] = kind
        data = graph_to_data(graph, torch.ones(3, 2))
        assert data.node_types[0].item() == NODE_TYPE_MAP[category]
        assert json.loads(data.graph_json)["dictionary"]["0"]["kind"] == kind


def test_constructed_directory_with_nested_graphs_and_reports_can_be_encoded(tmp_path):
    from preprocess.graph_core import dump_graph
    from preprocess.hcpg_generate import run
    inputs, outputs = tmp_path / "cpg", tmp_path / "hcpg"
    for subdir, sample_id, label in (("one", "a", 1), ("two", "b", 0)):
        graph = graph_fixture(sample_id)
        graph["metadata"].update(representation="CPG", label=label)
        bindings = graph["relations"][0]["bindings"]
        graph["relations"] = [{"id": f"ddg-{index}", "kind": "DDG",
                               "sources": [binding["source"]], "targets": [binding["target"]],
                               "bindings": [binding], "attributes": {}}
                              for index, binding in enumerate(bindings)]
        dump_graph(graph, inputs / subdir / f"{sample_id}-cpg-{label}.dot")
    result = run(inputs, outputs)
    assert result["succeeded"] == 2 and result["failed"] == 0
    assert (outputs / "construction_report.json").is_file()
    (outputs / "one" / "a-hcpg-1.construction_report.json").write_text('{}', encoding="utf-8")
    dataset = HCPGDataset(input_path=outputs, encoder=encoder_fixture())
    samples = dataset.encode_dot_to_feature()
    assert [(sample.sample_id, sample.y.item()) for sample in samples] == [("a", 1), ("b", 0)]
    assert all(sample.representation == "HCPG" for sample in samples)
    (outputs / "unexpected.json").write_text('{}', encoding="utf-8")
    dataset = HCPGDataset(input_path=outputs, encoder=encoder_fixture())
    with pytest.raises(ValueError, match="unexpected.json"):
        dataset.encode_dot_to_feature()


@pytest.mark.parametrize("change,match", [
    (lambda g: g["nodes"][0].update(code="different_retained_text()"), "content/kind mismatch"),
    (lambda g: g["dictionary"]["0"].update(kind="PARAM"), "content/kind mismatch"),
    (lambda g: g["dictionary"]["0"]["attributes"].update(position=False), "attributes mismatch"),
])
def test_retained_content_kind_and_json_attribute_types_must_agree(change, match):
    graph = graph_fixture()
    change(graph)
    with pytest.raises(ValueError, match=match):
        graph_to_data(graph, torch.ones(3, 2))


def test_core_carried_content_is_required_and_cannot_be_replaced():
    from preprocess.graph_core import normalize_graph, transform_graph
    original = normalize_graph({
        "schema_version": 2, "graph_id": "carrier-graph", "nodes": [
            {"id": "expression", "kind": "CALL", "code": "buf[2]", "attributes": {}},
            {"id": "size", "kind": "LITERAL", "code": "2", "attributes": {}}],
        "relations": [{"id": "ast", "kind": "AST", "sources": ["expression"], "targets": ["size"],
                       "attributes": {}}],
    })
    transformed = transform_graph(original)
    data = graph_to_data(transformed, torch.ones(len(transformed["nodes"]), 2))
    assert json.loads(data.graph_json)["dictionary"]["size"]["status"] == "carried"
    changed = copy.deepcopy(transformed)
    changed["nodes"][0]["attributes"]["retained_contents"][0]["code"] = "20"
    with pytest.raises(ValueError, match="Missing carried content record"):
        graph_to_data(changed, torch.ones(len(changed["nodes"]), 2))
    changed = copy.deepcopy(transformed)
    changed["dictionary"]["size"]["carrier_original"] = "lost"
    with pytest.raises(ValueError, match="dictionary entry"):
        graph_to_data(changed, torch.ones(len(changed["nodes"]), 2))


def test_core_merged_kind_type_context_and_original_content_must_agree():
    from preprocess.graph_core import normalize_graph, transform_graph
    original = normalize_graph({
        "schema_version": 2, "graph_id": "declarations", "relations": [], "nodes": [
            {"id": str(index), "kind": "LOCAL", "code": f"int v{index};", "attributes": {
                "declared_type": "int", "function_context": "f", "name": f"v{index}", "scope": "f/block"}}
            for index in (1, 2)],
    })
    transformed = transform_graph(original)
    assert len(transformed["nodes"]) == 1
    graph_to_data(transformed, torch.ones(1, 2))
    for field, value, error in (("kind", "PARAM", "common kind"),
                                 ("declared_type", "long", "declared_type"),
                                 ("function_context", "other", "function_context")):
        changed = copy.deepcopy(transformed)
        if field == "kind":
            changed["dictionary"]["1"][field] = value
        else:
            changed["dictionary"]["1"]["attributes"][field] = value
        with pytest.raises(ValueError, match=error):
            graph_to_data(changed, torch.ones(1, 2))
    changed = copy.deepcopy(transformed)
    changed["nodes"][0]["attributes"]["retained_contents"] = []
    with pytest.raises(ValueError, match="Missing merged content record"):
        graph_to_data(changed, torch.ones(1, 2))


def test_merged_scope_zero_is_known_and_unknown_markers_are_rejected():
    from preprocess.graph_core import normalize_graph, transform_graph
    original = normalize_graph({
        "schema_version": 2, "graph_id": "known-context", "relations": [], "nodes": [
            {"id": str(index), "kind": "LOCAL", "code": f"int v{index};", "attributes": {
                "declared_type": "int", "function_context": "f", "name": f"v{index}", "scope": 0}}
            for index in (1, 2)],
    })
    transformed = transform_graph(original)
    assert len(transformed["nodes"]) == 1
    graph_to_data(transformed, torch.ones(1, 2))
    for field, value in (("declared_type", " unknown "), ("declared_type", " ANY "),
                          ("function_context", "<UNKNOWN>"), ("scope", "?"), ("name", "UNKNOWN")):
        changed = copy.deepcopy(transformed)
        if field in {"declared_type", "function_context"}:
            changed["nodes"][0]["attributes"][field] = value
            for entry in changed["dictionary"].values():
                entry["attributes"][field] = value
        else:
            changed["dictionary"]["1"]["attributes"][field] = value
        with pytest.raises(ValueError, match="requires known|declaration/occurrence identity"):
            graph_to_data(changed, torch.ones(1, 2))
