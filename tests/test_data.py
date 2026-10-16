import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch

from sp4cpg.data import GraphData, NODE_KINDS, RELATION_KINDS, graph_to_data, node_kind_index
from preprocess.parse_dot import CodeGraphEncoder, node_input_text, parse_dot_file_enhanced
from preprocess.dot_embedding import embed_files, label_for, load_labels


def canonical(sample_id="stable-007"):
    nodes = [{"id": str(i), "kind": "CALL", "code": f"f{i}()", "attributes": {},
              "entries": [str(i)]} for i in range(3)]
    return {
        "schema_version": 2, "graph_id": sample_id, "nodes": nodes,
        "relations": [{"id": "r0", "kind": "DATA", "sources": ["0", "1"],
                       "targets": ["2"], "bindings": [
                           {"source": "0", "target": "2", "kind": "DDG", "label": "x"},
                           {"source": "1", "target": "2", "kind": "DDG", "label": "y"}],
                       "attributes": {}}],
        "dictionary": {n["id"]: dict(n, representative=n["id"], carrier=None,
                                      status="retained", rule=None) for n in nodes},
        "metadata": {"representation": "HCPG", "original_node_count": 3},
    }


def test_graph_to_data_preserves_stable_ids_types_and_two_index_spaces():
    data = graph_to_data(canonical(), torch.arange(6.).reshape(3, 2), 1)
    assert isinstance(data, GraphData)
    assert data.sample_id == "stable-007"
    assert data.original_node_count == 3
    assert data.schema_version == 2
    assert data.representation == "HCPG"
    assert data.num_hyperedges == 1
    assert data.y.tolist() == [1]
    assert data.relation_types.tolist() == [5]
    assert data.edge_index.tolist() == [[0, 1], [2, 2]]
    assert data.hyperedge_index.tolist() == [[0, 1, 2], [0, 0, 0]]
    assert data.incidence_role.tolist() == [0, 0, 1]


def test_independent_relation_offsets_and_batch_roundtrip():
    first = graph_to_data(canonical("first"), torch.ones(3, 2), 1)
    graph = canonical("second")
    duplicate = copy.deepcopy(graph["relations"][0])
    duplicate["id"] = "r1"
    graph["relations"].append(duplicate)
    second = graph_to_data(graph, torch.ones(3, 2), 0)
    batched = Batch.from_data_list([first, second])
    assert batched.hyperedge_index.tolist() == [[0, 1, 2, 3, 4, 5, 3, 4, 5],
                                                [0, 0, 0, 1, 1, 1, 2, 2, 2]]
    assert batched.edge_index.tolist() == [[0, 1, 3, 4], [2, 2, 5, 5]]
    restored = batched.to_data_list()
    assert restored[1].sample_id == "second"
    torch.testing.assert_close(restored[1].hyperedge_index, second.hyperedge_index)
    assert second.num_hyperedges == 2, "identical memberships remain distinct relations"


def test_incomplete_hyperedge_pairs_are_rejected_instead_of_fabricated():
    graph = canonical()
    node = {"id": "3", "kind": "CALL", "code": "f3()", "attributes": {}, "entries": ["3"]}
    graph["nodes"].append(node)
    graph["dictionary"]["3"] = dict(node, representative="3", carrier=None, status="retained")
    graph["metadata"]["original_node_count"] = 4
    relation = graph["relations"][0]
    relation["targets"] = ["2", "3"]
    relation["bindings"][1]["target"] = "3"
    with pytest.raises(ValueError, match="Unwitnessed physical source-target pair"):
        graph_to_data(graph, torch.ones(4, 1))


def test_complete_many_to_many_witnesses_become_physical_binary_edges():
    graph = canonical()
    node = {"id": "3", "kind": "CALL", "code": "f3()", "attributes": {}, "entries": ["3"]}
    graph["nodes"].append(node)
    graph["dictionary"]["3"] = dict(node, representative="3", carrier=None, status="retained")
    graph["metadata"]["original_node_count"] = 4
    graph["relations"][0].update(targets=["2", "3"], bindings=[
        {"source": s, "target": t, "kind": "DDG", "label": f"{s}-{t}"}
        for s in ("0", "1") for t in ("2", "3")])
    data = graph_to_data(graph, torch.ones(4, 1))
    assert data.edge_index.tolist() == [[0, 0, 1, 1], [2, 3, 2, 3]]


def test_carried_endpoint_uses_membership_carrier_and_keeps_original_count():
    graph = canonical()
    graph["dictionary"]["old"] = {"id": "old", "representative": None, "carrier": "0",
                                  "status": "carried"}
    graph["nodes"][0]["entries"].append("old")
    graph["relations"][0]["bindings"][0]["source"] = "old"
    graph["metadata"]["original_node_count"] = 4
    data = graph_to_data(graph, torch.ones(3, 1))
    assert data.original_node_count == 4
    assert data.edge_index.tolist() == [[0, 1], [2, 2]]


@pytest.mark.parametrize("mutate,match", [
    (lambda g: g.update(schema_version=1), "schema_version"),
    (lambda g: g.update(graph_id=""), "graph_id"),
    (lambda g: g["nodes"][0].update(kind="UNKNOWN"), "Unsupported node kind"),
    (lambda g: g["relations"][0].update(bindings=[]), "no endpoint witness"),
    (lambda g: g["relations"][0].update(sources=["missing"]), "missing node"),
    (lambda g: g["relations"][0]["bindings"][0].update(source="lost"), "Missing dictionary"),
    (lambda g: g["relations"][0]["bindings"][0].update(kind="CFG"), "Binding kind"),
])
def test_malformed_data_fails_explicitly(mutate, match):
    graph = canonical()
    mutate(graph)
    with pytest.raises(ValueError, match=match):
        graph_to_data(graph, torch.ones(3, 2))


def test_feature_shape_and_empty_graph_errors_are_not_dummy_samples():
    with pytest.raises(ValueError, match="features"):
        graph_to_data(canonical(), torch.ones(2, 2))
    graph = canonical()
    graph["nodes"] = []
    with pytest.raises(ValueError, match="empty graph"):
        graph_to_data(graph, torch.empty(0, 2))
    with pytest.raises(ValueError, match="NaN"):
        graph_to_data(canonical(), torch.full((3, 2), float("nan")))


def test_paper_type_inventory_and_operator_aliases():
    assert len(NODE_KINDS) == 17
    assert RELATION_KINDS == ("AST", "CFG", "CDG", "DDG", "CTRL", "DATA")
    assert node_kind_index("<operator>.assignment") == node_kind_index("CALL")
    assert node_kind_index("METHOD_PARAMETER_IN") == node_kind_index("PARAM")


def test_nonhyperedge_relation_cannot_hide_multiple_memberships():
    graph = canonical()
    graph["relations"][0]["kind"] = "DDG"
    with pytest.raises(ValueError, match="Binary relation.*singleton"):
        graph_to_data(graph, torch.ones(3, 1))


def test_hyperedge_cannot_mix_original_entries_via_shared_representative():
    graph = canonical()
    graph["nodes"][0]["entries"].append("old")
    graph["dictionary"]["old"] = {"id": "old", "representative": "0", "carrier": None,
                                  "status": "merged"}
    graph["metadata"]["original_node_count"] = 4
    # All physical pairs are witnessed, but one source representative combines
    # incompatible original ports, which must not become a logical star.
    graph["relations"][0]["bindings"].append(
        {"source": "old", "target": "2", "kind": "DDG", "label": "z"})
    with pytest.raises(ValueError, match="mixes different original endpoint ports"):
        graph_to_data(graph, torch.ones(3, 1))


def test_inconsistent_dictionary_ownership_and_counts_are_rejected():
    graph = canonical()
    graph["dictionary"]["0"]["representative"] = "1"
    with pytest.raises(ValueError, match="ownership mismatch"):
        graph_to_data(graph, torch.ones(3, 1))
    graph = canonical()
    graph["metadata"]["original_node_count"] = 999
    with pytest.raises(ValueError, match="dictionary size"):
        graph_to_data(graph, torch.ones(3, 1))


class FakeTokenizer:
    def __call__(self, texts, **options):
        self.options = options
        return {"input_ids": torch.tensor([[len(t)] for t in texts], dtype=torch.long)}


class FakeCodeBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2, _commit_hash="offline-test-revision")
        self.weight = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(.9)

    def forward(self, input_ids):
        base = input_ids.float().unsqueeze(-1).repeat(1, 1, 2) * self.weight
        return SimpleNamespace(last_hidden_state=self.dropout(base))


def test_codebert_is_eval_text_only_frozen_and_deterministic_without_download():
    tokenizer = FakeTokenizer()
    encoder = CodeGraphEncoder(tokenizer=tokenizer, model=FakeCodeBERT(), batch_size=2)
    encoder.train()  # Even a caller toggling train cannot enable encoding dropout.
    first = encoder.encode_code_text(["abc", "z", "hello"])
    second = encoder.encode_code_text(["abc", "z", "hello"])
    torch.testing.assert_close(first, torch.tensor([[3., 3.], [1., 1.], [5., 5.]]))
    torch.testing.assert_close(first, second)
    assert not encoder.codebert.training
    assert not first.requires_grad
    assert all(not p.requires_grad for p in encoder.parameters())
    assert not any("embedding" in name for name, _ in encoder.named_parameters())
    assert encoder.metadata["resolved_revision"] == "offline-test-revision"
    assert tokenizer.options["max_length"] == 128
    assert encoder.encode_code_text([]).shape == (0, 2)
    assert node_input_text({"code": "x", "kind": "CALL", "attributes": {"name": "other"}}) == "x"


def test_labels_are_joined_by_id_and_conflicts_fail(tmp_path):
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps([{"sample_id": "stable-007", "target": 1}]), encoding="utf-8")
    assert load_labels(manifest) == {"stable-007": 1}
    assert label_for(canonical(), "bad-name.dot", load_labels(manifest)) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        label_for(canonical(), "17-hcpg-0.dot", load_labels(manifest))


def test_embedding_failures_do_not_shift_ids_or_labels(monkeypatch, tmp_path):
    from preprocess import dot_embedding
    def load(path):
        if path.name == "broken.dot":
            raise ValueError("malformed input")
        return canonical(path.stem)
    monkeypatch.setattr(dot_embedding, "canonical_graph", load)
    encoder = CodeGraphEncoder(tokenizer=FakeTokenizer(), model=FakeCodeBERT())
    paths = [tmp_path / "a.dot", tmp_path / "broken.dot", tmp_path / "z.dot"]
    samples, failures = embed_files(paths, encoder, representation="CPG", labels={"a": 1, "z": 0})
    assert [(s.sample_id, s.y.item()) for s in samples] == [("a", 1), ("z", 0)]
    assert all(s.representation == "CPG" for s in samples)
    assert len(failures) == 1
    assert "malformed input" in failures[0]["error"]


def test_parse_api_does_not_swallow_encoding_errors():
    class BrokenEncoder:
        def encode_code_text(self, _):
            raise RuntimeError("encoder unavailable")
    with pytest.raises(RuntimeError, match="encoder unavailable"):
        parse_dot_file_enhanced(canonical(), 1, BrokenEncoder())
