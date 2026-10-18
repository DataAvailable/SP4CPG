import copy
import csv
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments import rq6
from preprocess.graph_core import dump_graph, normalize_graph, transform_graph
from preprocess.parse_dot import CodeGraphEncoder


def graph_pair(graph_id="entry-a", leaf=True):
    nodes = [{"id": "p", "kind": "CALL", "code": "f(2)", "attributes": {}}]
    relations = []
    if leaf:
        nodes.append({"id": "l", "kind": "LITERAL", "code": "2", "attributes": {}})
        relations.append({"id": "ast", "kind": "AST", "sources": ["p"], "targets": ["l"],
                          "label": "AST", "attributes": {}})
    original = normalize_graph({"schema_version": 2, "graph_id": graph_id, "nodes": nodes,
                                "relations": relations, "metadata": {"label": 1}})
    return original, transform_graph(original)


def write_pair(tmp_path, graph_id="entry-a", leaf=True, suffix=".json"):
    original, transformed = graph_pair(graph_id, leaf)
    cpg, hcpg = tmp_path / ("input-cpg" + suffix), tmp_path / ("input-hcpg" + suffix)
    dump_graph(original, cpg)
    dump_graph(transformed, hcpg)
    return cpg, hcpg


class FixtureTokenizer:
    def __call__(self, texts, **options):
        self.options = options
        return {"input_ids": torch.tensor([[len(text), 1] for text in texts])}


class FixtureCodeBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(0.95)
        self.config = SimpleNamespace(hidden_size=2, _commit_hash="local-encoder-version")
        self.calls = []

    def forward(self, input_ids):
        self.calls.append(input_ids.detach().clone())
        return SimpleNamespace(last_hidden_state=self.dropout(input_ids.float().unsqueeze(1) * self.weight))


def encoder():
    return CodeGraphEncoder(tokenizer=FixtureTokenizer(), model=FixtureCodeBERT())


def test_all_physical_nodes_are_encoded_once_with_one_frozen_eval_encoder(tmp_path):
    cpg, hcpg = write_pair(tmp_path)
    model = encoder()
    model.train()
    model.requires_grad_(True)
    report = rq6.run(cpg, hcpg, tmp_path / "results", "D1", encoder=model)
    expected = 11 / math.sqrt(7.25 * 17)
    assert report["pair_count"] == 1
    row = report["pairs"][0]
    assert row["graph_id"] == "entry-a" and row["cpg_nodes"] == 2 and row["hcpg_nodes"] == 1
    assert row["cosine_similarity"] == pytest.approx(expected)
    assert row["cosine_similarity_x100"] == pytest.approx(expected * 100)
    assert not model.training and not model.codebert.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert len(model.codebert.calls) == 2
    assert sorted(model.codebert.calls[0][:, 0].tolist()) == [1, 4]
    assert model.codebert.calls[1][:, 0].tolist() == [4]
    assert model.tokenizer.options["max_length"] == 128
    saved = json.loads((tmp_path / "results" / "rq6_results.json").read_text(encoding="utf-8"))
    assert saved == report
    assert report["encoder"]["resolved_model_version"] == "local-encoder-version"
    assert report["inputs"][0]["cpg"]["sha256"] != report["inputs"][0]["hcpg"]["sha256"]
    rows = list(csv.DictReader((tmp_path / "results" / "rq6_pairs.csv").open(encoding="utf-8")))
    assert len(rows) == 1 and float(rows[0]["cosine_similarity"]) == pytest.approx(expected)


def test_recursive_inputs_pair_by_graph_id_and_summarize_per_pair_values(tmp_path):
    cpg_dir, hcpg_dir = tmp_path / "cpg", tmp_path / "hcpg"
    for root in (cpg_dir, hcpg_dir):
        (root / "nested").mkdir(parents=True)
        (root / "construction_report.json").write_text("not a graph", encoding="utf-8")
        (root / "nested" / "entry.construction_report.json").write_text("not JSON", encoding="utf-8")
    for graph_id, leaf, name in (("z", False, "aaa"), ("a", True, "zzz")):
        original, transformed = graph_pair(graph_id, leaf)
        dump_graph(original, cpg_dir / "nested" / (name + ".dot"))
        dump_graph(transformed, hcpg_dir / (name + ".json"))
    report = rq6.run(cpg_dir, hcpg_dir, tmp_path / "results", "D2", encoder=encoder())
    assert [row["graph_id"] for row in report["pairs"]] == ["a", "z"]
    expected = 11 / math.sqrt(7.25 * 17)
    stats = report["statistics"]
    assert stats["min"] == pytest.approx(expected)
    assert stats["max"] == pytest.approx(1)
    assert stats["mean"] == pytest.approx((expected + 1) / 2)
    for key in ("min", "max", "mean"):
        assert stats[key + "_x100"] == pytest.approx(stats[key] * 100)


def test_cli_passes_encoder_options_to_a_single_instance(tmp_path, monkeypatch):
    cpg, hcpg = write_pair(tmp_path, suffix=".dot")
    calls = []
    def create(**options):
        calls.append(options)
        return CodeGraphEncoder(**options, tokenizer=FixtureTokenizer(), model=FixtureCodeBERT())
    monkeypatch.setattr(rq6, "CodeGraphEncoder", create)
    output = tmp_path / "results"
    assert rq6.main(["--cpg", str(cpg), "--hcpg", str(hcpg), "--output", str(output),
                     "--dataset", "D3", "--codebert", "local-codebert", "--model-version", "v7",
                     "--max-length", "64", "--batch-size", "1", "--device", "cpu",
                     "--local-files-only"]) == 0
    assert calls == [{"codebert_model_name": "local-codebert", "model_version": "v7", "max_length": 64,
                      "batch_size": 1, "device": "cpu", "local_files_only": True}]


def test_duplicate_missing_and_incorrect_representation_pairs_are_rejected(tmp_path):
    cpg, hcpg = write_pair(tmp_path)
    directory = tmp_path / "cpgraphs"
    directory.mkdir()
    original, transformed = graph_pair()
    dump_graph(original, directory / "one.json")
    dump_graph(original, directory / "two.json")
    with pytest.raises(ValueError, match="Duplicate graph_id"):
        rq6.pair_graphs(directory, hcpg)
    transformed["graph_id"] = "different-id"
    dump_graph(transformed, hcpg)
    with pytest.raises(ValueError, match="Unpaired graph IDs"):
        rq6.pair_graphs(cpg, hcpg)
    dump_graph(original, hcpg)
    with pytest.raises(ValueError, match="Expected HCPG"):
        rq6.pair_graphs(cpg, hcpg)


def test_dictionary_difference_and_label_conflicts_fail_before_encoder_construction(tmp_path, monkeypatch):
    cpg, hcpg = write_pair(tmp_path)
    def unavailable(**options):
        raise AssertionError("Input checks must precede encoder construction")
    monkeypatch.setattr(rq6, "CodeGraphEncoder", unavailable)
    original, transformed = graph_pair()
    transformed["dictionary"]["l"]["attributes"]["marker"] = 1
    transformed["nodes"][0]["attributes"]["retained_contents"][0]["attributes"]["marker"] = 1
    dump_graph(transformed, hcpg)
    with pytest.raises(ValueError, match="Original dictionary entries differ"):
        rq6.run(cpg, hcpg, tmp_path / "output", "D4")
    assert not (tmp_path / "output").exists()
    _, transformed = graph_pair()
    transformed["metadata"]["label"] = 0
    dump_graph(transformed, hcpg)
    with pytest.raises(ValueError, match="Conflicting CPG/HCPG labels"):
        rq6.run(cpg, hcpg, tmp_path / "output", "D4")
    original["metadata"]["target"] = 0
    dump_graph(original, cpg)
    with pytest.raises(ValueError, match="Conflicting labels"):
        rq6.pair_graphs(cpg, hcpg)


def test_filename_labels_cannot_override_graph_labels(tmp_path):
    original, transformed = graph_pair()
    cpg, hcpg = tmp_path / "entry-cpg-0.json", tmp_path / "entry-hcpg-1.json"
    dump_graph(original, cpg)
    dump_graph(transformed, hcpg)
    with pytest.raises(ValueError, match="Conflicting labels"):
        rq6.pair_graphs(cpg, hcpg)


@pytest.mark.parametrize("change", ["one_missing", "different"])
def test_paired_source_checksums_must_match(tmp_path, change):
    original, transformed = graph_pair()
    original["metadata"]["source_code_sha256"] = "a" * 64
    if change == "different":
        transformed["metadata"]["source_code_sha256"] = "b" * 64
    cpg, hcpg = tmp_path / "cpg.json", tmp_path / "hcpg.json"
    dump_graph(original, cpg)
    dump_graph(transformed, hcpg)
    with pytest.raises(ValueError, match="source checksums differ"):
        rq6.pair_graphs(cpg, hcpg)


class FixtureArrayEncoder(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def encode_code_text(self, texts):
        rows = len(texts)
        if self.mode == "zero":
            return torch.zeros(rows, 2)
        if self.mode == "nan":
            return torch.full((rows, 2), float("nan"))
        if self.mode == "integer":
            return torch.ones(rows, 2, dtype=torch.long)
        if self.mode == "shape":
            return torch.ones(rows + 1, 2)
        if self.mode == "width":
            return torch.ones(rows, 0)
        if self.mode == "cancel":
            return torch.tensor([[1.0, 0.0] if text == "f(2)" else [-1.0, 0.0] for text in texts])
        if self.mode == "partial_zero":
            return torch.tensor([[2.0, 1.0] if text == "f(2)" else [0.0, 0.0] for text in texts])
        return torch.full((rows, 2), 1e-300, dtype=torch.float64)


@pytest.mark.parametrize("mode", ["zero", "nan", "integer", "shape", "width", "cancel"])
def test_invalid_node_vectors_and_zero_graph_vectors_are_rejected(tmp_path, mode):
    cpg, hcpg = write_pair(tmp_path)
    with pytest.raises(ValueError, match="vector"):
        rq6.run(cpg, hcpg, tmp_path / "output", "D1", encoder=FixtureArrayEncoder(mode))
    assert not (tmp_path / "output").exists()


def test_zero_node_rows_participate_in_a_nonzero_graph_mean(tmp_path):
    cpg, hcpg = write_pair(tmp_path)
    original, _ = graph_pair()
    model = FixtureArrayEncoder("partial_zero")
    assert torch.equal(rq6._mean_vector(original, model), torch.tensor([1.0, 0.5], dtype=torch.float64))
    report = rq6.run(cpg, hcpg, tmp_path / "output", "D1", encoder=model)
    assert report["statistics"]["mean"] == pytest.approx(1)


def test_original_comment_node_is_encoded_before_rule2_removal(tmp_path):
    original = normalize_graph({
        "schema_version": 2, "graph_id": "comment-entry",
        "nodes": [{"id": "p", "kind": "CALL", "code": "f(2)", "attributes": {}},
                  {"id": "c", "kind": "COMMENT", "code": "// note", "attributes": {}}],
        "relations": [{"id": "ast", "kind": "AST", "sources": ["p"], "targets": ["c"],
                       "label": "AST", "attributes": {}}],
        "metadata": {"label": 1}})
    transformed = transform_graph(original)
    assert transformed["dictionary"]["c"]["status"] == "removed"
    cpg, hcpg = tmp_path / "cpg.json", tmp_path / "hcpg.json"
    dump_graph(original, cpg)
    dump_graph(transformed, hcpg)
    model = encoder()
    report = rq6.run(cpg, hcpg, tmp_path / "output", "D1", encoder=model)
    assert report["pairs"][0]["cpg_nodes"] == 2
    assert report["pairs"][0]["hcpg_nodes"] == 1
    assert sorted(model.codebert.calls[0][:, 0].tolist()) == [4, 7]
    assert model.codebert.calls[1][:, 0].tolist() == [4]
    assert report["statistics"]["mean"] == pytest.approx(23 / math.sqrt(31.25 * 17))


def test_nonzero_small_vectors_have_well_defined_cosine(tmp_path):
    cpg, hcpg = write_pair(tmp_path)
    report = rq6.run(cpg, hcpg, tmp_path / "output", "D1", encoder=FixtureArrayEncoder("small"))
    assert report["statistics"]["mean"] == pytest.approx(1)


def test_corrupt_graphs_and_report_only_directories_are_not_skipped(tmp_path):
    root = tmp_path / "graphs"
    root.mkdir()
    report = root / "construction_report.json"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="No DOT or JSON"):
        rq6.discover_graphs(root, "CPG")
    with pytest.raises(ValueError, match="Expected a DOT or JSON graph"):
        rq6.discover_graphs(report, "CPG")
    (root / "broken.json").write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read graph"):
        rq6.discover_graphs(root, "CPG")


def test_existing_results_are_preserved(tmp_path):
    cpg, hcpg = write_pair(tmp_path)
    output = tmp_path / "output"
    first = rq6.run(cpg, hcpg, output, "D1", encoder=encoder())
    before = (output / "rq6_results.json").read_bytes()
    with pytest.raises(ValueError, match="new or empty directory"):
        rq6.run(cpg, hcpg, output, "D1", encoder=encoder())
    assert (output / "rq6_results.json").read_bytes() == before
    assert json.loads(before)["statistics"] == first["statistics"]
