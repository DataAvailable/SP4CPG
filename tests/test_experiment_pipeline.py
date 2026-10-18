"""Exercise experiment commands with local model and graph files."""

import json
import copy

import pytest
import torch

from experiments import rq6
from experiments import llm, sp4cpg_predict
from models.hgcn_models import TAFHGCN
from preprocess.graph_core import dump_graph, normalize_graph, transform_graph
from preprocess.graph_data import graph_to_data


def test_rq6_loads_local_transformer_and_writes_paired_statistics(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

    checkpoint = tmp_path / "encoder"
    checkpoint.mkdir()
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(["int value() { return 2; }", "2", "f(2)"], vocab_size=280,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(checkpoint))
    tokenizer = RobertaTokenizerFast(vocab_file=str(checkpoint / "vocab.json"),
                                    merges_file=str(checkpoint / "merges.txt"))
    tokenizer.save_pretrained(checkpoint)
    torch.manual_seed(41)
    encoder = RobertaModel(RobertaConfig(vocab_size=len(tokenizer), hidden_size=8,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        intermediate_size=16, max_position_embeddings=34))
    encoder.save_pretrained(checkpoint)

    original = normalize_graph({"graph_id": "function-1", "nodes": [
        {"id": "node-1", "kind": "CALL", "code": "f(2)", "attributes": {}}
    ], "relations": [], "metadata": {"label": 1}})
    cpg, hcpg = tmp_path / "cpg.json", tmp_path / "hcpg.json"
    dump_graph(original, cpg)
    dump_graph(transform_graph(original), hcpg)
    output = tmp_path / "results"
    assert rq6.main(["--cpg", str(cpg), "--hcpg", str(hcpg), "--output", str(output),
                     "--dataset", "D1", "--codebert", str(checkpoint), "--max-length", "16",
                     "--batch-size", "2", "--local-files-only"]) == 0
    report = json.loads((output / "rq6_results.json").read_text(encoding="utf-8"))
    assert report["pair_count"] == 1
    assert report["statistics"]["min"] == pytest.approx(1)
    assert report["statistics"]["mean_x100"] == pytest.approx(100)
    assert report["pairs"][0]["graph_id"] == "function-1"
    assert (output / "rq6_pairs.csv").is_file()


def protocol_files(tmp_path):
    config = json.loads(llm.CONFIG_PATH.read_text(encoding="utf-8"))
    for key, model in config["models"].items():
        model.update(model_id=key, context_window=1024, input_token_limit=900,
                     max_output_tokens=64, tokenizer={"id": key, "version": "local-v1"})
        if model["endpoint"] is None:
            model["endpoint"] = "http://127.0.0.1:8000/v1/chat/completions"
        if model["provider"] == "anthropic_messages":
            model["api_version"] = "2023-06-01"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    records = [{"sample_id": "source-a", "func": "int value() { return 2; }", "target": 0},
               {"sample_id": "source-b", "func": "int dereference(int *p) { return *p; }", "target": 1}]
    test_path = tmp_path / "test.json"
    test_path.write_text(json.dumps(records), encoding="utf-8")
    counts = llm.token_inputs(test_path, tmp_path / "token-inputs.json", config_path=config_path, split="test")
    for model in counts["models"].values():
        model.update(method="tokenizer", includes_request_overhead=True)
    for row in counts["records"]:
        row["input_tokens"] = {key: 128 for key in llm.MODEL_KEYS}
    counts_path = tmp_path / "counts.json"
    counts_path.write_text(json.dumps(counts), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = llm.prepare(test_path, counts_path, manifest_path, config_path=config_path, split="test")
    return manifest_path, manifest


def graph_dataset(tmp_path):
    samples = []
    for graph_id, label, code in (("0", 0, "int value() { return 2; }"),
                                  ("1", 1, "int dereference(int *p) { return *p; }")):
        graph = normalize_graph({"graph_id": graph_id, "nodes": [
            {"id": "a", "kind": "METHOD", "code": code, "attributes": {}},
            {"id": "b", "kind": "RETURN", "code": "return value;", "attributes": {}}
        ], "relations": [{"kind": "DDG", "sources": ["a"], "targets": ["b"]}],
            "metadata": {"label": label, "source_code_sha256": llm.text_sha256(code)}})
        samples.append(graph_to_data(transform_graph(graph), torch.ones(2, 4), label))
    path = tmp_path / "graphs.pkl"
    torch.save(list(reversed(samples)), path)
    checkpoint = tmp_path / "detector.pt"
    torch.manual_seed(7)
    model = TAFHGCN(dropout=0, in_channels=4, hidden_channels=6, type_dim=3, num_layers=2, hop_size=1)
    torch.save(model.state_dict(), checkpoint)
    mapping = tmp_path / "ids.json"
    mapping.write_text(json.dumps({"source-a": "0", "source-b": "1"}), encoding="utf-8")
    return path, checkpoint, mapping


def test_shared_manifest_runs_sp4cpg_and_scores_all_seven_models(tmp_path):
    manifest_path, manifest = protocol_files(tmp_path)
    data_path, checkpoint, id_map = graph_dataset(tmp_path)
    output = tmp_path / "sp4cpg.jsonl"
    assert sp4cpg_predict.main(["--manifest", str(manifest_path), "--data", str(data_path),
                               "--checkpoint", str(checkpoint), "--id-map", str(id_map),
                               "--output", str(output), "--hop-size", "1", "--batch-size", "1"]) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_id"] for row in rows] == ["source-a", "source-b"]
    assert [row["graph_id"] for row in rows] == ["0", "1"]
    assert all(row["status"] == "ok" and 0 <= row["probability"] <= 1 for row in rows)
    predictions = {"sp4cpg": output}
    for key in llm.MODEL_KEYS:
        predictions[key] = tmp_path / f"{key}.jsonl"
        predictions[key].write_text("\n".join(json.dumps({
            "sample_id": row["sample_id"], "code_sha256": row["code_sha256"], "model_key": key,
            "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
            "rendered_prompt_sha256": row["rendered_prompt_sha256"], "status": "ok",
            "response": {"label": row["label"], "confidence": 1, "reason": [], "key_snippets": []}
        }) for row in manifest["samples"]) + "\n", encoding="utf-8")
    result = llm.evaluate(manifest_path, predictions, tmp_path / "metrics.json")
    assert result["complete"] and len(result["metrics"]) == 7
    assert all(metrics["samples"] == 2 for metrics in result["metrics"].values())
    assert result["metrics"]["gpt4o_mini"]["f1"] == 1
    # The same model produces identical results across inference batch sizes.
    batched = sp4cpg_predict.run(manifest_path, data_path, checkpoint, tmp_path / "batched.jsonl",
                                id_map=id_map, hop_size=1, batch_size=2)
    assert [row["prediction"] for row in batched] == [row["prediction"] for row in rows]
    assert [row["probability"] for row in batched] == pytest.approx([row["probability"] for row in rows])


@pytest.mark.parametrize("change,expected", [
    ("missing_map", "Missing graph"),
    ("shared_graph", "distinct graph IDs"),
    ("wrong_label", "label differs"),
    ("wrong_representation", "HCPG representation"),
    ("duplicate_graph", "Duplicate graph"),
    ("wrong_source", "source checksum differs"),
    ("missing_source", "source checksum differs"),
    ("checkpoint", "17 node types"),
])
def test_sp4cpg_rejects_mismatched_inputs_before_writing_predictions(tmp_path, change, expected):
    manifest_path, _ = protocol_files(tmp_path)
    data, checkpoint, id_map = graph_dataset(tmp_path)
    if change == "missing_map":
        id_map = None
    elif change == "shared_graph":
        id_map.write_text(json.dumps({"source-a": "0", "source-b": "0"}), encoding="utf-8")
    elif change == "checkpoint":
        state = torch.load(checkpoint)
        state["node_type_embedding.weight"] = state["node_type_embedding.weight"][:16]
        torch.save(state, checkpoint)
    else:
        dataset = torch.load(data)
        if change == "wrong_label":
            dataset[0].y = torch.tensor([0])
        elif change == "wrong_representation":
            dataset[0].representation = "CPG"
        elif change in {"wrong_source", "missing_source"}:
            canonical = json.loads(dataset[0].canonical_json)
            if change == "wrong_source":
                canonical["metadata"]["source_code_sha256"] = llm.text_sha256("int other() { return 0; }")
            else:
                del canonical["metadata"]["source_code_sha256"]
            dataset[0].canonical_json = json.dumps(canonical)
        else:
            dataset.append(copy.deepcopy(dataset[0]))
        torch.save(dataset, data)
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(ValueError, match=expected):
        sp4cpg_predict.run(manifest_path, data, checkpoint, output, id_map=id_map)
    assert not output.exists()


def test_sp4cpg_preserves_an_existing_prediction_file(tmp_path):
    manifest_path, _ = protocol_files(tmp_path)
    data, checkpoint, id_map = graph_dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    output.write_text("previous output\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sp4cpg_predict.run(manifest_path, data, checkpoint, output, id_map=id_map)
    assert output.read_text(encoding="utf-8") == "previous output\n"
