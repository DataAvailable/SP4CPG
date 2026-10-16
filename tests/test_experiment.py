import copy
import json
from types import SimpleNamespace

import pytest
import torch

from sp4cpg.experiment import (classification_metrics, evaluate_scopes, make_split, mean_run_metrics, membership_fingerprint,
                                sample_records, validate_split)
from sp4cpg.llm_protocol import collect_responses, parse_response, prepare_requests


def test_split_is_stable_and_rejects_overlap_missing_and_changed_labels():
    records = [{"sample_id": str(i), "label": i % 2, "original_node_count": i + 1} for i in range(20)]
    split = make_split(records, 7)
    assert split == make_split(list(reversed(records)), 7)
    assert [len(split[k]) for k in ("train", "validation", "test")] == [16, 2, 2]
    validate_split(split, records)
    assert not split["historical_paper_split"]
    bad = copy.deepcopy(split)
    bad["test"][0] = bad["train"][0]
    with pytest.raises(ValueError, match="overlapping"):
        validate_split(bad, records)
    with pytest.raises(ValueError, match="exactly"):
        validate_split(split, records[:-1])
    changed = copy.deepcopy(records)
    changed[0]["label"] = 1
    with pytest.raises(ValueError, match="fingerprint"):
        validate_split(split, changed)


def test_scope_uses_original_cpg_threshold_and_empty_is_na():
    rows = [{"sample_id": "500", "label": 1, "prediction": 0, "original_node_count": 500},
            {"sample_id": "501", "label": 1, "prediction": 1, "original_node_count": 501}]
    result = evaluate_scopes(rows)
    assert result["complete_test"]["recall"] == 0.5
    assert result["original_cpg_gt_500"]["sample_ids"] == ["501"]
    assert result["original_cpg_gt_500"]["f1"] == 1.0
    assert result["nodes_1001_plus"]["f1"] is None


def test_per_seed_f1_is_averaged_and_alignment_enforced():
    base = {"result_kind": "measured_run", "seed": 2025, "model": "HGNN", "representation": "HCPG",
            "membership_sha256": "ids", "split_sha256": "split", "dataset_sha256": "actual-data", "model_config": {},
            "training_config": {"seed": 2025, "batch": 64},
            "evaluation": {"complete_test": {"sample_ids": ["a"], "precision": 1.0,
                                             "recall": 0.5, "f1": 2 / 3, "accuracy": 0.5}}}
    second = copy.deepcopy(base)
    second["seed"] = second["training_config"]["seed"] = 2026
    second["evaluation"]["complete_test"].update(precision=0.5, recall=1.0)
    aggregate = mean_run_metrics([base, second])["evaluation"]["complete_test"]
    assert aggregate["f1_mean"] == pytest.approx(2 / 3)
    assert aggregate["precision_mean"] == 0.75
    second["evaluation"]["complete_test"]["sample_ids"] = ["b"]
    with pytest.raises(ValueError, match="records differ"):
        mean_run_metrics([base, second])


def test_metrics_reject_fractional_or_boolean_predictions():
    for value in (0.9, True, "1"):
        with pytest.raises(ValueError, match="binary"):
            classification_metrics([{"label": 1, "prediction": value}])


def test_llm_common_filter_and_error_response_does_not_shrink_comparison():
    functions = [{"sample_id": name, "label": 1, "func": "int f(){return 1;}"} for name in ("a", "b")]
    limits = {"m1": {"model_id": "provider/model", "version": "pinned", "input_token_limit": 100,
                     "max_output_tokens": 20},
              "m2": {"model_id": "provider/model2", "version": "pinned", "input_token_limit": 50,
                     "max_output_tokens": 20}}
    requests, manifest = prepare_requests(functions, ["a", "b"], limits,
                                          {"a": {"m1": 30, "m2": 30}, "b": {"m1": 30, "m2": 51}},
                                          "Example label 1. Analyze {CODE}")
    assert len(requests) == 2 and len(manifest["eligible"]) == 1
    assert all("label" not in row and row["temperature"] == 0 for row in requests)
    predictions = [{"sample_id": "a", "label": 1, "prediction": 1}]
    manifest.update(split_sha256="same-split", complete_test_ids=["a", "b"])
    summary = {"result_kind": "measured_run", "split_sha256": "same-split", "model": "TAF-HGCN",
               "representation": "HCPG", "ablation": "none",
               "evaluation": {"complete_test": {"sample_ids": ["a", "b"]}}}
    response = {"label": 1, "confidence": 0.8, "reason": [], "key_snippets": []}
    rows = [{"model": "m1", "sample_id": "a", "status": "ok", "model_id": "provider/model",
             "version": "pinned", "response": response}]
    incomplete = collect_responses(manifest, rows, predictions, summary)
    assert not incomplete["complete"] and incomplete["metrics"] is None
    rows += [{"model": "m2", "sample_id": "a", "status": "ok", "model_id": "provider/model2",
              "version": "pinned", "response": json.dumps(response)}]
    assert collect_responses(manifest, rows, predictions, summary)["metrics"]["m1"]["f1"] == 1
    with pytest.raises(ValueError, match="same split"):
        collect_responses(manifest, rows, predictions, {**summary, "split_sha256": "different"})
    with pytest.raises(ValueError):
        parse_response({**response, "confidence": float("nan")})
    with pytest.raises(ValueError):
        parse_response({**response, "label": True})


def test_cpu_training_checkpoint_and_eval_round_trip(tmp_path):
    from main import parse_args, run
    from sp4cpg.data import GraphData
    samples = []
    for i in range(12):
        samples.append(GraphData(x=torch.tensor([[i / 12, 0.2], [0.3, 0.5]]),
                                 edge_index=torch.tensor([[0], [1]]),
                                 hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
                                 incidence_role=torch.tensor([0, 1]), relation_types=torch.tensor([3]),
                                 node_types=torch.tensor([11, 11]), num_nodes=2, num_hyperedges=1,
                                 sample_id=str(i), original_node_count=600 if i % 2 else 2,
                                 schema_version=2, representation="HCPG", y=torch.tensor([i % 2])))
    data = tmp_path / "test_graphs.pt"
    torch.save({"schema_version": 2, "representation": "HCPG", "samples": samples,
                "metadata": {"encoder": {"source": "in-memory correctness-test graphs, separate from paper data"}}}, data)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(make_split(sample_records(samples), 2025)), encoding="utf-8")
    common = ["--data", str(data), "--split", str(split), "--device", "cpu", "--epoch", "2",
              "--hidden", "8", "--layers", "2", "--type-dim", "4", "--hop-size", "2", "--batch", "4"]
    for model in ("TAF-HGCN", "HGNN"):
        output = tmp_path / model
        summary = run(parse_args(common + ["--model", model, "--output", str(output)]))
        assert summary["evaluation"]["complete_test"]["samples"] == len(json.loads(split.read_text())["test"])
        assert summary["training_seconds"] > 0
        assert summary["training_peak_allocated_bytes"] is None
        if model == "TAF-HGCN":
            from client.sp4cpg_client.tools.predict import compute_metrics, run_real_hgcn
            inference_args = SimpleNamespace(repo=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
                                             data=str(data), checkpoint=str(output / "best_model.pt"),
                                             batch=4, model=model, source_data="")
            inferred = run_real_hgcn(inference_args)
            assert [row["sample_id"] for row in inferred] == [str(i) for i in range(12)]
            assert all(row["detector"] == "TAF-HGCN" for row in inferred)
            assert compute_metrics([{**inferred[0], "detector": "fallback_lexical_not_hgcn"}])["mode"] == "lexical_fallback_not_model_evaluation"
            unlabelled = [s.clone() for s in samples]
            for sample in unlabelled:
                del sample.y
            unlabelled_path = tmp_path / "unlabelled.pt"
            torch.save({"schema_version": 2, "representation": "HCPG", "samples": unlabelled,
                        "metadata": {"encoder": {"source": "in-memory correctness-test graphs, separate from paper data"}}}, unlabelled_path)
            inference_args.data = str(unlabelled_path)
            assert all(row["true_label"] is None for row in run_real_hgcn(inference_args))
        evaluated = run(parse_args(common + ["--model", model, "--output", str(tmp_path / (model + "-eval")),
                                            "--eval-only", "--seed", "2027", "--checkpoint", str(output / "best_model.pt")]))
        assert summary["evaluation"] == evaluated["evaluation"]
        assert evaluated["seed"] == 2025 and evaluated["evaluation_seed"] == 2027
        assert evaluated["result_kind"] == "measured_evaluation"
        with pytest.raises(ValueError, match="Only measured run"):
            mean_run_metrics([summary, evaluated])
