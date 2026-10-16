"""Provider-independent requests and strict response alignment for the paper's two-shot protocol.

Token counts must cover the FULL rendered prompt using each selected model's
actual tokenizer. API calls are deliberately external so provider versions and
errors are preserved, rather than guessed by this repository.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from sp4cpg.experiment import classification_metrics


def prepare_requests(functions, test_ids, limits, token_counts, template):
    if not limits or not test_ids or len(set(test_ids)) != len(test_ids):
        raise ValueError("Nonempty models and unique complete-test IDs are required")
    if template.count("{CODE}") != 1:
        raise ValueError("Prompt template must contain one {CODE} placeholder")
    lookup = {}
    for row in functions:
        sample_id = str(row["sample_id"])
        if not sample_id or sample_id in lookup or row["label"] not in (0, 1) or isinstance(row["label"], bool):
            raise ValueError("Functions require unique sample_id and binary label")
        if not isinstance(row["func"], str) or not row["func"].strip():
            raise ValueError(f"Missing function text for {sample_id}")
        lookup[sample_id] = row
    for model, config in limits.items():
        if not config.get("model_id") or not config.get("version"):
            raise ValueError(f"{model}: explicit provider model ID and version are required")
        for key in ("input_token_limit", "max_output_tokens"):
            if type(config.get(key)) is not int or config[key] < 1:
                raise ValueError(f"{model}: provide actual {key}")
    requests, eligible, excluded = [], [], []
    for sample_id in test_ids:
        if sample_id not in lookup:
            raise ValueError(f"Missing complete-test function {sample_id}")
        reasons = []
        for model, config in limits.items():
            count = token_counts.get(sample_id, {}).get(model)
            if type(count) is not int or count < 1:
                raise ValueError(f"Missing full-prompt token count: {sample_id}/{model}")
            if count > config["input_token_limit"]:
                reasons.append({"model": model, "reason": "full_prompt_exceeds_input_budget", "tokens": count})
        if reasons:
            excluded.append({"sample_id": sample_id, "reasons": reasons})
            continue
        row = lookup[sample_id]
        eligible.append({"sample_id": sample_id, "label": int(row["label"])})
        for model, config in limits.items():
            requests.append({"sample_id": sample_id, "model": model, "model_id": config["model_id"],
                             "version": config["version"], "temperature": 0,
                             "max_output_tokens": config["max_output_tokens"],
                             "full_prompt_tokens": token_counts[sample_id][model],
                             "prompt": template.replace("{CODE}", row["func"])})
    if not eligible:
        raise ValueError("No common eligible test functions")
    return requests, {"protocol": "appendix_B_two_shot", "temperature": 0, "calls_per_function_per_model": 1,
                      "task_specific_finetuning": False, "models": limits, "complete_test_count": len(test_ids),
                      "eligible": eligible, "excluded": excluded,
                      "example_sources": [{"project": "spice", "index": 71, "cve": "CVE-2013-4130", "label": 1},
                                          {"project": "openssl", "index": 0, "cve": "CVE-2013-6449", "label": 1}],
                      "example_dataset": "https://huggingface.co/datasets/colin/PrimeVul"}


def parse_response(response):
    value = json.loads(response) if isinstance(response, str) else response
    if not isinstance(value, dict) or set(value) != {"label", "confidence", "reason", "key_snippets"}:
        raise ValueError("Response must contain exactly label/confidence/reason/key_snippets")
    if type(value["label"]) is not int or value["label"] not in (0, 1):
        raise ValueError("Response label must be integer 0/1")
    confidence = value["confidence"]
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Response confidence must be finite and in [0,1]")
    for key in ("reason", "key_snippets"):
        items = value[key]
        if not isinstance(items, list) or len(items) > 3 or not all(isinstance(s, str) for s in items):
            raise ValueError(f"Response {key} must be a list of up to three strings")
    return value


def collect_responses(manifest, responses, sp4cpg_predictions, sp4cpg_summary):
    if sp4cpg_summary.get("result_kind") not in {"measured_run", "measured_evaluation"}:
        raise ValueError("SP4CPG comparison requires an actual measured model run")
    if sp4cpg_summary.get("split_sha256") != manifest.get("split_sha256") or not manifest.get("split_sha256"):
        raise ValueError("SP4CPG and LLM requests must use the same split manifest")
    if sp4cpg_summary.get("model") != "TAF-HGCN" or sp4cpg_summary.get("representation") != "HCPG" or sp4cpg_summary.get("ablation") != "none":
        raise ValueError("SP4CPG comparison requires the full TAF-HGCN on HCPG")
    test_ids = sp4cpg_summary.get("evaluation", {}).get("complete_test", {}).get("sample_ids", [])
    if set(test_ids) != set(manifest.get("complete_test_ids", [])) or len(test_ids) != len(set(test_ids)):
        raise ValueError("SP4CPG complete-test sample IDs differ from the LLM source split")
    truth = {r["sample_id"]: r["label"] for r in manifest["eligible"]}
    expected = {(model, sample_id) for model in manifest["models"] for sample_id in truth}
    seen, errors, predictions = set(), [], {m: [] for m in manifest["models"]}
    for row in responses:
        key = (row.get("model"), row.get("sample_id"))
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate response {key}")
        seen.add(key)
        try:
            if row.get("status") != "ok":
                raise ValueError(row.get("error", "provider_error"))
            config = manifest["models"][key[0]]
            if row.get("model_id") != config["model_id"] or row.get("version") != config["version"]:
                raise ValueError("Response provider model/version differs from request")
            result = parse_response(row["response"])
            predictions[key[0]].append({"sample_id": key[1], "label": truth[key[1]], "prediction": result["label"]})
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"model": key[0], "sample_id": key[1], "error": str(exc)})
    errors += [{"model": m, "sample_id": i, "error": "missing_response"} for m, i in sorted(expected - seen)]
    aligned, seen_ids = [], set()
    for row in sp4cpg_predictions:
        sample_id = row["sample_id"]
        if sample_id in seen_ids:
            raise ValueError("Duplicate SP4CPG prediction ID")
        seen_ids.add(sample_id)
        if sample_id in truth:
            if row["label"] != truth[sample_id]:
                raise ValueError(f"SP4CPG label differs for {sample_id}")
            aligned.append(row)
    if {r["sample_id"] for r in aligned} != set(truth):
        raise ValueError("SP4CPG predictions do not cover the common eligible IDs")
    return {"result_kind": "measured_llm_response_import", "complete": not errors, "errors": errors,
            "sample_ids": sorted(truth),
            "metrics": ({**{model: classification_metrics(rows) for model, rows in predictions.items()},
                         "SP4CPG": classification_metrics(aligned)} if not errors else None),
            "interpretation": "Selected function-text two-shot configurations only; no claim about fine-tuned or graph-conditioned LLMs"}
