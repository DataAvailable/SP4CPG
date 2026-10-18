import copy
import json
from pathlib import Path
import socket

import pytest

from experiments import llm


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def configured():
    config = json.loads(llm.CONFIG_PATH.read_text(encoding="utf-8"))
    for key, model in config["models"].items():
        model.update(model_id="configured-" + key, context_window=1000, input_token_limit=800,
                     max_output_tokens=100, tokenizer={"id": key + "-tokenizer", "version": "1"})
        if model["endpoint"] is None:
            model["endpoint"] = "http://localhost:8099/v1/chat/completions"
        if model["provider"] == "anthropic_messages":
            model["api_version"] = "2023-06-01"
    return config


def setup_files(tmp_path, rows=None):
    config = write_json(tmp_path / "config.json", configured())
    rows = rows or [{"sample_id": "stable-a", "func": "int a() { return 1; }", "target": 1},
                    {"sample_id": "stable-b", "func": "int b() { return 0; }", "target": 0}]
    test = write_json(tmp_path / "test.json", rows)
    spec = llm.token_inputs(test, tmp_path / "token-inputs.json", config_path=config, split="test")
    counts = {"schema_version": 1, "config_sha256": spec["config_sha256"], "prompt_sha256": spec["prompt_sha256"],
              "models": {key: dict(value, method="tokenizer", includes_request_overhead=True)
                         for key, value in spec["models"].items()},
              "records": [{"sample_id": row["sample_id"], "code_sha256": row["code_sha256"],
                           "rendered_prompt_sha256": row["rendered_prompt_sha256"],
                           "input_tokens": {key: 200 for key in llm.MODEL_KEYS}} for row in spec["records"]]}
    counts_path = write_json(tmp_path / "counts.json", counts)
    return test, config, counts_path, counts, spec


def prepared(tmp_path):
    test, config, counts_path, counts, spec = setup_files(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = llm.prepare(test, counts_path, manifest_path, config_path=config, split="test")
    return manifest_path, manifest, config


def classification(label=1):
    return {"label": label, "confidence": .8, "reason": ["Code evidence"], "key_snippets": ["return 1;"]}


class FixtureTransport:
    def __init__(self, provider="openai_chat", label=1):
        self.calls = []
        self.provider = provider
        self.label = label

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        text = json.dumps(classification(self.label))
        if self.provider == "anthropic_messages":
            response = {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}
        elif self.provider == "gemini_generate_content":
            response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}
        else:
            response = {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        return 200, json.dumps(response)


def test_shared_prompt_has_exact_two_demonstrations_and_no_target_label():
    template = llm.load_prompt()
    assert template.count("Label: 1") == 2
    assert "red_channel_pipes_add_empty_msg" in template
    assert "long ssl_get_algorithm2(SSL *s)" in template
    assert template.count("{CODE}") == 1
    assert llm.render_prompt("code {CODE}").endswith("code {CODE}\n")


def test_defaults_require_explicit_model_limits_and_tokenizers():
    with pytest.raises(ValueError, match="model_id"):
        llm.load_config()
    config = configured()
    config["models"]["gpt4o_mini"]["top_p"] = 1
    with pytest.raises(ValueError, match="configuration fields"):
        llm.load_config(config)
    config = configured()
    config["temperature"] = 1
    with pytest.raises(ValueError, match="temperature=0"):
        llm.load_config(config)


def test_complete_test_requires_explicit_split_and_stable_unique_ids(tmp_path):
    test = write_json(tmp_path / "test.json", [{"idx": 17, "func": "a()", "target": 1}])
    with pytest.raises(ValueError, match="--split test"):
        llm.read_test(test)
    assert llm.read_test(test, "test")[0]["sample_id"] == "17"
    write_json(test, [{"idx": 17, "func": "a()", "target": 1, "split": "train"}])
    with pytest.raises(ValueError, match="test"):
        llm.read_test(test, "test")
    write_json(test, [{"func": "a()", "target": 1, "split": "test"}])
    with pytest.raises(ValueError, match="stable"):
        llm.read_test(test)
    write_json(test, [{"id": "17", "func": "a()", "target": 1}, {"idx": 17, "func": "b()", "target": 0}])
    with pytest.raises(ValueError, match="Duplicate"):
        llm.read_test(test, "test")


def test_jsonl_split_and_conflicting_labels_are_checked(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text('\n'.join(json.dumps(row) for row in [
        {"id": "a", "func": "a()", "target": 1, "split": "test"},
        {"id": "b", "func": "b()", "target": 0, "split": "test"}]), encoding="utf-8")
    assert len(llm.read_test(path)) == 2
    write_json(path, [{"id": "a", "func": "a()", "target": 0, "label": False}])
    with pytest.raises(ValueError, match="Conflicting labels"):
        llm.read_test(path, "test")


def test_token_inputs_export_full_requests_without_gold(tmp_path):
    test, config, counts_path, counts, spec = setup_files(tmp_path)
    for row in spec["records"]:
        assert "label" not in row and "target" not in row
        assert row["rendered_prompt_sha256"] == llm.text_sha256(row["rendered_prompt"])
        assert set(row["requests"]) == set(llm.MODEL_KEYS)


def test_prepare_intersection_excludes_a_function_exceeding_any_one_model(tmp_path):
    test, config, counts_path, counts, spec = setup_files(tmp_path)
    counts["records"][1]["input_tokens"]["qwen25_coder7b"] = 801
    write_json(counts_path, counts)
    manifest = llm.prepare(test, counts_path, tmp_path / "manifest.json", config_path=config, split="test")
    assert manifest["selected_ids"] == ["stable-a"]
    assert manifest["excluded"][0]["sample_id"] == "stable-b"
    assert "input_token_limit:qwen25_coder7b" in manifest["excluded"][0]["reasons"]
    assert manifest["counts"] == {"total": 2, "selected": 1, "excluded": 1}


@pytest.mark.parametrize("change,match", [
    (lambda c: c.update(config_sha256="changed"), "configuration"),
    (lambda c: c["models"]["gpt4o_mini"].update(method="characters"), "Exact provider/tokenizer"),
    (lambda c: c["models"]["gpt4o_mini"].update(includes_request_overhead=False), "overhead"),
    (lambda c: c["records"].pop(), "complete test IDs"),
    (lambda c: c["records"][0].update(rendered_prompt_sha256="changed"), "content"),
    (lambda c: c["records"][0]["input_tokens"].pop("llama31_8b"), "Every function"),
    (lambda c: c["records"][0]["input_tokens"].update(gpt4o_mini=True), "positive integer"),
])
def test_counts_must_be_exact_complete_and_bound_to_full_prompt(tmp_path, change, match):
    test, config, counts_path, counts, _ = setup_files(tmp_path)
    change(counts)
    write_json(counts_path, counts)
    with pytest.raises(ValueError, match=match):
        llm.prepare(test, counts_path, tmp_path / "manifest.json", config_path=config, split="test")


@pytest.mark.parametrize("key", ["gpt4o_mini", "claude_sonnet4", "gemini25_pro", "llama31_8b"])
def test_provider_requests_use_only_temperature_output_cap_and_user_prompt(tmp_path, monkeypatch, key):
    manifest_path, manifest, _ = prepared(tmp_path)
    model = manifest["config"]["models"][key]
    if model["api_key_env"]:
        monkeypatch.setenv(model["api_key_env"], "test-credential")
    transport = FixtureTransport(model["provider"])
    results = llm.run_model(manifest_path, key, tmp_path / "runs", transport=transport)
    assert all(row["status"] == "ok" for row in results)
    assert len(transport.calls) == 2
    for url, headers, payload, timeout in transport.calls:
        if model["provider"] == "gemini_generate_content":
            assert set(payload) == {"contents", "generationConfig"}
            assert payload["generationConfig"] == {"temperature": 0, "maxOutputTokens": 100}
            assert headers["x-goog-api-key"] == "test-credential"
        else:
            assert set(payload) == {"model", "messages", "temperature", model.get("output_token_field", "max_tokens")}
            assert payload["messages"][0]["role"] == "user"
        assert "gold" not in payload and "target" not in payload and "label" not in payload
    llm.run_model(manifest_path, key, tmp_path / "runs", transport=transport)
    assert len(transport.calls) == 2


def test_write_ahead_attempt_survives_interruption_without_repeating_query(tmp_path, monkeypatch):
    manifest_path, _, _ = prepared(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-credential")
    calls = []
    def interrupted(*args):
        calls.append(args)
        raise KeyboardInterrupt()
    directory = tmp_path / "runs"
    with pytest.raises(KeyboardInterrupt):
        llm.run_model(manifest_path, "gpt4o_mini", directory, transport=interrupted)
    ledger = directory / "gpt4o_mini.attempts.jsonl"
    assert json.loads(ledger.read_text().splitlines()[0])["event"] == "attempted"
    transport = FixtureTransport()
    results = llm.run_model(manifest_path, "gpt4o_mini", directory, transport=transport)
    assert len(calls) == 1 and len(transport.calls) == 1
    assert [row["status"] for row in results] == ["attempted_no_result", "ok"]


@pytest.mark.parametrize("failure,status", [
    ("http", "http_error"), ("timeout", "timeout"), ("json", "invalid_response"),
])
def test_failed_attempts_remain_failures_and_are_never_retried(tmp_path, monkeypatch, failure, status):
    manifest_path, _, _ = prepared(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-credential")
    calls = []
    def transport(*args):
        calls.append(args)
        if failure == "http":
            return 429, '{"error":"rate limit"}'
        if failure == "timeout":
            raise socket.timeout("time limit")
        return 200, json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "```json\n{}\n```"}}]})
    directory = tmp_path / "runs"
    first = llm.run_model(manifest_path, "gpt4o_mini", directory, transport=transport)
    second = llm.run_model(manifest_path, "gpt4o_mini", directory, transport=transport)
    assert len(calls) == 2
    assert first == second
    assert all(row["status"] == status and "response" not in row for row in first)


def test_resume_rejects_changed_config_and_timeout(tmp_path, monkeypatch):
    manifest_path, _, config_path = prepared(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-credential")
    directory = tmp_path / "runs"
    llm.run_model(manifest_path, "gpt4o_mini", directory, transport=FixtureTransport())
    changed = json.loads(config_path.read_text())
    changed["models"]["gpt4o_mini"]["max_output_tokens"] = 101
    write_json(config_path, changed)
    with pytest.raises(ValueError, match="Configuration changed"):
        llm.run_model(manifest_path, "gpt4o_mini", directory, config_path=config_path, transport=FixtureTransport())
    with pytest.raises(ValueError, match="configuration changed"):
        llm.run_model(manifest_path, "gpt4o_mini", directory, timeout=121, transport=FixtureTransport())


def test_ledger_request_hash_corruption_prevents_another_request(tmp_path, monkeypatch):
    manifest_path, _, _ = prepared(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-credential")
    directory = tmp_path / "runs"
    llm.run_model(manifest_path, "gpt4o_mini", directory, transport=FixtureTransport())
    path = directory / "gpt4o_mini.attempts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["request_sha256"] = "changed"
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding="utf-8")
    transport = FixtureTransport()
    with pytest.raises(ValueError, match="request checksum"):
        llm.run_model(manifest_path, "gpt4o_mini", directory, transport=transport)
    assert transport.calls == []


def test_resealed_manifest_cannot_select_over_budget_functions(tmp_path):
    path, manifest, _ = prepared(tmp_path)
    manifest["samples"][0]["input_tokens"]["gpt4o_mini"] = 801
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = llm.digest(manifest)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="token budget"):
        llm.load_manifest(path)


def test_context_budget_reserves_the_configured_output_cap(tmp_path):
    test, config_path, counts_path, counts, _ = setup_files(tmp_path)
    config = json.loads(config_path.read_text())
    config["models"]["gpt4o_mini"]["input_token_limit"] = 1000
    write_json(config_path, config)
    spec = llm.token_inputs(test, tmp_path / "new-inputs.json", config_path=config_path, split="test")
    counts["config_sha256"] = spec["config_sha256"]
    counts["models"]["gpt4o_mini"]["model_sha256"] = spec["models"]["gpt4o_mini"]["model_sha256"]
    counts["records"][1]["input_tokens"]["gpt4o_mini"] = 901
    write_json(counts_path, counts)
    manifest = llm.prepare(test, counts_path, tmp_path / "manifest.json", config_path=config_path, split="test")
    assert manifest["selected_ids"] == ["stable-a"]
    assert "context_window:gpt4o_mini" in manifest["excluded"][0]["reasons"]


@pytest.mark.parametrize("failure", ["http", "exception", "text"])
def test_provider_echoed_credentials_are_removed_from_all_output_files(tmp_path, monkeypatch, failure):
    path, _, _ = prepared(tmp_path)
    secret = "sensitive-key-full"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    def transport(*args):
        if failure == "http":
            return 401, json.dumps({"error": "wrong key: " + secret})
        if failure == "exception":
            raise OSError("failed with " + secret)
        response = classification()
        response["reason"] = [secret]
        text = json.dumps(response).replace("s", "\\u0073")
        return 200, json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": text}}]})
    directory = tmp_path / "runs"
    results = llm.run_model(path, "gpt4o_mini", directory, transport=transport)
    for output in directory.iterdir():
        assert secret not in output.read_text(encoding="utf-8")
    llm.run_model(path, "gpt4o_mini", directory, transport=FixtureTransport())
    if failure == "text":
        assert results[0]["response"]["reason"] == ["[REDACTED]"]


@pytest.mark.parametrize("provider,extra", [
    ("openai_chat", "tool_calls"), ("openai_chat", "function_call"),
    ("anthropic_messages", "tool_use"), ("gemini_generate_content", "functionCall"),
    ("gemini_generate_content", "blockReason"),
])
def test_provider_tool_and_block_signals_cannot_be_hidden_by_valid_json(provider, extra):
    text = json.dumps(classification())
    if provider == "openai_chat":
        response = {"choices": [{"finish_reason": "stop", "message": {"content": text, extra: {"name": "tool"}}}]}
    elif provider == "anthropic_messages":
        response = {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}, {"type": extra}]}
    else:
        response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}
        if extra == "blockReason":
            response["promptFeedback"] = {"blockReason": "SAFETY"}
        else:
            response["candidates"][0]["content"]["parts"].append({extra: {"name": "tool"}})
    with pytest.raises(ValueError):
        llm._response_text(provider, response)


def test_resealed_manifest_cannot_exclude_an_eligible_function(tmp_path):
    path, manifest, _ = prepared(tmp_path)
    eligible = manifest["samples"].pop()
    manifest["selected_ids"].pop()
    manifest["excluded"].append({key: eligible[key] for key in (
        "sample_id", "code_sha256", "rendered_prompt_sha256", "input_tokens")})
    manifest["excluded"][-1]["reasons"] = []
    manifest["counts"] = {"selected": 1, "excluded": 1, "total": 2}
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = llm.digest(manifest)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="excluded IDs"):
        llm.load_manifest(path)


@pytest.mark.parametrize("value", [
    '{"label":true,"confidence":0.5,"reason":[],"key_snippets":[]}',
    '{"label":0,"label":1,"confidence":0.5,"reason":[],"key_snippets":[]}',
    '{"label":0,"confidence":NaN,"reason":[],"key_snippets":[]}',
    '{"label":0,"confidence":1.1,"reason":[],"key_snippets":[]}',
    '{"label":0,"confidence":0.5,"reason":["a","b","c","d"],"key_snippets":[]}',
    '{"label":0,"confidence":0.5,"reason":[],"key_snippets":[],"extra":1}',
])
def test_classification_json_is_strict(value):
    with pytest.raises(ValueError):
        llm.parse_classification(value)


def prediction_files(tmp_path, manifest):
    result = {}
    for key in llm.MODEL_KEYS:
        rows = [{"sample_id": sample["sample_id"], "model_key": key, "status": "ok",
                 "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
                 "code_sha256": sample["code_sha256"], "rendered_prompt_sha256": sample["rendered_prompt_sha256"],
                 "response": classification(1)} for sample in manifest["samples"]]
        result[key] = write_json(tmp_path / f"{key}.json", rows)
    result["sp4cpg"] = write_json(tmp_path / "sp4cpg.json", [
        {"sample_id": sample["sample_id"], "code_sha256": sample["code_sha256"], "prediction": sample["label"]}
        for sample in manifest["samples"]])
    return result


def test_evaluate_requires_same_ids_and_computes_actual_binary_metrics(tmp_path):
    manifest_path, manifest, _ = prepared(tmp_path)
    paths = prediction_files(tmp_path, manifest)
    result = llm.evaluate(manifest_path, paths, tmp_path / "metrics.json")
    assert result["complete"] is True
    assert result["metrics"]["sp4cpg"]["f1"] == 1
    score = result["metrics"]["gpt4o_mini"]
    assert score["tp"] == 1 and score["fp"] == 1 and score["fn"] == 0
    assert score["precision"] == .5 and score["recall"] == 1 and score["f1"] == pytest.approx(2./3.)


@pytest.mark.parametrize("change", ["missing", "extra", "error", "hash", "duplicate"])
def test_any_unmatched_or_failed_prediction_blocks_all_complete_metrics(tmp_path, change):
    manifest_path, manifest, _ = prepared(tmp_path)
    paths = prediction_files(tmp_path, manifest)
    path = paths["gpt4o_mini"]
    rows = json.loads(path.read_text())
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append(dict(rows[0], sample_id="extra"))
    elif change == "error":
        rows[0]["status"] = "invalid_response"
    elif change == "hash":
        rows[0]["code_sha256"] = "different"
    else:
        rows.append(copy.deepcopy(rows[0]))
    write_json(path, rows)
    result = llm.evaluate(manifest_path, paths, tmp_path / "metrics.json")
    assert result["complete"] is False
    assert result["metrics"] is None
    assert result["errors"]
