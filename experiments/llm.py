"""Prepare a shared test intersection, request each function once, and score it."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid


BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "llm_protocol.json"
PROMPT_PATH = BASE / "prompts" / "two_shot.txt"
MODEL_KEYS = ("gpt4o_mini", "claude_sonnet4", "gemini25_pro", "deepseek_coder67b",
              "llama31_8b", "qwen25_coder7b")
PROVIDERS = {"openai_chat", "openai_compatible", "anthropic_messages", "gemini_generate_content"}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def text_sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(text):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)


def _read_json(path):
    return _strict_json(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _rows(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        parsed = _strict_json(text)
    except (ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, list):
        return parsed, None
    if isinstance(parsed, dict) and isinstance(parsed.get("samples"), list):
        return parsed["samples"], parsed.get("split")
    if isinstance(parsed, dict):
        return [parsed], None
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                rows.append(_strict_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid JSONL at line {number}: {exc}") from exc
    if not rows:
        raise ValueError("Input contains no records")
    return rows, None


def _nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configure a nonempty {field}")
    return value


def load_config(path=None):
    value = _read_json(path or CONFIG_PATH) if not isinstance(path, dict) else _strict_json(_json(path))
    if set(value) != {"schema_version", "dataset", "temperature", "models"}:
        raise ValueError("Configuration requires schema_version, dataset, temperature and models only")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["dataset"] != "PrimeVul" or type(value["temperature"]) not in (int, float) or value["temperature"] != 0:
        raise ValueError("Expected schema_version=1, dataset=PrimeVul and temperature=0")
    if not isinstance(value["models"], dict) or set(value["models"]) != set(MODEL_KEYS):
        raise ValueError("Configure all six model keys")
    base_fields = {"display_name", "provider", "endpoint", "model_id", "api_key_env", "api_version",
                   "context_window", "input_token_limit", "max_output_tokens", "tokenizer"}
    for key, model in value["models"].items():
        if not isinstance(model, dict) or model.get("provider") not in PROVIDERS:
            raise ValueError(f"Unknown provider for {key}")
        openai = model["provider"] in {"openai_chat", "openai_compatible"}
        if set(model) != base_fields | ({"output_token_field"} if openai else set()):
            raise ValueError(f"Unsupported or missing configuration fields for {key}")
        for field in ("display_name", "model_id", "endpoint"):
            _nonempty(model[field], f"models.{key}.{field}")
        url = urllib.parse.urlsplit(model["endpoint"].replace("{model_id}", "model"))
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError(f"Invalid endpoint for {key}; credentials belong in environment variables")
        if model["provider"] == "gemini_generate_content" and model["endpoint"].count("{model_id}") != 1:
            raise ValueError("Gemini endpoint requires one {model_id} component")
        if openai and model["output_token_field"] not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"Invalid output_token_field for {key}")
        for field in ("context_window", "input_token_limit", "max_output_tokens"):
            if type(model[field]) is not int or model[field] < 1:
                raise ValueError(f"Configure a positive integer models.{key}.{field}")
        if model["input_token_limit"] > model["context_window"] or model["max_output_tokens"] >= model["context_window"]:
            raise ValueError(f"Invalid context/input/output limits for {key}")
        env = model["api_key_env"]
        if env is not None and (not isinstance(env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env)):
            raise ValueError(f"api_key_env must name an environment variable for {key}")
        if model["provider"] != "openai_compatible" and not env:
            raise ValueError(f"Configure api_key_env for {key}")
        if model["provider"] == "anthropic_messages":
            _nonempty(model["api_version"], f"models.{key}.api_version")
        elif model["api_version"] is not None:
            raise ValueError(f"api_version is only used by anthropic_messages: {key}")
        tokenizer = model["tokenizer"]
        if not isinstance(tokenizer, dict) or set(tokenizer) != {"id", "version"}:
            raise ValueError(f"Configure tokenizer.id and tokenizer.version for {key}")
        for field in tokenizer:
            _nonempty(tokenizer[field], f"models.{key}.tokenizer.{field}")
    return value


def load_prompt():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if text.count("{CODE}") != 1:
        raise ValueError("Prompt requires exactly one {CODE} substitution")
    return text


def render_prompt(code, template=None):
    return (load_prompt() if template is None else template).replace("{CODE}", code)


def read_test(path, split=None):
    if split not in (None, "test"):
        raise ValueError("Only the test split is accepted")
    rows, declared = _rows(path)
    if declared not in (None, "test"):
        raise ValueError("Input declares a non-test split")
    records, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every function must be a JSON object")
        row_split = row.get("split", declared)
        if row_split not in (None, "test") or (row_split is None and split != "test"):
            raise ValueError("Declare split=test on every record or pass --split test for the complete test file")
        sample_id = next((row[field] for field in ("sample_id", "id", "idx") if field in row), None)
        if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)) or not str(sample_id).strip():
            raise ValueError("Each function needs a stable sample_id, id or idx")
        sample_id = str(sample_id)
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID: {sample_id!r}")
        seen.add(sample_id)
        code = row.get("func", row.get("code"))
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"Missing function text for {sample_id!r}")
        if "func" in row and "code" in row and row["func"] != row["code"]:
            raise ValueError(f"Conflicting function text for {sample_id!r}")
        label = row.get("target", row.get("label"))
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"A binary integer target/label is required for {sample_id!r}")
        if "target" in row and "label" in row and (type(row["label"]) is not int or row["label"] != label):
            raise ValueError(f"Conflicting labels for {sample_id!r}")
        records.append({"sample_id": sample_id, "func": code, "label": label,
                        "code_sha256": text_sha256(code)})
    if not records:
        raise ValueError("Empty test split")
    return sorted(records, key=lambda record: record["sample_id"])


def request_body(model, prompt):
    provider = model["provider"]
    if provider in {"openai_chat", "openai_compatible"}:
        return {"model": model["model_id"], "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, model["output_token_field"]: model["max_output_tokens"]}
    if provider == "anthropic_messages":
        return {"model": model["model_id"], "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": model["max_output_tokens"]}
    return {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": model["max_output_tokens"]}}


def token_inputs(test_path, output_path, *, config_path=None, split=None):
    config, prompt = load_config(config_path), load_prompt()
    records = read_test(test_path, split)
    result = {"schema_version": 1, "config_sha256": digest(config), "prompt_sha256": text_sha256(prompt),
              "models": {key: {"model_sha256": digest(model), "tokenizer": model["tokenizer"]}
                         for key, model in config["models"].items()}, "records": []}
    for record in records:
        rendered = render_prompt(record["func"], prompt)
        result["records"].append({"sample_id": record["sample_id"], "code_sha256": record["code_sha256"],
                                  "rendered_prompt_sha256": text_sha256(rendered), "rendered_prompt": rendered,
                                  "requests": {key: request_body(model, rendered) for key, model in config["models"].items()}})
    _write_json(output_path, result)
    return result


def _budget_reasons(tokens, config):
    if not isinstance(tokens, dict) or set(tokens) != set(MODEL_KEYS) or any(type(value) is not int or value < 1 for value in tokens.values()):
        raise ValueError("Every function requires a positive integer token count for each model")
    reasons = []
    for key in MODEL_KEYS:
        model = config["models"][key]
        if tokens[key] > model["input_token_limit"]:
            reasons.append(f"input_token_limit:{key}")
        if tokens[key] + model["max_output_tokens"] > model["context_window"]:
            reasons.append(f"context_window:{key}")
    return reasons


def prepare(test_path, token_counts_path, output_path, *, config_path=None, split=None):
    config, prompt = load_config(config_path), load_prompt()
    records = read_test(test_path, split)
    counts = _read_json(token_counts_path)
    if type(counts.get("schema_version")) is not int or counts.get("schema_version") != 1 or counts.get("config_sha256") != digest(config) or counts.get("prompt_sha256") != text_sha256(prompt):
        raise ValueError("Token counts do not match the configuration and full prompt template")
    if set(counts.get("models", {})) != set(MODEL_KEYS):
        raise ValueError("Token counts require all six models")
    for key, model in config["models"].items():
        counter = counts["models"][key]
        if counter.get("model_sha256") != digest(model) or counter.get("tokenizer") != model["tokenizer"]:
            raise ValueError(f"Token counter configuration mismatch for {key}")
        if counter.get("method") not in {"provider", "tokenizer"} or counter.get("includes_request_overhead") is not True:
            raise ValueError(f"Exact provider/tokenizer counts including request overhead are required for {key}")
    by_id = {}
    for entry in counts.get("records", []):
        sample_id = entry.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in by_id:
            raise ValueError("Token count records require distinct string sample IDs")
        by_id[sample_id] = entry
    if set(by_id) != {record["sample_id"] for record in records}:
        raise ValueError("Token counts must cover exactly the complete test IDs")
    selected, excluded = [], []
    for record in records:
        rendered_hash = text_sha256(render_prompt(record["func"], prompt))
        count = by_id[record["sample_id"]]
        if count.get("code_sha256") != record["code_sha256"] or count.get("rendered_prompt_sha256") != rendered_hash:
            raise ValueError(f"Token counts are bound to different function/prompt content: {record['sample_id']}")
        tokens = count.get("input_tokens", {})
        reasons = _budget_reasons(tokens, config)
        entry = dict(record, rendered_prompt_sha256=rendered_hash, input_tokens=tokens)
        if reasons:
            excluded.append({"sample_id": record["sample_id"], "code_sha256": record["code_sha256"],
                             "rendered_prompt_sha256": rendered_hash, "input_tokens": tokens, "reasons": reasons})
        else:
            selected.append(entry)
    if not selected:
        raise ValueError("No test functions fit every model's input and context limits")
    manifest = {"schema_version": 1, "dataset": "PrimeVul", "split": "test", "config": config,
                "config_sha256": digest(config), "prompt_sha256": text_sha256(prompt), "prompt_template": prompt,
                "test_sha256": hashlib.sha256(Path(test_path).read_bytes()).hexdigest(),
                "token_counts_sha256": hashlib.sha256(Path(token_counts_path).read_bytes()).hexdigest(),
                "model_keys": list(MODEL_KEYS), "selected_ids": [row["sample_id"] for row in selected],
                "samples": selected, "excluded": excluded,
                "counts": {"total": len(records), "selected": len(selected), "excluded": len(excluded)}}
    manifest["manifest_sha256"] = digest(manifest)
    _write_json(output_path, manifest)
    return manifest


def load_manifest(path):
    manifest = _read_json(path)
    sealed = dict(manifest)
    claimed = sealed.pop("manifest_sha256", None)
    if claimed != digest(sealed) or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1 or manifest.get("split") != "test" or manifest.get("dataset") != "PrimeVul":
        raise ValueError("Invalid manifest checksum or test split")
    config = load_config(manifest["config"])
    prompt = load_prompt()
    if manifest.get("config_sha256") != digest(config) or manifest.get("prompt_template") != prompt or manifest.get("prompt_sha256") != text_sha256(prompt):
        raise ValueError("Manifest configuration or prompt has changed")
    ids = manifest.get("selected_ids", [])
    if not ids or any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)) or ids != [row["sample_id"] for row in manifest["samples"]] or manifest.get("model_keys") != list(MODEL_KEYS):
        raise ValueError("Manifest selected IDs are inconsistent")
    excluded_ids = [row["sample_id"] for row in manifest["excluded"]]
    if any(not isinstance(value, str) or not value for value in excluded_ids) or len(excluded_ids) != len(set(excluded_ids)) or set(ids) & set(excluded_ids):
        raise ValueError("Manifest selected/excluded IDs overlap or repeat")
    if _json(manifest.get("counts")) != _json({"selected": len(ids), "excluded": len(excluded_ids), "total": len(ids) + len(excluded_ids)}):
        raise ValueError("Manifest selection counts are inconsistent")
    for row in manifest["samples"]:
        if row["code_sha256"] != text_sha256(row["func"]) or row["rendered_prompt_sha256"] != text_sha256(render_prompt(row["func"], prompt)):
            raise ValueError("Manifest function or rendered prompt checksum mismatch")
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Manifest has an invalid binary label")
        if _budget_reasons(row.get("input_tokens", {}), config):
            raise ValueError("Manifest selects a function outside a model's token budget")
    for row in manifest["excluded"]:
        reasons = _budget_reasons(row.get("input_tokens", {}), config)
        if not reasons or row.get("reasons") != reasons:
            raise ValueError("Manifest excluded IDs must fail their recorded token budgets")
    return manifest


def parse_classification(text):
    value = _strict_json(text)
    if not isinstance(value, dict) or set(value) != {"label", "confidence", "reason", "key_snippets"}:
        raise ValueError("Response must contain exactly label/confidence/reason/key_snippets")
    if type(value["label"]) is not int or value["label"] not in (0, 1):
        raise ValueError("Response label must be integer 0 or 1")
    confidence = value["confidence"]
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Response confidence must be finite and in [0,1]")
    for key in ("reason", "key_snippets"):
        if not isinstance(value[key], list) or len(value[key]) > 3 or any(not isinstance(item, str) for item in value[key]):
            raise ValueError(f"Response {key} requires at most three strings")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _http_post(url, headers, payload, timeout):
    request = urllib.request.Request(url, data=_json(payload).encode("utf-8"), headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _response_text(provider, response):
    if not isinstance(response, dict) or response.get("error"):
        raise ValueError("Provider returned an error or a non-object response")
    if provider in {"openai_chat", "openai_compatible"}:
        if len(response["choices"]) != 1:
            raise ValueError("Expected one response choice")
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop" or any(choice["message"].get(field) for field in ("refusal", "tool_calls", "function_call")):
            raise ValueError(f"Incomplete or refused response: {choice['finish_reason']}")
        return choice["message"]["content"]
    if provider == "anthropic_messages":
        if response["stop_reason"] != "end_turn":
            raise ValueError(f"Incomplete response: {response['stop_reason']}")
        if not isinstance(response["content"], list) or not response["content"] or any(block.get("type") != "text" for block in response["content"]):
            raise ValueError("Expected text blocks without tools or other content")
        return "".join(block["text"] for block in response["content"])
    if response.get("promptFeedback", {}).get("blockReason") or len(response.get("candidates", [])) != 1 or response["candidates"][0].get("finishReason") != "STOP":
        raise ValueError("Incomplete or blocked Gemini response")
    candidate = response["candidates"][0]
    if any(rating.get("blocked") for rating in candidate.get("safetyRatings", [])):
        raise ValueError("Blocked Gemini response")
    parts = candidate["content"]["parts"]
    if not isinstance(parts, list) or not parts or any(
        not isinstance(part, dict) or "text" not in part or part.get("thought")
        or set(part) - {"text", "thought", "thoughtSignature"} for part in parts
    ):
        raise ValueError("Expected text parts without function calls or other content")
    return "".join(part["text"] for part in parts)


def _redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {key.replace(secret, "[REDACTED]"): _redact(item, secret) for key, item in value.items()}
    return value


def _safe_provider_text(text, secret):
    if not secret:
        return text
    try:
        parsed = _strict_json(text)
        safe = _redact(parsed, secret)
        if _json(safe) != _json(parsed):
            return _json(safe)
    except ValueError:
        pass
    return text.replace(secret, "[REDACTED]").replace(json.dumps(secret)[1:-1], "[REDACTED]")


@contextmanager
def _directory_lock(directory):
    with (directory / ".lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another process is using the output directory") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _append_event(path, event):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_json(event) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_ledger(path, manifest, model_key):
    entries = {}
    if not path.exists():
        return entries
    samples = {row["sample_id"]: row for row in manifest["samples"]}
    model = manifest["config"]["models"][model_key]
    for line in path.read_text(encoding="utf-8").splitlines():
        event = _strict_json(line)
        sample_id = event.get("sample_id")
        if event.get("manifest_sha256") != manifest["manifest_sha256"] or event.get("model_key") != model_key or sample_id not in manifest["selected_ids"]:
            raise ValueError("Attempt ledger does not match this manifest/model")
        request_hash = digest(request_body(model, render_prompt(samples[sample_id]["func"], manifest["prompt_template"])))
        if event.get("request_sha256") != request_hash or not isinstance(event.get("attempt_id"), str) or not event["attempt_id"]:
            raise ValueError("Attempt ledger request checksum or attempt ID mismatch")
        if event.get("event") == "attempted":
            if sample_id in entries:
                raise ValueError("Duplicate attempted sample/model pair in ledger")
            entries[sample_id] = {"attempt": event, "result": None}
        elif event.get("event") == "finished":
            if sample_id not in entries or entries[sample_id]["result"] is not None or event["attempt_id"] != entries[sample_id]["attempt"]["attempt_id"]:
                raise ValueError("Unmatched or duplicate ledger completion")
            result = event.get("result")
            expected = {"sample_id": sample_id, "model_key": model_key,
                        "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
                        "code_sha256": samples[sample_id]["code_sha256"],
                        "rendered_prompt_sha256": samples[sample_id]["rendered_prompt_sha256"]}
            if not isinstance(result, dict) or any(result.get(field) != value for field, value in expected.items()):
                raise ValueError("Attempt ledger result checksum/identity mismatch")
            if result.get("status") not in {"ok", "invalid_response", "http_error", "timeout", "transport_error"}:
                raise ValueError("Unknown attempt result status")
            if result["status"] == "ok" and _json(parse_classification(result.get("raw_text", ""))) != _json(result.get("response")):
                raise ValueError("Stored classification disagrees with its response text")
            entries[sample_id]["result"] = event["result"]
        else:
            raise ValueError("Unknown attempt ledger event")
    return entries


def run_model(manifest_path, model_key, output_dir, *, config_path=None, timeout=120., transport=None):
    manifest = load_manifest(manifest_path)
    if model_key not in MODEL_KEYS:
        raise ValueError("Unknown model key")
    if config_path is not None and digest(load_config(config_path)) != manifest["config_sha256"]:
        raise ValueError("Configuration changed; refusing to resume")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    model = manifest["config"]["models"][model_key]
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(model["api_key_env"], "") if model["api_key_env"] else None
    if model["api_key_env"] and not key:
        raise ValueError(f"Set environment variable {model['api_key_env']} before running")
    provider = model["provider"]
    if provider == "anthropic_messages":
        headers.update({"x-api-key": key, "anthropic-version": model["api_version"]})
    elif provider == "gemini_generate_content":
        headers["x-goog-api-key"] = key
    elif key is not None:
        headers["Authorization"] = "Bearer " + key
    url = model["endpoint"].replace("{model_id}", urllib.parse.quote(model["model_id"], safe=""))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    header = {"schema_version": 1, "manifest_sha256": manifest["manifest_sha256"],
              "config_sha256": manifest["config_sha256"], "prompt_sha256": manifest["prompt_sha256"], "timeout": timeout}
    ledger_path = directory / f"{model_key}.attempts.jsonl"
    with _directory_lock(directory):
        header_path = directory / "run.json"
        if header_path.exists():
            if _json(_read_json(header_path)) != _json(header):
                raise ValueError("Run configuration changed; refusing to resume")
        else:
            _write_json(header_path, header)
        entries = _read_ledger(ledger_path, manifest, model_key)
        for sample in manifest["samples"]:
            sample_id = sample["sample_id"]
            if sample_id in entries:
                continue
            payload = request_body(model, render_prompt(sample["func"], manifest["prompt_template"]))
            attempt = {"event": "attempted", "sample_id": sample_id, "model_key": model_key,
                       "manifest_sha256": manifest["manifest_sha256"], "attempt_id": uuid.uuid4().hex,
                       "request_sha256": digest(payload)}
            _append_event(ledger_path, attempt)
            result = {"sample_id": sample_id, "model_key": model_key,
                      "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
                      "code_sha256": sample["code_sha256"], "rendered_prompt_sha256": sample["rendered_prompt_sha256"]}
            try:
                status, body = (transport or _http_post)(url, headers, payload, timeout)
                result["http_status"] = status
                if status < 200 or status >= 300:
                    result.update(status="http_error", error=_safe_provider_text(body, key))
                else:
                    try:
                        raw = _response_text(provider, _strict_json(body))
                        if not isinstance(raw, str):
                            raise ValueError("Response text is not a string")
                        raw = _safe_provider_text(raw, key)
                        result["raw_text"] = raw
                        result.update(status="ok", response=parse_classification(raw))
                    except (ValueError, KeyError, TypeError) as exc:
                        result.update(status="invalid_response", error=_safe_provider_text(str(exc), key))
            except (TimeoutError, socket.timeout) as exc:
                result.update(status="timeout", error=_safe_provider_text(str(exc), key))
            except Exception as exc:
                timeout_error = isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout))
                result.update(status="timeout" if timeout_error else "transport_error", error=_safe_provider_text(f"{type(exc).__name__}: {exc}", key))
            _append_event(ledger_path, dict(attempt, event="finished", result=result))
            entries[sample_id] = {"attempt": attempt, "result": result}
        results = []
        for sample in manifest["samples"]:
            entry = entries[sample["sample_id"]]
            results.append(entry["result"] or {
                "sample_id": sample["sample_id"], "model_key": model_key, "manifest_sha256": manifest["manifest_sha256"],
                "config_sha256": manifest["config_sha256"], "code_sha256": sample["code_sha256"],
                "rendered_prompt_sha256": sample["rendered_prompt_sha256"], "status": "attempted_no_result"})
        prediction_path = directory / f"{model_key}.predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in results:
                stream.write(_json(row) + "\n")
    return results


def evaluate(manifest_path, predictions, output_path):
    manifest = load_manifest(manifest_path)
    expected = set(MODEL_KEYS) | {"sp4cpg"}
    errors, labels = [], {}
    samples = {row["sample_id"]: row for row in manifest["samples"]}
    if set(predictions) != expected:
        errors.append({"error": "Prediction files must contain all six models and sp4cpg"})
    for key, path in predictions.items():
        if key not in expected:
            errors.append({"model_key": key, "error": "Unknown model key"})
            continue
        try:
            rows, _ = _rows(path)
            by_id = {}
            for row in rows:
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or sample_id in by_id:
                    raise ValueError("Prediction IDs must be distinct strings")
                by_id[sample_id] = row
            if set(by_id) != set(samples):
                raise ValueError(f"Prediction ID mismatch; missing={sorted(set(samples)-set(by_id))}, extra={sorted(set(by_id)-set(samples))}")
            labels[key] = {}
            for sample_id, row in by_id.items():
                sample = samples[sample_id]
                if row.get("status", "ok") != "ok":
                    raise ValueError(f"{sample_id}: prediction status {row.get('status')}")
                if row.get("code_sha256") != sample["code_sha256"]:
                    raise ValueError(f"{sample_id}: function checksum mismatch")
                if key == "sp4cpg":
                    prediction = row.get("prediction")
                else:
                    if row.get("model_key") != key or row.get("manifest_sha256") != manifest["manifest_sha256"] or row.get("config_sha256") != manifest["config_sha256"] or row.get("rendered_prompt_sha256") != sample["rendered_prompt_sha256"]:
                        raise ValueError(f"{sample_id}: prediction protocol checksum mismatch")
                    prediction = parse_classification(_json(row.get("response")))["label"]
                if type(prediction) is not int or prediction not in (0, 1):
                    raise ValueError(f"{sample_id}: missing binary prediction")
                labels[key][sample_id] = prediction
        except (ValueError, OSError, TypeError, AttributeError) as exc:
            errors.append({"model_key": key, "error": str(exc)})
    metrics = None
    if not errors:
        metrics = {}
        for key in sorted(expected):
            tp = sum(row["label"] == 1 and labels[key][sample_id] == 1 for sample_id, row in samples.items())
            fp = sum(row["label"] == 0 and labels[key][sample_id] == 1 for sample_id, row in samples.items())
            fn = sum(row["label"] == 1 and labels[key][sample_id] == 0 for sample_id, row in samples.items())
            tn = len(samples) - tp - fp - fn
            precision = tp / (tp + fp) if tp + fp else 0.
            recall = tp / (tp + fn) if tp + fn else 0.
            metrics[key] = {"samples": len(samples), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                            "precision": precision, "recall": recall,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}
    result = {"schema_version": 1, "manifest_sha256": manifest["manifest_sha256"],
              "complete": not errors, "errors": errors, "metrics": metrics}
    _write_json(output_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("token-inputs", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--test", required=True, type=Path)
        command.add_argument("--split", choices=("test",))
        command.add_argument("--config", type=Path, default=CONFIG_PATH)
        command.add_argument("--output", required=True, type=Path)
        if name == "prepare":
            command.add_argument("--token-counts", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--model", choices=MODEL_KEYS, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--timeout", type=float, default=120.)
    score = commands.add_parser("evaluate")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--predictions", action="append", required=True, metavar="MODEL=PATH")
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "token-inputs":
            result = token_inputs(args.test, args.output, config_path=args.config, split=args.split)
            print(f"Exported {len(result['records'])} token-count inputs to {args.output}")
        elif args.command == "prepare":
            result = prepare(args.test, args.token_counts, args.output, config_path=args.config, split=args.split)
            print(_json(result["counts"]))
        elif args.command == "run":
            result = run_model(args.manifest, args.model, args.output_dir, config_path=args.config, timeout=args.timeout)
            print(_json({"attempted": len(result), "ok": sum(row["status"] == "ok" for row in result)}))
            return int(any(row["status"] != "ok" for row in result))
        else:
            predictions = {}
            for argument in args.predictions:
                key, separator, path = argument.partition("=")
                if not separator or not key or not path or key in predictions:
                    raise ValueError("Use each --predictions MODEL=PATH exactly once")
                predictions[key] = Path(path)
            result = evaluate(args.manifest, predictions, args.output)
            print(_json({"complete": result["complete"], "errors": result["errors"]}))
            return int(not result["complete"])
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"LLM protocol error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
