
from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_function_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw_bytes = path.read_bytes()
    fingerprint = hashlib.sha256(raw_bytes).hexdigest()
    raw = raw_bytes.decode("utf-8-sig").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(data, dict):
        # Support common wrappers: {"data": [...]}, {"samples": [...]}.
        for key in ("data", "samples", "items", "functions", "records"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array, JSONL file, single JSON object, or a dict containing a JSON array.")
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        code = item.get("func") or item.get("code") or item.get("source") or item.get("source_code") or item.get("function") or item.get("func_before") or ""
        if not isinstance(code, str) or not code.strip():
            continue
        target = item.get("target", item.get("label", item.get("y", item.get("vulnerable", None))))
        try:
            target = int(target) if target is not None else None
        except Exception:
            target = None
        rows.append({
            "sample_id": str(item.get("sample_id", item.get("id", f"generated:{fingerprint}:{i + 1}"))),
            "function_name": item.get("function_name") or item.get("name") or guess_function_name(code),
            "file": item.get("file") or item.get("path") or "",
            "func": code,
            "target": target,
        })
    return rows

def guess_function_name(code: str) -> str:
    m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{", code)
    return m.group(1) if m else "unknown_function"


VULN_PATTERNS = [
    (r"\bstrcpy\s*\(", 0.78, "strcpy call"),
    (r"\bstrcat\s*\(", 0.76, "strcat call"),
    (r"\bsprintf\s*\(", 0.74, "sprintf call"),
    (r"\bgets\s*\(", 0.90, "gets call"),
    (r"\bmemcpy\s*\(", 0.58, "memcpy call"),
    (r"\bmemmove\s*\(", 0.54, "memmove call"),
    (r"\bmalloc\s*\(", 0.40, "malloc allocation"),
    (r"\bfree\s*\(", 0.42, "free call"),
    (r"\bscanf\s*\(", 0.56, "scanf call"),
    (r"\bsystem\s*\(", 0.82, "system command execution"),
]
SAFE_PATTERNS = [
    (r"\bstrncpy\s*\(", -0.12, "bounded copy"),
    (r"\bsnprintf\s*\(", -0.18, "bounded formatting"),
    (r"\bif\s*\([^)]*(?:len|size|count|n)\b", -0.10, "size check"),
]


def lexical_score(code: str) -> Tuple[float, List[str]]:
    score = 0.12
    evidence: List[str] = []
    for pat, weight, desc in VULN_PATTERNS:
        if re.search(pat, code):
            score += weight
            evidence.append(desc)
    for pat, weight, desc in SAFE_PATTERNS:
        if re.search(pat, code):
            score += weight
            evidence.append(desc)
    # Crude use-after-free clue: variable freed and then appears later.
    free_vars = re.findall(r"\bfree\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", code)
    for var in free_vars:
        after = code.split(f"free({var}", 1)[-1]
        if re.search(rf"\b{re.escape(var)}\b\s*(?:\[|->|=|\))", after):
            score += 0.35
            evidence.append(f"possible use after free: {var}")
    score = 1.0 / (1.0 + math.exp(-2.4 * (score - 0.52)))
    return max(0.01, min(0.99, score)), evidence or ["no strong lexical indicator"]


def fallback_predictions(rows: List[Dict[str, Any]], source: str = "fallback_lexical") -> List[Dict[str, Any]]:
    preds: List[Dict[str, Any]] = []
    for row in rows:
        prob, evidence = lexical_score(row["func"])
        pred = 1 if prob >= 0.5 else 0
        preds.append({
            "sample_id": row["sample_id"],
            "function_name": row["function_name"],
            "file": row.get("file", ""),
            "prediction": pred,
            "vulnerability_probability": round(prob, 6),
            "confidence": round(prob if pred else 1 - prob, 6),
            "true_label": row.get("target"),
            "risk": "high" if prob >= 0.80 else "medium" if prob >= 0.50 else "low",
            "validation": "model_only",
            "detector": source,
            "evidence": "; ".join(evidence),
        })
    return preds


def compute_basic_metrics(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [p.get("true_label") for p in preds if p.get("true_label") is not None]
    y_pred = [p.get("prediction") for p in preds if p.get("true_label") is not None]
    if not y_true:
        return {"mode": "prediction_only", "samples": len(preds)}
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    acc = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "mode": "evaluated",
        "samples": len(preds),
        "accuracy": round(acc, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_rows(rows: List[Dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    import csv
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = sorted({k for r in rows for k in r.keys()}) or ["sample_id", "prediction"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
