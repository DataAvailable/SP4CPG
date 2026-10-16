
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .codebert_common import load_function_dataset, fallback_predictions, compute_basic_metrics, write_rows
except Exception:
    from codebert_common import load_function_dataset, fallback_predictions, compute_basic_metrics, write_rows


def run_real_codebert(rows: List[Dict[str, Any]], model_name: str, checkpoint: str, max_length: int) -> List[Dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_source = checkpoint.strip() or model_name
    tokenizer_source = checkpoint.strip() or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds: List[Dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            enc = tokenizer(
                row["func"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            prob = F.softmax(logits, dim=-1)[0]
            vuln_prob = float(prob[1].detach().cpu().item())
            pred = int(prob.argmax().detach().cpu().item())
            preds.append({
                "sample_id": row["sample_id"],
                "function_name": row["function_name"],
                "file": row.get("file", ""),
                "prediction": pred,
                "vulnerability_probability": round(vuln_prob, 6),
                "confidence": round(float(prob[pred].detach().cpu().item()), 6),
                "true_label": row.get("target"),
                "risk": "high" if vuln_prob >= 0.80 else "medium" if vuln_prob >= 0.50 else "low",
                "validation": "model_only",
                "detector": "codebert",
                "evidence": "CodeBERT sequence classification",
            })
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="CodeBERT vulnerability prediction helper")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()

    rows = load_function_dataset(Path(args.data))
    print(f"[codebert] samples={len(rows)} model={args.model_name}")
    try:
        if args.allow_fallback and not args.checkpoint.strip():
            raise RuntimeError("No CodeBERT checkpoint was provided. Fallback is enabled, so the workflow will use explicitly marked lexical predictions.")
        preds = run_real_codebert(rows, args.model_name, args.checkpoint, args.max_length)
    except Exception as exc:
        if not args.allow_fallback:
            raise
        print(f"[codebert:fallback] real CodeBERT is unavailable: {exc}")
        print("[codebert:fallback] generating deterministic fallback predictions so the client workflow can finish.")
        preds = fallback_predictions(rows, source="fallback_lexical_not_codebert")

    write_rows(preds, Path(args.output), Path(args.csv))
    metrics = compute_basic_metrics(preds)
    if args.metrics:
        Path(args.metrics).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        Path(args.output).with_name("codebert_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[codebert] predictions={args.output}")
    print(f"[codebert] csv={args.csv}")


if __name__ == "__main__":
    main()
