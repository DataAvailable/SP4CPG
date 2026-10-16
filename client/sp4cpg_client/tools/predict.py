from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def build_model(model_name: str, repo: Path, model_config: Dict[str, Any]):
    sys.path.insert(0, str(repo))
    from models.factory import build_model as factory
    return factory(model_name, **model_config)


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if any(str(r.get("detector", "")).startswith("fallback") for r in rows):
        return {"mode": "lexical_fallback_not_model_evaluation", "samples": len(rows)}
    labeled = [r for r in rows if r.get("true_label") is not None and str(r.get("true_label")) != ""]
    if not labeled:
        return {"mode": "prediction_only", "samples": len(rows)}
    y_true = [int(r["true_label"]) for r in labeled]
    y_pred = [int(r.get("prediction", 0)) for r in labeled]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "mode": "evaluated",
        "samples": len(rows),
        "labeled_samples": len(labeled),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_outputs(rows: List[Dict[str, Any]], output: Path, csv_path: Path, metrics_path: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = sorted({k for r in rows for k in r.keys()}) or ["sample_id", "prediction", "confidence"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(compute_metrics(rows), indent=2, ensure_ascii=False), encoding="utf-8")


def run_real_hgcn(args) -> List[Dict[str, Any]]:
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[hgcn] device={device}")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"HCPG dataset not found: {data_path}")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"HGCN checkpoint not found: {checkpoint}")

    from sp4cpg.experiment import load_torch, read_dataset
    dataset = read_dataset(data_path, require_labels=False)
    state = load_torch(checkpoint, map_location=device)
    if not isinstance(state, dict) or state.get("schema_version") != 2 or "model_config" not in state:
        raise ValueError("Unversioned checkpoint: rebuild v2 data and retrain; architecture cannot be inferred from raw weights")
    if state.get("model") != args.model or state.get("representation") != dataset["representation"]:
        raise ValueError("Checkpoint model/graph representation differs from inference request")
    if dataset["samples"][0].x.size(1) != state["model_config"]["in_channels"]:
        raise ValueError("Feature width differs from checkpoint")
    if state.get("dataset_metadata", {}).get("encoder") != dataset.get("metadata", {}).get("encoder"):
        raise ValueError("Node input / CodeBERT configuration differs from checkpoint")
    loader = DataLoader(dataset["samples"], batch_size=args.batch, shuffle=False, num_workers=0)
    model = build_model(args.model, repo, state["model_config"]).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    rows: List[Dict[str, Any]] = []
    source_rows = {}
    if getattr(args, "source_data", ""):
        try:
            from .codebert_common import load_function_dataset
        except ImportError:
            from codebert_common import load_function_dataset
        for row in load_function_dataset(Path(args.source_data)):
            if row["sample_id"] in source_rows:
                raise ValueError("Source dataset has duplicate sample IDs")
            source_rows[row["sample_id"]] = row
    idx_base = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            prob = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            y = getattr(batch, "y", None)
            batch_size = int(pred.numel())
            for i in range(batch_size):
                true_label = None
                if y is not None:
                    try:
                        true_label = int(y.view(-1)[i].detach().cpu().item())
                    except Exception:
                        true_label = None
                pred_i = int(pred[i].detach().cpu().item())
                confidence = float(prob[i, pred_i].detach().cpu().item())
                vuln_prob = float(prob[i, 1].detach().cpu().item()) if prob.shape[1] > 1 else confidence
                rows.append({
                    "sample_id": batch.sample_id[i],
                    "function_name": source_rows.get(batch.sample_id[i], {}).get("function_name", ""),
                    "file": source_rows.get(batch.sample_id[i], {}).get("file", ""),
                    "prediction": pred_i,
                    "vulnerability_probability": round(vuln_prob, 6),
                    "confidence": round(confidence, 6),
                    "true_label": true_label,
                    "risk": "high" if vuln_prob >= 0.80 else "medium" if vuln_prob >= 0.50 else "low",
                    "validation": "model_only",
                    "detector": args.model,
                })
            idx_base += batch_size
    return rows


def run_fallback(args, reason: Exception) -> List[Dict[str, Any]]:
    try:
        from .codebert_common import load_function_dataset, fallback_predictions
    except Exception:
        from codebert_common import load_function_dataset, fallback_predictions
    fallback_data = Path(args.fallback_data) if args.fallback_data else Path(args.data)
    rows = load_function_dataset(fallback_data)
    preds = fallback_predictions(rows, source="fallback_lexical_not_hgcn")
    for r in preds:
        r["fallback_reason"] = str(reason)
    return preds


def main():
    parser = argparse.ArgumentParser(description="SP4CPG HGCN trained-model inference helper")
    parser.add_argument("--repo", required=True, help="SP4CPG repository root")
    parser.add_argument("--model", required=True, choices=["TAF-HGCN", "HGNN", "GCN", "GAT", "GIN", "GraphSAGE", "GGNN", "HGCN"])
    parser.add_argument("--checkpoint", default="", help="Path to .pt checkpoint")
    parser.add_argument("--data", required=True, help="Path to version 2 embedded .pt dataset")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--csv", required=True, help="CSV output path")
    parser.add_argument("--metrics", default="", help="Optional metrics JSON path")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--allow-fallback", action="store_true", help="Enable explicitly marked lexical fallback; excluded from model evaluation")
    parser.add_argument("--fallback-data", default="", help="Raw JSON/JSONL dataset for fallback mode")
    parser.add_argument("--source-data", default="", help="Optional stable-ID source records for function/file metadata")
    args = parser.parse_args()

    try:
        rows = run_real_hgcn(args)
    except Exception as exc:
        if not args.allow_fallback:
            raise
        print(f"[hgcn:fallback] real HGCN prediction is unavailable: {exc}")
        print("[hgcn:fallback] generating marked fallback predictions so the client workflow can finish.")
        rows = run_fallback(args, exc)

    write_outputs(rows, Path(args.output), Path(args.csv), Path(args.metrics) if args.metrics else None)
    print(f"[hgcn] predictions={args.output}")
    print(f"[hgcn] csv={args.csv}")
    if args.metrics:
        print(f"[hgcn] metrics={args.metrics}")


if __name__ == "__main__":
    main()
