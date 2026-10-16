"""Explicit sample alignment and measured-result bookkeeping for revision experiments."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
from numbers import Integral
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch(path, map_location="cpu"):
    """Read trusted local dataset/checkpoint files on both torch 1.12 and 2.x."""
    import torch
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def sample_records(samples, require_labels=True):
    records = []
    seen = set()
    for sample in samples:
        sample_id = getattr(sample, "sample_id", None)
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("Every sample needs a unique nonempty string sample_id")
        seen.add(sample_id)
        if getattr(sample, "schema_version", None) != 2:
            raise ValueError(f"{sample_id}: rebuild this sample with schema version 2")
        y = getattr(sample, "y", None)
        if (y is None and require_labels) or (y is not None and (y.numel() != 1 or y.item() not in (0, 1))):
            raise ValueError(f"{sample_id}: a single binary label is required")
        count = getattr(sample, "original_node_count", None)
        if type(count) is not int or count < 1:
            raise ValueError(f"{sample_id}: original_node_count must be a positive integer")
        if sample.num_nodes < 1 or sample.x.ndim != 2 or not sample.x.isfinite().all():
            raise ValueError(f"{sample_id}: empty graph or nonfinite node features")
        records.append({"sample_id": sample_id, "label": int(y.item()) if y is not None else None, "original_node_count": count})
    if not records:
        raise ValueError("Empty dataset")
    return sorted(records, key=lambda r: r["sample_id"])


def membership_fingerprint(records):
    """Representation-independent fingerprint: IDs, labels and ORIGINAL CPG sizes."""
    payload = json.dumps(sorted(records, key=lambda r: r["sample_id"]), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_dataset(path, require_labels=True):
    value = load_torch(path)
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("Unversioned dataset: regenerate with preprocess/dot_embedding.py (schema v2)")
    if value.get("representation") not in ("CPG", "HCPG"):
        raise ValueError("Dataset representation must be CPG or HCPG")
    samples = value.get("samples", [])
    sample_records(samples, require_labels=require_labels)
    if len({getattr(sample, "y", None) is not None for sample in samples}) > 1:
        raise ValueError("Do not mix labeled and unlabeled samples in one inference dataset")
    if any(getattr(sample, "representation", None) != value["representation"] for sample in samples):
        raise ValueError("Per-sample representation differs from dataset metadata")
    widths = {int(sample.x.size(1)) for sample in samples}
    if len(widths) != 1:
        raise ValueError("Node feature widths differ across samples")
    return value


def make_split(records, seed):
    """Create a NEW reproducible 8:1:1 split; never label it a recovered paper split."""
    ids = sorted(r["sample_id"] for r in records)
    if len(ids) < 10 or len(set(ids)) != len(ids):
        raise ValueError("A new 8:1:1 split requires at least 10 unique samples")
    random.Random(seed).shuffle(ids)
    train_n, val_n = int(0.8 * len(ids)), int(0.1 * len(ids))
    return {
        "schema_version": 2, "source": "new_generated_split", "historical_paper_split": False,
        "split_seed": seed, "membership_sha256": membership_fingerprint(records),
        "train": ids[:train_n], "validation": ids[train_n:train_n + val_n], "test": ids[train_n + val_n:],
    }


def validate_split(manifest, records):
    expected = {r["sample_id"] for r in records}
    partitions = validate_id_partition(manifest)
    flattened = [i for values in partitions.values() for i in values]
    if set(flattened) != expected:
        raise ValueError("Split must cover exactly the dataset sample IDs; no implicit filtering")
    if manifest.get("membership_sha256") != membership_fingerprint(records):
        raise ValueError("Split fingerprint differs: IDs, labels or original CPG sizes changed")
    return partitions


def validate_id_partition(manifest):
    """Check split structure even when callers have only function text (no graph sizes)."""
    flattened = []
    for name in ("train", "validation", "test"):
        ids = manifest.get(name)
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            raise ValueError(f"Split {name} must contain sample IDs")
        flattened.extend(ids)
    if len(flattened) != len(set(flattened)):
        raise ValueError("Split contains duplicate or overlapping sample IDs")
    return {name: list(manifest[name]) for name in ("train", "validation", "test")}


def classification_metrics(rows):
    if not rows:
        return {"samples": 0, "precision": None, "recall": None, "f1": None, "accuracy": None,
                "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    pairs = [(r["label"], r["prediction"]) for r in rows]
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v not in (0, 1) for pair in pairs for v in pair):
        raise ValueError("Metrics require binary labels and predictions")
    tp = sum(t == p == 1 for t, p in pairs)
    tn = sum(t == p == 0 for t, p in pairs)
    fp = sum(t == 0 and p == 1 for t, p in pairs)
    fn = sum(t == 1 and p == 0 for t, p in pairs)
    # Match sklearn's zero_division=0 for classification; structural validation uses N/A instead.
    return {"samples": len(rows), "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate_scopes(rows):
    scopes = {"complete_test": rows,
              "original_cpg_gt_500": [r for r in rows if r["original_node_count"] > 500]}
    bounds = [(1, 100), (101, 300), (301, 500), (501, 1000), (1001, None)]
    for lower, upper in bounds:
        scopes[f"nodes_{lower}_{upper or 'plus'}"] = [r for r in rows if r["original_node_count"] >= lower
                                                     and (upper is None or r["original_node_count"] <= upper)]
    return {name: {**classification_metrics(group), "sample_ids": [r["sample_id"] for r in group]}
            for name, group in scopes.items()}


def mean_run_metrics(summaries):
    if not summaries:
        raise ValueError("No run summaries")
    first = summaries[0]
    for run in summaries:
        if run.get("result_kind") != "measured_run":
            raise ValueError("Only measured run summaries can be aggregated")
        for key in ("membership_sha256", "split_sha256", "dataset_sha256", "model", "representation"):
            if run[key] != first[key]:
                raise ValueError(f"Runs differ in {key}")
        if run["model_config"] != first["model_config"]:
            raise ValueError("Runs differ in model configuration")
        left = {k: v for k, v in run["training_config"].items() if k != "seed"}
        right = {k: v for k, v in first["training_config"].items() if k != "seed"}
        if left != right:
            raise ValueError("Runs differ in training configuration")
    if len({r["seed"] for r in summaries}) != len(summaries):
        raise ValueError("Duplicate training seeds")
    scopes = {}
    for scope in first["evaluation"]:
        expected_ids = first["evaluation"][scope]["sample_ids"]
        if any(run["evaluation"][scope]["sample_ids"] != expected_ids for run in summaries):
            raise ValueError(f"Test records differ in {scope}")
        entry = {"samples": len(expected_ids), "sample_ids": expected_ids}
        for metric in ("precision", "recall", "f1", "accuracy"):
            values = [run["evaluation"][scope][metric] for run in summaries]
            entry[metric + "_mean"] = sum(values) / len(values) if all(v is not None for v in values) else None
            entry[metric + "_sample_std"] = (math.sqrt(sum((v - entry[metric + "_mean"]) ** 2 for v in values)
                                                       / (len(values) - 1))
                                                 if len(values) > 1 and all(v is not None for v in values) else None)
        scopes[scope] = entry
    return {"result_kind": "measured_seed_aggregate", "model": first["model"],
            "representation": first["representation"], "seeds": [r["seed"] for r in summaries],
            "aggregation": "arithmetic mean of per-seed P/R/F1, not F1 reconstructed from mean P/R",
            "evaluation": scopes}
