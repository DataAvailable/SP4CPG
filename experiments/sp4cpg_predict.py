"""Predict the selected test functions with a trained TAF-HGCN checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
from torch_geometric.loader import DataLoader

from experiments.llm import load_manifest
from models.hgcn_models import TAFHGCN
from preprocess.graph_data import GraphData


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _id_map(path, samples):
    if path is None:
        return {row["sample_id"]: row["sample_id"] for row in samples}
    def distinct(pairs):
        value = {}
        for key, target in pairs:
            if key in value:
                raise ValueError(f"Duplicate dataset ID in graph map: {key!r}")
            value[key] = target
        return value
    mapping = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=distinct)
    if not isinstance(mapping, dict) or any(not isinstance(value, str) or not value.strip()
                                            for value in mapping.values()):
        raise ValueError("Graph ID map must contain nonempty string graph IDs")
    required = {row["sample_id"] for row in samples}
    if not required <= mapping.keys():
        raise ValueError(f"Graph ID map is missing selected IDs: {sorted(required - mapping.keys())}")
    selected = {sample_id: mapping[sample_id] for sample_id in required}
    if len(set(selected.values())) != len(selected):
        raise ValueError("Selected functions must have distinct graph IDs")
    return selected


def _select_data(path, samples, mapping):
    dataset = torch.load(path, map_location="cpu")
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Graph dataset must be a nonempty list of GraphData samples")
    indexed = {}
    for data in dataset:
        if not isinstance(data, GraphData) or data.schema_version != 2:
            raise ValueError("Graph dataset requires schema-version-2 GraphData samples")
        graph_id = getattr(data, "sample_id", None)
        if not isinstance(graph_id, str) or not graph_id:
            raise ValueError("Every graph needs a nonempty string sample_id")
        if graph_id in indexed:
            raise ValueError(f"Duplicate graph sample_id: {graph_id!r}")
        indexed[graph_id] = data
    selected, source_hashes = [], {}
    for row in samples:
        graph_id = mapping[row["sample_id"]]
        if graph_id not in indexed:
            raise ValueError(f"Missing graph for selected function {row['sample_id']!r}: {graph_id!r}")
        data = indexed[graph_id]
        if getattr(data, "representation", None) != "HCPG":
            raise ValueError(f"Selected graph must use the HCPG representation: {graph_id!r}")
        encoded = getattr(data, "canonical_json", None)
        if not isinstance(encoded, str):
            raise ValueError(f"Selected graph needs its complete canonical JSON: {graph_id!r}")
        canonical = json.loads(encoded)
        source_hash = canonical.get("metadata", {}).get("source_code_sha256")
        if (canonical.get("graph_id") != graph_id or source_hash != row["code_sha256"]
                or canonical.get("metadata", {}).get("representation") != "HCPG"):
            raise ValueError(f"Graph source checksum differs from the test function: {row['sample_id']!r}")
        label = getattr(data, "y", None)
        if (not torch.is_tensor(label) or label.numel() != 1 or label.dtype != torch.long
                or int(label.item()) != row["label"]):
            raise ValueError(f"Graph label differs from the test function: {row['sample_id']!r}")
        selected.append(data)
        source_hashes[row["sample_id"]] = source_hash
    return selected, source_hashes


def _load_model(path, hop_size, device):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state or any(not isinstance(key, str) for key in state):
        raise ValueError("Checkpoint must contain a TAF-HGCN state dictionary")
    try:
        node_table = state["node_type_embedding.weight"]
        relation_table = state["relation_type_embedding.weight"]
        classifier = state["classifier.weight"]
        if (node_table.ndim != 2 or node_table.size(0) != 17 or relation_table.ndim != 2
                or relation_table.shape != (6, node_table.size(1))):
            raise ValueError("Checkpoint requires 17 node types and six relation types")
        layers = sorted(int(match.group(1)) for key in state
                        if (match := re.fullmatch(r"layers\.(\d+)\.transform\.weight", key)))
        if not layers or layers != list(range(len(layers))):
            raise ValueError("Checkpoint layer indices must be contiguous")
        first = state["layers.0.transform.weight"]
        if first.ndim != 2 or classifier.ndim != 2 or classifier.shape != (2, first.size(0)):
            raise ValueError("Checkpoint classifier must produce two logits")
        in_channels = first.size(1) - node_table.size(1)
        model = TAFHGCN(dropout=0.0, in_channels=in_channels, hidden_channels=first.size(0),
                        num_layers=len(layers), hop_size=hop_size, type_dim=node_table.size(1))
        model.load_state_dict(state, strict=True)
    except (KeyError, AttributeError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid TAF-HGCN checkpoint: {exc}") from exc
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Checkpoint contains nonfinite parameters")
    return model.to(device).eval()


def run(manifest, data, checkpoint, output, *, id_map=None, batch_size=64, device="cpu", hop_size=3):
    for name, value in (("batch_size", batch_size), ("hop_size", hop_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"Prediction output already exists: {destination}")
    sealed = load_manifest(manifest)
    samples = sealed["samples"]
    mapping = _id_map(id_map, samples)
    selected, source_hashes = _select_data(data, samples, mapping)
    model = _load_model(checkpoint, hop_size, device)
    predictions = []
    with torch.no_grad():
        for batch in DataLoader(selected, batch_size=batch_size, shuffle=False):
            logits = model(batch.to(device))
            if logits.ndim != 2 or logits.size(1) != 2 or not torch.isfinite(logits).all():
                raise ValueError("Model must produce two finite logits per graph")
            probability = torch.softmax(logits, dim=1).cpu()
            for row in probability:
                predictions.append((int(row.argmax()), float(row[1])))
    if len(predictions) != len(samples):
        raise ValueError("Prediction count differs from selected test functions")
    checkpoint_hash = _file_hash(checkpoint)
    rows = [{"sample_id": sample["sample_id"], "code_sha256": source_hashes[sample["sample_id"]],
             "prediction": prediction, "probability": probability, "status": "ok",
             "graph_id": mapping[sample["sample_id"]], "manifest_sha256": sealed["manifest_sha256"],
             "checkpoint_sha256": checkpoint_hash, "hop_size": hop_size}
            for sample, (prediction, probability) in zip(samples, predictions)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-map", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hop-size", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        rows = run(**vars(args))
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        parser.exit(1, f"SP4CPG prediction failed: {exc}\n")
    print(f"Saved {len(rows)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
