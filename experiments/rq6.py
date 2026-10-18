"""Paired cosine similarity of mean-pooled CodeBERT node vectors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import torch
from torch import nn

from preprocess.graph_data import validate_graph
from preprocess.parse_dot import CodeGraphEncoder, canonical_graph, node_input_text


_MAPPING_FIELDS = {"representative", "carrier", "carrier_original", "status", "rule"}
_CSV_FIELDS = ("graph_id", "cpg_nodes", "hcpg_nodes", "cosine_similarity", "cosine_similarity_x100")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _construction_report(path):
    name = Path(path).name.lower()
    return name == "construction_report.json" or name.endswith(".construction_report.json")


def _label(graph, path):
    values = [graph["metadata"][key] for key in ("label", "target") if key in graph["metadata"]]
    name_label = re.fullmatch(r".+-(?:cpg|hcpg)-([01])", Path(path).stem, flags=re.I)
    if name_label:
        values.append(int(name_label.group(1)))
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError(f"Invalid binary label for graph {graph['graph_id']!r}")
    if len(set(values)) > 1:
        raise ValueError(f"Conflicting labels for graph {graph['graph_id']!r}")
    return values[0] if values else None


def discover_graphs(path, representation):
    """Read all graph assets and index them by their stable graph IDs."""
    root = Path(path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Graph input does not exist: {root}")
    if root.is_file():
        if root.suffix.lower() not in {".json", ".dot"} or _construction_report(root):
            raise ValueError(f"Expected a DOT or JSON graph file: {root}")
        paths = [root]
    else:
        paths = sorted((source for source in root.rglob("*")
                        if source.is_file() and source.suffix.lower() in {".json", ".dot"}
                        and not _construction_report(source)), key=lambda source: source.as_posix())
    if not paths:
        raise ValueError(f"No DOT or JSON graph files found: {root}")
    graphs = {}
    for source in paths:
        try:
            graph = validate_graph(canonical_graph(source), check_node_kinds=False)
            graph_id = graph["graph_id"]
            if graph_id in graphs:
                raise ValueError(f"Duplicate graph_id {graph_id!r}")
            if graph["metadata"].get("representation") != representation:
                raise ValueError(f"Expected {representation} representation for graph {graph_id!r}")
            label = _label(graph, source)
            graphs[graph_id] = {"graph": graph, "path": str(source), "sha256": _sha256(source), "label": label}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Cannot read graph {source}: {exc}") from exc
    return graphs


def _original_entries(graph):
    return {original_id: {field: value for field, value in entry.items() if field not in _MAPPING_FIELDS}
            for original_id, entry in graph["dictionary"].items()}


def pair_graphs(cpg, hcpg):
    originals = discover_graphs(cpg, "CPG")
    transformed = discover_graphs(hcpg, "HCPG")
    if set(originals) != set(transformed):
        missing_hcpg = sorted(set(originals) - set(transformed))
        missing_cpg = sorted(set(transformed) - set(originals))
        raise ValueError(f"Unpaired graph IDs: missing HCPG={missing_hcpg}, missing CPG={missing_cpg}")
    pairs = []
    for graph_id in sorted(originals):
        left, right = originals[graph_id], transformed[graph_id]
        original = left["graph"]
        if set(original["dictionary"]) != {node["id"] for node in original["nodes"]} or any(
                entry["status"] != "retained" or entry["representative"] != original_id
                or entry["carrier"] is not None or entry["rule"] is not None
                for original_id, entry in original["dictionary"].items()):
            raise ValueError(f"CPG must retain its original node identities: {graph_id!r}")
        entries = _original_entries(original)
        if _json(entries) != _json(_original_entries(right["graph"])):
            raise ValueError(f"Original dictionary entries differ for graph {graph_id!r}")
        if original["metadata"].get("source_code_sha256") != right["graph"]["metadata"].get("source_code_sha256"):
            raise ValueError(f"CPG/HCPG source checksums differ for graph {graph_id!r}")
        if left["label"] is not None and right["label"] is not None and left["label"] != right["label"]:
            raise ValueError(f"Conflicting CPG/HCPG labels for graph {graph_id!r}")
        pairs.append({"graph_id": graph_id, "cpg": left, "hcpg": right,
                      "dictionary_sha256": hashlib.sha256(_json(entries).encode("utf-8")).hexdigest()})
    return pairs


def _mean_vector(graph, encoder):
    texts = [node_input_text(node) for node in graph["nodes"]]
    features = encoder.encode_code_text(texts)
    if not torch.is_tensor(features) or not features.is_floating_point() or features.ndim != 2 or features.size(0) != len(texts) or features.size(1) < 1:
        raise ValueError(f"Invalid node-vector shape or dtype for graph {graph['graph_id']!r}")
    features = features.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(features).all():
        raise ValueError(f"Nonfinite node vector in graph {graph['graph_id']!r}")
    vector = (features / len(texts)).sum(dim=0)
    if not torch.isfinite(vector).all() or vector.abs().max() == 0:
        raise ValueError(f"Invalid or zero graph vector for {graph['graph_id']!r}")
    return vector


def _cosine(left, right):
    if left.shape != right.shape:
        raise ValueError("Paired graph-vector dimensions differ")
    left = left / left.abs().max()
    right = right / right.abs().max()
    left = left / torch.sqrt(left.square().sum())
    right = right / torch.sqrt(right.square().sum())
    value = float(torch.dot(left, right))
    if not -1 - 1e-12 <= value <= 1 + 1e-12:
        raise ValueError("Cosine similarity is outside its numeric range")
    return max(-1.0, min(1.0, value))


def compare_pairs(pairs, encoder):
    """Encode both graph forms with one frozen encoder, then pool and compare."""
    if not pairs:
        raise ValueError("At least one graph pair is required")
    if not isinstance(encoder, nn.Module) or not callable(getattr(encoder, "encode_code_text", None)):
        raise ValueError("Encoder must provide encode_code_text as a torch module")
    encoder.requires_grad_(False)
    encoder.eval()
    rows, inputs = [], []
    width = None
    with torch.no_grad():
        for pair in pairs:
            left = _mean_vector(pair["cpg"]["graph"], encoder)
            right = _mean_vector(pair["hcpg"]["graph"], encoder)
            if width is None:
                width = left.numel()
            if left.numel() != width or right.numel() != width:
                raise ValueError("Encoder graph-vector dimensions differ across inputs")
            similarity = _cosine(left, right)
            rows.append({"graph_id": pair["graph_id"], "cpg_nodes": len(pair["cpg"]["graph"]["nodes"]),
                         "hcpg_nodes": len(pair["hcpg"]["graph"]["nodes"]),
                         "cosine_similarity": similarity, "cosine_similarity_x100": similarity * 100})
            inputs.append({"graph_id": pair["graph_id"], "dictionary_sha256": pair["dictionary_sha256"],
                           **{side: {key: pair[side][key] for key in ("path", "sha256", "label")}
                              for side in ("cpg", "hcpg")}})
    values = [row["cosine_similarity"] for row in rows]
    statistics = {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
    statistics.update({f"{key}_x100": value * 100 for key, value in tuple(statistics.items())})
    return {"pair_count": len(rows), "statistics": statistics, "pairs": rows, "inputs": inputs}


def run(cpg, hcpg, output, dataset, *, codebert="microsoft/codebert-base", model_version=None,
        max_length=128, batch_size=32, device="cpu", local_files_only=False, encoder=None):
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a nonempty string")
    if type(max_length) is not int or max_length < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("max_length and batch_size must be positive integers")
    destination = Path(output).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {destination}")
    pairs = pair_graphs(cpg, hcpg)
    settings = {"codebert": str(codebert), "model_version": model_version, "max_length": max_length,
                "batch_size": batch_size, "device": str(device), "local_files_only": bool(local_files_only)}
    if encoder is None:
        encoder = CodeGraphEncoder(codebert_model_name=codebert, model_version=model_version,
                                   max_length=max_length, batch_size=batch_size, device=device,
                                   local_files_only=local_files_only)
    report = compare_pairs(pairs, encoder)
    config = getattr(getattr(encoder, "codebert", encoder), "config", None)
    settings["resolved_model_version"] = getattr(config, "_commit_hash", None)
    report = {"protocol": "rq6", "dataset": dataset.strip(), "encoder": settings, **report}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "rq6_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (destination / "rq6_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(report["pairs"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpg", type=Path, required=True)
    parser.add_argument("--hcpg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--codebert", default="microsoft/codebert-base")
    parser.add_argument("--model-version")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(**vars(args))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"RQ6 failed: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), "dataset": result["dataset"],
                      "pair_count": result["pair_count"], "statistics": result["statistics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
