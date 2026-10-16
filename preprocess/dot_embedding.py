"""Embed CPG/HCPG graphs into a versioned and auditable PyG dataset.

Labels are joined by stable graph IDs, never by parallel positional lists.
Failed inputs are reported explicitly and cannot silently become empty graphs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .parse_dot import CodeGraphEncoder, canonical_graph, node_input_text
except ImportError:
    from parse_dot import CodeGraphEncoder, canonical_graph, node_input_text
from sp4cpg.data import graph_to_data


def graph_files(directory):
    """Discover graphs recursively, excluding the construction CLI's own reports."""
    return [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".dot", ".json"}
            and p.name != "construction_report.json" and not p.name.endswith(".construction_report.json")]


def load_labels(path):
    """Read {sample_id:0|1} or a list with explicit IDs and target/label."""
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        source = json.load(handle)
    if isinstance(source, dict):
        pairs = source.items()
    elif isinstance(source, list):
        pairs = []
        for row in source:
            sample_id = row.get("sample_id", row.get("graph_id", row.get("id")))
            if sample_id is None:
                raise ValueError("Each label entry requires an explicit sample_id/graph_id/id")
            pairs.append((sample_id, row.get("target", row.get("label"))))
    else:
        raise ValueError("Labels must be an ID mapping or an ID-bearing list")
    result = {}
    for sample_id, label in pairs:
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"Invalid binary label for {sample_id!r}: {label!r}")
        key = str(sample_id)
        if key in result:
            raise ValueError(f"Duplicate label sample ID {key!r}")
        result[key] = int(label)
    return result


def label_for(graph, path, labels):
    sample_id = str(graph["graph_id"])
    candidates = []
    if sample_id in labels:
        candidates.append(labels[sample_id])
    metadata = graph.get("metadata", {})
    stored = metadata.get("label", metadata.get("target"))
    if stored is not None:
        candidates.append(stored)
    match = re.fullmatch(r".+-(?:cpg|hcpg)-([01])", Path(path).stem, flags=re.IGNORECASE)
    if match:
        candidates.append(int(match.group(1)))
    if any(isinstance(v, bool) or v not in (0, 1) for v in candidates):
        raise ValueError(f"Invalid binary label for {sample_id!r}")
    if len(set(candidates)) > 1:
        raise ValueError(f"Conflicting label sources for {sample_id!r}")
    return int(candidates[0]) if candidates else None


def embed_files(paths, encoder, *, representation, labels=None):
    """Return successes and explicit failures; the caller decides write policy."""
    labels = labels or {}
    samples, failures, used_ids = [], [], set()
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p).casefold()):
        try:
            graph = copy.deepcopy(canonical_graph(path))
            graph.setdefault("metadata", {})["representation"] = representation
            sample_id = str(graph["graph_id"])
            if sample_id in used_ids:
                raise ValueError(f"Duplicate graph_id {sample_id!r}")
            label = label_for(graph, path, labels)
            features = encoder.encode_code_text([node_input_text(n) for n in graph["nodes"]])
            data = graph_to_data(graph, features, label)
            samples.append(data)
            used_ids.add(sample_id)
        except Exception as exc:
            failures.append({"input": str(path), "error_type": type(exc).__name__,
                             "error": str(exc)})
    return samples, failures


class HCPGDataset:
    """Small compatibility wrapper; the schema-v2 CLI is recommended."""
    def __init__(self, dot_dir=None, encoder=None):
        self.dot_dir = Path(dot_dir) if dot_dir else REPO_ROOT / "preprocess" / "dots-hcpg"
        self.encoder = encoder

    def encode_dot_to_feature(self):
        if not self.dot_dir.is_dir():
            raise ValueError(f"Dataset directory not found: {self.dot_dir}")
        encoder = self.encoder or CodeGraphEncoder()
        paths = graph_files(self.dot_dir)
        samples, failures = embed_files(paths, encoder, representation="HCPG")
        if failures:
            raise ValueError(json.dumps({"embedding_failures": failures}, ensure_ascii=False))
        return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "preprocess" / "dots-hcpg")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "preprocess" / "hcpg_dataset.pt")
    parser.add_argument("--representation", choices=("CPG", "HCPG"), default="HCPG")
    parser.add_argument("--labels", type=Path, help="Stable-ID to label JSON manifest")
    parser.add_argument("--codebert", default="microsoft/codebert-base")
    parser.add_argument("--revision", help="Pinned model/tokenizer revision for this run")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--encoding-batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--on-error", choices=("error", "skip"), default="error")
    parser.add_argument("--report", type=Path, help="Default: OUTPUT.report.json")
    args = parser.parse_args()
    if args.input.is_dir():
        paths = graph_files(args.input)
    elif args.input.is_file():
        paths = [args.input]
    else:
        parser.error(f"Input not found: {args.input}")
    if not paths:
        parser.error("Input contains no .dot or .json graph files")
    labels = load_labels(args.labels)
    encoder = CodeGraphEncoder(args.codebert, revision=args.revision,
                               max_length=args.max_length, batch_size=args.encoding_batch,
                               device=args.device, local_files_only=args.local_files_only)
    samples, failures = embed_files(paths, encoder, representation=args.representation,
                                    labels=labels)
    report = {"schema_version": 2, "representation": args.representation,
              "attempted": len(paths), "succeeded": len(samples), "failed": len(failures),
              "failures": failures, "sample_ids": [s.sample_id for s in samples],
              "failure_policy": args.on_error, "encoder": encoder.metadata,
              "dataset_written": False}
    report_path = args.report or Path(str(args.output) + ".report.json")
    if samples and (not failures or args.on_error == "skip"):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": 2, "representation": args.representation,
                    "samples": samples, "metadata": {
                        "encoder": encoder.metadata, "failure_report": str(report_path),
                        "source_files": [str(p) for p in sorted(paths)],
                    }}, args.output)
        report["dataset_written"] = True
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"succeeded": len(samples), "failed": len(failures),
                      "dataset_written": report["dataset_written"],
                      "report": str(report_path)}, ensure_ascii=False))
    if not report["dataset_written"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
