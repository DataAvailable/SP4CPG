"""Encode graph files into a list of directed GraphData samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.graph_data import GraphData, validate_graph
from preprocess.parse_dot import CodeGraphEncoder, canonical_graph, parse_dot_file_enhanced


def load_labels(path):
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Labels must be a JSON object mapping stable sample IDs to 0 or 1")
    labels = {}
    for sample_id, label in value.items():
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Invalid binary label for {sample_id!r}")
        labels[str(sample_id)] = label
    return labels


def label_for(graph, path, labels):
    sample_id = graph["graph_id"]
    candidates = []
    if sample_id in labels:
        candidates.append(labels[sample_id])
    for key in ("label", "target"):
        if key in graph["metadata"]:
            candidates.append(graph["metadata"][key])
    match = re.fullmatch(r".+-(?:cpg|hcpg)-([01])", Path(path).stem, flags=re.IGNORECASE)
    if match:
        candidates.append(int(match.group(1)))
    if not candidates:
        raise ValueError(f"No binary label for graph {sample_id!r}: {path}")
    if any(type(label) is not int or label not in (0, 1) for label in candidates):
        raise ValueError(f"Invalid binary label for graph {sample_id!r}")
    if len(set(candidates)) != 1:
        raise ValueError(f"Conflicting labels for graph {sample_id!r}")
    return candidates[0]


class HCPGDataset:
    def __init__(self, transform=None, *, dot_dir=None, input_path=None, encoder=None,
                 labels=None, encoder_options=None):
        self.transform = transform
        self.dot_dir = Path(input_path or dot_dir or REPO_ROOT / "preprocess" / "dots-hcpg")
        if self.dot_dir.is_file():
            if self.dot_dir.suffix.lower() not in {".dot", ".json"}:
                raise ValueError("Graph input must be a DOT or JSON file")
            self.dot_files = [self.dot_dir]
        elif self.dot_dir.is_dir():
            self.dot_files = sorted((path for path in self.dot_dir.rglob("*")
                                     if path.is_file() and path.suffix.lower() in {".dot", ".json"}
                                     and path.name != "construction_report.json"
                                     and not path.name.endswith(".construction_report.json")),
                                    key=lambda path: path.relative_to(self.dot_dir).as_posix())
        else:
            raise FileNotFoundError(f"Graph input not found: {self.dot_dir}")
        if isinstance(labels, (str, Path)):
            label_path = Path(labels).resolve()
            self.dot_files = [path for path in self.dot_files if path.resolve() != label_path]
        if not self.dot_files:
            raise ValueError(f"No DOT or JSON graph files found: {self.dot_dir}")
        self.encoder = encoder
        self.encoder_options = dict(encoder_options or {})
        self.labels = load_labels(labels) if isinstance(labels, (str, Path)) else dict(labels or {})

    def encode_dot_to_feature(self):
        entries, ids = [], set()
        for path in self.dot_files:
            try:
                graph = validate_graph(canonical_graph(path))
                sample_id = graph["graph_id"]
                if sample_id in ids:
                    raise ValueError(f"Duplicate graph_id: {sample_id!r}")
                label = label_for(graph, path, self.labels)
                entries.append((path, graph, label))
                ids.add(sample_id)
            except Exception as exc:
                raise ValueError(f"Cannot read graph {path}: {exc}") from exc
        encoder = self.encoder or CodeGraphEncoder(**self.encoder_options)
        samples = []
        for path, graph, label in entries:
            try:
                data = parse_dot_file_enhanced(graph, label, encoder)
                if self.transform is not None:
                    data = self.transform(data)
                if not isinstance(data, GraphData) or data.schema_version != 2:
                    raise ValueError("Transforms must retain schema-version-2 GraphData")
                samples.append(data)
            except Exception as exc:
                raise ValueError(f"Cannot encode graph {path}: {exc}") from exc
        return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "preprocess" / "dots-hcpg")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "preprocess" / "hcpg_dataset.pkl")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--codebert", default="microsoft/codebert-base")
    parser.add_argument("--model-version")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output}; use --overwrite to replace it")
        dataset = HCPGDataset(input_path=args.input, labels=args.labels, encoder_options={
            "codebert_model_name": args.codebert, "model_version": args.model_version,
            "max_length": args.max_length, "batch_size": args.batch_size,
            "device": args.device, "local_files_only": args.local_files_only,
        })
        if args.output.resolve() in {path.resolve() for path in dataset.dot_files}:
            raise ValueError("Output must not overwrite an input graph")
        samples = dataset.encode_dot_to_feature()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, args.output)
        print(f"Saved {len(samples)} GraphData samples to {args.output}")
    except Exception as exc:
        parser.exit(1, f"Graph dataset encoding failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
