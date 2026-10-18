"""Canonical graph conversion and frozen CodeBERT node-text encoding."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.graph_data import EDGE_TYPE_MAP, NODE_TYPE_MAP, GraphData, graph_to_data, validate_graph


def node_input_text(node):
    value = node.get("code", "")
    if not isinstance(value, str):
        raise ValueError("Canonical node.code must be a string")
    return value


class CodeGraphEncoder(nn.Module):
    """Encode textual node content; the classifier learns all type embeddings."""

    def __init__(self, codebert_model_name="microsoft/codebert-base",
                 node_type_embed_dim=64, edge_type_embed_dim=32, hidden_dim=256, *,
                 model_version=None, max_length=128, batch_size=32, device="cpu",
                 local_files_only=False, tokenizer=None, model=None):
        super().__init__()
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        if (tokenizer is None) != (model is None):
            raise ValueError("Provide both tokenizer and model, or neither")
        if tokenizer is None:
            from transformers import AutoModel, AutoTokenizer
            options = {"local_files_only": local_files_only}
            if model_version is not None:
                options["revision"] = model_version
            tokenizer = AutoTokenizer.from_pretrained(codebert_model_name, **options)
            model = AutoModel.from_pretrained(codebert_model_name, **options)
        self.tokenizer = tokenizer
        self.codebert = model.to(device)
        self.codebert.requires_grad_(False)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.eval()

    def encode_code_text(self, code_texts, max_length=None):
        texts = list(code_texts)
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("CodeBERT inputs must be strings")
        width = int(self.codebert.config.hidden_size)
        if not texts:
            return torch.empty((0, width), dtype=torch.float32)
        limit = self.max_length if max_length is None else int(max_length)
        if limit < 1:
            raise ValueError("max_length must be positive")
        # no_grad does not disable transformer dropout.
        self.codebert.eval()
        device = next(self.codebert.parameters()).device
        encoded = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self.tokenizer(texts[start:start+self.batch_size], padding=True,
                                    truncation=True, max_length=limit, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output = self.codebert(**inputs)
                encoded.append(output.last_hidden_state[:, 0, :].detach().cpu().float())
        return torch.cat(encoded, dim=0)

    def forward(self, code_texts):
        return self.encode_code_text(code_texts)


def canonical_graph(value, sample_id=None):
    from preprocess.graph_core import from_pydot, load_graph, normalize_graph
    if isinstance(value, Mapping):
        graph = normalize_graph(dict(value))
    elif isinstance(value, (str, Path)):
        graph = load_graph(value)
    elif hasattr(value, "get_nodes") and hasattr(value, "get_edges"):
        graph = from_pydot(value, graph_id=str(sample_id or value.get_name()).strip('"'))
    else:
        raise TypeError("Expected a canonical graph, DOT/JSON path or pydot graph")
    if sample_id is not None and graph["graph_id"] != str(sample_id):
        raise ValueError("Explicit sample_id disagrees with the canonical graph_id")
    return graph


def parse_node_label(raw_label):
    """Read a node label through the same canonical DOT parser."""
    import pydot
    from preprocess.graph_core import from_pydot
    dot = pydot.Dot(graph_type="digraph")
    dot.add_node(pydot.Node("_node", label=raw_label))
    node = from_pydot(dot, graph_id="_node_label")["nodes"][0]
    if node["kind"] not in NODE_TYPE_MAP:
        raise ValueError(f"Unsupported node kind: {node['kind']!r}")
    return node["kind"], node["code"]


def parse_dot_file_enhanced(graph, label, encoder_model, *, sample_id=None):
    canonical = validate_graph(canonical_graph(graph, sample_id=sample_id))
    features = encoder_model.encode_code_text([node_input_text(node) for node in canonical["nodes"]])
    return graph_to_data(canonical, features, label=label)


def parse_dot_to_data(graph, label, encoder_model, *, sample_id=None):
    return parse_dot_file_enhanced(graph, label, encoder_model, sample_id=sample_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=REPO_ROOT / "preprocess" / "dots-hcpg" / "2-hcpg-0.dot")
    parser.add_argument("--label", type=int, choices=(0, 1))
    parser.add_argument("--codebert", default="microsoft/codebert-base")
    parser.add_argument("--model-version")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        graph = canonical_graph(args.input)
        encoder = CodeGraphEncoder(args.codebert, model_version=args.model_version,
                                   local_files_only=args.local_files_only)
        data = parse_dot_file_enhanced(graph, args.label, encoder)
        print(f"sample_id={data.sample_id}, nodes={data.num_nodes}, relations={data.num_hyperedges}")
    except Exception as exc:
        parser.exit(1, f"Graph encoding failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
