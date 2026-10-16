"""Canonical graph loading and deterministic, text-only CodeBERT features.

The historical class/function names remain importable, but generated data uses
schema v2 and the separate ``hyperedge_index`` convention. Legacy checkpoints
and legacy incidence-in-edge_index datasets must be handled explicitly elsewhere.
No pretrained model is downloaded, loaded, or called at module import time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sp4cpg.data import NODE_TYPE_MAP, RELATION_TYPE_MAP, graph_to_data

# Six kinds are ordered AST/CFG/CDG/DDG/CTRL/DATA; legacy numbering differs.
EDGE_TYPE_MAP = RELATION_TYPE_MAP


def node_input_text(node: Mapping) -> str:
    """New node-text policy: the canonical node's code field only.

    Kinds are separate model indices; bindings and relation labels are not
    fabricated into text features. Metadata records this new-run policy rather
    than claiming to recover undocumented historical serialization settings.
    """
    value = node.get("code", "")
    return "" if value is None else str(value)


class CodeGraphEncoder(nn.Module):
    """Frozen CodeBERT CLS features; learned type embeddings live in TAFHGCN.

    Tokenizer/model injection permits local offline use and unit testing.
    Defaults max_length=128 and CLS pooling follow the prior code policy.
    """
    def __init__(self, codebert_model_name="microsoft/codebert-base", *,
                 revision=None, max_length=128, batch_size=32, device="cpu",
                 local_files_only=False, tokenizer=None, model=None):
        super().__init__()
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        if (tokenizer is None) != (model is None):
            raise ValueError("Provide both tokenizer and model, or neither")
        if tokenizer is None:
            from transformers import AutoModel, AutoTokenizer
            load_options = {"local_files_only": local_files_only}
            if revision is not None:
                load_options["revision"] = revision
            tokenizer = AutoTokenizer.from_pretrained(codebert_model_name, **load_options)
            model = AutoModel.from_pretrained(codebert_model_name, **load_options)
        self.tokenizer = tokenizer
        self.codebert = model.to(device)
        self.codebert.requires_grad_(False)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.model_name = codebert_model_name
        self.revision = revision
        self.eval()

    @property
    def metadata(self):
        return {
            "encoder": self.model_name, "requested_revision": self.revision,
            "resolved_revision": getattr(self.codebert.config, "_commit_hash", None),
            "tokenizer": self.tokenizer.__class__.__name__,
            "node_input": "canonical node.code only", "token_pooling": "CLS",
            "max_length": self.max_length, "truncation": True,
            "encoding_batch_size": self.batch_size, "model_mode": "eval",
            "historical_configuration_recovered": False,
        }

    def encode_code_text(self, code_texts, max_length=None):
        texts = list(code_texts)
        width = int(self.codebert.config.hidden_size)
        if not texts:
            return torch.empty((0, width), dtype=torch.float32)
        limit = self.max_length if max_length is None else int(max_length)
        if limit < 1:
            raise ValueError("max_length must be positive")
        # no_grad alone does not disable transformer dropout.
        self.codebert.eval()
        device = next(self.codebert.parameters()).device
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self.tokenizer(texts[start:start+self.batch_size], padding=True,
                                    truncation=True, max_length=limit,
                                    return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.no_grad():
                output = self.codebert(**inputs)
                chunks.append(output.last_hidden_state[:, 0, :].detach().cpu().float())
        return torch.cat(chunks, dim=0)

    def forward(self, code_texts):
        return self.encode_code_text(code_texts)


def canonical_graph(value, graph_id=None):
    from sp4cpg.graph import from_pydot, load_graph
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, Path)):
        return load_graph(value)
    if hasattr(value, "get_nodes") and hasattr(value, "get_edges"):
        return from_pydot(value, graph_id=str(graph_id or value.get_name()).strip('"'))
    raise TypeError("Expected canonical graph, DOT/JSON path or pydot graph")


def parse_dot_file_enhanced(graph, label, encoder_model, *, graph_id=None):
    """Encode one canonical graph; malformed graphs raise, never become zero graphs."""
    canonical = canonical_graph(graph, graph_id=graph_id)
    texts = [node_input_text(node) for node in canonical["nodes"]]
    features = encoder_model.encode_code_text(texts)
    return graph_to_data(canonical, features, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--label", type=int, choices=(0, 1))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebert", default="microsoft/codebert-base")
    parser.add_argument("--revision")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    encoder = CodeGraphEncoder(args.codebert, revision=args.revision,
                               max_length=args.max_length,
                               local_files_only=args.local_files_only)
    sample = parse_dot_file_enhanced(args.input, args.label, encoder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 2, "representation": sample.representation,
                "samples": [sample], "metadata": encoder.metadata}, args.output)
    print(json.dumps({"sample_id": sample.sample_id, "output": str(args.output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
