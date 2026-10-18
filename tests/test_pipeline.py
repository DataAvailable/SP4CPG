"""Check graph construction, serialization, embeddings and directed learning together."""
import copy
import inspect
import json

import torch
from torch_geometric.data import Batch

from models.hgcn_models import HGCN
from preprocess.graph_core import dump_graph, load_graph, normalize_graph, transform_graph
from preprocess.graph_data import graph_to_data


def connected_graph():
    nodes = [
        {"id": "a", "kind": "CALL", "code": "x=5", "attributes": {}},
        {"id": "b", "kind": "CALL", "code": "y=2", "attributes": {}},
        {"id": "c", "kind": "CALL", "code": "z=x+y", "attributes": {}},
        {"id": "d", "kind": "RETURN", "code": "return x+y", "attributes": {}},
        {"id": "e", "kind": "IDENTIFIER", "code": "x", "attributes": {}},
    ]
    relations = []
    for source, label in (("a", "x"), ("b", "y")):
        for target in ("c", "d"):
            for kind in ("DDG", "CFG"):
                relation_id = f"{kind}:{source}:{target}"
                relations.append({"id": relation_id, "kind": kind, "sources": [source], "targets": [target],
                                  "bindings": [{"source": source, "target": target, "kind": kind,
                                                "label": label if kind == "DDG" else "",
                                                "entity": label if kind == "DDG" else "",
                                                "context": "branch:0", "attributes": {}, "path": [],
                                                "relation_id": relation_id}], "attributes": {}})
    relations.append({"id": "ast:a:e", "kind": "AST", "sources": ["a"], "targets": ["e"],
                      "bindings": [{"source": "a", "target": "e", "kind": "AST", "label": "",
                                    "entity": "", "context": "", "attributes": {}, "path": [],
                                    "relation_id": "ast:a:e"}], "attributes": {}})
    relations.append({"id": "cfg:d:a", "kind": "CFG", "sources": ["d"], "targets": ["a"],
                      "bindings": [{"source": "d", "target": "a", "kind": "CFG", "label": "",
                                    "entity": "", "context": "loop:0", "attributes": {}, "path": [],
                                    "relation_id": "cfg:d:a"}], "attributes": {}})
    return normalize_graph({"schema_version": 2, "graph_id": "function:1", "nodes": nodes,
                            "relations": relations, "metadata": {}})


def dependency_records(graph):
    return sorted(json.dumps(binding, sort_keys=True) for relation in graph["relations"]
                  for binding in relation["bindings"] if binding["kind"] == "DDG")


def test_construction_to_batched_learning_preserves_relation_records(tmp_path):
    original = connected_graph()
    unchanged = copy.deepcopy(original)
    transformed = transform_graph(original)
    assert original == unchanged
    assert dependency_records(transformed) == dependency_records(original)
    assert {r["kind"] for r in transformed["relations"]} >= {"CTRL", "DATA"}
    assert "e" not in {node["id"] for node in transformed["nodes"]}
    assert transformed["dictionary"]["e"]["carrier"] == "a"
    path = tmp_path / "hcpg.dot"
    dump_graph(transformed, path)
    restored = load_graph(path)
    assert restored == transformed
    width = 4
    features = torch.arange(len(restored["nodes"]) * width, dtype=torch.float32).view(-1, width) / 100
    data = graph_to_data(restored, features, 1)
    another_graph = copy.deepcopy(restored)
    another_graph["graph_id"] = "function:2"
    another = graph_to_data(another_graph, features * 2, 0)
    batch = Batch.from_data_list([data, another])
    assert dependency_records(json.loads(batch.graph_json[0])) == dependency_records(original)
    torch.manual_seed(12)
    model = HGCN(dropout=0.0, in_channels=width, hidden_channels=8, num_layers=2, hop_size=2)
    model.eval()
    individual = torch.cat([model(data), model(another)])
    torch.testing.assert_close(model(batch), individual)
    path = tmp_path / "hcpg_dataset.pkl"
    torch.save([data, another], path)
    options = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    loaded = torch.load(path, **options)
    loaded_batch = Batch.from_data_list(loaded)
    torch.testing.assert_close(model(loaded_batch), individual)
    loss = torch.nn.functional.cross_entropy(model(loaded_batch), loaded_batch.y.view(-1))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_hgcn_import_paths_share_the_same_architecture():
    from models.hgcn_models import TAFHGCN
    from models.hgnn_models import HGCN as OtherHGCN, TAFHGCN as OtherTAFHGCN
    assert OtherTAFHGCN is TAFHGCN
    assert OtherHGCN is HGCN
