"""Cross-module regression tests; fixtures are in-memory test graphs, separate from paper results."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

import torch

from sp4cpg.data import GraphData
from sp4cpg.experiment import make_split, sample_records


ROOT = Path(__file__).resolve().parents[1]


def test_factory_all_architectures_use_valid_binary_or_incidence_input():
    from models.factory import MODEL_NAMES, build_model
    from torch_geometric.data import Batch
    sample = GraphData(x=torch.ones(2, 4), edge_index=torch.tensor([[0, 1], [1, 0]]),
                       hyperedge_index=torch.tensor([[0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2]]),
                       incidence_role=torch.tensor([0, 1, 0, 1, 1, 0]),
                       relation_types=torch.tensor([0, 1, 3]), node_types=torch.tensor([11, 11]),
                       num_nodes=2, num_hyperedges=3)
    batch = Batch.from_data_list([sample, sample.clone()])
    for name in MODEL_NAMES:
        model = build_model(name, in_channels=4, hidden_channels=8, num_classes=2, num_layers=2,
                            dropout=0.0, hop_size=1, type_dim=4, use_types=True, directed=True)
        model.eval()
        output = model(batch)
        assert output.shape == (2, 2) and torch.isfinite(output).all(), name


def test_matched_cli_runs_all_four_configurations_on_identical_ids(tmp_path):
    samples = [GraphData(x=torch.tensor([[0.5, i / 12], [0.2, 0.4]]),
                         edge_index=torch.tensor([[0], [1]]),
                         hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
                         incidence_role=torch.tensor([0, 1]), relation_types=torch.tensor([3]),
                         node_types=torch.tensor([11, 11]), num_nodes=2, num_hyperedges=1,
                         sample_id=f"fixture-{i}", original_node_count=600, schema_version=2,
                         representation="CPG", y=torch.tensor([i % 2])) for i in range(12)]
    for representation in ("CPG", "HCPG"):
        copied = [s.clone() for s in samples]
        for sample in copied:
            sample.representation = representation
        torch.save({"schema_version": 2, "representation": representation, "samples": copied,
                    "metadata": {"encoder": {"fixture": "in-memory test features only"}}}, tmp_path / (representation + ".pt"))
    split = tmp_path / "split.json"
    split.write_text(json.dumps(make_split(sample_records(samples), 3)), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"hidden": 8, "layers": 1, "batch": 4, "epoch": 1, "hop_size": 1}), encoding="utf-8")
    output = tmp_path / "matched"
    process = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_matched.py"),
                              "--cpg-data", str(tmp_path / "CPG.pt"), "--hcpg-data", str(tmp_path / "HCPG.pt"),
                              "--split", str(split), "--output", str(output), "--config", str(config),
                              "--seeds", "2025", "--device", "cpu"], capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((output / "matched_summary.json").read_text(encoding="utf-8"))
    configs = report["configurations"]
    assert {(c["model"], c["representation"]) for c in configs} == {
        (m, g) for m in ("HGNN", "TAF-HGCN") for g in ("CPG", "HCPG")}
    assert len({tuple(c["evaluation"]["complete_test"]["sample_ids"]) for c in configs}) == 1


def test_client_export_training_and_static_steps_keep_same_ids(tmp_path, monkeypatch):
    # Test command construction without requiring a live Qt window/application.
    qt = types.ModuleType("PySide6.QtCore")
    qt.QObject = object
    qt.QThread = object
    qt.Signal = lambda *args: None
    parent = types.ModuleType("PySide6")
    parent.QtCore = qt
    monkeypatch.setitem(sys.modules, "PySide6", parent)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", qt)
    from client.sp4cpg_client.core.config import ClientConfig
    from client.sp4cpg_client.core.runner import WorkflowRunner
    config = ClientConfig(repo_root=str(ROOT), dataset_name="functions.jsonl", output_dir=str(tmp_path),
                          run_cpg=True, run_hcpg=True, run_embedding=True, hgcn_split_manifest="explicit-split.json",
                          hgcn_checkpoint="versioned-model.pt", allow_fallback=True)
    runner = WorkflowRunner(config, "predict")
    commands = [command for _, command, _ in runner._build_steps()]
    predict = next(c for c in commands if c[2].endswith("predict.py"))
    static = next(c for c in commands if c[2].endswith("svf_detector.py"))
    assert Path(static[2]).is_file()
    expected = str(tmp_path / "artifacts" / "cpg" / "samples.jsonl")
    assert predict[predict.index("--source-data") + 1] == static[static.index("--data") + 1] == expected
    runner.mode = "train"
    train = next(c for _, c, _ in runner._build_steps() if c[2].endswith("main.py"))
    assert train[train.index("--split") + 1] == "explicit-split.json"


def test_static_and_exporter_generate_identical_ids_for_idless_source(tmp_path):
    from preprocess.cpg_generate import load_samples
    from client.sp4cpg_client.tools.codebert_common import load_function_dataset
    path = tmp_path / "functions.json"
    path.write_text(json.dumps([{"func": "int first(){return 0;}", "target": 0},
                               {"func": "int second(){return 1;}", "target": 1}]), encoding="utf-8")
    _, exported = load_samples(path)
    static = load_function_dataset(path)
    assert [r["sample_id"] for r in exported] == [r["sample_id"] for r in static]
