"""Check source-text identity across DOT and tensor graph storage."""
import hashlib
import importlib
import io
import json
import pickle
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pydot
import pytest
import torch

from preprocess.graph_core import dump_graph, from_pydot, load_graph, normalize_graph, transform_graph
from preprocess.graph_data import graph_to_data
from preprocess.source_identity import stamp_dot_source
from preprocess.source_identity import run_export


DOT = '''digraph "function" {
  rankdir=LR;
  "call" [kind=CALL, code="consume(2)", label=<CALL,1<BR/>consume(2)>];
  "literal" [kind=LITERAL, code="2"];
  "return" [kind=RETURN, code="return 0;"];
  "call" -> "literal" [label="AST: argument", position="1"];
  "call" -> "return" [label="CFG: next", context="body"];
  "call" -> "return" [label="DDG: result", entity="result"];
  "call" -> "return" [label="DDG: result", entity="result"];
  subgraph "cluster_scope" {
    label="scope";
    "call" [color=blue];
  }
}
'''


def sha256(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def dot_structure(dot):
    return {
        "name": dot.get_name(),
        "attributes": {key: value for key, value in dot.get_attributes().items()
                       if key != "source_code_sha256"},
        "nodes": [(node.get_name(), node.get_attributes()) for node in dot.get_nodes()],
        "edges": [(edge.get_source(), edge.get_destination(), edge.get_attributes())
                  for edge in dot.get_edges()],
        "subgraphs": [dot_structure(child) for child in dot.get_subgraphs()],
    }


def test_stamp_uses_the_function_text_and_preserves_dot_structure(tmp_path):
    path = tmp_path / "0-cpg-1.dot"
    path.write_text(DOT, encoding="utf-8")
    before = dot_structure(pydot.graph_from_dot_file(str(path))[0])
    code = "int consume(int value) { return value + 2; }"
    assert stamp_dot_source(path, code) == sha256(code)
    parsed = pydot.graph_from_dot_file(str(path))[0]
    assert json.loads(parsed.get_attributes()["source_code_sha256"]) == sha256(code)
    assert dot_structure(parsed) == before
    assert load_graph(path)["metadata"]["source_code_sha256"] == sha256(code)

    changed_code = "int consume(int value) { return value + 3; }"
    stamp_dot_source(path, changed_code)
    assert load_graph(path)["metadata"]["source_code_sha256"] == sha256(changed_code)
    assert sha256(code) != sha256(changed_code)


def test_source_hash_uses_exact_utf8_and_dos_line_endings(tmp_path):
    path = tmp_path / "unicode-cpg-0.dot"
    path.write_bytes(b"\xef\xbb\xbf" + DOT.replace("\n", "\r\n").encode("utf-8"))
    code = 'int read_value() {\r\n  /* 中文 café */\r\n  return 2;\r\n}\r\n'
    stamp_dot_source(path, code)
    assert load_graph(path)["metadata"]["source_code_sha256"] == sha256(code)
    assert sha256(code) != sha256(code.replace("\r\n", "\n"))


def test_source_hash_survives_transform_dot_json_and_graphdata_storage(tmp_path):
    path = tmp_path / "case-cpg-1.dot"
    path.write_text(DOT, encoding="utf-8")
    code = "int run() { consume(2); return 0; }"
    stamp_dot_source(path, code)
    original = load_graph(path)
    graph = transform_graph(original)
    assert graph["dictionary"]["literal"]["status"] == "carried"
    assert graph["metadata"]["source_code_sha256"] == sha256(code)
    assert original["metadata"]["representation"] == "CPG"

    for suffix in ("dot", "json"):
        destination = tmp_path / ("case-hcpg-1." + suffix)
        dump_graph(graph, destination)
        restored = load_graph(destination)
        assert restored == graph
        data = graph_to_data(restored, torch.ones(len(restored["nodes"]), 4), 1)
        stream = io.BytesIO()
        torch.save(data, stream)
        stream.seek(0)
        encoded = torch.load(stream, pickle_module=pickle)
        assert json.loads(encoded.canonical_json)["metadata"]["source_code_sha256"] == sha256(code)


@pytest.mark.parametrize("checksum", [None, 17, "", "short", "a" * 63, "a" * 65,
                                      "g" * 64, " " + "a" * 64, "a" * 64 + "\n"])
def test_normalize_rejects_invalid_source_hash(checksum):
    with pytest.raises(ValueError, match="source_code_sha256"):
        normalize_graph({"nodes": [], "relations": [],
                         "metadata": {"source_code_sha256": checksum}})


@pytest.mark.parametrize("checksum", ["bad", "a" * 63, "g" * 64])
def test_dot_import_rejects_invalid_source_hash(tmp_path, checksum):
    path = tmp_path / "source.dot"
    path.write_text('digraph G { source_code_sha256="' + checksum + '"; }', encoding="utf-8")
    with pytest.raises(ValueError, match="source_code_sha256"):
        load_graph(path)


def test_hash_hexadecimal_case_is_normalized_without_changing_input():
    graph = {"nodes": [], "relations": [], "metadata": {"source_code_sha256": "A" * 64}}
    assert normalize_graph(graph)["metadata"]["source_code_sha256"] == "a" * 64
    assert graph["metadata"]["source_code_sha256"] == "A" * 64


def test_embedded_and_outer_source_hashes_must_agree():
    graph = {"nodes": [], "relations": [], "metadata": {"source_code_sha256": "a" * 64}}
    dot = pydot.Dot(graph_type="digraph")
    dot.set("sp4cpg_json", json.dumps(json.dumps(graph)))
    dot.set("source_code_sha256", json.dumps("b" * 64))
    with pytest.raises(ValueError, match="values differ"):
        from_pydot(dot)
    dot.set("source_code_sha256", json.dumps("A" * 64))
    assert from_pydot(dot)["metadata"]["source_code_sha256"] == "a" * 64


def test_outer_source_hash_is_imported_into_embedded_graph():
    dot = pydot.Dot(graph_type="digraph")
    dot.set("sp4cpg_json", json.dumps(json.dumps({"nodes": [], "relations": []})))
    dot.set("source_code_sha256", json.dumps("a" * 64))
    assert from_pydot(dot)["metadata"]["source_code_sha256"] == "a" * 64


def test_stamp_rejects_multiple_graphs_without_writing(tmp_path):
    path = tmp_path / "graphs.dot"
    content = "digraph A {}\ndigraph B {}\n"
    path.write_text(content, encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="exactly one DOT graph"):
        stamp_dot_source(path, "int a() { return 0; }")
    assert path.read_bytes() == before


class CommandRunner:
    def __init__(self, code, *, fail=None, missing=None):
        self.code, self.fail, self.missing = code, fail, missing
        self.calls = []
        self.source_dir = self.workspace_dir = None

    def __call__(self, command, *, capture_output, text):
        assert capture_output is True and text is True
        self.calls.append(command)
        if command[0] == "joern-parse":
            self.source_dir = Path(command[1])
            assert [path.name for path in self.source_dir.iterdir()] == ["function.c"]
            assert (self.source_dir / "function.c").read_bytes() == self.code.encode("utf-8")
            cpg_path = Path(command[3])
            self.workspace_dir = cpg_path.parent
            assert not cpg_path.exists()
            if self.missing != "joern-parse":
                cpg_path.write_bytes(b"cpg")
        else:
            assert command[0] == "joern-export"
            assert Path(command[1]).read_bytes() == b"cpg"
            output_dir = Path(command[-1])
            assert not output_dir.exists()
            output_dir.mkdir()
            if self.missing != "joern-export":
                (output_dir / "0-cpg.dot").write_text(DOT, encoding="utf-8")
        return subprocess.CompletedProcess(command, 3 if self.fail == command[0] else 0,
                                           stdout="", stderr="command failed" if self.fail == command[0] else "")


def test_export_uses_fresh_source_bin_and_dot_paths_before_stamping(tmp_path):
    source_root, workspace_root = tmp_path / "source", tmp_path / "workspace"
    old_sources = source_root / "0"
    old_sources.mkdir(parents=True)
    (old_sources / "0.c").write_text("int old_zero;", encoding="utf-8")
    (old_sources / "1.c").write_text("int old_one;", encoding="utf-8")
    workspace_root.mkdir()
    old_bin = workspace_root / "0-cpg-1.bin"
    old_bin.write_bytes(b"previous")
    destination = tmp_path / "dots-cpg" / "0-cpg-1.dot"
    destination.parent.mkdir()
    destination.write_text("digraph Old {}", encoding="utf-8")
    code = "int current() {\r\n  /* 值 */ return 2;\r\n}\r\n"
    runner = CommandRunner(code)
    assert run_export(code, destination, source_root=source_root, workspace_root=workspace_root,
                      command_runner=runner) == destination
    assert [command[0] for command in runner.calls] == ["joern-parse", "joern-export"]
    assert runner.source_dir.parent == source_root
    assert runner.workspace_dir.parent == workspace_root
    assert not runner.source_dir.exists() and not runner.workspace_dir.exists()
    assert (old_sources / "0.c").read_text(encoding="utf-8") == "int old_zero;"
    assert (old_sources / "1.c").read_text(encoding="utf-8") == "int old_one;"
    assert old_bin.read_bytes() == b"previous"
    assert load_graph(destination)["metadata"]["source_code_sha256"] == sha256(code)


@pytest.mark.parametrize("stage,missing", [("joern-parse", False), ("joern-export", False),
                                           ("joern-parse", True), ("joern-export", True)])
def test_failed_or_missing_exports_do_not_stamp_or_replace_previous_graph(tmp_path, monkeypatch, stage, missing):
    from preprocess import source_identity
    destination = tmp_path / "previous.dot"
    destination.write_text("digraph Previous {}", encoding="utf-8")
    previous = destination.read_bytes()
    code = "int current() { return 2; }"
    runner = CommandRunner(code, missing=stage if missing else None, fail=None if missing else stage)
    stamped = []
    monkeypatch.setattr(source_identity, "stamp_dot_source", lambda *args: stamped.append(args))
    with pytest.raises(RuntimeError, match=stage):
        run_export(code, destination, source_root=tmp_path / "source", workspace_root=tmp_path / "workspace",
                   command_runner=runner)
    assert stamped == []
    assert destination.read_bytes() == previous
    expected = ["joern-parse"] if stage == "joern-parse" else ["joern-parse", "joern-export"]
    assert [command[0] for command in runner.calls] == expected
    assert not runner.source_dir.exists() and not runner.workspace_dir.exists()


@pytest.mark.parametrize("module_name", ["preprocess.cpg_generate", "preprocess.cpg_generate_parallel"])
def test_exporter_import_has_no_command_or_directory_side_effects(tmp_path, monkeypatch, module_name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["exporter", "--not-an-export-option"])
    module = importlib.import_module(module_name)
    assert callable(module.main)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module_name", ["preprocess.cpg_generate", "preprocess.cpg_generate_parallel"])
@pytest.mark.parametrize("failed", [False, True])
def test_exporter_main_reports_status_and_only_writes_successful_functions(tmp_path, monkeypatch, module_name, failed):
    module = importlib.import_module(module_name)
    data_dir, preprocess_dir = tmp_path / "data", tmp_path / "preprocess"
    data_dir.mkdir()
    preprocess_dir.mkdir()
    monkeypatch.chdir(preprocess_dir)
    first_code, second_code = "int first() { return 1; }", "int second() { return 2; }"
    records = [{"func": first_code, "target": 0}, {"func": second_code, "target": 1}]
    (data_dir / "PrimeVul.json").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

    def export(code, destination, **paths):
        runner = CommandRunner(code, fail="joern-parse" if failed and code == second_code else None)
        return run_export(code, destination, command_runner=runner, **paths)

    monkeypatch.setattr(module, "run_export", export)
    if module_name.endswith("_parallel"):
        monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    assert module.main(["--dataset", "PrimeVul.json", "--workers", "2"]) == int(failed)
    assert load_graph(preprocess_dir / "dots-cpg" / "0-cpg-0.dot")["metadata"]["source_code_sha256"] == sha256(first_code)
    second_dot = preprocess_dir / "dots-cpg" / "1-cpg-1.dot"
    assert second_dot.exists() is not failed
    if not failed:
        assert load_graph(second_dot)["metadata"]["source_code_sha256"] == sha256(second_code)


def test_parallel_worker_reports_the_same_export_failure(tmp_path, monkeypatch):
    from preprocess import cpg_generate_parallel
    code = "int value() { return 2; }"
    runner = CommandRunner(code, fail="joern-export")

    def export(code, destination, **paths):
        return run_export(code, destination, command_runner=runner, **paths)

    monkeypatch.setattr(cpg_generate_parallel, "run_export", export)
    idx, ok, message = cpg_generate_parallel.process_one((
        7, code, 1, tmp_path / "source", tmp_path / "dots-cpg", tmp_path / "workspace"))
    assert idx == 7 and ok is False and "joern-export failed" in message
    assert not (tmp_path / "dots-cpg" / "7-cpg-1.dot").exists()
