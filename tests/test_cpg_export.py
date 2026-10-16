import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from preprocess import cpg_generate
from sp4cpg.graph import load_graph


def method_dot(name="target", external=False):
    return ('digraph ' + json.dumps(name) + ' {\n'
            '"1" [kind="METHOD", label=' + json.dumps("(METHOD," + name + ")") + ', code="int target() {}", '
            'isExternal=' + ('true' if external else 'false') + '];\n'
            '"2" [kind="CALL", code="work()"];\n'
            '"1" -> "2" [label="AST"];\n}')


class JoernCommandFixture:
    def __init__(self, exports=None):
        self.calls = []
        self.exports = exports or {"0-cpg.dot": method_dot()}

    def __call__(self, argv, **kwargs):
        assert isinstance(argv, list)
        assert kwargs["shell"] is False
        self.calls.append((argv, kwargs))
        if argv[0] == "parser":
            source = next(Path(argv[1]).glob("function.*")).read_text(encoding="utf-8")
            if "FAIL_PARSE" in source:
                return subprocess.CompletedProcess(argv, 7, stdout="", stderr="parse failed")
            Path(argv[argv.index("-o") + 1]).write_text(source, encoding="utf-8")
        elif argv[0] == "exporter":
            source = Path(argv[1]).read_text(encoding="utf-8")
            if "FAIL_EXPORT" in source:
                return subprocess.CompletedProcess(argv, 8, stdout="", stderr="export failed")
            output = Path(argv[argv.index("--out") + 1])
            output.mkdir(parents=True)
            for filename, content in self.exports.items():
                path = output / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")


def export(tmp_path, monkeypatch, rows, *, exports=None, **options):
    dataset = tmp_path / "arbitrary-dataset-name.json"
    dataset.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    joern = JoernCommandFixture(exports)
    monkeypatch.setattr(cpg_generate.subprocess, "run", joern)
    result = cpg_generate.run_export(
        dataset, output_root=tmp_path / "output", source_root=tmp_path / "source",
        workspace_root=tmp_path / "workspace", joern_parse="parser", joern_export="exporter",
        joern_version="author-supplied-exact-version", **options)
    return result, joern


def test_safe_filenames_stable_ids_labels_hashes_and_fixed_output(tmp_path, monkeypatch):
    malicious = '../../CON; & echo "unsafe" $(bad)'
    rows = [{"sample_id": malicious, "func": "int a() { return 1; }", "target": 1},
            {"id": "CON", "func": "int b() { return 0; }", "target": 0}]
    result, joern = export(tmp_path, monkeypatch, rows, workers=2)
    assert result["counts"] == {"total": 2, "success": 2, "failed": 0, "filtered": 0}
    assert Path(result["graph_dir"]) == tmp_path / "output" / "graphs"
    assert result["joern"]["version"] == "author-supplied-exact-version"
    assert json.loads(Path(result["labels"]).read_text(encoding="utf-8")) == {malicious: 1, "CON": 0}
    for record, original in zip(result["records"], rows):
        dot_path = Path(record["dot_path"])
        assert dot_path.parent == tmp_path / "output" / "graphs"
        assert dot_path.name.startswith("s-")
        assert not any(token in dot_path.name for token in ("..", "&", "$", "CON"))
        graph = load_graph(dot_path)
        assert graph["graph_id"] == record["sample_id"]
        assert graph["metadata"]["label"] == original["target"]
        assert record["source_sha256"] == hashlib.sha256(original["func"].encode()).hexdigest()
        assert record["selected_method_dot"] == "0-cpg.dot"
        assert len(record["commands"]) == 2
    assert len(joern.calls) == 4
    for argv, kwargs in joern.calls:
        assert malicious not in argv
        assert all(original["func"] not in argv for original in rows)
    exported_rows = [json.loads(line) for line in Path(result["samples_jsonl"]).read_text(encoding="utf-8").splitlines()]
    assert exported_rows[0]["id"] == malicious
    assert exported_rows[0]["label"] == 1


def test_parse_export_failures_are_reported_and_parse_failure_stops_export(tmp_path, monkeypatch):
    rows = [{"id": "a", "func": "FAIL_PARSE", "target": 1},
            {"id": "b", "func": "FAIL_EXPORT", "target": 0},
            {"id": "c", "func": "int c() {}", "target": 1}]
    result, joern = export(tmp_path, monkeypatch, rows)
    assert result["counts"]["failed"] == 2
    assert result["counts"]["success"] == 1
    assert [len(r["commands"]) for r in result["records"]] == [1, 2, 2]
    assert [r["status"] for r in result["records"]] == ["error", "error", "ok"]
    assert "Joern parse failed" in result["records"][0]["error"]
    assert "Joern export failed" in result["records"][1]["error"]
    assert json.loads(Path(result["labels"]).read_text()) == {"c": 1}
    assert len(joern.calls) == 5
    assert Path(result["manifest"]).exists()


def test_multiple_function_dots_are_ambiguous_without_explicit_selection(tmp_path, monkeypatch):
    result, _ = export(tmp_path, monkeypatch, [{"id": "a", "func": "int a(){}", "target": 0}],
                       exports={"0-cpg.dot": method_dot("first"), "1-cpg.dot": method_dot("second")})
    record = result["records"][0]
    assert record["status"] == "error"
    assert "found 2" in record["error"]
    assert len(record["method_candidates"]) == 2
    assert "selected_method_dot" not in record
    assert list(Path(result["graph_dir"]).glob("*.dot")) == []


def test_global_and_explicit_external_methods_are_excluded(tmp_path, monkeypatch):
    result, _ = export(tmp_path, monkeypatch, [{"id": "a", "func": "int a(){}", "target": 1}],
                       exports={"0-cpg.dot": method_dot("<global>"),
                                "1-cpg.dot": method_dot("library", external=True),
                                "8-cpg.dot": method_dot("target")})
    record = result["records"][0]
    assert record["status"] == "ok"
    assert record["selected_method_dot"] == "8-cpg.dot"
    assert [candidate["eligible"] for candidate in record["method_candidates"]] == [False, False, True]


def test_explicit_method_map_selects_and_records_actual_dot(tmp_path, monkeypatch):
    result, _ = export(tmp_path, monkeypatch, [{"id": "a", "func": "int a(){}", "target": 1}],
                       exports={"0-cpg.dot": method_dot("wrong"),
                                "nested/7-cpg.dot": method_dot("right")},
                       method_map={"a": "nested/7-cpg.dot"})
    record = result["records"][0]
    assert record["status"] == "ok"
    assert record["requested_method_dot"] == record["selected_method_dot"] == "nested/7-cpg.dot"
    assert "right" in next(c for c in record["method_candidates"] if c["eligible"] and
                           c["path"] == "nested/7-cpg.dot")["method_names"]


def test_jsonl_failures_filters_and_generated_ids_do_not_shift_labels(tmp_path, monkeypatch):
    dataset = tmp_path / "Chrome-Debian.any"
    dataset.write_text('\n'.join([json.dumps({"id": "skip", "func": "a()", "target": 1}),
                                  "{broken", json.dumps({"func": "b()", "target": 0}),
                                  json.dumps({"id": "last", "func": "c()", "target": 0})]), encoding="utf-8")
    joern = JoernCommandFixture()
    monkeypatch.setattr(cpg_generate.subprocess, "run", joern)
    result = cpg_generate.run_export(dataset, output_root=tmp_path / "output",
                                    source_root=tmp_path / "source", workspace_root=tmp_path / "workspace",
                                    joern_parse="parser", joern_export="exporter", include_labels=[0])
    assert result["counts"] == {"total": 4, "success": 2, "failed": 1, "filtered": 1}
    generated = result["records"][2]
    assert generated["generated_id"] is True
    assert generated["sample_id"] == f"generated:{result['dataset_sha256']}:3"
    assert result["records"][3]["target"] == 0
    assert len(joern.calls) == 4


def test_duplicate_ids_are_not_overwritten_or_exported(tmp_path, monkeypatch):
    result, joern = export(tmp_path, monkeypatch,
                           [{"id": "dup", "func": "a()", "target": 0},
                            {"id": "dup", "func": "b()", "target": 1}])
    assert result["counts"]["failed"] == 2
    assert joern.calls == []
    assert all("Duplicate sample_id" in r["error"] for r in result["records"])


def test_nonempty_output_is_refused_without_deleting_artifacts(tmp_path, monkeypatch):
    dataset = tmp_path / "data.json"
    dataset.write_text('[{"id":"a","func":"a()","target":0}]')
    output = tmp_path / "output"
    output.mkdir()
    preserved = output / "previous.txt"
    preserved.write_text("keep")
    with pytest.raises(FileExistsError, match="not empty"):
        cpg_generate.run_export(dataset, output_root=output)
    assert preserved.read_text() == "keep"


def test_missing_executable_is_a_sample_failure_not_success(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("Joern executable unavailable")
    dataset = tmp_path / "data.json"
    dataset.write_text('[{"id":"a","func":"a()","target":0}]')
    monkeypatch.setattr(cpg_generate.subprocess, "run", missing)
    result = cpg_generate.run_export(dataset, output_root=tmp_path / "output",
                                    source_root=tmp_path / "source", workspace_root=tmp_path / "workspace")
    record = result["records"][0]
    assert record["status"] == "error"
    assert record["commands"][0]["error_type"] == "FileNotFoundError"
    assert result["counts"]["failed"] == 1


def test_parallel_wrapper_imports_the_shared_main_without_side_effects():
    from preprocess import cpg_generate_parallel
    assert cpg_generate_parallel.main is cpg_generate.main


def test_one_dot_containing_multiple_methods_is_not_a_whole_file_fallback(tmp_path, monkeypatch):
    multiple = method_dot().replace('"2" [kind="CALL", code="work()"]',
                                    '"2" [kind="METHOD", code="int other(){}"]')
    result, _ = export(tmp_path, monkeypatch,
                       [{"id": "a", "func": "int a(){}", "target": 0}],
                       exports={"0-cpg.dot": multiple})
    record = result["records"][0]
    assert record["status"] == "error"
    assert "2 METHOD nodes" in record["error"]
    assert len(record["selected_graph_method_ids"]) == 2
