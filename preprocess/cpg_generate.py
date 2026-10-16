"""Auditable per-function Joern export for JSON-list and JSONL datasets.

Every invocation writes independent run directories. It retains source files,
raw exports, commands, failures and selected method provenance. No directory is
deleted, no sample ID becomes a pathname, and no command is passed to a shell.
Use the printed run's graphs/ directory for the following HCPG/embedding steps.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sp4cpg.graph import dump_graph, from_pydot

# pydot's parser is not thread-safe; Joern subprocesses still run concurrently.
_DOT_PARSE_LOCK = threading.Lock()


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def safe_sample_name(sample_id):
    """A fixed ASCII basename, including for traversal/shell/Windows-device IDs."""
    return "s-" + _sha256(str(sample_id).encode("utf-8"))


def load_samples(path):
    """Return dataset fingerprint and an ordered report record for every row.

    Missing IDs are explicitly generated from this exact file fingerprint and
    its one-based record number, never presented as recovered historical IDs.
    Invalid/filtered records keep their original position and cannot shift labels.
    """
    path = Path(path)
    raw = path.read_bytes()
    fingerprint = _sha256(raw)
    text = raw.decode("utf-8-sig")
    rows = []
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("JSON input must be a list of function records")
        rows = [(i + 1, row, None) for i, row in enumerate(value)]
    else:
        record_number = 0
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            record_number += 1
            try:
                rows.append((record_number, json.loads(line), None))
            except ValueError as exc:
                rows.append((record_number, None,
                             f"Invalid JSONL at source line {line_number}: {exc}"))
    records = []
    for number, row, parse_error in rows:
        record = {"record_number": number, "status": "pending",
                  "sample_id": f"generated:{fingerprint}:{number}",
                  "id_source": "generated_from_dataset_sha256_and_record_number",
                  "generated_id": True, "target": None, "func": None,
                  "source_sha256": None, "commands": []}
        try:
            if parse_error:
                raise ValueError(parse_error)
            if not isinstance(row, dict):
                raise ValueError("Each record must be a JSON object")
            id_field = "sample_id" if "sample_id" in row else ("id" if "id" in row else None)
            if id_field is not None:
                value = row[id_field]
                if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
                    raise ValueError("sample_id/id must be a nonempty string or integer")
                record.update(sample_id=str(value), id_source=id_field, generated_id=False)
            code = row.get("func", row.get("code"))
            if not isinstance(code, str) or not code.strip():
                raise ValueError("func/code must contain a nonempty function string")
            record["func"] = code
            record["source_sha256"] = _sha256(code.encode("utf-8"))
            label = row.get("target", row.get("label"))
            if type(label) is not int or label not in (0, 1):
                raise ValueError("target/label must be integer 0 or 1")
            if "target" in row and "label" in row and row["target"] != row["label"]:
                raise ValueError("target and label fields disagree")
            record["target"] = label
        except (ValueError, TypeError) as exc:
            record.update(status="error", error_type=type(exc).__name__, error=str(exc))
        records.append(record)
    counts = Counter(record["sample_id"] for record in records)
    for record in records:
        if counts[record["sample_id"]] > 1:
            record.update(status="error", error_type="ValueError",
                          error=f"Duplicate sample_id {record['sample_id']!r}")
    if not records:
        raise ValueError("Dataset contains no records")
    return fingerprint, records


def _unquote(value):
    value = str(value or "")
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except ValueError:
            return value[1:-1]
    return value


def _read_dot(path):
    import pydot
    with _DOT_PARSE_LOCK:
        graphs = pydot.graph_from_dot_data(Path(path).read_text(encoding="utf-8-sig"))
    if not graphs or len(graphs) != 1:
        raise ValueError(f"Expected one graph in exported DOT {Path(path).name}")
    return graphs[0]


def describe_method_dot(path, export_dir):
    """Exclude only explicit global/external evidence; uncertainty stays visible."""
    dot = _read_dot(path)
    graph_name = html.unescape(_unquote(dot.get_name())).strip()
    names = [graph_name] if graph_name else []
    external_flags = []
    def visit(graph):
        for node in graph.get_nodes():
            attrs = {str(k): _unquote(v) for k, v in node.get_attributes().items()}
            label = html.unescape(str(attrs.get("label", "")))
            header = re.split(r"<BR\s*/?>", label.strip("<>"), maxsplit=1, flags=re.I)[0]
            kind = str(attrs.get("kind", "")).upper()
            if kind == "METHOD" or re.match(r"^\(?METHOD(?:[,)]|$)", header, flags=re.I):
                name = attrs.get("name", "")
                if not name:
                    match = re.match(r"^\(?METHOD\s*,\s*([^,)]+)", header, flags=re.I)
                    name = match.group(1).strip() if match else ""
                if name:
                    names.append(html.unescape(name))
                flag = attrs.get("isExternal", attrs.get("IS_EXTERNAL", attrs.get("is_external")))
                external_flags.append(str(flag).lower() in {"true", "1"} if flag is not None else None)
        for child in graph.get_subgraphs():
            visit(child)
    visit(dot)
    reasons = []
    if any(name.strip() in {"<global>", ":<global>", "<global>:<global>"} for name in names):
        reasons.append("explicit_global_method")
    if external_flags and all(flag is True for flag in external_flags):
        reasons.append("explicit_external_method")
    return {"path": Path(path).relative_to(export_dir).as_posix(),
            "graph_name": graph_name, "method_names": sorted(set(names)),
            "eligible": not reasons, "excluded_reasons": reasons}


def select_method_dot(export_dir, selector=None):
    """Return the selected path plus inventory; ambiguity is always an error."""
    export_dir = Path(export_dir)
    candidates = []
    for path in sorted(export_dir.rglob("*.dot")):
        try:
            candidates.append(describe_method_dot(path, export_dir))
        except Exception as exc:
            candidates.append({"path": path.relative_to(export_dir).as_posix(), "eligible": False,
                               "excluded_reasons": ["invalid_dot"], "error": str(exc)})
    eligible = [item for item in candidates if item["eligible"]]
    if selector is not None:
        matches = [item for item in eligible if item["path"] == selector]
        if not matches:
            matches = [item for item in eligible if Path(item["path"]).name == selector]
        if len(matches) != 1:
            return None, candidates, f"--method-dot {selector!r} must select exactly one eligible DOT"
        return export_dir / matches[0]["path"], candidates, None
    if any("invalid_dot" in item["excluded_reasons"] for item in candidates):
        return None, candidates, "An exported DOT could not be inspected; explicitly select a valid method DOT"
    if len(eligible) != 1:
        return None, candidates, (
            f"Expected one eligible function DOT, found {len(eligible)}; "
            "provide --method-dot or per-sample --method-map when ambiguous")
    return export_dir / eligible[0]["path"], candidates, None


def _command(argv, cwd, log_prefix, timeout=None):
    started = time.monotonic()
    result = {"argv": [str(value) for value in argv], "cwd": str(cwd), "shell": False}
    try:
        process = subprocess.run(result["argv"], cwd=str(cwd), shell=False,
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=timeout)
        stdout, stderr = process.stdout or "", process.stderr or ""
        result["returncode"] = int(process.returncode)
    except (OSError, subprocess.TimeoutExpired) as exc:
        stdout, stderr = "", str(exc)
        result.update(returncode=None, error_type=type(exc).__name__, error=str(exc))
    out_path, err_path = Path(str(log_prefix) + ".stdout.log"), Path(str(log_prefix) + ".stderr.log")
    out_path.write_text(stdout, encoding="utf-8")
    err_path.write_text(stderr, encoding="utf-8")
    result.update(stdout_log=str(out_path), stderr_log=str(err_path),
                  stderr_tail=stderr[-4000:], elapsed_seconds=time.monotonic() - started)
    return result


def _export_one(record, *, graph_dir, source_dir, workspace_dir, joern_parse,
                joern_export, export_repr, method_dot, timeout, source_extension,
                dataset_sha256, joern_version):
    record = dict(record)
    token = safe_sample_name(record["sample_id"])
    sample_source = source_dir / token
    sample_workspace = workspace_dir / token
    sample_source.mkdir(parents=True, exist_ok=False)
    sample_workspace.mkdir(parents=True, exist_ok=False)
    source_path = sample_source / ("function." + source_extension)
    source_path.write_bytes(record["func"].encode("utf-8"))
    binary_path = sample_workspace / "cpg.bin"
    export_dir = sample_workspace / "export"
    record.update(source_path=str(source_path), cpg_path=str(binary_path),
                  raw_export_dir=str(export_dir), requested_method_dot=method_dot)
    commands = [
        [joern_parse, str(sample_source), "-o", str(binary_path)],
        [joern_export, str(binary_path), "--repr", export_repr, "--out", str(export_dir)],
    ]
    try:
        for index, argv in enumerate(commands):
            stage = "parse" if index == 0 else "export"
            command = _command(argv, sample_workspace, sample_workspace / stage, timeout)
            record["commands"].append(command)
            if command["returncode"] != 0:
                raise RuntimeError(f"Joern {stage} failed; see {command['stderr_log']}")
        selected, candidates, selection_error = select_method_dot(export_dir, method_dot)
        record["method_candidates"] = candidates
        if selection_error:
            raise ValueError(selection_error)
        record["selected_method_dot"] = selected.relative_to(export_dir).as_posix()
        with _DOT_PARSE_LOCK:
            # This operation only traverses the already parsed object; keeping the
            # lock also makes future parser integration safe for parallel exports.
            dot = _read_dot_unlocked(selected)
            graph = from_pydot(dot, graph_id=record["sample_id"])
        graph["graph_id"] = record["sample_id"]
        method_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "METHOD"]
        record["selected_graph_method_ids"] = method_ids
        if len(method_ids) != 1:
            raise ValueError(
                f"Selected DOT contains {len(method_ids)} METHOD nodes; "
                "per-function export requires exactly one, without a whole-file fallback")
        graph["metadata"].update(
            representation="CPG", label=record["target"],
            source_sha256=record["source_sha256"], dataset_sha256=dataset_sha256,
            id_source=record["id_source"], generated_id=record["generated_id"],
            joern_version=joern_version, selected_method_dot=record["selected_method_dot"],
            exported_method_id=method_ids[0],
        )
        if not graph["nodes"]:
            raise ValueError("Selected DOT has no nodes")
        output = graph_dir / f"{token}-cpg-{record['target']}.dot"
        dump_graph(graph, output)
        record.update(status="ok", dot_path=str(output), original_node_count=len(graph["nodes"]))
    except Exception as exc:
        record.update(status="error", error_type=type(exc).__name__, error=str(exc))
    return record


def _read_dot_unlocked(path):
    import pydot
    graphs = pydot.graph_from_dot_data(Path(path).read_text(encoding="utf-8-sig"))
    if not graphs or len(graphs) != 1:
        raise ValueError("Selected DOT must contain exactly one graph")
    return graphs[0]


def run_export(input_path, *, output_root=None, source_root=None, workspace_root=None,
               joern_parse="joern-parse", joern_export="joern-export", joern_version=None,
               workers=1, include_labels=None, method_dot=None, method_map=None,
               export_repr="cpg14", timeout=None, source_extension="c"):
    """Export all valid rows, record every failure, and return the manifest."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    if source_extension not in {"c", "cpp"}:
        raise ValueError("source_extension must be c or cpp")
    if include_labels is not None and not set(include_labels) <= {0, 1}:
        raise ValueError("include_labels must contain only 0 and/or 1")
    input_path = Path(input_path).resolve()
    # Resolve dataset/command/map validation before creating output directories.
    dataset_sha256, records = load_samples(input_path)
    selectors = {str(k): str(v) for k, v in (method_map or {}).items()}
    unknown_selectors = set(selectors) - {record["sample_id"] for record in records}
    if unknown_selectors:
        raise ValueError(f"--method-map contains unknown sample IDs: {sorted(unknown_selectors)}")
    def executable(value):
        value = str(value)
        return str(Path(value).expanduser().resolve()) if "/" in value or "\\" in value else value
    parse_exe, export_exe = executable(joern_parse), executable(joern_export)
    for name, value in (("joern_parse", parse_exe), ("joern_export", export_exe)):
        if not value.strip():
            raise ValueError(f"{name} cannot be empty")
    run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    run_dir = (Path(output_root).resolve() if output_root else
               REPO_ROOT / "preprocess" / "dots-cpg" / run_id)
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise FileExistsError("Output directory is not empty; choose a fresh --output directory")
    source_dir = (Path(source_root) if source_root else REPO_ROOT / "preprocess" / "source").resolve() / run_id
    workspace_dir = (Path(workspace_root) if workspace_root else REPO_ROOT / "preprocess" / "workspace").resolve() / run_id
    graph_dir = run_dir / "graphs"
    for directory in (graph_dir, source_dir, workspace_dir):
        directory.mkdir(parents=True, exist_ok=False)
    jobs = []
    for record in records:
        if record["status"] == "error":
            continue
        if include_labels is not None and record["target"] not in include_labels:
            record.update(status="filtered", filter_reason="label_not_in_include_labels")
            continue
        jobs.append(record)
    options = dict(graph_dir=graph_dir, source_dir=source_dir, workspace_dir=workspace_dir,
                   joern_parse=parse_exe, joern_export=export_exe, export_repr=export_repr,
                   timeout=timeout, source_extension=source_extension,
                   dataset_sha256=dataset_sha256, joern_version=joern_version)
    def export(record):
        try:
            return _export_one(record, method_dot=selectors.get(record["sample_id"], method_dot), **options)
        except Exception as exc:
            return dict(record, status="error", error_type=type(exc).__name__, error=str(exc))
    if workers == 1:
        completed = list(map(export, jobs))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            completed = list(pool.map(export, jobs))
    by_number = {r["record_number"]: r for r in completed}
    records = [by_number.get(r["record_number"], r) for r in records]
    successes = [r for r in records if r["status"] == "ok"]
    labels_path, samples_path = run_dir / "labels.json", run_dir / "samples.jsonl"
    labels_path.write_text(json.dumps({r["sample_id"]: r["target"] for r in successes},
                                     indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with samples_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in successes:
            sample = {k: record[k] for k in (
                "sample_id", "target", "func", "source_sha256", "dot_path", "generated_id")}
            sample.update(id=record["sample_id"], label=record["target"])
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": 2, "run_id": run_id, "input": str(input_path),
        "dataset_sha256": dataset_sha256, "historical_paper_split": False,
        "run_dir": str(run_dir), "graph_dir": str(graph_dir), "source_dir": str(source_dir),
        "workspace_dir": str(workspace_dir), "manifest": str(manifest_path),
        "labels": str(labels_path), "samples_jsonl": str(samples_path),
        "joern": {"parse_executable": parse_exe, "export_executable": export_exe,
                  "version": joern_version,
                  "version_source": "user_supplied" if joern_version else "unrecorded",
                  "export_repr": export_repr},
        "workers": workers, "include_labels": include_labels,
        "counts": {"total": len(records), "success": len(successes),
                   "failed": sum(r["status"] == "error" for r in records),
                   "filtered": sum(r["status"] == "filtered" for r in records)},
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None, *, default_workers=1):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", type=Path, help="JSON list or JSONL, independent of dataset name")
    group.add_argument("--dataset", default=None, help="Compatibility: filename below repository data/")
    parser.add_argument("--output", "--output-root", dest="output", type=Path,
                        help="Fresh fixed run directory; default: preprocess/dots-cpg/run-<unique ID>")
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "preprocess" / "source")
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT / "preprocess" / "workspace")
    parser.add_argument("--joern-parse", default="joern-parse")
    parser.add_argument("--joern-export", default="joern-export")
    parser.add_argument("--joern-version", help="Exact installed version, recorded as user supplied")
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--include-label", action="append", type=int, choices=(0, 1))
    parser.add_argument("--method-dot", help="Export-relative filename/path; never implicitly selects 0-cpg.dot")
    parser.add_argument("--method-map", type=Path, help="JSON {sample_id: export-relative DOT filename}")
    parser.add_argument("--export-repr", default="cpg14")
    parser.add_argument("--timeout", type=float, help="Per Joern command timeout in seconds")
    parser.add_argument("--source-extension", choices=("c", "cpp"), default="c")
    args = parser.parse_args(argv)
    input_path = args.input or REPO_ROOT / "data" / (args.dataset or "function.json")
    try:
        method_map = json.loads(args.method_map.read_text(encoding="utf-8")) if args.method_map else None
        if method_map is not None and not isinstance(method_map, dict):
            raise ValueError("--method-map must be a JSON ID-to-filename object")
        manifest = run_export(input_path, output_root=args.output, source_root=args.source_root,
                              workspace_root=args.workspace_root, joern_parse=args.joern_parse,
                              joern_export=args.joern_export, joern_version=args.joern_version,
                              workers=args.workers, include_labels=args.include_label,
                              method_dot=args.method_dot, method_map=method_map,
                              export_repr=args.export_repr, timeout=args.timeout,
                              source_extension=args.source_extension)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"CPG export configuration/input error: {exc}\n")
    print(json.dumps({"manifest": manifest["manifest"], "graph_dir": manifest["graph_dir"],
                      "labels": manifest["labels"], "samples_jsonl": manifest["samples_jsonl"],
                      "counts": manifest["counts"]}, indent=2, ensure_ascii=False))
    return 1 if manifest["counts"]["failed"] or not manifest["counts"]["success"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
