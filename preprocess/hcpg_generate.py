"""Construct HCPG DOT or JSON files from original binary CPG files."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path
import re

try:
    from .graph_core import dump_graph, load_graph, transform_graph
except ImportError:
    from graph_core import dump_graph, load_graph, transform_graph


def _output_name(path, output_format):
    stem = re.sub(r"-cpg-([01])$", r"-hcpg-\1", path.stem, flags=re.I)
    return stem + "." + output_format


def _one_file(paths):
    source, destination = (Path(value) for value in paths)
    try:
        original = load_graph(source)
        graph = transform_graph(original)
        dump_graph(graph, destination)
        return {"input": str(source), "output": str(destination), "graph_id": graph["graph_id"],
                "status": "ok", "input_nodes": len(original["nodes"]), "output_nodes": len(graph["nodes"]),
                "input_relations": len(original["relations"]), "output_relations": len(graph["relations"])}
    except Exception as error:
        return {"input": str(source), "output": str(destination), "status": "error",
                "error_type": type(error).__name__, "error": str(error)}


def run(input_path, output_path, *, output_format="dot", workers=1, executor="process", report_path=None):
    """Process sorted files with the same serial or parallel construction core."""
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output_format not in {"dot", "json"} or executor not in {"process", "thread"}:
        raise ValueError("Expected dot/json output and process/thread executor")
    if workers < 1:
        raise ValueError("workers must be positive")
    if source.is_file():
        inputs = [source]
        outputs = [destination if not destination.is_dir() and destination.suffix.lower() in {".dot", ".json"}
                   else destination / _output_name(source, output_format)]
    else:
        if destination == source or source in destination.parents:
            raise ValueError("Output graphs must be outside the input graph directory")
        inputs = sorted((path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in {".dot", ".json"}
                         and path.name != "construction_report.json"), key=lambda path: path.relative_to(source).as_posix())
        outputs = [destination / path.relative_to(source).parent / _output_name(path, output_format) for path in inputs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Multiple input files would share an output path")
    if any(path in inputs for path in outputs):
        raise ValueError("Original graph files cannot be overwritten")
    report = Path(report_path).resolve() if report_path else (
        outputs[0].with_suffix(".construction_report.json") if source.is_file() else destination / "construction_report.json")
    if report in inputs or report in outputs:
        raise ValueError("The report path must differ from graph paths")
    if workers > 1 and executor == "thread" and any(path.suffix.lower() == ".dot" for path in inputs):
        raise ValueError("DOT parsing requires the process executor when using multiple workers")
    jobs = [(str(a), str(b)) for a, b in zip(inputs, outputs)]
    if workers == 1:
        records = [_one_file(job) for job in jobs]
    else:
        pool_type = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
        with pool_type(max_workers=workers) as pool:
            records = list(pool.map(_one_file, jobs))
    result = {"schema_version": 2, "succeeded": sum(record["status"] == "ok" for record in records),
              "failed": sum(record["status"] == "error" for record in records), "files": records}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


class HCPGDot:
    """Directory interface for callers constructing DOT datasets."""
    def __init__(self, dot_dir=None, output_dir=None):
        base = Path(__file__).resolve().parent
        self.dot_dir = Path(dot_dir) if dot_dir is not None else base / "dots-cpg"
        self.output_dir = Path(output_dir) if output_dir is not None else base / "dots-hcpg"

    def dependency_preserving_graph_pruning(self):
        return run(self.dot_dir, self.output_dir)

    def semantic_preserving_graph_pruning(self):
        return self.dependency_preserving_graph_pruning()


def main(argv=None):
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--dot-dir", dest="input", default=str(base / "dots-cpg"))
    parser.add_argument("--output", "--output-dir", dest="output", default=str(base / "dots-hcpg"))
    parser.add_argument("--format", choices=("dot", "json"), default="dot")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--report")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.input, args.output, output_format=args.format, workers=args.workers,
                 executor=args.executor, report_path=args.report)
    if not args.quiet:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(result["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
