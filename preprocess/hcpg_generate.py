"""Generate dependency-preserving HCPGs using the shared JSON v2 core.

Examples (from repository root)::

    python preprocess/hcpg_generate.py --input original.dot --output hcpg.json
    python preprocess/hcpg_generate.py --input dots-cpg --output dots-hcpg

Directory mode writes JSON v2 by default and a deterministic construction_report.json.
Incomplete legacy hyperedges fail explicitly; regenerate them from original CPGs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sp4cpg.graph import dump_graph, load_graph, transform_graph


def _construct_one(item: tuple[str, str]) -> dict:
    source, destination = map(Path, item)
    try:
        original = load_graph(source)
        graph = transform_graph(original)
        dump_graph(graph, destination)
        return {"input": str(source), "output": str(destination), "graph_id": graph["graph_id"], "status": "ok",
                "original_nodes": len(original["nodes"]), "physical_nodes": len(graph["nodes"]),
                "original_relations": len(original["relations"]), "relations": len(graph["relations"])}
    except Exception as exc:
        return {"input": str(source), "output": str(destination), "status": "error",
                "error_type": type(exc).__name__, "error": str(exc)}


def run(input_path: str | Path, output_path: str | Path, *, output_format: str = "json",
        workers: int = 1, executor: str = "process", report_path: str | Path | None = None) -> dict:
    """Construct files in sorted input order, returning an ordered failure report.

    Existing output files are refused: accidental replacement of an original CPG
    would invalidate independent dependency verification.
    """
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if workers < 1:
        raise ValueError("workers must be at least one")
    if output_format not in {"json", "dot"}:
        raise ValueError("output_format must be json or dot")
    if source.is_file():
        inputs = [source]
        if destination.suffix.lower() in {".dot", ".json"}:
            outputs = [destination]
        else:
            outputs = [destination / f"{source.stem}.{output_format}"]
    else:
        if destination == source or source in destination.parents:
            raise ValueError("Directory output must be outside the original graph directory")
        inputs = sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in {".dot", ".json"}
                         and p.name != "construction_report.json"), key=lambda p: p.relative_to(source).as_posix())
        outputs = [destination / p.relative_to(source).with_suffix("." + output_format) for p in inputs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Input stems collide in output; keep DOT and JSON inputs in separate directories")
    if any(p.exists() for p in outputs):
        raise FileExistsError("An output graph already exists; choose a fresh output path")
    if source.is_file() and source in outputs:
        raise ValueError("Cannot overwrite the original graph")
    jobs = [(str(a), str(b)) for a, b in zip(inputs, outputs)]
    if workers == 1:
        reports = [_construct_one(job) for job in jobs]
    else:
        pool_class = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor
        if executor not in {"thread", "process"}:
            raise ValueError("executor must be thread or process")
        if executor == "thread" and any(p.suffix.lower() == ".dot" for p in inputs):
            raise ValueError("pydot parsing is not thread-safe; use --executor process for DOT inputs")
        with pool_class(max_workers=workers) as pool:
            reports = list(pool.map(_construct_one, jobs))
    result = {"schema_version": 2, "success": sum(r["status"] == "ok" for r in reports),
              "failed": sum(r["status"] == "error" for r in reports), "reports": reports}
    if report_path is None:
        report_path = (outputs[0].with_suffix(".construction_report.json") if source.is_file()
                       else destination / "construction_report.json")
    report = Path(report_path).resolve()
    if report == source or report in inputs or report in outputs:
        raise ValueError("Report must not overwrite an input or output graph")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--dot-dir", dest="input", default="dots-cpg")
    parser.add_argument("--output", "--output-dir", dest="output", default="dots-hcpg")
    parser.add_argument("--format", choices=("json", "dot"), default="json")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--report")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.input, args.output, output_format=args.format, workers=args.workers,
                 executor=args.executor, report_path=args.report)
    if not args.quiet:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
