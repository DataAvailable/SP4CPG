#!/usr/bin/env python
"""Validate supplied original and transformed graph files/directories."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sp4cpg.graph import load_graph
from sp4cpg.validation import aggregate_reports, validate_pair


def is_construction_report(path):
    """Recognize only the sidecar names emitted by the graph construction CLI."""
    name = Path(path).name.lower()
    return name == "construction_report.json" or name.endswith(".construction_report.json")


def load_collection(path):
    root = Path(path)
    if not root.exists():
        raise ValueError(f"Graph input does not exist: {root}")
    if root.is_file() and is_construction_report(root):
        raise ValueError(f"Construction report is not a graph input: {root}")
    files = [root] if root.is_file() else sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".dot"} and not is_construction_report(p)
    )
    if not files:
        raise ValueError(f"No JSON/DOT graphs found in {root}")
    graphs = {}
    for source in files:
        graph = load_graph(source)
        graph_id = graph.get("graph_id")
        if not isinstance(graph_id, str) or not graph_id:
            raise ValueError(f"Missing stable graph_id in {source}")
        if graph_id in graphs:
            raise ValueError(f"Duplicate graph_id {graph_id!r} in {root}; pairs must be unambiguous")
        graphs[graph_id] = graph
    return graphs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="Independent original CPG file or directory")
    parser.add_argument("--transformed", required=True, help="Active HCPG file or directory")
    parser.add_argument("--output", required=True, help="Computed JSON report destination")
    args = parser.parse_args(argv)
    reports, input_errors = [], []
    try:
        originals, outputs = load_collection(args.original), load_collection(args.transformed)
        for graph_id in sorted(set(originals) | set(outputs)):
            if graph_id not in originals or graph_id not in outputs:
                input_errors.append(f"Unmatched graph_id: {graph_id}")
                continue
            reports.append(validate_pair(originals[graph_id], outputs[graph_id]))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        input_errors.append(str(exc))
    result = {
        "schema_version": 2,
        "provenance": "computed_from_supplied_original_and_active_output_graphs",
        "original_input": str(Path(args.original).resolve()),
        "transformed_input": str(Path(args.transformed).resolve()),
        "aggregate": aggregate_reports(reports), "reports": reports, "input_errors": input_errors,
    }
    result["aggregate"]["all_exact_match"] = result["aggregate"]["all_exact_match"] and not input_errors
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination), "graphs": len(reports), "all_exact_match": result["aggregate"]["all_exact_match"], "input_errors": input_errors}))
    return 0 if result["aggregate"]["all_exact_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
