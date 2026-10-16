#!/usr/bin/env python
"""Write physical directed depth/hop statistics for supplied graph assets."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sp4cpg.statistics import graph_statistics
from scripts.validate_dependencies import load_collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Canonical JSON/DOT graph file or directory")
    parser.add_argument("--output", required=True, help="JSON statistics report destination")
    args = parser.parse_args(argv)
    try:
        graphs = load_collection(args.input)
        statistics = [graph_statistics(graphs[graph_id]) for graph_id in sorted(graphs)]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Unable to compute graph statistics: {exc}\n")
    result = {"schema_version": 2, "provenance": "computed_from_supplied_active_graphs", "graphs": statistics}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination), "graphs": len(statistics)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
