"""Run HGNN/TAF-HGCN x CPG/HCPG on identical IDs, features, split and seeds."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sp4cpg.experiment import mean_run_metrics, read_dataset, sample_records, validate_split, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpg-data", required=True)
    p.add_argument("--hcpg-data", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=str(ROOT / "configs" / "paper.json"))
    p.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026, 2027])
    p.add_argument("--device", default="auto")
    p.add_argument("--epoch", type=int, help="Optional shorter NEW run, recorded in every run")
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        p.error("Training seeds must be unique")
    manifest = json.loads(Path(args.split).read_text(encoding="utf-8"))
    datasets = {"CPG": read_dataset(args.cpg_data), "HCPG": read_dataset(args.hcpg_data)}
    for representation, data in datasets.items():
        if data["representation"] != representation:
            p.error(f"{representation} data has the wrong representation metadata")
        validate_split(manifest, sample_records(data["samples"]))
    if not datasets["CPG"].get("metadata", {}).get("encoder"):
        p.error("Record node-input/encoder settings before running a matched experiment")
    for key in ("encoder",):
        if datasets["CPG"].get("metadata", {}).get(key) != datasets["HCPG"].get("metadata", {}).get(key):
            p.error(f"CPG/HCPG feature configuration differs in {key}")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        p.error("Choose a new output directory; matched runs never silently overwrite/reuse results")
    aggregates = []
    for model in ("HGNN", "TAF-HGCN"):
        for representation, path in (("CPG", args.cpg_data), ("HCPG", args.hcpg_data)):
            summaries = []
            for seed in args.seeds:
                run_dir = output / f"{model}_{representation}" / str(seed)
                command = [sys.executable, str(ROOT / "main.py"), "--config", str(Path(args.config).resolve()),
                           "--data", str(Path(path).resolve()), "--split", str(Path(args.split).resolve()),
                           "--output", str(run_dir), "--model", model, "--seed", str(seed), "--device", args.device]
                if args.epoch is not None:
                    command += ["--epoch", str(args.epoch)]
                subprocess.run(command, cwd=ROOT, check=True)
                summaries.append(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))
            aggregates.append(mean_run_metrics(summaries))
    write_json(output / "matched_summary.json", {"result_kind": "measured_matched_configurations", "configurations": aggregates})


if __name__ == "__main__":
    main()
