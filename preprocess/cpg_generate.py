"""Export function-level CPG files with their source-text identities."""
import argparse
import json
import os
from pathlib import Path

try:
    from .source_identity import run_export
except ImportError:
    from source_identity import run_export


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PrimeVul.json", type=str, help="dataset name")
    parser.add_argument("--workers", default=max(1, (os.cpu_count() or 2) // 2), type=int,
                        help="parallel workers (joern is heavy; start with 2~4)")
    args = parser.parse_args(argv)
    dataset_file = Path("../data") / args.dataset
    source_dir, dot_output_dir, joern_workspace = Path("source"), Path("dots-cpg"), Path("workspace")
    source_dir.mkdir(parents=True, exist_ok=True)
    dot_output_dir.mkdir(parents=True, exist_ok=True)
    joern_workspace.mkdir(parents=True, exist_ok=True)
    with dataset_file.open("r", encoding="utf-8") as stream:
        if args.dataset == "PrimeVul.json":
            data = [json.loads(line) for line in stream if line.strip()]
        elif args.dataset == "FFmpeg_Qemu.json":
            data = json.load(stream)
        else:
            raise ValueError(f"Unknown dataset format for {args.dataset}")
    failed = 0
    for idx, item in enumerate(data):
        code, label = item["func"], item["target"]
        renamed_dot = dot_output_dir / f"{idx}-cpg-{label}.dot"
        print(f"[+] Processing function {idx}")
        try:
            run_export(code, renamed_dot, source_root=source_dir, workspace_root=joern_workspace)
        except Exception as error:
            failed += 1
            print(f"[!] Function {idx} failed: {error}")
            continue
        print(f"    --> Saved to {renamed_dot}")
    print(f"Done. success={len(data) - failed}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())