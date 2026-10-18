"""Export function-level CPG files in independent worker processes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

try:
    from .source_identity import run_export
except ImportError:
    from source_identity import run_export


def process_one(task):
    """Export one function using task-local paths."""
    idx, code, label, source_root, dot_output_dir, joern_workspace = task
    renamed_dot = Path(dot_output_dir) / f"{idx}-cpg-{label}.dot"
    try:
        run_export(code, renamed_dot, source_root=source_root, workspace_root=joern_workspace)
    except Exception as error:
        return (idx, False, f"[!] Function {idx} failed: {error}")
    return (idx, True, f"[+] Function {idx} --> {renamed_dot}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PrimeVul.json", type=str, help="dataset name")
    parser.add_argument("--workers", default=max(1, (os.cpu_count() or 2) // 2), type=int,
                        help="parallel workers (joern is heavy; start with 2~4)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    dataset_file = Path("../data") / args.dataset
    source_root, dot_output_dir, joern_workspace = Path("source"), Path("dots-cpg"), Path("workspace")
    for directory in (source_root, dot_output_dir, joern_workspace):
        directory.mkdir(parents=True, exist_ok=True)
    with dataset_file.open("r", encoding="utf-8") as stream:
        if args.dataset in {"PrimeVul.json", "Chrome-Debian.json"}:
            data = [json.loads(line) for line in stream if line.strip()]
        elif args.dataset == "FFmpeg-Qemu.json":
            data = json.load(stream)
        else:
            raise ValueError(f"Unknown dataset format for {args.dataset}")
    tasks = [(idx, item["func"], item["target"], str(source_root), str(dot_output_dir), str(joern_workspace))
             for idx, item in enumerate(data)]
    print(f"Total samples: {len(tasks)}, workers: {args.workers}")
    succeeded = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]
        for future in as_completed(futures):
            _, ok, message = future.result()
            print(message, flush=True)
            if ok:
                succeeded += 1
            else:
                failed += 1
    print(f"Done. success={succeeded}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
