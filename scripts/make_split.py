"""Create a new reproducible split, or validate a supplied historical split."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sp4cpg.experiment import make_split, read_dataset, sample_records, validate_split, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025, help="New split seed, independent of training seeds")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    records = sample_records(read_dataset(args.data)["samples"])
    output = Path(args.output)
    if args.validate_only:
        validate_split(json.loads(output.read_text(encoding="utf-8")), records)
        print(f"Validated {output}: {len(records)} aligned samples")
    else:
        if output.exists():
            parser.error("Output already exists; reuse/validate it or choose a new path")
        manifest = make_split(records, args.seed)
        write_json(output, manifest)
        print(f"Created NEW split {output}; this does not recover the historical paper split")


if __name__ == "__main__":
    main()
