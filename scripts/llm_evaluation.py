"""Prepare one-call two-shot requests or import provider responses on common IDs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sp4cpg.experiment import file_sha256, validate_id_partition, write_json
from sp4cpg.llm_protocol import collect_responses, prepare_requests


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--functions", required=True, help="JSON list of sample_id/label/func")
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--limits", required=True, help="Actual provider model/version and token budgets")
    prepare.add_argument("--token-counts", required=True, help="Full rendered prompt counts by sample ID and model key")
    prepare.add_argument("--prompt", default=str(ROOT / "preprocess" / "prompt.txt"))
    prepare.add_argument("--output", required=True, help="New directory for requests.jsonl and manifest.json")
    collect = sub.add_parser("collect")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--responses", required=True, help="JSONL model/sample_id/status/model_id/version/response")
    collect.add_argument("--sp4cpg-predictions", required=True)
    collect.add_argument("--sp4cpg-summary", required=True, help="Same model run's summary.json, binding split/prediction hashes")
    collect.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "prepare":
        template = Path(args.prompt).read_text(encoding="utf-8")
        split = validate_id_partition(read_json(args.split))
        requests, manifest = prepare_requests(read_json(args.functions), split["test"],
                                             read_json(args.limits), read_json(args.token_counts), template)
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            p.error("Choose a new requests directory")
        output.mkdir(parents=True, exist_ok=True)
        (output / "requests.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests), encoding="utf-8")
        manifest.update({"split_sha256": file_sha256(args.split), "functions_sha256": file_sha256(args.functions),
                         "complete_test_ids": split["test"],
                         "prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                         "token_counts_sha256": file_sha256(args.token_counts)})
        write_json(output / "manifest.json", manifest)
    else:
        responses = [json.loads(line) for line in Path(args.responses).read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = read_json(args.sp4cpg_summary)
        if summary.get("predictions_sha256") != file_sha256(args.sp4cpg_predictions):
            p.error("SP4CPG predictions do not match the supplied measured run summary")
        report = collect_responses(read_json(args.manifest), responses, read_json(args.sp4cpg_predictions), summary)
        report["response_file_sha256"] = file_sha256(args.responses)
        write_json(args.output, report)
        if not report["complete"]:
            raise SystemExit("Incomplete/error responses: report saved; no comparison metrics produced")


if __name__ == "__main__":
    main()
