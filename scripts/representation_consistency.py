"""RQ6: cosine similarity of paired mean CodeBERT graph vectors (not TAF logits)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sp4cpg.experiment import file_sha256, read_dataset, sample_records, validate_split, write_json


def compare(cpg, hcpg, ids=None):
    import torch
    if cpg["representation"] != "CPG" or hcpg["representation"] != "HCPG":
        raise ValueError("Expected CPG and HCPG datasets, respectively")
    left, right = ({s.sample_id: s for s in data["samples"]} for data in (cpg, hcpg))
    if set(left) != set(right) or sample_records(cpg["samples"]) != sample_records(hcpg["samples"]):
        raise ValueError("Paired datasets must share all sample IDs, labels and original CPG sizes")
    if not cpg.get("metadata", {}).get("encoder") or cpg["metadata"]["encoder"] != hcpg.get("metadata", {}).get("encoder"):
        raise ValueError("Both datasets need identical recorded node-input and encoder settings")
    rows = []
    for sample_id in sorted(ids if ids is not None else left):
        a, b = left[sample_id].x.mean(dim=0), right[sample_id].x.mean(dim=0)
        if a.shape != b.shape:
            raise ValueError("CPG/HCPG feature widths differ")
        denominator = a.norm() * b.norm()
        similarity = float(torch.dot(a, b) / denominator) if float(denominator) > 0 else None
        rows.append({"sample_id": sample_id, "cosine_similarity": similarity,
                     "status": "ok" if similarity is not None else "undefined_zero_vector"})
    values = [r["cosine_similarity"] for r in rows if r["cosine_similarity"] is not None]
    return {"result_kind": "measured_representational_consistency", "pairs": len(rows),
            "defined_pairs": len(values), "undefined_pairs": len(rows) - len(values),
            "min": min(values) if values else None, "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None, "per_pair": rows,
            "interpretation": "CodeBERT node vectors mean-pooled; no TAF propagation/classifier; not a dependency-preservation proof"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpg-data", required=True)
    p.add_argument("--hcpg-data", required=True)
    p.add_argument("--split", help="If supplied, restrict to its complete test IDs")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cpg, hcpg = read_dataset(args.cpg_data), read_dataset(args.hcpg_data)
    ids = None
    if args.split:
        manifest = json.loads(Path(args.split).read_text(encoding="utf-8"))
        ids = validate_split(manifest, sample_records(cpg["samples"]))["test"]
        validate_split(manifest, sample_records(hcpg["samples"]))
    report = compare(cpg, hcpg, ids)
    report.update({"cpg_sha256": file_sha256(args.cpg_data), "hcpg_sha256": file_sha256(args.hcpg_data),
                   "split_sha256": file_sha256(args.split) if args.split else None,
                   "encoder": cpg["metadata"]["encoder"]})
    write_json(args.output, report)


if __name__ == "__main__":
    main()
