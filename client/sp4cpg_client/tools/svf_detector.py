
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

try:
    from .codebert_common import load_function_dataset
except Exception:
    from codebert_common import load_function_dataset


def find_wpa(MEDE_home: str) -> str:
    candidates = []
    if MEDE_home:
        root = Path(MEDE_home)
        candidates.extend([
            root / "Release-build" / "bin" / "wpa.exe",
            root / "Release-build" / "bin" / "wpa",
            root / "build" / "bin" / "wpa.exe",
            root / "build" / "bin" / "wpa",
            root / "bin" / "wpa.exe",
            root / "bin" / "wpa",
        ])
    for p in candidates:
        if p.exists(): return str(p)
    return shutil.which("wpa") or ""


def run_real_MEDE(bitcode: str, MEDE_home: str, out_dir: Path) -> List[Dict[str, Any]]:
    if not bitcode:
        raise ValueError("No LLVM bitcode path was provided.")
    bc = Path(bitcode)
    if not bc.exists():
        raise FileNotFoundError(f"Bitcode file not found: {bc}")
    wpa = find_wpa(MEDE_home)
    if not wpa:
        raise FileNotFoundError("MEDE wpa executable not found. Set MEDE_HOME or add wpa to PATH.")
    raw = out_dir / "MEDE_raw_output.txt"
    cmd = [wpa, "-ander", str(bc)]
    print(f"[MEDE] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, errors="replace")
    raw.write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"MEDE returned {proc.returncode}. Raw output: {raw}")
    alerts: List[Dict[str, Any]] = []
    for idx, line in enumerate((proc.stdout or "").splitlines()):
        if re.search(r"warning|error|leak|uaf|use.after.free|null|overflow|bug|vuln", line, re.I):
            alerts.append({
                "sample_id": None,
                "mapping": "unresolved_warning_line_not_sample_id",
                "function_name": "unknown_function",
                "severity": "medium",
                "category": "MEDE_output_signal",
                "message": line[:500],
                "detector": "MEDE",
                "raw_output": str(raw),
            })
    if not alerts:
        alerts.append({
            "sample_id": "MEDE-global",
            "function_name": "project",
            "severity": "info",
            "category": "MEDE_completed",
            "message": "MEDE finished without parser-recognized warning lines. See raw output for details.",
            "detector": "MEDE",
            "raw_output": str(raw),
        })
    return alerts


def fallback_static_scan(data: Path) -> List[Dict[str, Any]]:
    rows = load_function_dataset(data)
    alerts: List[Dict[str, Any]] = []
    for row in rows:
        code = row["func"]
        sid = row["sample_id"]
        fname = row["function_name"]
        # Memory-management oriented checks designed to imitate the expected shape of MEDE evidence.
        malloc_vars = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:\([^)]*\)\s*)?malloc\s*\(", code)
        free_vars = re.findall(r"\bfree\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", code)
        for var in free_vars:
            tail = code.split(f"free({var}", 1)[-1]
            if re.search(rf"\b{re.escape(var)}\b\s*(?:\[|->|=|\))", tail):
                alerts.append({
                    "sample_id": sid,
                    "function_name": fname,
                    "file": row.get("file", ""),
                    "severity": "high",
                    "category": "possible_use_after_free",
                    "message": f"Pointer '{var}' is referenced after free in fallback static scan.",
                    "detector": "fallback_static_not_MEDE",
                })
        for var in malloc_vars:
            if var not in free_vars and "return " not in code:
                alerts.append({
                    "sample_id": sid,
                    "function_name": fname,
                    "file": row.get("file", ""),
                    "severity": "medium",
                    "category": "possible_memory_leak",
                    "message": f"Allocated pointer '{var}' is not freed in this function in fallback static scan.",
                    "detector": "fallback_static_not_MEDE",
                })
        if re.search(r"\bstrcpy\s*\(|\bgets\s*\(|\bsprintf\s*\(", code):
            alerts.append({
                "sample_id": sid,
                "function_name": fname,
                "file": row.get("file", ""),
                "severity": "medium",
                "category": "unsafe_c_api",
                "message": "Unsafe C library API observed in fallback static scan.",
                "detector": "fallback_static_not_MEDE",
            })
    if not alerts:
        alerts.append({
            "sample_id": "static-summary",
            "function_name": "project",
            "severity": "info",
            "category": "no_static_alert",
            "message": "Fallback static scan completed and did not find memory-management alerts.",
            "detector": "fallback_static_not_MEDE",
        })
    return alerts


def write_alerts(alerts: List[Dict[str, Any]], output: Path, csv_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"alerts": alerts, "count": len(alerts)}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = sorted({k for a in alerts for k in a.keys()}) or ["sample_id", "message"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(alerts)


def main() -> None:
    parser = argparse.ArgumentParser(description="MEDE/static analysis integration helper")
    parser.add_argument("--data", required=True)
    parser.add_argument("--bitcode", default="")
    parser.add_argument("--MEDE-home", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output).parent
    try:
        alerts = run_real_MEDE(args.bitcode, args.MEDE_home, out_dir)
    except Exception as exc:
        if not args.allow_fallback:
            raise
        print(f"[MEDE:fallback] real MEDE is unavailable: {exc}")
        print("[MEDE:fallback] running built-in memory-oriented static scan so the workflow can finish.")
        alerts = fallback_static_scan(Path(args.data))
    write_alerts(alerts, Path(args.output), Path(args.csv))
    print(f"[MEDE] alerts={len(alerts)} output={args.output}")


if __name__ == "__main__":
    main()
