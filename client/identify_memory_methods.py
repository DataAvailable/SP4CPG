#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MEDE Stage 1: Custom Memory Method Identification

This script identifies project-specific custom memory management functions
from C/C++ projects. It extracts Candidate Function Records (CFRs), applies
conservative filtering, classifies candidate functions as allocator / deallocator
/ other, and generates source-sink rules for downstream pointer/value-flow analysis.

Basic usage:
    python mede_identify_memory_methods.py --project E:/Projects/libpng --out outputs/mede

Optional LLM verification:
    set LLM_API_KEY=your_key
    set LLM_BASE_URL=https://api.openai.com/v1
    set LLM_MODEL=gpt-4o-mini
    python mede_identify_memory_methods.py --project E:/Projects/libpng --out outputs/mede --use-llm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SOURCE_EXTS = {
    ".c", ".cc", ".cpp", ".cxx",
    ".h", ".hpp", ".hh", ".hxx",
}

STANDARD_ALLOCATORS = {
    "malloc", "calloc", "realloc", "aligned_alloc",
    "strdup", "strndup", "new", "operator new",
    "kmalloc", "kcalloc", "krealloc", "vmalloc",
    "g_malloc", "g_malloc0", "g_realloc",
    "OPENSSL_malloc", "OPENSSL_zalloc", "BIO_new",
    "PyMem_Malloc", "PyObject_Malloc",
}

STANDARD_DEALLOCATORS = {
    "free", "delete", "operator delete",
    "kfree", "vfree",
    "g_free",
    "OPENSSL_free", "BIO_free", "BIO_free_all",
    "PyMem_Free", "PyObject_Free",
}

ALLOC_NAME_HINTS = {
    "alloc", "malloc", "calloc", "realloc", "new",
    "create", "init", "make", "build", "dup", "clone",
    "copy", "reserve", "grow", "construct", "open",
}

FREE_NAME_HINTS = {
    "free", "release", "destroy", "delete", "dealloc",
    "dispose", "clear", "close", "cleanup", "fini",
    "unref", "drop", "reset",
}

NEGATIVE_NAME_HINTS = {
    "print", "log", "debug", "trace", "format",
    "strcmp", "strlen", "hash", "dump", "show",
    "get", "set", "is", "has", "check", "test",
}

BUFFER_PARAM_HINTS = {
    "buf", "buffer", "data", "ptr", "mem", "memory",
    "out", "dst", "src", "str", "string", "array",
}

HANDLE_TYPE_HINTS = {
    "handle", "ctx", "context", "object", "buffer",
    "stream", "session", "pool", "node", "list", "map",
}


@dataclass
class ParamInfo:
    raw: str
    name: str = ""
    type: str = ""
    pointer_depth: int = 0
    is_const: bool = False


@dataclass
class CandidateFunctionRecord:
    project: str
    file_path: str
    entity_type: str  # function | declaration | macro
    function_name: str
    return_type: str
    params: List[ParamInfo]
    direct_callees: List[str]
    signature: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    file_path: str
    function_name: str
    classification: str  # allocator | deallocator | other
    confidence: float
    reason: str
    evidence: List[str]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//.*", "", code)
    return code


def split_params(param_text: str) -> List[str]:
    params: List[str] = []
    cur: List[str] = []
    depth = 0

    for ch in param_text:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)

        if ch == "," and depth == 0:
            p = "".join(cur).strip()
            if p:
                params.append(p)
            cur = []
        else:
            cur.append(ch)

    last = "".join(cur).strip()
    if last:
        params.append(last)

    if len(params) == 1 and params[0] in {"void", ""}:
        return []

    return params


def parse_param(raw: str) -> ParamInfo:
    raw = raw.strip()
    pointer_depth = raw.count("*")
    is_const = bool(re.search(r"\bconst\b", raw))
    raw_no_default = raw.split("=")[0].strip()

    names = re.findall(r"[A-Za-z_]\w*", raw_no_default)
    name = names[-1] if names else ""

    ptype = raw_no_default
    if name:
        ptype = re.sub(r"\b" + re.escape(name) + r"\b\s*$", "", raw_no_default).strip()

    return ParamInfo(
        raw=raw,
        name=name,
        type=ptype,
        pointer_depth=pointer_depth,
        is_const=is_const,
    )


def extract_direct_callees(body: str) -> List[str]:
    keywords = {
        "if", "for", "while", "switch", "return", "sizeof",
        "catch", "delete", "new",
    }
    names = re.findall(r"\b([A-Za-z_]\w*)\s*\(", body)
    callees: List[str] = []
    for n in names:
        if n not in keywords and n not in callees:
            callees.append(n)
    return callees


def is_probably_control_statement(prefix: str) -> bool:
    last = prefix.strip().split()[-1:] or [""]
    return last[0] in {"if", "for", "while", "switch", "catch"}


def find_matching_brace(code: str, start: int) -> int:
    depth = 0
    for i in range(start, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


FUNC_HEADER_RE = re.compile(
    r"""
    (?P<ret>
        (?:[A-Za-z_]\w*|struct\s+[A-Za-z_]\w*|enum\s+[A-Za-z_]\w*|union\s+[A-Za-z_]\w*)
        (?:
            [\s\*\&]+
            (?:[A-Za-z_]\w*|const|volatile|static|inline|extern|unsigned|signed|long|short)
        )*
        [\s\*\&]*
    )
    \b(?P<name>[A-Za-z_]\w*)\s*
    \(
        (?P<params>[^;{}]*?)
    \)
    \s*
    \{
    """,
    re.X | re.S,
)

DECL_RE = re.compile(
    r"""
    (?P<ret>
        (?:[A-Za-z_]\w*|struct\s+[A-Za-z_]\w*|enum\s+[A-Za-z_]\w*|union\s+[A-Za-z_]\w*)
        (?:
            [\s\*\&]+
            (?:[A-Za-z_]\w*|const|volatile|static|inline|extern|unsigned|signed|long|short)
        )*
        [\s\*\&]*
    )
    \b(?P<name>[A-Za-z_]\w*)\s*
    \(
        (?P<params>[^{}]*?)
    \)
    \s*;
    """,
    re.X | re.S,
)

MACRO_RE = re.compile(
    r"^\s*#\s*define\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^\)]*)\)\s*(?P<body>.*)$",
    re.M,
)


def extract_functions_from_file(project: str, path: Path, root: Path) -> List[CandidateFunctionRecord]:
    raw_code = read_text(path)
    code = strip_comments(raw_code)
    records: List[CandidateFunctionRecord] = []
    rel_path = str(path.relative_to(root))

    for m in FUNC_HEADER_RE.finditer(code):
        name = m.group("name").strip()
        ret = " ".join(m.group("ret").split())
        params_text = m.group("params").strip()

        prefix = code[max(0, m.start() - 20):m.start()]
        if is_probably_control_statement(prefix):
            continue

        brace_pos = code.find("{", m.end() - 1)
        end_pos = find_matching_brace(code, brace_pos)
        body = code[brace_pos + 1:end_pos] if end_pos != -1 else ""

        params = [parse_param(p) for p in split_params(params_text)]
        callees = extract_direct_callees(body)
        signature = f"{ret} {name}({params_text})"

        records.append(CandidateFunctionRecord(
            project=project,
            file_path=rel_path,
            entity_type="function",
            function_name=name,
            return_type=ret,
            params=params,
            direct_callees=callees,
            signature=signature,
        ))

    for m in DECL_RE.finditer(code):
        name = m.group("name").strip()
        ret = " ".join(m.group("ret").split())
        params_text = m.group("params").strip()

        if name in {"if", "for", "while", "switch"}:
            continue

        params = [parse_param(p) for p in split_params(params_text)]
        signature = f"{ret} {name}({params_text})"

        records.append(CandidateFunctionRecord(
            project=project,
            file_path=rel_path,
            entity_type="declaration",
            function_name=name,
            return_type=ret,
            params=params,
            direct_callees=[],
            signature=signature,
        ))

    for m in MACRO_RE.finditer(raw_code):
        name = m.group("name").strip()
        params_text = m.group("params").strip()
        body = m.group("body").strip()
        params = [parse_param(p.strip()) for p in params_text.split(",") if p.strip()]
        callees = extract_direct_callees(body)

        records.append(CandidateFunctionRecord(
            project=project,
            file_path=rel_path,
            entity_type="macro",
            function_name=name,
            return_type="macro",
            params=params,
            direct_callees=callees,
            signature=f"#define {name}({params_text}) {body}",
        ))

    return records


def build_cfrs(project_dir: Path) -> List[CandidateFunctionRecord]:
    project = project_dir.name
    records: List[CandidateFunctionRecord] = []

    for path in project_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SOURCE_EXTS:
            records.extend(extract_functions_from_file(project, path, project_dir))

    seen = set()
    unique: List[CandidateFunctionRecord] = []
    for r in records:
        key = (r.file_path, r.function_name, r.signature)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def has_pointer_return(cfr: CandidateFunctionRecord) -> bool:
    rt = cfr.return_type.lower()
    return "*" in rt or any(h in rt for h in HANDLE_TYPE_HINTS)


def has_pointer_param(cfr: CandidateFunctionRecord) -> bool:
    return any(p.pointer_depth >= 1 for p in cfr.params)


def has_double_pointer_param(cfr: CandidateFunctionRecord) -> bool:
    return any(p.pointer_depth >= 2 for p in cfr.params)


def has_buffer_like_param(cfr: CandidateFunctionRecord) -> bool:
    for p in cfr.params:
        low = f"{p.name} {p.type} {p.raw}".lower()
        if any(h in low for h in BUFFER_PARAM_HINTS) and p.pointer_depth >= 1:
            return True
    return False


def has_memory_callee(cfr: CandidateFunctionRecord, known_memory: Optional[set] = None) -> bool:
    known_memory = known_memory or set()
    callees = set(cfr.direct_callees)
    return bool(
        callees & STANDARD_ALLOCATORS
        or callees & STANDARD_DEALLOCATORS
        or callees & known_memory
    )


def name_contains(name: str, hints: set) -> bool:
    n = name.lower()
    return any(h in n for h in hints)


def filter_candidate(cfr: CandidateFunctionRecord, known_memory: Optional[set] = None) -> Optional[CandidateFunctionRecord]:
    evidence: List[str] = []

    if has_pointer_return(cfr):
        evidence.append("returns pointer/object handle")
    if has_double_pointer_param(cfr):
        evidence.append("has double-pointer output parameter")
    if has_buffer_like_param(cfr):
        evidence.append("has buffer/object pointer parameter")
    if has_memory_callee(cfr, known_memory):
        evidence.append("calls known memory primitive or known custom memory method")
    if name_contains(cfr.function_name, ALLOC_NAME_HINTS):
        evidence.append("function name contains allocation/creation hint")
    if name_contains(cfr.function_name, FREE_NAME_HINTS):
        evidence.append("function name contains release/destruction hint")

    if evidence:
        cfr.evidence = evidence
        return cfr

    return None


def conservative_filter(cfrs: List[CandidateFunctionRecord]) -> List[CandidateFunctionRecord]:
    filtered: List[CandidateFunctionRecord] = []
    for cfr in cfrs:
        kept = filter_candidate(cfr)
        if kept is not None:
            filtered.append(kept)
    return filtered


def classify_by_heuristic(cfr: CandidateFunctionRecord) -> ClassificationResult:
    name = cfr.function_name.lower()
    callees = set(cfr.direct_callees)
    alloc_score = 0.0
    free_score = 0.0
    reasons: List[str] = []

    if has_pointer_return(cfr):
        alloc_score += 0.25
        reasons.append("pointer or handle return")
    if has_double_pointer_param(cfr):
        alloc_score += 0.25
        reasons.append("double-pointer output parameter")
    if callees & STANDARD_ALLOCATORS:
        alloc_score += 0.40
        reasons.append(f"calls allocator primitive: {sorted(callees & STANDARD_ALLOCATORS)}")
    if callees & STANDARD_DEALLOCATORS:
        free_score += 0.45
        reasons.append(f"calls deallocator primitive: {sorted(callees & STANDARD_DEALLOCATORS)}")
    if name_contains(name, ALLOC_NAME_HINTS):
        alloc_score += 0.20
        reasons.append("name suggests allocation/creation")
    if name_contains(name, FREE_NAME_HINTS):
        free_score += 0.25
        reasons.append("name suggests deallocation/destruction")
    if has_buffer_like_param(cfr):
        free_score += 0.10
        alloc_score += 0.05
        reasons.append("has buffer/object pointer parameter")
    if name_contains(name, NEGATIVE_NAME_HINTS) and not (callees & STANDARD_ALLOCATORS or callees & STANDARD_DEALLOCATORS):
        alloc_score -= 0.20
        free_score -= 0.20
        reasons.append("negative name hint without direct memory primitive")

    alloc_score = max(0.0, min(1.0, alloc_score))
    free_score = max(0.0, min(1.0, free_score))

    if alloc_score >= 0.45 and alloc_score >= free_score + 0.10:
        cls = "allocator"
        conf = alloc_score
    elif free_score >= 0.45 and free_score >= alloc_score:
        cls = "deallocator"
        conf = free_score
    else:
        cls = "other"
        conf = max(0.2, max(alloc_score, free_score))

    return ClassificationResult(
        file_path=cfr.file_path,
        function_name=cfr.function_name,
        classification=cls,
        confidence=round(conf, 4),
        reason="; ".join(reasons) if reasons else "no strong memory-management evidence",
        evidence=cfr.evidence,
    )


def build_llm_prompt(cfr: CandidateFunctionRecord) -> str:
    params = [asdict(p) for p in cfr.params]
    cfr_obj = {
        "project": cfr.project,
        "file_path": cfr.file_path,
        "entity_type": cfr.entity_type,
        "function_name": cfr.function_name,
        "return_type": cfr.return_type,
        "params": params,
        "direct_callees": cfr.direct_callees,
        "signature": cfr.signature,
        "filter_evidence": cfr.evidence,
    }
    return f"""
You are identifying project-specific custom memory management functions in C/C++ projects.

Classify the following Candidate Function Record into one of:
- allocator: creates a new dynamic memory object and transfers ownership to caller through return value or output parameter.
- deallocator: releases caller-provided objects, buffers, object fields, or terminates their lifetime.
- other: not a project-specific memory allocation/deallocation interface.

Important:
- If a function only allocates and frees temporary memory internally, classify it as other.
- Do not rely solely on the function name.
- Use return type, parameters, and direct callees as evidence.

Return JSON only:
{{
  "classification": "allocator|deallocator|other",
  "confidence": 0.0,
  "reason": "brief reason"
}}

CFR:
{json.dumps(cfr_obj, ensure_ascii=False, indent=2)}
""".strip()


def call_openai_compatible_api(prompt: str, timeout: int = 60) -> Dict[str, Any]:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set")

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You are a precise C/C++ static-analysis assistant. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))

    content = data["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def classify_by_llm(cfr: CandidateFunctionRecord) -> ClassificationResult:
    try:
        obj = call_openai_compatible_api(build_llm_prompt(cfr))
        cls = obj.get("classification", "other")
        conf = float(obj.get("confidence", 0.0))
        reason = str(obj.get("reason", ""))

        if cls not in {"allocator", "deallocator", "other"}:
            cls = "other"

        return ClassificationResult(
            file_path=cfr.file_path,
            function_name=cfr.function_name,
            classification=cls,
            confidence=round(max(0.0, min(1.0, conf)), 4),
            reason=reason,
            evidence=cfr.evidence,
        )
    except Exception as e:
        fallback = classify_by_heuristic(cfr)
        fallback.reason = f"LLM failed, fallback to heuristic: {e}; {fallback.reason}"
        return fallback


def build_source_sink_rules(results: List[ClassificationResult]) -> Dict[str, Any]:
    allocators = []
    deallocators = []

    for r in results:
        if r.classification == "allocator":
            allocators.append({
                "function": r.function_name,
                "file_path": r.file_path,
                "confidence": r.confidence,
                "source_kind": "heap_object_creation",
            })
        elif r.classification == "deallocator":
            deallocators.append({
                "function": r.function_name,
                "file_path": r.file_path,
                "confidence": r.confidence,
                "sink_kind": "object_lifetime_termination",
            })

    return {
        "version": "1.0",
        "description": "Project-specific custom memory methods identified by MEDE Stage 1",
        "sources": allocators,
        "sinks": deallocators,
        "summary": {
            "allocator_count": len(allocators),
            "deallocator_count": len(deallocators),
        },
    }


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="target C/C++ project directory")
    parser.add_argument("--out", default="mede_outputs", help="output directory")
    parser.add_argument("--use-llm", action="store_true", help="enable LLM semantic validation")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="minimum confidence for final custom memory methods")
    args = parser.parse_args()

    project_dir = Path(args.project).resolve()
    out_dir = Path(args.out).resolve()

    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    print(f"[*] Project: {project_dir}")
    print("[*] Building Candidate Function Records...")
    cfrs = build_cfrs(project_dir)
    save_json(out_dir / "candidates.json", [asdict(x) for x in cfrs])

    print(f"[*] Total CFRs: {len(cfrs)}")
    print("[*] Conservative filtering...")
    filtered = conservative_filter(cfrs)
    save_json(out_dir / "filtered_candidates.json", [asdict(x) for x in filtered])

    print(f"[*] Filtered CFRs: {len(filtered)}")
    print("[*] Semantic classification...")

    results: List[ClassificationResult] = []
    for i, cfr in enumerate(filtered, 1):
        if args.use_llm:
            r = classify_by_llm(cfr)
            time.sleep(0.2)
        else:
            r = classify_by_heuristic(cfr)

        results.append(r)
        print(f"[{i}/{len(filtered)}] {r.classification:11s} {r.confidence:.2f} {r.function_name} ({r.file_path})")

    final_results = [
        r for r in results
        if r.classification in {"allocator", "deallocator"} and r.confidence >= args.min_confidence
    ]

    save_json(out_dir / "validation_results_all.json", [asdict(x) for x in results])
    save_json(out_dir / "custom_memory_methods.json", [asdict(x) for x in final_results])

    rules = build_source_sink_rules(final_results)
    save_json(out_dir / "source_sink_rules.json", rules)

    print("\n✅ Done.")
    print(f"    candidates:              {out_dir / 'candidates.json'}")
    print(f"    filtered candidates:     {out_dir / 'filtered_candidates.json'}")
    print(f"    all validation results:  {out_dir / 'validation_results_all.json'}")
    print(f"    custom memory methods:   {out_dir / 'custom_memory_methods.json'}")
    print(f"    source-sink rules:       {out_dir / 'source_sink_rules.json'}")
    print(f"    allocators: {rules['summary']['allocator_count']}, deallocators: {rules['summary']['deallocator_count']}")


if __name__ == "__main__":
    main()
