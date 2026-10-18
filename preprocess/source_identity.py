"""Bind exported DOT graphs to the exact input function text."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


def stamp_dot_source(path, code):
    """Store the SHA256 of the function's UTF-8 text as a DOT graph attribute."""
    if not isinstance(code, str):
        raise TypeError("Function code must be a string")
    import pydot

    path = Path(path)
    graphs = pydot.graph_from_dot_data(path.read_text(encoding="utf-8-sig"))
    if not graphs or len(graphs) != 1:
        raise ValueError("Expected exactly one DOT graph")
    checksum = hashlib.sha256(code.encode("utf-8")).hexdigest()
    graphs[0].set("source_code_sha256", json.dumps(checksum))
    path.write_text(graphs[0].to_string(), encoding="utf-8")
    return checksum


def run_export(code, destination, *, source_root, workspace_root, command_runner=None):
    """Parse, export and stamp one function in fresh input and output directories."""
    if not isinstance(code, str):
        raise TypeError("Function code must be a string")
    destination, source_root, workspace_root = Path(destination), Path(source_root), Path(workspace_root)
    source_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    runner = subprocess.run if command_runner is None else command_runner
    with TemporaryDirectory(prefix="function-", dir=source_root) as source_path, \
            TemporaryDirectory(prefix="export-", dir=workspace_root) as workspace_path:
        source_dir, workspace_dir = Path(source_path), Path(workspace_path)
        (source_dir / "function.c").write_bytes(code.encode("utf-8"))
        cpg_path, dot_dir = workspace_dir / "function.bin", workspace_dir / "dots"
        commands = [
            ["joern-parse", str(source_dir), "-o", str(cpg_path)],
            ["joern-export", str(cpg_path), "--repr", "cpg14", "--out", str(dot_dir)],
        ]
        for command in commands:
            result = runner(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"{command[0]} failed ({result.returncode}): {result.stderr.strip()}")
            if command[0] == "joern-parse" and not cpg_path.is_file():
                raise RuntimeError("joern-parse did not create the CPG file")
        generated_dot = dot_dir / "0-cpg.dot"
        if not generated_dot.is_file():
            raise RuntimeError("joern-export did not create 0-cpg.dot")
        stamp_dot_source(generated_dot, code)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated_dot, destination)
    return destination
