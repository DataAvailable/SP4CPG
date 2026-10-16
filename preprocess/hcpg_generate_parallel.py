"""Parallel entry point for the same checked graph-construction implementation.

Accepts all hcpg_generate.py options; defaults to process workers. No pruning
logic is copied here, preventing serial/parallel divergence.
"""
from __future__ import annotations

import os
import sys

try:
    from .hcpg_generate import main
except ImportError:
    from hcpg_generate import main


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--workers" not in arguments:
        arguments = arguments + ["--workers", str(min(4, os.cpu_count() or 1))]
    raise SystemExit(main(arguments))
