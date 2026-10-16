"""Parallel Joern export through the same audited core as cpg_generate.py."""
from __future__ import annotations

import os

try:
    from .cpg_generate import main
except ImportError:
    from cpg_generate import main


if __name__ == "__main__":
    raise SystemExit(main(default_workers=min(4, max(1, (os.cpu_count() or 2) // 2))))
