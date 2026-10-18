"""Parallel HCPG construction with the common graph core."""
import os
import sys

try:
    from .hcpg_generate import main, run
except ImportError:
    from hcpg_generate import main, run


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not any(argument == "--workers" or argument.startswith("--workers=") for argument in arguments):
        arguments += ["--workers", str(min(4, os.cpu_count() or 1))]
    raise SystemExit(main(arguments))
