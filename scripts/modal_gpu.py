#!/usr/bin/env python3
"""
Compatibility shim for running the repo-local experiment runner on the Modal backend.

Prefer:
    python scripts/run_experiment.py --backend modal --gpu GPU --timeout MINS -- COMMAND...
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import run_experiment
except ModuleNotFoundError:  # pragma: no cover - importlib-based tests/loaders
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_experiment


def main(argv: list[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    return run_experiment.main(["--backend", "modal", *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
