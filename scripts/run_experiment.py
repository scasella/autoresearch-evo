#!/usr/bin/env python3
"""
Run an experiment command through a repo-local backend.

Usage:
    python scripts/run_experiment.py [--backend BACKEND] [--gpu GPU] [--timeout MINS] -- COMMAND...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from backends.local_backend import run_local
    from backends.modal_backend import DEFAULT_TIMEOUT_MINUTES, SUPPORTED_GPUS, run_modal
except ModuleNotFoundError:  # pragma: no cover - importlib-based tests/loaders
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backends.local_backend import run_local
    from backends.modal_backend import DEFAULT_TIMEOUT_MINUTES, SUPPORTED_GPUS, run_modal


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if any(arg in {"-h", "--help"} for arg in argv):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--backend", choices=("local", "modal"), default="local")
        parser.add_argument("--gpu", choices=SUPPORTED_GPUS, default="T4")
        parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_MINUTES, help="backend timeout in minutes")
        parser.print_help()
        raise SystemExit(0)
    if "--" not in argv:
        raise SystemExit(
            "usage: python scripts/run_experiment.py [--backend BACKEND] [--gpu GPU] [--timeout MINS] -- COMMAND..."
        )
    split = argv.index("--")
    args = argv[:split]
    command = argv[split + 1 :]
    if not command:
        raise SystemExit("expected a command after `--`")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "modal"), default="local")
    parser.add_argument("--gpu", choices=SUPPORTED_GPUS, default="T4")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_MINUTES, help="backend timeout in minutes")
    return parser.parse_args(args), command


def run_local_command(command: list[str]) -> int:
    return run_local(command)


def run_modal_command(command: list[str], *, gpu: str, timeout: int, repo_root: Path) -> int:
    return run_modal(command, gpu=gpu, timeout=timeout, repo_root=repo_root)


def main(argv: list[str] | None = None) -> int:
    args, command = parse_args(list(sys.argv[1:] if argv is None else argv))
    repo_root = Path(__file__).resolve().parents[1]
    if args.backend == "local":
        return run_local_command(command)
    return run_modal_command(command, gpu=args.gpu, timeout=args.timeout, repo_root=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
