from __future__ import annotations

import subprocess


def run_local(command: list[str]) -> int:
    """Run the experiment command locally with inherited stdio."""
    return subprocess.run(command, check=False).returncode

