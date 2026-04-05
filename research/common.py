from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import fcntl


EMITTERS = {
    "local_tuner": "Exploit a nearby promising configuration.",
    "optimizer_hacker": "Change optimizer, scheduler, or update dynamics.",
    "architecture_mutator": "Attempt one structural model change.",
    "simplifier": "Delete or compress complexity around a working idea.",
    "contrarian": "Try the opposite of the recent local trend.",
    "recombinator": "Combine two promising near-miss motifs.",
    "anomaly_chaser": "Investigate a surprising win, loss, or crash.",
}

DEFAULT_PHASE = "setup"
DISCOVERY_RELATIVE_DIR = Path("results") / "discovery"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()



def stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]



def repo_root_from(start: Path) -> Path:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(proc.stdout.strip())
    except Exception:
        # .codex/hooks/<script>.py -> repo root is two levels up
        if start.is_file():
            return start.resolve().parents[2]
        return start.resolve()



def discovery_dir(root: Path) -> Path:
    path = root / DISCOVERY_RELATIVE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path



def state_path(root: Path) -> Path:
    return discovery_dir(root) / "state.json"



def events_path(root: Path) -> Path:
    return discovery_dir(root) / "events.jsonl"



def run_review_path(root: Path) -> Path:
    return discovery_dir(root) / "last_review.md"



def plan_path(root: Path) -> Path:
    return discovery_dir(root) / "current_plan.json"



def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)



def save_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)



def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")



def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")



def git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()



def file_exists(root: Path, relative: str | Path) -> bool:
    return (root / relative).exists()


@contextlib.contextmanager
def state_lock(root: Path) -> Iterator[None]:
    lock_path = discovery_dir(root) / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
