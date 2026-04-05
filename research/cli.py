from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import repo_root_from
from .core import (
    archive_snapshot,
    build_next_prompt,
    build_session_context,
    load_last_review,
    load_state,
    parse_run_log,
    select_next_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect repo-local autoresearch discovery state.")
    parser.add_argument("command", choices=["summary", "archive", "next", "parse-run", "last-review"])
    parser.add_argument("path", nargs="?", help="Optional path for parse-run.")
    args = parser.parse_args()

    root = repo_root_from(Path.cwd())

    if args.command == "summary":
        state = load_state(root)
        print(build_session_context(root, state))
        return 0

    if args.command == "archive":
        state = load_state(root)
        rows = archive_snapshot(state)
        payload = [
            {
                "niche": niche,
                "best_val_bpb": data.get("best_val_bpb"),
                "commit": data.get("commit"),
                "emitter": data.get("emitter"),
                "description": data.get("description"),
                "count": data.get("count"),
            }
            for niche, data in rows
        ]
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "next":
        state = load_state(root)
        print(json.dumps(select_next_plan(state).to_dict(), indent=2))
        prompt = build_next_prompt(root, persist=False)
        if prompt:
            print("\nNext continuation prompt:\n")
            print(prompt)
        return 0

    if args.command == "parse-run":
        path = Path(args.path or root / "run.log")
        metrics, text = parse_run_log(path)
        payload = {"metrics": metrics.to_dict() if metrics else None, "tail": text.splitlines()[-20:]}
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "last-review":
        review = load_last_review(root)
        if review:
            print(review)
        else:
            print("No review recorded yet.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
