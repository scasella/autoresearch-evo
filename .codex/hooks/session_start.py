#!/usr/bin/env python3
from __future__ import annotations

from _util import emit_json, read_payload, repo_root

root = repo_root()
payload = read_payload()

from research.core import build_session_context, load_state  # noqa: E402

state = load_state(root)
context = build_session_context(root, state)
source = payload.get("source") or "startup"
context = f"Session source: {source}\n{context}"

emit_json(
    {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
)
