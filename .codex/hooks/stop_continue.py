#!/usr/bin/env python3
from __future__ import annotations

from _util import emit_json, read_payload, repo_root

root = repo_root()
payload = read_payload()

from research.core import build_next_prompt  # noqa: E402

reason = build_next_prompt(
    root,
    stop_hook_active=bool(payload.get("stop_hook_active")),
    last_assistant_message=(payload.get("last_assistant_message") or None),
)
if reason:
    emit_json(
        {
            "decision": "block",
            "reason": reason,
            "systemMessage": "Autoresearch loop continuing.",
        }
    )
