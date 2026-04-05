#!/usr/bin/env python3
from __future__ import annotations

from _util import emit_json, read_payload, repo_root

root = repo_root()
payload = read_payload()
prompt = str(payload.get("prompt") or "")

from research.core import build_prompt_context, record_autonomy_preference  # noqa: E402

enabled, state = record_autonomy_preference(root, prompt)
context = build_prompt_context(root, state, prompt)
messages = [context]
if enabled is True:
    messages.append("Autonomous continuation is now enabled for this repo until the user explicitly pauses or stops it.")
elif enabled is False:
    messages.append("Autonomous continuation is now disabled. End the session naturally unless the user explicitly restarts it.")

emit_json(
    {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join(messages),
        }
    }
)
