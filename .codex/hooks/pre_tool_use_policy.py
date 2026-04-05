#!/usr/bin/env python3
from __future__ import annotations

from _util import emit_json, read_payload, repo_root

root = repo_root()
payload = read_payload()
command = str((payload.get("tool_input") or {}).get("command") or "")

from research.core import check_command_policy  # noqa: E402

reason = check_command_policy(root, command)
if reason:
    emit_json(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
            "systemMessage": reason,
        }
    )
