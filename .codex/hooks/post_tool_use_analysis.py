#!/usr/bin/env python3
from __future__ import annotations

from _util import emit_json, read_payload, repo_root

root = repo_root()
payload = read_payload()
command = str((payload.get("tool_input") or {}).get("command") or "")

from research.core import build_training_review, is_training_command, record_experiment_from_run  # noqa: E402

if is_training_command(command):
    experiment, state = record_experiment_from_run(root)
    if experiment is not None:
        review = build_training_review(experiment, state)
        emit_json(
            {
                "decision": "block",
                "reason": review,
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": review,
                },
                "systemMessage": f"Experiment analyzed: {experiment.recommendation}",
            }
        )
