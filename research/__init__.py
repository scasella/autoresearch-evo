"""Repo-local discovery control plane for autoresearch."""

from .core import (
    build_next_prompt,
    build_session_context,
    build_training_review,
    load_state,
    parse_metrics_text,
    record_autonomy_preference,
    record_experiment_from_run,
    select_next_plan,
)

__all__ = [
    "build_next_prompt",
    "build_session_context",
    "build_training_review",
    "load_state",
    "parse_metrics_text",
    "record_autonomy_preference",
    "record_experiment_from_run",
    "select_next_plan",
]
