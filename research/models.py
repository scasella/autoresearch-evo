from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunMetrics:
    val_bpb: float
    training_seconds: float | None = None
    total_seconds: float | None = None
    peak_vram_mb: float | None = None
    mfu_percent: float | None = None
    total_tokens_M: float | None = None
    num_steps: int | None = None
    num_params_M: float | None = None
    depth: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    plan_id: str
    phase: str
    emitter: str
    niche: str
    hypothesis: str
    predicted_direction: str
    rationale: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    branch: str
    commit: str | None
    commit_parent: str | None
    description: str
    status: str
    category: str
    niche: str
    phase: str
    emitter: str
    recommendation: str
    diff_summary: str
    metrics: dict[str, Any]
    fitness_gain: float
    novelty_bonus: float
    information_gain: float
    surprise_bonus: float
    complexity_cost: float
    vram_cost: float
    discovery_score: float
    improvement_vs_best: float | None
    improvement_vs_baseline: float | None
    crash_signature: str | None = None
    conjecture_updates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
