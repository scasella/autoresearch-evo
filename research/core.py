from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_PHASE,
    EMITTERS,
    append_jsonl,
    events_path,
    git,
    load_json,
    plan_path,
    run_review_path,
    save_json_atomic,
    stable_id,
    state_lock,
    state_path,
    utcnow_iso,
    write_text,
)
from .models import ExperimentRecord, Plan, RunMetrics


CATEGORY_KEYWORDS = {
    "optimizer": [
        "lr",
        "learning_rate",
        "warmup",
        "decay",
        "adamw",
        "muon",
        "beta",
        "momentum",
        "weight_decay",
        "scheduler",
        "clip",
    ],
    "architecture": [
        "depth",
        "n_head",
        "n_heads",
        "d_model",
        "dim",
        "ffn",
        "mlp",
        "embed",
        "width",
        "residual",
        "block",
        "layer",
    ],
    "attention": [
        "attention",
        "attn",
        "window_pattern",
        "rope",
        "flash",
        "kv",
        "qk",
        "causal",
    ],
    "batch": [
        "batch",
        "device_batch_size",
        "total_batch_size",
        "grad_accum",
        "accum",
        "tokens_per_step",
        "microbatch",
    ],
    "normalization": ["norm", "rmsnorm", "layernorm"],
    "regularization": ["dropout", "stochastic", "drop_path", "label_smoothing"],
}

ENABLE_PATTERNS = [
    r"\bcontinue autonom(?:ous|ously|y)\b",
    r"\bgo autonom(?:ous|ously|y)\b",
    r"\bnever stop\b",
    r"\bcontinue the loop\b",
    r"\bresume (?:the )?(?:autonomous )?(?:autoresearch|experiment|research) loop\b",
    r"\brestart (?:the )?(?:autoresearch|experiment|research) loop\b",
    r"\bstart (?:the )?(?:autonomous )?(?:autoresearch|experiment|research) loop\b",
    r"\bkick off (?:a )?new experiment\b",
]
DISABLE_PATTERNS = [
    r"\bpause\b",
    r"\bstop after\b",
    r"\bdon't continue\b",
    r"\bdo not continue\b",
    r"\bsummarize only\b",
    r"\bhold here\b",
    r"\bstop the loop\b",
    r"\bend the loop\b",
    r"\bdisable autonomy\b",
]


def detect_autonomy_preference(prompt: str) -> bool | None:
    prompt_lower = prompt.lower()
    for pattern in DISABLE_PATTERNS:
        if re.search(pattern, prompt_lower):
            return False
    for pattern in ENABLE_PATTERNS:
        if re.search(pattern, prompt_lower):
            return True
    return None


def initial_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "autonomy_enabled": False,
        "autonomy_reason": None,
        "phase": DEFAULT_PHASE,
        "experiment_count": 0,
        "stall_count": 0,
        "best_val_bpb": None,
        "best_commit": None,
        "best_peak_vram_mb": None,
        "baseline_val_bpb": None,
        "baseline_commit": None,
        "baseline_peak_vram_mb": None,
        "recent_experiments": [],
        "near_misses": [],
        "archive": {},
        "emitter_stats": {
            name: {
                "attempts": 0,
                "improvements": 0,
                "crashes": 0,
                "total_reward": 0.0,
                "last_used_index": 0,
            }
            for name in EMITTERS
        },
        "category_stats": {},
        "crash_signatures": {},
        "conjectures": [],
        "current_plan": None,
        "last_experiment_id": None,
        "ingested_results_tsv_rows": 0,
        "results_tsv_exists": False,
        "updated_at": utcnow_iso(),
    }


def load_state(root: Path) -> dict[str, Any]:
    state = load_json(state_path(root), initial_state())
    state.setdefault("schema_version", 1)
    state.setdefault("phase", DEFAULT_PHASE)
    state.setdefault("emitter_stats", {})
    for name, defaults in initial_state()["emitter_stats"].items():
        state["emitter_stats"].setdefault(name, defaults.copy())
    state.setdefault("archive", {})
    state.setdefault("recent_experiments", [])
    state.setdefault("near_misses", [])
    state.setdefault("category_stats", {})
    state.setdefault("crash_signatures", {})
    state.setdefault("conjectures", [])
    state.setdefault("current_plan", None)
    state.setdefault("ingested_results_tsv_rows", 0)
    state.setdefault("results_tsv_exists", False)
    _ingest_results_tsv(root, state)
    state["conjectures"] = _derive_conjectures(state)
    return state


def save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utcnow_iso()
    save_json_atomic(state_path(root), state)


def _read_results_tsv(root: Path) -> list[dict[str, str]]:
    path = root / "results.tsv"
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append({key: (value or "") for key, value in row.items()})
    return rows


def _ingest_results_tsv(root: Path, state: dict[str, Any]) -> None:
    rows = _read_results_tsv(root)
    state["results_tsv_exists"] = bool(rows)
    if not rows:
        return
    if state.get("baseline_val_bpb") is None:
        first = rows[0]
        try:
            state["baseline_val_bpb"] = float(first.get("val_bpb") or "nan")
            state["baseline_commit"] = first.get("commit") or None
        except ValueError:
            pass
    best_val = state.get("best_val_bpb")
    best_commit = state.get("best_commit")
    for row in rows:
        try:
            val = float(row.get("val_bpb") or "nan")
        except ValueError:
            continue
        if val <= 0:
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_commit = row.get("commit") or None
    state["best_val_bpb"] = best_val
    state["best_commit"] = best_commit
    state["ingested_results_tsv_rows"] = len(rows)


def parse_metrics_text(text: str) -> RunMetrics | None:
    if not text:
        return None
    metrics: dict[str, Any] = {}
    allowed_keys = set(RunMetrics.__dataclass_fields__)
    found = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "---":
            continue
        match = re.match(r"^([a-zA-Z0-9_]+):\s+([-+0-9.eE]+)$", line)
        if not match:
            continue
        key, value = match.groups()
        if key not in allowed_keys:
            continue
        found = True
        if key in {"num_steps", "depth"}:
            metrics[key] = int(float(value))
        else:
            metrics[key] = float(value)
    if not found or "val_bpb" not in metrics:
        return None
    return RunMetrics(**metrics)


def parse_run_log(path: Path) -> tuple[RunMetrics | None, str]:
    if not path.exists():
        return None, ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_metrics_text(text), text


def is_training_command(command: str) -> bool:
    return "uv run train.py" in command


def current_branch(root: Path) -> str:
    try:
        return git(root, "branch", "--show-current")
    except Exception:
        return ""


def current_commit(root: Path) -> str | None:
    try:
        return git(root, "rev-parse", "--short", "HEAD")
    except Exception:
        return None


def current_commit_parent(root: Path) -> str | None:
    try:
        return git(root, "rev-parse", "--short", "HEAD^")
    except Exception:
        return None


def current_commit_subject(root: Path) -> str:
    try:
        return git(root, "log", "-1", "--pretty=%s")
    except Exception:
        return ""


def train_diff(root: Path) -> str:
    for args in (
        ("diff", "--unified=0", "HEAD^", "HEAD", "--", "train.py"),
        ("diff", "--unified=0", "HEAD", "--", "train.py"),
    ):
        try:
            diff = git(root, *args)
            if diff:
                return diff
        except Exception:
            continue
    return ""


def train_diff_stats(root: Path) -> tuple[int, int]:
    for args in (
        ("diff", "--numstat", "HEAD^", "HEAD", "--", "train.py"),
        ("diff", "--numstat", "HEAD", "--", "train.py"),
    ):
        try:
            output = git(root, *args)
        except Exception:
            continue
        if not output:
            continue
        line = output.splitlines()[0]
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                return 0, 0
    return 0, 0


def _classify_category(diff_text: str, additions: int, deletions: int) -> tuple[str, list[str]]:
    lowered = diff_text.lower()
    scores = {category: 0 for category in CATEGORY_KEYWORDS}
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            scores[category] += lowered.count(keyword.lower())
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    hits = [name for name, score in ranked if score > 0]
    if deletions > additions and "simplifier" not in hits:
        hits.insert(0, "simplification")
    primary = hits[0] if hits else ("simplification" if deletions > additions else "mixed")
    return primary, hits[:3]


def _size_bucket(value: float | None, thresholds: list[float], labels: list[str]) -> str:
    if value is None:
        return "unknown"
    for threshold, label in zip(thresholds, labels):
        if value < threshold:
            return label
    return labels[-1]


def _derive_niche(category: str, metrics: RunMetrics | None, plan: dict[str, Any] | None, additions: int, deletions: int) -> str:
    if plan and plan.get("niche"):
        return str(plan["niche"])
    params_bucket = _size_bucket(
        metrics.num_params_M if metrics else None,
        [25, 75, 150],
        ["tiny", "small", "medium", "large"],
    )
    vram_bucket = _size_bucket(
        metrics.peak_vram_mb / 1024 if metrics and metrics.peak_vram_mb is not None else None,
        [20, 40, 60],
        ["lean", "balanced", "heavy", "extreme"],
    )
    complexity_bucket = "shrinking" if deletions > additions else "growing"
    return f"{category}/{params_bucket}/{vram_bucket}/{complexity_bucket}"


def _complexity_cost(additions: int, deletions: int, diff_text: str) -> float:
    hunk_count = diff_text.count("@@")
    net_add = max(0, additions - deletions)
    simplicity_bonus = max(0, deletions - additions) * 0.01
    hack_penalty = 0.15 if re.search(r"\b(todo|hack|fixme)\b", diff_text, flags=re.IGNORECASE) else 0.0
    return round(0.02 * net_add + 0.05 * hunk_count + hack_penalty - simplicity_bonus, 4)


def _vram_cost(state: dict[str, Any], metrics: RunMetrics | None) -> float:
    if metrics is None or metrics.peak_vram_mb is None:
        return 0.0
    baseline = state.get("baseline_peak_vram_mb")
    if baseline is None or baseline <= 0:
        return 0.0
    ratio = (metrics.peak_vram_mb - baseline) / baseline
    return round(max(0.0, ratio) * 0.5, 4)


def _novelty_bonus(state: dict[str, Any], niche: str, emitter: str, category: str) -> float:
    archive = state.get("archive", {})
    niche_count = archive.get(niche, {}).get("count", 0)
    category_attempts = state.get("category_stats", {}).get(category, {}).get("attempts", 0)
    emitter_attempts = state.get("emitter_stats", {}).get(emitter, {}).get("attempts", 0)
    bonus = 0.35 / math.sqrt(niche_count + 1)
    bonus += 0.2 / math.sqrt(category_attempts + 1)
    bonus += 0.1 / math.sqrt(emitter_attempts + 1)
    return round(bonus, 4)


def _information_gain(state: dict[str, Any], category: str, crash_signature: str | None) -> float:
    category_attempts = state.get("category_stats", {}).get(category, {}).get("attempts", 0)
    bonus = 0.25 / math.sqrt(category_attempts + 1)
    if crash_signature:
        prior = state.get("crash_signatures", {}).get(crash_signature, {}).get("count", 0)
        if prior == 0:
            bonus += 0.2
    if state.get("stall_count", 0) >= 4:
        bonus += 0.1
    return round(bonus, 4)


def _surprise_bonus(plan: dict[str, Any] | None, improvement_vs_best: float | None, crash_signature: str | None) -> float:
    if crash_signature:
        return 0.15
    if not plan or improvement_vs_best is None:
        return 0.0
    predicted = (plan.get("predicted_direction") or "").lower()
    improved = improvement_vs_best > 0
    if improved and predicted in {"flat", "uncertain", "risky"}:
        return 0.2
    if not improved and predicted in {"improve", "likely improve"}:
        return 0.15
    if abs(improvement_vs_best) > 0.0015:
        return 0.1
    return 0.0


def _fitness_gain(improvement_vs_best: float | None, additions: int, deletions: int) -> float:
    if improvement_vs_best is None:
        return 0.0
    gain = max(0.0, improvement_vs_best) * 100.0
    if abs(improvement_vs_best) <= 0.0002 and deletions > additions:
        gain += 0.1
    return round(gain, 4)


def _recommendation(improvement_vs_best: float | None, crash_signature: str | None, complexity_cost: float, discovery_score: float, additions: int, deletions: int) -> str:
    if crash_signature:
        return "discard"
    if improvement_vs_best is not None and improvement_vs_best > 0.001:
        return "keep"
    if improvement_vs_best is not None and improvement_vs_best > 0 and complexity_cost <= 0.2:
        return "keep"
    if improvement_vs_best is not None and improvement_vs_best >= -0.0002 and deletions > additions:
        return "investigate"
    if discovery_score >= 0.5:
        return "investigate"
    return "discard"


def _derive_conjectures(state: dict[str, Any]) -> list[str]:
    category_stats = state.get("category_stats", {})
    emitter_stats = state.get("emitter_stats", {})
    conjectures: list[str] = []

    ranked_categories = sorted(
        category_stats.items(),
        key=lambda item: (item[1].get("avg_reward", 0.0), item[1].get("attempts", 0)),
        reverse=True,
    )
    if ranked_categories:
        best_name, best_stats = ranked_categories[0]
        if best_stats.get("attempts", 0) >= 2:
            conjectures.append(
                f"{best_name} edits are currently the strongest lane (avg reward {best_stats.get('avg_reward', 0.0):.2f})."
            )
    if len(ranked_categories) > 1:
        weakest_name, weakest_stats = ranked_categories[-1]
        if weakest_stats.get("attempts", 0) >= 2 and weakest_stats.get("avg_reward", 0.0) < 0:
            conjectures.append(
                f"{weakest_name} edits are underperforming; avoid repeating them without a new rationale."
            )

    simplifier = emitter_stats.get("simplifier", {})
    if simplifier.get("attempts", 0) >= 2 and simplifier.get("total_reward", 0.0) > 0.4:
        conjectures.append("Simplification is paying off; prefer cleaner follow-ups around strong motifs.")

    crashes = state.get("crash_signatures", {})
    if crashes:
        signature, data = max(crashes.items(), key=lambda item: item[1].get("count", 0))
        if data.get("count", 0) >= 2:
            conjectures.append(f"Repeated crash pattern detected: {signature}. Treat it as a no-fly zone unless debugging it directly.")

    if state.get("stall_count", 0) >= 5:
        conjectures.append("The search is plateauing; favor contrarian or recombination moves over another local tweak.")

    return conjectures[:4]


def _phase_from_state(state: dict[str, Any]) -> str:
    if state.get("experiment_count", 0) == 0:
        return "baseline"
    recent = state.get("recent_experiments", [])[-5:]
    if not recent:
        return "explore"
    recent_crashes = sum(1 for row in recent if row.get("status") == "crash")
    recent_improvements = sum(1 for row in recent if (row.get("improvement_vs_best") or 0.0) > 0)
    if recent_crashes >= 2:
        return "stabilize"
    if state.get("stall_count", 0) >= 6 and len(state.get("near_misses", [])) >= 2:
        return "recombine"
    if state.get("stall_count", 0) >= 4:
        return "explore"
    if recent_improvements >= 2:
        return "exploit"
    return "explore"


def _format_best(state: dict[str, Any]) -> str:
    best = state.get("best_val_bpb")
    commit = state.get("best_commit")
    if best is None:
        return "no best yet"
    return f"{best:.6f}" + (f" @ {commit}" if commit else "")


def _format_baseline(state: dict[str, Any]) -> str:
    baseline = state.get("baseline_val_bpb")
    commit = state.get("baseline_commit")
    if baseline is None:
        return "no baseline yet"
    return f"{baseline:.6f}" + (f" @ {commit}" if commit else "")


def _underexplored_categories(state: dict[str, Any]) -> list[str]:
    category_stats = state.get("category_stats", {})
    if not category_stats:
        return ["optimizer", "architecture", "attention", "batch", "normalization"]
    counts = {category: stats.get("attempts", 0) for category, stats in category_stats.items()}
    all_categories = list(CATEGORY_KEYWORDS) + ["mixed", "simplification"]
    return sorted(all_categories, key=lambda name: counts.get(name, 0))


def select_next_plan(state: dict[str, Any]) -> Plan:
    phase = _phase_from_state(state)
    total_attempts = max(1, state.get("experiment_count", 0))

    phase_biases: dict[str, dict[str, float]] = {
        "baseline": {"local_tuner": 0.2, "simplifier": 0.1},
        "exploit": {"local_tuner": 0.35, "optimizer_hacker": 0.25, "simplifier": 0.15},
        "explore": {"architecture_mutator": 0.35, "contrarian": 0.25, "optimizer_hacker": 0.1},
        "recombine": {"recombinator": 0.4, "contrarian": 0.2, "architecture_mutator": 0.1},
        "stabilize": {"anomaly_chaser": 0.4, "simplifier": 0.2, "local_tuner": 0.1},
    }

    best_emitter = "local_tuner"
    best_score = -1e9
    for emitter, stats in state.get("emitter_stats", {}).items():
        attempts = stats.get("attempts", 0)
        mean_reward = stats.get("total_reward", 0.0) / max(1, attempts)
        exploration = math.sqrt(2.0 * math.log(total_attempts + 2) / (attempts + 1))
        recency_penalty = 0.05 if stats.get("last_used_index", 0) == total_attempts else 0.0
        bias = phase_biases.get(phase, {}).get(emitter, 0.0)
        score = mean_reward + exploration + bias - recency_penalty
        if score > best_score:
            best_score = score
            best_emitter = emitter

    underexplored = _underexplored_categories(state)
    niche_category = underexplored[0]
    if phase == "exploit":
        niche_category = max(
            state.get("category_stats", {}).items(),
            key=lambda item: item[1].get("avg_reward", -999.0),
            default=(underexplored[0], {}),
        )[0]
    elif phase == "recombine" and state.get("near_misses"):
        categories = [row.get("category", "mixed") for row in state["near_misses"][-2:]]
        niche_category = "+".join(dict.fromkeys(categories))

    niche = f"{niche_category}/{phase}"
    hypothesis_map = {
        "local_tuner": f"Make one tight improvement inside the strongest recent {niche_category} motif. Favor low-blast-radius edits.",
        "optimizer_hacker": f"Search for a schedule or optimizer update that improves convergence inside the fixed 5-minute budget.",
        "architecture_mutator": f"Attempt one structural {niche_category} change that could unlock a new niche rather than only a local gain.",
        "simplifier": "Remove, merge, or simplify machinery around a promising motif and keep only what appears to earn its keep.",
        "contrarian": f"Try a deliberate opposite move to the recent local trend in {niche_category} and see whether the search is stuck in a false basin.",
        "recombinator": "Combine two promising near-miss ideas into one clean mutation instead of repeating either in isolation.",
        "anomaly_chaser": "Investigate the most surprising recent win, failure, or crash and turn it into a sharper hypothesis.",
    }
    predicted_map = {
        "local_tuner": "likely improve",
        "optimizer_hacker": "improve",
        "architecture_mutator": "risky",
        "simplifier": "flat",
        "contrarian": "uncertain",
        "recombinator": "improve",
        "anomaly_chaser": "uncertain",
    }
    rationale = (
        f"Phase={phase}; emitter={best_emitter}; underexplored={underexplored[:3]}; "
        f"best={_format_best(state)}; stall_count={state.get('stall_count', 0)}."
    )
    return Plan(
        plan_id=stable_id(utcnow_iso(), phase, best_emitter, niche, state.get("experiment_count", 0)),
        phase=phase,
        emitter=best_emitter,
        niche=niche,
        hypothesis=hypothesis_map[best_emitter],
        predicted_direction=predicted_map[best_emitter],
        rationale=rationale,
        created_at=utcnow_iso(),
    )


def _persist_plan(root: Path, state: dict[str, Any], plan: Plan) -> None:
    state["current_plan"] = plan.to_dict()
    save_json_atomic(plan_path(root), plan.to_dict())
    append_jsonl(
        events_path(root),
        {
            "event": "plan_selected",
            "timestamp": utcnow_iso(),
            "plan": plan.to_dict(),
        },
    )


def record_autonomy_preference(root: Path, prompt: str) -> tuple[bool | None, dict[str, Any]]:
    enabled = detect_autonomy_preference(prompt)

    with state_lock(root):
        state = load_state(root)
        if enabled is not None:
            state["autonomy_enabled"] = enabled
            state["autonomy_reason"] = prompt.strip()[:240]
            if not enabled:
                state["current_plan"] = None
                save_json_atomic(plan_path(root), {})
            append_jsonl(
                events_path(root),
                {
                    "event": "autonomy_preference",
                    "timestamp": utcnow_iso(),
                    "enabled": enabled,
                    "prompt": prompt.strip()[:500],
                },
            )
            save_state(root, state)
        return enabled, state


def _crash_signature(log_text: str) -> str | None:
    for pattern in (
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"RuntimeError: .*",
        r"AssertionError: .*",
        r"ValueError: .*",
        r"TypeError: .*",
    ):
        match = re.search(pattern, log_text)
        if match:
            return match.group(0).strip()
    tail = [line.strip() for line in log_text.splitlines() if line.strip()]
    if not tail:
        return None
    if "traceback" in tail[-1].lower():
        return "Traceback"
    return tail[-1][:140]


def _description_from_commit(root: Path) -> str:
    subject = current_commit_subject(root)
    if subject:
        return subject
    return "experiment"


def record_experiment_from_run(root: Path) -> tuple[ExperimentRecord | None, dict[str, Any]]:
    run_log = root / "run.log"
    metrics, log_text = parse_run_log(run_log)
    with state_lock(root):
        state = load_state(root)
        branch = current_branch(root)
        commit = current_commit(root)
        plan = state.get("current_plan") or load_json(plan_path(root), {}) or None
        record_token = stable_id(commit or "no-commit", run_log.stat().st_mtime_ns if run_log.exists() else "no-log")
        if record_token == state.get("last_experiment_id"):
            return None, state

        additions, deletions = train_diff_stats(root)
        diff_text = train_diff(root)
        category, categories = _classify_category(diff_text, additions, deletions)
        niche = _derive_niche(category, metrics, plan, additions, deletions)
        description = _description_from_commit(root)
        phase = str((plan or {}).get("phase") or _phase_from_state(state))
        emitter = str((plan or {}).get("emitter") or "local_tuner")
        crash_signature = None
        status = "ok"
        if metrics is None:
            status = "crash"
            crash_signature = _crash_signature(log_text)

        best_before = state.get("best_val_bpb")
        baseline = state.get("baseline_val_bpb")
        improvement_vs_best = None if metrics is None or best_before is None else round(best_before - metrics.val_bpb, 6)
        improvement_vs_baseline = None if metrics is None or baseline is None else round(baseline - metrics.val_bpb, 6)

        complexity_cost = _complexity_cost(additions, deletions, diff_text)
        vram_cost = _vram_cost(state, metrics)
        novelty_bonus = _novelty_bonus(state, niche, emitter, category)
        information_gain = _information_gain(state, category, crash_signature)
        surprise_bonus = _surprise_bonus(plan, improvement_vs_best, crash_signature)
        fitness_gain = _fitness_gain(improvement_vs_best, additions, deletions)
        discovery_score = round(
            fitness_gain + novelty_bonus + information_gain + surprise_bonus - complexity_cost - vram_cost,
            4,
        )
        recommendation = _recommendation(
            improvement_vs_best,
            crash_signature,
            complexity_cost,
            discovery_score,
            additions,
            deletions,
        )

        experiment = ExperimentRecord(
            experiment_id=record_token,
            timestamp=utcnow_iso(),
            branch=branch,
            commit=commit,
            commit_parent=current_commit_parent(root),
            description=description,
            status=status,
            category=category,
            niche=niche,
            phase=phase,
            emitter=emitter,
            recommendation=recommendation,
            diff_summary=_summarize_diff(diff_text, categories, additions, deletions),
            metrics=metrics.to_dict() if metrics else {},
            fitness_gain=fitness_gain,
            novelty_bonus=novelty_bonus,
            information_gain=information_gain,
            surprise_bonus=surprise_bonus,
            complexity_cost=complexity_cost,
            vram_cost=vram_cost,
            discovery_score=discovery_score,
            improvement_vs_best=improvement_vs_best,
            improvement_vs_baseline=improvement_vs_baseline,
            crash_signature=crash_signature,
            conjecture_updates=[],
        )

        _update_state_from_experiment(state, experiment)
        state["last_experiment_id"] = record_token
        state["current_plan"] = None
        state["conjectures"] = _derive_conjectures(state)
        experiment.conjecture_updates = state["conjectures"][:3]

        append_jsonl(
            events_path(root),
            {
                "event": "experiment_recorded",
                "timestamp": experiment.timestamp,
                "experiment": experiment.to_dict(),
            },
        )
        save_state(root, state)
        write_text(run_review_path(root), build_training_review(experiment, state))
        return experiment, state


def _summarize_diff(diff_text: str, categories: list[str], additions: int, deletions: int) -> str:
    summary_parts: list[str] = []
    if categories:
        summary_parts.append("categories=" + ",".join(categories))
    summary_parts.append(f"+{additions}/-{deletions}")
    interesting_lines = []
    for line in diff_text.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            stripped = line[1:].strip()
            if stripped:
                interesting_lines.append(stripped)
        if len(interesting_lines) >= 2:
            break
    if interesting_lines:
        summary_parts.append(" | ".join(interesting_lines))
    return "; ".join(summary_parts)[:280]


def _update_state_from_experiment(state: dict[str, Any], experiment: ExperimentRecord) -> None:
    state["experiment_count"] = int(state.get("experiment_count", 0)) + 1

    stats = state["emitter_stats"].setdefault(
        experiment.emitter,
        {"attempts": 0, "improvements": 0, "crashes": 0, "total_reward": 0.0, "last_used_index": 0},
    )
    stats["attempts"] += 1
    stats["total_reward"] = round(stats.get("total_reward", 0.0) + experiment.discovery_score, 4)
    stats["last_used_index"] = state["experiment_count"]
    if experiment.status == "crash":
        stats["crashes"] += 1
    elif (experiment.improvement_vs_best or 0.0) > 0 or experiment.recommendation == "keep":
        stats["improvements"] += 1

    category_stats = state["category_stats"].setdefault(
        experiment.category,
        {"attempts": 0, "improvements": 0, "total_reward": 0.0, "avg_reward": 0.0, "best_delta": None},
    )
    category_stats["attempts"] += 1
    category_stats["total_reward"] = round(category_stats.get("total_reward", 0.0) + experiment.discovery_score, 4)
    category_stats["avg_reward"] = round(
        category_stats["total_reward"] / max(1, category_stats["attempts"]),
        4,
    )
    if (experiment.improvement_vs_best or 0.0) > 0:
        category_stats["improvements"] += 1
        current_best_delta = category_stats.get("best_delta")
        if current_best_delta is None or experiment.improvement_vs_best > current_best_delta:
            category_stats["best_delta"] = experiment.improvement_vs_best

    if experiment.status == "crash" and experiment.crash_signature:
        crash = state["crash_signatures"].setdefault(
            experiment.crash_signature,
            {"count": 0, "last_seen": None, "category": experiment.category, "example_commit": experiment.commit},
        )
        crash["count"] += 1
        crash["last_seen"] = experiment.timestamp

    if experiment.metrics:
        val = experiment.metrics.get("val_bpb")
        peak = experiment.metrics.get("peak_vram_mb")
        if state.get("baseline_val_bpb") is None and val is not None:
            state["baseline_val_bpb"] = val
            state["baseline_commit"] = experiment.commit
            state["baseline_peak_vram_mb"] = peak
        if val is not None and (state.get("best_val_bpb") is None or val < state["best_val_bpb"]):
            state["best_val_bpb"] = val
            state["best_commit"] = experiment.commit
            state["best_peak_vram_mb"] = peak

    archive = state["archive"]
    niche_entry = archive.get(experiment.niche, {"count": 0})
    niche_entry["count"] = int(niche_entry.get("count", 0)) + 1
    elite_val = niche_entry.get("best_val_bpb")
    current_val = experiment.metrics.get("val_bpb") if experiment.metrics else None
    if current_val is not None and (elite_val is None or current_val < elite_val):
        niche_entry.update(
            {
                "best_val_bpb": current_val,
                "commit": experiment.commit,
                "emitter": experiment.emitter,
                "category": experiment.category,
                "description": experiment.description,
                "discovery_score": experiment.discovery_score,
            }
        )
    archive[experiment.niche] = niche_entry

    if experiment.status != "crash" and (experiment.improvement_vs_best or 0.0) <= 0 and experiment.metrics:
        best = state.get("best_val_bpb")
        if best is not None and experiment.metrics["val_bpb"] <= best + 0.0015:
            state["near_misses"].append(
                {
                    "commit": experiment.commit,
                    "category": experiment.category,
                    "niche": experiment.niche,
                    "val_bpb": experiment.metrics["val_bpb"],
                    "description": experiment.description,
                }
            )
            state["near_misses"] = state["near_misses"][-12:]

    if experiment.status == "crash":
        state["stall_count"] = int(state.get("stall_count", 0)) + 1
    elif (experiment.improvement_vs_best or 0.0) > 0 or experiment.recommendation == "keep":
        state["stall_count"] = 0
    else:
        state["stall_count"] = int(state.get("stall_count", 0)) + 1

    recent_payload = experiment.to_dict()
    state["recent_experiments"].append(recent_payload)
    state["recent_experiments"] = state["recent_experiments"][-20:]
    state["phase"] = _phase_from_state(state)


def build_training_review(experiment: ExperimentRecord, state: dict[str, Any]) -> str:
    lines = ["Discovery review:"]
    if experiment.status == "crash":
        lines.append(f"- Result: crash ({experiment.crash_signature or 'unknown failure'}).")
    else:
        metrics = experiment.metrics
        lines.append(
            "- Metrics: "
            f"val_bpb={metrics.get('val_bpb', 0.0):.6f}, "
            f"peak_vram_gb={(metrics.get('peak_vram_mb') or 0.0) / 1024:.1f}, "
            f"mfu={metrics.get('mfu_percent', 0.0):.2f}, "
            f"params_M={metrics.get('num_params_M', 0.0):.1f}, "
            f"depth={int(metrics.get('depth', 0) or 0)}."
        )
        if experiment.improvement_vs_best is not None:
            lines.append(f"- Delta vs best before run: {experiment.improvement_vs_best:+.6f} bpb.")
        if experiment.improvement_vs_baseline is not None:
            lines.append(f"- Delta vs baseline: {experiment.improvement_vs_baseline:+.6f} bpb.")
    lines.append(f"- Classification: emitter={experiment.emitter}, phase={experiment.phase}, category={experiment.category}, niche={experiment.niche}.")
    lines.append(f"- Diff summary: {experiment.diff_summary}")
    lines.append(
        "- Discovery score: "
        f"{experiment.discovery_score:.3f} = fitness {experiment.fitness_gain:.3f} + novelty {experiment.novelty_bonus:.3f} + "
        f"info {experiment.information_gain:.3f} + surprise {experiment.surprise_bonus:.3f} - complexity {experiment.complexity_cost:.3f} - vram {experiment.vram_cost:.3f}."
    )
    lines.append(f"- Recommendation: {experiment.recommendation.upper()}.")
    if experiment.conjecture_updates:
        lines.append("- Updated conjectures:")
        for item in experiment.conjecture_updates:
            lines.append(f"  - {item}")
    lines.append(f"- Current best: {_format_best(state)}.")
    return "\n".join(lines)


def build_session_context(root: Path, state: dict[str, Any]) -> str:
    branch = current_branch(root) or "(no branch)"
    lines = [
        "Autoresearch discovery state:",
        f"- Branch: {branch}",
        f"- Autonomy enabled: {state.get('autonomy_enabled', False)}",
        f"- Baseline: {_format_baseline(state)}",
        f"- Best: {_format_best(state)}",
        f"- Experiments recorded by hooks: {state.get('experiment_count', 0)}",
        f"- Search phase: {_phase_from_state(state)}",
    ]

    plan = state.get("current_plan")
    if plan:
        lines.append(
            f"- Pending plan: emitter={plan.get('emitter')}, niche={plan.get('niche')}, predicted={plan.get('predicted_direction')}"
        )

    if state.get("conjectures"):
        lines.append("- Current conjectures:")
        for item in state["conjectures"][:4]:
            lines.append(f"  - {item}")

    archive = state.get("archive", {})
    if archive:
        lines.append("- Archive elites:")
        elite_items = sorted(
            archive.items(),
            key=lambda item: item[1].get("best_val_bpb", float("inf")),
        )[:4]
        for niche, data in elite_items:
            lines.append(
                f"  - {niche}: {data.get('best_val_bpb', 'n/a')} @ {data.get('commit', 'n/a')} ({data.get('description', '')})"
            )

    crashes = state.get("crash_signatures", {})
    if crashes:
        lines.append("- No-fly patterns:")
        for signature, data in sorted(crashes.items(), key=lambda item: item[1].get("count", 0), reverse=True)[:3]:
            lines.append(f"  - {signature} (count={data.get('count', 0)})")

    lines.append("- Rich state is stored in results/discovery/. Treat it as machine memory; do not commit it.")
    return "\n".join(lines)


def _coerce_plan(plan_payload: dict[str, Any] | None) -> Plan | None:
    if not plan_payload:
        return None
    try:
        return Plan(**plan_payload)
    except TypeError:
        return None


def build_prompt_context(root: Path, state: dict[str, Any], user_prompt: str) -> str:
    branch = current_branch(root)
    active = state.get("autonomy_enabled", False)
    if active:
        mode = "autonomous loop is ON"
    else:
        mode = "autonomous loop is OFF until the user explicitly starts or resumes it"
    return (
        f"Repo-local discovery hooks are active. {mode}. Current branch: {branch or '(none)'}. "
        f"Best={_format_best(state)}. Phase={_phase_from_state(state)}. "
        "Use train.py as the only experiment target, trust the hook-generated run review, and keep results.tsv as the public ledger."
    )


def build_next_prompt(
    root: Path,
    *,
    stop_hook_active: bool = False,
    last_assistant_message: str | None = None,
    persist: bool = True,
) -> str | None:
    with state_lock(root):
        state = load_state(root)
        if not state.get("autonomy_enabled"):
            return None

        branch = current_branch(root)
        results_path = root / "results.tsv"
        cache_path = Path.home() / ".cache" / "autoresearch"

        if not branch or not branch.startswith("autoresearch/"):
            prompt = (
                "Continue the autoresearch setup autonomously. Create or switch to a fresh "
                "autoresearch/<tag> branch, verify ~/.cache/autoresearch/ exists, initialize results.tsv with the header row if needed, "
                "and then run the baseline before making any train.py edits. Do not ask for confirmation."
            )
            if persist:
                append_jsonl(events_path(root), {"event": "stop_continue", "timestamp": utcnow_iso(), "reason": prompt})
            return prompt

        if not cache_path.exists():
            reminder = "run `uv run prepare.py`"
            if stop_hook_active and last_assistant_message and reminder in last_assistant_message:
                return None
            prompt = (
                "The data cache appears missing. Tell the human to run `uv run prepare.py`, then as soon as the cache exists continue the loop automatically."
            )
            if persist:
                append_jsonl(events_path(root), {"event": "stop_continue", "timestamp": utcnow_iso(), "reason": prompt})
            return prompt

        if not results_path.exists() or not state.get("results_tsv_exists"):
            prompt = (
                "Continue the setup on the current autoresearch branch. Create results.tsv with the standard header, run the baseline `uv run train.py > run.log 2>&1`, "
                "log the baseline to results.tsv, and then continue autonomously."
            )
            if persist:
                append_jsonl(events_path(root), {"event": "stop_continue", "timestamp": utcnow_iso(), "reason": prompt})
            return prompt

        plan = _coerce_plan(state.get("current_plan"))
        if plan is None:
            plan = select_next_plan(state)
            if persist:
                _persist_plan(root, state, plan)
                save_state(root, state)

        no_fly = []
        for signature, data in sorted(state.get("crash_signatures", {}).items(), key=lambda item: item[1].get("count", 0), reverse=True)[:2]:
            no_fly.append(signature)
        no_fly_text = "; avoid repeating: " + " | ".join(no_fly) if no_fly else ""

        prompt = (
            "Continue the autoresearch loop autonomously. "
            f"Phase={plan.phase}. Emitter={plan.emitter}. Target niche={plan.niche}. "
            f"Hypothesis: {plan.hypothesis} "
            f"Predicted effect: {plan.predicted_direction}. "
            f"Current best: {_format_best(state)}. "
            "Make exactly one coherent mutation to train.py aligned to this plan, commit it with an `exp:` message, run `uv run train.py > run.log 2>&1`, "
            "use the hook-generated discovery review to decide keep/discard/investigate, update results.tsv, and keep going without asking for confirmation."
            f"{no_fly_text}"
        )
        if persist:
            append_jsonl(events_path(root), {"event": "stop_continue", "timestamp": utcnow_iso(), "reason": prompt, "plan_id": plan.plan_id})
        return prompt


def check_command_policy(root: Path, command: str) -> str | None:
    command_lower = command.lower()

    disallowed_substrings = {
        " tee ": "Use `> run.log 2>&1` instead of tee so training output does not flood context.",
        "| tee": "Use `> run.log 2>&1` instead of tee so training output does not flood context.",
        "pip install": "Do not install new dependencies in autoresearch.",
        "uv add": "Do not add dependencies in autoresearch.",
        "poetry add": "Do not add dependencies in autoresearch.",
        "git push": "Experimental branches are local only; do not push from the autonomous loop.",
        "gh pr": "Do not open pull requests from the autonomous loop.",
    }
    for needle, reason in disallowed_substrings.items():
        if needle in command_lower:
            return reason

    if is_training_command(command):
        has_run_log = re.search(r">\s*run\.log\b", command_lower) is not None
        has_stderr_redirect = "2>&1" in command_lower
        if not (has_run_log and has_stderr_redirect):
            return "Train runs must be launched as `uv run train.py > run.log 2>&1` so the hook layer can parse them cleanly."

    write_verbs = ["sed -i", "> prepare.py", ">> prepare.py", "mv ", "cp ", "perl -pi", "python - <<"]
    if "prepare.py" in command_lower and any(verb in command_lower for verb in write_verbs):
        return "prepare.py is read-only in autoresearch."

    if ".codex/hooks" in command_lower or "research/" in command_lower:
        if any(verb in command_lower for verb in ["sed -i", "perl -pi", ">", "mv ", "cp ", "rm "]):
            return "Do not mutate the hook/control-plane code during the autonomous experiment loop."

    if "git reset --hard" in command_lower and not current_branch(root).startswith("autoresearch/"):
        return "Hard resets are only allowed on dedicated autoresearch/* branches."

    if re.search(r"rm\s+-rf\s+(/|~)", command_lower):
        return "Refusing destructive rm -rf command."

    return None


def load_last_review(root: Path) -> str | None:
    path = run_review_path(root)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def archive_snapshot(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    archive = state.get("archive", {})
    return sorted(archive.items(), key=lambda item: item[1].get("best_val_bpb", float("inf")))
