# autoresearch-evo

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that keeps the same tiny 5-minute `train.py` benchmark, but gives the research loop memory, novelty-search-inspired exploration, and autonomous review so it can search more broadly than plain hill-climbing.

Instead of treating each run as an isolated local tweak, this fork tries to make the loop behave more like a lightweight research system:

- it explores through multiple mutation styles instead of only one local-tuning behavior
- it remembers near-misses, crashes, and stepping stones instead of forgetting them immediately
- it reviews each run and feeds that evidence back into the next decision
- it can keep operating autonomously on a dedicated research branch
- it still preserves the same simple public benchmark: one file, one metric, one 5-minute run

At this release cut, that loop improved the baseline from `0.997426` to `0.985596` `val_bpb` under the same benchmark contract.

This is a heuristic discovery control plane with novelty-search-inspired exploration, not a formal evolutionary algorithm framework. The goal is still pragmatic: help one small autoresearch loop search more intelligently without bloating the repo.

The public branch is curated to keep only the intentional fork surface: the control-plane, the runner, tests, the best validated `train.py` found in our run history, and release charts summarizing the search.

## Upstream Base

The clean base for this fork is `karpathy/autoresearch`, which provides:

- `prepare.py` for one-time data prep and evaluation utilities
- `train.py` as the only experiment target
- a fixed 5-minute training budget
- `val_bpb` as the optimization metric

This repo keeps that contract. The benchmark is still the same 5-minute `train.py` run. Any backend-specific timeout is only an outer guardrail around startup, sync, and evaluation.

## What This Fork Adds

The main value-add is a repo-local discovery layer that changes how the loop searches:

- **Broader search than hill-climbing.** The loop chooses among several mutation styles instead of repeating one kind of local tweak.
- **Persistent research memory.** Near-misses, crashes, archive elites, and selected conjectures carry across turns.
- **Structured run review.** Every completed run is parsed and scored for fitness, novelty, information gain, and surprise before the next step is chosen.
- **Autonomous continuation.** Repo-local hooks can keep the loop moving on `autoresearch/*` branches without turning the repo into a heavyweight framework.
- **Optional remote backend.** A generic repo runner supports a local path by default and an optional remote GPU backend when local hardware is not the target environment.

The emitter portfolio is:

- `local_tuner`
- `optimizer_hacker`
- `architecture_mutator`
- `simplifier`
- `contrarian`
- `recombinator`
- `anomaly_chaser`

Under the hood, the core implementation lives in `.codex/`, `research/`, and the repo-local runner scripts, but the intended public mental model stays small: `prepare.py`, `train.py`, `program.md`, and a loop that now searches with memory.

## Best Validated Configuration Shipped Here

This public branch ships the best validated `train.py` configuration we found during the since-inception run history used for the charts below.

Best validated run:

- commit: `0261b2c`
- `val_bpb`: `0.985596`
- baseline: `80b9a32` at `0.997426`
- absolute improvement: `0.011830`
- relative improvement: about `1.186%`

The most important differences versus the discovery-path base are:

- shared norm changed from pure RMSNorm to LayerNorm with dtype preservation
- q/k normalization changed to a mixed RMSNorm/LayerNorm path
- total batch reduced to `2**17`
- Muon weight decay retuned to `0.109`
- warmup restored to `2%`
- device batch reduced to `64`

This is not presented as a final optimum, only as the best validated point discovered in this release cut.

## Since-Inception Research Progress

Overall run history, including keeps, discards, crashes, and the best-so-far frontier:

![Since-Inception val_bpb Progress](docs/figures/val_bpb_since_inception.svg)

Frontier-only view to make the cumulative best improvements easier to inspect:

![Since-Inception Frontier](docs/figures/val_bpb_frontier.svg)

Release-cut summary:

- total runs: `102`
- keeps: `12`
- discards: `83`
- crashes: `7`
- best run: `0261b2c` at `0.985596`

## Quick Start

Requirements:

- Python 3.10+
- `uv`
- a single NVIDIA GPU locally for the upstream-style path

Setup:

```bash
uv sync
uv run prepare.py
```

Local benchmark run:

```bash
uv run train.py
```

Generic repo runner on the local backend:

```bash
python scripts/run_experiment.py --backend local -- uv run train.py > run.log 2>&1
```

Optional remote GPU run through the Modal backend:

```bash
python scripts/run_experiment.py --backend modal --gpu H100 --timeout 10 -- uv run train.py > run.log 2>&1
```

The inner command must remain exactly `uv run train.py`. The runner only handles transport, cache reuse, and backend lifecycle.

For Modal-specific operational details and environment variables, see [docs/infra/modal.md](docs/infra/modal.md).

## Repo Layout

```text
prepare.py             # fixed data prep + evaluation utilities
train.py               # single experiment target, shipped at the best validated config
program.md             # agent-facing research protocol
.codex/                # repo-local Codex config + hooks
research/              # discovery-state logic
scripts/run_experiment.py      # generic experiment runner
scripts/backends/modal_backend.py # optional Modal backend
scripts/modal_gpu.py           # compatibility shim for the Modal backend
tests/                 # control-plane and runner tests
docs/figures/          # release charts
```

## Testing

```bash
python3 -m py_compile research/core.py scripts/run_experiment.py scripts/modal_gpu.py scripts/backends/modal_backend.py train.py
python3 -m unittest discover -s tests -v
```

## License

MIT
