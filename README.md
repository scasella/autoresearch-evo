# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies.

The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026.*

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is still the same: you are not touching the Python files the way you normally would as a researcher. Instead, you are programming the research organization around the code.

This update keeps the original repo shape and constraints, but adds a **Codex-native discovery control plane** around the loop:

- Codex reads `program.md` automatically through repo-local `.codex/config.toml`.
- Repo-local hooks provide persistent research memory, search-phase control, Bash safety rails, automated run parsing, and a `Stop`-hook continuation loop.
- The original `results.tsv` stays intact for compatibility, while richer machine-readable state lives in `results/discovery/`.
- `train.py` remains the only experiment target. `prepare.py`, the eval harness, and dependency set remain fixed.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits during experiments. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, optimizer, batch size, model size, etc.
- **`program.md`** — baseline instructions for the agent. In this update, Codex reads it automatically as project guidance.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup and compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte): lower is better, and vocab-size-independent, so architectural changes are fairly compared.

## What the hook layer adds

The hook layer makes autoresearch behave less like a plain hill-climber and more like a tiny discovery engine:

- **Never-stop continuation.** A repo-local `Stop` hook keeps the loop going on `autoresearch/*` branches instead of letting Codex end the session.
- **Emitter-based search.** The next experiment is chosen from a small portfolio of search roles (`local_tuner`, `optimizer_hacker`, `architecture_mutator`, `simplifier`, `contrarian`, `recombinator`, `anomaly_chaser`) instead of one monolithic style of mutation.
- **Quality-diversity archive.** Results are grouped into niches so the system preserves stepping stones instead of tracking only one incumbent best.
- **Conjecture memory.** The hooks distill simple, machine-generated theories from outcomes and feed them back at session start.
- **Automated run review.** After `uv run train.py > run.log 2>&1`, a `PostToolUse` hook parses the run, scores novelty / fitness / information gain, and returns a structured review to Codex.
- **Safety rails.** A `PreToolUse` hook blocks destructive or off-policy Bash commands such as dependency installation, `tee`, `git push`, and writes against `prepare.py`.

The result is still recognizably autoresearch, but the search process now has memory, structure, and a stronger bias toward discovery.

## Quick start

Requirements: a single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/), and a trusted Codex project so `.codex/config.toml` and `.codex/hooks.json` are loaded.

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Optional sanity check: manually run one training experiment (~5 min)
uv run train.py
```

Once the above works, open the repo in Codex and start with something like:

```text
Read program.md, create a fresh autoresearch/<tag> branch, run the baseline, and continue autonomously.
```

The hook layer will activate automatically if the repo is trusted and project config is loaded.

## Human-visible state

Two state surfaces now coexist on purpose:

- **`results.tsv`** — the compact, human-readable experiment ledger used by the original repo.
- **`results/discovery/`** — untracked machine state written by the hooks: event log, current search phase, emitter stats, niche archive, crash signatures, and selected next plan.

You can inspect the richer state with:

```bash
python -m research.cli summary
python -m research.cli archive
python -m research.cli next
```

## Project structure

```text
prepare.py             # constants, data prep + runtime utilities (do not modify)
train.py               # model, optimizer, training loop (agent modifies this)
program.md             # research instructions (loaded automatically by Codex)
.codex/config.toml     # enables hooks and treats program.md as project guidance
.codex/hooks.json      # lifecycle hook wiring
.codex/hooks/*.py      # repo-local hook entrypoints
research/*.py          # shared discovery-state logic for hooks + CLI
pyproject.toml         # dependencies
```

## Design choices

- **Single file to modify during experiments.** The agent still only changes `train.py`. That keeps diffs reviewable and preserves the charm of the original project.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of platform. This makes experiments comparable within one machine.
- **Repo-local control plane.** Search strategy, memory, and autonomy live outside `train.py`, so the training code stays simple and the research organization evolves independently.
- **Compatibility first.** The new machinery augments `results.tsv` instead of replacing it.
- **No new dependencies required.** The hook/control-plane code uses only the Python standard library.

## Platform support

This code still requires a single NVIDIA GPU. In principle it is possible to support CPU, MPS, and other platforms, but that would bloat the core repo. Forks remain the right place for platform-specific adaptations.

If you are running on smaller hardware, the original recommendations still apply:

1. Use a lower-entropy dataset such as [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean).
2. Consider lowering `vocab_size`.
3. Reduce `MAX_SEQ_LEN` and `EVAL_TOKENS` in `prepare.py` in your own fork if you are intentionally targeting smaller machines.
4. Lower `DEPTH` and other scale knobs in `train.py`.
5. Try a simpler `WINDOW_PATTERN` such as `"L"`.
6. Reduce `TOTAL_BATCH_SIZE` while keeping it power-of-two.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
