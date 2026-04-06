# autoresearch

This is an experiment to have the LLM do its own research.

The repo still follows the original autoresearch contract:

- `train.py` is the only experiment target.
- `prepare.py` is read-only.
- The metric is still `val_bpb`.
- Each experiment still gets a fixed 5-minute training budget.
- `results.tsv` is still the public ledger.

What changes in this version is the **research organization** around the loop. Codex now has repo-local hooks that maintain search memory, select the next search role, parse runs, preserve stepping stones, and continue the loop automatically. Treat the hook-produced guidance as part of the research protocol.

## GPU execution

Anything in this repo that requires a GPU should be routed through the `$modal-gpu` skill instead of assuming local CUDA access. That includes experiments, CUDA sanity checks, GPU debugging, training smoke tests, and benchmarks.

When GPU work is needed, follow the `$modal-gpu` workflow:

1. Verify Modal install and auth.
2. Ensure the repo-local runner contract is available (`scripts/modal_gpu.py`).
3. Run a cheap CUDA sanity check first.
4. Launch the target command through the Modal GPU runner with an explicit GPU type and timeout.
5. Report evidence and clean up any remote sandboxes.

The Modal runner is an allowed infrastructure exception to the "`train.py` only" rule, but treat it narrowly:

- You may modify `scripts/modal_gpu.py` only to make remote GPU execution, cache visibility, or sandbox transport work correctly.
- Runner edits are **infrastructure**, not experiments. Commit them separately with a non-`exp:` message.
- Once the runner is healthy, freeze it. Do not keep tuning the launcher during normal search.
- Avoid unnecessary repo-wide churn between runs. Changes outside `train.py` can trigger avoidable Modal image rebuilds, cache misses, and slow startup.
- If infrastructure changes are required after the official baseline, make the fix separately and then re-run the untouched baseline before trusting new comparisons.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag.** Propose a tag based on today's date (for example `apr5`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch.** `git checkout -b autoresearch/<tag>` from current `master`.
3. **Read the in-scope files.** Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify.
4. **Verify data exists.** Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Stabilize the GPU path before the official baseline.** If `scripts/modal_gpu.py`, remote mounts, or cache wiring need work, do it now. Do not treat infrastructure debugging or repeated baseline re-stamps as normal search progress.
6. **Initialize `results.tsv`.** Create `results.tsv` with just the header row. The baseline will be recorded after the first real run on stable infrastructure.
7. **Confirm and go.** Once setup looks good, start the experiment loop.

## Experimentation

Each experiment runs on a single GPU. Use `$modal-gpu` for experiment execution and any other GPU-bound task. The training script still runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup and compilation). Once the repo-local Modal runner exists, launch experiments like this so `run.log` stays local for the hooks:

```bash
python scripts/modal_gpu.py --gpu H100 --timeout 10 -- uv run train.py > run.log 2>&1
```

The wrapper is only transport. The inner experiment command must remain exactly `uv run train.py`.
The 10-minute Modal timeout is an outer wall-clock guardrail for remote startup / sync / eval overhead. The benchmark itself is still the 5-minute training budget enforced inside `train.py`.

### What you CAN do

- Modify `train.py`.
- Change architecture, optimizer, hyperparameters, training loop details, batch size, and model size.
- Use the hook-provided research guidance to choose the next experimental direction.
- Make narrowly-scoped `scripts/modal_gpu.py` fixes only when remote GPU execution is genuinely blocked.

### What you CANNOT do

- Modify `prepare.py`.
- Install new packages or add dependencies.
- Modify the evaluation harness. `evaluate_bpb` in `prepare.py` is the ground-truth metric.
- Modify `scripts/modal_gpu.py` as part of an experiment or to change training / evaluation semantics.
- Touch unrelated tracked files during normal search. A real experiment diff should normally be `train.py` only.
- Use `tee` or let training output flood the context window. Always redirect to `run.log`.
- Push, publish, or otherwise sync experimental branches. This is a local autonomous loop.

The goal is still simple: get the lowest `val_bpb` while respecting the fixed time budget.

### Secondary criteria

- **VRAM** is a soft constraint. Some increase is fine for meaningful gains, but avoid dramatic blow-ups.
- **Simplicity** matters. A tiny improvement that adds ugly complexity is usually not worth it. An equal result with simpler code is a real win.
- **Discovery value** matters. Not every useful run becomes the branch tip. A discarded run can still be valuable if it opens a new niche, falsifies a conjecture, or creates a stepping stone the hook system can remember.

### The first run

Your first run on a fresh branch should always be the baseline: run the current `train.py` without changing it.

## Hook-aware discovery protocol

The hooks maintain untracked machine state in `results/discovery/`. You do not edit those files during experiments, but you should use the guidance they provide.

### Search roles

The hook layer selects one of these emitter styles for the next experiment:

- `local_tuner` — exploit a nearby promising configuration.
- `optimizer_hacker` — change schedule, optimizer, or update mechanics.
- `architecture_mutator` — attempt one structural model change.
- `simplifier` — delete or compress complexity around a winning idea.
- `contrarian` — try the opposite of the recent local trend.
- `recombinator` — combine two promising near-misses.
- `anomaly_chaser` — investigate a surprising win, loss, or crash.

Respect the active emitter and target niche unless you have direct local evidence that the hook-picked direction is impossible.

### One mutation per experiment

Each experiment should make **one coherent idea-level change**. Small supporting edits are fine, but do not bundle several independent ideas into one run.

### Pre-register intent in the commit message

Before running training, commit the candidate with a short message in this style:

```bash
git commit -am "exp: <short description> [emitter=<name>]"
```

The hooks can infer metadata if needed, but a structured message makes the archive cleaner.

The experiment identity rules are strict:

- One launched run must map to exactly one `exp:` commit.
- One launched run must map to exactly one `results.tsv` row for that same commit hash.
- If you revise a candidate **before** launching, amend the existing `exp:` commit instead of stacking another `exp:` commit on top.
- Do not leave abandoned or superseded `exp:` commits in branch history.
- Non-experiment commits such as restore commits or infrastructure fixes must not be logged as experiment rows.

### Use the hook review as the primary post-run analysis

After `python scripts/modal_gpu.py -- uv run train.py > run.log 2>&1`, the post-tool hook will parse `run.log` and return a structured discovery review. That review includes:

- parsed metrics
- category / niche classification
- novelty / information / surprise signals
- updated conjectures
- a keep / discard / investigate recommendation

Use that as your primary feedback. Do not waste context manually re-parsing logs when the hook has already done it.

### Preserve stepping stones without polluting the branch

The hook layer tracks archive niches and useful failed ideas in `results/discovery/`. That means a run can be a valuable stepping stone even if it does **not** become the new branch tip.

Keep branch history clean:

- **Keep** the commit if it improves `val_bpb`, or if it is materially simpler with no real regression.
- **Discard** the commit if it clearly regresses and does not buy compelling novelty or insight.
- **Investigate / replicate** marginal wins or surprising results before you commit to a new long local search around them.
- If a run lands within roughly `0.001` bpb of the current best, or trades a tiny regression for a large VRAM / simplicity gain, spend the next run replicating, sharpening, or isolating it before abandoning the lane.
- Restore back to the best validated tip before the next candidate. Do not let the branch drift forward from a known loser.
- Keep diffs narrow. Do not bundle broad cleanup, multiple hypotheses, or unrelated code motion into the middle of the search loop.

## Output format

Once the script finishes it prints a summary like this:

```text
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

The script is configured to stop after 5 minutes, so the exact numbers depend on this machine.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, not comma-separated).

The TSV has a header row and 5 columns:

```text
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `val_bpb` achieved (`0.000000` for crashes)
3. peak memory in GB, rounded to `.1f` (`0.0` for crashes)
4. status: `keep`, `discard`, or `crash`
5. short text description of what the experiment tried

Example:

```text
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

`results.tsv` is the compatibility ledger. The richer machine-readable memory is kept separately in `results/discovery/`.

Logging discipline matters:

- Update `results.tsv` immediately after each completed run and before making the next restore or candidate commit.
- Every completed run should also appear in the discovery event log for the same commit.
- If a run never actually launched, do not log it and do not leave its `exp:` commit as if it were a completed experiment.

## The experiment loop

The experiment runs on a dedicated branch such as `autoresearch/apr5`.

LOOP FOREVER:

1. Look at the git state: current branch and current commit.
2. Read the latest hook context: current phase, active emitter, target niche, and anti-patterns.
3. Tune `train.py` with **one** coherent experimental idea. Before committing, make sure the tracked experiment diff is limited to `train.py` unless you are handling a separate infrastructure fix.
4. Commit the candidate.
5. Run the experiment through `$modal-gpu`: `python scripts/modal_gpu.py --gpu H100 --timeout 10 -- uv run train.py > run.log 2>&1`.
6. Read the hook-generated discovery review.
7. If the run crashed, inspect the tail of `run.log` only as needed to diagnose the error.
8. Record the result in `results.tsv` for the exact commit that was run.
9. If the result genuinely advances the branch, keep it.
10. If the result is equal or worse and does not justify branch advancement, restore back to the best validated commit you started from.
11. Let the stop hook continue the session. Do **not** ask the human whether to keep going.

## Timeout and crash policy

- Each experiment should take about 5 minutes total plus startup / eval overhead.
- If a run exceeds 10 minutes, kill it and treat it as a failure.
- Keep the Modal sandbox timeout aligned with this outer cap. Do not pay for long-idle remote sandboxes while waiting on infrastructure stalls.
- If a crash is a trivial bug in the experimental edit, fix it and rerun.
- If the idea itself looks broken, log `crash`, revert, and move on.
- If a crash pattern repeats, the hook archive will remember it; do not keep retrying the same dead end.
- If the failure is clearly infrastructure-related (Modal auth, image build, remote cache visibility, sandbox transport), treat it as an infrastructure blocker, not as a train.py research result. Fix the runner separately or ask the human for the missing prerequisite instead of burning experiment slots on repeated startup failures.

## Autonomy policy

NEVER STOP once the experiment loop has begun unless the human explicitly tells you to pause or stop.

The human might be asleep or away from the keyboard and expects you to continue indefinitely until manually interrupted. The repo-local `Stop` hook is there to reinforce this exact behavior. Do not ask "should I keep going?". Continue working.

If you run out of easy ideas, the correct response is not to stop. Re-read the in-scope files, inspect the discovery archive, combine near-misses, simplify successful motifs, or make a more radical architectural move.
