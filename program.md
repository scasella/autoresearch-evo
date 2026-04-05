# autoresearch

This is an experiment to have the LLM do its own research.

The repo still follows the original autoresearch contract:

- `train.py` is the only experiment target.
- `prepare.py` is read-only.
- The metric is still `val_bpb`.
- Each experiment still gets a fixed 5-minute training budget.
- `results.tsv` is still the public ledger.

What changes in this version is the **research organization** around the loop. Codex now has repo-local hooks that maintain search memory, select the next search role, parse runs, preserve stepping stones, and continue the loop automatically. Treat the hook-produced guidance as part of the research protocol.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag.** Propose a tag based on today's date (for example `apr5`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch.** `git checkout -b autoresearch/<tag>` from current `master`.
3. **Read the in-scope files.** Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify.
4. **Verify data exists.** Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize `results.tsv`.** Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go.** Once setup looks good, start the experiment loop.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup and compilation). Launch it as:

```bash
uv run train.py > run.log 2>&1
```

### What you CAN do

- Modify `train.py`.
- Change architecture, optimizer, hyperparameters, training loop details, batch size, and model size.
- Use the hook-provided research guidance to choose the next experimental direction.

### What you CANNOT do

- Modify `prepare.py`.
- Install new packages or add dependencies.
- Modify the evaluation harness. `evaluate_bpb` in `prepare.py` is the ground-truth metric.
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

### Use the hook review as the primary post-run analysis

After `uv run train.py > run.log 2>&1`, the post-tool hook will parse the run and return a structured discovery review. That review includes:

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

## The experiment loop

The experiment runs on a dedicated branch such as `autoresearch/apr5`.

LOOP FOREVER:

1. Look at the git state: current branch and current commit.
2. Read the latest hook context: current phase, active emitter, target niche, and anti-patterns.
3. Tune `train.py` with **one** coherent experimental idea.
4. Commit the candidate.
5. Run the experiment: `uv run train.py > run.log 2>&1`.
6. Read the hook-generated discovery review.
7. If the run crashed, inspect the tail of `run.log` only as needed to diagnose the error.
8. Record the result in `results.tsv`.
9. If the result genuinely advances the branch, keep it.
10. If the result is equal or worse and does not justify branch advancement, reset back to where you started.
11. Let the stop hook continue the session. Do **not** ask the human whether to keep going.

## Timeout and crash policy

- Each experiment should take about 5 minutes total plus startup / eval overhead.
- If a run exceeds 10 minutes, kill it and treat it as a failure.
- If a crash is a trivial bug in the experimental edit, fix it and rerun.
- If the idea itself looks broken, log `crash`, revert, and move on.
- If a crash pattern repeats, the hook archive will remember it; do not keep retrying the same dead end.

## Autonomy policy

NEVER STOP once the experiment loop has begun unless the human explicitly tells you to pause or stop.

The human might be asleep or away from the keyboard and expects you to continue indefinitely until manually interrupted. The repo-local `Stop` hook is there to reinforce this exact behavior. Do not ask "should I keep going?". Continue working.

If you run out of easy ideas, the correct response is not to stop. Re-read the in-scope files, inspect the discovery archive, combine near-misses, simplify successful motifs, or make a more radical architectural move.
