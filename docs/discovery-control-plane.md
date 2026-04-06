# Codex Discovery Control Plane

This fork keeps the original `prepare.py` / `train.py` benchmark loop from `karpathy/autoresearch`, but wraps it in a small **repo-local research organization** that helps the agent search more broadly and remember what it has already learned.

The goal is not to turn the project into a heavyweight framework. The goal is to keep the core benchmark simple while giving the loop a few capabilities that plain hill-climbing lacks:

- multiple search styles instead of one mutation habit
- memory of near-misses, crashes, and niche elites
- automated review of completed runs
- autonomous next-step selection on a dedicated research branch
- consistent experiment identity and bookkeeping

## High-Level Loop

At a high level, the control plane changes the loop from:

1. edit `train.py`
2. run `train.py`
3. eyeball the result
4. think of another tweak

into:

1. load current research state
2. choose a search role and niche
3. make one coherent mutation
4. run the benchmark
5. parse and score the result
6. update archive / near-miss / crash memory
7. choose the next move automatically

The underlying benchmark is unchanged:

- `train.py` is still the only experiment target
- the metric is still `val_bpb`
- each run still gets a fixed 5-minute training budget

## Hook Architecture

The repo uses Codex repo-local hooks to make the loop stateful.

### SessionStart

Purpose:

- load current discovery state when a session starts or resumes
- inject a compact summary of best run, phase, archive elites, and warnings

Why it matters:

- the agent does not restart from a blank prompt every time
- it can resume with the same research context instead of rediscovering it manually

### UserPromptSubmit

Purpose:

- detect whether autonomy should be enabled or disabled
- inject repo-specific research context before the agent answers

Why it matters:

- the loop can be explicitly turned on or off
- user prompts become part of the repo-local operating mode rather than one-off chat state

### PreToolUse (Bash policy)

Purpose:

- block off-policy shell commands before they run

Examples:

- disallow dependency installation during experiments
- require `run.log` redirection for benchmark runs
- protect `prepare.py`
- block destructive or irrelevant commands in the research loop

Why it matters:

- the loop stays inside its intended search surface
- infrastructure or logging drift does not silently invalidate experiments

### PostToolUse (run analysis)

Purpose:

- detect completed benchmark runs
- parse `run.log`
- classify the experiment
- score it for fitness, novelty, information gain, and surprise
- write updated state and a structured run review

Why it matters:

- every run becomes machine-readable feedback, not just an unstructured text log
- the next step can depend on what actually happened, not just the agent's memory

### Stop

Purpose:

- decide the next move when the agent would otherwise stop
- keep the loop moving on `autoresearch/*` branches

Why it matters:

- autonomy is not just “keep going”
- the next step is chosen from the current phase, archive, emitter stats, and no-fly patterns

## Search Roles (Emitters)

The control plane does not mutate `train.py` in only one style. It rotates through different search roles.

### `local_tuner`

Purpose:

- make low-blast-radius improvements near a promising configuration

Typical use:

- small schedule changes
- small hyperparameter trims
- nearby exploitation of a recent near-win

### `optimizer_hacker`

Purpose:

- change optimizer or schedule behavior

Typical use:

- warmup/warmdown changes
- learning-rate floor/schedule changes
- Muon/AdamW parameterization tweaks

### `architecture_mutator`

Purpose:

- try one structural model change

Typical use:

- attention layout changes
- value-embedding coverage changes
- grouped-query or similar structural probes

### `simplifier`

Purpose:

- remove or compress machinery around a working idea

Typical use:

- delete a gating path
- reduce complexity in a schedule or auxiliary path
- test whether a winning motif still works in simpler form

### `contrarian`

Purpose:

- deliberately try the opposite of the recent local trend

Typical use:

- reverse a recent schedule direction
- move batch/regularization in the opposite direction from the current local basin

### `recombinator`

Purpose:

- combine promising near-misses into one coherent experiment

Typical use:

- merge a good architectural near-miss with a good optimizer near-miss
- test whether two partial wins reinforce each other

### `anomaly_chaser`

Purpose:

- investigate surprising results, crashes, or unexplained near-wins

Typical use:

- re-stamp a suspiciously strong run
- isolate why a crash happened
- probe around a result that seems too good or too bad to ignore

## Discovery Memory

The loop stores more than just “best so far.”

### Archive elites

For each niche, the control plane tracks the best result seen there.

Why it matters:

- the search can preserve stepping stones
- a non-global-best motif can still remain available for recombination later

### Near misses

Runs close to the best can be preserved as “live” alternatives even if they are not immediate winners.

Why it matters:

- the loop can revisit strong but slightly worse efficiency/simplicity tradeoffs
- promising lanes are not immediately forgotten

### Crash signatures

Repeated failures are remembered and surfaced as no-fly patterns.

Why it matters:

- the loop can stop wasting time on the same infrastructure or model failure
- crash investigation becomes a first-class signal instead of noise

### Conjectures

The system derives short, machine-generated working theories from accumulated outcomes.

Why it matters:

- the agent gets a compact synthesis of what seems to be working
- the next experiment can build on aggregated evidence rather than isolated anecdotes

## How Experiments Are Scored

The control plane does not rank runs only by raw `val_bpb`.

Each completed run can contribute through:

- **fitness gain**: did it improve the best-known result?
- **novelty**: did it explore a less-used niche or emitter?
- **information gain**: did it reveal something useful, even without winning?
- **surprise**: was the result unexpectedly good or unexpectedly bad?
- **complexity cost**: did it add too much code or moving parts?
- **VRAM cost**: did it increase memory substantially?

Why this matters:

- the search is closer to novelty/evolutionary search than to a pure greedy metric optimizer
- a run can be valuable because it teaches the system something, not only because it wins immediately

## Bookkeeping Discipline

The loop also tries to stay honest about what happened.

Key rules:

- one launched run maps to one `exp:` commit
- one completed run maps to one `results.tsv` row
- commit subject emitter tags are treated as authoritative for experiment identity
- stale plans are cleared if `HEAD`, `results.tsv`, and the current plan drift apart
- warnings are surfaced when the branch state and the discovery state disagree

Why it matters:

- the archive remains usable
- emitter statistics remain meaningful
- the agent is less likely to plan from stale or contradictory state

## Why This Is the Main Value-Add

The public branch is not mainly valuable because it has a remote GPU backend or a slightly better `train.py`.

Its main value-add is that it treats the repo like a **small autonomous research lab**:

- there is memory
- there are multiple search behaviors
- there is automatic run review
- there is structured continuation
- there is enough bookkeeping discipline to keep the loop intelligible over many runs

That is the central difference from upstream, and the rest of the fork exists mostly to support that loop cleanly.
