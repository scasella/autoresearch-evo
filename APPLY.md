# Apply this update to upstream autoresearch

Copy the contents of this overlay into the root of `karpathy/autoresearch`, preserving paths.

## What changes

- `program.md` is upgraded from a bare loop prompt into a hook-aware discovery protocol.
- `.codex/config.toml` enables Codex hooks and tells Codex to treat `program.md` as project guidance.
- `.codex/hooks.json` wires repo-local hooks for session start, prompt submit, Bash policy checks, post-run analysis, and stop-time continuation.
- `research/` contains the shared control-plane logic used by the hooks.
- `tests/` provides a small stdlib-only test suite for the new logic.

## Expected runtime behavior

- Open the repo as a trusted project in Codex.
- Start an `autoresearch/<tag>` branch.
- The stop hook keeps the loop alive automatically once autonomy is enabled.
- The post-tool hook parses `run.log`, updates `results/discovery/`, and feeds a structured review back into Codex.

## Useful manual commands

```bash
python -m unittest discover -s tests -v
python -m research.cli summary
python -m research.cli archive
python -m research.cli next
```
