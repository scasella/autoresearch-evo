# Modal Backend

This fork exposes a generic experiment runner:

```bash
python scripts/run_experiment.py --backend modal --gpu H100 --timeout 10 -- uv run train.py > run.log 2>&1
```

The older top-level command still works as a compatibility shim:

```bash
python scripts/modal_gpu.py --gpu H100 --timeout 10 -- uv run train.py > run.log 2>&1
```

## What the Modal backend does

- builds a dependency-stable image with `uv_sync`
- mounts mutable repo code at startup
- reuses a persistent autoresearch cache volume
- reuses a persistent Hugging Face kernel-cache volume when available
- runs the inner benchmark command unchanged

The benchmark itself is still the 5-minute `train.py` time budget. The `10m` timeout is only an outer guardrail for remote startup, sync, and evaluation.

## Environment Variables

- `AUTORESEARCH_MODAL_APP_NAME`
- `AUTORESEARCH_MODAL_DATA_VOLUME_NAME`
- `AUTORESEARCH_MODAL_HF_VOLUME_NAME`

These are optional. The backend derives deterministic defaults if they are unset.

## First-Run Behavior

The first run on a new Modal app/volume set may seed remote cache volumes from the local machine. Later runs should be much faster because the backend can reuse the remote caches.

## When to use it

Use the Modal backend when:

- the local machine does not have the target NVIDIA GPU
- you want a reproducible remote H100/T4/A100 path
- you want the runner's remote cache reuse behavior

If you already have a suitable local NVIDIA GPU, the simpler local path remains:

```bash
uv run train.py
```
