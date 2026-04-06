#!/usr/bin/env python3
"""
Run an arbitrary repo command inside a Modal GPU sandbox.

Usage:
    python scripts/modal_gpu.py [--gpu GPU] [--timeout MINS] -- COMMAND...
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import modal
from modal.exception import NotFoundError
from modal_proto import api_pb2


APP_NAME = "autoresearch-gpu-runner"
SUPPORTED_GPUS = ("T4", "A10G", "A100", "H100")
DEFAULT_TIMEOUT_MINUTES = 10
REMOTE_REPO_ROOT = "/app"
REMOTE_AUTORESEARCH_CACHE = "/root/.cache/autoresearch"
REMOTE_HF_CACHE = "/root/.cache/huggingface"
SEED_MANIFEST_PATH = "/.seed-manifest.json"
HF_KERNEL_REPOS = (
    "models--varunneal--flash-attention-3",
    "models--kernels-community--flash-attn3",
)


@dataclass(frozen=True)
class RunnerConfig:
    app_name: str
    data_volume_name: str
    hf_volume_name: str
    repo_root: Path
    cache_root: Path
    hf_cache_root: Path


def resolve_config(repo_root: Path, env: dict[str, str] | None = None) -> RunnerConfig:
    env_map = env or os.environ
    app_name = env_map.get("AUTORESEARCH_MODAL_APP_NAME", APP_NAME)
    data_volume_name = env_map.get("AUTORESEARCH_MODAL_DATA_VOLUME_NAME", f"{app_name}-autoresearch-cache-v2")
    hf_volume_name = env_map.get("AUTORESEARCH_MODAL_HF_VOLUME_NAME", f"{app_name}-hf-cache-v2")
    return RunnerConfig(
        app_name=app_name,
        data_volume_name=data_volume_name,
        hf_volume_name=hf_volume_name,
        repo_root=repo_root,
        cache_root=Path.home() / ".cache" / "autoresearch",
        hf_cache_root=Path.home() / ".cache" / "huggingface",
    )


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if any(arg in {"-h", "--help"} for arg in argv):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--gpu", choices=SUPPORTED_GPUS, default="T4")
        parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_MINUTES, help="sandbox timeout in minutes")
        parser.print_help()
        raise SystemExit(0)
    if "--" not in argv:
        raise SystemExit("usage: python scripts/modal_gpu.py [--gpu GPU] [--timeout MINS] -- COMMAND...")
    split = argv.index("--")
    args = argv[:split]
    command = argv[split + 1 :]
    if not command:
        raise SystemExit("expected a command after `--`")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=SUPPORTED_GPUS, default="T4")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_MINUTES, help="sandbox timeout in minutes")
    return parser.parse_args(args), command


def log(message: str) -> None:
    print(f"[modal-gpu] {message}", file=sys.stderr, flush=True)


def build_image(repo_root: Path) -> modal.Image:
    ignore = [
        ".git",
        ".venv",
        ".omx",
        "results",
        "run.log",
        ".DS_Store",
        "**/__pycache__",
        "**/*.pyc",
    ]
    return (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install("uv>=0.7.0")
        .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        .uv_sync(str(repo_root), frozen=True)
        .add_local_dir(str(repo_root), remote_path=REMOTE_REPO_ROOT, copy=False, ignore=ignore)
    )


def open_volume(name: str) -> modal.Volume:
    return modal.Volume.from_name(
        name,
        create_if_missing=True,
        version=api_pb2.VolumeFsVersion.VOLUME_FS_VERSION_V2,
    )


def read_volume_json(volume: modal.Volume, path: str) -> dict | None:
    payload = io.BytesIO()
    try:
        volume.read_file_into_fileobj(path, payload)
    except (NotFoundError, FileNotFoundError):
        return None
    return json.loads(payload.getvalue().decode("utf-8"))


def write_manifest(batch, payload: dict) -> None:
    batch.put_file(
        io.BytesIO(json.dumps(payload, sort_keys=True).encode("utf-8")),
        SEED_MANIFEST_PATH,
    )


def seed_autoresearch_volume(volume: modal.Volume, cache_root: Path) -> None:
    manifest = read_volume_json(volume, SEED_MANIFEST_PATH)
    if manifest and manifest.get("kind") == "autoresearch-cache":
        log("reusing autoresearch cache volume")
        return
    log(f"seeding autoresearch cache volume from {cache_root}")
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(cache_root), "/")
        write_manifest(
            batch,
            {
                "kind": "autoresearch-cache",
                "seed_source": str(cache_root),
            },
        )


def collect_hf_seed_dirs(hf_cache_root: Path) -> list[tuple[Path, str]]:
    hub_root = hf_cache_root / "hub"
    available: list[tuple[Path, str]] = []
    for repo_dir in HF_KERNEL_REPOS:
        local_repo_cache = hub_root / repo_dir
        if local_repo_cache.exists():
            available.append((local_repo_cache, f"/hub/{repo_dir}"))
    return available


def seed_hf_volume(volume: modal.Volume, hf_cache_root: Path) -> None:
    available = collect_hf_seed_dirs(hf_cache_root)
    if not available:
        log("no local HF kernel cache available; leaving HF volume unseeded")
        return
    manifest = read_volume_json(volume, SEED_MANIFEST_PATH) or {}
    seeded = set(manifest.get("repos", []))
    missing = [(path, remote) for path, remote in available if Path(remote).name not in seeded]
    if not missing and manifest.get("kind") == "huggingface-cache":
        log(f"reusing HF cache volume for {', '.join(sorted(seeded))}")
        return
    log(
        "seeding HF cache volume for "
        + ", ".join(Path(remote).name for _, remote in missing)
    )
    with volume.batch_upload(force=True) as batch:
        for local_path, remote_path in missing:
            batch.put_directory(str(local_path), remote_path)
        write_manifest(
            batch,
            {
                "kind": "huggingface-cache",
                "repos": sorted({Path(remote).name for _, remote in available}),
                "seed_source": str(hf_cache_root),
            },
        )


def prepare_volumes(config: RunnerConfig) -> tuple[dict[str, modal.Volume], bool]:
    data_volume = open_volume(config.data_volume_name)
    hf_volume = open_volume(config.hf_volume_name)
    seed_autoresearch_volume(data_volume, config.cache_root)
    seed_hf_volume(hf_volume, config.hf_cache_root)
    hf_manifest = read_volume_json(hf_volume, SEED_MANIFEST_PATH) or {}
    hf_ready = hf_manifest.get("kind") == "huggingface-cache"
    return (
        {
            REMOTE_AUTORESEARCH_CACHE: data_volume,
            REMOTE_HF_CACHE: hf_volume,
        },
        hf_ready,
    )


def wrap_command(command: list[str], *, hf_offline: bool) -> list[str]:
    command_text = " ".join(shlex.quote(arg) for arg in command)
    exports = ""
    if hf_offline:
        exports = "export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1; "
    bootstrap = (
        f"mkdir -p {shlex.quote(REMOTE_REPO_ROOT)} "
        f"&& ln -sfn /.uv/.venv {shlex.quote(REMOTE_REPO_ROOT + '/.venv')} "
        f"&& {exports}exec {command_text}"
    )
    return ["/bin/bash", "-lc", bootstrap]


def _drain_output(stream, target, label: str) -> None:
    try:
        payload = stream.read()
    except Exception as exc:  # pragma: no cover - best effort stream cleanup
        print(f"[modal-gpu] {label} read error: {exc}", file=sys.stderr)
        return
    if payload:
        target.write(payload)
        target.flush()


def main(argv: list[str]) -> int:
    args, command = parse_args(argv)
    config = resolve_config(Path(__file__).resolve().parents[1])
    if not config.cache_root.exists():
        raise SystemExit("missing ~/.cache/autoresearch; run `uv run prepare.py` locally first")
    image = build_image(config.repo_root)
    app = None
    sandbox = None

    def terminate_sandbox(*_signal_args) -> None:
        if sandbox is not None:
            sandbox.terminate()
        raise SystemExit(130)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, terminate_sandbox)

    log(f"launching command on {args.gpu} with timeout={args.timeout}m: {' '.join(command)}")

    try:
        with modal.enable_output():
            log(f"resolving app {config.app_name}")
            app = modal.App.lookup(config.app_name, create_if_missing=True)
            log("building image")
            build_started = time.time()
            image = image.build(app)
            log(f"image ready in {time.time() - build_started:.1f}s")
            log("preparing volumes")
            volume_started = time.time()
            volumes, hf_ready = prepare_volumes(config)
            log(f"volumes ready in {time.time() - volume_started:.1f}s (hf_offline={'on' if hf_ready else 'off'})")
            log("creating sandbox")
            sandbox = modal.Sandbox.create(
                *wrap_command(command, hf_offline=hf_ready),
                app=app,
                image=image,
                gpu=args.gpu,
                timeout=args.timeout * 60,
                volumes=volumes,
                workdir=REMOTE_REPO_ROOT,
            )

        log("waiting for sandbox completion")
        sandbox.wait(raise_on_termination=False)
        _drain_output(sandbox.stdout, sys.stdout, "stdout")
        _drain_output(sandbox.stderr, sys.stderr, "stderr")
        return int(sandbox.returncode or 0)
    finally:
        if sandbox is not None:
            try:
                if sandbox.poll() is None:
                    sandbox.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
