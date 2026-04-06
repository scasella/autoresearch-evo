from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from modal.exception import NotFoundError


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "modal_gpu.py"
SPEC = importlib.util.spec_from_file_location("modal_gpu", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
modal_gpu = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = modal_gpu
SPEC.loader.exec_module(modal_gpu)


class FakeBatch:
    def __init__(self) -> None:
        self.directories: list[tuple[str, str]] = []
        self.files: list[tuple[bytes, str]] = []

    def put_directory(self, local_path: str, remote_path: str, recursive: bool = True) -> None:
        self.directories.append((local_path, remote_path))

    def put_file(self, local_file, remote_path: str, mode=None) -> None:
        if hasattr(local_file, "read"):
            payload = local_file.read()
        else:
            payload = Path(local_file).read_bytes()
        self.files.append((payload, remote_path))


class FakeBatchContext:
    def __init__(self, batch: FakeBatch) -> None:
        self.batch = batch

    def __enter__(self) -> FakeBatch:
        return self.batch

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeVolume:
    def __init__(self, manifest: dict | None = None) -> None:
        self.manifest = manifest
        self.batch = FakeBatch()

    def read_file_into_fileobj(self, path: str, fileobj: io.BytesIO) -> int:
        if path != modal_gpu.SEED_MANIFEST_PATH or self.manifest is None:
            raise NotFoundError("missing")
        payload = json.dumps(self.manifest).encode("utf-8")
        fileobj.write(payload)
        return len(payload)

    def batch_upload(self, force: bool = False) -> FakeBatchContext:
        return FakeBatchContext(self.batch)


class ModalGpuTests(unittest.TestCase):
    def test_resolve_config_defaults(self) -> None:
        repo_root = Path("/tmp/repo")
        config = modal_gpu.resolve_config(repo_root, {})
        self.assertEqual(config.app_name, modal_gpu.APP_NAME)
        self.assertEqual(config.data_volume_name, f"{modal_gpu.APP_NAME}-autoresearch-cache-v2")
        self.assertEqual(config.hf_volume_name, f"{modal_gpu.APP_NAME}-hf-cache-v2")
        self.assertEqual(config.repo_root, repo_root)

    def test_resolve_config_env_overrides(self) -> None:
        repo_root = Path("/tmp/repo")
        env = {
            "AUTORESEARCH_MODAL_APP_NAME": "my-app",
            "AUTORESEARCH_MODAL_DATA_VOLUME_NAME": "data-vol",
            "AUTORESEARCH_MODAL_HF_VOLUME_NAME": "hf-vol",
        }
        config = modal_gpu.resolve_config(repo_root, env)
        self.assertEqual(config.app_name, "my-app")
        self.assertEqual(config.data_volume_name, "data-vol")
        self.assertEqual(config.hf_volume_name, "hf-vol")

    def test_collect_hf_seed_dirs_only_returns_known_existing_repos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_root = Path(tmpdir) / "hub"
            wanted = hub_root / modal_gpu.HF_KERNEL_REPOS[0]
            wanted.mkdir(parents=True)
            (hub_root / "models--other--repo").mkdir(parents=True)
            found = modal_gpu.collect_hf_seed_dirs(Path(tmpdir))
        self.assertEqual(found, [(wanted, f"/hub/{wanted.name}")])

    def test_wrap_command_bootstraps_shared_venv(self) -> None:
        wrapped = modal_gpu.wrap_command(["uv", "run", "train.py"], hf_offline=True)
        self.assertEqual(wrapped[:2], ["/bin/bash", "-lc"])
        self.assertIn("ln -sfn /.uv/.venv /app/.venv", wrapped[2])
        self.assertIn("export HF_HUB_OFFLINE=1", wrapped[2])
        self.assertIn("exec uv run train.py", wrapped[2])

    def test_seed_autoresearch_volume_skips_when_manifest_present(self) -> None:
        volume = FakeVolume({"kind": "autoresearch-cache"})
        modal_gpu.seed_autoresearch_volume(volume, Path("/tmp/cache"))
        self.assertEqual(volume.batch.directories, [])
        self.assertEqual(volume.batch.files, [])

    def test_seed_autoresearch_volume_uploads_directory_and_manifest(self) -> None:
        volume = FakeVolume()
        modal_gpu.seed_autoresearch_volume(volume, Path("/tmp/cache"))
        self.assertEqual(volume.batch.directories, [("/tmp/cache", "/")])
        payloads = {remote: json.loads(data.decode("utf-8")) for data, remote in volume.batch.files}
        self.assertEqual(payloads[modal_gpu.SEED_MANIFEST_PATH]["kind"], "autoresearch-cache")

    def test_seed_hf_volume_only_uploads_missing_repos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_root = Path(tmpdir) / "hub"
            first = hub_root / modal_gpu.HF_KERNEL_REPOS[0]
            second = hub_root / modal_gpu.HF_KERNEL_REPOS[1]
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            volume = FakeVolume({"kind": "huggingface-cache", "repos": [first.name]})
            modal_gpu.seed_hf_volume(volume, Path(tmpdir))
        self.assertEqual(volume.batch.directories, [(str(second), f"/hub/{second.name}")])
        payloads = {remote: json.loads(data.decode("utf-8")) for data, remote in volume.batch.files}
        self.assertEqual(
            payloads[modal_gpu.SEED_MANIFEST_PATH]["repos"],
            sorted([first.name, second.name]),
        )


if __name__ == "__main__":
    unittest.main()
