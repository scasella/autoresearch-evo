from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_experiment = load_module("run_experiment", ROOT / "scripts" / "run_experiment.py")
modal_gpu = load_module("modal_gpu_shim", ROOT / "scripts" / "modal_gpu.py")


class RunnerTests(unittest.TestCase):
    def test_run_experiment_local_dispatch(self) -> None:
        with mock.patch.object(run_experiment, "run_local_command", return_value=7) as local_run:
            rc = run_experiment.main(["--backend", "local", "--", "python3", "-c", "print('ok')"])
        self.assertEqual(rc, 7)
        local_run.assert_called_once_with(["python3", "-c", "print('ok')"])

    def test_run_experiment_modal_dispatch(self) -> None:
        with mock.patch.object(run_experiment, "run_modal_command", return_value=3) as modal_run:
            rc = run_experiment.main(
                ["--backend", "modal", "--gpu", "A100", "--timeout", "12", "--", "uv", "run", "train.py"]
            )
        self.assertEqual(rc, 3)
        modal_run.assert_called_once()
        args, kwargs = modal_run.call_args
        self.assertEqual(args[0], ["uv", "run", "train.py"])
        self.assertEqual(kwargs["gpu"], "A100")
        self.assertEqual(kwargs["timeout"], 12)
        self.assertEqual(kwargs["repo_root"], ROOT)

    def test_modal_shim_forces_modal_backend(self) -> None:
        with mock.patch.object(run_experiment, "main", return_value=11) as runner_main:
            rc = modal_gpu.main(["--gpu", "H100", "--timeout", "10", "--", "uv", "run", "train.py"])
        self.assertEqual(rc, 11)
        runner_main.assert_called_once_with(
            ["--backend", "modal", "--gpu", "H100", "--timeout", "10", "--", "uv", "run", "train.py"]
        )


if __name__ == "__main__":
    unittest.main()
