from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.core import (  # noqa: E402
    build_next_prompt,
    check_command_policy,
    detect_autonomy_preference,
    initial_state,
    load_state,
    parse_metrics_text,
    record_experiment_from_run,
    save_state,
    select_next_plan,
)


class ResearchCoreTests(unittest.TestCase):
    def test_parse_metrics_text(self) -> None:
        text = """
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
        """
        metrics = parse_metrics_text(text)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertAlmostEqual(metrics.val_bpb, 0.9979)
        self.assertEqual(metrics.depth, 8)
        self.assertEqual(metrics.num_steps, 953)

    def test_parse_metrics_text_ignores_unknown_metrics(self) -> None:
        text = """
        ---
        val_bpb:          0.997900
        unexpected_metric: 123.45
        depth:            8
        """
        metrics = parse_metrics_text(text)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertFalse(hasattr(metrics, "unexpected_metric"))
        self.assertEqual(metrics.depth, 8)

    def test_check_command_policy_blocks_bad_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train.py").write_text("print('hi')\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
            subprocess.run(["git", "add", "train.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "checkout", "-b", "autoresearch/test"], cwd=root, check=True, capture_output=True)

            reason = check_command_policy(root, "uv run train.py")
            self.assertIn("run.log", reason or "")
            reason = check_command_policy(root, "uv run train.py > run.log")
            self.assertIn("2>&1", reason or "")
            reason = check_command_policy(root, "uv run train.py | tee run.log")
            self.assertIn("tee", reason or "")
            self.assertIsNone(check_command_policy(root, "uv run train.py > run.log 2>&1"))

    def test_select_next_plan_prefers_explore_after_stall(self) -> None:
        state = initial_state()
        state["autonomy_enabled"] = True
        state["experiment_count"] = 8
        state["stall_count"] = 6
        plan = select_next_plan(state)
        self.assertIn(plan.phase, {"explore", "recombine", "stabilize"})
        self.assertIn(plan.emitter, state["emitter_stats"])

    def test_select_next_plan_breaks_emitter_streak(self) -> None:
        state = initial_state()
        state["experiment_count"] = 12
        state["stall_count"] = 5
        state["last_emitter"] = "anomaly_chaser"
        state["current_emitter_streak"] = 4
        state["recent_emitters"] = ["anomaly_chaser"] * 8
        for name in state["emitter_stats"]:
            state["emitter_stats"][name]["attempts"] = 2
            state["emitter_stats"][name]["total_reward"] = 0.5
        state["emitter_stats"]["anomaly_chaser"]["attempts"] = 10
        state["emitter_stats"]["anomaly_chaser"]["total_reward"] = 12.0

        plan = select_next_plan(state)
        self.assertNotEqual(plan.emitter, "anomaly_chaser")

    def test_detect_autonomy_preference_is_precise(self) -> None:
        self.assertTrue(detect_autonomy_preference("Read program.md, run the baseline, and continue autonomously."))
        self.assertIsNone(detect_autonomy_preference("Start by reviewing the overlay and summarize it."))
        self.assertFalse(detect_autonomy_preference("Pause here and summarize only."))

    def test_build_next_prompt_does_not_repeat_missing_cache_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train.py").write_text("print('hi')\n", encoding="utf-8")
            (root / "results.tsv").write_text(
                "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
            subprocess.run(["git", "add", "train.py", "results.tsv"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "baseline"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "checkout", "-b", "autoresearch/test"], cwd=root, check=True, capture_output=True)

            state = load_state(root)
            state["autonomy_enabled"] = True
            save_state(root, state)

            old_home = os.environ.get("HOME")
            try:
                os.environ["HOME"] = tmpdir
                prompt = build_next_prompt(root, persist=False)
                self.assertIn("uv run prepare.py", prompt or "")
                repeated = build_next_prompt(
                    root,
                    stop_hook_active=True,
                    last_assistant_message=prompt,
                    persist=False,
                )
                self.assertIsNone(repeated)
            finally:
                if old_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = old_home

    def test_record_experiment_from_run_updates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train.py").write_text("DEPTH = 8\nLR = 0.01\n", encoding="utf-8")
            (root / "results.tsv").write_text(
                "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
            subprocess.run(["git", "add", "train.py", "results.tsv"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "baseline"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "checkout", "-b", "autoresearch/test"], cwd=root, check=True, capture_output=True)
            (root / "train.py").write_text("DEPTH = 9\nLR = 0.01\n", encoding="utf-8")
            subprocess.run(["git", "add", "train.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "exp: raise depth [emitter=architecture_mutator]"], cwd=root, check=True, capture_output=True)
            (root / "run.log").write_text(
                "---\nval_bpb:          0.997900\ntraining_seconds: 300.1\npeak_vram_mb:     45060.2\nmfu_percent:      39.80\nnum_params_M:     50.3\ndepth:            9\n",
                encoding="utf-8",
            )

            experiment, state = record_experiment_from_run(root)
            self.assertIsNotNone(experiment)
            assert experiment is not None
            self.assertEqual(experiment.category, "architecture")
            self.assertEqual(state["best_commit"], experiment.commit)
            self.assertGreaterEqual(state["experiment_count"], 1)
            self.assertTrue(state["archive"])

    def test_record_experiment_from_run_prefers_commit_subject_emitter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train.py").write_text("DEPTH = 8\nLR = 0.01\n", encoding="utf-8")
            (root / "results.tsv").write_text(
                "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
            subprocess.run(["git", "add", "train.py", "results.tsv"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "baseline"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "checkout", "-b", "autoresearch/test"], cwd=root, check=True, capture_output=True)
            (root / "train.py").write_text("DEPTH = 8\nLR = 0.02\n", encoding="utf-8")
            subprocess.run(["git", "add", "train.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "exp: schedule tweak [emitter=contrarian]"], cwd=root, check=True, capture_output=True)
            save_state(root, {**load_state(root), "current_plan": {"emitter": "local_tuner", "phase": "explore"}})
            (root / "run.log").write_text(
                "---\nval_bpb:          0.997800\ntraining_seconds: 300.1\ntotal_seconds:    350.1\npeak_vram_mb:     45060.2\nmfu_percent:      39.80\nnum_params_M:     50.3\ndepth:            8\n",
                encoding="utf-8",
            )

            experiment, _state = record_experiment_from_run(root)
            self.assertIsNotNone(experiment)
            assert experiment is not None
            self.assertEqual(experiment.emitter, "contrarian")

    def test_build_next_prompt_prefers_in_flight_head_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train.py").write_text("DEPTH = 8\nLR = 0.01\n", encoding="utf-8")
            (root / "results.tsv").write_text(
                "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
                "aaaaaaa\t0.997900\t44.0\tkeep\tbaseline\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
            subprocess.run(["git", "add", "train.py", "results.tsv"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "baseline"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "checkout", "-b", "autoresearch/test"], cwd=root, check=True, capture_output=True)
            (root / "train.py").write_text("DEPTH = 9\nLR = 0.01\n", encoding="utf-8")
            subprocess.run(["git", "add", "train.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "exp: raise depth [emitter=architecture_mutator]"], cwd=root, check=True, capture_output=True)

            state = load_state(root)
            state["autonomy_enabled"] = True
            state["current_plan"] = {"emitter": "local_tuner", "phase": "explore"}
            save_state(root, state)

            prompt = build_next_prompt(root, persist=False)
            self.assertIn("existing in-flight experiment", prompt or "")
            self.assertIn("raise depth", subprocess.check_output(["git", "log", "-1", "--pretty=%s"], cwd=root, text=True))


if __name__ == "__main__":
    unittest.main()
