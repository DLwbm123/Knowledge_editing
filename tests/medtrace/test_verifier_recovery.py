"""CPU integration fixtures only; never used in scientific performance statistics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.medtrace import run_frozen_expert_visual_verifier as runner


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.old, self.execution, self.run, self.public = [root / name for name in ("old", "execution", "run", "public")]
        private = self.old / "private"
        private.mkdir(parents=True)
        image = root / "fixture.png"
        image.write_bytes(b"fixture-image-not-loaded")
        gate = root / "gate"; gate.mkdir()
        runtime_lock = gate / "lock.json"; runtime_lock.write_text('{}')
        base = private / "base.jsonl"; base.write_text('{}\n')
        runner.atomic_json(private / "CAMPAIGN_RUNTIME_CONFIG.json", {"cpu_gate": str(gate), "runtime_lock": str(runtime_lock), "base_predictions": str(base)})
        dev, scopes = [], {}
        for index in range(1, 13):
            record = f"fixture-{index}"
            dev.append({"edit_record": {"record_id": record, "question": "native"}})
            scopes[record] = {
                "status": "HARD_EVALUABLE" if index <= 7 else "BROAD_ONLY",
                "primary": {"target": "yes", "image_path": str(image), "image_name": image.name},
                "positives": {role: [{"question": role}] for role in ("fit", "calibration", "evaluation")},
                "negative_roles": {role: [{"fact_relation": runner.HARD_RELATION, "source_answer": "no", "image_path": str(image)}] for role in ("fit", "calibration", "evaluation")},
            }
            feature = private / f"features/e{index:02d}.pt"
            feature.parent.mkdir(exist_ok=True); feature.write_bytes(b"cpu-preflight-fixture")
            for seed in runner.SEEDS:
                folder = self.execution / "private/tasks" / f"P0_s{seed}_e{index:02d}" / "C1_R2_FIXED_Q_LONG_RECOVERY"
                folder.mkdir(parents=True)
                for name in ("step0800.pt", "e2e_step0800.json", "calibration_step0800.json"):
                    (folder / name).write_text('{}')
        runner.atomic_json(private / "frozen_data.json", {"dev": dev, "scopes": scopes})
        judge = self.execution / "private/judge"; judge.mkdir()
        for name in ("JUDGE_SIDECAR_PRIVATE.json", "JUDGE_OUTPUT_PRIVATE.jsonl", "JUDGE_EXECUTION_LOCK_PRIVATE.json"):
            (judge / name).write_text('{}')
        self.args = runner.parser().parse_args(["prepare", "--run-root", str(self.run), "--old-run", str(self.old), "--execution-run", str(self.execution), "--public-dir", str(self.public), "--base-commit", "HEAD"])
        self.args.func(self.args)
        config = json.loads((self.run / "private/CAMPAIGN_CONFIG.json").read_text())
        self.worker = argparse.Namespace(run_root=self.run, old_run=self.old, execution_run=self.execution, expected_code_commit=config["code_commit"], preflight_only=True, max_tasks=0)

    def start(self):
        args = runner.parser().parse_args(["start", "--run-root", str(self.run)])
        args.func(args)
        return runner.validate_start(self.run)

    def test_actual_prepare_start_worker_preflight_without_loader(self):
        self.start()
        with patch.object(runner, "verify_gpu", return_value=("2", runner.GPU_UUIDS["2"])), patch.object(runner, "load_real_runtime", side_effect=AssertionError("expensive loader forbidden")) as loader:
            runner.worker(self.worker)
            loader.assert_not_called()

    def test_missing_corrupt_or_misbound_start_fails_before_loader(self):
        with patch.object(runner, "load_real_runtime", side_effect=AssertionError("loader forbidden")) as loader:
            for value in (None, '{bad', '{}'):
                path = self.run / "private/CAMPAIGN_START.json"
                if value is not None: path.write_text(value)
                with self.assertRaisesRegex(RuntimeError, "campaign preflight"):
                    runner.worker(self.worker)
            loader.assert_not_called()

    def test_concurrent_start_and_resume_preserve_epoch(self):
        with ThreadPoolExecutor(2) as pool:
            records = list(pool.map(lambda _: self.start(), range(2)))
        self.assertEqual(records[0], records[1])
        self.assertEqual(records[0], self.start())
        records[0]["config_sha256"] = "wrong"
        runner.atomic_json(self.run / "private/CAMPAIGN_START.json", records[0])
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            self.start()

    def test_two_workers_claim_exclusively_from_fixed_21(self):
        self.start()
        queue = runner.TaskQueue(self.run / "private/TASK_QUEUE.json", self.run)
        with ThreadPoolExecutor(2) as pool:
            tasks = list(pool.map(lambda worker: queue.claim(f"fixture{worker}"), range(21)))
        self.assertEqual(len({task["task_id"] for task in tasks}), 21)
        self.assertIsNone(queue.claim("third"))
        self.assertEqual(len(queue.snapshot()["tasks"]), 21)

    def test_all_off_all_on_save_and_mismatch_fails(self):
        rows = [{"logical_id": "fixture"}]
        outputs = {"fixture": {"base": {"raw_token_ids": [1], "raw_answer": "base"}, "forced": {"raw_token_ids": [2], "raw_answer": "on"}}}
        calibration = {name: {"PRIMARY_SAFETY_FIRST": {"thresholds": [0, 0]}} for name in runner.CONDITIONS[1:]}
        for score in (-1, 1):
            scores = {"fixture": {name: [score, score] for name in runner.CONDITIONS[1:]}}
            result = runner.natural_replays(rows, scores, calibration, outputs, lambda row, on: outputs["fixture"]["forced" if on else "base"])
            runner.atomic_json(self.run / f"fixture-{score}.json", {"status": "COMPLETE", "replays": result})
            self.assertEqual(sum(row["exact_replay"] is True for row in result), 3)
            self.assertEqual(sum(row["status"].startswith("NO_NATURAL") for row in result), 3)
            with self.assertRaisesRegex(RuntimeError, "replay differ"):
                runner.natural_replays(rows, scores, calibration, outputs, lambda *args: {"raw_token_ids": [9], "raw_answer": "wrong"})

    def test_exit_queue_and_judge_all_required(self):
        self.start()
        args = argparse.Namespace(run_root=self.run, execution_run=self.execution, packet=self.run / "packet", sidecar=self.run / "sidecar")
        runner.atomic_json(self.run / "PROCESS_EXIT_CODES.json", {"gpu2": 0, "gpu3": 0})
        with self.assertRaisesRegex(RuntimeError, "expected 21"):
            runner.prepare_judge(args)
        runner.atomic_json(self.run / "PROCESS_EXIT_CODES.json", {"gpu2": 1})
        with patch.object(runner, "_task_results", return_value=[{}] * 21), self.assertRaisesRegex(RuntimeError, "exit codes"):
            runner.prepare_judge(args)
        runner.atomic_json(args.sidecar, {"tuples": {"required": {"reused_verdict": None}}})
        output = self.run / "judge-output"; output.write_text('')
        with self.assertRaisesRegex(RuntimeError, "every tuple"):
            runner._verdicts(args.sidecar, output)


if __name__ == "__main__":
    unittest.main()
