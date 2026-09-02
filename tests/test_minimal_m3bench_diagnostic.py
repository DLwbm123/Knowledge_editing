import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimal_m3bench_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("minimal_m3bench_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MinimalDiagnosticTest(unittest.TestCase):
    def test_method_blind_scoring_primitives(self):
        events = []
        for group in ("group_01", "group_02", "group_03", "group_04"):
            events.extend([
                {
                    "anonymous_group_id": group, "mode": "single", "checkpoint": "single",
                    "task_id": "T0", "formal_edit_position": 1, "probe_id": "p0", "anchor_record_id": "a0",
                    "question": "Is there an abnormality?", "gold_or_reference": "yes",
                    "raw_model_answer": "Yes.", "dataset": "fixture", "source_raw_path": "raw.json",
                },
                {
                    "anonymous_group_id": group, "mode": "sequential", "checkpoint": "final",
                    "task_id": "T1L", "formal_edit_position": 1, "probe_id": "p1", "anchor_record_id": "a1",
                    "question": "Describe the finding.", "gold_or_reference": "A long free-text medical reference.",
                    "raw_model_answer": "A matching free-text response.", "dataset": "fixture", "source_raw_path": "raw.json",
                },
            ])
        check = MODULE.validate_events(events, 8)
        self.assertTrue(check["event_key_symmetry"])
        self.assertTrue(MODULE.deterministic_correct("YES", "yes"))
        event_to_id, payloads = MODULE.semantic_payloads(events)
        self.assertEqual(len(event_to_id), 4)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(set(payloads[0]), {
            "opaque_event_id", "question", "gold_or_reference", "raw_model_answer", "task_metadata"
        })
        self.assertNotIn("anonymous_group_id", json.dumps(payloads))

        with tempfile.TemporaryDirectory() as directory:
            mapping = Path(directory) / "map.json"
            mapping.write_text(json.dumps({"group_mapping": {
                "group_01": "lora", "group_02": "grace", "group_03": "balancedit", "group_04": "belora"
            }}))
            self.assertEqual(set(MODULE.read_method_map(mapping, check["anonymous_groups"]).values()), {
                "LoRA", "GRACE", "BalanceEdit", "BELoRA"
            })


if __name__ == "__main__":
    unittest.main()
