import json
import tempfile
import unittest
from pathlib import Path

from scripts.medtrace.closeout_dev16 import task_metrics, whole_words
from scripts.medtrace.run_fixed_judge_vllm import select_model_length
from scripts.medtrace.run_scope_pilot import validate_eq_rows


class CloseoutTests(unittest.TestCase):
    def test_whole_answer_and_length_lane(self):
        self.assertTrue(whole_words("no mass", "There is no mass."))
        self.assertFalse(whole_words("no", "normal"))
        self.assertEqual(select_model_length(1001), 2048)
        self.assertEqual(select_model_length(2024), 2048)
        self.assertIsNone(select_model_length(4090))

    def test_macro_uses_only_edits_with_probes(self):
        rows = [
            {"task": "T0", "event_index": 1, "normalized_exact_reference_match": True, "semantic_correct": True, "reached_length_limit": False, "ended_with_eos": True, "truncated_without_eos": False},
            {"task": "T1L", "event_index": 1, "normalized_exact_reference_match": True, "semantic_correct": True, "reached_length_limit": False, "ended_with_eos": True, "truncated_without_eos": False},
            {"task": "T1L", "event_index": 1, "normalized_exact_reference_match": False, "semantic_correct": False, "reached_length_limit": False, "ended_with_eos": True, "truncated_without_eos": False},
            {"task": "T1L", "event_index": 2, "normalized_exact_reference_match": True, "semantic_correct": True, "reached_length_limit": False, "ended_with_eos": True, "truncated_without_eos": False},
        ]
        rows += [{"task": task, "event_index": 1, "normalized_exact_reference_match": True, "semantic_correct": True, "reached_length_limit": False, "ended_with_eos": True, "truncated_without_eos": False} for task in ("T1G", "T2G")]
        value = task_metrics(rows)["T1L"]
        self.assertEqual(value["eligible_edit_count"], 2)
        self.assertAlmostEqual(value["exact_micro"], 2 / 3)
        self.assertAlmostEqual(value["exact_macro"], 0.75)

    def test_eqkey_role_isolation(self):
        rows = [{"eqkey": str(i), "role": "fit", "label": "positive"} for i in range(2)]
        validate_eq_rows(rows, expected=2)
        rows[1]["eqkey"] = rows[0]["eqkey"]
        rows[1]["role"] = "evaluation"
        with self.assertRaisesRegex(RuntimeError, "EqKey"):
            validate_eq_rows(rows, expected=2)


if __name__ == "__main__":
    unittest.main()
