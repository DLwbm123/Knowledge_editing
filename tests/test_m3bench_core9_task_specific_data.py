import csv
import tempfile
import unittest
from pathlib import Path

from scripts.m3bench_core9_base_predictions import inference_targets, replay_record_pass, semantic_verdict_index
from scripts.m3bench_core9_freeze_report import freeze_status
from scripts.m3bench_core9_public_query_inventory import assert_no_method_fields
from scripts.m3bench_core9_task_specific_cohorts import build_t3, macro_per_edit


class TaskSpecificDataTests(unittest.TestCase):
    def test_method_output_cannot_enter_selection(self):
        with self.assertRaises(ValueError):
            assert_no_method_fields([{"query_id": "q", "model_answer": "leak"}])

    def test_reuse_requires_token_level_replay(self):
        self.assertTrue(replay_record_pass({"token_ids_equal": True, "decoded_equal": True, "normalized_equal": True}))
        self.assertFalse(replay_record_pass({"token_ids_equal": False, "decoded_equal": True, "normalized_equal": True}))

    def test_replay_mismatch_forces_full_inventory(self):
        rows = [{"query_id": "reused", "legacy_source_record_ids": ["r"], "legacy_derived_probe_ids": []}, {"query_id": "new", "legacy_source_record_ids": [], "legacy_derived_probe_ids": []}]
        source = {"r": {"model_answer_raw": "x"}}
        self.assertEqual([row["query_id"] for row in inference_targets(rows, source, {}, True)], ["reused", "new"])
        self.assertEqual([row["query_id"] for row in inference_targets(rows, source, {}, False)], ["new"])

    def test_semantic_verdicts_are_boolean_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verdicts.jsonl"
            path.write_text('{"opaque_query_id":"q","is_correct":true}\n', encoding="utf-8")
            self.assertEqual(semantic_verdict_index(path), {"q": True})
            path.write_text('{"opaque_query_id":"q","is_correct":true}\n{"opaque_query_id":"q","is_correct":false}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                semantic_verdict_index(path)

    def test_macro_is_per_edit_not_pooled(self):
        self.assertAlmostEqual(macro_per_edit({"a": [True], "b": [True, False, False]}), 2 / 3)

    def test_freeze_status_names_all_zero_denominator_tasks(self):
        summaries = {task: {"eligible_edit_count": 1, "eligible_probe_count": 1} for task in ("T0", "T1L", "T1G", "T2L", "T2G", "T3L", "T3G", "T4L", "T4G")}
        summaries["T2L"]["eligible_probe_count"] = 0
        summaries["T4L"]["eligible_edit_count"] = 0
        self.assertEqual(freeze_status(summaries, False), "M3BENCH_CORE9_DATA_BLOCKED__T2L_T4L__ZERO_ELIGIBLE_COHORT")

    def test_t3_uses_paired_source_gold(self):
        anchor = {"query_id": "a", "dataset": "SLAKE", "image_id": "imgA", "question": "Is tumor present?", "gold_answer": "No"}
        paired = {"query_id": "b", "dataset": "SLAKE", "image_id": "imgB", "question": "Is tumor present?", "gold_answer": "Yes"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "t3_cross_modality_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_A", "modality_A", "diseases", "same_disease_images_in_other_modalities"])
                writer.writeheader()
                writer.writerow({
                    "image_A": "imgA", "modality_A": "MRI", "diseases": "['tumor']",
                    "same_disease_images_in_other_modalities": "[{'disease':'tumor','modality':'CT','image_id':'imgB'}]",
                })
            rows, audit = build_t3(root, {("SLAKE", "imgA"): [anchor], ("SLAKE", "imgB"): [paired]}, {"a": False, "b": True})
        self.assertEqual(rows["T3L"][0]["probe_query_ids"], ["b"])
        self.assertEqual(audit[0]["paired_gold_source"], "paired_image_source_qa")
        self.assertNotEqual(anchor["gold_answer"], paired["gold_answer"])


if __name__ == "__main__":
    unittest.main()
