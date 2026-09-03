import csv
import tempfile
import unittest
from pathlib import Path

from scripts.m3bench_core9_base_predictions import replay_record_pass
from scripts.m3bench_core9_public_query_inventory import assert_no_method_fields
from scripts.m3bench_core9_task_specific_cohorts import build_t3, macro_per_edit


class TaskSpecificDataTests(unittest.TestCase):
    def test_method_output_cannot_enter_selection(self):
        with self.assertRaises(ValueError):
            assert_no_method_fields([{"query_id": "q", "model_answer": "leak"}])

    def test_reuse_requires_token_level_replay(self):
        self.assertTrue(replay_record_pass({"token_ids_equal": True, "decoded_equal": True, "normalized_equal": True}))
        self.assertFalse(replay_record_pass({"token_ids_equal": False, "decoded_equal": True, "normalized_equal": True}))

    def test_macro_is_per_edit_not_pooled(self):
        self.assertAlmostEqual(macro_per_edit({"a": [True], "b": [True, False, False]}), 2 / 3)

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
