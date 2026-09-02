import unittest

from scripts.m3bench_lora_strong import (
    CURRENT_RUNTIME_CLASSIFICATION,
    FROZEN_RUNTIME_CLASSIFICATION,
    adaptive_should_stop,
    rank_config_summaries,
    validate_runtime_locks,
)


class LoraStrongTests(unittest.TestCase):
    def test_two_consecutive_adaptive_checkpoints_are_required(self):
        self.assertFalse(adaptive_should_stop([True]))
        self.assertFalse(adaptive_should_stop([True, False, True]))
        self.assertTrue(adaptive_should_stop([False, True, True]))

    def test_calibration_ranking_prioritizes_semantic_success(self):
        rows = [
            {"semantic_t0_correct": 7, "nll_decrease_count": 8, "exact_fuzzy_correct": 7,
             "median_post_nll": 0.1, "config": {"max_steps": 10, "learning_rate": 5e-5}},
            {"semantic_t0_correct": 8, "nll_decrease_count": 7, "exact_fuzzy_correct": 6,
             "median_post_nll": 0.2, "config": {"max_steps": 20, "learning_rate": 1e-4}},
        ]
        self.assertEqual(rank_config_summaries(rows)[0]["semantic_t0_correct"], 8)

    def test_only_runtime_classification_metadata_may_change(self):
        frozen = {"classification": FROZEN_RUNTIME_CLASSIFICATION, "blocks": 32}
        current = {"classification": CURRENT_RUNTIME_CLASSIFICATION, "blocks": 32}
        self.assertEqual(
            validate_runtime_locks(current, {"inventory_sha256": "new", "targets": 96},
                                   frozen, {"inventory_sha256": "old", "targets": 96}),
            "classification-metadata-amendment-v1-to-v2",
        )
        with self.assertRaises(RuntimeError):
            validate_runtime_locks({**current, "blocks": 31}, {"targets": 96}, frozen, {"targets": 96})


if __name__ == "__main__":
    unittest.main()
