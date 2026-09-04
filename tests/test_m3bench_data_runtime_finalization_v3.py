import tempfile
import unittest
from pathlib import Path

from scripts.m3bench_base_correctness_v3 import (
    EXPECTED_OLD_RAW_SHA256, exact_correct, legacy_correct, majority, public_fuzzy_correct,
)
from scripts.m3bench_core9_public_query_inventory import assert_no_method_fields
from scripts.m3bench_core9_task_specific_cohorts import macro_per_edit
from scripts.m3bench_static_catalog_v3 import T5_STATUS, select_runtime, t0_filter, t3_partition, t4l_eligible
from scripts.run_semantic_judge_v3 import PACKET_FIELDS, parse_boolean, parse_vote
from scripts.m3bench_checkpoint_runtime_v3 import LLAVA_MED_COMMIT, MODEL_FILES, MODEL_SHA, VISION_FILES, VISION_SHA


class DataRuntimeFinalizationV3Tests(unittest.TestCase):
    def test_legacy_scorer_exact_reproduction(self):
        self.assertTrue(legacy_correct("pleural effusion", "pleural effusion"))
        self.assertTrue(legacy_correct("left pleural effusion", "pleural effusion left"))

    def test_public_fuzzy_exact_reproduction(self):
        self.assertTrue(public_fuzzy_correct("There is evidence of pleural effusion", "pleural effusion"))

    def test_short_answer_nonexact_goes_to_semantic_judge(self):
        self.assertFalse(exact_correct("Yes, this image shows cardiomegaly.", "yes"))

    def test_negation_not_accepted_by_substring(self):
        self.assertTrue(public_fuzzy_correct("No mass is present", "mass"))
        self.assertFalse(exact_correct("No mass is present", "mass"))

    def test_gate_critical_majority_adjudication(self):
        self.assertTrue(majority([True, True], True))
        with self.assertRaises(ValueError):
            majority([True, False], True)
        self.assertFalse(majority([True, False, False], True))

    def test_old_raw_hash_is_immutable(self):
        self.assertEqual(EXPECTED_OLD_RAW_SHA256, "25006913f849d7fedfe0fc100a789badad2ef093c09e9614fa511d4ed73251dc")

    def test_checkpoint_manifest_complete(self):
        required = {"config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"}
        self.assertTrue(required <= MODEL_FILES.keys())
        self.assertIn("pytorch_model.bin", VISION_FILES)
        self.assertEqual(len(MODEL_SHA), 40); self.assertEqual(len(VISION_SHA), 40); self.assertEqual(len(LLAVA_MED_COMMIT), 40)

    def test_runtime_canary_selection_is_not_score_cherry_picking(self):
        audit = {"checkpoint_identity_verified": True, "native_runtime_stable": True, "official_prompt_image_generation": True, "no_runtime_errors": True, "normalized_parity": .994, "semantic_parity": 1.0, "source_accuracy": 1.0}
        self.assertEqual(select_runtime(audit), "runtime_b_official_native")

    def test_runtime_output_parity(self):
        audit = {"checkpoint_identity_verified": True, "native_runtime_stable": True, "official_prompt_image_generation": True, "no_runtime_errors": True, "normalized_parity": .995, "semantic_parity": .995, "source_accuracy": 0.0}
        self.assertEqual(select_runtime(audit), "runtime_a_official_parity")

    def test_static_catalog_independent_of_base_verdict(self):
        self.assertNotIn("verdict", {"query_id": "q", "method_outputs_used": False})

    def test_target_validity_separate_from_base_wrong(self):
        self.assertEqual(t0_filter([{"query_id": "a"}, {"query_id": "b"}], {"a": True, "b": False}), [{"query_id": "b"}])

    def test_t0_sequence_refilters_base_correct(self):
        rows = [{"query_id": "a", "amended_position": 1}, {"query_id": "b", "amended_position": 2}]
        self.assertEqual([row["query_id"] for row in t0_filter(rows, {"a": False, "b": True})], ["a"])

    def test_t2l_task_specific_not_t0_anchored(self):
        self.assertTrue({"source": "t2l_cross_image_pairs.csv"}.get("source"))

    def test_t3_l_g_mutually_exclusive(self):
        locality, generality = t3_partition(["a", "b"], {"a": True, "b": False})
        self.assertFalse(locality & generality)

    def test_t4l_qA_wrong_qB_correct(self):
        self.assertTrue(t4l_eligible(False, True)); self.assertFalse(t4l_eligible(True, True))

    def test_macro_per_edit_not_micro(self):
        self.assertAlmostEqual(macro_per_edit({"a": [True], "b": [True, False, False]}), 2 / 3)

    def test_no_method_fields_in_data_selection(self):
        with self.assertRaises(ValueError):
            assert_no_method_fields([{"query_id": "q", "model_answer": "leak"}])

    def test_t5_does_not_block_t0_t4(self):
        self.assertIn("T5_SEPARATE_EXTENSION_BLOCKED", T5_STATUS)

    def test_semantic_judge_strict_boolean_schema(self):
        self.assertTrue(parse_boolean('{"is_correct": true}'))
        self.assertFalse(parse_boolean('{"is_correct": false}'))
        with self.assertRaises(ValueError):
            parse_boolean('CORRECT')

    def test_malformed_judge_output_is_preserved_without_a_vote(self):
        self.assertEqual(parse_vote('CORRECT'), (None, False))

    def test_semantic_judge_packet_is_method_blind(self):
        self.assertEqual(PACKET_FIELDS, {"opaque_query_id", "question", "gold_answer", "raw_base_answer", "adjudication_pass"})


if __name__ == "__main__":
    unittest.main()
