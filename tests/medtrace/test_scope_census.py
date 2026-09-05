import unittest

from scripts.medtrace.build_scope_census import freeze_roles, scan_candidates, select_candidates, source_key, stratified_cap


class ScopeCensusTests(unittest.TestCase):
    def test_source_evidence_and_exclusion(self):
        primary = {
            "relative_image_path": "img0/source.jpg", "question": "Is X present?", "gold_answer": "yes",
            "source_triple": ["x", "present", "yes"], "source_base_type": "presence",
        }
        rows = [
            {"qid": 1, "img_name": "img0/source.jpg", "question": "What organ?", "answer": "lung", "triple": ["organ"], "base_type": "organ"},
            {"qid": 2, "img_name": "img1", "question": "Is X present?", "answer": "no", "triple": ["x", "present", "no"], "base_type": "presence"},
            {"qid": 3, "img_name": "img2", "question": "What modality?", "answer": "CT", "triple": ["modality"], "base_type": "modality"},
        ]
        excluded = {source_key("SLAKE", "img2", "What modality?", "CT")}
        selected = select_candidates(primary, rows, excluded)
        self.assertEqual([row["source_qid"] for row in selected], [1, 2])
        self.assertTrue(all(row["role"] == "UNASSIGNED_PRE_EQKEY" for row in selected))

    def test_stratified_cap_keeps_hard_rows_after_many_broad_rows(self):
        broad = [{"fact_relation": "broad_unrelated_source_qa", "source_qid": i} for i in range(130)]
        hard = [{"fact_relation": "same_image_other_source_fact", "source_qid": "hard"}]
        selected = stratified_cap(broad + hard)
        self.assertEqual(len(selected), 120)
        self.assertEqual(selected[0]["source_qid"], "hard")

    def test_equivalent_count_is_observed_and_unknown_never_freezes(self):
        primary = {"relative_image_path": "img0/source.jpg", "question": "Q", "gold_answer": "A", "source_triple": ["x"]}
        rows = [
            {"qid": 1, "img_name": "img0/source.jpg", "question": "Q", "answer": "A", "triple": ["x"], "base_type": "kvqa"},
            {"qid": 2, "img_name": "img2/source.jpg", "question": "other", "answer": "B", "triple": None, "base_type": None},
        ]
        scanned = scan_candidates(primary, rows, set())
        self.assertEqual([row["fact_relation"] for row in scanned], ["same_source_equivalent_candidate"])

    def test_role_freeze_uses_disjoint_families_and_images(self):
        candidates = [{
            "source_dataset": "SLAKE", "image_name": f"i{i}/source.jpg", "question": f"n{i}",
            "source_answer": "a", "scope_relation_verification_status": "SOURCE_RELATION_VERIFIED",
        } for i in range(60)]
        review = {
            "review_input_visibility": "SOURCE_QUESTION_ONLY__NO_MODEL_OUTPUTS",
            "candidates": [
                {"question": f"p-{role}-{i}", "role": role, "rewrite_family": role, "review_status": "APPROVED_EQUIVALENT"}
                for role in ("fit", "calibration", "evaluation") for i in range(4)
            ],
        }
        frozen = freeze_roles({"primary": {"record_id": "r"}, "negative_candidates_full": candidates}, review)
        self.assertEqual(frozen["role_counts"]["evaluation"], {"positive": 4, "negative": 20})
        images = [row["image_name"] for role in frozen["negatives"].values() for row in role]
        self.assertEqual(len(images), len(set(images)))


if __name__ == "__main__":
    unittest.main()
