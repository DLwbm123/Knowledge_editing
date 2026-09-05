import unittest

from scripts.medtrace.build_scope_census import select_candidates, source_key


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


if __name__ == "__main__":
    unittest.main()
