import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/c0r_review"))
import professional_review


class ProfessionalReviewTests(unittest.TestCase):
    def test_direct_valid_uses_existing_preset(self):
        verdict = professional_review.normalize_row({"review_id": "target_review_0001", "verdict_code": 1})
        self.assertTrue(verdict["valid"])
        self.assertEqual(verdict["relation"], "direct_answer")

    def test_invalid_requires_human_note_without_inferred_issue(self):
        with self.assertRaises(ValueError):
            professional_review.normalize_row({"review_id": "target_review_0001", "verdict_code": 4, "note": "short"})
        verdict = professional_review.normalize_row({"review_id": "target_review_0001", "verdict_code": 4, "note": "Human reviewer documented a mismatch."})
        self.assertFalse(verdict["valid"])
        self.assertEqual(verdict["issue_type"], "other")

    def test_uncertain_routes_to_manual_review(self):
        verdict = professional_review.normalize_row({"review_id": "target_review_0001", "verdict_code": 5, "note": "Human reviewer could not decide reliably."})
        self.assertEqual(verdict["recommended_action"], "manual_review")

    def test_custom_verdict_hard_stops(self):
        with self.assertRaises(ValueError):
            professional_review.normalize_row({"review_id": "target_review_0001", "verdict_code": 6, "note": "custom"})

    def test_batch_requires_exact_order_and_unique_ids(self):
        rows = [
            {"review_sequence": 1, "review_id": "target_review_0001", "verdict_code": 1},
            {"review_sequence": 2, "review_id": "target_review_0002", "verdict_code": 2},
        ]
        self.assertEqual(len(professional_review.normalize_rows(rows, expected_count=2)), 2)


if __name__ == "__main__":
    unittest.main()
