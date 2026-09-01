import sys
import json
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/c0r_review"))
import professional_review
import reviewctl


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

    def test_pilot_difference_is_not_an_outcome_conflict(self):
        pilot = professional_review.normalize_row({"review_id": "target_review_0001", "verdict_code": 1})
        expert = professional_review.normalize_row({"review_id": "target_review_0001", "verdict_code": 3})
        report = professional_review.compare_pilot_overlap([pilot], [expert])
        self.assertEqual(len(report["structured_difference_ids"]), 1)
        self.assertEqual(report["outcome_conflict_ids"], [])

    def test_blind_b_selection_has_mandatory_items_and_twenty_controls(self):
        verdicts = []
        for index in range(1, 27):
            code = 3 if index == 2 else 4 if index == 3 else 1
            note = "Human reviewer documented a mismatch." if code == 4 else ""
            verdicts.append(professional_review.normalize_row({"review_id": f"target_review_{index:04d}", "verdict_code": code, "note": note}))
        datasets = {row["review_id"]: "A" if index % 2 else "B" for index, row in enumerate(verdicts)}
        result = professional_review.build_blind_reviewer_b_selection(["target_review_0001"], verdicts, datasets, 7, 11)
        self.assertEqual(len(result["control_ids"]), 20)
        self.assertTrue({"target_review_0001", "target_review_0002", "target_review_0003"} <= set(result["mandatory_ids"]))
        self.assertEqual(len(result["queue"]), 23)

    def test_reviewctl_resolves_active_output_without_touching_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            active = review / "sessions/Reviewer_A/ACTIVE_REVIEWER_A_OUTPUT.json"
            output = review / "sessions/Reviewer_A/imports/run/REVIEWER_A_EXPERT_OUTPUT.jsonl"
            output.parent.mkdir(parents=True); output.write_text("", encoding="utf-8")
            active.parent.mkdir(parents=True, exist_ok=True)
            active.write_text(json.dumps({"output": str(output.relative_to(review))}), encoding="utf-8")
            paths = {"review": review, "active_output": active, "output": review / "pilot.jsonl"}
            self.assertEqual(reviewctl.active_artifact(paths, "output", paths["output"]), output.resolve())


if __name__ == "__main__":
    unittest.main()
