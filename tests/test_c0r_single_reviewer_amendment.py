import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/c0r_review"))
import single_reviewer_amendment as amendment


class SingleReviewerAmendmentTests(unittest.TestCase):
    def test_outcome_policy(self):
        valid = {"valid": True, "confidence": "high", "relation": "direct_answer", "recommended_action": "retain"}
        invalid = {**valid, "valid": False, "recommended_action": "exclude"}
        unresolved = {**invalid, "confidence": "low", "relation": "ambiguous", "recommended_action": "manual_review"}
        self.assertEqual(amendment.final_census([valid, invalid, unresolved]), {"VALID": 1, "CONFIRMED_INVALID": 1, "UNRESOLVED": 1})

    def test_evidence_precedence(self):
        self.assertEqual(amendment.classify_nonretained(True, True), "AUTHORITATIVE_SOURCE_REPAIR_AVAILABLE")
        self.assertEqual(amendment.classify_nonretained(False, True), "PRE_FORMAL_FROZEN_RESERVE_AVAILABLE")
        self.assertEqual(amendment.classify_nonretained(False, False), "EXCLUSION_REQUIRED")

    def test_event_impact_uses_single_positions_and_sequential_checkpoint_state(self):
        events = []
        for group in range(1, 5):
            gid = f"group_{group:02d}"
            events.extend([
                {"mode": "single", "formal_edit_position": 19, "checkpoint": "single", "anonymous_group_id": gid},
                {"mode": "single", "formal_edit_position": 20, "checkpoint": "single", "anonymous_group_id": gid},
                {"mode": "sequential", "formal_edit_position": 1, "checkpoint": "1", "anonymous_group_id": gid},
                {"mode": "sequential", "formal_edit_position": 1, "checkpoint": "50", "anonymous_group_id": gid},
            ])
        result = amendment.calculate_event_impact(events, [19])
        self.assertEqual((result["affected_single_events"], result["affected_sequential_events"]), (4, 4))
        self.assertTrue(result["single_group_symmetric"] and result["sequential_group_symmetric"])


if __name__ == "__main__":
    unittest.main()
