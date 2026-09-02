import unittest

from scripts.editor_paperspec_formal import grace_state_checks, method_state_contract


def grace_summary(*, edits: int, entries: int) -> dict:
    return {
        "entry_count": entries,
        "logical_edit_ids": [f"entry-{index}" for index in range(entries)],
        "radii": [1.0] * entries,
        "value_entry_count": entries,
        "requested_to_effective_count": edits,
        "requested_mapping_keys_match_history": True,
        "requested_mapping_values_resident": True,
    }


class GraceStateContractTests(unittest.TestCase):
    def test_same_label_reuse_preserves_valid_request_contract(self):
        summary = grace_summary(edits=15, entries=14)
        checks = grace_state_checks(
            summary,
            15,
            prior_entry_count=14,
            insertion_action="same_label_reuse",
        )
        self.assertTrue(all(checks.values()), checks)
        self.assertTrue(method_state_contract("grace", summary, 15))

    def test_new_entry_requires_exact_increment(self):
        checks = grace_state_checks(
            grace_summary(edits=15, entries=15),
            15,
            prior_entry_count=14,
            insertion_action="collision_split",
        )
        self.assertTrue(all(checks.values()), checks)

    def test_reuse_cannot_hide_missing_request_mapping(self):
        summary = grace_summary(edits=14, entries=14)
        checks = grace_state_checks(
            summary,
            15,
            prior_entry_count=14,
            insertion_action="same_label_reuse",
        )
        self.assertFalse(checks["requested_mapping_count_exact"])
        self.assertFalse(method_state_contract("grace", summary, 15))

    def test_reuse_cannot_reference_nonresident_entry(self):
        summary = grace_summary(edits=15, entries=14)
        summary["requested_mapping_values_resident"] = False
        self.assertFalse(method_state_contract("grace", summary, 15))

    def test_other_method_contracts_are_unchanged(self):
        self.assertTrue(method_state_contract("balancedit", {"entry_count": 15}, 15))
        self.assertFalse(method_state_contract("balancedit", {"entry_count": 14}, 15))
        self.assertTrue(method_state_contract("belora", {"entry_count": 15}, 15))
        self.assertTrue(method_state_contract("lora", {"adapter_parameter_count": 1}, 15))
        self.assertFalse(method_state_contract("lora", {"adapter_parameter_count": 0}, 15))


if __name__ == "__main__":
    unittest.main()
