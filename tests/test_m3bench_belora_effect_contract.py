import unittest

import torch
import torch.nn as nn

from m3bench_repro.editors.methods import BeloraPaperSpecEditor
from m3bench_repro.editors.routed_layers import RoutedLoRALinear, safe_slot


class BeloraEffectContractTests(unittest.TestCase):
    def test_belora_effect_gate_uses_first_non_noop_step_count(self):
        self.assertEqual(BeloraPaperSpecEditor.steps_per_edit, 50)

    def test_belora_stable_mapping_and_new_edit_preserves_old_adapter(self):
        editor = object.__new__(BeloraPaperSpecEditor)
        editor.wrappers = {
            "one": RoutedLoRALinear(nn.Linear(2, 2, bias=False), rank=1, alpha=1),
            "two": RoutedLoRALinear(nn.Linear(2, 2, bias=False), rank=1, alpha=1),
        }
        editor.edit_to_adapter = {"edit-a": safe_slot("edit-a")}
        for wrapper in editor.wrappers.values():
            a, b = wrapper.add_adapter(editor.edit_to_adapter["edit-a"], seed=1)
            a.data.fill_(1.0)
            b.data.fill_(2.0)
        before = editor.adapter_state_sha256("edit-a")
        editor.edit_to_adapter["edit-b"] = safe_slot("edit-b")
        for wrapper in editor.wrappers.values():
            wrapper.add_adapter(editor.edit_to_adapter["edit-b"], seed=2)
        self.assertEqual(len(set(editor.edit_to_adapter.values())), 2)
        self.assertEqual(editor.adapter_state_sha256("edit-a"), before)
        editor._set_active("edit-a")
        self.assertEqual(
            {wrapper.active_logical_id for wrapper in editor.wrappers.values()},
            {editor.edit_to_adapter["edit-a"]},
        )
        editor._set_active(None)


if __name__ == "__main__":
    unittest.main()
