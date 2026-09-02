import unittest
import torch.nn as nn

from m3bench_repro.editors.methods import GracePaperSpecEditor
from m3bench_repro.editors.routed_layers import GraceValueLinear
from m3bench_repro.editors.routing import RouteDecision
from scripts.editor_effect_probe import adapter_active_before_generation


class GenerationRouteContextTests(unittest.TestCase):
    def test_base_route_probe_may_precede_active_generation_prefill(self):
        self.assertTrue(
            adapter_active_before_generation(
                [
                    {"active_adapter": None},
                    {"active_adapter": "expected"},
                ],
                "expected",
            )
        )
        self.assertFalse(
            adapter_active_before_generation(
                [{"active_adapter": None}], "expected"
            )
        )

    def test_generation_route_is_fixed_and_exception_clears_it(self):
        editor = object.__new__(GracePaperSpecEditor)
        editor.wrapper = GraceValueLinear(nn.Linear(2, 2), replacement="replace_prompt")
        editor.wrapper.add_cold_value("current", seed=1)
        decision = RouteDecision("current", "current", 0.0, 1.0, True, "cosine")
        editor._route = lambda _record: decision
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with editor.route_generation(object()) as selected:
                self.assertEqual(selected, decision)
                self.assertEqual(editor.wrapper.active_logical_id, "current")
                raise RuntimeError("boom")
        self.assertIsNone(editor.wrapper.active_logical_id)


if __name__ == "__main__":
    unittest.main()
