from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
from m3bench_repro.editors.routed_layers import RoutedFullLinear
from m3bench_repro.editors.routing import RouteDecision


class BalanceEditEffectContractTests(unittest.TestCase):
    def test_balanceedit_nonempty_generation_and_route_cleanup(self):
        editor = object.__new__(BalanceEditPaperSpecEditor)
        editor.wrapper = RoutedFullLinear(nn.Linear(2, 2, bias=False))
        editor.wrapper.add_edit("current")
        editor._route = lambda _record: RouteDecision(
            "current", "current", 0.0, 1.0, True, "euclidean"
        )
        editor.runtime = SimpleNamespace(
            generate=lambda _record, use_cache=True: {
                "decoded_text": "ok",
                "raw_token_ids": [7],
                "sequence_contract": "new_tokens_only",
                "use_cache": use_cache,
            }
        )
        result = editor.generate(object())
        self.assertTrue(result["generation"]["raw_token_ids"])
        self.assertIsNone(editor.wrapper.active_logical_id)


if __name__ == "__main__":
    unittest.main()
