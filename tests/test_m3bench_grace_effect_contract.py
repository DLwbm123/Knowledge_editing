import unittest

import torch
import torch.nn as nn

from m3bench_repro.editors.routed_layers import GraceValueLinear
from m3bench_repro.editors.routing import GraceCodebook


class GraceEffectContractTests(unittest.TestCase):
    def test_grace_exact_target_signature_and_self_route(self):
        codebook = GraceCodebook(distance="cosine", eps_init=0.2)
        codebook.insert_with_source_semantics("current", torch.tensor([2.0, 0.0]), [1, 3])
        self.assertFalse(codebook.source_label_match([1, 3], [2, 2]))
        decision = codebook.route(torch.tensor([8.0, 0.0]))
        self.assertTrue(decision.activated)
        self.assertEqual(decision.logical_edit_id, "current")
        self.assertEqual(decision.nearest_distance, 0.0)


    def test_grace_expanded_key_is_included_and_miss_is_exact_base(self):
        base = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(base.weight)
        wrapper = GraceValueLinear(base, replacement="replace_prompt")
        wrapper.add_cold_value("edit", seed=3).data.fill_(2.0)
        inputs = torch.ones((1, 3, 2))
        base_output = wrapper(inputs)
        wrapper.set_active("edit", token_index=2)
        self.assertTrue(torch.equal(wrapper(inputs), torch.full((1, 3, 2), 2.0)))
        wrapper.disable()
        self.assertTrue(torch.equal(wrapper(inputs), base_output))


if __name__ == "__main__":
    unittest.main()
