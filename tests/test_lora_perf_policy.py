import unittest

import torch

from m3bench_repro.editors.llava_runtime import target_nll_parts
from m3bench_repro.editors.perf_policy import calibration_allowed, new_method_code_allowed, new_method_gpu_allowed


class LoraPerfPolicyTests(unittest.TestCase):
    def test_old_semantic_failure_does_not_block_calibration(self):
        self.assertTrue(calibration_allowed(integration_pass=True, effect_active=True, v4_inputs_verified=True))

    def test_medtrace_gates_are_independent_of_formal_lora_completion(self):
        self.assertTrue(new_method_code_allowed(v4_inputs_verified=True, method_spec_resolved=True))
        self.assertTrue(new_method_gpu_allowed(lora_development_ready=True, zero_effect_tests_pass=True))

    def test_target_nll_splits_content_and_eos(self):
        logits = torch.zeros(1, 4, 5)
        labels = torch.tensor([[-100, 1, 2, 4]])
        parts = target_nll_parts(logits, labels, eos_token_id=4)
        self.assertEqual(parts["target_content_token_count"], 2)
        self.assertEqual(parts["eos_template_tail_token_count"], 1)


if __name__ == "__main__":
    unittest.main()
