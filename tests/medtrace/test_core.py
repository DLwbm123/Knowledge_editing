import unittest

import torch
import torch.nn as nn

from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook, calibrate_threshold, factor_pair


class MedTraceCoreTests(unittest.TestCase):
    def test_factorized_dense_and_normalization(self):
        torch.manual_seed(1)
        expert = AsymmetricCPExpert(12, 8, 4)
        expert.rho.data.normal_()
        activation = torch.randn(2, 3, 12)
        expected = expert.normalize_activation(activation) @ expert.materialize_dense().T
        self.assertTrue(torch.allclose(expert.residual(activation), expected, atol=1e-6))
        before = expert.residual(activation)
        expert.normalize_factors_()
        self.assertTrue(torch.allclose(before, expert.residual(activation), atol=1e-5))
        restored = AsymmetricCPExpert(12, 8, 4)
        restored.load_state_dict(expert.state_dict())
        self.assertTrue(torch.equal(expert.residual(activation), restored.residual(activation)))

        factorized = AsymmetricCPExpert(12, 8, 4)
        factorized.rho.data.normal_()
        dense = AsymmetricCPExpert(12, 8, 4)
        dense.load_state_dict(factorized.state_dict())
        x1 = torch.randn(2, 3, 12, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_()
        factorized.residual(x1).square().sum().backward()
        (dense.normalize_activation(x2) @ dense.materialize_dense().T).square().sum().backward()
        self.assertTrue(torch.allclose(x1.grad, x2.grad, atol=1e-5))
        for left, right in zip(factorized.parameters(), dense.parameters(), strict=True):
            self.assertTrue(torch.allclose(left.grad, right.grad, atol=1e-5))

    def test_zero_effect_and_assistant_only_mask(self):
        layer = nn.Linear(12, 8, bias=False)
        expert = AsymmetricCPExpert(12, 8, 4)
        hook = MedTraceLayerHook(layer, expert)
        hook.attach()
        activation = torch.randn(1, 4, 12)
        base = layer(activation).detach()
        hook.set_request_routing(torch.tensor([[False, False, False, True]]))
        self.assertTrue(torch.equal(base, layer(activation)))
        expert.rho.data.fill_(0.1)
        edited = layer(activation)
        self.assertTrue(torch.equal(base[:, :3], edited[:, :3]))
        self.assertFalse(torch.equal(base[:, 3:], edited[:, 3:]))
        hook.clear_request_routing()
        self.assertTrue(torch.equal(base, layer(activation)))
        hook.set_teacher_routing(torch.tensor([[-100, -100, 5, 6]]))
        teacher = layer(activation)
        self.assertTrue(torch.equal(base[:, :1], teacher[:, :1]))
        self.assertFalse(torch.equal(base[:, 1:3], teacher[:, 1:3]))
        self.assertTrue(torch.equal(base[:, 3:], teacher[:, 3:]))
        hook.set_generation_routing()
        generated = layer(activation)
        self.assertTrue(torch.equal(base[:, :3], generated[:, :3]))
        self.assertFalse(torch.equal(base[:, 3:], generated[:, 3:]))
        hook.detach()
        restored = MedTraceLayerHook(layer, AsymmetricCPExpert(12, 8, 4))
        self.assertFalse(restored.enabled)
        self.assertIsNone(restored.token_mask)

    def test_zero_rho_gradient_and_optimizer_boundary(self):
        base = nn.Linear(12, 8, bias=False)
        expert = AsymmetricCPExpert(12, 8, 4)
        activation = torch.randn(1, 2, 12)
        expert.residual(activation).sum().backward()
        self.assertGreater(expert.rho.grad.abs().sum().item(), 0)
        self.assertEqual(expert.u_in.grad.abs().sum().item(), 0)
        optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3)
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(optimized, {id(parameter) for parameter in expert.parameters()})
        self.assertTrue(all(id(parameter) not in optimized for parameter in base.parameters()))

    def test_threshold_is_metadata_not_parameter(self):
        calibration = calibrate_threshold([0.8, 0.9], [0.1, 0.2, 0.3], target_fpr=0.0)
        self.assertEqual(calibration.false_positive_rate, 0.0)
        self.assertEqual(calibration.true_positive_rate, 1.0)
        self.assertEqual(factor_pair(14336), (112, 128))
        expert = AsymmetricCPExpert(12, 8, 4)
        self.assertNotIn("threshold", dict(expert.named_parameters()))


if __name__ == "__main__":
    unittest.main()
