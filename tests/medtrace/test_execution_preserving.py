import unittest

import torch
import torch.nn as nn

from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook
from scripts.medtrace.run_execution_preserving_longrun import CONDITIONS, condition_order, parameter_groups, route_loss


class ExecutionPreservingTests(unittest.TestCase):
    def test_anchor_is_zero_for_reference_and_positive_for_perturbation(self):
        torch.manual_seed(9)
        layer = nn.Linear(12, 8, bias=False)
        expert = AsymmetricCPExpert(12, 8, 4)
        expert.rho.data.normal_()
        activation = torch.randn(1, 5, 12)
        labels = torch.tensor([[-100, -100, 1, 2, 3]])
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, :-1] = labels[:, 1:] != -100
        reference = expert.residual(activation)[mask].detach()
        hook = MedTraceLayerHook(layer, expert)
        hook.attach()
        try:
            hook.set_teacher_routing(labels)
            hook.set_anchor_reference(reference)
            layer(activation)
            self.assertLess(float(hook.pop_anchor_loss()), 1e-8)
            hook.set_teacher_routing(labels)
            hook.set_anchor_reference(reference + 0.1)
            layer(activation)
            loss = hook.pop_anchor_loss()
            self.assertGreater(float(loss), 0)
            loss.backward()
            self.assertGreater(float(expert.rho.grad.abs().sum()), 0)
        finally:
            hook.detach()
        self.assertIsNone(hook.anchor_reference)

    def test_optimizer_boundaries_and_route_gradient(self):
        expert = AsymmetricCPExpert(12, 8, 4)
        _, c1 = parameter_groups(expert, CONDITIONS[0])
        self.assertEqual({id(value) for value in c1}, {id(expert.u_out), id(expert.v_out), id(expert.rho)})
        self.assertFalse(expert.u_in.requires_grad)
        _, c2 = parameter_groups(expert, CONDITIONS[1])
        self.assertEqual({id(value) for value in c2}, {id(value) for value in expert.parameters()})
        prompt = torch.randn(6, 12)
        visual = [torch.randn(3, 12) for _ in prompt]
        loss, _ = route_loss(expert, prompt, visual, 4)
        loss.backward()
        self.assertGreater(float(expert.u_in.grad.abs().sum()), 0)
        self.assertGreater(float(expert.v_in.grad.abs().sum()), 0)
        self.assertIsNone(expert.u_out.grad)

    def test_condition_rotation_is_complete_and_stable(self):
        first = condition_order(20260906, 1)
        self.assertEqual(first, condition_order(20260906, 1))
        self.assertEqual(set(first), set(CONDITIONS))
        self.assertEqual(len(first), 3)


if __name__ == "__main__":
    unittest.main()
