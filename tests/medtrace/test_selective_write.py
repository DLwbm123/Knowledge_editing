import copy
import unittest

import torch

from methods.medtrace.core import AsymmetricCPExpert, MedTraceLayerHook
from methods.medtrace.selective_write import (LowRankExpert, full_vocab_kl, predictor_mask,
    optimizer_for, fit_groups, balanced_schedule, Protection, CONDITIONS, select_global_lambda)


class SelectiveWriteTest(unittest.TestCase):
    def test_lossless_transfer_and_optimizer(self):
        torch.manual_seed(7)
        cp = AsymmetricCPExpert(12, 8, 4, beta=2.3)
        cp.rho.data.normal_()
        low = LowRankExpert(cp, 13)
        x = torch.randn(2, 9, 12)
        self.assertTrue(torch.allclose(cp.residual(x), low.residual(x), atol=1e-6, rtol=1e-5))
        self.assertEqual(torch.count_nonzero(low.B[:, 4:]), 0)
        self.assertTrue(torch.allclose(low.A[4:].norm(dim=1), torch.ones(12)))
        before = low.residual(x).detach()
        low.normalize_factors_()
        self.assertTrue(torch.allclose(before, low.residual(x), atol=1e-6, rtol=1e-5))
        backbone = torch.nn.Linear(12, 8).requires_grad_(False)
        opt = optimizer_for(low, backbone)
        self.assertEqual([g['lr'] for g in opt.param_groups], [1e-4,1e-3])
        self.assertEqual({id(p) for g in opt.param_groups for p in g['params']}, {id(p) for p in low.parameters()})
        with self.assertRaises(ValueError):
            optimizer_for(low, backbone.requires_grad_(True))

    def test_prefix_mask_full_kl_and_live_gradient(self):
        # Positions 0..3 are prompt/image, 4 first answer, 5 EOS, 6 padding.
        labels = torch.tensor([[-100,-100,-100,-100,9,2,-100]])
        attention = torch.tensor([[1,1,1,1,1,1,0]])
        mask = predictor_mask(labels, attention)
        self.assertEqual(mask.nonzero().tolist(), [[0,3],[0,4]])
        layer = torch.nn.Linear(12, 8).requires_grad_(False)
        cp = AsymmetricCPExpert(12,8,4)
        cp.rho.data.fill_(.3)
        hook = MedTraceLayerHook(layer, cp)
        hook.attach()
        x = torch.randn(1,7,12)
        teacher = layer(x).detach()[mask].float().log_softmax(-1)
        self.assertLess(abs(float(full_vocab_kl(layer(x)[mask],teacher))), 1e-6)
        hook.set_teacher_routing(labels)
        self.assertTrue(torch.equal(hook.token_mask, mask))
        loss = full_vocab_kl(layer(x)[mask], teacher, chunk=1)
        self.assertGreater(float(loss), 0)
        loss.backward()
        self.assertGreater(float(cp.rho.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in layer.parameters()))
        self.assertFalse(teacher.requires_grad)
        hook.clear_request_routing()
        self.assertLess(abs(float(full_vocab_kl(layer(x)[mask],teacher))), 1e-6)
        with hook.generation_request():
            layer(x)
        hook.detach()
        self.assertFalse(hook.enabled)
        self.assertIsNone(hook.token_mask)
        self.assertFalse(layer._forward_hooks)
        with self.assertRaises(ValueError):
            full_vocab_kl(torch.randn(2,8), teacher.requires_grad_(True))
        with self.assertRaises(FloatingPointError):
            full_vocab_kl(torch.full((2,8),float('nan')),teacher.detach())

    def test_roles_balance_and_dual(self):
        rows = [dict(role='fit',label='negative',fact_relation='broad_unrelated_source_qa',
                     source_group=str(i%2),eqkey=str(i),logical_id=str(i)) for i in range(6)]
        groups = fit_groups(rows, 'U')
        schedule = balanced_schedule(groups, 24, 11)
        self.assertEqual(sum(r['source_group']=='0' for r in schedule),12)
        self.assertEqual(schedule,balanced_schedule(groups,24,11))
        self.assertEqual({r['eqkey'] for r in schedule}, {r['eqkey'] for r in rows})
        for role in ('calibration','evaluation','challenge'):
            with self.assertRaises(ValueError):
                fit_groups([rows[0] | dict(role=role)],'U')
        protection = Protection(dict(H=.2,U=.1),CONDITIONS[-1])
        self.assertEqual(protection.ema,dict(H=1.,U=1.))
        self.assertEqual(protection.epsilon,dict(H=.1,U=.05))
        protection.update(dict(H=torch.tensor(.2,requires_grad=True),U=.1))
        self.assertAlmostEqual(protection.dual['H'],.525,6)
        with self.assertRaises(ValueError):
            protection.update(dict(H=.2,U=.1),role='evaluation')
        w1 = Protection(dict(H=.2,U=.1),CONDITIONS[1])
        self.assertAlmostEqual(w1.coefficient('H'),.25)
        self.assertEqual(Protection(dict(H=.2,U=.1),CONDITIONS[0]).coefficient('U'),0)
        frozen_gate = {'a':False,'b':True}
        previous = copy.deepcopy(frozen_gate)
        protection.update(dict(H=.1,U=.02))
        self.assertEqual(frozen_gate,previous)

    def test_global_calibration_selection(self):
        cal = [dict(edit=i,role='calibration') for i in range(7)]
        for row in cal:
            for c in ('A2',*CONDITIONS):
                row[c] = dict(positive_semantic=.8,U_kl=.1,H_kl=.2)
            row['W1_KL_1']['H_kl']=.1
            row['W1_KL_10']['positive_semantic']=.7
        self.assertEqual(select_global_lambda(cal)['lambda'],1.)
        self.assertTrue(select_global_lambda(cal)['qualified'])
        for row in cal:
            for c in CONDITIONS[1:4]:
                row[c]['positive_semantic']=.5
        self.assertFalse(select_global_lambda(cal)['qualified'])
        cal[0]['role']='evaluation'
        with self.assertRaises(ValueError):
            select_global_lambda(cal)


if __name__ == '__main__':
    unittest.main()
