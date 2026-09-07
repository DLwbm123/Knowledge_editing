import unittest
from unittest.mock import patch
import torch

from methods.medtrace.core import AsymmetricCPExpert
from methods.medtrace.frozen_verifier import verifier_features
from methods.medtrace.visual_pooling import GROUPS, pooled_feature, fit_image_head
from scripts.medtrace import run_frozen_expert_visual_verifier as vf


class PoolingTests(unittest.TestCase):
    def test_pooling_and_frozen_fit_contract(self):
        torch.manual_seed(9)
        expert = AsymmetricCPExpert(12, 8, 4).requires_grad_(False)
        before = vf.state_hash(expert.state_dict())
        prompt, visual = torch.randn(12), torch.randn(5,12)
        self.assertTrue(torch.equal(pooled_feature(expert,prompt,visual,"G0"),verifier_features(expert,prompt,visual).pre_cp_visual))
        for kind in GROUPS:
            value = pooled_feature(expert,prompt,visual,kind)
            self.assertEqual(value.shape,(12,)); self.assertAlmostEqual(value.norm().item(),1,places=5)
        with patch.object(expert,"input_basis",side_effect=AssertionError("Q forbidden")):
            for kind in ("G1","G2"):
                pooled_feature(expert,prompt,visual,kind)
        rows = [dict(logical_id="a",role="fit",label="positive",source_group="p",positive_logical_id="a"),
                dict(logical_id="b",role="fit",label="negative",source_group="n",positive_logical_id="a")]
        question = dict(weight=torch.randn(1,12),bias=torch.randn(1))
        x = torch.randn(2,12)
        model, curve = fit_image_head(question,x,rows)
        self.assertEqual(curve[-1]["step"],800)
        self.assertEqual(before,vf.state_hash(expert.state_dict()))
        self.assertEqual(vf.state_hash(question),vf.state_hash(model.question.state_dict()))
        self.assertFalse(model.question.weight.requires_grad)
        self.assertTrue(torch.allclose(model.image(x)[0],model.image(x.flip(0))[1],atol=1e-6))
        with self.assertRaisesRegex(ValueError,"fit-only"):
            fit_image_head(question,x,[rows[0],rows[1]|{"role":"evaluation"}])
        outputs={"a":{"base":{"raw_token_ids":[1],"raw_answer":"base"},"forced":{"raw_token_ids":[2],"raw_answer":"forced"}}}
        for value in (-1,1):
            scores={"a":{g:[value,value] for g in GROUPS}}
            cal={g:{vf.POINTS[0]:{"thresholds":[0,0]}} for g in GROUPS}
            replay=vf.natural_replays([{"logical_id":"a"}],scores,cal,outputs,lambda r,on:outputs["a"]["forced" if on else "base"],GROUPS)
            self.assertEqual(sum(r["exact_replay"] is True for r in replay),3)
            self.assertEqual(sum(r["status"].startswith("NO_NATURAL") for r in replay),3)


if __name__ == "__main__": unittest.main()
