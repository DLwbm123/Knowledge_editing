import unittest

import torch
from types import SimpleNamespace

from methods.medtrace import AsymmetricCPExpert
from methods.medtrace.frozen_verifier import (
    LinearApplicabilityVerifier,
    calibrate_decisions,
    conjunction_decision,
    mean_decision,
    train_verifier,
    verifier_features,
)
from scripts.medtrace.run_frozen_expert_visual_verifier import _hierarchical_by_edit, _score_rows, _training_data, matched_panel_rows
from scripts.medtrace.run_longrun_campaign import REPRESENTATIONS, score_with_frozen_prototypes


class FrozenVerifierTests(unittest.TestCase):
    def test_cp_and_pre_cp_features_share_q_and_pooling(self):
        torch.manual_seed(4)
        expert = AsymmetricCPExpert(12, 8, 4)
        prompt, visual = torch.randn(12), torch.randn(5, 12)
        features = verifier_features(expert, prompt, visual)
        q = expert.input_basis()
        self.assertTrue(torch.allclose(features.cp_prompt, torch.nn.functional.normalize(features.pre_cp_prompt @ q, dim=0), atol=1e-5))
        self.assertTrue(torch.allclose(features.cp_visual, torch.nn.functional.normalize(features.pre_cp_visual @ q, dim=0), atol=1e-5))
        self.assertAlmostEqual(float(features.pooling_weights.sum()), 1.0, places=6)

    def test_verifier_training_does_not_change_frozen_expert(self):
        torch.manual_seed(5)
        expert = AsymmetricCPExpert(12, 8, 4)
        before = {name: value.clone() for name, value in expert.state_dict().items()}
        question = torch.tensor([[1.0, 0, 0, 0], [0.8, 0, 0, 0], [-1.0, 0, 0, 0], [-0.8, 0, 0, 0]])
        image = torch.tensor([[0, 1.0, 0, 0], [0, 0.8, 0, 0], [0, -1.0, 0, 0], [0, -0.8, 0, 0]])
        verifier, curve = train_verifier(
            question_features=question, question_labels=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
            question_groups=["p1", "p2", "n1", "n2"], image_features=image,
            image_labels=torch.tensor([1, 1, 0, 0], dtype=torch.bool), image_groups=["p1", "p2", "n1", "n2"],
            matched_pairs=[(0, 2), (1, 3)],
        )
        self.assertTrue(all(torch.equal(before[name], value) for name, value in expert.state_dict().items()))
        self.assertLess(curve[-1]["loss"], curve[0]["loss"])
        batch = verifier.decisions(question, image, 0, 0)
        single = torch.stack([verifier.decisions(q[None], v[None], 0, 0)[0] for q, v in zip(question, image, strict=True)])
        self.assertTrue(torch.equal(batch, single))

    def test_and_rule_is_not_the_mean_rule(self):
        prompt = torch.tensor([0.9, 0.4])
        visual = torch.tensor([0.1, 0.6])
        self.assertEqual(mean_decision(prompt, visual, 0.45).tolist(), [True, True])
        self.assertEqual(conjunction_decision(prompt, visual, 0.5, 0.5).tolist(), [False, False])

    def test_m0_control_uses_exact_legacy_score_path(self):
        torch.manual_seed(7)
        expert = AsymmetricCPExpert(12, 8, 4)
        prompt, visual = torch.randn(12), torch.randn(5, 12)
        features = {"row": verifier_features(expert, prompt, visual)}
        prototypes = {
            "prompt_prototype": torch.nn.functional.normalize(torch.randn(4), dim=0),
            "visual_prototype": torch.nn.functional.normalize(torch.randn(4), dim=0),
        }
        checkpoint = {"representation": REPRESENTATIONS[2], "prototypes": prototypes}
        cache = {"values": {"row": {"prompt": prompt, "visual": visual}}}
        verifiers = {"M2": LinearApplicabilityVerifier(4), "M3": LinearApplicabilityVerifier(12)}
        actual = _score_rows(expert, checkpoint, cache, features, verifiers)["row"]["M0_C1_R2_MEAN_CONTROL"][0]
        expected = score_with_frozen_prototypes(expert, prompt[None], [visual], REPRESENTATIONS[2], prototypes)[0].item()
        self.assertAlmostEqual(actual, expected, places=6)

    def test_calibration_is_deterministic_and_coverage90_means_four_of_four(self):
        positive = [(0.9, 0.9), (0.8, 0.8), (0.7, 0.7), (0.6, 0.6)]
        hard = [(0.9, 0.1), (0.1, 0.9)]
        broad = [(0.2, 0.2), (0.1, 0.3)]
        first = calibrate_decisions(positive, hard, broad)
        self.assertEqual(first, calibrate_decisions(positive, hard, broad))
        self.assertEqual(first["SECONDARY_COVERAGE90"]["positive_tpr"], 1.0)
        self.assertEqual(first["PRIMARY_SAFETY_FIRST"]["hard_fpr"], 0.0)
        self.assertEqual(first["PRIMARY_SAFETY_FIRST"]["broad_fpr"], 0.0)

    def test_invalid_calibration_and_budget_fail_closed(self):
        with self.assertRaises(ValueError):
            calibrate_decisions([], [(0.0,)], [(0.0,)])
        with self.assertRaises(ValueError):
            train_verifier(
                question_features=torch.randn(4, 4), question_labels=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
                question_groups=["a", "b", "c", "d"], image_features=torch.randn(4, 4),
                image_labels=torch.tensor([1, 1, 0, 0], dtype=torch.bool), image_groups=["a", "b", "c", "d"],
                matched_pairs=[(0, 2)], steps=10,
            )

    def test_matched_panel_preserves_role_and_identical_question(self):
        hard = lambda role: {"fact_relation": "same_question_different_image_conflicting_source_answer", "source_qid": role, "source_answer": "negative", "image_path": f"/{role}.png", "image_name": f"{role}.png"}
        scope = {
            "status": "HARD_EVALUABLE",
            "primary": {"target": "positive", "image_path": "/native.png", "image_name": "native.png"},
            "positives": {role: [{"family": role, "question": f"question-{role}", "review_status": "APPROVED_EQUIVALENT_SOURCE_QUESTION_ONLY"}] for role in ("fit", "calibration", "evaluation")},
            "negative_roles": {role: [hard(role)] for role in ("fit", "calibration", "evaluation")},
        }
        rows, exclusions = matched_panel_rows(scope, {"edit_record": {"question": "native-question"}})
        self.assertFalse(exclusions)
        for negative in (row for row in rows if row["label"] == "negative"):
            positive = next(row for row in rows if row["logical_id"] == negative["positive_logical_id"])
            self.assertEqual((negative["role"], negative["question"]), (positive["role"], positive["question"]))
        self.assertEqual(sum(row["role"] == "fit" and row["label"] == "positive" for row in rows), 2)

    def test_image_head_uses_only_matched_fit_pairs(self):
        feature = lambda value: SimpleNamespace(cp_prompt=value, cp_visual=value, pre_cp_prompt=value, pre_cp_visual=value)
        matched = {"values": {
            "p": {"row": {"logical_id": "p", "role": "fit", "label": "positive", "source_group": "native", "positive_logical_id": "p"}},
            "n": {"row": {"logical_id": "n", "role": "fit", "label": "negative", "source_group": "hard", "positive_logical_id": "p"}},
        }}
        original = {"values": {
            "b": {"row": {"logical_id": "b", "role": "fit", "fact_relation": "broad_unrelated_source_qa", "image_path": "/b"}},
            "c": {"row": {"logical_id": "c", "role": "challenge", "fact_relation": "same_image_other_source_fact", "image_path": "/c"}},
        }}
        values = {"p": feature(torch.ones(4)), "n": feature(-torch.ones(4))}
        old = {"b": feature(-torch.ones(4)), "c": feature(-torch.ones(4))}
        data = _training_data(matched, original, values, old, "M2")
        self.assertEqual(data["image_features"].shape[0], 2)
        self.assertEqual(data["question_features"].shape[0], 3)
        positive, negative = data["matched_pairs"][0]
        self.assertTrue(data["image_labels"][positive])
        self.assertFalse(data["image_labels"][negative])

    def test_hierarchical_metric_does_not_count_rewrites_as_facts(self):
        base = [
            {"event_index": 1, "seed": 1, "source_group": "a", "on": True},
            {"event_index": 1, "seed": 1, "source_group": "b", "on": False},
            {"event_index": 1, "seed": 2, "source_group": "a", "on": True},
            {"event_index": 1, "seed": 2, "source_group": "b", "on": False},
        ]
        duplicated = [*base, *[dict(row) for row in base if row["source_group"] == "a"]]
        self.assertEqual(_hierarchical_by_edit(base, "on"), _hierarchical_by_edit(duplicated, "on"))


if __name__ == "__main__":
    unittest.main()
