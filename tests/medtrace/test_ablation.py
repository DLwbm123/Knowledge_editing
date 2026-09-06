import unittest

import torch

from m3bench_repro.editors.llava_runtime import expanded_image_span
from methods.medtrace import AsymmetricCPExpert
from scripts.medtrace.build_ablation_data import freeze_scope_roles
from scripts.medtrace.finalize_ablation import task_metrics
from scripts.medtrace.run_longrun_campaign import REPRESENTATIONS, allocate_hard, calibrate_operating_points, representation_score
from scripts.medtrace.run_generality_ablation import CONDITIONS, micro_plan
from scripts.medtrace.run_hard_scope_ablation import validate_eqkeys


def candidate(qid, relation, image):
    return {
        "source_qid": qid,
        "source_dataset": "SLAKE",
        "image_name": f"{image}/source.jpg",
        "image_path": f"/data/{image}/source.jpg",
        "question": f"q{qid}",
        "source_answer": f"a{qid}",
        "fact_relation": relation,
        "scope_relation_verification_status": "SOURCE_RELATION_VERIFIED",
    }


class AblationTests(unittest.TestCase):
    def test_paired_schedule_never_early_stops(self):
        native = [micro_plan(CONDITIONS[0], step) for step in range(1, 81)]
        augmented = [micro_plan(CONDITIONS[1], step) for step in range(1, 81)]
        self.assertEqual(native, [(0, 0)] * 80)
        self.assertEqual(len(augmented) * 2, 160)
        self.assertEqual([pair[1] for pair in augmented[:8]], [1, 2, 3, 4, 1, 2, 3, 4])

    def test_hard_roles_survive_final_selector(self):
        hard = [candidate(i, "same_question_different_image_conflicting_source_answer", f"h{i}") for i in range(13)]
        challenge = [candidate(100 + i, "same_image_other_source_fact", "primary") for i in range(12)]
        excluded = candidate(999, "same_image_other_source_fact", "primary")
        broad = [candidate(1000 + i, "broad_unrelated_source_qa", f"b{i}") for i in range(120)]
        census = {"primary": {"record_id": "p"}, "negative_candidates_full": hard + challenge + [excluded] + broad}
        old_roles = {
            "primary": {"record_id": "p"},
            "positives": {role: [{"question": f"{role}-{i}"} for i in range(4)] for role in ("fit", "calibration", "evaluation")},
            "negatives": {role: [] for role in ("fit", "calibration", "evaluation")},
        }
        review = {
            "same_question_different_image": {"approved_qids": list(range(13))},
            "same_image_challenge": {"approved_qids": list(range(100, 112))},
            "excluded": [{"qid": 999}],
        }
        frozen = freeze_scope_roles(census, old_roles, review)
        self.assertEqual(sum(row["fact_relation"].startswith("same_question") for row in frozen["negative_roles"]["mixed_fit"]), 4)
        self.assertEqual(sum(row["fact_relation"].startswith("same_question") for row in frozen["negative_roles"]["calibration"]), 4)
        self.assertEqual(sum(row["fact_relation"].startswith("same_question") for row in frozen["negative_roles"]["evaluation"]), 5)
        groups = {role: {row["image_name"] for row in frozen["negative_roles"][role]} for role in ("broad_fit_control", "calibration", "evaluation")}
        self.assertFalse(groups["broad_fit_control"] & groups["calibration"])
        self.assertFalse(groups["broad_fit_control"] & groups["evaluation"])

    def test_eqkey_roles_and_labels_are_locked(self):
        rows = [{"logical_id": "a", "role": "fit", "label": "positive"}, {"logical_id": "b", "role": "evaluation", "label": "negative"}]
        eq = [{"logical_id": "a", "role": "fit", "label": "positive", "eqkey": "1"}, {"logical_id": "b", "role": "evaluation", "label": "negative", "eqkey": "2"}]
        validate_eqkeys(rows, eq)
        eq[1]["eqkey"] = "1"
        with self.assertRaisesRegex(RuntimeError, "EqKey"):
            validate_eqkeys(rows, eq)

    def test_macro_and_micro_keep_edit_denominators(self):
        rows = [
            {"task": "T1L", "edit_id": "a", "exact": True, "semantic": True, "truncated_without_eos": False},
            {"task": "T1L", "edit_id": "a", "exact": False, "semantic": False, "truncated_without_eos": False},
            {"task": "T1L", "edit_id": "b", "exact": True, "semantic": True, "truncated_without_eos": False},
        ]
        rows += [{"task": task, "edit_id": "a", "exact": True, "semantic": True, "truncated_without_eos": False} for task in ("T0", "T1G", "T2G")]
        value = task_metrics(rows)["T1L"]
        self.assertEqual(value["eligible_edits"], 2)
        self.assertAlmostEqual(value["semantic_micro"], 2 / 3)
        self.assertAlmostEqual(value["semantic_macro"], 0.75)

    def test_realized_visual_span_and_route_scores(self):
        raw = torch.tensor([[0, 10, -200, 11, 12]])
        raw_attention = torch.tensor([[0, 1, 1, 1, 1]])
        expanded_attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1]])
        self.assertEqual(expanded_image_span(raw, raw_attention, expanded_attention, 7, image_token_index=-200), (2, 5))
        expert = AsymmetricCPExpert(12, 8, 4)
        prompt = torch.randn(3, 12)
        visual = [torch.randn(2, 12) for _ in range(3)]
        for representation in REPRESENTATIONS:
            scores, metadata = representation_score(expert, prompt, visual, representation, 2)
            self.assertEqual(scores.shape, (3,))
            self.assertEqual("visual_prototype" in metadata, representation == REPRESENTATIONS[2])

    def test_coverage_calibration_is_not_all_off(self):
        points = calibrate_operating_points([0.8, 0.9, 1.0, 1.1], [0.85, 0.95], [0.1, 0.2], hard_evaluable=True)
        self.assertGreaterEqual(points["COVERAGE_CONSTRAINED"]["positive_tpr"], 0.75)
        self.assertEqual(points["SAFETY_FIRST"]["broad_fpr"], 0)

    def test_sparse_hard_groups_are_spread_across_roles(self):
        rows = [candidate(i, "same_question_different_image_conflicting_source_answer", f"h{i}") for i in range(3)]
        self.assertEqual({role: len(values) for role, values in allocate_hard("p", rows).items()}, {"fit": 1, "calibration": 1, "evaluation": 1})


if __name__ == "__main__":
    unittest.main()
