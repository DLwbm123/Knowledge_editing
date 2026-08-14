from methods.liveedit_med.routing_attribution import attribute_route, failure_class
from methods.liveedit_med.posthoc_validation import plan_audit
from methods.liveedit_med.source_ops import BaseRoutePlan
import torch
import pytest


def test_visual_miss_attribution():
    route = attribute_route(target_index=0, visual_scores=[0.1, 0.4], sentinel_score=0.2,
                            candidate_mask=[False, True], text_scores=[1.0], absolute_weights=[0.7],
                            relative_weights=[1.0], final_weights=[0.7])
    assert route["delta_visual"] == -0.1
    assert not route["target_in_candidates"]
    assert failure_class(route, forced_success=True, routed_success=False,
                         target_residual_norm=0, fused_residual_norm=0) == "VISUAL_SENTINEL_RECALL_FAILURE"


def test_empty_candidate_plan_preserves_visual_evidence():
    plan = BaseRoutePlan(candidate_mask=torch.tensor([False]), visual_scores=torch.tensor([[0.1]]),
                         sentinel_score=torch.tensor([[0.2]]))
    audit = plan_audit(plan, ["target"])
    assert audit["visual_scores"] == [[pytest.approx(0.1)]]
    assert audit["sentinel_score"] == [[pytest.approx(0.2)]]
