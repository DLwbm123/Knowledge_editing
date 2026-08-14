from methods.liveedit_med.routing_attribution import attribute_route, failure_class


def test_visual_miss_attribution():
    route = attribute_route(target_index=0, visual_scores=[0.1, 0.4], sentinel_score=0.2,
                            candidate_mask=[False, True], text_scores=[1.0], absolute_weights=[0.7],
                            relative_weights=[1.0], final_weights=[0.7])
    assert route["delta_visual"] == -0.1
    assert not route["target_in_candidates"]
    assert failure_class(route, forced_success=True, routed_success=False,
                         target_residual_norm=0, fused_residual_norm=0) == "VISUAL_SENTINEL_RECALL_FAILURE"
