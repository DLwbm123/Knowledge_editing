"""Pure routing attribution and repository-size helpers."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence


def stable_repository(records: Sequence[Mapping[str, Any]], target_id: str, size: int) -> list[Mapping[str, Any]]:
    if size < 1 or size > len(records):
        raise ValueError("LIVEEDIT_MED_INVALID_REPOSITORY_SIZE")
    by_id = {str(row["record_id"]): row for row in records}
    if target_id not in by_id:
        raise KeyError(target_id)
    distractors = [row for row in records if str(row["record_id"]) != target_id]
    distractors.sort(key=lambda row: (hashlib.sha256(f"{target_id}||{row['record_id']}||{row.get('selection_hash','')}".encode()).hexdigest(), str(row["record_id"])))
    return [by_id[target_id], *distractors[: size - 1]]


def attribute_route(*, target_index: int, visual_scores: Sequence[float], sentinel_score: float,
                    candidate_mask: Sequence[bool], text_scores: Sequence[float],
                    absolute_weights: Sequence[float], relative_weights: Sequence[float],
                    final_weights: Sequence[float]) -> dict[str, Any]:
    if not (0 <= target_index < len(visual_scores)):
        raise ValueError("LIVEEDIT_MED_TARGET_INDEX_OUT_OF_RANGE")
    candidates = [index for index, selected in enumerate(candidate_mask) if selected]
    target_in = target_index in candidates
    rank = 1 + sum(float(score) > float(visual_scores[target_index]) for score in visual_scores)
    if not target_in:
        return {
            "delta_visual": float(visual_scores[target_index] - sentinel_score), "target_visual_rank": rank,
            "target_in_candidates": False, "candidate_count": len(candidates), "text_raw_score": None,
            "sigmoid_absolute_weight": 0.0, "softmax_relative_weight": 0.0, "final_weight": 0.0,
            "max_distractor_weight": max([0.0, *map(float, final_weights)]),
            "total_distractor_weight": float(sum(map(float, final_weights))),
        }
    position = candidates.index(target_index)
    distractors = [float(value) for index, value in enumerate(final_weights) if index != position]
    return {
        "delta_visual": float(visual_scores[target_index] - sentinel_score), "target_visual_rank": rank,
        "target_in_candidates": True, "candidate_count": len(candidates), "text_raw_score": float(text_scores[position]),
        "sigmoid_absolute_weight": float(absolute_weights[position]), "softmax_relative_weight": float(relative_weights[position]),
        "final_weight": float(final_weights[position]), "max_distractor_weight": max([0.0, *distractors]),
        "total_distractor_weight": float(sum(distractors)),
    }


def failure_class(route: Mapping[str, Any], *, forced_success: bool, routed_success: bool,
                  target_residual_norm: float, fused_residual_norm: float) -> str:
    if routed_success:
        return "SUCCESS"
    if not forced_success:
        return "GENERATOR_OR_EXPERT_FAILURE"
    if not route["target_in_candidates"]:
        return "VISUAL_SENTINEL_RECALL_FAILURE"
    if float(route["sigmoid_absolute_weight"]) < 0.5:
        return "TEXT_ABSOLUTE_SUPPRESSION"
    if float(route["softmax_relative_weight"]) < 0.5:
        return "TEXT_RELATIVE_COMPETITION"
    if fused_residual_norm and target_residual_norm / fused_residual_norm < 0.5:
        return "RESIDUAL_INTERFERENCE"
    return "ROUTED_GENERATION_FAILURE_UNRESOLVED"
