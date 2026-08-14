"""Leakage-safe helpers for post-hoc LiveEdit-Med checkpoint validation."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .source_ops import BaseRoutePlan, RoutePlan, apply_low_rank_expert_residual, route_repository


CHECKPOINT_STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3200)
PROTOCOL = "POSTHOC_VALIDATION_RECOVERY__NO_TEST_LEAKAGE"


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_validation_panel(source_records: Mapping[str, Any], count: int = 8) -> dict[str, Any]:
    rows = list(source_records["records"]["validation"])
    ordered = sorted(rows, key=lambda row: (str(row["selection_hash"]), str(row["record_id"])))
    selected = ordered[:count]
    forbidden = {"953"}
    if len(selected) != count or any(str(row["record_id"]) in forbidden for row in selected):
        raise RuntimeError("LIVEEDIT_MED_VALIDATION_PANEL_LEAKAGE")
    if selected != rows[:count]:
        raise RuntimeError("LIVEEDIT_MED_FROZEN_SPLIT_ORDER_DRIFT")
    compact = [{"panel_index": index, "record_id": str(row["record_id"]),
                "selection_hash": str(row["selection_hash"])} for index, row in enumerate(selected)]
    panel = {"protocol": PROTOCOL, "selection_rule": "first_eight_validation_edits_by_existing_stable_hash_order",
             "record953_excluded": True, "count": count, "edits": compact}
    panel["panel_hash"] = canonical_json_hash(panel)
    return panel


def verify_checkpoint_set(run_dir: Path) -> dict[str, Any]:
    from .serialization import load_safe_state
    rows = []
    for step in CHECKPOINT_STEPS:
        directory = run_dir / "training" / f"checkpoint_{step:04d}"
        state, manifest = load_safe_state(directory)
        if int(manifest.get("step", -1)) != step or manifest.get("stage") != "S" or not manifest.get("source_objective"):
            raise RuntimeError(f"LIVEEDIT_MED_CHECKPOINT_MANIFEST_MISMATCH:{step}")
        rows.append({"step": step, "directory": str(directory),
                     "model_sha256": file_sha256(directory / "model.safetensors"),
                     "manifest_sha256": file_sha256(directory / "manifest.json"),
                     "tensor_count": len(state), "tensor_hashes_verified": True})
    return {"protocol": PROTOCOL, "checkpoint_count": len(rows), "checkpoints": rows,
            "set_hash": canonical_json_hash(rows)}


def immutable_tree_manifest(directory: Path) -> dict[str, Any]:
    rows = [{"path": str(path.relative_to(directory)), "size": path.stat().st_size,
             "sha256": file_sha256(path)} for path in sorted(item for item in directory.rglob("*") if item.is_file())]
    return {"directory": str(directory), "files": rows, "tree_hash": canonical_json_hash(rows)}


def sample_to_model_row(sample: Mapping[str, Any]) -> dict[str, list[Any]]:
    return {"image_path": [sample.get("image")], "prompt": [str(sample["prompt"])],
            "target": [str(sample["target"])]}


def native_sample(record: Mapping[str, Any]) -> dict[str, Any]:
    row = record["requests"][0]
    return {"image": row["image"], "prompt": row["prompt"], "target": row["target_new"]}


def normalize_answer(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    return re.sub(r"^(the answer is|answer is|answer|it is)\s+", "", text).strip()


def unrestricted_match(output: str, target: str, *, eos: bool, cap_hit: bool) -> dict[str, Any]:
    actual, wanted = normalize_answer(output), normalize_answer(target)
    aw, ww = actual.split(), wanted.split()
    span = bool(ww) and any(aw[index:index + len(ww)] == ww for index in range(max(0, len(aw) - len(ww) + 1)))
    contradiction = bool(wanted and (f"not {wanted}" in actual or f"no {wanted}" in actual))
    return {"normalized_output": actual, "normalized_target": wanted, "target_span": span,
            "contradiction": contradiction, "eos_normal": bool(eos and not cap_hit),
            "success": bool(span and not contradiction and eos and not cap_hit)}


def plan_audit(plan: BaseRoutePlan | RoutePlan, expert_ids: Sequence[str]) -> dict[str, Any]:
    if isinstance(plan, BaseRoutePlan):
        mask = plan.candidate_mask.tolist() if plan.candidate_mask is not None else [False] * len(expert_ids)
        visual = plan.visual_scores.detach().float().cpu().tolist() if plan.visual_scores is not None else []
        sentinel = plan.sentinel_score.detach().float().cpu().tolist() if plan.sentinel_score is not None else None
        return {"kind": "base", "reason": plan.reason, "candidate_mask": mask,
                "candidate_ids": [rid for rid, selected in zip(expert_ids, mask) if selected],
                "visual_scores": visual, "sentinel_score": sentinel, "raw_text_scores": [],
                "sigmoid_weights": [], "softmax_weights": [], "final_weights": [], "sum_final_weights": 0.0}
    mask = [bool(value) for value in plan.candidate_mask.tolist()]
    return {"kind": "routed", "candidate_mask": mask,
            "candidate_ids": [rid for rid, selected in zip(expert_ids, mask) if selected],
            "visual_scores": plan.visual_scores.detach().float().cpu().tolist(),
            "sentinel_score": plan.sentinel_score.detach().float().cpu().tolist(),
            "raw_text_scores": plan.text_scores.detach().float().cpu().tolist(),
            "sigmoid_weights": plan.absolute_weights.detach().float().cpu().tolist(),
            "softmax_weights": plan.relative_weights.detach().float().cpu().tolist(),
            "final_weights": plan.final_weights.detach().float().cpu().tolist(),
            "sum_final_weights": float(plan.final_weights.sum().item())}


def route_residual(plan: BaseRoutePlan | RoutePlan, hidden: torch.Tensor, moe_cs: torch.Tensor,
                   moe_rs: torch.Tensor, instant_norm: torch.nn.Module) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(plan, BaseRoutePlan):
        return torch.zeros_like(hidden), {"per_expert_residual_norms": [], "fused_residual_norm": 0.0}
    selected_cs, selected_rs = moe_cs[plan.candidate_mask], moe_rs[plan.candidate_mask]
    residual = apply_low_rank_expert_residual(hidden.float(), selected_cs, selected_rs,
                                              plan.final_weights, instant_norm).to(hidden.dtype)
    per = []
    for index in range(selected_cs.shape[0]):
        one = apply_low_rank_expert_residual(hidden.float(), selected_cs[index:index+1],
                                             selected_rs[index:index+1],
                                             plan.final_weights[:, index:index+1], instant_norm)
        per.append(float(one.norm().item()))
    return residual, {"per_expert_residual_norms": per, "fused_residual_norm": float(residual.float().norm().item())}


def checkpoint_score(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (int(row["routed_native_success_count"]), int(row["routed_generality_success_count"]),
            int(row["locality_exact_preservation_count"]), -int(row["routing_false_positive_count"]),
            -int(row["target_contamination_count"]), int(row["forced_native_success_count"]),
            -float(row["validation_source_loss"]), -int(row["step"]))


def select_checkpoint(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if {int(row["step"]) for row in rows} != set(CHECKPOINT_STEPS):
        raise RuntimeError("LIVEEDIT_MED_INCOMPLETE_VALIDATION_SET")
    routed = max(int(row["routed_native_success_count"]) for row in rows)
    forced = max(int(row["forced_native_success_count"]) for row in rows)
    if forced == 0:
        return {"status": "STOP", "label": "LIVEEDIT_SHARED_GENERATOR_NO_NATURAL_GENERATION_ON_VALIDATION",
                "selected_step": None, "stage_f_permitted": False}
    if routed == 0:
        best = max(rows, key=lambda row: (int(row["forced_native_success_count"]),
                    int(row["forced_generality_success_count"]), int(row["locality_exact_preservation_count"]),
                    -float(row["validation_source_loss"]), -int(row["step"])))
        return {"status": "SELECTED_FOR_STAGE_F_ONLY", "label": "GENERATOR_CAPABLE__ROUTER_UNDERFIT",
                "selected_step": int(best["step"]), "stage_f_permitted": True}
    best = max(rows, key=checkpoint_score)
    return {"status": "SELECTED", "label": "LIVEEDIT_POSTHOC_VALIDATION_CHECKPOINT_SELECTED",
            "selected_step": int(best["step"]), "stage_f_permitted": True}


def assert_no_record953_environment() -> None:
    for name, value in os.environ.items():
        if "953" in name or "953" in value:
            raise RuntimeError("LIVEEDIT_MED_RECORD953_ENVIRONMENT_LEAKAGE")
