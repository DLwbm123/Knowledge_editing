"""Pure contracts for LiveEdit-Med router-only domain adaptation R1.

The functions in this module deliberately do not load the model, the canonical
bank, record 953, or the sealed blind set.  They define the deterministic
repository construction, trainable boundary, and validation selection contract
used by the execution scripts.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn


PROTOCOL = "LIVEEDIT_MED_ROUTER_ONLY_DOMAIN_ADAPTATION_R1"
EXPECTED_BANK_HASH = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"
EXPECTED_BLIND_SELECTION_HASH = "67f54b61d07132eaceb940c8164b8d788a967c307fbcf870649dd99e7dcfd8ab"
EXPECTED_BLIND_SEALED_HASH = "da5a8db2c14fd879bf804fc9238382e4cdffc754303a930e2f4eef97f7b108e6"
REPOSITORY_SIZES = (1, 4, 8, 16, 32)
CHECKPOINT_STEPS = (80, 160, 240, 320, 400, 480, 560, 640)
POSITIVE_CATEGORIES = ("native", "textual", "visual", "paired")
NEGATIVE_CATEGORIES = (
    "same_image_different_question",
    "same_question_different_image",
    "visual_nearest",
    "text_nearest",
    "joint_near_miss",
)
TRAINABLE_PREFIXES = ("edit_extractor.", "input_extractor.")
FROZEN_PREFIXES = ("moegen_c.", "moegen_r.", "instant_reps_norm.")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def stable_key(*values: Any) -> str:
    return canonical_hash([str(value) for value in values])


def repository_size(step: int) -> int:
    """Return the fixed 1,4,8,16,32 cycle for one-indexed optimizer steps."""
    if step < 1:
        raise ValueError("step must be one-indexed")
    return REPOSITORY_SIZES[(step - 1) % len(REPOSITORY_SIZES)]


def semantic_category(step: int) -> str:
    if step < 1:
        raise ValueError("step must be one-indexed")
    return ("textual", "visual", "paired")[(step - 1) % 3]


def negative_category(step: int) -> str:
    if step < 1:
        raise ValueError("step must be one-indexed")
    return NEGATIVE_CATEGORIES[(step - 1) % len(NEGATIVE_CATEGORIES)]


def configure_router_only(modules: nn.Module) -> tuple[str, ...]:
    """Freeze the source module container, then enable only the two extractors."""
    for parameter in modules.parameters():
        parameter.requires_grad_(False)
    names = []
    for name, parameter in modules.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
            names.append(name)
    assert_router_only(modules)
    return tuple(names)


def assert_router_only(modules: nn.Module) -> None:
    trainable = {name for name, parameter in modules.named_parameters() if parameter.requires_grad}
    if not trainable or any(not name.startswith(TRAINABLE_PREFIXES) for name in trainable):
        raise RuntimeError("ROUTER_R1_UNAUTHORIZED_TRAINABLE_PARAMETER")
    frozen = {name for name, _parameter in modules.named_parameters() if name.startswith(FROZEN_PREFIXES)}
    if not frozen or trainable & frozen:
        raise RuntimeError("ROUTER_R1_UNAUTHORIZED_TRAINABLE_PARAMETER")


def router_state(modules: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value for name, value in modules.state_dict().items()
            if name.startswith(TRAINABLE_PREFIXES)}


def frozen_module_state(modules: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value for name, value in modules.state_dict().items()
            if name.startswith(FROZEN_PREFIXES)}


def deterministic_repository(target_id: str, size: int,
                             nearest: Mapping[str, Mapping[str, Sequence[str]]],
                             all_ids: Sequence[str]) -> list[str]:
    """Target + visual, text, joint, then stable-hash distractors."""
    if size not in REPOSITORY_SIZES:
        raise ValueError(f"unsupported repository size: {size}")
    target_id = str(target_id)
    ordered = [target_id]
    row = nearest[target_id]
    for category in ("visual", "text", "joint"):
        for candidate in row.get(category, ()):
            candidate = str(candidate)
            if candidate != target_id and candidate not in ordered:
                ordered.append(candidate)
            if len(ordered) == size:
                return ordered
    remaining = sorted((str(value) for value in all_ids if str(value) not in ordered),
                       key=lambda value: (stable_key(target_id, value), value))
    ordered.extend(remaining[: max(0, size - len(ordered))])
    if len(ordered) != size or ordered[0] != target_id or len(set(ordered)) != size:
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN")
    return ordered


def validation_eligibility(metrics: Mapping[str, Any], forced: Mapping[str, int]) -> bool:
    return (
        int(metrics["routed_native"]) >= math.ceil(.90 * int(forced["native"]))
        and int(metrics["routed_textual"]) >= math.ceil(.75 * int(forced["textual"]))
        and int(metrics["routed_visual"]) >= math.ceil(.75 * int(forced["visual"]))
        and int(metrics["routed_paired"]) >= math.ceil(.75 * int(forced["paired"]))
        and int(metrics["target_contamination"]) == 0
        and int(metrics["clinical_canonical_failures"]) == 0
    )


def selection_key(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    """Ascending Python key implementing the required lexicographic ranking."""
    return (
        -int(metrics["negative_locality_exact_s0"]),
        -(int(metrics["routed_visual"]) + int(metrics["routed_paired"])),
        -(int(metrics["routed_native"]) + int(metrics["routed_textual"])),
        float(metrics["mean_candidate_count"]),
        int(metrics["text_relative_competition_failures"]),
        float(metrics["negative_locality_kl"]),
        int(metrics["step"]),
    )


def select_checkpoint(rows: Sequence[Mapping[str, Any]], forced: Mapping[str, int]) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if validation_eligibility(row, forced)]
    return min(eligible, key=selection_key) if eligible else None


def positive_visual_loss(input_visual: torch.Tensor, edit_visual: torch.Tensor,
                         sentinel_visual: torch.Tensor, target_index: int) -> torch.Tensor:
    from .source_ops import SIM_SCALE
    scores = torch.einsum("bed,med->bme", input_visual, edit_visual).mean(2) * SIM_SCALE
    sentinel = torch.einsum("bed,bed->be", input_visual, sentinel_visual).mean(1, True) * SIM_SCALE
    logits = torch.cat([scores, sentinel], 1)
    return torch.nn.functional.cross_entropy(logits, torch.tensor([target_index], device=logits.device))


def negative_visual_loss(input_visual: torch.Tensor, edit_visual: torch.Tensor,
                         sentinel_visual: torch.Tensor) -> torch.Tensor:
    from .source_ops import SIM_SCALE
    scores = torch.einsum("bed,med->bme", input_visual, edit_visual).mean(2) * SIM_SCALE
    sentinel = torch.einsum("bed,bed->be", input_visual, sentinel_visual).mean(1, True) * SIM_SCALE
    logits = torch.cat([scores, sentinel], 1)
    return torch.nn.functional.cross_entropy(logits, torch.tensor([edit_visual.shape[0]], device=logits.device))


def positive_text_losses(input_text: torch.Tensor, edit_text: torch.Tensor,
                         target_index: int, locality_key: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    from .source_ops import SIM_SCALE
    scores = torch.einsum("ned,med->nme", input_text, edit_text).mean(2) * SIM_SCALE
    absolute = torch.nn.functional.binary_cross_entropy_with_logits(
        scores[:, target_index], torch.ones_like(scores[:, target_index]))
    relative_scores = scores
    if locality_key is not None:
        extra = torch.einsum("ned,med->nme", input_text, locality_key).mean(2) * SIM_SCALE
        relative_scores = torch.cat([scores, extra], 1)
    relative = torch.nn.functional.cross_entropy(
        relative_scores, torch.tensor([target_index], device=scores.device))
    return absolute, relative


def negative_text_absolute_loss(input_text: torch.Tensor, edit_text: torch.Tensor) -> torch.Tensor:
    from .source_ops import SIM_SCALE
    scores = torch.einsum("ned,med->nme", input_text, edit_text).mean(2) * SIM_SCALE
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, torch.zeros_like(scores))

