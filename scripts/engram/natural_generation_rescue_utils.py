"""Pure helpers for the ENGRAM V2 one-shot natural-generation rescue."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer, tensor_sha256
from scripts.engram.stage1_behavioral_margin_utils import (
    clip_global_relative_displacement,
    relative_parameter_displacement,
    shifted_boundary_training_tensors,
)


@dataclass(frozen=True)
class ScaffoldAlignment:
    start: int
    end: int
    method: str
    coverage: float


def _starts(source: Sequence[Any], target: Sequence[Any]) -> list[int]:
    left, right = list(source), list(target)
    return [index for index in range(max(0, len(left) - len(right) + 1)) if left[index : index + len(right)] == right]


def _normalized_piece(tokenizer: Any, token_id: int) -> str:
    text = unicodedata.normalize("NFKC", tokenizer.decode([int(token_id)], skip_special_tokens=True)).lower()
    text = re.sub(r"[^\w]+", "", text)
    return "" if text in {"", "a", "an", "the"} else text


def align_model_short_to_unrestricted(
    unrestricted_ids: Sequence[int],
    short_ids: Sequence[int],
    *,
    tokenizer: Any,
    stop_ids: Sequence[int] = (),
    minimum_content_tokens: int = 2,
    minimum_coverage: float = 0.60,
) -> ScaffoldAlignment | None:
    stops = set(map(int, stop_ids))
    unrestricted = [int(item) for item in unrestricted_ids if int(item) not in stops]
    short = [int(item) for item in short_ids if int(item) not in stops]
    exact = _starts(unrestricted, short)
    if len(exact) == 1:
        return ScaffoldAlignment(exact[0], exact[0] + len(short), "exact_token", 1.0)
    if len(exact) > 1:
        return None

    u_content = [(index, _normalized_piece(tokenizer, token_id)) for index, token_id in enumerate(unrestricted)]
    s_content = [(index, _normalized_piece(tokenizer, token_id)) for index, token_id in enumerate(short)]
    u_content = [(index, value) for index, value in u_content if value]
    s_content = [(index, value) for index, value in s_content if value]
    if not s_content:
        return None
    u_values, s_values = [value for _, value in u_content], [value for _, value in s_content]
    normalized_starts = _starts(u_values, s_values)
    if len(normalized_starts) == 1:
        start_content = normalized_starts[0]
        return ScaffoldAlignment(
            u_content[start_content][0],
            u_content[start_content + len(s_values) - 1][0] + 1,
            "normalized_token",
            1.0,
        )
    if len(normalized_starts) > 1:
        return None

    best_length = 0
    candidates: list[tuple[int, int]] = []
    for u_start in range(len(u_values)):
        for s_start in range(len(s_values)):
            length = 0
            while u_start + length < len(u_values) and s_start + length < len(s_values) and u_values[u_start + length] == s_values[s_start + length]:
                length += 1
            if length > best_length:
                best_length, candidates = length, [(u_start, u_start + length)]
            elif length == best_length and length > 0:
                candidates.append((u_start, u_start + length))
    unique = sorted(set(candidates))
    coverage = best_length / len(s_values)
    if best_length < int(minimum_content_tokens) or coverage < float(minimum_coverage) or len(unique) != 1:
        return None
    start_content, end_content = unique[0]
    return ScaffoldAlignment(
        u_content[start_content][0],
        u_content[end_content - 1][0] + 1,
        "longest_content_overlap",
        coverage,
    )


def deterministic_best_prefix(rows: Sequence[Mapping[str, Any]], tolerance: float = 1e-12) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("No prefix candidates")
    scored = []
    for row in rows:
        score = (
            float(row["m_first"])
            + 0.25 * float(row["m_4"])
            - 0.10 * float(row["nll_target"])
            - 0.01 * int(row["prefix_length"])
        )
        scored.append({**dict(row), "score": score})
    maximum = max(float(row["score"]) for row in scored)
    tied = [row for row in scored if maximum - float(row["score"]) <= float(tolerance)]
    return min(tied, key=lambda row: int(row["prefix_length"]))


def construct_target_path(
    prompt_ids: torch.Tensor,
    natural_prefix_ids: Sequence[int],
    target_ids: torch.Tensor,
    natural_suffix_ids: Sequence[int],
) -> dict[str, Any]:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or target_ids.ndim != 1:
        raise ValueError("Expected prompt [1,n] and target [m]")
    prefix_generated = torch.tensor([list(map(int, natural_prefix_ids))], dtype=prompt_ids.dtype, device=prompt_ids.device)
    prefix = torch.cat([prompt_ids, prefix_generated], dim=1)
    inputs, labels, target_predictor_positions = shifted_boundary_training_tensors(prefix, target_ids)
    suffix = torch.tensor([list(map(int, natural_suffix_ids))], dtype=prompt_ids.dtype, device=prompt_ids.device)
    complete = torch.cat([prefix, target_ids.unsqueeze(0).to(prompt_ids.device), suffix], dim=1)
    return {
        "prefix_ids": prefix,
        "training_input_ids": inputs,
        "shifted_labels": labels,
        "target_predictor_positions": target_predictor_positions,
        "target_span": [int(prefix.shape[1]), int(prefix.shape[1] + target_ids.numel())],
        "complete_ids": complete,
    }


def assert_target_free_generation_prompts(unrestricted_prompt: str, short_prompt: str, target: str) -> None:
    normalized_target = normalize_medical_answer(target)
    for name, prompt in (("unrestricted", unrestricted_prompt), ("short_answer", short_prompt)):
        normalized_prompt = normalize_medical_answer(prompt)
        if normalized_target and re.search(r"(?<!\w)" + re.escape(normalized_target) + r"(?!\w)", normalized_prompt):
            raise AssertionError(f"Edited target leaked into {name} generation prompt")


def project_shadow(delta: torch.Tensor, anchor: torch.Tensor, relative_cap: float) -> torch.Tensor:
    return clip_global_relative_displacement(delta, anchor, maximum=float(relative_cap))


def choose_backtracking_proposal(
    baseline: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for proposal in proposals:
        improves_signal = (
            float(proposal["natural_first_margin"]) > float(baseline["natural_first_margin"])
            or float(proposal.get("sequence_contrast", 0.0)) > float(baseline.get("sequence_contrast", 0.0))
        )
        if (
            float(proposal["loss"]) < float(baseline["loss"])
            and improves_signal
            and bool(proposal["prefix_preserved"])
            and bool(proposal["locality_top1_preserved"])
            and float(proposal["locality_nll_drift"]) <= 0.01
        ):
            return proposal
    return None


def snapshot_identity(parameter: torch.nn.Parameter, bank_hash: str) -> dict[str, str]:
    return {"parameter_hash": tensor_sha256(parameter), "bank_hash": str(bank_hash)}


def assert_shadow_only(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    if dict(before) != dict(after):
        raise RuntimeError("Canonical model weight or bank changed outside an isolated shadow application")


def relative_cap_reached(delta: torch.Tensor, anchor: torch.Tensor, cap: float, tolerance: float = 1e-8) -> bool:
    return relative_parameter_displacement(delta, anchor) >= float(cap) - float(tolerance)


def save_candidate_state(path: Path, delta: torch.Tensor, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    torch.save({"delta": delta.detach().cpu().float(), "metadata": dict(metadata)}, path)


def load_candidate_state(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def exact_rollback(parameter: torch.nn.Parameter, snapshot: torch.Tensor) -> bool:
    return bool(torch.equal(parameter.detach(), snapshot) and tensor_sha256(parameter) == tensor_sha256(snapshot))


def optimizer_state_hash(shadow: torch.Tensor, accepted_steps: int, relative_cap: float) -> str:
    value = {"shadow_hash": tensor_sha256(shadow), "accepted_steps": int(accepted_steps), "relative_cap": float(relative_cap)}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
