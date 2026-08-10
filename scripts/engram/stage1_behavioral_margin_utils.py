"""Pure safety helpers for the ENGRAM V2 Stage-1 one-edit gate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer, tensor_sha256


HARD_STOP_SPAN_NOT_FOUND = "NATURAL_ANSWER_SPAN_NOT_FOUND"
HARD_STOP_SPAN_AMBIGUOUS = "NATURAL_ANSWER_SPAN_AMBIGUOUS"


class NaturalAnswerSpanError(RuntimeError):
    def __init__(self, label: str, message: str):
        super().__init__(message)
        self.label = label


@dataclass(frozen=True)
class AnswerSpan:
    start: int
    end: int
    method: str
    decoded: str


def _subsequence_starts(values: Sequence[int], needle: Sequence[int]) -> list[int]:
    source = [int(item) for item in values]
    target = [int(item) for item in needle]
    if not target:
        raise ValueError("Answer token sequence must not be empty")
    return [
        index
        for index in range(max(0, len(source) - len(target) + 1))
        if source[index : index + len(target)] == target
    ]


def align_unique_answer_span(
    response_ids: Sequence[int],
    answer_ids: Sequence[int],
    *,
    tokenizer: Any,
    answer_text: str,
) -> AnswerSpan:
    """Find one old-answer span, preferring exact tokens then normalization."""
    exact = _subsequence_starts(response_ids, answer_ids)
    if len(exact) == 1:
        start = exact[0]
        end = start + len(answer_ids)
        return AnswerSpan(start, end, "exact_token", tokenizer.decode(list(response_ids[start:end]), skip_special_tokens=True))
    if len(exact) > 1:
        raise NaturalAnswerSpanError(HARD_STOP_SPAN_AMBIGUOUS, f"Old answer occurs {len(exact)} times by exact token alignment")

    normalized_answer = normalize_medical_answer(answer_text)
    normalized: list[AnswerSpan] = []
    values = [int(item) for item in response_ids]
    for start in range(len(values)):
        for end in range(start + 1, len(values) + 1):
            decoded = tokenizer.decode(values[start:end], skip_special_tokens=True)
            if normalize_medical_answer(decoded) == normalized_answer:
                normalized.append(AnswerSpan(start, end, "registered_normalization", decoded))
    unique = {(item.start, item.end): item for item in normalized}
    if len(unique) == 1:
        return next(iter(unique.values()))
    if len(unique) > 1:
        raise NaturalAnswerSpanError(HARD_STOP_SPAN_AMBIGUOUS, f"Old answer has {len(unique)} normalized token spans")
    raise NaturalAnswerSpanError(HARD_STOP_SPAN_NOT_FOUND, "Dataset old/reference answer is absent from the deterministic S0 response")


def prompt_plus_natural_prefix(
    prompt_ids: torch.Tensor,
    response_ids: Sequence[int],
    span: AnswerSpan,
) -> torch.Tensor:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("Natural prefix construction requires batch size one")
    if span.start < 0 or span.end <= span.start or span.end > len(response_ids):
        raise ValueError("Answer span is outside the generated response")
    prefix = torch.tensor(
        [list(map(int, response_ids[: span.start]))],
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    )
    return torch.cat([prompt_ids, prefix], dim=1)


def shifted_boundary_training_tensors(prefix_ids: torch.Tensor, target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build causal inputs/labels and predictor indices for a continuation."""
    if prefix_ids.ndim != 2 or target_ids.ndim != 1 or prefix_ids.shape[0] != 1 or target_ids.numel() == 0:
        raise ValueError("Expected prefix [1,n] and non-empty target [m]")
    full = torch.cat([prefix_ids, target_ids.unsqueeze(0).to(prefix_ids.device)], dim=1)
    inputs = full[:, :-1]
    labels = full[:, 1:]
    first = int(prefix_ids.shape[1]) - 1
    positions = list(range(first, first + int(target_ids.numel())))
    if labels[0, positions[0]].item() != target_ids[0].item():
        raise AssertionError("First shifted label is not the natural-boundary target token")
    return inputs, labels, positions


def assert_generation_inputs_target_free(
    unrestricted_ids: torch.Tensor,
    expected_unrestricted_ids: torch.Tensor,
    short_ids: torch.Tensor,
    expected_short_ids: torch.Tensor,
    target_ids: Sequence[int],
) -> None:
    for name, observed, expected in (
        ("unrestricted", unrestricted_ids, expected_unrestricted_ids),
        ("short_answer", short_ids, expected_short_ids),
    ):
        if not torch.equal(observed, expected):
            raise AssertionError(f"Target leakage: {name} generation input differs from its prompt-only tensor")
        flat = observed.detach().cpu().reshape(-1).tolist()
        if _subsequence_starts(flat, target_ids):
            raise AssertionError(f"Target leakage: full target token sequence occurs in {name} generation input")


def tensor_l2(value: torch.Tensor) -> float:
    return float(value.detach().double().norm().cpu())


def match_exact_budget(delta: torch.Tensor, budget: float, epsilon: float = 1.0e-15) -> torch.Tensor:
    if budget < 0:
        raise ValueError("Budget must be nonnegative")
    norm = tensor_l2(delta)
    if budget == 0:
        return torch.zeros_like(delta)
    if norm <= epsilon:
        raise RuntimeError("Cannot match a positive trust-region budget from a zero direction")
    scaled = delta.detach().double() * (float(budget) / norm)
    return scaled.to(dtype=delta.dtype, device=delta.device)


def relative_parameter_displacement(delta: torch.Tensor, anchor_weight: torch.Tensor, epsilon: float = 1.0e-15) -> float:
    return tensor_l2(delta) / max(tensor_l2(anchor_weight), epsilon)


def clip_global_relative_displacement(delta: torch.Tensor, anchor_weight: torch.Tensor, maximum: float = 0.003) -> torch.Tensor:
    if maximum <= 0:
        raise ValueError("Maximum relative displacement must be positive")
    budget = float(maximum) * tensor_l2(anchor_weight)
    norm = tensor_l2(delta)
    return delta.detach().clone() if norm <= budget else match_exact_budget(delta, budget)


def normalized_shadow_step(shadow: torch.Tensor, gradient: torch.Tensor, radius: float) -> torch.Tensor:
    """Deterministic ascent step that mutates neither shadow nor model weight."""
    if shadow.shape != gradient.shape:
        raise ValueError("Shadow and gradient shapes differ")
    if not torch.isfinite(gradient).all():
        raise RuntimeError("Non-finite behavior-margin gradient")
    direction_norm = tensor_l2(gradient)
    if direction_norm == 0:
        raise RuntimeError("Zero behavior-margin gradient")
    candidate = shadow.detach().double() + gradient.detach().double() / direction_norm
    return match_exact_budget(candidate.to(shadow.dtype), float(radius))


def canonical_optimizer_hash(shadow: torch.Tensor, step: int, radius: float) -> str:
    payload = {
        "shadow_hash": tensor_sha256(shadow),
        "step": int(step),
        "radius": float(radius),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_bank_immutable(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if dict(before) != dict(after):
        raise RuntimeError("Original ENGRAM V2 bank changed during Stage-1")


def assert_candidate_namespace(original_bank: Path, candidate_bank: Path) -> None:
    original = original_bank.resolve()
    candidate = candidate_bank.resolve()
    if candidate == original or original in candidate.parents:
        raise ValueError("Candidate bank must use a separate experimental namespace")


def assert_three_path_parity(no_cache: Sequence[int], cached: Sequence[int], hf: Sequence[int]) -> None:
    if list(map(int, no_cache)) != list(map(int, cached)) or list(map(int, no_cache)) != list(map(int, hf)):
        raise RuntimeError("Cached/no-cache/HF token trajectories differ")


def assert_locality_preserved(baseline: Mapping[str, Any], candidate: Mapping[str, Any], max_nll_drift: float = 0.01) -> None:
    exact_fields = ("token_ids", "normalized_output", "first_top1_id", "stop_reason")
    for field in exact_fields:
        if baseline.get(field) != candidate.get(field):
            raise RuntimeError(f"Locality changed: {field}")
    if bool(candidate.get("cap_hit")):
        raise RuntimeError("Locality generation hit the uniform cap")
    if abs(float(candidate["nll"]) - float(baseline["nll"])) > float(max_nll_drift):
        raise RuntimeError("Locality NLL drift exceeds the Stage-1 gate")


def save_candidate_payload_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    torch.save(dict(payload), path)


def load_candidate_payload(path: Path) -> Mapping[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def fresh_reproduction_equal(reference: Mapping[str, Any], fresh: Mapping[str, Any], tolerance: float) -> bool:
    return bool(
        reference.get("token_ids") == fresh.get("token_ids")
        and abs(float(reference.get("nll", 0.0)) - float(fresh.get("nll", 0.0))) <= float(tolerance)
    )


def exact_parameter_rollback(parameter: torch.nn.Parameter, snapshot: torch.Tensor) -> bool:
    return bool(torch.equal(parameter.detach(), snapshot) and tensor_sha256(parameter) == tensor_sha256(snapshot))


def run_deterministic_twice(operation: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    first = operation()
    second = operation()
    if not torch.equal(first, second):
        raise RuntimeError("Deterministic optimizer replay mismatch")
    return first, second
