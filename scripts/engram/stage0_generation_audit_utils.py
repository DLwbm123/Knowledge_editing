"""Pure helpers for the ENGRAM V2 Stage-0 loss/generation audit.

The functions in this module do not edit model weights.  Model-facing helpers
use the exact token and image tensors supplied by the caller so that the audit
can assert alignment instead of silently rebuilding inputs.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


NEGATION_TERMS = {"no", "not", "without", "negative", "absent", "absence"}
LATERALITY_TERMS = {"left", "right", "bilateral"}
PRESERVED_MODIFIERS = NEGATION_TERMS | LATERALITY_TERMS | {
    "mild", "moderate", "severe", "grade", "dose", "unit", "units",
}


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def ids_sha256(value: torch.Tensor) -> str:
    return tensor_sha256(value.to(dtype=torch.int64))


@dataclass(frozen=True)
class CanonicalInputs:
    prompt_text: str
    full_text: str
    prompt_ids: torch.Tensor
    full_ids: torch.Tensor
    image: torch.Tensor
    answer_start: int
    target_ids: torch.Tensor
    prompt_hash: str
    full_hash: str
    pixel_hash: str


def build_canonical_inputs(model: Any, sample: Mapping[str, Any]) -> CanonicalInputs:
    """Use the repository's real LLaVA-Med template/tokenizer/image path once."""
    prompt = str(sample["prompt"][0])
    target = str(sample["target"][0])
    prompt_text = model._conversation_prompt(prompt, None)
    full_text = model._conversation_prompt(prompt, target)
    prompt_ids = model.tokenizer_image_token(
        prompt_text,
        model.llava_tokenizer,
        model.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.lm_device)
    full_ids = model.tokenizer_image_token(
        full_text,
        model.llava_tokenizer,
        model.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.lm_device)
    answer_start = int(prompt_ids.shape[1])
    if full_ids.shape[1] <= answer_start:
        raise RuntimeError("Canonical prompt+answer contains no answer suffix")
    if not torch.equal(full_ids[:, :answer_start], prompt_ids):
        raise RuntimeError("Canonical prompt-only IDs are not a prefix of prompt+answer IDs")
    image = model._image_for_row(dict(sample), 0)
    return CanonicalInputs(
        prompt_text=prompt_text,
        full_text=full_text,
        prompt_ids=prompt_ids,
        full_ids=full_ids,
        image=image,
        answer_start=answer_start,
        target_ids=full_ids[0, answer_start:].clone(),
        prompt_hash=ids_sha256(prompt_ids),
        full_hash=ids_sha256(full_ids),
        pixel_hash=tensor_sha256(image),
    )


def assert_no_gold_leakage(generation_ids: torch.Tensor, canonical: CanonicalInputs) -> None:
    if generation_ids.ndim != 2 or generation_ids.shape[0] != 1:
        raise AssertionError("Stage-0 generation requires batch size one")
    if int(generation_ids.shape[1]) != canonical.answer_start:
        raise AssertionError("Gold-answer leakage: generation input extends beyond answer_start")
    if not torch.equal(generation_ids, canonical.prompt_ids):
        raise AssertionError("Generation IDs differ from the canonical prompt-only IDs")


def first_supervised_position(labels: torch.Tensor, ignore_index: int) -> Optional[int]:
    positions = torch.where(labels[0].ne(int(ignore_index)))[0]
    return int(positions[0].item()) if positions.numel() else None


def _competitor_and_rank(logits: torch.Tensor, target_id: int) -> Dict[str, Any]:
    values = logits.float()
    target_logit = values[int(target_id)]
    top_values, top_ids = torch.topk(values, min(2, values.numel()))
    top1_id = int(top_ids[0].item())
    competitor = top_values[1] if top1_id == int(target_id) and top_values.numel() > 1 else top_values[0]
    log_probs = torch.log_softmax(values, dim=-1)
    return {
        "target_logit": float(target_logit.item()),
        "target_probability": float(log_probs[int(target_id)].exp().item()),
        "target_rank": int(values.gt(target_logit).sum().item()) + 1,
        "top1_id": top1_id,
        "top1_probability": float(log_probs[top1_id].exp().item()),
        "margin": float((target_logit - competitor).item()),
        "nll": float(-log_probs[int(target_id)].item()),
    }


def eos_diagnostics(logits: torch.Tensor, eos_ids: Sequence[int]) -> Dict[str, Any]:
    if not eos_ids:
        return {"eos_probability": None, "eos_rank": None, "eos_logit": None}
    values = logits.float()
    log_probs = torch.log_softmax(values, dim=-1)
    candidates = []
    for eos_id in eos_ids:
        eos_id = int(eos_id)
        if 0 <= eos_id < values.numel():
            candidates.append((
                float(log_probs[eos_id].exp().item()),
                int(values.gt(values[eos_id]).sum().item()) + 1,
                float(values[eos_id].item()),
            ))
    if not candidates:
        return {"eos_probability": None, "eos_rank": None, "eos_logit": None}
    probability, rank, logit = max(candidates, key=lambda item: item[0])
    return {"eos_probability": probability, "eos_rank": rank, "eos_logit": logit}


@torch.inference_mode()
def model_next_logits(model: Any, input_ids: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    attention = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    output = model.llava_model(
        input_ids=input_ids,
        images=image,
        attention_mask=attention,
        return_dict=True,
        use_cache=False,
    )
    return output.logits[0, -1].float()


@torch.inference_mode()
def score_target_incrementally(
    model: Any,
    canonical: CanonicalInputs,
    eos_ids: Sequence[int],
    *,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokenizer = model.llava_tokenizer
    for index, target_id_value in enumerate(canonical.target_ids.tolist()):
        target_id = int(target_id_value)
        prefix = canonical.full_ids[:, : canonical.answer_start + index]
        logits = model_next_logits(model, prefix, canonical.image)
        stats = _competitor_and_rank(logits, target_id)
        top_values, top_ids = torch.topk(logits, min(int(top_k), logits.numel()))
        rows.append({
            "token_index": index,
            "target_id": target_id,
            "target_text": tokenizer.decode([target_id], skip_special_tokens=False),
            **stats,
            "top1_text": tokenizer.decode([stats["top1_id"]], skip_special_tokens=False),
            **eos_diagnostics(logits, eos_ids),
            "top_k": [
                {
                    "token_id": int(token_id),
                    "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                    "logit": float(value),
                }
                for value, token_id in zip(top_values.tolist(), top_ids.tolist())
            ],
            "prefix_hash": ids_sha256(prefix),
        })
    return rows


def incremental_mean_nll(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise RuntimeError("Cannot compute incremental NLL for an empty target")
    return float(sum(float(row["nll"]) for row in rows) / len(rows))


def classify_termination(
    generated_ids: Sequence[int],
    target_ids: Sequence[int],
    eos_ids: Sequence[int],
    max_new_tokens: int,
) -> Dict[str, Any]:
    generated = [int(item) for item in generated_ids]
    target = [int(item) for item in target_ids]
    eos_set = {int(item) for item in eos_ids}
    eos_step = next((idx for idx, token in enumerate(generated) if token in eos_set), None)
    cap_hit = len(generated) >= int(max_new_tokens) and eos_step is None
    target_complete = len(generated) >= len(target) and generated[: len(target)] == target
    early_eos = eos_step is not None and not target_complete
    termination_failure = target_complete and eos_step is None
    stop_reason = "eos" if eos_step is not None else ("max_new_tokens" if cap_hit else "other")
    return {
        "eos_step": eos_step,
        "cap_hit": cap_hit,
        "target_completed_before_stop": target_complete,
        "early_eos_failure": early_eos,
        "termination_failure": termination_failure,
        "stop_reason": stop_reason,
        "valid_effectiveness": not cap_hit,
    }


def first_repeated_ngram_step(token_ids: Sequence[int], n: int) -> Optional[int]:
    seen = set()
    values = [int(item) for item in token_ids]
    for end in range(n, len(values) + 1):
        gram = tuple(values[end - n:end])
        if gram in seen:
            return end - 1
        seen.add(gram)
    return None


@torch.inference_mode()
def manual_greedy_trace(
    model: Any,
    canonical: CanonicalInputs,
    max_new_tokens: int,
    eos_ids: Sequence[int],
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    assert_no_gold_leakage(canonical.prompt_ids, canonical)
    generated: List[int] = []
    trace: List[Dict[str, Any]] = []
    tokenizer = model.llava_tokenizer
    eos_set = {int(item) for item in eos_ids}
    for step in range(int(max_new_tokens)):
        suffix = torch.tensor([generated], device=canonical.prompt_ids.device, dtype=canonical.prompt_ids.dtype)
        input_ids = torch.cat([canonical.prompt_ids, suffix], dim=1)
        logits = model_next_logits(model, input_ids, canonical.image)
        selected_id = int(torch.argmax(logits).item())
        target_id = int(canonical.target_ids[step].item()) if step < canonical.target_ids.numel() else None
        row: Dict[str, Any] = {
            "step": step,
            "selected_id": selected_id,
            "selected_text": tokenizer.decode([selected_id], skip_special_tokens=False),
            "target_id": target_id,
            "prefix_hash": ids_sha256(input_ids),
            **eos_diagnostics(logits, eos_ids),
        }
        if target_id is not None:
            row.update(_competitor_and_rank(logits, target_id))
        top_values, top_ids = torch.topk(logits, min(int(top_k), logits.numel()))
        row["top_k"] = [
            {"token_id": int(token_id), "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False), "logit": float(value)}
            for value, token_id in zip(top_values.tolist(), top_ids.tolist())
        ]
        trace.append(row)
        generated.append(selected_id)
        if selected_id in eos_set:
            break
    termination = classify_termination(generated, canonical.target_ids.tolist(), eos_ids, max_new_tokens)
    first_divergence = next(
        (idx for idx, (actual, target) in enumerate(zip(generated, canonical.target_ids.tolist())) if int(actual) != int(target)),
        None,
    )
    if first_divergence is None and len(generated) < canonical.target_ids.numel():
        first_divergence = len(generated)
    return {
        "token_ids": generated,
        "raw_output": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "trajectory": trace,
        "first_divergence": first_divergence,
        "first_repeated_bigram_step": first_repeated_ngram_step(generated, 2),
        "first_repeated_trigram_step": first_repeated_ngram_step(generated, 3),
        "repeated_trigram_count": _repeated_ngram_count(generated, 3),
        **termination,
    }


def _cache_length(past_key_values: Any) -> Optional[int]:
    try:
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        return int(past_key_values[0][0].shape[-2])
    except Exception:
        return None


@torch.inference_mode()
def manual_cached_greedy_trace(
    model: Any,
    canonical: CanonicalInputs,
    max_new_tokens: int,
    eos_ids: Sequence[int],
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Deterministic batch-one greedy decoding with the model's real KV cache."""
    assert_no_gold_leakage(canonical.prompt_ids, canonical)
    tokenizer = model.llava_tokenizer
    eos_set = {int(item) for item in eos_ids}
    generated: List[int] = []
    trace: List[Dict[str, Any]] = []
    attention = torch.ones_like(canonical.prompt_ids, dtype=torch.long, device=canonical.prompt_ids.device)
    output = model.llava_model(
        input_ids=canonical.prompt_ids,
        images=canonical.image,
        attention_mask=attention,
        return_dict=True,
        use_cache=True,
    )
    past = output.past_key_values
    logits = output.logits[0, -1].float()
    for step in range(int(max_new_tokens)):
        selected_id = int(torch.argmax(logits).item())
        target_id = int(canonical.target_ids[step].item()) if step < canonical.target_ids.numel() else None
        row: Dict[str, Any] = {
            "step": step,
            "selected_id": selected_id,
            "selected_text": tokenizer.decode([selected_id], skip_special_tokens=False),
            "target_id": target_id,
            "cache_position": _cache_length(past),
            **eos_diagnostics(logits, eos_ids),
        }
        if target_id is not None:
            row.update(_competitor_and_rank(logits, target_id))
        top_values, top_ids = torch.topk(logits, min(int(top_k), logits.numel()))
        row["top_k"] = [
            {"token_id": int(token_id), "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False), "logit": float(value)}
            for value, token_id in zip(top_values.tolist(), top_ids.tolist())
        ]
        trace.append(row)
        generated.append(selected_id)
        if selected_id in eos_set:
            break
        step_ids = torch.tensor([[selected_id]], device=canonical.prompt_ids.device, dtype=canonical.prompt_ids.dtype)
        attention = torch.ones(
            (1, canonical.answer_start + len(generated)),
            dtype=torch.long,
            device=canonical.prompt_ids.device,
        )
        output = model.llava_model(
            input_ids=step_ids,
            images=None,
            attention_mask=attention,
            past_key_values=past,
            return_dict=True,
            use_cache=True,
        )
        past = output.past_key_values
        logits = output.logits[0, -1].float()
    termination = classify_termination(generated, canonical.target_ids.tolist(), eos_ids, max_new_tokens)
    return {
        "token_ids": generated,
        "raw_output": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "trajectory": trace,
        "first_repeated_bigram_step": first_repeated_ngram_step(generated, 2),
        "first_repeated_trigram_step": first_repeated_ngram_step(generated, 3),
        "repeated_trigram_count": _repeated_ngram_count(generated, 3),
        **termination,
    }


def _repeated_ngram_count(token_ids: Sequence[int], n: int) -> int:
    values = [int(item) for item in token_ids]
    grams = [tuple(values[index:index + n]) for index in range(max(0, len(values) - n + 1))]
    return len(grams) - len(set(grams))


def normalize_medical_answer(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    value = re.sub(r"^(the\s+answer\s+is|answer\s*:|it\s+is)\s+", "", value)
    value = re.sub(r"[^\w\s]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: Any) -> List[str]:
    return normalize_medical_answer(text).split()


def _polarity(text: Any) -> str:
    return "negative" if NEGATION_TERMS.intersection(_tokens(text)) else "positive"


def _laterality(text: Any) -> Optional[str]:
    found = LATERALITY_TERMS.intersection(_tokens(text))
    return sorted(found)[0] if len(found) == 1 else ("ambiguous" if found else None)


def medical_answer_match(
    output: Any,
    target: Any,
    *,
    aliases: Optional[Sequence[str]] = None,
    required_terms: Optional[Sequence[str]] = None,
    forbidden_terms: Optional[Sequence[str]] = None,
    required_polarity: Optional[str] = None,
    required_laterality: Optional[str] = None,
) -> Dict[str, Any]:
    raw_output = str(output or "").strip()
    raw_target = str(target or "").strip()
    normalized_output = normalize_medical_answer(raw_output)
    normalized_target = normalize_medical_answer(raw_target)
    alias_values = [str(item) for item in (aliases or [])]
    normalized_aliases = [normalize_medical_answer(item) for item in alias_values if normalize_medical_answer(item)]
    output_tokens = set(_tokens(output))
    required = [normalize_medical_answer(item) for item in (required_terms or [])]
    forbidden = [normalize_medical_answer(item) for item in (forbidden_terms or [])]
    expected_polarity = required_polarity or _polarity(target)
    expected_laterality = required_laterality or _laterality(target)
    actual_polarity = _polarity(output)
    actual_laterality = _laterality(output)
    polarity_ok = actual_polarity == expected_polarity
    laterality_ok = expected_laterality in (None, "none") or actual_laterality == expected_laterality
    required_ok = all(set(item.split()).issubset(output_tokens) for item in required if item)
    forbidden_ok = all(not set(item.split()).issubset(output_tokens) for item in forbidden if item)
    normalized_exact = bool(normalized_output and normalized_output == normalized_target)
    alias_match = bool(normalized_output and normalized_output in normalized_aliases)
    lexical_match = normalized_exact or alias_match
    clinical_ok = bool(lexical_match and polarity_ok and laterality_ok and required_ok and forbidden_ok)
    return {
        "raw_exact_match": bool(raw_output and raw_output == raw_target),
        "normalized_exact_match": normalized_exact,
        "preregistered_alias_match": alias_match,
        "clinical_constraint_match": clinical_ok,
        "normalized_output": normalized_output,
        "normalized_target": normalized_target,
        "output_polarity": actual_polarity,
        "required_polarity": expected_polarity,
        "polarity_match": polarity_ok,
        "output_laterality": actual_laterality,
        "required_laterality": expected_laterality,
        "laterality_match": laterality_ok,
        "required_terms_match": required_ok,
        "forbidden_terms_match": forbidden_ok,
        "annotation_coverage_incomplete": not bool(alias_values or required_terms or forbidden_terms or required_polarity or required_laterality),
        "preserved_modifier_vocabulary": sorted(PRESERVED_MODIFIERS),
    }


def normalized_candidate_score(token_rows: Sequence[Mapping[str, Any]]) -> float:
    if not token_rows:
        return -math.inf
    return float(-sum(float(row["nll"]) for row in token_rows) / len(token_rows))


def shared_generation_budget(tokenizer: Any, targets_and_aliases: Iterable[str]) -> int:
    maximum = max((len(tokenizer(str(text), add_special_tokens=False).input_ids) for text in targets_and_aliases), default=0)
    return max(64, 2 * maximum + 16)
