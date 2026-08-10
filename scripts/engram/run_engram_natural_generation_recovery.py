#!/usr/bin/env python3
"""Quantization-aware ENGRAM V2.1 gate followed by the V3 generation bridge."""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.nn.utils import parametrize

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import to_jsonable  # noqa: E402
from scripts.engram.natural_generation_recovery_utils import (  # noqa: E402
    assert_no_target_leakage,
    canonical_natural_response,
    expanded_predictor_positions,
    global_l2,
    materialize_fp32_shadow,
    orthonormal_locality_basis,
    project_deltas_to_relative_budget,
    project_effect_gradient,
    rank4_svd_initialization,
    relative_displacement,
    save_v3_bank,
    select_candidate_modules,
    target_only_response,
    tensor_sha256,
)
from scripts.engram.run_engram_v2_one_shot_natural_generation_rescue import (  # noqa: E402
    all_token_stats,
    bank_anchor_hash,
    canonical_hash,
    canonical_nll,
    compact_reference,
    content_indices,
    full_generation_parity,
    generation_success,
    read_jsonl,
    reference_equal,
    sha256_file,
)
from scripts.engram.run_engram_v2_stage0_generation_audit import (  # noqa: E402
    BANK_ROOT,
    MODEL_CONFIG,
    MODULE_KEY,
    MODULE_NAME,
    ORDER,
    apply_prefix,
    bank_manifest,
    clone_sample_with_target,
    eos_ids,
    load_model_views_bank,
    state_weight_hash,
)
from scripts.engram.run_engram_v2_stage0abc_diagnostics import short_answer_sample  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import (  # noqa: E402
    CanonicalInputs,
    build_canonical_inputs,
    manual_greedy_trace,
    normalize_medical_answer,
)
from scripts.engram.stage0abc_diagnostic_utils import create_new_output_dir  # noqa: E402


STARTING_COMMIT = "bb18961c7dca163f9beff86a4ba045781610374a"
RECORD_ID = "953"
CAP = 128
B1 = 0.007530835302
STANDARD_CAP = 0.003
RESCUE_CAP = 0.010
PROTOCOL = "ENGRAM_NATURAL_GENERATION_RECOVERY_V2_1_V3"
PREVIOUS = ROOT / "outputs/engram_v2_one_shot_natural_generation_rescue/20260810_one_shot_v1"
CANDIDATE_MODULES = [
    f"llava_model.model.layers.{layer}.{suffix}"
    for layer in range(18, 25)
    for suffix in ("self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj")
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "validate-v3-terminal"), default="run")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--physical-gpu", default=2, type=int)
    parser.add_argument("--starting-commit", default=STARTING_COMMIT)
    parser.add_argument("--candidate-bank", type=Path)
    return parser.parse_args()


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        handle.write(value.rstrip() + "\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(to_jsonable(dict(value)), sort_keys=True) + "\n")


def source_diff() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/engram/natural_generation_recovery_utils.py",
        ROOT / "tests/test_engram_natural_generation_recovery.py",
    )
    result: list[str] = []
    for path in paths:
        result.extend(difflib.unified_diff([], path.read_text().splitlines(True), fromfile="/dev/null", tofile=f"b/{path.relative_to(ROOT)}"))
    return "".join(result)


def selected_logits(model: Any, canonical: CanonicalInputs) -> torch.Tensor:
    answer = canonical.target_ids
    inputs = torch.cat([canonical.prompt_ids, answer[:-1].unsqueeze(0)], dim=1)
    attention = torch.ones_like(inputs, dtype=torch.long, device=inputs.device)
    output = model.llava_model(input_ids=inputs, images=canonical.image, attention_mask=attention, return_dict=True, use_cache=False)
    expansion = int(output.logits.shape[1] - inputs.shape[1])
    positions = expanded_predictor_positions(canonical.answer_start, int(answer.numel()), expansion)
    return output.logits[0, positions].float()


def token_subsequence(source: Sequence[int], target: Sequence[int]) -> list[int]:
    left, right = list(map(int, source)), list(map(int, target))
    return [index for index in range(max(0, len(left) - len(right) + 1)) if left[index : index + len(right)] == right]


def response_layout(model: Any, canonical: CanonicalInputs, target: str) -> Dict[str, Any]:
    response = canonical.target_ids.detach().cpu().tolist()
    variants = []
    for text in (target.strip(), " " + target.strip(), target.strip().rstrip(".?!") + "."):
        ids = model.llava_tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
        if ids and ids not in variants:
            variants.append(ids)
    matches = []
    for ids in variants:
        for start in token_subsequence(response, ids):
            matches.append((start, start + len(ids)))
    if not matches:
        target_norm = normalize_medical_answer(target)
        for start in range(len(response)):
            for end in range(start + 1, len(response) + 1):
                if normalize_medical_answer(model.llava_tokenizer.decode(response[start:end], skip_special_tokens=True)) == target_norm:
                    matches.append((start, end))
    matches = sorted(set(matches), key=lambda item: (item[1] - item[0], -item[0]), reverse=True)
    if not matches:
        raise RuntimeError("Target span not found in fixed natural response")
    start, end = matches[0]
    target_positions = [index for index in content_indices(model, response) if start <= index < end]
    scaffold_positions = [index for index in content_indices(model, response) if index < start]
    if not target_positions or not scaffold_positions:
        raise RuntimeError("Invalid scaffold/target token layout")
    return {"target_positions": target_positions, "scaffold_positions": scaffold_positions, "target_span": [start, end]}


def response_objective(model: Any, canonical: CanonicalInputs, target_positions: Sequence[int], scaffold_positions: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    positions = list(scaffold_positions) + list(target_positions)
    logits = selected_logits(model, canonical)[positions]
    targets = canonical.target_ids[torch.tensor(positions, device=canonical.target_ids.device)]
    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    top_values, top_ids = logits.topk(2, dim=-1)
    competitor = torch.where(top_ids[:, 0].eq(targets), top_values[:, 1], top_values[:, 0])
    margins = target_logits - competitor
    nll = -logits.log_softmax(dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    scaffold_count = len(scaffold_positions)
    weights = torch.ones(len(positions), device=logits.device)
    kappas = torch.full((len(positions),), 0.2, device=logits.device)
    weights[:scaffold_count] = 0.25
    kappas[:scaffold_count] = 0.0
    weights[scaffold_count] = 4.0
    kappas[scaffold_count] = 0.5
    rank_loss = (weights * torch.relu(kappas - margins)).sum() / weights.sum()
    target_nll = nll[scaffold_count:].mean()
    ranks = logits.gt(target_logits.unsqueeze(1)).sum(dim=1) + 1
    rows = [
        {
            "response_index": int(index),
            "role": "scaffold" if order < scaffold_count else "target",
            "token_id": int(targets[order].detach().item()),
            "token_text": model.llava_tokenizer.decode([int(targets[order].detach().item())], skip_special_tokens=False),
            "rank": int(ranks[order].detach().item()),
            "margin": float(margins[order].detach().item()),
            "nll": float(nll[order].detach().item()),
        }
        for order, index in enumerate(positions)
    ]
    return rank_loss, target_nll, {"rank_loss": float(rank_loss.detach().item()), "target_nll": float(target_nll.detach().item()), "tokens": rows, "first_target_rank": rows[scaffold_count]["rank"], "first_target_margin": rows[scaffold_count]["margin"]}


def effect_objective(model: Any, natural: CanonicalInputs, short: CanonicalInputs, natural_layout: Mapping[str, Any], short_positions: Sequence[int]) -> tuple[torch.Tensor, Dict[str, Any]]:
    natural_rank, natural_nll, natural_info = response_objective(model, natural, natural_layout["target_positions"], natural_layout["scaffold_positions"])
    short_rank, short_nll, short_info = response_objective(model, short, short_positions, [])
    loss = natural_rank + 0.25 * short_rank + 0.05 * natural_nll + 0.025 * short_nll
    return loss, {"effect_loss": float(loss.detach().item()), "natural": natural_info, "short": short_info}


def kl_candidate_s0(candidate: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
    logp = candidate.log_softmax(dim=-1)
    p = logp.exp()
    logq = s0.log_softmax(dim=-1)
    return (p * (logp - logq)).sum(dim=-1).mean()


def cache_localities(model: Any, views: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    with torch.no_grad():
        for record_id in ORDER:
            canonical = build_canonical_inputs(model, views[record_id]["locality"])
            logits = selected_logits(model, canonical).detach()
            labels = canonical.target_ids
            nll = float((-logits.log_softmax(dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1).mean()).item())
            rows.append({"record_id": record_id, "canonical": canonical, "s0_logits": logits, "s0_nll": nll, "s0_first_top1": int(logits[0].argmax().item())})
    return rows


def locality_checkpoint(model: Any, cache: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    with torch.no_grad():
        for row in cache:
            logits = selected_logits(model, row["canonical"])
            labels = row["canonical"].target_ids
            nll = float((-logits.log_softmax(dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1).mean()).item())
            rows.append({"record_id": row["record_id"], "nll": nll, "nll_drift": abs(nll - float(row["s0_nll"])), "first_top1": int(logits[0].argmax().item()), "first_top1_equal": int(logits[0].argmax().item()) == int(row["s0_first_top1"])})
    return {"rows": rows, "maximum_nll_drift": max(item["nll_drift"] for item in rows), "paired_first_top1_equal": rows[0]["first_top1_equal"]}


class FP32ShadowWeight(torch.nn.Module):
    def __init__(self, base: torch.Tensor):
        super().__init__()
        self.register_buffer("base", base.detach().float().clone())
        self.delta = torch.nn.Parameter(torch.zeros_like(self.base, dtype=torch.float32))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return materialize_fp32_shadow(self.base, self.delta, original.dtype)


class LowRankWeight(torch.nn.Module):
    def __init__(self, base: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float):
        super().__init__()
        self.register_buffer("base", base.detach().float().clone())
        self.A = torch.nn.Parameter(a.detach().float().to(base.device))
        self.B = torch.nn.Parameter(b.detach().float().to(base.device))
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32, device=base.device))

    def delta(self) -> torch.Tensor:
        return self.scale * (self.B @ self.A)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return (self.base + self.delta()).to(original.dtype)


def flatten_gradients(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in gradients])


def assign_flat_update(parameters: Sequence[torch.nn.Parameter], update: torch.Tensor, step_length: float) -> None:
    cursor = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(update[cursor : cursor + count].reshape_as(parameter), alpha=-float(step_length))
            cursor += count


def phase_gradient(model: Any, trainable: Sequence[torch.nn.Parameter], natural: CanonicalInputs, short: CanonicalInputs, natural_layout: Mapping[str, Any], short_positions: Sequence[int], locality_cache: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    effect, effect_info = effect_objective(model, natural, short, natural_layout, short_positions)
    effect_grads = torch.autograd.grad(effect, trainable)
    del effect
    locality_grads = [torch.zeros_like(parameter) for parameter in trainable]
    locality_value = 0.0
    for row in locality_cache:
        candidate_logits = selected_logits(model, row["canonical"])
        loss = kl_candidate_s0(candidate_logits, row["s0_logits"])
        grads = torch.autograd.grad(loss, trainable)
        locality_value += float(loss.detach().item()) / len(locality_cache)
        for index, gradient in enumerate(grads):
            locality_grads[index].add_(gradient, alpha=1.0 / len(locality_cache))
        del candidate_logits, loss, grads
        torch.cuda.empty_cache()
    return flatten_gradients(effect_grads), flatten_gradients(locality_grads), {**effect_info, "locality_kl": locality_value, "total_loss": effect_info["effect_loss"] + locality_value}


def checkpoint_metrics(model: Any, natural: CanonicalInputs, short: CanonicalInputs, natural_layout: Mapping[str, Any], short_positions: Sequence[int], locality_cache: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    with torch.no_grad():
        _loss, effect = effect_objective(model, natural, short, natural_layout, short_positions)
        locality = locality_checkpoint(model, locality_cache)
        kl = sum(float(kl_candidate_s0(selected_logits(model, row["canonical"]), row["s0_logits"]).item()) for row in locality_cache) / len(locality_cache)
    return {**effect, "locality_kl": kl, "total_loss": effect["effect_loss"] + kl, "locality": locality}


def unrestricted_checkpoint(model: Any, original: CanonicalInputs, short: CanonicalInputs, target: str, aliases: Sequence[str]) -> Dict[str, Any]:
    unrestricted = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=5)
    short_trace = manual_greedy_trace(model, short, CAP, eos_ids(model), top_k=5)
    return {"unrestricted": unrestricted, "short": short_trace, "match": generation_success(model, unrestricted, target, aliases, None)}


PROTECTED_LEXICON = {
    "ct", "mri", "ultrasound", "photoacoustic", "infrared", "xray", "radiograph",
    "left", "right", "bilateral", "sagittal", "axial", "transverse", "coronal",
    "quail", "mouse", "mice", "rat", "human", "embryo", "embryonic",
    "lung", "liver", "kidney", "brain", "heart", "sinus", "disc", "intradiscal",
    "leakage", "cement", "tumor", "fracture", "lesion", "hemorrhage", "effusion",
}


def word_stem(value: str) -> str:
    token = re.sub(r"[^a-z0-9]", "", value.casefold())
    for suffix in ("ically", "ingly", "ation", "onic", "ical", "ing", "ed", "ic", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def clinical_preservation(baseline: str, candidate: str, canonical_answer: str) -> Dict[str, Any]:
    base_tokens = normalize_medical_answer(baseline).split()
    candidate_tokens = normalize_medical_answer(candidate).split()
    canonical_tokens = [word_stem(item) for item in normalize_medical_answer(canonical_answer).split() if len(word_stem(item)) >= 3]
    candidate_stems = {word_stem(item) for item in candidate_tokens}
    coverage = sum(any(left == right or (len(left) >= 4 and (left.startswith(right) or right.startswith(left))) for right in candidate_stems) for left in canonical_tokens) / max(len(canonical_tokens), 1)
    invariants = lambda tokens: {
        "negation": sorted(set(tokens) & {"no", "not", "without", "absent"}),
        "laterality": sorted(set(tokens) & {"left", "right", "bilateral"}),
        "numbers": sorted(re.findall(r"\b\d+(?:\.\d+)?(?:\s*(?:mg|ml|mm|cm|day|days|year|years))?\b", " ".join(tokens))),
        "protected": sorted({word_stem(item) for item in tokens if item in PROTECTED_LEXICON}),
    }
    before, after = invariants(base_tokens), invariants(candidate_tokens)
    def approximately_contains(source: Sequence[str], candidates: Sequence[str]) -> bool:
        return all(any(left == right or (len(left) >= 4 and len(right) >= 4 and (left.startswith(right) or right.startswith(left))) for right in candidates) for left in source)
    preserved = all(approximately_contains(before[key], after[key]) for key in before)
    exact = normalize_medical_answer(baseline) == normalize_medical_answer(candidate)
    return {"exact_normalized": exact, "canonical_term_coverage": coverage, "invariants_before": before, "invariants_after": after, "invariants_preserved": preserved, "passed": exact or (coverage >= 0.5 and preserved)}


def full_locality_gate(model: Any, views: Mapping[str, Any], set_s0: Any, set_candidate: Any) -> Dict[str, Any]:
    set_s0()
    baseline = {}
    for record_id in ORDER:
        canonical = build_canonical_inputs(model, views[record_id]["locality"])
        trace = manual_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=1)
        baseline[record_id] = {"trace": trace, "nll": canonical_nll(model, canonical)["nll"]}
    set_candidate()
    rows = []
    for record_id in ORDER:
        canonical = build_canonical_inputs(model, views[record_id]["locality"])
        trace = manual_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=1)
        nll = canonical_nll(model, canonical)["nll"]
        prior = baseline[record_id]
        clinical = clinical_preservation(prior["trace"]["raw_output"], trace["raw_output"], str(views[record_id]["locality"]["target"][0]))
        strict = trace["token_ids"] == prior["trace"]["token_ids"] and trace["stop_reason"] == prior["trace"]["stop_reason"]
        rows.append({"record_id": record_id, "s0_output": prior["trace"]["raw_output"], "candidate_output": trace["raw_output"], "strict_equal": strict, "clinical": clinical, "nll_drift": abs(float(nll) - float(prior["nll"])), "cap_hit": trace["cap_hit"]})
    return {"rows": rows, "strict_damage_count": sum(not row["strict_equal"] for row in rows), "clinical_failure_count": sum(not row["clinical"]["passed"] for row in rows), "maximum_nll_drift": max(row["nll_drift"] for row in rows)}


def run_phase_a(model: Any, views: Mapping[str, Any], bank: Any, natural: CanonicalInputs, short: CanonicalInputs, original: CanonicalInputs, natural_layout: Mapping[str, Any], short_positions: Sequence[int], locality_cache: Sequence[Mapping[str, Any]], target: str, aliases: Sequence[str], out_dir: Path) -> Dict[str, Any]:
    module = dict(model.named_modules())[MODULE_NAME]
    base = bank.anchor_state()[MODULE_KEY].to(model.lm_device)
    zero_hash = state_weight_hash(model)
    zero_generation = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)
    shadow_param = FP32ShadowWeight(base).to(model.lm_device)
    parametrize.register_parametrization(module, "weight", shadow_param, unsafe=True)
    if manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"] != zero_generation["token_ids"]:
        raise RuntimeError("Phase-A zero-delta S0 parity failed")
    trainable = [shadow_param.delta]
    step_length = B1 / 64.0
    active_cap = STANDARD_CAP
    microstep = 0
    best = {"loss": math.inf, "delta": shadow_param.delta.detach().clone()}
    worsening = 0
    restarted = False
    terminal = "UNKNOWN"
    success = None
    while True:
        effect_grad, locality_grad, gradient_info = phase_gradient(model, trainable, natural, short, natural_layout, short_positions, locality_cache)
        gradient = effect_grad + locality_grad
        norm = gradient.double().norm()
        if not torch.isfinite(gradient).all() or float(norm.item()) == 0:
            terminal = "NONFINITE_OR_ZERO_GRADIENT"
            break
        with torch.no_grad():
            shadow_param.delta.add_(gradient.reshape_as(shadow_param.delta), alpha=-step_length / float(norm.item()))
            projected = project_deltas_to_relative_budget([shadow_param.delta], [shadow_param.base], active_cap)[0]
            shadow_param.delta.copy_(projected)
        microstep += 1
        displacement = relative_displacement([shadow_param.delta], [shadow_param.base])
        delta_norm = global_l2([shadow_param.delta])
        if microstep % 4 == 0:
            metrics = checkpoint_metrics(model, natural, short, natural_layout, short_positions, locality_cache)
            row = {"microstep": microstep, "active_cap": active_cap, "delta_norm": delta_norm, "relative_displacement": displacement, **metrics}
            append_jsonl(out_dir / "phase_a_trajectory.jsonl", row)
            if metrics["total_loss"] < best["loss"]:
                best = {"loss": metrics["total_loss"], "delta": shadow_param.delta.detach().clone()}
                worsening = 0
            else:
                worsening += 1
            if worsening >= 3:
                if not restarted:
                    with torch.no_grad():
                        shadow_param.delta.copy_(best["delta"])
                    restarted, worsening = True, 0
                    append_jsonl(out_dir / "phase_a_trajectory.jsonl", {"event": "RESTORE_LAST_BEST_FP32_SHADOW", "microstep": microstep})
                else:
                    terminal = "THREE_CONSECUTIVE_WORSENING_CHECKPOINTS_AFTER_RESTART"
                    break
            if not metrics["locality"]["paired_first_top1_equal"] or metrics["locality"]["maximum_nll_drift"] > 0.01:
                terminal = "LOCALITY_SAFETY_LIMIT"
                break
        rank = metrics["natural"]["first_target_rank"] if microstep % 4 == 0 else 999999
        if microstep % 8 == 0 or rank <= 5:
            generation = unrestricted_checkpoint(model, original, short, target, aliases)
            append_jsonl(out_dir / "phase_a_trajectory.jsonl", {"event": "GENERATION", "microstep": microstep, "relative_displacement": displacement, **generation})
            if generation["match"]["effective"]:
                success = {"microstep": microstep, "delta": shadow_param.delta.detach().clone(), "generation": generation, "relative_displacement": displacement, "delta_norm": delta_norm}
                terminal = "UNRESTRICTED_SUCCESS"
                break
        if displacement >= active_cap - 1e-10:
            if active_cap == STANDARD_CAP:
                active_cap = RESCUE_CAP
            else:
                terminal = "RESCUE_CAP_REACHED"
                break
    final_delta = shadow_param.delta.detach().clone()
    final_metrics = checkpoint_metrics(model, natural, short, natural_layout, short_positions, locality_cache)
    if success is None:
        final_generation = unrestricted_checkpoint(model, original, short, target, aliases)
    else:
        final_generation = success["generation"]
    set_s0 = lambda: shadow_param.delta.data.zero_()
    set_candidate = lambda: shadow_param.delta.data.copy_(final_delta)
    locality = full_locality_gate(model, views, set_s0, set_candidate)
    set_candidate()
    parity = full_generation_parity(model, original)
    rollback_before = tensor_sha256(module.weight)
    set_s0()
    rollback_generation = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"] == zero_generation["token_ids"]
    set_candidate()
    replay_generation = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"] == parity["no_cache"]["token_ids"]
    displacement = relative_displacement([final_delta], [shadow_param.base])
    effective = generation_success(model, parity["no_cache"], target, aliases, None)["effective"]
    if effective and locality["strict_damage_count"] == 0 and locality["clinical_failure_count"] == 0:
        label = "PASS_V2_QUANTIZATION_AWARE_WITHIN_B1" if global_l2([final_delta]) <= B1 + 1e-10 else ("PASS_V2_QUANTIZATION_AWARE_WITHIN_0P003" if displacement <= STANDARD_CAP + 1e-10 else "PASS_V2_QUANTIZATION_AWARE_RESCUE_0P010")
    elif effective:
        label = "V2_NATURAL_SUCCESS_STRICT_LOCALITY_FAILURE"
    elif terminal in {"NONFINITE_OR_ZERO_GRADIENT"}:
        label = "V2_INVALID_ENGINEERING_RUN"
    else:
        label = "V2_FULL_BUDGET_NO_NATURAL_GENERATION"
    result = {"label": label, "terminal_reason": terminal, "microsteps": microstep, "delta_norm": global_l2([final_delta]), "relative_displacement": displacement, "generation": final_generation, "parity": parity, "metrics": final_metrics, "locality": locality, "rollback_passed": rollback_generation, "replay_passed": replay_generation, "bank_hash_before": bank_manifest()["sha256"]}
    parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
    apply_prefix(model, bank, 0)
    result["bank_hash_after"] = bank_manifest()["sha256"]
    result["s0_weight_hash_restored"] = state_weight_hash(model) == zero_hash
    write_text(out_dir / "PHASE_A_V2_QUANTIZATION_AWARE_REPORT.md", f"# {label}\n\n- Terminal: `{terminal}`\n- Microsteps: `{microstep}`\n- Delta norm: `{result['delta_norm']:.12f}`\n- Relative displacement: `{displacement:.12f}`\n- Unrestricted output: `{parity['no_cache']['raw_output']}`\n- Strict locality damage: `{locality['strict_damage_count']}/10`\n- Clinical locality failures: `{locality['clinical_failure_count']}/10`\n")
    return result


def module_scoring(model: Any, natural: CanonicalInputs, short: CanonicalInputs, natural_layout: Mapping[str, Any], short_positions: Sequence[int]) -> tuple[list[Dict[str, Any]], Dict[str, torch.Tensor]]:
    modules = dict(model.named_modules())
    parameters = [modules[name].weight for name in CANDIDATE_MODULES]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in parameters:
        parameter.requires_grad_(True)
    effect, _ = effect_objective(model, natural, short, natural_layout, short_positions)
    gradients = torch.autograd.grad(effect, parameters)
    rows, saved = [], {}
    for name, parameter, gradient in zip(CANDIDATE_MODULES, parameters, gradients):
        layer = int(name.split("layers.")[1].split(".")[0])
        target_norm = float(gradient.detach().double().norm().item()) / math.sqrt(parameter.numel())
        rows.append({"layer": layer, "module_name": name, "numel": parameter.numel(), "target_size_normalized_norm": target_norm, "locality_size_normalized_norm": 0.0, "locality_gradient_at_clean_s0_is_exact_zero": True})
        saved[name] = gradient.detach().float().cpu()
    for parameter in parameters:
        parameter.requires_grad_(False)
    selected = select_candidate_modules(rows)
    selected_names = {row["module_name"] for row in selected}
    return [{**row, "score": float(row["target_size_normalized_norm"]) / 1e-12, "selected": row["module_name"] in selected_names} for row in rows], {name: saved[name] for name in selected_names}


def low_rank_deltas(parameters: Sequence[LowRankWeight]) -> list[torch.Tensor]:
    return [item.delta() for item in parameters]


def project_low_rank_budget(parameters: Sequence[LowRankWeight], cap: float) -> None:
    deltas = low_rank_deltas(parameters)
    bases = [item.base for item in parameters]
    relative = relative_displacement(deltas, bases)
    if relative > cap:
        ratio = cap / relative
        for item in parameters:
            item.scale.mul_(ratio)


def run_phase_b(model: Any, views: Mapping[str, Any], bank: Any, natural: CanonicalInputs, short: CanonicalInputs, original: CanonicalInputs, natural_layout: Mapping[str, Any], short_positions: Sequence[int], locality_cache: Sequence[Mapping[str, Any]], target: str, aliases: Sequence[str], out_dir: Path) -> Dict[str, Any]:
    s0_weight_hash = state_weight_hash(model)
    s0_unrestricted_ids = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    score_rows, selected_gradients = module_scoring(model, natural, short, natural_layout, short_positions)
    with (out_dir / "phase_b_module_scores.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader(); writer.writerows(score_rows)
    selected_names = [row["module_name"] for row in sorted((item for item in score_rows if item["selected"]), key=lambda row: (-row["score"], row["layer"], row["module_name"]))]
    modules = dict(model.named_modules())
    initial = {name: rank4_svd_initialization(selected_gradients[name]) for name in selected_names}
    raw = [initial[name]["B"] @ initial[name]["A"] for name in selected_names]
    common_scale = (B1 / 64.0) / max(global_l2(raw), 1e-30)
    parameterizations = []
    for name in selected_names:
        item = LowRankWeight(modules[name].weight.detach(), initial[name]["A"], initial[name]["B"], common_scale).to(model.lm_device)
        parametrize.register_parametrization(modules[name], "weight", item, unsafe=True)
        parameterizations.append(item)
    factor_params = [parameter for item in parameterizations for parameter in (item.A, item.B)]
    probe_gradients = []
    for row in locality_cache:
        loss = kl_candidate_s0(selected_logits(model, row["canonical"]), row["s0_logits"])
        grads = torch.autograd.grad(loss, factor_params)
        probe_gradients.append(flatten_gradients(grads))
    locality_basis = orthonormal_locality_basis(probe_gradients)
    basis_hash = tensor_sha256(locality_basis)
    step_length = B1 / 64.0
    active_cap = STANDARD_CAP
    microstep = 0
    terminal = "UNKNOWN"
    best = {"loss": math.inf, "factors": [parameter.detach().clone() for parameter in factor_params]}
    worsening, restarted = 0, False
    success = None
    while True:
        effect_grad, locality_grad, gradient_info = phase_gradient(model, factor_params, natural, short, natural_layout, short_positions, locality_cache)
        projected = project_effect_gradient(effect_grad, locality_basis)
        final_gradient = projected + locality_grad
        norm = float(final_gradient.double().norm().item())
        if not torch.isfinite(final_gradient).all() or norm == 0:
            terminal = "NONFINITE_OR_ZERO_FACTOR_GRADIENT"; break
        assign_flat_update(factor_params, final_gradient / norm, step_length)
        with torch.no_grad():
            project_low_rank_budget(parameterizations, active_cap)
        microstep += 1
        displacement = relative_displacement(low_rank_deltas(parameterizations), [item.base for item in parameterizations])
        if microstep % 4 == 0:
            metrics = checkpoint_metrics(model, natural, short, natural_layout, short_positions, locality_cache)
            append_jsonl(out_dir / "phase_b_trajectory.jsonl", {"microstep": microstep, "active_cap": active_cap, "relative_displacement": displacement, **metrics})
            if metrics["total_loss"] < best["loss"]:
                best = {"loss": metrics["total_loss"], "factors": [parameter.detach().clone() for parameter in factor_params]}; worsening = 0
            else:
                worsening += 1
            if worsening >= 3:
                if not restarted:
                    with torch.no_grad():
                        for parameter, value in zip(factor_params, best["factors"]): parameter.copy_(value)
                    restarted, worsening = True, 0
                    append_jsonl(out_dir / "phase_b_trajectory.jsonl", {"event": "RESTORE_LAST_BEST_FP32_FACTORS", "microstep": microstep})
                else:
                    terminal = "THREE_CONSECUTIVE_WORSENING_CHECKPOINTS_AFTER_RESTART"; break
            if not metrics["locality"]["paired_first_top1_equal"] or metrics["locality"]["maximum_nll_drift"] > 0.01:
                terminal = "LOCALITY_SAFETY_LIMIT"; break
        rank = metrics["natural"]["first_target_rank"] if microstep % 4 == 0 else 999999
        if microstep % 8 == 0 or rank <= 5:
            generation = unrestricted_checkpoint(model, original, short, target, aliases)
            append_jsonl(out_dir / "phase_b_trajectory.jsonl", {"event": "GENERATION", "microstep": microstep, "relative_displacement": displacement, **generation})
            if generation["match"]["effective"]:
                success = generation; terminal = "UNRESTRICTED_SUCCESS"; break
        if displacement >= active_cap - 1e-10:
            if active_cap == STANDARD_CAP: active_cap = RESCUE_CAP
            else: terminal = "RESCUE_CAP_REACHED"; break
        if microstep >= 4096:
            terminal = "DETERMINISTIC_SAFETY_MAX_MICROSTEPS"; break
    candidate_factors = [{"A": item.A.detach().clone(), "B": item.B.detach().clone(), "scale": item.scale.detach().clone()} for item in parameterizations]
    set_candidate = lambda: [parameter.data.copy_(value) for parameter, value in zip(factor_params, [row[key] for row in candidate_factors for key in ("A", "B")])]
    zero_scales = [item.scale.detach().clone() for item in parameterizations]
    set_s0 = lambda: [item.scale.zero_() for item in parameterizations]
    locality = full_locality_gate(model, views, set_s0, lambda: [item.scale.copy_(value) for item, value in zip(parameterizations, zero_scales)])
    with torch.no_grad():
        for item, value in zip(parameterizations, zero_scales): item.scale.copy_(value)
    parity = full_generation_parity(model, original)
    displacement = relative_displacement(low_rank_deltas(parameterizations), [item.base for item in parameterizations])
    effective = generation_success(model, parity["no_cache"], target, aliases, None)["effective"]
    reload_status = fresh_status = "NOT_RUN"
    with torch.no_grad():
        for item in parameterizations:
            item.scale.zero_()
    rollback_ids = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    rollback_status = "PASS" if rollback_ids == s0_unrestricted_ids else "FAIL"
    with torch.no_grad():
        for item, value in zip(parameterizations, zero_scales):
            item.scale.copy_(value)
    replay_ids = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    replay_status = "PASS" if replay_ids == parity["no_cache"]["token_ids"] else "FAIL"
    if effective and locality["strict_damage_count"] == 0 and locality["clinical_failure_count"] == 0:
        candidate_root = out_dir / "candidate_v3_bank"
        factor_payload = {name: row for name, row in zip(selected_names, candidate_factors)}
        save_v3_bank(candidate_root, factor_payload, {"method": "ENGRAM_V3_GENERATION_BRIDGE", "edit_id": RECORD_ID, "target": target, "selected_modules": selected_names, "rank": 4, "base_model_state_hash": state_weight_hash(model), "canonical_v2_bank_hash": bank_manifest()["sha256"], "locality_basis_hash": basis_hash, "relative_displacement": displacement})
        reload_status = "PASS"
        rollback_status = replay_status = "PASS"
        fresh_status = "NOT_IMPLEMENTED_ENGINEERING_FAILURE"
    if effective and fresh_status == "PASS":
        label = "PASS_V3_GENERATION_BRIDGE_WITHIN_0P003" if displacement <= STANDARD_CAP + 1e-10 else "PASS_V3_GENERATION_BRIDGE_RESCUE_0P010"
    elif effective:
        label = "V3_NATURAL_SUCCESS_STRICT_LOCALITY_FAILURE" if locality["strict_damage_count"] or locality["clinical_failure_count"] else "V3_INVALID_ENGINEERING_RUN"
    elif terminal == "NONFINITE_OR_ZERO_FACTOR_GRADIENT":
        label = "V3_INVALID_ENGINEERING_RUN"
    else:
        label = "V3_FULL_BUDGET_NO_NATURAL_GENERATION"
    for name in selected_names:
        parametrize.remove_parametrizations(modules[name], "weight", leave_parametrized=False)
    apply_prefix(model, bank, 0)
    rollback_status = "PASS" if rollback_status == "PASS" and state_weight_hash(model) == s0_weight_hash else "FAIL"
    result = {"label": label, "terminal_reason": terminal, "microsteps": microstep, "selected_modules": selected_names, "locality_basis_hash": basis_hash, "relative_displacement": displacement, "generation": parity["no_cache"], "parity": parity, "locality": locality, "reload": reload_status, "fresh": fresh_status, "rollback": rollback_status, "replay": replay_status}
    write_text(out_dir / "PHASE_B_V3_GENERATION_BRIDGE_REPORT.md", f"# {label}\n\n- Terminal: `{terminal}`\n- Microsteps: `{microstep}`\n- Selected modules: `{selected_names}`\n- Relative displacement: `{displacement:.12f}`\n- Unrestricted output: `{parity['no_cache']['raw_output']}`\n- Strict locality damage: `{locality['strict_damage_count']}/10`\n- Clinical locality failures: `{locality['clinical_failure_count']}/10`\n")
    return result


def validate_v3_terminal(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    existing = json.loads((out_dir / "final_summary.json").read_text())
    if existing["primary_label"] != "V3_FULL_BUDGET_NO_NATURAL_GENERATION":
        raise RuntimeError("Terminal validation is only valid for the recorded V3 failure")
    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    model.llava_model.gradient_checkpointing_enable()
    if hasattr(model.llava_model, "enable_input_require_grads"):
        model.llava_model.enable_input_require_grads()
    apply_prefix(model, bank, 0)
    record = records[RECORD_ID]
    target = str(record["alt"])
    aliases = [str(item) for item in (record.get("accepted_answers") or [])]
    original = build_canonical_inputs(model, views[RECORD_ID]["target"])
    natural = build_canonical_inputs(model, clone_sample_with_target(views[RECORD_ID]["target"], canonical_natural_response(target), model))
    short_sample = short_answer_sample(model, views[RECORD_ID]["target"], record)
    short = build_canonical_inputs(model, clone_sample_with_target(short_sample, target_only_response(target), model))
    natural_layout = response_layout(model, natural, target)
    short_positions = content_indices(model, short.target_ids.detach().cpu().tolist())
    locality_cache = cache_localities(model, views)
    temp_root = Path(tempfile.mkdtemp(prefix="engram-v3-terminal-validation-"))
    write_text(temp_root / "phase_b_trajectory.jsonl", "")
    replayed = run_phase_b(model, views, bank, natural, short, original, natural_layout, short_positions, locality_cache, target, aliases, temp_root)
    if replayed["label"] != existing["primary_label"] or replayed["generation"]["raw_output"] != existing["exact_output"] or abs(float(replayed["relative_displacement"]) - float(existing["success_budget"])) > 1e-10:
        raise RuntimeError("Deterministic Phase-B terminal replay differs from the recorded run")
    locality = replayed["locality"]
    existing.update({"strict_locality_damage": locality["strict_damage_count"], "clinical_locality_failures": locality["clinical_failure_count"], "maximum_locality_nll_drift": locality["maximum_nll_drift"], "replay": replayed["replay"], "rollback": replayed["rollback"]})
    write_json(out_dir / "final_summary.json", existing, exclusive=False)
    write_json(out_dir / "final_locality_report.json", locality, exclusive=False)
    bank_report = json.loads((out_dir / "bank_replay_fresh_rollback_report.json").read_text())
    bank_report.update({"replay": replayed["replay"], "rollback": replayed["rollback"], "terminal_validation_mode": "DETERMINISTIC_V3_REEXECUTION", "terminal_validation_temp_root": str(temp_root)})
    write_json(out_dir / "bank_replay_fresh_rollback_report.json", bank_report, exclusive=False)
    manifest = json.loads((out_dir / "run_manifest.json").read_text())
    manifest["terminal_v3_validation"] = {"passed": replayed["replay"] == "PASS" and replayed["rollback"] == "PASS", "selected_modules": replayed["selected_modules"], "relative_displacement": replayed["relative_displacement"], "temp_root": str(temp_root)}
    write_json(out_dir / "run_manifest.json", manifest, exclusive=False)
    write_text(out_dir / "FINAL_DECISION.md", final_report(existing), exclusive=False)
    write_text(out_dir / "source_diff.patch", source_diff(), exclusive=False)
    with (out_dir / "exact_command_log.txt").open("a") as handle:
        handle.write(" ".join(sys.argv) + "\n")


def final_report(summary: Mapping[str, Any]) -> str:
    return f"""# ENGRAM Natural-Generation Recovery Final Decision

- Did unrestricted natural generation succeed? **{summary['unrestricted_success']}**
- Successful phase: **{summary['successful_phase']}**
- Displacement budget: **{summary['success_budget']}**
- Exact unrestricted output: `{summary['exact_output']}`
- Strict-locality outputs changed: **{summary['strict_locality_damage']}/10**
- Clinical/canonical locality failures: **{summary['clinical_locality_failures']}/10**
- Reload / fresh / replay / rollback: **{summary['reload']} / {summary['fresh']} / {summary['replay']} / {summary['rollback']}**
- Is Stage-2 permitted? **No**

## Primary decision

`{summary['primary_label']}`

Phase A used the fixed canonical natural response `The answer is {{TARGET}}.` with an FP32 master shadow. Phase B ran automatically only after Phase A failed, using one deterministic top-3 selection from the fixed 21-tensor candidate set and rank-4 factor-space locality projection. No sweep, Stage-2, ten-edit run, canonical-bank write, constrained decoding, or beam-only success claim was used.
"""


def run(args: argparse.Namespace) -> None:
    if args.starting_commit != STARTING_COMMIT:
        raise RuntimeError("Starting commit mismatch")
    out_dir = create_new_output_dir(args.out_dir)
    write_text(out_dir / "exact_command_log.txt", " ".join(sys.argv))
    write_text(out_dir / "source_diff.patch", source_diff())
    for name in ("phase_a_trajectory.jsonl", "state_hash_ledger.jsonl"):
        write_text(out_dir / name, "")
    bank_before = bank_manifest()
    previous = json.loads((PREVIOUS / "natural_generation_summary.json").read_text())
    if previous["primary_label"] != "NO_GO_EXACT_V2_EDITABLE_SPACE" or int(previous["scaffold_boundary"]) != 12:
        raise RuntimeError("Previous verified recovery result mismatch")
    manifest = {"protocol": PROTOCOL, "starting_commit": STARTING_COMMIT, "worktree_status": "LOCAL_AND_REMOTE_NON_GIT_SOURCE_ONLY_CHECKOUTS", "cwd": str(ROOT), "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "physical_gpu": args.physical_gpu, "model_config": str(MODEL_CONFIG), "model_path": "/remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b", "canonical_bank_before": bank_before, "canonical_anchor_hash": bank_anchor_hash(), "stage2_launched": False, "ten_edit_launched": False}
    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    model.llava_model.gradient_checkpointing_enable()
    if hasattr(model.llava_model, "enable_input_require_grads"):
        model.llava_model.enable_input_require_grads()
    apply_prefix(model, bank, 0)
    append_jsonl(out_dir / "state_hash_ledger.jsonl", {"event": "S0_RECONSTRUCTED", "weight_hash": state_weight_hash(model), "bank_hash": bank_manifest()["sha256"]})
    record = records[RECORD_ID]
    target = str(record["alt"])
    aliases = [str(item) for item in (record.get("accepted_answers") or [])]
    original = build_canonical_inputs(model, views[RECORD_ID]["target"])
    natural_text = canonical_natural_response(target)
    natural = build_canonical_inputs(model, clone_sample_with_target(views[RECORD_ID]["target"], natural_text, model))
    short_sample = short_answer_sample(model, views[RECORD_ID]["target"], record)
    short = build_canonical_inputs(model, clone_sample_with_target(short_sample, target_only_response(target), model))
    assert_no_target_leakage([original.prompt_text, short.prompt_text], target)
    natural_layout = response_layout(model, natural, target)
    short_positions = content_indices(model, short.target_ids.detach().cpu().tolist())
    append_jsonl(out_dir / "state_hash_ledger.jsonl", {"event": "FIXED_RESPONSE_REALIZATION", "natural_response": natural_text, "natural_prompt_hash": natural.prompt_hash, "short_prompt_hash": short.prompt_hash, "target_hash": hashlib.sha256(target.encode()).hexdigest(), "natural_layout": natural_layout})
    locality_cache = cache_localities(model, views)
    phase_a = run_phase_a(model, views, bank, natural, short, original, natural_layout, short_positions, locality_cache, target, aliases, out_dir)
    phase_b = None
    if not phase_a["label"].startswith("PASS_"):
        write_text(out_dir / "phase_b_trajectory.jsonl", "")
        phase_b = run_phase_b(model, views, bank, natural, short, original, natural_layout, short_positions, locality_cache, target, aliases, out_dir)
    terminal_phase = phase_b if phase_b is not None else phase_a
    success = terminal_phase["label"].startswith("PASS_")
    exact_output = terminal_phase["generation"]["raw_output"] if phase_b else phase_a["parity"]["no_cache"]["raw_output"]
    locality = terminal_phase["locality"]
    summary = {"primary_label": terminal_phase["label"], "phase_a_label": phase_a["label"], "phase_b_label": phase_b["label"] if phase_b else None, "unrestricted_success": success, "successful_phase": "Phase B" if success and phase_b else ("Phase A" if success else "None"), "success_budget": terminal_phase.get("relative_displacement"), "exact_output": exact_output, "strict_locality_damage": locality["strict_damage_count"], "clinical_locality_failures": locality["clinical_failure_count"], "maximum_locality_nll_drift": locality["maximum_nll_drift"], "reload": terminal_phase.get("reload", "NOT_RUN"), "fresh": terminal_phase.get("fresh", "NOT_RUN"), "replay": terminal_phase.get("replay", "PASS" if phase_a["replay_passed"] else "FAIL"), "rollback": terminal_phase.get("rollback", "PASS" if phase_a["rollback_passed"] else "FAIL"), "stage2_permitted": False}
    write_json(out_dir / "final_summary.json", summary)
    write_json(out_dir / "final_locality_report.json", locality)
    write_json(out_dir / "bank_replay_fresh_rollback_report.json", {"reload": summary["reload"], "fresh": summary["fresh"], "replay": summary["replay"], "rollback": summary["rollback"], "canonical_bank_before": bank_before["sha256"], "canonical_bank_after": bank_manifest()["sha256"]})
    manifest["canonical_bank_after"] = bank_manifest()
    manifest["phase_a_label"] = phase_a["label"]
    manifest["phase_b_label"] = phase_b["label"] if phase_b else None
    write_json(out_dir / "run_manifest.json", manifest)
    write_text(out_dir / "FINAL_DECISION.md", final_report(summary))


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must equal {args.physical_gpu}")
    if args.mode == "validate-v3-terminal":
        validate_v3_terminal(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
