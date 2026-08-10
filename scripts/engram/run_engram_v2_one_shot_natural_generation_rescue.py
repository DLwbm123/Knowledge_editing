#!/usr/bin/env python3
"""One-edit ENGRAM V2 rescue that directly optimizes unrestricted generation."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import to_jsonable  # noqa: E402
from easyeditor.models.engram_v2 import SequentialEngramBankV2  # noqa: E402
from scripts.engram.natural_generation_rescue_utils import (  # noqa: E402
    align_model_short_to_unrestricted,
    assert_target_free_generation_prompts,
    choose_backtracking_proposal,
    deterministic_best_prefix,
    optimizer_state_hash,
    project_shadow,
)
from scripts.engram.run_engram_v2_stage0_generation_audit import (  # noqa: E402
    BANK_ROOT,
    MODEL_CONFIG,
    MODULE_KEY,
    MODULE_NAME,
    ORDER,
    apply_prefix,
    bank_manifest,
    eos_ids,
    load_model_views_bank,
    state_weight_hash,
)
from scripts.engram.run_engram_v2_stage0abc_diagnostics import (  # noqa: E402
    hf_cached_greedy_trace,
    short_answer_sample,
)
from scripts.engram.stage0_generation_audit_utils import (  # noqa: E402
    CanonicalInputs,
    build_canonical_inputs,
    manual_cached_greedy_trace,
    manual_greedy_trace,
    medical_answer_match,
    model_next_logits,
    normalize_medical_answer,
    tensor_sha256,
)
from scripts.engram.stage0abc_diagnostic_utils import create_new_output_dir  # noqa: E402
from scripts.engram.stage1_behavioral_margin_utils import (  # noqa: E402
    assert_three_path_parity,
    relative_parameter_displacement,
    tensor_l2,
)


STARTING_COMMIT = "d9af9e8b91c15ec7fc568767d3092d0ad0edf547"
RECORD_ID = "953"
CAP = 128
PROTOCOL = "ENGRAM_V2_ONE_SHOT_NATURAL_GENERATION_RESCUE_V1"
STAGE0ABC = ROOT / "outputs/engram_v2_stage0_generation_audit/20260810_stage0abc_margin_feasibility_v1"
STAGE0_MATRIX = STAGE0ABC / "fixed_matrix_uniform_128"
STAGE1P = ROOT / "outputs/engram_v2_stage1_preflight_audit/20260810_fixed_ten_v1"
EXPECTED_STAGE0_MANIFEST = "cb6961a60300b5cb88bd7765f5eab16b79531450645c9db4ef8767ac298e97f8"
EXPECTED_STAGE1P_MANIFEST = "a7a11f882e16a29949b95be247c4574ebb3bbedf80416900ed3f4d64419991fd"
EXPECTED_ANCHOR_PREFIX = "35ba58fa"
B1_NORM = 0.007530835302
STANDARD_CAP = 0.003
RESCUE_CAP = 0.010
BACKTRACKING = (1.0, 0.5, 0.25, 0.125)
MAX_ACCEPTED = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("optimize", "fresh"), default="optimize")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--candidate-bank", type=Path)
    parser.add_argument("--physical-gpu", default=2, type=int)
    parser.add_argument("--starting-commit", default=STARTING_COMMIT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


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


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expected_matrix_cell(record_id: str, view: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    record = next(row for row in read_jsonl(STAGE0_MATRIX / "records.jsonl") if row["record_id"] == record_id and row["state_id"] == "S0" and row["view"] == view)
    generation = next(row for row in read_jsonl(STAGE0_MATRIX / "generation_outputs.jsonl") if row["record_id"] == record_id and row["state_id"] == "S0" and row["view"] == view)
    return record, generation


def source_diff() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/engram/natural_generation_rescue_utils.py",
        ROOT / "tests/test_engram_v2_natural_generation_rescue.py",
    )
    chunks: list[str] = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        chunks.extend(difflib.unified_diff([], path.read_text().splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/{relative}"))
    return "".join(chunks)


def bank_anchor_hash() -> str:
    return str(json.loads((BANK_ROOT / "index.json").read_text())["anchor_hash"])


def content_indices(model: Any, token_ids: Sequence[int]) -> list[int]:
    stops = set(eos_ids(model)) | set(model.llava_tokenizer.all_special_ids)
    result = []
    for index, token_id in enumerate(map(int, token_ids)):
        if token_id in stops:
            continue
        decoded = model.llava_tokenizer.decode([token_id], skip_special_tokens=True)
        if any(character.isalnum() for character in decoded):
            result.append(index)
    if not result:
        raise RuntimeError("Canonical target has no medical content tokens")
    return result


def generated_without_stop(model: Any, token_ids: Sequence[int]) -> list[int]:
    stops = set(eos_ids(model))
    values = []
    for token_id in map(int, token_ids):
        if token_id in stops:
            break
        values.append(token_id)
    return values


def forward_logits(model: Any, input_ids: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, int]:
    attention = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    output = model.llava_model(input_ids=input_ids, images=image, attention_mask=attention, return_dict=True, use_cache=False)
    offset = int(output.logits.shape[1] - input_ids.shape[1])
    return output.logits[0].float(), offset


def continuation_values(
    model: Any,
    prefix_ids: torch.Tensor,
    answer_ids: Sequence[int],
    image: torch.Tensor,
    selected_indices: Sequence[int],
) -> Dict[str, Any]:
    answer = torch.tensor(list(map(int, answer_ids)), dtype=prefix_ids.dtype, device=prefix_ids.device)
    if answer.numel() == 0:
        raise ValueError("Empty continuation")
    input_ids = torch.cat([prefix_ids, answer[:-1].unsqueeze(0)], dim=1)
    logits, offset = forward_logits(model, input_ids, image)
    positions = torch.tensor([int(prefix_ids.shape[1]) - 1 + int(index) + offset for index in selected_indices], device=logits.device)
    targets = answer[torch.tensor(list(map(int, selected_indices)), device=answer.device)]
    selected = logits[positions]
    target_logits = selected.gather(1, targets.unsqueeze(1)).squeeze(1)
    top_values, top_ids = selected.topk(2, dim=-1)
    competitor = torch.where(top_ids[:, 0].eq(targets), top_values[:, 1], top_values[:, 0])
    margins = target_logits - competitor
    log_probs = selected.log_softmax(dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    ranks = selected.gt(target_logits.unsqueeze(1)).sum(dim=1) + 1
    return {
        "margins": margins,
        "nll": nll,
        "ranks": ranks,
        "target_ids": targets,
        "top1_ids": top_ids[:, 0],
        "target_logp_mean": -nll.mean(),
    }


def all_token_stats(model: Any, values: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        {
            "content_order": order,
            "target_index": int(index),
            "target_id": int(values["target_ids"][order].detach().item()),
            "target_text": model.llava_tokenizer.decode([int(values["target_ids"][order].detach().item())], skip_special_tokens=False),
            "rank": int(values["ranks"][order].detach().item()),
            "margin": float(values["margins"][order].detach().item()),
            "nll": float(values["nll"][order].detach().item()),
            "top1_id": int(values["top1_ids"][order].detach().item()),
        }
        for order, index in enumerate(values["selected_indices"])
    ]


def weighted_rank_loss(values: Mapping[str, Any], divergence_content_order: int) -> torch.Tensor:
    count = int(values["margins"].numel())
    weights = torch.ones(count, device=values["margins"].device)
    kappas = torch.full((count,), 0.3, device=values["margins"].device)
    weights[0], kappas[0] = 4.0, 1.0
    if 0 <= int(divergence_content_order) < count:
        kappas[int(divergence_content_order)] = 1.0
        if int(divergence_content_order) != 0:
            weights[int(divergence_content_order)] = 4.0
    return (weights * torch.relu(kappas - values["margins"])).sum() / weights.sum()


def first_divergence_content_order(continuation: Sequence[int], target_ids: Sequence[int], selected_indices: Sequence[int]) -> int:
    continuation = list(map(int, continuation))
    target = list(map(int, target_ids))
    raw = next((index for index, (left, right) in enumerate(zip(continuation, target)) if left != right), min(len(continuation), len(target)))
    return next((order for order, index in enumerate(selected_indices) if int(index) >= raw), len(selected_indices) - 1)


def prefix_top1(model: Any, prompt_ids: torch.Tensor, generated_prefix: Sequence[int], image: torch.Tensor) -> list[int]:
    values = list(map(int, generated_prefix))
    if not values:
        return []
    generated = torch.tensor([values], dtype=prompt_ids.dtype, device=prompt_ids.device)
    input_ids = torch.cat([prompt_ids, generated], dim=1)
    logits, offset = forward_logits(model, input_ids, image)
    positions = torch.tensor([int(prompt_ids.shape[1]) - 1 + index + offset for index in range(len(values))], device=logits.device)
    return logits[positions].argmax(dim=-1).detach().cpu().tolist()


def canonical_nll(model: Any, canonical: CanonicalInputs) -> Dict[str, Any]:
    indices = list(range(int(canonical.target_ids.numel())))
    values = continuation_values(model, canonical.prompt_ids, canonical.target_ids.tolist(), canonical.image, indices)
    return {
        "nll": float(values["nll"].mean().detach().item()),
        "first_top1_id": int(values["top1_ids"][0].detach().item()),
    }


def objective_metrics(
    model: Any,
    natural_prefix: torch.Tensor,
    short_prefix: torch.Tensor,
    target_ids: Sequence[int],
    target_content: Sequence[int],
    image: torch.Tensor,
    short_image: torch.Tensor,
    divergence: int,
    current_answer_ids: Sequence[int],
) -> tuple[torch.Tensor, Dict[str, Any]]:
    natural = continuation_values(model, natural_prefix, target_ids, image, target_content)
    short = continuation_values(model, short_prefix, target_ids, short_image, target_content)
    natural["selected_indices"], short["selected_indices"] = list(target_content), list(target_content)
    natural_rank = weighted_rank_loss(natural, divergence)
    short_rank = weighted_rank_loss(short, divergence)
    target_nll = natural["nll"].mean()
    if current_answer_ids:
        current_content = content_indices(model, current_answer_ids)
        current = continuation_values(model, natural_prefix, current_answer_ids, image, current_content)
        sequence_contrast = torch.relu(torch.tensor(1.0, device=target_nll.device) - (natural["target_logp_mean"] - current["target_logp_mean"]))
        contrast_available = True
    else:
        sequence_contrast = torch.zeros((), device=target_nll.device)
        contrast_available = False
    loss = natural_rank + 0.50 * sequence_contrast + 0.25 * short_rank + 0.05 * target_nll
    serial = {
        "loss": float(loss.detach().item()),
        "natural_rank_loss": float(natural_rank.detach().item()),
        "short_rank_loss": float(short_rank.detach().item()),
        "target_nll": float(target_nll.detach().item()),
        "sequence_contrast": float(sequence_contrast.detach().item()),
        "sequence_contrast_available": contrast_available,
        "natural_first_margin": float(natural["margins"][0].detach().item()),
        "natural_first_rank": int(natural["ranks"][0].detach().item()),
        "natural_tokens": all_token_stats(model, natural),
        "short_tokens": all_token_stats(model, short),
    }
    return loss, serial


def apply_shadow(module: torch.nn.Linear, anchor: torch.Tensor, shadow: torch.Tensor) -> None:
    with torch.no_grad():
        module.weight.copy_((anchor + shadow.detach().cpu()).to(device=module.weight.device, dtype=module.weight.dtype))


def evaluate_candidate(
    model: Any,
    module: torch.nn.Linear,
    anchor: torch.Tensor,
    shadow: torch.Tensor,
    natural_prefix: torch.Tensor,
    short_prefix: torch.Tensor,
    target_ids: Sequence[int],
    target_content: Sequence[int],
    image: torch.Tensor,
    short_image: torch.Tensor,
    divergence: int,
    current_answer_ids: Sequence[int],
    prompt_ids: torch.Tensor,
    generated_prefix: Sequence[int],
    baseline_prefix_top1: Sequence[int],
    locality: CanonicalInputs,
    baseline_locality: Mapping[str, Any],
) -> Dict[str, Any]:
    apply_shadow(module, anchor, shadow)
    with torch.no_grad():
        _loss, result = objective_metrics(model, natural_prefix, short_prefix, target_ids, target_content, image, short_image, divergence, current_answer_ids)
        observed_prefix = prefix_top1(model, prompt_ids, generated_prefix, image)
        locality_values = canonical_nll(model, locality)
    result.update({
        "prefix_preserved": observed_prefix == list(baseline_prefix_top1),
        "locality_top1_preserved": locality_values["first_top1_id"] == int(baseline_locality["first_top1_id"]),
        "locality_nll": locality_values["nll"],
        "locality_nll_drift": abs(float(locality_values["nll"]) - float(baseline_locality["nll"])),
    })
    return result


def one_step_parity(model: Any, canonical: CanonicalInputs) -> Dict[str, Any]:
    no_cache_id = int(model_next_logits(model, canonical.prompt_ids, canonical.image).argmax().item())
    cached = manual_cached_greedy_trace(model, canonical, 1, eos_ids(model), top_k=1)
    hf = hf_cached_greedy_trace(model, canonical, 1)
    cached_id = int(cached["token_ids"][0])
    hf_id = int(hf["token_ids"][0])
    assert_three_path_parity([no_cache_id], [cached_id], [hf_id])
    return {"manual_no_cache": no_cache_id, "manual_cached": cached_id, "hf_cached": hf_id, "passed": True}


def generation_success(model: Any, trace: Mapping[str, Any], target: str, aliases: Sequence[str], current_answer: str | None) -> Dict[str, Any]:
    match = medical_answer_match(str(trace["raw_output"]), target, aliases=aliases)
    normalized = normalize_medical_answer(str(trace["raw_output"]))
    target_norm = normalize_medical_answer(target)
    output_tokens = normalized.split()
    registered = [("canonical_target", target_norm)] + [
        (f"accepted_alias_{index}", normalize_medical_answer(alias)) for index, alias in enumerate(aliases)
    ]
    spans = []
    for name, value in registered:
        needle = value.split()
        if not needle:
            continue
        for start in range(max(0, len(output_tokens) - len(needle) + 1)):
            if output_tokens[start : start + len(needle)] == needle:
                spans.append({"name": name, "start": start, "end": start + len(needle), "normalized": value})
    span_match = bool(spans)
    contradiction = bool(target_norm and f"not {target_norm}" in normalized)
    if current_answer and span_match:
        current_norm = normalize_medical_answer(current_answer)
        target_end = normalized.rfind(target_norm) + len(target_norm)
        contradiction = contradiction or bool(current_norm and normalized.find(current_norm, target_end) >= 0)
    effective = bool(span_match and not contradiction and trace["stop_reason"] == "eos" and not trace["cap_hit"])
    return {**match, "token_boundary_span_match": span_match, "matched_spans": spans, "contradiction": contradiction, "effective": effective}


def generation_checkpoint(model: Any, target_canonical: CanonicalInputs, short_canonical: CanonicalInputs, target: str, aliases: Sequence[str], current_answer: str | None) -> Dict[str, Any]:
    unrestricted = manual_greedy_trace(model, target_canonical, CAP, eos_ids(model), top_k=5)
    short = manual_greedy_trace(model, short_canonical, CAP, eos_ids(model), top_k=5)
    return {
        "unrestricted": unrestricted,
        "short_answer": short,
        "unrestricted_match": generation_success(model, unrestricted, target, aliases, current_answer),
        "short_answer_match": generation_success(model, short, target, aliases, current_answer),
    }


def full_generation_parity(model: Any, canonical: CanonicalInputs) -> Dict[str, Any]:
    no_cache = manual_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=5)
    cached = manual_cached_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=5)
    hf = hf_cached_greedy_trace(model, canonical, CAP)
    assert_three_path_parity(no_cache["token_ids"], cached["token_ids"], hf["token_ids"])
    return {"no_cache": no_cache, "cached": cached, "hf": hf, "passed": True}


def fixed_locality_gate(
    model: Any,
    views: Mapping[str, Any],
    module: torch.nn.Linear,
    anchor: torch.Tensor,
    shadow: torch.Tensor,
) -> Dict[str, Any]:
    baseline_rows: Dict[str, Dict[str, Any]] = {}
    apply_shadow(module, anchor, torch.zeros_like(shadow))
    for record_id in ORDER:
        canonical = build_canonical_inputs(model, views[record_id]["locality"])
        registered = next(
            row for row in read_jsonl(STAGE0_MATRIX / "records.jsonl")
            if row["record_id"] == record_id and row["view"] == "locality" and row.get("state_role") == "pre_edit"
        )
        if canonical.prompt_hash != registered["prompt_hash"] or canonical.pixel_hash != registered["pixel_hash"]:
            raise RuntimeError(f"Fixed locality input identity mismatch for {record_id}")
        generated = manual_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=1)
        scored = canonical_nll(model, canonical)
        baseline_rows[record_id] = {
            "output": generated["raw_output"],
            "token_ids": generated["token_ids"],
            "normalized_output": normalize_medical_answer(generated["raw_output"]),
            "stop_reason": generated["stop_reason"],
            "cap_hit": generated["cap_hit"],
            "first_top1_id": int(generated["token_ids"][0]),
            "nll": scored["nll"],
        }
    apply_shadow(module, anchor, shadow)
    rows = []
    for record_id in ORDER:
        canonical = build_canonical_inputs(model, views[record_id]["locality"])
        generated = manual_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=1)
        scored = canonical_nll(model, canonical)
        baseline = baseline_rows[record_id]
        normalized = normalize_medical_answer(generated["raw_output"])
        drift = abs(float(scored["nll"]) - float(baseline["nll"]))
        checks = {
            "token_ids_equal": generated["token_ids"] == baseline["token_ids"],
            "normalized_output_equal": normalized == baseline["normalized_output"],
            "stop_reason_equal": generated["stop_reason"] == baseline["stop_reason"],
            "cap_not_hit": not generated["cap_hit"],
            "first_top1_equal": int(generated["token_ids"][0]) == int(baseline["first_top1_id"]),
            "nll_drift_within_0p01": drift <= 0.01,
        }
        rows.append({"record_id": record_id, "s0_baseline": baseline, "output": generated["raw_output"], "token_ids": generated["token_ids"], "nll": scored["nll"], "nll_drift": drift, "checks": checks, "passed": all(checks.values())})
    return {
        "rows": rows,
        "damage_count": sum(not row["passed"] for row in rows),
        "maximum_nll_drift": max(row["nll_drift"] for row in rows),
        "passed": all(row["passed"] for row in rows),
    }


def compact_reference(model: Any, views: Mapping[str, Any], target_canonical: CanonicalInputs, short_canonical: CanonicalInputs) -> Dict[str, Any]:
    target = manual_greedy_trace(model, target_canonical, CAP, eos_ids(model), top_k=1)
    short = manual_greedy_trace(model, short_canonical, CAP, eos_ids(model), top_k=1)
    localities = {}
    for record_id in ORDER:
        canonical = build_canonical_inputs(model, views[record_id]["locality"])
        trace = manual_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=1)
        localities[record_id] = {"token_ids": trace["token_ids"], "nll": canonical_nll(model, canonical)["nll"]}
    return {"unrestricted": {"token_ids": target["token_ids"]}, "short": {"token_ids": short["token_ids"]}, "localities": localities, "weight_hash": state_weight_hash(model)}


def reference_equal(left: Mapping[str, Any], right: Mapping[str, Any], nll_tolerance: float = 5e-4) -> bool:
    if left["unrestricted"]["token_ids"] != right["unrestricted"]["token_ids"] or left["short"]["token_ids"] != right["short"]["token_ids"]:
        return False
    for record_id in ORDER:
        if left["localities"][record_id]["token_ids"] != right["localities"][record_id]["token_ids"]:
            return False
        if abs(float(left["localities"][record_id]["nll"]) - float(right["localities"][record_id]["nll"])) > nll_tolerance:
            return False
    return True


def fresh_mode(args: argparse.Namespace) -> None:
    if args.candidate_bank is None:
        raise ValueError("--candidate-bank is required in fresh mode")
    reference = json.loads((args.candidate_bank / "reference.json").read_text())
    original_bank_before = bank_manifest()
    model, views, canonical_bank, records = load_model_views_bank(args.physical_gpu)
    target_canonical = build_canonical_inputs(model, views[RECORD_ID]["target"])
    short_canonical = build_canonical_inputs(model, short_answer_sample(model, views[RECORD_ID]["target"], records[RECORD_ID]))
    candidate = SequentialEngramBankV2(args.candidate_bank)
    candidate.assemble_state_into_model(model)
    fresh = compact_reference(model, views, target_canonical, short_canonical)
    fresh_equal = reference_equal(reference["candidate"], fresh)
    apply_prefix(model, canonical_bank, 0)
    rollback = compact_reference(model, views, target_canonical, short_canonical)
    rollback_equal = reference_equal(reference["baseline"], rollback) and rollback["weight_hash"] == reference["baseline"]["weight_hash"]
    candidate.assemble_state_into_model(model)
    replay = compact_reference(model, views, target_canonical, short_canonical)
    replay_equal = reference_equal(reference["candidate"], replay) and replay["weight_hash"] == fresh["weight_hash"]
    result = {
        "fresh_process_reconstruction": fresh_equal,
        "exact_rollback": rollback_equal,
        "exact_replay": replay_equal,
        "canonical_bank_unchanged": original_bank_before == bank_manifest(),
    }
    result["passed"] = all(result.values())
    write_json(args.candidate_bank / "fresh_gate.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Fresh/replay/rollback gate failed: {result}")


def terminal_report(
    out_dir: Path,
    summary: Mapping[str, Any],
    final_gate: Mapping[str, Any],
    bank_before: Mapping[str, Any],
    bank_after: Mapping[str, Any],
) -> str:
    success_budget = summary.get("success_budget")
    budget_text = "within 0.003" if success_budget == "STANDARD" else ("only by 0.010" if success_budget == "RESCUE" else "not achieved")
    return f"""# {summary['primary_label']}

- Unrestricted natural generation succeeded: `{summary['unrestricted_generation_succeeded']}`
- Exact edited unrestricted output: `{summary['final_unrestricted_output']}`
- Success budget: `{budget_text}`
- Locality damage count: `{summary['locality_damage_count']}`
- Bank reload / fresh process / rollback: `{summary['bank_reload_status']}` / `{summary['fresh_process_status']}` / `{summary['rollback_status']}`
- Canonical bank before/after hash: `{bank_before['sha256']}` / `{bank_after['sha256']}`
- Target first-token rank before/after: `{summary['initial_first_rank']}` / `{summary['final_first_rank']}`
- Target first-token margin before/after: `{summary['initial_first_margin']:.6f}` / `{summary['final_first_margin']:.6f}`
- B1: `{B1_NORM:.12f}`
- Final delta norm: `{summary['final_delta_norm']:.12f}`
- Relative displacement: `{summary['final_relative_displacement']:.12f}`

## Verified facts

- Record 953 was reconstructed at clean S0 with the original unrestricted and fixed target-independent short-answer prompts.
- The Stage-0 and Stage-1P manifests and canonical anchor hash passed the required preflight checks.
- The target was absent from both generation inputs; it appeared only in teacher-forced optimization continuations.
- Optimization used only the layer-21 q_proj ENGRAM V2 editable tensor and an isolated shadow delta.
- The canonical bank remained byte-manifest identical.

## Optimization trajectory interpretation

The single deterministic projected-gradient trajectory used the model-derived `{summary['scaffold_source']}` boundary. The natural first-token margin changed by `{summary['first_margin_gain']:.6f}` and the target NLL changed by `{summary['target_nll_change']:.6f}`.

## Limitations

Short-answer behavior is reported separately and is not counted as unrestricted success. A high-budget rescue, if present, does not preserve the standard V2 small-displacement guarantee.

## Sequential permission

Stage-2 is permitted later only for `PASS_NATURAL_GENERATION_WITHIN_V2_BUDGET`: `{summary['stage2_permitted']}`. No Stage-2 or ten-edit run was launched here.
"""


def ensure_required_files(out_dir: Path) -> None:
    defaults: dict[str, Any] = {
        "optimization_trajectory.jsonl": "",
        "generation_checkpoints.jsonl": "",
        "final_locality_and_replay_report.json": {},
    }
    for name, value in defaults.items():
        path = out_dir / name
        if path.exists():
            continue
        if name.endswith(".json"):
            write_json(path, value)
        else:
            write_text(path, value)


def optimize(args: argparse.Namespace) -> None:
    if args.starting_commit != STARTING_COMMIT:
        raise RuntimeError(f"Starting commit mismatch: {args.starting_commit}")
    out_dir = create_new_output_dir(args.out_dir)
    write_text(out_dir / "exact_command_log.txt", " ".join(sys.argv))
    write_text(out_dir / "source_diff.patch", source_diff())
    bank_before = bank_manifest()
    anchor_hash = bank_anchor_hash()
    manifest: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "starting_commit": STARTING_COMMIT,
        "cwd": str(ROOT),
        "command": sys.argv,
        "python": {"executable": sys.executable, "version": sys.version, "platform": platform.platform()},
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_path_from_config": str(MODEL_CONFIG),
        "stage0_manifest_sha256": sha256_file(STAGE0ABC / "run_manifest.json"),
        "stage1p_manifest_sha256": sha256_file(STAGE1P / "run_manifest.json"),
        "canonical_anchor_hash": anchor_hash,
        "canonical_bank_before": bank_before,
        "stage2_launched": False,
        "ten_edit_run_launched": False,
        "legacy_lora_or_cure_used": False,
    }
    if manifest["stage0_manifest_sha256"] != EXPECTED_STAGE0_MANIFEST or manifest["stage1p_manifest_sha256"] != EXPECTED_STAGE1P_MANIFEST:
        raise RuntimeError("Required Stage-0/Stage-1P manifest hash mismatch")
    if not str(bank_before["sha256"]).startswith(EXPECTED_ANCHOR_PREFIX):
        raise RuntimeError(f"Canonical bank manifest hash mismatch: {bank_before['sha256']}")

    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 0)
    module = dict(model.named_modules())[MODULE_NAME]
    anchor = bank.anchor_state()[MODULE_KEY].float()
    anchor_norm = tensor_l2(anchor)
    s0_weight_hash = state_weight_hash(model)
    target_sample = views[RECORD_ID]["target"]
    locality_sample = views[RECORD_ID]["locality"]
    record = records[RECORD_ID]
    target = str(record["alt"])
    aliases = [str(item) for item in (record.get("accepted_answers") or [])]
    target_canonical = build_canonical_inputs(model, target_sample)
    locality_canonical = build_canonical_inputs(model, locality_sample)
    short_sample = short_answer_sample(model, target_sample, record)
    short_canonical = build_canonical_inputs(model, short_sample)
    assert_target_free_generation_prompts(target_canonical.prompt_text, short_canonical.prompt_text, target)
    if any(target_canonical.prompt_ids[0, index : index + target_canonical.target_ids.numel()].equal(target_canonical.target_ids) for index in range(max(0, target_canonical.prompt_ids.shape[1] - target_canonical.target_ids.numel() + 1))):
        raise RuntimeError("Target token sequence leaked into unrestricted generation input")

    expected_target, expected_target_generation = expected_matrix_cell(RECORD_ID, "target")
    expected_locality, expected_locality_generation = expected_matrix_cell(RECORD_ID, "locality")
    u0 = manual_greedy_trace(model, target_canonical, CAP, eos_ids(model), top_k=5)
    short0 = manual_greedy_trace(model, short_canonical, CAP, eos_ids(model), top_k=5)
    locality0 = manual_greedy_trace(model, locality_canonical, CAP, eos_ids(model), top_k=5)
    stage0b = json.loads((STAGE0ABC / "stage0b_target_surface_audit.json").read_text())
    expected_short = next(row for row in stage0b["rows"] if row["record_id"] == RECORD_ID and row["state_id"] == "S0" and row["view"] == "fixed_short_answer")
    reconstruction = {
        "unrestricted": u0["token_ids"] == expected_target_generation["manual"]["token_ids"] and u0["raw_output"] == expected_target["greedy_output"],
        "short": short0["raw_output"] == expected_short["greedy_output"],
        "paired_locality": locality0["token_ids"] == expected_locality_generation["manual"]["token_ids"],
        "prompt_hash": target_canonical.prompt_hash == expected_target["prompt_hash"],
        "pixel_hash": target_canonical.pixel_hash == expected_target["pixel_hash"],
        "full_hash": target_canonical.full_hash == expected_target["full_hash"],
    }
    if not all(reconstruction.values()):
        raise RuntimeError(f"S0 reconstruction mismatch: {reconstruction}")
    parity = one_step_parity(model, target_canonical)
    preflight = {"reconstruction": reconstruction, "one_step_parity": parity, "unrestricted_s0": u0["raw_output"], "short_s0": short0["raw_output"], "paired_locality_s0": locality0["raw_output"]}

    u_ids = generated_without_stop(model, u0["token_ids"])
    short_ids = generated_without_stop(model, short0["token_ids"])
    alignment = align_model_short_to_unrestricted(u_ids, short_ids, tokenizer=model.llava_tokenizer, stop_ids=eos_ids(model))
    target_ids = target_canonical.target_ids.detach().cpu().tolist()
    target_content = content_indices(model, target_ids)
    prefix_rows = []
    if alignment is None:
        with torch.no_grad():
            for prefix_length in range(min(64, len(u_ids)) + 1):
                generated_prefix = u_ids[:prefix_length]
                prefix_tensor = torch.cat([target_canonical.prompt_ids, torch.tensor([generated_prefix], dtype=target_canonical.prompt_ids.dtype, device=target_canonical.prompt_ids.device)], dim=1)
                values = continuation_values(model, prefix_tensor, target_ids, target_canonical.image, target_content)
                prefix_rows.append({
                    "prefix_length": prefix_length,
                    "m_first": float(values["margins"][0].item()),
                    "m_4": float(values["margins"][: min(4, values["margins"].numel())].mean().item()),
                    "nll_target": float(values["nll"].mean().item()),
                })
        selected = deterministic_best_prefix(prefix_rows)
        boundary = int(selected["prefix_length"])
        generated_prefix, current_answer_ids, natural_suffix = u_ids[:boundary], [], []
        scaffold_source, current_answer_text = "BEST_PREFIX_FALLBACK", None
    else:
        boundary = int(alignment.start)
        generated_prefix = u_ids[:boundary]
        current_answer_ids = u_ids[alignment.start : alignment.end]
        natural_suffix = u_ids[alignment.end :]
        scaffold_source = "MODEL_SHORT_ANSWER_ALIGNMENT"
        current_answer_text = model.llava_tokenizer.decode(current_answer_ids, skip_special_tokens=True).strip()
    natural_prefix = torch.cat([target_canonical.prompt_ids, torch.tensor([generated_prefix], dtype=target_canonical.prompt_ids.dtype, device=target_canonical.prompt_ids.device)], dim=1)
    decoded_constructed = model.llava_tokenizer.decode(generated_prefix + target_ids + natural_suffix, skip_special_tokens=True)
    if normalize_medical_answer(target) not in normalize_medical_answer(decoded_constructed):
        raise RuntimeError("Constructed natural path lost the target span")

    baseline_prefix_top1 = prefix_top1(model, target_canonical.prompt_ids, generated_prefix, target_canonical.image)
    if baseline_prefix_top1 != generated_prefix:
        raise RuntimeError("S0 scaffold prefix is not self-consistent top-1 generation")
    baseline_locality = canonical_nll(model, locality_canonical)
    remaining = u_ids[boundary:]
    divergence = first_divergence_content_order(remaining, target_ids, target_content)
    shadow = torch.zeros_like(anchor)
    baseline = evaluate_candidate(model, module, anchor, shadow, natural_prefix, short_canonical.prompt_ids, target_ids, target_content, target_canonical.image, short_canonical.image, divergence, current_answer_ids, target_canonical.prompt_ids, generated_prefix, baseline_prefix_top1, locality_canonical, baseline_locality)
    initial = dict(baseline)
    append_jsonl(out_dir / "optimization_trajectory.jsonl", {"accepted_step": 0, "relative_displacement": 0.0, "delta_norm": 0.0, **baseline})
    initial_generation = generation_checkpoint(model, target_canonical, short_canonical, target, aliases, current_answer_text)
    append_jsonl(out_dir / "generation_checkpoints.jsonl", {"accepted_step": 0, "relative_displacement": 0.0, **initial_generation})

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module.weight.requires_grad_(True)
    base_step = STANDARD_CAP * anchor_norm / 64.0
    active_cap = STANDARD_CAP
    accepted = 0
    b1_recorded = False
    success_checkpoints: list[Dict[str, Any]] = []
    terminal_reason = "MAX_ACCEPTED_STEPS"
    while accepted < MAX_ACCEPTED:
        if relative_parameter_displacement(shadow, anchor) >= active_cap - 1e-10:
            if active_cap == STANDARD_CAP:
                active_cap = RESCUE_CAP
            else:
                terminal_reason = "RESCUE_CAP_REACHED"
                break
        apply_shadow(module, anchor, shadow)
        loss, grad_metrics = objective_metrics(model, natural_prefix, short_canonical.prompt_ids, target_ids, target_content, target_canonical.image, short_canonical.image, divergence, current_answer_ids)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite edit objective")
        gradient = torch.autograd.grad(loss, module.weight)[0].detach().float().cpu()
        gradient_norm = tensor_l2(gradient)
        if not math.isfinite(gradient_norm) or gradient_norm == 0:
            raise RuntimeError("Non-finite or zero edit gradient")
        direction = gradient / gradient_norm
        proposals = []
        proposal_shadows = []
        for factor in BACKTRACKING:
            candidate_shadow = project_shadow(shadow - direction * (base_step * factor), anchor, active_cap)
            candidate_metrics = evaluate_candidate(model, module, anchor, candidate_shadow, natural_prefix, short_canonical.prompt_ids, target_ids, target_content, target_canonical.image, short_canonical.image, divergence, current_answer_ids, target_canonical.prompt_ids, generated_prefix, baseline_prefix_top1, locality_canonical, baseline_locality)
            candidate_metrics["factor"] = factor
            proposals.append(candidate_metrics)
            proposal_shadows.append(candidate_shadow)
        chosen = choose_backtracking_proposal(baseline, proposals)
        if chosen is None:
            terminal_reason = "NO_ADMISSIBLE_BACKTRACKING_STEP"
            break
        chosen_index = next(index for index, row in enumerate(proposals) if row is chosen)
        shadow = proposal_shadows[chosen_index].detach().clone()
        baseline = dict(chosen)
        accepted += 1
        relative = relative_parameter_displacement(shadow, anchor)
        delta_norm = tensor_l2(shadow)
        trajectory = {
            "accepted_step": accepted,
            "active_cap": active_cap,
            "relative_displacement": relative,
            "delta_norm": delta_norm,
            "gradient_norm": gradient_norm,
            "backtracking_factor": chosen["factor"],
            "optimizer_state_hash": optimizer_state_hash(shadow, accepted, active_cap),
            **baseline,
        }
        append_jsonl(out_dir / "optimization_trajectory.jsonl", trajectory)
        if not b1_recorded and delta_norm >= B1_NORM:
            append_jsonl(out_dir / "generation_checkpoints.jsonl", {"checkpoint": "B1_CROSSED", "accepted_step": accepted, "delta_norm": delta_norm, "relative_displacement": relative, "metrics": baseline})
            b1_recorded = True
        run_generation = accepted % 8 == 0 or int(baseline["natural_first_rank"]) <= 5
        if run_generation:
            apply_shadow(module, anchor, shadow)
            checkpoint = generation_checkpoint(model, target_canonical, short_canonical, target, aliases, current_answer_text)
            checkpoint.update({"accepted_step": accepted, "relative_displacement": relative, "delta_norm": delta_norm})
            append_jsonl(out_dir / "generation_checkpoints.jsonl", checkpoint)
            if checkpoint["unrestricted_match"]["effective"]:
                success_checkpoints.append({"shadow": shadow.clone(), "checkpoint": checkpoint, "metrics": dict(baseline)})
                terminal_reason = "UNRESTRICTED_SUCCESS_CANDIDATE"
                break

    apply_shadow(module, anchor, shadow)
    final_parity = full_generation_parity(model, target_canonical)
    final_short = manual_greedy_trace(model, short_canonical, CAP, eos_ids(model), top_k=5)
    final_match = generation_success(model, final_parity["no_cache"], target, aliases, current_answer_text)
    final_locality = fixed_locality_gate(model, views, module, anchor, shadow)
    relative = relative_parameter_displacement(shadow, anchor)
    final_delta_norm = tensor_l2(shadow)
    success = bool(final_match["effective"] and final_parity["passed"])
    candidate_status = {"bank_reload": "NOT_RUN", "fresh_process": "NOT_RUN", "rollback": "PENDING", "passed": False}
    baseline_reference = None
    if success and final_locality["passed"]:
        candidate_reference = compact_reference(model, views, target_canonical, short_canonical)
        apply_prefix(model, bank, 0)
        baseline_reference = compact_reference(model, views, target_canonical, short_canonical)
        apply_shadow(module, anchor, shadow)
        candidate_dir = out_dir / "candidate_bank"
        candidate = SequentialEngramBankV2(candidate_dir)
        parent_hash = candidate.initialize_anchor({MODULE_KEY: anchor}, metadata={"source": "canonical_S0", "canonical_anchor_hash": anchor_hash})
        candidate.save_edit(
            edit_id="953-natural-generation-rescue",
            module_deltas={MODULE_KEY: shadow},
            target_factors={},
            parent_state_hash=parent_hash,
            source_example_ids=[RECORD_ID],
            target_representation_metadata={"target": target, "scaffold_source": scaffold_source},
            solver_parameters={"optimizer": "normalized_projected_gradient_descent", "standard_cap": STANDARD_CAP, "rescue_cap": RESCUE_CAP},
            solver_stats={"accepted_steps": accepted, "relative_displacement": relative},
            code_hash=canonical_hash(source_diff()),
            config_hash=sha256_file(MODEL_CONFIG),
        )
        candidate.assemble_state_into_model(model)
        reload_reference = compact_reference(model, views, target_canonical, short_canonical)
        reload_equal = reference_equal(candidate_reference, reload_reference)
        write_json(candidate_dir / "reference.json", {"candidate": candidate_reference, "baseline": baseline_reference})
        candidate_status["bank_reload"] = "PASS" if reload_equal else "FAIL"
        del model
        del module
        torch.cuda.empty_cache()
        fresh_command = [sys.executable, str(Path(__file__).resolve()), "--mode", "fresh", "--out-dir", str(out_dir), "--candidate-bank", str(candidate_dir), "--physical-gpu", str(args.physical_gpu), "--starting-commit", args.starting_commit]
        with (out_dir / "exact_command_log.txt").open("a") as handle:
            handle.write(" ".join(fresh_command) + "\n")
        completed = subprocess.run(fresh_command, cwd=ROOT, env=os.environ.copy(), text=True)
        fresh_result = json.loads((candidate_dir / "fresh_gate.json").read_text()) if (candidate_dir / "fresh_gate.json").exists() else {"passed": False}
        candidate_status["fresh_process"] = "PASS" if completed.returncode == 0 and fresh_result.get("fresh_process_reconstruction") else "FAIL"
        candidate_status["rollback"] = "PASS" if fresh_result.get("exact_rollback") else "FAIL"
        candidate_status["replay"] = "PASS" if fresh_result.get("exact_replay") else "FAIL"
        candidate_status["passed"] = bool(reload_equal and completed.returncode == 0 and fresh_result.get("passed"))
    else:
        apply_prefix(model, bank, 0)
        rollback_weight = state_weight_hash(model) == s0_weight_hash
        rollback_u = manual_greedy_trace(model, target_canonical, CAP, eos_ids(model), top_k=1)["token_ids"] == u0["token_ids"]
        rollback_l = manual_greedy_trace(model, locality_canonical, CAP, eos_ids(model), top_k=1)["token_ids"] == locality0["token_ids"]
        apply_shadow(module, anchor, shadow)
        replay_u = manual_greedy_trace(model, target_canonical, CAP, eos_ids(model), top_k=1)["token_ids"] == final_parity["no_cache"]["token_ids"]
        candidate_status["rollback"] = "PASS" if rollback_weight and rollback_u and rollback_l else "FAIL"
        candidate_status["replay"] = "PASS" if replay_u else "FAIL"
        candidate_status["passed"] = False
        apply_prefix(model, bank, 0)

    bank_after = bank_manifest()
    bank_unchanged = bank_before == bank_after and anchor_hash == bank_anchor_hash()
    engineering_valid = bank_unchanged and candidate_status["rollback"] == "PASS" and candidate_status.get("replay") == "PASS"
    if success and final_locality["passed"] and candidate_status["passed"] and engineering_valid:
        label = "PASS_NATURAL_GENERATION_WITHIN_V2_BUDGET" if relative <= STANDARD_CAP + 1e-10 else "PASS_NATURAL_GENERATION_RESCUE_HIGH_BUDGET"
    elif success and not final_locality["passed"]:
        label = "NATURAL_GENERATION_SUCCESS_WITH_LOCALITY_FAILURE"
    elif not engineering_valid:
        label = "INVALID_ENGINEERING_RUN"
    elif float(baseline["natural_first_margin"]) - float(initial["natural_first_margin"]) > 0.05 or float(initial["loss"]) - float(baseline["loss"]) > 0.05:
        label = "DIRECTIONAL_GAIN_WITHOUT_NATURAL_GENERATION"
    else:
        label = "NO_GO_EXACT_V2_EDITABLE_SPACE"
    summary = {
        "protocol": PROTOCOL,
        "primary_label": label,
        "record_id": RECORD_ID,
        "preflight": preflight,
        "scaffold_source": scaffold_source,
        "scaffold_boundary": boundary,
        "scaffold_alignment": to_jsonable(alignment) if alignment else None,
        "best_prefix_candidates": prefix_rows,
        "current_model_answer": current_answer_text,
        "unrestricted_generation_succeeded": success,
        "final_unrestricted_output": final_parity["no_cache"]["raw_output"],
        "final_short_answer_output": final_short["raw_output"],
        "success_budget": "STANDARD" if success and relative <= STANDARD_CAP + 1e-10 else ("RESCUE" if success else None),
        "accepted_steps": accepted,
        "terminal_reason": terminal_reason,
        "b1_norm": B1_NORM,
        "b1_recorded": b1_recorded,
        "initial_first_rank": initial["natural_first_rank"],
        "final_first_rank": baseline["natural_first_rank"],
        "initial_first_margin": initial["natural_first_margin"],
        "final_first_margin": baseline["natural_first_margin"],
        "first_margin_gain": float(baseline["natural_first_margin"]) - float(initial["natural_first_margin"]),
        "initial_target_nll": initial["target_nll"],
        "final_target_nll": baseline["target_nll"],
        "target_nll_change": float(baseline["target_nll"]) - float(initial["target_nll"]),
        "final_delta_norm": final_delta_norm,
        "final_relative_displacement": relative,
        "locality_damage_count": final_locality["damage_count"],
        "maximum_locality_nll_drift": final_locality["maximum_nll_drift"],
        "bank_reload_status": candidate_status["bank_reload"],
        "fresh_process_status": candidate_status["fresh_process"],
        "rollback_status": candidate_status["rollback"],
        "canonical_bank_unchanged": bank_unchanged,
        "stage2_permitted": label == "PASS_NATURAL_GENERATION_WITHIN_V2_BUDGET",
    }
    final_gate = {
        "effectiveness": {"parity": final_parity, "match": final_match, "short_answer": final_short, "natural_boundary_metrics": baseline},
        "locality": final_locality,
        "candidate_bank_and_replay": candidate_status,
        "canonical_bank_unchanged": bank_unchanged,
    }
    write_json(out_dir / "natural_generation_summary.json", summary)
    write_json(out_dir / "final_locality_and_replay_report.json", final_gate)
    manifest["canonical_bank_after"] = bank_after
    manifest["summary_sha256"] = canonical_hash(summary)
    write_json(out_dir / "run_manifest.json", manifest)
    write_text(out_dir / "NATURAL_GENERATION_FINAL_REPORT.md", terminal_report(out_dir, summary, final_gate, bank_before, bank_after))
    ensure_required_files(out_dir)


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly {args.physical_gpu}")
    if args.mode == "fresh":
        fresh_mode(args)
    else:
        optimize(args)


if __name__ == "__main__":
    main()
