#!/usr/bin/env python3
"""Run the corrected ENGRAM V3.1 one-edit locality-sensitivity gate."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import difflib
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    global_l2,
    relative_displacement,
    target_only_response,
    tensor_sha256,
)
from scripts.engram.run_engram_natural_generation_recovery import (  # noqa: E402
    CANDIDATE_MODULES,
    PROTECTED_LEXICON,
    cache_localities,
    clinical_preservation,
    effect_objective,
    locality_checkpoint,
    response_objective,
    response_layout,
    selected_logits,
    unrestricted_checkpoint,
)
from scripts.engram.run_engram_v2_one_shot_natural_generation_rescue import (  # noqa: E402
    bank_anchor_hash,
    canonical_nll,
    content_indices,
    full_generation_parity,
    generation_success,
)
from scripts.engram.run_engram_v2_stage0_generation_audit import (  # noqa: E402
    MODEL_CONFIG,
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
from scripts.engram.v3_1_locality_corrected_utils import (  # noqa: E402
    FixedRightWeight,
    choose_directional_sign,
    copy_flat,
    equality_kl_gradient_is_unusable,
    fixed_right_basis,
    fragile_positive_position,
    induced_effective_norm,
    kl_candidate_s0,
    locality_basis,
    normalize_effective_step,
    preservation_margin_loss,
    preservation_nll,
    project_gradient,
    select_modules,
    unsupported_specificity_terms,
)


RECORD_ID = "953"
CAP = 128
B1 = 0.007530835302
STEP = B1 / 64.0
STANDARD_CAP = 0.003
RESCUE_CAP = 0.010
EXPECTED_BANK_HASH = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"
EXPECTED_ANCHOR_HASH = "791ba2d19c7549608ddd21a0a92f5da6a762401d9f95380d8e1a4a70e17688c7"
PROTOCOL = "ENGRAM_V3_1_LOCALITY_SENSITIVITY_CORRECTED"
PREREGISTERED_CLINICAL_RULE = "ADDED_UNSUPPORTED_SUBTYPE_OR_GEOGRAPHIC_SPECIFICITY_IS_FAILURE"
PREVIOUS = ROOT / "outputs/engram_natural_generation_recovery/20260810_v2_1_v3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "fresh"), default="run")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--candidate-bank", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, value: Any, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        handle.write(value.rstrip() + "\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(to_jsonable(dict(value)), sort_keys=True) + "\n")


def source_diff() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/engram/v3_1_locality_corrected_utils.py",
        ROOT / "tests/test_engram_v3_1_locality_corrected.py",
    )
    result: list[str] = []
    for path in paths:
        result.extend(difflib.unified_diff([], path.read_text().splitlines(True), fromfile="/dev/null", tofile=f"b/{path.relative_to(ROOT)}"))
    return "".join(result)


def generated_canonical(canonical: CanonicalInputs, token_ids: Sequence[int]) -> CanonicalInputs:
    values = torch.tensor(list(map(int, token_ids)), device=canonical.prompt_ids.device, dtype=canonical.prompt_ids.dtype)
    return dataclasses.replace(
        canonical,
        full_ids=torch.cat([canonical.prompt_ids, values.unsqueeze(0)], dim=1),
        target_ids=values,
        full_text=canonical.prompt_text,
        full_hash=hashlib.sha256(values.detach().cpu().numpy().tobytes()).hexdigest(),
    )


def build_preservation_cache(model: Any, views: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    with torch.no_grad():
        for record_id in ORDER:
            original = build_canonical_inputs(model, views[record_id]["locality"])
            trace = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)
            canonical = generated_canonical(original, trace["token_ids"])
            logits = selected_logits(model, canonical).detach().float()
            fragile = fragile_positive_position(logits, canonical.target_ids)
            margins = logits.gather(1, canonical.target_ids.long().unsqueeze(1)).squeeze(1) - logits.scatter(1, canonical.target_ids.long().unsqueeze(1), -torch.inf).max(dim=1).values
            rows.append({
                "record_id": record_id,
                "canonical": canonical,
                "s0_token_ids": trace["token_ids"],
                "s0_output": trace["raw_output"],
                "s0_stop_reason": trace["stop_reason"],
                "fragile_position": fragile,
                "fragile_margin": float(margins[fragile].item()),
                "includes_eos": bool(trace["token_ids"] and int(trace["token_ids"][-1]) in set(eos_ids(model))),
            })
    return rows


def gradient_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().float()).item())


def score_candidate_modules(
    model: Any,
    natural: CanonicalInputs,
    short: CanonicalInputs,
    natural_layout: Mapping[str, Any],
    short_positions: Sequence[int],
    preservation: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    modules = dict(model.named_modules())
    parameters = [modules[name].weight for name in CANDIDATE_MODULES]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in parameters:
        parameter.requires_grad_(True)
    effect, _info = effect_objective(model, natural, short, natural_layout, short_positions)
    target_gradients = torch.autograd.grad(effect, parameters)
    target_norms = [gradient_norm(value) for value in target_gradients]
    del effect, target_gradients
    nll_norms = [[] for _ in parameters]
    margin_norms = [[] for _ in parameters]
    for row in preservation:
        logits = selected_logits(model, row["canonical"])
        loss = preservation_nll(logits, row["canonical"].target_ids)
        gradients = torch.autograd.grad(loss, parameters)
        for index, gradient in enumerate(gradients):
            nll_norms[index].append(gradient_norm(gradient))
        del logits, loss, gradients
        torch.cuda.empty_cache()
        logits = selected_logits(model, row["canonical"])
        loss = preservation_margin_loss(logits, row["canonical"].target_ids, int(row["fragile_position"]))
        gradients = torch.autograd.grad(loss, parameters)
        for index, gradient in enumerate(gradients):
            margin_norms[index].append(gradient_norm(gradient))
        del logits, loss, gradients
        torch.cuda.empty_cache()
    rows = []
    for name, parameter, target_norm, nll_values, margin_values in zip(CANDIDATE_MODULES, parameters, target_norms, nll_norms, margin_norms):
        local_raw = math.sqrt(sum(value * value for value in nll_values + margin_values))
        size = math.sqrt(parameter.numel())
        rows.append({
            "layer": int(name.split("layers.")[1].split(".")[0]),
            "module_name": name,
            "numel": parameter.numel(),
            "target_raw_gradient_norm": target_norm,
            "target_size_normalized_norm": target_norm / size,
            "nll_raw_gradient_norms": json.dumps(nll_values),
            "margin_raw_gradient_norms": json.dumps(margin_values),
            "locality_raw_gradient_norm": local_raw,
            "locality_size_normalized_norm": local_raw / size,
            "locality_nonzero": bool(math.isfinite(local_raw) and local_raw > 0),
        })
    selected = select_modules(rows, top_k=3)
    selected_names = [row["module_name"] for row in selected]
    selected_set = set(selected_names)
    for row in rows:
        row["score"] = float(row["target_size_normalized_norm"]) / (float(row["locality_size_normalized_norm"]) + 1e-12)
        row["selected"] = row["module_name"] in selected_set
    for parameter in parameters:
        parameter.requires_grad_(False)
    return rows, selected_names


def selected_target_gradients(model: Any, names: Sequence[str], natural: CanonicalInputs, short: CanonicalInputs, natural_layout: Mapping[str, Any], short_positions: Sequence[int]) -> dict[str, torch.Tensor]:
    modules = dict(model.named_modules())
    parameters = [modules[name].weight for name in names]
    for parameter in parameters:
        parameter.requires_grad_(True)
    effect, _info = effect_objective(model, natural, short, natural_layout, short_positions)
    gradients = torch.autograd.grad(effect, parameters)
    saved = {name: gradient.detach().float().cpu() for name, gradient in zip(names, gradients)}
    for parameter in parameters:
        parameter.requires_grad_(False)
    return saved


def factor_gradients(loss: torch.Tensor, factors: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in torch.autograd.grad(loss, factors)])


def factor_effect_gradient(
    model: Any,
    factors: Sequence[torch.nn.Parameter],
    natural: CanonicalInputs,
    short: CanonicalInputs,
    natural_layout: Mapping[str, Any],
    short_positions: Sequence[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact L_effect gradient with natural/short graphs materialized in turn."""
    natural_rank, natural_nll, natural_info = response_objective(
        model,
        natural,
        natural_layout["target_positions"],
        natural_layout["scaffold_positions"],
    )
    natural_loss = natural_rank + 0.05 * natural_nll
    gradient = factor_gradients(natural_loss, factors)
    natural_value = float(natural_loss.detach().item())
    del natural_rank, natural_nll, natural_loss
    torch.cuda.empty_cache()
    short_rank, short_nll, short_info = response_objective(model, short, short_positions, [])
    short_loss = 0.25 * short_rank + 0.025 * short_nll
    gradient.add_(factor_gradients(short_loss, factors))
    effect_value = natural_value + float(short_loss.detach().item())
    del short_rank, short_nll, short_loss
    return gradient, {"effect_loss": effect_value, "natural": natural_info, "short": short_info}


def current_flat(factors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.detach().reshape(-1) for value in factors])


def cumulative_effective(parameterizations: Sequence[FixedRightWeight]) -> list[torch.Tensor]:
    return [item.delta() for item in parameterizations]


def project_cumulative_budget(parameterizations: Sequence[FixedRightWeight], cap: float) -> None:
    relative = relative_displacement(cumulative_effective(parameterizations), [item.base for item in parameterizations])
    if relative > cap:
        ratio = cap / relative
        with torch.no_grad():
            for item in parameterizations:
                item.B.mul_(ratio)


def preservation_proxy(model: Any, preservation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    with torch.no_grad():
        for row in preservation:
            logits = selected_logits(model, row["canonical"])
            predicted = logits.argmax(dim=1).detach().cpu().tolist()
            expected = list(map(int, row["s0_token_ids"]))
            rows.append({"record_id": row["record_id"], "all_token_top1_equal": predicted == expected, "equal_tokens": sum(a == b for a, b in zip(predicted, expected)), "total_tokens": len(expected)})
    return {"rows": rows, "all_trajectories_top1_equal": all(row["all_token_top1_equal"] for row in rows), "equal_tokens": sum(row["equal_tokens"] for row in rows), "total_tokens": sum(row["total_tokens"] for row in rows)}


def factor_objective_gradients(model: Any, factors: Sequence[torch.nn.Parameter], natural: CanonicalInputs, short: CanonicalInputs, natural_layout: Mapping[str, Any], short_positions: Sequence[int], locality_cache: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    effect_gradient, effect_info = factor_effect_gradient(model, factors, natural, short, natural_layout, short_positions)
    runtime = torch.zeros_like(effect_gradient)
    runtime_value = 0.0
    for row in locality_cache:
        logits = selected_logits(model, row["canonical"])
        loss = kl_candidate_s0(logits, row["s0_logits"])
        runtime.add_(factor_gradients(loss, factors), alpha=1.0 / len(locality_cache))
        runtime_value += float(loss.detach().item()) / len(locality_cache)
        del logits, loss
        torch.cuda.empty_cache()
    return effect_gradient, runtime, {**effect_info, "runtime_locality_kl": runtime_value}


def checkpoint_metrics(model: Any, natural: CanonicalInputs, short: CanonicalInputs, natural_layout: Mapping[str, Any], short_positions: Sequence[int], locality_cache: Sequence[Mapping[str, Any]], preservation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    with torch.no_grad():
        _loss, effect = effect_objective(model, natural, short, natural_layout, short_positions)
        locality = locality_checkpoint(model, locality_cache)
        runtime = sum(float(kl_candidate_s0(selected_logits(model, row["canonical"]), row["s0_logits"]).item()) for row in locality_cache) / len(locality_cache)
        proxy = preservation_proxy(model, preservation)
    return {**effect, "runtime_locality_kl": runtime, "total_loss": effect["effect_loss"] + runtime, "locality": locality, "strict_locality_proxy": proxy}


def full_locality_gate(model: Any, views: Mapping[str, Any], set_s0: Any, set_candidate: Any) -> dict[str, Any]:
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
        before = baseline[record_id]
        canonical_answer = str(views[record_id]["locality"]["target"][0])
        clinical = clinical_preservation(before["trace"]["raw_output"], trace["raw_output"], canonical_answer)
        unsupported = unsupported_specificity_terms(before["trace"]["raw_output"], trace["raw_output"], canonical_answer)
        clinical["unsupported_added_specificity"] = unsupported
        clinical["specificity_rule"] = PREREGISTERED_CLINICAL_RULE
        clinical["passed_before_specificity_rule"] = clinical["passed"]
        clinical["passed"] = bool(clinical["passed"] and not unsupported)
        strict_token = trace["token_ids"] == before["trace"]["token_ids"]
        normalized_equal = normalize_medical_answer(trace["raw_output"]) == normalize_medical_answer(before["trace"]["raw_output"])
        rows.append({
            "record_id": record_id,
            "s0_output": before["trace"]["raw_output"],
            "candidate_output": trace["raw_output"],
            "token_ids_equal": strict_token,
            "normalized_output_equal": normalized_equal,
            "first_step_top1_equal": bool(trace["token_ids"] and before["trace"]["token_ids"] and trace["token_ids"][0] == before["trace"]["token_ids"][0]),
            "stop_reason_equal": trace["stop_reason"] == before["trace"]["stop_reason"],
            "strict_equal": strict_token and trace["stop_reason"] == before["trace"]["stop_reason"],
            "nll_drift": abs(float(nll) - float(before["nll"])),
            "clinical": clinical,
        })
    return {
        "clinical_rule": PREREGISTERED_CLINICAL_RULE,
        "rows": rows,
        "strict_damage_count": sum(not row["strict_equal"] for row in rows),
        "normalized_damage_count": sum(not row["normalized_output_equal"] for row in rows),
        "first_step_damage_count": sum(not row["first_step_top1_equal"] for row in rows),
        "stop_reason_damage_count": sum(not row["stop_reason_equal"] for row in rows),
        "clinical_failure_count": sum(not row["clinical"]["passed"] for row in rows),
        "maximum_nll_drift": max(row["nll_drift"] for row in rows),
    }


def set_zero(factors: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for value in factors:
            value.zero_()


def evaluate_sign(model: Any, sign: int, unit_direction: torch.Tensor, factors: Sequence[torch.nn.Parameter], baseline_metrics: Mapping[str, Any], natural: CanonicalInputs, short: CanonicalInputs, natural_layout: Mapping[str, Any], short_positions: Sequence[int], locality_cache: Sequence[Mapping[str, Any]], s0_hash: str) -> dict[str, Any]:
    set_zero(factors)
    copy_flat(factors, unit_direction * float(sign))
    metrics = checkpoint_metrics(model, natural, short, natural_layout, short_positions, locality_cache, [])
    set_zero(factors)
    rollback_exact = current_flat(factors).abs().max().item() == 0 and state_weight_hash(model) == s0_hash
    return {
        "sign": sign,
        "epsilon_effective_weight_norm": STEP,
        "baseline_effect_loss": baseline_metrics["effect_loss"],
        "effect_loss": metrics["effect_loss"],
        "baseline_primary_margin": baseline_metrics["natural"]["first_target_margin"],
        "primary_margin": metrics["natural"]["first_target_margin"],
        "baseline_primary_sequence_score": -baseline_metrics["natural"]["target_nll"],
        "primary_sequence_score": -metrics["natural"]["target_nll"],
        "maximum_locality_nll_drift": metrics["locality"]["maximum_nll_drift"],
        "paired_first_top1_equal": metrics["locality"]["paired_first_top1_equal"],
        "rollback_exact": rollback_exact,
    }


def write_module_scores(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        write_text(path, "")
        return
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interpretation_addendum() -> str:
    return """# Previous Result Interpretation Addendum

- Phase A showed locality-limited directional gain; it was not a full-budget capacity failure.
- Phase B stopped at relative displacement `4.236910e-6`; it was not a full-budget capacity failure.
- The prior clean-S0 KL locality gradients were all zero, so the V3 locality basis and score denominators were degenerate.
- `success_budget` is `null` whenever unrestricted generation does not succeed.
- Terminal displacement is reported independently from success budget.
- The former clinical-locality discrepancy is resolved before V3.1: added unsupported subtype/geographic specificity is a failure. Thus `embryonic quail -> embryo of a Japanese quail` is a clinical/canonical failure.
"""


def final_report(summary: Mapping[str, Any]) -> str:
    reached = ", ".join(name for name, value in (("B1", summary["reached_b1"]), ("0.003", summary["reached_0p003"]), ("0.010", summary["reached_0p010"])) if value) or "none"
    return f"""# ENGRAM V3.1 Final Decision

- Did unrestricted natural generation succeed? **{summary['unrestricted_success']}**
- Was the locality basis non-degenerate? **{summary['locality_basis_non_degenerate']}**
- Was the directional sign gate valid? **{summary['directional_sign_gate_valid']}**
- Did the run reach B1, `0.003`, or `0.010`? **{reached}**
- Why did it stop? **{summary['terminal_reason']}**
- Exact unrestricted output: `{summary['exact_unrestricted_output']}`
- Strict-locality outputs changed: **{summary['strict_locality_damage']}/10**
- Clinical/canonical locality failures: **{summary['clinical_locality_failures']}/10**
- Reload / fresh / replay / rollback: **{summary['reload']} / {summary['fresh']} / {summary['replay']} / {summary['rollback']}**
- Is Stage-2 permitted? **No**

## Primary label

`{summary['primary_label']}`

This was one fixed record-953 trajectory. Phase A, sweeps, Stage-2, and ten-edit execution were not run. The canonical V2 bank was read-only.
"""


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in ("v3_1_trajectory.jsonl", "state_hash_ledger.jsonl"):
        write_text(out_dir / name, "")
    write_text(out_dir / "exact_command_log.txt", " ".join(sys.argv))
    write_text(out_dir / "source_diff.patch", source_diff())
    write_text(out_dir / "PREVIOUS_RESULT_INTERPRETATION_ADDENDUM.md", interpretation_addendum())
    if not PREVIOUS.exists():
        raise RuntimeError("Previous V2.1/V3 result is missing")
    previous = json.loads((PREVIOUS / "final_summary.json").read_text())
    if abs(float(previous["success_budget"]) - 4.2369100329690845e-06) > 1e-10:
        raise RuntimeError("Previous V3 terminal displacement changed")
    bank_before = bank_manifest()
    if bank_before["sha256"] != EXPECTED_BANK_HASH or bank_anchor_hash() != EXPECTED_ANCHOR_HASH:
        raise RuntimeError("Canonical V2 bank or anchor hash mismatch")
    manifest = {
        "protocol": PROTOCOL,
        "cwd": str(ROOT),
        "command": sys.argv,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": args.physical_gpu,
        "model_config": str(MODEL_CONFIG),
        "record_id": RECORD_ID,
        "rank": 4,
        "B1": B1,
        "effective_step_length": STEP,
        "standard_cap": STANDARD_CAP,
        "rescue_cap": RESCUE_CAP,
        "clinical_rule": PREREGISTERED_CLINICAL_RULE,
        "record_1333_example_classification": "FAILURE",
        "canonical_bank_before": bank_before,
        "canonical_anchor_hash": bank_anchor_hash(),
        "stage2_launched": False,
        "ten_edit_launched": False,
        "phase_a_rerun": False,
    }
    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    model.llava_model.gradient_checkpointing_enable()
    if hasattr(model.llava_model, "enable_input_require_grads"):
        model.llava_model.enable_input_require_grads()
    apply_prefix(model, bank, 0)
    s0_hash = state_weight_hash(model)
    append_jsonl(out_dir / "state_hash_ledger.jsonl", {"event": "S0_RECONSTRUCTED", "weight_hash": s0_hash, "bank_hash": bank_manifest()["sha256"]})
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
    baseline_generation = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)
    locality_cache = cache_localities(model, views)
    if not all(equality_kl_gradient_is_unusable(row["s0_logits"].detach().cpu()) for row in locality_cache):
        raise RuntimeError("KL-at-equality preflight unexpectedly has usable gradient")
    preservation = build_preservation_cache(model, views)
    append_jsonl(out_dir / "state_hash_ledger.jsonl", {"event": "PRESERVATION_TRAJECTORIES_FIXED", "rows": [{key: row[key] for key in ("record_id", "fragile_position", "fragile_margin", "includes_eos")} for row in preservation]})
    try:
        score_rows, selected_names = score_candidate_modules(model, natural, short, natural_layout, short_positions, preservation)
    except ValueError as error:
        write_module_scores(out_dir / "v3_1_module_scores.csv", [])
        raise RuntimeError("V3_1_INVALID_LOCALITY_SENSITIVITY_ZERO") from error
    write_module_scores(out_dir / "v3_1_module_scores.csv", score_rows)
    target_gradients = selected_target_gradients(model, selected_names, natural, short, natural_layout, short_positions)
    modules = dict(model.named_modules())
    parameterizations: list[FixedRightWeight] = []
    svd_report = {}
    for name in selected_names:
        decomposition = fixed_right_basis(-target_gradients[name])
        item = FixedRightWeight(modules[name].weight.detach(), decomposition["A_fixed"]).to(model.lm_device)
        parametrize.register_parametrization(modules[name], "weight", item, unsafe=True)
        parameterizations.append(item)
        svd_report[name] = {"singular_values": decomposition["singular_values"].tolist(), "captured_target_gradient_energy_fraction": decomposition["captured_energy_fraction"], "A_fixed_hash": tensor_sha256(item.A_fixed)}
    del target_gradients
    torch.cuda.empty_cache()
    factors = [item.B for item in parameterizations]
    a_fixed = [item.A_fixed for item in parameterizations]
    if any(value.detach().abs().max().item() != 0 for value in factors) or manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"] != baseline_generation["token_ids"]:
        raise RuntimeError("V3.1 exact-zero S0 parity failure")
    factor_preservation_gradients = []
    direction_rows = []
    for row in preservation:
        logits = selected_logits(model, row["canonical"])
        loss = preservation_nll(logits, row["canonical"].target_ids)
        gradient = factor_gradients(loss, factors)
        factor_preservation_gradients.append(gradient.detach().cpu())
        direction_rows.append({"record_id": row["record_id"], "kind": "baseline_token_nll", "factor_gradient_norm": gradient_norm(gradient)})
        del logits, loss, gradient
        logits = selected_logits(model, row["canonical"])
        loss = preservation_margin_loss(logits, row["canonical"].target_ids, int(row["fragile_position"]))
        gradient = factor_gradients(loss, factors)
        factor_preservation_gradients.append(gradient.detach().cpu())
        direction_rows.append({"record_id": row["record_id"], "kind": "fragile_token_margin", "factor_gradient_norm": gradient_norm(gradient)})
        del logits, loss, gradient
        torch.cuda.empty_cache()
    try:
        basis_result = locality_basis(factor_preservation_gradients)
    except ValueError as error:
        raise RuntimeError("V3_1_INVALID_LOCALITY_BASIS") from error
    basis = basis_result["basis"].to(model.lm_device)
    del factor_preservation_gradients
    torch.cuda.empty_cache()
    effect_gradient, baseline_effect = factor_effect_gradient(model, factors, natural, short, natural_layout, short_positions)
    projected, projection_residual = project_gradient(effect_gradient, basis)
    basis_report = {
        "requested_preservation_directions": 20,
        "nonzero_preservation_directions": basis_result["nonzero_directions"],
        "numerical_rank": basis_result["rank"],
        "singular_values": basis_result["singular_values"].tolist(),
        "orthogonality_error": basis_result["orthogonality_error"],
        "target_projection_residual": projection_residual,
        "basis_hash": tensor_sha256(basis),
        "directions": direction_rows,
        "selected_module_svd": svd_report,
    }
    write_json(out_dir / "v3_1_locality_basis_report.json", basis_report)
    if not torch.isfinite(basis).all() or basis_result["rank"] == 0 or basis_result["orthogonality_error"] > 1e-4 or projection_residual > 1e-4:
        raise RuntimeError("V3_1_INVALID_LOCALITY_BASIS")
    descent = normalize_effective_step(-projected, factors, a_fixed, STEP)
    plus = evaluate_sign(model, 1, descent, factors, baseline_effect, natural, short, natural_layout, short_positions, locality_cache, s0_hash)
    minus = evaluate_sign(model, -1, descent, factors, baseline_effect, natural, short, natural_layout, short_positions, locality_cache, s0_hash)
    try:
        chosen_sign = choose_directional_sign(plus, minus)
        sign_valid = True
    except ValueError:
        chosen_sign, sign_valid = 0, False
    sign_report = {"plus": plus, "minus": minus, "chosen_sign": chosen_sign, "valid": sign_valid, "post_gate_s0_hash": state_weight_hash(model), "rollback_exact": state_weight_hash(model) == s0_hash}
    write_json(out_dir / "v3_1_directional_sign_gate.json", sign_report)
    if not sign_valid:
        raise RuntimeError("V3_1_DIRECTION_OR_SIGN_INVALID")
    active_cap = STANDARD_CAP
    microstep = 0
    terminal = "UNKNOWN"
    success = None
    best = {"loss": math.inf, "flat": current_flat(factors).clone()}
    worsening = 0
    restarted = False
    reached_b1 = reached_standard = reached_rescue = False
    while True:
        effect_gradient, runtime_gradient, gradient_info = factor_objective_gradients(model, factors, natural, short, natural_layout, short_positions, locality_cache)
        projected, residual = project_gradient(effect_gradient, basis)
        direction = -float(chosen_sign) * projected - runtime_gradient
        try:
            step = normalize_effective_step(direction, factors, a_fixed, STEP)
        except ValueError:
            terminal = "NONFINITE_OR_ZERO_EFFECTIVE_DIRECTION"
            break
        before_flat = current_flat(factors)
        copy_flat(factors, before_flat + step)
        project_cumulative_budget(parameterizations, active_cap)
        microstep += 1
        delta_norm = global_l2(cumulative_effective(parameterizations))
        factor_norm = global_l2(factors)
        displacement = relative_displacement(cumulative_effective(parameterizations), [item.base for item in parameterizations])
        reached_b1 = reached_b1 or delta_norm >= B1 - 1e-9
        if microstep % 4 == 0:
            metrics = checkpoint_metrics(model, natural, short, natural_layout, short_positions, locality_cache, preservation)
            append_jsonl(out_dir / "v3_1_trajectory.jsonl", {"microstep": microstep, "active_cap": active_cap, "factor_norm": factor_norm, "effective_delta_norm": delta_norm, "relative_displacement": displacement, "effect_gradient_norm": gradient_norm(effect_gradient), "projected_effect_gradient_norm": gradient_norm(projected), "runtime_kl_gradient_norm": gradient_norm(runtime_gradient), "projection_residual": residual, **metrics})
            if metrics["total_loss"] < best["loss"]:
                best = {"loss": metrics["total_loss"], "flat": current_flat(factors).clone()}
                worsening = 0
            else:
                worsening += 1
            if worsening >= 3:
                if not restarted:
                    copy_flat(factors, best["flat"])
                    restarted, worsening = True, 0
                    append_jsonl(out_dir / "v3_1_trajectory.jsonl", {"event": "RESTORE_LAST_BEST_EFFECTIVE_STATE", "microstep": microstep})
                else:
                    terminal = "OPTIMIZER_STALLED_AFTER_SINGLE_RESTART"
                    break
            if metrics["locality"]["maximum_nll_drift"] > 0.01 or not metrics["locality"]["paired_first_top1_equal"]:
                terminal = "LOCALITY_SAFETY_LIMIT"
                break
        rank = metrics["natural"]["first_target_rank"] if microstep % 4 == 0 else 999999
        if microstep % 8 == 0 or rank <= 5:
            generation = unrestricted_checkpoint(model, original, short, target, aliases)
            append_jsonl(out_dir / "v3_1_trajectory.jsonl", {"event": "GENERATION", "microstep": microstep, "effective_delta_norm": delta_norm, "relative_displacement": displacement, **generation})
            if generation["match"]["effective"]:
                success = generation
                terminal = "UNRESTRICTED_SUCCESS"
                break
        if displacement >= active_cap - 1e-10:
            if active_cap == STANDARD_CAP:
                reached_standard = True
                active_cap = RESCUE_CAP
                append_jsonl(out_dir / "v3_1_trajectory.jsonl", {"event": "ENTER_RESCUE_0P010", "microstep": microstep})
            else:
                reached_rescue = True
                terminal = "RESCUE_CAP_REACHED"
                break
        if microstep >= 20000:
            terminal = "DETERMINISTIC_SAFETY_MAX_MICROSTEPS"
            break
        if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
            terminal = "CANONICAL_BANK_MUTATION"
            break
    candidate_flat = current_flat(factors).clone()
    set_candidate = lambda: copy_flat(factors, candidate_flat)
    set_s0 = lambda: set_zero(factors)
    locality = full_locality_gate(model, views, set_s0, set_candidate)
    set_candidate()
    parity = full_generation_parity(model, original)
    match = generation_success(model, parity["no_cache"], target, aliases, None)
    terminal_displacement = relative_displacement(cumulative_effective(parameterizations), [item.base for item in parameterizations])
    terminal_delta_norm = global_l2(cumulative_effective(parameterizations))
    candidate_ids = parity["no_cache"]["token_ids"]
    set_s0()
    rollback_ids = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    rollback = "PASS" if rollback_ids == baseline_generation["token_ids"] and state_weight_hash(model) == s0_hash else "FAIL"
    set_candidate()
    replay_ids = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    replay = "PASS" if replay_ids == candidate_ids else "FAIL"
    reload_status = fresh_status = "NOT_RUN"
    success_budget = None
    natural_success = bool(match["effective"])
    strict_pass = locality["strict_damage_count"] == 0 and locality["clinical_failure_count"] == 0
    candidate_bank = None
    if natural_success and strict_pass:
        success_budget = "0.003" if terminal_displacement <= STANDARD_CAP + 1e-10 else "0.010"
        candidate_bank = out_dir / "candidate_v3_1_bank"
        candidate_bank.mkdir(exist_ok=False)
        payload = {name: {"A_fixed": item.A_fixed.detach().cpu(), "B": item.B.detach().cpu(), "base_hash": tensor_sha256(item.base)} for name, item in zip(selected_names, parameterizations)}
        torch.save(payload, candidate_bank / "factors.pt")
        bank_meta = {"protocol": PROTOCOL, "record_id": RECORD_ID, "target": target, "selected_modules": selected_names, "rank": 4, "locality_basis_hash": basis_report["basis_hash"], "relative_displacement": terminal_displacement, "canonical_bank_hash": EXPECTED_BANK_HASH, "candidate_token_ids": candidate_ids, "factor_hashes": {name: {key: tensor_sha256(value) for key, value in row.items() if isinstance(value, torch.Tensor)} for name, row in payload.items()}}
        write_json(candidate_bank / "manifest.json", bank_meta)
        loaded = torch.load(candidate_bank / "factors.pt", map_location="cpu", weights_only=False)
        reload_status = "PASS" if all(tensor_sha256(loaded[name][key]) == bank_meta["factor_hashes"][name][key] for name in selected_names for key in ("A_fixed", "B")) else "FAIL"
    if natural_success and not strict_pass:
        label = "V3_1_NATURAL_SUCCESS_STRICT_LOCALITY_FAILURE"
    elif natural_success and strict_pass:
        label = "PASS_V3_1_WITHIN_0P003" if terminal_displacement <= STANDARD_CAP + 1e-10 else "PASS_V3_1_RESCUE_0P010"
    elif terminal == "LOCALITY_SAFETY_LIMIT":
        label = "V3_1_LOCALITY_LIMIT_BEFORE_NATURAL_GENERATION"
    elif terminal == "OPTIMIZER_STALLED_AFTER_SINGLE_RESTART":
        label = "V3_1_OPTIMIZER_STALLED_BEFORE_BUDGET"
    elif terminal in {"RESCUE_CAP_REACHED"} or (reached_standard and active_cap == RESCUE_CAP):
        label = "V3_1_FULL_BUDGET_NO_NATURAL_GENERATION"
    else:
        label = "V3_1_INVALID_ENGINEERING_RUN"
    for name in selected_names:
        parametrize.remove_parametrizations(modules[name], "weight", leave_parametrized=False)
    apply_prefix(model, bank, 0)
    rollback = "PASS" if rollback == "PASS" and state_weight_hash(model) == s0_hash else "FAIL"
    bank_after = bank_manifest()
    append_jsonl(out_dir / "state_hash_ledger.jsonl", {"event": "S0_RESTORED", "weight_hash": state_weight_hash(model), "bank_hash": bank_after["sha256"]})
    if bank_after["sha256"] != EXPECTED_BANK_HASH:
        label = "V3_1_INVALID_ENGINEERING_RUN"
    summary = {
        "primary_label": label,
        "unrestricted_success": natural_success,
        "success_budget": success_budget,
        "terminal_displacement": terminal_displacement,
        "terminal_effective_delta_norm": terminal_delta_norm,
        "terminal_factor_norm": float(candidate_flat.double().norm()),
        "terminal_reason": terminal,
        "microsteps": microstep,
        "selected_modules": selected_names,
        "locality_basis_non_degenerate": basis_result["rank"] > 0,
        "locality_basis_rank": basis_result["rank"],
        "directional_sign_gate_valid": sign_valid,
        "chosen_sign": chosen_sign,
        "reached_b1": reached_b1,
        "reached_0p003": reached_standard,
        "reached_0p010": reached_rescue,
        "exact_unrestricted_output": parity["no_cache"]["raw_output"],
        "strict_locality_damage": locality["strict_damage_count"],
        "normalized_locality_damage": locality["normalized_damage_count"],
        "first_step_locality_damage": locality["first_step_damage_count"],
        "stop_reason_locality_damage": locality["stop_reason_damage_count"],
        "clinical_locality_failures": locality["clinical_failure_count"],
        "maximum_locality_nll_drift": locality["maximum_nll_drift"],
        "reload": reload_status,
        "fresh": fresh_status,
        "replay": replay,
        "rollback": rollback,
        "canonical_bank_unchanged": bank_after["sha256"] == bank_before["sha256"],
        "stage2_permitted": False,
    }
    write_json(out_dir / "v3_1_summary.json", summary)
    write_json(out_dir / "v3_1_final_locality_report.json", locality)
    write_json(out_dir / "v3_1_bank_replay_fresh_rollback.json", {"reload": reload_status, "fresh": fresh_status, "replay": replay, "rollback": rollback, "candidate_bank": str(candidate_bank) if candidate_bank else None, "canonical_bank_before": bank_before["sha256"], "canonical_bank_after": bank_after["sha256"]})
    manifest.update({"selected_modules": selected_names, "locality_basis_rank": basis_result["rank"], "directional_sign_gate_valid": sign_valid, "primary_label": label, "canonical_bank_after": bank_after})
    write_json(out_dir / "run_manifest.json", manifest)
    write_text(out_dir / "V3_1_FINAL_DECISION.md", final_report(summary))


def fresh(args: argparse.Namespace) -> None:
    if args.candidate_bank is None:
        raise ValueError("--candidate-bank is required")
    metadata = json.loads((args.candidate_bank / "manifest.json").read_text())
    payload = torch.load(args.candidate_bank / "factors.pt", map_location="cpu", weights_only=False)
    model, views, bank, _records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 0)
    original = build_canonical_inputs(model, views[RECORD_ID]["target"])
    baseline_ids = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    modules = dict(model.named_modules())
    items = []
    for name in metadata["selected_modules"]:
        item = FixedRightWeight(modules[name].weight.detach(), payload[name]["A_fixed"]).to(model.lm_device)
        with torch.no_grad():
            item.B.copy_(payload[name]["B"].to(model.lm_device))
        parametrize.register_parametrization(modules[name], "weight", item, unsafe=True)
        items.append(item)
    candidate_ids = full_generation_parity(model, original)["no_cache"]["token_ids"]
    reconstruction = candidate_ids == metadata["candidate_token_ids"]
    for name in metadata["selected_modules"]:
        parametrize.remove_parametrizations(modules[name], "weight", leave_parametrized=False)
    apply_prefix(model, bank, 0)
    rollback = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"] == baseline_ids
    write_json(args.candidate_bank / "fresh_gate.json", {"fresh_process_reconstruction": reconstruction, "rollback": rollback, "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH, "passed": reconstruction and rollback and bank_manifest()["sha256"] == EXPECTED_BANK_HASH})


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must equal {args.physical_gpu}")
    if args.mode == "fresh":
        fresh(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
