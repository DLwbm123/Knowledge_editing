#!/usr/bin/env python3
"""Build and evaluate the frozen 32-expert post-hoc LiveEdit-Med repository."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import to_jsonable
from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook, resolve_layer21_block
from methods.liveedit_med.posthoc_validation import (PROTOCOL, canonical_json_hash, native_sample,
    normalize_answer, plan_audit, route_residual, sample_to_model_row, unrestricted_match)
from methods.liveedit_med.serialization import load_safe_state, save_safe_state, tensor_hashes
from methods.liveedit_med.source_ops import BaseRoutePlan, apply_low_rank_expert_residual, route_repository
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_natural_generation_recovery import clinical_preservation
from scripts.engram.run_engram_v2_one_shot_natural_generation_rescue import canonical_nll
from scripts.engram.run_engram_v2_stage0_generation_audit import (apply_prefix, bank_manifest,
    eos_ids, load_model_views_bank, state_weight_hash)
from scripts.engram.run_engram_v2_stage0abc_diagnostics import SHORT_INSTRUCTION, hf_cached_greedy_trace
from scripts.engram.stage0_generation_audit_utils import (build_canonical_inputs,
    manual_cached_greedy_trace, manual_greedy_trace, normalize_medical_answer)
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import (capture_prompt,
    capture_teacher_forced, compact_trace, load_clean_model)


PROTOCOL_Q = "POSTHOC_VALIDATION_RECOVERY__NO_TEST_LEAKAGE__STAGE_Q"
RECORD_ID = "953"
MAX_NEW_TOKENS = 128
EXPECTED_BANK_HASH = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"
POSITIVE_NAMES = ("native", "textual", "visual", "paired")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def compact_generation(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {key: trace.get(key) for key in
            ("raw_output", "token_ids", "eos_step", "stop_reason", "cap_hit")}


def sample(image: str, prompt: str, target: str) -> dict[str, str]:
    return {"image": str(image), "prompt": str(prompt), "target": str(target)}


def validate_stage_f(recovery_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = read_json(recovery_dir / "checkpoint_selection.json")
    stage_f = read_json(recovery_dir / "stage_f/record953_forced_on_selected_checkpoint.json")
    selected_step = selection.get("selected_step")
    if (selection.get("protocol") != PROTOCOL or selection.get("record953_used_for_selection") is not False
            or selected_step is None or stage_f.get("selected_step") != selected_step
            or stage_f.get("selection_hash") != selection.get("selection_hash")
            or stage_f.get("stage_q_permitted") is not True
            or stage_f.get("native_unrestricted_success") is not True):
        raise RuntimeError("LIVEEDIT_MED_STAGE_Q_GATE_NOT_SATISFIED")
    return selection, stage_f


def stage_q_inputs(views: Mapping[str, Any], records: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    target_row = views[RECORD_ID]["target"]
    target = str(target_row["target"][0]); raw = records[RECORD_ID]
    image_root = Path(str(target_row["image_path"][0])).parents[1]
    positive = {
        "native": sample(target_row["image_path"][0], target_row["prompt"][0], target),
        "textual": sample(target_row["image_path"][0], raw["rephrase"], raw["alt"]),
        "visual": sample(image_root / raw["image_rephrase"], raw["src"], raw["alt"]),
        "paired": sample(image_root / raw["image_rephrase"],
                         raw["port_new"][0]["Q&A"]["Question"], raw["alt"]),
    }
    questions = {
        "native": str(raw["src"]), "textual": str(raw["rephrase"]),
        "visual": str(raw["src"]), "paired": str(raw["port_new"][0]["Q&A"]["Question"]),
    }
    positive_short = {name: sample(row["image"],
        f"Question: {questions[name]} {SHORT_INSTRUCTION} Short answer: ", row["target"])
        for name, row in positive.items()}
    locality_answers = {f"locality:{rid}:native": str(views[rid]["locality"]["target"][0])
                        for rid in views if f"locality:{rid}:native" in
                        {str(row["input_id"]) for row in split["fixed_locality_native_inputs"]}}
    safety = [{"input_id": str(row["input_id"]), "equivalence_key": str(row["router_input_equivalence_key"]),
               "image": str(row["image_path"]), "prompt": str(row["question"]),
               "canonical_answer": None} for row in split["unique_heldout_classes"]]
    locality = [{"input_id": str(row["input_id"]), "equivalence_key": str(row["router_input_equivalence_key"]),
                 "image": str(row["image_path"]), "prompt": str(row["question"]),
                 "canonical_answer": locality_answers[str(row["input_id"])]}
                for row in split["fixed_locality_native_inputs"]]
    if len(safety) != 40 or len(locality) != 10 or len({row["equivalence_key"] for row in safety}) != 40:
        raise RuntimeError("LIVEEDIT_MED_STAGE_Q_FROZEN_EXTERNAL_SET_MISMATCH")
    result = {"protocol": PROTOCOL_Q, "positive": positive, "positive_short": positive_short,
              "safety": safety, "locality": locality,
              "counts": {"positive": 4, "safety": 40, "locality": 10},
              "selection_rule": "existing_v1_1_unique_heldout_equivalence_classes_and_fixed_locality_native_inputs"}
    result["input_manifest_hash"] = canonical_json_hash(result)
    return result


def repository_hash(state: Mapping[str, torch.Tensor], ids: Sequence[str]) -> str:
    return canonical_json_hash({"ids": list(ids), "tensor_hashes": tensor_hashes(state)})


def build_repository(args: argparse.Namespace) -> None:
    if args.stage_q_dir.exists():
        raise FileExistsError(args.stage_q_dir)
    args.stage_q_dir.mkdir(parents=True, exist_ok=False)
    selection, stage_f = validate_stage_f(args.recovery_dir)
    selected_step = int(selection["selected_step"])
    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 0); clean_hash = state_weight_hash(model)
    _, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    checkpoint = args.run_dir / "training" / f"checkpoint_{selected_step:04d}"
    state, checkpoint_manifest = load_safe_state(checkpoint)
    modules.load_state_dict(state, strict=True); modules.eval()
    source = read_json(args.source_records)
    heldout = sorted(source["records"]["heldout"],
                     key=lambda row: (str(row["selection_hash"]), str(row["record_id"])))
    distractors = heldout[:31]
    if len(distractors) != min(31, len(heldout)) or any(str(row["record_id"]) == RECORD_ID for row in distractors):
        raise RuntimeError("LIVEEDIT_MED_STAGE_Q_DISTRACTOR_SELECTION_FAILURE")
    target_sample = {"image": views[RECORD_ID]["target"]["image_path"][0],
                     "prompt": views[RECORD_ID]["target"]["prompt"][0],
                     "target": views[RECORD_ID]["target"]["target"][0]}
    expert_rows = [(RECORD_ID, target_sample, None)] + [(str(row["record_id"]), native_sample(row),
        str(row["selection_hash"])) for row in distractors]
    tensors: dict[str, list[torch.Tensor]] = {name: [] for name in ("eqr", "evr", "moe_c", "moe_r")}
    target_by_id = {}
    for index, (rid, edit_sample, selection_hash) in enumerate(expert_rows):
        captured = capture_teacher_forced(model, block, edit_sample)
        eqr, evr, moe_c, moe_r = modules.generated_edit(captured["vision"].float(),
                                                        captured["question"].float(),
                                                        captured["answer"].float())
        for name, value in zip(tensors, (eqr, evr, moe_c, moe_r)):
            tensors[name].append(value.detach().float().cpu())
        target_by_id[rid] = str(edit_sample["target"])
        print(json.dumps({"event": "repository_expert", "index": index, "record_id": rid}), flush=True)
    repo_state = {name: torch.cat(values, 0) for name, values in tensors.items()}
    ids = [row[0] for row in expert_rows]
    repo_hash = repository_hash(repo_state, ids)
    manifest = {"protocol": PROTOCOL_Q, "selected_step": selected_step,
                "selection_hash": selection["selection_hash"], "stage_f_label": stage_f["label"],
                "ids": ids, "target_record_id": RECORD_ID, "targets": target_by_id,
                "distractor_selection_rule": "first_31_heldout_by_existing_stable_hash_order",
                "distractors": [{"record_id": row[0], "selection_hash": row[2]} for row in expert_rows[1:]],
                "repository_hash": repo_hash, "checkpoint_manifest": checkpoint_manifest,
                "canonical_bank_hash": bank_manifest()["sha256"], "base_state_hash": clean_hash}
    save_safe_state(args.stage_q_dir / "repository", repo_state, manifest)
    frozen_split = read_json(args.frozen_external_split)
    inputs = stage_q_inputs(views, records, frozen_split)
    inputs["frozen_external_split"] = str(args.frozen_external_split.resolve())
    inputs["frozen_external_split_file_hash"] = canonical_json_hash(frozen_split)
    inputs.pop("input_manifest_hash", None)
    inputs["input_manifest_hash"] = canonical_json_hash(inputs)
    write_json(args.stage_q_dir / "input_manifest.json", inputs)
    audit = {"protocol": PROTOCOL_Q, "status": "FROZEN_REPOSITORY_READY",
             "selected_step": selected_step, "expert_count": len(ids), "target_count": 1,
             "distractor_count": len(distractors), "repository_hash": repo_hash,
             "input_manifest_hash": inputs["input_manifest_hash"],
             "record953_used_for_checkpoint_selection": False,
             "record_specific_optimization": False,
             "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH,
             "base_state_unchanged": state_weight_hash(model) == clean_hash}
    if not audit["canonical_bank_unchanged"] or not audit["base_state_unchanged"]:
        raise RuntimeError("LIVEEDIT_MED_STAGE_Q_BUILD_MUTATION")
    write_json(args.stage_q_dir / "repository_audit.json", audit)
    print(json.dumps(audit), flush=True)


def load_repository(directory: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    state, manifest = load_safe_state(directory / "repository")
    ids = [str(value) for value in manifest["ids"]]
    if manifest.get("protocol") != PROTOCOL_Q or len(ids) != 32 or ids[0] != RECORD_ID:
        raise RuntimeError("LIVEEDIT_MED_STAGE_Q_REPOSITORY_MANIFEST_INVALID")
    if repository_hash(state, ids) != manifest.get("repository_hash"):
        raise RuntimeError("LIVEEDIT_MED_STAGE_Q_REPOSITORY_HASH_MISMATCH")
    repo: dict[str, Any] = {"ids": ids}
    for name, value in state.items():
        repo[name] = value.to(device)
    return repo, manifest


def route_context(model: Any, block: torch.nn.Module, modules: LiveEditMedicalModules,
                  row: Mapping[str, Any], repository: Mapping[str, Any]):
    canonical = build_canonical_inputs(model, sample_to_model_row(row))
    prompt_hidden, vision, question = capture_prompt(model, block, canonical)
    plan = route_repository(modules.input_extractor, question.float(), vision.float(),
                            repository["evr"], repository["eqr"])
    audit = plan_audit(plan, repository["ids"])
    _residual, norms = route_residual(plan, prompt_hidden, repository["moe_c"],
                                      repository["moe_r"], modules.instant_reps_norm)
    audit["residual_norms"] = norms
    raw_weights = audit["final_weights"]
    if len(raw_weights) == 1 and isinstance(raw_weights[0], list):
        raw_weights = raw_weights[0]
    weights = [float(value) for value in raw_weights]
    candidates = list(audit["candidate_ids"])
    if RECORD_ID in candidates:
        pos = candidates.index(RECORD_ID); target_weight = weights[pos]
        target_rank = 1 + sorted(weights, reverse=True).index(target_weight)
    else:
        target_weight, target_rank = 0.0, None
    audit.update({"target_expert_rank": target_rank, "target_expert_final_weight": target_weight,
                  "distractor_total_weight": sum(value for rid, value in zip(candidates, weights)
                                                 if rid != RECORD_ID)})
    hook = None
    if not isinstance(plan, BaseRoutePlan):
        selected_c, selected_r = repository["moe_c"][plan.candidate_mask], repository["moe_r"][plan.candidate_mask]
        hook = Layer21ResidualHook(block, lambda hidden: apply_low_rank_expert_residual(
            hidden.float(), selected_c, selected_r, plan.final_weights,
            modules.instant_reps_norm).to(hidden.dtype)).install()
        hook.enabled = True
    return canonical, audit, hook


def remove_hook(hook: Layer21ResidualHook | None) -> None:
    if hook is not None:
        hook.remove()


@torch.inference_mode()
def positive_result(model: Any, block: torch.nn.Module, modules: LiveEditMedicalModules,
                    row: Mapping[str, Any], short_row: Mapping[str, Any], repository: Mapping[str, Any]) -> dict[str, Any]:
    canonical, route, hook = route_context(model, block, modules, row, repository)
    try:
        no_cache = manual_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=5)
        cached = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=5)
        hf = hf_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS)
    finally:
        remove_hook(hook)
    parity = no_cache["token_ids"] == cached["token_ids"] == hf["token_ids"]
    unrestricted = {**compact_generation(no_cache), "match": unrestricted_match(
        no_cache["raw_output"], row["target"], eos=no_cache["stop_reason"] == "eos", cap_hit=no_cache["cap_hit"])}
    short_canonical, short_route, short_hook = route_context(model, block, modules, short_row, repository)
    try:
        short_trace = manual_cached_greedy_trace(model, short_canonical, MAX_NEW_TOKENS,
                                                eos_ids(model), top_k=5)
    finally:
        remove_hook(short_hook)
    short = {**compact_generation(short_trace), "match": unrestricted_match(
        short_trace["raw_output"], short_row["target"], eos=short_trace["stop_reason"] == "eos",
        cap_hit=short_trace["cap_hit"])}
    return {"route": route, "unrestricted": unrestricted, "short_route": short_route,
            "short_output": short, "parity": {"passed": parity,
                "no_cache": compact_generation(no_cache), "cached": compact_generation(cached),
                "hf": compact_generation(hf)},
            "success": bool(unrestricted["match"]["success"] and short["match"]["success"] and parity)}


def nll_with_hook(model: Any, canonical: Any, hook: Layer21ResidualHook | None) -> float:
    try:
        return float(canonical_nll(model, canonical)["nll"])
    finally:
        remove_hook(hook)


@torch.inference_mode()
def negative_result(model: Any, block: torch.nn.Module, modules: LiveEditMedicalModules,
                    entry: Mapping[str, Any], repository: Mapping[str, Any], target: str) -> dict[str, Any]:
    probe = sample(entry["image"], entry["prompt"], "placeholder")
    clean_canonical = build_canonical_inputs(model, sample_to_model_row(probe))
    clean = manual_cached_greedy_trace(model, clean_canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
    routed_canonical, route, hook = route_context(model, block, modules, probe, repository)
    try:
        routed = manual_cached_greedy_trace(model, routed_canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
    finally:
        remove_hook(hook)
    score_target = clean["raw_output"] if clean["raw_output"].strip() else str(entry.get("canonical_answer") or "unknown")
    score_row = sample(entry["image"], entry["prompt"], score_target)
    clean_nll = float(canonical_nll(model, build_canonical_inputs(model, sample_to_model_row(score_row)))["nll"])
    routed_score_canonical, _score_route, score_hook = route_context(model, block, modules, score_row, repository)
    routed_nll = nll_with_hook(model, routed_score_canonical, score_hook)
    normalized_equal = normalize_medical_answer(clean["raw_output"]) == normalize_medical_answer(routed["raw_output"])
    canonical_answer = entry.get("canonical_answer")
    clinical = clinical_preservation(clean["raw_output"], routed["raw_output"],
                                     str(canonical_answer or clean["raw_output"]))
    contamination = bool(normalize_answer(target) and normalize_answer(target) in normalize_answer(routed["raw_output"]))
    checks = {"token_ids_equal": clean["token_ids"] == routed["token_ids"],
              "normalized_output_equal": normalized_equal,
              "clinical_canonical_preserved": bool(clinical["passed"]),
              "target_contamination_absent": not contamination,
              "stop_reason_equal": clean["stop_reason"] == routed["stop_reason"]}
    return {"input_id": entry["input_id"], "equivalence_key": entry["equivalence_key"],
            "route": route, "s0": compact_generation(clean), "routed": compact_generation(routed),
            "clinical": clinical, "canonical_answer": canonical_answer,
            "record953_target_contamination": contamination,
            "nll": {"target_basis": "s0_generated_continuation", "s0": clean_nll,
                    "routed": routed_nll, "absolute_drift": abs(routed_nll - clean_nll)},
            "checks": checks, "passed": all(checks.values())}


def evaluate_shard(args: argparse.Namespace) -> None:
    if args.shard not in (0, 1):
        raise ValueError("--shard must be 0 or 1")
    out = args.stage_q_dir / f"worker_{args.shard}.json"
    if out.exists():
        raise FileExistsError(out)
    inputs = read_json(args.stage_q_dir / "input_manifest.json")
    selection, _stage_f = validate_stage_f(args.recovery_dir)
    selected_step = int(selection["selected_step"])
    model, _bank = load_clean_model(args.physical_gpu); clean_hash = state_weight_hash(model)
    _, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(
        args.run_dir / "training" / f"checkpoint_{selected_step:04d}"
    )
    modules.load_state_dict(state, strict=True); modules.eval()
    repository, repo_manifest = load_repository(args.stage_q_dir, model.lm_device)
    positive_names = POSITIVE_NAMES[:2] if args.shard == 0 else POSITIVE_NAMES[2:]
    safety_entries = inputs["safety"][:20] if args.shard == 0 else inputs["safety"][20:]
    locality_entries = inputs["locality"][:5] if args.shard == 0 else inputs["locality"][5:]
    positive = {}
    for name in positive_names:
        positive[name] = positive_result(model, block, modules, inputs["positive"][name],
                                         inputs["positive_short"][name], repository)
        print(json.dumps({"event": "positive", "shard": args.shard, "name": name,
                          "success": positive[name]["success"]}), flush=True)
    target = repo_manifest["targets"][RECORD_ID]
    safety = []
    for index, entry in enumerate(safety_entries):
        safety.append(negative_result(model, block, modules, entry, repository, target))
        print(json.dumps({"event": "safety", "shard": args.shard, "index": index,
                          "input_id": entry["input_id"], "passed": safety[-1]["passed"]}), flush=True)
    locality = []
    for index, entry in enumerate(locality_entries):
        locality.append(negative_result(model, block, modules, entry, repository, target))
        print(json.dumps({"event": "locality", "shard": args.shard, "index": index,
                          "input_id": entry["input_id"], "passed": locality[-1]["passed"]}), flush=True)
    result = {"protocol": PROTOCOL_Q, "shard": args.shard, "physical_gpu": args.physical_gpu,
              "selected_step": selected_step, "checkpoint_manifest": checkpoint_manifest,
              "repository_hash": repo_manifest["repository_hash"],
              "input_manifest_hash": inputs["input_manifest_hash"],
              "positive": positive, "safety": safety, "locality": locality,
              "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH,
              "base_state_unchanged": state_weight_hash(model) == clean_hash,
              "generation_config": {"do_sample": False, "num_beams": 1,
                                    "max_new_tokens": MAX_NEW_TOKENS}}
    if not result["canonical_bank_unchanged"] or not result["base_state_unchanged"]:
        raise RuntimeError("LIVEEDIT_MED_STAGE_Q_WORKER_MUTATION")
    write_json(out, result)
    print(json.dumps({"event": "worker_complete", "shard": args.shard}), flush=True)


def finalize(args: argparse.Namespace) -> None:
    out = args.stage_q_dir / "stage_q_summary.json"
    if out.exists():
        raise FileExistsError(out)
    inputs = read_json(args.stage_q_dir / "input_manifest.json")
    workers = [read_json(args.stage_q_dir / f"worker_{index}.json") for index in (0, 1)]
    selected_steps = {int(row.get("selected_step", -1)) for row in workers}
    if any(row.get("protocol") != PROTOCOL_Q for row in workers) or len(selected_steps) != 1:
        raise RuntimeError("LIVEEDIT_MED_STAGE_Q_WORKER_INVALID")
    selected_step = selected_steps.pop()
    if len({row["repository_hash"] for row in workers}) != 1 or len({row["input_manifest_hash"] for row in workers}) != 1:
        raise RuntimeError("LIVEEDIT_MED_STAGE_Q_WORKER_IDENTITY_MISMATCH")
    positives = {name: row for worker in workers for name, row in worker["positive"].items()}
    safety = [row for worker in workers for row in worker["safety"]]
    locality = [row for worker in workers for row in worker["locality"]]
    expected_safety = {row["input_id"] for row in inputs["safety"]}
    expected_locality = {row["input_id"] for row in inputs["locality"]}
    if set(positives) != set(POSITIVE_NAMES) or {row["input_id"] for row in safety} != expected_safety or {row["input_id"] for row in locality} != expected_locality:
        raise RuntimeError("LIVEEDIT_MED_STAGE_Q_INCOMPLETE_RESULT_SET")
    positive_pass = all(positives[name]["success"] for name in POSITIVE_NAMES)
    safety_pass = sum(row["passed"] for row in safety)
    locality_pass = sum(row["passed"] for row in locality)
    clinical_failures = sum(not row["checks"]["clinical_canonical_preserved"] for row in safety + locality)
    contaminations = sum(row["record953_target_contamination"] for row in safety + locality)
    parity_pass = all(positives[name]["parity"]["passed"] for name in POSITIVE_NAMES)
    passed = bool(positive_pass and safety_pass == 40 and locality_pass == 10
                  and clinical_failures == 0 and contaminations == 0 and parity_pass)
    summary = {"protocol": PROTOCOL_Q,
               "label": "LIVEEDIT_FULL_REPOSITORY_ROUTING_GATE_PASS" if passed
                        else "LIVEEDIT_FULL_REPOSITORY_ROUTING_GATE_FAILURE",
               "passed": passed, "selected_step": selected_step,
               "record953_used_for_checkpoint_selection": False,
               "repository_hash": workers[0]["repository_hash"],
               "input_manifest_hash": workers[0]["input_manifest_hash"],
               "positive_success": {name: positives[name]["success"] for name in POSITIVE_NAMES},
               "positive_outputs": positives, "safety": {"passed": safety_pass, "total": 40,
                   "rows": sorted(safety, key=lambda row: row["input_id"])},
               "locality": {"passed": locality_pass, "total": 10,
                   "rows": sorted(locality, key=lambda row: row["input_id"])},
               "clinical_canonical_failure_count": clinical_failures,
               "target_contamination_count": contaminations,
               "required_parity_passed": parity_pass,
               "canonical_bank_unchanged": all(row["canonical_bank_unchanged"] for row in workers),
               "base_state_unchanged": all(row["base_state_unchanged"] for row in workers),
               "generation_config": {"do_sample": False, "num_beams": 1,
                                     "max_new_tokens": MAX_NEW_TOKENS}}
    write_json(out, summary)
    report = args.stage_q_dir / "STAGE_Q_FINAL_DECISION.md"
    report.write_text("# Stage Q Final Decision\n\n"
        f"- Label: `{summary['label']}`\n"
        f"- Selected checkpoint: **{selected_step}**\n"
        f"- Positive native/textual/visual/paired: **"
        f"{' / '.join(str(summary['positive_success'][name]) for name in POSITIVE_NAMES)}**\n"
        f"- Safety exact S0: **{safety_pass}/40**\n"
        f"- Locality exact S0: **{locality_pass}/10**\n"
        f"- Clinical/canonical failures: **{clinical_failures}**\n"
        f"- Target contaminations: **{contaminations}**\n"
        f"- Required parity: **{parity_pass}**\n")
    print(json.dumps({key: summary[key] for key in ("label", "positive_success",
          "clinical_canonical_failure_count", "target_contamination_count")}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "worker", "finalize"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--stage-q-dir", type=Path, required=True)
    parser.add_argument("--source-records", type=Path)
    parser.add_argument("--frozen-external-split", type=Path)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--shard", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "build":
        if args.source_records is None or args.frozen_external_split is None or args.physical_gpu is None:
            raise ValueError("build requires source records, frozen external split, and physical GPU")
        build_repository(args)
    elif args.mode == "worker":
        if args.physical_gpu is None or args.shard is None:
            raise ValueError("worker requires physical GPU and shard")
        evaluate_shard(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
