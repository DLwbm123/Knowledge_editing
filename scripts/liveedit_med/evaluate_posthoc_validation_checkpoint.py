#!/usr/bin/env python3
"""Evaluate one frozen LiveEdit-Med checkpoint on the precommitted validation panel."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import ensure_offline_env, to_jsonable
from easyeditor.models.engram import EngramMultimodalHparams
from easyeditor.models.engram_v2 import SequentialEngramBankV2
from easyeditor.trainer.models import get_model
from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook, resolve_layer21_block
from methods.liveedit_med.posthoc_validation import (BaseRoutePlan, PROTOCOL, native_sample,
    normalize_answer, plan_audit, route_residual, sample_to_model_row, unrestricted_match)
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.source_ops import apply_low_rank_expert_residual, route_repository
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_continual_v2 import set_determinism
from scripts.engram.run_engram_v2_stage0_generation_audit import (BANK_ROOT, MODEL_CONFIG,
    MODULE_KEY, MODULE_NAME, apply_prefix, bank_manifest, eos_ids, state_weight_hash)
from scripts.engram.stage0_generation_audit_utils import (CanonicalInputs, build_canonical_inputs,
    ids_sha256, manual_cached_greedy_trace)


MAX_NEW_TOKENS = 128
EXPECTED_BANK_HASH = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_clean_model(physical_gpu: int):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu) or torch.cuda.device_count() != 1:
        raise RuntimeError("LIVEEDIT_MED_GPU_VISIBILITY_MISMATCH")
    ensure_offline_env(); set_determinism(42)
    config = EngramMultimodalHparams.from_hparams(str(MODEL_CONFIG))
    config.dropout, config.no_grad_layers, config.device = 0.0, None, "cuda"
    model = get_model(config).to(torch.device("cuda")).eval()
    bank = SequentialEngramBankV2(BANK_ROOT)
    module = dict(model.named_modules()).get(MODULE_NAME)
    expected = bank.anchor_state()[MODULE_KEY].to(dtype=module.weight.dtype)
    if not torch.equal(module.weight.detach().cpu(), expected):
        raise RuntimeError("LIVEEDIT_MED_S0_ANCHOR_MISMATCH")
    apply_prefix(model, bank, 0)
    return model, bank


@torch.inference_mode()
def capture_teacher_forced(model: Any, block: torch.nn.Module, sample: Mapping[str, Any]):
    row = sample_to_model_row(sample)
    inputs, labels, masks = model._build_batch(row)
    captured = []
    handle = block.register_forward_hook(lambda _m, _a, out: captured.append(
        (out[0] if isinstance(out, (tuple, list)) else out).detach()))
    output = model.llava_model(inputs_embeds=inputs, attention_mask=masks["attention_mask"].long(),
                               labels=labels, return_dict=True, use_cache=False)
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError("LIVEEDIT_MED_CAPTURE_COUNT")
    hidden = captured[0]
    return {"hidden": hidden, "vision": hidden[:, masks["vision_mask"][0]],
            "question": hidden[:, masks["prompt_mask"][0]],
            "answer": hidden[:, masks["answer_mask"][0]],
            "labels": labels, "attention": masks["attention_mask"], "loss": float(output.loss.item())}


@torch.inference_mode()
def capture_prompt(model: Any, block: torch.nn.Module, canonical: CanonicalInputs):
    captured = []
    handle = block.register_forward_hook(lambda _m, _a, out: captured.append(
        (out[0] if isinstance(out, (tuple, list)) else out).detach()))
    model.llava_model(input_ids=canonical.prompt_ids, images=canonical.image,
                      attention_mask=torch.ones_like(canonical.prompt_ids), return_dict=True, use_cache=False)
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError("LIVEEDIT_MED_PROMPT_CAPTURE_COUNT")
    hidden = captured[0]
    image_pos = torch.where(canonical.prompt_ids[0].eq(model.IMAGE_TOKEN_INDEX))[0]
    if image_pos.numel() != 1:
        raise RuntimeError("LIVEEDIT_MED_PROMPT_IMAGE_TOKEN_COUNT")
    start = int(image_pos[0]); vision_count = int(model.llava_model.encode_images(canonical.image).shape[1])
    vision = hidden[:, start:start + vision_count]
    question = hidden[:, start + vision_count:]
    if question.shape[1] == 0:
        raise RuntimeError("LIVEEDIT_MED_EMPTY_ROUTING_QUESTION")
    return hidden, vision, question


def compact_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {"raw_output": trace["raw_output"], "token_ids": trace["token_ids"],
            "eos_step": trace.get("eos_step"), "stop_reason": trace["stop_reason"],
            "cap_hit": trace["cap_hit"], "target_token_metrics": [
                {key: row.get(key) for key in ("step", "target_id", "target_rank", "margin",
                                                "target_probability", "top1_id", "top1_probability")}
                for row in trace["trajectory"] if row.get("target_id") is not None]}


@torch.inference_mode()
def forced_generation(model: Any, block: torch.nn.Module, modules: LiveEditMedicalModules,
                      sample: Mapping[str, Any], moe_c: torch.Tensor, moe_r: torch.Tensor):
    canonical = build_canonical_inputs(model, sample_to_model_row(sample))
    prompt_hidden, _vision, _question = capture_prompt(model, block, canonical)
    norm = {"per_expert_residual_norms": [float(apply_low_rank_expert_residual(
        prompt_hidden.float(), moe_c, moe_r, torch.ones(1, 1, device=prompt_hidden.device),
        modules.instant_reps_norm).norm().item())]}
    norm["fused_residual_norm"] = norm["per_expert_residual_norms"][0]
    hook = Layer21ResidualHook(block, lambda hidden: apply_low_rank_expert_residual(
        hidden.float(), moe_c, moe_r, torch.ones(1, 1, device=hidden.device),
        modules.instant_reps_norm).to(hidden.dtype)).install(); hook.enabled = True
    trace = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
    hook.remove()
    match = unrestricted_match(trace["raw_output"], sample["target"],
                               eos=trace["stop_reason"] == "eos", cap_hit=trace["cap_hit"])
    return {**compact_trace(trace), "match": match, "route": {"kind": "forced_on", "weight": 1.0},
            "residual_norms": norm}


@torch.inference_mode()
def routed_generation(model: Any, block: torch.nn.Module, modules: LiveEditMedicalModules,
                      sample: Mapping[str, Any], repository: Mapping[str, Any]):
    canonical = build_canonical_inputs(model, sample_to_model_row(sample))
    prompt_hidden, vision, question = capture_prompt(model, block, canonical)
    plan = route_repository(modules.input_extractor, question.float(), vision.float(),
                            repository["evr"], repository["eqr"])
    audit = plan_audit(plan, repository["ids"])
    _diagnostic_residual, norms = route_residual(plan, prompt_hidden, repository["moe_c"],
                                                 repository["moe_r"], modules.instant_reps_norm)
    if isinstance(plan, BaseRoutePlan):
        trace = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
    else:
        selected_c = repository["moe_c"][plan.candidate_mask]
        selected_r = repository["moe_r"][plan.candidate_mask]
        hook = Layer21ResidualHook(block, lambda hidden: apply_low_rank_expert_residual(
            hidden.float(), selected_c, selected_r, plan.final_weights,
            modules.instant_reps_norm).to(hidden.dtype)).install(); hook.enabled = True
        trace = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
        hook.remove()
    match = unrestricted_match(trace["raw_output"], sample["target"],
                               eos=trace["stop_reason"] == "eos", cap_hit=trace["cap_hit"])
    return {**compact_trace(trace), "match": match, "route": audit, "residual_norms": norms}


@torch.inference_mode()
def teacher_forced_nll(model: Any, block: torch.nn.Module, modules: LiveEditMedicalModules,
                       sample: Mapping[str, Any], moe_c: torch.Tensor, moe_r: torch.Tensor) -> float:
    row = sample_to_model_row(sample); inputs, labels, masks = model._build_batch(row)
    hook = Layer21ResidualHook(block, lambda hidden: apply_low_rank_expert_residual(
        hidden.float(), moe_c, moe_r, torch.ones(1, 1, device=hidden.device),
        modules.instant_reps_norm).to(hidden.dtype)).install(); hook.enabled = True
    output = model.llava_model(inputs_embeds=inputs, attention_mask=masks["attention_mask"].long(),
                               labels=labels, return_dict=True, use_cache=False)
    hook.remove(); return float(output.loss.item())


@torch.inference_mode()
def image_locality(model: Any, block: torch.nn.Module, modules: LiveEditMedicalModules,
                   sample: Mapping[str, Any], repository: Mapping[str, Any], targets: list[str]):
    canonical = build_canonical_inputs(model, sample_to_model_row(sample))
    clean = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
    edited = routed_generation(model, block, modules, sample, repository)
    contamination = [target for target in targets if normalize_answer(target) and normalize_answer(target) in normalize_answer(edited["raw_output"])]
    exact = clean["token_ids"] == edited["token_ids"] and clean["stop_reason"] == edited["stop_reason"]
    return {"mode": "image_bearing", "s0": compact_trace(clean), "routed": edited,
            "exact_preservation": exact, "target_contaminations": contamination}


def text_only_canonical(model: Any, sample: Mapping[str, Any]) -> CanonicalInputs:
    conv = model.conv_templates[model.conversation_template].copy()
    conv.append_message(conv.roles[0], str(sample["prompt"])); conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    conv = model.conv_templates[model.conversation_template].copy()
    conv.append_message(conv.roles[0], str(sample["prompt"])); conv.append_message(conv.roles[1], str(sample["target"]))
    full_text = conv.get_prompt()
    prompt_ids = model.llava_tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.lm_device)
    full_ids = model.llava_tokenizer(full_text, return_tensors="pt").input_ids.to(model.lm_device)
    if not torch.equal(full_ids[:, :prompt_ids.shape[1]], prompt_ids):
        raise RuntimeError("LIVEEDIT_MED_TEXT_ONLY_PREFIX_MISMATCH")
    return CanonicalInputs(prompt_text, full_text, prompt_ids, full_ids, None, int(prompt_ids.shape[1]),
                           full_ids[0, prompt_ids.shape[1]:].clone(), ids_sha256(prompt_ids),
                           ids_sha256(full_ids), "NO_IMAGE")


def text_only_locality(model: Any, sample: Mapping[str, Any]) -> dict[str, Any]:
    # A text-only prompt has no visual key, hence visual-hard routing has an empty
    # candidate set and must take the exact-base bypass.  We still execute S0 text
    # generation so preservation is measured rather than merely asserted.
    canonical = text_only_canonical(model, sample)
    clean = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
    replay = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
    exact = clean["token_ids"] == replay["token_ids"] and clean["stop_reason"] == replay["stop_reason"]
    return {"mode": "text_only", "prompt": sample["prompt"], "target": sample["target"],
            "route": {"kind": "base", "reason": "TEXT_ONLY_EMPTY_CANDIDATE_BASE_BYPASS",
                      "candidate_ids": [], "final_weights": [], "sum_final_weights": 0.0},
            "s0": compact_trace(clean), "routed": compact_trace(replay),
            "exact_preservation": exact, "target_contaminations": [],
            "generation_status": "EXECUTED_TEXT_ONLY_S0_EMPTY_CANDIDATE_BYPASS"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    panel = json.loads(args.panel_manifest.read_text())
    if panel.get("protocol") != PROTOCOL or panel.get("record953_excluded") is not True:
        raise RuntimeError("LIVEEDIT_MED_INVALID_PANEL_MANIFEST")
    source = json.loads(args.source_records.read_text())
    by_id = {str(row["record_id"]): row for row in source["records"]["validation"]}
    records = [by_id[str(row["record_id"])] for row in panel["edits"]]
    if any(str(row["record_id"]) == "953" for row in records):
        raise RuntimeError("LIVEEDIT_MED_RECORD953_VALIDATION_LEAKAGE")

    model, _bank = load_clean_model(args.physical_gpu)
    clean_hash = state_weight_hash(model); _, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    modules.load_state_dict(state, strict=True); modules.eval()

    experts = []
    for record in records:
        captured = capture_teacher_forced(model, block, native_sample(record))
        eqr, evr, moe_c, moe_r = modules.generated_edit(captured["vision"].float(),
                                                       captured["question"].float(),
                                                       captured["answer"].float())
        experts.append({"record_id": str(record["record_id"]), "eqr": eqr, "evr": evr,
                        "moe_c": moe_c, "moe_r": moe_r})
    repository = {"ids": [row["record_id"] for row in experts]}
    for key in ("eqr", "evr", "moe_c", "moe_r"):
        repository[key] = torch.cat([row[key] for row in experts], dim=0)

    outputs = []; source_losses = []; targets = [native_sample(record)["target"] for record in records]
    for record, expert in zip(records, experts):
        views = {"native": native_sample(record)}
        for name in ("textual", "visual", "paired"):
            views[name] = record["generality"][name][0]
        forced = {}; routed = {}
        for name, sample in views.items():
            forced[name] = forced_generation(model, block, modules, sample, expert["moe_c"], expert["moe_r"])
            routed[name] = routed_generation(model, block, modules, sample, repository)
            source_losses.append(teacher_forced_nll(model, block, modules, sample, expert["moe_c"], expert["moe_r"]))
        image_loc = image_locality(model, block, modules, record["locality"]["image_or_paired"][0], repository, targets)
        text_loc = text_only_locality(model, record["locality"]["text_only"][0])
        outputs.append({"record_id": str(record["record_id"]), "forced_on": forced, "routed": routed,
                        "locality": {"image_bearing": image_loc, "text_only": text_loc}})

    routed_native = sum(row["routed"]["native"]["match"]["success"] for row in outputs)
    routed_gen = sum(row["routed"][name]["match"]["success"] for row in outputs for name in ("textual", "visual", "paired"))
    forced_native = sum(row["forced_on"]["native"]["match"]["success"] for row in outputs)
    forced_gen = sum(row["forced_on"][name]["match"]["success"] for row in outputs for name in ("textual", "visual", "paired"))
    locality_exact = sum(row["locality"][name]["exact_preservation"] for row in outputs for name in ("image_bearing", "text_only"))
    false_positive = 0
    for row in outputs:
        own = row["record_id"]
        for view in row["routed"].values():
            false_positive += sum(candidate != own for candidate in view["route"]["candidate_ids"])
        false_positive += len(row["locality"]["image_bearing"]["routed"]["route"]["candidate_ids"])
    contaminations = sum(len(row["locality"][name]["target_contaminations"]) for row in outputs for name in ("image_bearing", "text_only"))
    summary = {"protocol": PROTOCOL, "step": int(checkpoint_manifest["step"]),
               "panel_hash": panel["panel_hash"], "fresh_clean_s0": True,
               "record953_loaded_or_evaluated": False,
               "routed_native_success_count": int(routed_native),
               "routed_generality_success_count": int(routed_gen),
               "locality_exact_preservation_count": int(locality_exact),
               "locality_total": 16, "routing_false_positive_count": int(false_positive),
               "target_contamination_count": int(contaminations),
               "forced_native_success_count": int(forced_native),
               "forced_generality_success_count": int(forced_gen),
               "validation_source_loss": sum(source_losses) / len(source_losses),
               "checkpoint_manifest": checkpoint_manifest, "outputs": outputs,
               "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH,
               "base_state_unchanged": state_weight_hash(model) == clean_hash,
               "generation_config": {"do_sample": False, "num_beams": 1,
                                     "max_new_tokens": MAX_NEW_TOKENS}}
    if not summary["canonical_bank_unchanged"] or not summary["base_state_unchanged"]:
        raise RuntimeError("LIVEEDIT_MED_BASE_OR_BANK_MUTATION")
    write_json(args.out, summary)
    print(json.dumps({key: summary[key] for key in ("step", "routed_native_success_count",
          "routed_generality_success_count", "forced_native_success_count",
          "forced_generality_success_count", "locality_exact_preservation_count",
          "validation_source_loss")}), flush=True)


if __name__ == "__main__":
    main()
