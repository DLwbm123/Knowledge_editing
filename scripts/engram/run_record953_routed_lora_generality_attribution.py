#!/usr/bin/env python3
"""Read-only attribution of record-953 routed-LoRA generality failure."""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import to_jsonable  # noqa: E402
from scripts.engram.generality_attribution_utils import (  # noqa: E402
    attribution_label,
    canonical_hash,
    classify_generality,
    diagnostic_separability,
    failed_router_conjuncts,
    nearest_by_components,
    primary_failed_side,
    sha256_file,
)
from scripts.engram.lora_positive_control_utils import adapter_hash, load_adapter_payload, load_adapter_state, positive_control_match, resolve_target_modules  # noqa: E402
from scripts.engram.routed_banked_lora_utils import load_router_bank, route_on  # noqa: E402
from scripts.engram.run_engram_v2_one_shot_natural_generation_rescue import bank_anchor_hash, content_indices, continuation_values, full_generation_parity  # noqa: E402
from scripts.engram.run_engram_v2_stage0_generation_audit import DATASET_PATH, MODEL_CONFIG, ORDER, apply_prefix, bank_manifest, eos_ids, load_model_views_bank, state_weight_hash  # noqa: E402
from scripts.engram.run_engram_v2_stage0abc_diagnostics import SHORT_INSTRUCTION  # noqa: E402
from scripts.engram.run_llavamed_record953_lora_positive_control import insert_lora, seed_everything  # noqa: E402
from scripts.engram.run_record953_routed_banked_lora_one_edit import CAP, EXPECTED_ANCHOR_HASH, EXPECTED_BANK_HASH, RECORD_ID, TARGET, TOL, extract_router_keys, generation_sample, locate_positive_control  # noqa: E402
from scripts.engram.run_record953_routed_banked_lora_v1_1 import PREVIOUS_COMMIT, visible_input_audit  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_greedy_trace  # noqa: E402


PROTOCOL = "RECORD953_ROUTED_LORA_GENERALITY_ATTRIBUTION_V1"
V11_ROOT = ROOT / "outputs/record953_routed_banked_lora_one_edit_v1_1"
V11_RUN_NAME = "20260811_equivalence_v1"
V11_PUBLISHED_COMMIT = "55ce6e8c20ad7b8b98e203422a7096db65e8cd39"
V11_SOURCE_HASHES = {
    "scripts/engram/equivalence_aware_router_utils.py": "fe005aaa1e32fe6dfa7e932bfbe745280e35f5d19e450ddb84a98e8f32bc43bb",
    "scripts/engram/run_record953_routed_banked_lora_v1_1.py": "3a0a54150c924f57da7184c78fb1e7da49739df43181a48812fa67404774d80f",
    "tests/test_record953_routed_banked_lora_v1_1.py": "6c2af8d7609933d70789d9838d4d05d3bbdfa7b659a55a8dd73ff6e8c371670b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--physical-gpu", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(value.rstrip() + "\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(to_jsonable(dict(value)), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def source_diff() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/engram/generality_attribution_utils.py",
        ROOT / "tests/test_record953_routed_lora_generality_attribution.py",
    )
    chunks: list[str] = []
    for path in paths:
        chunks.extend(difflib.unified_diff([], path.read_text().splitlines(True), fromfile="/dev/null", tofile=f"b/{path.relative_to(ROOT)}"))
    return "".join(chunks)


def read_scores(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in csv.DictReader(path.open()):
        row = dict(raw)
        for name in ("s_img", "s_text", "s_fused", "s_min", "s_joint"):
            row[name] = float(row[name])
        row["route_on"] = str(row["route_on"]).casefold() == "true"
        rows.append(row)
    return rows


def locate_v11_anchor() -> tuple[Path, dict[str, Any], dict[str, Any], Path, dict[str, torch.Tensor], dict[str, Any]]:
    candidates = []
    for summary_path in V11_ROOT.glob("*/routed_lora_v1_1_summary.json"):
        run = summary_path.parent
        try:
            summary = json.loads(summary_path.read_text())
            manifest = json.loads((run / "run_manifest.json").read_text())
            bank_item = run / "routed_adapter_bank_item"
            router_tensors, bank_meta = load_router_bank(bank_item)
            checks = (
                run.name == V11_RUN_NAME
                and summary.get("primary_label") == "PASS_ROUTED_BANKED_LORA_CORE_ONLY"
                and summary.get("target_route_on") is True
                and summary.get("target_output") == "The answer is completely ectocervical and fully visible."
                and summary.get("unique_calibration_negatives") == 13
                and summary.get("heldout_false_positive_count") == 0
                and summary.get("fixed_locality_exact_s0") is True
                and all(summary.get("reproducibility", {}).get(name) == "PASS" for name in ("reload", "fresh", "replay", "rollback"))
                and summary.get("canonical_bank_unchanged") is True
                and manifest.get("counts", {}).get("unique_heldout") == 40
                and manifest.get("canonical_bank_before") == manifest.get("canonical_bank_after") == EXPECTED_BANK_HASH
            )
            if checks:
                candidates.append((run, summary, manifest, bank_item, router_tensors, bank_meta))
        except (FileNotFoundError, KeyError, RuntimeError, ValueError, json.JSONDecodeError):
            continue
    if len(candidates) != 1:
        raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: V1.1 anchor is not unique")
    return candidates[0]


def short_sample(model: Any, entry: Mapping[str, Any]) -> dict[str, Any]:
    prompt = f"Question: {entry['question']} {SHORT_INSTRUCTION} Short answer: "
    if TARGET.casefold() in prompt.casefold():
        raise RuntimeError("Target leakage into short prompt")
    return {
        "image_path": [str(entry["image_path"])],
        "prompt": [prompt],
        "target": [TARGET],
        "text_input": [prompt + TARGET],
        "labels": model.llava_tokenizer(TARGET, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device),
    }


def compact_trace(model: Any, trace: Mapping[str, Any], aliases: Sequence[str]) -> dict[str, Any]:
    matched = positive_control_match(str(trace["raw_output"]), TARGET, eos=trace["stop_reason"] == "eos", cap_hit=bool(trace["cap_hit"]), aliases=aliases)
    return {
        "raw_output": trace["raw_output"],
        "token_ids": trace["token_ids"],
        "stop_reason": trace["stop_reason"],
        "eos_step": trace.get("eos_step"),
        "cap_hit": trace["cap_hit"],
        "match": {**matched, "target_completed_before_eos": bool((matched["canonical_target_span_match"] or matched["alias_match"]) and matched["eos_normal"])},
    }


def generate_manual(model: Any, sample: Mapping[str, Any], aliases: Sequence[str]) -> dict[str, Any]:
    canonical = build_canonical_inputs(model, sample)
    return compact_trace(model, manual_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=5), aliases)


def generate_with_parity(model: Any, sample: Mapping[str, Any], aliases: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = build_canonical_inputs(model, sample)
    parity = full_generation_parity(model, canonical)
    result = compact_trace(model, parity["no_cache"], aliases)
    proof = {
        "passed": parity["passed"],
        "manual_no_cache_token_ids": parity["no_cache"]["token_ids"],
        "manual_cached_token_ids": parity["cached"]["token_ids"],
        "hf_token_ids": parity["hf"]["token_ids"],
    }
    return result, proof


@torch.inference_mode()
def teacher_metrics(model: Any, sample: Mapping[str, Any]) -> dict[str, Any]:
    canonical = build_canonical_inputs(model, sample)
    indices = list(range(int(canonical.target_ids.numel())))
    content = content_indices(model, canonical.target_ids.tolist())
    values = continuation_values(model, canonical.prompt_ids, canonical.target_ids.tolist(), canonical.image, indices)
    per_token = []
    for index in indices:
        token_id = int(values["target_ids"][index].item())
        per_token.append({
            "target_index": index,
            "target_id": token_id,
            "target_text": model.llava_tokenizer.decode([token_id], skip_special_tokens=False),
            "rank": int(values["ranks"][index].item()),
            "margin": float(values["margins"][index].item()),
            "nll": float(values["nll"][index].item()),
            "top1_id": int(values["top1_ids"][index].item()),
        })
    first = int(content[0])
    return {
        "target_sequence_nll": float(values["nll"].mean().item()),
        "first_target_content_index": first,
        "first_target_content_token_id": int(values["target_ids"][first].item()),
        "first_target_content_rank": int(values["ranks"][first].item()),
        "first_target_content_margin": float(values["margins"][first].item()),
        "per_target_token": per_token,
    }


def source_metadata(entry: Mapping[str, Any], record: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    input_id = str(entry["input_id"])
    if input_id == "generality:textual":
        field, raw, category = "rephrase", record["rephrase"], "TEXTUAL_GENERALITY"
    elif input_id == "generality:visual":
        field, raw, category = "image_rephrase", record["image_rephrase"], "VISUAL_GENERALITY"
    elif input_id.startswith("generality:paired:"):
        index = int(input_id.rsplit(":", 1)[1])
        field, raw, category = f"port_new[{index}]", record["port_new"][index], "PAIRED_GENERALITY"
    else:
        raise RuntimeError(f"Unexpected generality input {input_id}")
    native_image = str(entry["pair_type"]) not in ("rephrased_image_original_question",)
    native_question = str(entry["pair_type"]) == "rephrased_image_original_question"
    classified = classify_generality(image_differs=not native_image, question_differs=not native_question, source_field="port_new" if input_id.startswith("generality:paired:") else field)
    if classified != category:
        raise RuntimeError("Generality category mismatch")
    return {"source_field_name": field, "raw_source_row_index": row_index, "raw_source_value": raw, "category": category}


def final_report(summary: Mapping[str, Any]) -> str:
    rows = summary["categories"]
    recommendation = summary["recommended_next_branch"]
    return f"""# Routed LoRA Generality Attribution Decision

- Frozen adapter succeeds on textual generality when forced ON: **{rows['TEXTUAL_GENERALITY']['always_on_unrestricted_success'] == rows['TEXTUAL_GENERALITY']['count']}**
- Frozen adapter succeeds on visual generality when forced ON: **{rows['VISUAL_GENERALITY']['always_on_unrestricted_success'] == rows['VISUAL_GENERALITY']['count']}**
- Frozen adapter succeeds on paired generality when forced ON: **{rows['PAIRED_GENERALITY']['always_on_unrestricted_success'] == rows['PAIRED_GENERALITY']['count']}**
- Examples failing because the frozen router is OFF: **{summary['router_limited_examples']}**
- Examples failing even with the adapter ON: **{summary['adapter_limited_examples']}**
- Short-answer-only successes: **{summary['format_conditioned_only_examples']}**
- Exact primary label: **`{summary['primary_label']}`**
- Single recommended next branch: **{recommendation}**
- Is Stage-2 permitted? **No**

This is a one-edit attribution audit with one source-grounded example per modality; it is not a statistically strong generality claim.
"""


def run(args: argparse.Namespace) -> None:
    seed_everything()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_text(out / "exact_command_log.txt", f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} {sys.executable} " + " ".join(sys.argv))
    write_text(out / "source_diff.patch", source_diff())
    write_text(out / "state_and_bank_hash_ledger.jsonl", "")
    for relative, expected in V11_SOURCE_HASHES.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: V1.1 source hash mismatch {relative}")
    run_dir, anchor_summary, anchor_manifest, bank_item, router_tensors, bank_meta = locate_v11_anchor()
    threshold_path = run_dir / "router_thresholds.json"
    thresholds_file = json.loads(threshold_path.read_text())
    thresholds = bank_meta["thresholds"]
    if any(abs(float(thresholds[name]) - float(thresholds_file[name])) > 1e-12 for name in ("tau_fused", "tau_min", "tau_joint")):
        raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: threshold mismatch")
    if sha256_file(run_dir / "router_prototype_keys.safetensors") != sha256_file(bank_item / "router_keys.safetensors"):
        raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: prototype key mismatch")
    adapter_reference = Path(bank_meta["adapter_reference"])
    adapter_state, adapter_meta = load_adapter_payload(adapter_reference)
    source_dir, source_summary, _source_manifest, source_adapter_meta = locate_positive_control()
    source_adapter = source_dir / "successful_adapter_bank_item"
    if (
        adapter_hash(adapter_state) != bank_meta["adapter_sha256"]
        or bank_meta["adapter_sha256"] != source_adapter_meta["adapter_sha256"]
        or sha256_file(adapter_reference / "adapter.pt") != bank_meta["adapter_file_sha256"]
        or adapter_reference.resolve() != source_adapter.resolve()
        or adapter_meta["rank"] != 16 or adapter_meta["alpha"] != 32 or adapter_meta["training_step"] != 10
    ):
        raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: frozen adapter mismatch")
    identifiers = next(entry["identifiers"] for entry in bank_meta["routing_inputs"] if entry["input_id"] == "target:953:original")
    anchor_hashes = {
        "v1_1_run_directory": str(run_dir), "v1_1_summary_sha256": sha256_file(run_dir / "routed_lora_v1_1_summary.json"),
        "v1_1_manifest_sha256": sha256_file(run_dir / "run_manifest.json"), "routed_bank_manifest_sha256": sha256_file(bank_item / "manifest.json"),
        "prototype_key_file_sha256": sha256_file(bank_item / "router_keys.safetensors"), "threshold_file_sha256": sha256_file(threshold_path),
        "clean_s0_hash": bank_meta["base_s0_hash"], "adapter_reference": str(adapter_reference), "adapter_sha256": bank_meta["adapter_sha256"],
        "adapter_file_sha256": bank_meta["adapter_file_sha256"], "rank": adapter_meta["rank"], "alpha": adapter_meta["alpha"],
        "processor_tokenizer_chat_template_identifiers": identifiers, "bank_recorded_code_commit": bank_meta.get("code_commit"),
        "v1_1_published_source_commit": V11_PUBLISHED_COMMIT, "v1_1_source_hashes": V11_SOURCE_HASHES,
        "positive_control_primary_label": source_summary["primary_label"], "positive_control_first_success_step": source_summary["success_step"],
        "canonical_engram_bank_hash": bank_meta["canonical_engram_bank_hash"],
    }
    write_json(out / "source_anchor_and_hashes.json", anchor_hashes)

    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 0)
    clean_hash = state_weight_hash(model)
    if clean_hash != bank_meta["base_s0_hash"] or bank_manifest()["sha256"] != EXPECTED_BANK_HASH or bank_anchor_hash() != EXPECTED_ANCHOR_HASH:
        raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: clean S0 mismatch")
    append_jsonl(out / "state_and_bank_hash_ledger.jsonl", {"event": "CLEAN_S0", "state_weight_hash": clean_hash, "canonical_bank_hash": bank_manifest()["sha256"], "adapter_present": False})
    record = records[RECORD_ID]
    rows_raw = json.loads(DATASET_PATH.read_text())
    row_index = next(index for index, row in enumerate(rows_raw) if str(row["id"]) == RECORD_ID)
    target_entry = next(entry for entry in bank_meta["routing_inputs"] if entry["input_id"] == "target:953:original")
    generality_entries = [entry for entry in bank_meta["routing_inputs"] if str(entry["input_id"]).startswith("generality:")]
    if len(generality_entries) != 3:
        raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: unexpected generality count")
    positive_key = target_entry["router_input_equivalence_key"]
    negative_keys = {entry["router_input_equivalence_key"] for entry in [*bank_meta["calibration_equivalence_classes"], *bank_meta["heldout_equivalence_classes"]]}
    anchor_score_rows = read_scores(run_dir / "routing_scores.csv")
    calibration_scores = [row for row in anchor_score_rows if row["group"] == "calibration"]
    heldout_scores = [row for row in anchor_score_rows if row["group"] in ("heldout_fixed_ten", "heldout_locality")]
    previous_gen = json.loads((run_dir / "target_and_generality_generation.json").read_text())["generality"]
    previous_by_id = {row["input_id"]: row for row in previous_gen}
    prototype = {name: router_tensors[f"p_{name}"] for name in ("img", "text", "fused")}

    input_audit, router_rows, runtime_entries = [], [], []
    for bank_entry in generality_entries:
        entry = dict(bank_entry)
        visible = visible_input_audit(model, entry)
        if visible["router_input_equivalence_key"] != entry["router_input_equivalence_key"]:
            raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: input equivalence drift")
        metadata = source_metadata(entry, record, row_index)
        collision_positive = entry["router_input_equivalence_key"] == positive_key
        collision_negative = entry["router_input_equivalence_key"] in negative_keys
        if collision_positive or collision_negative:
            raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: generality equivalence collision")
        keys, representation_spec = extract_router_keys(model, entry)
        from scripts.engram.equivalence_aware_router_utils import clamped_router_scores
        scores = clamped_router_scores(keys, prototype)
        decision = route_on(scores, thresholds, TOL)
        previous = next(row for row in anchor_score_rows if row["input_id"] == entry["input_id"])
        if decision != previous["route_on"] or any(abs(float(scores[name]) - float(previous[name])) > TOL for name in ("s_img", "s_text", "s_fused", "s_min", "s_joint")):
            raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: V1.1 route mismatch")
        failed = failed_router_conjuncts(scores, thresholds, TOL)
        router_row = {
            "input_id": entry["input_id"], "category": metadata["category"], **scores,
            "tau_fused": thresholds["tau_fused"], "tau_min": thresholds["tau_min"], "tau_joint": thresholds["tau_joint"],
            "d_fused": scores["s_fused"] - thresholds["tau_fused"], "d_min": scores["s_min"] - thresholds["tau_min"], "d_joint": scores["s_joint"] - thresholds["tau_joint"],
            "route_on": decision, "failed_conjuncts": failed, "primary_failed_side": primary_failed_side(scores, thresholds, TOL),
            "nearest_calibration": nearest_by_components(scores, calibration_scores), "nearest_heldout": nearest_by_components(scores, heldout_scores),
            "distance_to_prototype": {name: 1.0 - float(scores[name]) for name in ("s_img", "s_text", "s_fused", "s_min", "s_joint")},
        }
        router_rows.append(router_row)
        input_audit.append({
            "input_id": entry["input_id"], **metadata, "image_path": entry["image_path"], "image_sha256": entry["image_sha256"],
            "processed_pixel_tensor_sha256": visible["processed_pixel_tensor_sha256"], "original_image_size_wh": visible["original_image_size_wh"],
            "image_sizes": visible["image_sizes_equivalence_field"], "raw_question": entry["question"], "normalized_question": visible["normalized_original_question"],
            "canonical_routing_prompt_sha256": hashlib.sha256(visible["rendered_canonical_routing_prompt"].encode()).hexdigest(),
            "routing_input_ids_sha256": canonical_hash(visible["routing_input_ids"]), "router_input_equivalence_key": visible["router_input_equivalence_key"],
            "distinct_from_exact_target_prototype": not collision_positive, "negative_equivalence_collision": collision_negative,
        })
        runtime_entries.append({"entry": entry, "metadata": metadata, "router": router_row, "representation": representation_spec})
    write_csv(out / "generality_input_audit.csv", input_audit, ["input_id", "category", "source_field_name", "raw_source_row_index", "raw_source_value", "image_path", "image_sha256", "processed_pixel_tensor_sha256", "original_image_size_wh", "image_sizes", "raw_question", "normalized_question", "canonical_routing_prompt_sha256", "routing_input_ids_sha256", "router_input_equivalence_key", "distinct_from_exact_target_prototype", "negative_equivalence_collision"])
    write_csv(out / "generality_router_scores.csv", router_rows, ["input_id", "category", "s_img", "s_text", "s_fused", "s_min", "s_joint", "tau_fused", "tau_min", "tau_joint", "d_fused", "d_min", "d_joint", "route_on", "failed_conjuncts", "primary_failed_side", "nearest_calibration", "nearest_heldout", "distance_to_prototype"])

    aliases = [str(value) for value in (record.get("accepted_answers") or [])]
    conditions: dict[str, Any] = {}
    parity_results: dict[str, Any] = {}
    teacher: dict[str, Any] = {}
    for item in runtime_entries:
        entry, category = item["entry"], item["metadata"]["category"]
        unrestricted = generation_sample(model, entry["image_path"], entry["question"], TARGET)
        short = short_sample(model, entry)
        if TARGET.casefold() in str(unrestricted["prompt"][0]).casefold():
            raise RuntimeError("Target leakage into unrestricted prompt")
        s0_unrestricted = generate_manual(model, unrestricted, aliases)
        s0_short = generate_manual(model, short, aliases)
        s0_teacher = teacher_metrics(model, unrestricted)
        conditions[entry["input_id"]] = {"category": category, "route_on": item["router"]["route_on"], "S0_ADAPTER_OFF": {"unrestricted": s0_unrestricted, "short_answer": s0_short}}
        teacher[entry["input_id"]] = {"category": category, "S0_ADAPTER_OFF": s0_teacher}

    resolved = resolve_target_modules(model.llava_model.named_modules())
    if resolved != adapter_meta["resolved_lora_modules"]:
        raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: module mismatch")
    model.llava_model = insert_lora(model.llava_model, resolved)
    peft = model.llava_model
    load_adapter_state(peft.named_parameters(), adapter_state)
    peft.enable_adapter_layers()
    append_jsonl(out / "state_and_bank_hash_ledger.jsonl", {"event": "FROZEN_ADAPTER_ALWAYS_ON", "adapter_sha256": adapter_hash(adapter_state), "router_threshold_hash": sha256_file(threshold_path), "parameters_updated": False})
    for item in runtime_entries:
        entry = item["entry"]
        unrestricted = generation_sample(model, entry["image_path"], entry["question"], TARGET)
        short = short_sample(model, entry)
        always_unrestricted, parity = generate_with_parity(model, unrestricted, aliases)
        always_short = generate_manual(model, short, aliases)
        on_teacher = teacher_metrics(model, unrestricted)
        base = teacher[entry["input_id"]]["S0_ADAPTER_OFF"]
        teacher[entry["input_id"]].update({
            "ADAPTER_ALWAYS_ON": on_teacher,
            "delta_on_minus_s0": {
                "target_sequence_nll": on_teacher["target_sequence_nll"] - base["target_sequence_nll"],
                "first_target_content_rank": on_teacher["first_target_content_rank"] - base["first_target_content_rank"],
                "first_target_content_margin": on_teacher["first_target_content_margin"] - base["first_target_content_margin"],
            },
        })
        conditions[entry["input_id"]]["ADAPTER_ALWAYS_ON"] = {"unrestricted": always_unrestricted, "short_answer": always_short, "router_bypassed_for_evaluation_only": True}
        parity_results[entry["input_id"]] = {"category": item["metadata"]["category"], "ADAPTER_ALWAYS_ON_UNRESTRICTED": parity}

    for item in runtime_entries:
        entry = item["entry"]
        unrestricted = generation_sample(model, entry["image_path"], entry["question"], TARGET)
        short = short_sample(model, entry)
        if item["router"]["route_on"]:
            peft.enable_adapter_layers()
            routed_unrestricted, routed_parity = generate_with_parity(model, unrestricted, aliases)
            routed_short = generate_manual(model, short, aliases)
            parity_results[entry["input_id"]]["ROUTER_GATED_UNRESTRICTED"] = routed_parity
        else:
            peft.disable_adapter_layers()
            routed_unrestricted = generate_manual(model, unrestricted, aliases)
            routed_short = generate_manual(model, short, aliases)
        previous = previous_by_id[entry["input_id"]]["generation"]
        if routed_unrestricted["token_ids"] != previous["token_ids"] or routed_unrestricted["raw_output"] != previous["raw_output"]:
            raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: routed output mismatch")
        conditions[entry["input_id"]]["ROUTER_GATED"] = {"unrestricted": routed_unrestricted, "short_answer": routed_short, "adapter_state": "ON" if item["router"]["route_on"] else "OFF_S0"}
    peft.disable_adapter_layers()
    append_jsonl(out / "state_and_bank_hash_ledger.jsonl", {"event": "ROUTER_GATED_COMPLETE", "adapter_enabled": False, "router_unchanged": True, "parameters_updated": False})
    write_json(out / "three_condition_generality_outputs.json", conditions)
    write_json(out / "generality_generation_parity.json", parity_results)
    write_json(out / "generality_teacher_forced_metrics.json", teacher)

    separability = {}
    for row in router_rows:
        separability[row["input_id"]] = {"category": row["category"], **diagnostic_separability(row["category"], row, calibration_scores, TOL), "diagnostic_only": True, "deployable_threshold_derived": False}
    write_json(out / "generality_representation_separability.json", separability)
    modality_rows = []
    for category in ("TEXTUAL_GENERALITY", "VISUAL_GENERALITY", "PAIRED_GENERALITY"):
        ids = [item["entry"]["input_id"] for item in runtime_entries if item["metadata"]["category"] == category]
        deltas = [teacher[value]["delta_on_minus_s0"] for value in ids]
        modality_rows.append({
            "category": category, "example_count": len(ids),
            "frozen_router_on_count": sum(conditions[value]["route_on"] for value in ids),
            "s0_unrestricted_success_count": sum(conditions[value]["S0_ADAPTER_OFF"]["unrestricted"]["match"]["success"] for value in ids),
            "always_on_unrestricted_success_count": sum(conditions[value]["ADAPTER_ALWAYS_ON"]["unrestricted"]["match"]["success"] for value in ids),
            "routed_unrestricted_success_count": sum(conditions[value]["ROUTER_GATED"]["unrestricted"]["match"]["success"] for value in ids),
            "always_on_short_success_count": sum(conditions[value]["ADAPTER_ALWAYS_ON"]["short_answer"]["match"]["success"] for value in ids),
            "mean_target_nll_change_on_minus_s0": sum(row["target_sequence_nll"] for row in deltas) / len(deltas),
            "mean_first_content_rank_change_on_minus_s0": sum(row["first_target_content_rank"] for row in deltas) / len(deltas),
            "mean_first_content_margin_change_on_minus_s0": sum(row["first_target_content_margin"] for row in deltas) / len(deltas),
            "always_on_parity_pass": all(parity_results[value]["ADAPTER_ALWAYS_ON_UNRESTRICTED"]["passed"] for value in ids),
            "failed_router_conjuncts": {value: next(row["failed_conjuncts"] for row in router_rows if row["input_id"] == value) for value in ids},
            "diagnostic_separability_status": {value: separability[value]["status"] for value in ids},
        })
    write_csv(out / "modality_level_summary.csv", modality_rows, ["category", "example_count", "frozen_router_on_count", "s0_unrestricted_success_count", "always_on_unrestricted_success_count", "routed_unrestricted_success_count", "always_on_short_success_count", "mean_target_nll_change_on_minus_s0", "mean_first_content_rank_change_on_minus_s0", "mean_first_content_margin_change_on_minus_s0", "always_on_parity_pass", "failed_router_conjuncts", "diagnostic_separability_status"])

    always_success = [value for value in conditions if conditions[value]["ADAPTER_ALWAYS_ON"]["unrestricted"]["match"]["success"]]
    always_short = [value for value in conditions if conditions[value]["ADAPTER_ALWAYS_ON"]["short_answer"]["match"]["success"]]
    label = attribution_label(len(always_success), len(conditions), len(always_short))
    router_limited = [value for value in always_success if not conditions[value]["route_on"]]
    adapter_limited = [value for value in conditions if value not in always_success]
    format_only = [value for value in adapter_limited if value in always_short]
    if label == "GENERALITY_ROUTER_RECALL_BOTTLENECK":
        recommendation = "Keep the adapter frozen; pre-register a modality-aware router and evaluate it on additional held-out positives."
    elif label == "GENERALITY_ADAPTER_MEMORIZATION_BOTTLENECK":
        recommendation = "Train a generality-aware adapter from source-grounded positive views while preserving the high-precision router/locality protocol."
    elif label == "GENERALITY_FORMAT_CONDITIONED_ONLY":
        recommendation = "Study format-robust adapter training before changing router thresholds."
    else:
        succeeded_categories = sorted({conditions[value]["category"] for value in always_success})
        failed_categories = sorted({conditions[value]["category"] for value in adapter_limited})
        recommendation = f"Run a modality-specific follow-up: router-only for {succeeded_categories}, adapter-generalization for {failed_categories}; do not globally lower thresholds."
    category_summary = {row["category"]: {"count": row["example_count"], "router_on": row["frozen_router_on_count"], "always_on_unrestricted_success": row["always_on_unrestricted_success_count"], "routed_unrestricted_success": row["routed_unrestricted_success_count"], "always_on_short_success": row["always_on_short_success_count"]} for row in modality_rows}
    summary = {
        "primary_label": label, "categories": category_summary, "router_limited_examples": router_limited,
        "adapter_limited_examples": adapter_limited, "format_conditioned_only_examples": format_only,
        "always_on_unrestricted_success_count": len(always_success), "total_generality_examples": len(conditions),
        "recommended_next_branch": recommendation, "one_edit_attribution_only": True,
        "router_modified": False, "thresholds_recalibrated": False, "adapter_modified": False,
        "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH, "stage2_permitted": False,
    }
    base = peft.unload()
    model.llava_model = base
    apply_prefix(model, bank, 0)
    rollback = state_weight_hash(model) == clean_hash
    if not rollback or bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN: rollback or bank mismatch")
    append_jsonl(out / "state_and_bank_hash_ledger.jsonl", {"event": "FINAL_ROLLBACK", "state_weight_hash": state_weight_hash(model), "clean_hash_match": rollback, "canonical_bank_hash": bank_manifest()["sha256"]})
    write_json(out / "generality_attribution_summary.json", summary)
    write_json(out / "run_manifest.json", {
        "protocol": PROTOCOL, "cwd": str(ROOT), "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "physical_gpu": args.physical_gpu,
        "seed": 42, "record_id": RECORD_ID, "v1_1_anchor": anchor_hashes, "v1_1_primary_label": anchor_summary["primary_label"],
        "frozen_thresholds": thresholds, "generality_count": len(conditions), "primary_label": label,
        "adapter_trained": False, "router_modified": False, "thresholds_recalibrated": False, "heldout_negatives_rerun": False, "locality_suite_rerun": False,
        "new_bank_created": False, "stage2_launched": False, "canonical_bank_before": EXPECTED_BANK_HASH, "canonical_bank_after": bank_manifest()["sha256"],
        "required_outputs_complete": True,
    })
    write_text(out / "GENERALITY_ATTRIBUTION_FINAL_DECISION.md", final_report(summary))


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must equal {args.physical_gpu}")
    run(args)


if __name__ == "__main__":
    main()
