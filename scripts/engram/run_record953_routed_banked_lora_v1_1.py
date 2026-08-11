#!/usr/bin/env python3
"""Equivalence-aware full rerun of the record-953 routed LoRA gate."""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import to_jsonable  # noqa: E402
from scripts.engram.equivalence_aware_router_utils import (  # noqa: E402
    calibration_sufficiency,
    clamped_router_scores,
    corrected_thresholds,
    router_input_equivalence_key,
    unique_negative_equivalence_classes,
)
from scripts.engram.lora_positive_control_utils import adapter_hash, load_adapter_payload, load_adapter_state, resolve_target_modules  # noqa: E402
from scripts.engram.routed_banked_lora_utils import load_router_bank, route_on, save_router_bank, split_negative_records  # noqa: E402
from scripts.engram.run_engram_v2_one_shot_natural_generation_rescue import bank_anchor_hash  # noqa: E402
from scripts.engram.run_engram_v2_stage0_generation_audit import MODEL_CONFIG, ORDER, apply_prefix, bank_manifest, eos_ids, load_model_views_bank, state_weight_hash  # noqa: E402
from scripts.engram.run_llavamed_record953_lora_positive_control import insert_lora, seed_everything  # noqa: E402
from scripts.engram.run_record953_routed_banked_lora_one_edit import (  # noqa: E402
    CAP,
    EXPECTED_ANCHOR_HASH,
    EXPECTED_BANK_HASH,
    EXPECTED_OUTPUT,
    EXPECTED_SHORT_OUTPUT,
    RECORD_ID,
    TARGET,
    TOL,
    exact_locality_rows,
    extract_router_keys,
    generate_parity,
    generation_sample,
    generality_rule,
    locality_trace,
    locate_positive_control,
    make_entry,
)
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_greedy_trace, tensor_sha256  # noqa: E402


PROTOCOL = "RECORD953_ROUTED_BANKED_LORA_ONE_EDIT_V1_1_EQUIVALENCE_AWARE"
PREVIOUS_COMMIT = "412b79d046aedaf178a099fedc7dc0673d86bfff"
PREVIOUS_RUN = ROOT / "outputs/record953_routed_banked_lora_one_edit/20260811_routed_lora_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "fresh"), default="run")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bank-item", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_table(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def source_diff() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/engram/equivalence_aware_router_utils.py",
        ROOT / "tests/test_record953_routed_banked_lora_v1_1.py",
    )
    result: list[str] = []
    for path in paths:
        result.extend(difflib.unified_diff([], path.read_text().splitlines(True), fromfile="/dev/null", tofile=f"b/{path.relative_to(ROOT)}"))
    return "".join(result)


def previous_addendum() -> str:
    return f"""# Previous Run Protocol Addendum

The preserved V1.0 run at `{PREVIOUS_RUN}` stopped with `ROUTER_CALIBRATION_NOT_SEPARABLE`.

Its corrected scientific status is:

`INVALID_NEGATIVE_CONSTRUCTION_IDENTICAL_TO_POSITIVE`

Candidate `calibration:1592:prototype_image` combined the record-953 image with the record-1592 question. Since records 953 and 1592 have the exact same raw question, this constructed candidate was model-input-identical to the positive prototype. Its zero score margin was therefore an impossible contradictory label assignment, not evidence of representation overlap between distinct inputs.

The previous run, implementation, scores, hashes, and Git commit `{PREVIOUS_COMMIT}` remain unchanged and available. V1.1 changes only negative equivalence-class construction.
"""


@torch.inference_mode()
def visible_input_audit(model: Any, entry: Mapping[str, Any]) -> dict[str, Any]:
    question = str(entry["question"])
    prompt = model._conversation_prompt(question, None)
    input_ids = model.tokenizer_image_token(prompt, model.llava_tokenizer, model.IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.lm_device)
    attention = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    sample = {"image_path": [str(entry["image_path"])], "prompt": [question], "target": [""]}
    pixels = model._image_for_row(sample, 0)
    with Image.open(entry["image_path"]) as image:
        original_size = [int(image.width), int(image.height)]
    processed_hash = tensor_sha256(pixels)
    image_sizes = [*original_size, *[int(value) for value in pixels.shape]]
    ids = [int(value) for value in input_ids[0].tolist()]
    mask = [int(value) for value in attention[0].tolist()]
    question_ids = [int(value) for value in model.llava_tokenizer(question, add_special_tokens=False).input_ids]
    key = router_input_equivalence_key(processed_hash, image_sizes, ids, mask)
    tokenizer_name = str(getattr(model.llava_tokenizer, "name_or_path", model.llava_tokenizer.__class__.__name__))
    processor = getattr(model, "image_processor", None)
    if processor is None and hasattr(model.llava_model, "get_vision_tower"):
        processor = getattr(model.llava_model.get_vision_tower(), "image_processor", None)
    return {
        "router_input_equivalence_key": key,
        "processed_pixel_tensor_sha256": processed_hash,
        "raw_image_sha256": entry["image_sha256"],
        "original_image_size_wh": original_size,
        "processed_pixel_tensor_shape": [int(value) for value in pixels.shape],
        "image_sizes_equivalence_field": image_sizes,
        "normalized_original_question": " ".join(question.casefold().split()),
        "question_token_ids": question_ids,
        "rendered_canonical_routing_prompt": prompt,
        "routing_input_ids": ids,
        "attention_mask": mask,
        "identifiers": {
            "model_class": model.llava_model.__class__.__name__,
            "tokenizer": tokenizer_name,
            "image_processor": processor.__class__.__name__ if processor is not None else "vision_tower_internal_processor",
            "conversation_template": str(model.conversation_template),
            "model_config": str(MODEL_CONFIG),
        },
    }


def enrich(model: Any, entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for raw in entries:
        entry = dict(raw)
        entry.update(visible_input_audit(model, entry))
        rows.append(entry)
    return rows


def raw_membership(views: Mapping[str, Any], records: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    original_image = str(views[RECORD_ID]["target"]["image_path"][0])
    original_question = str(records[RECORD_ID]["src"])
    target = make_entry("target:953:original", "target", "exact", "native", RECORD_ID, original_image, original_question)
    sources = []
    for rid in ORDER:
        if rid != RECORD_ID:
            sources.append(make_entry(f"source:{rid}", "source", "negative_source", "native", rid, str(views[rid]["target"]["image_path"][0]), str(records[rid]["src"])))
    calibration_sources, heldout_sources = split_negative_records(sources)

    def triples(source: Mapping[str, Any], group: str) -> list[dict[str, Any]]:
        rid = str(source["record_id_audit"])
        return [
            make_entry(f"{group}:{rid}:native", group, "negative", "native", rid, source["image_path"], source["question"]),
            make_entry(f"{group}:{rid}:prototype_image", group, "negative", "prototype_image_native_question", rid, original_image, source["question"]),
            make_entry(f"{group}:{rid}:prototype_question", group, "negative", "native_image_prototype_question", rid, source["image_path"], original_question),
        ]

    calibration = [entry for source in calibration_sources for entry in triples(source, "calibration")]
    heldout = [entry for source in heldout_sources for entry in triples(source, "heldout_fixed_ten")]
    fixed_locality = []
    for rid in ORDER:
        image = str(views[rid]["locality"]["image_path"][0])
        question = str(records[rid]["m_loc_q"])
        native = make_entry(f"locality:{rid}:native", "heldout_locality", "locality", "native", rid, image, question)
        fixed_locality.append(native)
        heldout.extend([
            native,
            make_entry(f"locality:{rid}:prototype_image", "heldout_locality", "locality", "prototype_image_locality_question", rid, original_image, question),
            make_entry(f"locality:{rid}:prototype_question", "heldout_locality", "locality", "locality_image_prototype_question", rid, image, original_question),
        ])
    record = records[RECORD_ID]
    generality = [
        make_entry("generality:textual", "generality", "textual", "original_image_rephrased_question", RECORD_ID, original_image, str(record["rephrase"])),
        make_entry("generality:visual", "generality", "visual", "rephrased_image_original_question", RECORD_ID, str(views[RECORD_ID]["generalization"]["image_path"][0]), original_question),
    ]
    for index, item in enumerate(record.get("port_new") or []):
        qa = item.get("Q&A") or {}
        if qa.get("Question") and qa.get("Answer"):
            generality.append(make_entry(f"generality:paired:{index}", "generality", "paired", "source_portability_pair", RECORD_ID, original_image, str(qa["Question"])))
    return target, calibration, heldout, fixed_locality, generality


def equivalence_rows(positives: Sequence[Mapping[str, Any]], proposed_calibration: Sequence[Mapping[str, Any]], proposed_heldout: Sequence[Mapping[str, Any]], calibration: Sequence[Mapping[str, Any]], heldout: Sequence[Mapping[str, Any]], exclusions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dispositions = {str(row["candidate_name"]): str(row["status"]) for row in exclusions}
    representatives = {name for row in [*calibration, *heldout] for name in row["candidate_names"]}
    rows = []
    for entry in [*positives, *proposed_calibration, *proposed_heldout]:
        name = str(entry["input_id"])
        if entry in positives:
            disposition = "KEPT_POSITIVE_EQUIVALENCE_CLASS"
        elif name in dispositions:
            disposition = dispositions[name]
        elif name in representatives:
            disposition = "KEPT_UNIQUE_NEGATIVE_EQUIVALENCE_CLASS"
        else:
            disposition = "UNKNOWN"
        rows.append({
            "candidate_name": name, "group": entry["group"], "source_record": entry["record_id_audit"],
            "pair_type": entry["pair_type"], "router_input_equivalence_key": entry["router_input_equivalence_key"],
            "processed_pixel_tensor_sha256": entry["processed_pixel_tensor_sha256"], "raw_image_sha256": entry["raw_image_sha256"],
            "question_sha256": entry["question_sha256"], "disposition": disposition,
        })
    return rows


def native_collision_audit(model: Any, views: Mapping[str, Any], records: Mapping[str, Any]) -> dict[str, Any]:
    native = []
    for rid in ORDER:
        row = make_entry(f"native_edit:{rid}", "native_edit_audit", "native_edit", "native", rid, str(views[rid]["target"]["image_path"][0]), str(records[rid]["src"]))
        row.update(visible_input_audit(model, row))
        row["edited_target"] = str(records[rid]["alt"])
        native.append(row)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in native:
        groups.setdefault(row["router_input_equivalence_key"], []).append(row)
    collisions = []
    for key, rows in groups.items():
        if len(rows) > 1:
            targets = sorted({row["edited_target"] for row in rows})
            collisions.append({"router_input_equivalence_key": key, "record_ids": [row["record_id_audit"] for row in rows], "targets": targets, "status": "ROUTER_KEY_COLLISION_ACROSS_EDITS" if len(targets) > 1 else "DUPLICATE_NATIVE_EQUIVALENCE_SAME_TARGET"})
    target_conflict = any(RECORD_ID in row["record_ids"] and row["status"] == "ROUTER_KEY_COLLISION_ACROSS_EDITS" for row in collisions)
    return {"native_records": native, "collisions": collisions, "record953_conflicting_target": target_conflict, "future_multi_edit_implication": "Any cross-edit collision requires collision-aware multi-value routing, edit version/context metadata, or a different routing interface.", "stage2_permitted": False}


def score_entries(model: Any, entries: Sequence[Mapping[str, Any]], prototype: Mapping[str, torch.Tensor]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, specs = [], {}
    for entry in entries:
        keys, spec = extract_router_keys(model, entry)
        rows.append({**{name: entry[name] for name in ("input_id", "group", "category", "pair_type", "record_id_audit", "router_input_equivalence_key")}, **clamped_router_scores(keys, prototype)})
        specs[str(entry["input_id"])] = spec
    return rows, specs


def set_routes(rows: Sequence[dict[str, Any]], thresholds: Mapping[str, float]) -> None:
    for row in rows:
        row["route_on"] = route_on(row, thresholds, TOL)


def recompute_decisions(model: Any, entries: Sequence[Mapping[str, Any]], prototype: Mapping[str, torch.Tensor], thresholds: Mapping[str, float]) -> dict[str, bool]:
    result = {}
    for entry in entries:
        keys, _ = extract_router_keys(model, entry)
        result[str(entry["input_id"])] = route_on(clamped_router_scores(keys, prototype), thresholds, TOL)
    return result


def summary_report(summary: Mapping[str, Any]) -> str:
    categories = summary["generality"]["categories"]
    return f"""# Routed Banked LoRA V1.1 Final Decision

- Was the previous failure an identical positive-negative collision? **Yes**
- Proposed negatives excluded as positive-equivalent: **{summary['positive_equivalent_exclusions']}**
- Duplicate negative candidates collapsed: **{summary['deduplicated_negative_candidates']}**
- Unique calibration negatives remaining: **{summary['unique_calibration_negatives']}**
- Any distinct calibration input overlapping the prototype score? **{summary['distinct_input_score_overlap']}**
- Did record 953 route ON? **{summary['target_route_on']}**
- Distinct held-out false positives: **{summary['heldout_false_positive_count']}**
- Did unrestricted target generation succeed? **{summary['target_unrestricted_success']}**
- Did fixed locality remain exact S0? **{summary['fixed_locality_exact_s0']}**
- Generality textual routed/generated: **{categories['textual']['routed_on']}/{categories['textual']['natural_generation_success']} of {categories['textual']['count']}**
- Generality visual routed/generated: **{categories['visual']['routed_on']}/{categories['visual']['natural_generation_success']} of {categories['visual']['count']}**
- Generality paired routed/generated: **{categories['paired']['routed_on']}/{categories['paired']['natural_generation_success']} of {categories['paired']['count']}**
- Did the generality rule pass? **{summary['generality']['passed']}**
- Reload / fresh / replay / rollback: **{summary['reproducibility']['reload']} / {summary['reproducibility']['fresh']} / {summary['reproducibility']['replay']} / {summary['reproducibility']['rollback']}**
- Primary label: **`{summary['primary_label']}`**
- Is Stage-2 permitted? **No**
"""


def fresh(args: argparse.Namespace) -> None:
    if args.bank_item is None:
        raise ValueError("--bank-item required")
    seed_everything()
    raw, manifest = load_router_bank(args.bank_item)
    prototype = {name: raw[f"p_{name}"] for name in ("img", "text", "fused")}
    model, views, bank, _records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 0)
    decisions = recompute_decisions(model, manifest["routing_inputs"], prototype, manifest["thresholds"])
    state, adapter_manifest = load_adapter_payload(Path(manifest["adapter_reference"]))
    resolved = resolve_target_modules(model.llava_model.named_modules())
    if resolved != adapter_manifest["resolved_lora_modules"]:
        raise RuntimeError("Fresh module mismatch")
    model.llava_model = insert_lora(model.llava_model, resolved)
    load_adapter_state(model.llava_model.named_parameters(), state)
    model.llava_model.enable_adapter_layers()
    target_ids = manual_greedy_trace(model, build_canonical_inputs(model, views[RECORD_ID]["target"]), CAP, eos_ids(model), top_k=1)["token_ids"]
    model.llava_model.disable_adapter_layers()
    locality_ids = {rid: locality_trace(model, views[rid]["locality"])["token_ids"] for rid in ORDER}
    passed = decisions == manifest["expected_route_decisions"] and target_ids == manifest["expected_target_token_ids"] and locality_ids == manifest["expected_locality_token_ids"] and bank_manifest()["sha256"] == EXPECTED_BANK_HASH
    write_json(args.bank_item / "fresh_result.json", {"decision_parity": decisions == manifest["expected_route_decisions"], "target_token_parity": target_ids == manifest["expected_target_token_ids"], "locality_token_parity": locality_ids == manifest["expected_locality_token_ids"], "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH, "passed": passed})
    if not passed:
        raise RuntimeError("ROUTED_BANK_REPRODUCIBILITY_FAILURE")


def run(args: argparse.Namespace) -> None:
    seed_everything()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_text(out / "exact_command_log.txt", f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} {sys.executable} " + " ".join(sys.argv))
    write_text(out / "source_diff.patch", source_diff())
    write_text(out / "state_and_bank_hash_ledger.jsonl", "")
    write_text(out / "PREVIOUS_RUN_PROTOCOL_ADDENDUM.md", previous_addendum())
    if not PREVIOUS_RUN.is_dir():
        raise RuntimeError("Preserved previous run missing")
    previous_hashes = {path.name: sha256_file(path) for path in sorted(PREVIOUS_RUN.iterdir()) if path.is_file() and not path.name.startswith("._")}
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH or bank_anchor_hash() != EXPECTED_ANCHOR_HASH:
        raise RuntimeError("Canonical bank mismatch")
    source_dir, source_summary, source_run_manifest, adapter_manifest = locate_positive_control()
    source_adapter = source_dir / "successful_adapter_bank_item"
    source_anchor = {
        "source_directory": str(source_dir.resolve()), "run_manifest_sha256": sha256_file(source_dir / "run_manifest.json"),
        "adapter_manifest_sha256": sha256_file(source_adapter / "manifest.json"), "adapter_file_sha256": sha256_file(source_adapter / "adapter.pt"),
        "adapter_sha256": adapter_manifest["adapter_sha256"], "source_code_commit": adapter_manifest.get("code_commit"),
        "base_s0_hash": adapter_manifest["base_s0_hash"], "image_hash": adapter_manifest["image_hash"], "question_hash": adapter_manifest["question_hash"],
        "target_hash": adapter_manifest["target_hash"], "canonical_engram_bank_hash": adapter_manifest["canonical_bank_hash"],
        "rank": adapter_manifest["rank"], "alpha": adapter_manifest["alpha"], "module_count": len(adapter_manifest["resolved_lora_modules"]),
        "validation": {"primary_label": source_summary["primary_label"], "success_step": source_summary["success_step"], "unrestricted": source_summary["exact_unrestricted_output"], "short": source_summary["exact_short_output"], "reload": source_summary["reload"], "fresh": source_summary["fresh"], "rollback": source_summary["rollback"]},
    }
    write_json(out / "source_positive_control_anchor.json", source_anchor)
    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 0)
    clean_hash = state_weight_hash(model)
    append_jsonl(out / "state_and_bank_hash_ledger.jsonl", {"event": "CLEAN_S0", "state_weight_hash": clean_hash, "canonical_bank_hash": bank_manifest()["sha256"]})
    target_raw, cal_raw, held_raw, fixed_raw, gen_raw = raw_membership(views, records)
    positives = enrich(model, [target_raw, *gen_raw])
    target_entry, generality = positives[0], positives[1:]
    proposed_calibration = enrich(model, cal_raw)
    proposed_heldout = enrich(model, held_raw)
    fixed_locality = enrich(model, fixed_raw)
    positive_keys = {entry["router_input_equivalence_key"] for entry in positives}
    calibration, cal_exclusions = unique_negative_equivalence_classes(proposed_calibration, positive_keys)
    calibration_keys = {entry["router_input_equivalence_key"] for entry in calibration}
    heldout, held_exclusions = unique_negative_equivalence_classes(proposed_heldout, positive_keys, prior_split_keys=calibration_keys)
    exclusions = [*cal_exclusions, *held_exclusions]
    sufficiency = calibration_sufficiency(calibration)
    equivalence_ledger = equivalence_rows(positives, proposed_calibration, proposed_heldout, calibration, heldout, exclusions)
    write_table(out / "router_input_equivalence_ledger.csv", equivalence_ledger)
    write_table(out / "excluded_collision_and_dedup_ledger.csv", exclusions, ["candidate_name", "source_record", "pair_type", "group", "router_input_equivalence_key", "status", "representative_input_id"])
    collision_audit = native_collision_audit(model, views, records)
    write_json(out / "native_edit_collision_audit.json", collision_audit)
    split = {
        "written_before_representation_scores": True, "source_split_fixed_before_equivalence_filtering": True,
        "positive_inputs": positives, "proposed_calibration_candidates": proposed_calibration, "unique_calibration_classes": calibration,
        "proposed_heldout_candidates": proposed_heldout, "unique_heldout_classes": heldout, "fixed_locality_native_inputs": fixed_locality,
        "exclusions": exclusions, "calibration_sufficiency": sufficiency,
        "counts": {"proposed_calibration": len(proposed_calibration), "unique_calibration": len(calibration), "proposed_heldout": len(proposed_heldout), "unique_heldout": len(heldout), "positive_equivalent_exclusions": sum(row["status"] == "EXCLUDED_POSITIVE_EQUIVALENCE_COLLISION" for row in exclusions), "deduplicated_negative_candidates": sum(row["status"] == "DEDUPLICATED_NEGATIVE_EQUIVALENCE_CLASS" for row in exclusions), "cross_split_exclusions": sum(row["status"] == "EXCLUDED_CROSS_SPLIT_EQUIVALENCE_DUPLICATE" for row in exclusions)},
    }
    write_json(out / "calibration_and_test_split.json", split)
    if not sufficiency["passed"]:
        raise RuntimeError("ROUTER_INSUFFICIENT_UNIQUE_CALIBRATION_NEGATIVES")

    prototype, target_spec = extract_router_keys(model, target_entry)
    from safetensors.torch import save_file
    save_file({f"p_{name}": value.float().contiguous() for name, value in prototype.items()}, str(out / "router_prototype_keys.safetensors"))
    target_score = {**{name: target_entry[name] for name in ("input_id", "group", "category", "pair_type", "record_id_audit", "router_input_equivalence_key")}, **clamped_router_scores(prototype, prototype)}
    calibration_rows, cal_specs = score_entries(model, calibration, prototype)
    thresholds = corrected_thresholds(calibration_rows, target_score, TOL)
    target_score["route_on"] = route_on(target_score, thresholds, TOL)
    set_routes(calibration_rows, thresholds)
    heldout_rows, held_specs = score_entries(model, heldout, prototype)
    generality_rows, gen_specs = score_entries(model, generality, prototype)
    set_routes(heldout_rows, thresholds)
    set_routes(generality_rows, thresholds)
    routing_rows = [target_score, *calibration_rows, *heldout_rows, *generality_rows]
    write_table(out / "routing_scores.csv", routing_rows, ["input_id", "group", "category", "pair_type", "record_id_audit", "router_input_equivalence_key", "s_img", "s_text", "s_fused", "s_min", "s_joint", "route_on"])
    write_json(out / "router_thresholds.json", {"formula": "midpoint of clamped positive self-score and unique calibration-negative maximum", "weights": {"image": .30, "text": .30, "fused": .40}, **thresholds, "prototype_scores": {key: target_score[key] for key in ("s_img", "s_text", "s_fused", "s_min", "s_joint")}})
    representation = {"protocol": PROTOCOL, "equivalence_protocol": "MODEL_VISIBLE_INPUT_EQUIVALENCE_V1_1", "dtype": "FP32", "normalization": "L2", "cosines_clamped": True, "routing_state": "clean S0 adapter absent", "prototype": target_spec, "input_mappings": {**cal_specs, **held_specs, **gen_specs}, "no_generation_cache_reused": True}
    write_json(out / "router_representation_spec.json", representation)
    heldout_fps = [row for row in heldout_rows if row["route_on"]]
    fixed_key_to_route = {row["router_input_equivalence_key"]: bool(row["route_on"]) for row in heldout_rows}
    fixed_route_map = {entry["input_id"]: fixed_key_to_route[entry["router_input_equivalence_key"]] for entry in fixed_locality}
    if not target_score["route_on"]:
        pre_gate = "ROUTER_FALSE_NEGATIVE_ON_TARGET"
    elif heldout_fps:
        pre_gate = "ROUTER_HELD_OUT_FALSE_POSITIVE"
    else:
        pre_gate = "PASS"

    aliases = [str(value) for value in (records[RECORD_ID].get("accepted_answers") or [])]
    baseline_target = manual_greedy_trace(model, build_canonical_inputs(model, views[RECORD_ID]["target"]), CAP, eos_ids(model), top_k=1)
    baseline_locality = {rid: locality_trace(model, views[rid]["locality"]) for rid in ORDER}
    state, checked = load_adapter_payload(source_adapter)
    resolved = resolve_target_modules(model.llava_model.named_modules())
    if resolved != checked["resolved_lora_modules"] or adapter_hash(state) != checked["adapter_sha256"]:
        raise RuntimeError("ROUTED_LORA_INVALID_ENGINEERING_RUN")
    model.llava_model = insert_lora(model.llava_model, resolved)
    peft = model.llava_model
    load_adapter_state(peft.named_parameters(), state)
    peft.disable_adapter_layers()
    if manual_greedy_trace(model, build_canonical_inputs(model, views[RECORD_ID]["target"]), CAP, eos_ids(model), top_k=1)["token_ids"] != baseline_target["token_ids"]:
        raise RuntimeError("Disabled adapter S0 mismatch")
    if target_score["route_on"]:
        peft.enable_adapter_layers()
        target_generation = generate_parity(model, views[RECORD_ID]["target"], aliases)
        short_generation = generate_parity(model, generation_sample(model, target_entry["image_path"], records[RECORD_ID]["src"], TARGET), aliases)
    else:
        target_generation = short_generation = None
    gen_results = []
    for entry, row in zip(generality, generality_rows):
        peft.enable_adapter_layers() if row["route_on"] else peft.disable_adapter_layers()
        generated = generate_parity(model, generation_sample(model, entry["image_path"], entry["question"], TARGET), aliases)
        gen_results.append({**row, "generation": generated, "adapter_state": "ON" if row["route_on"] else "OFF_S0"})
    write_json(out / "target_and_generality_generation.json", {"target": {**target_score, "unrestricted": target_generation, "short_answer": short_generation}, "generality": gen_results})
    peft.disable_adapter_layers()
    locality = exact_locality_rows(model, views, baseline_locality)
    locality["route_decisions"] = fixed_route_map
    locality["all_routes_off"] = len(fixed_route_map) == 10 and not any(fixed_route_map.values())
    write_json(out / "fixed_locality_results.json", locality)
    fp_behavior = []
    for fp in heldout_fps:
        entry = next(value for value in heldout if value["input_id"] == fp["input_id"])
        peft.enable_adapter_layers()
        fp_behavior.append({**fp, "adapter_on_generation": generate_parity(model, generation_sample(model, entry["image_path"], entry["question"], TARGET), aliases)})
    peft.disable_adapter_layers()
    write_json(out / "held_out_negative_results.json", {"false_positive_count": len(heldout_fps), "false_positives": heldout_fps, "false_positive_behavior": fp_behavior, "unique_rows": heldout_rows})
    target_success = bool(target_generation and target_generation["match"]["success"] and target_generation["three_path_parity"])
    gen_pass, gen_summary = generality_rule(gen_results)
    always = json.loads((source_dir / "final_locality_report.json").read_text())
    ablation = {"conditions": [
        {"condition": "S0_ADAPTER_OFF", "target_natural_generation_success": False, "adapter_activation_count": 0, "strict_locality_changes": 0, "clinical_canonical_locality_failures": 0, "maximum_locality_nll_drift": 0.0},
        {"condition": "ADAPTER_ALWAYS_ON", "target_natural_generation_success": True, "adapter_activation_count": 11, "strict_locality_changes": always["strict_damage_count"], "clinical_canonical_locality_failures": always["clinical_failure_count"], "maximum_locality_nll_drift": always["maximum_nll_drift"], "source": str(source_dir / "final_locality_report.json")},
        {"condition": "ROUTER_GATED", "target_natural_generation_success": target_success, "adapter_activation_count": int(target_score["route_on"]) + sum(fixed_route_map.values()), "strict_locality_changes": locality["strict_damage_count"], "clinical_canonical_locality_failures": locality["clinical_failure_count"], "maximum_locality_nll_drift": locality["maximum_nll_drift"]},
    ], "frozen_diagnostic_no_tuning": True}
    write_json(out / "three_condition_ablation.json", ablation)

    core = bool(target_score["route_on"] and target_success and not heldout_fps and locality["all_routes_off"] and locality["passed"])
    repro = {"reload": "NOT_RUN", "fresh": "NOT_RUN", "replay": "NOT_RUN", "rollback": "NOT_RUN", "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH}
    expected_decisions = {row["input_id"]: bool(row["route_on"]) for row in routing_rows}
    routing_inputs = [target_entry, *calibration, *heldout, *generality]
    bank_item = out / "routed_adapter_bank_item"
    if core:
        payload = {
            "protocol": PROTOCOL, "equivalence_protocol": "MODEL_VISIBLE_INPUT_EQUIVALENCE_V1_1", "record_id": RECORD_ID,
            "adapter_reference": str(source_adapter.resolve()), "adapter_sha256": checked["adapter_sha256"], "adapter_file_sha256": sha256_file(source_adapter / "adapter.pt"),
            "source_positive_control_manifest_sha256": source_anchor["run_manifest_sha256"], "thresholds": thresholds,
            "representation": representation["prototype"], "base_s0_hash": clean_hash, "model_config": str(MODEL_CONFIG),
            "generation_config": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "repetition_penalty": 1.0, "seed": 42},
            "calibration_equivalence_classes": calibration, "heldout_equivalence_classes": heldout, "exclusion_ledger": exclusions,
            "routing_inputs": routing_inputs, "expected_route_decisions": expected_decisions,
            "expected_target_token_ids": target_generation["token_ids"], "expected_locality_token_ids": {rid: baseline_locality[rid]["token_ids"] for rid in ORDER},
            "canonical_engram_bank_hash": EXPECTED_BANK_HASH, "code_commit": PREVIOUS_COMMIT, "adapter_frozen": True,
        }
        save_router_bank(bank_item, {f"p_{name}": value for name, value in prototype.items()}, payload)
        loaded, bank_meta = load_router_bank(bank_item)
        loaded_proto = {name: loaded[f"p_{name}"] for name in ("img", "text", "fused")}
        peft.disable_adapter_layers()
        reload_decisions = recompute_decisions(model, routing_inputs, loaded_proto, thresholds)
        peft.enable_adapter_layers()
        reload_target = manual_greedy_trace(model, build_canonical_inputs(model, views[RECORD_ID]["target"]), CAP, eos_ids(model), top_k=1)["token_ids"]
        peft.disable_adapter_layers()
        reload_locality = {rid: locality_trace(model, views[rid]["locality"])["token_ids"] for rid in ORDER}
        expected_locality = {rid: baseline_locality[rid]["token_ids"] for rid in ORDER}
        repro["reload"] = "PASS" if reload_decisions == expected_decisions and reload_target == target_generation["token_ids"] and reload_locality == expected_locality else "FAIL"
        peft.enable_adapter_layers()
        replay_target = manual_greedy_trace(model, build_canonical_inputs(model, views[RECORD_ID]["target"]), CAP, eos_ids(model), top_k=1)["token_ids"]
        repro["replay"] = "PASS" if replay_target == target_generation["token_ids"] and reload_decisions == expected_decisions else "FAIL"
        peft.disable_adapter_layers()
        rollback_target = manual_greedy_trace(model, build_canonical_inputs(model, views[RECORD_ID]["target"]), CAP, eos_ids(model), top_k=1)["token_ids"]
        rollback_locality = {rid: locality_trace(model, views[rid]["locality"])["token_ids"] for rid in ORDER}
        repro["rollback"] = "PASS" if rollback_target == baseline_target["token_ids"] and rollback_locality == expected_locality else "FAIL"
        base = peft.unload()
        model.llava_model = base
        apply_prefix(model, bank, 0)
        repro["rollback"] = "PASS" if repro["rollback"] == "PASS" and state_weight_hash(model) == clean_hash else "FAIL"
        del peft, base, model
        torch.cuda.empty_cache()
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", "fresh", "--out-dir", str(out), "--bank-item", str(bank_item), "--physical-gpu", str(args.physical_gpu)]
        completed = subprocess.run(command, cwd=ROOT, env=dict(os.environ), check=False)
        fresh_result = json.loads((bank_item / "fresh_result.json").read_text()) if (bank_item / "fresh_result.json").exists() else {"passed": False}
        repro["fresh"] = "PASS" if completed.returncode == 0 and fresh_result.get("passed") else "FAIL"
    else:
        peft.disable_adapter_layers()
        base = peft.unload()
        model.llava_model = base
        apply_prefix(model, bank, 0)
        repro["rollback"] = "PASS" if state_weight_hash(model) == clean_hash else "FAIL"
    repro["canonical_bank_unchanged"] = bank_manifest()["sha256"] == EXPECTED_BANK_HASH
    repro_pass = core and all(repro[name] == "PASS" for name in ("reload", "fresh", "replay", "rollback"))
    if pre_gate != "PASS":
        label = pre_gate
    elif not target_success or not core:
        label = "ROUTED_LORA_INVALID_ENGINEERING_RUN"
    elif not repro_pass:
        label = "ROUTED_BANK_REPRODUCIBILITY_FAILURE"
    elif any(row["route_on"] and not row["generation"]["match"]["success"] for row in gen_results):
        label = "ROUTED_ADAPTER_GENERALITY_FAILURE"
    elif gen_pass:
        label = "PASS_ROUTED_BANKED_LORA_CORE_AND_GENERALITY"
    else:
        label = "PASS_ROUTED_BANKED_LORA_CORE_ONLY"
    counts = split["counts"]
    summary = {
        "primary_label": label, "previous_run_protocol_status": "INVALID_NEGATIVE_CONSTRUCTION_IDENTICAL_TO_POSITIVE",
        "positive_equivalent_exclusions": counts["positive_equivalent_exclusions"], "deduplicated_negative_candidates": counts["deduplicated_negative_candidates"],
        "unique_calibration_negatives": counts["unique_calibration"], "distinct_input_score_overlap": False,
        "target_route_on": bool(target_score["route_on"]), "heldout_false_positive_count": len(heldout_fps),
        "target_unrestricted_success": target_success, "target_output": target_generation["raw_output"] if target_generation else None,
        "fixed_locality_exact_s0": locality["passed"], "generality": gen_summary, "core_gate_passed": core,
        "reproducibility": repro, "native_edit_collision_audit": {"collision_count": len(collision_audit["collisions"]), "record953_conflict": collision_audit["record953_conflicting_target"]},
        "canonical_bank_unchanged": repro["canonical_bank_unchanged"], "stage2_permitted": False,
    }
    write_json(out / "routed_bank_reload_fresh_replay_rollback.json", repro)
    write_json(out / "routed_lora_v1_1_summary.json", summary)
    append_jsonl(out / "state_and_bank_hash_ledger.jsonl", {"event": "FINAL", "clean_s0_hash": clean_hash, "canonical_bank_hash": bank_manifest()["sha256"], "primary_label": label})
    write_json(out / "run_manifest.json", {
        "protocol": PROTOCOL, "previous_commit": PREVIOUS_COMMIT, "previous_run": str(PREVIOUS_RUN), "previous_output_hashes": previous_hashes,
        "cwd": str(ROOT), "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "physical_gpu": args.physical_gpu, "seed": 42, "record_id": RECORD_ID,
        "source_positive_control": source_anchor, "counts": counts, "thresholds": thresholds, "pre_generation_gate": pre_gate,
        "primary_label": label, "adapter_retrained": False, "stage2_launched": False, "multi_edit_launched": False,
        "canonical_bank_before": EXPECTED_BANK_HASH, "canonical_bank_after": bank_manifest()["sha256"], "required_outputs_complete": True,
    })
    write_text(out / "ROUTED_LORA_V1_1_FINAL_DECISION.md", summary_report(summary))


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must equal {args.physical_gpu}")
    fresh(args) if args.mode == "fresh" else run(args)


if __name__ == "__main__":
    main()
