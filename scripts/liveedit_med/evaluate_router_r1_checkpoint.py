#!/usr/bin/env python3
"""Evaluate one frozen router-R1 checkpoint on validation or held-out edits."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.cached_suffix import answer_kl
from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import native_sample, normalize_answer
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH, PROTOCOL, deterministic_repository
from methods.liveedit_med.serialization import load_safe_state, tensor_hashes
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_natural_generation_recovery import clinical_preservation
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import forced_generation, load_clean_model, routed_generation
from scripts.liveedit_med.run_official_style_medical_aggregate import routed_tf
from scripts.liveedit_med.run_validation_routing_attribution import route_attribution


SIZES = (1, 10, 32)
POSITIVE = ("native", "textual", "visual", "paired")
FAILURE_MAP = {
    "VISUAL_SENTINEL_RECALL_FAILURE": "VISUAL_SENTINEL_RECALL_FAILURE",
    "TEXT_ABSOLUTE_SUPPRESSION": "TEXT_ABSOLUTE_SUPPRESSION_FAILURE",
    "TEXT_RELATIVE_COMPETITION": "TEXT_RELATIVE_COMPETITION_FAILURE",
    "RESIDUAL_INTERFERENCE": "RESIDUAL_INTERFERENCE_FAILURE",
    "GENERATOR_OR_EXPERT_FAILURE": "GENERATED_EXPERT_FAILURE",
    "ROUTED_GENERATION_FAILURE_UNRESOLVED": "RESIDUAL_INTERFERENCE_FAILURE",
}


def variant(tensors: Mapping[str, torch.Tensor], name: str, device: torch.device):
    prefix = name + "__"
    raw = {key[len(prefix):]: value.to(device) for key, value in tensors.items() if key.startswith(prefix)}
    hidden = raw["hidden"].float().unsqueeze(0)
    return {"hidden": hidden, "vision": hidden[:, raw["vision_mask"].bool()],
            "question": hidden[:, raw["question_mask"].bool()], "answer": hidden[:, raw["answer_mask"].bool()],
            "base_answer_logits": raw["base_answer_logits"].float(), "answer_mask": raw["answer_mask"].bool()}


def views(record: Mapping[str, Any]):
    return {"native": native_sample(record), **{name: record["generality"][name][0]
            for name in ("textual", "visual", "paired")}}


def compact_clean(item: Mapping[str, Any]):
    return dict(item["clean_generation"])


def find_input(entry: Mapping[str, Any], category: str):
    return next(item for item in entry["inputs"] if item["category"] == category)


def repo_ids(target: str, size: int, nearest: Mapping[str, Any], all_ids: list[str]):
    return (deterministic_repository(target, 16, nearest, all_ids)[:10] if size == 10
            else deterministic_repository(target, size, nearest, all_ids))


def build_experts(modules, regular, device):
    result = {}
    for rid, entry in regular.items():
        tensors = load_file(entry["file_path"], device="cpu")
        native = variant(tensors, "native", device)
        eqr = modules.edit_extractor.extract_query(native["question"])
        evr = modules.edit_extractor.extract_vision(native["question"], native["vision"])
        result[rid] = {"eqr": eqr, "evr": evr,
            "moe_c": tensors["expert__moe_c"].float().to(device),
            "moe_r": tensors["expert__moe_r"].float().to(device),
            "frozen_expert_hashes": {name: value for name, value in entry["tensor_hashes"].items()
                                      if name.startswith("expert__moe_")}}
    return result


def repository(ids, experts):
    result = {"ids": ids}
    for name in ("eqr", "evr", "moe_c", "moe_r"):
        result[name] = torch.cat([experts[rid][name] for rid in ids])
    return result


def negative_samples(record, source_by_id, nearest_row):
    native = native_sample(record)
    chosen = nearest_row["chosen"]
    other_text = native_sample(source_by_id[chosen["same_image_different_question"]])
    other_visual = native_sample(source_by_id[chosen["same_question_different_image"]])
    visual_near = native_sample(source_by_id[chosen["visual_nearest"]])
    text_near = native_sample(source_by_id[chosen["text_nearest"]])
    joint_image = native_sample(source_by_id[chosen["joint_near_miss_image"]])
    joint_question = native_sample(source_by_id[chosen["joint_near_miss_question"]])
    return {
        "same_image_different_question": {"image": native["image"], "prompt": other_text["prompt"], "target": native["target"]},
        "same_question_different_image": {"image": other_visual["image"], "prompt": native["prompt"], "target": native["target"]},
        "visual_nearest": {"image": native["image"], "prompt": visual_near["prompt"], "target": native["target"]},
        "text_nearest": {"image": text_near["image"], "prompt": native["prompt"], "target": native["target"]},
        "joint_near_miss": {"image": joint_image["image"], "prompt": joint_question["prompt"], "target": native["target"]},
        "image_locality": record["locality"]["image_or_paired"][0],
    }


def clean_for_negative(category, rid, regular, hard, nearest_row):
    if category in ("same_image_different_question", "same_question_different_image",
                    "visual_nearest", "text_nearest", "joint_near_miss"):
        return compact_clean(find_input(hard[(rid)], category))
    if category == "image_locality":
        return compact_clean(find_input(regular[rid], "image_locality"))
    return compact_clean(find_input(regular[nearest_row["chosen"][category]], "native"))


@torch.inference_mode()
def worker(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES mismatch")
    if args.out.exists():
        raise FileExistsError(args.out)
    source = json.loads(args.source_records.read_text())["records"][args.split]
    source_by_id = {str(row["record_id"]): row for row in source}
    ids = list(source_by_id)
    rep = json.loads(args.representation_manifest.read_text())
    regular = {str(row["record_id"]): row for row in rep["splits"][args.split]}
    hard_manifest = json.loads(args.hard_cache.read_text())
    hard = {str(row["record_id"]): row for row in hard_manifest["records"] if row["split"] == args.split}
    nearest = json.loads(args.nearest.read_text())["neighbors"]
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:bank")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    modules.load_state_dict(state, strict=True); modules.eval()
    experts = build_experts(modules, regular, model.lm_device)
    rows = []
    assigned = [record for index, record in enumerate(source) if index % args.worker_count == args.worker_index]
    for complete, record in enumerate(assigned, 1):
        rid = str(record["record_id"])
        positive_samples = views(record)
        forced = {name: forced_generation(model, block, modules, sample,
            experts[rid]["moe_c"], experts[rid]["moe_r"]) for name, sample in positive_samples.items()}
        negatives = negative_samples(record, source_by_id, nearest[rid])
        clean_negative = {name: clean_for_negative(name, rid, regular, hard, nearest[rid]) for name in negatives}
        for size in SIZES:
            members = repo_ids(rid, size, nearest, ids)
            repo = repository(members, experts)
            positive_rows = {}
            for name, sample in positive_samples.items():
                generation = routed_generation(model, block, modules, sample, repo)
                _logits, _labels, route, norms = routed_tf(model, block, modules, sample, repo)
                attribution, target_norm = route_attribution(route, norms, rid)
                success = bool(generation["match"]["success"])
                if success:
                    failure = "SUCCESS"
                elif not forced[name]["match"]["success"]:
                    failure = "GENERATED_EXPERT_FAILURE"
                elif not attribution["target_in_candidates"]:
                    failure = "VISUAL_SENTINEL_RECALL_FAILURE"
                elif attribution["sigmoid_absolute_weight"] < .5:
                    failure = "TEXT_ABSOLUTE_SUPPRESSION_FAILURE"
                elif attribution["softmax_relative_weight"] < .5:
                    failure = "TEXT_RELATIVE_COMPETITION_FAILURE"
                else:
                    failure = "RESIDUAL_INTERFERENCE_FAILURE"
                positive_rows[name] = {"forced": forced[name], "routed": generation, "route": route,
                    "attribution": attribution, "target_residual_norm": target_norm,
                    "residual_norms": norms, "failure_attribution": failure}
            negative_rows = {}
            for name, sample in negatives.items():
                generation = routed_generation(model, block, modules, sample, repo)
                clean = clean_negative[name]
                exact = clean["token_ids"] == generation["token_ids"] and clean["stop_reason"] == generation["stop_reason"]
                clinical = clinical_preservation(clean["raw_output"], generation["raw_output"],
                    sample["target"] if name == "image_locality" else clean["raw_output"])
                contamination = bool(normalize_answer(native_sample(record)["target"])
                    and normalize_answer(native_sample(record)["target"]) in normalize_answer(generation["raw_output"]))
                is_hard = name in ("same_image_different_question", "same_question_different_image",
                                   "visual_nearest", "text_nearest", "joint_near_miss")
                cached_tensors = load_file((hard[rid] if is_hard else regular[rid])["file_path"], device="cpu")
                cache_category = name if is_hard else "image_locality"
                cache_row = variant(cached_tensors, cache_category, model.lm_device)
                routed_logits, routed_labels, route, norms = routed_tf(model, block, modules, sample, repo)
                positions = torch.where(routed_labels[0].ne(-100))[0] - 1
                kl = float(answer_kl(routed_logits[0, positions], cache_row["base_answer_logits"].to(routed_logits.device)).item())
                candidate_count = len(route.get("candidate_ids", []))
                total_weight = float(route.get("sum_final_weights", 0.0))
                if exact:
                    failure = "SUCCESS"
                elif route.get("kind") == "base":
                    failure = "BASE_BYPASS_ENGINEERING_FAILURE"
                elif candidate_count == 0:
                    failure = "BASE_BYPASS_ENGINEERING_FAILURE"
                elif total_weight >= .5:
                    failure = "UNSAFE_RESIDUAL_STRENGTH"
                elif max([0.0, *[float(v) for v in (route.get("sigmoid_weights") or [[]])[0]]]) >= .5:
                    failure = "UNSAFE_TEXT_ABSOLUTE_WEIGHT"
                else:
                    failure = "UNSAFE_VISUAL_CANDIDATE"
                negative_rows[name] = {"s0": clean, "routed": generation, "route": route,
                    "residual_norms": norms, "exact_s0": exact, "clinical": clinical,
                    "target_contamination": contamination, "answer_kl": kl,
                    "failure_attribution": failure}
            text_item = find_input(regular[rid], "text_locality")
            negative_rows["text_locality"] = {"s0": text_item["clean_generation"],
                "routed": text_item["clean_generation"], "route": {"kind": "base",
                "reason": "TEXT_ONLY_EMPTY_CANDIDATE_BASE_BYPASS", "candidate_ids": [],
                "sum_final_weights": 0.0}, "exact_s0": True,
                "clinical": {"passed": True}, "target_contamination": False,
                "answer_kl": 0.0, "failure_attribution": "SUCCESS"}
            rows.append({"record_id": rid, "repository_size": size, "repository_ids": members,
                         "positive": positive_rows, "negative_locality": negative_rows})
        print(json.dumps({"event": "router_r1_eval", "split": args.split,
            "checkpoint_step": checkpoint_manifest["step"], "worker": args.worker_index,
            "complete": complete, "total": len(assigned), "record_id": rid}), flush=True)
    payload = {"protocol": PROTOCOL, "split": args.split, "checkpoint_step": checkpoint_manifest["step"],
        "checkpoint_tensor_hashes": checkpoint_manifest["tensor_hashes"], "worker_index": args.worker_index,
        "worker_count": args.worker_count, "rows": rows, "canonical_bank_hash": bank_manifest()["sha256"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def finalize(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = []; steps = set(); hashes = set()
    for path in args.shard:
        value = json.loads(path.read_text()); rows.extend(value["rows"])
        steps.add(int(value["checkpoint_step"])); hashes.add(json.dumps(value["checkpoint_tensor_hashes"], sort_keys=True))
    if len(steps) != 1 or len(hashes) != 1 or len(rows) != 64 * 3:
        raise RuntimeError(f"ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:eval_shards:{len(rows)}")
    step = steps.pop(); sizes = {}
    for size in SIZES:
        subset = [row for row in rows if row["repository_size"] == size]
        routed = {name: sum(row["positive"][name]["routed"]["match"]["success"] for row in subset) for name in POSITIVE}
        forced = {name: sum(row["positive"][name]["forced"]["match"]["success"] for row in subset) for name in POSITIVE}
        negatives = [value for row in subset for value in row["negative_locality"].values()]
        positive_values = [value for row in subset for value in row["positive"].values()]
        sizes[str(size)] = {"step": step, "edit_count": len(subset),
            "forced": forced, "routed": routed,
            "negative_locality_exact_s0": sum(value["exact_s0"] for value in negatives),
            "negative_locality_count": len(negatives),
            "target_contamination": sum(value["target_contamination"] for value in negatives),
            "clinical_canonical_failures": sum(not value["clinical"]["passed"] for value in negatives),
            "mean_candidate_count": sum(len(value["route"].get("candidate_ids", [])) for value in positive_values + negatives) / max(1, len(positive_values) + len(negatives)),
            "text_relative_competition_failures": sum(value["failure_attribution"] == "TEXT_RELATIVE_COMPETITION_FAILURE" for value in positive_values),
            "negative_locality_kl": sum(value["answer_kl"] for value in negatives) / max(1, len(negatives)),
            "failure_attribution": dict(Counter(value["failure_attribution"] for value in positive_values + negatives))}
    output = {"protocol": PROTOCOL, "split": args.split, "checkpoint_step": step,
              "repository_sizes": sizes, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ROUTER_R1_CHECKPOINT_EVALUATION_COMPLETE", "split": args.split,
                      "step": step, "repo32": sizes["32"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--source-records", type=Path, required=True); p.add_argument("--split", choices=("validation", "heldout"), required=True)
    p.add_argument("--representation-manifest", type=Path, required=True); p.add_argument("--hard-cache", type=Path, required=True)
    p.add_argument("--nearest", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, required=True); p.add_argument("--worker-index", type=int, required=True)
    p.add_argument("--worker-count", type=int, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("finalize"); p.add_argument("--shard", type=Path, action="append", required=True); p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); (worker if args.mode == "worker" else finalize)(args)


if __name__ == "__main__":
    main()
