#!/usr/bin/env python3
"""Stage C: complete validation-only routing attribution, no adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path: sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import native_sample, normalize_answer, sample_to_model_row
from methods.liveedit_med.routing_attribution import attribute_route, failure_class, stable_repository
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import eos_ids
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_cached_greedy_trace
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import (MAX_NEW_TOKENS, compact_trace,
    forced_generation, load_clean_model, routed_generation)
from scripts.liveedit_med.run_official_style_medical_aggregate import build_experts, repository_for, views, write_new


PROTOCOL = "LIVEEDIT_MED_VALIDATION_ROUTING_ATTRIBUTION_V1"


@torch.inference_mode()
def clean_generation(model, sample):
    canonical = build_canonical_inputs(model, sample_to_model_row(sample))
    return compact_trace(manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1))


def route_attribution(route, norms, target_id):
    def flat(value):
        value = value or []
        return value[0] if value and isinstance(value[0], list) else value
    ids = route.get("candidate_ids", [])
    target_norm = norms["per_expert_residual_norms"][ids.index(target_id)] if target_id in ids else 0.0
    visual, sentinel = flat(route.get("visual_scores")), flat(route.get("sentinel_score"))
    return attribute_route(target_index=0, visual_scores=visual or [0.0],
        sentinel_score=(sentinel or [0.0])[0], candidate_mask=route.get("candidate_mask", [False]),
        text_scores=flat(route.get("raw_text_scores")), absolute_weights=flat(route.get("sigmoid_weights")),
        relative_weights=flat(route.get("softmax_weights")), final_weights=flat(route.get("final_weights"))), target_norm


def worker(args):
    source = json.loads(args.source_records.read_text()); records = source["records"]["validation"]
    assigned = [row for index, row in enumerate(records) if index % args.worker_count == args.worker_index]
    model, _bank = load_clean_model(args.physical_gpu); _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, manifest = load_safe_state(args.checkpoint)
    if int(manifest["step"]) != 3000: raise RuntimeError("LIVEEDIT_MED_STAGE_C_REQUIRES_STEP3000")
    modules.load_state_dict(state, strict=True); modules.eval(); experts = build_experts(model, block, modules, records)
    by_id = {str(row["record_id"]): row for row in records}; ids = list(by_id)
    nearest = {}
    for rid in ids:
        nearest[rid] = max((other for other in ids if other != rid),
                           key=lambda other: float(torch.dot(experts[rid]["clean_s0_representation"], experts[other]["clean_s0_representation"])))
    rows = []
    for record in assigned:
        rid = str(record["record_id"]); own = experts[rid]
        forced = {name: forced_generation(model, block, modules, sample, own["moe_c"], own["moe_r"])
                  for name, sample in views(record).items()}
        distractor = by_id[nearest[rid]]; native = native_sample(record); other = native_sample(distractor)
        negative = {
            "same_image_different_question": {"image": native["image"], "prompt": other["prompt"], "target": native["target"]},
            "same_question_different_image": {"image": other["image"], "prompt": native["prompt"], "target": native["target"]},
            "visual_near_miss": other,
            "ordinary_locality": record["locality"]["image_or_paired"][0],
        }
        clean_negative = {name: clean_generation(model, sample) for name, sample in negative.items()}
        for size in (1, 4, 8, 16, 32):
            members = stable_repository(records, rid, size); repo = repository_for(members, experts)
            for category, sample in {**views(record), **negative}.items():
                routed = routed_generation(model, block, modules, sample, repo)
                route = routed["route"]; norms = routed["residual_norms"]
                attribution, target_norm = route_attribution(route, norms, rid)
                positive = category in ("native", "textual", "visual", "paired")
                if positive:
                    forced_success = bool(forced[category]["match"]["success"]); routed_success = bool(routed["match"]["success"])
                    exact_s0 = None; contamination = None
                else:
                    clean = clean_negative[category]
                    exact_s0 = clean["token_ids"] == routed["token_ids"] and clean["stop_reason"] == routed["stop_reason"]
                    contamination = normalize_answer(native["target"]) in normalize_answer(routed["raw_output"])
                    forced_success = True; routed_success = bool(exact_s0)
                classification = failure_class(attribution, forced_success=forced_success, routed_success=routed_success,
                    target_residual_norm=target_norm, fused_residual_norm=float(norms["fused_residual_norm"]))
                row = {"record_id": rid, "repository_size": size, "repository_ids": repo["ids"], "category": category,
                       "positive": positive, **attribution, "expert_residual_norm": target_norm,
                       "fused_residual_norm": norms["fused_residual_norm"], "forced_generation_success": forced_success,
                       "generation_result": routed, "exact_s0_preservation": exact_s0, "target_contamination": contamination,
                       "failure_class": classification, "near_miss_record_id": nearest[rid] if category == "visual_near_miss" else None}
                rows.append(row)
            write_new(args.progress_dir / f"record_{rid}_repo_{size}.json",
                      {"record_id": rid, "repository_size": size, "rows": rows[-8:]})
    write_new(args.out, {"protocol": PROTOCOL, "worker_index": args.worker_index, "physical_gpu": args.physical_gpu,
                         "checkpoint_step": 3000, "rows": rows})


def finalize(args):
    rows = []
    for path in args.shard: rows.extend(json.loads(path.read_text())["rows"])
    expected = 64 * 5 * 8
    if len(rows) != expected or len({(r["record_id"],r["repository_size"],r["category"]) for r in rows}) != expected:
        raise RuntimeError(f"LIVEEDIT_MED_STAGE_C_INCOMPLETE:{len(rows)}!={expected}")
    scaling = {}
    for size in (1,4,8,16,32):
        subset=[r for r in rows if r["repository_size"]==size]
        scaling[str(size)]={"target_visual_recall":sum(r["target_in_candidates"] for r in subset if r["positive"]),
            "positive_count":sum(r["positive"] for r in subset), "candidate_count_mean":sum(r["candidate_count"] for r in subset)/len(subset),
            "positive_generation_success":sum(r["generation_result"]["match"]["success"] for r in subset if r["positive"]),
            "negative_exact_s0":sum(r["exact_s0_preservation"] is True for r in subset),
            "negative_count":sum(not r["positive"] for r in subset), "failure_classes":dict(Counter(r["failure_class"] for r in subset))}
    write_new(args.out_dir/"routing_attribution.json",{"protocol":PROTOCOL,"split":"validation","edit_count":64,
        "repository_sizes":[1,4,8,16,32],"rows":rows,"scaling":scaling,"adaptation_or_threshold_change":False})
    lines=["# Validation-only routing attribution","","No parameter, threshold, checkpoint, or repository tensor was modified.","",
           "| Repo | Visual recall | Positive generation | Negative exact S0 | Mean candidates |","|---:|---:|---:|---:|---:|"]
    for size in (1,4,8,16,32):
        s=scaling[str(size)]; lines.append(f"| {size} | {s['target_visual_recall']}/{s['positive_count']} | {s['positive_generation_success']}/{s['positive_count']} | {s['negative_exact_s0']}/{s['negative_count']} | {s['candidate_count_mean']:.2f} |")
    (args.out_dir/"ROUTING_ATTRIBUTION.md").write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="mode",required=True)
    w=sub.add_parser("worker"); w.add_argument("--source-records",type=Path,required=True); w.add_argument("--checkpoint",type=Path,required=True)
    w.add_argument("--physical-gpu",type=int,required=True); w.add_argument("--worker-index",type=int,required=True); w.add_argument("--worker-count",type=int,required=True)
    w.add_argument("--progress-dir",type=Path,required=True); w.add_argument("--out",type=Path,required=True)
    f=sub.add_parser("finalize"); f.add_argument("--shard",type=Path,action="append",required=True); f.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); (worker if a.mode=="worker" else finalize)(a)
if __name__=="__main__": main()
