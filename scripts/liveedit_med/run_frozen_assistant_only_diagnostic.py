#!/usr/bin/env python3
"""Stage D: source-full versus assistant-only residual scope, routes frozen."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path: sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook, resolve_layer21_block
from methods.liveedit_med.posthoc_validation import BaseRoutePlan, native_sample, plan_audit, route_residual, sample_to_model_row, unrestricted_match
from methods.liveedit_med.routing_attribution import stable_repository
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.source_ops import apply_low_rank_expert_residual, route_repository
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import eos_ids
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_cached_greedy_trace
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import MAX_NEW_TOKENS, capture_prompt, compact_trace, load_clean_model
from scripts.liveedit_med.run_official_style_medical_aggregate import build_experts, repository_for, views, write_new
from scripts.liveedit_med.run_posthoc_stage_q import load_repository


PROTOCOL = "LIVEEDIT_MED_FROZEN_ASSISTANT_ONLY_DIAGNOSTIC_V1"


@torch.inference_mode()
def compare_modes(model, block, modules, sample, repository):
    canonical = build_canonical_inputs(model, sample_to_model_row(sample))
    prompt_hidden, vision, question = capture_prompt(model, block, canonical)
    plan = route_repository(modules.input_extractor, question.float(), vision.float(), repository["evr"], repository["eqr"])
    route = plan_audit(plan, repository["ids"])
    _residual, norms = route_residual(plan, prompt_hidden, repository["moe_c"], repository["moe_r"], modules.instant_reps_norm)
    traces = {}
    for mode, assistant_only in (("source_full", False), ("assistant_only", True)):
        hook = None
        if not isinstance(plan, BaseRoutePlan):
            c, r = repository["moe_c"][plan.candidate_mask], repository["moe_r"][plan.candidate_mask]
            hook = Layer21ResidualHook(block, lambda hidden: apply_low_rank_expert_residual(
                hidden.float(), c, r, plan.final_weights, modules.instant_reps_norm).to(hidden.dtype),
                assistant_only=assistant_only).install()
            hook.enabled = True
            if assistant_only: hook.set_prompt_boundary(10**9)
        trace = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
        if hook is not None: hook.remove()
        traces[mode] = compact_trace(trace)
    return {"route": route, "route_reused_identically": True, "residual_norms": norms, **traces,
            "outputs_identical": traces["source_full"]["token_ids"] == traces["assistant_only"]["token_ids"] and
                                 traces["source_full"]["stop_reason"] == traces["assistant_only"]["stop_reason"]}


def worker(args):
    model, _bank = load_clean_model(args.physical_gpu); _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, manifest = load_safe_state(args.checkpoint)
    checkpoint_step = int(manifest["step"])
    modules.load_state_dict(state, strict=True); modules.eval(); rows=[]
    if args.dataset == "validation":
        records=json.loads(args.source_records.read_text())["records"]["validation"]
        experts=build_experts(model,block,modules,records)
        assigned=[row for index,row in enumerate(records) if index%args.worker_count==args.worker_index]
        for record in assigned:
            rid=str(record["record_id"]); repo=repository_for(stable_repository(records,rid,32),experts)
            for category,sample in views(record).items():
                result=compare_modes(model,block,modules,sample,repo)
                for mode in ("source_full","assistant_only"):
                    result[mode]["match"]=unrestricted_match(result[mode]["raw_output"],sample["target"],
                        eos=result[mode]["stop_reason"]=="eos",cap_hit=result[mode]["cap_hit"])
                rows.append({"dataset":"validation","record_id":rid,"category":category,**result})
    else:
        repo,_repo_manifest=load_repository(args.stage_q_dir,model.lm_device)
        inputs=json.loads((args.stage_q_dir/"input_manifest.json").read_text())
        dev=[("safety",row) for row in inputs["safety"]]+[("locality",row) for row in inputs["locality"]]
        assigned=[row for index,row in enumerate(dev) if index%args.worker_count==args.worker_index]
        for category,row in assigned:
            sample={"image":row["image"],"prompt":row["prompt"],"target":row.get("canonical_answer") or "unknown"}
            rows.append({"dataset":"record953_development_regression","input_id":row["input_id"],"category":category,
                         "target_is_placeholder":row.get("canonical_answer") is None,**compare_modes(model,block,modules,sample,repo)})
    write_new(args.out,{"protocol":PROTOCOL,"worker_index":args.worker_index,"dataset":args.dataset,
                        "physical_gpu":args.physical_gpu,"checkpoint_step":checkpoint_step,"rows":rows})


def finalize(args):
    rows=[]; checkpoint_steps=set()
    for path in args.shard:
        shard=json.loads(path.read_text()); rows.extend(shard["rows"]); checkpoint_steps.add(int(shard["checkpoint_step"]))
    if len(checkpoint_steps)!=1: raise RuntimeError("LIVEEDIT_MED_STAGE_D_CHECKPOINT_DRIFT")
    checkpoint_step=checkpoint_steps.pop()
    validation=[r for r in rows if r["dataset"]=="validation"]; dev=[r for r in rows if r["dataset"]!="validation"]
    if len(validation)!=256 or len(dev)!=50: raise RuntimeError(f"LIVEEDIT_MED_STAGE_D_INCOMPLETE:{len(validation)}:{len(dev)}")
    summary={"protocol":PROTOCOL,"checkpoint_step":checkpoint_step,"route_parameters_and_decisions_reused":True,
      "validation":{"count":len(validation),"source_full_success":sum(r["source_full"]["match"]["success"] for r in validation),
        "assistant_only_success":sum(r["assistant_only"]["match"]["success"] for r in validation),
        "outputs_identical":sum(r["outputs_identical"] for r in validation)},
      "record953_development_regression":{"count":len(dev),"outputs_identical":sum(r["outputs_identical"] for r in dev),
        "scope":"DEVELOPMENT_REGRESSION_ONLY_NOT_BLIND"},"rows":rows}
    write_new(args.out_dir/"assistant_only_diagnostic.json",summary)
    (args.out_dir/"ASSISTANT_ONLY_DIAGNOSTIC.md").write_text(
      "# Frozen assistant-only diagnostic\n\nRouting plans were computed once and reused for both residual scopes. No parameter changed.\n\n"
      f"- Validation source-full success: {summary['validation']['source_full_success']}/256\n"
      f"- Validation assistant-only success: {summary['validation']['assistant_only_success']}/256\n"
      f"- Validation identical outputs: {summary['validation']['outputs_identical']}/256\n"
      f"- Old record-953 development/regression identical outputs: {summary['record953_development_regression']['outputs_identical']}/50\n")


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="mode",required=True)
    w=sub.add_parser("worker"); w.add_argument("--dataset",choices=("validation","dev"),required=True)
    w.add_argument("--source-records",type=Path); w.add_argument("--stage-q-dir",type=Path); w.add_argument("--checkpoint",type=Path,required=True)
    w.add_argument("--physical-gpu",type=int,required=True); w.add_argument("--worker-index",type=int,required=True); w.add_argument("--worker-count",type=int,required=True); w.add_argument("--out",type=Path,required=True)
    f=sub.add_parser("finalize"); f.add_argument("--shard",type=Path,action="append",required=True); f.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); (worker if a.mode=="worker" else finalize)(a)
if __name__=="__main__": main()
