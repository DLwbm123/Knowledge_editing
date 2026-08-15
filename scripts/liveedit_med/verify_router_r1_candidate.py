#!/usr/bin/env python3
"""Two-process parity, replay, reload, and rollback audit for selected R1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import native_sample
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH, PROTOCOL, deterministic_repository
from methods.liveedit_med.serialization import load_safe_state, tensor_hashes
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, eos_ids
from scripts.engram.run_engram_v2_stage0abc_diagnostics import hf_cached_greedy_trace
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_cached_greedy_trace, manual_greedy_trace
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import load_clean_model
from scripts.liveedit_med.evaluate_router_r1_checkpoint import build_experts, repo_ids, repository
from scripts.liveedit_med.run_posthoc_stage_q import remove_hook, route_context


def compact(trace):
    return {key: trace.get(key) for key in ("raw_output", "token_ids", "stop_reason", "eos_step", "cap_hit")}


@torch.inference_mode()
def worker(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES mismatch")
    if args.out.exists(): raise FileExistsError(args.out)
    source = json.loads(args.source_records.read_text())["records"]["heldout"]
    rep = json.loads(args.representation_manifest.read_text())
    regular = {str(row["record_id"]): row for row in rep["splits"]["heldout"]}
    nearest = json.loads(args.nearest.read_text())["neighbors"]
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH: raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:bank")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, manifest = load_safe_state(args.checkpoint); modules.load_state_dict(state, strict=True); modules.eval()
    experts = build_experts(modules, regular, model.lm_device)
    record = source[0]; rid = str(record["record_id"]); ids = [str(row["record_id"]) for row in source]
    repo = repository(repo_ids(rid, 32, nearest, ids), experts); sample = native_sample(record)
    clean_canonical = build_canonical_inputs(model, {"image_path": [sample["image"]], "prompt": [sample["prompt"]], "target": [sample["target"]]})
    clean_before = manual_cached_greedy_trace(model, clean_canonical, 128, eos_ids(model), top_k=5)
    traces = {}
    for mode in ("no_cache", "cached", "hf", "replay"):
        canonical, route, hook = route_context(model, block, modules, sample, repo)
        try:
            if mode == "no_cache": trace = manual_greedy_trace(model, canonical, 128, eos_ids(model), top_k=5)
            elif mode == "hf": trace = hf_cached_greedy_trace(model, canonical, 128)
            else: trace = manual_cached_greedy_trace(model, canonical, 128, eos_ids(model), top_k=5)
        finally: remove_hook(hook)
        traces[mode] = compact(trace)
        if mode == "cached": cached_route = route
    clean_after = manual_cached_greedy_trace(model, clean_canonical, 128, eos_ids(model), top_k=5)
    cached_tensor = load_file(regular[rid]["file_path"], device="cpu")
    expert_hash_equal = (tensor_hashes({"c": experts[rid]["moe_c"].cpu(), "r": experts[rid]["moe_r"].cpu()}) ==
        tensor_hashes({"c": cached_tensor["expert__moe_c"], "r": cached_tensor["expert__moe_r"]}))
    output = {"protocol": PROTOCOL, "process_index": args.process_index, "selected_step": manifest["step"],
        "record_id": rid, "route": cached_route, "traces": traces,
        "manual_no_cache_cached_hf_parity": traces["no_cache"]["token_ids"] == traces["cached"]["token_ids"] == traces["hf"]["token_ids"],
        "replay": traces["cached"] == traces["replay"],
        "rollback": compact(clean_before) == compact(clean_after),
        "generated_expert_hash_parity": expert_hash_equal,
        "checkpoint_hash": hashlib.sha256((args.checkpoint / "manifest.json").read_bytes()).hexdigest(),
        "canonical_bank_hash": bank_manifest()["sha256"]}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def finalize(args):
    if args.out.exists(): raise FileExistsError(args.out)
    rows = [json.loads(path.read_text()) for path in args.process]
    if len(rows) != 2: raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:repro_process_count")
    fresh = (rows[0]["route"] == rows[1]["route"] and rows[0]["traces"] == rows[1]["traces"]
             and rows[0]["checkpoint_hash"] == rows[1]["checkpoint_hash"])
    result = {"protocol": PROTOCOL, "manual_no_cache_cached_hf_parity": all(row["manual_no_cache_cached_hf_parity"] for row in rows),
        "reload": fresh, "fresh_process": fresh, "route_decision_parity": rows[0]["route"] == rows[1]["route"],
        "output_token_parity": rows[0]["traces"] == rows[1]["traces"],
        "generated_expert_hash_parity": all(row["generated_expert_hash_parity"] for row in rows),
        "replay": all(row["replay"] for row in rows), "rollback": all(row["rollback"] for row in rows),
        "canonical_bank_unchanged": all(row["canonical_bank_hash"] == EXPECTED_BANK_HASH for row in rows),
        "process_results": rows}
    result["passed"] = all(result[name] for name in ("manual_no_cache_cached_hf_parity", "reload", "fresh_process",
        "route_decision_parity", "output_token_parity", "generated_expert_hash_parity", "replay", "rollback", "canonical_bank_unchanged"))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("worker"); p.add_argument("--source-records",type=Path,required=True); p.add_argument("--representation-manifest",type=Path,required=True)
    p.add_argument("--nearest",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--physical-gpu",type=int,required=True)
    p.add_argument("--process-index",type=int,required=True); p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("finalize"); p.add_argument("--process",type=Path,action="append",required=True); p.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); (worker if args.mode=="worker" else finalize)(args)


if __name__=="__main__": main()
