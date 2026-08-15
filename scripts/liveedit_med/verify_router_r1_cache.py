#!/usr/bin/env python3
"""Online-vs-cache parity gate for one deterministic edit per R1 split."""
from __future__ import annotations

import argparse
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
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH, PROTOCOL
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest
from scripts.liveedit_med.cache_router_r1 import capture
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import load_clean_model, routed_generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--representation-manifest", type=Path, required=True)
    parser.add_argument("--strict-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES mismatch")
    source = json.loads(args.source_records.read_text())["records"]
    manifest = json.loads(args.representation_manifest.read_text())
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("ROUTER_R1_ANCHOR_MISMATCH:bank")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.strict_checkpoint)
    modules.load_state_dict(state, strict=True); modules.eval()
    rows = []
    for split in ("train", "validation", "heldout"):
        entry = manifest["splits"][split][0]
        record = source[split][0]
        if str(entry["record_id"]) != str(record["record_id"]):
            raise RuntimeError("ROUTER_R1_CACHE_PARITY_FAILURE:source_order")
        tensors = load_file(entry["file_path"], device="cpu")
        online = capture(model, block, native_sample(record))
        representation = all(torch.equal(online[name], tensors[f"native__{name}"])
                             for name in online)
        hidden = online["hidden"].float().unsqueeze(0).to(model.lm_device)
        vision = hidden[:, online["vision_mask"].to(model.lm_device)]
        question = hidden[:, online["question_mask"].to(model.lm_device)]
        answer = hidden[:, online["answer_mask"].to(model.lm_device)]
        eqr, evr, c, r = modules.generated_edit(vision, question, answer)
        expert = torch.equal(c.cpu(), tensors["expert__moe_c"]) and torch.equal(r.cpu(), tensors["expert__moe_r"])
        repo_online = {"ids": [str(record["record_id"])], "eqr": eqr, "evr": evr, "moe_c": c, "moe_r": r}
        repo_cached = {"ids": [str(record["record_id"])], "eqr": eqr, "evr": evr,
                       "moe_c": tensors["expert__moe_c"].to(model.lm_device),
                       "moe_r": tensors["expert__moe_r"].to(model.lm_device)}
        online_route = routed_generation(model, block, modules, native_sample(record), repo_online)
        cached_route = routed_generation(model, block, modules, native_sample(record), repo_cached)
        routed = (online_route["token_ids"] == cached_route["token_ids"]
                  and online_route["stop_reason"] == cached_route["stop_reason"]
                  and online_route["route"] == cached_route["route"])
        base_item = next(item for item in entry["inputs"] if item["category"] == "native")
        clean_generation_present = bool(base_item["clean_generation"]["token_ids"])
        row = {"split": split, "record_id": str(record["record_id"]),
               "representation_exact": representation, "expert_exact": expert,
               "clean_logits_exact": torch.equal(online["base_answer_logits"], tensors["native__base_answer_logits"]),
               "clean_generation_cached": clean_generation_present, "routed_output_exact": routed}
        row["passed"] = all(value for key, value in row.items() if key.endswith("_exact") or key == "clean_generation_cached")
        rows.append(row)
    passed = all(row["passed"] for row in rows)
    output = {"protocol": PROTOCOL, "status": "PASS" if passed else "ROUTER_R1_CACHE_PARITY_FAILURE",
              "checkpoint_step": checkpoint_manifest["step"], "sample_rule": "first_stable_hash_edit_each_split",
              "rows": rows, "canonical_bank_hash": bank_manifest()["sha256"]}
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("ROUTER_R1_CACHE_PARITY_FAILURE")
    print(json.dumps({"status": "ROUTER_R1_CACHE_PARITY_PASS", "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
