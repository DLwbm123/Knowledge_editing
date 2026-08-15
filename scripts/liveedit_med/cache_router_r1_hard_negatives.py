#!/usr/bin/env python3
"""Cache the two compositional router-R1 hard-negative input families."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import native_sample, sample_to_model_row
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH, PROTOCOL, canonical_hash
from methods.liveedit_med.serialization import tensor_hashes
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, eos_ids
from scripts.engram.run_record953_routed_banked_lora_v1_1 import visible_input_audit
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_cached_greedy_trace
from scripts.liveedit_med.cache_router_r1 import capture, sha256_file, tensor_prefix
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import MAX_NEW_TOKENS, compact_trace, load_clean_model


@torch.inference_mode()
def worker(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES mismatch")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    source = json.loads(args.source_records.read_text())["records"]
    nearest = json.loads(args.nearest.read_text())["neighbors"]
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("ROUTER_R1_ANCHOR_MISMATCH:bank")
    _name, block = resolve_layer21_block(model)
    rows = []
    for split in ("train", "validation", "heldout"):
        records = source[split]
        by_id = {str(row["record_id"]): row for row in records}
        assigned = [(index, row) for index, row in enumerate(records)
                    if index % args.worker_count == args.worker_index]
        for ordinal, record in assigned:
            rid = str(record["record_id"])
            native = native_sample(record)
            chosen = nearest[rid]["chosen"]
            text_other = native_sample(by_id[chosen["same_image_different_question"]])
            visual_other = native_sample(by_id[chosen["same_question_different_image"]])
            visual_near = native_sample(by_id[chosen["visual_nearest"]])
            text_near = native_sample(by_id[chosen["text_nearest"]])
            joint_image = native_sample(by_id[chosen["joint_near_miss_image"]])
            joint_question = native_sample(by_id[chosen["joint_near_miss_question"]])
            samples = {
                "same_image_different_question": {
                    "image": native["image"], "prompt": text_other["prompt"], "target": native["target"]},
                "same_question_different_image": {
                    "image": visual_other["image"], "prompt": native["prompt"], "target": native["target"]},
                "visual_nearest": {
                    "image": native["image"], "prompt": visual_near["prompt"], "target": native["target"]},
                "text_nearest": {
                    "image": text_near["image"], "prompt": native["prompt"], "target": native["target"]},
                "joint_near_miss": {
                    "image": joint_image["image"], "prompt": joint_question["prompt"], "target": native["target"]},
            }
            tensors = {}
            metadata = []
            positive_eq = next(item["eqkey"] for item in next(
                row for row in args.regular_manifest_data["splits"][split] if str(row["record_id"]) == rid)["inputs"]
                if item["category"] == "native")
            all_positive_eq = {item["eqkey"] for split_rows in args.regular_manifest_data["splits"].values()
                               for source_row in split_rows for item in source_row["inputs"]
                               if item["category"] in ("native", "textual", "visual", "paired")}
            seen = set(all_positive_eq)
            for category, sample in samples.items():
                captured = capture(model, block, sample)
                tensor_prefix(tensors, category, captured)
                visible = visible_input_audit(model, {"question": sample["prompt"], "image_path": sample["image"],
                    "image_sha256": sha256_file(Path(sample["image"]))})
                eqkey = visible["router_input_equivalence_key"]
                if eqkey in seen:
                    raise RuntimeError(f"ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:EqKey:{split}:{rid}:{category}")
                seen.add(eqkey)
                canonical = build_canonical_inputs(model, sample_to_model_row(sample))
                trace = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
                metadata.append({"category": category, "eqkey": eqkey, "image": sample["image"],
                    "prompt": sample["prompt"], "target": sample["target"],
                    "source_hash": canonical_hash({"image": sha256_file(Path(sample["image"])),
                                                    "prompt": sample["prompt"]}),
                    "clean_generation": compact_trace(trace)})
            path = args.out / f"{split}_record_{rid}.safetensors"
            save_file(tensors, str(path))
            loaded = load_file(str(path), device="cpu")
            if loaded.keys() != tensors.keys() or any(not torch.equal(loaded[name], tensors[name]) for name in tensors):
                raise RuntimeError("ROUTER_R1_CACHE_PARITY_FAILURE:hard_reload")
            rows.append({"split": split, "ordinal": ordinal, "record_id": rid,
                "file": path.name, "file_path": str(path.resolve()), "file_sha256": sha256_file(path),
                "tensor_hashes": tensor_hashes(tensors), "inputs": metadata,
                "construction_sources": chosen})
            print(json.dumps({"event": "hard_cache", "worker": args.worker_index, "split": split,
                              "record_id": rid, "complete": len(rows)}), flush=True)
    manifest = {"protocol": PROTOCOL, "kind": "hard_negative_cache_shard",
                "worker_index": args.worker_index, "worker_count": args.worker_count,
                "records": rows, "count": len(rows), "clean_s0_only": True,
                "record953_used": False, "blind_loaded": False}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def finalize(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = []
    for path in args.shard:
        value = json.loads(path.read_text())
        if value.get("protocol") != PROTOCOL:
            raise RuntimeError("ROUTER_R1_CACHE_PARITY_FAILURE:hard_manifest")
        rows.extend(value["records"])
    expected = 512 + 64 + 64
    if len(rows) != expected or len({(row["split"], row["record_id"]) for row in rows}) != expected:
        raise RuntimeError(f"ROUTER_R1_CACHE_PARITY_FAILURE:hard_count:{len(rows)}")
    eqkeys = [(row["split"], row["record_id"], item["category"], item["eqkey"])
              for row in rows for item in row["inputs"]]
    classes = {}
    for split, record_id, category, eqkey in eqkeys:
        row = classes.setdefault(eqkey, {"splits": set(), "provenance": []})
        row["splits"].add(split); row["provenance"].append({"split": split, "record_id": record_id, "category": category})
    if any(len(row["splits"]) > 1 for row in classes.values()):
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:hard_negative_cross_split_eqkey")
    output = {"protocol": PROTOCOL, "kind": "hard_negative_cache", "count": len(rows),
              "records": sorted(rows, key=lambda row: (row["split"], int(row["ordinal"]))),
              "equivalence_classes": {key: {"split": next(iter(value["splits"])),
                  "provenance": value["provenance"]} for key, value in classes.items()},
              "clean_s0_only": True, "record953_used": False, "blind_loaded": False}
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--source-records", type=Path, required=True)
    p.add_argument("--nearest", type=Path, required=True)
    p.add_argument("--regular-manifest", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--worker-index", type=int, required=True)
    p.add_argument("--worker-count", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("finalize")
    p.add_argument("--shard", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "worker":
        args.regular_manifest_data = json.loads(args.regular_manifest.read_text())
        worker(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
