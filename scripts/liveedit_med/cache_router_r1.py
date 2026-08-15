#!/usr/bin/env python3
"""Deterministic representation/expert/base-output cache for router R1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import native_sample, sample_to_model_row
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH, PROTOCOL, canonical_hash
from methods.liveedit_med.serialization import load_safe_state, tensor_hashes
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.equivalence_aware_router_utils import router_input_equivalence_key
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, eos_ids
from scripts.engram.run_record953_routed_banked_lora_v1_1 import visible_input_audit
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_cached_greedy_trace
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import (
    MAX_NEW_TOKENS,
    compact_trace,
    load_clean_model,
    text_only_canonical,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variants(record: Mapping[str, Any]):
    yield "native", native_sample(record), "positive"
    for name in ("textual", "visual", "paired"):
        rows = record["generality"].get(name) or []
        if rows:
            yield name, rows[0], "positive"
    rows = record["locality"].get("image_or_paired") or []
    if rows:
        yield "image_locality", rows[0], "locality"
    rows = record["locality"].get("text_only") or []
    if rows:
        yield "text_locality", rows[0], "text_only"


@torch.inference_mode()
def capture(model: Any, block: torch.nn.Module, sample: Mapping[str, Any]):
    row = sample_to_model_row(sample)
    inputs, labels, masks = model._build_batch(row)
    captured = []
    handle = block.register_forward_hook(lambda _m, _a, output: captured.append(
        (output[0] if isinstance(output, (tuple, list)) else output).detach()))
    output = model.llava_model(inputs_embeds=inputs, attention_mask=masks["attention_mask"].long(),
                               labels=labels, return_dict=True, use_cache=False)
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:capture_count")
    length = int(masks["attention_mask"][0].sum())
    hidden = captured[0][0, :length]
    labels = labels[0, :length]
    attention = masks["attention_mask"][0, :length].long()
    answer = masks["answer_mask"][0, :length].bool()
    predictor = torch.where(answer)[0] - 1
    if bool((predictor < 0).any()):
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:predictor")
    result = {
        "hidden": hidden.half().cpu().contiguous(),
        "labels": labels.cpu().contiguous(),
        "attention": attention.cpu().contiguous(),
        "position_ids": (attention.cumsum(0) - 1).clamp_min(0).cpu().contiguous(),
        "vision_mask": masks["vision_mask"][0, :length].bool().cpu().contiguous(),
        "question_mask": masks["prompt_mask"][0, :length].bool().cpu().contiguous(),
        "answer_mask": answer.cpu().contiguous(),
        "base_answer_logits": output.logits[0, predictor].half().cpu().contiguous(),
    }
    if not all(torch.isfinite(value).all() for value in result.values() if value.is_floating_point()):
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:nonfinite_cache")
    return result


def tensor_prefix(destination: dict[str, torch.Tensor], prefix: str, values: Mapping[str, torch.Tensor]) -> None:
    for name, value in values.items():
        destination[f"{prefix}__{name}"] = value


@torch.inference_mode()
def worker(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES mismatch")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    source = json.loads(args.source_records.read_text())
    records = source["records"][args.split]
    assigned = [(index, record) for index, record in enumerate(records)
                if index % args.worker_count == args.worker_index]
    if args.limit is not None:
        assigned = assigned[:args.limit]
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("ROUTER_R1_ANCHOR_MISMATCH:bank")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    if checkpoint_manifest.get("step") != 3200:
        raise RuntimeError("ROUTER_R1_ANCHOR_MISMATCH:checkpoint")
    modules.load_state_dict(state, strict=True)
    modules.eval()
    rows = []
    for ordinal, record in assigned:
        rid = str(record["record_id"])
        tensors: dict[str, torch.Tensor] = {}
        metadata = []
        native_capture = None
        for category, sample, role in variants(record):
            if role == "text_only":
                canonical = text_only_canonical(model, sample)
                trace = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
                metadata.append({"category": category, "role": role, "image": None,
                                 "prompt": sample["prompt"], "target": sample["target"],
                                 "eqkey": "NO_IMAGE:" + canonical.prompt_hash,
                                 "clean_generation": compact_trace(trace)})
                continue
            captured = capture(model, block, sample)
            tensor_prefix(tensors, category, captured)
            if category == "native":
                native_capture = captured
            visible = visible_input_audit(model, {
                "question": sample["prompt"], "image_path": sample["image"],
                "image_sha256": sha256_file(Path(sample["image"])),
            })
            canonical = build_canonical_inputs(model, sample_to_model_row(sample))
            trace = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
            metadata.append({
                "category": category, "role": role, "image": sample["image"],
                "prompt": sample["prompt"], "target": sample["target"],
                "eqkey": visible["router_input_equivalence_key"],
                "processed_pixel_tensor_sha256": visible["processed_pixel_tensor_sha256"],
                "routing_input_ids_sha256": canonical_hash(visible["routing_input_ids"]),
                "source_hash": canonical_hash({"image_sha256": visible["raw_image_sha256"],
                                                 "prompt": sample["prompt"], "target": sample["target"]}),
                "clean_generation": compact_trace(trace),
            })
        if native_capture is None:
            raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:missing_native")
        hidden = native_capture["hidden"].float().unsqueeze(0).to(model.lm_device)
        vision = hidden[:, native_capture["vision_mask"].to(model.lm_device)]
        question = hidden[:, native_capture["question_mask"].to(model.lm_device)]
        answer = hidden[:, native_capture["answer_mask"].to(model.lm_device)]
        eqr, evr, moe_c, moe_r = modules.generated_edit(vision, question, answer)
        tensor_prefix(tensors, "expert", {"moe_c": moe_c.float().cpu(), "moe_r": moe_r.float().cpu(),
                                          "initial_eqr": eqr.float().cpu(), "initial_evr": evr.float().cpu()})
        path = args.out / f"record_{rid}.safetensors"
        save_file(tensors, str(path))
        loaded = load_file(str(path), device="cpu")
        if loaded.keys() != tensors.keys() or any(not torch.equal(loaded[name], tensors[name]) for name in tensors):
            raise RuntimeError("ROUTER_R1_CACHE_PARITY_FAILURE:reload")
        rows.append({"ordinal": ordinal, "record_id": rid, "selection_hash": record["selection_hash"],
                     "file": path.name, "file_path": str(path.resolve()), "file_sha256": sha256_file(path),
                     "tensor_hashes": tensor_hashes(tensors), "inputs": metadata})
        print(json.dumps({"event": "cache", "split": args.split, "worker": args.worker_index,
                          "complete": len(rows), "total": len(assigned), "record_id": rid}), flush=True)
    manifest = {"protocol": PROTOCOL, "kind": "router_r1_cache_shard", "split": args.split,
                "worker_index": args.worker_index, "worker_count": args.worker_count,
                "checkpoint_step": 3200, "records": rows, "count": len(rows),
                "canonical_bank_hash": EXPECTED_BANK_HASH, "generated_experts_frozen": True,
                "clean_base_state": "S0", "record953_used": False, "blind_set_loaded": False}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def finalize(args: argparse.Namespace) -> None:
    if args.cache_dir.exists() and any(args.cache_dir.iterdir()):
        raise FileExistsError(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    splits = {}
    for shard_path in args.shard:
        manifest = json.loads(shard_path.read_text())
        if manifest.get("protocol") != PROTOCOL or manifest.get("checkpoint_step") != 3200:
            raise RuntimeError("ROUTER_R1_CACHE_PARITY_FAILURE:manifest")
        split = manifest["split"]
        splits.setdefault(split, []).extend(manifest["records"])
    expected = {"train": 512, "validation": 64, "heldout": 64}
    for split, count in expected.items():
        rows = sorted(splits.get(split, []), key=lambda row: int(row["ordinal"]))
        if len(rows) != count or len({row["record_id"] for row in rows}) != count:
            raise RuntimeError(f"ROUTER_R1_CACHE_PARITY_FAILURE:{split}:{len(rows)}")
        splits[split] = rows
        records.extend({**row, "split": split} for row in rows)
    representation = {"protocol": PROTOCOL, "kind": "representation_cache", "counts": expected,
                      "splits": splits, "clean_base_state": "S0", "layer": 21,
                      "dtype": "float16_storage_float32_router", "record953_used": False,
                      "blind_set_loaded": False}
    expert = {"protocol": PROTOCOL, "kind": "frozen_expert_cache", "checkpoint_step": 3200,
              "records": [{"split": row["split"], "record_id": row["record_id"],
                           "file": row["file"], "expert_tensor_hashes": {
                               name: value for name, value in row["tensor_hashes"].items()
                               if name.startswith("expert__moe_")}}
                          for row in records], "immutable": True}
    base = {"protocol": PROTOCOL, "kind": "base_output_cache", "counts": expected,
            "records": [{"split": row["split"], "record_id": row["record_id"],
                         "input_count": len(row["inputs"]),
                         "generation_hashes": {item["category"]: canonical_hash(item["clean_generation"])
                                               for item in row["inputs"]}}
                        for row in records], "clean_base_state": "S0"}
    for name, value in (("representation_cache_manifest.json", representation),
                        ("expert_cache_manifest.json", expert),
                        ("base_output_cache_manifest.json", base)):
        (args.cache_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ROUTER_R1_CACHE_FINALIZED", "counts": expected}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--source-records", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("train", "validation", "heldout"), required=True)
    p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--worker-index", type=int, required=True)
    p.add_argument("--worker-count", type=int, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("finalize")
    p.add_argument("--shard", type=Path, action="append", required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    (worker if args.mode == "worker" else finalize)(args)


if __name__ == "__main__":
    main()
