#!/usr/bin/env python3
"""Cache exact frozen LLaVA-Med layer-21 outputs for Stage-S source training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from scripts.engram.run_engram_v2_stage0_generation_audit import apply_prefix, load_model_views_bank


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variants(record: dict[str, Any]):
    request = record["requests"][0]
    yield "native_0", {"image": request["image"], "prompt": request["prompt"], "target": request["target_new"]}, False
    for group_name, rows in record["generality"].items():
        for index, row in enumerate(rows):
            yield f"gen_{group_name}_{index}", row, False
    for group_name, rows in record["locality"].items():
        for index, row in enumerate(rows):
            if row.get("image"):
                yield f"loc_{group_name}_{index}", row, True


@torch.inference_mode()
def capture_many(model: Any, block: torch.nn.Module, rows: list[tuple[str, dict[str, Any], bool]]):
    sample = {"image_path": [row[1]["image"] for row in rows], "prompt": [row[1]["prompt"] for row in rows], "target": [row[1]["target"] for row in rows]}
    inputs, labels, masks = model._build_batch(sample)
    captured: list[torch.Tensor] = []
    handle = block.register_forward_hook(lambda _m, _a, output: captured.append((output[0] if isinstance(output, (tuple, list)) else output).detach()))
    output = model.llava_model(
        inputs_embeds=inputs, attention_mask=masks["attention_mask"].long(),
        labels=labels, return_dict=True, use_cache=False,
    )
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError("LIVEEDIT_MED_LAYER21_CAPTURE_COUNT")
    result = {}
    for index, (key, _row, locality) in enumerate(rows):
        length = int(masks["attention_mask"][index].sum())
        tensors = {
            "hidden": captured[0][index, :length].cpu().contiguous(),
            "labels": labels[index, :length].cpu().contiguous(),
            "attention": masks["attention_mask"][index, :length].cpu().contiguous(),
            "vision": masks["vision_mask"][index, :length].cpu().contiguous(),
            "prompt": masks["prompt_mask"][index, :length].cpu().contiguous(),
            "answer": masks["answer_mask"][index, :length].cpu().contiguous(),
        }
        if locality:
            positions = torch.where(tensors["answer"])[0] - 1
            if bool((positions < 0).any()): raise RuntimeError("LIVEEDIT_MED_INVALID_LOCALITY_PREDICTOR_POSITION")
            tensors["base_answer_logits"] = output.logits[index, positions].half().cpu().contiguous()
        if not all(bool(torch.isfinite(value).all()) for value in tensors.values() if value.is_floating_point()): raise RuntimeError("LIVEEDIT_MED_NONFINITE_REPRESENTATION")
        result[key] = tensors
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES mismatch")
    args.cache_dir.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source_records.read_text())
    records = source["records"]["train"][args.offset:]
    if args.limit is not None:
        records = records[: args.limit]
    model, _views, bank, _raw = load_model_views_bank(args.physical_gpu)
    # The expensive visual encoder is no longer needed after multimodal preparation.
    # Moving it to CPU leaves more GPU memory and bandwidth for layer-21 capture.
    vision_tower = model.llava_model.get_vision_tower()
    apply_prefix(model, bank, 0)
    _name, block = resolve_layer21_block(model)
    rows = []
    for record_index, record in enumerate(records):
        tensors: dict[str, torch.Tensor] = {}; variant_rows = []; items=list(variants(record))
        for key, item, locality in items:
            if not Path(item["image"]).is_file():
                raise RuntimeError(f"LIVEEDIT_MED_MISSING_IMAGE:{item['image']}")
            variant_rows.append({"key": key, "kind": key.split("_", 1)[0], "locality": locality, **item})
        captured_rows=capture_many(model,block,items)
        for key, item, locality in items:
            captured = captured_rows[key]
            for tensor_name, value in captured.items():
                tensors[f"{key}__{tensor_name}"] = value
        path = args.cache_dir / f"record_{record['record_id']}.safetensors"
        save_file(tensors, str(path))
        loaded = load_file(str(path), device="cpu")
        if tensors.keys() != loaded.keys() or any(not torch.equal(tensors[name], loaded[name]) for name in tensors):
            raise RuntimeError("LIVEEDIT_MED_CACHE_RELOAD_MISMATCH")
        row = {"record_id": record["record_id"], "selection_hash": record["selection_hash"], "file": path.name,
               "file_sha256": file_sha256(path), "variants": variant_rows}
        rows.append(row)
        print(json.dumps({"cached": record_index + 1, "total": len(records), "record_id": record["record_id"], "variants": len(variant_rows)}), flush=True)
    manifest = {"protocol": "LIVEEDIT_MED_V4_LAYER21_CACHE", "source_records": str(args.source_records),
                "records": rows, "count": len(rows), "base_model_frozen": True, "layer_path": "model.layers.21"}
    (args.cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
