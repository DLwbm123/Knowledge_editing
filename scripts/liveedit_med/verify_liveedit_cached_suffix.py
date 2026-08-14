#!/usr/bin/env python3
"""Real-model equality gate for direct versus cached layer-21 suffix evaluation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.cached_suffix import forward_suffix
from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from scripts.engram.run_engram_v2_stage0_generation_audit import apply_prefix, load_model_views_bank


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=2)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES mismatch")
    source = json.loads(args.source_records.read_text())
    row = source["records"]["train"][0]["requests"][0]
    sample = {"image_path": [row["image"]], "prompt": [row["prompt"]], "target": [row["target_new"]]}
    model, _views, bank, _records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 0)
    _name, block = resolve_layer21_block(model)
    inputs, labels, masks = model._build_one(row["prompt"], row["target_new"], model._image_for_row(sample, 0))
    captured = []
    handle = block.register_forward_hook(lambda _m, _a, output: captured.append((output[0] if isinstance(output, (tuple, list)) else output).detach()))
    direct = model.llava_model(inputs_embeds=inputs.unsqueeze(0), attention_mask=masks["attention_mask"].unsqueeze(0).long(), return_dict=True, use_cache=False).logits
    handle.remove()
    cached = forward_suffix(model.llava_model, captured[0], masks["attention_mask"].unsqueeze(0).long())
    result = {
        "record_id": source["records"]["train"][0]["record_id"],
        "direct_shape": list(direct.shape), "cached_shape": list(cached.shape),
        "exact_equal": bool(torch.equal(direct, cached)),
        "max_abs_error": float((direct.float() - cached.float()).abs().max()),
        "layer21_exact_finite": bool(torch.isfinite(captured[0]).all()),
        "passed": bool(torch.equal(direct, cached)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise RuntimeError("LIVEEDIT_MED_CACHED_SUFFIX_PARITY_FAILURE")


if __name__ == "__main__":
    main()
