#!/usr/bin/env python3
"""One-shot engineering diagnostic for Stage-D zero-effect generation parity."""
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

from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook, resolve_layer21_block
from methods.liveedit_med.source_ops import direct_expert_residual
from methods.liveedit_med.upstream_modules import reset_layer_norm
from scripts.engram.run_engram_v2_stage0_generation_audit import (
    apply_prefix, build_canonical_inputs, eos_ids, load_model_views_bank,
)
from scripts.engram.run_llavamed_record953_lora_positive_control import CAP, RECORD_ID, seed_everything
from scripts.engram.stage0_generation_audit_utils import manual_greedy_trace


def trace(model, canonical):
    row = manual_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=1)
    return {"token_ids": row["token_ids"], "raw_output": row["raw_output"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, default=2)
    args = parser.parse_args()
    seed_everything()
    model, views, bank, _records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 0)
    _name, block = resolve_layer21_block(model)
    canonical = build_canonical_inputs(model, views[RECORD_ID]["target"])
    rows = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    rows["baseline_1"] = trace(model, canonical)
    rows["baseline_2"] = trace(model, canonical)
    zero_hook = Layer21ResidualHook(block, lambda hidden: torch.zeros_like(hidden)).install()
    zero_hook.enabled = True
    rows["zeros_like_hook"] = trace(model, canonical)
    zero_hook.remove()
    raw_c = torch.empty(4, 4096, device=model.lm_device, dtype=torch.float32)
    torch.nn.init.kaiming_normal_(raw_c)
    raw_r = torch.zeros_like(raw_c)
    norm = reset_layer_norm(torch.nn.LayerNorm(4096, device=model.lm_device, dtype=torch.float32)).eval().requires_grad_(False)
    residual_stats = []
    def low_rank_residual(hidden):
        value = direct_expert_residual(hidden.float(), raw_c, raw_r, norm).to(hidden.dtype)
        if len(residual_stats) < 3:
            residual_stats.append({
                "shape": list(value.shape),
                "hidden_dtype": str(hidden.dtype),
                "value_dtype": str(value.dtype),
                "finite": bool(torch.isfinite(value).all()),
                "exact_zero": bool(torch.equal(value, torch.zeros_like(value))),
                "max_abs": float(torch.nan_to_num(value).abs().max()),
                "raw_r_exact_zero": bool(torch.equal(raw_r, torch.zeros_like(raw_r))),
                "norm_finite": bool(torch.isfinite(norm(hidden.float())).all()),
                "hidden_finite": bool(torch.isfinite(hidden).all()),
                "hidden_nan_count": int(torch.isnan(hidden).sum()),
                "hidden_posinf_count": int(torch.isposinf(hidden).sum()),
                "hidden_neginf_count": int(torch.isneginf(hidden).sum()),
                "hidden_finite_max_abs": float(torch.nan_to_num(hidden.float()).abs().max()),
                "norm_weight_finite": bool(torch.isfinite(norm.weight).all()),
                "norm_bias_finite": bool(torch.isfinite(norm.bias).all()),
            })
        return value
    residual_hook = Layer21ResidualHook(block, low_rank_residual).install()
    residual_hook.enabled = True
    rows["low_rank_zero_hook"] = trace(model, canonical)
    rows["residual_stats"] = residual_stats
    residual_hook.remove()
    rows["equal"] = {
        "repeat": rows["baseline_1"]["token_ids"] == rows["baseline_2"]["token_ids"],
        "zeros_like": rows["baseline_1"]["token_ids"] == rows["zeros_like_hook"]["token_ids"],
        "low_rank_zero": rows["baseline_1"]["token_ids"] == rows["low_rank_zero_hook"]["token_ids"],
    }
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
