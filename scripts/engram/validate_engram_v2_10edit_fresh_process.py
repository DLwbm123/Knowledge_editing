#!/usr/bin/env python3
"""Independent fresh-process prefix replay validator for the frozen ENGRAM V2 10-edit gate."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dsca_medmkeb_diag_common import ensure_offline_env
from easyeditor.models.engram import EngramMultimodalHparams
from easyeditor.models.engram_v2 import SequentialEngramBankV2
from easyeditor.trainer.models import get_model
from scripts.engram.run_engram_continual_v2 import build_views, set_determinism
from scripts.engram.run_engram_v2_10edit_gate import ORDER, clean, evaluate_set, hash_state, stage_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config", default="hparams/ENGRAM/llava_med_continual_v2.yaml")
    parser.add_argument("--model-config", default="hparams/ENGRAM/llava_med_continual_v1.yaml")
    return parser.parse_args()


def compare_clean(left: Mapping[str, Dict[str, Any]], right: Mapping[str, Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    if set(left) != set(right):
        return {"passed": False, "key_sets_equal": False, "max_nll_abs_diff": math.inf, "max_logits_abs_diff": math.inf}
    max_nll = 0.0
    all_logits_exact = True
    all_preparation_equal = True
    rows = {}
    prep_fields = (
        "input_ids_sha256", "labels_sha256", "attention_mask_sha256",
        "inputs_embeds_sha256", "pixel_values", "image_metadata", "use_cache_argument",
    )
    for name in sorted(left):
        lrow, rrow = left[name], right[name]
        nll_diff = abs(float(lrow["metric"]["target_nll"]) - float(rrow["metric"]["target_nll"]))
        logits_exact = lrow["logits_sha256"] == rrow["logits_sha256"]
        logits_diff = 0.0 if logits_exact else math.inf
        preparation_equal = all(lrow[field] == rrow[field] for field in prep_fields)
        max_nll = max(max_nll, nll_diff)
        all_logits_exact = all_logits_exact and logits_exact
        all_preparation_equal = all_preparation_equal and preparation_equal
        rows[name] = {
            "nll_abs_diff": nll_diff,
            "logits_exact": logits_exact,
            "logits_max_abs_diff": logits_diff,
            "preparation_equal": preparation_equal,
        }
    return {
        "passed": max_nll <= tolerance and all_logits_exact and all_preparation_equal,
        "key_sets_equal": True,
        "max_nll_abs_diff": max_nll,
        "max_logits_abs_diff": 0.0 if all_logits_exact else math.inf,
        "all_logits_exact": all_logits_exact,
        "all_preparation_equal": all_preparation_equal,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    output = args.out_dir / "preflight" / "fresh_process_prefix_validation.json"
    if output.exists():
        raise FileExistsError(output)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    main_result_path = args.out_dir / "run" / "main_result.json"
    if not main_result_path.exists():
        raise FileNotFoundError(main_result_path)
    ensure_offline_env()
    config = yaml.safe_load((ROOT / args.config).read_text())
    set_determinism(int(config["seed"]))
    backend = {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }
    expected = {"flash": False, "mem_efficient": False, "math": True, "cudnn": False, "deterministic": True}
    if torch.cuda.device_count() != 1 or backend != expected:
        raise RuntimeError(f"fresh-process backend mismatch: count={torch.cuda.device_count()} backend={backend}")

    model_config = EngramMultimodalHparams.from_hparams(str((ROOT / args.model_config).resolve()))
    model_config.dropout, model_config.no_grad_layers, model_config.device = 0.0, None, "cuda"
    model = get_model(model_config).to(torch.device("cuda")).eval()
    records = {str(row["id"]): row for row in json.loads((ROOT / config["dataset"]).read_text())}
    image_root = Path(model_config.coco_image)
    if not image_root.is_absolute():
        image_root = ROOT / image_root
    views = {record_id: build_views(model, records[record_id], image_root) for record_id in ORDER}
    main_result = json.loads(main_result_path.read_text())
    baseline_state = main_result["baseline_determinism"]["state_before"]
    fresh_initial_state = hash_state(model)
    bank_root = args.out_dir / "run" / "bank"
    tolerance = float(config["replay_nll_tolerance"])
    prefixes = []

    for step in range(1, len(ORDER) + 1):
        fresh_bank = SequentialEngramBankV2(bank_root)
        fresh_bank.rollback_to_prefix(model, 0)
        fresh_bank.assemble_state_into_model(model, prefix=step)
        snapshots = clean(evaluate_set(model, stage_samples(views, ORDER[:step])))
        state = hash_state(model)
        stage = json.loads((args.out_dir / "run" / f"stage_{step:02d}.json").read_text())
        direct = stage["direct_snapshots"]
        comparison = compare_clean(direct, snapshots, tolerance)
        bank_row = stage["bank_equivalence"]
        metadata = fresh_bank.list_edits()
        metadata_equal = (
            [item["edit_id"] for item in metadata] == bank_row["edit_order"]
            and [item["parent_state_hash"] for item in metadata] == bank_row["parent_state_hashes"]
            and [item["resulting_state_hash"] for item in metadata] == bank_row["resulting_state_hashes"]
            and [item["delta_checksums"] for item in metadata] == bank_row["delta_checksums"]
        )
        state_equal = state == bank_row["direct_state"]
        prefixes.append({
            "prefix_length": step,
            "record_ids": ORDER[:step],
            "state_equal": state_equal,
            "metadata_equal": metadata_equal,
            "comparison": comparison,
            "fresh_state": state,
            "direct_state": bank_row["direct_state"],
            "passed": state_equal and metadata_equal and comparison["passed"],
        })

    bank = SequentialEngramBankV2(bank_root)
    bank.rollback_to_prefix(model, 0)
    final_rollback_state = hash_state(model)
    payload = {
        "process_kind": "independent fresh model process after main 10-edit process exited",
        "pid": os.getpid(),
        "fixed_order": ORDER,
        "backend": backend,
        "fresh_initial_anchor_state_equal": fresh_initial_state == baseline_state,
        "prefixes": prefixes,
        "prefixes_passed": sum(bool(row["passed"]) for row in prefixes),
        "all_prefixes_passed": all(bool(row["passed"]) for row in prefixes),
        "max_nll_abs_diff": max(float(row["comparison"]["max_nll_abs_diff"]) for row in prefixes),
        "max_logits_abs_diff": max(float(row["comparison"]["max_logits_abs_diff"]) for row in prefixes),
        "final_anchor_rollback_state_equal": final_rollback_state == baseline_state,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "fresh_process_prefixes_passed": f"{payload['prefixes_passed']}/10",
        "all_prefixes_passed": payload["all_prefixes_passed"],
        "max_nll_abs_diff": payload["max_nll_abs_diff"],
        "max_logits_abs_diff": payload["max_logits_abs_diff"],
        "final_anchor_rollback_state_equal": payload["final_anchor_rollback_state_equal"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
