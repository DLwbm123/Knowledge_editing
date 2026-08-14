#!/usr/bin/env python3
"""Run Stage F once, gated by a no-leakage post-hoc checkpoint selection."""
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

from dsca_medmkeb_diag_common import to_jsonable
from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook, resolve_layer21_block
from methods.liveedit_med.posthoc_validation import PROTOCOL, canonical_json_hash, unrestricted_match
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.source_ops import apply_low_rank_expert_residual
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_one_shot_natural_generation_rescue import full_generation_parity
from scripts.engram.run_engram_v2_stage0_generation_audit import (apply_prefix, bank_manifest,
    build_views, load_model_views_bank, state_weight_hash)
from scripts.engram.run_engram_v2_stage0abc_diagnostics import SHORT_INSTRUCTION
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import (capture_teacher_forced,
    forced_generation, sample_to_model_row)


RECORD_ID = "953"
EXPECTED_BANK_HASH = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True); handle.write("\n")


def compact_parity(value):
    return {"passed": value["passed"], **{name: {key: row[key] for key in
        ("raw_output", "token_ids", "stop_reason", "cap_hit", "eos_step")}
        for name, row in value.items() if name in ("no_cache", "cached", "hf")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()

    # The no-leakage selection is fully authenticated before the code path that
    # loads record 953 is imported/executed.
    selection_path = args.recovery_dir / "checkpoint_selection.json"
    panel_path = args.recovery_dir / "validation_panel_manifest.json"
    selection = json.loads(selection_path.read_text()); panel = json.loads(panel_path.read_text())
    if selection.get("protocol") != PROTOCOL or selection.get("record953_used_for_selection") is not False:
        raise RuntimeError("LIVEEDIT_MED_INVALID_NO_LEAKAGE_SELECTION")
    if selection.get("panel_hash") != panel.get("panel_hash") or panel.get("record953_excluded") is not True:
        raise RuntimeError("LIVEEDIT_MED_PANEL_SELECTION_HASH_MISMATCH")
    saved_hash = selection.pop("selection_hash")
    if canonical_json_hash(selection) != saved_hash:
        raise RuntimeError("LIVEEDIT_MED_SELECTION_ARTIFACT_HASH_MISMATCH")
    selection["selection_hash"] = saved_hash
    if not selection.get("stage_f_permitted") or selection.get("selected_step") is None:
        raise RuntimeError("LIVEEDIT_MED_STAGE_F_NOT_PERMITTED")
    if args.out.exists():
        raise FileExistsError(args.out)

    step = int(selection["selected_step"])
    checkpoint = args.run_dir / "training" / f"checkpoint_{step:04d}"
    state, checkpoint_manifest = load_safe_state(checkpoint)
    if int(checkpoint_manifest["step"]) != step:
        raise RuntimeError("LIVEEDIT_MED_STAGE_F_CHECKPOINT_MISMATCH")

    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 0); clean_hash = state_weight_hash(model)
    _, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    modules.load_state_dict(state, strict=True); modules.eval()
    target_row = views[RECORD_ID]["target"]
    target_sample = {"image": target_row["image_path"][0], "prompt": target_row["prompt"][0],
                     "target": target_row["target"][0]}
    captured = capture_teacher_forced(model, block, target_sample)
    eqr, evr, moe_c, moe_r = modules.generated_edit(captured["vision"].float(),
                                                    captured["question"].float(),
                                                    captured["answer"].float())
    raw = records[RECORD_ID]
    image_root = Path(target_sample["image"]).parents[1]
    views_to_test = {
        "native": target_sample,
        "short_answer": {"image": target_sample["image"],
            "prompt": f"Question: {raw['src']} {SHORT_INSTRUCTION} Short answer: ",
            "target": target_sample["target"]},
        "textual": {"image": target_sample["image"], "prompt": raw["rephrase"], "target": raw["alt"]},
        "visual": {"image": str(image_root / raw["image_rephrase"]), "prompt": raw["src"], "target": raw["alt"]},
        "paired": {"image": str(image_root / raw["image_rephrase"]),
            "prompt": raw["port_new"][0]["Q&A"]["Question"], "target": raw["alt"]},
    }
    outputs = {name: forced_generation(model, block, modules, sample, moe_c, moe_r)
               for name, sample in views_to_test.items()}

    native_canonical = build_canonical_inputs(model, sample_to_model_row(target_sample))
    hook = Layer21ResidualHook(block, lambda hidden: apply_low_rank_expert_residual(
        hidden.float(), moe_c, moe_r, torch.ones(1, 1, device=hidden.device),
        modules.instant_reps_norm).to(hidden.dtype)).install(); hook.enabled = True
    parity = compact_parity(full_generation_parity(model, native_canonical)); hook.remove()
    native = outputs["native"]
    native_success = bool(native["match"]["success"] and parity["passed"])
    result = {"protocol": PROTOCOL, "stage": "F", "executed_once": True,
              "selection_label": selection["label"], "selection_hash": saved_hash,
              "selected_step": step, "checkpoint_manifest": checkpoint_manifest,
              "record_id": RECORD_ID, "record_specific_optimization": False,
              "generated_expert_shapes": {"eqr": list(eqr.shape), "evr": list(evr.shape),
                  "moe_c": list(moe_c.shape), "moe_r": list(moe_r.shape)},
              "outputs": outputs, "native_three_path_parity": parity,
              "native_unrestricted_success": native_success,
              "label": "LIVEEDIT_GENERATOR_FORCED_ON_PASS_SOURCE_OBJECTIVE" if native_success
                       else "LIVEEDIT_GENERATOR_TRANSFER_FAILURE",
              "stage_q_permitted": native_success,
              "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH,
              "base_state_unchanged": state_weight_hash(model) == clean_hash,
              "generation_config": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128}}
    if not result["canonical_bank_unchanged"] or not result["base_state_unchanged"]:
        raise RuntimeError("LIVEEDIT_MED_BASE_OR_BANK_MUTATION")
    write_json(args.out, result)
    print(json.dumps({"label": result["label"], "native": native["raw_output"],
                      "stage_q_permitted": result["stage_q_permitted"]}), flush=True)


if __name__ == "__main__":
    main()
