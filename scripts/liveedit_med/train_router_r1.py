#!/usr/bin/env python3
"""Train only LiveEdit routing extractors on frozen medical-domain caches."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.router_r1 import (
    CHECKPOINT_STEPS,
    EXPECTED_BANK_HASH,
    NEGATIVE_CATEGORIES,
    PROTOCOL,
    assert_router_only,
    canonical_hash,
    configure_router_only,
    frozen_module_state,
    negative_category,
    negative_text_absolute_loss,
    negative_visual_loss,
    positive_text_losses,
    positive_visual_loss,
    repository_size,
    router_state,
    semantic_category,
)
from methods.liveedit_med.serialization import load_safe_state, save_safe_state, tensor_hashes
from methods.liveedit_med.source_ops import compute_text_soft_weights
from methods.liveedit_med.source_training_continuation import SourceTrainingContinuationMode
from methods.liveedit_med.trace_parity import state_dict_sha256
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import apply_prefix, bank_manifest, state_weight_hash
from scripts.engram.run_llavamed_record953_lora_positive_control import seed_everything
from scripts.liveedit_med.train_liveedit_med_v4_source import grouped_residual_losses, load_training_model


STRICT_MODE = SourceTrainingContinuationMode.STRICT_SOURCE_REAPPLY_LAYER21


def recursive_hash(value: Any) -> str:
    def normalize(item: Any):
        if torch.is_tensor(item):
            return {"tensor": tensor_hashes({"value": item})["value"]}
        if isinstance(item, Mapping):
            return {str(key): normalize(val) for key, val in sorted(item.items(), key=lambda row: str(row[0]))}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        return item
    return canonical_hash(normalize(value))


def variant(tensors: Mapping[str, torch.Tensor], name: str, device: torch.device):
    prefix = name + "__"
    raw = {key[len(prefix):]: value.to(device) for key, value in tensors.items() if key.startswith(prefix)}
    if not raw:
        raise RuntimeError(f"ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:missing_variant:{name}")
    return {"hidden": raw["hidden"].float().unsqueeze(0), "labels": raw["labels"],
            "attention": raw["attention"], "vision": raw["vision_mask"],
            "prompt": raw["question_mask"], "answer": raw["answer_mask"],
            "base_answer_logits": raw["base_answer_logits"].float()}


def spans(row: Mapping[str, torch.Tensor]):
    hidden = row["hidden"]
    return hidden[:, row["vision"]], hidden[:, row["prompt"]], hidden[:, row["answer"]]


def load_maps(manifest_path: Path, hard_path: Path):
    manifest = json.loads(manifest_path.read_text())
    regular = {split: {str(row["record_id"]): row for row in manifest["splits"][split]}
               for split in ("train", "validation", "heldout")}
    hard_manifest = json.loads(hard_path.read_text())
    hard = {(row["split"], str(row["record_id"])): row for row in hard_manifest["records"]}
    return manifest, regular, hard


def membership_map(path: Path):
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "train":
                result[(int(row["step"]), str(row["record_id"]))] = row
    return result


def load_regular(entry: Mapping[str, Any]):
    return load_file(entry["file_path"], device="cpu")


def module_keys(modules, tensors, device):
    native = variant(tensors, "native", device)
    vision, question, _answer = spans(native)
    return modules.edit_extractor.extract_query(question), modules.edit_extractor.extract_vision(question, vision)


def query_keys(modules, row):
    vision, question, _answer = spans(row)
    return (modules.input_extractor.extract_query(question),
            modules.input_extractor.extract_vision(question, vision),
            modules.input_extractor.extract_from_visprot(question),
            modules.edit_extractor.extract_query(question))


def global_expert_hash(expert_manifest: Mapping[str, Any]) -> str:
    return canonical_hash([{"split": row["split"], "record_id": row["record_id"],
                            "hashes": row["expert_tensor_hashes"]}
                           for row in expert_manifest["records"]])


def save_checkpoint(run_dir: Path, step: int, epoch: int, modules, optimizer,
                    frozen_hash: str, expert_hash: str, base_hash: str) -> dict[str, Any]:
    assert_router_only(modules)
    if state_dict_sha256(frozen_module_state(modules)) != frozen_hash:
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:frozen_module_mutation")
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:bank_mutation")
    directory = run_dir / f"checkpoint_{step:04d}"
    manifest = save_safe_state(directory, modules.state_dict(), {
        "protocol": PROTOCOL, "step": step, "epoch": epoch, "router_only": True,
        "trainable_tensor_hashes": tensor_hashes(router_state(modules)),
        "frozen_module_hash": frozen_hash, "frozen_expert_hash": expert_hash,
        "base_model_hash": base_hash, "canonical_bank_hash": EXPECTED_BANK_HASH,
        "optimizer_state_hash": recursive_hash(optimizer.state_dict()),
        "source_training_continuation": STRICT_MODE.value,
        "inference_continuation": "official_layer21_output_hook_then_layer22",
    })
    return {"step": step, "epoch": epoch, "directory": directory.name,
            "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
            "tensor_hashes": manifest["tensor_hashes"],
            "optimizer_state_hash": manifest["optimizer_state_hash"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-manifest", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--hard-cache", type=Path, required=True)
    parser.add_argument("--nearest", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--strict-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--suffix-physical-gpu", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=640)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != f"{args.physical_gpu},{args.suffix_physical_gpu}":
        raise RuntimeError("LIVEEDIT_MED_TWO_GPU_VISIBILITY_MISMATCH")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything()
    manifest, regular, hard = load_maps(args.representation_manifest, args.hard_cache)
    expert_manifest = json.loads(args.expert_manifest.read_text())
    expert_hash = global_expert_hash(expert_manifest)
    nearest = json.loads(args.nearest.read_text())["neighbors"]
    memberships = membership_map(args.membership)
    model, bank, suffix_device = load_training_model(args.physical_gpu, args.suffix_physical_gpu, STRICT_MODE)
    apply_prefix(model, bank, 0)
    base_hash = state_weight_hash(model)
    for parameter in model.llava_model.parameters():
        parameter.requires_grad_(False)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig(
        learning_rate=5e-5, source_training_continuation_mode=STRICT_MODE)).to(model.lm_device).float()
    initial_state, strict_checkpoint_manifest = load_safe_state(args.strict_checkpoint)
    if strict_checkpoint_manifest.get("step") != 3200:
        raise RuntimeError("ROUTER_R1_ANCHOR_MISMATCH:strict_checkpoint")
    modules.load_state_dict(initial_state, strict=True)
    trainable_names = configure_router_only(modules)
    modules.train()
    frozen_hash = state_dict_sha256(frozen_module_state(modules))
    initial_router_hash = state_dict_sha256(router_state(modules))
    optimizer = torch.optim.Adam([parameter for parameter in modules.parameters() if parameter.requires_grad],
                                 lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=0)
    if optimizer.state:
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:old_optimizer_moments")
    parameter_audit = {"protocol": PROTOCOL, "trainable_parameters": list(trainable_names),
        "trainable_parameter_count": sum(parameter.numel() for parameter in modules.parameters() if parameter.requires_grad),
        "frozen_parameters": [name for name, parameter in modules.named_parameters() if not parameter.requires_grad],
        "only_edit_and_input_extractors_trainable": True, "fresh_optimizer_empty_state": True,
        "initial_router_hash": initial_router_hash, "frozen_module_hash": frozen_hash}
    (args.run_dir.parent / "trainable_parameter_audit.json").write_text(
        json.dumps(parameter_audit, indent=2, sort_keys=True) + "\n")
    config = """protocol: LIVEEDIT_MED_ROUTER_ONLY_DOMAIN_ADAPTATION_R1
optimizer: Adam
learning_rate: 5.0e-5
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0
max_grad_norm: 1.0
batch_size: 8
epochs: 10
optimizer_steps: 640
seed: 42
scheduler: constant
checkpoint_steps: [80, 160, 240, 320, 400, 480, 560, 640]
source_training_continuation: strict_source_reapply_layer21
inference_continuation: official_layer21_output_hook_then_layer22
"""
    (args.run_dir / "router_r1_config.yaml").write_text(config)
    trajectory_path = args.run_dir / "training_trajectory.jsonl"
    trajectory_path.write_text("")
    frozen_ledger = args.run_dir / "frozen_hash_ledger.jsonl"
    frozen_ledger.write_text("")
    train_ids = [str(row["record_id"]) for row in manifest["splits"]["train"]]
    rng = np.random.default_rng(42)
    step = 0
    checkpoints = []
    for epoch in range(1, 11):
        order = rng.permutation(len(train_ids))
        for begin in range(0, len(train_ids), 8):
            step += 1
            if step > args.max_steps:
                break
            ids = [train_ids[int(index)] for index in order[begin:begin + 8]]
            size = repository_size(step)
            semantic = semantic_category(step)
            negative = negative_category(step)
            optimizer.zero_grad(set_to_none=True)
            all_eqr, all_evr, all_c, all_r = [], [], [], []
            target_positions = []
            target_rows = []
            route_losses = {"hard": [], "absolute": [], "relative": []}
            regular_cache: dict[str, Mapping[str, torch.Tensor]] = {}
            hard_cache: dict[str, Mapping[str, torch.Tensor]] = {}
            for rid in ids:
                member = memberships[(step, rid)]
                repo_ids = [str(value) for value in member["repository_ids"]]
                if len(repo_ids) != size or repo_ids[0] != rid:
                    raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:repository")
                begin_repo = len(all_eqr)
                for member_id in repo_ids:
                    tensors = regular_cache.setdefault(member_id, load_regular(regular["train"][member_id]))
                    eqr, evr = module_keys(modules, tensors, model.lm_device)
                    all_eqr.append(eqr); all_evr.append(evr)
                    all_c.append(tensors["expert__moe_c"].float().to(model.lm_device))
                    all_r.append(tensors["expert__moe_r"].float().to(model.lm_device))
                target_positions.append(begin_repo)
                target_tensor = regular_cache[rid]
                native_row = variant(target_tensor, "native", model.lm_device)
                semantic_row = variant(target_tensor, semantic, model.lm_device)
                if negative in ("same_image_different_question", "same_question_different_image"):
                    hard_entry = hard[("train", rid)]
                    hard_tensor = hard_cache.setdefault(rid, load_file(hard_entry["file_path"], device="cpu"))
                    negative_row = variant(hard_tensor, negative, model.lm_device)
                else:
                    other_id = nearest[rid]["chosen"][negative]
                    negative_row = variant(regular_cache.setdefault(other_id,
                        load_regular(regular["train"][other_id])), "native", model.lm_device)
                locality_row = variant(target_tensor, "image_locality", model.lm_device)
                target_rows.append((native_row, semantic_row, negative_row, locality_row, begin_repo, begin_repo + size))
            eqrs = torch.cat(all_eqr); evrs = torch.cat(all_evr)
            moe_cs = torch.cat(all_c); moe_rs = torch.cat(all_r)
            masks = []
            native_rows, semantic_rows, negative_rows, locality_rows = [], [], [], []
            for native_row, semantic_row, negative_row, locality_row, repo_begin, repo_end in target_rows:
                mask = torch.zeros(len(all_eqr), dtype=torch.bool, device=model.lm_device)
                mask[repo_begin:repo_end] = True
                masks.extend([mask, mask, mask, mask])
                native_rows.append(native_row); semantic_rows.append(semantic_row)
                negative_rows.append(negative_row); locality_rows.append(locality_row)
                repo_eqr, repo_evr = eqrs[repo_begin:repo_end], evrs[repo_begin:repo_end]
                for row in (native_row, semantic_row):
                    iqr, ivr, sentinel, locality_edit_key = query_keys(modules, row)
                    loc_q = spans(locality_row)[1]
                    loc_edit = modules.edit_extractor.extract_query(loc_q)
                    route_losses["hard"].append(positive_visual_loss(ivr, repo_evr, sentinel, 0))
                    absolute, relative = positive_text_losses(iqr, repo_eqr, 0, loc_edit)
                    route_losses["absolute"].append(absolute); route_losses["relative"].append(relative)
                for row in (negative_row, locality_row):
                    iqr, ivr, sentinel, _locality_edit_key = query_keys(modules, row)
                    route_losses["hard"].append(negative_visual_loss(ivr, repo_evr, sentinel))
                    route_losses["absolute"].append(negative_text_absolute_loss(iqr, repo_eqr))
            # grouped_residual_losses expects one mask per row in group order.
            native_masks = masks[0::4]; semantic_masks = masks[1::4]
            negative_masks = masks[2::4]; locality_masks = masks[3::4]
            output = grouped_residual_losses(modules, model, [
                ("native", native_rows, native_masks, False),
                ("semantic", semantic_rows, semantic_masks, False),
                ("hard_negative", negative_rows, negative_masks, True),
                ("locality", locality_rows, locality_masks, True),
            ], moe_cs, moe_rs, eqrs, suffix_device, STRICT_MODE)
            loss_hard = torch.stack(route_losses["hard"]).mean()
            loss_abs = torch.stack(route_losses["absolute"]).mean()
            loss_rel = torch.stack(route_losses["relative"]).mean()
            loss_positive = output["native"] + output["semantic"]
            loss_negative = output["hard_negative"] + output["locality"]
            losses = {"hard": loss_hard, "soft_absolute": loss_abs, "soft_relative": loss_rel,
                      "positive": loss_positive, "negative": loss_negative}
            total = sum(losses.values())
            if not torch.isfinite(total) or any(not torch.isfinite(value) or float(value.detach()) == 0.0 for value in losses.values()):
                raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:nonfinite_or_zero_loss")
            total.backward()
            assert_router_only(modules)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in modules.parameters() if parameter.requires_grad], 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:nonfinite_gradient")
            optimizer.step()
            row = {"protocol": PROTOCOL, "epoch": epoch, "step": step, "record_ids": ids,
                   "repository_size": size, "semantic_category": semantic, "negative_category": negative,
                   "total_loss": float(total.detach()), **{f"loss_{name}": float(value.detach()) for name, value in losses.items()},
                   "gradient_norm": float(grad_norm), "parameter_norm": float(torch.sqrt(sum(
                       parameter.detach().float().square().sum() for parameter in modules.parameters() if parameter.requires_grad))),
                   "nan_or_inf": False, "learning_rate": 5e-5}
            with trajectory_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if step % 20 == 0:
                print(json.dumps({key: row[key] for key in ("epoch", "step", "repository_size", "total_loss",
                    "loss_hard", "loss_soft_absolute", "loss_soft_relative", "loss_positive", "loss_negative",
                    "gradient_norm")}, sort_keys=True), flush=True)
            if step in CHECKPOINT_STEPS:
                checkpoint = save_checkpoint(args.run_dir, step, epoch, modules, optimizer,
                                             frozen_hash, expert_hash, base_hash)
                checkpoints.append(checkpoint)
                with frozen_ledger.open("a") as handle:
                    handle.write(json.dumps({"step": step, "frozen_module_hash": frozen_hash,
                        "frozen_expert_hash": expert_hash, "base_model_hash": base_hash,
                        "canonical_bank_hash": EXPECTED_BANK_HASH, "blind_manifest_hash_audited_only": True,
                        "unchanged": True}, sort_keys=True) + "\n")
        if step >= args.max_steps:
            break
    if args.max_steps == 640 and (step != 640 or tuple(row["step"] for row in checkpoints) != CHECKPOINT_STEPS):
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:incomplete_checkpoints")
    if state_dict_sha256(frozen_module_state(modules)) != frozen_hash or state_weight_hash(model) != base_hash:
        raise RuntimeError("ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:frozen_mutation")
    (args.run_dir / "checkpoint_manifest.json").write_text(json.dumps({"protocol": PROTOCOL,
        "complete": step == 640, "optimizer_steps": step, "checkpoints": checkpoints,
        "frozen_module_hash": frozen_hash, "frozen_expert_hash": expert_hash,
        "base_model_hash": base_hash, "canonical_bank_hash": bank_manifest()["sha256"]},
        indent=2, sort_keys=True) + "\n")
    report = ["# Router R1 training report", "", f"- Optimizer steps: **{step}/640**",
              f"- Checkpoints: **{len(checkpoints)}/8**", "- Trainable modules: `edit_extractor`, `input_extractor`",
              "- Generator, residual norm, backbone, expert tensors, and bank: **frozen and hash-checked**",
              "- Source-training continuation: `strict_source_reapply_layer21`",
              "- Inference continuation: `official_layer21_output_hook_then_layer22`"]
    (args.run_dir / "ROUTER_R1_TRAINING_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": "ROUTER_R1_TRAINING_COMPLETE" if step == 640 else "ROUTER_R1_SMOKE_COMPLETE",
                      "steps": step, "checkpoints": len(checkpoints)}, sort_keys=True))


if __name__ == "__main__":
    main()
