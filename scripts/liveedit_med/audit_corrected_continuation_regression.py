#!/usr/bin/env python3
"""One-real-batch regression gate for the archived corrected continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.cached_suffix import answer_kl
from methods.liveedit_med.source_ops import (
    apply_low_rank_expert_residual,
    compute_text_soft_weights,
    source_routing_losses,
    source_soft_losses,
)
from methods.liveedit_med.source_training_continuation import (
    SourceTrainingContinuationMode,
    forward_source_training_hidden,
)
from methods.liveedit_med.trace_parity import state_dict_sha256
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import apply_prefix, state_weight_hash
from scripts.engram.run_llavamed_record953_lora_positive_control import seed_everything
from scripts.liveedit_med.train_liveedit_med_v4_source import (
    edited_values,
    load_training_model,
    pad_rows,
    routing_pairs,
    spans,
    split_variant,
)


MODE = SourceTrainingContinuationMode.CORRECTED_SEMANTICS_CONTINUE_LAYER22


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def legacy_forward_suffix_hidden(llava_model, hidden, attention_mask):
    """Verbatim archived implementation before the shared-mode refactor."""
    core = llava_model.model
    batch, length, _ = hidden.shape
    cache_position = torch.arange(length, device=hidden.device)
    position_ids = cache_position.unsqueeze(0).expand(batch, -1)
    causal_mask = core._update_causal_mask(attention_mask, hidden, cache_position, None, False)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    for layer in core.layers[22 : core.config.num_hidden_layers]:
        def layer_forward(value, current_layer=layer):
            return current_layer(
                value,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        hidden = checkpoint(layer_forward, hidden, use_reentrant=False)
    return core.norm(hidden)


def grouped_losses(modules, model, groups, moe_cs, moe_rs, eqrs, suffix_device, implementation):
    all_rows, all_clean, all_residual, ranges = [], [], [], []
    for name, rows, masks, locality in groups:
        begin = len(all_rows)
        all_rows.extend(rows)
        values = edited_values(modules, model, rows, moe_cs, moe_rs, masks, eqrs)
        all_clean.extend(value[0] for value in values)
        all_residual.extend(value[1] for value in values)
        ranges.append((name, begin, len(all_rows), locality))
    edited = [
        (clean + residual).to(model.llava_model.dtype)
        for clean, residual in zip(all_clean, all_residual)
    ]
    hidden, attention, _labels = pad_rows(all_rows, edited, suffix_device)
    if implementation == "legacy":
        suffix_hidden = legacy_forward_suffix_hidden(model.llava_model, hidden, attention)
    elif implementation == "shared":
        suffix_hidden = forward_source_training_hidden(
            model.llava_model, hidden, attention, mode=MODE,
            gradient_checkpointing=True,
        )
    else:
        raise RuntimeError(f"UNKNOWN_IMPLEMENTATION:{implementation}")
    logits = []
    for index, row in enumerate(all_rows):
        predictors = torch.where(row["answer"])[0] - 1
        logits.append(model.llava_model.lm_head(suffix_hidden[index, predictors.to(suffix_hidden.device)]))
    result = {}
    for name, begin, end, locality in ranges:
        losses = []
        for index in range(begin, end):
            row = all_rows[index]
            if locality:
                losses.append(answer_kl(logits[index], row["base_answer_logits"].to(logits[index].device)))
            else:
                target = row["labels"][row["answer"]].long().to(logits[index].device)
                losses.append(torch.nn.functional.cross_entropy(logits[index].float(), target))
        result[name] = torch.stack(losses).mean().to(model.lm_device)
    return result


def optimizer_state_hash(modules, optimizer) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(modules.named_parameters()):
        digest.update(name.encode())
        for label, value in [("parameter", parameter.detach())] + sorted(optimizer.state[parameter].items()):
            digest.update(str(label).encode())
            if isinstance(value, torch.Tensor):
                tensor = value.detach().contiguous().cpu()
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def run_once(modules, model, batch, suffix_device, implementation):
    rng_data = np.random.default_rng(42)
    rng_train = np.random.default_rng(43)
    optimizer, scheduler = modules.optimizer()
    optimizer.zero_grad(set_to_none=True)
    edits = []
    for record in batch:
        vision, question, answer = spans(record["variants"]["native_0"])
        edits.append(modules.generated_edit(vision, question, answer))
    eqrs = torch.cat([item[0] for item in edits])
    moe_cs = torch.cat([item[2] for item in edits])
    moe_rs = torch.cat([item[3] for item in edits])
    count = len(batch)
    rel_mask = torch.eye(count, device=model.lm_device, dtype=torch.bool)
    gen_mask = rel_mask.clone()
    loc_mask = torch.zeros_like(rel_mask)
    prefixes = []
    for index in range(count):
        ns = rng_train.integers(0, count + 1, 3)
        prefixes.append(ns.tolist())
        rel_mask[index, : ns[0]] = True
        gen_mask[index, : ns[1]] = True
        loc_mask[index, : ns[2]] = True
    rel_rows = [record["variants"]["native_0"] for record in batch]
    loc_rows = [record["variants"]["loc_image_or_paired_0"] for record in batch]
    groups = [("rel", rel_rows, rel_mask, False)]
    groups += [
        (f"gen_{name}", [record["variants"][f"gen_{name}_0"] for record in batch], gen_mask, False)
        for name in ("textual", "visual", "paired")
    ]
    groups += [("loc", loc_rows, loc_mask, True)]
    task = grouped_losses(modules, model, groups, moe_cs, moe_rs, eqrs, suffix_device, implementation)
    neighbors, prototypes = routing_pairs(batch, rng_data)
    input_keys = torch.cat([modules.input_extractor.extract_query(pair[1]) for pair in neighbors[0]])
    edit_keys = torch.cat([modules.edit_extractor.extract_query(pair[1]) for pair in neighbors[1]])
    soft_rel, soft_abs = source_soft_losses(input_keys, edit_keys)
    hard_neighbor, hard_prototype = source_routing_losses(
        modules.input_extractor, modules.edit_extractor,
        neighbors[0], neighbors[1], prototypes[0], prototypes[1],
    )
    components = {
        "reliability": task["rel"],
        "generality_textual": task["gen_textual"],
        "generality_visual": task["gen_visual"],
        "generality_paired": task["gen_paired"],
        "locality": task["loc"],
        "soft_relative": soft_rel,
        "soft_absolute": soft_abs,
        "hard_neighbor": hard_neighbor,
        "hard_prototype": hard_prototype,
    }
    total = sum(components.values())
    total.backward()
    grad_hash = state_dict_sha256({
        name: parameter.grad for name, parameter in modules.named_parameters()
    })
    optimizer.step()
    scheduler.step()
    result = {
        "implementation": implementation,
        "loss": float(total.detach()),
        "components": {name: float(value.detach()) for name, value in components.items()},
        "gradient_hash": grad_hash,
        "parameter_hash_after_adam": state_dict_sha256(modules.state_dict()),
        "optimizer_state_hash_after_adam": optimizer_state_hash(modules, optimizer),
        "prefixes": prefixes,
    }
    del optimizer, scheduler, total
    modules.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--suffix-physical-gpu", type=int, default=3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    entries = manifest["records"]
    if len(entries) != 512:
        raise RuntimeError(f"CORRECTED_REGRESSION_CACHE_COUNT:{len(entries)}")
    model, bank, suffix_device = load_training_model(args.physical_gpu, args.suffix_physical_gpu, MODE)
    apply_prefix(model, bank, 0)
    clean_model_hash = state_weight_hash(model)
    for parameter in model.llava_model.parameters():
        parameter.requires_grad_(False)
    seed_everything()
    modules = LiveEditMedicalModules(LiveEditMedicalConfig(source_training_continuation_mode=MODE)).to(model.lm_device).float()
    modules.train()
    initial = {name: value.detach().cpu().clone() for name, value in modules.state_dict().items()}
    initial_hash = state_dict_sha256(initial)
    order = np.random.default_rng(42).permutation(len(entries))[:8]
    batch = []
    for index in order:
        entry = entries[int(index)]
        tensors = load_file(str(args.cache_dir / entry["file"]), device="cpu")
        keys = [variant["key"] for variant in entry["variants"]]
        batch.append({
            "record_id": entry["record_id"],
            "variants": {key: split_variant(tensors, key, model.lm_device) for key in keys},
        })
    legacy = run_once(modules, model, batch, suffix_device, "legacy")
    modules.load_state_dict(initial, strict=True)
    shared = run_once(modules, model, batch, suffix_device, "shared")
    checks = {
        "loss_exact": legacy["loss"] == shared["loss"],
        "components_exact": legacy["components"] == shared["components"],
        "gradient_hash_exact": legacy["gradient_hash"] == shared["gradient_hash"],
        "parameter_hash_after_adam_exact": legacy["parameter_hash_after_adam"] == shared["parameter_hash_after_adam"],
        "optimizer_state_hash_after_adam_exact": legacy["optimizer_state_hash_after_adam"] == shared["optimizer_state_hash_after_adam"],
        "sampling_exact": legacy["prefixes"] == shared["prefixes"],
        "base_model_unchanged": state_weight_hash(model) == clean_model_hash,
    }
    status = "CORRECTED_SEMANTICS_REGRESSION_PASS" if all(checks.values()) else "CORRECTED_SEMANTICS_REGRESSION_FAIL"
    result = {
        "status": status,
        "mode": MODE.value,
        "record_ids": [item["record_id"] for item in batch],
        "initial_module_hash": initial_hash,
        "legacy": legacy,
        "shared": shared,
        "checks": checks,
        "edited_checkpoint_loaded": False,
    }
    write_json(args.out_dir / "corrected_semantics_regression.json", result)
    print(json.dumps({"status": status, "checks": checks}, sort_keys=True))
    if not all(checks.values()):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
