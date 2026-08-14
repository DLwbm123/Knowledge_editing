#!/usr/bin/env python3
"""Diagnose LLaVA-Med repeated-forward and ENGRAM V1 metric determinism."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dsca_medmkeb_diag_common import clone_batch, ensure_offline_env
from easyeditor.models.engram import EngramMultimodalHparams
from easyeditor.trainer.models import get_model
from scripts.engram.engram_eval_utils import (
    compare_tensors,
    first_tensor,
    full_state_sha256,
    legacy_tail_metrics,
    module_state_inventory,
    nested_input_hash,
    shifted_teacher_forced_metrics,
    tensor_metadata,
    tensor_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hparams/ENGRAM/llava_med_continual_v1.yaml")
    parser.add_argument("--dataset", default="datasets/MedMKEB/eval.json")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--full-state-hash", action="store_true")
    return parser.parse_args()


def set_determinism(seed: int = 42) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_image(root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if root.name == "images" and str(value).startswith("images/"):
        return root.parent / path
    return root / path


def prompt(value: Any) -> str:
    return f"Question: {str(value or '')} Short answer: "


def make_sample(model: Any, record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    question = prompt(record.get("src"))
    target = str(record.get("alt") or "")
    raw_labels = model.llava_tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return {
        "image_path": [str(resolve_image(image_root, record["image"]))],
        "prompt": [question],
        "target": [target],
        "text_input": [question + target],
        "labels": raw_labels,
        "prompts_len": [len(model.llava_tokenizer(question, add_special_tokens=False).input_ids)],
    }


def backend_report() -> Dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def capture_build_path(model: Any, sample: Dict[str, Any]) -> tuple[Dict[str, str], tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
    captures: Dict[str, str] = {}
    handles = []
    for name, module in model.named_modules():
        interesting = (
            name.endswith("vision_tower.vision_tower")
            or name.endswith("model.mm_projector.0")
            or name.endswith("model.mm_projector.2")
        )
        if not interesting:
            continue
        def pre_hook(_module, inputs, key=f"{name}:input"):
            tensor = first_tensor(inputs)
            if tensor is not None:
                captures[key] = tensor_sha256(tensor)
        def hook(_module, _inputs, output, key=f"{name}:output"):
            tensor = first_tensor(output)
            if tensor is not None:
                captures[key] = tensor_sha256(tensor)
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(hook))
    batch = clone_batch(sample)
    with torch.inference_mode():
        frozen = model._build_batch(batch)
    for handle in handles:
        handle.remove()
    return captures, frozen


def raw_forward(model: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
    batch = clone_batch(sample)
    before_hash = nested_input_hash(batch)
    captures: Dict[str, str] = {}
    handles = []
    for name, module in model.named_modules():
        interesting = (
            name.endswith("vision_tower.vision_tower")
            or name.endswith("model.mm_projector.0")
            or name.endswith("model.mm_projector.2")
        )
        if not interesting:
            continue
        def pre_hook(_module, inputs, key=f"{name}:input"):
            tensor = first_tensor(inputs)
            if tensor is not None:
                captures[key] = tensor_sha256(tensor)
        def hook(_module, _inputs, output, key=f"{name}:output"):
            tensor = first_tensor(output)
            if tensor is not None:
                captures[key] = tensor_sha256(tensor)
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(hook))
    original_image_for_row = model._image_for_row
    image_hashes: List[str] = []
    def traced_image_for_row(samples, row):
        value = original_image_for_row(samples, row)
        image_hashes.append(tensor_sha256(value))
        return value
    model._image_for_row = traced_image_for_row
    try:
        with torch.inference_mode():
            outputs = model(batch)
    finally:
        model._image_for_row = original_image_for_row
        for handle in handles:
            handle.remove()
    after_hash = nested_input_hash(batch)
    correct = shifted_teacher_forced_metrics(outputs.logits, outputs.labels, ignore_index=model.IGNORE_INDEX)
    legacy = legacy_tail_metrics(outputs.logits, sample["labels"], ignore_index=model.IGNORE_INDEX)
    return {
        "input_hash_before": before_hash,
        "input_hash_after": after_hash,
        "input_mutated": before_hash != after_hash,
        "image_hashes": image_hashes,
        "captures": captures,
        "logits": outputs.logits.detach().cpu(),
        "expanded_labels": outputs.labels.detach().cpu(),
        "attention_mask": outputs.attention_mask.detach().cpu(),
        "answer_mask": outputs.answer_mask.detach().cpu(),
        "correct_metric": correct,
        "legacy_metric": legacy,
    }


def frozen_forward(model: Any, frozen: tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    inputs_embeds, labels, masks = frozen
    embeds = inputs_embeds.detach().clone()
    expanded_labels = labels.detach().clone()
    attention = masks["attention_mask"].detach().clone()
    before = nested_input_hash({"inputs_embeds": embeds, "labels": expanded_labels, "attention_mask": attention})
    with torch.inference_mode():
        outputs = model.llava_model(
            inputs_embeds=embeds,
            attention_mask=attention.long(),
            labels=expanded_labels,
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )
    after = nested_input_hash({"inputs_embeds": embeds, "labels": expanded_labels, "attention_mask": attention})
    return {
        "input_hash_before": before,
        "input_hash_after": after,
        "input_mutated": before != after,
        "logits": outputs.logits.detach().cpu(),
        "correct_metric": shifted_teacher_forced_metrics(outputs.logits, expanded_labels, ignore_index=model.IGNORE_INDEX),
    }


def serializable_forward(row: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in row.items():
        if key == "logits":
            continue
        result[key] = tensor_metadata(value) if isinstance(value, torch.Tensor) else value
    return result


def pairwise(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [compare_tensors(rows[0]["logits"], row["logits"]) for row in rows[1:]]


def main() -> None:
    args = parse_args()
    ensure_offline_env()
    set_determinism(42)
    config = EngramMultimodalHparams.from_hparams(str((ROOT / args.config).resolve()))
    config.dropout = 0.0
    config.no_grad_layers = None
    config.device = "cuda"
    model = get_model(config).to(torch.device("cuda")).eval()
    records = json.loads((ROOT / args.dataset).read_text())
    record = records[args.record_index]
    image_root = Path(config.coco_image)
    if not image_root.is_absolute():
        image_root = ROOT / image_root
    sample = make_sample(model, record, image_root)

    inventory = module_state_inventory(model)
    state_before = None
    state_before_meta = None
    if args.full_state_hash:
        state_before, state_before_meta = full_state_sha256(model)

    pixel_rows = []
    build_rows = []
    frozen_candidates = []
    for _ in range(args.repeats):
        with torch.inference_mode():
            pixels = model._image_for_row(clone_batch(sample), 0)
        pixel_rows.append(tensor_metadata(pixels))
        captures, frozen = capture_build_path(model, sample)
        inputs_embeds, labels, masks = frozen
        build_rows.append({
            "captures": captures,
            "inputs_embeds": tensor_metadata(inputs_embeds),
            "labels": tensor_metadata(labels),
            "attention_mask": tensor_metadata(masks["attention_mask"]),
            "answer_mask": tensor_metadata(masks["answer_mask"]),
        })
        frozen_candidates.append(frozen)

    raw_rows = [raw_forward(model, sample) for _ in range(args.repeats)]
    frozen = frozen_candidates[0]
    frozen_input_hash = nested_input_hash({"inputs_embeds": frozen[0], "labels": frozen[1], "masks": frozen[2]})
    frozen_rows = [frozen_forward(model, frozen) for _ in range(args.repeats)]

    state_after = None
    state_after_meta = None
    if args.full_state_hash:
        state_after, state_after_meta = full_state_sha256(model)

    payload = {
        "run_id": args.run_id,
        "record_id": str(record.get("id")),
        "backend": backend_report(),
        "model_training": model.training,
        "llava_model_training": model.llava_model.training,
        "processor": model.image_processor.to_dict() if hasattr(model.image_processor, "to_dict") else repr(model.image_processor),
        "module_inventory_summary": {
            "training_module_count": len(inventory["training_modules"]),
            "training_modules": inventory["training_modules"],
            "dropout_modules": inventory["dropout_modules"],
            "parameter_count": len(inventory["parameters"]),
            "buffer_count": len(inventory["buffers"]),
        },
        "module_inventory": inventory,
        "full_state_sha256_before": state_before,
        "full_state_sha256_after": state_after,
        "full_state_before_meta": state_before_meta,
        "full_state_after_meta": state_after_meta,
        "full_state_unchanged": state_before == state_after if state_before is not None else None,
        "sample_hash": nested_input_hash(sample),
        "pixel_rows": pixel_rows,
        "pixel_all_equal": len({row["sha256"] for row in pixel_rows}) == 1,
        "build_rows": build_rows,
        "prepared_inputs_all_equal": len({row["inputs_embeds"]["sha256"] for row in build_rows}) == 1,
        "frozen_input_hash": frozen_input_hash,
        "raw_rows": [serializable_forward(row) for row in raw_rows],
        "raw_logits_comparisons": pairwise(raw_rows),
        "frozen_rows": [serializable_forward(row) for row in frozen_rows],
        "frozen_logits_comparisons": pairwise(frozen_rows),
        "raw_metric_nll_range": max(row["correct_metric"]["target_nll"] for row in raw_rows) - min(row["correct_metric"]["target_nll"] for row in raw_rows),
        "legacy_metric_nll_range": max(row["legacy_metric"]["target_nll"] for row in raw_rows) - min(row["legacy_metric"]["target_nll"] for row in raw_rows),
        "frozen_metric_nll_range": max(row["correct_metric"]["target_nll"] for row in frozen_rows) - min(row["correct_metric"]["target_nll"] for row in frozen_rows),
        "legacy_vs_correct_nll_first": {
            "legacy": raw_rows[0]["legacy_metric"]["target_nll"],
            "correct": raw_rows[0]["correct_metric"]["target_nll"],
            "abs_diff": abs(raw_rows[0]["legacy_metric"]["target_nll"] - raw_rows[0]["correct_metric"]["target_nll"]),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "run_id": args.run_id,
        "record_id": payload["record_id"],
        "pixel_all_equal": payload["pixel_all_equal"],
        "prepared_inputs_all_equal": payload["prepared_inputs_all_equal"],
        "full_state_unchanged": payload["full_state_unchanged"],
        "raw_logits_comparisons": payload["raw_logits_comparisons"],
        "frozen_logits_comparisons": payload["frozen_logits_comparisons"],
        "raw_metric_nll_range": payload["raw_metric_nll_range"],
        "legacy_metric_nll_range": payload["legacy_metric_nll_range"],
        "frozen_metric_nll_range": payload["frozen_metric_nll_range"],
        "legacy_vs_correct_nll_first": payload["legacy_vs_correct_nll_first"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
