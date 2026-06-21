#!/usr/bin/env python3
"""LLaVA-Med DSCA decoded-generation smoke and small generation-path diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from dsca_medmkeb_diag_common import (
    answer_fields,
    append_jsonl,
    normalize_medical_answer,
    target_nll_from_outputs,
    to_jsonable,
    write_json,
)
from easyeditor.trainer.algs.dsca_utils import DSCAContext, dsca_intervention_context
from easyeditor.trainer.llava_med_models.llava_med import build_llava_med_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llava-med", choices=["llava-med"])
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--hparams", default="hparams/DSCA/llava_med.yaml")
    parser.add_argument("--training-hparams", default="hparams/TRAINING/DSCA/llava_med_stage1_smoke.yaml")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generation-use-cache", default=None, choices=["true", "false"])
    parser.add_argument("--debug-generation-hooks", action="store_true")
    parser.add_argument("--force-route-assigned-cluster", action="store_true")
    parser.add_argument("--force-route-all-active", action="store_true")
    parser.add_argument("--disable-normal-routing-for-diagnostic", action="store_true")
    parser.add_argument(
        "--residual-apply-mask",
        default=None,
        choices=["attention", "vision_prompt", "all_nonpad", "current_token"],
    )
    parser.add_argument("--dsca-generation-mode", default=None, choices=["normal", "prefill_only", "cache_reuse_route"])
    parser.add_argument("--require-all-samples-active-residual", action="store_true")
    return parser.parse_args()


def ensure_env() -> None:
    os.environ.setdefault("HF_HOME", "/remote-home/wangbomin/hugging_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def device_arg(text: str) -> Any:
    if text == "cuda":
        return "cuda"
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        return int(suffix) if suffix.isdigit() else text
    return int(text) if text.isdigit() else text


def torch_device(value: Any) -> torch.device:
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    return torch.device(str(value))


def as_bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_dataset_path(dataset: str) -> Path:
    root = Path.cwd()
    candidates = [
        root / "datasets" / "MedMKEB" / "eval.json",
        root / "datasets" / "MEDMKEB" / "eval.json",
        root / "datasets" / "medmkeb" / "eval.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("MedMKEB eval.json not found; checked " + ", ".join(str(item) for item in candidates))


def load_records(dataset: str, limit: int) -> Tuple[Path, List[Dict[str, Any]]]:
    path = resolve_dataset_path(dataset)
    records = json.loads(path.read_text(errors="replace"))
    if not isinstance(records, list):
        raise RuntimeError(f"Dataset JSON root must be a list: {path}")
    records = [row for row in records if isinstance(row, dict) and row.get("alt") and row.get("image")]
    return path, records[:limit]


def resolve_image_path(image_root: Path, record_value: str) -> Path:
    value = Path(record_value)
    if value.is_absolute():
        return value
    image_root = image_root.resolve()
    if image_root.name == "images" and str(record_value).startswith("images/"):
        return image_root.parent / value
    return image_root / value


def prompt_for(record: Dict[str, Any]) -> str:
    return "Question: {} Short answer: ".format(record.get("src", ""))


def target_for(record: Dict[str, Any]) -> str:
    return str(record.get("alt", ""))


def clone_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.clone()
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def make_sample(model: Any, record: Dict[str, Any], image_root: Path, target: Optional[str] = None) -> Dict[str, Any]:
    prompt = prompt_for(record)
    target_text = target_for(record) if target is None else str(target)
    tokenizer = model.llava_tokenizer
    labels = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return {
        "image_path": [str(resolve_image_path(image_root, str(record["image"])))],
        "prompt": [prompt],
        "target": [target_text],
        "text_input": [prompt + target_text],
        "labels": labels,
        "prompts_len": [len(tokenizer(prompt, add_special_tokens=False).input_ids)],
    }


def load_model_alg(args: argparse.Namespace):
    ensure_env()
    from easyeditor.trainer.algs.dsca import DSCA
    from easyeditor.trainer.models import get_model
    from easyeditor.trainer.training_hparams.dsca_multimodal_training_hparams import DSCAMultimodalTrainingHparams

    config = DSCAMultimodalTrainingHparams.from_hparams(args.training_hparams or args.hparams)
    config.device = device_arg(args.device)
    config.coco_image = str(args.image_root)
    config.rephrase_image = str(args.image_root)
    device = torch_device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    model = get_model(config).to(device).eval()
    alg = DSCA(model, config, lambda: None).to(device).eval()
    return model, alg, config


def resolve_generation_policy(args: argparse.Namespace, config: Any) -> Tuple[bool, str, Optional[str], bool]:
    use_cache = as_bool_text(args.generation_use_cache) if args.generation_use_cache is not None else True
    generation_mode = args.dsca_generation_mode or str(getattr(config, "dsca_generation_mode", "normal"))
    residual_apply_mask = args.residual_apply_mask
    if residual_apply_mask is None:
        residual_apply_mask = getattr(config, "dsca_generation_residual_apply_mask", None)
    if residual_apply_mask in {"", "none", "None"}:
        residual_apply_mask = None
    reuse_prefill_route = bool(getattr(config, "dsca_generation_reuse_prefill_route", generation_mode == "cache_reuse_route"))
    if generation_mode == "cache_reuse_route":
        reuse_prefill_route = True if reuse_prefill_route is None else bool(reuse_prefill_route)
    return use_cache, generation_mode, residual_apply_mask, reuse_prefill_route


def make_basis(
    rank: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed_vector: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    basis = torch.zeros(rank, hidden_size, device=device, dtype=dtype)
    start = 0
    if seed_vector is not None:
        seed = seed_vector.detach().flatten().to(device=device, dtype=torch.float32)
        seed = seed / seed.norm().clamp_min(1.0e-8)
        if torch.isfinite(seed).all() and seed.numel() == hidden_size:
            basis[0] = seed.to(dtype)
            start = 1
    for idx in range(start, rank):
        basis[idx, idx % hidden_size] = 1
    return basis


def create_cluster_from_sample(alg: Any, sample: Dict[str, Any], active: bool) -> int:
    reps = alg.capture_representations(clone_sample(sample))
    cid = alg.repository.create_cluster(reps["h_f"][0], reps["h_v"][0], metadata={"source": "generation_gate"})
    if active:
        basis = make_basis(
            alg.rank,
            alg.hidden_size,
            alg.repository.p_f.device,
            alg.repository.p_f.dtype,
            seed_vector=reps["h_f"][0],
        )
        dsam = alg.repository.dsams[cid]
        dsam.set_basis(basis)
        with torch.no_grad():
            dsam.W.copy_(dsam.R)
            dsam.b.fill_(1.0e-3)
            dsam.gate_down.weight.zero_()
            dsam.gate_down.bias.zero_()
            dsam.gate_up.weight.zero_()
            dsam.gate_up.bias.fill_(2.0)
        alg.repository.active[cid] = True
    return cid


def temporary_dsam_residual_scale(alg: Any, value: float):
    class _Ctx:
        def __enter__(self_inner):
            self_inner.old_values = [float(dsam.residual_scale) for dsam in alg.repository.dsams]
            for dsam in alg.repository.dsams:
                dsam.residual_scale = float(value)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            for dsam, old in zip(alg.repository.dsams, self_inner.old_values):
                dsam.residual_scale = old
            return False

    return _Ctx()


def prepare_generation_inputs(model: Any, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    prompt = sample["prompt"][0]
    prompt_text = model._conversation_prompt(prompt, None)
    input_ids = model.tokenizer_image_token(prompt_text, model.llava_tokenizer, model.IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model.lm_device)
    attention = torch.ones_like(input_ids, dtype=torch.long, device=model.lm_device)
    labels = torch.full_like(input_ids, model.IGNORE_INDEX)
    image_tensor = model._image_for_row(sample, 0)
    (
        _,
        _position_ids,
        expanded_attention,
        _,
        inputs_embeds,
        expanded_labels,
    ) = model.llava_model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention,
        past_key_values=None,
        labels=labels,
        images=image_tensor,
    )
    image_feature_len = int(inputs_embeds.shape[1] - (input_ids.shape[1] - 1))
    masks = build_llava_med_masks(
        token_ids=input_ids[0],
        labels=expanded_labels[0],
        expanded_attention_mask=expanded_attention[0],
        image_token_index=model.IMAGE_TOKEN_INDEX,
        image_feature_len=image_feature_len,
    )
    masks = {name: value.unsqueeze(0) for name, value in masks.items()}
    return input_ids, image_tensor, masks


def residual_norm(alg: Any) -> float:
    residual = getattr(alg, "_last_residual", None)
    if residual is None:
        return 0.0
    return float(residual.detach().float().norm().cpu())


def candidate_count(alg: Any) -> int:
    weights = getattr(alg, "_last_route_weights", None)
    if weights is None or not torch.is_tensor(weights) or weights.numel() == 0:
        return 0
    return int(torch.count_nonzero(weights[0] > 0).item())


def route_ids_from_events(events: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[int], List[float]]:
    candidate_ids: List[int] = []
    active_ids: List[int] = []
    route_weights: List[float] = []
    for event in events:
        ids = event.get("candidate_ids") or []
        active = event.get("active_candidate_ids") or []
        weights = event.get("route_weights") or []
        if ids:
            candidate_ids = [int(item) for item in ids]
        if active:
            active_ids = [int(item) for item in active]
        if weights:
            route_weights = [float(item) for item in weights]
    return candidate_ids, active_ids, route_weights


def hook_event_summary(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_ids, active_ids, route_weights = route_ids_from_events(events)
    residuals = [float(item.get("residual_norm") or 0.0) for item in events if item.get("residual_norm") is not None]
    prefill_events = [item for item in events if item.get("phase") == "prefill"]
    cached_events = [item for item in events if item.get("phase") == "cached_decode"]
    prefill_ids, prefill_active_ids, prefill_weights = route_ids_from_events(prefill_events)
    cached_residuals = [float(item.get("residual_norm") or 0.0) for item in cached_events if item.get("residual_norm") is not None]
    cached_apply_sums = [
        int(item.get("apply_mask_sum")) for item in cached_events if item.get("apply_mask_sum") is not None
    ]
    residual_nonzero_by_step = [value > 1.0e-8 for value in cached_residuals]
    hook_error_count = sum(1 for item in events if item.get("error"))
    nonfinite_residual_count = sum(1 for value in residuals if not math.isfinite(value))
    return {
        "hook_event_count": len(events),
        "prefill_hook_event_count": len(prefill_events),
        "cached_decode_hook_event_count": len(cached_events),
        "hook_entered": any(bool(item.get("hook_entered")) for item in events),
        "selected_candidate_ids": candidate_ids,
        "active_candidate_ids": active_ids,
        "normal_candidate_count": len(candidate_ids),
        "active_route_selected": bool(active_ids),
        "route_weights": route_weights,
        "max_event_residual_norm": max(residuals) if residuals else 0.0,
        "prefill_route_ids": prefill_ids,
        "prefill_candidate_ids": prefill_ids,
        "prefill_active_candidate_ids": prefill_active_ids,
        "prefill_route_weights": prefill_weights,
        "cached_decode_route_reused": any(bool(item.get("cached_decode_route_reused")) for item in cached_events),
        "cached_decode_residual_norm_by_step": cached_residuals,
        "current_token_apply_mask_sum": cached_apply_sums,
        "residual_nonzero_by_step": residual_nonzero_by_step,
        "repository_update_enabled": any(bool(item.get("repository_update_enabled")) for item in events),
        "hook_error_count": hook_error_count,
        "nonfinite_event_residual_count": nonfinite_residual_count,
        "apply_mask_all_false_active_route": any(bool(item.get("apply_mask_all_false_active_route")) for item in events),
    }


def force_route_ids_for(alg: Any, assigned_cluster_id: Optional[int], args: argparse.Namespace) -> Optional[List[int]]:
    if args.force_route_all_active:
        return [idx for idx in range(len(alg.repository)) if bool(alg.repository.active[idx].item())]
    if args.force_route_assigned_cluster and assigned_cluster_id is not None:
        return [int(assigned_cluster_id)]
    return None


def reset_generation_debug_state(alg: Any) -> None:
    alg._last_residual = None
    alg._last_route_weights = None
    alg._last_route_selected = None
    alg._last_apply_mask = None
    alg._cached_generation_route_weights = None
    alg._cached_generation_route_selected = None
    alg._generation_hook_call_index = 0


def generate_text(
    model: Any,
    sample: Dict[str, Any],
    max_new_tokens: int,
    alg: Optional[Any] = None,
    scale0: bool = False,
    sample_id: Optional[int] = None,
    call_label: Optional[str] = None,
    use_cache: bool = True,
    debug_generation_hooks: bool = False,
    event_path: Optional[Path] = None,
    force_route_ids: Optional[List[int]] = None,
    disable_normal_routing: bool = False,
    residual_apply_mask_mode: Optional[str] = None,
    generation_mode: str = "normal",
    generation_reuse_prefill_route: Optional[bool] = None,
) -> Dict[str, Any]:
    input_ids, image_tensor, masks = prepare_generation_inputs(model, sample)
    events: List[Dict[str, Any]] = []
    if alg is not None:
        reset_generation_debug_state(alg)
    managers = []
    if alg is not None:
        managers.append(
            dsca_intervention_context(
                alg,
                DSCAContext(
                    batch=masks,
                    is_generation=True,
                    debug_events=events,
                    sample_id=sample_id,
                    call_label=call_label,
                    generation_mode=generation_mode,
                    generation_reuse_prefill_route=generation_reuse_prefill_route,
                    generation_use_cache=use_cache,
                    force_route_ids=force_route_ids,
                    disable_normal_routing=disable_normal_routing,
                    residual_apply_mask_mode=residual_apply_mask_mode,
                    extend_generation_masks=(not use_cache) or generation_mode == "cache_reuse_route",
                ),
            )
        )
    if alg is not None and scale0:
        managers.append(temporary_dsam_residual_scale(alg, 0.0))
    with ExitStack() as stack:
        for manager in managers:
            stack.enter_context(manager)
        with torch.inference_mode():
            output_ids = model.llava_model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                pad_token_id=model.llava_tokenizer.pad_token_id,
                eos_token_id=model.llava_tokenizer.eos_token_id,
            )
    if event_path is not None and events:
        for event in events:
            append_jsonl(event_path, event)
    summary = hook_event_summary(events)
    input_len = int(input_ids.shape[1])
    sequence_includes_prompt = bool(
        output_ids.shape[1] >= input_len and torch.equal(output_ids[0, :input_len].detach().cpu(), input_ids[0].detach().cpu())
    )
    generated_ids = output_ids[:, input_len:] if sequence_includes_prompt else output_ids
    full_text = model.llava_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    text = model.llava_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return {
        "text": text,
        "full_text": full_text,
        "input_ids_len": input_len,
        "sequence_includes_prompt": sequence_includes_prompt,
        "generated_ids": generated_ids.detach().cpu().tolist(),
        "residual_norm": residual_norm(alg) if alg is not None else 0.0,
        "candidate_count": candidate_count(alg) if alg is not None else 0,
        "active_dsam_count": alg.repository.num_active() if alg is not None else 0,
        "masks": masks,
        "events": events,
        **summary,
    }


def tensor_delta_norm(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    seq = min(lhs.shape[1], rhs.shape[1])
    return float((lhs[:, :seq].detach().float() - rhs[:, :seq].detach().float()).norm().cpu())


def forward_compare(model: Any, alg: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
    with torch.no_grad():
        base = model(clone_sample(sample))
        edited = alg(clone_sample(sample))
    masks = getattr(alg, "_last_masks", {}) or {}
    return {
        "active_hidden_delta_norm": residual_norm(alg),
        "active_logits_delta_norm": tensor_delta_norm(base.logits, edited.logits),
        "masks_align": all(
            torch.is_tensor(value) and tuple(value.shape) == tuple(edited.logits.shape[:2])
            for value in masks.values()
        ),
        "vision_mask_sum": int(masks.get("vision_mask", torch.zeros(1)).sum().detach().cpu()) if masks else 0,
        "prompt_mask_sum": int(masks.get("prompt_mask", torch.zeros(1)).sum().detach().cpu()) if masks else 0,
        "answer_mask_sum": int(masks.get("answer_mask", torch.zeros(1)).sum().detach().cpu()) if masks else 0,
        "attention_mask_sum": int(masks.get("attention_mask", torch.zeros(1)).sum().detach().cpu()) if masks else 0,
        "base_target_nll": target_nll_from_outputs(base, sample)["target_nll"],
        "edited_target_nll": target_nll_from_outputs(edited, sample)["target_nll"],
    }


def summarize_smoke_rows(dataset_path: Path, rows: Sequence[Dict[str, Any]], require_all_samples_active_residual: bool = False) -> Dict[str, Any]:
    active_route_rows = [row for row in rows if row["active_route_selected"]]
    no_route_rows = [row for row in rows if not row["active_route_selected"]]
    active_route_nonzero = [row for row in active_route_rows if row["residual_nonzero"]]
    summary = {
        "dataset_path": str(dataset_path),
        "num_samples": len(rows),
        "empty_repository_identity": all(row["empty_equals_base"] and row["empty_repo_residual_norm"] == 0.0 for row in rows),
        "inactive_dsam_identity": all(row["inactive_equals_base"] and row["inactive_residual_norm"] == 0.0 for row in rows),
        "residual_scale0_identity": all(row["scale0_equals_base"] and row["scale0_residual_norm"] == 0.0 for row in rows),
        "all_identity_checks_pass": all(
            row["empty_equals_base"]
            and row["empty_repo_residual_norm"] == 0.0
            and row["inactive_equals_base"]
            and row["inactive_residual_norm"] == 0.0
            and row["scale0_equals_base"]
            and row["scale0_residual_norm"] == 0.0
            for row in rows
        ),
        "active_route_case_count": len(active_route_rows),
        "active_route_nonzero_residual_count": len(active_route_nonzero),
        "active_route_nonzero_residual_rate": len(active_route_nonzero) / len(active_route_rows) if active_route_rows else None,
        "no_route_case_count": len(no_route_rows),
        "no_route_identity_or_zero_residual_count": sum(
            1 for row in no_route_rows if row["generation_residual_norm_mean"] == 0.0 or row["active_dsam_text"] == row["base_text"]
        ),
        "active_routed_dsam_residual_nonzero": (
            True if not active_route_rows else all(row["residual_nonzero"] for row in active_route_rows)
        ),
        "active_hidden_delta_nonzero": (
            True if not active_route_rows else all(row["active_hidden_delta_norm"] > 1.0e-8 for row in active_route_rows)
        ),
        "active_logits_delta_nonzero": (
            True if not active_route_rows else all(row["logits_delta_nonzero"] for row in active_route_rows)
        ),
        "masks_align": all(row["masks_align"] for row in rows),
        "generation_hook_active": any(row["hook_entered"] for row in rows),
        "hook_error_count": sum(int(row["hook_error_count"]) for row in rows),
        "no_hook_errors": all(int(row["hook_error_count"]) == 0 for row in rows),
        "nonfinite_event_residual_count": sum(int(row["nonfinite_event_residual_count"]) for row in rows),
        "nonfinite_logits_delta_count": sum(
            1 for row in rows if not math.isfinite(float(row["active_logits_delta_norm"]))
        ),
        "all_generation_values_finite": all(
            math.isfinite(float(row["generation_residual_norm_mean"]))
            and math.isfinite(float(row["active_hidden_delta_norm"]))
            and math.isfinite(float(row["active_logits_delta_norm"]))
            and int(row["nonfinite_event_residual_count"]) == 0
            for row in rows
        ),
        "no_active_routed_samples": len(active_route_rows) == 0,
        "require_all_samples_active_residual": bool(require_all_samples_active_residual),
        "repository_unchanged_during_generation": all(row["repository_unchanged_during_generation"] for row in rows),
    }
    if require_all_samples_active_residual:
        summary["active_routed_dsam_residual_nonzero"] = all(row["residual_nonzero"] for row in rows)
        summary["active_hidden_delta_nonzero"] = all(row["active_hidden_delta_norm"] > 1.0e-8 for row in rows)
        summary["active_logits_delta_nonzero"] = all(row["logits_delta_nonzero"] for row in rows)
    return summary


def first_token_rank(model: Any, sample: Dict[str, Any], token_id: int, alg: Optional[Any] = None) -> int:
    prompt_only = clone_sample(sample)
    prompt_only["target"] = [""]
    prompt_only["text_input"] = [prompt_only["prompt"][0]]
    with torch.no_grad():
        outputs = alg(prompt_only) if alg is not None else model(prompt_only)
    logits = outputs.logits[0, -1].float()
    order = torch.argsort(logits, descending=True)
    found = (order == int(token_id)).nonzero(as_tuple=False)
    return int(found[0].item() + 1) if found.numel() else int(logits.numel() + 1)


def smoke_rows(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dataset_path, records = load_records(args.dataset, args.num_samples)
    model, alg, config = load_model_alg(args)
    rows: List[Dict[str, Any]] = []
    debug_path = args.output_dir / "dsca_generation_smoke_debug.jsonl"
    event_path = args.output_dir / "generation_hook_events.jsonl"
    if debug_path.exists():
        debug_path.unlink()
    if event_path.exists():
        event_path.unlink()
    use_cache, generation_mode, residual_apply_mask, reuse_prefill_route = resolve_generation_policy(args, config)

    for idx, record in enumerate(records):
        sample = make_sample(model, record, args.image_root)
        base = generate_text(model, sample, args.max_new_tokens, sample_id=idx, call_label="base", use_cache=use_cache)

        alg.repository.clear()
        empty = generate_text(
            model,
            sample,
            args.max_new_tokens,
            alg=alg,
            sample_id=idx,
            call_label="empty_repo",
            use_cache=use_cache,
            debug_generation_hooks=args.debug_generation_hooks,
            event_path=event_path,
            residual_apply_mask_mode=residual_apply_mask,
            generation_mode=generation_mode,
            generation_reuse_prefill_route=reuse_prefill_route,
        )
        empty_clusters = len(alg.repository)

        alg.repository.clear()
        create_cluster_from_sample(alg, sample, active=False)
        inactive = generate_text(
            model,
            sample,
            args.max_new_tokens,
            alg=alg,
            sample_id=idx,
            call_label="inactive_dsam",
            use_cache=use_cache,
            debug_generation_hooks=args.debug_generation_hooks,
            event_path=event_path,
            residual_apply_mask_mode=residual_apply_mask,
            generation_mode=generation_mode,
            generation_reuse_prefill_route=reuse_prefill_route,
        )

        alg.repository.clear()
        assigned_cluster_id = create_cluster_from_sample(alg, sample, active=True)
        repo_before = len(alg.repository)
        p_f_before = alg.repository.p_f.detach().clone()
        forced_ids = force_route_ids_for(alg, assigned_cluster_id, args)
        active = generate_text(
            model,
            sample,
            args.max_new_tokens,
            alg=alg,
            sample_id=idx,
            call_label="active_dsam",
            use_cache=use_cache,
            debug_generation_hooks=args.debug_generation_hooks,
            event_path=event_path,
            force_route_ids=forced_ids,
            disable_normal_routing=args.disable_normal_routing_for_diagnostic,
            residual_apply_mask_mode=residual_apply_mask,
            generation_mode=generation_mode,
            generation_reuse_prefill_route=reuse_prefill_route,
        )
        compare = forward_compare(model, alg, sample)
        repo_after = len(alg.repository)
        prototypes_unchanged = bool(torch.allclose(p_f_before, alg.repository.p_f.detach()))
        scale0 = generate_text(
            model,
            sample,
            args.max_new_tokens,
            alg=alg,
            scale0=True,
            sample_id=idx,
            call_label="scale0",
            use_cache=use_cache,
            debug_generation_hooks=args.debug_generation_hooks,
            event_path=event_path,
            force_route_ids=forced_ids,
            disable_normal_routing=args.disable_normal_routing_for_diagnostic,
            residual_apply_mask_mode=residual_apply_mask,
            generation_mode=generation_mode,
            generation_reuse_prefill_route=reuse_prefill_route,
        )
        residual_nonzero = active["residual_norm"] > 1.0e-8 or compare["active_hidden_delta_norm"] > 1.0e-8
        logits_delta_nonzero = compare["active_logits_delta_norm"] > 1.0e-8

        row = {
            "sample_id": idx,
            "record_id": record.get("id", idx),
            "target": target_for(record),
            "generation_use_cache": use_cache,
            "dsca_generation_mode": generation_mode,
            "residual_apply_mask_mode": residual_apply_mask or getattr(alg, "residual_apply_mask", "attention"),
            "generation_reuse_prefill_route": reuse_prefill_route,
            "force_route_ids": forced_ids or [],
            "disable_normal_routing": args.disable_normal_routing_for_diagnostic,
            "base_text": base["text"],
            "empty_repo_text": empty["text"],
            "inactive_dsam_text": inactive["text"],
            "active_dsam_text": active["text"],
            "scale0_text": scale0["text"],
            "empty_equals_base": empty["text"] == base["text"],
            "inactive_equals_base": inactive["text"] == base["text"],
            "scale0_equals_base": scale0["text"] == base["text"],
            "active_hidden_delta_norm": compare["active_hidden_delta_norm"],
            "active_logits_delta_norm": compare["active_logits_delta_norm"],
            "generation_residual_norm_mean": active["residual_norm"],
            "candidate_count_mean": active["candidate_count"],
            "active_dsam_count_mean": active["active_dsam_count"],
            "normal_candidate_count": active["normal_candidate_count"],
            "selected_candidate_ids": active["selected_candidate_ids"],
            "active_candidate_ids": active["active_candidate_ids"],
            "active_route_selected": active["active_route_selected"],
            "residual_nonzero": residual_nonzero,
            "logits_delta_nonzero": logits_delta_nonzero,
            "hook_entered": active["hook_entered"],
            "hook_event_count": active["hook_event_count"],
            "prefill_hook_event_count": active["prefill_hook_event_count"],
            "cached_decode_hook_event_count": active["cached_decode_hook_event_count"],
            "max_event_residual_norm": active["max_event_residual_norm"],
            "prefill_route_ids": active["prefill_route_ids"],
            "prefill_candidate_ids": active["prefill_candidate_ids"],
            "prefill_active_candidate_ids": active["prefill_active_candidate_ids"],
            "prefill_route_weights": active["prefill_route_weights"],
            "cached_decode_route_reused": active["cached_decode_route_reused"],
            "cached_decode_residual_norm_by_step": active["cached_decode_residual_norm_by_step"],
            "current_token_apply_mask_sum": active["current_token_apply_mask_sum"],
            "residual_nonzero_by_step": active["residual_nonzero_by_step"],
            "logits_delta_norm_by_step": [compare["active_logits_delta_norm"]] if active["hook_entered"] else [],
            "repository_update_enabled": active["repository_update_enabled"],
            "hook_error_count": active["hook_error_count"],
            "nonfinite_event_residual_count": active["nonfinite_event_residual_count"],
            "apply_mask_all_false_active_route": active["apply_mask_all_false_active_route"],
            "masks_align": compare["masks_align"],
            "vision_mask_sum": compare["vision_mask_sum"],
            "prompt_mask_sum": compare["prompt_mask_sum"],
            "answer_mask_sum": compare["answer_mask_sum"],
            "attention_mask_sum": compare["attention_mask_sum"],
            "empty_repo_residual_norm": empty["residual_norm"],
            "inactive_residual_norm": inactive["residual_norm"],
            "scale0_residual_norm": scale0["residual_norm"],
            "empty_repo_cluster_count": empty_clusters,
            "repo_count_before_generation": repo_before,
            "repo_count_after_generation": repo_after,
            "repository_unchanged_during_generation": repo_before == repo_after and prototypes_unchanged,
        }
        rows.append(row)
        append_jsonl(debug_path, row)

    summary = summarize_smoke_rows(dataset_path, rows, args.require_all_samples_active_residual)
    return summary, rows


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows([{k: to_jsonable(v) for k, v in row.items()} for row in rows])


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = smoke_rows(args)
    write_json(args.output_dir / "dsca_generation_smoke_summary.json", summary)
    write_rows(args.output_dir / "dsca_generation_smoke_per_sample.csv", rows)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    failures = []
    for key in (
        "empty_repository_identity",
        "inactive_dsam_identity",
        "residual_scale0_identity",
        "masks_align",
        "generation_hook_active",
        "repository_unchanged_during_generation",
        "no_hook_errors",
        "all_generation_values_finite",
    ):
        if not summary.get(key):
            failures.append(key)
    if int(summary.get("active_route_case_count") or 0) <= 0:
        failures.append("active_route_case_count")
    if summary.get("active_route_nonzero_residual_rate") != 1.0:
        failures.append("active_route_nonzero_residual_rate")
    if failures:
        raise RuntimeError("LLaVA-Med DSCA generation smoke failed: " + ", ".join(failures))
    return summary


def train_small_repository(alg: Any, model: Any, records: Sequence[Dict[str, Any]], image_root: Path, lr: float) -> None:
    alg.set_editor_train(True)
    optimizer = torch.optim.Adam(alg.outer_parameters(), lr=lr)
    for record in records:
        sample = make_sample(model, record, image_root)
        batch = {"edit_inner": sample, "loc_image": clone_sample(sample), "loc": clone_sample(sample)}
        optimizer.zero_grad(set_to_none=True)
        alg.edit_step(batch, training=True, optimizer=optimizer)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    alg.set_editor_train(False)


def run_llava_med_generation_path_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path, records = load_records(args.dataset, args.num_samples)
    model, alg, config = load_model_alg(args)
    use_cache, generation_mode, residual_apply_mask, reuse_prefill_route = resolve_generation_policy(args, config)
    train_small_repository(alg, model, records, args.image_root, float(getattr(config, "lr", 1.0e-4)))
    rows: List[Dict[str, Any]] = []
    debug_path = args.output_dir / "generation_path_debug.jsonl"
    event_path = args.output_dir / "generation_hook_events.jsonl"
    if debug_path.exists():
        debug_path.unlink()
    if event_path.exists():
        event_path.unlink()

    for idx, record in enumerate(records):
        sample = make_sample(model, record, args.image_root)
        target_ids = sample["labels"][0].detach().cpu().tolist()
        first_target = target_ids[0] if target_ids else None
        base = generate_text(model, sample, getattr(args, "max_new_tokens", 8), use_cache=use_cache)
        edited = generate_text(
            model,
            sample,
            getattr(args, "max_new_tokens", 8),
            alg=alg,
            sample_id=idx,
            call_label="generation_path_edited",
            use_cache=use_cache,
            debug_generation_hooks=True,
            event_path=event_path,
            residual_apply_mask_mode=residual_apply_mask,
            generation_mode=generation_mode,
            generation_reuse_prefill_route=reuse_prefill_route,
        )
        base_outputs = model(clone_sample(sample))
        edited_outputs = alg(clone_sample(sample))
        compare = forward_compare(model, alg, sample)
        base_nll = target_nll_from_outputs(base_outputs, sample)["target_nll"]
        edited_nll = target_nll_from_outputs(edited_outputs, sample)["target_nll"]
        base_rank = first_token_rank(model, sample, first_target, alg=None) if first_target is not None else None
        edited_rank = first_token_rank(model, sample, first_target, alg=alg) if first_target is not None else None
        fields = answer_fields(base["text"], edited["text"], target_for(record))
        residual = getattr(alg, "_last_residual", None)
        masks = getattr(alg, "_last_masks", {}) or {}
        answer_norm = prompt_norm = vision_norm = 0.0
        if residual is not None and masks:
            for name, out_name in (("answer_mask", "answer"), ("prompt_mask", "prompt"), ("vision_mask", "vision")):
                mask = masks.get(name)
                if torch.is_tensor(mask) and mask.any():
                    value = float(residual.detach().float().norm(dim=-1).masked_select(mask.bool()).mean().cpu())
                else:
                    value = 0.0
                if out_name == "answer":
                    answer_norm = value
                elif out_name == "prompt":
                    prompt_norm = value
                else:
                    vision_norm = value
        row = {
            "sample_id": idx,
            "record_id": record.get("id", idx),
            "target": target_for(record),
            "base_free_text": base["text"],
            "edited_free_text": edited["text"],
            "edited_equals_base": fields["edited_equals_base"],
            "edited_contains_target": fields["edited_contains_target"],
            "base_target_nll": base_nll,
            "edited_target_nll": edited_nll,
            "delta_nll": edited_nll - base_nll,
            "first_target_base_rank": base_rank,
            "first_target_edited_rank": edited_rank,
            "generation_residual_norm_mean": edited["residual_norm"],
            "generation_candidate_count": edited["candidate_count"],
            "active_dsam_count": edited["active_dsam_count"],
            "hook_entered": edited["hook_entered"],
            "hook_event_count": edited["hook_event_count"],
            "prefill_hook_event_count": edited["prefill_hook_event_count"],
            "cached_decode_hook_event_count": edited["cached_decode_hook_event_count"],
            "prefill_candidate_ids": edited["prefill_candidate_ids"],
            "prefill_active_candidate_ids": edited["prefill_active_candidate_ids"],
            "cached_decode_route_reused": edited["cached_decode_route_reused"],
            "cached_decode_residual_norm_by_step": edited["cached_decode_residual_norm_by_step"],
            "current_token_apply_mask_sum": edited["current_token_apply_mask_sum"],
            "residual_nonzero_by_step": edited["residual_nonzero_by_step"],
            "repository_update_enabled": edited["repository_update_enabled"],
            "hook_error_count": edited["hook_error_count"],
            "nonfinite_event_residual_count": edited["nonfinite_event_residual_count"],
            "teacher_forced_answer_residual_norm": answer_norm,
            "teacher_forced_prompt_residual_norm": prompt_norm,
            "teacher_forced_vision_residual_norm": vision_norm,
            "assigned_cluster_in_candidates": edited["active_route_selected"],
            "active_dsam_available": edited["active_dsam_count"] > 0,
            "masks_align": compare["masks_align"],
            "active_logits_delta_norm": compare["active_logits_delta_norm"],
        }
        rows.append(row)
        append_jsonl(debug_path, row)

    summary = {
        "dataset_path": str(dataset_path),
        "num_rows": len(rows),
        "generation_path_uses_dsca_hook": any(row["hook_entered"] for row in rows),
        "edited_equals_base_rate": sum(1 for row in rows if row["edited_equals_base"]) / len(rows) if rows else None,
        "generation_residual_zero_rate": sum(1 for row in rows if row["generation_residual_norm_mean"] <= 1.0e-8) / len(rows) if rows else None,
        "teacher_forced_nll_improved_count": sum(1 for row in rows if row["delta_nll"] < 0.0),
        "prompt_only_first_token_rank_improved_count": sum(
            1
            for row in rows
            if row["first_target_base_rank"] is not None
            and row["first_target_edited_rank"] is not None
            and row["first_target_edited_rank"] < row["first_target_base_rank"]
        ),
        "force_route_improved_count": 0,
        "answer_residual_dominance_rate": sum(
            1
            for row in rows
            if row["teacher_forced_answer_residual_norm"]
            > 2.0 * max(row["teacher_forced_prompt_residual_norm"], row["teacher_forced_vision_residual_norm"], 1.0e-8)
        )
        / len(rows)
        if rows
        else None,
        "assigned_cluster_routed_rate": sum(1 for row in rows if row["assigned_cluster_in_candidates"]) / len(rows) if rows else None,
        "active_dsam_available_rate": sum(1 for row in rows if row["active_dsam_available"]) / len(rows) if rows else None,
        "edited_free_generation_contains_target_count": sum(1 for row in rows if row["edited_contains_target"]),
        "mean_base_target_nll": sum(row["base_target_nll"] for row in rows) / len(rows) if rows else None,
        "mean_edited_target_nll": sum(row["edited_target_nll"] for row in rows) / len(rows) if rows else None,
        "mask_alignment_failures": sum(1 for row in rows if not row["masks_align"]),
        "hook_error_count": sum(int(row["hook_error_count"]) for row in rows),
        "no_hook_errors": all(int(row["hook_error_count"]) == 0 for row in rows),
        "nonfinite_event_residual_count": sum(int(row["nonfinite_event_residual_count"]) for row in rows),
        "nonempty_generation_count": sum(1 for row in rows if row["base_free_text"] and row["edited_free_text"]),
    }
    write_json(args.output_dir / "generation_path_summary.json", summary)
    write_rows(args.output_dir / "generation_path_per_sample.csv", rows)
    report = [
        "# LLaVA-Med DSCA Generation Path Diagnostic",
        "",
        f"- output directory: `{args.output_dir.resolve()}`",
        f"- generation path uses DSCA hook: {summary['generation_path_uses_dsca_hook']}",
        f"- edited free generation equals base rate: {summary['edited_equals_base_rate']}",
        f"- generation residual zero rate: {summary['generation_residual_zero_rate']}",
        f"- teacher-forced NLL improved count: {summary['teacher_forced_nll_improved_count']}",
        f"- prompt-only first target rank improved count: {summary['prompt_only_first_token_rank_improved_count']}",
        f"- edited free generation contains target count: {summary['edited_free_generation_contains_target_count']}",
        f"- answer-residual dominance rate: {summary['answer_residual_dominance_rate']}",
        f"- assigned cluster routed rate: {summary['assigned_cluster_routed_rate']}",
        f"- active DSAM available rate: {summary['active_dsam_available_rate']}",
        f"- mean base target NLL: {summary['mean_base_target_nll']}",
        f"- mean edited target NLL: {summary['mean_edited_target_nll']}",
    ]
    (args.output_dir / "generation_path_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    failures = []
    if not summary["generation_path_uses_dsca_hook"]:
        failures.append("generation hook inactive")
    if summary["generation_residual_zero_rate"] == 1:
        failures.append("all generation residuals are zero")
    if summary["mask_alignment_failures"]:
        failures.append("mask alignment failures")
    if not summary["no_hook_errors"]:
        failures.append("hook errors")
    if summary["nonempty_generation_count"] != len(rows):
        failures.append("empty generation output")
    if summary["answer_residual_dominance_rate"] and summary["answer_residual_dominance_rate"] > 0:
        failures.append("answer-token leakage")
    if failures:
        raise RuntimeError("LLaVA-Med generation-path diagnostic failed: " + "; ".join(failures))
    return summary


def main() -> None:
    run_smoke(parse_args())


if __name__ == "__main__":
    main()
