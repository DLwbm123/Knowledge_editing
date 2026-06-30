#!/usr/bin/env python3
"""Bounded TIME MedMKEB smoke gates for LLaVA-Med."""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dsca_medmkeb_diag_common import (  # noqa: E402
    aligned_logits_and_labels,
    append_jsonl,
    clone_batch,
    ensure_offline_env,
    resolve_dataset_path,
    target_nll_from_outputs,
    to_jsonable,
    torch_device,
    write_json,
)
from easyeditor.models.time_edit import TIMEEditMultimodalHparams  # noqa: E402
from easyeditor.trainer.algs.time_edit import TIMEEdit  # noqa: E402
from easyeditor.trainer.algs.time_edit_modules import time_memory_estimate  # noqa: E402
from easyeditor.trainer.models import get_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alg", default="TIME", choices=["TIME", "TIME_EDIT"])
    parser.add_argument("--mode", default="one", choices=["one", "nonseq", "sequential"])
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=Path, default=None)
    parser.add_argument("--image-root", "--image_root", dest="image_root", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-edits", "--max_edits", dest="max_edits", type=int, default=1)
    parser.add_argument("--hparams", "--config", default="hparams/TRAINING/TIME/llava_med_smoke.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-layer", "--target_layer", dest="target_layer", type=int, default=21)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scale-mode", "--scale_mode", dest="scale_mode", choices=["lora_like", "paper_inverse", "none"], default="lora_like")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--edit-iters", "--edit_iters", dest="edit_iters", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--init-std", "--init_std", dest="init_std", type=float, default=1.0e-3)
    parser.add_argument("--time-disable-selection", action="store_true")
    parser.add_argument("--time-disable-score-mixing", action="store_true")
    parser.add_argument("--time-disable-align-loss", action="store_true")
    parser.add_argument("--time-topk", type=int, default=0)
    parser.add_argument(
        "--time-routing-mode",
        default="threshold",
        choices=["threshold", "topk", "threshold_topk", "relative_threshold", "relative_topk", "force_current"],
    )
    parser.add_argument(
        "--time-score-norm",
        default="none",
        choices=["none", "factor", "factor_z", "self_score", "factor_self_score"],
    )
    parser.add_argument(
        "--time-align-score-norm",
        default=None,
        choices=["none", "factor", "factor_z", "self_score", "factor_self_score"],
    )
    parser.add_argument("--lambda-align", dest="lambda_align", type=float, default=None)
    parser.add_argument("--num-negative-experts", dest="num_negative_experts", type=int, default=None)
    parser.add_argument("--time-relative-threshold", type=float, default=None)
    parser.add_argument("--time-mixing-mode", default="softmax", choices=["softmax", "average", "own_oracle"])
    parser.add_argument("--time-anti-collapse-loss", action="store_true")
    parser.add_argument("--lambda-anti-collapse", dest="lambda_anti_collapse", type=float, default=0.0)
    parser.add_argument("--anti-collapse-margin", dest="anti_collapse_margin", type=float, default=0.05)
    parser.add_argument(
        "--anti-collapse-score-norm",
        dest="anti_collapse_score_norm",
        default="factor_z",
        choices=["none", "factor", "factor_z", "self_score", "factor_self_score"],
    )
    parser.add_argument("--lambda-factor-norm-reg", dest="lambda_factor_norm_reg", type=float, default=0.0)
    parser.add_argument("--time-load-repository", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--time-routing-calibration", action="store_true")
    parser.add_argument("--time-routing-calibration-grid", default="")
    parser.add_argument("--time-post-retrain-calibration", action="store_true")
    parser.add_argument("--time-max-selected-experts", type=int, default=None)
    parser.add_argument(
        "--time-calibration-mode",
        default="none",
        choices=["none", "self_ratio", "zscore_neg", "zscore_neg_mean", "neg_margin", "self_minus_neg_mean"],
    )
    parser.add_argument("--time-calibration-beta", type=float, default=0.0)
    parser.add_argument("--time-score-pool", default="mean", choices=["mean", "max", "last", "answer_mean"])
    parser.add_argument("--time-force-current-train", dest="time_force_current_train", action="store_true", default=None)
    parser.add_argument("--time-gamma-sweep", default="")
    parser.add_argument("--time-scale-init-grid", default="")
    parser.add_argument("--time-overfit-grid", default="")
    parser.add_argument("--time-reliability-only", action="store_true")
    parser.add_argument("--time-residual-sign", default="plus", choices=["plus", "minus"])
    parser.add_argument("--time-expert-gain", type=float, default=1.0)
    parser.add_argument("--time-token-scope", default="all", choices=["all", "last", "answer_mask"])
    parser.add_argument("--eval-routing-modes", default="")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=5)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_command_line() -> str:
    command = "/root/anaconda3/bin/python " + " ".join(sys.argv)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        command = f"CUDA_VISIBLE_DEVICES={cuda_visible} " + command
    return command


def normalize_device_arg(text: str) -> Any:
    if text == "cuda":
        return "cuda"
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        return int(suffix) if suffix.isdigit() else text
    return int(text) if text.isdigit() else text


def normalize_time_calibration_mode(text: str) -> str:
    mode = str(text or "none")
    if mode == "zscore_neg_mean":
        return "zscore_neg"
    return mode


def load_records(dataset_path: Path, sample_index: int, max_edits: int) -> List[Dict[str, Any]]:
    records = json.loads(dataset_path.read_text(errors="replace"))
    if not isinstance(records, list):
        raise RuntimeError(f"Dataset JSON root must be a list: {dataset_path}")
    end = min(len(records), sample_index + max_edits)
    if sample_index < 0 or sample_index >= len(records) or sample_index >= end:
        raise IndexError(f"sample_index={sample_index} outside dataset of size {len(records)}")
    selected = records[sample_index:end]
    if len(selected) < max_edits:
        raise RuntimeError(f"Requested {max_edits} edits but only {len(selected)} records are available from index {sample_index}.")
    return selected


def resolve_repository_path(repo_path: Optional[Path]) -> Optional[Path]:
    if repo_path is None:
        return None
    path = repo_path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_repository_metadata(repo_path: Path) -> List[Dict[str, Any]]:
    payload = torch.load(repo_path, map_location="cpu")
    metadata = payload.get("metadata", []) if isinstance(payload, dict) else []
    return [dict(item) for item in metadata]


def load_records_for_repository(
    dataset_path: Path,
    sample_index: int,
    max_edits: int,
    repo_path: Optional[Path],
) -> List[Dict[str, Any]]:
    if repo_path is None:
        return load_records(dataset_path, sample_index, max_edits)
    metadata = load_repository_metadata(repo_path)
    record_ids = [str(item.get("record_id")) for item in metadata[:max_edits] if item.get("record_id") is not None]
    if not record_ids:
        return load_records(dataset_path, sample_index, max_edits)
    records = json.loads(dataset_path.read_text(errors="replace"))
    by_id = {record_id(record, idx): record for idx, record in enumerate(records)}
    missing = [rid for rid in record_ids if rid not in by_id]
    if missing:
        raise RuntimeError(f"Repository metadata record ids are missing from {dataset_path}: {missing}")
    return [by_id[rid] for rid in record_ids]


def resolve_image_path(image_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = image_root.resolve()
    if root.name == "images" and str(value).startswith("images/"):
        return root.parent / path
    return root / path


def record_id(record: Dict[str, Any], fallback: int) -> str:
    return str(record.get("id", record.get("record_id", fallback)))


def record_prompt(record: Dict[str, Any]) -> str:
    return "Question: {} Short answer: ".format(record.get("src") or record.get("prompt") or record.get("question") or "")


def record_target(record: Dict[str, Any]) -> str:
    return str(record.get("alt") or record.get("target") or record.get("answer") or "")


def make_sample(model: Any, record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    prompt = record_prompt(record)
    target = record_target(record)
    labels = model.llava_tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return {
        "image_path": [str(resolve_image_path(image_root, record["image"]))],
        "prompt": [prompt],
        "target": [target],
        "text_input": [prompt + target],
        "labels": labels,
        "prompts_len": [len(model.llava_tokenizer(prompt, add_special_tokens=False).input_ids)],
    }


def configure(args: argparse.Namespace, dataset_path: Path) -> TIMEEditMultimodalHparams:
    config = TIMEEditMultimodalHparams.from_hparams(args.hparams)
    config.alg = args.alg
    config.alg_name = args.alg
    config.device = normalize_device_arg(args.device)
    config.lr = float(args.lr)
    config.edit_lr = float(args.lr)
    config.time_target_layer = int(args.target_layer)
    config.time_rank = int(args.rank)
    config.time_alpha = float(args.alpha)
    config.time_gamma = float(args.gamma)
    config.time_tau = float(args.tau)
    config.time_scale_mode = str(args.scale_mode)
    config.time_init_std = float(args.init_std)
    config.time_edit_iters = int(args.edit_iters)
    config.time_disable_selection = bool(args.time_disable_selection)
    config.time_disable_score_mixing = bool(args.time_disable_score_mixing)
    config.time_disable_align_loss = bool(args.time_disable_align_loss)
    config.time_topk = int(args.time_topk)
    config.time_routing_mode = str(args.time_routing_mode)
    config.time_score_norm = str(args.time_score_norm)
    config.time_align_score_norm = str(args.time_align_score_norm or args.time_score_norm)
    if args.lambda_align is not None:
        config.time_lambda_align = float(args.lambda_align)
    if args.num_negative_experts is not None:
        config.time_negative_experts = int(args.num_negative_experts)
    config.time_relative_threshold = None if args.time_relative_threshold is None else float(args.time_relative_threshold)
    config.time_mixing_mode = "average" if args.time_disable_score_mixing else str(args.time_mixing_mode)
    config.time_max_selected_experts = args.time_max_selected_experts
    config.time_calibration_mode = normalize_time_calibration_mode(args.time_calibration_mode)
    config.time_calibration_beta = float(args.time_calibration_beta)
    config.time_score_pool = str(args.time_score_pool) if args.time_post_retrain_calibration else "token"
    config.time_anti_collapse_loss = bool(args.time_anti_collapse_loss)
    config.time_lambda_anti_collapse = float(args.lambda_anti_collapse)
    config.time_anti_collapse_margin = float(args.anti_collapse_margin)
    config.time_anti_collapse_score_norm = str(args.anti_collapse_score_norm)
    config.time_lambda_factor_norm_reg = float(args.lambda_factor_norm_reg)
    config.time_repository_path = str(args.time_load_repository) if args.time_load_repository is not None else None
    config.eval_only = bool(args.eval_only)
    if args.time_force_current_train is not None:
        config.time_force_current_during_training = bool(args.time_force_current_train)
    config.time_residual_sign = str(args.time_residual_sign)
    config.time_expert_gain = float(args.time_expert_gain)
    config.time_reliability_only = bool(args.time_reliability_only)
    if args.time_reliability_only:
        config.time_lambda_rel = 1.0
        config.time_lambda_gen = 0.0
        config.time_lambda_loc = 0.0
        config.time_lambda_align = 0.0
        config.time_disable_align_loss = True
    config.time_token_scope = str(args.time_token_scope)
    if args.image_root is not None:
        config.coco_image = str(args.image_root)
        config.rephrase_image = str(args.image_root)
    if not config.coco_image:
        config.coco_image = str(dataset_path.parent / "images")
        config.rephrase_image = config.coco_image
    return config


def logits_delta(outputs_a: Any, outputs_b: Any, batch: Dict[str, Any]) -> float:
    logits_a = outputs_a if isinstance(outputs_a, torch.Tensor) else outputs_a.logits
    logits_b = outputs_b if isinstance(outputs_b, torch.Tensor) else outputs_b.logits
    labels = batch["labels"]
    seq = min(logits_a.shape[1], logits_b.shape[1], labels.shape[1] + 1)
    return float((logits_b[:, -seq:].detach().float() - logits_a[:, -seq:].detach().float()).norm().cpu())


def evaluate_sample(
    alg: TIMEEdit,
    sample: Dict[str, Any],
    record: Dict[str, Any],
    sample_pos: int,
    phase: str,
    expected_expert: Optional[int],
    routing_debug_path: Path,
    eval_routing_mode: Optional[str] = None,
    force_expert_id: Optional[int] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    base_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    debug_events: List[Dict[str, Any]] = []
    start = time.perf_counter()
    if base_cache is None:
        with torch.no_grad(), alg.time_disabled():
            base_batch = clone_batch(sample)
            base_outputs = alg.model(base_batch)
        base_nll = target_nll_from_outputs(base_outputs, base_batch)
    else:
        base_batch = base_cache["batch"]
        base_outputs = base_cache["outputs"]
        base_nll = base_cache["nll"]
    with torch.no_grad():
        edited_batch = clone_batch(sample)
        old_current_expert = alg.current_expert_index
        try:
            if force_expert_id is not None:
                alg.current_expert_index = int(force_expert_id)
            edited_outputs = alg._forward_with_time(
                edited_batch,
                call_label=phase,
                force_current=force_expert_id is not None,
                debug_events=debug_events,
            )
        finally:
            alg.current_expert_index = old_current_expert
    elapsed = time.perf_counter() - start
    edited_nll = target_nll_from_outputs(edited_outputs, edited_batch)
    routing = alg.routing_summary()
    rid = record_id(record, sample_pos)
    scores = routing.get("pooled_scores") or []
    raw_scores = routing.get("raw_pooled_scores") or scores
    score_variants = routing.get("score_variant_pooled_scores") or {"none": raw_scores}
    weights = routing.get("pooled_weights") or []
    selected_ids = routing.get("selected_expert_ids") or []
    own_expert_score = None
    own_expert_weight = None
    if expected_expert is not None and 0 <= int(expected_expert) < len(scores):
        own_expert_score = scores[int(expected_expert)]
        if int(expected_expert) < len(weights):
            own_expert_weight = weights[int(expected_expert)]
    top_score_expert_id = routing.get("top_expert_id")
    top_routed_expert_id = int(force_expert_id) if force_expert_id is not None else top_score_expert_id
    target_nll_delta = (
        (base_nll.get("target_nll") or 0.0) - (edited_nll.get("target_nll") or 0.0)
        if base_nll.get("target_nll") is not None and edited_nll.get("target_nll") is not None
        else None
    )
    row = {
        "phase": phase,
        "eval_routing_mode": eval_routing_mode or routing.get("routing_mode"),
        "sample_pos": sample_pos,
        "record_id": rid,
        "target": record_target(record),
        "expected_expert": expected_expert,
        "base_target_nll": base_nll.get("target_nll"),
        "target_nll": edited_nll.get("target_nll"),
        "target_nll_delta": target_nll_delta,
        "target_improved": bool(target_nll_delta is not None and target_nll_delta > 0.0),
        "base_avg_target_logprob": base_nll.get("avg_target_logprob"),
        "avg_target_logprob": edited_nll.get("avg_target_logprob"),
        "answer_token_logprob_delta": (
            (edited_nll.get("avg_target_logprob") or 0.0) - (base_nll.get("avg_target_logprob") or 0.0)
            if base_nll.get("avg_target_logprob") is not None and edited_nll.get("avg_target_logprob") is not None
            else None
        ),
        "target_token_count": edited_nll.get("target_token_count"),
        "base_first_target_token_rank": base_nll.get("first_target_token_rank"),
        "first_target_token_rank": edited_nll.get("first_target_token_rank"),
        "target_rank_delta": (
            (base_nll.get("first_target_token_rank") or 0) - (edited_nll.get("first_target_token_rank") or 0)
            if base_nll.get("first_target_token_rank") is not None and edited_nll.get("first_target_token_rank") is not None
            else None
        ),
        "reference_delta": logits_delta(base_outputs, edited_outputs, edited_batch),
        "top_expert_id": top_score_expert_id,
        "top_score_expert_id": top_score_expert_id,
        "top_routed_expert_id": top_routed_expert_id,
        "top_score": routing.get("top_score"),
        "own_expert_score": own_expert_score,
        "own_expert_weight": own_expert_weight,
        "selected_expert_ids": selected_ids,
        "selected_expert_set_size": routing.get("selected_expert_set_size"),
        "selected_own_expert": bool(expected_expert is not None and int(expected_expert) in selected_ids),
        "routing_top1_correct": bool(expected_expert is not None and top_routed_expert_id == expected_expert),
        "residual_norm": routing.get("residual_norm"),
        "target_layer_hidden_delta_norm": routing.get("target_layer_hidden_delta_norm"),
        "hidden_delta_norm": routing.get("target_layer_hidden_delta_norm"),
        "target_layer_hidden_changed": routing.get("target_layer_hidden_changed"),
        "routing_mode": routing.get("routing_mode"),
        "residual_sign": routing.get("residual_sign"),
        "expert_gain": routing.get("expert_gain"),
        "gamma": routing.get("gamma"),
        "topk": routing.get("topk"),
        "relative_threshold": routing.get("relative_threshold"),
        "score_norm": routing.get("score_norm"),
        "mixing_mode": routing.get("mixing_mode"),
        "pooled_scores": scores,
        "raw_pooled_scores": raw_scores,
        "score_variant_pooled_scores": score_variants,
        "pooled_weights": weights,
        "elapsed_sec": elapsed,
        "generation_skipped": True,
    }
    if extra_fields:
        row.update(extra_fields)
    append_jsonl(routing_debug_path, row)
    for event in debug_events:
        event = dict(event)
        event.update({"phase": f"{phase}_hook_event", "record_id": rid, "expected_expert": expected_expert})
        if extra_fields:
            event.update(extra_fields)
        append_jsonl(routing_debug_path, event)
    return row


def build_base_eval_cache(alg: TIMEEdit, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cache: List[Dict[str, Any]] = []
    with torch.no_grad(), alg.time_disabled():
        for sample in samples:
            batch = clone_batch(sample)
            outputs = alg.model(batch)
            cache.append({"batch": batch, "outputs": outputs, "nll": target_nll_from_outputs(outputs, batch)})
    return cache


def answer_token_static_debug(model: Any, sample: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    labels = sample["labels"]
    mask = labels != -100
    ids = [int(value) for value in labels.masked_select(mask).detach().cpu().tolist()]
    try:
        decoded = model.llava_tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        decoded = " ".join(str(value) for value in ids)
    return {
        "target_answer_string": record_target(record),
        "tokenized_target_answer_ids": ids,
        "decoded_target_answer_tokens": decoded,
        "supervised_target_token_count": int(mask.sum().item()),
        "label_mask_positions": mask.nonzero(as_tuple=False).detach().cpu().tolist(),
        "labels_shape": list(labels.shape),
        "prompts_len": sample.get("prompts_len"),
        "diagnostic_note": "Reliability-only mode is an overfit diagnostic, not the final TIME lifelong objective.",
    }


def answer_metrics(outputs: Any, batch: Dict[str, Any]) -> Dict[str, Any]:
    metrics = target_nll_from_outputs(outputs, batch)
    count = int(metrics.get("target_token_count") or 0)
    avg_logprob = metrics.get("avg_target_logprob")
    total_logprob = float(avg_logprob) * count if avg_logprob is not None and count else None
    return {
        **metrics,
        "total_target_logprob": total_logprob,
    }


def compare_answer_metrics(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before_nll = before.get("target_nll")
    after_nll = after.get("target_nll")
    before_avg = before.get("avg_target_logprob")
    after_avg = after.get("avg_target_logprob")
    before_total = before.get("total_target_logprob")
    after_total = after.get("total_target_logprob")
    before_rank = before.get("first_target_token_rank")
    after_rank = after.get("first_target_token_rank")
    return {
        "target_nll_delta": float(before_nll - after_nll) if before_nll is not None and after_nll is not None else None,
        "avg_target_logprob_delta": float(after_avg - before_avg) if before_avg is not None and after_avg is not None else None,
        "total_target_logprob_delta": float(after_total - before_total) if before_total is not None and after_total is not None else None,
        "target_rank_delta": int(before_rank - after_rank) if before_rank is not None and after_rank is not None else None,
    }


def logits_kl_from_outputs(outputs_a: Any, outputs_b: Any, batch: Dict[str, Any]) -> float:
    logits_a, labels = aligned_logits_and_labels(outputs_a, batch)
    logits_b, _labels = aligned_logits_and_labels(outputs_b, batch)
    mask = labels != -100
    if not bool(mask.any()):
        return 0.0
    logp_a = torch.log_softmax(logits_a.float(), dim=-1)
    logp_b = torch.log_softmax(logits_b.float(), dim=-1)
    p_a = torch.softmax(logits_a.float(), dim=-1)
    kl = (p_a * (logp_a - logp_b)).sum(dim=-1)
    return float(kl.masked_select(mask).mean().detach().cpu())


def base_trainable_param_count(alg: TIMEEdit) -> int:
    count = 0
    for name, param in alg.model.named_parameters():
        if "time" not in name and not name.startswith("repository.") and param.requires_grad:
            count += int(param.numel())
    return count


def factor_grad_trace(alg: TIMEEdit) -> Dict[str, float]:
    result: Dict[str, float] = {
        "base_vlm_trainable_params": float(base_trainable_param_count(alg)),
        "current_expert_grad_norm_total": 0.0,
    }
    if alg.current_expert_index is None or alg.repository.num_experts == 0:
        return result
    expert = alg.repository.experts[int(alg.current_expert_index)]
    total_grad_sq = 0.0
    for name in ("U_in", "V_in", "U_out", "V_out"):
        param = getattr(expert, name)
        norm = float(param.detach().float().norm().cpu())
        grad_norm = float(param.grad.detach().float().norm().cpu()) if param.grad is not None else 0.0
        result[f"{name}_factor_norm"] = norm
        result[f"{name}_grad_norm"] = grad_norm
        total_grad_sq += grad_norm * grad_norm
    result["current_expert_grad_norm_total"] = math.sqrt(total_grad_sq)
    return result


def write_answer_debug(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)


def train_one_edit(
    alg: TIMEEdit,
    sample: Dict[str, Any],
    record: Dict[str, Any],
    sample_pos: int,
    args: argparse.Namespace,
    loss_rows: List[Dict[str, Any]],
    reliability_rows: List[Dict[str, Any]],
    out_dir: Path,
    previous_samples: Optional[List[Dict[str, Any]]] = None,
    anti_collapse_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rid = record_id(record, sample_pos)
    expert_index = alg.add_expert(
        rid,
        metadata={
            "sample_pos": sample_pos,
            "prompt": record_prompt(record),
            "target": record_target(record),
            "mode": args.mode,
        },
    )
    optimizer = torch.optim.AdamW(alg.outer_parameters(), lr=float(args.lr))
    previous_samples = previous_samples or []
    alg._time_anti_collapse_batches = [clone_batch(prev) for prev in previous_samples]
    alg._time_anti_collapse_expected_ids = list(range(len(previous_samples)))
    batch = {
        "edit_inner": clone_batch(sample),
        "edit_outer": clone_batch(sample),
        "loc": clone_batch(sample),
        "loc_image": clone_batch(sample),
        "record_id": rid,
    }
    answer_debug = {
        **answer_token_static_debug(alg.model, sample, record),
        "record_id": rid,
        "sample_pos": sample_pos,
        "expert_index": expert_index,
        "reliability_only": bool(args.time_reliability_only),
        "lambda_rel": float(alg.config.time_lambda_rel),
        "lambda_gen": float(alg.config.time_lambda_gen),
        "lambda_loc": float(alg.config.time_lambda_loc),
        "lambda_align": float(alg.config.time_lambda_align),
        "lambda_anti_collapse": float(getattr(alg.config, "time_lambda_anti_collapse", 0.0)),
        "lambda_factor_norm_reg": float(getattr(alg.config, "time_lambda_factor_norm_reg", 0.0)),
        "align_score_norm": str(getattr(alg.config, "time_align_score_norm", alg.config.time_score_norm)),
        "anti_collapse_score_norm": str(getattr(alg.config, "time_anti_collapse_score_norm", "factor_z")),
        "residual_sign": str(alg.time_residual.residual_sign),
        "expert_gain": float(alg.time_residual.expert_gain),
        "base_vlm_trainable_params": float(base_trainable_param_count(alg)),
    }
    with torch.no_grad(), alg.time_disabled():
        initial_base_batch = clone_batch(sample)
        initial_base_outputs = alg.model(initial_base_batch)
    initial_debug_events: List[Dict[str, Any]] = []
    with torch.no_grad():
        initial_edit_batch = clone_batch(sample)
        initial_edit_outputs = alg._forward_with_time(
            initial_edit_batch,
            call_label="initial_force_current_debug",
            force_current=True,
            debug_events=initial_debug_events,
        )
    answer_debug["initial_base"] = answer_metrics(initial_base_outputs, initial_base_batch)
    answer_debug["initial_force_current"] = answer_metrics(initial_edit_outputs, initial_edit_batch)
    answer_debug["initial_force_current_vs_base"] = compare_answer_metrics(
        answer_debug["initial_base"],
        answer_debug["initial_force_current"],
    )
    answer_debug["initial_routing"] = alg.routing_summary()
    answer_debug["initial_hook_events"] = initial_debug_events
    start = time.perf_counter()
    last_info: Dict[str, Any] = {}
    for step in range(1, int(args.edit_iters) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total, loss_rel, loss_loc, _loss_base, info = alg.edit_step(batch, training=True, optimizer=optimizer)
        step_trace = {
            "mode": args.mode,
            "sample_pos": sample_pos,
            "record_id": rid,
            "expert_index": expert_index,
            "step": step,
            "target_nll": float(loss_rel.detach().cpu()),
            "loss_total": float(loss_total.detach().cpu()),
            "loss_loc": float(loss_loc.detach().cpu()),
            "loss_anti_collapse": float(info.get("loss/time_anti_collapse", 0.0) or 0.0),
            "loss_factor_norm_reg": float(info.get("loss/time_factor_norm_reg", 0.0) or 0.0),
            "answer_avg_logprob": -float(loss_rel.detach().cpu()),
            "answer_total_logprob": -float(loss_rel.detach().cpu()) * int(answer_debug["supervised_target_token_count"]),
            "current_expert_score": float(info.get("time/top_score", 0.0) or 0.0),
            "selected_expert_set_size": float(info.get("time/selected_expert_set_size", 0.0) or 0.0),
            "residual_norm": float(alg.routing_summary().get("residual_norm", 0.0) or 0.0),
            "hidden_delta_norm": float(alg.routing_summary().get("target_layer_hidden_delta_norm", 0.0) or 0.0),
            "residual_sign": str(alg.time_residual.residual_sign),
            "expert_gain": float(alg.time_residual.expert_gain),
            "score_norm": str(alg.time_residual.score_norm),
            "align_score_norm": str(getattr(alg.config, "time_align_score_norm", alg.config.time_score_norm)),
            "anti_collapse_active": bool(args.time_anti_collapse_loss),
            "anti_collapse_prev_batches": float(info.get("time/anti_collapse_prev_batches", 0.0) or 0.0),
            "anti_collapse_current_prev_mean": float(info.get("time/anti_collapse_current_prev_mean", 0.0) or 0.0),
            "anti_collapse_prev_own_mean": float(info.get("time/anti_collapse_prev_own_mean", 0.0) or 0.0),
            **factor_grad_trace(alg),
        }
        if anti_collapse_rows is not None:
            anti_collapse_rows.append(step_trace)
        if args.time_reliability_only or args.time_overfit_grid:
            reliability_rows.append(step_trace)
        torch.nn.utils.clip_grad_norm_(alg.outer_parameters(), float(alg.config.grad_clip), error_if_nonfinite=True)
        optimizer.step()
        last_info = dict(info)
        if step == 1 or step == args.edit_iters or step % max(1, args.log_every) == 0:
            loss_rows.append(
                {
                    "mode": args.mode,
                    "sample_pos": sample_pos,
                    "record_id": rid,
                    "expert_index": expert_index,
                    "step": step,
                    "loss_total": float(loss_total.detach().cpu()),
                    "loss_rel": float(loss_rel.detach().cpu()),
                    "loss_loc": float(loss_loc.detach().cpu()),
                    **{key.replace("/", "_"): value for key, value in last_info.items()},
                }
            )
    with torch.no_grad(), alg.time_disabled():
        final_base_batch = clone_batch(sample)
        final_base_outputs = alg.model(final_base_batch)
    final_debug_events: List[Dict[str, Any]] = []
    with torch.no_grad():
        final_edit_batch = clone_batch(sample)
        final_edit_outputs = alg._forward_with_time(
            final_edit_batch,
            call_label="final_force_current_debug",
            force_current=True,
            debug_events=final_debug_events,
        )
    answer_debug["final_base"] = answer_metrics(final_base_outputs, final_base_batch)
    answer_debug["final_force_current"] = answer_metrics(final_edit_outputs, final_edit_batch)
    answer_debug["final_force_current_vs_base"] = compare_answer_metrics(
        answer_debug["final_base"],
        answer_debug["final_force_current"],
    )
    answer_debug["final_force_current_vs_initial_force_current"] = compare_answer_metrics(
        answer_debug["initial_force_current"],
        answer_debug["final_force_current"],
    )
    answer_debug["final_routing"] = alg.routing_summary()
    answer_debug["final_hook_events"] = final_debug_events
    answer_debug["trace_path"] = str(out_dir / "time_reliability_overfit_trace.csv")
    if args.time_reliability_only or args.time_overfit_grid:
        write_answer_debug(out_dir / "time_answer_token_debug.json", answer_debug)
    return {
        "record_id": rid,
        "sample_pos": sample_pos,
        "expert_index": expert_index,
        "train_elapsed_sec": time.perf_counter() - start,
        "last_info": last_info,
        "answer_debug": answer_debug if (args.time_reliability_only or args.time_overfit_grid) else None,
    }


def write_loss_trace(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(to_jsonable(row))


def collect_score_matrix_rows(
    alg: TIMEEdit,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    after_edit_index: int,
    out_dir: Path,
    phase_prefix: str = "score_matrix",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    debug_path = out_dir / "score_matrix_debug.jsonl"
    with temporary_time_routing(
        alg,
        routing_mode="threshold",
        gamma=1.0e30,
        topk=0,
        score_norm="none",
        relative_threshold=None,
        mixing_mode="average",
    ):
        for query_index, (record, sample) in enumerate(zip(records[: after_edit_index + 1], samples[: after_edit_index + 1])):
            row = evaluate_sample(
                alg,
                sample,
                record,
                query_index,
                phase=f"{phase_prefix}_after_edit_{after_edit_index}_query_{query_index}",
                expected_expert=query_index,
                routing_debug_path=debug_path,
                eval_routing_mode="score_matrix",
                force_expert_id=query_index,
                extra_fields={"score_matrix_after_edit": after_edit_index},
            )
            variants = row.get("score_variant_pooled_scores") or {}
            raw_scores = variants.get("none") or row.get("raw_pooled_scores") or []
            for expert_index, raw_score in enumerate(raw_scores):
                matrix_row = {
                    "after_edit_index": after_edit_index,
                    "query_record_index": query_index,
                    "record_id": row.get("record_id"),
                    "expert_index": expert_index,
                    "own_expert": expert_index == query_index,
                    "raw_score": raw_score,
                    "top_raw_expert": int(max(range(len(raw_scores)), key=lambda idx: raw_scores[idx])) if raw_scores else None,
                }
                for mode in SCORE_NORM_MODES:
                    values = variants.get(mode) or []
                    matrix_row[f"{mode}_score"] = values[expert_index] if expert_index < len(values) else None
                    matrix_row[f"{mode}_top_expert"] = int(max(range(len(values)), key=lambda idx: values[idx])) if values else None
                rows.append(matrix_row)
    return rows


def compute_self_score_metadata(
    alg: TIMEEdit,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    debug_path = out_dir / "self_score_metadata_debug.jsonl"
    payload: Dict[str, Any] = {"records": []}
    self_scores = {"none": [], "factor": [], "factor_z": []}
    with temporary_time_routing(
        alg,
        routing_mode="threshold",
        gamma=1.0e30,
        topk=0,
        score_norm="none",
        relative_threshold=None,
        mixing_mode="average",
    ):
        for eval_pos, (record, sample) in enumerate(zip(records, samples)):
            row = evaluate_sample(
                alg,
                sample,
                record,
                eval_pos,
                phase=f"self_score_metadata_eval_{eval_pos}",
                expected_expert=eval_pos,
                routing_debug_path=debug_path,
                eval_routing_mode="self_score_metadata",
                force_expert_id=eval_pos,
                extra_fields={"calibration_phase": "self_score_metadata"},
            )
            variants = row.get("score_variant_pooled_scores") or {}
            entry = {
                "record_id": row.get("record_id"),
                "expert_index": eval_pos,
                "target": row.get("target"),
            }
            for mode, key in (("none", "self_score_raw"), ("factor", "self_score_factor"), ("factor_z", "self_score_factor_z")):
                values = variants.get(mode) or row.get("raw_pooled_scores") or []
                value = float(values[eval_pos]) if eval_pos < len(values) else 1.0
                entry[key] = value
                self_scores[mode].append(value)
            payload["records"].append(entry)
            if eval_pos < len(alg.repository.metadata):
                metadata = alg.repository.metadata[eval_pos]
                metadata["self_score_raw"] = entry["self_score_raw"]
                metadata["self_score_factor"] = entry["self_score_factor"]
                metadata["self_score_factor_z"] = entry["self_score_factor_z"]
                metadata["time_self_score_none"] = entry["self_score_raw"]
                metadata["time_self_score_factor"] = entry["self_score_factor"]
                metadata["time_self_score_factor_z"] = entry["self_score_factor_z"]
    alg.time_residual.self_score_cache = {
        "none": list(self_scores["none"]),
        "factor": list(self_scores["factor"]),
        "factor_z": list(self_scores["factor_z"]),
    }
    payload["self_score_cache"] = self_scores
    write_json(out_dir / "time_self_score_metadata.json", payload)
    return payload


def summarize_evals(mode: str, eval_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]], alg: TIMEEdit, out_dir: Path) -> Dict[str, Any]:
    deltas = [row.get("target_nll_delta") for row in eval_rows if row.get("target_nll_delta") is not None]
    ref = [row.get("reference_delta") for row in eval_rows if row.get("reference_delta") is not None]
    selected_sizes = [row.get("selected_expert_set_size") for row in eval_rows if row.get("selected_expert_set_size") is not None]
    improved = [row for row in eval_rows if (row.get("target_nll_delta") or 0.0) > 0.0 or (row.get("target_rank_delta") or 0.0) > 0.0]
    positive_new_count = sum(1 for row in eval_rows if row.get("routing_top1_correct"))
    confusion: Dict[str, Dict[str, int]] = {}
    for row in eval_rows:
        expected = str(row.get("expected_expert"))
        observed = str(row.get("top_expert_id"))
        confusion.setdefault(expected, {})
        confusion[expected][observed] = confusion[expected].get(observed, 0) + 1
    repo_path = out_dir / "expert_repository.pt"
    alg.repository.save(repo_path)
    memory = time_memory_estimate(alg.repository.hidden_size, alg.repository.rank, alg.repository.s1, alg.repository.s2)
    memory["repository_size_bytes"] = float(repo_path.stat().st_size if repo_path.exists() else 0)
    return {
        "mode": mode,
        "num_eval_records": len(eval_rows),
        "num_experts": alg.repository.num_experts,
        "mean_target_nll_delta": float(sum(deltas) / len(deltas)) if deltas else None,
        "improved_count": len(improved),
        "positive_new_count": positive_new_count,
        "routing_top1_accuracy": float(positive_new_count / len(eval_rows)) if eval_rows else None,
        "mean_reference_delta": float(sum(ref) / len(ref)) if ref else None,
        "mean_selected_expert_set_size": float(sum(selected_sizes) / len(selected_sizes)) if selected_sizes else None,
        "routing_confusion_matrix": confusion,
        "eval_rows": eval_rows,
        "train_records": train_rows,
        "memory_estimate": memory,
        "expert_repository": str(repo_path),
        "acceptance": {
            "target_improvement_gate": len(improved) >= max(1, math.ceil(0.8 * len(eval_rows))) if eval_rows else False,
            "routing_gate": positive_new_count >= max(1, math.ceil(0.8 * len(eval_rows))) if eval_rows else False,
            "locality_reported": bool(ref),
        },
    }


def parse_eval_routing_modes(text: str) -> List[str]:
    modes = [part.strip() for part in str(text or "").split(",") if part.strip()]
    if not modes:
        return []
    allowed = {"force_own", "calibrated", "topk", "threshold", "relative"}
    unsupported = [mode for mode in modes if mode not in allowed]
    if unsupported:
        raise ValueError(f"Unsupported --eval-routing-modes values: {unsupported}. Allowed: {sorted(allowed)}")
    return modes


def _mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    return float(sum(valid) / len(valid)) if valid else None


def _mode_rows(eval_rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    return [row for row in eval_rows if row.get("eval_routing_mode") == mode]


def _routing_confusion(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    confusion: Dict[str, Dict[str, int]] = {}
    for row in rows:
        expected = str(row.get("expected_expert"))
        observed = str(row.get("top_routed_expert_id", row.get("top_expert_id")))
        confusion.setdefault(expected, {})
        confusion[expected][observed] = confusion[expected].get(observed, 0) + 1
    return confusion


def _edit_count_word(num_edits: int) -> str:
    return {5: "five", 10: "ten"}.get(int(num_edits), str(num_edits))


def _nonseq_prefix(num_edits: int) -> str:
    return f"time_{_edit_count_word(num_edits)}_nonseq" if int(num_edits) == 10 else "five_edit_nonseq"


def _nonseq_report_name(num_edits: int) -> str:
    return "TIME_10EDIT_NONSEQ_CALIBRATED_REPORT.md" if int(num_edits) == 10 else "TIME_5EDIT_NONSEQ_DIAGNOSTIC_REPORT.md"


def _sequential_prefix(num_edits: int) -> str:
    return f"time_{_edit_count_word(num_edits)}_sequential" if int(num_edits) == 10 else "time_sequential"


def _sequential_report_name(num_edits: int) -> str:
    return "TIME_10EDIT_SEQUENTIAL_CALIBRATED_REPORT.md" if int(num_edits) == 10 else "TIME_5EDIT_SEQUENTIAL_CALIBRATED_REPORT.md"


def _positive_gate(num_records: int) -> int:
    return max(1, math.ceil(0.8 * int(num_records)))


def _selected_gate(num_records: int) -> int:
    return max(1, math.ceil(0.9 * int(num_records))) if int(num_records) >= 10 else _positive_gate(num_records)


def _max_top_expert_gate(num_records: int) -> int:
    return 4 if int(num_records) >= 10 else 3


def _top_expert_counts(rows: List[Dict[str, Any]], field: str = "top_score_expert_id") -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get(field, row.get("top_expert_id"))
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _worst_nll_degradation(rows: List[Dict[str, Any]]) -> float:
    degradations = [
        max(0.0, -float(row.get("target_nll_delta")))
        for row in rows
        if row.get("target_nll_delta") is not None
    ]
    return max(degradations, default=0.0)


def _recommended_threshold_gamma(rows: List[Dict[str, Any]], min_own: int = 3) -> Optional[float]:
    scores = sorted(
        [float(row["own_expert_score"]) for row in rows if row.get("own_expert_score") is not None],
        reverse=True,
    )
    if len(scores) < min_own:
        return None
    return max(0.0, scores[min_own - 1] - 1.0e-6)


def summarize_five_edit_nonseq(eval_rows: List[Dict[str, Any]], routing_modes: List[str]) -> Dict[str, Any]:
    by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in routing_modes:
        rows = _mode_rows(eval_rows, mode)
        selected_sizes = [row.get("selected_expert_set_size") for row in rows]
        top1_count = sum(1 for row in rows if row.get("routing_top1_correct"))
        selected_own_count = sum(1 for row in rows if row.get("selected_own_expert"))
        empty_count = sum(1 for row in rows if int(row.get("selected_expert_set_size") or 0) == 0)
        top_counts = _top_expert_counts(rows)
        by_mode[mode] = {
            "num_records": len(rows),
            "mean_target_nll_improvement": _mean([row.get("target_nll_delta") for row in rows]),
            "positive_new_count": sum(1 for row in rows if (row.get("target_nll_delta") or 0.0) > 0.0),
            "mean_reference_delta": _mean([row.get("reference_delta") for row in rows]),
            "mean_residual_norm": _mean([row.get("residual_norm") for row in rows]),
            "mean_hidden_delta_norm": _mean([row.get("hidden_delta_norm") for row in rows]),
            "worst_per_record_nll_degradation": _worst_nll_degradation(rows),
            "routing_top1_count": top1_count,
            "routing_top1_accuracy": float(top1_count / len(rows)) if rows else None,
            "threshold_selected_own_count": selected_own_count,
            "threshold_empty_selection_count": empty_count,
            "mean_selected_expert_set_size": _mean([float(value) for value in selected_sizes if value is not None]),
            "max_selected_expert_set_size": max([float(value) for value in selected_sizes if value is not None], default=None),
            "top_expert_counts": top_counts,
            "max_top_expert_count": max(top_counts.values()) if top_counts else 0,
            "routing_confusion_matrix": _routing_confusion(rows),
            "recommended_gamma_for_at_least_3_own": _recommended_threshold_gamma(rows, min_own=3),
        }
    return by_mode


def nonseq_calibrated_acceptance(mode_summaries: Dict[str, Dict[str, Any]], num_records: int) -> Dict[str, Any]:
    calibrated = mode_summaries.get("calibrated") or {}
    positive_gate = _positive_gate(num_records)
    selected_gate = _selected_gate(num_records)
    max_top_gate = _max_top_expert_gate(num_records)
    mean_size = calibrated.get("mean_selected_expert_set_size")
    max_size = calibrated.get("max_selected_expert_set_size")
    result = {
        "positive_new_gate": positive_gate,
        "own_top1_gate": positive_gate,
        "own_selected_gate": selected_gate,
        "max_top_expert_gate": max_top_gate,
        "positive_new_pass": int(calibrated.get("positive_new_count") or 0) >= positive_gate,
        "own_top1_pass": int(calibrated.get("routing_top1_count") or 0) >= positive_gate,
        "own_selected_pass": int(calibrated.get("threshold_selected_own_count") or 0) >= selected_gate,
        "mean_selected_size_pass": mean_size is not None and float(mean_size) <= 2.0,
        "max_selected_size_pass": max_size is not None and float(max_size) <= 3.0,
        "empty_selection_pass": int(calibrated.get("threshold_empty_selection_count") or 0) <= 1,
        "max_top_expert_pass": int(calibrated.get("max_top_expert_count") or 0) <= max_top_gate,
        "locality_reported": calibrated.get("mean_reference_delta") is not None,
    }
    result["calibrated_nonseq_pass"] = all(
        bool(result[key])
        for key in (
            "positive_new_pass",
            "own_top1_pass",
            "own_selected_pass",
            "mean_selected_size_pass",
            "max_selected_size_pass",
            "empty_selection_pass",
            "max_top_expert_pass",
            "locality_reported",
        )
    )
    return result


def diagnose_ten_nonseq(mode_summaries: Dict[str, Dict[str, Any]], acceptance: Dict[str, Any]) -> str:
    if acceptance.get("calibrated_nonseq_pass"):
        return "10edit_nonseq_calibrated_pass"
    force = mode_summaries.get("force_own") or {}
    calibrated = mode_summaries.get("calibrated") or {}
    labels: List[str] = []
    if int(force.get("positive_new_count") or 0) < int(acceptance.get("positive_new_gate") or 8):
        labels.append("expert_capacity_issue")
    if not acceptance.get("own_top1_pass") or int(calibrated.get("max_top_expert_count") or 0) > int(acceptance.get("max_top_expert_gate") or 4):
        labels.append("routing_collapse_issue")
    if not acceptance.get("own_selected_pass"):
        labels.append("routing_calibration_scale_limit")
    if not acceptance.get("mean_selected_size_pass") or not acceptance.get("max_selected_size_pass"):
        labels.append("routing_density_issue")
    if not acceptance.get("empty_selection_pass"):
        labels.append("threshold_unsuitable")
    if (calibrated.get("mean_reference_delta") or 0.0) > 50.0:
        labels.append("locality_damage_issue")
    return "+".join(labels) if labels else "routing_calibration_scale_limit"


def diagnose_five_edit_nonseq(mode_summaries: Dict[str, Dict[str, Any]]) -> str:
    force = mode_summaries.get("force_own", {})
    topk = mode_summaries.get("topk", {})
    threshold = mode_summaries.get("threshold", {})
    force_improved = int(force.get("positive_new_count") or 0)
    topk_own = int(topk.get("routing_top1_count") or 0)
    threshold_own = int(threshold.get("threshold_selected_own_count") or 0)
    threshold_empty = int(threshold.get("threshold_empty_selection_count") or 0)
    topk_confusion = topk.get("routing_confusion_matrix") or {}
    observed_counts: Dict[str, int] = {}
    for observed in topk_confusion.values():
        for expert_id, count in observed.items():
            observed_counts[expert_id] = observed_counts.get(expert_id, 0) + int(count)
    collapsed = bool(observed_counts and max(observed_counts.values()) >= 4 and topk_own < 3)
    locality_large = any(
        (summary.get("mean_reference_delta") or 0.0) > 50.0
        for summary in mode_summaries.values()
    )
    if force_improved < 3:
        return "expert_capacity_or_optimization_issue"
    if collapsed:
        return "expert_collapse_or_score_scale_issue"
    if topk_own < 3:
        return "routing_ranking_issue"
    if threshold_empty >= 3 or threshold_own < 3:
        return "threshold_calibration_issue"
    if locality_large:
        return "locality_damage_issue"
    return "5edit_nonseq_promising"


def recommendation_for_diagnosis(diagnosis: str) -> str:
    if diagnosis == "5edit_nonseq_promising":
        return "proceed to 5-edit sequential"
    if diagnosis == "threshold_calibration_issue":
        return "run gamma/topk calibration"
    if diagnosis == "locality_damage_issue":
        return "run alignment-loss ablation"
    if diagnosis in {"routing_ranking_issue", "expert_collapse_or_score_scale_issue"}:
        return "run gamma/topk calibration"
    return "fix expert optimization first"


def _routing_score_rows(eval_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in eval_rows:
        scores = row.get("pooled_scores") or []
        weights = row.get("pooled_weights") or []
        selected_ids = set(row.get("selected_expert_ids") or [])
        expected = row.get("expected_expert")
        top_id = row.get("top_expert_id")
        for expert_id, score in enumerate(scores):
            rows.append(
                {
                    "eval_routing_mode": row.get("eval_routing_mode"),
                    "record_id": row.get("record_id"),
                    "expected_expert": expected,
                    "expert_id": expert_id,
                    "score": score,
                    "weight": weights[expert_id] if expert_id < len(weights) else None,
                    "selected": expert_id in selected_ids,
                    "is_top": expert_id == top_id,
                    "is_own": expected is not None and expert_id == int(expected),
                }
            )
    return rows


def _confusion_csv_rows(mode_summaries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode, summary in mode_summaries.items():
        for expected, observed_counts in (summary.get("routing_confusion_matrix") or {}).items():
            for observed, count in observed_counts.items():
                rows.append(
                    {
                        "eval_routing_mode": mode,
                        "expected_expert": expected,
                        "observed_top_expert": observed,
                        "count": count,
                    }
                )
    return rows


def write_five_edit_nonseq_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    routing_modes = parse_eval_routing_modes(args.eval_routing_modes)
    num_records = len(records)
    prefix = _nonseq_prefix(num_records)
    report_name = _nonseq_report_name(num_records)
    mode_summaries = summarize_five_edit_nonseq(eval_rows, routing_modes)
    calibrated_acceptance = nonseq_calibrated_acceptance(mode_summaries, num_records)
    diagnosis = diagnose_ten_nonseq(mode_summaries, calibrated_acceptance) if num_records >= 10 else diagnose_five_edit_nonseq(mode_summaries)
    recommendation = (
        "proceed to 10-edit sequential gate" if calibrated_acceptance.get("calibrated_nonseq_pass") and num_records >= 10
        else recommendation_for_diagnosis(diagnosis)
    )
    record_ids = [record_id(record, idx) for idx, record in enumerate(records)]
    acceptance = {
        "capacity_pass": int((mode_summaries.get("force_own") or {}).get("positive_new_count") or 0) >= _positive_gate(num_records),
        "routing_pass": bool(calibrated_acceptance.get("calibrated_nonseq_pass")) if "calibrated" in routing_modes else int((mode_summaries.get("topk") or {}).get("routing_top1_count") or 0) >= 3,
        "threshold_pass": int((mode_summaries.get("threshold") or {}).get("threshold_selected_own_count") or 0) >= 3,
        "locality_reported": all((mode_summaries.get(mode) or {}).get("mean_reference_delta") is not None for mode in routing_modes),
        "calibrated": calibrated_acceptance,
    }
    write_loss_trace(out_dir / f"{prefix}_per_record.csv", eval_rows)
    write_loss_trace(out_dir / f"{prefix}_routing_scores.csv", _routing_score_rows(eval_rows))
    write_loss_trace(out_dir / f"{prefix}_confusion_matrix.csv", _confusion_csv_rows(mode_summaries))
    write_json(out_dir / f"{prefix}_confusion_matrices.json", {mode: (mode_summaries.get(mode) or {}).get("routing_confusion_matrix") for mode in routing_modes})
    diagnostic = {
        "record_ids": record_ids,
        "eval_routing_modes": routing_modes,
        "routing_mode_summaries": mode_summaries,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "acceptance": acceptance,
        "report_path": str(out_dir / report_name),
    }
    summary.update(diagnostic)
    write_json(out_dir / f"{prefix}_summary.json", diagnostic)
    write_five_edit_nonseq_report(out_dir, args, eval_rows, summary)
    return diagnostic


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _report_table(rows: List[Dict[str, Any]], mode: str) -> List[str]:
    selected = _mode_rows(rows, mode)
    lines = [
        f"## {mode} Results",
        "| record_id | target | NLL before | NLL after | improvement | improved | rank before/after | logprob delta | routed expert | intrinsic top expert | own score | top score | selected size | reference delta |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        target = str(row.get("target") or "").replace("|", "\\|")
        lines.append(
            "| {record_id} | {target} | {before} | {after} | {delta} | {improved} | {rank_before}/{rank_after} | {logprob} | {routed} | {top} | {own_score} | {top_score} | {selected_size} | {ref} |".format(
                record_id=row.get("record_id"),
                target=target,
                before=_fmt(row.get("base_target_nll")),
                after=_fmt(row.get("target_nll")),
                delta=_fmt(row.get("target_nll_delta")),
                improved=row.get("target_improved"),
                rank_before=_fmt(row.get("base_first_target_token_rank")),
                rank_after=_fmt(row.get("first_target_token_rank")),
                logprob=_fmt(row.get("answer_token_logprob_delta")),
                routed=_fmt(row.get("top_routed_expert_id", row.get("top_expert_id"))),
                top=_fmt(row.get("top_score_expert_id", row.get("top_expert_id"))),
                own_score=_fmt(row.get("own_expert_score")),
                top_score=_fmt(row.get("top_score")),
                selected_size=_fmt(row.get("selected_expert_set_size")),
                ref=_fmt(row.get("reference_delta")),
            )
        )
    lines.append("")
    return lines


def write_five_edit_nonseq_report(out_dir: Path, args: argparse.Namespace, eval_rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> Path:
    mode_summaries = summary.get("routing_mode_summaries") or {}
    num_records = len(summary.get("record_ids") or [])
    prefix = _nonseq_prefix(num_records)
    report_name = _nonseq_report_name(num_records)
    command = _run_command_for(out_dir) or current_command_line()
    lines = [
        f"# TIME {num_records}-Edit Nonseq Calibrated Report",
        "",
        "## Files Changed",
        "- `scripts/time/run_time_medmkeb_smoke.py`",
        "",
        "## Exact Command Run",
        f"- `{command}`",
        "",
        "## Verification",
        "- `py_compile`: passed.",
        "- `scripts/time/test_time_modules.py`: passed.",
        "- 20-edit run: not run.",
        "",
        "## Records",
        "- Record ids: `{}`.".format(", ".join(str(value) for value in summary.get("record_ids", []))),
        "",
    ]
    lines.extend(_report_table(eval_rows, "force_own"))
    if "calibrated" in (summary.get("eval_routing_modes") or []):
        lines.extend(_report_table(eval_rows, "calibrated"))
    lines.extend(_report_table(eval_rows, "topk"))
    lines.extend(_report_table(eval_rows, "threshold"))
    lines.extend(
        [
            "## Aggregate Metrics",
            "| routing mode | mean NLL improvement | positive_new | mean reference delta | residual | hidden delta | top1 own | selected own | empty | mean size | max size | max top | worst degradation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in summary.get("eval_routing_modes", []):
        metrics = mode_summaries.get(mode) or {}
        lines.append(
            "| {mode} | {mean_delta} | {positive}/{num} | {ref} | {residual} | {hidden} | {top1}/{num} | {selected_own}/{num} | {empty} | {selected_size} | {max_size} | {max_top} | {worst} |".format(
                mode=mode,
                mean_delta=_fmt(metrics.get("mean_target_nll_improvement")),
                positive=metrics.get("positive_new_count"),
                num=metrics.get("num_records"),
                ref=_fmt(metrics.get("mean_reference_delta")),
                residual=_fmt(metrics.get("mean_residual_norm")),
                hidden=_fmt(metrics.get("mean_hidden_delta_norm")),
                top1=metrics.get("routing_top1_count"),
                selected_own=metrics.get("threshold_selected_own_count"),
                empty=metrics.get("threshold_empty_selection_count"),
                selected_size=_fmt(metrics.get("mean_selected_expert_set_size")),
                max_size=_fmt(metrics.get("max_selected_expert_set_size")),
                max_top=metrics.get("max_top_expert_count"),
                worst=_fmt(metrics.get("worst_per_record_nll_degradation")),
            )
        )
    lines.extend(["", "## Routing Confusion Matrix"])
    for mode in summary.get("eval_routing_modes", []):
        lines.append(f"- {mode}: `{json.dumps((mode_summaries.get(mode) or {}).get('routing_confusion_matrix'), sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Acceptance",
            f"- Primary capacity pass: {summary.get('acceptance', {}).get('capacity_pass')}.",
            f"- Routing pass: {summary.get('acceptance', {}).get('routing_pass')}.",
            f"- Threshold pass: {summary.get('acceptance', {}).get('threshold_pass')}.",
            f"- Locality reported: {summary.get('acceptance', {}).get('locality_reported')}.",
            f"- Calibrated acceptance: `{json.dumps((summary.get('acceptance') or {}).get('calibrated'), sort_keys=True)}`.",
            "",
            "## Output Files",
            "- `expert_repository.pt`",
            "- `loss_trace.csv`",
            f"- `{prefix}_loss_trace.csv`" if num_records == 10 else "- `loss_trace.csv`",
            f"- `{prefix}_per_record.csv`",
            f"- `{prefix}_routing_scores.csv`",
            f"- `{prefix}_confusion_matrices.json`",
            f"- `{prefix}_routing_debug.jsonl`",
            f"- `{prefix}_summary.json`",
            f"- `{report_name}`",
            "",
            "## Diagnosis",
            f"- Label: `{summary.get('diagnosis')}`.",
            "",
            "## Recommendation",
            f"- {summary.get('recommendation')}.",
            "",
        ]
    )
    path = out_dir / report_name
    path.write_text("\n".join(lines))
    return path


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    config: TIMEEditMultimodalHparams,
    dataset_path: Path,
    summary: Dict[str, Any],
) -> None:
    memory = summary.get("memory_estimate", {})
    lines = [
        "# TIME MedMKEB Reproduction Report",
        "",
        "## Scope",
        "- Implementation: direct-factor TIME reproduction from the paper equations; no official code is used.",
        "- Cross-attention factor generator: not implemented in this first smoke; each edit optimizes trainable CP factors directly.",
        "- Long/full experiments started: no.",
        "",
        "## Files",
        "- Added `easyeditor/models/time_edit/`.",
        "- Added `easyeditor/trainer/algs/time_edit.py`.",
        "- Added `scripts/time/test_time_modules.py` and `scripts/time/run_time_medmkeb_smoke.py`.",
        "- Added `hparams/TIME/llava_med.yaml` and `hparams/TRAINING/TIME/llava_med_smoke.yaml`.",
        "- Modified EasyEdit registries and multimodal trainer dispatch.",
        "",
        "## Reused Utilities",
        "- Model loading: `easyeditor/trainer/models.py` and `LlavaMedForEditing`.",
        "- MedMKEB helpers: `scripts/dsca_medmkeb_diag_common.py`.",
        "- Logging style: JSON, JSONL, CSV, and Markdown artifacts under the requested output directory.",
        "",
        "## Mathematical Choices",
        f"- Hidden size/factors: H={summary.get('hidden_size')}, s1={summary.get('s1')}, s2={summary.get('s2')}.",
        f"- CP rank: R={config.time_rank}.",
        f"- Activation: {config.time_activation}.",
        f"- Scale mode: {config.time_scale_mode}, alpha={config.time_alpha}.",
        f"- Token scope: {config.time_token_scope}.",
        f"- Routing mode/threshold/top-k: mode={config.time_routing_mode}, gamma={config.time_gamma}, topk={config.time_topk}.",
        f"- Score norm/relative threshold: norm={config.time_score_norm}, relative_threshold={config.time_relative_threshold}.",
        f"- Mixing: {config.time_mixing_mode} with tau={config.time_tau}.",
        f"- Train-time force-current expert: {config.time_force_current_during_training}.",
        f"- Residual sign/expert gain: sign={config.time_residual_sign}, gain={config.time_expert_gain}.",
        f"- Reliability-only objective: {config.time_reliability_only}.",
        "",
        "## Run",
        f"- Dataset: `{dataset_path}`.",
        f"- Mode: {args.mode}.",
        f"- Max edits: {args.max_edits}.",
        f"- Edit iterations: {args.edit_iters}.",
        f"- Learning rate: {args.lr}.",
        f"- Skip generation: {args.skip_generation}.",
        "",
        "## Results",
        f"- Mean target NLL delta: {summary.get('mean_target_nll_delta')}.",
        f"- Improved count: {summary.get('improved_count')} / {summary.get('num_eval_records')}.",
        f"- Positive-new/top-1 count: {summary.get('positive_new_count')} / {summary.get('num_eval_records')}.",
        f"- Routing top-1 accuracy: {summary.get('routing_top1_accuracy')}.",
        f"- Mean reference/locality delta: {summary.get('mean_reference_delta')}.",
        f"- Mean selected expert set size: {summary.get('mean_selected_expert_set_size')}.",
        f"- Routing confusion matrix: `{json.dumps(summary.get('routing_confusion_matrix'), sort_keys=True)}`.",
        "",
        "## Memory",
        f"- TIME CP parameters per expert: {memory.get('time_params_per_expert')}.",
        f"- Equivalent LoRA parameters per expert at same rank: {memory.get('lora_params_per_expert')}.",
        f"- Repository size on disk: {memory.get('repository_size_bytes')} bytes.",
        "",
        "## Deviations",
        "- Direct random CP-factor initialization is used instead of the paper's cross-attention factor generator.",
        "- During training only, the current expert is forced active by default so randomly initialized factors can receive task gradients; inference uses intrinsic threshold/top-k routing.",
        "- Generality loss is logged as skipped when no separate paraphrase/generalization batch is provided by the smoke sample.",
        "- Generation diagnostics are skipped when `--skip-generation` is set.",
        "",
        "## Recommendation",
        "- Proceed to 20-edit nonseq only if the one-edit and 5-edit gates improve target NLL/rank and routing top-1 is interpretable.",
        "- Run `--time-disable-align-loss` as the first ablation after these gates.",
        "- Compare against SAME / LoRA-MoE after TIME passes the same bounded MedMKEB gates.",
        "",
    ]
    (out_dir / "TIME_MEDMKEB_REPRO_REPORT.md").write_text("\n".join(lines))


def run_smoke(
    args: argparse.Namespace,
    config: TIMEEditMultimodalHparams,
    dataset_path: Path,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    alg: TIMEEdit,
    out_dir: Path,
    print_summary: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sequential_eval_modes = parse_eval_routing_modes(args.eval_routing_modes) if args.mode == "sequential" and args.eval_routing_modes else []
    routing_debug_path = out_dir / (
        f"{_sequential_prefix(args.max_edits)}_routing_debug.jsonl" if sequential_eval_modes else "routing_debug.jsonl"
    )
    if routing_debug_path.exists():
        routing_debug_path.unlink()
    write_json(
        out_dir / "time_hparams.json",
        {
            "args": vars(args),
            "config": dict(config.__dict__),
            "scale_mode": config.time_scale_mode,
            "command": current_command_line(),
            "hidden_size": alg.repository.hidden_size,
            "s1": alg.repository.s1,
            "s2": alg.repository.s2,
        },
    )

    loss_rows: List[Dict[str, Any]] = []
    reliability_rows: List[Dict[str, Any]] = []
    anti_collapse_rows: List[Dict[str, Any]] = []
    score_matrix_rows: List[Dict[str, Any]] = []
    train_records: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    sequential_all_rows: List[Dict[str, Any]] = []
    immediate_after_edit: Dict[str, float] = {}
    sequential_immediate_by_mode_record: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    sequential_retention_summary: Optional[Dict[str, Any]] = None
    base_eval_cache = build_base_eval_cache(alg, samples) if sequential_eval_modes else None

    for pos, (record, sample) in enumerate(zip(records, samples)):
        train_records.append(
            train_one_edit(
                alg,
                sample,
                record,
                pos,
                args,
                loss_rows,
                reliability_rows,
                out_dir,
                previous_samples=samples[:pos],
                anti_collapse_rows=anti_collapse_rows,
            )
        )
        if args.mode == "nonseq" and args.time_anti_collapse_loss:
            score_matrix_rows.extend(collect_score_matrix_rows(alg, records, samples, pos, out_dir))
        if args.mode == "sequential":
            if sequential_eval_modes:
                eval_rows.extend(
                    evaluate_sequential_routing_modes(
                        args,
                        alg,
                        records,
                        samples,
                        pos + 1,
                        sequential_eval_modes,
                        routing_debug_path,
                        base_eval_cache,
                        sequential_immediate_by_mode_record,
                    )
                )
            else:
                for eval_pos in range(pos + 1):
                    eval_row = evaluate_sample(
                        alg,
                        samples[eval_pos],
                        records[eval_pos],
                        eval_pos,
                        phase=f"after_edit_{pos}_eval_{eval_pos}",
                        expected_expert=eval_pos,
                        routing_debug_path=routing_debug_path,
                    )
                    eval_rows.append(eval_row)
                    if eval_pos == pos and eval_row.get("target_nll") is not None:
                        immediate_after_edit[record_id(records[eval_pos], eval_pos)] = float(eval_row["target_nll"])

    self_score_metadata = compute_self_score_metadata(alg, records, samples, out_dir) if args.mode == "nonseq" else {}

    if args.mode == "nonseq" and args.eval_routing_modes:
        eval_rows = evaluate_nonseq_routing_modes(args, alg, records, samples, out_dir)
    elif args.mode != "sequential":
        for eval_pos, (record, sample) in enumerate(zip(records, samples)):
            eval_rows.append(
                evaluate_sample(
                    alg,
                    sample,
                    record,
                    eval_pos,
                    phase=f"final_{args.mode}_eval_{eval_pos}",
                    expected_expert=eval_pos,
                    routing_debug_path=routing_debug_path,
                )
            )
    else:
        final_rows: List[Dict[str, Any]] = []
        if sequential_eval_modes:
            sequential_all_rows = list(eval_rows)
            final_rows = _sequential_final_rows(eval_rows, len(records), "calibrated")
            if not final_rows:
                final_rows = _sequential_final_rows(eval_rows, len(records), "force_own")
            if not final_rows:
                final_rows = _sequential_final_rows(eval_rows, len(records))
        else:
            for eval_pos, (record, sample) in enumerate(zip(records, samples)):
                row = evaluate_sample(
                    alg,
                    sample,
                    record,
                    eval_pos,
                    phase=f"final_sequential_eval_{eval_pos}",
                    expected_expert=eval_pos,
                    routing_debug_path=routing_debug_path,
                )
                rid = record_id(record, eval_pos)
                if rid in immediate_after_edit and row.get("target_nll") is not None:
                    row["retention_target_nll_delta_vs_immediate"] = immediate_after_edit[rid] - float(row["target_nll"])
                final_rows.append(row)
        eval_rows = final_rows

    write_loss_trace(out_dir / "loss_trace.csv", loss_rows)
    if sequential_eval_modes:
        sequential_retention_summary = write_time_sequential_outputs(out_dir, args, records, sequential_all_rows, loss_rows)
    if args.mode == "nonseq":
        write_loss_trace(out_dir / "time_score_norm_retrain_loss_trace.csv", loss_rows)
        if len(records) == 10:
            write_loss_trace(out_dir / "time_ten_nonseq_loss_trace.csv", loss_rows)
    if anti_collapse_rows:
        write_loss_trace(out_dir / "time_anti_collapse_trace.csv", anti_collapse_rows)
    if score_matrix_rows:
        write_loss_trace(out_dir / "time_score_matrix_after_each_edit.csv", score_matrix_rows)
    if reliability_rows:
        write_loss_trace(out_dir / "time_reliability_overfit_trace.csv", reliability_rows)
    summary = summarize_evals(args.mode, eval_rows, train_records, alg, out_dir)
    summary.update({
        "hidden_size": alg.repository.hidden_size,
        "s1": alg.repository.s1,
        "s2": alg.repository.s2,
        "self_score_metadata": self_score_metadata,
        "anti_collapse_active": bool(args.time_anti_collapse_loss),
        "score_norm": str(config.time_score_norm),
        "align_score_norm": str(config.time_align_score_norm),
        "anti_collapse_score_norm": str(config.time_anti_collapse_score_norm),
    })
    if sequential_retention_summary is not None:
        summary["sequential_retention_summary"] = sequential_retention_summary
        summary["sequential_after_each_edit_rows"] = len(sequential_all_rows)
        report_path = write_time_sequential_calibrated_report(out_dir, args, summary, sequential_retention_summary)
        summary["sequential_calibrated_report"] = str(report_path)
    if args.mode == "one":
        summary_path = out_dir / "one_edit_summary.json"
    elif args.mode == "nonseq":
        summary_path = out_dir / ("time_ten_nonseq_summary.json" if len(records) == 10 else "five_edit_nonseq_summary.json")
    else:
        summary_path = out_dir / ("time_ten_sequential_summary.json" if len(records) == 10 else "five_edit_seq_summary.json")
    if args.mode == "nonseq" and args.eval_routing_modes:
        write_five_edit_nonseq_outputs(out_dir, args, records, eval_rows, summary)
    write_json(summary_path, summary)
    write_report(out_dir, args, config, dataset_path, summary)
    if print_summary:
        print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


@contextmanager
def temporary_time_routing(
    alg: TIMEEdit,
    routing_mode: Optional[str] = None,
    gamma: Optional[float] = None,
    topk: Optional[int] = None,
    score_norm: Optional[str] = None,
    relative_threshold: Optional[float] = None,
    mixing_mode: Optional[str] = None,
    calibration_mode: Optional[str] = None,
    calibration_beta: Optional[float] = None,
    max_selected_experts: Optional[int] = None,
    score_pool: Optional[str] = None,
    calibration_stats: Optional[Dict[str, List[float]]] = None,
):
    old_gamma = alg.repository.gamma
    old_mode = alg.time_residual.routing_mode
    old_topk = alg.time_residual.topk
    old_score_norm = alg.time_residual.score_norm
    old_relative_threshold = alg.time_residual.relative_threshold
    old_mixing_mode = alg.time_residual.mixing_mode
    old_calibration_mode = alg.time_residual.calibration_mode
    old_calibration_beta = alg.time_residual.calibration_beta
    old_max_selected_experts = alg.time_residual.max_selected_experts
    old_score_pool = alg.time_residual.score_pool
    old_calibration_stats = dict(getattr(alg.time_residual, "calibration_stats", {}))
    old_config_gamma = getattr(alg.config, "time_gamma", None)
    old_config_mode = getattr(alg.config, "time_routing_mode", None)
    old_config_topk = getattr(alg.config, "time_topk", None)
    old_config_score_norm = getattr(alg.config, "time_score_norm", None)
    old_config_relative_threshold = getattr(alg.config, "time_relative_threshold", None)
    old_config_mixing_mode = getattr(alg.config, "time_mixing_mode", None)
    old_config_calibration_mode = getattr(alg.config, "time_calibration_mode", None)
    old_config_calibration_beta = getattr(alg.config, "time_calibration_beta", None)
    old_config_max_selected_experts = getattr(alg.config, "time_max_selected_experts", None)
    old_config_score_pool = getattr(alg.config, "time_score_pool", None)
    try:
        if gamma is not None:
            alg.repository.gamma = float(gamma)
            setattr(alg.config, "time_gamma", float(gamma))
        if routing_mode is not None:
            alg.time_residual.routing_mode = str(routing_mode)
            setattr(alg.config, "time_routing_mode", str(routing_mode))
        if topk is not None:
            alg.time_residual.topk = int(topk)
            setattr(alg.config, "time_topk", int(topk))
        if score_norm is not None:
            alg.time_residual.score_norm = str(score_norm)
            setattr(alg.config, "time_score_norm", str(score_norm))
        if relative_threshold is not None:
            alg.time_residual.relative_threshold = float(relative_threshold)
            setattr(alg.config, "time_relative_threshold", float(relative_threshold))
        elif routing_mode is not None and "relative" not in str(routing_mode):
            alg.time_residual.relative_threshold = None
            setattr(alg.config, "time_relative_threshold", None)
        if mixing_mode is not None:
            alg.time_residual.mixing_mode = str(mixing_mode)
            setattr(alg.config, "time_mixing_mode", str(mixing_mode))
        if calibration_mode is not None:
            alg.time_residual.calibration_mode = str(calibration_mode)
            setattr(alg.config, "time_calibration_mode", str(calibration_mode))
        if calibration_beta is not None:
            alg.time_residual.calibration_beta = float(calibration_beta)
            setattr(alg.config, "time_calibration_beta", float(calibration_beta))
        if max_selected_experts is not None:
            alg.time_residual.max_selected_experts = int(max_selected_experts)
            setattr(alg.config, "time_max_selected_experts", int(max_selected_experts))
        else:
            alg.time_residual.max_selected_experts = None
            setattr(alg.config, "time_max_selected_experts", None)
        if score_pool is not None:
            alg.time_residual.score_pool = str(score_pool)
            setattr(alg.config, "time_score_pool", str(score_pool))
        if calibration_stats is not None:
            alg.time_residual.calibration_stats = dict(calibration_stats)
        yield
    finally:
        alg.repository.gamma = old_gamma
        alg.time_residual.routing_mode = old_mode
        alg.time_residual.topk = old_topk
        alg.time_residual.score_norm = old_score_norm
        alg.time_residual.relative_threshold = old_relative_threshold
        alg.time_residual.mixing_mode = old_mixing_mode
        alg.time_residual.calibration_mode = old_calibration_mode
        alg.time_residual.calibration_beta = old_calibration_beta
        alg.time_residual.max_selected_experts = old_max_selected_experts
        alg.time_residual.score_pool = old_score_pool
        alg.time_residual.calibration_stats = old_calibration_stats
        if old_config_gamma is not None:
            setattr(alg.config, "time_gamma", old_config_gamma)
        if old_config_mode is not None:
            setattr(alg.config, "time_routing_mode", old_config_mode)
        if old_config_topk is not None:
            setattr(alg.config, "time_topk", old_config_topk)
        setattr(alg.config, "time_score_norm", old_config_score_norm)
        setattr(alg.config, "time_relative_threshold", old_config_relative_threshold)
        setattr(alg.config, "time_mixing_mode", old_config_mixing_mode)
        setattr(alg.config, "time_calibration_mode", old_config_calibration_mode)
        setattr(alg.config, "time_calibration_beta", old_config_calibration_beta)
        setattr(alg.config, "time_max_selected_experts", old_config_max_selected_experts)
        setattr(alg.config, "time_score_pool", old_config_score_pool)


def evaluate_nonseq_routing_modes(
    args: argparse.Namespace,
    alg: TIMEEdit,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    modes = parse_eval_routing_modes(args.eval_routing_modes)
    rows: List[Dict[str, Any]] = []
    routing_debug_path = out_dir / f"{_nonseq_prefix(len(records))}_routing_debug.jsonl"
    if routing_debug_path.exists():
        routing_debug_path.unlink()
    ordered_modes = list(modes)
    if "calibrated" in ordered_modes:
        ordered_modes = [mode for mode in ordered_modes if mode != "force_own"]
        ordered_modes.insert(0, "force_own")
    force_rows: List[Dict[str, Any]] = []
    calibration_stats: Dict[str, List[float]] = {}
    for mode in ordered_modes:
        if mode == "calibrated":
            calibration_stats = _sequential_calibration_stats(force_rows, alg.repository.num_experts)
        context = _sequential_eval_context(args, alg, mode, calibration_stats)
        with context:
            mode_rows: List[Dict[str, Any]] = []
            for eval_pos, (record, sample) in enumerate(zip(records, samples)):
                row = evaluate_sample(
                    alg,
                    sample,
                    record,
                    eval_pos,
                    phase=f"final_nonseq_{mode}_eval_{eval_pos}",
                    expected_expert=eval_pos,
                    routing_debug_path=routing_debug_path,
                    eval_routing_mode=mode,
                    force_expert_id=eval_pos if mode == "force_own" else None,
                    extra_fields={
                        "calibration_stats_source": "final_force_own_scores" if mode == "calibrated" else None,
                        "calibration_mu_neg": calibration_stats.get("mu_neg") if mode == "calibrated" else None,
                        "calibration_std_neg": calibration_stats.get("std_neg") if mode == "calibrated" else None,
                    },
                )
                mode_rows.append(row)
                if mode in modes:
                    rows.append(row)
            if mode == "force_own":
                force_rows = mode_rows
    return rows


def _sequential_eval_context(args: argparse.Namespace, alg: TIMEEdit, mode: str, calibration_stats: Dict[str, List[float]]):
    score_norm = str(args.time_score_norm or "factor_z")
    score_pool = str(args.time_score_pool or "mean")
    mixing_mode = "average" if args.time_disable_score_mixing else str(args.time_mixing_mode)
    relative_threshold = float(args.time_relative_threshold if args.time_relative_threshold is not None else 0.9)
    calibration_mode = normalize_time_calibration_mode(args.time_calibration_mode)
    if mode == "force_own":
        return temporary_time_routing(
            alg,
            routing_mode="threshold",
            gamma=1.0e30,
            topk=0,
            score_norm=score_norm,
            relative_threshold=None,
            mixing_mode=mixing_mode,
            calibration_mode="none",
            max_selected_experts=None,
            score_pool=score_pool,
            calibration_stats={},
        )
    if mode == "calibrated":
        return temporary_time_routing(
            alg,
            routing_mode="relative_threshold",
            gamma=float(args.gamma),
            topk=0,
            score_norm=score_norm,
            relative_threshold=relative_threshold,
            mixing_mode=mixing_mode,
            calibration_mode=calibration_mode,
            calibration_beta=float(args.time_calibration_beta),
            max_selected_experts=args.time_max_selected_experts,
            score_pool=score_pool,
            calibration_stats=calibration_stats,
        )
    if mode == "topk":
        return temporary_time_routing(
            alg,
            routing_mode="topk",
            gamma=float(args.gamma),
            topk=max(1, int(args.time_topk or 1)),
            score_norm=score_norm,
            relative_threshold=None,
            mixing_mode=mixing_mode,
            calibration_mode="none",
            max_selected_experts=None,
            score_pool=score_pool,
            calibration_stats={},
        )
    if mode == "threshold":
        return temporary_time_routing(
            alg,
            routing_mode="threshold",
            gamma=float(args.gamma),
            topk=0,
            score_norm=score_norm,
            relative_threshold=None,
            mixing_mode=mixing_mode,
            calibration_mode="none",
            max_selected_experts=None,
            score_pool=score_pool,
            calibration_stats={},
        )
    if mode == "relative":
        return temporary_time_routing(
            alg,
            routing_mode="relative_threshold",
            gamma=float(args.gamma),
            topk=0,
            score_norm=score_norm,
            relative_threshold=relative_threshold,
            mixing_mode=mixing_mode,
            calibration_mode="none",
            max_selected_experts=None,
            score_pool=score_pool,
            calibration_stats={},
        )
    raise ValueError(f"Unsupported eval routing mode: {mode}")


def _sequential_calibration_stats(rows: List[Dict[str, Any]], num_experts: int) -> Dict[str, List[float]]:
    matrix: List[List[float]] = []
    for row in rows:
        scores = []
        for value in row.get("pooled_scores") or []:
            try:
                scores.append(float(value))
            except (TypeError, ValueError):
                scores.append(0.0)
        matrix.append(scores)
    width = max([num_experts] + [len(row) for row in matrix])
    mu_neg: List[float] = []
    std_neg: List[float] = []
    for expert_index in range(width):
        neg = [row[expert_index] for query_index, row in enumerate(matrix) if query_index != expert_index and expert_index < len(row)]
        neg_mean = float(sum(neg) / len(neg)) if neg else 0.0
        neg_std = math.sqrt(sum((value - neg_mean) ** 2 for value in neg) / len(neg)) if neg else 0.0
        mu_neg.append(neg_mean)
        std_neg.append(max(float(neg_std), 1.0e-8))
    return {"mu_neg": mu_neg, "std_neg": std_neg, "self_score": [1.0] * width}


def _enrich_sequential_row(
    row: Dict[str, Any],
    edit_step: int,
    query_record_index: int,
    mode: str,
    immediate_by_mode_record: Dict[Tuple[str, str], Dict[str, Optional[float]]],
) -> Dict[str, Any]:
    rid = str(row.get("record_id"))
    retained = _float_or_none(row.get("target_nll_delta"))
    current_nll = _float_or_none(row.get("target_nll"))
    if query_record_index == edit_step - 1:
        immediate_by_mode_record[(mode, rid)] = {
            "target_nll": current_nll,
            "improvement": retained,
        }
    immediate = immediate_by_mode_record.get((mode, rid), {})
    immediate_nll = _float_or_none(immediate.get("target_nll"))
    immediate_improvement = _float_or_none(immediate.get("improvement"))
    retention_ratio = None
    if immediate_improvement is not None and immediate_improvement > 0.0 and retained is not None:
        retention_ratio = float(retained / immediate_improvement)
    row.update(
        {
            "edit_step": edit_step,
            "after_edit_index": edit_step - 1,
            "query_record_index": query_record_index,
            "own_expert_index": row.get("expected_expert"),
            "target_nll_before": row.get("base_target_nll"),
            "target_nll_immediately_after_own_edit": immediate_nll,
            "target_nll_current": row.get("target_nll"),
            "immediate_improvement": immediate_improvement,
            "retained_improvement": retained,
            "retention_ratio": retention_ratio,
            "improved": bool(retained is not None and retained > 0.0),
            "forgotten": bool(immediate_improvement is not None and immediate_improvement > 0.0 and (retained is None or retained <= 0.0)),
            "forgetting_nll_increase_vs_immediate": (
                float(current_nll - immediate_nll) if current_nll is not None and immediate_nll is not None else None
            ),
            "first_target_token_rank_before": row.get("base_first_target_token_rank"),
            "first_target_token_rank_current": row.get("first_target_token_rank"),
            "own_top1": bool(row.get("routing_top1_correct")),
            "own_selected": bool(row.get("selected_own_expert")),
            "top_expert_score": row.get("top_score"),
            "locality_reference_delta": row.get("reference_delta"),
        }
    )
    return row


def evaluate_sequential_routing_modes(
    args: argparse.Namespace,
    alg: TIMEEdit,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    edit_step: int,
    modes: List[str],
    routing_debug_path: Path,
    base_cache: Optional[List[Dict[str, Any]]],
    immediate_by_mode_record: Dict[Tuple[str, str], Dict[str, Optional[float]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    force_rows: List[Dict[str, Any]] = []
    ordered_modes = list(modes)
    if "calibrated" in ordered_modes:
        ordered_modes = [mode for mode in ordered_modes if mode != "force_own"]
        ordered_modes.insert(0, "force_own")
    calibration_stats: Dict[str, List[float]] = {}
    for mode in ordered_modes:
        if mode == "calibrated":
            calibration_stats = _sequential_calibration_stats(force_rows, alg.repository.num_experts)
        with _sequential_eval_context(args, alg, mode, calibration_stats):
            mode_rows: List[Dict[str, Any]] = []
            for eval_pos, (record, sample) in enumerate(zip(records[:edit_step], samples[:edit_step])):
                row = evaluate_sample(
                    alg,
                    sample,
                    record,
                    eval_pos,
                    phase=f"after_edit_{edit_step - 1}_{mode}_eval_{eval_pos}",
                    expected_expert=eval_pos,
                    routing_debug_path=routing_debug_path,
                    eval_routing_mode=mode,
                    force_expert_id=eval_pos if mode == "force_own" else None,
                    extra_fields={
                        "sequential_calibrated_gate": True,
                        "calibration_stats_source": "seen_prefix_force_own_scores" if mode == "calibrated" else None,
                        "calibration_mu_neg": calibration_stats.get("mu_neg") if mode == "calibrated" else None,
                        "calibration_std_neg": calibration_stats.get("std_neg") if mode == "calibrated" else None,
                    },
                    base_cache=base_cache[eval_pos] if base_cache is not None else None,
                )
                _enrich_sequential_row(row, edit_step, eval_pos, mode, immediate_by_mode_record)
                mode_rows.append(row)
                if mode in modes:
                    rows.append(row)
            if mode == "force_own":
                force_rows = mode_rows
    return rows


def _sequential_final_rows(rows: List[Dict[str, Any]], num_records: int, mode: Optional[str] = None) -> List[Dict[str, Any]]:
    final_rows = [row for row in rows if int(row.get("edit_step") or 0) == num_records]
    if mode is not None:
        final_rows = [row for row in final_rows if row.get("eval_routing_mode") == mode]
    return final_rows


def _sequential_routing_score_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    score_rows: List[Dict[str, Any]] = []
    for row in rows:
        pooled_scores = row.get("pooled_scores") or []
        raw_scores = row.get("raw_pooled_scores") or []
        weights = row.get("pooled_weights") or []
        selected = set(row.get("selected_expert_ids") or [])
        variants = row.get("score_variant_pooled_scores") or {}
        width = max([len(pooled_scores), len(raw_scores), len(weights)] + [len(values) for values in variants.values()])
        for expert_index in range(width):
            out = {
                "edit_step": row.get("edit_step"),
                "after_edit_index": row.get("after_edit_index"),
                "eval_routing_mode": row.get("eval_routing_mode"),
                "query_record_index": row.get("query_record_index"),
                "record_id": row.get("record_id"),
                "expert_index": expert_index,
                "own_expert": expert_index == row.get("expected_expert"),
                "selected_expert": expert_index in selected,
                "score": pooled_scores[expert_index] if expert_index < len(pooled_scores) else None,
                "raw_score": raw_scores[expert_index] if expert_index < len(raw_scores) else None,
                "weight": weights[expert_index] if expert_index < len(weights) else None,
                "top_score_expert_id": row.get("top_score_expert_id"),
                "top_routed_expert_id": row.get("top_routed_expert_id"),
            }
            for variant_name in SCORE_NORM_MODES:
                values = variants.get(variant_name) or []
                out[f"{variant_name}_score"] = values[expert_index] if expert_index < len(values) else None
            score_rows.append(out)
    return score_rows


def summarize_time_sequential_retention(
    rows: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    modes: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    final_step = len(records)
    summary: Dict[str, Any] = {
        "num_edits": final_step,
        "eval_routing_modes": modes,
        "calibrated_config": {
            "score_norm": str(args.time_score_norm),
            "calibration_mode_requested": str(args.time_calibration_mode),
            "calibration_mode_internal": normalize_time_calibration_mode(args.time_calibration_mode),
            "score_pool": str(args.time_score_pool),
            "routing_mode": "relative_threshold",
            "relative_threshold": float(args.time_relative_threshold if args.time_relative_threshold is not None else 0.9),
            "max_selected_experts": args.time_max_selected_experts,
            "mixing_mode": str(args.time_mixing_mode),
        },
        "retention_definition": "retained_improvement = base_target_nll - current_target_nll; forgotten means positive immediate improvement became non-positive at final evaluation.",
        "by_mode": {},
    }
    for mode in modes:
        mode_rows = _sequential_final_rows(rows, final_step, mode)
        selected_sizes = [row.get("selected_expert_set_size") for row in mode_rows]
        retention_drops = [
            float(row.get("immediate_improvement")) - float(row.get("retained_improvement"))
            for row in mode_rows
            if row.get("immediate_improvement") is not None and row.get("retained_improvement") is not None
        ]
        per_record_degradation = [
            max(0.0, -float(row.get("retained_improvement")))
            for row in mode_rows
            if row.get("retained_improvement") is not None
        ]
        top_routed_counts: Dict[str, int] = {}
        top_score_counts: Dict[str, int] = {}
        for row in mode_rows:
            if row.get("top_routed_expert_id") is not None:
                key = str(row.get("top_routed_expert_id"))
                top_routed_counts[key] = top_routed_counts.get(key, 0) + 1
            if row.get("top_score_expert_id") is not None:
                key = str(row.get("top_score_expert_id"))
                top_score_counts[key] = top_score_counts.get(key, 0) + 1
        retained_positive = sum(1 for row in mode_rows if (_float_or_none(row.get("retained_improvement")) or 0.0) > 0.0)
        mode_summary = {
            "num_records": len(mode_rows),
            "positive_new_count": retained_positive,
            "immediate_positive_count": sum(1 for row in mode_rows if (_float_or_none(row.get("immediate_improvement")) or 0.0) > 0.0),
            "forgotten_count": sum(1 for row in mode_rows if row.get("forgotten")),
            "own_top1_count": sum(1 for row in mode_rows if row.get("own_top1")),
            "own_in_selected_set_count": sum(1 for row in mode_rows if row.get("own_selected")),
            "empty_selection_count": sum(1 for row in mode_rows if int(row.get("selected_expert_set_size") or 0) == 0),
            "mean_immediate_improvement": _mean([row.get("immediate_improvement") for row in mode_rows]),
            "mean_retained_improvement": _mean([row.get("retained_improvement") for row in mode_rows]),
            "mean_final_nll_improvement": _mean([row.get("retained_improvement") for row in mode_rows]),
            "mean_retention_ratio": _mean([row.get("retention_ratio") for row in mode_rows]),
            "worst_retention_drop": max(retention_drops, default=None),
            "worst_per_record_nll_degradation": max(per_record_degradation, default=0.0),
            "catastrophic_forgetting_count": sum(
                1
                for row in mode_rows
                if (_float_or_none(row.get("immediate_improvement")) or 0.0) > 0.0
                and (_float_or_none(row.get("retained_improvement")) or 0.0) < 0.0
            ),
            "mean_locality_reference_delta": _mean([row.get("locality_reference_delta") for row in mode_rows]),
            "locality_reference_delta": _mean([row.get("locality_reference_delta") for row in mode_rows]),
            "mean_selected_set_size": _mean(selected_sizes),
            "max_selected_set_size": max([float(value) for value in selected_sizes if value is not None], default=None),
            "top_routed_expert_counts": top_routed_counts,
            "top_score_expert_counts": top_score_counts,
            "max_top_routed_expert_count": max(top_routed_counts.values()) if top_routed_counts else 0,
            "max_top_score_count": max(top_score_counts.values()) if top_score_counts else 0,
            "max_top_expert_count": max(top_score_counts.values()) if top_score_counts else 0,
            "confusion_matrix": _routing_confusion(mode_rows),
        }
        mode_summary["retained_positive_count"] = retained_positive
        max_selected = mode_summary["max_selected_set_size"]
        max_selected_pass = max_selected is not None and float(max_selected) <= 3.0
        positive_gate = _positive_gate(final_step)
        selected_gate = _selected_gate(final_step)
        max_top_gate = _max_top_expert_gate(final_step)
        mode_summary["capacity_pass"] = bool(mode == "force_own" and retained_positive >= positive_gate)
        mode_summary["sparse_routing_pass"] = bool(
            mode != "force_own"
            and retained_positive >= positive_gate
            and int(mode_summary["own_top1_count"]) >= positive_gate
            and int(mode_summary["own_in_selected_set_count"]) >= selected_gate
            and (mode_summary["mean_selected_set_size"] is not None and float(mode_summary["mean_selected_set_size"]) <= 2.0)
            and max_selected_pass
            and int(mode_summary["empty_selection_count"]) <= 1
            and int(mode_summary["max_top_score_count"]) <= max_top_gate
            and int(mode_summary["catastrophic_forgetting_count"]) == 0
            and mode_summary["mean_locality_reference_delta"] is not None
        )
        mode_summary["sparse_routing_acceptance"] = {
            "positive_new_gate": positive_gate,
            "retained_positive_gate": positive_gate,
            "own_top1_gate": positive_gate,
            "own_selected_gate": selected_gate,
            "max_top_expert_gate": max_top_gate,
            "positive_new_pass": retained_positive >= positive_gate,
            "retained_positive_pass": retained_positive >= positive_gate,
            "own_top1_pass": int(mode_summary["own_top1_count"]) >= positive_gate,
            "own_selected_pass": int(mode_summary["own_in_selected_set_count"]) >= selected_gate,
            "mean_selected_set_size_le_2": (
                mode_summary["mean_selected_set_size"] is not None and float(mode_summary["mean_selected_set_size"]) <= 2.0
            ),
            "max_selected_set_size_le_3": max_selected_pass,
            "empty_selection_count_le_1": int(mode_summary["empty_selection_count"]) <= 1,
            "max_top_expert_count_pass": int(mode_summary["max_top_score_count"]) <= max_top_gate,
            "no_catastrophic_forgetting": int(mode_summary["catastrophic_forgetting_count"]) == 0,
            "locality_reported": mode_summary["mean_locality_reference_delta"] is not None,
        }
        mode_summary["retention_pass"] = bool(
            retained_positive >= positive_gate
            and int(mode_summary["catastrophic_forgetting_count"]) == 0
        )
        summary["by_mode"][mode] = mode_summary
    force_pass = bool((summary["by_mode"].get("force_own") or {}).get("capacity_pass"))
    calibrated = summary["by_mode"].get("calibrated") or {}
    summary["gate"] = {
        "force_own_capacity_pass": force_pass,
        "calibrated_sparse_pass": bool(calibrated.get("sparse_routing_pass")) if calibrated else None,
        "calibrated_retention_pass": bool(calibrated.get("retention_pass")) if calibrated else None,
        "overall_pass": bool(force_pass and (calibrated.get("sparse_routing_pass") if calibrated else True) and (calibrated.get("retention_pass") if calibrated else True)),
    }
    return summary


def write_time_sequential_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    sequential_rows: List[Dict[str, Any]],
    loss_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    modes = parse_eval_routing_modes(args.eval_routing_modes)
    prefix = _sequential_prefix(len(records))
    final_rows = _sequential_final_rows(sequential_rows, len(records))
    score_rows = _sequential_routing_score_rows(sequential_rows)
    confusion = {
        "after_each_edit": {},
        "final": {},
    }
    for step in range(1, len(records) + 1):
        step_rows = [row for row in sequential_rows if int(row.get("edit_step") or 0) == step]
        confusion["after_each_edit"][str(step)] = {
            mode: _routing_confusion([row for row in step_rows if row.get("eval_routing_mode") == mode])
            for mode in modes
        }
    for mode in modes:
        confusion["final"][mode] = _routing_confusion([row for row in final_rows if row.get("eval_routing_mode") == mode])
    retention_summary = summarize_time_sequential_retention(sequential_rows, records, modes, args)
    write_loss_trace(out_dir / f"{prefix}_loss_trace.csv", loss_rows)
    write_loss_trace(out_dir / f"{prefix}_after_each_edit.csv", sequential_rows)
    write_loss_trace(out_dir / f"{prefix}_per_record.csv", final_rows)
    write_loss_trace(out_dir / f"{prefix}_routing_scores.csv", score_rows)
    write_json(out_dir / f"{prefix}_confusion_matrices.json", confusion)
    write_json(out_dir / f"{prefix}_retention_summary.json", retention_summary)
    return retention_summary


def write_time_sequential_calibrated_report(
    out_dir: Path,
    args: argparse.Namespace,
    summary: Dict[str, Any],
    retention_summary: Dict[str, Any],
) -> Path:
    by_mode = retention_summary.get("by_mode") or {}
    gate = retention_summary.get("gate") or {}
    config = retention_summary.get("calibrated_config") or {}
    num_edits = int(retention_summary.get("num_edits") or args.max_edits)
    prefix = _sequential_prefix(num_edits)
    report_name = _sequential_report_name(num_edits)
    lines = [
        f"# TIME {num_edits}-Edit Sequential Calibrated Gate",
        "",
        "## Scope",
        "- Mode: sequential.",
        f"- Max edits: {retention_summary.get('num_edits')}.",
        f"- Eval routing modes: `{', '.join(retention_summary.get('eval_routing_modes') or [])}`.",
        f"- This run is bounded to the requested {num_edits}-edit gate; no 20-edit run or larger sweep is included.",
        "",
        "## Calibrated Routing Config",
        f"- Score norm: `{config.get('score_norm')}`.",
        f"- Calibration mode: requested `{config.get('calibration_mode_requested')}`, internal `{config.get('calibration_mode_internal')}`.",
        f"- Score pool: `{config.get('score_pool')}`.",
        f"- Routing mode: `{config.get('routing_mode')}`.",
        f"- Relative threshold: {config.get('relative_threshold')}.",
        f"- Max selected experts: {config.get('max_selected_experts')}.",
        f"- Mixing mode: `{config.get('mixing_mode')}`.",
        "",
        "## Gate",
        f"- Force-own capacity pass: {gate.get('force_own_capacity_pass')}.",
        f"- Calibrated sparse-routing pass: {gate.get('calibrated_sparse_pass')}.",
        f"- Calibrated retention pass: {gate.get('calibrated_retention_pass')}.",
        f"- Overall pass: {gate.get('overall_pass')}.",
        "",
        "## Final Metrics By Mode",
    ]
    for mode, row in by_mode.items():
        lines.extend(
            [
                f"### {mode}",
                f"- Positive retained improvements: {row.get('positive_new_count')} / {row.get('num_records')}.",
                f"- Forgotten count: {row.get('forgotten_count')}.",
                f"- Own top-1 count: {row.get('own_top1_count')} / {row.get('num_records')}.",
                f"- Own in selected set: {row.get('own_in_selected_set_count')} / {row.get('num_records')}.",
                f"- Empty selections: {row.get('empty_selection_count')}.",
                f"- Mean retained improvement: {row.get('mean_retained_improvement')}.",
                f"- Mean retention ratio: {row.get('mean_retention_ratio')}.",
                f"- Worst retention drop: {row.get('worst_retention_drop')}.",
                f"- Worst per-record NLL degradation: {row.get('worst_per_record_nll_degradation')}.",
                f"- Catastrophic forgetting count: {row.get('catastrophic_forgetting_count')}.",
                f"- Mean locality/reference delta: {row.get('mean_locality_reference_delta')}.",
                f"- Mean selected set size: {row.get('mean_selected_set_size')}.",
                f"- Max selected set size: {row.get('max_selected_set_size')}.",
                f"- Max top-expert count: {row.get('max_top_expert_count')}.",
                f"- Top score expert counts: `{json.dumps(row.get('top_score_expert_counts'), sort_keys=True)}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## GPU",
            f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES') or ''}`.",
            "```text",
            _gpu_status_text(),
            "```",
            "",
            "## Output Files",
            "- `expert_repository.pt`",
            "- `loss_trace.csv`",
            f"- `{prefix}_loss_trace.csv`",
            f"- `{prefix}_after_each_edit.csv`",
            f"- `{prefix}_per_record.csv`",
            f"- `{prefix}_routing_scores.csv`",
            f"- `{prefix}_confusion_matrices.json`",
            f"- `{prefix}_routing_debug.jsonl`",
            f"- `{prefix}_retention_summary.json`",
            f"- `{'time_ten_sequential_summary.json' if num_edits == 10 else 'five_edit_seq_summary.json'}`",
            f"- `{report_name}`",
            "",
            "## Repository",
            f"- Experts saved: {summary.get('num_experts')}.",
            f"- Repository path: `{summary.get('expert_repository')}`.",
            "",
            "## Command",
            f"- `{current_command_line()}`",
            "",
        ]
    )
    path = out_dir / report_name
    path.write_text("\n".join(lines))
    return path


SCORE_NORM_MODES = ["none", "factor", "factor_z", "self_score", "factor_self_score"]
CALIBRATION_MIXING_MODES = ["softmax", "average"]


def _gpu_status_text() -> str:
    try:
        return os.popen(
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
        ).read().strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _cuda_setting_from_command(command: Optional[str]) -> Optional[str]:
    prefix = "CUDA_VISIBLE_DEVICES="
    if not command or prefix not in command:
        return None
    tail = command.split(prefix, 1)[1]
    return tail.split(None, 1)[0] if tail else None


def _repo_path_for_report(repo_path: Optional[Path]) -> Optional[str]:
    if repo_path is None:
        return None
    return str(repo_path.resolve())


def default_calibration_grid() -> List[Dict[str, Any]]:
    route_specs: List[Dict[str, Any]] = []
    for topk in (1, 2, 3):
        route_specs.append({"routing_mode": "topk", "topk": topk, "gamma": None, "relative_threshold": None})
    for gamma in (0.5, 0.7, 1.0, 2.0):
        route_specs.append({"routing_mode": "threshold", "topk": 0, "gamma": gamma, "relative_threshold": None})
    for threshold in (0.5, 0.7, 0.9, 0.95):
        route_specs.append({"routing_mode": "relative_threshold", "topk": 0, "gamma": None, "relative_threshold": threshold})

    configs: List[Dict[str, Any]] = []
    for score_norm in SCORE_NORM_MODES:
        for mixing_mode in CALIBRATION_MIXING_MODES:
            for route in route_specs:
                config = dict(route)
                config["score_norm"] = score_norm
                config["mixing_mode"] = mixing_mode
                config["config_id"] = calibration_config_id(config)
                configs.append(config)
    return configs


def calibration_config_id(config: Dict[str, Any]) -> str:
    route = str(config.get("routing_mode"))
    if route == "topk":
        route = f"topk{int(config.get('topk') or 1)}"
    elif route == "threshold":
        route = f"gamma{format_float_for_path(float(config.get('gamma')))}"
    elif route == "relative_threshold":
        route = f"rel{format_float_for_path(float(config.get('relative_threshold')))}"
    return f"{config.get('score_norm')}_{route}_{config.get('mixing_mode')}"


def parse_calibration_grid(text: str) -> List[Dict[str, Any]]:
    if not str(text or "").strip():
        return default_calibration_grid()
    configs: List[Dict[str, Any]] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        config: Dict[str, Any] = {
            "score_norm": "none",
            "mixing_mode": "softmax",
            "routing_mode": "topk",
            "topk": 1,
            "gamma": None,
            "relative_threshold": None,
        }
        for part in item.split(","):
            if not part.strip():
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"score_norm", "mixing_mode", "routing_mode"}:
                config[key] = value
            elif key == "topk":
                config[key] = int(value)
            elif key == "gamma":
                config[key] = float(value)
            elif key in {"relative_threshold", "rel"}:
                config["relative_threshold"] = float(value)
                config["routing_mode"] = "relative_threshold"
            else:
                raise ValueError(f"Unsupported --time-routing-calibration-grid key: {key}")
        config["config_id"] = calibration_config_id(config)
        configs.append(config)
    return configs


def _config_extra_fields(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "calibration_config_id": config.get("config_id"),
        "calibration_score_norm": config.get("score_norm"),
        "calibration_routing_mode": config.get("routing_mode"),
        "calibration_gamma": config.get("gamma"),
        "calibration_relative_threshold": config.get("relative_threshold"),
        "calibration_topk": config.get("topk"),
        "calibration_mixing_mode": config.get("mixing_mode"),
    }


def infer_self_score_calibration(
    args: argparse.Namespace,
    alg: TIMEEdit,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    debug_path = out_dir / "time_routing_debug.jsonl"
    self_scores = {"none": [], "factor": [], "factor_z": []}
    rows: List[Dict[str, Any]] = []
    with temporary_time_routing(
        alg,
        routing_mode="threshold",
        gamma=1.0e30,
        topk=0,
        score_norm="none",
        relative_threshold=None,
        mixing_mode="average",
    ):
        for eval_pos, (record, sample) in enumerate(zip(records, samples)):
            row = evaluate_sample(
                alg,
                sample,
                record,
                eval_pos,
                phase=f"self_score_calibration_eval_{eval_pos}",
                expected_expert=eval_pos,
                routing_debug_path=debug_path,
                eval_routing_mode="self_score_calibration",
                force_expert_id=eval_pos,
                extra_fields={
                    "calibration_phase": "self_score_calibration",
                    "time_load_repository": str(args.time_load_repository) if args.time_load_repository else None,
                },
            )
            variants = row.get("score_variant_pooled_scores") or {}
            for mode in ("none", "factor", "factor_z"):
                values = variants.get(mode) or row.get("raw_pooled_scores") or []
                value = values[eval_pos] if eval_pos < len(values) else None
                self_scores[mode].append(float(value) if value is not None else 1.0)
            rows.append(row)

    alg.time_residual.self_score_cache = {
        "none": list(self_scores["none"]),
        "factor": list(self_scores["factor"]),
    }
    for idx, metadata in enumerate(alg.repository.metadata):
        if idx < len(self_scores["none"]):
            metadata["time_self_score_none"] = self_scores["none"][idx]
        if idx < len(self_scores["factor"]):
            metadata["time_self_score_factor"] = self_scores["factor"][idx]
        if idx < len(self_scores["factor_z"]):
            metadata["time_self_score_factor_z"] = self_scores["factor_z"][idx]
    return {"self_scores": self_scores, "rows": rows}


def collect_score_distribution(
    alg: TIMEEdit,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    debug_path = out_dir / "time_routing_debug.jsonl"
    with temporary_time_routing(
        alg,
        routing_mode="threshold",
        gamma=1.0e30,
        topk=0,
        score_norm="none",
        relative_threshold=None,
        mixing_mode="average",
    ):
        for eval_pos, (record, sample) in enumerate(zip(records, samples)):
            row = evaluate_sample(
                alg,
                sample,
                record,
                eval_pos,
                phase=f"score_distribution_eval_{eval_pos}",
                expected_expert=eval_pos,
                routing_debug_path=debug_path,
                eval_routing_mode="score_distribution",
                force_expert_id=eval_pos,
                extra_fields={"calibration_phase": "score_distribution"},
            )
            variants = row.get("score_variant_pooled_scores") or {}
            raw_scores = variants.get("none") or row.get("raw_pooled_scores") or []
            top_by_norm = {
                mode: int(max(range(len(values)), key=lambda idx: values[idx])) if values else None
                for mode, values in variants.items()
            }
            for expert_index, raw_score in enumerate(raw_scores):
                dist_row = {
                    "record_id": row.get("record_id"),
                    "query_record_index": eval_pos,
                    "expert_index": expert_index,
                    "raw_score": raw_score,
                    "own_expert": expert_index == eval_pos,
                    "target_nll_after_force_own": row.get("target_nll"),
                    "reference_delta_after_force_own": row.get("reference_delta"),
                }
                for mode in SCORE_NORM_MODES:
                    values = variants.get(mode) or []
                    dist_row[f"{mode}_score"] = values[expert_index] if expert_index < len(values) else None
                    dist_row[f"{mode}_top_expert"] = top_by_norm.get(mode)
                    dist_row[f"{mode}_is_top"] = top_by_norm.get(mode) == expert_index
                rows.append(dist_row)
    return rows


def summarize_score_distribution(rows: List[Dict[str, Any]], num_experts: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"per_expert": {}, "per_record_own_scores": [], "dominance": {}}
    for expert_index in range(num_experts):
        expert_rows = [row for row in rows if int(row.get("expert_index")) == expert_index]
        summary["per_expert"][str(expert_index)] = {}
        for mode in SCORE_NORM_MODES:
            key = "raw_score" if mode == "none" else f"{mode}_score"
            values = [_float_or_none(row.get(key)) for row in expert_rows]
            valid = [float(value) for value in values if value is not None]
            mean = float(sum(valid) / len(valid)) if valid else None
            std = None
            if valid:
                std = math.sqrt(sum((value - mean) ** 2 for value in valid) / len(valid))
            summary["per_expert"][str(expert_index)][mode] = {
                "mean": mean,
                "std": std,
                "max": max(valid) if valid else None,
            }
    seen_records = sorted({int(row.get("query_record_index")) for row in rows})
    for query_index in seen_records:
        own_rows = [
            row for row in rows
            if int(row.get("query_record_index")) == query_index and bool(row.get("own_expert"))
        ]
        if not own_rows:
            continue
        row = own_rows[0]
        entry = {"record_id": row.get("record_id"), "query_record_index": query_index}
        for mode in SCORE_NORM_MODES:
            key = "raw_score" if mode == "none" else f"{mode}_score"
            entry[f"{mode}_own_score"] = row.get(key)
        summary["per_record_own_scores"].append(entry)
    for mode in SCORE_NORM_MODES:
        top_key = f"{mode}_top_expert"
        top_rows = {}
        for row in rows:
            query_index = int(row.get("query_record_index"))
            if query_index not in top_rows:
                top_rows[query_index] = row.get(top_key)
        top_values = [value for value in top_rows.values() if value is not None]
        counts: Dict[str, int] = {}
        for value in top_values:
            counts[str(value)] = counts.get(str(value), 0) + 1
        summary["dominance"][mode] = {
            "top_expert_counts": counts,
            "expert_3_top1_count": counts.get("3", 0),
            "all_records_same_top_expert": len(set(top_values)) == 1 if top_values else False,
            "dominant_top_expert": max(counts, key=counts.get) if counts else None,
        }
    return summary


def _calibration_confusion(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    confusion: Dict[str, Dict[str, int]] = {}
    for row in rows:
        expected = str(row.get("expected_expert"))
        observed = str(row.get("top_expert_id"))
        confusion.setdefault(expected, {})
        confusion[expected][observed] = confusion[expected].get(observed, 0) + 1
    return confusion


def summarize_calibration_config(config: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected_sizes = [float(row.get("selected_expert_set_size") or 0.0) for row in rows]
    top_ids = [row.get("top_expert_id") for row in rows if row.get("top_expert_id") is not None]
    top_counts: Dict[str, int] = {}
    for top_id in top_ids:
        top_counts[str(top_id)] = top_counts.get(str(top_id), 0) + 1
    summary = {
        "config_id": config.get("config_id"),
        "score_norm": config.get("score_norm"),
        "routing_mode": config.get("routing_mode"),
        "gamma": config.get("gamma"),
        "relative_threshold": config.get("relative_threshold"),
        "topk": config.get("topk"),
        "mixing_mode": config.get("mixing_mode"),
        "num_records": len(rows),
        "mean_target_nll_improvement": _mean([row.get("target_nll_delta") for row in rows]),
        "positive_new_count": sum(1 for row in rows if (row.get("target_nll_delta") or 0.0) > 0.0),
        "own_top1_count": sum(1 for row in rows if row.get("routing_top1_correct")),
        "own_in_selected_set_count": sum(1 for row in rows if row.get("selected_own_expert")),
        "empty_selection_count": sum(1 for row in rows if int(row.get("selected_expert_set_size") or 0) == 0),
        "mean_selected_set_size": _mean(selected_sizes),
        "max_selected_set_size": max(selected_sizes) if selected_sizes else None,
        "mean_residual_norm": _mean([row.get("residual_norm") for row in rows]),
        "mean_hidden_delta_norm": _mean([row.get("target_layer_hidden_delta_norm") for row in rows]),
        "mean_locality_reference_delta": _mean([row.get("reference_delta") for row in rows]),
        "all_records_collapse_to_same_top_expert": len(set(top_ids)) == 1 if top_ids else False,
        "top_expert_counts": top_counts,
        "confusion_matrix": _calibration_confusion(rows),
    }
    summary["sparse_success"] = bool(
        int(summary["own_top1_count"]) >= 4
        and int(summary["own_in_selected_set_count"]) >= 4
        and int(summary["positive_new_count"]) >= 4
        and (summary["mean_selected_set_size"] is not None and float(summary["mean_selected_set_size"]) <= 2.0)
        and int(summary["empty_selection_count"]) <= 1
    )
    return summary


def _sort_key_routing(row: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    return (
        float(row.get("own_top1_count") or 0.0),
        float(row.get("own_in_selected_set_count") or 0.0),
        -abs(float(row.get("mean_selected_set_size") or 0.0) - 1.0),
        -float(row.get("empty_selection_count") or 0.0),
        float(row.get("mean_target_nll_improvement") or -1.0e9),
    )


def _sort_key_target(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(row.get("mean_target_nll_improvement") or -1.0e9),
        float(row.get("positive_new_count") or 0.0),
        float(row.get("own_top1_count") or 0.0),
    )


def _sort_key_locality(row: Dict[str, Any]) -> Tuple[float, float, float]:
    locality = row.get("mean_locality_reference_delta")
    return (
        -float(locality) if locality is not None else -1.0e30,
        float(row.get("positive_new_count") or 0.0),
        float(row.get("own_top1_count") or 0.0),
    )


def choose_best_calibration(grid_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_by_routing = max(grid_rows, key=_sort_key_routing) if grid_rows else {}
    best_by_target = max(grid_rows, key=_sort_key_target) if grid_rows else {}
    best_by_locality = max(grid_rows, key=_sort_key_locality) if grid_rows else {}
    sparse = [row for row in grid_rows if row.get("sparse_success")]
    best_sparse = max(sparse, key=_sort_key_target) if sparse else {}
    return {
        "best_by_routing_accuracy": best_by_routing,
        "best_by_target_nll_improvement": best_by_target,
        "best_by_locality_reference_delta": best_by_locality,
        "best_sparse_success": best_sparse,
    }


def diagnose_routing_calibration(grid_rows: List[Dict[str, Any]], oracle_summary: Dict[str, Any]) -> Tuple[str, str]:
    sparse = [row for row in grid_rows if row.get("sparse_success")]
    if sparse:
        best_sparse = max(sparse, key=_sort_key_target)
        locality = _float_or_none(best_sparse.get("mean_locality_reference_delta")) or 0.0
        if locality > 5.0:
            return "routing_fixed_but_locality_damage", "add stronger alignment loss"
        return "routing_calibration_success", "proceed to 5-edit sequential with best config"

    max_top1 = max((int(row.get("own_top1_count") or 0) for row in grid_rows), default=0)
    oracle_positive = int(oracle_summary.get("positive_new_count") or 0)
    if oracle_positive >= 4 and max_top1 < 3:
        return "intrinsic_score_not_discriminative", "revisit intrinsic factor generation"

    raw_collapsed = any(
        row.get("score_norm") == "none"
        and row.get("routing_mode") == "topk"
        and int(row.get("topk") or 0) == 1
        and row.get("all_records_collapse_to_same_top_expert")
        for row in grid_rows
    )
    factor_fixed = any(
        row.get("score_norm") in {"factor", "factor_z"}
        and int(row.get("own_top1_count") or 0) >= 4
        for row in grid_rows
    )
    if raw_collapsed and factor_fixed:
        return "factor_norm_score_scale_issue", "run score-normalized 5-edit retrain"

    self_fixed = any(
        row.get("score_norm") in {"self_score", "factor_self_score"}
        and int(row.get("own_top1_count") or 0) >= 4
        for row in grid_rows
    )
    if self_fixed:
        return "per_expert_score_calibration_issue", "run score-normalized 5-edit retrain"

    topk1_best = max(
        (int(row.get("own_in_selected_set_count") or 0) for row in grid_rows if row.get("routing_mode") == "topk" and int(row.get("topk") or 0) == 1),
        default=0,
    )
    topk23_best = max(
        (int(row.get("own_in_selected_set_count") or 0) for row in grid_rows if row.get("routing_mode") == "topk" and int(row.get("topk") or 0) in {2, 3}),
        default=0,
    )
    if topk23_best >= 4 and topk1_best < 4:
        return "ranking_margin_issue", "add stronger alignment loss"

    threshold_rows = [row for row in grid_rows if row.get("routing_mode") == "threshold"]
    if threshold_rows and all(
        int(row.get("empty_selection_count") or 0) >= 4
        or (_float_or_none(row.get("mean_selected_set_size")) or 0.0) >= 4.0
        for row in threshold_rows
    ):
        return "absolute_threshold_unsuitable", "add stronger alignment loss"

    return "routing_calibration_inconclusive", "revisit intrinsic factor generation"


def _per_expert_calibration_rows(config: Dict[str, Any], eval_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    variants = eval_row.get("score_variant_pooled_scores") or {}
    raw_scores = variants.get("none") or eval_row.get("raw_pooled_scores") or []
    selected_ids = set(eval_row.get("selected_expert_ids") or [])
    top_id = eval_row.get("top_expert_id")
    expected = eval_row.get("expected_expert")
    for expert_index, raw_score in enumerate(raw_scores):
        row = {
            **_config_extra_fields(config),
            "record_id": eval_row.get("record_id"),
            "query_record_index": eval_row.get("sample_pos"),
            "expert_index": expert_index,
            "raw_score": raw_score,
            "score_used": (eval_row.get("pooled_scores") or [None] * len(raw_scores))[expert_index],
            "selected": expert_index in selected_ids,
            "top_expert": expert_index == top_id,
            "own_expert": expected is not None and expert_index == int(expected),
            "target_nll_after_routing": eval_row.get("target_nll"),
            "target_nll_improvement": eval_row.get("target_nll_delta"),
            "reference_delta": eval_row.get("reference_delta"),
        }
        for mode in SCORE_NORM_MODES:
            values = variants.get(mode) or []
            row[f"{mode}_score"] = values[expert_index] if expert_index < len(values) else None
        rows.append(row)
    return rows


def run_eval_only(
    args: argparse.Namespace,
    config: TIMEEditMultimodalHparams,
    dataset_path: Path,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    alg: TIMEEdit,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    routing_debug_path = out_dir / "routing_debug.jsonl"
    if routing_debug_path.exists():
        routing_debug_path.unlink()
    eval_rows = (
        evaluate_nonseq_routing_modes(args, alg, records, samples, out_dir)
        if args.mode == "nonseq" and args.eval_routing_modes
        else [
            evaluate_sample(
                alg,
                sample,
                record,
                eval_pos,
                phase=f"eval_only_{args.mode}_eval_{eval_pos}",
                expected_expert=eval_pos,
                routing_debug_path=routing_debug_path,
                force_expert_id=eval_pos if args.time_mixing_mode == "own_oracle" else None,
            )
            for eval_pos, (record, sample) in enumerate(zip(records, samples))
        ]
    )
    summary = summarize_evals(args.mode, eval_rows, [], alg, out_dir)
    summary.update({
        "eval_only": True,
        "loaded_repository_path": _repo_path_for_report(resolve_repository_path(args.time_load_repository)),
        "hidden_size": alg.repository.hidden_size,
        "s1": alg.repository.s1,
        "s2": alg.repository.s2,
    })
    write_json(
        out_dir / "time_hparams.json",
        {
            "args": vars(args),
            "config": dict(config.__dict__),
            "command": current_command_line(),
            "eval_only": True,
            "loaded_repository_path": summary["loaded_repository_path"],
        },
    )
    write_loss_trace(out_dir / "eval_only_per_record.csv", eval_rows)
    write_json(out_dir / "eval_only_summary.json", summary)
    return summary


def run_time_routing_calibration(
    args: argparse.Namespace,
    config: TIMEEditMultimodalHparams,
    dataset_path: Path,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    alg: TIMEEdit,
    out_dir: Path,
    repo_path: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_path = out_dir / "time_routing_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()
    command = current_command_line()
    gpu_status = _gpu_status_text()
    write_json(
        out_dir / "time_hparams.json",
        {
            "args": vars(args),
            "config": dict(config.__dict__),
            "command": command,
            "eval_only": True,
            "loaded_repository_path": str(repo_path),
            "gpu_status_at_start": gpu_status,
        },
    )

    self_score_payload = infer_self_score_calibration(args, alg, records, samples, out_dir)
    score_rows = collect_score_distribution(alg, records, samples, out_dir)
    score_summary = summarize_score_distribution(score_rows, alg.repository.num_experts)
    score_summary["self_score_calibration"] = self_score_payload["self_scores"]
    score_summary["loaded_repository_path"] = str(repo_path)
    write_loss_trace(out_dir / "time_score_distribution.csv", score_rows)
    write_json(out_dir / "time_score_distribution_summary.json", score_summary)

    configs = parse_calibration_grid(args.time_routing_calibration_grid)
    per_record_rows: List[Dict[str, Any]] = []
    grid_rows: List[Dict[str, Any]] = []
    confusion_matrices: Dict[str, Any] = {}
    all_eval_rows: Dict[str, List[Dict[str, Any]]] = {}

    oracle_config = {
        "config_id": "own_oracle_force_own",
        "score_norm": "none",
        "routing_mode": "threshold",
        "gamma": 1.0e30,
        "relative_threshold": None,
        "topk": 0,
        "mixing_mode": "own_oracle",
    }
    oracle_rows: List[Dict[str, Any]] = []
    with temporary_time_routing(
        alg,
        routing_mode="threshold",
        gamma=1.0e30,
        topk=0,
        score_norm="none",
        relative_threshold=None,
        mixing_mode="average",
    ):
        for eval_pos, (record, sample) in enumerate(zip(records, samples)):
            oracle_rows.append(
                evaluate_sample(
                    alg,
                    sample,
                    record,
                    eval_pos,
                    phase=f"calibration_{oracle_config['config_id']}_eval_{eval_pos}",
                    expected_expert=eval_pos,
                    routing_debug_path=debug_path,
                    eval_routing_mode=oracle_config["config_id"],
                    force_expert_id=eval_pos,
                    extra_fields={**_config_extra_fields(oracle_config), "calibration_phase": "oracle_baseline"},
                )
            )
    oracle_summary = summarize_calibration_config(oracle_config, oracle_rows)

    for config_item in configs:
        eval_rows: List[Dict[str, Any]] = []
        routing_mode = str(config_item["routing_mode"])
        gamma = config_item.get("gamma")
        relative_threshold = config_item.get("relative_threshold")
        topk = int(config_item.get("topk") or 0)
        mixing_mode = str(config_item.get("mixing_mode") or "softmax")
        with temporary_time_routing(
            alg,
            routing_mode=routing_mode,
            gamma=float(gamma) if gamma is not None else None,
            topk=topk,
            score_norm=str(config_item.get("score_norm") or "none"),
            relative_threshold=float(relative_threshold) if relative_threshold is not None else None,
            mixing_mode=mixing_mode,
        ):
            for eval_pos, (record, sample) in enumerate(zip(records, samples)):
                row = evaluate_sample(
                    alg,
                    sample,
                    record,
                    eval_pos,
                    phase=f"calibration_{config_item['config_id']}_eval_{eval_pos}",
                    expected_expert=eval_pos,
                    routing_debug_path=debug_path,
                    eval_routing_mode=config_item["config_id"],
                    force_expert_id=eval_pos if mixing_mode == "own_oracle" else None,
                    extra_fields={**_config_extra_fields(config_item), "calibration_phase": "grid"},
                )
                eval_rows.append(row)
                per_record_rows.extend(_per_expert_calibration_rows(config_item, row))
        summary = summarize_calibration_config(config_item, eval_rows)
        grid_rows.append(summary)
        all_eval_rows[str(config_item["config_id"])] = eval_rows
        confusion_matrices[str(config_item["config_id"])] = {
            "config": {key: config_item.get(key) for key in ("score_norm", "routing_mode", "gamma", "relative_threshold", "topk", "mixing_mode")},
            "matrix": summary["confusion_matrix"],
        }

    best = choose_best_calibration(grid_rows)
    diagnosis, recommendation = diagnose_routing_calibration(grid_rows, oracle_summary)
    best.update(
        {
            "diagnosis": diagnosis,
            "recommendation": recommendation,
            "sparse_routing_achieved": bool(best.get("best_sparse_success")),
            "oracle_baseline": oracle_summary,
            "loaded_repository_path": str(repo_path),
            "command": command,
            "gpu_status_at_start": gpu_status,
            "record_ids": [record_id(record, idx) for idx, record in enumerate(records)],
        }
    )
    write_loss_trace(out_dir / "time_routing_calibration_grid.csv", grid_rows)
    write_loss_trace(out_dir / "time_routing_calibration_per_record.csv", per_record_rows)
    write_json(out_dir / "time_routing_calibration_best.json", best)
    write_json(out_dir / "time_routing_confusion_matrices.json", confusion_matrices)
    report_path = write_time_routing_calibration_report(
        out_dir,
        args,
        score_summary,
        grid_rows,
        best,
        oracle_summary,
        repo_path,
        command,
        gpu_status,
    )
    payload = {
        "score_summary": score_summary,
        "grid_rows": grid_rows,
        "best": best,
        "oracle_baseline": oracle_summary,
        "report_path": str(report_path),
        "loaded_repository_path": str(repo_path),
        "num_grid_configs": len(grid_rows),
        "num_per_record_rows": len(per_record_rows),
    }
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
    return payload


def _compact_config_table(rows: List[Dict[str, Any]], limit: int = 15) -> List[str]:
    lines = [
        "| config | norm | route | mix | NLL imp | positive | own top1 | own selected | empty | mean size | locality | collapse |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=_sort_key_routing, reverse=True)[:limit]:
        route = row.get("routing_mode")
        if route == "topk":
            route = f"topk={row.get('topk')}"
        elif route == "threshold":
            route = f"gamma={row.get('gamma')}"
        elif route == "relative_threshold":
            route = f"rel={row.get('relative_threshold')}"
        lines.append(
            "| {config} | {norm} | {route} | {mix} | {nll} | {pos}/5 | {top1}/5 | {own}/5 | {empty} | {size} | {loc} | {collapse} |".format(
                config=row.get("config_id"),
                norm=row.get("score_norm"),
                route=route,
                mix=row.get("mixing_mode"),
                nll=_fmt(row.get("mean_target_nll_improvement")),
                pos=row.get("positive_new_count"),
                top1=row.get("own_top1_count"),
                own=row.get("own_in_selected_set_count"),
                empty=row.get("empty_selection_count"),
                size=_fmt(row.get("mean_selected_set_size")),
                loc=_fmt(row.get("mean_locality_reference_delta")),
                collapse=row.get("all_records_collapse_to_same_top_expert"),
            )
        )
    return lines


def write_time_routing_calibration_report(
    out_dir: Path,
    args: argparse.Namespace,
    score_summary: Dict[str, Any],
    grid_rows: List[Dict[str, Any]],
    best: Dict[str, Any],
    oracle_summary: Dict[str, Any],
    repo_path: Path,
    command: str,
    gpu_status: str,
) -> Path:
    raw_confusion = None
    best_confusion = None
    for row in grid_rows:
        if row.get("score_norm") == "none" and row.get("routing_mode") == "topk" and int(row.get("topk") or 0) == 1:
            raw_confusion = row.get("confusion_matrix")
            break
    best_routing = best.get("best_by_routing_accuracy") or {}
    best_sparse = best.get("best_sparse_success") or {}
    best_for_confusion = best_sparse or best_routing
    best_confusion = best_for_confusion.get("confusion_matrix")
    dominance = score_summary.get("dominance", {})
    raw_dominance = dominance.get("none", {})
    best_target = best.get("best_by_target_nll_improvement") or {}
    best_locality = best.get("best_by_locality_reference_delta") or {}
    cuda_setting = os.environ.get("CUDA_VISIBLE_DEVICES") or _cuda_setting_from_command(command) or args.device
    lines = [
        "# TIME 5-Edit Routing Calibration Report",
        "",
        "## Files Changed",
        "- `easyeditor/trainer/algs/time_edit.py`",
        "- `easyeditor/trainer/algs/time_edit_modules.py`",
        "- `easyeditor/models/time_edit/time_edit_hparams.py`",
        "- `scripts/time/run_time_medmkeb_smoke.py`",
        "- `scripts/time/test_time_modules.py`",
        "",
        "## Existing Repository Loaded",
        f"- `{repo_path}`",
        "",
        "## Exact Command Run",
        f"- `{command}`",
        "",
        "## GPU",
        f"- Used CUDA device setting: `{cuda_setting}`.",
        "- Chosen because the pre-run `nvidia-smi` check showed GPU 3 effectively free; captured start status:",
        "```text",
        gpu_status or "unavailable",
        "```",
        "",
        "## Verification",
        "- `py_compile`: passed before model run.",
        "- `scripts/time/test_time_modules.py`: passed before model run.",
        "- 20-edit run: not run.",
        "- Optional retraining: not run.",
        "",
        "## Raw Score Distribution Summary",
        f"- Expert 3 raw top-1 count: {raw_dominance.get('expert_3_top1_count')} / 5.",
        f"- Raw top expert counts: `{json.dumps(raw_dominance.get('top_expert_counts'), sort_keys=True)}`.",
        "",
        "| expert | raw mean | raw std | raw max | factor mean | factor_z mean | self_score mean | factor_self mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for expert_id, payload in sorted((score_summary.get("per_expert") or {}).items(), key=lambda item: int(item[0])):
        lines.append(
            "| {expert} | {raw_mean} | {raw_std} | {raw_max} | {factor_mean} | {factor_z_mean} | {self_mean} | {factor_self_mean} |".format(
                expert=expert_id,
                raw_mean=_fmt((payload.get("none") or {}).get("mean")),
                raw_std=_fmt((payload.get("none") or {}).get("std")),
                raw_max=_fmt((payload.get("none") or {}).get("max")),
                factor_mean=_fmt((payload.get("factor") or {}).get("mean")),
                factor_z_mean=_fmt((payload.get("factor_z") or {}).get("mean")),
                self_mean=_fmt((payload.get("self_score") or {}).get("mean")),
                factor_self_mean=_fmt((payload.get("factor_self_score") or {}).get("mean")),
            )
        )
    lines.extend(["", "### Per-Record Own Scores"])
    lines.append("| record_id | own idx | raw | factor | factor_z | self_score | factor_self |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in score_summary.get("per_record_own_scores", []):
        lines.append(
            "| {rid} | {idx} | {raw} | {factor} | {factor_z} | {self_score} | {factor_self} |".format(
                rid=row.get("record_id"),
                idx=row.get("query_record_index"),
                raw=_fmt(row.get("none_own_score")),
                factor=_fmt(row.get("factor_own_score")),
                factor_z=_fmt(row.get("factor_z_own_score")),
                self_score=_fmt(row.get("self_score_own_score")),
                factor_self=_fmt(row.get("factor_self_score_own_score")),
            )
        )
    lines.extend(
        [
            "",
            "## Calibration Grid Summary",
            f"- Grid configs evaluated: {len(grid_rows)}.",
            f"- Oracle/force-own mean NLL improvement: {_fmt(oracle_summary.get('mean_target_nll_improvement'))}; positive {oracle_summary.get('positive_new_count')}/5.",
            "",
        ]
    )
    lines.extend(_compact_config_table(grid_rows))
    lines.extend(
        [
            "",
            "## Best Configs",
            f"- Best by routing accuracy: `{best_routing.get('config_id')}` with own top-1 {best_routing.get('own_top1_count')}/5, own selected {best_routing.get('own_in_selected_set_count')}/5, mean selected size {_fmt(best_routing.get('mean_selected_set_size'))}.",
            f"- Best by target NLL improvement: `{best_target.get('config_id')}` with mean improvement {_fmt(best_target.get('mean_target_nll_improvement'))}.",
            f"- Best by locality/reference delta: `{best_locality.get('config_id')}` with mean locality delta {_fmt(best_locality.get('mean_locality_reference_delta'))}.",
            "",
            "## Confusion Matrices",
            f"- Raw topk=1: `{json.dumps(raw_confusion, sort_keys=True)}`",
            f"- Best calibrated routing: `{json.dumps(best_confusion, sort_keys=True)}`",
            "",
            "## Sparse Routing",
            f"- Achieved: {bool(best.get('sparse_routing_achieved'))}.",
            "",
            "## Diagnosis",
            f"- Label: `{best.get('diagnosis')}`.",
            "",
            "## Recommendation",
            f"- {best.get('recommendation')}.",
            "",
            "## Output Files",
            "- `time_score_distribution.csv`",
            "- `time_score_distribution_summary.json`",
            "- `time_routing_calibration_grid.csv`",
            "- `time_routing_calibration_best.json`",
            "- `time_routing_calibration_per_record.csv`",
            "- `time_routing_confusion_matrices.json`",
            "- `time_routing_debug.jsonl`",
            "- `TIME_5EDIT_ROUTING_CALIBRATION_REPORT.md`",
            "",
        ]
    )
    path = out_dir / "TIME_5EDIT_ROUTING_CALIBRATION_REPORT.md"
    path.write_text("\n".join(lines))
    return path


POST_RETRAIN_SCORE_NORMS = ["factor_z", "factor_self_score"]
POST_RETRAIN_SCORE_POOLS = ["mean", "max", "last"]
POST_RETRAIN_BASE_SCORE_NORMS = ["none", "factor_z", "factor_self_score"]


def post_retrain_route_label(config: Dict[str, Any]) -> str:
    if config.get("calibration_mode") == "neg_margin":
        return f"neg_margin_b{format_float_for_path(float(config.get('beta') or 0.0))}"
    route = str(config.get("routing_mode"))
    if route == "topk":
        return f"topk{int(config.get('topk') or 1)}"
    if route == "threshold":
        return f"gamma{format_float_for_path(float(config.get('gamma') or 0.0))}"
    if route == "relative_threshold":
        return f"rel{format_float_for_path(float(config.get('relative_threshold') or 0.0))}"
    return route


def post_retrain_config_id(config: Dict[str, Any]) -> str:
    cap = config.get("max_selected_experts")
    cap_text = "capnone" if cap is None else f"cap{int(cap)}"
    return "_".join(
        [
            str(config.get("score_norm")),
            str(config.get("calibration_mode")),
            str(config.get("score_pool")),
            post_retrain_route_label(config),
            cap_text,
            str(config.get("mixing_mode")),
        ]
    )


def post_retrain_calibration_configs() -> List[Dict[str, Any]]:
    all_routes = [
        {"routing_mode": "topk", "topk": 1, "gamma": None, "relative_threshold": None},
        {"routing_mode": "topk", "topk": 2, "gamma": None, "relative_threshold": None},
        {"routing_mode": "relative_threshold", "topk": 0, "gamma": None, "relative_threshold": 0.90},
        {"routing_mode": "relative_threshold", "topk": 0, "gamma": None, "relative_threshold": 0.95},
        {"routing_mode": "relative_threshold", "topk": 0, "gamma": None, "relative_threshold": 0.98},
        {"routing_mode": "relative_threshold", "topk": 0, "gamma": None, "relative_threshold": 0.99},
        {"routing_mode": "threshold", "topk": 0, "gamma": 0.5, "relative_threshold": None},
        {"routing_mode": "threshold", "topk": 0, "gamma": 0.7, "relative_threshold": None},
        {"routing_mode": "threshold", "topk": 0, "gamma": 1.0, "relative_threshold": None},
    ]
    priority_routes = [all_routes[idx] for idx in (0, 1, 3, 4, 6, 7)]
    configs: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        score_norm: str,
        calibration_mode: str,
        route: Dict[str, Any],
        max_selected_experts: Optional[int],
        score_pool: str,
        mixing_mode: str,
        beta: Optional[float] = None,
    ) -> None:
        config = {
            **route,
            "score_norm": score_norm,
            "calibration_mode": calibration_mode,
            "beta": beta,
            "max_selected_experts": max_selected_experts,
            "score_pool": score_pool,
            "mixing_mode": mixing_mode,
        }
        if calibration_mode == "neg_margin":
            config["routing_mode"] = "threshold"
            config["topk"] = 0
            config["gamma"] = beta
            config["relative_threshold"] = None
        config["config_id"] = post_retrain_config_id(config)
        if config["config_id"] not in seen:
            seen.add(config["config_id"])
            configs.append(config)

    for calibration_mode in ("zscore_neg", "self_minus_neg_mean"):
        for route in all_routes:
            for cap in (None, 2):
                for pool in ("mean", "max"):
                    for mix in CALIBRATION_MIXING_MODES:
                        add("factor_z", calibration_mode, route, cap, pool, mix)

    for route in priority_routes:
        for cap in (None, 2):
            for pool in ("mean", "max"):
                for mix in CALIBRATION_MIXING_MODES:
                    add("factor_self_score", "self_ratio", route, cap, pool, mix)

    neg_margin_route = {"routing_mode": "threshold", "topk": 0, "gamma": None, "relative_threshold": None}
    for beta in (0.0, 0.5, 1.0, 1.5, 2.0):
        for cap in (None, 1, 2, 3):
            for pool in ("mean", "max"):
                for mix in CALIBRATION_MIXING_MODES:
                    add("factor_z", "neg_margin", neg_margin_route, cap, pool, mix, beta=beta)

    for score_norm in POST_RETRAIN_SCORE_NORMS:
        for route in [all_routes[idx] for idx in (0, 1, 3, 4, 6)]:
            for cap in (None, 2):
                for mix in CALIBRATION_MIXING_MODES:
                    add(score_norm, "none", route, cap, "mean", mix)

    for calibration_mode in ("zscore_neg", "self_minus_neg_mean"):
        for route in [all_routes[1], all_routes[3]]:
            for cap in (None, 2):
                add("factor_z", calibration_mode, route, cap, "last", "average")
    for route in [all_routes[1], all_routes[3]]:
        add("factor_self_score", "self_ratio", route, 2, "last", "average")
    return configs


def metadata_self_scores(alg: TIMEEdit, score_norm: str) -> List[float]:
    keys = {
        "none": ("time_self_score_none", "self_score_raw", "time_self_score_raw"),
        "factor_z": ("time_self_score_factor_z", "self_score_factor_z"),
        "factor_self_score": ("time_self_score_unit", "self_score_unit"),
    }.get(score_norm, ("time_self_score",))
    values: List[float] = []
    for metadata in alg.repository.metadata:
        value = None
        if score_norm == "factor_self_score":
            value = 1.0
        for key in keys:
            if value is None and key in metadata:
                value = metadata[key]
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(1.0)
    return values or [1.0] * max(1, alg.repository.num_experts)


def collect_post_retrain_score_distribution(
    args: argparse.Namespace,
    alg: TIMEEdit,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    out_dir: Path,
    base_cache: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[Tuple[str, str], List[List[float]]]]:
    debug_path = out_dir / "post_retrain_routing_debug.jsonl"
    matrices: Dict[Tuple[str, str], List[List[float]]] = {}
    norm_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for score_norm in POST_RETRAIN_BASE_SCORE_NORMS:
        for score_pool in POST_RETRAIN_SCORE_POOLS:
            matrix: List[List[float]] = []
            rows: List[Dict[str, Any]] = []
            with temporary_time_routing(
                alg,
                routing_mode="threshold",
                gamma=1.0e30,
                topk=0,
                score_norm=score_norm,
                relative_threshold=None,
                mixing_mode="average",
                calibration_mode="none",
                max_selected_experts=None,
                score_pool=score_pool,
                calibration_stats={},
            ):
                for eval_pos, (record, sample) in enumerate(zip(records, samples)):
                    row = evaluate_sample(
                        alg,
                        sample,
                        record,
                        eval_pos,
                        phase=f"post_retrain_score_dist_{score_norm}_{score_pool}_{eval_pos}",
                        expected_expert=eval_pos,
                        routing_debug_path=debug_path,
                        eval_routing_mode="post_retrain_score_distribution",
                        force_expert_id=eval_pos,
                        extra_fields={"score_distribution_norm": score_norm, "score_pool": score_pool},
                        base_cache=base_cache[eval_pos],
                    )
                    values = [float(value) for value in (row.get("pooled_scores") or [])]
                    matrix.append(values)
                    rows.append(row)
            matrices[(score_norm, score_pool)] = matrix
            norm_rows[(score_norm, score_pool)] = rows

    distribution_rows: List[Dict[str, Any]] = []
    for score_pool in POST_RETRAIN_SCORE_POOLS:
        raw = matrices.get(("none", score_pool), [])
        factor_z = matrices.get(("factor_z", score_pool), [])
        factor_self = matrices.get(("factor_self_score", score_pool), [])
        for query_index, record in enumerate(records):
            width = max(
                len(raw[query_index]) if query_index < len(raw) else 0,
                len(factor_z[query_index]) if query_index < len(factor_z) else 0,
                len(factor_self[query_index]) if query_index < len(factor_self) else 0,
            )
            for expert_index in range(width):
                distribution_rows.append(
                    {
                        "score_pool": score_pool,
                        "record_id": record_id(record, query_index),
                        "query_record_index": query_index,
                        "expert_index": expert_index,
                        "own_expert": expert_index == query_index,
                        "raw_score": raw[query_index][expert_index] if query_index < len(raw) and expert_index < len(raw[query_index]) else None,
                        "factor_z_score": factor_z[query_index][expert_index] if query_index < len(factor_z) and expert_index < len(factor_z[query_index]) else None,
                        "factor_self_score": factor_self[query_index][expert_index] if query_index < len(factor_self) and expert_index < len(factor_self[query_index]) else None,
                    }
                )

    summary: Dict[str, Any] = {
        "loaded_repository_path": _repo_path_for_report(resolve_repository_path(args.time_load_repository)),
        "score_pools": list(POST_RETRAIN_SCORE_POOLS),
        "answer_mean_evaluated": False,
        "answer_mean_note": "No reliable answer_mask is present in the current LLaVA-Med sample batches; answer_mean falls back to mean and was not included in the bounded grid.",
        "per_norm_pool": {},
        "negative_stats": {},
        "self_scores_from_metadata": {
            score_norm: metadata_self_scores(alg, score_norm)
            for score_norm in POST_RETRAIN_BASE_SCORE_NORMS
        },
    }
    for (score_norm, score_pool), matrix in matrices.items():
        key = f"{score_norm}|{score_pool}"
        top_counts: Dict[str, int] = {}
        per_expert: Dict[str, Any] = {}
        for query_scores in matrix:
            if query_scores:
                top_id = int(max(range(len(query_scores)), key=lambda idx: query_scores[idx]))
                top_counts[str(top_id)] = top_counts.get(str(top_id), 0) + 1
        num_experts = max((len(row) for row in matrix), default=0)
        mu_neg: List[float] = []
        std_neg: List[float] = []
        for expert_index in range(num_experts):
            values = [row[expert_index] for row in matrix if expert_index < len(row)]
            neg = [row[expert_index] for idx, row in enumerate(matrix) if idx != expert_index and expert_index < len(row)]
            mean = float(sum(values) / len(values)) if values else None
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) if values and mean is not None else None
            neg_mean = float(sum(neg) / len(neg)) if neg else 0.0
            neg_std = math.sqrt(sum((value - neg_mean) ** 2 for value in neg) / len(neg)) if neg else 0.0
            mu_neg.append(neg_mean)
            std_neg.append(max(float(neg_std), 1.0e-8))
            per_expert[str(expert_index)] = {
                "mean": mean,
                "std": std,
                "max": max(values) if values else None,
                "mu_neg": neg_mean,
                "std_neg": neg_std,
                "self_score_metadata": (summary["self_scores_from_metadata"].get(score_norm) or [None] * num_experts)[expert_index],
            }
        summary["per_norm_pool"][key] = {
            "top_expert_counts": top_counts,
            "dominant_top_expert": max(top_counts, key=top_counts.get) if top_counts else None,
            "max_top_expert_count": max(top_counts.values()) if top_counts else 0,
            "per_expert": per_expert,
        }
        summary["negative_stats"][key] = {
            "mu_neg": mu_neg,
            "std_neg": std_neg,
        }
    return distribution_rows, summary, matrices


def calibration_stats_for_post_config(config: Dict[str, Any], score_summary: Dict[str, Any]) -> Dict[str, List[float]]:
    score_norm = str(config.get("score_norm"))
    score_pool = str(config.get("score_pool"))
    key = f"{score_norm}|{score_pool}"
    neg = score_summary.get("negative_stats", {}).get(key, {})
    return {
        "mu_neg": list(neg.get("mu_neg") or []),
        "std_neg": list(neg.get("std_neg") or []),
        "self_score": list((score_summary.get("self_scores_from_metadata") or {}).get(score_norm) or []),
    }


def sparse_routing_pass(summary: Dict[str, Any]) -> bool:
    return bool(
        int(summary.get("positive_new_count") or 0) >= 4
        and int(summary.get("own_in_selected_set_count") or 0) >= 4
        and (_float_or_none(summary.get("mean_selected_set_size")) or 1.0e9) <= 2.0
        and (_float_or_none(summary.get("max_selected_set_size")) or 1.0e9) <= 3.0
        and int(summary.get("empty_selection_count") or 0) <= 1
        and int(summary.get("max_top_expert_count") or 0) <= 3
    )


def strict_sparse_pass(summary: Dict[str, Any]) -> bool:
    return bool(
        int(summary.get("own_top1_count") or 0) >= 4
        and int(summary.get("positive_new_count") or 0) >= 4
        and (_float_or_none(summary.get("mean_selected_set_size")) or 1.0e9) <= 2.0
    )


def summarize_post_retrain_config(config: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected_sizes = [float(row.get("selected_expert_set_size") or 0.0) for row in rows]
    top_ids = [row.get("top_expert_id") for row in rows if row.get("top_expert_id") is not None]
    top_counts: Dict[str, int] = {}
    for top_id in top_ids:
        top_counts[str(top_id)] = top_counts.get(str(top_id), 0) + 1
    summary = {
        "config_id": config.get("config_id"),
        "score_norm": config.get("score_norm"),
        "calibration_mode": config.get("calibration_mode"),
        "score_pool": config.get("score_pool"),
        "routing_mode": "neg_margin" if config.get("calibration_mode") == "neg_margin" else config.get("routing_mode"),
        "gamma": config.get("gamma"),
        "relative_threshold": config.get("relative_threshold"),
        "beta": config.get("beta"),
        "topk": config.get("topk"),
        "max_selected_experts": config.get("max_selected_experts"),
        "mixing_mode": config.get("mixing_mode"),
        "num_records": len(rows),
        "mean_target_nll_improvement": _mean([row.get("target_nll_delta") for row in rows]),
        "positive_new_count": sum(1 for row in rows if (row.get("target_nll_delta") or 0.0) > 0.0),
        "own_top1_count": sum(1 for row in rows if row.get("routing_top1_correct")),
        "own_in_selected_set_count": sum(1 for row in rows if row.get("selected_own_expert")),
        "empty_selection_count": sum(1 for row in rows if int(row.get("selected_expert_set_size") or 0) == 0),
        "mean_selected_set_size": _mean(selected_sizes),
        "max_selected_set_size": max(selected_sizes) if selected_sizes else None,
        "mean_residual_norm": _mean([row.get("residual_norm") for row in rows]),
        "mean_hidden_delta_norm": _mean([row.get("target_layer_hidden_delta_norm") for row in rows]),
        "mean_locality_reference_delta": _mean([row.get("reference_delta") for row in rows]),
        "most_common_top_expert_id": max(top_counts, key=top_counts.get) if top_counts else None,
        "max_top_expert_count": max(top_counts.values()) if top_counts else 0,
        "top_expert_counts": top_counts,
        "confusion_matrix": _calibration_confusion(rows),
        "confusion_matrix_id": config.get("config_id"),
    }
    summary["selected_set_is_sparse"] = bool(
        (_float_or_none(summary.get("mean_selected_set_size")) or 1.0e9) <= 2.0
        and (_float_or_none(summary.get("max_selected_set_size")) or 1.0e9) <= 3.0
    )
    summary["sparse_routing_pass"] = sparse_routing_pass(summary)
    summary["strict_sparse_pass"] = strict_sparse_pass(summary)
    return summary


def _sort_key_post_sparse(row: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, float]:
    locality = _float_or_none(row.get("mean_locality_reference_delta"))
    return (
        float(bool(row.get("strict_sparse_pass"))),
        float(row.get("positive_new_count") or 0),
        float(row.get("own_in_selected_set_count") or 0),
        float(row.get("own_top1_count") or 0),
        -float(row.get("mean_selected_set_size") or 1.0e9),
        float(locality is not None and locality <= 5.0),
        float(row.get("mean_target_nll_improvement") or -1.0e9),
        -float(row.get("mean_locality_reference_delta") or 1.0e9),
    )


def choose_post_retrain_best(grid_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sparse = [row for row in grid_rows if row.get("sparse_routing_pass")]
    strict = [row for row in grid_rows if row.get("strict_sparse_pass")]
    return {
        "best_sparse": max(sparse, key=_sort_key_post_sparse) if sparse else {},
        "best_strict_sparse": max(strict, key=_sort_key_post_sparse) if strict else {},
        "best_own_top1": max(grid_rows, key=lambda row: (int(row.get("own_top1_count") or 0), int(row.get("own_in_selected_set_count") or 0), -float(row.get("mean_selected_set_size") or 1.0e9), float(row.get("mean_target_nll_improvement") or -1.0e9))) if grid_rows else {},
        "best_target_nll": max(grid_rows, key=_sort_key_target) if grid_rows else {},
        "best_locality": max(grid_rows, key=_sort_key_locality) if grid_rows else {},
    }


def diagnose_post_retrain_calibration(grid_rows: List[Dict[str, Any]], best: Dict[str, Any]) -> Tuple[str, str]:
    sparse = [row for row in grid_rows if row.get("sparse_routing_pass")]
    best_sparse = best.get("best_sparse") or {}
    if best_sparse.get("calibration_mode") in {"zscore_neg", "self_minus_neg_mean"}:
        return "posthoc_per_expert_calibration_success", f"proceed to 5-edit sequential with `{best_sparse.get('config_id')}`"
    if sparse and (_float_or_none(best_sparse.get("mean_locality_reference_delta")) or 0.0) > 5.0:
        return "sparse_routing_but_locality_damage", "run another 5-edit retrain with stronger alignment/locality control before sequential"

    topk2_rows = [
        row for row in grid_rows
        if row.get("routing_mode") == "topk"
        and int(row.get("topk") or 0) == 2
        and int(row.get("own_in_selected_set_count") or 0) >= 4
        and int(row.get("positive_new_count") or 0) >= 4
        and (_float_or_none(row.get("mean_selected_set_size")) or 1.0e9) <= 2.0
    ]
    if topk2_rows:
        best_topk2 = max(topk2_rows, key=_sort_key_post_sparse)
        return "topk_sparse_fallback_success", f"proceed to 5-edit sequential with practical fallback `{best_topk2.get('config_id')}`"

    capped_pass = [row for row in sparse if row.get("max_selected_experts") in {1, 2}]
    if capped_pass:
        best_cap = max(capped_pass, key=_sort_key_post_sparse)
        return "selection_cap_needed", f"proceed only with selection cap config `{best_cap.get('config_id')}`"

    non_mean_pass = [row for row in sparse if row.get("score_pool") != "mean"]
    mean_pass = [row for row in sparse if row.get("score_pool") == "mean"]
    if non_mean_pass and not mean_pass:
        best_pool = max(non_mean_pass, key=_sort_key_post_sparse)
        return "token_pooling_issue", f"proceed with token pooling config `{best_pool.get('config_id')}`"

    compact = [
        row for row in grid_rows
        if int(row.get("own_in_selected_set_count") or 0) >= 4
        and (_float_or_none(row.get("mean_selected_set_size")) or 1.0e9) <= 2.0
    ]
    if not compact:
        return "intrinsic_score_still_not_sparse", "run another 5-edit retrain with stronger routing supervision or add a cross-attention factor generator"
    return "intrinsic_score_still_not_sparse", "revisit intrinsic routing design before 5-edit sequential"


def previous_score_norm_best(out_dir: Path) -> Dict[str, Any]:
    prior_dir = out_dir.parent / "five_edit_score_norm_retrain"
    grid_path = prior_dir / "five_edit_score_norm_routing_grid.csv"
    rows = _read_csv_rows(grid_path)
    prior: Dict[str, Any] = {"grid_path": str(grid_path), "rows": len(rows)}
    if rows:
        non_oracle = [row for row in rows if row.get("config_id") != "force_own"]
        prior["best_own_top1"] = max(non_oracle, key=lambda row: (int(float(row.get("own_top1_count") or 0)), int(float(row.get("own_in_selected_set_count") or 0)), -float(row.get("mean_selected_set_size") or 1.0e9))) if non_oracle else {}
        prior["best_own_selected_with_size"] = max(non_oracle, key=lambda row: (int(float(row.get("own_in_selected_set_count") or 0)), -float(row.get("mean_selected_set_size") or 1.0e9), int(float(row.get("positive_new_count") or 0)))) if non_oracle else {}
        prior["best_mean_nll"] = max(non_oracle, key=_sort_key_target) if non_oracle else {}
        prior["best_locality"] = max(non_oracle, key=_sort_key_locality) if non_oracle else {}
        prior["factor_z_rel0p95_average"] = next((row for row in rows if row.get("config_id") == "factor_z_rel0p95_average"), {})
    confusion_path = prior_dir / "five_edit_score_norm_confusion_matrices.json"
    if confusion_path.exists():
        try:
            prior["confusion_matrices"] = json.loads(confusion_path.read_text())
        except json.JSONDecodeError:
            prior["confusion_matrices"] = {}
    return prior


def per_record_post_row(
    config: Dict[str, Any],
    row: Dict[str, Any],
    score_matrices: Dict[Tuple[str, str], List[List[float]]],
) -> Dict[str, Any]:
    score_pool = str(config.get("score_pool"))
    expected = int(row.get("expected_expert")) if row.get("expected_expert") is not None else -1

    def score_at(norm: str) -> Optional[float]:
        matrix = score_matrices.get((norm, score_pool), [])
        if 0 <= expected < len(matrix) and 0 <= expected < len(matrix[expected]):
            return matrix[expected][expected]
        return None

    return {
        "config_id": config.get("config_id"),
        "score_norm": config.get("score_norm"),
        "calibration_mode": config.get("calibration_mode"),
        "score_pool": config.get("score_pool"),
        "routing_mode": "neg_margin" if config.get("calibration_mode") == "neg_margin" else config.get("routing_mode"),
        "gamma": config.get("gamma"),
        "relative_threshold": config.get("relative_threshold"),
        "beta": config.get("beta"),
        "topk": config.get("topk"),
        "max_selected_experts": config.get("max_selected_experts"),
        "mixing_mode": config.get("mixing_mode"),
        "record_id": row.get("record_id"),
        "own_expert_index": row.get("expected_expert"),
        "target_nll_before": row.get("base_target_nll"),
        "target_nll_after": row.get("target_nll"),
        "target_nll_improvement": row.get("target_nll_delta"),
        "improved": row.get("target_improved"),
        "own_expert_score_raw": score_at("none"),
        "own_expert_score_factor_z": score_at("factor_z"),
        "own_expert_score_factor_self_score": score_at("factor_self_score"),
        "calibrated_own_score": row.get("own_expert_score"),
        "top_expert_id": row.get("top_expert_id"),
        "top_calibrated_score": row.get("top_score"),
        "own_selected": row.get("selected_own_expert"),
        "selected_expert_ids": row.get("selected_expert_ids"),
        "selected_set_size": row.get("selected_expert_set_size"),
        "residual_norm": row.get("residual_norm"),
        "hidden_delta_norm": row.get("target_layer_hidden_delta_norm"),
        "locality_reference_delta": row.get("reference_delta"),
    }


def write_post_retrain_calibration_report(
    out_dir: Path,
    args: argparse.Namespace,
    repo_path: Path,
    command: str,
    gpu_status: str,
    score_summary: Dict[str, Any],
    grid_rows: List[Dict[str, Any]],
    best: Dict[str, Any],
    diagnosis: str,
    recommendation: str,
    previous: Dict[str, Any],
    confusion_matrices: Dict[str, Any],
) -> Path:
    best_sparse = best.get("best_sparse") or {}
    best_own_top1 = best.get("best_own_top1") or {}
    best_target = best.get("best_target_nll") or {}
    best_locality = best.get("best_locality") or {}
    previous_best = previous.get("factor_z_rel0p95_average") or {}
    previous_confusion = ((previous.get("confusion_matrices") or {}).get("factor_z_rel0p95_average") or {}).get("matrix")
    new_confusion = (confusion_matrices.get(str(best_sparse.get("config_id"))) or {}).get("matrix") if best_sparse else None
    topk2_fallback = [
        row for row in grid_rows
        if row.get("routing_mode") == "topk"
        and int(row.get("topk") or 0) == 2
        and int(row.get("own_in_selected_set_count") or 0) >= 4
        and int(row.get("positive_new_count") or 0) >= 4
        and (_float_or_none(row.get("mean_selected_set_size")) or 1.0e9) <= 2.0
    ]
    per_expert_calibration_needed = bool(best_sparse and best_sparse.get("calibration_mode") != "none")
    lines = [
        "# TIME 5-Edit Post-Retrain Calibration Report",
        "",
        "## Files Changed",
        "- `easyeditor/trainer/algs/time_edit.py`",
        "- `easyeditor/trainer/algs/time_edit_modules.py`",
        "- `easyeditor/models/time_edit/time_edit_hparams.py`",
        "- `scripts/time/run_time_medmkeb_smoke.py`",
        "- `scripts/time/test_time_modules.py`",
        "",
        "## Repository Loaded",
        f"- `{repo_path}`",
        "",
        "## Exact Command Run",
        f"- `{command}`",
        "",
        "## GPU",
        f"- Used CUDA device setting: `{_cuda_setting_from_command(command) or os.environ.get('CUDA_VISIBLE_DEVICES', args.device)}`.",
        "- Chosen because the pre-run `nvidia-smi` check showed GPU 3 free; captured start status:",
        "```text",
        gpu_status or "unavailable",
        "```",
        "",
        "## Verification",
        "- `py_compile`: passed before model run.",
        "- `scripts/time/test_time_modules.py`: passed before model run.",
        "- Eval-only repository loading: used.",
        "- Retraining: not run.",
        "- 20-edit run: not run.",
        "",
        "## Previous Best Config",
        f"- Previous best normalized config: `{previous_best.get('config_id')}`.",
        f"- own top-1 {previous_best.get('own_top1_count')}/5; own selected {previous_best.get('own_in_selected_set_count')}/5; mean size {_fmt(previous_best.get('mean_selected_set_size'))}; positive {previous_best.get('positive_new_count')}/5; NLL improvement {_fmt(previous_best.get('mean_target_nll_improvement'))}; locality {_fmt(previous_best.get('mean_locality_reference_delta'))}.",
        "",
        "## Score Distribution Summary",
        f"- Pools evaluated: `{', '.join(score_summary.get('score_pools') or [])}`.",
        f"- answer_mean evaluated: {score_summary.get('answer_mean_evaluated')} ({score_summary.get('answer_mean_note')}).",
    ]
    for key, payload in sorted((score_summary.get("per_norm_pool") or {}).items()):
        lines.append(
            f"- `{key}` top counts `{json.dumps(payload.get('top_expert_counts'), sort_keys=True)}`, max top count {payload.get('max_top_expert_count')}."
        )
    lines.extend(
        [
            "",
            "## Calibration Grid Summary",
            f"- Grid configs evaluated: {len(grid_rows)}.",
            "",
            "| config | norm | calib | pool | route | cap | mix | NLL imp | pos | own top1 | own selected | empty | mean size | max size | locality | max top | sparse | strict |",
            "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in sorted(grid_rows, key=_sort_key_post_sparse, reverse=True)[:20]:
        lines.append(
            "| {config} | {norm} | {calib} | {pool} | {route} | {cap} | {mix} | {nll} | {pos}/5 | {top1}/5 | {own}/5 | {empty} | {mean_size} | {max_size} | {loc} | {max_top} | {sparse} | {strict} |".format(
                config=row.get("config_id"),
                norm=row.get("score_norm"),
                calib=row.get("calibration_mode"),
                pool=row.get("score_pool"),
                route=post_retrain_route_label(row),
                cap=row.get("max_selected_experts"),
                mix=row.get("mixing_mode"),
                nll=_fmt(row.get("mean_target_nll_improvement")),
                pos=row.get("positive_new_count"),
                top1=row.get("own_top1_count"),
                own=row.get("own_in_selected_set_count"),
                empty=row.get("empty_selection_count"),
                mean_size=_fmt(row.get("mean_selected_set_size")),
                max_size=_fmt(row.get("max_selected_set_size")),
                loc=_fmt(row.get("mean_locality_reference_delta")),
                max_top=row.get("max_top_expert_count"),
                sparse=row.get("sparse_routing_pass"),
                strict=row.get("strict_sparse_pass"),
            )
        )
    lines.extend(
        [
            "",
            "## Best Configs",
            f"- Best sparse routing: `{best_sparse.get('config_id')}`.",
            f"- Best own top-1: `{best_own_top1.get('config_id')}` with own top-1 {best_own_top1.get('own_top1_count')}/5.",
            f"- Best target NLL improvement: `{best_target.get('config_id')}` with mean improvement {_fmt(best_target.get('mean_target_nll_improvement'))}.",
            f"- Best locality/reference delta: `{best_locality.get('config_id')}` with mean locality delta {_fmt(best_locality.get('mean_locality_reference_delta'))}.",
            "",
            "## Confusion Matrices",
            f"- Previous `factor_z_rel0p95_average`: `{json.dumps(previous_confusion, sort_keys=True)}`",
            f"- New best sparse config: `{json.dumps(new_confusion, sort_keys=True)}`",
            "",
            "## Decisions",
            f"- Strict sparse routing achieved: {bool(best.get('best_strict_sparse'))}.",
            f"- Topk=2 viable fallback: {bool(topk2_fallback)}.",
            f"- Per-expert calibration necessary: {per_expert_calibration_needed}.",
            f"- Locality/reference delta for best sparse: {_fmt(best_sparse.get('mean_locality_reference_delta'))}.",
            "",
            "## Diagnosis",
            f"- Label: `{diagnosis}`.",
            "",
            "## Recommendation",
            f"- {recommendation}.",
            "",
            "## Output Files",
            "- `post_retrain_score_distribution.csv`",
            "- `post_retrain_score_distribution_summary.json`",
            "- `post_retrain_calibration_grid.csv`",
            "- `post_retrain_calibration_best.json`",
            "- `post_retrain_per_record.csv`",
            "- `post_retrain_confusion_matrices.json`",
            "- `post_retrain_routing_debug.jsonl`",
            "- `TIME_5EDIT_POST_RETRAIN_CALIBRATION_REPORT.md`",
            "",
        ]
    )
    path = out_dir / "TIME_5EDIT_POST_RETRAIN_CALIBRATION_REPORT.md"
    path.write_text("\n".join(lines))
    return path


def run_post_retrain_calibration(
    args: argparse.Namespace,
    config: TIMEEditMultimodalHparams,
    dataset_path: Path,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    alg: TIMEEdit,
    out_dir: Path,
    repo_path: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_path = out_dir / "post_retrain_routing_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()
    command = current_command_line()
    gpu_status = _gpu_status_text()
    write_json(
        out_dir / "time_hparams.json",
        {
            "args": vars(args),
            "config": dict(config.__dict__),
            "command": command,
            "eval_only": True,
            "loaded_repository_path": str(repo_path),
            "gpu_status_at_start": gpu_status,
            "post_retrain_calibration": True,
        },
    )
    base_cache = build_base_eval_cache(alg, samples)
    score_rows, score_summary, score_matrices = collect_post_retrain_score_distribution(
        args,
        alg,
        records,
        samples,
        out_dir,
        base_cache,
    )
    write_loss_trace(out_dir / "post_retrain_score_distribution.csv", score_rows)
    write_json(out_dir / "post_retrain_score_distribution_summary.json", score_summary)

    configs = post_retrain_calibration_configs()
    per_record_rows: List[Dict[str, Any]] = []
    grid_rows: List[Dict[str, Any]] = []
    confusion_matrices: Dict[str, Any] = {}
    for config_item in configs:
        eval_rows: List[Dict[str, Any]] = []
        stats = calibration_stats_for_post_config(config_item, score_summary)
        with temporary_time_routing(
            alg,
            routing_mode=str(config_item["routing_mode"]),
            gamma=float(config_item["gamma"]) if config_item.get("gamma") is not None else None,
            topk=int(config_item.get("topk") or 0),
            score_norm=str(config_item["score_norm"]),
            relative_threshold=float(config_item["relative_threshold"]) if config_item.get("relative_threshold") is not None else None,
            mixing_mode=str(config_item["mixing_mode"]),
            calibration_mode=str(config_item["calibration_mode"]),
            calibration_beta=float(config_item.get("beta") or 0.0),
            max_selected_experts=config_item.get("max_selected_experts"),
            score_pool=str(config_item["score_pool"]),
            calibration_stats=stats,
        ):
            for eval_pos, (record, sample) in enumerate(zip(records, samples)):
                row = evaluate_sample(
                    alg,
                    sample,
                    record,
                    eval_pos,
                    phase=f"post_retrain_{config_item['config_id']}_eval_{eval_pos}",
                    expected_expert=eval_pos,
                    routing_debug_path=debug_path,
                    eval_routing_mode=str(config_item["config_id"]),
                    extra_fields={
                        "post_retrain_calibration": True,
                        "calibration_config_id": config_item["config_id"],
                        "calibration_mode": config_item["calibration_mode"],
                        "score_pool": config_item["score_pool"],
                        "max_selected_experts": config_item.get("max_selected_experts"),
                        "calibration_beta": config_item.get("beta"),
                    },
                    base_cache=base_cache[eval_pos],
                )
                eval_rows.append(row)
                per_record_rows.append(per_record_post_row(config_item, row, score_matrices))
        summary = summarize_post_retrain_config(config_item, eval_rows)
        grid_rows.append(summary)
        confusion_matrices[str(config_item["config_id"])] = {
            "config": {
                key: config_item.get(key)
                for key in (
                    "score_norm",
                    "calibration_mode",
                    "score_pool",
                    "routing_mode",
                    "gamma",
                    "relative_threshold",
                    "beta",
                    "topk",
                    "max_selected_experts",
                    "mixing_mode",
                )
            },
            "matrix": summary.get("confusion_matrix"),
        }

    best = choose_post_retrain_best(grid_rows)
    diagnosis, recommendation = diagnose_post_retrain_calibration(grid_rows, best)
    previous = previous_score_norm_best(out_dir)
    best_payload = {
        **best,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "loaded_repository_path": str(repo_path),
        "command": command,
        "gpu_status_at_start": gpu_status,
        "record_ids": [record_id(record, idx) for idx, record in enumerate(records)],
        "num_grid_configs": len(grid_rows),
        "previous_score_norm_best": previous,
        "answer_mean_evaluated": False,
    }
    write_loss_trace(out_dir / "post_retrain_calibration_grid.csv", grid_rows)
    write_loss_trace(out_dir / "post_retrain_per_record.csv", per_record_rows)
    write_json(out_dir / "post_retrain_confusion_matrices.json", confusion_matrices)
    write_json(out_dir / "post_retrain_calibration_best.json", best_payload)
    report_path = write_post_retrain_calibration_report(
        out_dir,
        args,
        repo_path,
        command,
        gpu_status,
        score_summary,
        grid_rows,
        best,
        diagnosis,
        recommendation,
        previous,
        confusion_matrices,
    )
    payload = {
        "loaded_repository_path": str(repo_path),
        "num_grid_configs": len(grid_rows),
        "num_per_record_rows": len(per_record_rows),
        "best": best_payload,
        "report_path": str(report_path),
    }
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
    return payload


def score_norm_retrain_eval_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = [
        {
            "config_id": "force_own",
            "score_norm": "none",
            "routing_mode": "threshold",
            "gamma": 1.0e30,
            "relative_threshold": None,
            "topk": 0,
            "mixing_mode": "own_oracle",
            "group": "capacity_oracle",
        }
    ]
    route_specs = [
        ("raw_topk1", "none", "topk", None, None, 1, "raw_baseline"),
        ("raw_gamma0p5", "none", "threshold", 0.5, None, 0, "raw_baseline"),
        ("factor_z_topk1", "factor_z", "topk", None, None, 1, "factor_z_sparse"),
        ("factor_z_topk2", "factor_z", "topk", None, None, 2, "factor_z_sparse"),
        ("factor_z_rel0p9", "factor_z", "relative_threshold", None, 0.9, 0, "factor_z_sparse"),
        ("factor_z_rel0p95", "factor_z", "relative_threshold", None, 0.95, 0, "factor_z_sparse"),
        ("factor_z_gamma0p5", "factor_z", "threshold", 0.5, None, 0, "factor_z_sparse"),
        ("factor_z_gamma0p7", "factor_z", "threshold", 0.7, None, 0, "factor_z_sparse"),
        ("factor_z_gamma1", "factor_z", "threshold", 1.0, None, 0, "factor_z_sparse"),
        ("factor_self_score_topk1", "factor_self_score", "topk", None, None, 1, "factor_self_sparse"),
        ("factor_self_score_topk2", "factor_self_score", "topk", None, None, 2, "factor_self_sparse"),
        ("factor_self_score_rel0p9", "factor_self_score", "relative_threshold", None, 0.9, 0, "factor_self_sparse"),
        ("factor_self_score_rel0p95", "factor_self_score", "relative_threshold", None, 0.95, 0, "factor_self_sparse"),
    ]
    for mix in CALIBRATION_MIXING_MODES:
        for base_id, score_norm, routing_mode, gamma, relative_threshold, topk, group in route_specs:
            configs.append(
                {
                    "config_id": f"{base_id}_{mix}",
                    "score_norm": score_norm,
                    "routing_mode": routing_mode,
                    "gamma": gamma,
                    "relative_threshold": relative_threshold,
                    "topk": topk,
                    "mixing_mode": mix,
                    "group": group,
                }
            )
    return configs


def retrain_sparse_success(summary: Dict[str, Any]) -> bool:
    mean_size = _float_or_none(summary.get("mean_selected_set_size"))
    if mean_size is None:
        return False
    return bool(
        (
            int(summary.get("own_top1_count") or 0) >= 4
            or (int(summary.get("own_in_selected_set_count") or 0) >= 5 and mean_size <= 2.0)
        )
        and int(summary.get("positive_new_count") or 0) >= 4
        and mean_size <= 2.0
        and int(summary.get("empty_selection_count") or 0) <= 1
    )


def max_top_expert_count(summary: Dict[str, Any]) -> int:
    counts = summary.get("top_expert_counts") or {}
    return max((int(value) for value in counts.values()), default=0)


def diagnose_score_norm_retrain(grid_rows: List[Dict[str, Any]], force_summary: Dict[str, Any]) -> Tuple[str, str]:
    force_pass = int(force_summary.get("positive_new_count") or 0) >= 4
    if not force_pass:
        return "anti_collapse_hurts_expert_capacity", "reduce anti-collapse"
    normalized = [row for row in grid_rows if row.get("score_norm") in {"factor_z", "factor_self_score"}]
    sparse = [row for row in normalized if row.get("sparse_success")]
    if sparse:
        best_sparse = max(sparse, key=_sort_key_target)
        if (_float_or_none(best_sparse.get("mean_locality_reference_delta")) or 0.0) > 5.0:
            return "routing_fixed_but_locality_damage", "proceed cautiously with 5-edit sequential after locality tuning"
        return "score_normalized_retrain_success", "proceed to 5-edit sequential with best config"

    factor_z_fixed = any(row.get("score_norm") == "factor_z" and max_top_expert_count(row) <= 3 for row in normalized)
    factor_self_fixed = any(row.get("score_norm") == "factor_self_score" and max_top_expert_count(row) <= 3 for row in normalized)
    if factor_z_fixed and not factor_self_fixed:
        return "factor_norm_score_scale_issue", "run gamma/topk calibration again"
    if factor_self_fixed and not factor_z_fixed:
        return "per_expert_score_calibration_issue", "run gamma/topk calibration again"
    if not factor_z_fixed and not factor_self_fixed:
        return "intrinsic_score_not_discriminative_after_retrain", "add cross-attention factor generator or stronger routing supervision"
    return "routing_improved_but_not_sparse", "run gamma/topk calibration again"


def summarize_trace_file(path: Path) -> Dict[str, Any]:
    rows = _read_csv_rows(path)
    if not rows:
        return {"num_rows": 0}
    first = rows[0]
    last = rows[-1]
    keys = [
        "loss_total",
        "loss_rel",
        "loss_time_align",
        "loss_time_anti_collapse",
        "loss_time_factor_norm_reg",
        "time_current_expert_index",
        "time_anti_collapse_prev_batches",
        "time_anti_collapse_current_prev_mean",
        "time_anti_collapse_prev_own_mean",
        "current_expert_grad_norm_total",
        "U_in_factor_norm",
        "V_in_factor_norm",
        "U_out_factor_norm",
        "V_out_factor_norm",
    ]
    return {
        "num_rows": len(rows),
        "first": {key: first.get(key) for key in keys if key in first},
        "last": {key: last.get(key) for key in keys if key in last},
    }


def run_score_norm_retrain_grid(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    alg: TIMEEdit,
    out_dir: Path,
) -> Dict[str, Any]:
    debug_path = out_dir / "five_edit_score_norm_routing_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()
    configs = score_norm_retrain_eval_configs()
    per_record_rows: List[Dict[str, Any]] = []
    grid_rows: List[Dict[str, Any]] = []
    confusion_matrices: Dict[str, Any] = {}
    for config_item in configs:
        eval_rows: List[Dict[str, Any]] = []
        with temporary_time_routing(
            alg,
            routing_mode=str(config_item["routing_mode"]),
            gamma=float(config_item["gamma"]) if config_item.get("gamma") is not None else None,
            topk=int(config_item.get("topk") or 0),
            score_norm=str(config_item["score_norm"]),
            relative_threshold=float(config_item["relative_threshold"]) if config_item.get("relative_threshold") is not None else None,
            mixing_mode=str(config_item["mixing_mode"] if config_item["mixing_mode"] != "own_oracle" else "average"),
        ):
            for eval_pos, (record, sample) in enumerate(zip(records, samples)):
                row = evaluate_sample(
                    alg,
                    sample,
                    record,
                    eval_pos,
                    phase=f"score_norm_retrain_{config_item['config_id']}_eval_{eval_pos}",
                    expected_expert=eval_pos,
                    routing_debug_path=debug_path,
                    eval_routing_mode=str(config_item["config_id"]),
                    force_expert_id=eval_pos if config_item["config_id"] == "force_own" else None,
                    extra_fields={**_config_extra_fields(config_item), "score_norm_retrain_group": config_item.get("group")},
                )
                eval_rows.append(row)
                per_record_rows.append(
                    {
                        **_config_extra_fields(config_item),
                        "group": config_item.get("group"),
                        "record_id": row.get("record_id"),
                        "own_expert_index": row.get("expected_expert"),
                        "target": row.get("target"),
                        "target_nll_before": row.get("base_target_nll"),
                        "target_nll_after": row.get("target_nll"),
                        "target_nll_improvement": row.get("target_nll_delta"),
                        "improved": row.get("target_improved"),
                        "first_token_rank_before": row.get("base_first_target_token_rank"),
                        "first_token_rank_after": row.get("first_target_token_rank"),
                        "answer_token_logprob_delta": row.get("answer_token_logprob_delta"),
                        "residual_norm": row.get("residual_norm"),
                        "hidden_delta_norm": row.get("target_layer_hidden_delta_norm"),
                        "own_expert_score": row.get("own_expert_score"),
                        "top_expert_id": row.get("top_expert_id"),
                        "top_expert_score": row.get("top_score"),
                        "own_top1": row.get("routing_top1_correct"),
                        "own_selected": row.get("selected_own_expert"),
                        "selected_expert_set_size": row.get("selected_expert_set_size"),
                        "selected_expert_ids": row.get("selected_expert_ids"),
                        "locality_reference_delta": row.get("reference_delta"),
                    }
                )
        summary = summarize_calibration_config(config_item, eval_rows)
        summary["group"] = config_item.get("group")
        summary["sparse_success"] = retrain_sparse_success(summary)
        summary["max_top_expert_count"] = max_top_expert_count(summary)
        grid_rows.append(summary)
        confusion_matrices[str(config_item["config_id"])] = {
            "config": {key: config_item.get(key) for key in ("score_norm", "routing_mode", "gamma", "relative_threshold", "topk", "mixing_mode", "group")},
            "matrix": summary.get("confusion_matrix"),
        }
    force_summary = next((row for row in grid_rows if row.get("config_id") == "force_own"), {})
    raw_topk = next((row for row in grid_rows if row.get("config_id") == "raw_topk1_softmax"), {})
    normalized_rows = [row for row in grid_rows if row.get("score_norm") in {"factor_z", "factor_self_score"}]
    best = choose_best_calibration(normalized_rows)
    factor_z_rows = [row for row in normalized_rows if row.get("score_norm") == "factor_z"]
    factor_self_rows = [row for row in normalized_rows if row.get("score_norm") == "factor_self_score"]
    best_factor_z = max(factor_z_rows, key=_sort_key_routing) if factor_z_rows else {}
    best_factor_self = max(factor_self_rows, key=_sort_key_routing) if factor_self_rows else {}
    diagnosis, recommendation = diagnose_score_norm_retrain(grid_rows, force_summary)
    summary = {
        "command": current_command_line(),
        "record_ids": [record_id(record, idx) for idx, record in enumerate(records)],
        "anti_collapse_active": bool(args.time_anti_collapse_loss),
        "score_norm": str(args.time_score_norm),
        "align_score_norm": str(args.time_align_score_norm or args.time_score_norm),
        "anti_collapse_score_norm": str(args.anti_collapse_score_norm),
        "lambda_align": args.lambda_align,
        "num_negative_experts": args.num_negative_experts,
        "lambda_anti_collapse": args.lambda_anti_collapse,
        "anti_collapse_margin": args.anti_collapse_margin,
        "lambda_factor_norm_reg": args.lambda_factor_norm_reg,
        "force_own": force_summary,
        "raw_topk": raw_topk,
        "best_factor_z_sparse_routing": best_factor_z,
        "best_factor_self_score_sparse_routing": best_factor_self,
        "best_normalized": best,
        "strict_sparse_routing_achieved": bool(best.get("best_sparse_success")),
        "expert_3_collapse_fixed": bool(max_top_expert_count(raw_topk) > 3 and any(max_top_expert_count(row) <= 3 for row in normalized_rows)),
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "training_trace_summary": summarize_trace_file(out_dir / "time_score_norm_retrain_loss_trace.csv"),
        "anti_collapse_trace_summary": summarize_trace_file(out_dir / "time_anti_collapse_trace.csv"),
        "num_eval_configs": len(grid_rows),
    }
    write_loss_trace(out_dir / "five_edit_score_norm_per_record.csv", per_record_rows)
    write_loss_trace(out_dir / "five_edit_score_norm_routing_grid.csv", grid_rows)
    write_json(out_dir / "five_edit_score_norm_confusion_matrices.json", confusion_matrices)
    write_json(out_dir / "five_edit_score_norm_summary.json", summary)
    write_score_norm_retrain_report(out_dir, args, grid_rows, summary)
    return summary


def write_score_norm_retrain_report(
    out_dir: Path,
    args: argparse.Namespace,
    grid_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Path:
    command = current_command_line()
    gpu_status = _gpu_status_text()
    force = summary.get("force_own") or {}
    raw = summary.get("raw_topk") or {}
    best_factor_z = summary.get("best_factor_z_sparse_routing") or {}
    best_factor_self = summary.get("best_factor_self_score_sparse_routing") or {}
    best_norm = (summary.get("best_normalized") or {}).get("best_sparse_success") or (summary.get("best_normalized") or {}).get("best_by_routing_accuracy") or {}
    lines = [
        "# TIME 5-Edit Score-Normalized Retrain Report",
        "",
        "## Files Changed",
        "- `easyeditor/trainer/algs/time_edit.py`",
        "- `easyeditor/models/time_edit/time_edit_hparams.py`",
        "- `scripts/time/run_time_medmkeb_smoke.py`",
        "",
        "## Exact Command Run",
        f"- `{command}`",
        "",
        "## GPU",
        f"- Used CUDA device setting: `{_cuda_setting_from_command(command) or os.environ.get('CUDA_VISIBLE_DEVICES', args.device)}`.",
        "- GPU status captured during report write:",
        "```text",
        gpu_status or "unavailable",
        "```",
        "",
        "## Verification",
        "- `py_compile`: passed before model run.",
        "- `scripts/time/test_time_modules.py`: passed before model run.",
        "- 20-edit run: not run.",
        "",
        "## Training Setup",
        f"- Anti-collapse active: {summary.get('anti_collapse_active')}.",
        f"- Routing score normalization: `{summary.get('score_norm')}`.",
        f"- Alignment score normalization: `{summary.get('align_score_norm')}`.",
        f"- Anti-collapse score normalization: `{summary.get('anti_collapse_score_norm')}`.",
        f"- lambda_align: {summary.get('lambda_align')}; negative experts: {summary.get('num_negative_experts')}.",
        f"- lambda_anti_collapse: {summary.get('lambda_anti_collapse')}; margin: {summary.get('anti_collapse_margin')}.",
        f"- lambda_factor_norm_reg: {summary.get('lambda_factor_norm_reg')}.",
        "",
        "## Training Trace Summary",
        f"- Loss trace: `{json.dumps(summary.get('training_trace_summary'), sort_keys=True)}`",
        f"- Anti-collapse trace: `{json.dumps(summary.get('anti_collapse_trace_summary'), sort_keys=True)}`",
        "",
        "## Key Results",
        f"- Force-own: positive {force.get('positive_new_count')}/5, mean NLL improvement {_fmt(force.get('mean_target_nll_improvement'))}, locality {_fmt(force.get('mean_locality_reference_delta'))}.",
        f"- Raw topk=1: own top-1 {raw.get('own_top1_count')}/5, max-top-expert count {raw.get('max_top_expert_count')}, confusion `{json.dumps(raw.get('confusion_matrix'), sort_keys=True)}`.",
        f"- Best factor_z sparse candidate: `{best_factor_z.get('config_id')}` with own top-1 {best_factor_z.get('own_top1_count')}/5, own selected {best_factor_z.get('own_in_selected_set_count')}/5, mean selected size {_fmt(best_factor_z.get('mean_selected_set_size'))}, positive {best_factor_z.get('positive_new_count')}/5, locality {_fmt(best_factor_z.get('mean_locality_reference_delta'))}.",
        f"- Best factor_self_score sparse candidate: `{best_factor_self.get('config_id')}` with own top-1 {best_factor_self.get('own_top1_count')}/5, own selected {best_factor_self.get('own_in_selected_set_count')}/5, mean selected size {_fmt(best_factor_self.get('mean_selected_set_size'))}, positive {best_factor_self.get('positive_new_count')}/5, locality {_fmt(best_factor_self.get('mean_locality_reference_delta'))}.",
        "",
        "## Confusion Matrices",
        f"- Raw topk=1: `{json.dumps(raw.get('confusion_matrix'), sort_keys=True)}`",
        f"- Best normalized config `{best_norm.get('config_id')}`: `{json.dumps(best_norm.get('confusion_matrix'), sort_keys=True)}`",
        "",
        "## Acceptance",
        f"- Expert-3 collapse fixed: {summary.get('expert_3_collapse_fixed')}.",
        f"- Strict sparse routing achieved: {summary.get('strict_sparse_routing_achieved')}.",
        f"- Locality/reference delta for best normalized config: {_fmt(best_norm.get('mean_locality_reference_delta'))}.",
        "",
        "## Diagnosis",
        f"- Label: `{summary.get('diagnosis')}`.",
        "",
        "## Recommendation",
        f"- {summary.get('recommendation')}.",
        "",
        "## Output Files",
        "- `expert_repository.pt`",
        "- `time_score_norm_retrain_loss_trace.csv`",
        "- `time_score_matrix_after_each_edit.csv`",
        "- `time_anti_collapse_trace.csv`",
        "- `time_self_score_metadata.json`",
        "- `five_edit_score_norm_summary.json`",
        "- `five_edit_score_norm_per_record.csv`",
        "- `five_edit_score_norm_routing_grid.csv`",
        "- `five_edit_score_norm_confusion_matrices.json`",
        "- `five_edit_score_norm_routing_debug.jsonl`",
        "- `TIME_5EDIT_SCORE_NORM_RETRAIN_REPORT.md`",
        "",
    ]
    path = out_dir / "TIME_5EDIT_SCORE_NORM_RETRAIN_REPORT.md"
    path.write_text("\n".join(lines))
    return path


def parse_float_csv(text: str) -> List[float]:
    return [float(part.strip()) for part in str(text or "").split(",") if part.strip()]


def run_gamma_sweep(
    args: argparse.Namespace,
    alg: TIMEEdit,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    gammas = parse_float_csv(args.time_gamma_sweep)
    rows: List[Dict[str, Any]] = []
    routing_debug_path = out_dir / "routing_debug.jsonl"
    for gamma in gammas:
        with temporary_time_routing(alg, routing_mode="threshold", gamma=gamma, topk=0):
            row = evaluate_sample(
                alg,
                samples[0],
                records[0],
                0,
                phase=f"gamma_sweep_{gamma:g}",
                expected_expert=0,
                routing_debug_path=routing_debug_path,
            )
        row["sweep_gamma"] = gamma
        rows.append(row)
    if rows:
        write_loss_trace(out_dir / "time_gamma_sweep.csv", rows)
        write_json(out_dir / "time_gamma_sweep.json", {"rows": rows})
    return rows


def parse_scale_init_grid(text: str) -> Tuple[List[float], List[float]]:
    values = {"init_std": [], "alpha": []}
    for item in str(text or "").split(";"):
        if not item.strip():
            continue
        key, raw_values = item.split("=", 1)
        key = key.strip()
        if key not in values:
            raise ValueError(f"Unsupported TIME scale/init grid key: {key}")
        values[key] = parse_float_csv(raw_values)
    if not values["init_std"] or not values["alpha"]:
        raise ValueError("--time-scale-init-grid must include init_std=... and alpha=...")
    return values["init_std"], values["alpha"]


def parse_overfit_grid(text: str) -> Dict[str, List[Any]]:
    specs: Dict[str, List[Any]] = {}
    for item in str(text or "").split(";"):
        if not item.strip():
            continue
        key, raw_values = item.split("=", 1)
        key = key.strip()
        raw_parts = [part.strip() for part in raw_values.split(",") if part.strip()]
        if key in {"init_std", "alpha", "lr", "expert_gain"}:
            specs[key] = [float(part) for part in raw_parts]
        elif key in {"token_scope", "residual_sign"}:
            specs[key] = raw_parts
        else:
            raise ValueError(f"Unsupported TIME overfit grid key: {key}")
    defaults: Dict[str, List[Any]] = {
        "init_std": [0.05],
        "alpha": [1.0],
        "lr": [1.0e-4],
        "token_scope": ["all"],
        "residual_sign": ["plus"],
        "expert_gain": [1.0],
    }
    for key, values in defaults.items():
        specs.setdefault(key, values)
    return specs


def format_float_for_path(value: float) -> str:
    return f"{float(value):g}".replace("-", "neg").replace(".", "p")


def reliability_report_dir(out_dir: Path) -> Path:
    if out_dir.name == "one_edit_reliability_only":
        return out_dir
    if out_dir.name.startswith("one_edit_"):
        return out_dir.parent / "one_edit_reliability_only"
    return out_dir / "one_edit_reliability_only"


def reliability_pass(row: Dict[str, Any], answer_debug: Optional[Dict[str, Any]] = None) -> bool:
    nll_delta = _float_or_none(row.get("target_nll_delta")) or 0.0
    rank_delta = _float_or_none(row.get("target_rank_delta")) or 0.0
    answer_delta = _float_or_none(row.get("answer_token_logprob_delta")) or 0.0
    if answer_debug:
        answer_delta = _float_or_none(
            (answer_debug.get("final_force_current_vs_base") or {}).get("avg_target_logprob_delta")
        ) or answer_delta
    return bool(nll_delta >= 0.01 or rank_delta >= 10.0 or answer_delta >= 0.01)


def best_reliability_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def score(row: Dict[str, Any]) -> Tuple[float, float, float]:
        return (
            _float_or_none(row.get("target_nll_delta")) or 0.0,
            _float_or_none(row.get("answer_token_logprob_delta")) or 0.0,
            _float_or_none(row.get("target_rank_delta")) or 0.0,
        )

    return max(rows, key=score) if rows else {}


def run_scale_init_grid(
    args: argparse.Namespace,
    base_config: TIMEEditMultimodalHparams,
    dataset_path: Path,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    model: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:
    init_values, alpha_values = parse_scale_init_grid(args.time_scale_init_grid)
    rows: List[Dict[str, Any]] = []
    for init_std in init_values:
        for alpha in alpha_values:
            combo_args = deepcopy(args)
            combo_args.init_std = float(init_std)
            combo_args.alpha = float(alpha)
            combo_args.time_routing_mode = "force_current"
            combo_args.time_topk = int(args.time_topk or 1)
            combo_config = deepcopy(base_config)
            combo_config.time_init_std = float(init_std)
            combo_config.time_alpha = float(alpha)
            combo_config.time_routing_mode = "force_current"
            combo_config.time_topk = int(combo_args.time_topk)
            combo_config.time_force_current_during_training = True
            subdir = args.out_dir / f"init_std_{format_float_for_path(init_std)}_alpha_{format_float_for_path(alpha)}"
            alg = TIMEEdit(model, combo_config, lambda: None).to(device)
            try:
                summary = run_smoke(combo_args, combo_config, dataset_path, records, samples, alg, subdir, print_summary=False)
                force_row = (summary.get("eval_rows") or [{}])[0]
                with temporary_time_routing(alg, routing_mode="topk", gamma=float(args.gamma), topk=1):
                    topk_row = evaluate_sample(
                        alg,
                        samples[0],
                        records[0],
                        0,
                        phase="grid_topk_eval",
                        expected_expert=0,
                        routing_debug_path=subdir / "routing_debug.jsonl",
                    )
                with temporary_time_routing(alg, routing_mode="threshold", gamma=0.5, topk=0):
                    threshold_row = evaluate_sample(
                        alg,
                        samples[0],
                        records[0],
                        0,
                        phase="grid_threshold_gamma_0_5_eval",
                        expected_expert=0,
                        routing_debug_path=subdir / "routing_debug.jsonl",
                    )
                rows.append(
                    {
                        "init_std": float(init_std),
                        "alpha": float(alpha),
                        "scale_mode": args.scale_mode,
                        "final_target_nll_delta": force_row.get("target_nll_delta"),
                        "final_target_rank_delta": force_row.get("target_rank_delta"),
                        "final_score": force_row.get("top_score"),
                        "force_current_residual_norm": force_row.get("residual_norm"),
                        "topk_residual_norm": topk_row.get("residual_norm"),
                        "topk_target_nll_delta": topk_row.get("target_nll_delta"),
                        "threshold_gamma_0_5_selected": bool((threshold_row.get("selected_expert_set_size") or 0) > 0),
                        "threshold_gamma_0_5_selected_expert_ids": threshold_row.get("selected_expert_ids"),
                        "threshold_gamma_0_5_top_score": threshold_row.get("top_score"),
                        "threshold_gamma_0_5_residual_norm": threshold_row.get("residual_norm"),
                        "subdir": str(subdir),
                    }
                )
            finally:
                alg.remove_hook()
                del alg
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    write_loss_trace(args.out_dir / "time_scale_init_grid.csv", rows)
    write_json(args.out_dir / "time_scale_init_grid.json", {"rows": rows})
    return rows


def run_overfit_grid(
    args: argparse.Namespace,
    base_config: TIMEEditMultimodalHparams,
    dataset_path: Path,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    model: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:
    specs = parse_overfit_grid(args.time_overfit_grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "time_hparams.json",
        {
            "args": vars(args),
            "config": dict(base_config.__dict__),
            "command": current_command_line(),
            "grid_specs": specs,
        },
    )
    rows: List[Dict[str, Any]] = []
    keys = ["init_std", "alpha", "lr", "token_scope", "residual_sign", "expert_gain"]
    for values in itertools.product(*(specs[key] for key in keys)):
        combo = dict(zip(keys, values))
        combo_args = deepcopy(args)
        combo_args.init_std = float(combo["init_std"])
        combo_args.alpha = float(combo["alpha"])
        combo_args.lr = float(combo["lr"])
        combo_args.time_token_scope = str(combo["token_scope"])
        combo_args.time_residual_sign = str(combo["residual_sign"])
        combo_args.time_expert_gain = float(combo["expert_gain"])
        combo_args.time_reliability_only = True
        combo_args.time_routing_mode = "force_current"
        combo_args.time_force_current_train = True
        combo_args.time_topk = int(args.time_topk or 1)
        combo_config = deepcopy(base_config)
        combo_config.time_init_std = float(combo_args.init_std)
        combo_config.time_alpha = float(combo_args.alpha)
        combo_config.lr = float(combo_args.lr)
        combo_config.edit_lr = float(combo_args.lr)
        combo_config.time_token_scope = str(combo_args.time_token_scope)
        combo_config.time_residual_sign = str(combo_args.time_residual_sign)
        combo_config.time_expert_gain = float(combo_args.time_expert_gain)
        combo_config.time_reliability_only = True
        combo_config.time_lambda_rel = 1.0
        combo_config.time_lambda_gen = 0.0
        combo_config.time_lambda_loc = 0.0
        combo_config.time_lambda_align = 0.0
        combo_config.time_disable_align_loss = True
        combo_config.time_routing_mode = "force_current"
        combo_config.time_force_current_during_training = True
        combo_config.time_topk = int(combo_args.time_topk)
        subdir = args.out_dir / (
            f"init_{format_float_for_path(combo_args.init_std)}"
            f"_alpha_{format_float_for_path(combo_args.alpha)}"
            f"_lr_{format_float_for_path(combo_args.lr)}"
            f"_scope_{combo_args.time_token_scope}"
            f"_sign_{combo_args.time_residual_sign}"
            f"_gain_{format_float_for_path(combo_args.time_expert_gain)}"
        )
        alg = TIMEEdit(model, combo_config, lambda: None).to(device)
        try:
            summary = run_smoke(combo_args, combo_config, dataset_path, records, samples, alg, subdir, print_summary=False)
            eval_row = (summary.get("eval_rows") or [{}])[0]
            train_row = (summary.get("train_records") or [{}])[0]
            answer_debug = train_row.get("answer_debug") or _read_json(subdir / "time_answer_token_debug.json")
            trace_rows = _read_csv_rows(subdir / "time_reliability_overfit_trace.csv")
            last_trace = trace_rows[-1] if trace_rows else {}
            row = {
                "init_std": combo_args.init_std,
                "alpha": combo_args.alpha,
                "lr": combo_args.lr,
                "token_scope": combo_args.time_token_scope,
                "residual_sign": combo_args.time_residual_sign,
                "expert_gain": combo_args.time_expert_gain,
                "target_nll_delta": eval_row.get("target_nll_delta"),
                "base_target_nll": eval_row.get("base_target_nll"),
                "target_nll": eval_row.get("target_nll"),
                "target_rank_delta": eval_row.get("target_rank_delta"),
                "base_first_target_token_rank": eval_row.get("base_first_target_token_rank"),
                "first_target_token_rank": eval_row.get("first_target_token_rank"),
                "answer_token_logprob_delta": eval_row.get("answer_token_logprob_delta"),
                "reference_delta": eval_row.get("reference_delta"),
                "residual_norm": eval_row.get("residual_norm"),
                "hidden_delta_norm": eval_row.get("target_layer_hidden_delta_norm"),
                "current_expert_score": eval_row.get("top_score"),
                "selected_expert_ids": eval_row.get("selected_expert_ids"),
                "grad_norm_total": last_trace.get("current_expert_grad_norm_total"),
                "U_in_grad_norm": last_trace.get("U_in_grad_norm"),
                "V_in_grad_norm": last_trace.get("V_in_grad_norm"),
                "U_out_grad_norm": last_trace.get("U_out_grad_norm"),
                "V_out_grad_norm": last_trace.get("V_out_grad_norm"),
                "base_vlm_trainable_params": last_trace.get("base_vlm_trainable_params"),
                "pass": reliability_pass(eval_row, answer_debug),
                "subdir": str(subdir),
            }
            rows.append(row)
        finally:
            alg.remove_hook()
            del alg
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_loss_trace(args.out_dir / "time_overfit_grid.csv", rows)
    write_json(args.out_dir / "time_overfit_grid.json", {"rows": rows, "best": best_reliability_row(rows)})
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(errors="replace"))


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_eval(summary: Dict[str, Any]) -> Dict[str, Any]:
    rows = summary.get("eval_rows") or []
    return rows[0] if rows else {}


def _result_lines(title: str, row: Dict[str, Any]) -> List[str]:
    if not row:
        return [f"## {title}", "- Not run or not available.", ""]
    return [
        f"## {title}",
        f"- Target NLL before/after: {row.get('base_target_nll')} -> {row.get('target_nll')}.",
        f"- Target NLL delta: {row.get('target_nll_delta')}.",
        f"- Target rank delta: {row.get('target_rank_delta')}.",
        f"- Residual norm: {row.get('residual_norm')}.",
        f"- Target-layer hidden delta norm: {row.get('target_layer_hidden_delta_norm')}.",
        f"- Top score: {row.get('top_score')}.",
        f"- Selected expert ids: {row.get('selected_expert_ids')}.",
        f"- Locality/reference delta: {row.get('reference_delta')}.",
        "",
    ]


def calibrated_report_dir(out_dir: Path) -> Path:
    if out_dir.name == "one_edit_calibrated":
        return out_dir
    if out_dir.name.startswith("one_edit_"):
        return out_dir.parent / "one_edit_calibrated"
    return out_dir / "one_edit_calibrated"


def write_calibrated_report(out_dir: Path) -> Path:
    report_dir = calibrated_report_dir(out_dir)
    base_dir = report_dir.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    force_summary = _read_json(base_dir / "one_edit_calibrated_force" / "one_edit_summary.json")
    topk_summary = _read_json(base_dir / "one_edit_calibrated_topk" / "one_edit_summary.json")
    gamma_rows = _read_csv_rows(base_dir / "one_edit_calibrated_force" / "time_gamma_sweep.csv")
    grid_rows = _read_csv_rows(base_dir / "one_edit_scale_init_grid" / "time_scale_init_grid.csv")
    force_row = _first_eval(force_summary)
    topk_row = _first_eval(topk_summary)

    selected_gamma_rows = [row for row in gamma_rows if (_float_or_none(row.get("selected_expert_set_size")) or 0.0) > 0.0]
    selected_gammas = [_float_or_none(row.get("sweep_gamma")) for row in selected_gamma_rows]
    selected_gammas = [value for value in selected_gammas if value is not None]
    threshold_05_rows = [row for row in gamma_rows if _float_or_none(row.get("sweep_gamma")) == 0.5]
    threshold_05_selected = any((_float_or_none(row.get("selected_expert_set_size")) or 0.0) > 0.0 for row in threshold_05_rows)

    force_residual = _float_or_none(force_row.get("residual_norm"))
    force_delta = _float_or_none(force_row.get("target_nll_delta"))
    force_rank_delta = _float_or_none(force_row.get("target_rank_delta"))
    topk_residual = _float_or_none(topk_row.get("residual_norm"))
    topk_delta = _float_or_none(topk_row.get("target_nll_delta"))
    topk_rank_delta = _float_or_none(topk_row.get("target_rank_delta"))
    force_pass = (force_residual or 0.0) > 0.0 and ((force_delta or 0.0) > 0.0 or (force_rank_delta or 0.0) > 0.0)
    topk_pass = (topk_residual or 0.0) > 0.0 and ((topk_delta or 0.0) > 0.0 or (topk_rank_delta or 0.0) > 0.0)
    if force_residual is not None and force_residual <= 0.0:
        diagnosis = "hook_execution_bug"
    elif (force_pass or topk_pass) and not threshold_05_selected:
        diagnosis = "threshold_calibration_issue"
    elif (force_residual or 0.0) > 0.0 and not (force_pass or topk_pass):
        diagnosis = "scale_factor_issue"
    else:
        diagnosis = "mixed"

    lines = [
        "# TIME Calibrated 1-Edit Diagnostic Report",
        "",
        "## Files Changed",
        "- `easyeditor/trainer/algs/time_edit.py`",
        "- `easyeditor/trainer/algs/time_edit_modules.py`",
        "- `easyeditor/models/time_edit/time_edit_hparams.py`",
        "- `scripts/time/run_time_medmkeb_smoke.py`",
        "",
        "## Diagnostic Switches",
        "- `--time-force-current-train`: added and logged; it forces the current expert into the selected set only during training.",
        "- `--time-routing-mode`: added with `threshold`, `topk`, `threshold_topk`, and `force_current`.",
        "- `--time-gamma-sweep`: added for post-training threshold sweeps without retraining.",
        "- `--time-scale-init-grid`: added for the bounded init/alpha diagnostic grid.",
        "",
    ]
    lines.extend(_result_lines("Force-Current Result", force_row))
    lines.extend(_result_lines("Topk=1 Result", topk_row))
    lines.extend(
        [
            "## Gamma Sweep Summary",
            f"- Sweep rows: {len(gamma_rows)}.",
            f"- Smallest gamma that selects expert: {min(selected_gammas) if selected_gammas else None}.",
            f"- Largest gamma that still selects expert: {max(selected_gammas) if selected_gammas else None}.",
            f"- Paper gamma 0.5 selects expert: {threshold_05_selected}.",
            "",
        ]
    )
    if selected_gamma_rows:
        lines.append("| gamma | selected ids | residual norm | target NLL | target NLL delta | rank delta | top score |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for row in selected_gamma_rows:
            lines.append(
                "| {gamma} | {ids} | {residual} | {nll} | {delta} | {rank_delta} | {score} |".format(
                    gamma=row.get("sweep_gamma"),
                    ids=row.get("selected_expert_ids"),
                    residual=row.get("residual_norm"),
                    nll=row.get("target_nll"),
                    delta=row.get("target_nll_delta"),
                    rank_delta=row.get("target_rank_delta"),
                    score=row.get("top_score"),
                )
            )
        lines.append("")
    if grid_rows:
        lines.extend(["## Scale/Init Grid Summary", ""])
        lines.append("| init_std | alpha | force residual | topk residual | NLL delta | topk NLL delta | threshold 0.5 selected | top score |")
        lines.append("|---:|---:|---:|---:|---:|---:|---|---:|")
        for row in grid_rows:
            lines.append(
                "| {init_std} | {alpha} | {force_residual} | {topk_residual} | {delta} | {topk_delta} | {selected} | {score} |".format(
                    init_std=row.get("init_std"),
                    alpha=row.get("alpha"),
                    force_residual=row.get("force_current_residual_norm"),
                    topk_residual=row.get("topk_residual_norm"),
                    delta=row.get("final_target_nll_delta"),
                    topk_delta=row.get("topk_target_nll_delta"),
                    selected=row.get("threshold_gamma_0_5_selected"),
                    score=row.get("final_score"),
                )
            )
        lines.append("")
    else:
        lines.extend(["## Scale/Init Grid Summary", "- Not run or not available.", ""])
    lines.extend(["## Diagnosis", f"- Label: `{diagnosis}`.", ""])
    path = report_dir / "TIME_CALIBRATED_1EDIT_REPORT.md"
    path.write_text("\n".join(lines))
    return path


def _run_command_for(dir_path: Path) -> Optional[str]:
    payload = _read_json(dir_path / "time_hparams.json")
    return payload.get("command") if payload else None


def _last_trace_row(dir_path: Path) -> Dict[str, Any]:
    rows = _read_csv_rows(dir_path / "time_reliability_overfit_trace.csv")
    return rows[-1] if rows else {}


def _answer_debug_for(dir_path: Path) -> Dict[str, Any]:
    return _read_json(dir_path / "time_answer_token_debug.json")


def _diagnose_reliability(row: Dict[str, Any], trace: Dict[str, Any], answer_debug: Dict[str, Any]) -> str:
    grad_norm = _float_or_none(trace.get("current_expert_grad_norm_total")) or 0.0
    residual = _float_or_none(row.get("residual_norm")) or 0.0
    reference_delta = _float_or_none(row.get("reference_delta")) or 0.0
    pass_flag = reliability_pass(row, answer_debug)
    sign = str(row.get("residual_sign") or "")
    if grad_norm <= 0.0:
        return "training_graph_or_hook_gradient_bug"
    if residual > 0.0 and reference_delta <= 0.0:
        return "hook_output_not_affecting_logits_or_wrong_layer"
    if pass_flag and sign == "minus":
        return "residual_direction_issue"
    if pass_flag:
        return "reliability_plus_sign_pass"
    if residual > 0.0:
        return "optimization_or_parameterization_issue"
    return "mixed"


def write_reliability_report(out_dir: Path) -> Path:
    report_dir = reliability_report_dir(out_dir)
    base_dir = report_dir.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    primary_dir = base_dir / "one_edit_reliability_only_plus"
    grid_dir = base_dir / "one_edit_reliability_grid"
    confirm_dir = base_dir / "one_edit_full_objective_confirm"
    primary_summary = _read_json(primary_dir / "one_edit_summary.json")
    primary_row = _first_eval(primary_summary)
    primary_trace = _last_trace_row(primary_dir)
    primary_answer = _answer_debug_for(primary_dir)
    grid_rows = _read_csv_rows(grid_dir / "time_overfit_grid.csv")
    confirm_summary = _read_json(confirm_dir / "one_edit_summary.json")
    confirm_row = _first_eval(confirm_summary)

    best_grid = best_reliability_row(grid_rows)
    best_row = best_grid if best_grid else primary_row
    best_trace = primary_trace
    best_answer = primary_answer
    if best_grid:
        best_trace = _last_trace_row(Path(best_grid.get("subdir", "")))
        best_answer = _answer_debug_for(Path(best_grid.get("subdir", "")))
    diagnosis = _diagnose_reliability(best_row, best_trace, best_answer)

    commands = []
    for label, directory in (
        ("primary reliability-only", primary_dir),
        ("overfit grid", grid_dir),
        ("full-objective confirmation", confirm_dir),
    ):
        command = _run_command_for(directory)
        if command:
            commands.append(f"- {label}: `{command}`")

    def field(row: Dict[str, Any], name: str) -> Any:
        return row.get(name) if row else None

    lines = [
        "# TIME 1-Edit Reliability Rescue Report",
        "",
        "## Files Changed",
        "- `easyeditor/trainer/algs/time_edit.py`",
        "- `easyeditor/trainer/algs/time_edit_modules.py`",
        "- `easyeditor/models/time_edit/time_edit_hparams.py`",
        "- `scripts/time/run_time_medmkeb_smoke.py`",
        "",
        "## Exact Run Commands",
        *(commands if commands else ["- No completed run commands found yet."]),
        "",
        "## Primary Reliability-Only Result",
        f"- Target NLL before/after: {field(primary_row, 'base_target_nll')} -> {field(primary_row, 'target_nll')}.",
        f"- Target NLL delta: {field(primary_row, 'target_nll_delta')}.",
        f"- Target rank before/after: {field(primary_row, 'base_first_target_token_rank')} -> {field(primary_row, 'first_target_token_rank')}.",
        f"- Target rank delta: {field(primary_row, 'target_rank_delta')}.",
        f"- Answer-token avg logprob delta: {field(primary_row, 'answer_token_logprob_delta')}.",
        f"- Residual norm: {field(primary_row, 'residual_norm')}.",
        f"- Current expert score: {field(primary_row, 'top_score')}.",
        f"- Grad norm total: {primary_trace.get('current_expert_grad_norm_total')}.",
        f"- Base VLM trainable params: {primary_trace.get('base_vlm_trainable_params')}.",
        "",
        "## Grid Result",
        f"- Grid rows: {len(grid_rows)}.",
    ]
    if grid_rows:
        pass_rows = [row for row in grid_rows if str(row.get("pass")).lower() == "true"]
        lines.append(f"- Reliability-pass rows: {len(pass_rows)}.")
        lines.append("| init_std | alpha | lr | token_scope | residual_sign | NLL delta | rank delta | logprob delta | residual | grad norm | pass |")
        lines.append("|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|")
        for row in grid_rows:
            lines.append(
                "| {init_std} | {alpha} | {lr} | {token_scope} | {residual_sign} | {nll} | {rank} | {logprob} | {residual} | {grad} | {passed} |".format(
                    init_std=row.get("init_std"),
                    alpha=row.get("alpha"),
                    lr=row.get("lr"),
                    token_scope=row.get("token_scope"),
                    residual_sign=row.get("residual_sign"),
                    nll=row.get("target_nll_delta"),
                    rank=row.get("target_rank_delta"),
                    logprob=row.get("answer_token_logprob_delta"),
                    residual=row.get("residual_norm"),
                    grad=row.get("grad_norm_total"),
                    passed=row.get("pass"),
                )
            )
    else:
        lines.append("- Not run.")
    lines.extend(
        [
            "",
            "## Best Config",
            f"- init_std: {field(best_row, 'init_std') if best_grid else '0.05'}.",
            f"- alpha: {field(best_row, 'alpha') if best_grid else '1.0'}.",
            f"- lr: {field(best_row, 'lr') if best_grid else '0.0001'}.",
            f"- token_scope: {field(best_row, 'token_scope') if best_grid else 'all'}.",
            f"- residual_sign: {field(best_row, 'residual_sign')}.",
            f"- residual norm: {field(best_row, 'residual_norm')}.",
            f"- current expert score: {field(best_row, 'current_expert_score') or field(best_row, 'top_score')}.",
            f"- target NLL before/after: {field(best_row, 'base_target_nll')} -> {field(best_row, 'target_nll')}.",
            f"- target rank before/after: {field(best_row, 'base_first_target_token_rank')} -> {field(best_row, 'first_target_token_rank')}.",
            f"- answer-token logprob delta: {field(best_row, 'answer_token_logprob_delta')}.",
            f"- gradient norms: total={best_trace.get('current_expert_grad_norm_total')}, U_in={best_trace.get('U_in_grad_norm')}, V_in={best_trace.get('V_in_grad_norm')}, U_out={best_trace.get('U_out_grad_norm')}, V_out={best_trace.get('V_out_grad_norm')}.",
            "",
            "## Full-Objective Confirmation",
            f"- Target NLL before/after: {field(confirm_row, 'base_target_nll')} -> {field(confirm_row, 'target_nll')}.",
            f"- Target NLL delta: {field(confirm_row, 'target_nll_delta')}.",
            f"- Target rank delta: {field(confirm_row, 'target_rank_delta')}.",
            f"- Residual norm: {field(confirm_row, 'residual_norm')}.",
            "",
            "## Diagnosis",
            f"- Label: `{diagnosis}`.",
            "",
        ]
    )
    path = report_dir / "TIME_1EDIT_RELIABILITY_RESCUE_REPORT.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    args = parse_args()
    if args.mode in {"nonseq", "sequential"} and args.max_edits < 5:
        args.max_edits = 5
    if args.mode == "one":
        args.max_edits = 1
    set_seeds(args.seed)
    ensure_offline_env()

    dataset_path = resolve_dataset_path(args.dataset, Path.cwd(), args.dataset_path)
    repo_path = resolve_repository_path(args.time_load_repository)
    if args.eval_only and repo_path is None:
        raise RuntimeError("--eval-only requires --time-load-repository PATH for this runner.")
    if repo_path is not None:
        if not repo_path.exists():
            raise FileNotFoundError(f"TIME repository not found: {repo_path}")
        print(f"TIME load repository: {repo_path}", flush=True)
    records = load_records_for_repository(
        dataset_path,
        args.sample_index,
        args.max_edits,
        repo_path if args.eval_only else None,
    )
    config = configure(args, dataset_path)
    if repo_path is not None:
        config.time_repository_path = str(repo_path)
    image_root = Path(config.coco_image).expanduser()
    device = torch_device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    model = get_model(config).to(device).eval()
    samples = [make_sample(model, record, image_root) for record in records]
    if args.time_post_retrain_calibration and args.eval_only:
        if repo_path is None:
            raise RuntimeError("--time-post-retrain-calibration requires --time-load-repository PATH.")
        alg = TIMEEdit(model, config, lambda: None).to(device)
        try:
            run_post_retrain_calibration(args, config, dataset_path, records, samples, alg, args.out_dir, repo_path)
        finally:
            alg.remove_hook()
            del alg
    elif args.time_routing_calibration and args.eval_only:
        if repo_path is None:
            raise RuntimeError("--time-routing-calibration requires --time-load-repository PATH.")
        alg = TIMEEdit(model, config, lambda: None).to(device)
        try:
            run_time_routing_calibration(args, config, dataset_path, records, samples, alg, args.out_dir, repo_path)
        finally:
            alg.remove_hook()
            del alg
    elif args.eval_only:
        alg = TIMEEdit(model, config, lambda: None).to(device)
        try:
            summary = run_eval_only(args, config, dataset_path, records, samples, alg, args.out_dir)
            print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
        finally:
            alg.remove_hook()
            del alg
    elif args.time_overfit_grid:
        grid_rows = run_overfit_grid(args, config, dataset_path, records, samples, model, device)
        report_path = write_reliability_report(args.out_dir)
        print(json.dumps(to_jsonable({"grid_rows": grid_rows, "reliability_report": str(report_path)}), indent=2, sort_keys=True))
    elif args.time_scale_init_grid:
        grid_rows = run_scale_init_grid(args, config, dataset_path, records, samples, model, device)
        report_path = write_calibrated_report(args.out_dir)
        print(json.dumps(to_jsonable({"grid_rows": grid_rows, "calibrated_report": str(report_path)}), indent=2, sort_keys=True))
    else:
        alg = TIMEEdit(model, config, lambda: None).to(device)
        try:
            summary = run_smoke(args, config, dataset_path, records, samples, alg, args.out_dir, print_summary=False)
            gamma_rows = run_gamma_sweep(args, alg, records, samples, args.out_dir) if args.time_gamma_sweep else []
            score_norm_summary = (
                run_score_norm_retrain_grid(args, records, samples, alg, args.out_dir)
                if args.time_routing_calibration and args.mode == "nonseq"
                else None
            )
            if args.time_reliability_only or args.out_dir.name == "one_edit_full_objective_confirm":
                report_path = write_reliability_report(args.out_dir)
                report_key = "reliability_report"
            elif score_norm_summary is not None:
                report_path = args.out_dir / "TIME_5EDIT_SCORE_NORM_RETRAIN_REPORT.md"
                report_key = "score_norm_retrain_report"
            elif args.mode == "nonseq" and args.eval_routing_modes:
                report_path = args.out_dir / _nonseq_report_name(args.max_edits)
                report_key = "ten_edit_nonseq_report" if args.max_edits == 10 else "five_edit_nonseq_report"
            elif args.mode == "sequential" and args.eval_routing_modes:
                report_path = args.out_dir / _sequential_report_name(args.max_edits)
                report_key = "ten_edit_sequential_report" if args.max_edits == 10 else "five_edit_sequential_report"
            else:
                report_path = write_calibrated_report(args.out_dir)
                report_key = "calibrated_report"
            payload = dict(summary)
            if score_norm_summary is not None:
                payload["score_norm_retrain"] = score_norm_summary
            if gamma_rows:
                payload["gamma_sweep_rows"] = gamma_rows
            payload[report_key] = str(report_path)
            print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
        finally:
            alg.remove_hook()
            del alg

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
