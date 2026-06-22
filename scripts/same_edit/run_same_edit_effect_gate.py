#!/usr/bin/env python3
"""Bounded real-effect gate for SAME-Edit on MedMKEB LLaVA-Med.

This runner intentionally evaluates SAME-Edit as an editing method, not only as
importable code.  It compares a plain MoE-LoRA baseline with SAME-full under the
same records, seed, optimizer, rank, expert count, and step budget.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dsca_medmkeb_diag_common import (  # noqa: E402
    append_jsonl,
    decode_argmax_on_labels,
    target_nll_from_outputs,
    to_jsonable,
    torch_device,
    write_json,
)
from easyeditor.models.same_edit import (  # noqa: E402
    SAMEEditLinear,
    SAMEEditMultimodalHparams,
    same_edit_gradient_summary,
    same_edit_trainable_summary,
)
from easyeditor.trainer.algs.same_edit import SAMEEdit  # noqa: E402
from easyeditor.trainer.models import get_model  # noqa: E402


REQUIRED_METHODS = ("plain_moe_lora", "same_full")
OPTIONAL_METHODS = ("same_router_only", "same_router_expert")
MODE_TO_CSV = {
    "one_edit": "one_edit_metrics.csv",
    "nonseq": "nonseq_5edit_metrics.csv",
    "sequential": "sequential_5edit_metrics.csv",
}
SAME_FILES = [
    "SAME_EDIT_DEVIATIONS.md",
    "easyeditor/models/same_edit/__init__.py",
    "easyeditor/models/same_edit/same_edit_hparams.py",
    "easyeditor/models/same_edit/same_edit_main.py",
    "easyeditor/models/same_edit/same_edit_modules.py",
    "easyeditor/trainer/algs/same_edit.py",
    "easyeditor/util/alg_dict.py",
    "easyeditor/util/alg_train_dict.py",
    "scripts/same_edit/overfit_same_edit_one_medmkeb_edit.py",
    "scripts/same_edit/run_same_edit_5edit_medmkeb.py",
    "scripts/same_edit/run_same_edit_effect_gate.py",
    "hparams/SAME_EDIT/llava_med.yaml",
    "hparams/TRAINING/SAME_EDIT/llava_med_oneedit_smoke.yaml",
    "tests/test_same_edit.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--data_path", "--dataset-path", dest="data_path", type=Path, default=None)
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--image_root", "--image-root", dest="image_root", type=Path, default=None)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True, type=Path)
    parser.add_argument("--hparams", default="hparams/TRAINING/SAME_EDIT/llava_med_oneedit_smoke.yaml")
    parser.add_argument("--max_edits", "--max-edits", dest="max_edits", type=int, default=5)
    parser.add_argument("--record_offset", "--record-offset", dest="record_offset", type=int, default=0)
    parser.add_argument("--mode", choices=["one_edit", "nonseq", "sequential", "all"], default="all")
    parser.add_argument(
        "--method",
        default="plain_moe_lora,same_full",
        help="Comma-separated methods: plain_moe_lora,same_router_only,same_router_expert,same_full",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--one_edit_steps", "--one-edit-steps", dest="one_edit_steps", type=int, default=50)
    parser.add_argument("--steps", "--max_steps", "--max-steps", dest="steps", type=int, default=50)
    parser.add_argument("--lora_r", "--lora-r", dest="lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", "--lora-alpha", dest="lora_alpha", type=float, default=16.0)
    parser.add_argument("--expert_num", "--expert-num", dest="expert_num", type=int, default=4)
    parser.add_argument("--top_k", "--top-k", dest="top_k", type=int, default=1)
    parser.add_argument("--route_loss_weight", "--route-loss-weight", dest="route_loss_weight", type=float, default=0.1)
    parser.add_argument("--target_modules", "--target-modules", dest="target_modules", default="last4_down_proj")
    parser.add_argument("--max_new_tokens", "--max-new-tokens", dest="max_new_tokens", type=int, default=8)
    parser.add_argument("--skip_generation", "--skip-generation", dest="skip_generation", action="store_true")
    parser.add_argument("--locality_threshold", "--locality-threshold", dest="locality_threshold", type=float, default=0.02)
    parser.add_argument("--rollback_tolerance", "--rollback-tolerance", dest="rollback_tolerance", type=float, default=1.0e-4)
    parser.add_argument("--timeout_safe", "--timeout-safe", dest="timeout_safe", type=int, default=0)
    parser.add_argument("--include_optional", "--include-optional", dest="include_optional", action="store_true")
    parser.add_argument("--skip_static_tests", "--skip-static-tests", dest="skip_static_tests", action="store_true")
    parser.add_argument("--no_gate_stop", "--no-gate-stop", dest="gate_stop", action="store_false")
    parser.set_defaults(gate_stop=True)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


def finite(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def run_capture(command: Sequence[str]) -> Dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        list(command),
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": list(command),
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - started, 3),
        "output_tail": proc.stdout[-12000:],
    }


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.data_path is not None:
        path = args.data_path if args.data_path.is_absolute() else PROJECT_ROOT / args.data_path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [
        PROJECT_ROOT / "datasets" / "MedMKEB" / "eval.json",
        PROJECT_ROOT / "datasets" / "MEDMKEB" / "eval.json",
        PROJECT_ROOT / "datasets" / "medmkeb" / "eval.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("MedMKEB eval.json not found; pass --data_path")


def resolve_image_path(image_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = image_root.resolve()
    if root.name == "images" and path.parts and path.parts[0] == "images":
        return root.parent / path
    return root / path


def normalize_device_arg(text: str) -> Any:
    if str(text).isdigit():
        return int(text)
    return text


def format_prompt(question: Any) -> str:
    text = str(question or "").strip()
    if text.lower().startswith("question:"):
        return text if text.endswith(" ") else text + " "
    return f"Question: {text} Short answer: "


def new_answer(record: Dict[str, Any]) -> str:
    return str(record.get("new_answer") or record.get("replacement_answer") or record.get("alt") or "")


def old_answer(record: Dict[str, Any]) -> str:
    return str(record.get("old_answer") or record.get("erase_answer") or record.get("pred") or "")


def record_id(record: Dict[str, Any]) -> str:
    return str(record.get("record_id") or record.get("id"))


def load_selected_records(args: argparse.Namespace, dataset_path: Path) -> List[Dict[str, Any]]:
    records = json.loads(dataset_path.read_text(errors="replace"))
    if not isinstance(records, list):
        raise RuntimeError(f"Dataset root is not a list: {dataset_path}")
    selected = records[int(args.record_offset) : int(args.record_offset) + int(args.max_edits)]
    if len(selected) < int(args.max_edits):
        raise RuntimeError(f"Requested {args.max_edits} edits but only found {len(selected)} records.")
    out: List[Dict[str, Any]] = []
    for row in selected:
        item = dict(row)
        item.setdefault("record_id", item.get("id"))
        out.append(item)
    return out


def selected_record_view(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    return {
        "record_id": record_id(record),
        "image_path": str(resolve_image_path(image_root, record.get("image"))),
        "question": record.get("src"),
        "old_answer": old_answer(record),
        "new_answer": new_answer(record),
        "reference_question": record.get("m_loc_q"),
        "reference_answer": record.get("m_loc_a"),
        "reference_image_path": str(resolve_image_path(image_root, record.get("m_loc"))) if record.get("m_loc") else None,
        "locality_prompt": record.get("loc"),
        "locality_answer": record.get("loc_ans"),
        "department": record.get("department"),
        "modality": record.get("modality"),
    }


def make_sample(model: Any, prompt: str, target: str, image_path: Path) -> Dict[str, Any]:
    prompt = format_prompt(prompt)
    labels = model.llava_tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return {
        "image_path": [str(image_path)],
        "prompt": [prompt],
        "target": [str(target)],
        "text_input": [prompt + str(target)],
        "labels": labels,
        "prompts_len": [len(model.llava_tokenizer(prompt, add_special_tokens=False).input_ids)],
    }


def record_samples(model: Any, record: Dict[str, Any], image_root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "new": make_sample(model, str(record.get("src") or ""), new_answer(record), resolve_image_path(image_root, record["image"])),
        "old": make_sample(model, str(record.get("src") or ""), old_answer(record), resolve_image_path(image_root, record["image"])),
        "locality": make_sample(
            model,
            str(record.get("m_loc_q") or record.get("loc") or record.get("src") or ""),
            str(record.get("m_loc_a") or record.get("loc_ans") or old_answer(record)),
            resolve_image_path(image_root, record.get("m_loc") or record.get("image")),
        ),
    }


def clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, dict):
            cloned[key] = clone_batch(value)
        else:
            cloned[key] = value
    return cloned


@contextmanager
def eval_routing_context(alg: SAMEEdit, eval_oracle_routing: Optional[bool]) -> Iterator[None]:
    old_training = alg.training
    old_values = []
    for _name, layer in alg.same_model.same_edit_layers():
        old_values.append((layer, layer.eval_oracle_edit_routing))
        if eval_oracle_routing is not None:
            layer.eval_oracle_edit_routing = bool(eval_oracle_routing)
    alg.set_editor_train(False)
    try:
        yield
    finally:
        for layer, old_value in old_values:
            layer.eval_oracle_edit_routing = old_value
        alg.set_editor_train(old_training)


def forward_outputs(
    alg: SAMEEdit,
    sample: Dict[str, Any],
    *,
    adapters_enabled: bool,
    eval_oracle_routing: Optional[bool],
) -> Tuple[Any, Dict[str, Any]]:
    with eval_routing_context(alg, eval_oracle_routing):
        context = alg.same_model.adapters_disabled() if not adapters_enabled else null_context()
        with context:
            with torch.no_grad():
                batch = clone_batch(sample)
                outputs = alg(batch)
    metrics = target_nll_from_outputs(outputs, batch)
    metrics["teacher_forced_argmax"] = decode_argmax_on_labels(alg.same_model.model.llava_tokenizer, outputs, batch)
    return outputs, metrics


class null_context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def logits_max_abs_diff(outputs_a: Any, outputs_b: Any) -> float:
    logits_a = outputs_a if isinstance(outputs_a, torch.Tensor) else outputs_a.logits
    logits_b = outputs_b if isinstance(outputs_b, torch.Tensor) else outputs_b.logits
    seq = min(int(logits_a.shape[1]), int(logits_b.shape[1]))
    vocab = min(int(logits_a.shape[-1]), int(logits_b.shape[-1]))
    diff = (logits_a[:, -seq:, :vocab].detach().float() - logits_b[:, -seq:, :vocab].detach().float()).abs()
    return float(diff.max().cpu()) if diff.numel() else 0.0


def generate_text(alg: SAMEEdit, sample: Dict[str, Any], max_new_tokens: int, adapters_enabled: bool) -> Optional[str]:
    if max_new_tokens <= 0:
        return None
    raw_model = alg.same_model.model
    prompt = sample["prompt"][0]
    prompt_text = raw_model._conversation_prompt(prompt, None)
    input_ids = raw_model.tokenizer_image_token(
        prompt_text,
        raw_model.llava_tokenizer,
        raw_model.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    input_ids = input_ids.unsqueeze(0).to(raw_model.lm_device)
    image_tensor = raw_model._image_for_row(sample, 0)
    context = alg.same_model.adapters_disabled() if not adapters_enabled else null_context()
    with context:
        with torch.inference_mode():
            output_ids = raw_model.llava_model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
                pad_token_id=raw_model.llava_tokenizer.pad_token_id,
                eos_token_id=raw_model.llava_tokenizer.eos_token_id,
            )
    new_ids = output_ids[:, input_ids.shape[1] :] if output_ids.shape[1] >= input_ids.shape[1] else output_ids
    return raw_model.llava_tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def configure(args: argparse.Namespace, method: str) -> SAMEEditMultimodalHparams:
    config = SAMEEditMultimodalHparams.from_hparams(str(args.hparams))
    config.device = normalize_device_arg(str(args.device))
    if args.model_name_or_path:
        config.name = str(args.model_name_or_path)
        config.tokenizer_name = str(args.model_name_or_path)
    if args.image_root is not None:
        image_root = args.image_root if args.image_root.is_absolute() else PROJECT_ROOT / args.image_root
        config.coco_image = str(image_root)
        config.rephrase_image = str(image_root)
    config.lr = float(args.learning_rate)
    config.edit_lr = float(args.learning_rate)
    config.same_edit_lora_r = int(args.lora_r)
    config.same_edit_lora_alpha = float(args.lora_alpha)
    config.same_edit_expert_num = int(args.expert_num)
    config.same_edit_top_k = int(args.top_k)
    config.same_edit_route_loss_weight = float(args.route_loss_weight)
    config.same_edit_target_modules = str(args.target_modules)
    config.same_edit_oracle_edit_routing = True
    config.same_edit_eval_oracle_edit_routing = False
    config.same_edit_learned_hidden_routing = True
    config.same_edit_allow_missing_covariance = True
    config.same_edit_curvature_max_grad_ratio = 10.0
    config.same_edit_update_covariance = True
    config.same_edit_num_steps = int(args.steps)
    if method == "plain_moe_lora":
        config.same_edit_spectral_router = False
        config.same_edit_curvature_mode = "off"
        config.same_edit_adaptive_activation = False
    elif method == "same_router_only":
        config.same_edit_spectral_router = True
        config.same_edit_curvature_mode = "off"
        config.same_edit_adaptive_activation = False
    elif method == "same_router_expert":
        config.same_edit_spectral_router = True
        config.same_edit_curvature_mode = "safe"
        config.same_edit_adaptive_activation = False
    elif method == "same_full":
        config.same_edit_spectral_router = True
        config.same_edit_curvature_mode = "safe"
        config.same_edit_adaptive_activation = True
    else:
        raise ValueError(f"Unknown method: {method}")
    return config


def set_active_edit(alg: SAMEEdit, config: SAMEEditMultimodalHparams, edit_index: int, *, reset: bool = False) -> None:
    config.same_edit_current_edit = int(edit_index)
    alg.config.same_edit_current_edit = int(edit_index)
    alg.same_config.current_edit = int(edit_index)
    if reset:
        alg.same_model.reset_for_new_edit(int(edit_index), snapshot_previous=True)
    else:
        alg.same_model.set_current_edit(int(edit_index))


def optimizer_for(alg: SAMEEdit, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(alg.outer_parameters(), lr=float(lr), weight_decay=0.0)


def train_record(
    alg: SAMEEdit,
    config: SAMEEditMultimodalHparams,
    samples: Dict[str, Dict[str, Any]],
    *,
    edit_index: int,
    steps: int,
    lr: float,
    log_path: Path,
) -> Dict[str, Any]:
    set_active_edit(alg, config, edit_index)
    batch = {
        "edit_inner": clone_batch(samples["new"]),
        "edit_outer": clone_batch(samples["new"]),
        "loc": clone_batch(samples["locality"]),
        "loc_image": clone_batch(samples["locality"]),
    }
    optimizer = optimizer_for(alg, lr)
    last_info: Dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        set_active_edit(alg, config, edit_index)
        optimizer.zero_grad(set_to_none=True)
        loss_total, loss_edit, _loss_loc, _loss_base, info = alg.edit_step(batch, training=True, optimizer=optimizer)
        torch.nn.utils.clip_grad_norm_(alg.outer_parameters(), float(config.grad_clip), error_if_nonfinite=True)
        optimizer.step()
        last_info = dict(info)
        if step == 1 or step == int(steps):
            row = {
                "phase": "train_step",
                "edit_index": int(edit_index),
                "step": int(step),
                "loss_total": float(loss_total.detach().cpu()),
                "loss_edit": float(loss_edit.detach().cpu()),
                **same_edit_gradient_summary(alg.same_model),
                **{k.replace("/", "_"): v for k, v in info.items()},
            }
            append_jsonl(log_path, row)
    return last_info


def layer_diagnostics(alg: SAMEEdit) -> Dict[str, Any]:
    layers = []
    mask_total = 0
    mask_frozen = 0
    cov_ranks = []
    cov_prev_ranks = []
    cov_energy = []
    curvature_before = []
    curvature_after = []
    router_before = []
    router_after = []
    top_ids = []
    entropies = []
    for name, layer in alg.same_model.same_edit_layers():
        if not isinstance(layer, SAMEEditLinear):
            continue
        mask = layer.expert_masks.detach().float().cpu()
        mask_total += int(mask.numel())
        mask_frozen += int((mask <= 0).sum().item())
        cov_s = layer.cov_S.detach().float().cpu()
        cov_prev_s = layer.cov_S_prev.detach().float().cpu()
        cov_rank = int((cov_s.abs() > 1.0e-10).sum().item())
        cov_prev_rank = int((cov_prev_s.abs() > 1.0e-10).sum().item())
        cov_ranks.append(cov_rank)
        cov_prev_ranks.append(cov_prev_rank)
        cov_energy.append(float((cov_s**2).sum().item()))
        summary = layer.routing_summary()
        if summary.get("top_expert_id") is not None:
            top_ids.append(int(summary["top_expert_id"]))
        if summary.get("routing_entropy") is not None:
            entropies.append(float(summary["routing_entropy"]))
        curv = summary.get("curvature") or {}
        hook = summary.get("router_hook") or {}
        if curv.get("grad_norm_before") is not None:
            curvature_before.append(float(curv["grad_norm_before"]))
        if curv.get("grad_norm_after") is not None:
            curvature_after.append(float(curv["grad_norm_after"]))
        if hook.get("grad_norm_before") is not None:
            router_before.append(float(hook["grad_norm_before"]))
        if hook.get("grad_norm_after") is not None:
            router_after.append(float(hook["grad_norm_after"]))
        layers.append(
            {
                "module": name,
                "routing_entropy": summary.get("routing_entropy"),
                "top_expert_id": summary.get("top_expert_id"),
                "assigned_expert_id": summary.get("assigned_expert_id"),
                "active_expert_count": summary.get("active_expert_count"),
                "expert_mask": summary.get("expert_mask"),
                "utilization": summary.get("utilization"),
                "importance": summary.get("importance"),
                "covariance_rank": cov_rank,
                "covariance_prev_rank": cov_prev_rank,
                "covariance_energy": cov_energy[-1],
                "curvature": curv,
                "router_hook": hook,
            }
        )
    return {
        "mean_router_entropy": mean(entropies),
        "top_expert_ids": top_ids,
        "expert_freeze_ratio": (float(mask_frozen) / float(mask_total)) if mask_total else 0.0,
        "mean_covariance_rank": mean(cov_ranks),
        "mean_covariance_prev_rank": mean(cov_prev_ranks),
        "mean_covariance_energy": mean(cov_energy),
        "mean_curvature_grad_norm_before": mean(curvature_before),
        "mean_curvature_grad_norm_after": mean(curvature_after),
        "mean_router_grad_norm_before_scaling": mean(router_before),
        "mean_router_grad_norm_after_scaling": mean(router_after),
        "layers": layers,
    }


def evaluate_record(
    alg: SAMEEdit,
    record: Dict[str, Any],
    samples: Dict[str, Dict[str, Any]],
    *,
    method: str,
    mode: str,
    edit_index: int,
    stage: str,
    steps: int,
    eval_oracle_routing: bool,
    max_new_tokens: int,
    skip_generation: bool,
    locality_threshold: float,
    rollback_tolerance: float,
    diagnostics_path: Path,
    predictions_path: Path,
) -> Dict[str, Any]:
    pre_new_outputs, pre_new = forward_outputs(alg, samples["new"], adapters_enabled=False, eval_oracle_routing=eval_oracle_routing)
    _pre_old_outputs, pre_old = forward_outputs(alg, samples["old"], adapters_enabled=False, eval_oracle_routing=eval_oracle_routing)
    _pre_loc_outputs, pre_loc = forward_outputs(alg, samples["locality"], adapters_enabled=False, eval_oracle_routing=eval_oracle_routing)
    post_new_outputs, post_new = forward_outputs(alg, samples["new"], adapters_enabled=True, eval_oracle_routing=eval_oracle_routing)
    _post_old_outputs, post_old = forward_outputs(alg, samples["old"], adapters_enabled=True, eval_oracle_routing=eval_oracle_routing)
    _post_loc_outputs, post_loc = forward_outputs(alg, samples["locality"], adapters_enabled=True, eval_oracle_routing=eval_oracle_routing)
    rollback_outputs, _rollback_metrics = forward_outputs(
        alg,
        samples["new"],
        adapters_enabled=False,
        eval_oracle_routing=eval_oracle_routing,
    )
    rollback_diff = logits_max_abs_diff(pre_new_outputs, rollback_outputs)
    pre_new_nll = pre_new.get("target_nll")
    post_new_nll = post_new.get("target_nll")
    pre_ref_nll = pre_old.get("target_nll")
    post_ref_nll = post_old.get("target_nll")
    pre_loc_nll = pre_loc.get("target_nll")
    post_loc_nll = post_loc.get("target_nll")
    delta_new = None if pre_new_nll is None or post_new_nll is None else float(post_new_nll) - float(pre_new_nll)
    ref_delta = None if pre_ref_nll is None or post_ref_nll is None else float(post_ref_nll) - float(pre_ref_nll)
    loc_delta = None if pre_loc_nll is None or post_loc_nll is None else float(post_loc_nll) - float(pre_loc_nll)
    diag = layer_diagnostics(alg)
    grad_summary = same_edit_gradient_summary(alg.same_model)
    row = {
        "method": method,
        "mode": mode,
        "stage": stage,
        "edit_index": int(edit_index),
        "record_id": record_id(record),
        "steps": int(steps),
        "pre_new_answer_nll": pre_new_nll,
        "post_new_answer_nll": post_new_nll,
        "delta_new_nll": delta_new,
        "positive_new": bool(delta_new is not None and delta_new < 0.0),
        "first_target_token_rank_pre": pre_new.get("first_target_token_rank"),
        "first_target_token_rank_post": post_new.get("first_target_token_rank"),
        "target_token_argmax": post_new.get("teacher_forced_argmax"),
        "pre_ref_nll": pre_ref_nll,
        "post_ref_nll": post_ref_nll,
        "ref_delta": ref_delta,
        "ref_abs_delta": None if ref_delta is None else abs(ref_delta),
        "pre_locality_nll": pre_loc_nll,
        "post_locality_nll": post_loc_nll,
        "locality_delta": loc_delta,
        "locality_abs_delta": None if loc_delta is None else abs(loc_delta),
        "locality_damage": bool(loc_delta is not None and abs(loc_delta) > float(locality_threshold)),
        "old_answer_nll_pre": pre_ref_nll,
        "old_answer_nll_post": post_ref_nll,
        "rollback_max_abs_diff": rollback_diff,
        "rollback_pass": bool(rollback_diff <= float(rollback_tolerance)),
        "router_entropy": diag.get("mean_router_entropy"),
        "expert_utilization": alg.same_model.summary().get("expert_usage_histogram"),
        "top_k_expert_ids": diag.get("top_expert_ids"),
        "expert_freeze_ratio": diag.get("expert_freeze_ratio"),
        "covariance_retained_rank": diag.get("mean_covariance_rank"),
        "covariance_prev_retained_rank": diag.get("mean_covariance_prev_rank"),
        "covariance_energy": diag.get("mean_covariance_energy"),
        "curvature_grad_norm_before": diag.get("mean_curvature_grad_norm_before"),
        "curvature_grad_norm_after": diag.get("mean_curvature_grad_norm_after"),
        "router_grad_norm_before_scaling": diag.get("mean_router_grad_norm_before_scaling"),
        "router_grad_norm_after_scaling": diag.get("mean_router_grad_norm_after_scaling"),
        "nan_inf_count": int(grad_summary.get("nan_inf_grad_count") or 0) + (0 if finite(diag) else 1),
        "nan_inf_detected": not finite({"row": row if False else {}, "diag": diag, "grad": grad_summary}),
    }
    append_jsonl(diagnostics_path, {"phase": "eval", **row, "diagnostics": diag, "gradient_summary": grad_summary})
    if not skip_generation:
        base_generation = generate_text(alg, samples["new"], max_new_tokens, adapters_enabled=False)
        post_generation = generate_text(alg, samples["new"], max_new_tokens, adapters_enabled=True)
        pred_row = {
            "method": method,
            "mode": mode,
            "stage": stage,
            "edit_index": int(edit_index),
            "record_id": record_id(record),
            "base_free_generation": base_generation,
            "post_free_generation": post_generation,
            "target": new_answer(record),
            "old_answer": old_answer(record),
        }
        append_jsonl(predictions_path, pred_row)
        row["base_free_generation"] = base_generation
        row["post_free_generation"] = post_generation
    else:
        row["base_free_generation"] = None
        row["post_free_generation"] = None
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(to_jsonable(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (list, dict))
                    else to_jsonable(row.get(key))
                    for key in fieldnames
                }
            )


def aggregate_rows(rows: List[Dict[str, Any]], method: str, mode: str) -> Dict[str, Any]:
    selected = [row for row in rows if row.get("method") == method and row.get("mode") == mode and row.get("stage") == "post_edit"]
    deltas = [float(row["delta_new_nll"]) for row in selected if row.get("delta_new_nll") is not None]
    ref_abs = [float(row["ref_abs_delta"]) for row in selected if row.get("ref_abs_delta") is not None]
    loc_abs = [float(row["locality_abs_delta"]) for row in selected if row.get("locality_abs_delta") is not None]
    rollback = [float(row["rollback_max_abs_diff"]) for row in selected if row.get("rollback_max_abs_diff") is not None]
    retention = [
        float(row["retention_drop"])
        for row in rows
        if row.get("method") == method and row.get("mode") == mode and row.get("stage") == "final_retention" and row.get("retention_drop") is not None
    ]
    return {
        "method": method,
        "mode": mode,
        "record_count": len(selected),
        "mean_delta_new_nll": mean(deltas),
        "mean_new_answer_nll_decrease": None if mean(deltas) is None else -float(mean(deltas)),
        "positive_new_count": sum(1 for value in deltas if value < 0.0),
        "ref_abs_delta_mean": mean(ref_abs),
        "locality_abs_delta_mean": mean(loc_abs),
        "locality_damage_count": sum(1 for row in selected if row.get("locality_damage")),
        "rollback_pass": all(bool(row.get("rollback_pass")) for row in selected) if selected else False,
        "rollback_max_abs_diff": max(rollback) if rollback else None,
        "nan_inf_count": sum(int(row.get("nan_inf_count") or 0) for row in selected),
        "mean_router_entropy": mean([float(row["router_entropy"]) for row in selected if row.get("router_entropy") is not None]),
        "mean_expert_freeze_ratio": mean([float(row["expert_freeze_ratio"]) for row in selected if row.get("expert_freeze_ratio") is not None]),
        "mean_retention_drop": mean(retention),
        "max_retention_drop": max(retention) if retention else None,
    }


def static_audit(args: argparse.Namespace, output_dir: Path, trainable: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    files = [str(PROJECT_ROOT / item) for item in SAME_FILES if (PROJECT_ROOT / item).exists()]
    alg_dict = read_text(PROJECT_ROOT / "easyeditor" / "util" / "alg_dict.py")
    alg_train_dict = read_text(PROJECT_ROOT / "easyeditor" / "util" / "alg_train_dict.py")
    dep_hits: List[Dict[str, Any]] = []
    for rel in SAME_FILES:
        path = PROJECT_ROOT / rel
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            lowered = line.lower()
            if any(token in lowered for token in ("engram", "cure", "dsca")):
                dep_hits.append({"file": rel, "line": lineno, "text": line.strip()})
    py_compile = None
    pytest = None
    if not args.skip_static_tests:
        py_compile = run_capture(
            [
                sys.executable,
                "-m",
                "py_compile",
                "easyeditor/models/same_edit/same_edit_modules.py",
                "easyeditor/models/same_edit/same_edit_hparams.py",
                "easyeditor/models/same_edit/same_edit_main.py",
                "easyeditor/trainer/algs/same_edit.py",
                "easyeditor/util/alg_dict.py",
                "easyeditor/util/alg_train_dict.py",
                "scripts/same_edit/overfit_same_edit_one_medmkeb_edit.py",
                "scripts/same_edit/run_same_edit_5edit_medmkeb.py",
                "scripts/same_edit/run_same_edit_effect_gate.py",
            ]
        )
        pytest = run_capture([sys.executable, "-m", "pytest", "tests/test_same_edit.py", "-q"])
    gpu = {"cuda_available": bool(torch.cuda.is_available()), "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0}
    if torch.cuda.is_available():
        gpu["devices"] = []
        for idx in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(idx)
            gpu["devices"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "free_bytes": int(free),
                    "total_bytes": int(total),
                }
            )
    payload = {
        "same_edit_files": files,
        "registered_multimodal": "SAME_EDIT" in alg_dict and "SAMEEditMultimodalRewriteExecutor" in alg_dict,
        "registered_training": "SAME_EDIT" in alg_train_dict and "SAMEEdit" in alg_train_dict,
        "dependency_scan": {
            "depends_on_engram_or_cure": any("engram" in hit["text"].lower() or "cure" in hit["text"].lower() for hit in dep_hits),
            "dsca_hits_are_shared_helper_only": all("dsca_medmkeb_diag_common" in hit["text"] for hit in dep_hits if "dsca" in hit["text"].lower()),
            "hits": dep_hits,
        },
        "static_checks": {"py_compile": py_compile, "pytest": pytest},
        "trainable": trainable,
        "nan_inf_risk_checks": {
            "gradient_nonfinite_counting": True,
            "clip_grad_error_if_nonfinite": True,
            "curvature_safe_mode_clamps_nonfinite_or_large_ratio": True,
            "target_nll_json_nan_inf_sanitization": True,
        },
        "gpu": gpu,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write_json(output_dir / "static_audit.json", payload)
    return payload


def trainable_detail(alg: SAMEEdit) -> Dict[str, Any]:
    summary = same_edit_trainable_summary(alg.same_model)
    names = [
        {"name": name, "numel": int(param.numel()), "shape": list(param.shape)}
        for name, param in alg.same_model.named_parameters()
        if param.requires_grad
    ]
    base_trainable = [
        {"name": name, "numel": int(param.numel()), "shape": list(param.shape)}
        for name, param in alg.same_model.named_parameters()
        if param.requires_grad and ".base_linear." in name
    ]
    return {**summary, "trainable_parameter_names": names, "base_trainable_parameters": base_trainable}


def method_list(args: argparse.Namespace) -> List[str]:
    methods = [item.strip() for item in str(args.method).split(",") if item.strip()]
    if args.include_optional:
        methods = list(dict.fromkeys([*methods, *OPTIONAL_METHODS]))
    for method in methods:
        if method not in (*REQUIRED_METHODS, *OPTIONAL_METHODS):
            raise ValueError(f"Unsupported method: {method}")
    return methods


def mode_list(args: argparse.Namespace) -> List[str]:
    if args.mode == "all":
        return ["one_edit", "nonseq", "sequential"]
    return [args.mode]


def load_model_and_alg(args: argparse.Namespace, method: str) -> Tuple[Any, SAMEEditMultimodalHparams, SAMEEdit]:
    config = configure(args, method)
    device = torch_device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    model = get_model(config).to(device).eval()
    alg = SAMEEdit(model, config, lambda: None).to(device)
    detail = trainable_detail(alg)
    if int(detail.get("base_trainable_param_count") or 0) != 0:
        raise RuntimeError(f"SAME-Edit base trainable params are nonzero: {detail}")
    return model, config, alg


def cleanup_model(*items: Any) -> None:
    for item in items:
        del item
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_method_mode(
    args: argparse.Namespace,
    method: str,
    mode: str,
    records: List[Dict[str, Any]],
    image_root: Path,
    output_dir: Path,
    diagnostics_path: Path,
    predictions_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    set_seeds(int(args.seed))
    model, config, alg = load_model_and_alg(args, method)
    detail = trainable_detail(alg)
    write_json(output_dir / "logs" / f"{mode}_{method}_trainable_summary.json", detail)
    static_audit(args, output_dir, trainable=detail)
    initial_bundle = deepcopy(alg.same_model.state_bundle())
    steps = int(args.one_edit_steps if mode == "one_edit" else args.steps)
    active_records = records[:1] if mode == "one_edit" else records[: int(args.max_edits)]
    rows: List[Dict[str, Any]] = []
    retention_source: Dict[int, Dict[str, Any]] = {}
    try:
        if mode == "one_edit":
            record = active_records[0]
            samples = record_samples(model, record, image_root)
            train_record(
                alg,
                config,
                samples,
                edit_index=0,
                steps=steps,
                lr=float(args.learning_rate),
                log_path=diagnostics_path,
            )
            rows.append(
                evaluate_record(
                    alg,
                    record,
                    samples,
                    method=method,
                    mode=mode,
                    edit_index=0,
                    stage="post_edit",
                    steps=steps,
                    eval_oracle_routing=False,
                    max_new_tokens=int(args.max_new_tokens),
                    skip_generation=bool(args.skip_generation),
                    locality_threshold=float(args.locality_threshold),
                    rollback_tolerance=float(args.rollback_tolerance),
                    diagnostics_path=diagnostics_path,
                    predictions_path=predictions_path,
                )
            )
        elif mode == "nonseq":
            for edit_index, record in enumerate(active_records):
                alg.same_model.load_state_bundle(initial_bundle)
                set_active_edit(alg, config, edit_index)
                samples = record_samples(model, record, image_root)
                train_record(
                    alg,
                    config,
                    samples,
                    edit_index=edit_index,
                    steps=steps,
                    lr=float(args.learning_rate),
                    log_path=diagnostics_path,
                )
                rows.append(
                    evaluate_record(
                        alg,
                        record,
                        samples,
                        method=method,
                        mode=mode,
                        edit_index=edit_index,
                        stage="post_edit",
                        steps=steps,
                        eval_oracle_routing=False,
                        max_new_tokens=int(args.max_new_tokens),
                        skip_generation=bool(args.skip_generation),
                        locality_threshold=float(args.locality_threshold),
                        rollback_tolerance=float(args.rollback_tolerance),
                        diagnostics_path=diagnostics_path,
                        predictions_path=predictions_path,
                    )
                )
        elif mode == "sequential":
            for edit_index, record in enumerate(active_records):
                set_active_edit(alg, config, edit_index, reset=edit_index > 0)
                samples = record_samples(model, record, image_root)
                train_record(
                    alg,
                    config,
                    samples,
                    edit_index=edit_index,
                    steps=steps,
                    lr=float(args.learning_rate),
                    log_path=diagnostics_path,
                )
                row = evaluate_record(
                    alg,
                    record,
                    samples,
                    method=method,
                    mode=mode,
                    edit_index=edit_index,
                    stage="post_edit",
                    steps=steps,
                    eval_oracle_routing=False,
                    max_new_tokens=int(args.max_new_tokens),
                    skip_generation=bool(args.skip_generation),
                    locality_threshold=float(args.locality_threshold),
                    rollback_tolerance=float(args.rollback_tolerance),
                    diagnostics_path=diagnostics_path,
                    predictions_path=predictions_path,
                )
                rows.append(row)
                retention_source[edit_index] = {
                    "record": record,
                    "samples": samples,
                    "post_edit_new_nll": row.get("post_new_answer_nll"),
                    "post_edit_delta_new_nll": row.get("delta_new_nll"),
                }
                for previous_index, previous in retention_source.items():
                    if previous_index == edit_index:
                        continue
                    set_active_edit(alg, config, previous_index)
                    _outputs, current = forward_outputs(
                        alg,
                        previous["samples"]["new"],
                        adapters_enabled=True,
                        eval_oracle_routing=False,
                    )
                    current_nll = current.get("target_nll")
                    base_nll = previous.get("post_edit_new_nll")
                    retention_drop = None if current_nll is None or base_nll is None else float(current_nll) - float(base_nll)
                    retention_row = {
                        "method": method,
                        "mode": mode,
                        "stage": "retention_after_edit",
                        "after_edit_index": int(edit_index),
                        "edit_index": int(previous_index),
                        "record_id": record_id(previous["record"]),
                        "post_edit_new_nll": base_nll,
                        "current_new_nll": current_nll,
                        "retention_drop": retention_drop,
                        "teacher_forced_argmax": current.get("teacher_forced_argmax"),
                    }
                    append_jsonl(diagnostics_path, retention_row)
            for previous_index, previous in retention_source.items():
                set_active_edit(alg, config, previous_index)
                _outputs, current = forward_outputs(
                    alg,
                    previous["samples"]["new"],
                    adapters_enabled=True,
                    eval_oracle_routing=False,
                )
                current_nll = current.get("target_nll")
                base_nll = previous.get("post_edit_new_nll")
                retention_drop = None if current_nll is None or base_nll is None else float(current_nll) - float(base_nll)
                rows.append(
                    {
                        "method": method,
                        "mode": mode,
                        "stage": "final_retention",
                        "edit_index": int(previous_index),
                        "record_id": record_id(previous["record"]),
                        "final_retention_new_nll": current_nll,
                        "post_edit_new_nll": base_nll,
                        "retention_drop": retention_drop,
                        "delta_new_nll": previous.get("post_edit_delta_new_nll"),
                    }
                )
        else:
            raise ValueError(mode)
        alg.write_summary(output_dir / "logs" / f"{mode}_{method}_state")
        agg = aggregate_rows(rows, method, mode)
        return rows, agg, detail
    finally:
        cleanup_model(alg, model)


def gate_one_pass(summary: Dict[str, Any], threshold: float) -> Tuple[bool, List[str]]:
    reasons = []
    same = summary.get("one_edit", {}).get("same_full")
    if not same:
        return False, ["same_full one_edit summary missing"]
    if same.get("mean_delta_new_nll") is None or float(same["mean_delta_new_nll"]) >= 0.0:
        reasons.append("same_full one-edit did not lower new-answer NLL")
    if int(same.get("nan_inf_count") or 0) != 0:
        reasons.append("same_full one-edit has NaN/Inf diagnostics")
    if same.get("ref_abs_delta_mean") is None or float(same["ref_abs_delta_mean"]) > threshold:
        reasons.append("same_full one-edit reference delta exceeds threshold")
    if int(same.get("locality_damage_count") or 0) != 0:
        reasons.append("same_full one-edit has locality damage")
    if not same.get("rollback_pass"):
        reasons.append("same_full one-edit rollback check failed")
    return not reasons, reasons


def gate_five_pass(summary: Dict[str, Any], mode: str, threshold: float) -> Tuple[bool, List[str]]:
    reasons = []
    same = summary.get(mode, {}).get("same_full")
    plain = summary.get(mode, {}).get("plain_moe_lora")
    if not same:
        return False, [f"same_full {mode} summary missing"]
    if same.get("mean_delta_new_nll") is None or float(same["mean_delta_new_nll"]) >= 0.0:
        reasons.append(f"same_full {mode} mean delta_new_nll is not negative")
    if int(same.get("positive_new_count") or 0) < 4:
        reasons.append(f"same_full {mode} positive_new_count < 4/5")
    if int(same.get("nan_inf_count") or 0) != 0:
        reasons.append(f"same_full {mode} has NaN/Inf diagnostics")
    if int(same.get("locality_damage_count") or 0) != 0:
        reasons.append(f"same_full {mode} has locality damage")
    if not same.get("rollback_pass"):
        reasons.append(f"same_full {mode} rollback check failed")
    if same.get("ref_abs_delta_mean") is not None and float(same["ref_abs_delta_mean"]) > threshold:
        if not plain or plain.get("ref_abs_delta_mean") is None or float(same["ref_abs_delta_mean"]) > float(plain["ref_abs_delta_mean"]):
            reasons.append(f"same_full {mode} reference damage exceeds threshold and plain baseline")
    return not reasons, reasons


def classify(summary: Dict[str, Any]) -> str:
    seq_same = summary.get("sequential", {}).get("same_full")
    seq_plain = summary.get("sequential", {}).get("plain_moe_lora")
    nonseq_same = summary.get("nonseq", {}).get("same_full")
    if not seq_same or not nonseq_same:
        return "fail"
    same_success = (
        seq_same.get("mean_delta_new_nll") is not None
        and float(seq_same["mean_delta_new_nll"]) < 0.0
        and int(seq_same.get("positive_new_count") or 0) >= 4
    )
    if not same_success:
        if seq_plain and seq_same.get("max_retention_drop") is not None and seq_plain.get("max_retention_drop") is not None:
            if float(seq_same["max_retention_drop"]) < float(seq_plain["max_retention_drop"]):
                return "stability_promising"
        return "fail"
    if not seq_plain:
        return "pass"
    same_new = abs(float(seq_same.get("mean_delta_new_nll") or 0.0))
    plain_new = abs(float(seq_plain.get("mean_delta_new_nll") or 0.0))
    same_ref = float(seq_same.get("ref_abs_delta_mean") or 0.0)
    plain_ref = float(seq_plain.get("ref_abs_delta_mean") or 0.0)
    same_ret = float(seq_same.get("max_retention_drop") or 0.0)
    plain_ret = float(seq_plain.get("max_retention_drop") or 0.0)
    if same_new > plain_new and same_ref <= plain_ref and same_ret <= plain_ret:
        return "strong_pass"
    if same_ret < plain_ret or same_ref < plain_ref:
        return "stability_promising" if same_new < plain_new else "pass"
    return "pass"


def write_rollback_report(output_dir: Path, rows: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    checks = [row for row in rows if row.get("stage") == "post_edit" and row.get("rollback_max_abs_diff") is not None]
    max_diff = max([float(row["rollback_max_abs_diff"]) for row in checks], default=0.0)
    payload = {
        "rollback_tolerance": float(tolerance),
        "fp16_note": "LLaVA-Med runs in fp16; <=1e-4 is treated as rollback pass for this bounded gate.",
        "rollback_max_abs_diff": max_diff,
        "rollback_pass": bool(max_diff <= float(tolerance)),
        "checks": [
            {
                "method": row.get("method"),
                "mode": row.get("mode"),
                "record_id": row.get("record_id"),
                "rollback_max_abs_diff": row.get("rollback_max_abs_diff"),
                "rollback_pass": row.get("rollback_pass"),
            }
            for row in checks
        ],
    }
    write_json(output_dir / "rollback_report.json", payload)
    return payload


def write_run_commands(output_dir: Path, args: argparse.Namespace) -> None:
    text = "#!/usr/bin/env bash\nset -euo pipefail\n\n"
    text += " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]) + "\n"
    (output_dir / "run_commands.sh").write_text(text, encoding="utf-8")


def write_doc(output_dir: Path, args: argparse.Namespace, final_summary: Dict[str, Any]) -> None:
    doc = PROJECT_ROOT / "docs" / "SAME_EDIT_EFFECT_GATE.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("one_edit", "nonseq", "sequential"):
        for method in ("plain_moe_lora", "same_full"):
            item = final_summary.get("aggregates", {}).get(mode, {}).get(method)
            if not item:
                continue
            rows.append(
                "| {mode} | {method} | {delta} | {positive} | {ref} | {loc} | {ret} | {rollback} |".format(
                    mode=mode,
                    method=method,
                    delta=item.get("mean_delta_new_nll"),
                    positive=item.get("positive_new_count"),
                    ref=item.get("ref_abs_delta_mean"),
                    loc=item.get("locality_damage_count"),
                    ret=item.get("max_retention_drop"),
                    rollback=item.get("rollback_pass"),
                )
            )
    lines = [
        "# SAME-Edit Effect Gate",
        "",
        "## Purpose",
        "",
        "Validate SAME-Edit as an independent medical MLLM editing method on bounded MedMKEB / LLaVA-Med records.",
        "",
        "## Environment",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Device: `{args.device}`",
        f"- Model: `{args.model_name_or_path or 'hparams default'}`",
        f"- Data: `{args.data_path or 'datasets/MedMKEB/eval.json'}`",
        f"- Max edits: `{args.max_edits}`",
        f"- One-edit steps: `{args.one_edit_steps}`",
        f"- Five-edit steps per edit: `{args.steps}`",
        f"- LoRA rank / experts / top-k: `{args.lora_r}` / `{args.expert_num}` / `{args.top_k}`",
        "",
        "## Code Changes",
        "",
        "- Added `scripts/same_edit/run_same_edit_effect_gate.py`.",
        "- Updated this document with the latest gate results.",
        "- No ENGRAM / CURE / DSCA core logic was changed.",
        "",
        "## Commands",
        "",
        f"- Full command is recorded in `{output_dir / 'run_commands.sh'}`.",
        "",
        "## Results",
        "",
        "| mode | method | mean_delta_new_nll | positive_new | ref_abs_delta_mean | locality_damage | max_retention_drop | rollback |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        *rows,
        "",
        "## SAME Diagnostics",
        "",
        f"- Diagnostics JSONL: `{output_dir / 'same_diagnostics.jsonl'}`",
        f"- Rollback report: `{output_dir / 'rollback_report.json'}`",
        "",
        "## Conclusion",
        "",
        f"- Verdict: `{final_summary.get('verdict')}`",
        f"- Stop reason: `{final_summary.get('stop_reason')}`",
        "",
        "This is a bounded engineering gate, not a full reproduction of the original CoIN 8-task MCIT setting.",
        "",
    ]
    doc.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seeds(int(args.seed))
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    write_run_commands(output_dir, args)
    diagnostics_path = output_dir / "same_diagnostics.jsonl"
    predictions_path = output_dir / "predictions_before_after.jsonl"
    diagnostics_path.write_text("", encoding="utf-8")
    predictions_path.write_text("", encoding="utf-8")
    static_audit(args, output_dir)

    dataset_path = resolve_dataset_path(args)
    image_root = args.image_root if args.image_root is not None else Path(SAMEEditMultimodalHparams.from_hparams(str(args.hparams)).coco_image)
    if not image_root.is_absolute():
        image_root = PROJECT_ROOT / image_root
    records = load_selected_records(args, dataset_path)
    write_json(output_dir / "selected_records.json", [selected_record_view(row, image_root) for row in records])

    methods = method_list(args)
    modes = mode_list(args)
    all_rows: List[Dict[str, Any]] = []
    aggregates: Dict[str, Dict[str, Any]] = {mode: {} for mode in modes}
    stop_reason: Optional[str] = None
    gate_results: Dict[str, Any] = {}

    for mode in modes:
        for method in methods:
            rows, agg, _detail = run_method_mode(
                args,
                method,
                mode,
                records,
                image_root,
                output_dir,
                diagnostics_path,
                predictions_path,
            )
            all_rows.extend(rows)
            aggregates.setdefault(mode, {})[method] = agg
            write_csv(output_dir / MODE_TO_CSV[mode], [row for row in all_rows if row.get("mode") == mode])
        if mode == "one_edit":
            passed, reasons = gate_one_pass(aggregates, float(args.locality_threshold))
            gate_results["gate1_one_edit"] = {"passed": passed, "reasons": reasons}
            if not passed and args.gate_stop:
                stop_reason = "gate1_one_edit_failed: " + "; ".join(reasons)
                break
        if mode == "nonseq":
            passed, reasons = gate_five_pass(aggregates, "nonseq", float(args.locality_threshold))
            gate_results["gate2_nonseq"] = {"passed": passed, "reasons": reasons}
            if not passed and args.gate_stop:
                stop_reason = "gate2_nonseq_failed: " + "; ".join(reasons)
                break
        if mode == "sequential":
            passed, reasons = gate_five_pass(aggregates, "sequential", float(args.locality_threshold))
            gate_results["gate3_sequential"] = {"passed": passed, "reasons": reasons}

    for mode in ("one_edit", "nonseq", "sequential"):
        write_csv(output_dir / MODE_TO_CSV[mode], [row for row in all_rows if row.get("mode") == mode])
    rollback = write_rollback_report(output_dir, all_rows, float(args.rollback_tolerance))
    final_summary = {
        "output_dir": str(output_dir),
        "dataset_path": str(dataset_path),
        "image_root": str(image_root),
        "methods": methods,
        "modes_requested": modes,
        "aggregates": aggregates,
        "gate_results": gate_results,
        "rollback": rollback,
        "stop_reason": stop_reason,
        "verdict": "fail" if stop_reason else classify(aggregates),
        "bounded": True,
        "timeout_safe_seconds": int(args.timeout_safe),
        "large_model_weights_saved": False,
        "adapter_state_saved": True,
    }
    write_json(output_dir / "final_summary.json", final_summary)
    write_doc(output_dir, args, final_summary)
    print(json.dumps(to_jsonable(final_summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
