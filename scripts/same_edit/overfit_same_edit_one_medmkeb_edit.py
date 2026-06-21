#!/usr/bin/env python3
"""One-edit SAME-Edit overfit smoke for MedMKEB LLaVA-Med."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dsca_medmkeb_diag_common import (  # noqa: E402
    answer_fields,
    append_jsonl,
    clone_batch,
    decode_argmax_on_labels,
    ensure_offline_env,
    resolve_dataset_path,
    target_nll_from_outputs,
    to_jsonable,
    torch_device,
    write_json,
)
from easyeditor.models.same_edit import SAMEEditMultimodalHparams  # noqa: E402
from easyeditor.models.same_edit import (  # noqa: E402
    print_same_edit_trainable_summary,
    same_edit_gradient_summary,
)
from easyeditor.trainer.algs.same_edit import SAMEEdit  # noqa: E402
from easyeditor.trainer.llava_med_models.llava_med import build_llava_med_masks  # noqa: E402
from easyeditor.trainer.models import get_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=Path, default=None)
    parser.add_argument("--image-root", "--image_root", dest="image_root", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", "--max-steps", "--max_steps", dest="steps", type=int, default=100)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--hparams", "--config", dest="hparams", default="hparams/TRAINING/SAME_EDIT/llava_med_oneedit_smoke.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expert-num", "--expert_num", dest="expert_num", type=int, default=None)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=None)
    parser.add_argument("--lora-r", "--lora_r", dest="lora_r", type=int, default=None)
    parser.add_argument("--lora-alpha", "--lora_alpha", dest="lora_alpha", type=float, default=None)
    parser.add_argument("--target-modules", "--target_layers", dest="target_modules", default=None)
    parser.add_argument("--adaptive-activation", "--adaptive_activation", dest="adaptive_activation", type=str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--curvature-mode", "--curvature_mode", dest="curvature_mode", choices=["off", "prism", "safe"], default=None)
    parser.add_argument("--spectral-router", "--spectral_router", dest="spectral_router", type=str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--oracle-edit-routing", "--oracle_edit_routing", dest="oracle_edit_routing", type=str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--eval-oracle-routing", "--eval_oracle_routing", dest="eval_oracle_routing", type=str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--route-loss-weight", "--route_loss_weight", dest="route_loss_weight", type=float, default=None)
    parser.add_argument("--validate-save-load", "--validate_save_load", dest="validate_save_load", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=16)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=5)
    return parser.parse_args()


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_device_arg(text: str) -> Any:
    if text == "cuda":
        return "cuda"
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        return int(suffix) if suffix.isdigit() else text
    return int(text) if text.isdigit() else text


def load_raw_record(dataset_path: Path, sample_index: int) -> Dict[str, Any]:
    records = json.loads(dataset_path.read_text(errors="replace"))
    if not isinstance(records, list):
        raise RuntimeError(f"Dataset JSON root must be a list: {dataset_path}")
    if sample_index < 0 or sample_index >= len(records):
        raise IndexError(f"sample_index={sample_index} outside dataset of size {len(records)}")
    record = records[sample_index]
    if not isinstance(record, dict):
        raise RuntimeError(f"Dataset row {sample_index} is not an object.")
    return record


def resolve_image_path(image_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = image_root.resolve()
    if root.name == "images" and str(value).startswith("images/"):
        return root.parent / path
    return root / path


def record_prompt(record: Dict[str, Any]) -> str:
    return "Question: {} Short answer: ".format(record.get("src") or record.get("prompt") or record.get("question") or "")


def record_target(record: Dict[str, Any]) -> str:
    return str(record.get("alt") or record.get("target") or "")


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


def prepare_generation_inputs(model: Any, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    prompt = sample["prompt"][0]
    prompt_text = model._conversation_prompt(prompt, None)
    input_ids = model.tokenizer_image_token(
        prompt_text,
        model.llava_tokenizer,
        model.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
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
    return input_ids, image_tensor, {name: value.unsqueeze(0) for name, value in masks.items()}


def generate_text(alg: SAMEEdit, sample: Dict[str, Any], max_new_tokens: int, adapters_enabled: bool) -> str:
    raw_model = alg.same_model.model
    input_ids, image_tensor, _masks = prepare_generation_inputs(raw_model, sample)
    context = alg.same_model.adapters_disabled() if not adapters_enabled else null_context()
    with context:
        with torch.inference_mode():
            output_ids = raw_model.llava_model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=raw_model.llava_tokenizer.pad_token_id,
                eos_token_id=raw_model.llava_tokenizer.eos_token_id,
            )
    input_len = int(input_ids.shape[1])
    generated_ids = output_ids[:, input_len:] if output_ids.shape[1] >= input_len else output_ids
    return raw_model.llava_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


class null_context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def logits_delta_on_targets(outputs_a: Any, outputs_b: Any, batch: Dict[str, Any]) -> float:
    logits_a = outputs_a.logits if not isinstance(outputs_a, torch.Tensor) else outputs_a
    logits_b = outputs_b.logits if not isinstance(outputs_b, torch.Tensor) else outputs_b
    labels = batch["labels"]
    seq = min(logits_a.shape[1], logits_b.shape[1], labels.shape[1] + 1)
    logits_a = logits_a[:, -seq:]
    logits_b = logits_b[:, -seq:]
    return float((logits_b.detach().float() - logits_a.detach().float()).norm().cpu())


def grad_nonfinite_count(parameters) -> int:
    count = 0
    for param in parameters:
        if param.grad is not None:
            count += int((~torch.isfinite(param.grad)).sum().detach().cpu().item())
    return count


def configure(args: argparse.Namespace) -> SAMEEditMultimodalHparams:
    config = SAMEEditMultimodalHparams.from_hparams(args.hparams)
    config.device = normalize_device_arg(args.device)
    if args.image_root is not None:
        config.coco_image = str(args.image_root)
        config.rephrase_image = str(args.image_root)
    config.lr = float(args.learning_rate)
    config.same_edit_num_steps = int(args.steps)
    if args.expert_num is not None:
        config.same_edit_expert_num = int(args.expert_num)
    if args.top_k is not None:
        config.same_edit_top_k = int(args.top_k)
    if args.lora_r is not None:
        config.same_edit_lora_r = int(args.lora_r)
    if args.lora_alpha is not None:
        config.same_edit_lora_alpha = float(args.lora_alpha)
    if args.target_modules:
        config.same_edit_target_modules = str(args.target_modules)
    if args.adaptive_activation is not None:
        config.same_edit_adaptive_activation = bool(args.adaptive_activation)
    if args.curvature_mode:
        config.same_edit_curvature_mode = str(args.curvature_mode)
    if args.spectral_router is not None:
        config.same_edit_spectral_router = bool(args.spectral_router)
    if args.oracle_edit_routing is not None:
        config.same_edit_oracle_edit_routing = bool(args.oracle_edit_routing)
    if args.eval_oracle_routing is not None:
        config.same_edit_eval_oracle_edit_routing = bool(args.eval_oracle_routing)
    if args.route_loss_weight is not None:
        config.same_edit_route_loss_weight = float(args.route_loss_weight)
    return config


@contextmanager
def eval_routing_context(alg: SAMEEdit, eval_oracle_routing: Optional[bool]):
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


def snapshot(
    alg: SAMEEdit,
    tokenizer: Any,
    sample: Dict[str, Any],
    target: str,
    max_new_tokens: int,
    step: int,
    eval_oracle_routing: Optional[bool] = None,
) -> Dict[str, Any]:
    with eval_routing_context(alg, eval_oracle_routing):
        with torch.no_grad(), alg.same_model.adapters_disabled():
            base_batch = clone_batch(sample)
            base_outputs = alg.model(base_batch)
        with torch.no_grad():
            edited_batch = clone_batch(sample)
            edited_outputs = alg(edited_batch)
        base_free = generate_text(alg, clone_batch(sample), max_new_tokens, adapters_enabled=False)
        edited_free = generate_text(alg, clone_batch(sample), max_new_tokens, adapters_enabled=True)
    base_nll = target_nll_from_outputs(base_outputs, base_batch)
    edited_nll = target_nll_from_outputs(edited_outputs, edited_batch)
    teacher_prediction = decode_argmax_on_labels(tokenizer, edited_outputs, edited_batch)
    teacher_fields = answer_fields(None, teacher_prediction, target)
    free_fields = answer_fields(base_free, edited_free, target)
    summary = alg.same_model.summary()
    first_layer = summary.get("layers", [{}])[0] if summary.get("layers") else {}
    routing = first_layer.get("routing") or []
    assigned = summary.get("assigned_expert_id")
    routing_overlap = float(routing[int(assigned)]) if routing and assigned is not None and int(assigned) < len(routing) else 0.0
    return {
        "step": step,
        "eval_oracle_routing": eval_oracle_routing,
        "base_target_nll": base_nll.get("target_nll"),
        "target_nll": edited_nll.get("target_nll"),
        "target_nll_decrease": (base_nll.get("target_nll") or 0.0) - (edited_nll.get("target_nll") or 0.0),
        "base_first_target_token_rank": base_nll.get("first_target_token_rank"),
        "first_target_token_rank": edited_nll.get("first_target_token_rank"),
        "teacher_forced_argmax_prediction": teacher_prediction,
        "teacher_forced_exact": bool(teacher_fields["exact_match_normalized"]),
        "teacher_forced_contains": bool(teacher_fields["contains_target"]),
        "base_free_generation": base_free,
        "free_generation": edited_free,
        "free_generation_exact": bool(free_fields["exact_match_normalized"]),
        "free_generation_contains": bool(free_fields["contains_target"]),
        "free_generation_edited_equals_base": bool(free_fields["edited_equals_base"]),
        "reference_delta": logits_delta_on_targets(base_outputs, edited_outputs, edited_batch),
        "routing_vector": routing,
        "routing_entropy": first_layer.get("routing_entropy"),
        "top_expert_id": first_layer.get("top_expert_id"),
        "assigned_expert_id": assigned,
        "routing_overlap": routing_overlap,
        "active_expert_count": first_layer.get("active_expert_count"),
        "expert_usage_histogram": summary.get("expert_usage_histogram"),
        "covariance_valid_count": summary.get("covariance_valid_count"),
        "cov_prev_valid_count": summary.get("covariance_valid_count"),
        "expert_masks": [row.get("expert_mask") for row in summary.get("layers", [])],
    }


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    ensure_offline_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = args.output_dir / "same_edit_oneedit_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()

    dataset_path = resolve_dataset_path(args.dataset, Path.cwd(), args.dataset_path)
    record = load_raw_record(dataset_path, args.sample_index)
    target = record_target(record)
    config = configure(args)
    image_root = Path(config.coco_image).expanduser()
    if not str(image_root):
        raise RuntimeError("image_root is required either via --image-root or config.coco_image.")
    device = torch_device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    model = get_model(config).to(device).eval()
    alg = SAMEEdit(model, config, lambda: None).to(device)
    alg.same_model.set_current_edit(0)
    trainable_summary = print_same_edit_trainable_summary(alg.same_model)
    write_json(args.output_dir / "same_edit_trainable_summary.json", trainable_summary)
    if int(trainable_summary.get("base_trainable_param_count") or 0) != 0:
        raise RuntimeError(f"SAME-Edit base trainable params are nonzero: {trainable_summary}")
    tokenizer = model.llava_tokenizer
    sample = make_sample(model, record, image_root)
    batch = {
        "edit_inner": clone_batch(sample),
        "edit_outer": clone_batch(sample),
        "loc": clone_batch(sample),
        "loc_image": clone_batch(sample),
    }
    optimizer = torch.optim.Adam(alg.outer_parameters(), lr=float(config.lr))
    primary_eval_oracle = (
        bool(args.eval_oracle_routing)
        if args.eval_oracle_routing is not None
        else bool(config.same_edit_oracle_edit_routing)
    )

    rows: List[Dict[str, Any]] = []

    def log(
        step: int,
        loss_total: Optional[torch.Tensor] = None,
        loss_edit: Optional[torch.Tensor] = None,
        phase: str = "train",
        eval_oracle_routing: Optional[bool] = None,
    ) -> Dict[str, Any]:
        grad_summary = same_edit_gradient_summary(alg.same_model)
        row = {
            "phase": phase,
            "sample_index": args.sample_index,
            "record_id": record.get("id"),
            "prompt": record_prompt(record),
            "target": target,
            "total_loss": float(loss_total.detach().cpu()) if loss_total is not None else float("nan"),
            "edit_loss": float(loss_edit.detach().cpu()) if loss_edit is not None else float("nan"),
            **snapshot(
                alg,
                tokenizer,
                sample,
                target,
                args.max_new_tokens,
                step,
                eval_oracle_routing=eval_oracle_routing,
            ),
            **grad_summary,
            "nan_inf_count": grad_summary["nan_inf_grad_count"],
        }
        rows.append(row)
        append_jsonl(debug_path, row)
        return row

    initial = log(0, eval_oracle_routing=primary_eval_oracle)
    last_training_row = initial
    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total, loss_edit, _loss_loc, _loss_base, _info = alg.edit_step(batch, training=True, optimizer=optimizer)
        torch.nn.utils.clip_grad_norm_(alg.outer_parameters(), float(config.grad_clip), error_if_nonfinite=True)
        optimizer.step()
        should_log = step == 1 or step == args.steps or step % max(1, args.log_every) == 0
        if should_log:
            last_training_row = log(
                step,
                loss_total=loss_total,
                loss_edit=loss_edit,
                eval_oracle_routing=primary_eval_oracle,
            )

    alg.same_model.save_covariance_snapshot()
    alg.write_summary(args.output_dir)
    oracle_final = log(int(args.steps), phase="final_oracle", eval_oracle_routing=True)
    learned_final = log(int(args.steps), phase="final_learned", eval_oracle_routing=False)
    final = oracle_final if primary_eval_oracle else learned_final
    csv_path = args.output_dir / "same_edit_oneedit_trace.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows([to_jsonable(row) for row in rows])
    summary = {
        "dataset_path": str(dataset_path),
        "image_root": str(image_root),
        "sample_index": args.sample_index,
        "record_id": record.get("id"),
        "steps": int(args.steps),
        "output_dir": str(args.output_dir),
        "initial_target_nll": initial.get("target_nll"),
        "final_target_nll": final.get("target_nll"),
        "target_nll_decrease": (
            (initial.get("target_nll") or 0.0) - (final.get("target_nll") or 0.0)
            if initial.get("target_nll") is not None and final.get("target_nll") is not None
            else None
        ),
        "teacher_forced_exact": final.get("teacher_forced_exact"),
        "teacher_forced_contains": final.get("teacher_forced_contains"),
        "free_generation_exact": final.get("free_generation_exact"),
        "free_generation_contains": final.get("free_generation_contains"),
        "free_decode_exact": final.get("free_generation_exact"),
        "free_decode_contains": final.get("free_generation_contains"),
        "reference_delta": final.get("reference_delta"),
        "locality_damage_count": None,
        "locality_damage_note": "No separate locality set is used by the one-edit smoke; inspect reference_delta.",
        "routing_vector": final.get("routing_vector"),
        "routing_entropy": final.get("routing_entropy"),
        "top_expert": final.get("top_expert_id"),
        "top_expert_id": final.get("top_expert_id"),
        "assigned_expert": final.get("assigned_expert_id"),
        "assigned_expert_id": final.get("assigned_expert_id"),
        "active_expert_count": final.get("active_expert_count"),
        "lora_A_grad_norm": last_training_row.get("lora_A_grad_norm"),
        "lora_B_grad_norm": last_training_row.get("lora_B_grad_norm"),
        "router_grad_norm": last_training_row.get("router_grad_norm"),
        "base_grad_norm": last_training_row.get("base_grad_norm"),
        "nan_inf_grad_count": last_training_row.get("nan_inf_grad_count"),
        "nan_inf_count": last_training_row.get("nan_inf_count"),
        "oracle_eval": oracle_final,
        "learned_eval": learned_final,
        "trainable_summary": trainable_summary,
        "same_edit_state": str(args.output_dir / "same_edit_state.pt"),
        "same_edit_summary": str(args.output_dir / "same_edit_summary.json"),
        "trace_csv": str(csv_path),
        "save_load_validation": None,
        "long_runs_started": False,
    }
    if args.validate_save_load:
        state_path = args.output_dir / "same_edit_state.pt"
        before_reload = final
        del optimizer
        del alg
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reload_model = get_model(config).to(device).eval()
        reload_alg = SAMEEdit(reload_model, config, lambda: None).to(device)
        reload_alg.same_model.set_current_edit(0)
        reload_alg.same_model.load_same_edit_state(state_path)
        reload_sample = make_sample(reload_model, record, image_root)
        after_reload = snapshot(
            reload_alg,
            reload_model.llava_tokenizer,
            reload_sample,
            target,
            args.max_new_tokens,
            int(args.steps),
            eval_oracle_routing=primary_eval_oracle,
        )
        before_routing = before_reload.get("routing_vector") or []
        after_routing = after_reload.get("routing_vector") or []
        routing_l1 = (
            sum(abs(float(a) - float(b)) for a, b in zip(before_routing, after_routing))
            if len(before_routing) == len(after_routing)
            else None
        )
        reload_validation = {
            "state_path": str(state_path),
            "before": before_reload,
            "after": after_reload,
            "target_nll_abs_diff": (
                abs(float(before_reload["target_nll"]) - float(after_reload["target_nll"]))
                if before_reload.get("target_nll") is not None and after_reload.get("target_nll") is not None
                else None
            ),
            "routing_l1_diff": routing_l1,
            "top_expert_match": before_reload.get("top_expert_id") == after_reload.get("top_expert_id"),
            "cov_prev_valid_count_match": before_reload.get("cov_prev_valid_count") == after_reload.get("cov_prev_valid_count"),
            "expert_masks_match": before_reload.get("expert_masks") == after_reload.get("expert_masks"),
            "free_decode_match": before_reload.get("free_generation") == after_reload.get("free_generation"),
        }
        write_json(args.output_dir / "same_edit_reload_validation.json", reload_validation)
        summary["save_load_validation"] = reload_validation
    write_json(args.output_dir / "same_edit_oneedit_summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
