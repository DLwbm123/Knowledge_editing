#!/usr/bin/env python3
"""Bounded 5-edit SAME-Edit smoke for MedMKEB LLaVA-Med."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(SCRIPT_DIR))

from dsca_medmkeb_diag_common import append_jsonl, clone_batch, ensure_offline_env, resolve_dataset_path, to_jsonable, torch_device, write_json  # noqa: E402
from easyeditor.models.same_edit import print_same_edit_trainable_summary, same_edit_gradient_summary  # noqa: E402
from easyeditor.trainer.algs.same_edit import SAMEEdit  # noqa: E402
from easyeditor.trainer.models import get_model  # noqa: E402
from overfit_same_edit_one_medmkeb_edit import make_sample, normalize_device_arg, record_prompt, record_target, snapshot, str2bool  # noqa: E402
from easyeditor.models.same_edit import SAMEEditMultimodalHparams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=Path, default=None)
    parser.add_argument("--image-root", "--image_root", dest="image_root", type=Path, default=None)
    parser.add_argument("--sample-start", "--sample_start", dest="sample_start", type=int, default=0)
    parser.add_argument("--num-edits", "--num_edits", dest="num_edits", type=int, default=5)
    parser.add_argument("--max-steps-per-edit", "--max_steps_per_edit", dest="max_steps_per_edit", type=int, default=100)
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
    parser.add_argument("--route-loss-weight", "--route_loss_weight", dest="route_loss_weight", type=float, default=None)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=16)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=25)
    parser.add_argument("--locality-nll-threshold", "--locality_nll_threshold", dest="locality_nll_threshold", type=float, default=1.0)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(args: argparse.Namespace) -> SAMEEditMultimodalHparams:
    config = SAMEEditMultimodalHparams.from_hparams(args.hparams)
    config.device = normalize_device_arg(args.device)
    if args.image_root is not None:
        config.coco_image = str(args.image_root)
        config.rephrase_image = str(args.image_root)
    config.lr = float(args.learning_rate)
    config.same_edit_num_steps = int(args.max_steps_per_edit)
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
    if args.route_loss_weight is not None:
        config.same_edit_route_loss_weight = float(args.route_loss_weight)
    return config


def load_records(dataset_path: Path) -> List[Dict[str, Any]]:
    records = json.loads(dataset_path.read_text(errors="replace"))
    if not isinstance(records, list):
        raise RuntimeError(f"Dataset JSON root must be a list: {dataset_path}")
    return records


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def set_active_edit(
    alg: SAMEEdit,
    config: SAMEEditMultimodalHparams,
    edit_index: int,
    reset: bool = False,
    snapshot_previous: bool = True,
) -> None:
    edit_index = int(edit_index)
    config.same_edit_current_edit = edit_index
    alg.config.same_edit_current_edit = edit_index
    alg.same_config.current_edit = edit_index
    if reset:
        alg.same_model.reset_for_new_edit(edit_index, snapshot_previous=snapshot_previous)
    else:
        alg.same_model.set_current_edit(edit_index)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    ensure_offline_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = args.output_dir / "same_edit_5edit_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()

    dataset_path = resolve_dataset_path(args.dataset, Path.cwd(), args.dataset_path)
    records = load_records(dataset_path)
    selected = records[args.sample_start : args.sample_start + args.num_edits]
    if len(selected) < args.num_edits:
        raise RuntimeError(f"Requested {args.num_edits} edits from {dataset_path}, found {len(selected)}.")

    config = configure(args)
    image_root = Path(config.coco_image).expanduser()
    if not str(image_root):
        raise RuntimeError("image_root is required either via --image-root or config.coco_image.")
    device = torch_device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    model = get_model(config).to(device).eval()
    alg = SAMEEdit(model, config, lambda: None).to(device)
    trainable_summary = print_same_edit_trainable_summary(alg.same_model)
    write_json(args.output_dir / "same_edit_trainable_summary.json", trainable_summary)
    if int(trainable_summary.get("base_trainable_param_count") or 0) != 0:
        raise RuntimeError(f"SAME-Edit base trainable params are nonzero: {trainable_summary}")

    tokenizer = model.llava_tokenizer
    edit_results: List[Dict[str, Any]] = []
    retained: Dict[int, Dict[str, Any]] = {}
    locality_damage_count = 0
    locality_checks = 0

    for edit_idx, record in enumerate(selected):
        if edit_idx == 0:
            set_active_edit(alg, config, edit_idx)
        else:
            set_active_edit(alg, config, edit_idx, reset=True, snapshot_previous=True)
        sample = make_sample(model, record, image_root)
        batch = {
            "edit_inner": clone_batch(sample),
            "edit_outer": clone_batch(sample),
            "loc": clone_batch(sample),
            "loc_image": clone_batch(sample),
        }
        target = record_target(record)
        initial = snapshot(
            alg,
            tokenizer,
            sample,
            target,
            args.max_new_tokens,
            step=0,
            eval_oracle_routing=True,
        )
        optimizer = torch.optim.Adam(alg.outer_parameters(), lr=float(config.lr))
        last_loss_total: Optional[torch.Tensor] = None
        last_loss_edit: Optional[torch.Tensor] = None
        for step in range(1, int(args.max_steps_per_edit) + 1):
            set_active_edit(alg, config, edit_idx)
            optimizer.zero_grad(set_to_none=True)
            loss_total, loss_edit, _loss_loc, _loss_base, _info = alg.edit_step(batch, training=True, optimizer=optimizer)
            torch.nn.utils.clip_grad_norm_(alg.outer_parameters(), float(config.grad_clip), error_if_nonfinite=True)
            optimizer.step()
            last_loss_total = loss_total
            last_loss_edit = loss_edit
            if step == 1 or step == args.max_steps_per_edit or step % max(1, args.log_every) == 0:
                row = {
                    "phase": "train_step",
                    "edit_index": edit_idx,
                    "record_id": record.get("id"),
                    "step": step,
                    "loss": float(loss_total.detach().cpu()),
                    "edit_loss": float(loss_edit.detach().cpu()),
                    **same_edit_gradient_summary(alg.same_model),
                }
                append_jsonl(debug_path, row)
        alg.same_model.save_covariance_snapshot()
        set_active_edit(alg, config, edit_idx)
        final = snapshot(
            alg,
            tokenizer,
            sample,
            target,
            args.max_new_tokens,
            step=int(args.max_steps_per_edit),
            eval_oracle_routing=True,
        )
        learned = snapshot(
            alg,
            tokenizer,
            sample,
            target,
            args.max_new_tokens,
            step=int(args.max_steps_per_edit),
            eval_oracle_routing=False,
        )
        grad_summary = same_edit_gradient_summary(alg.same_model)
        result = {
            "edit_index": edit_idx,
            "record_id": record.get("id"),
            "assigned_expert": final.get("assigned_expert_id"),
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
            "reference_delta": final.get("reference_delta"),
            "routing_vector": final.get("routing_vector"),
            "top_expert": final.get("top_expert_id"),
            "active_expert_count": final.get("active_expert_count"),
            "routing_entropy": final.get("routing_entropy"),
            "nan_inf_count": grad_summary.get("nan_inf_grad_count"),
            "last_loss": float(last_loss_total.detach().cpu()) if last_loss_total is not None else None,
            "last_edit_loss": float(last_loss_edit.detach().cpu()) if last_loss_edit is not None else None,
            "learned_eval": learned,
            **grad_summary,
        }
        retained[edit_idx] = {"sample": sample, "target": target, "post_edit_target_nll": final.get("target_nll")}
        for previous_idx, previous in retained.items():
            if previous_idx == edit_idx:
                continue
            set_active_edit(alg, config, previous_idx)
            retained_snapshot = snapshot(
                alg,
                tokenizer,
                previous["sample"],
                previous["target"],
                args.max_new_tokens,
                step=int(args.max_steps_per_edit),
                eval_oracle_routing=True,
            )
            locality_checks += 1
            nll_before = previous.get("post_edit_target_nll")
            nll_after = retained_snapshot.get("target_nll")
            damaged = (
                nll_before is not None
                and nll_after is not None
                and float(nll_after) - float(nll_before) > float(args.locality_nll_threshold)
            ) or not bool(retained_snapshot.get("teacher_forced_contains"))
            locality_damage_count += int(damaged)
            append_jsonl(
                debug_path,
                {
                    "phase": "retention_check",
                    "after_edit_index": edit_idx,
                    "checked_edit_index": previous_idx,
                    "damaged": bool(damaged),
                    "post_edit_target_nll": nll_before,
                    "current_target_nll": nll_after,
                    "teacher_forced_contains": retained_snapshot.get("teacher_forced_contains"),
                    "routing_vector": retained_snapshot.get("routing_vector"),
                },
            )
        set_active_edit(alg, config, edit_idx)
        edit_results.append(result)
        append_jsonl(debug_path, {"phase": "edit_final", **result})

    alg.same_model.save_covariance_snapshot()
    alg.write_summary(args.output_dir)
    target_decreases = [float(row["target_nll_decrease"]) for row in edit_results if row.get("target_nll_decrease") is not None]
    reference_deltas = [abs(float(row["reference_delta"])) for row in edit_results if row.get("reference_delta") is not None]
    routing_hits = [
        int(row.get("top_expert") == row.get("assigned_expert"))
        for row in edit_results
        if row.get("top_expert") is not None and row.get("assigned_expert") is not None
    ]
    routing_entropies = [float(row["routing_entropy"]) for row in edit_results if row.get("routing_entropy") is not None]
    assigned_hist = [0 for _ in range(max(1, int(config.same_edit_expert_num)))]
    for row in edit_results:
        assigned = row.get("assigned_expert")
        if assigned is not None and 0 <= int(assigned) < len(assigned_hist):
            assigned_hist[int(assigned)] += 1
    final_model_summary = alg.same_model.summary()
    summary = {
        "dataset_path": str(dataset_path),
        "image_root": str(image_root),
        "output_dir": str(args.output_dir),
        "num_edits": len(edit_results),
        "num_positive_edits": sum(1 for value in target_decreases if value > 0.0),
        "mean_target_nll_decrease": mean(target_decreases),
        "mean_reference_delta_abs": mean(reference_deltas),
        "locality_damage_count": locality_damage_count,
        "locality_check_count": locality_checks,
        "teacher_forced_exact_count": sum(1 for row in edit_results if row.get("teacher_forced_exact")),
        "teacher_forced_contains_count": sum(1 for row in edit_results if row.get("teacher_forced_contains")),
        "free_exact_count": sum(1 for row in edit_results if row.get("free_generation_exact")),
        "free_contains_count": sum(1 for row in edit_results if row.get("free_generation_contains")),
        "routing_accuracy_against_assigned_expert": mean([float(hit) for hit in routing_hits]),
        "mean_routing_entropy": mean(routing_entropies),
        "assigned_expert_usage_histogram": assigned_hist,
        "expert_usage_histogram": final_model_summary.get("expert_usage_histogram"),
        "cov_prev_valid_layer_count": final_model_summary.get("covariance_valid_count"),
        "nan_inf_count": sum(int(row.get("nan_inf_count") or 0) for row in edit_results),
        "edits": edit_results,
        "same_edit_state": str(args.output_dir / "same_edit_state.pt"),
        "same_edit_summary": str(args.output_dir / "same_edit_summary.json"),
        "debug_jsonl": str(debug_path),
        "long_runs_started": False,
    }
    write_json(args.output_dir / "same_edit_5edit_summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
