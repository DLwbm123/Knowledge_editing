#!/usr/bin/env python3
"""Diagnostic one-edit DSCA overfit run for MedMKEB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from dsca_medmkeb_diag_common import (
    active_ids,
    aligned_logits_and_labels,
    answer_fields,
    append_jsonl,
    clone_batch,
    collate_record,
    decode_argmax_on_labels,
    load_dataset_and_model,
    target_nll_from_outputs,
    temporarily_force_route,
    to_jsonable,
    write_json,
)
from easyeditor.trainer.algs.dsca_utils import dsca_route, orthonormalize_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="blip2", choices=["blip2", "llava-med"])
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--refine-interval", type=int, default=999999)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--task-only", action="store_true")
    parser.add_argument("--force-route-assigned-cluster", action="store_true")
    parser.add_argument(
        "--diagnostic-cold-start-basis",
        default="none",
        choices=["none", "random_orthonormal", "noisy_duplicate"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hparams", default="hparams/DSCA/blip2_20edit_pilot.yaml")
    parser.add_argument("--training-hparams", default=None)
    parser.add_argument("--dsca-generation-mode", default=None, choices=["normal", "prefill_only", "cache_reuse_route"])
    parser.add_argument("--residual-apply-mask", default=None, choices=["attention", "vision_prompt", "all_nonpad", "current_token"])
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--generation-use-cache", default=None, choices=["true", "false"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sampled_base_param_signature(model: torch.nn.Module, max_tensors: int = 16, values_per_tensor: int = 16) -> Tuple[float, float]:
    total_sum = 0.0
    total_norm = 0.0
    seen = 0
    with torch.no_grad():
        for param in model.parameters():
            if param.numel() == 0:
                continue
            data = param.detach().flatten()[:values_per_tensor].float().cpu()
            total_sum += float(data.sum())
            total_norm += float(data.norm())
            seen += 1
            if seen >= max_tensors:
                break
    return total_sum, total_norm


def signature_delta(lhs: Tuple[float, float], rhs: Tuple[float, float]) -> float:
    return abs(lhs[0] - rhs[0]) + abs(lhs[1] - rhs[1])


def duplicate_optimizer_params(optimizer: torch.optim.Optimizer) -> bool:
    seen = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            ident = id(param)
            if ident in seen:
                return True
            seen.add(ident)
    return False


def logits_delta_on_targets(outputs_a: Any, outputs_b: Any, batch: Dict[str, Any]) -> float:
    logits_a, labels = aligned_logits_and_labels(outputs_a, batch)
    logits_b, _ = aligned_logits_and_labels(outputs_b, batch)
    mask = labels != -100
    if int(mask.sum().item()) == 0:
        return float("nan")
    delta = (logits_b.float() - logits_a.float()).masked_select(mask.unsqueeze(-1)).float()
    return float(delta.norm().detach().cpu())


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def initialize_diagnostic_cluster(alg: Any, batch: Dict[str, Any], mode: str) -> int:
    reps = alg.capture_representations(clone_batch(batch))
    ids, _created = alg.repository.assign_batch(reps["h_f"], reps["h_v"], initialize_basis=True)
    cluster_id = ids[0]
    if mode == "noisy_duplicate":
        while alg.repository.pca_buffers[cluster_id].shape[0] < alg.repository.min_samples:
            noise = torch.randn_like(reps["h_f"][0]) * 1.0e-4
            alg.repository.append_to_buffer(cluster_id, reps["h_f"][0] + noise)
        alg.repository.initialize_basis_if_ready(cluster_id, force=True, reason="diagnostic_noisy_duplicate")
    elif mode == "random_orthonormal":
        basis_seed = torch.randn(
            alg.rank,
            alg.hidden_size,
            device=alg.repository.p_f.device,
            dtype=alg.repository.p_f.dtype,
        )
        basis = orthonormalize_rows(basis_seed, alg.rank, alg.hidden_size)
        alg.repository.dsams[cluster_id].set_basis(basis.to(alg.repository.p_f.device, alg.repository.p_f.dtype))
        alg.repository.active[cluster_id] = True
    return cluster_id


def route_info(alg: Any, batch: Dict[str, Any], cluster_id: Optional[int]) -> Dict[str, Any]:
    reps = alg.capture_representations(clone_batch(batch))
    weights, selected, _aux = dsca_route(
        reps["h_v"],
        reps["h_f"],
        alg.repository,
        tau_visual=alg.tau_visual,
        route_temperature=alg.route_temperature,
        candidate_topk=alg.candidate_topk,
    )
    candidate_ids = selected[0].nonzero(as_tuple=False).flatten().detach().cpu().tolist() if selected.numel() else []
    return {
        "route_candidate_count": len(candidate_ids),
        "candidate_cluster_ids": ";".join(str(item) for item in candidate_ids),
        "route_weight_to_assigned_cluster": float(weights[0, cluster_id].detach().cpu())
        if cluster_id is not None and weights.numel() and cluster_id < weights.shape[1]
        else 0.0,
        "active_dsam_ids": ";".join(str(item) for item in active_ids(alg.repository)),
    }


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


def record_prompt(record: Dict[str, Any], fallback: Dict[str, Any]) -> Any:
    return record.get("src") or record.get("prompt") or record.get("question") or fallback.get("prompt")


def record_target(record: Dict[str, Any], fallback: Dict[str, Any]) -> Any:
    return record.get("alt") or record.get("target") or fallback.get("target")


def target_success(prediction: str, target: Any) -> Dict[str, Any]:
    fields = answer_fields(None, prediction, target)
    return {
        "contains_target": bool(fields["contains_target"]),
        "exact_match": bool(fields["exact_match_normalized"]),
        "exact_match_normalized": bool(fields["exact_match_normalized"]),
    }


def evaluate_snapshot(
    model: torch.nn.Module,
    alg: Any,
    tokenizer: Any,
    batch: Dict[str, Any],
    target: Any,
) -> Dict[str, Any]:
    with torch.no_grad():
        base_outputs = model(clone_batch(batch))
        edited_outputs = alg(clone_batch(batch))
    base_nll = target_nll_from_outputs(base_outputs, batch)
    nll = target_nll_from_outputs(edited_outputs, batch)
    prediction = decode_argmax_on_labels(tokenizer, edited_outputs, batch)
    fields = answer_fields(None, prediction, target)
    return {
        "base_target_nll": base_nll["target_nll"],
        "target_nll": nll["target_nll"],
        "base_first_target_token_rank": base_nll.get("first_target_token_rank"),
        "first_target_token_rank": nll.get("first_target_token_rank"),
        "generated_prediction": prediction,
        "exact_match_normalized": fields["exact_match_normalized"],
        "contains_target": fields["contains_target"],
        "residual_norm": float(alg._last_info.get("dsca/residual_norm_mean", 0.0)),
        "hidden_delta_norm": float(alg._last_info.get("dsca/residual_norm_mean", 0.0)),
        "logits_delta_norm": logits_delta_on_targets(base_outputs, edited_outputs, batch),
    }


def resolve_llava_generation_policy(args: argparse.Namespace, config: Any):
    from smoke_llava_med_dsca_generation import resolve_generation_policy

    return resolve_generation_policy(args, config)


def llava_free_sample(model: Any, raw_record: Dict[str, Any], image_root: Path):
    from smoke_llava_med_dsca_generation import make_sample

    return make_sample(model, raw_record, image_root)


def llava_generate_snapshot(
    args: argparse.Namespace,
    model: Any,
    alg: Any,
    config: Any,
    sample: Dict[str, Any],
    raw_record: Dict[str, Any],
    cluster_id: Optional[int],
    step: int,
    target: Any,
) -> Dict[str, Any]:
    from smoke_llava_med_dsca_generation import generate_text

    event_path = args.output_dir / "generation_hook_events.jsonl"
    use_cache, generation_mode, residual_apply_mask, reuse_prefill_route = resolve_llava_generation_policy(args, config)
    forced_ids = [int(cluster_id)] if args.force_route_assigned_cluster and cluster_id is not None else None
    base = generate_text(
        model,
        sample,
        args.max_new_tokens,
        sample_id=step,
        call_label=f"overfit_base_step_{step:03d}",
        use_cache=use_cache,
    )
    edited = generate_text(
        model,
        sample,
        args.max_new_tokens,
        alg=alg,
        sample_id=step,
        call_label=f"overfit_edited_step_{step:03d}",
        use_cache=use_cache,
        debug_generation_hooks=True,
        event_path=event_path,
        force_route_ids=forced_ids,
        residual_apply_mask_mode=residual_apply_mask,
        generation_mode=generation_mode,
        generation_reuse_prefill_route=reuse_prefill_route,
    )
    fields = answer_fields(base["text"], edited["text"], target)
    active_route_nonzero = [
        bool(value) for value in edited.get("residual_nonzero_by_step", []) if value is not None
    ]
    active_rate = (
        sum(1 for value in active_route_nonzero if value) / len(active_route_nonzero)
        if active_route_nonzero
        else None
    )
    return {
        "base_free_generation": base["text"],
        "free_generation": edited["text"],
        "free_generation_contains_target": bool(fields["contains_target"]),
        "free_generation_exact_match": bool(fields["exact_match_normalized"]),
        "free_generation_edited_equals_base": bool(fields["edited_equals_base"]),
        "generation_hook_active": bool(edited.get("hook_entered")),
        "generation_hook_error_count": int(edited.get("hook_error_count", 0)),
        "active_route_nonzero_residual_rate": active_rate,
        "generation_residual_norm_mean": float(edited.get("residual_norm", 0.0)),
        "cached_decode_route_reused": bool(edited.get("cached_decode_route_reused")),
        "cached_decode_hook_event_count": int(edited.get("cached_decode_hook_event_count", 0)),
        "cached_decode_residual_norm_by_step": edited.get("cached_decode_residual_norm_by_step", []),
        "current_token_apply_mask_sum": edited.get("current_token_apply_mask_sum", []),
        "dsca_generation_mode": generation_mode,
        "residual_apply_mask": residual_apply_mask,
        "generation_use_cache": use_cache,
        "target": target,
        "prompt": record_prompt(raw_record, {}),
    }


def write_prediction_file(path: Path, row: Dict[str, Any]) -> None:
    lines = [
        f"step: {row.get('step')}",
        f"prompt: {row.get('prompt')}",
        f"target: {row.get('target')}",
        f"base_free_generation: {row.get('base_free_generation')}",
        f"free_generation: {row.get('free_generation')}",
        f"teacher_forced_argmax_prediction: {row.get('generated_prediction')}",
        f"contains_target: {row.get('contains_target')}",
        f"exact_match: {row.get('exact_match_normalized')}",
        f"free_generation_contains_target: {row.get('free_generation_contains_target')}",
        f"free_generation_exact_match: {row.get('free_generation_exact_match')}",
        f"target_nll: {row.get('target_nll')}",
        f"first_target_token_rank: {row.get('first_target_token_rank')}",
        f"generation_hook_active: {row.get('generation_hook_active')}",
        f"active_route_nonzero_residual_rate: {row.get('active_route_nonzero_residual_rate')}",
    ]
    path.write_text("\n".join(str(item) for item in lines) + "\n")


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    output_dir = args.output_dir.resolve()
    args.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / "overfit_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()
    hook_event_path = output_dir / "generation_hook_events.jsonl"
    if hook_event_path.exists():
        hook_event_path.unlink()

    dataset_size = max(args.sample_index + 1, 1)
    dataset, model, alg, tokenizer, config, dataset_path = load_dataset_and_model(args, num_samples=dataset_size, load_repo=False)
    raw_record = load_raw_record(dataset_path, args.sample_index)
    record = raw_record if args.model == "llava-med" else dataset[args.sample_index]
    target = record_target(raw_record, record)
    prompt = record_prompt(raw_record, record)
    free_sample = llava_free_sample(model, raw_record, args.image_root) if args.model == "llava-med" else None
    if free_sample is not None:
        edit_batch = clone_batch(free_sample)
        batch = {
            "edit_inner": clone_batch(free_sample),
            "loc_image": clone_batch(free_sample),
            "loc": clone_batch(free_sample),
        }
    else:
        batch = collate_record(dataset, record)
        edit_batch = clone_batch(batch["edit_inner"])
    cluster_id = initialize_diagnostic_cluster(alg, edit_batch, args.diagnostic_cold_start_basis)
    optimizer = torch.optim.Adam(alg.outer_parameters(), lr=float(args.learning_rate))
    alg.dsca_register_new_params_with_optimizer(optimizer)
    base_signature_before = sampled_base_param_signature(model)

    rows: List[Dict[str, Any]] = []
    success = False
    first_success_step: Optional[int] = None
    final_snapshot: Dict[str, Any] = {}

    def log_snapshot(step: int, loss_total: Optional[torch.Tensor] = None, loss_task: Optional[torch.Tensor] = None, info: Optional[Dict[str, Any]] = None, active_grad: float = 0.0) -> Dict[str, Any]:
        with temporarily_force_route(alg, cluster_id) if args.force_route_assigned_cluster else _null_context():
            snapshot = evaluate_snapshot(model, alg, tokenizer, edit_batch, target)
            route = route_info(alg, edit_batch, cluster_id)
            if free_sample is not None:
                snapshot.update(llava_generate_snapshot(args, model, alg, config, free_sample, raw_record, cluster_id, step, target))
            else:
                snapshot.update(
                    {
                        "base_free_generation": "",
                        "free_generation": snapshot["generated_prediction"],
                        "free_generation_contains_target": snapshot["contains_target"],
                        "free_generation_exact_match": snapshot["exact_match_normalized"],
                        "free_generation_edited_equals_base": False,
                        "generation_hook_active": False,
                        "generation_hook_error_count": 0,
                        "active_route_nonzero_residual_rate": None,
                        "generation_residual_norm_mean": snapshot["residual_norm"],
                    }
                )
        base_delta = signature_delta(base_signature_before, sampled_base_param_signature(model))
        row = {
            "step": step,
            "sample_index": args.sample_index,
            "assigned_cluster_id": cluster_id,
            "prompt": prompt,
            "target": target,
            "task_loss": float(loss_task.detach().cpu()) if loss_task is not None else float("nan"),
            "total_loss": float(loss_total.detach().cpu()) if loss_total is not None else float("nan"),
            **snapshot,
            **route,
            "active_dsam_grad_norm": active_grad,
            "base_vlm_params_changed": bool(base_delta > 1.0e-8),
            "base_param_delta_norm": base_delta,
            "R_k_requires_grad_any": any(dsam.R.requires_grad for dsam in alg.repository.dsams),
            "duplicate_optimizer_param_groups": duplicate_optimizer_params(optimizer),
            "num_clusters": len(alg.repository),
            "num_active_dsams": alg.repository.num_active(),
            "loss_finite": True if loss_total is None else math.isfinite(float(loss_total.detach().cpu())),
            "dsca_residual_norm_mean": float((info or {}).get("dsca/residual_norm_mean", 0.0)),
        }
        rows.append(row)
        append_jsonl(debug_path, row)
        write_prediction_file(output_dir / f"prediction_at_step_{step:03d}.txt", row)
        return row

    final_snapshot = log_snapshot(0)
    if bool(final_snapshot["free_generation_exact_match"]) or bool(final_snapshot["free_generation_contains_target"]):
        success = True
        first_success_step = 0

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        context = temporarily_force_route(alg, cluster_id) if args.force_route_assigned_cluster else None
        if context is not None:
            context.__enter__()
        try:
            loss_total, loss_task, _loss_loc, _loss_sparse, info = alg.edit_step(batch, training=True, optimizer=optimizer)
        finally:
            if context is not None:
                context.__exit__(None, None, None)
        active_params = []
        if cluster_id < len(alg.repository.dsams):
            active_params = [param for param in alg.repository.dsams[cluster_id].parameters() if param.requires_grad]
        active_grad = grad_norm(active_params)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        should_log = step == 1 or step == args.steps or step % max(args.log_every, 1) == 0
        if should_log:
            row = log_snapshot(step, loss_total=loss_total, loss_task=loss_task, info=info, active_grad=active_grad)
            final_snapshot = row
            if bool(row["free_generation_exact_match"]) or bool(row["free_generation_contains_target"]):
                success = True
                if first_success_step is None:
                    first_success_step = step

    csv_path = output_dir / "overfit_trace.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    repo_path = output_dir / "repository_after_overfit.pt"
    alg.repository.save(str(repo_path))
    final_repo_path = output_dir / "final_repository.pt"
    alg.repository.save(str(final_repo_path))
    summary = {
        "dataset_path": str(dataset_path),
        "sample_index": args.sample_index,
        "prompt": prompt,
        "target": target,
        "model": args.model,
        "steps": args.steps,
        "task_only": bool(args.task_only),
        "force_route_assigned_cluster": bool(args.force_route_assigned_cluster),
        "diagnostic_cold_start_basis": args.diagnostic_cold_start_basis,
        "residual_scale": args.residual_scale if args.residual_scale is not None else getattr(config, "dsca_residual_scale", None),
        "dsca_generation_mode": args.dsca_generation_mode or getattr(config, "dsca_generation_mode", None),
        "residual_apply_mask": args.residual_apply_mask or getattr(config, "dsca_generation_residual_apply_mask", None),
        "assigned_cluster_id": cluster_id,
        "success": success,
        "first_success_step": first_success_step,
        "final_target_nll": final_snapshot.get("target_nll"),
        "final_base_target_nll": final_snapshot.get("base_target_nll"),
        "final_first_target_token_rank": final_snapshot.get("first_target_token_rank"),
        "final_base_first_target_token_rank": final_snapshot.get("base_first_target_token_rank"),
        "final_prediction": final_snapshot.get("generated_prediction"),
        "final_free_generation": final_snapshot.get("free_generation"),
        "final_base_free_generation": final_snapshot.get("base_free_generation"),
        "final_contains_target": final_snapshot.get("contains_target"),
        "final_exact_match_normalized": final_snapshot.get("exact_match_normalized"),
        "final_free_generation_contains_target": final_snapshot.get("free_generation_contains_target"),
        "final_free_generation_exact_match": final_snapshot.get("free_generation_exact_match"),
        "final_route_weight_to_assigned_cluster": final_snapshot.get("route_weight_to_assigned_cluster"),
        "base_vlm_params_changed": final_snapshot.get("base_vlm_params_changed"),
        "R_k_requires_grad_any": final_snapshot.get("R_k_requires_grad_any"),
        "duplicate_optimizer_param_groups": final_snapshot.get("duplicate_optimizer_param_groups"),
        "generation_hook_active": any(bool(row.get("generation_hook_active")) for row in rows),
        "generation_hook_error_count": sum(int(row.get("generation_hook_error_count") or 0) for row in rows),
        "active_route_nonzero_residual_rate": final_snapshot.get("active_route_nonzero_residual_rate"),
        "repository_path": str(repo_path),
        "final_repository_path": str(final_repo_path),
        "num_clusters": len(alg.repository),
        "num_active_dsams": alg.repository.num_active(),
    }
    torch.save(
        {
            "repository_state_dict": alg.repository.state_dict(),
            "assigned_cluster_id": cluster_id,
            "sample_index": args.sample_index,
            "target": target,
            "prompt": prompt,
            "summary": to_jsonable(summary),
        },
        output_dir / "final_dsca_state.pt",
    )
    try:
        import yaml

        config_payload = {
            key: to_jsonable(value)
            for key, value in vars(config).items()
            if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict, Path))
        }
        (output_dir / "final_config_resolved.yaml").write_text(yaml.safe_dump(config_payload, sort_keys=True))
    except Exception as exc:
        (output_dir / "final_config_resolved.yaml").write_text(f"config_export_error: {exc}\n")
    write_json(output_dir / "overfit_summary.json", summary)
    write_json(output_dir / "final_summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    main()
