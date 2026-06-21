#!/usr/bin/env python3
"""Teacher-forced target NLL diagnostics for a completed DSCA MedMKEB run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from dsca_medmkeb_diag_common import (
    active_ids,
    aligned_logits_and_labels,
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
from easyeditor.trainer.algs.dsca_utils import DSCAConceptRepository, dsca_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--model", default="blip2", choices=["blip2"])
    parser.add_argument("--hparams", default="hparams/DSCA/blip2_20edit_pilot.yaml")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--repository-step", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def replay_assignments(dataset: Any, alg: Any, config: Any, limit: int) -> List[int]:
    repo = DSCAConceptRepository(
        hidden_size=alg.hidden_size,
        rank=alg.rank,
        gate_bottleneck=alg.gate_bottleneck,
        cluster_alpha=float(getattr(config, "dsca_cluster_alpha", 2.0)),
        proto_ema=float(getattr(config, "dsca_proto_ema", 0.95)),
        min_samples=int(getattr(config, "dsca_min_samples", 4)),
        refine_interval=int(getattr(config, "dsca_refine_interval", 10)),
        dsam_init_std=float(getattr(config, "dsca_dsam_init_std", 0.02)),
        residual_scale=float(getattr(config, "dsca_residual_scale", 1.0)),
        device=alg.repository.p_f.device,
        dtype=alg.repository.p_f.dtype,
    )
    ids: List[int] = []
    for idx in range(limit):
        batch = collate_record(dataset, dataset[idx])
        reps = alg.capture_representations(clone_batch(batch["edit_inner"]))
        assigned, _created = repo.assign_batch(reps["h_f"], reps["h_v"], initialize_basis=False)
        ids.extend(assigned)
    return ids


def logits_delta_on_targets(outputs_a: Any, outputs_b: Any, batch: Dict[str, Any]) -> float:
    logits_a, labels = aligned_logits_and_labels(outputs_a, batch)
    logits_b, _ = aligned_logits_and_labels(outputs_b, batch)
    mask = labels != -100
    if int(mask.sum().item()) == 0:
        return float("nan")
    delta = (logits_b.float() - logits_a.float()).masked_select(mask.unsqueeze(-1)).float()
    return float(delta.norm().detach().cpu())


def temporary_residual_scale(alg: Any, value: float):
    class _Ctx:
        def __enter__(self_inner):
            self_inner.old_values = [float(dsam.residual_scale) for dsam in alg.repository.dsams]
            for dsam in alg.repository.dsams:
                dsam.residual_scale = float(value)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            for dsam, old_value in zip(alg.repository.dsams, self_inner.old_values):
                dsam.residual_scale = old_value

    return _Ctx()


def route_snapshot(alg: Any, batch: Dict[str, Any], assigned: Optional[int]) -> Dict[str, Any]:
    reps = alg.capture_representations(clone_batch(batch))
    weights, selected, aux = dsca_route(
        reps["h_v"],
        reps["h_f"],
        alg.repository,
        tau_visual=alg.tau_visual,
        route_temperature=alg.route_temperature,
        candidate_topk=alg.candidate_topk,
    )
    candidate_ids = selected[0].nonzero(as_tuple=False).flatten().detach().cpu().tolist() if selected.numel() else []
    return {
        "routed_candidate_count": len(candidate_ids),
        "routed_cluster_ids": ";".join(str(item) for item in candidate_ids),
        "route_weights": ";".join(f"{float(x):.8g}" for x in weights[0].detach().cpu().tolist()) if weights.numel() else "",
        "active_dsam_ids": ";".join(str(item) for item in active_ids(alg.repository)),
        "selected_active_dsam_count": int(torch.count_nonzero(weights[0] > 0).item()) if weights.numel() else 0,
        "assigned_cluster_id": assigned,
        "assigned_in_candidates": bool(assigned in candidate_ids) if assigned is not None else False,
        "route_weight_assigned_cluster": float(weights[0, assigned].detach().cpu())
        if assigned is not None and weights.numel() and assigned < weights.shape[1]
        else 0.0,
        "visual_similarity_to_assigned_cluster": float(aux["visual_sim"][0, assigned].detach().cpu())
        if assigned is not None and aux["visual_sim"].numel() and assigned < aux["visual_sim"].shape[1]
        else None,
        "fused_similarity_to_assigned_cluster": float(aux["fused_sim"][0, assigned].detach().cpu())
        if assigned is not None and aux["fused_sim"].numel() and assigned < aux["fused_sim"].shape[1]
        else None,
    }


def nll_metrics(prefix: str, outputs: Any, batch: Dict[str, Any]) -> Dict[str, Any]:
    metrics = target_nll_from_outputs(outputs, batch)
    return {
        f"{prefix}_target_nll": metrics["target_nll"],
        f"{prefix}_avg_target_logprob": metrics["avg_target_logprob"],
        f"{prefix}_first_target_token_rank": metrics["first_target_token_rank"],
    }


def mean(values: List[float]) -> Optional[float]:
    finite = [float(item) for item in values if item == item]
    return sum(finite) / len(finite) if finite else None


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / "target_nll_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()

    dataset, model, alg, tokenizer, config, dataset_path = load_dataset_and_model(args, num_samples=args.num_samples, load_repo=True)
    limit = min(args.num_samples, len(dataset))
    assigned_ids = replay_assignments(dataset, alg, config, limit)
    rows: List[Dict[str, Any]] = []

    for idx in range(limit):
        step = idx + 1
        record = dataset[idx]
        batch = collate_record(dataset, record)
        edit_batch = clone_batch(batch["edit_inner"])
        assigned = assigned_ids[idx] if idx < len(assigned_ids) else None
        with torch.no_grad():
            base_outputs = model(clone_batch(edit_batch))
            edited_outputs = alg(clone_batch(edit_batch))
            edited_info = dict(alg._last_info)
            edited_weights = alg._last_route_weights.detach().clone() if alg._last_route_weights is not None else None
            with temporary_residual_scale(alg, 0.0):
                residual0_outputs = alg(clone_batch(edit_batch))
            if assigned is not None and assigned < len(alg.repository) and bool(alg.repository.active[assigned].item()):
                with temporarily_force_route(alg, assigned):
                    force_outputs = alg(clone_batch(edit_batch))
            else:
                force_outputs = None
        route = route_snapshot(alg, clone_batch(edit_batch), assigned)
        base_nll = target_nll_from_outputs(base_outputs, edit_batch)
        edited_nll = target_nll_from_outputs(edited_outputs, edit_batch)
        residual0_nll = target_nll_from_outputs(residual0_outputs, edit_batch)
        force_nll = target_nll_from_outputs(force_outputs, edit_batch) if force_outputs is not None else {}
        row: Dict[str, Any] = {
            "step": step,
            "prompt": record.get("prompt"),
            "target": record.get("target"),
            "target_token_count": base_nll["target_token_count"],
            **nll_metrics("base", base_outputs, edit_batch),
            **nll_metrics("edited", edited_outputs, edit_batch),
            **nll_metrics("residual0", residual0_outputs, edit_batch),
            "force_target_nll": force_nll.get("target_nll"),
            "force_avg_target_logprob": force_nll.get("avg_target_logprob"),
            "force_first_target_token_rank": force_nll.get("first_target_token_rank"),
            "delta_nll": edited_nll["target_nll"] - base_nll["target_nll"],
            "residual0_delta_nll": residual0_nll["target_nll"] - base_nll["target_nll"],
            "force_delta_nll": force_nll.get("target_nll", float("nan")) - base_nll["target_nll"]
            if force_outputs is not None
            else None,
            "edited_prediction_argmax": decode_argmax_on_labels(tokenizer, edited_outputs, edit_batch),
            "dsam_residual_norm_at_edit_layer": float(edited_info.get("dsca/residual_norm_mean", 0.0)),
            "hidden_delta_norm": float(edited_info.get("dsca/residual_norm_mean", 0.0)),
            "logits_delta_norm_on_target_positions": logits_delta_on_targets(base_outputs, edited_outputs, edit_batch),
            "residual0_logits_delta_norm_on_target_positions": logits_delta_on_targets(base_outputs, residual0_outputs, edit_batch),
            "force_logits_delta_norm_on_target_positions": logits_delta_on_targets(base_outputs, force_outputs, edit_batch)
            if force_outputs is not None
            else None,
            **route,
        }
        if edited_weights is not None:
            row["edited_route_weight_l1"] = float(edited_weights.abs().sum().detach().cpu())
        rows.append(row)
        append_jsonl(debug_path, row)

    csv_path = output_dir / "target_nll_per_sample.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    deltas = [float(row["delta_nll"]) for row in rows if row.get("delta_nll") is not None]
    logits_deltas = [float(row["logits_delta_norm_on_target_positions"]) for row in rows if row.get("logits_delta_norm_on_target_positions") is not None]
    summary = {
        "dataset_path": str(dataset_path),
        "num_rows": len(rows),
        "mean_base_target_nll": mean([row["base_target_nll"] for row in rows]),
        "mean_edited_target_nll": mean([row["edited_target_nll"] for row in rows]),
        "mean_delta_nll": mean(deltas),
        "num_improved_target_nll": sum(1 for value in deltas if value < 0.0),
        "num_route_missing": sum(1 for row in rows if int(row.get("routed_candidate_count") or 0) == 0),
        "mean_logits_delta_norm_on_target_positions": mean(logits_deltas),
        "num_residual_has_no_logits_effect": sum(
            1
            for row in rows
            if float(row.get("dsam_residual_norm_at_edit_layer") or 0.0) > 1.0e-8
            and float(row.get("logits_delta_norm_on_target_positions") or 0.0) <= 1.0e-8
        ),
        "mean_force_target_nll": mean([row["force_target_nll"] for row in rows if row.get("force_target_nll") is not None]),
        "num_force_improved_target_nll": sum(
            1
            for row in rows
            if row.get("force_delta_nll") is not None and float(row["force_delta_nll"]) < 0.0
        ),
        "assigned_cluster_in_candidates_rate": sum(1 for row in rows if row.get("assigned_in_candidates")) / len(rows) if rows else None,
        "active_dsam_available_rate": sum(
            1
            for row in rows
            if row.get("assigned_cluster_id") is not None
            and int(row["assigned_cluster_id"]) < len(alg.repository)
            and bool(alg.repository.active[int(row["assigned_cluster_id"])].item())
        )
        / len(rows)
        if rows
        else None,
    }
    write_json(output_dir / "target_nll_summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
