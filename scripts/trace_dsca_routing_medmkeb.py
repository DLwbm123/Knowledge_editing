#!/usr/bin/env python3
"""Trace DSCA routing for the completed MedMKEB pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

from dsca_medmkeb_diag_common import (
    active_ids,
    clone_batch,
    collate_record,
    load_dataset_and_model,
    read_jsonl,
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--repository-step", type=int, default=20)
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
        device=next(alg.repository.parameters(), alg.repository.p_f).device if len(alg.repository) else alg.repository.p_f.device,
        dtype=alg.repository.p_f.dtype,
    )
    ids: List[int] = []
    for idx in range(limit):
        batch = collate_record(dataset, dataset[idx])
        reps = alg.capture_representations(clone_batch(batch["edit_inner"]))
        assigned, _created = repo.assign_batch(reps["h_f"], reps["h_v"], initialize_basis=False)
        ids.extend(assigned)
    return ids


def prediction_success_by_step(run_dir: Path) -> Dict[int, bool]:
    rows = read_jsonl(run_dir / "predictions.jsonl")
    result = {}
    for row in rows:
        if row.get("sample_type") != "rel":
            continue
        step = int(row.get("step"))
        score = row.get("score")
        result[step] = bool(score is not None and float(score) >= 1.0)
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, _model, alg, _tokenizer, config, dataset_path = load_dataset_and_model(args, num_samples=args.num_samples, load_repo=True)
    limit = min(args.num_samples, len(dataset))
    assigned_ids = replay_assignments(dataset, alg, config, limit)
    success = prediction_success_by_step(args.run_dir)
    rows: List[Dict[str, Any]] = []
    for idx in range(limit):
        step = idx + 1
        record = dataset[idx]
        batch = collate_record(dataset, record)
        rel_batch = clone_batch(batch["edit_inner"])
        reps = alg.capture_representations(rel_batch)
        weights, selected, aux = dsca_route(
            reps["h_v"],
            reps["h_f"],
            alg.repository,
            tau_visual=alg.tau_visual,
            route_temperature=alg.route_temperature,
            candidate_topk=alg.candidate_topk,
        )
        with torch.no_grad():
            _ = alg(clone_batch(batch["edit_inner"]))
        assigned = assigned_ids[idx] if idx < len(assigned_ids) else None
        candidate_ids = selected[0].nonzero(as_tuple=False).flatten().detach().cpu().tolist() if selected.numel() else []
        active = active_ids(alg.repository)
        row = {
            "step": step,
            "assigned_cluster_id": assigned,
            "cluster_buffer_size": int(alg.repository.pca_buffers[assigned].shape[0]) if assigned is not None and assigned < len(alg.repository.pca_buffers) else None,
            "assigned_cluster_active": bool(alg.repository.active[assigned].item()) if assigned is not None and assigned < len(alg.repository) else False,
            "candidate_cluster_ids": ";".join(str(x) for x in candidate_ids),
            "assigned_in_candidates": bool(assigned in candidate_ids) if assigned is not None else False,
            "route_weight_assigned_cluster": float(weights[0, assigned].detach().cpu()) if assigned is not None and weights.numel() and assigned < weights.shape[1] else 0.0,
            "visual_similarity_to_assigned_cluster": float(aux["visual_sim"][0, assigned].detach().cpu()) if assigned is not None and aux["visual_sim"].numel() and assigned < aux["visual_sim"].shape[1] else None,
            "fused_similarity_to_assigned_cluster": float(aux["fused_sim"][0, assigned].detach().cpu()) if assigned is not None and aux["fused_sim"].numel() and assigned < aux["fused_sim"].shape[1] else None,
            "tau_visual": alg.tau_visual,
            "active_dsam_ids": ";".join(str(x) for x in active),
            "residual_norm": float(alg._last_info.get("dsca/residual_norm_mean", 0.0)),
            "prediction_success": success.get(step),
        }
        rows.append(row)
    csv_path = output_dir / "routing_trace.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    candidate_counts = [len(str(row["candidate_cluster_ids"]).split(";")) if row["candidate_cluster_ids"] else 0 for row in rows]
    summary = {
        "dataset_path": str(dataset_path),
        "num_rows": len(rows),
        "assigned_cluster_in_candidates_rate": sum(1 for row in rows if row["assigned_in_candidates"]) / len(rows) if rows else None,
        "active_dsam_available_rate": sum(1 for row in rows if row["assigned_cluster_active"]) / len(rows) if rows else None,
        "mean_route_weight_assigned_cluster": sum(float(row["route_weight_assigned_cluster"]) for row in rows) / len(rows) if rows else None,
        "candidate_count_distribution": dict(Counter(candidate_counts)),
        "active_dsam_ids_final": active_ids(alg.repository),
        "num_clusters_final": len(alg.repository),
        "num_active_dsams_final": alg.repository.num_active(),
    }
    write_json(output_dir / "routing_trace_summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
