from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .bank import EngramBank

LOG = logging.getLogger(__name__)


def tensor_overlap(left: torch.Tensor, right: torch.Tensor, eps: float = 1.0e-12) -> float:
    if left.shape != right.shape:
        return 0.0
    a = left.detach().cpu().float().reshape(-1)
    b = right.detach().cpu().float().reshape(-1)
    denom = float(a.norm() * b.norm()) + eps
    return float(torch.dot(a, b) / denom)


def projector_overlap(left: torch.Tensor, right: torch.Tensor, eps: float = 1.0e-12) -> float:
    if left.dim() != 2 or right.dim() != 2 or left.shape[1] != right.shape[0]:
        return tensor_overlap(left, right, eps=eps) if left.shape == right.shape else 0.0
    numerator = float(left.matmul(right).norm().detach().cpu())
    denom = float(left.norm().detach().cpu() * right.norm().detach().cpu()) + eps
    return numerator / denom


def _layer_tensor(raw: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], str]:
    projector = raw.get("projector")
    if isinstance(projector, torch.Tensor):
        return projector, "projector"
    weight = raw.get("weight")
    if isinstance(weight, torch.Tensor):
        return weight, "weight"
    return None, "none"


def compute_pair_overlap(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_updates = left["updates"]
    right_updates = right["updates"]
    common = sorted(set(left_updates) & set(right_updates))
    module_rows: List[Dict[str, Any]] = []
    for module_name in common:
        left_tensor, source = _layer_tensor(left_updates[module_name])
        right_tensor, right_source = _layer_tensor(right_updates[module_name])
        if left_tensor is None or right_tensor is None:
            continue
        if source == "projector" and right_source == "projector":
            score = projector_overlap(left_tensor, right_tensor)
        else:
            score = tensor_overlap(left_tensor, right_tensor)
            source = "weight"
        module_rows.append({"module_name": module_name, "overlap": score, "source": source})
    aggregate = sum(row["overlap"] for row in module_rows) / max(len(module_rows), 1)
    return {"aggregate_overlap": aggregate, "modules": module_rows}


def compute_bank_overlap(
    bank_dir: str | Path,
    *,
    edit_ids: Optional[Iterable[str]] = None,
    threshold: float = 0.35,
) -> Dict[str, Any]:
    bank = EngramBank(bank_dir)
    ids = list(edit_ids) if edit_ids is not None else [item["edit_id"] for item in bank.list_edits()]
    edits = {edit_id: bank.load_edit(edit_id) for edit_id in ids}
    pairs: List[Dict[str, Any]] = []
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            result = compute_pair_overlap(edits[left_id], edits[right_id])
            row = {
                "edit_id_i": left_id,
                "edit_id_j": right_id,
                "aggregate_overlap": result["aggregate_overlap"],
                "warning": bool(result["aggregate_overlap"] > threshold),
                "modules": result["modules"],
            }
            if row["warning"]:
                LOG.warning(
                    "[ENGRAM] overlap %.4f exceeds threshold %.4f for %s vs %s",
                    row["aggregate_overlap"],
                    threshold,
                    left_id,
                    right_id,
                )
            pairs.append(row)
    return {"bank_dir": str(bank_dir), "threshold": threshold, "edit_ids": ids, "pairs": pairs}


def write_overlap_report(report: Dict[str, Any], out_dir: str | Path, *, heatmap: bool = False) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "engram_overlap.json"
    csv_path = out / "engram_overlap.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["edit_id_i", "edit_id_j", "aggregate_overlap", "warning"])
        writer.writeheader()
        for pair in report.get("pairs", []):
            writer.writerow(
                {
                    "edit_id_i": pair["edit_id_i"],
                    "edit_id_j": pair["edit_id_j"],
                    "aggregate_overlap": pair["aggregate_overlap"],
                    "warning": pair["warning"],
                }
            )

    paths = {"json": str(json_path), "csv": str(csv_path)}
    if heatmap:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            ids = report.get("edit_ids", [])
            matrix = np.eye(len(ids), dtype=float)
            pos = {edit_id: idx for idx, edit_id in enumerate(ids)}
            for pair in report.get("pairs", []):
                i = pos[pair["edit_id_i"]]
                j = pos[pair["edit_id_j"]]
                matrix[i, j] = matrix[j, i] = pair["aggregate_overlap"]
            fig, ax = plt.subplots(figsize=(max(4, len(ids) * 0.7), max(4, len(ids) * 0.7)))
            im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(len(ids)))
            ax.set_yticks(range(len(ids)))
            ax.set_xticklabels(ids, rotation=45, ha="right")
            ax.set_yticklabels(ids)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            png_path = out / "engram_overlap_heatmap.png"
            fig.savefig(png_path, dpi=160)
            plt.close(fig)
            paths["png"] = str(png_path)
        except Exception as exc:
            paths["png_error"] = f"{type(exc).__name__}: {exc}"
    return paths

