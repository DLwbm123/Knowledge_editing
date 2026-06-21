from __future__ import annotations

import csv
import json
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def flatten_metric_row(metric: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metric = jsonable(metric)
    post = metric.get("post") or {}
    pre = metric.get("pre") or {}
    row = {
        "case_id": metric.get("case_id", metric.get("case_nums")),
        "runtime_sec": metric.get("time"),
        "rewrite_acc": post.get("rewrite_acc"),
        "current_edit_success": post.get("rewrite_acc"),
        "rephrase_acc": post.get("rephrase_acc"),
        "image_rephrase_acc": post.get("image_rephrase_acc"),
        "locality_acc": post.get("locality_acc"),
        "multimodal_locality_acc": post.get("multimodal_locality_acc"),
        "portability_acc": post.get("portability_acc"),
        "portability_status": "reported_by_easyedit" if post.get("portability_acc") is not None else "not_implemented_for_this_model",
        "robustness_acc": post.get("robustness_acc"),
        "retained_edit_success": post.get("retained_edit_success"),
        "pre_rewrite_acc": pre.get("rewrite_acc"),
    }
    if metadata:
        row.update(
            {
                "record_id": metadata.get("id"),
                "modality": metadata.get("modality"),
                "department": metadata.get("department"),
                "clinical_VQA_task": metadata.get("clinical_VQA_task"),
                "perceptual_granularity": metadata.get("perceptual_granularity"),
            }
        )
    return row


def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    metric_keys = [
        "rewrite_acc",
        "current_edit_success",
        "rephrase_acc",
        "image_rephrase_acc",
        "locality_acc",
        "multimodal_locality_acc",
        "portability_acc",
        "robustness_acc",
        "runtime_sec",
    ]
    summary: Dict[str, Any] = {"n": len(rows)}
    for key in metric_keys:
        values = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        summary[f"mean_{key}"] = mean(values) if values else None
    return [summary]


def write_metrics_jsonl(path, metrics: Iterable[Dict[str, Any]], metadata: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, metric in enumerate(metrics):
            meta = metadata[idx] if metadata and idx < len(metadata) else None
            payload = {"metric": jsonable(metric), "summary_row": flatten_metric_row(metric, meta)}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows.append(payload["summary_row"])
    return rows


def write_summary_csv(path, rows: List[Dict[str, Any]]) -> None:
    summary_rows = summarize_rows(rows)
    fieldnames = sorted({key for row in rows + summary_rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        if summary_rows:
            writer.writerow(summary_rows[0])
