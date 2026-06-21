#!/usr/bin/env python3
"""Offline failure analysis for a completed DSCA MedMKEB run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from dsca_medmkeb_diag_common import (
    alias_list,
    answer_fields,
    normalize_medical_answer,
    read_jsonl,
    to_jsonable,
    token_count,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def resolve_image_path(image_path: Any, run_dir: Path, image_root: Path) -> Optional[str]:
    if image_path is None or str(image_path).strip() == "":
        return None
    raw = Path(str(image_path))
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([image_root / raw, run_dir / raw])
        if image_root.name == "images":
            candidates.append(image_root.parent / raw)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def prediction_row(row: Dict[str, Any], run_dir: Path, image_root: Path) -> Dict[str, Any]:
    aliases = row.get("aliases", row.get("target_aliases"))
    fields = answer_fields(row.get("base_prediction"), row.get("edited_prediction"), row.get("target"), aliases)
    normalized_target = fields["normalized_target"]
    normalized_base = fields["normalized_base_prediction"]
    normalized_edited = fields["normalized_edited_prediction"]
    exact_raw = bool_or_none(row.get("exact_match_raw"))
    exact_norm = bool_or_none(row.get("exact_match_normalized"))
    contains = bool_or_none(row.get("contains_target"))
    if exact_raw is None:
        exact_raw = fields["exact_match_raw"]
    if exact_norm is None:
        exact_norm = fields["exact_match_normalized"]
    if contains is None:
        contains = fields["contains_target"]
    return {
        "step": row.get("step"),
        "sample_type": row.get("sample_type"),
        "prompt": row.get("prompt"),
        "target": row.get("target"),
        "aliases": "; ".join(alias_list(aliases)),
        "base_prediction": row.get("base_prediction"),
        "edited_prediction": row.get("edited_prediction"),
        "exact_match_raw": exact_raw,
        "exact_match_normalized": exact_norm,
        "contains_target": contains,
        "normalized_target": normalized_target,
        "normalized_base_prediction": normalized_base,
        "normalized_edited_prediction": normalized_edited,
        "target_token_count_estimate": int(row.get("target_token_count") or len(normalized_target.split())),
        "prediction_empty_or_missing": not bool(str(row.get("edited_prediction") or "").strip()),
        "prediction_identical_base_edited": normalized_base == normalized_edited if normalized_base or normalized_edited else False,
        "base_contains_target": bool_or_none(row.get("base_contains_target")) if row.get("base_contains_target") is not None else fields["base_contains_target"],
        "edited_contains_target": bool_or_none(row.get("edited_contains_target")) if row.get("edited_contains_target") is not None else fields["edited_contains_target"],
        "missing_field_notes": row.get("missing_field_notes", row.get("warning")),
        "image_path_resolved": resolve_image_path(row.get("image_path"), run_dir, image_root),
    }


def missing_fields(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    expected = [
        "exact_match_raw",
        "exact_match_normalized",
        "contains_target",
        "base_contains_target",
        "edited_contains_target",
        "normalized_target",
        "normalized_base_prediction",
        "normalized_edited_prediction",
        "aliases",
        "alias_exact_match",
        "alias_contains_match",
        "target_token_count",
        "generated_token_count",
        "generation_config",
        "edited_equals_base",
    ]
    counts = {}
    for field in expected:
        counts[field] = sum(1 for row in rows if field not in row)
    return counts


def write_examples(path: Path, table_rows: Sequence[Dict[str, Any]]) -> None:
    failures = [
        row
        for row in table_rows
        if row["sample_type"] in {"rel", "t_gen", "m_gen"}
        and not bool(row["edited_contains_target"])
        and not bool(row["exact_match_normalized"])
    ]
    lines = ["# DSCA MedMKEB Representative Failures", ""]
    for idx, row in enumerate(failures[:20], start=1):
        reasons = []
        if row["prediction_empty_or_missing"]:
            reasons.append("edited prediction is empty/missing")
        if row["prediction_identical_base_edited"]:
            reasons.append("edited prediction is identical to base")
        if not row["edited_contains_target"]:
            reasons.append("edited prediction does not contain normalized target")
        lines.extend(
            [
                f"## Failure {idx}: step {row['step']} `{row['sample_type']}`",
                "",
                f"- image path: `{row['image_path_resolved'] or 'not available'}`",
                f"- prompt: {row['prompt']}",
                f"- target: `{row['target']}`",
                f"- base prediction: `{row['base_prediction']}`",
                f"- edited prediction: `{row['edited_prediction']}`",
                f"- normalized target: `{row['normalized_target']}`",
                f"- normalized base: `{row['normalized_base_prediction']}`",
                f"- normalized edited: `{row['normalized_edited_prediction']}`",
                f"- why counted as failure: {', '.join(reasons) if reasons else 'no exact/contains match'}",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics_per_step.csv"
    diagnostics_path = run_dir / "dsca_diagnostics.jsonl"
    final_summary_path = run_dir / "final_summary.json"
    config_path = run_dir / "config_resolved.yaml"
    rows = read_jsonl(prediction_path)
    table_rows = [prediction_row(row, run_dir, image_root) for row in rows]

    csv_path = output_dir / "failure_case_table.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0].keys()) if table_rows else [])
        if table_rows:
            writer.writeheader()
            writer.writerows(table_rows)

    sample_counts = Counter(str(row.get("sample_type")) for row in rows)
    summary = {
        "artifact_paths": {
            "predictions": str(prediction_path),
            "metrics_per_step": str(metrics_path),
            "dsca_diagnostics": str(diagnostics_path),
            "final_summary": str(final_summary_path),
            "config_resolved": str(config_path),
        },
        "prediction_rows": len(rows),
        "number_missing_predictions": sum(1 for row in table_rows if row["prediction_empty_or_missing"]),
        "number_empty_edited_predictions": sum(1 for row in table_rows if str(row.get("edited_prediction") or "").strip() == ""),
        "number_base_equals_edited": sum(1 for row in table_rows if row["prediction_identical_base_edited"]),
        "number_edited_contains_target": sum(1 for row in table_rows if row["edited_contains_target"]),
        "number_base_contains_target": sum(1 for row in table_rows if row["base_contains_target"]),
        "average_target_length": float(pd.Series([row["target_token_count_estimate"] for row in table_rows]).mean()) if table_rows else None,
        "average_edited_prediction_length": float(
            pd.Series([len(normalize_medical_answer(row.get("edited_prediction")).split()) for row in table_rows]).mean()
        )
        if table_rows
        else None,
        "sample_types_available": dict(sample_counts),
        "sample_types_missing": [name for name in ["rel", "t_gen", "m_gen", "t_loc", "m_loc"] if sample_counts.get(name, 0) == 0],
        "exact_fields_missing_from_predictions_jsonl": missing_fields(rows),
    }
    write_json(output_dir / "failure_summary.json", summary)
    write_examples(output_dir / "failure_examples.md", table_rows)

    print(json.dumps(to_jsonable({"success": True, "output_dir": output_dir, **summary}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
