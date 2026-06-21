#!/usr/bin/env python3
"""Check DSCA MedMKEB label and multimodal mask integrity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from dsca_medmkeb_diag_common import (
    clone_batch,
    collate_record,
    labels_mask_report,
    load_dataset_and_model,
    to_jsonable,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--model", default="blip2", choices=["blip2"])
    parser.add_argument("--hparams", default="hparams/DSCA/blip2_20edit_pilot.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, model, _alg, tokenizer, _config, dataset_path = load_dataset_and_model(args, num_samples=args.num_samples, load_repo=False)
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    limit = min(args.num_samples, len(dataset))
    for idx in range(limit):
        record = dataset[idx]
        batch = collate_record(dataset, record)
        edit_batch = clone_batch(batch["edit_inner"])
        with torch.no_grad():
            outputs = model(edit_batch)
        report = labels_mask_report(edit_batch, outputs, tokenizer, record.get("target"))
        report.update(
            {
                "index": idx,
                "step": idx + 1,
                "prompt": record.get("prompt"),
                "target": record.get("target"),
                "tokenized_input_length": int(getattr(outputs, "attention_mask", torch.empty(1, 0)).shape[1]),
            }
        )
        if report["target_token_count"] == 0:
            failures.append(f"step {idx + 1}: target_token_count == 0")
        if report["labels_not_ignore_count"] == 0:
            failures.append(f"step {idx + 1}: labels != -100 count == 0")
        if (report.get("answer_mask_sum") or 0) == 0:
            failures.append(f"step {idx + 1}: answer_mask sum == 0")
        if not report.get("labels_align_answer_mask"):
            failures.append(f"step {idx + 1}: labels != -100 does not align with answer_mask tail")
        rows.append(report)

    csv_path = output_dir / "label_mask_check.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "dataset_path": str(dataset_path),
        "num_checked": len(rows),
        "pass": not failures,
        "failures": failures,
        "min_target_token_count": min((row["target_token_count"] for row in rows), default=None),
        "min_labels_not_ignore_count": min((row["labels_not_ignore_count"] for row in rows), default=None),
        "min_answer_mask_sum": min((row.get("answer_mask_sum") or 0 for row in rows), default=None),
        "all_labels_align_answer_mask": all(bool(row.get("labels_align_answer_mask")) for row in rows),
    }
    write_json(output_dir / "label_mask_summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
