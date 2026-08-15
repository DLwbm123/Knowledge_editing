#!/usr/bin/env python3
"""Mine clean-S0 hard negatives and freeze router-R1 repositories."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.router_r1 import (
    NEGATIVE_CATEGORIES,
    PROTOCOL,
    deterministic_repository,
    repository_size,
    stable_key,
)


def normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


def pooled(tensors: dict[str, torch.Tensor], mask_name: str) -> torch.Tensor:
    hidden = tensors["native__hidden"].float()
    mask = tensors[f"native__{mask_name}"].bool()
    value = hidden[mask].mean(0)
    return torch.nn.functional.normalize(value, dim=0)


def rank(scores: torch.Tensor, target_index: int, ids: list[str]) -> list[str]:
    values = []
    for index, rid in enumerate(ids):
        if index != target_index:
            values.append((-float(scores[index]), stable_key(ids[target_index], rid), rid))
    values.sort()
    return [row[2] for row in values]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--representation-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    nearest_path = args.out_dir / "nearest_neighbor_audit.json"
    if nearest_path.exists():
        raise FileExistsError(nearest_path)
    source = json.loads(args.source_records.read_text())["records"]
    manifest = json.loads(args.representation_manifest.read_text())
    nearest: dict[str, dict[str, Any]] = {}
    hard_rows: list[dict[str, Any]] = []
    eq_rows: list[dict[str, Any]] = []
    source_by_split = {split: {str(row["record_id"]): row for row in rows}
                       for split, rows in source.items()}
    for split in ("train", "validation", "heldout"):
        entries = manifest["splits"][split]
        ids = [str(row["record_id"]) for row in entries]
        tensors = [load_file(row["file_path"], device="cpu") for row in entries]
        visual = torch.stack([pooled(value, "vision_mask") for value in tensors])
        text = torch.stack([pooled(value, "question_mask") for value in tensors])
        visual_scores, text_scores = visual @ visual.T, text @ text.T
        joint_scores = .5 * visual_scores + .5 * text_scores
        for index, rid in enumerate(ids):
            ranks = {"visual": rank(visual_scores[index], index, ids),
                     "text": rank(text_scores[index], index, ids),
                     "joint": rank(joint_scores[index], index, ids)}
            row = source_by_split[split][rid]
            target = normalize(row["requests"][0]["target_new"])
            question = normalize(row["requests"][0]["prompt"])
            image = str(row["requests"][0]["image"])
            def valid(other_id: str, *, need_question: bool = False, need_image: bool = False) -> bool:
                other = source_by_split[split][other_id]
                return (normalize(other["requests"][0]["target_new"]) != target
                        and (not need_question or normalize(other["requests"][0]["prompt"]) != question)
                        and (not need_image or str(other["requests"][0]["image"]) != image))
            filtered = {name: [other for other in values if valid(other)] for name, values in ranks.items()}
            same_image = next(other for other in ranks["text"] if valid(other, need_question=True))
            same_question = next(other for other in ranks["visual"] if valid(other, need_image=True))
            chosen = {
                "same_image_different_question": same_image,
                "same_question_different_image": same_question,
                "visual_nearest": filtered["visual"][0],
                "text_nearest": filtered["text"][0],
                "joint_near_miss": filtered["joint"][0],
            }
            nearest[rid] = {"split": split, "visual": filtered["visual"],
                            "text": filtered["text"], "joint": filtered["joint"], "chosen": chosen}
            native_eq = next(item["eqkey"] for item in entries[index]["inputs"] if item["category"] == "native")
            eq_rows.append({"split": split, "record_id": rid, "category": "native_positive",
                            "eqkey": native_eq, "disposition": "KEPT_POSITIVE"})
            for category, other_id in chosen.items():
                hard_rows.append({"split": split, "record_id": rid, "category": category,
                                  "other_record_id": other_id, "target_distinct": True,
                                  "clean_s0_only": True, "eqkey": "PENDING_CROSS_INPUT_CACHE",
                                  "provenance": f"{category}:{other_id}"})
            for locality in ("image_locality", "text_locality"):
                item = next((item for item in entries[index]["inputs"] if item["category"] == locality), None)
                if item is not None:
                    hard_rows.append({"split": split, "record_id": rid, "category": locality,
                                      "other_record_id": "", "target_distinct": True,
                                      "clean_s0_only": True, "eqkey": item["eqkey"],
                                      "provenance": "source_locality"})

    nearest_path.write_text(json.dumps({"protocol": PROTOCOL, "clean_s0_only": True,
        "target_answers_used_as_features": False, "edited_behavior_used": False,
        "record953_used": False, "heldout_outcomes_used": False, "blind_used": False,
        "neighbors": nearest}, indent=2, sort_keys=True) + "\n")
    write_csv(args.out_dir / "hard_negative_ledger.csv", hard_rows,
              ["split", "record_id", "category", "other_record_id", "target_distinct",
               "clean_s0_only", "eqkey", "provenance"])
    write_csv(args.out_dir / "eqkey_exclusion_ledger.csv", eq_rows,
              ["split", "record_id", "category", "eqkey", "disposition"])

    membership = args.out_dir / "repository_membership_training.jsonl"
    all_train_ids = [str(row["record_id"]) for row in manifest["splits"]["train"]]
    rng = np.random.default_rng(42)
    step = 0
    with membership.open("x") as handle:
        for epoch in range(1, 11):
            order = rng.permutation(len(all_train_ids))
            for begin in range(0, len(order), 8):
                step += 1
                size = repository_size(step)
                for index in order[begin:begin + 8]:
                    rid = all_train_ids[int(index)]
                    repo = deterministic_repository(rid, size, nearest, all_train_ids)
                    handle.write(json.dumps({"split": "train", "epoch": epoch, "step": step,
                        "record_id": rid, "repository_size": size, "repository_ids": repo}, sort_keys=True) + "\n")
        validation_ids = [str(row["record_id"]) for row in manifest["splits"]["validation"]]
        for rid in validation_ids:
            for size in (1, 10, 32):
                # Size 10 is evaluation-only; deterministic_repository accepts
                # source cycle sizes, so construct it through the same ordering.
                if size == 10:
                    ordered = deterministic_repository(rid, 16, nearest, validation_ids)[:10]
                else:
                    ordered = deterministic_repository(rid, size, nearest, validation_ids)
                handle.write(json.dumps({"split": "validation", "record_id": rid,
                    "repository_size": size, "repository_ids": ordered}, sort_keys=True) + "\n")
    if step != 640:
        raise RuntimeError(f"ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:steps:{step}")
    print(json.dumps({"status": "ROUTER_R1_DATA_PREPARED", "steps": step,
                      "hard_negative_rows": len(hard_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
