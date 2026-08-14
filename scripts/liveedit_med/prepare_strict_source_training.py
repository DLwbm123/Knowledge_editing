#!/usr/bin/env python3
"""Freeze the matched strict-source schedule and leakage-safe validation panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.posthoc_validation import freeze_validation_panel
from methods.liveedit_med.source_training_continuation import SourceTrainingContinuationMode


EXPECTED_PANEL = ["1883", "1625", "549", "866", "2036", "1829", "1453", "1064"]
EXPECTED_PANEL_HASH = "b811ff63f17cd579adcbd46ca53427b5d824699dd93e21b7f96dbffcb331ccc8"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def routing_draws(rng: np.random.Generator, count: int):
    names = ("textual", "visual", "paired")
    rows = []
    for _ in range(count):
        first = int(rng.integers(0, 3))
        first_g = names[int(rng.integers(0, 3))]
        second = int(rng.integers(0, 2)) if first != 2 else 2
        second_g = names[int(rng.integers(0, 3))]
        kind = int(rng.integers(0, 2))
        p0_g = names[int(rng.integers(0, 3))]
        p0_variant = int(rng.integers(0, 2))
        p1_g = names[int(rng.integers(0, 3))]
        p1_variant = int(rng.integers(0, 2))
        rows.append({
            "neighbor_first": first, "neighbor_first_generality": first_g,
            "neighbor_second": second, "neighbor_second_generality": second_g,
            "prototype_kind": kind, "prototype_0_generality": p0_g,
            "prototype_0_positive_variant": p0_variant,
            "prototype_1_generality": p1_g, "prototype_1_positive_variant": p1_variant,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    training = args.out_root / "training"
    training.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source_records.read_text())
    cache = json.loads((args.cache_dir / "manifest.json").read_text())
    splits = source["records"]
    counts = {name: len(splits[name]) for name in ("train", "validation", "heldout")}
    if counts != {"train": 512, "validation": 64, "heldout": 64}:
        raise RuntimeError(f"STRICT_SOURCE_SPLIT_COUNT_MISMATCH:{counts}")
    train_ids = [str(row["record_id"]) for row in splits["train"]]
    cache_ids = [str(row["record_id"]) for row in cache["records"]]
    if train_ids != cache_ids or not cache.get("source_order_exact"):
        raise RuntimeError("STRICT_SOURCE_CACHE_ORDER_MISMATCH")
    if "953" in train_ids or "953" in [str(row["record_id"]) for row in splits["validation"]]:
        raise RuntimeError("STRICT_SOURCE_RECORD953_SELECTION_LEAKAGE")

    panel = freeze_validation_panel(source)
    panel_ids = [str(row["record_id"]) for row in panel["edits"]]
    if panel_ids != EXPECTED_PANEL or panel["panel_hash"] != EXPECTED_PANEL_HASH:
        raise RuntimeError(f"STRICT_SOURCE_PANEL_DRIFT:{panel_ids}:{panel['panel_hash']}")
    write_new(training / "validation_panel_manifest.json", panel)

    rng_order = np.random.default_rng(42)
    rng_train = np.random.default_rng(43)
    rng_data = np.random.default_rng(42)
    schedule = []
    step = 0
    for epoch in range(1, 51):
        order = rng_order.permutation(len(cache_ids))
        for begin in range(0, len(cache_ids), 8):
            step += 1
            ids = [cache_ids[int(index)] for index in order[begin : begin + 8]]
            prefixes = [rng_train.integers(0, 9, 3).tolist() for _ in ids]
            routes = routing_draws(rng_data, len(ids))
            if step <= 20:
                schedule.append({
                    "epoch": epoch, "step": step, "record_ids": ids,
                    "prefix_counts": [
                        {"reliability": int(row[0]), "generality": int(row[1]), "locality": int(row[2])}
                        for row in prefixes
                    ],
                    "generality_views": ["textual", "visual", "paired"],
                    "locality_view": "image_or_paired",
                    "routing_draws": routes,
                })
        if step >= 20:
            break
    schedule_artifact = {
        "seed": 42,
        "routing_seed": 42,
        "prefix_seed": 43,
        "order_seed": 42,
        "first_20_batches": schedule,
        "first_20_schedule_hash": canonical_hash(schedule),
        "total_expected_steps": 3200,
    }
    write_new(training / "rng_and_batch_schedule_audit.json", schedule_artifact)
    split_manifest = {
        "counts": counts,
        "train_ids": train_ids,
        "validation_ids": [str(row["record_id"]) for row in splits["validation"]],
        "heldout_ids": [str(row["record_id"]) for row in splits["heldout"]],
        "source_records_sha256": file_sha256(args.source_records),
        "cache_manifest_sha256": file_sha256(args.cache_dir / "manifest.json"),
        "cache_count": len(cache_ids),
        "cache_source_order_exact": True,
        "record953_excluded_from_train_and_selection": True,
        "validation_panel_ids": panel_ids,
        "validation_panel_hash": panel["panel_hash"],
    }
    write_new(training / "split_manifest.json", split_manifest)
    config = {
        "source_training_continuation_mode": SourceTrainingContinuationMode.STRICT_SOURCE_REAPPLY_LAYER21.value,
        "inference_mode": "official_layer21_output_hook",
        "rank": 4, "module_dim": 1024, "edit_layer": 21, "optimizer": "Adam",
        "learning_rate": 1e-4, "batch_size": 8, "epochs": 50, "optimizer_steps": 3200,
        "seed": 42, "checkpoint_steps": [500, 1000, 1500, 2000, 2500, 3000, 3200],
        "router_training": False, "stage2_permitted": False,
    }
    write_new(training / "strict_source_config.json", config)
    (training / "strict_source_config.yaml").write_text("\n".join(f"{key}: {json.dumps(value)}" for key, value in config.items()) + "\n")
    (training / "validation_generation_panel.jsonl").write_text("")
    print(json.dumps({
        "status": "STRICT_SOURCE_TRAINING_PREPARATION_COMPLETE",
        "counts": counts, "panel_hash": panel["panel_hash"],
        "schedule_hash": schedule_artifact["first_20_schedule_hash"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
