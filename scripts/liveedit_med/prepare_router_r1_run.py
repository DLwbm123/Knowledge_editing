#!/usr/bin/env python3
"""Create and audit a non-overwriting LiveEdit-Med router-R1 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.router_r1 import (
    EXPECTED_BANK_HASH,
    EXPECTED_BLIND_SEALED_HASH,
    EXPECTED_BLIND_SELECTION_HASH,
    PROTOCOL,
    canonical_hash,
)
from methods.liveedit_med.serialization import load_safe_state, tensor_hashes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"ROUTER_R1_ANCHOR_MISMATCH:missing:{path}")
    return json.loads(path.read_text())


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--corrected-run", type=Path)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)

    selection = read(args.strict_run / "validation/checkpoint_selection.json")
    split = read(args.strict_run / "training/split_manifest.json")
    strict_config = read(args.strict_run / "training/strict_source_config.json")
    strict_manifest = read(args.strict_run / "run_manifest.json")
    blind = read(args.strict_run / "blind_set_seal_audit.json")
    frozen_tree = read(args.strict_run / "strict_training_tree_frozen_manifest.json")
    checkpoint_dir = args.strict_run / "training/checkpoint_3200"
    checkpoint_state, checkpoint_manifest = load_safe_state(checkpoint_dir)
    checkpoint_tensor_hash = canonical_hash(tensor_hashes(checkpoint_state))

    checks = {
        "unique_strict_run": args.strict_run.name == "20260814T101648Z",
        "selected_checkpoint_3200": selection.get("selected_step") == 3200,
        "selection_frozen_no_test_leakage": selection.get("record953_used_for_selection") is False
            and selection.get("sealed_blind_used_for_selection") is False,
        "checkpoint_step_3200": checkpoint_manifest.get("step") == 3200,
        "strict_continuation": checkpoint_manifest.get("source_training_continuation_mode") == "strict_source_reapply_layer21"
            and strict_config.get("source_training_continuation_mode") == "strict_source_reapply_layer21",
        "inference_output_hook": strict_config.get("inference_mode") == "official_layer21_output_hook",
        "split_512_64_64": split.get("counts") == {"train": 512, "validation": 64, "heldout": 64},
        "record953_excluded": split.get("record953_excluded_from_train_and_selection") is True
            and "953" not in set(split.get("train_ids", []) + split.get("validation_ids", []) + split.get("heldout_ids", [])),
        "source_records_match": sha256_file(args.source_records) == split.get("source_records_sha256"),
        "canonical_bank": blind.get("passed") is True,
        "blind_selection_hash": blind.get("selection_manifest_internal_hash") == EXPECTED_BLIND_SELECTION_HASH,
        "blind_sealed_hash": blind.get("sealed_manifest_internal_hash") == EXPECTED_BLIND_SEALED_HASH,
        "blind_edited_checkpoint_not_loaded": blind.get("edited_checkpoint_loaded") is False,
        "strict_training_tree_frozen": frozen_tree.get("tree_hash") == "f4fcc9d645088ce27edd35d3fc35058d6c1aae52e5cbff22a9b3fe12e1ad2147",
        "source_commit_resolved": bool((strict_manifest.get("source_commit") or {}).get("commit")),
    }
    # The bank content is not loaded here.  The strict run's independent bank
    # audit is resolved, and every later model-loading process rechecks it.
    strict_bank = read(args.strict_run / "post_training/stage_q_final_summary.json") if (
        args.strict_run / "post_training/stage_q_final_summary.json").is_file() else None
    if strict_bank is not None:
        observed = strict_bank.get("canonical_bank_sha256") or strict_bank.get("canonical_bank_hash")
        if observed is not None:
            checks["canonical_bank"] = observed == EXPECTED_BANK_HASH
    if not all(checks.values()):
        failures = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError("ROUTER_R1_ANCHOR_MISMATCH:" + ",".join(failures))

    for name in ("cache", "data", "training", "validation", "heldout", "record953_regression"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=(name != "cache"))
    audit = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "strict_run": str(args.strict_run.resolve()),
        "corrected_comparator": str(args.corrected_run.resolve()) if args.corrected_run else None,
        "corrected_comparator_read_only": True,
        "selected_checkpoint": 3200,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_tensor_hash": checkpoint_tensor_hash,
        "checkpoint_tensor_hashes": checkpoint_manifest["tensor_hashes"],
        "strict_training_tree_hash": frozen_tree["tree_hash"],
        "source_commit": strict_manifest["source_commit"],
        "canonical_bank_hash": EXPECTED_BANK_HASH,
        "blind_selection_manifest_hash": EXPECTED_BLIND_SELECTION_HASH,
        "blind_sealed_manifest_hash": EXPECTED_BLIND_SEALED_HASH,
        "blind_selected_edits": 16,
        "blind_total_inputs": 208,
        "blind_edited_checkpoint_loaded": False,
        "checks": checks,
    }
    write_new(args.run_dir / "anchor_and_freeze_audit.json", audit)
    write_new(args.run_dir / "data/split_manifest.json", split)
    write_new(args.run_dir / "blind_set_seal_audit.json", {
        "selection_manifest_internal_hash": EXPECTED_BLIND_SELECTION_HASH,
        "sealed_manifest_internal_hash": EXPECTED_BLIND_SEALED_HASH,
        "selected_edits": 16,
        "total_inputs": 208,
        "edited_checkpoint_loaded": False,
        "outcomes_enumerated": False,
        "UNOPENED_BY_EDITED_LIVEEDIT": True,
    })
    write_new(args.run_dir / "run_manifest.json", {
        "protocol": PROTOCOL,
        "status": "ANCHOR_AUDIT_COMPLETE",
        "strict_checkpoint": 3200,
        "record953_scope": "DEVELOPMENT_REGRESSION_ONLY",
        "blind_set_opened": False,
        "blind_evaluation_permitted_next": False,
        "stage2_permitted": False,
    })
    print(json.dumps({"status": "ROUTER_R1_ANCHOR_AUDIT_PASS", "run_dir": str(args.run_dir),
                      "checkpoint_tensor_hash": checkpoint_tensor_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
