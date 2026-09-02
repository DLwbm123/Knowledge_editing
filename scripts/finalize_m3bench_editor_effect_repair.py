#!/usr/bin/env python3
"""Freeze compact effect-smoke locks and public reports without reading raw answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml


METHODS = ("lora", "grace", "balancedit", "belora")
CODE_FILES = (
    "m3bench_repro/editors/llava_runtime.py",
    "m3bench_repro/editors/methods.py",
    "m3bench_repro/editors/routed_layers.py",
    "m3bench_repro/editors/routing.py",
    "scripts/editor_effect_probe.py",
    "scripts/editor_paperspec_runtime.py",
    "scripts/finalize_m3bench_editor_effect_repair.py",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, payload: str, *, frozen: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to replace differing artifact: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    if frozen:
        os.chmod(path, 0o444)


def write_json(path: Path, value: object, *, frozen: bool) -> None:
    write_once(path, json.dumps(value, indent=2, sort_keys=True) + "\n", frozen=frozen)


def git(worktree: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(worktree), *args], text=True).strip()


def basename(value: str | None) -> str | None:
    return Path(value).name if value else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    args = parser.parse_args()

    root, worktree, public = args.run_root, args.worktree, args.public_dir
    index = read_json(args.index)
    commit = git(worktree, "rev-parse", "HEAD")
    branch = git(worktree, "rev-parse", "--abbrev-ref", "HEAD")
    clean_before = not git(worktree, "status", "--porcelain")
    if not clean_before:
        raise RuntimeError("worktree must be clean before finalization")

    method_configs, one_edit, smoke_8, stream_4 = {}, {}, {}, {}
    for method in METHODS:
        stream_dir = root / index["stream_4"][method]
        config_path = stream_dir / "method_lock/METHOD_CONFIG_LOCK_V2.yaml"
        method_configs[method] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        one = read_json(root / index["one_edit"][method])
        one_edit[method] = {
            "status": one["status"],
            "all_checks_pass": all(one["checks"].values()),
            "pre_target_nll": one["pre_target_nll"],
            "post_target_nll": one["post_target_nll"],
            "target_logprob_delta": one["target_logprob_delta"],
            "state_delta_norm": one["state_delta_norm"],
            "self_route_hit": one.get("self_route_hit"),
            "post_generation_nonempty": one["post_generation_nonempty"],
            "base_weights_unchanged": one["base_weights_unchanged"],
        }

        smoke = read_json(root / index["smoke_8"][method] / "SMOKE_8_SUMMARY_PREJUDGE.json")
        smoke_8[method] = {
            key: smoke[key]
            for key in (
                "status", "raw_output_changed_count", "median_target_nll_decrease",
                "route_hit_count", "empty_output_count", "runtime_seconds",
            )
        }
        smoke_8[method].update({
            "denominator": 8,
            "t0_corrected_count": index["t0_corrected_count"][method],
            "effect_noop": index["t0_corrected_count"][method] == 0
            and smoke["raw_output_changed_count"] == 0,
        })

        run, replay = read_json(stream_dir / "stream_run.json"), read_json(stream_dir / "stream_replay.json")
        final_state = run["steps"][-1]["state_summary"]
        stream_4[method] = {
            "run_status": run["status"],
            "step_count": len(run["steps"]),
            "all_step_checks_pass": all(value for step in run["steps"] for value in step["checks"].values()),
            "runtime_seconds": run["runtime_seconds"],
            "replay_status": replay["status"],
            "replay_checks": replay["checks"],
            "final_state_count": int(final_state.get("entry_count", len(final_state.get("edit_history", [])))),
        }
        if method == "belora":
            stream_4[method].update({
                "unique_logical_ids": len(final_state["edit_to_adapter"]),
                "unique_adapter_names": len(set(final_state["edit_to_adapter"].values())),
                "adapter_hash_count": len(final_state["adapter_hashes"]),
            })

    all_pass = (
        all(row["status"] == "PASS" and row["all_checks_pass"] for row in one_edit.values())
        and all(row["status"] == "PASS" and not row["effect_noop"] for row in smoke_8.values())
        and all(row["run_status"] == row["replay_status"] == "PASS" and row["all_step_checks_pass"] for row in stream_4.values())
    )
    if not all_pass:
        raise RuntimeError("effect smoke is not closed")

    inventory = read_json(root / "runtime/LLAVA_MED_MODULE_INVENTORY.json")
    target_lock = read_json(root / "runtime/LLAVA_MED_EDIT_TARGET_LOCK.json")
    generation = read_json(root / "carry_forward/foundation_v4/runtime/llava_med_generation_frozen.json")
    locks = {
        "EFFECT_REPAIRED_METHOD_CONFIG_BUNDLE.json": {
            "schema_version": "m3bench-effect-repaired-method-config-bundle-v1",
            "classification": index["classification"], "source_commit": commit,
            "method_configs": method_configs,
        },
        "EFFECT_REPAIRED_MODEL_RUNTIME_LOCK.json": {
            "schema_version": "m3bench-effect-repaired-model-runtime-lock-v1",
            "classification": index["classification"], "source_commit": commit,
            "model_class": inventory["model_class"], "model_dtype": inventory["model_dtype"],
            "model_id": basename(inventory.get("model_path")),
            "vision_tower_id": basename(inventory.get("vision_tower_cache")),
            "language_block_count": inventory["language_block_count"],
            "total_model_parameters": inventory["total_model_parameters"],
        },
        "EFFECT_REPAIRED_TARGET_MODULE_LOCK.json": {
            **target_lock, "classification": index["classification"], "source_commit": commit,
        },
        "EFFECT_REPAIRED_GENERATION_LOCK.json": {
            **{key: value for key, value in generation.items() if key not in {"model_path", "vision_tower_path"}},
            "model_id": basename(generation.get("model_path")),
            "vision_tower_id": basename(generation.get("vision_tower_path")),
            "classification": index["classification"], "source_commit": commit,
        },
        "EFFECT_REPAIRED_SOURCE_MANIFEST.json": {
            "schema_version": "m3bench-effect-repaired-source-manifest-v1",
            "classification": index["classification"], "branch": branch, "commit": commit,
            "worktree_clean_before_finalization": clean_before,
            "code_sha256": {path: sha256(worktree / path) for path in CODE_FILES},
        },
        "EFFECT_REPAIRED_SMOKE_RESULTS.json": {
            "schema_version": "m3bench-effect-repaired-smoke-results-v1",
            "status": "M3BENCH_EDITOR_EFFECT_SMOKE_PASS_V2",
            "classification": index["classification"], "source_commit": commit,
            "cohort_sha256": index["cohort_sha256"],
            "one_edit": one_edit, "smoke_8": smoke_8, "stream_4": stream_4,
            "formal_experiment": False,
        },
    }
    for name, value in locks.items():
        write_json(root / "locks" / name, value, frozen=True)
        write_json(public / "locks" / name, value, frozen=False)

    authority = {
        "schema_version": "m3bench-current-formal-sequence-authority-v1",
        "source_branch": index["authority"]["source_branch"],
        "source_commit": index["authority"]["source_commit"],
        "superseded_reports": [
            {"decision": "EARLY_TEXT_ONLY_POSITIONS_19_57_67_INVALID", "status": "SUPERSEDED_BY_REVIEWER_A_IMAGE_AWARE_AUDIT"},
            {"decision": "REVIEWER_B_REQUIRED_GATE", "status": "SUPERSEDED_BY_APPROVED_SINGLE_REVIEWER_PROTOCOL"},
        ],
        "active_decision": "C",
        "active_sequence": {"path": None, "sha256": None, "count": 0, "status": "NO_OPERATOR_APPROVED_AMENDMENT"},
        "original_sequence_candidate": {
            "path": "private metadata-only formal selection",
            "sha256": index["authority"]["original_sequence_sha256"],
            "count": index["authority"]["original_sequence_count"],
            "reconstruction_status": index["authority"]["reconstruction_status"],
            "authorized_for_effect_repaired_formal_rerun": False,
        },
        "reviewer_a_counts": index["authority"]["reviewer_a_counts"],
        "position_handling": index["authority"]["position_handling"],
        "recommended_amended_sequence_count": index["authority"]["recommended_amended_sequence_count"],
        "operator_selected_option": index["authority"]["operator_selected_option"],
        "blocker": "M3BENCH_EFFECT_REPAIR_BLOCKED__FORMAL_SEQUENCE_UNRESOLVED",
    }
    write_json(root / "reports/CURRENT_FORMAL_SEQUENCE_AUTHORITY.json", authority, frozen=True)
    write_json(public / "CURRENT_FORMAL_SEQUENCE_AUTHORITY.json", authority, frozen=False)
    authority_md = (
        "# Current Formal Sequence Authority\n\n"
        "Decision **C**: no operator-approved amended sequence exists. The exact reconstructed original sequence "
        f"contains {authority['original_sequence_candidate']['count']} targets and has SHA-256 "
        f"`{authority['original_sequence_candidate']['sha256']}`, but it is not authorized for the repaired formal rerun. "
        f"Reviewer A counts are 189 VALID, 10 CONFIRMED_INVALID, and 1 UNRESOLVED; the only supported proposal is length "
        f"{authority['recommended_amended_sequence_count']}, still unselected. Position 19 is confirmed invalid; positions 57 "
        "and 67 are valid under the authoritative image-aware review.\n\n"
        "Formal single, sequential, raw closure, Judge, evaluator, and scoring were not started.\n"
    )
    write_once(root / "reports/CURRENT_FORMAL_SEQUENCE_AUTHORITY.md", authority_md, frozen=True)
    write_once(public / "CURRENT_FORMAL_SEQUENCE_AUTHORITY.md", authority_md, frozen=False)

    summary = {
        "status": "M3BENCH_EDITOR_EFFECT_REPAIR_PASS__FORMAL_RERUN_BLOCKED_BY_GOVERNANCE",
        "effect_smoke_status": "M3BENCH_EDITOR_EFFECT_SMOKE_PASS_V2",
        "classification": index["classification"], "source_branch": branch, "source_commit": commit,
        "one_edit": one_edit, "smoke_8": smoke_8, "stream_4": stream_4,
        "formal_sequence_authority": "C", "formal_single_started": False,
        "formal_sequential_started": False, "raw_closure": None, "judge_started": False,
        "evaluator_started": False, "semantic_metrics_started": False,
        "remaining_blocker": authority["blocker"],
    }
    write_json(public / "EDITOR_EFFECT_REPAIR_SUMMARY.json", summary, frozen=False)
    table = ["| Method | T0/8 | Raw changed | Median NLL decrease | Route hit | Empty | 4-edit run | Fresh replay |", "|---|---:|---:|---:|---:|---:|---|---|"]
    labels = {"lora": "LoRA", "grace": "GRACE", "balancedit": "BalanceEdit", "belora": "BELoRA"}
    for method in METHODS:
        smoke, stream = smoke_8[method], stream_4[method]
        table.append(
            f"| {labels[method]} | {smoke['t0_corrected_count']}/8 | {smoke['raw_output_changed_count']}/8 | "
            f"{smoke['median_target_nll_decrease']:.4f} | {smoke['route_hit_count']}/8 | {smoke['empty_output_count']} | "
            f"{stream['run_status']} | {stream['replay_status']} |"
        )
    summary_md = (
        "# M3Bench Editor Effect Repair V2\n\n"
        "Status: `M3BENCH_EDITOR_EFFECT_REPAIR_PASS__FORMAL_RERUN_BLOCKED_BY_GOVERNANCE`.\n\n"
        + "\n".join(table)
        + "\n\nBELoRA is a paper-spec independent reimplementation; 50 steps/edit is an explicit deviation from the "
        "5-step paper-spec setting because 5–20 steps were a generation no-op on the approved smoke cohort. "
        "The first tested checkpoint that changed generation was 50.\n\n"
        "All one-edit and 4-edit effect checks passed, all outputs were nonempty, and base weights remained unchanged. "
        "Formal-200 and scoring were not started because the current sequence authority is C.\n"
    )
    write_once(public / "EDITOR_EFFECT_REPAIR_SUMMARY.md", summary_md, frozen=False)
    marker = "M3BENCH_EDITOR_EFFECT_SMOKE_PASS_V2\n"
    write_once(root / "gates/M3BENCH_EDITOR_EFFECT_SMOKE_PASS_V2", marker, frozen=True)
    write_once(public / "M3BENCH_EDITOR_EFFECT_SMOKE_PASS_V2", marker, frozen=False)
    print(json.dumps({"status": summary["status"], "source_commit": commit, "public_dir": str(public)}, indent=2))


if __name__ == "__main__":
    main()
