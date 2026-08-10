#!/usr/bin/env python3
"""ENGRAM V2 Stage-1 behavioral-margin one-edit gate for fixed record 953.

The runner is deliberately fail-closed.  It performs the complete real-model
preflight and natural-answer-boundary gate before any shadow optimization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import to_jsonable  # noqa: E402
from scripts.engram.run_engram_v2_stage0_generation_audit import (  # noqa: E402
    BANK_ROOT,
    MODEL_CONFIG,
    MODULE_KEY,
    MODULE_NAME,
    apply_prefix,
    bank_manifest,
    eos_ids,
    load_model_views_bank,
    state_weight_hash,
)
from scripts.engram.run_engram_v2_stage0abc_diagnostics import (  # noqa: E402
    SHORT_INSTRUCTION,
    hf_cached_greedy_trace,
    short_answer_sample,
)
from scripts.engram.stage0_generation_audit_utils import (  # noqa: E402
    build_canonical_inputs,
    ids_sha256,
    incremental_mean_nll,
    manual_cached_greedy_trace,
    manual_greedy_trace,
    medical_answer_match,
    normalize_medical_answer,
    score_target_incrementally,
)
from scripts.engram.stage0abc_diagnostic_utils import create_new_output_dir  # noqa: E402
from scripts.engram.stage1_behavioral_margin_utils import (  # noqa: E402
    NaturalAnswerSpanError,
    align_unique_answer_span,
    assert_bank_immutable,
    assert_generation_inputs_target_free,
    assert_three_path_parity,
    tensor_l2,
)


STARTING_COMMIT = "a70a02236306e7bcec9350fb4599da8dde576dcf"
RECORD_ID = "953"
CAP = 128
PROTOCOL = "ENGRAM_V2_STAGE1_BEHAVIORAL_MARGIN_ONE_EDIT_V1"
STAGE0ABC = ROOT / "outputs/engram_v2_stage0_generation_audit/20260810_stage0abc_margin_feasibility_v1"
STAGE0_MATRIX = STAGE0ABC / "fixed_matrix_uniform_128"
REQUIRED_ARTIFACTS = (
    "STAGE1_BEHAVIORAL_MARGIN_REPORT.md",
    "stage1_summary.json",
    "stage1_checkpoint_trajectory.jsonl",
    "stage1_comparison_table.csv",
    "stage1_target_token_trajectory.jsonl",
    "stage1_locality_ledger.jsonl",
    "stage1_state_hash_ledger.jsonl",
    "stage1_bank_replay_report.json",
    "stage1_fresh_process_report.json",
    "stage1_rollback_report.json",
    "run_manifest.json",
    "commands.log",
    "source_config_diff.txt",
    "STAGE1_NEXT_DECISION.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--physical-gpu", default=2, type=int)
    parser.add_argument("--starting-commit", default=STARTING_COMMIT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(to_jsonable(dict(value)), sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    with path.open("x") as handle:
        handle.write(value.rstrip() + "\n")


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_manifest() -> Dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/engram/stage1_behavioral_margin_utils.py",
        ROOT / "tests/test_engram_v2_stage1_behavioral_margin.py",
        ROOT / "scripts/engram/run_engram_v2_stage0_generation_audit.py",
        ROOT / "scripts/engram/run_engram_v2_stage0abc_diagnostics.py",
        ROOT / "scripts/engram/stage0_generation_audit_utils.py",
        ROOT / "scripts/engram/stage0abc_diagnostic_utils.py",
        MODEL_CONFIG,
        ROOT / "easyeditor/models/engram_v2/bank.py",
        ROOT / "easyeditor/models/engram_v2/editor.py",
        ROOT / "easyeditor/models/engram_v2/solver.py",
    ]
    rows = [{"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]
    return {"files": rows, "sha256": canonical_hash(rows)}


def expected_cell(record_id: str, state_id: str, view: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    record = next(
        row for row in read_jsonl(STAGE0_MATRIX / "records.jsonl")
        if row["record_id"] == record_id and row["state_id"] == state_id and row["view"] == view
    )
    generation = next(
        row for row in read_jsonl(STAGE0_MATRIX / "generation_outputs.jsonl")
        if row["record_id"] == record_id and row["state_id"] == state_id and row["view"] == view
    )
    return record, generation


def evaluate_preflight_cell(model: Any, sample: Mapping[str, Any], *, state_id: str, view: str) -> Dict[str, Any]:
    canonical = build_canonical_inputs(model, sample)
    no_cache = manual_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=5)
    cached = manual_cached_greedy_trace(model, canonical, CAP, eos_ids(model), top_k=5)
    hf = hf_cached_greedy_trace(model, canonical, CAP)
    assert_three_path_parity(no_cache["token_ids"], cached["token_ids"], hf["token_ids"])
    if any(bool(item["cap_hit"]) for item in (no_cache, cached, hf)):
        raise RuntimeError(f"{RECORD_ID}:{state_id}:{view}:uniform_128_cap_hit")
    target_rows = score_target_incrementally(model, canonical, eos_ids(model), top_k=5)
    expected_record, expected_generation = expected_cell(RECORD_ID, state_id, view)
    checks = {
        "prompt_hash": canonical.prompt_hash == expected_record["prompt_hash"],
        "pixel_hash": canonical.pixel_hash == expected_record["pixel_hash"],
        "first_target_rank": int(target_rows[0]["target_rank"]) == int(expected_record["first_target_rank"]),
        "first_target_margin": abs(float(target_rows[0]["margin"]) - float(expected_record["first_target_margin"])) <= 5e-4,
        "manual_token_ids": no_cache["token_ids"] == expected_generation["manual"]["token_ids"],
        "manual_output": no_cache["raw_output"] == expected_record["greedy_output"],
        "path_parity": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage-0 deterministic reconstruction mismatch at {state_id}:{view}: {checks}")
    return {
        "record_id": RECORD_ID,
        "state_id": state_id,
        "view": view,
        "prompt_hash": canonical.prompt_hash,
        "pixel_hash": canonical.pixel_hash,
        "full_hash": canonical.full_hash,
        "answer_start": canonical.answer_start,
        "target_ids": canonical.target_ids.detach().cpu().tolist(),
        "target_rows": target_rows,
        "target_nll": incremental_mean_nll(target_rows),
        "no_cache": no_cache,
        "cached": cached,
        "hf": hf,
        "checks": checks,
        "canonical": canonical,
    }


def serializable_cell(cell: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in cell.items() if key != "canonical"}


def tokenizer_hash(model: Any) -> str:
    vocab = sorted((str(key), int(value)) for key, value in model.llava_tokenizer.get_vocab().items())
    return canonical_hash({"vocab": vocab, "special_tokens": model.llava_tokenizer.special_tokens_map})


def write_failure_artifacts(
    out_dir: Path,
    *,
    hard_stop: NaturalAnswerSpanError,
    manifest: Dict[str, Any],
    s0_target: Mapping[str, Any],
    s1_target: Mapping[str, Any],
    s0_locality: Mapping[str, Any],
    s1_locality: Mapping[str, Any],
    b1_norm: float,
    b1_relative: float,
    rollback: Mapping[str, Any],
) -> None:
    summary = {
        "protocol": PROTOCOL,
        "record_id": RECORD_ID,
        "primary_final_label": "INVALID_ENGINEERING_RUN",
        "hard_stop_label": hard_stop.label,
        "hard_stop_message": str(hard_stop),
        "preflight_passed_before_natural_boundary": True,
        "natural_answer_boundary_passed": False,
        "optimization_started": False,
        "candidate_bank_created": False,
        "unrestricted_generation_succeeded": False,
        "success_budget": None,
        "stage2_permitted": False,
        "uniform_cap": CAP,
        "b1_norm": b1_norm,
        "b1_relative_parameter_displacement": b1_relative,
        "original_bank_unchanged": rollback["bank_unchanged"],
        "rollback_passed": rollback["passed"],
    }
    write_json(out_dir / "stage1_summary.json", summary)
    append_jsonl(out_dir / "stage1_checkpoint_trajectory.jsonl", {
        "checkpoint": "preflight_hard_stop",
        "budget": None,
        "optimization_started": False,
        "hard_stop_label": hard_stop.label,
    })
    for cell in (s0_target, s1_target):
        for token in cell["target_rows"]:
            append_jsonl(out_dir / "stage1_target_token_trajectory.jsonl", {
                "record_id": RECORD_ID,
                "state_id": cell["state_id"],
                "view": "unrestricted",
                **token,
            })
    for cell in (s0_locality, s1_locality):
        append_jsonl(out_dir / "stage1_locality_ledger.jsonl", {
            "record_id": RECORD_ID,
            "state_id": cell["state_id"],
            "view": "paired_locality_preflight",
            "token_ids": cell["no_cache"]["token_ids"],
            "output": cell["no_cache"]["raw_output"],
            "stop_reason": cell["no_cache"]["stop_reason"],
            "cap_hit": cell["no_cache"]["cap_hit"],
            "path_parity": True,
            "stage0_equal": all(cell["checks"].values()),
        })
    write_json(out_dir / "stage1_bank_replay_report.json", {
        "status": "NOT_RUN",
        "reason": hard_stop.label,
        "candidate_bank_created": False,
        "original_bank_unchanged": rollback["bank_unchanged"],
    })
    write_json(out_dir / "stage1_fresh_process_report.json", {
        "status": "NOT_RUN",
        "reason": hard_stop.label,
        "candidate_did_not_reach_commit_gate": True,
    })
    write_json(out_dir / "stage1_rollback_report.json", dict(rollback))

    stage0b = json.loads((STAGE0ABC / "stage0b_target_surface_audit.json").read_text())
    surface = {(row["state_id"], row["view"]): row for row in stage0b["rows"] if row["record_id"] == RECORD_ID}
    rows = []
    for label, cell, delta_norm, relative in (
        ("S0 base", s0_target, 0.0, 0.0),
        ("existing NLL-only S1", s1_target, b1_norm, b1_relative),
    ):
        short = surface[(cell["state_id"], "fixed_short_answer")]
        rows.append({
            "candidate": label,
            "delta_norm": delta_norm,
            "relative_parameter_displacement": relative,
            "target_nll": cell["target_nll"],
            "natural_boundary_target_rank": "NOT_AVAILABLE",
            "natural_boundary_target_margin": "NOT_AVAILABLE",
            "target_vs_old_margin": "NOT_AVAILABLE",
            "short_answer_rank": short["first_target_rank"],
            "short_answer_margin": short["first_target_margin"],
            "unrestricted_output": cell["no_cache"]["raw_output"],
            "short_answer_output": short["greedy_output"],
            "locality_damage_count": 0,
            "maximum_locality_nll_drift": "NOT_EVALUATED_CANDIDATE_ABSENT",
            "replay_reload_fresh_rollback": "preflight rollback PASS; candidate gates NOT RUN",
        })
    fields = list(rows[0])
    with (out_dir / "stage1_comparison_table.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = f"""# ENGRAM V2 Stage-1 Behavioral-Margin One-Edit Gate

**Primary label:** `INVALID_ENGINEERING_RUN`

**Hard stop:** `{hard_stop.label}`

The deterministic S0 unrestricted response does not contain the dataset-provided old/reference answer, so a unique natural answer boundary cannot be constructed without manually rewriting the response. The specification requires an immediate stop before optimization.

- Record: `{RECORD_ID}`
- Uniform generation cap: `{CAP}`
- S0/S1 target and paired-locality reconstruction: `PASS`
- Cached/no-cache/HF parity: `PASS`
- Original bank unchanged: `{rollback['bank_unchanged']}`
- Exact rollback: `{rollback['passed']}`
- Shadow optimization started: `False`
- Candidate bank created: `False`
- Stage-2 permitted: `False`
"""
    write_text(out_dir / "STAGE1_BEHAVIORAL_MARGIN_REPORT.md", report)
    decision = f"""# Stage-1 Next Decision

## Verified facts

- Stage-0 starting commit: `{STARTING_COMMIT}`.
- Record 953 S0/S1 and paired locality reproduce the Stage-0 token trajectories and margins.
- All cached/no-cache/HF trajectories match and stop normally within 128 tokens.
- The dataset old/reference answer is absent from the unrestricted S0 response.
- The original bank is unchanged and rollback passed.

## Diagnostic inferences

- The requested natural answer-boundary objective is not defined for this fixed edit under the prescribed construction.
- Continuing would require manually inventing or rewriting a response scaffold, which is forbidden.

## Primary decision label

- `INVALID_ENGINEERING_RUN`
- Hard-stop subtype: `{hard_stop.label}`

## Gates

- Unrestricted generation success: `False`
- Success by B1 or Bmax: `None`; optimization was not authorized to start.
- Locality status: preflight reconstruction passed; candidate locality gate not run.
- Replay/reload/fresh-process: candidate gates not run because no candidate exists.
- Rollback: `PASS`
- Exact S1 displacement norm B1: `{b1_norm:.12f}`
- Exact S1 relative displacement: `{b1_relative:.12f}`
- Stage-2 permitted: `False`
"""
    write_text(out_dir / "STAGE1_NEXT_DECISION.md", decision)
    manifest["summary"] = summary


def main() -> None:
    args = parse_args()
    if args.starting_commit != STARTING_COMMIT:
        raise RuntimeError(f"Required starting commit is {STARTING_COMMIT}, got {args.starting_commit}")
    out_dir = create_new_output_dir(args.out_dir)
    write_text(out_dir / "commands.log", " ".join(sys.argv))
    bank_before = bank_manifest()
    manifest: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "starting_commit": STARTING_COMMIT,
        "cwd": str(ROOT),
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "uniform_max_new_tokens": CAP,
        "record_ids_used_for_optimization": [RECORD_ID],
        "stage2_launched": False,
        "ten_edit_run_launched": False,
        "legacy_lora_or_cure_used": False,
        "original_bank_manifest": bank_before,
    }

    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    manifest["sources"] = source_manifest()
    manifest["model_and_input_identity"] = {
        "model_class": type(model).__qualname__,
        "tokenizer_class": type(model.llava_tokenizer).__qualname__,
        "tokenizer_hash": tokenizer_hash(model),
        "processor_class": type(getattr(model, "vis_processor", None)).__qualname__,
        "config_sha256": sha256_file(MODEL_CONFIG),
    }
    record = records[RECORD_ID]
    target = str(record["alt"])
    old_answer = record.get("pred")
    if old_answer is None or not str(old_answer).strip():
        raise RuntimeError("Dataset-provided old/reference answer is unavailable for record 953")

    apply_prefix(model, bank, 0)
    s0_hash = state_weight_hash(model)
    s0_target = evaluate_preflight_cell(model, views[RECORD_ID]["target"], state_id="S0", view="target")
    s0_locality = evaluate_preflight_cell(model, views[RECORD_ID]["locality"], state_id="S0", view="locality")
    append_jsonl(out_dir / "stage1_state_hash_ledger.jsonl", {"stage": "preflight", "state_id": "S0", "editable_weight_hash": s0_hash})

    short_sample = short_answer_sample(model, views[RECORD_ID]["target"], record)
    short_canonical = build_canonical_inputs(model, short_sample)
    assert SHORT_INSTRUCTION in short_canonical.prompt_text
    assert_generation_inputs_target_free(
        s0_target["canonical"].prompt_ids,
        s0_target["canonical"].prompt_ids.clone(),
        short_canonical.prompt_ids,
        short_canonical.prompt_ids.clone(),
        s0_target["canonical"].target_ids.tolist(),
    )

    apply_prefix(model, bank, 1)
    s1_hash = state_weight_hash(model)
    s1_target = evaluate_preflight_cell(model, views[RECORD_ID]["target"], state_id="S1", view="target")
    s1_locality = evaluate_preflight_cell(model, views[RECORD_ID]["locality"], state_id="S1", view="locality")
    append_jsonl(out_dir / "stage1_state_hash_ledger.jsonl", {"stage": "preflight", "state_id": "S1", "editable_weight_hash": s1_hash})

    anchor = bank.anchor_state()[MODULE_KEY]
    s1_state = bank.assemble_state([bank.list_edits()[0]["edit_id"]])[MODULE_KEY]
    existing_delta = s1_state - anchor
    b1_norm = tensor_l2(existing_delta)
    b1_relative = b1_norm / tensor_l2(anchor)

    hard_stop: NaturalAnswerSpanError
    try:
        old_ids = model.llava_tokenizer(str(old_answer), add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
        _span = align_unique_answer_span(
            s0_target["no_cache"]["token_ids"],
            old_ids,
            tokenizer=model.llava_tokenizer,
            answer_text=str(old_answer),
        )
    except NaturalAnswerSpanError as error:
        hard_stop = error
    else:
        raise RuntimeError("Natural answer boundary unexpectedly passed; this bounded fail-closed runner refuses to optimize without a separately reviewed continuation implementation")

    apply_prefix(model, bank, 0)
    rollback_target = manual_greedy_trace(model, s0_target["canonical"], CAP, eos_ids(model), top_k=5)
    rollback_locality = manual_greedy_trace(model, s0_locality["canonical"], CAP, eos_ids(model), top_k=5)
    bank_after = bank_manifest()
    assert_bank_immutable(bank_before, bank_after)
    rollback = {
        "passed": bool(
            state_weight_hash(model) == s0_hash
            and rollback_target["token_ids"] == s0_target["no_cache"]["token_ids"]
            and rollback_locality["token_ids"] == s0_locality["no_cache"]["token_ids"]
            and rollback_target["stop_reason"] == s0_target["no_cache"]["stop_reason"]
            and rollback_locality["stop_reason"] == s0_locality["no_cache"]["stop_reason"]
        ),
        "s0_hash_before": s0_hash,
        "s0_hash_after": state_weight_hash(model),
        "target_token_ids_equal": rollback_target["token_ids"] == s0_target["no_cache"]["token_ids"],
        "locality_token_ids_equal": rollback_locality["token_ids"] == s0_locality["no_cache"]["token_ids"],
        "target_stop_reason_equal": rollback_target["stop_reason"] == s0_target["no_cache"]["stop_reason"],
        "locality_stop_reason_equal": rollback_locality["stop_reason"] == s0_locality["no_cache"]["stop_reason"],
        "bank_before_sha256": bank_before["sha256"],
        "bank_after_sha256": bank_after["sha256"],
        "bank_unchanged": bank_before == bank_after,
    }
    append_jsonl(out_dir / "stage1_state_hash_ledger.jsonl", {"stage": "rollback", **rollback})
    if not rollback["passed"]:
        raise RuntimeError("Stage-1 preflight rollback mismatch")

    manifest["preflight"] = {
        "s0_target": serializable_cell(s0_target),
        "s1_target": serializable_cell(s1_target),
        "s0_locality": serializable_cell(s0_locality),
        "s1_locality": serializable_cell(s1_locality),
        "target_repr": repr(target),
        "old_answer_repr": repr(str(old_answer)),
        "gold_absent_from_generation_inputs": True,
    }
    write_failure_artifacts(
        out_dir,
        hard_stop=hard_stop,
        manifest=manifest,
        s0_target=s0_target,
        s1_target=s1_target,
        s0_locality=s0_locality,
        s1_locality=s1_locality,
        b1_norm=b1_norm,
        b1_relative=b1_relative,
        rollback=rollback,
    )
    write_text(out_dir / "source_config_diff.txt", "\n".join([
        f"required_starting_commit={STARTING_COMMIT}",
        "new scripts/engram/stage1_behavioral_margin_utils.py",
        "new scripts/engram/run_engram_v2_stage1_behavioral_margin_gate.py",
        "new tests/test_engram_v2_stage1_behavioral_margin.py",
        "no existing ENGRAM bank, Stage-0 output, model, tokenizer, processor, or config file was modified",
    ]))
    manifest["bank_manifest_after"] = bank_after
    manifest["original_bank_unchanged"] = bank_before == bank_after
    manifest["outputs"] = [
        {"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
        for path in sorted(out_dir.iterdir()) if path.is_file() and path.name != "run_manifest.json"
    ]
    write_json(out_dir / "run_manifest.json", manifest)
    missing = [name for name in REQUIRED_ARTIFACTS if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing Stage-1 artifacts: {missing}")
    print(json.dumps({
        "primary_final_label": "INVALID_ENGINEERING_RUN",
        "hard_stop_label": hard_stop.label,
        "optimization_started": False,
        "bank_unchanged": True,
        "stage2_permitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
