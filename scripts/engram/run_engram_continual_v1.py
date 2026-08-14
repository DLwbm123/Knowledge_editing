#!/usr/bin/env python3
"""Matched ENGRAM V1 continual-editing evaluator.

The core ENGRAM algorithm is not changed here. This runner freezes a protocol
around its existing update and EngramBank save/apply/rollback APIs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dsca_medmkeb_diag_common import clone_batch, ensure_offline_env, target_nll_from_outputs, to_jsonable
from easyeditor.models.engram import EngramBank, EngramMultimodalHparams, EngramMultimodalRewriteExecutor
from easyeditor.models.engram.engram_main import select_linear_layers
from easyeditor.trainer.models import get_model


def record_id(record: Dict[str, Any], fallback: int) -> str:
    return str(record.get("id", record.get("record_id", fallback)))


def resolve_image_path(image_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = image_root.resolve()
    if root.name == "images" and str(value).startswith("images/"):
        return root.parent / path
    return root / path

CONDITIONS = ("SEQUENTIAL_BASE_EDITOR", "ENGRAM_RESET_BANK", "ENGRAM_PERSISTENT_BANK")
EXPECTED_IDS = ["953", "1293", "1592", "2174", "1628", "942", "1382", "1333", "671", "1343"]
SUCCESS_EPS = 0.0
NUMERIC_EQ_TOL = 1.0e-5
LOCALITY_NLL_LIMIT = 0.5
MIN_CURRENT_SUCCESS = 7
MIN_PERSISTENT_COUNT_ADVANTAGE_RESET = 2
MIN_PERSISTENT_COUNT_ADVANTAGE_BASE = 1
MIN_MEAN_IMPROVEMENT_ADVANTAGE = 1.0e-3
EVAL_FORWARD_SEED = 424242


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hparams/ENGRAM/llava_med_continual_v1.yaml")
    parser.add_argument("--dataset", default="datasets/MedMKEB/eval.json")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-edits", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-label", required=True)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    names = list(fieldnames or (sorted({key for row in rows for key in row}) if rows else []))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        if names:
            writer.writeheader()
            writer.writerows([{key: _cell(row.get(key)) for key in names} for row in rows])


def _cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bank_checksum(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []
    total = 0
    for path in files:
        rel = str(path.relative_to(root)).encode()
        digest.update(rel)
        data = path.read_bytes()
        digest.update(data)
        total += len(data)
    return digest.hexdigest(), len(files), total


def prompt(text: Any) -> str:
    return f"Question: {str(text or '')} Short answer: "


def make_sample(model: Any, question: Any, answer: Any, image: Path) -> Dict[str, Any]:
    q = prompt(question)
    a = str(answer or "")
    labels = model.llava_tokenizer(a, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return {
        "image_path": [str(image)],
        "prompt": [q],
        "target": [a],
        "text_input": [q + a],
        "labels": labels,
        "prompts_len": [len(model.llava_tokenizer(q, add_special_tokens=False).input_ids)],
    }


def build_views(model: Any, record: Dict[str, Any], image_root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "target": make_sample(model, record.get("src"), record.get("alt"), resolve_image_path(image_root, record["image"])),
        "generalization": make_sample(model, record.get("rephrase"), record.get("alt"), resolve_image_path(image_root, record["image_rephrase"])),
        "locality": make_sample(model, record.get("m_loc_q"), record.get("m_loc_a"), resolve_image_path(image_root, record["m_loc"])),
    }


def build_request(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    return {
        "record_id": str(record["id"]),
        "prompt": prompt(record.get("src")),
        "target": str(record.get("alt") or ""),
        "image": str(resolve_image_path(image_root, record["image"])),
        "rephrase_prompt": prompt(record.get("rephrase")),
        "image_rephrase": str(resolve_image_path(image_root, record["image_rephrase"])),
        "multimodal_locality_prompt": prompt(record.get("m_loc_q")),
        "multimodal_locality_ground_truth": str(record.get("m_loc_a") or ""),
        "multimodal_locality_image": str(resolve_image_path(image_root, record["m_loc"])),
    }


def nll(model: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
    # The LLaVA-Med image path shows RNG-dependent discrete NLL branches even in eval mode.
    torch.manual_seed(EVAL_FORWARD_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EVAL_FORWARD_SEED)
    batch = clone_batch(sample)
    with torch.no_grad():
        outputs = model(batch)
    return target_nll_from_outputs(outputs, batch)


def baseline_metrics(model: Any, views: Sequence[Dict[str, Dict[str, Any]]]) -> List[Dict[str, Dict[str, Any]]]:
    result = []
    for item in views:
        result.append({name: nll(model, sample) for name, sample in item.items()})
    return result


def eval_condition(
    model: Any,
    condition: str,
    edit_step: int,
    records: Sequence[Dict[str, Any]],
    views: Sequence[Dict[str, Dict[str, Any]]],
    baseline: Sequence[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows = []
    for index in range(edit_step):
        current = {name: nll(model, sample) for name, sample in views[index].items()}
        target_imp = baseline[index]["target"]["target_nll"] - current["target"]["target_nll"]
        gen_imp = baseline[index]["generalization"]["target_nll"] - current["generalization"]["target_nll"]
        loc_drift = abs(current["locality"]["target_nll"] - baseline[index]["locality"]["target_nll"])
        rows.append({
            "condition": condition,
            "edit_step": edit_step,
            "query_index": index + 1,
            "record_id": record_id(records[index], index),
            "is_current_edit": index + 1 == edit_step,
            "base_target_nll": baseline[index]["target"]["target_nll"],
            "target_nll": current["target"]["target_nll"],
            "target_nll_improvement": target_imp,
            "target_success": target_imp > SUCCESS_EPS,
            "base_generalization_nll": baseline[index]["generalization"]["target_nll"],
            "generalization_nll": current["generalization"]["target_nll"],
            "generalization_nll_improvement": gen_imp,
            "generalization_success": gen_imp > SUCCESS_EPS,
            "base_locality_nll": baseline[index]["locality"]["target_nll"],
            "locality_nll": current["locality"]["target_nll"],
            "locality_nll_abs_drift": loc_drift,
        })
    return rows


def snapshot_layers(layers: Sequence[Any]) -> Dict[str, Dict[str, torch.Tensor | None]]:
    return {
        layer.name: {
            "weight": layer.module.weight.detach().clone(),
            "bias": layer.module.bias.detach().clone() if layer.module.bias is not None else None,
        }
        for layer in layers
    }


def restore_layers(layers: Sequence[Any], state: Dict[str, Dict[str, torch.Tensor | None]]) -> None:
    with torch.no_grad():
        for layer in layers:
            saved = state[layer.name]
            layer.module.weight.copy_(saved["weight"])
            if layer.module.bias is not None and saved["bias"] is not None:
                layer.module.bias.copy_(saved["bias"])


def layer_state_max_diff(layers: Sequence[Any], state: Dict[str, Dict[str, torch.Tensor | None]]) -> float:
    diffs = []
    for layer in layers:
        saved = state[layer.name]
        diffs.append(float((layer.module.weight.detach().float() - saved["weight"].float()).abs().max().cpu()))
        if layer.module.bias is not None and saved["bias"] is not None:
            diffs.append(float((layer.module.bias.detach().float() - saved["bias"].float()).abs().max().cpu()))
    return max(diffs, default=0.0)


def apply_ids(bank: EngramBank, model: Any, edit_ids: Iterable[str], counts: Dict[str, int]) -> None:
    for edit_id in edit_ids:
        bank.apply_edit(model, edit_id)
        counts["apply"] += 1


def rollback_ids(bank: EngramBank, model: Any, edit_ids: Iterable[str], counts: Dict[str, int]) -> None:
    for edit_id in edit_ids:
        bank.rollback_edit(model, edit_id)
        counts["rollback"] += 1


def matrix_rows(all_rows: Sequence[Dict[str, Any]], condition: str, max_edits: int) -> List[Dict[str, Any]]:
    output = []
    for step in range(1, max_edits + 1):
        row = {"condition": condition, "edit_step": step}
        indexed = {(int(item["edit_step"]), int(item["query_index"])): item for item in all_rows if item["condition"] == condition}
        for query in range(1, max_edits + 1):
            item = indexed.get((step, query))
            row[f"edit_{query:02d}"] = item.get("target_nll_improvement") if item else None
        output.append(row)
    return output


def condition_summary(rows: Sequence[Dict[str, Any]], condition: str, max_edits: int) -> Dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    final = [row for row in selected if int(row["edit_step"]) == max_edits]
    current = [row for row in selected if row["is_current_edit"]]
    best_by_query: Dict[int, float] = {}
    final_by_query = {int(row["query_index"]): float(row["target_nll_improvement"]) for row in final}
    for row in selected:
        q = int(row["query_index"])
        best_by_query[q] = max(best_by_query.get(q, float("-inf")), float(row["target_nll_improvement"]))
    forgetting = [max(0.0, best_by_query[q] - final_by_query[q]) for q in final_by_query]
    return {
        "condition": condition,
        "num_edits": max_edits,
        "current_success_count": sum(bool(row["target_success"]) for row in current),
        "final_retained_count": sum(bool(row["target_success"]) for row in final),
        "final_generalization_retained_count": sum(bool(row["generalization_success"]) for row in final),
        "mean_final_target_nll_improvement": _mean(row["target_nll_improvement"] for row in final),
        "mean_final_generalization_nll_improvement": _mean(row["generalization_nll_improvement"] for row in final),
        "mean_final_locality_nll_abs_drift": _mean(row["locality_nll_abs_drift"] for row in final),
        "worst_final_locality_nll_abs_drift": max((float(row["locality_nll_abs_drift"]) for row in final), default=None),
        "average_forgetting": _mean(forgetting),
        "worst_forgetting": max(forgetting, default=None),
        "failure_onset_step": next((int(row["edit_step"]) for row in current if not row["target_success"]), None),
    }


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def max_condition_diff(rows: Sequence[Dict[str, Any]], left: str, right: str, max_edits: int) -> float:
    fields = ("target_nll_improvement", "generalization_nll_improvement", "locality_nll_abs_drift")
    lmap = {(r["edit_step"], r["query_index"]): r for r in rows if r["condition"] == left}
    rmap = {(r["edit_step"], r["query_index"]): r for r in rows if r["condition"] == right}
    diffs = []
    for key in set(lmap) & set(rmap):
        if int(key[0]) <= max_edits:
            diffs.extend(abs(float(lmap[key][f]) - float(rmap[key][f])) for f in fields)
    return max(diffs, default=0.0)


def choose_gate(summaries: Dict[str, Dict[str, Any]], direct_persistent_diff: float, persistent_reset_diff: float, reload_diff: float) -> str:
    direct = summaries["SEQUENTIAL_BASE_EDITOR"]
    reset = summaries["ENGRAM_RESET_BANK"]
    persistent = summaries["ENGRAM_PERSISTENT_BANK"]
    required_current = min(MIN_CURRENT_SUCCESS, max(1, (7 * int(persistent["num_edits"]) + 9) // 10))
    if int(persistent["current_success_count"]) < required_current:
        return "CURRENT_EDIT_CAPACITY_COLLAPSES"
    if float(persistent["mean_final_locality_nll_abs_drift"] or 0.0) > LOCALITY_NLL_LIMIT:
        return "CONTINUAL_RETENTION_IMPROVES_BUT_LOCALITY_FAILS"
    if persistent_reset_diff <= NUMERIC_EQ_TOL:
        return "BANK_PERSISTENCE_HAS_NO_MEASURABLE_EFFECT"
    count_adv_reset = int(persistent["final_retained_count"]) - int(reset["final_retained_count"])
    count_adv_base = int(persistent["final_retained_count"]) - int(direct["final_retained_count"])
    mean_adv = float(persistent["mean_final_target_nll_improvement"] or 0.0) - float(reset["mean_final_target_nll_improvement"] or 0.0)
    if (
        count_adv_reset >= MIN_PERSISTENT_COUNT_ADVANTAGE_RESET
        and count_adv_base >= MIN_PERSISTENT_COUNT_ADVANTAGE_BASE
        and mean_adv >= MIN_MEAN_IMPROVEMENT_ADVANTAGE
        and reload_diff <= NUMERIC_EQ_TOL
    ):
        return "CONTINUAL_EDITING_SIGNAL_CONFIRMED"
    if int(persistent["final_retained_count"]) <= max(1, int(persistent["current_success_count"]) // 2):
        return "CATASTROPHIC_FORGETTING_REMAINS"
    return "ENGINEERING_PATH_VALID_EFFECT_INCONCLUSIVE"


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    summaries = payload["condition_summaries"]
    lines = [
        "# ENGRAM Continual Editing 10-Edit Report",
        "",
        "## Decision",
        "",
        f"**{payload['gate']}**",
        "",
        f"- Run label: `{payload['run_label']}`",
        f"- Completed edits: {payload['max_edits']}",
        f"- Record IDs: `{payload['record_ids']}`",
        f"- GPU visibility: `{payload['cuda_visible_devices']}`",
        f"- Core algorithm modified: no",
        f"- Training/optimization: no; ENGRAM is forward-only in this V1.",
        "",
        "## Frozen Interpretation",
        "",
        "`SEQUENTIAL_BASE_EDITOR` is the ENGRAM projector update applied directly and cumulatively with no bank read. The current source has no separate underlying base editor. `ENGRAM_RESET_BANK` replays only the current serialized entry. `ENGRAM_PERSISTENT_BANK` replays every serialized entry in order. All entries are extracted once along the same direct sequential trajectory, so the comparison does not change the update rule.",
        "",
        "## Condition Results",
        "",
        "| condition | current success | final retained | gen retained | mean final improvement | avg forgetting | mean locality drift | worst locality drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONDITIONS:
        row = summaries[name]
        lines.append(
            "| {condition} | {current}/{n} | {retained}/{n} | {gen}/{n} | {mean:.6g} | {forget:.6g} | {loc:.6g} | {worst:.6g} |".format(
                condition=name,
                current=row["current_success_count"], retained=row["final_retained_count"], gen=row["final_generalization_retained_count"], n=payload["max_edits"],
                mean=float(row["mean_final_target_nll_improvement"] or 0.0), forget=float(row["average_forgetting"] or 0.0),
                loc=float(row["mean_final_locality_nll_abs_drift"] or 0.0), worst=float(row["worst_final_locality_nll_abs_drift"] or 0.0),
            )
        )
    lines.extend([
        "",
        "## Causal Checks",
        "",
        f"- Direct vs persistent max metric difference: {payload['direct_persistent_max_abs_diff']}",
        f"- Persistent vs reset max metric difference: {payload['persistent_reset_max_abs_diff']}",
        f"- Fresh bank reload max target-NLL difference: {payload['reload_max_target_nll_diff']}",
        f"- Bank reload reproducible: {payload['bank_reload_reproducible']}",
        "",
        "The bank is not a router and no routing metric is reported. Equality between direct accumulation and persistent replay establishes serialization/replay correctness, not an advantage over the update primitive.",
        "",
        "## Protocol Thresholds Frozen Before Runs",
        "",
        f"- Target success: target NLL improvement > {SUCCESS_EPS}.",
        f"- Current capacity floor: {MIN_CURRENT_SUCCESS}/10.",
        f"- Locality failure: mean absolute locality NLL drift > {LOCALITY_NLL_LIMIT}.",
        f"- Numerical equivalence/reload tolerance: {NUMERIC_EQ_TOL}.",
        f"- Signal count advantage: persistent >= reset + {MIN_PERSISTENT_COUNT_ADVANTAGE_RESET}, and persistent >= direct + {MIN_PERSISTENT_COUNT_ADVANTAGE_BASE}.",
        f"- Signal mean-improvement advantage over reset: {MIN_MEAN_IMPROVEMENT_ADVANTAGE}.",
        "",
        "## Artifacts",
        "",
        "- `retention_matrix.csv`",
        "- `per_edit_trajectory.csv`",
        "- `locality_trajectory.csv`",
        "- `summary.json`",
        "- `bank_state_manifest.csv`",
        "- `steps/step_01.json` through the final step",
        "- `banks/persistent/` and `banks/reset/`",
        "",
        "## Recommendation",
        "",
        "Do not run 20-edit or additional seeds from this result unless the gate is CONTINUAL_EDITING_SIGNAL_CONFIRMED. Do not tune alpha, direction, module, or thresholds after observing this run; a changed method definition requires a separately versioned protocol.",
        "",
    ])
    (out_dir / "ENGRAM_CONTINUAL_EDITING_10EDIT_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.max_edits == 10 and args.run_label != "ten_edit":
        raise SystemExit("10-edit requires --run-label ten_edit")
    ensure_offline_env()
    set_seeds(args.seed)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "steps").mkdir(exist_ok=True)
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    dataset_path = (ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    config = EngramMultimodalHparams.from_hparams(str(config_path))
    # Trainer model factory expects this generic field, which ENGRAM hparams omit.
    config.dropout = 0.0
    config.no_grad_layers = None
    config.device = "cuda"
    config.bank_dir = str(out_dir / "banks" / "persistent")
    records_all = json.loads(dataset_path.read_text())
    records = records_all[: args.max_edits]
    ids = [record_id(record, i) for i, record in enumerate(records)]
    if ids != EXPECTED_IDS[: args.max_edits]:
        raise RuntimeError(f"record order mismatch: {ids}")
    image_root = Path(config.coco_image)
    if not image_root.is_absolute():
        image_root = ROOT / image_root

    start = time.time()
    model = get_model(config).to(torch.device("cuda")).eval()
    tokenizer = model.llava_tokenizer
    selected_layers = select_linear_layers(model, config)
    if len(selected_layers) != 1:
        raise RuntimeError(f"V1 requires exactly one selected layer, got {[layer.name for layer in selected_layers]}")
    base_layer_state = snapshot_layers(selected_layers)
    views = [build_views(model, record, image_root) for record in records]
    requests = [build_request(record, image_root) for record in records]
    baseline = baseline_metrics(model, views)
    write_json(out_dir / "baseline_metrics.json", baseline)

    persistent_root = out_dir / "banks" / "persistent"
    reset_root = out_dir / "banks" / "reset"
    persistent_bank = EngramBank(persistent_root)
    executor = EngramMultimodalRewriteExecutor()
    all_rows: List[Dict[str, Any]] = []
    bank_rows: List[Dict[str, Any]] = []
    op_counts = {"update": 0, "apply": 0, "rollback": 0, "compose": 0}
    edit_ids: List[str] = []

    for step, (record, request) in enumerate(zip(records, requests), start=1):
        edit_id = f"edit_{step:02d}_{record_id(record, step - 1)}"
        config.edit_id = edit_id
        config.concept_id = record_id(record, step - 1)
        set_seeds(args.seed * 1000 + step)
        executor.apply_to_model(model, tokenizer, [request], config, copy=False, return_orig_weights=False)
        op_counts["update"] += 1
        edit_ids.append(edit_id)

        if reset_root.exists():
            shutil.rmtree(reset_root)
        reset_bank = EngramBank(reset_root)
        reset_bank.save_edit(edit_id=edit_id, metadata=deepcopy(executor.last_report["metadata"]), updates=executor.last_updates)

        direct_state = snapshot_layers(selected_layers)
        direct_rows = eval_condition(model, CONDITIONS[0], step, records, views, baseline)
        all_rows.extend(direct_rows)

        rollback_ids(persistent_bank, model, reversed(edit_ids), op_counts)
        direct_rollback_max_abs_error = layer_state_max_diff(selected_layers, base_layer_state)
        restore_layers(selected_layers, base_layer_state)

        apply_ids(persistent_bank, model, edit_ids, op_counts)
        persistent_replay_state_max_abs_diff = layer_state_max_diff(selected_layers, direct_state)
        persistent_rows = eval_condition(model, CONDITIONS[2], step, records, views, baseline)
        all_rows.extend(persistent_rows)
        rollback_ids(persistent_bank, model, reversed(edit_ids), op_counts)
        persistent_rollback_max_abs_error = layer_state_max_diff(selected_layers, base_layer_state)
        restore_layers(selected_layers, base_layer_state)

        apply_ids(reset_bank, model, [edit_id], op_counts)
        reset_rows = eval_condition(model, CONDITIONS[1], step, records, views, baseline)
        all_rows.extend(reset_rows)
        rollback_ids(reset_bank, model, [edit_id], op_counts)
        reset_rollback_max_abs_error = layer_state_max_diff(selected_layers, base_layer_state)
        restore_layers(selected_layers, base_layer_state)

        restore_layers(selected_layers, direct_state)
        p_checksum, p_files, p_bytes = bank_checksum(persistent_root)
        r_checksum, r_files, r_bytes = bank_checksum(reset_root)
        bank_rows.extend([
            {"step": step, "condition": CONDITIONS[2], "entries": len(persistent_bank.list_edits()), "files": p_files, "bytes": p_bytes, "sha256": p_checksum, "direct_rollback_max_abs_error": direct_rollback_max_abs_error, "replay_state_max_abs_diff": persistent_replay_state_max_abs_diff, "rollback_max_abs_error": persistent_rollback_max_abs_error, **op_counts},
            {"step": step, "condition": CONDITIONS[1], "entries": len(reset_bank.list_edits()), "files": r_files, "bytes": r_bytes, "sha256": r_checksum, "rollback_max_abs_error": reset_rollback_max_abs_error, **op_counts},
        ])
        write_json(out_dir / "steps" / f"step_{step:02d}.json", {
            "step": step, "record_id": record_id(record, step - 1), "edit_id": edit_id,
            "conditions": {name: [row for row in all_rows if row["condition"] == name and row["edit_step"] == step] for name in CONDITIONS},
            "engram_report": executor.last_report, "bank_state": bank_rows[-2:], "operation_counts": dict(op_counts),
        })

    restore_layers(selected_layers, base_layer_state)
    reloaded_bank = EngramBank(persistent_root)
    apply_ids(reloaded_bank, model, edit_ids, op_counts)
    reload_rows = eval_condition(model, "ENGRAM_PERSISTENT_BANK_RELOADED", args.max_edits, records, views, baseline)
    persistent_final = {(row["query_index"]): row for row in all_rows if row["condition"] == CONDITIONS[2] and row["edit_step"] == args.max_edits}
    reload_diff = max((abs(float(row["target_nll"]) - float(persistent_final[row["query_index"]]["target_nll"])) for row in reload_rows), default=0.0)

    summaries = {condition: condition_summary(all_rows, condition, args.max_edits) for condition in CONDITIONS}
    direct_persistent_diff = max_condition_diff(all_rows, CONDITIONS[0], CONDITIONS[2], args.max_edits)
    persistent_reset_diff = max_condition_diff(all_rows, CONDITIONS[2], CONDITIONS[1], args.max_edits)
    gate = choose_gate(summaries, direct_persistent_diff, persistent_reset_diff, reload_diff)
    matrix = [row for condition in CONDITIONS for row in matrix_rows(all_rows, condition, args.max_edits)]
    locality_rows = [{key: row[key] for key in ("condition", "edit_step", "query_index", "record_id", "locality_nll_abs_drift")} for row in all_rows]
    payload = {
        "gate": gate, "run_label": args.run_label, "max_edits": args.max_edits, "record_ids": ids,
        "seed": args.seed, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "condition_summaries": summaries, "direct_persistent_max_abs_diff": direct_persistent_diff,
        "persistent_reset_max_abs_diff": persistent_reset_diff, "reload_max_target_nll_diff": reload_diff,
        "bank_reload_reproducible": reload_diff <= NUMERIC_EQ_TOL, "operation_counts": op_counts,
        "runtime_sec": time.time() - start, "max_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "config_sha256": sha256_file(config_path), "dataset_path": str(dataset_path), "config_path": str(config_path),
        "thresholds": {"success_eps": SUCCESS_EPS, "numeric_eq_tol": NUMERIC_EQ_TOL, "locality_nll_limit": LOCALITY_NLL_LIMIT,
                       "min_current_success": MIN_CURRENT_SUCCESS, "min_persistent_count_advantage_reset": MIN_PERSISTENT_COUNT_ADVANTAGE_RESET,
                       "min_persistent_count_advantage_base": MIN_PERSISTENT_COUNT_ADVANTAGE_BASE,
                       "min_mean_improvement_advantage": MIN_MEAN_IMPROVEMENT_ADVANTAGE},
    }
    write_csv(out_dir / "retention_matrix.csv", matrix)
    write_csv(out_dir / "per_edit_trajectory.csv", all_rows)
    write_csv(out_dir / "locality_trajectory.csv", locality_rows)
    write_csv(out_dir / "bank_state_manifest.csv", bank_rows)
    write_json(out_dir / "summary.json", payload)
    write_json(out_dir / "reload_results.json", reload_rows)
    if args.max_edits == 10:
        write_report(out_dir, payload)
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
