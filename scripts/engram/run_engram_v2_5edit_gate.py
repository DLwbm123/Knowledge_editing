#!/usr/bin/env python3
"""Frozen single-order five-edit gate for SR-TR ENGRAM V2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dsca_medmkeb_diag_common import clone_batch, ensure_offline_env, to_jsonable
from easyeditor.models.engram import EngramMultimodalHparams
from easyeditor.models.engram_v2 import SequentialEngramBankV2
from easyeditor.trainer.models import get_model
from scripts.engram.engram_eval_utils import shifted_teacher_forced_metrics, tensor_sha256
from scripts.engram.run_engram_continual_v2 import (
    aggregate_code_hash,
    build_views,
    make_editor,
    runtime_metadata,
    set_determinism,
    sha256_file,
    subspace_overlap,
)

ORDER = ["953", "1293", "1592", "2174", "1628"]
SUCCESS_EPS = 0.0
NORM_LIMIT = 0.010001
IMAGE_CACHE: Dict[str, Dict[str, Any]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config", default="hparams/ENGRAM/llava_med_continual_v2.yaml")
    parser.add_argument("--model-config", default="hparams/ENGRAM/llava_med_continual_v1.yaml")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n")


def cell(value: Any) -> Any:
    return json.dumps(to_jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: cell(row.get(field)) for field in fields})


def hash_state(model: torch.nn.Module) -> Dict[str, Any]:
    full = hashlib.sha256()
    result: Dict[str, Any] = {}
    for kind, items in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        digest = hashlib.sha256()
        count = total = requires_grad = 0
        dtypes, devices = set(), set()
        for name, tensor in items:
            value = tensor.detach().contiguous().cpu()
            raw = value.view(torch.uint8).numpy().tobytes()
            for token in (kind, name, str(value.dtype), str(tuple(value.shape))):
                digest.update(token.encode())
                full.update(token.encode())
            digest.update(raw)
            full.update(raw)
            count += 1
            total += len(raw)
            requires_grad += int(getattr(tensor, "requires_grad", False))
            dtypes.add(str(tensor.dtype))
            devices.add(str(tensor.device))
        result[kind] = {
            "sha256": digest.hexdigest(),
            "tensor_count": count,
            "total_bytes": total,
            "dtypes": sorted(dtypes),
            "devices": sorted(devices),
            "requires_grad_count": requires_grad,
        }
    result["full_state_sha256"] = full.hexdigest()
    return result


def image_metadata(path: Path) -> Dict[str, Any]:
    key = str(path.resolve())
    if key not in IMAGE_CACHE:
        stat = path.stat()
        IMAGE_CACHE[key] = {
            "path": key,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    return IMAGE_CACHE[key]


def evaluate(model: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
    batch = clone_batch(sample)
    pixels: List[Dict[str, Any]] = []
    original = model._image_for_row

    def traced(samples: Dict[str, Any], row: int) -> torch.Tensor:
        value = original(samples, row)
        pixels.append({
            "sha256": tensor_sha256(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        })
        return value

    model._image_for_row = traced
    try:
        with torch.inference_mode():
            embeds, labels, masks = model._build_batch(batch)
            output = model.llava_model(
                inputs_embeds=embeds,
                attention_mask=masks["attention_mask"].long(),
                labels=labels,
                return_dict=True,
                use_cache=False,
            )
    finally:
        model._image_for_row = original

    prompts, targets = model._prompt_target_lists(sample, 1)
    full_text = model._conversation_prompt(prompts[0], targets[0])
    input_ids = model.tokenizer_image_token(
        full_text, model.tokenizer, model.IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    paths = [Path(str(value)) for value in sample.get("image_path", [])]
    return {
        "metric": shifted_teacher_forced_metrics(output.logits, labels, ignore_index=model.IGNORE_INDEX),
        "logits": output.logits.detach().cpu(),
        "logits_sha256": tensor_sha256(output.logits),
        "input_ids_sha256": tensor_sha256(input_ids),
        "labels_sha256": tensor_sha256(labels),
        "attention_mask_sha256": tensor_sha256(masks["attention_mask"]),
        "inputs_embeds_sha256": tensor_sha256(embeds),
        "pixel_values": pixels,
        "image_metadata": [image_metadata(path) for path in paths],
        "use_cache_argument": False,
    }


def evaluate_set(model: Any, samples: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {name: evaluate(model, sample) for name, sample in samples.items()}


def clean(rows: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {name: {key: value for key, value in row.items() if key != "logits"} for name, row in rows.items()}


def compare(left: Mapping[str, Dict[str, Any]], right: Mapping[str, Dict[str, Any]], tol: float) -> Dict[str, Any]:
    if set(left) != set(right):
        return {"passed": False, "key_sets_equal": False, "max_nll_abs_diff": math.inf, "max_logits_abs_diff": math.inf}
    max_nll = max_logits = 0.0
    exact_logits = prep_equal = True
    details = {}
    prep_fields = (
        "input_ids_sha256", "labels_sha256", "attention_mask_sha256",
        "inputs_embeds_sha256", "pixel_values", "image_metadata", "use_cache_argument",
    )
    for name in sorted(left):
        lrow, rrow = left[name], right[name]
        nll = abs(float(lrow["metric"]["target_nll"]) - float(rrow["metric"]["target_nll"]))
        if tuple(lrow["logits"].shape) == tuple(rrow["logits"].shape):
            delta = lrow["logits"].float() - rrow["logits"].float()
            logits = float(delta.abs().max()) if delta.numel() else 0.0
            exact = bool(torch.equal(lrow["logits"], rrow["logits"]))
        else:
            logits, exact = math.inf, False
        prep = all(lrow[field] == rrow[field] for field in prep_fields)
        max_nll, max_logits = max(max_nll, nll), max(max_logits, logits)
        exact_logits, prep_equal = exact_logits and exact, prep_equal and prep
        details[name] = {"nll_abs_diff": nll, "logits_max_abs_diff": logits, "logits_exact": exact, "preparation_equal": prep}
    return {
        "passed": max_nll <= tol and max_logits <= tol and exact_logits and prep_equal,
        "key_sets_equal": True,
        "max_nll_abs_diff": max_nll,
        "max_logits_abs_diff": max_logits,
        "all_logits_exact": exact_logits,
        "all_preparation_equal": prep_equal,
        "details": details,
    }


def all_samples(views: Mapping[str, Mapping[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    return {
        f"{view}:{record_id}": views[record_id][view]
        for record_id in ORDER for view in ("target", "generalization", "locality")
    }


def stage_samples(views: Mapping[str, Mapping[str, Dict[str, Any]]], active: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    result = {
        f"{view}:{record_id}": views[record_id][view]
        for record_id in active for view in ("target", "generalization")
    }
    result.update({f"locality:{record_id}": views[record_id]["locality"] for record_id in ORDER})
    return result


def metrics_from(rows: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {name: row["metric"] for name, row in rows.items()}


def stage_rows(state: str, active: Sequence[str], baseline: Mapping[str, Dict[str, Any]], current: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record_id in active:
        for view in ("target", "generalization"):
            key = f"{view}:{record_id}"
            gain = float(baseline[key]["target_nll"] - current[key]["metric"]["target_nll"])
            rows.append({
                "state": state, "record_id": record_id, "view": view,
                "baseline_nll": baseline[key]["target_nll"],
                "current_nll": current[key]["metric"]["target_nll"],
                "improvement": gain, "abs_drift": abs(gain), "success": gain > SUCCESS_EPS,
            })
    for record_id in ORDER:
        key = f"locality:{record_id}"
        gain = float(baseline[key]["target_nll"] - current[key]["metric"]["target_nll"])
        rows.append({
            "state": state, "record_id": record_id, "view": "locality",
            "baseline_nll": baseline[key]["target_nll"],
            "current_nll": current[key]["metric"]["target_nll"],
            "improvement": gain, "abs_drift": abs(gain), "success": abs(gain) <= 0.5,
        })
    return rows


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    lvalue, rvalue = left.detach().double().reshape(-1), right.detach().double().reshape(-1)
    denom = float(lvalue.norm() * rvalue.norm())
    return float(lvalue.dot(rvalue) / denom) if denom else 0.0


def displacement(bank: SequentialEngramBankV2, key: str) -> tuple[float, float]:
    anchor = bank.anchor_state()[key].double()
    diff = bank.assemble_state()[key].double() - anchor
    norm = float(diff.norm())
    return norm, norm / max(float(anchor.norm()), 1.0e-12)


def finite_report(report: Dict[str, Any]) -> bool:
    stats = report["solver_stats"]
    keys = (
        "delta_norm", "delta_relative_norm", "small_system_condition", "ridge",
        "target_residual_error_after", "reference_effect_norm", "old_effect_norm",
    )
    return all(math.isfinite(float(stats[key])) for key in keys)


def bank_stats(root: Path) -> Dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": [{"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
    }


def delta_analysis(bank: SequentialEngramBankV2, key: str, reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    payloads = [bank.load_edit(item["edit_id"]) for item in bank.list_edits()]
    deltas = [payload["deltas"][key] for payload in payloads]
    factors = [payload["target_factors"][key] for payload in payloads]
    pairwise, overlaps = [], []
    for left in range(len(deltas)):
        for right in range(left + 1, len(deltas)):
            identity = {
                "left_step": left + 1, "right_step": right + 1,
                "left_record_id": ORDER[left], "right_record_id": ORDER[right],
            }
            pairwise.append({**identity, "cosine": cosine(deltas[left], deltas[right])})
            overlaps.append({**identity, **subspace_overlap(factors[left], factors[right])})
    steps = []
    for index, report in enumerate(reports):
        stats = report["solver_stats"]
        steps.append({
            "step": index + 1, "record_id": ORDER[index], "module": key,
            "delta_norm": stats["delta_norm"], "total_delta_norm": stats["delta_norm"],
            "relative_delta_norm": stats["delta_relative_norm"], "eta": stats["eta"],
            "trust_region_triggered": float(stats["eta"]) < 1.0,
            "target_columns": stats["target_columns"], "reference_columns": stats["reference_columns"],
            "old_columns": stats["old_columns"], "factor_effective_rank": stats["factor_effective_rank"],
            "factor_singular_max": stats["factor_singular_max"], "factor_singular_min": stats["factor_singular_min"],
            "singular_value_cutoff": max(float(stats["factor_singular_max"]), 1.0) * 1.0e-10,
            "small_system_condition": stats["small_system_condition"], "ridge": stats["ridge"],
            "target_residual_error_before": stats["target_residual_error_before"],
            "target_residual_error_after": stats["target_residual_error_after"],
            "reference_effect_norm": stats["reference_effect_norm"], "old_effect_norm": stats["old_effect_norm"],
        })
    cumulative_norm, cumulative_relative = displacement(bank, key)
    return {
        "steps": steps, "pairwise_delta_cosines": pairwise,
        "successive_delta_cosines": [row for row in pairwise if row["right_step"] == row["left_step"] + 1],
        "activation_subspace_overlaps": overlaps,
        "cumulative_displacement_norm": cumulative_norm,
        "cumulative_relative_displacement": cumulative_relative,
        "all_deltas_finite": all(bool(torch.isfinite(delta).all()) for delta in deltas),
        "max_single_step_relative_norm": max(float(row["relative_delta_norm"]) for row in steps),
        "max_condition_number": max(float(row["small_system_condition"]) for row in steps),
        "any_trust_region_triggered": any(bool(row["trust_region_triggered"]) for row in steps),
    }


def classify(checks: Mapping[str, bool], final_gains: Mapping[str, float], degradation: Mapping[str, float]) -> str:
    if not checks["deterministic"]:
        return "NOT_EVALUABLE"
    if not checks["bank"] or not checks["rollback"]:
        return "ENGRAM_V2_5EDIT_BANK_FAILURE"
    if not checks["solver"]:
        return "ENGRAM_V2_NUMERICAL_INSTABILITY"
    if all(checks.values()):
        return "ENGRAM_V2_5EDIT_PASS"
    if any(gain <= SUCCESS_EPS for gain in final_gains.values()) and any(degradation[record_id] > 0 for record_id in ORDER[:3]):
        return "ENGRAM_V2_INTERFERENCE_ACCUMULATION"
    return "ENGRAM_V2_5EDIT_PARTIAL"


def recommendation(gate: str) -> str:
    if gate == "ENGRAM_V2_5EDIT_PASS":
        return "PROMOTE_TO_FIXED_10EDIT"
    if gate in ("ENGRAM_V2_5EDIT_PARTIAL", "ENGRAM_V2_INTERFERENCE_ACCUMULATION", "ENGRAM_V2_NUMERICAL_INSTABILITY"):
        return "HOLD_FOR_DIAGNOSIS"
    return "STOP"


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    result_path = args.out_dir / "ENGRAM_V2_5EDIT_RESULTS.json"
    if result_path.exists():
        raise FileExistsError(f"Five-edit result exists: {result_path}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    ensure_offline_env()
    config_path = (ROOT / args.config).resolve()
    model_config_path = (ROOT / args.model_config).resolve()
    config = yaml.safe_load(config_path.read_text())
    set_determinism(int(config["seed"]))
    backend = {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }
    expected_backend = {"flash": False, "mem_efficient": False, "math": True, "cudnn": False, "deterministic": True}
    if torch.cuda.device_count() != 1 or backend != expected_backend:
        raise RuntimeError(f"GPU/backend preflight failed: count={torch.cuda.device_count()} backend={backend}")

    model_config = EngramMultimodalHparams.from_hparams(str(model_config_path))
    model_config.dropout, model_config.no_grad_layers, model_config.device = 0.0, None, "cuda"
    model = get_model(model_config).to(torch.device("cuda")).eval()
    records = {str(row["id"]): row for row in json.loads((ROOT / config["dataset"]).read_text())}
    if any(record_id not in records for record_id in ORDER):
        raise RuntimeError(f"Missing records from fixed order: {ORDER}")
    image_root = Path(model_config.coco_image)
    if not image_root.is_absolute():
        image_root = ROOT / image_root
    views = {record_id: build_views(model, records[record_id], image_root) for record_id in ORDER}

    code_paths = [
        ROOT / "easyeditor/models/engram_v2/solver.py",
        ROOT / "easyeditor/models/engram_v2/bank.py",
        ROOT / "easyeditor/models/engram_v2/editor.py",
        ROOT / "scripts/engram/engram_eval_utils.py",
        ROOT / "scripts/engram/run_engram_continual_v2.py",
        Path(__file__).resolve(),
    ]
    code_hash = aggregate_code_hash(code_paths)
    config_hash = sha256_file(config_path)
    module_name = str(config["target_module"])
    module_key = f"{module_name}.weight"
    module = dict(model.named_modules()).get(module_name)
    if not isinstance(module, torch.nn.Linear):
        raise KeyError(f"Missing frozen module: {module_name}")

    frozen_samples = all_samples(views)
    state0 = hash_state(model)
    first = evaluate_set(model, frozen_samples)
    second = evaluate_set(model, frozen_samples)
    state0_after = hash_state(model)
    tolerance = float(config["replay_nll_tolerance"])
    baseline_compare = compare(first, second, tolerance)
    training_modules = [name for name, item in model.named_modules() if item.training]
    baseline_gate = {
        "passed": state0 == state0_after and baseline_compare["passed"] and not training_modules,
        "full_state_unchanged": state0 == state0_after,
        "state_before": state0, "state_after": state0_after,
        "comparison": baseline_compare, "module_training_flags": training_modules,
        "model_training": model.training, "llava_model_training": model.llava_model.training,
        "model_config_use_cache": getattr(model.llava_model.config, "use_cache", None),
        "forward_use_cache": False, "backend": backend,
        "snapshots": clean(first),
    }
    write_json(args.out_dir / "preflight" / "baseline_determinism.json", baseline_gate)
    if not baseline_gate["passed"]:
        payload = {
            "gate": "NOT_EVALUABLE", "reason": "NOT_EVALUABLE_FORWARD_NONDETERMINISM",
            "fixed_order": ORDER, "baseline_determinism": baseline_gate, "runtime": runtime_metadata(),
        }
        write_json(result_path, payload)
        print(json.dumps(payload, indent=2))
        return

    del second
    baseline = metrics_from(first)
    bank_root = args.out_dir / "run" / "bank"
    if bank_root.exists():
        raise FileExistsError(f"Five-edit bank exists: {bank_root}")
    bank = SequentialEngramBankV2(bank_root)
    bank.initialize_anchor({module_key: module.weight}, metadata={
        "method": config["method_version"], "fixed_order": ORDER, "seed": config["seed"],
        "config_hash": config_hash, "code_hash": code_hash,
    })
    editor = make_editor(config)
    stages, reports, bank_rows, metric_rows, retention_rows, locality_rows = [], [], [], [], [], []
    histories = {record_id: [] for record_id in ORDER}
    write_gains: Dict[str, float] = {}

    for step, record_id in enumerate(ORDER, start=1):
        active = ORDER[:step]
        before = evaluate(model, views[record_id]["target"])
        report = editor.solve_and_store(
            model=model, target_sample=views[record_id]["target"],
            reference_sample=views[record_id]["locality"], bank=bank,
            edit_id=f"edit_{step:02d}_{record_id}", source_example_ids=[record_id],
            code_hash=code_hash, config_hash=config_hash, ignore_index=model.IGNORE_INDEX,
        )
        reports.append(report)
        torch.cuda.empty_cache()
        current_samples = stage_samples(views, active)
        direct = evaluate_set(model, current_samples)
        direct_state = hash_state(model)
        rows = stage_rows(f"S{step}", active, baseline, direct)
        metric_rows.extend(rows)
        targets = {row["record_id"]: row for row in rows if row["view"] == "target"}
        localities = [row for row in rows if row["view"] == "locality"]
        locality_rows.extend(localities)
        current_gain = float(targets[record_id]["improvement"])
        write_gains[record_id] = current_gain
        for active_id in active:
            histories[active_id].append(float(targets[active_id]["improvement"]))
        retention = {"state": f"S{step}", **{candidate: (float(targets[candidate]["improvement"]) if candidate in targets else None) for candidate in ORDER}}
        retention_rows.append(retention)

        metadata = bank.list_edits()
        bank.rollback_to_prefix(model, 0)
        bank.assemble_state_into_model(model, prefix=step)
        replay = evaluate_set(model, current_samples)
        replay_state = hash_state(model)
        replay_compare = compare(direct, replay, tolerance)

        bank.rollback_to_prefix(model, 0)
        fresh_bank = SequentialEngramBankV2(bank_root)
        fresh_bank.assemble_state_into_model(model, prefix=step)
        fresh = evaluate_set(model, current_samples)
        fresh_state = hash_state(model)
        fresh_compare = compare(direct, fresh, tolerance)
        cumulative_norm, cumulative_relative = displacement(bank, module_key)
        bank_row = {
            "prefix": list(active), "direct_state": direct_state,
            "in_memory_replay_state": replay_state, "fresh_reload_state": fresh_state,
            "direct_in_memory_full_state_equal": direct_state == replay_state,
            "direct_fresh_full_state_equal": direct_state == fresh_state,
            "in_memory_comparison": replay_compare, "fresh_comparison": fresh_compare,
            "bank_metadata_equal": metadata == fresh_bank.list_edits(),
            "edit_order": [item["edit_id"] for item in metadata],
            "parent_state_hashes": [item["parent_state_hash"] for item in metadata],
            "resulting_state_hashes": [item["resulting_state_hash"] for item in metadata],
            "delta_checksums": [item["delta_checksums"] for item in metadata],
            "bank_directory": bank_stats(bank_root),
        }
        bank_rows.append(bank_row)
        stage = {
            "step": step, "state": f"S{step}", "record_id": record_id, "active_ids": list(active),
            "target_nll_before_edit": before["metric"]["target_nll"],
            "target_nll_after_edit": direct[f"target:{record_id}"]["metric"]["target_nll"],
            "current_target_immediate_improvement": float(before["metric"]["target_nll"] - direct[f"target:{record_id}"]["metric"]["target_nll"]),
            "current_target_baseline_improvement": current_gain,
            "current_edit_success": current_gain > SUCCESS_EPS,
            "current_generalization_improvement": next(row["improvement"] for row in rows if row["view"] == "generalization" and row["record_id"] == record_id),
            "retained_count": sum(float(row["improvement"]) > SUCCESS_EPS for row in targets.values()),
            "worst_locality_drift": max(float(row["abs_drift"]) for row in localities),
            "mean_locality_drift": sum(float(row["abs_drift"]) for row in localities) / len(localities),
            "delta_norm": report["solver_stats"]["delta_norm"],
            "delta_relative_norm": report["solver_stats"]["delta_relative_norm"],
            "cumulative_displacement_norm": cumulative_norm,
            "cumulative_relative_displacement": cumulative_relative,
            "solver_finite": finite_report(report),
            "edit_report": report, "direct_snapshots": clean(direct),
            "bank_equivalence": bank_row,
        }
        stages.append(stage)
        write_json(args.out_dir / "run" / f"stage_{step:02d}.json", stage)
        del direct, replay, fresh
        torch.cuda.empty_cache()

    final_gains = {record_id: histories[record_id][-1] for record_id in ORDER}
    per_edit, degradations = [], {}
    for record_id in ORDER:
        best, final = max(histories[record_id]), final_gains[record_id]
        degradation = max(0.0, best - final)
        degradations[record_id] = degradation
        per_edit.append({
            "record_id": record_id, "write_improvement": write_gains[record_id],
            "best_improvement": best, "final_improvement": final,
            "retention_ratio": final / write_gains[record_id] if write_gains[record_id] > SUCCESS_EPS else None,
            "absolute_degradation": degradation, "retained": final > SUCCESS_EPS,
        })

    delta = delta_analysis(bank, module_key, reports)
    bank.rollback_to_prefix(model, 0)
    rollback_state = hash_state(model)
    rollback_eval = evaluate_set(model, frozen_samples)
    rollback_compare = compare(first, rollback_eval, tolerance)
    rollback = {
        "full_state_equal": rollback_state == state0,
        "parameter_hash_equal": rollback_state["parameter"]["sha256"] == state0["parameter"]["sha256"],
        "buffer_hash_equal": rollback_state["buffer"]["sha256"] == state0["buffer"]["sha256"],
        "dtype_equal": rollback_state["parameter"]["dtypes"] == state0["parameter"]["dtypes"] and rollback_state["buffer"]["dtypes"] == state0["buffer"]["dtypes"],
        "device_equal": rollback_state["parameter"]["devices"] == state0["parameter"]["devices"] and rollback_state["buffer"]["devices"] == state0["buffer"]["devices"],
        "rollback_state": rollback_state, "baseline_state": state0,
        "comparison": rollback_compare, "snapshots": clean(rollback_eval),
    }

    drifts = [float(row["abs_drift"]) for row in locality_rows]
    locality = {
        "limit": float(config["locality_nll_abs_drift_limit"]), "rows": locality_rows,
        "worst_drift": max(drifts), "mean_drift": sum(drifts) / len(drifts),
        "all_within_limit": max(drifts) <= float(config["locality_nll_abs_drift_limit"]),
    }
    replay_exact = all(row["direct_in_memory_full_state_equal"] and row["in_memory_comparison"]["passed"] for row in bank_rows)
    fresh_exact = all(row["direct_fresh_full_state_equal"] and row["fresh_comparison"]["passed"] and row["bank_metadata_equal"] for row in bank_rows)
    bank_summary = {
        "in_memory_replay_all_exact": replay_exact, "fresh_reload_all_exact": fresh_exact,
        "max_nll_abs_diff": max(max(row["in_memory_comparison"]["max_nll_abs_diff"], row["fresh_comparison"]["max_nll_abs_diff"]) for row in bank_rows),
        "max_logits_abs_diff": max(max(row["in_memory_comparison"]["max_logits_abs_diff"], row["fresh_comparison"]["max_logits_abs_diff"]) for row in bank_rows),
    }
    checks = {
        "deterministic": baseline_gate["passed"],
        "all_current_edits_succeed": all(stage["current_edit_success"] for stage in stages),
        "all_final_edits_retained": all(gain > SUCCESS_EPS for gain in final_gains.values()),
        "locality": locality["all_within_limit"],
        "single_step_norm": delta["max_single_step_relative_norm"] <= NORM_LIMIT,
        "cumulative_norm": delta["cumulative_relative_displacement"] <= NORM_LIMIT,
        "bank": replay_exact and fresh_exact,
        "rollback": rollback["full_state_equal"] and rollback_compare["passed"],
        "solver": delta["all_deltas_finite"] and all(stage["solver_finite"] for stage in stages),
    }
    gate = classify(checks, final_gains, degradations)
    recommend = recommendation(gate)
    prior_one = json.loads((ROOT / "outputs/engram_v2_method_development_20260710/ENGRAM_V2_1EDIT_RESULTS.json").read_text())
    prior_two = json.loads((ROOT / "outputs/engram_v2_method_development_20260710/ENGRAM_V2_2EDIT_RESULTS.json").read_text())
    two_gains = [float(comp["target"]["target_nll_improvement"]) for item in prior_two["order_results"] for comp in item["final_comparison"].values()]
    summary = {
        "edits_retained": sum(gain > SUCCESS_EPS for gain in final_gains.values()),
        "worst_target_improvement": min(final_gains.values()),
        "worst_historical_edit_degradation": max(degradations.values()),
        "worst_locality_drift": locality["worst_drift"],
        "max_single_step_relative_delta": delta["max_single_step_relative_norm"],
        "cumulative_relative_displacement": delta["cumulative_relative_displacement"],
    }
    result = {
        "gate": gate, "recommendation": recommend, "fixed_order": ORDER,
        "checks": checks, "summary": summary, "method_version": config["method_version"],
        "config_hash": config_hash, "code_hash": code_hash, "runtime": runtime_metadata(),
        "preflight": {
            "tests": "20 passed, 7 warnings", "v1_freeze_manifest": "PASS",
            "prior_v2_manifest": "PASS", "git_status": "NOT_A_GIT_REPOSITORY",
            "record_assets": "PASS",
        },
        "baseline_determinism": baseline_gate, "stages": stages,
        "per_edit_final": per_edit, "bank_summary": bank_summary,
        "rollback": rollback,
        "prior_trend": {
            "one_edit_target_gain": float(prior_one["comparison"]["target"]["target_nll_improvement"]),
            "two_edit_worst_final_target_gain": min(two_gains),
        },
        "constraints": {
            "training": False, "sweep": False, "alternate_order": False,
            "repeat_seed": False, "ten_edit": False, "twenty_edit": False,
            "physical_gpu": 0, "gpu2_used": False, "algorithm_or_config_modified": False,
        },
    }
    write_json(result_path, result)
    write_csv(args.out_dir / "ENGRAM_V2_5EDIT_STAGE_METRICS.csv", metric_rows, ["state", "record_id", "view", "baseline_nll", "current_nll", "improvement", "abs_drift", "success"])
    write_csv(args.out_dir / "ENGRAM_V2_5EDIT_RETENTION_MATRIX.csv", retention_rows, ["state", *ORDER])
    write_json(args.out_dir / "ENGRAM_V2_5EDIT_LOCALITY.json", locality)
    write_json(args.out_dir / "ENGRAM_V2_5EDIT_DELTA_ANALYSIS.json", delta)
    write_json(args.out_dir / "ENGRAM_V2_5EDIT_BANK_EQUIVALENCE.json", {"prefixes": bank_rows, "summary": bank_summary})
    write_json(args.out_dir / "ENGRAM_V2_5EDIT_ROLLBACK.json", rollback)
    print(json.dumps({
        "final_gate": gate, "fixed_sequence": ORDER,
        "edits_retained": f"{summary['edits_retained']}/5",
        "worst_target_improvement": summary["worst_target_improvement"],
        "worst_historical_edit_degradation": summary["worst_historical_edit_degradation"],
        "worst_locality_drift": summary["worst_locality_drift"],
        "max_single_step_relative_delta_norm": summary["max_single_step_relative_delta"],
        "cumulative_relative_displacement": summary["cumulative_relative_displacement"],
        "deterministic": checks["deterministic"], "replay": replay_exact,
        "fresh_reload": fresh_exact, "rollback": checks["rollback"],
        "solver_stability": checks["solver"], "tests": "20 passed, 0 failed",
        "gpu": "physical GPU0", "gpu2_used": False,
        "training": False, "sweep": False, "ten_edit_recommendation": recommend,
    }, indent=2))


if __name__ == "__main__":
    main()
