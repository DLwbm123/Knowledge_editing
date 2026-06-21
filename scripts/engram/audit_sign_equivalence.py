#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.engram_main import EngramMultimodalRewriteExecutor  # noqa: E402
from easyeditor.models.engram.erasure_metrics import safe_model_answer_nll_and_logprob  # noqa: E402
from easyeditor.models.engram.solver import EngramLayerUpdate, apply_update_to_module  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _load_records(path: Path, max_edits: Optional[int]) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Expected non-empty JSON list: {path}")
    return records[:max_edits] if max_edits else records


def _resolve_image(root: Path, rel_path: str) -> str:
    path = Path(rel_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists() and root.name == "images":
        rel = Path(rel_path)
        if rel.parts and rel.parts[0] == "images":
            path = root / Path(*rel.parts[1:])
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def _record_to_request(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    record_id = record.get("id")
    return {
        "id": record_id,
        "record_id": record_id,
        "source_record_id": record_id,
        "prompt": record["src"],
        "target": record.get("pred") or record.get("alt") or record.get("erase_answer"),
        "pred": record.get("pred"),
        "image": _resolve_image(image_root, record["image"]),
        "rephrase_prompt": record.get("rephrase"),
        "image_rephrase": _resolve_image(image_root, record["image_rephrase"]) if record.get("image_rephrase") else None,
        "locality_prompt": record.get("loc"),
        "locality_ground_truth": record.get("loc_ans"),
        "multimodal_locality_prompt": record.get("m_loc_q"),
        "multimodal_locality_ground_truth": record.get("m_loc_a"),
        "multimodal_locality_image": _resolve_image(image_root, record["m_loc"]) if record.get("m_loc") else None,
    }


def _target_sample(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    return {
        "text_input": record["src"],
        "prompt": record["src"],
        "target": record.get("pred") or record.get("erase_answer") or record.get("alt"),
        "image_path": _resolve_image(image_root, record["image"]),
    }


def _reference_sample(record: Dict[str, Any], image_root: Path) -> Optional[Dict[str, Any]]:
    if not (record.get("m_loc_q") and record.get("m_loc_a") and record.get("m_loc")):
        return None
    return {
        "text_input": record["m_loc_q"],
        "prompt": record["m_loc_q"],
        "target": record["m_loc_a"],
        "image_path": _resolve_image(image_root, record["m_loc"]),
    }


def _module_map(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def _snapshot_modules(model: torch.nn.Module, module_names: Iterable[str]) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
    modules = _module_map(model)
    snapshots: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
    for name in module_names:
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"Module not found or not Linear: {name}")
        snapshots[name] = {
            "weight": module.weight.detach().clone().cpu(),
            "bias": module.bias.detach().clone().cpu() if module.bias is not None else None,
        }
    return snapshots


def _restore_modules(model: torch.nn.Module, snapshots: Dict[str, Dict[str, Optional[torch.Tensor]]]) -> None:
    modules = _module_map(model)
    with torch.no_grad():
        for name, tensors in snapshots.items():
            module = modules[name]
            module.weight.copy_(tensors["weight"].to(module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None and tensors["bias"] is not None:
                module.bias.copy_(tensors["bias"].to(module.bias.device, dtype=module.bias.dtype))


def _max_snapshot_diff(model: torch.nn.Module, snapshots: Dict[str, Dict[str, Optional[torch.Tensor]]]) -> float:
    return _max_between_snapshots(_snapshot_modules(model, snapshots.keys()), snapshots)


def _max_between_snapshots(
    left: Dict[str, Dict[str, Optional[torch.Tensor]]],
    right: Dict[str, Dict[str, Optional[torch.Tensor]]],
) -> float:
    diffs: List[float] = []
    for name in left:
        diffs.append((left[name]["weight"] - right[name]["weight"]).abs().max().item())
        if left[name].get("bias") is not None and right[name].get("bias") is not None:
            diffs.append((left[name]["bias"] - right[name]["bias"]).abs().max().item())
    return float(max(diffs) if diffs else 0.0)


def _tensor_checksum(tensor: Optional[torch.Tensor]) -> Optional[str]:
    if tensor is None:
        return None
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _update_checksum(update: EngramLayerUpdate) -> Dict[str, Any]:
    return {
        "weight_shape": list(update.weight.shape),
        "bias_shape": list(update.bias.shape) if update.bias is not None else None,
        "projector_shape": list(update.projector.shape) if update.projector is not None else None,
        "weight_sha256": _tensor_checksum(update.weight),
        "bias_sha256": _tensor_checksum(update.bias),
        "projector_sha256": _tensor_checksum(update.projector),
    }


def _strip(metrics: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if not metrics.get("available"):
        return None
    return {"nll": float(metrics["nll"]), "logprob": float(metrics["logprob"]), "num_tokens": int(metrics["num_tokens"])}


def _eval_pair(model: torch.nn.Module, record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    target = safe_model_answer_nll_and_logprob(model, _target_sample(record, image_root))
    reference_sample = _reference_sample(record, image_root)
    reference = safe_model_answer_nll_and_logprob(model, reference_sample) if reference_sample else None
    return {"target": target, "reference": reference}


def _clone_update(update: EngramLayerUpdate, *, alpha: float, direction: str, sign: int) -> EngramLayerUpdate:
    return replace(
        update,
        alpha=float(alpha),
        engram_update_direction=direction,
        direction_sign=int(sign),
        paper_direction_equivalent=(
            "paper_style_W_minus_alpha_E"
            if direction == "subtract"
            else "equivalent_to_paper_subtract_with_signed_alpha_negative"
        ),
    )


def _apply_updates(model: torch.nn.Module, updates: Dict[str, EngramLayerUpdate], *, direction: int = -1) -> None:
    modules = _module_map(model)
    for name, update in updates.items():
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"Module not found or not Linear: {name}")
        apply_update_to_module(module, update, direction=direction)


def _apply_manual_plus(model: torch.nn.Module, updates: Dict[str, EngramLayerUpdate], alpha: float) -> None:
    modules = _module_map(model)
    with torch.no_grad():
        for name, update in updates.items():
            module = modules[name]
            module.weight.add_((float(alpha) * update.weight).to(module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None and update.bias is not None:
                module.bias.add_((float(alpha) * update.bias).to(module.bias.device, dtype=module.bias.dtype))


def _apply_manual_minus(model: torch.nn.Module, updates: Dict[str, EngramLayerUpdate], alpha: float) -> None:
    _apply_manual_plus(model, updates, -float(alpha))


def _apply_composed_delta(model: torch.nn.Module, composed: Dict[str, Dict[str, Any]], scale: float = 1.0) -> None:
    modules = _module_map(model)
    with torch.no_grad():
        for name, raw in composed.items():
            module = modules[name]
            module.weight.add_((float(scale) * raw["weight"]).to(module.weight.device, dtype=module.weight.dtype))
            bias = raw.get("bias")
            if module.bias is not None and bias is not None:
                module.bias.add_((float(scale) * bias).to(module.bias.device, dtype=module.bias.dtype))


def _metric_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Optional[float]]:
    target_before = _strip(before["target"])
    target_after = _strip(after["target"])
    reference_before = _strip(before.get("reference") or {})
    reference_after = _strip(after.get("reference") or {})
    return {
        "target_nll_before": None if target_before is None else target_before["nll"],
        "target_nll_after": None if target_after is None else target_after["nll"],
        "target_logprob_before": None if target_before is None else target_before["logprob"],
        "target_logprob_after": None if target_after is None else target_after["logprob"],
        "target_nll_increase": None if target_before is None or target_after is None else target_after["nll"] - target_before["nll"],
        "target_logprob_drop": None if target_before is None or target_after is None else target_before["logprob"] - target_after["logprob"],
        "reference_nll_before": None if reference_before is None else reference_before["nll"],
        "reference_nll_after": None if reference_after is None else reference_after["nll"],
        "reference_delta_abs": None if reference_before is None or reference_after is None else abs(reference_after["nll"] - reference_before["nll"]),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _extract_updates(
    wrapper: torch.nn.Module,
    tok: Any,
    base_hparams: EngramMultimodalHparams,
    record: Dict[str, Any],
    image_root: Path,
    seed: int,
) -> Tuple[Dict[str, EngramLayerUpdate], Dict[str, Any]]:
    _set_seed(seed)
    hparams = deepcopy(base_hparams)
    hparams.alpha = 0.0
    hparams.engram_update_direction = "subtract"
    hparams.bank_dir = None
    hparams.engram_bank_path = None
    hparams.edit_id = None
    executor = EngramMultimodalRewriteExecutor()
    request = _record_to_request(record, image_root)
    executor.apply_to_model(wrapper, tok, [request], hparams, copy=False, return_orig_weights=True, keep_original_weight=True)
    updates = {name: deepcopy(update) for name, update in executor.last_updates.items()}
    return updates, deepcopy(executor.last_report.get("metadata", {}))


def _mode_updates(updates: Dict[str, EngramLayerUpdate], mode: str) -> Dict[str, EngramLayerUpdate]:
    if mode == "subtract_negative":
        return {name: _clone_update(update, alpha=-0.05, direction="subtract", sign=-1) for name, update in updates.items()}
    if mode == "add_positive":
        return {name: _clone_update(update, alpha=0.05, direction="add", sign=1) for name, update in updates.items()}
    raise ValueError(mode)


def _run_single_e(
    wrapper: torch.nn.Module,
    tok: Any,
    hparams: EngramMultimodalHparams,
    records: List[Dict[str, Any]],
    image_root: Path,
    out_dir: Path,
    seed: int,
    tolerance: float,
) -> Tuple[Dict[str, Dict[str, EngramLayerUpdate]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    extracted: Dict[str, Dict[str, EngramLayerUpdate]] = {}
    for case_index, record in enumerate(records):
        record_id = str(record.get("id"))
        updates, metadata = _extract_updates(wrapper, tok, hparams, record, image_root, seed)
        extracted[record_id] = updates
        module_names = list(updates)
        base_snapshot = _snapshot_modules(wrapper, module_names)
        baseline = _eval_pair(wrapper, record, image_root)
        checksums = {name: _update_checksum(update) for name, update in updates.items()}

        mode_states: Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]] = {}
        mode_metrics: Dict[str, Dict[str, Any]] = {}

        for mode in ["subtract_negative", "add_positive"]:
            _restore_modules(wrapper, base_snapshot)
            mode_update = _mode_updates(updates, mode)
            _apply_updates(wrapper, mode_update, direction=-1)
            mode_states[mode] = _snapshot_modules(wrapper, module_names)
            mode_metrics[mode] = _metric_delta(baseline, _eval_pair(wrapper, record, image_root))
            _apply_updates(wrapper, mode_update, direction=1)
            rollback = _max_snapshot_diff(wrapper, base_snapshot)
            mode_metrics[mode]["rollback_max_abs_diff"] = rollback

        _restore_modules(wrapper, base_snapshot)
        _apply_manual_plus(wrapper, updates, 0.05)
        mode_states["manual_plus"] = _snapshot_modules(wrapper, module_names)
        mode_metrics["manual_plus"] = _metric_delta(baseline, _eval_pair(wrapper, record, image_root))
        _apply_manual_minus(wrapper, updates, 0.05)
        mode_metrics["manual_plus"]["rollback_max_abs_diff"] = _max_snapshot_diff(wrapper, base_snapshot)
        _restore_modules(wrapper, base_snapshot)

        rows.append(
            {
                "case_index": case_index,
                "record_id": record_id,
                "selected_modules": module_names,
                "target_activation_counts": [metadata.get("layers", [])[i].get("num_target_vectors") for i in range(len(module_names))],
                "reference_activation_counts": [metadata.get("layers", [])[i].get("num_reference_vectors") for i in range(len(module_names))],
                "e_checksums": checksums,
                "max_abs_param_diff_A_vs_B": _max_between_snapshots(mode_states["subtract_negative"], mode_states["add_positive"]),
                "max_abs_param_diff_A_vs_C": _max_between_snapshots(mode_states["subtract_negative"], mode_states["manual_plus"]),
                "max_abs_param_diff_B_vs_C": _max_between_snapshots(mode_states["add_positive"], mode_states["manual_plus"]),
                "target_nll_after_A": mode_metrics["subtract_negative"]["target_nll_after"],
                "target_nll_after_B": mode_metrics["add_positive"]["target_nll_after"],
                "target_nll_after_C": mode_metrics["manual_plus"]["target_nll_after"],
                "reference_nll_after_A": mode_metrics["subtract_negative"]["reference_nll_after"],
                "reference_nll_after_B": mode_metrics["add_positive"]["reference_nll_after"],
                "reference_nll_after_C": mode_metrics["manual_plus"]["reference_nll_after"],
                "target_nll_increase_A": mode_metrics["subtract_negative"]["target_nll_increase"],
                "target_nll_increase_B": mode_metrics["add_positive"]["target_nll_increase"],
                "target_nll_increase_C": mode_metrics["manual_plus"]["target_nll_increase"],
                "rollback_A": mode_metrics["subtract_negative"]["rollback_max_abs_diff"],
                "rollback_B": mode_metrics["add_positive"]["rollback_max_abs_diff"],
                "rollback_C": mode_metrics["manual_plus"]["rollback_max_abs_diff"],
                "within_tolerance": (
                    _max_between_snapshots(mode_states["subtract_negative"], mode_states["add_positive"]) <= tolerance
                    and _max_between_snapshots(mode_states["subtract_negative"], mode_states["manual_plus"]) <= tolerance
                    and _max_between_snapshots(mode_states["add_positive"], mode_states["manual_plus"]) <= tolerance
                ),
            }
        )
    output = {
        "status": "pass" if all(row["within_tolerance"] for row in rows) else "fail",
        "tolerance": tolerance,
        "modes": {
            "A": "subtract alpha=-0.05, W <- W - alpha * E",
            "B": "add alpha=+0.05, W <- W + alpha * E",
            "C": "manual W <- W + 0.05 * E",
        },
        "rows": rows,
    }
    _write_json(out_dir / "single_E_equivalence.json", output)
    _write_csv(out_dir / "single_E_equivalence.csv", rows)
    return extracted, rows


def _run_bank_compose(
    wrapper: torch.nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    extracted: Dict[str, Dict[str, EngramLayerUpdate]],
    out_dir: Path,
    tolerance: float,
) -> List[Dict[str, Any]]:
    tmp_root = out_dir / "_tmp_bank_compose"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    rows: List[Dict[str, Any]] = []
    for case_index, record in enumerate(records):
        record_id = str(record.get("id"))
        updates_add = _mode_updates(extracted[record_id], "add_positive")
        module_names = list(updates_add)
        base_snapshot = _snapshot_modules(wrapper, module_names)
        baseline = _eval_pair(wrapper, record, image_root)

        def state_and_metric_for_direct() -> Tuple[Dict[str, Dict[str, Optional[torch.Tensor]]], Dict[str, Any], float]:
            _restore_modules(wrapper, base_snapshot)
            _apply_updates(wrapper, updates_add, direction=-1)
            state = _snapshot_modules(wrapper, module_names)
            metric = _metric_delta(baseline, _eval_pair(wrapper, record, image_root))
            _apply_updates(wrapper, updates_add, direction=1)
            return state, metric, _max_snapshot_diff(wrapper, base_snapshot)

        direct_state, direct_metric, direct_rollback = state_and_metric_for_direct()

        bank_root = tmp_root / record_id / "single"
        bank = EngramBank(bank_root)
        bank.save_edit(
            edit_id=f"{record_id}_add005",
            metadata={"record_id": record_id, "source_record_id": record_id, "source_request_ids": [record_id]},
            updates=updates_add,
            overwrite=True,
        )
        _restore_modules(wrapper, base_snapshot)
        bank.apply_edit(wrapper, f"{record_id}_add005")
        bank_state = _snapshot_modules(wrapper, module_names)
        bank_metric = _metric_delta(baseline, _eval_pair(wrapper, record, image_root))
        bank.rollback_edit(wrapper, f"{record_id}_add005")
        bank_rollback = _max_snapshot_diff(wrapper, base_snapshot)

        _restore_modules(wrapper, base_snapshot)
        _apply_composed_delta(wrapper, bank.compose_updates([f"{record_id}_add005"]))
        compose_state = _snapshot_modules(wrapper, module_names)
        compose_metric = _metric_delta(baseline, _eval_pair(wrapper, record, image_root))
        _apply_composed_delta(wrapper, bank.compose_updates([f"{record_id}_add005"]), scale=-1.0)
        compose_rollback = _max_snapshot_diff(wrapper, base_snapshot)

        half_updates = {name: replace(update, alpha=0.025) for name, update in updates_add.items()}
        half_bank = EngramBank(tmp_root / record_id / "half")
        for suffix in ["a", "b"]:
            half_bank.save_edit(
                edit_id=f"{record_id}_half_{suffix}",
                metadata={"record_id": record_id, "source_record_id": record_id, "source_request_ids": [record_id]},
                updates=half_updates,
                overwrite=True,
            )
        _restore_modules(wrapper, base_snapshot)
        _apply_composed_delta(wrapper, half_bank.compose_updates([f"{record_id}_half_a", f"{record_id}_half_b"]))
        half_state = _snapshot_modules(wrapper, module_names)
        half_metric = _metric_delta(baseline, _eval_pair(wrapper, record, image_root))
        _apply_composed_delta(wrapper, half_bank.compose_updates([f"{record_id}_half_a", f"{record_id}_half_b"]), scale=-1.0)
        half_rollback = _max_snapshot_diff(wrapper, base_snapshot)

        meta = bank.load_edit(f"{record_id}_add005")["metadata"]
        rows.append(
            {
                "case_index": case_index,
                "record_id": record_id,
                "metadata_direction_sign": meta.get("direction_sign"),
                "metadata_alpha": meta.get("alpha"),
                "metadata_effective_update_norm_ratio": meta.get("effective_update_norm_ratio"),
                "max_abs_param_diff_direct_vs_bank": _max_between_snapshots(direct_state, bank_state),
                "max_abs_param_diff_direct_vs_compose_single": _max_between_snapshots(direct_state, compose_state),
                "max_abs_param_diff_direct_vs_compose_two_half": _max_between_snapshots(direct_state, half_state),
                "target_nll_after_direct": direct_metric["target_nll_after"],
                "target_nll_after_bank": bank_metric["target_nll_after"],
                "target_nll_after_compose_single": compose_metric["target_nll_after"],
                "target_nll_after_compose_two_half": half_metric["target_nll_after"],
                "reference_nll_after_direct": direct_metric["reference_nll_after"],
                "reference_nll_after_bank": bank_metric["reference_nll_after"],
                "reference_nll_after_compose_single": compose_metric["reference_nll_after"],
                "reference_nll_after_compose_two_half": half_metric["reference_nll_after"],
                "rollback_direct": direct_rollback,
                "rollback_bank": bank_rollback,
                "rollback_compose_single": compose_rollback,
                "rollback_compose_two_half": half_rollback,
                "within_tolerance": (
                    _max_between_snapshots(direct_state, bank_state) <= tolerance
                    and _max_between_snapshots(direct_state, compose_state) <= tolerance
                    and _max_between_snapshots(direct_state, half_state) <= tolerance
                ),
            }
        )
        _restore_modules(wrapper, base_snapshot)
    output = {"status": "pass" if all(row["within_tolerance"] for row in rows) else "fail", "tolerance": tolerance, "rows": rows}
    _write_json(out_dir / "bank_compose_equivalence.json", output)
    _write_csv(out_dir / "bank_compose_equivalence.csv", rows)
    return rows


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().float().flatten()
    b = right.detach().float().flatten()
    denom = torch.clamp(a.norm() * b.norm(), min=1.0e-30)
    return float(torch.dot(a, b).div(denom).detach().cpu())


def _run_reextraction(
    wrapper: torch.nn.Module,
    tok: Any,
    hparams: EngramMultimodalHparams,
    records: List[Dict[str, Any]],
    image_root: Path,
    out_dir: Path,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    for case_index, record in enumerate(records):
        record_id = str(record.get("id"))
        repeats = []
        for repeat in range(3):
            updates, metadata = _extract_updates(wrapper, tok, hparams, record, image_root, seed)
            repeats.append({"updates": updates, "metadata": metadata})
        for module_name in repeats[0]["updates"]:
            weights = [item["updates"][module_name].weight for item in repeats]
            projectors = [item["updates"][module_name].projector for item in repeats]
            stats = [item["updates"][module_name].stats for item in repeats]
            pair_max = max((weights[i] - weights[j]).abs().max().item() for i in range(3) for j in range(i + 1, 3))
            pair_cos = min(_cosine(weights[i], weights[j]) for i in range(3) for j in range(i + 1, 3))
            projector_max = None
            projector_cos = None
            if all(p is not None for p in projectors):
                projector_max = max((projectors[i] - projectors[j]).abs().max().item() for i in range(3) for j in range(i + 1, 3))
                projector_cos = min(_cosine(projectors[i], projectors[j]) for i in range(3) for j in range(i + 1, 3))
            rows.append(
                {
                    "case_index": case_index,
                    "record_id": record_id,
                    "module_name": module_name,
                    "target_activation_counts": [s.get("num_target_vectors") for s in stats],
                    "reference_activation_counts": [s.get("num_reference_vectors") for s in stats],
                    "rank_plus": [s.get("rank_plus") for s in stats],
                    "rank_total": [s.get("rank_total") for s in stats],
                    "norm_E": [s.get("norm_E") for s in stats],
                    "norm_ratio": [s.get("norm_ratio") for s in stats],
                    "E_min_pairwise_cosine": pair_cos,
                    "E_max_abs_diff": pair_max,
                    "P_min_pairwise_cosine": projector_cos,
                    "P_max_abs_diff": projector_max,
                }
            )
        details[record_id] = [
            {
                "metadata": item["metadata"],
                "checksums": {name: _update_checksum(update) for name, update in item["updates"].items()},
            }
            for item in repeats
        ]
    output = {
        "status": "pass" if all(float(row["E_max_abs_diff"]) == 0.0 for row in rows) else "non_identical",
        "rows": rows,
        "repeat_details": details,
    }
    _write_json(out_dir / "reextraction_reproducibility.json", output)
    _write_csv(out_dir / "reextraction_reproducibility.csv", rows)
    return rows


def _aggregate_jitter(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_record: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        entry = by_record.setdefault(row["record_id"], {"target_nll": [], "reference_nll": [], "target_logprob": [], "reference_logprob": []})
        for key in entry:
            if row.get(key) is not None:
                entry[key].append(float(row[key]))
    per_record = []
    for record_id, values in by_record.items():
        per_record.append(
            {
                "record_id": record_id,
                "target_nll_std": pstdev(values["target_nll"]) if len(values["target_nll"]) > 1 else 0.0,
                "reference_nll_std": pstdev(values["reference_nll"]) if len(values["reference_nll"]) > 1 else 0.0,
                "target_logprob_std": pstdev(values["target_logprob"]) if len(values["target_logprob"]) > 1 else 0.0,
                "reference_logprob_std": pstdev(values["reference_logprob"]) if len(values["reference_logprob"]) > 1 else 0.0,
            }
        )
    return {
        "per_record": per_record,
        "mean_target_nll_std": sum(r["target_nll_std"] for r in per_record) / len(per_record),
        "max_target_nll_std": max(r["target_nll_std"] for r in per_record),
        "mean_reference_nll_std": sum(r["reference_nll_std"] for r in per_record) / len(per_record),
        "max_reference_nll_std": max(r["reference_nll_std"] for r in per_record),
    }


def _run_metric_jitter(
    wrapper: torch.nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    out_dir: Path,
    repeats: int,
) -> Dict[str, Any]:
    rows = []
    for repeat in range(repeats):
        for case_index, record in enumerate(records):
            metrics = _eval_pair(wrapper, record, image_root)
            target = _strip(metrics["target"])
            reference = _strip(metrics.get("reference") or {})
            rows.append(
                {
                    "repeat": repeat,
                    "case_index": case_index,
                    "record_id": record.get("id"),
                    "target_nll": None if target is None else target["nll"],
                    "target_logprob": None if target is None else target["logprob"],
                    "reference_nll": None if reference is None else reference["nll"],
                    "reference_logprob": None if reference is None else reference["logprob"],
                }
            )
    aggregate = _aggregate_jitter(rows)
    output = {"status": "complete", "repeats": repeats, "aggregate": aggregate, "rows": rows}
    _write_json(out_dir / "metric_jitter.json", output)
    _write_csv(out_dir / "metric_jitter.csv", rows)
    return aggregate


def _run_repeated_signed_effect(
    wrapper: torch.nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    extracted: Dict[str, Dict[str, EngramLayerUpdate]],
    out_dir: Path,
    repeats: int,
) -> List[Dict[str, Any]]:
    rows = []
    for repeat in range(repeats):
        for case_index, record in enumerate(records):
            record_id = str(record.get("id"))
            updates = extracted[record_id]
            module_names = list(updates)
            base_snapshot = _snapshot_modules(wrapper, module_names)
            baseline = _eval_pair(wrapper, record, image_root)
            for mode in ["subtract_negative", "add_positive"]:
                mode_updates = _mode_updates(updates, mode)
                _restore_modules(wrapper, base_snapshot)
                _apply_updates(wrapper, mode_updates, direction=-1)
                delta = _metric_delta(baseline, _eval_pair(wrapper, record, image_root))
                _apply_updates(wrapper, mode_updates, direction=1)
                rows.append(
                    {
                        "repeat": repeat,
                        "case_index": case_index,
                        "record_id": record_id,
                        "mode": mode,
                        "target_nll_increase": delta["target_nll_increase"],
                        "target_logprob_drop": delta["target_logprob_drop"],
                        "reference_delta_abs": delta["reference_delta_abs"],
                        "rollback_max_abs_diff": _max_snapshot_diff(wrapper, base_snapshot),
                    }
                )
                _restore_modules(wrapper, base_snapshot)
    output = {"status": "complete", "repeats": repeats, "rows": rows}
    _write_json(out_dir / "repeated_signed_effect.json", output)
    _write_csv(out_dir / "repeated_signed_effect.csv", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict ENGRAM sign-equivalence audit.")
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default="outputs/engram_sign_equivalence_audit")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-edits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    _set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(Path(args.data_file), args.max_edits)
    image_root = Path(args.image_root)
    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    hparams.device = int(args.device) if str(args.device).isdigit() else args.device
    hparams.coco_image = ""
    hparams.rephrase_image = ""

    editor = MultimodalEditor.from_hparams(hparams)
    wrapper = editor.model
    wrapper.eval()

    extracted, single_rows = _run_single_e(wrapper, editor.tok, hparams, records, image_root, out_dir, args.seed, args.tolerance)
    bank_rows = _run_bank_compose(wrapper, records, image_root, extracted, out_dir, args.tolerance)
    reextract_rows = _run_reextraction(wrapper, editor.tok, hparams, records, image_root, out_dir, args.seed)
    jitter = _run_metric_jitter(wrapper, records, image_root, out_dir, args.repeats)
    repeated_rows = _run_repeated_signed_effect(wrapper, records, image_root, extracted, out_dir, args.repeats)
    summary = {
        "status": "complete",
        "single_E_status": "pass" if all(row["within_tolerance"] for row in single_rows) else "fail",
        "bank_compose_status": "pass" if all(row["within_tolerance"] for row in bank_rows) else "fail",
        "reextraction_status": "pass" if all(float(row["E_max_abs_diff"]) == 0.0 for row in reextract_rows) else "non_identical",
        "metric_jitter": jitter,
        "repeated_signed_effect_rows": len(repeated_rows),
    }
    _write_json(out_dir / "audit_sign_equivalence_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
