#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from easyeditor.dataset.coco_caption import CaptionDataset  # noqa: E402
from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.engram_main import select_linear_layers  # noqa: E402
from scripts.engram.run_token_module_ablation_5edit import (  # noqa: E402
    _answer_metrics,
    _apply_add_alpha,
    _finite,
    _format,
    _json_dump,
    _max_snapshot_diff,
    _maybe_generate,
    _mean,
    _module_map,
    _reference_sample,
    _resolve_image,
    _restore_modules,
    _restore_weight_copy,
    _sample,
    _snapshot_modules,
    _strip,
    _write_csv,
)


EXPECTED_MODULES = [
    "llava_model.model.layers.0.mlp.gate_proj",
    "llava_model.model.layers.8.mlp.gate_proj",
    "llava_model.model.layers.16.mlp.gate_proj",
    "llava_model.model.layers.24.mlp.gate_proj",
    "llava_model.model.layers.16.self_attn.q_proj",
    "llava_model.model.layers.24.self_attn.q_proj",
    "llava_model.model.layers.16.self_attn.k_proj",
    "llava_model.model.layers.24.self_attn.k_proj",
]

SYNTHETIC_REPLACEMENTS = {
    "synthetic-5edit-1": "synthetic-code-1",
    "synthetic-5edit-2": "synthetic-code-2",
    "synthetic-5edit-3": "synthetic-code-3",
    "synthetic-5edit-4": "synthetic-code-4",
    "synthetic-5edit-5": "synthetic-code-5",
}


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 5:
        raise RuntimeError(f"Expected exactly 5 records at {path}, got {len(records) if isinstance(records, list) else type(records)}")
    return records


def _run_capture(command: List[str], cwd: Path) -> str:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.stdout


def _write_git_outputs(out_dir: Path) -> None:
    (out_dir / "git_status.txt").write_text(_run_capture(["git", "status"], PROJECT_ROOT), encoding="utf-8")
    (out_dir / "git_diff.patch").write_text(_run_capture(["git", "diff"], PROJECT_ROOT), encoding="utf-8")


def _exact_module_patterns() -> List[str]:
    return [rf"^{re.escape(name)}$" for name in EXPECTED_MODULES]


def _new_answer(record: Dict[str, Any]) -> str:
    return str(record.get("new_answer") or record.get("replacement_answer") or record.get("alt") or "")


def _old_answer(record: Dict[str, Any]) -> str:
    return str(record.get("old_answer") or record.get("erase_answer") or record.get("pred") or "")


def _old_sample(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    return _sample(record["src"], _old_answer(record), _resolve_image(image_root, record["image"]))


def _new_sample(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    return _sample(record["src"], _new_answer(record), _resolve_image(image_root, record["image"]))


def _metric_value(raw: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not raw or not raw.get("available") or raw.get(key) is None:
        return None
    return float(raw[key])


def _delta(after: Optional[float], before: Optional[float]) -> Optional[float]:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den in (None, 0.0):
        return None
    return float(num) / float(den)


def _configure_hparams(
    hparams: EngramMultimodalHparams,
    *,
    image_root: Path,
    bank_dir: Path,
    device: str,
    edit_mode: str,
) -> None:
    hparams.device = int(device) if str(device).isdigit() else device
    dataset_image_root = image_root.parent if image_root.name == "images" else image_root
    hparams.coco_image = str(dataset_image_root)
    hparams.rephrase_image = str(dataset_image_root)
    hparams.edit_mode = edit_mode
    hparams.token_scope = "all"
    hparams.engram_update_direction = "add"
    hparams.alpha = 0.0
    hparams.beta = 1.0
    hparams.covariance_device = "cpu"
    hparams.solve_device = "cpu"
    hparams.max_cov_dim = 4097
    hparams.skip_if_dim_larger_than = 4097
    hparams.module_patterns = _exact_module_patterns()
    hparams.exclude_module_patterns = [r"lm_head$", r"down_proj$"]
    hparams.prioritize_module_selection = True
    hparams.module_priority_patterns = [r"gate_proj$", r"q_proj$", r"k_proj$"]
    hparams.engram_layers = [0, 8, 16, 24]
    hparams.engram_max_modules = None
    hparams.sequential_edit = False
    hparams.bank_dir = str(bank_dir)
    hparams.engram_bank_path = str(bank_dir)
    hparams.edit_id = None
    hparams.engram_edit_id = None


def _write_failure_summary(out_dir: Path, sequential_report: Path, best_config_path: Path) -> Dict[str, Any]:
    text = sequential_report.read_text(encoding="utf-8") if sequential_report.exists() else ""
    best = json.loads(best_config_path.read_text(encoding="utf-8")) if best_config_path.exists() else {}
    summary = {
        "sequential_report_path": str(sequential_report),
        "sequential_report_found": sequential_report.exists(),
        "best_nonseq_config_path": str(best_config_path),
        "best_nonseq_config_found": best_config_path.exists(),
        "best_nonseq_token_scope": best.get("token_scope"),
        "best_nonseq_module_group": best.get("module_group"),
        "best_nonseq_alpha": best.get("alpha"),
        "best_nonseq_mean_target_nll_increase": best.get("mean_target_nll_increase"),
        "best_nonseq_mean_reference_delta_abs": best.get("mean_reference_delta_abs"),
        "best_nonseq_positive_target_edits": best.get("positive_target_edits"),
        "best_nonseq_locality_damage_edits": best.get("locality_damage_edits"),
        "sequential_decision_mentions_do_not_scale": "do not scale" in text.lower() or "decision c" in text.lower(),
    }
    lines = [
        "# Direct Erase Failure Summary",
        "",
        "This summary is generated from the saved 5-edit sequential smoke report and the best non-sequential direct-erase config.",
        "",
        "## Prior Non-Sequential Direct Erase",
        "",
        f"- token_scope: `{summary['best_nonseq_token_scope']}`",
        f"- module_group: `{summary['best_nonseq_module_group']}`",
        f"- alpha: `{summary['best_nonseq_alpha']}`",
        f"- mean target NLL increase: `{_format(summary['best_nonseq_mean_target_nll_increase'])}`",
        f"- mean reference delta abs: `{_format(summary['best_nonseq_mean_reference_delta_abs'])}`",
        f"- positive target edits: `{summary['best_nonseq_positive_target_edits']}`",
        f"- locality damage edits: `{summary['best_nonseq_locality_damage_edits']}`",
        "",
        "## Sequential Smoke Failure Mode",
        "",
        "The saved sequential smoke report is treated as the source of truth for direct erase scaling.",
        "It showed enough target-side signal to justify investigation, but locality damage and empty generation made direct erase unsuitable for scaling as-is.",
        "",
        "## Consequence For This Run",
        "",
        "This runner does not launch 20-edit. It tests a localized replacement candidate on the same 5 synthetic records before any sequential scaling.",
        "",
    ]
    (out_dir / "DIRECT_ERASE_FAILURE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    _json_dump(out_dir / "direct_erase_failure_summary.json", summary)
    return summary


def _prepare_replacement_data(source_data: Path, source_image_root: Path, out_dir: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    records = _load_records(source_data)
    image_root = out_dir / "synthetic_root" / "data" / "medmkeb" / "images"
    raw_dir = out_dir / "synthetic_root" / "data" / "medmkeb" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if image_root.is_symlink() or image_root.is_file():
        image_root.unlink()
    elif image_root.exists():
        shutil.rmtree(image_root)
    source_image_root = source_image_root.resolve()
    image_materialization = "symlink"
    try:
        image_root.symlink_to(source_image_root, target_is_directory=True)
    except OSError:
        image_materialization = "copy_without_appledouble"
        shutil.copytree(
            source_image_root,
            image_root,
            ignore=lambda _src, names: [name for name in names if name.startswith("._")],
        )

    replacement_records: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        record_id = str(record.get("id") or f"synthetic-5edit-{idx}")
        old = str(record.get("erase_answer") or record.get("pred") or record.get("alt") or "")
        new = SYNTHETIC_REPLACEMENTS.get(record_id, f"synthetic-code-{idx}")
        copied = dict(record)
        copied["id"] = record_id
        copied["old_answer"] = old
        copied["new_answer"] = new
        copied["replacement_answer"] = new
        copied["alt"] = new
        copied["pred"] = old
        copied["erase_answer"] = old
        copied["synthetic_replacement_non_phi"] = True
        copied["synthetic_replacement_note"] = "Synthetic non-PHI target used only for ENGRAM replacement mechanics."
        replacement_records.append(copied)

        paths = {
            "image": _resolve_image(image_root, copied["image"]),
            "image_rephrase": _resolve_image(image_root, copied["image_rephrase"]),
            "m_loc": _resolve_image(image_root, copied["m_loc"]),
        }
        rows.append(
            {
                "record_id": record_id,
                "old_answer": old,
                "new_answer": new,
                "x_minus_non_empty": bool(copied.get("m_loc_q") and copied.get("m_loc_a") and copied.get("m_loc")),
                "image_paths_resolve": all(Path(path).exists() for path in paths.values()),
                "resolved_paths": paths,
            }
        )

    replacement_path = raw_dir / "engram_replacement_5edit.json"
    replacement_path.write_text(json.dumps(replacement_records, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "status": "pass"
        if len(rows) == 5
        and all(row["record_id"] and row["old_answer"] and row["new_answer"] for row in rows)
        and all(row["x_minus_non_empty"] and row["image_paths_resolve"] for row in rows)
        else "fail",
        "source_data_file": str(source_data),
        "replacement_data_file": str(replacement_path),
        "image_root": str(image_root),
        "source_image_root": str(source_image_root),
        "image_materialization": image_materialization,
        "record_count": len(rows),
        "records": rows,
        "original_data_modified": False,
        "private_or_patient_data_used": False,
    }
    _json_dump(out_dir / "replacement_data_summary.json", summary)
    if summary["status"] != "pass":
        raise RuntimeError(f"Replacement data preflight failed: {summary}")
    return replacement_path, image_root, summary


class TinyLoraTrainingPatch:
    def __init__(self, model: nn.Module, module_names: Iterable[str], *, rank: int, scale: float) -> None:
        self.model = model
        self.module_names = list(module_names)
        self.rank = int(rank)
        self.scale = float(scale)
        self.original_forwards: Dict[str, Any] = {}
        self.factors: Dict[str, Dict[str, torch.nn.Parameter]] = {}
        self.requires_grad: List[Tuple[torch.nn.Parameter, bool]] = []

    def install(self) -> None:
        modules = _module_map(self.model)
        for param in self.model.parameters():
            self.requires_grad.append((param, bool(param.requires_grad)))
            param.requires_grad_(False)
        for name in self.module_names:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"LoRA target is not nn.Linear: {name}")
            device = module.weight.device
            in_features = int(module.in_features)
            out_features = int(module.out_features)
            a = torch.nn.Parameter(torch.randn(self.rank, in_features, device=device, dtype=torch.float32) * 0.01)
            b = torch.nn.Parameter(torch.zeros(out_features, self.rank, device=device, dtype=torch.float32))
            self.factors[name] = {"A": a, "B": b}
            self.original_forwards[name] = module.forward

            def patched_forward(x, *, _base=module.forward, _a=a, _b=b, _scale=self.scale):
                base = _base(x)
                low = torch.nn.functional.linear(x.to(torch.float32), _a)
                delta = torch.nn.functional.linear(low, _b) * float(_scale)
                return base + delta.to(dtype=base.dtype)

            module.forward = patched_forward  # type: ignore[method-assign]

    def parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for factor in self.factors.values():
            params.extend([factor["A"], factor["B"]])
        return params

    def state(self) -> Dict[str, Dict[str, torch.Tensor | float]]:
        return {
            name: {
                "A": factor["A"].detach().cpu().clone(),
                "B": factor["B"].detach().cpu().clone(),
                "scale": self.scale,
            }
            for name, factor in self.factors.items()
        }

    def remove(self) -> None:
        modules = _module_map(self.model)
        for name, forward in reversed(list(self.original_forwards.items())):
            modules[name].forward = forward  # type: ignore[method-assign]
        for param, value in self.requires_grad:
            param.requires_grad_(value)
        self.original_forwards.clear()
        self.requires_grad.clear()


class EvalLoraPatch:
    def __init__(self, model: nn.Module, factors: Dict[str, Dict[str, torch.Tensor | float]], *, beta: float) -> None:
        self.model = model
        self.factors = factors
        self.beta = float(beta)
        self.original_forwards: Dict[str, Any] = {}

    def install(self) -> None:
        modules = _module_map(self.model)
        for name, factor in self.factors.items():
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"LoRA eval target is not nn.Linear: {name}")
            a = factor["A"].to(module.weight.device, dtype=torch.float32)  # type: ignore[union-attr]
            b = factor["B"].to(module.weight.device, dtype=torch.float32)  # type: ignore[union-attr]
            scale = float(factor.get("scale", 1.0)) * self.beta  # type: ignore[union-attr]
            self.original_forwards[name] = module.forward

            def patched_forward(x, *, _base=module.forward, _a=a, _b=b, _scale=scale):
                base = _base(x)
                low = torch.nn.functional.linear(x.to(torch.float32), _a)
                delta = torch.nn.functional.linear(low, _b) * float(_scale)
                return base + delta.to(dtype=base.dtype)

            module.forward = patched_forward  # type: ignore[method-assign]

    def remove(self) -> None:
        modules = _module_map(self.model)
        for name, forward in reversed(list(self.original_forwards.items())):
            modules[name].forward = forward  # type: ignore[method-assign]
        self.original_forwards.clear()


def _train_tiny_lora(
    model: nn.Module,
    record: Dict[str, Any],
    image_root: Path,
    module_names: List[str],
    *,
    rank: int,
    steps: int,
    lr: float,
    scale: float,
    lambda_ref: float,
) -> Tuple[Dict[str, Dict[str, torch.Tensor | float]], Dict[str, Any]]:
    patch = TinyLoraTrainingPatch(model, module_names, rank=rank, scale=scale)
    target = _new_sample(record, image_root)
    reference = _reference_sample(record, image_root)
    losses: List[Dict[str, Optional[float]]] = []
    patch.install()
    try:
        optimizer = torch.optim.AdamW(patch.parameters(), lr=float(lr), weight_decay=0.0)
        for step in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            output = model(dict(target))
            loss = output.loss
            ref_loss = None
            if lambda_ref > 0.0 and reference is not None:
                ref_output = model(dict(reference))
                ref_loss = ref_output.loss
                loss = loss + float(lambda_ref) * ref_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"tiny LoRA loss became non-finite at step {step}: {loss}")
            loss.backward()
            optimizer.step()
            losses.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "target_loss": float(output.loss.detach().cpu()) if output.loss is not None else None,
                    "reference_loss": float(ref_loss.detach().cpu()) if ref_loss is not None else None,
                }
            )
        state = patch.state()
    finally:
        patch.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state, {"loss_trace": losses, "steps": int(steps), "rank": int(rank), "lr": float(lr), "scale": float(scale)}


def _low_rank_norm(a: torch.Tensor, b: torch.Tensor, scale: float) -> float:
    aa = a.float().matmul(a.float().t())
    bb = b.float().t().matmul(b.float())
    value = torch.trace(bb.matmul(aa)).clamp_min(0.0).sqrt() * abs(float(scale))
    return float(value.detach().cpu())


def _project_factors(
    factors: Dict[str, Dict[str, torch.Tensor | float]],
    projector_edit: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, torch.Tensor | float]], Dict[str, Any]]:
    updates = projector_edit["updates"]
    metadata = projector_edit["metadata"]
    projected: Dict[str, Dict[str, torch.Tensor | float]] = {}
    module_rows: List[Dict[str, Any]] = []
    all_available = True
    for name, factor in factors.items():
        raw = updates.get(name)
        a = factor["A"].detach().cpu().float()  # type: ignore[union-attr]
        b = factor["B"].detach().cpu().float()  # type: ignore[union-attr]
        scale = float(factor.get("scale", 1.0))  # type: ignore[union-attr]
        projector_available = raw is not None and isinstance(raw.get("projector"), torch.Tensor)
        reason = None
        if not projector_available:
            all_available = False
            projected[name] = {"A": a, "B": b, "scale": scale}
            reason = "missing_projector"
            safe_a = a
        else:
            projector = raw["projector"].detach().cpu().float()
            if projector.shape[0] == a.shape[1] and projector.shape[1] == a.shape[1]:
                safe_a = a.matmul(projector)
            elif projector.shape[0] == a.shape[1] + 1 and projector.shape[1] == a.shape[1] + 1:
                safe_a = a.matmul(projector[: a.shape[1], : a.shape[1]])
                reason = "used_top_left_projector_block_for_absorbed_bias"
            else:
                all_available = False
                safe_a = a
                reason = f"projector_shape_mismatch={tuple(projector.shape)} for A={tuple(a.shape)}"
            projected[name] = {"A": safe_a.detach().cpu(), "B": b, "scale": scale}
        candidate_norm = _low_rank_norm(a, b, scale)
        safe_norm = _low_rank_norm(safe_a, b, scale)
        stats = dict(raw.get("stats") or {}) if raw else {}
        module_rows.append(
            {
                "module_name": name,
                "projector_available": bool(projector_available and reason not in {"missing_projector"}),
                "projector_reason": reason,
                "candidate_delta_norm": candidate_norm,
                "projected_delta_norm": safe_norm,
                "projection_norm_ratio": _safe_div(safe_norm, candidate_norm),
                "target_activation_count": int(stats.get("num_target_vectors", 0) or 0),
                "reference_activation_count": int(stats.get("num_reference_vectors", 0) or 0),
                "projector_norm": stats.get("projector_norm"),
            }
        )
    summary = {
        "projector_available": bool(all_available),
        "record_id": metadata.get("record_id") or metadata.get("source_record_id"),
        "edit_id": metadata.get("edit_id"),
        "modules": module_rows,
        "candidate_delta_norm_total": math.sqrt(sum(float(row["candidate_delta_norm"]) ** 2 for row in module_rows)),
        "projected_delta_norm_total": math.sqrt(sum(float(row["projected_delta_norm"]) ** 2 for row in module_rows)),
    }
    summary["projection_norm_ratio_total"] = _safe_div(summary["projected_delta_norm_total"], summary["candidate_delta_norm_total"])
    return projected, summary


def _evaluate_current(
    model: nn.Module,
    record: Dict[str, Any],
    image_root: Path,
    *,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Dict[str, Any]:
    reference = _reference_sample(record, image_root)
    return {
        "old_raw": _answer_metrics(model, _old_sample(record, image_root)),
        "new_raw": _answer_metrics(model, _new_sample(record, image_root)),
        "reference_raw": _answer_metrics(model, dict(reference)) if reference else None,
        "generation": _maybe_generate(model, record, image_root, max_new_tokens, min_new_tokens, skip_generation),
    }


def _make_eval_row(
    *,
    method: str,
    record: Dict[str, Any],
    case_index: int,
    before: Dict[str, Any],
    after: Dict[str, Any],
    rollback_diff: float,
    rollback_tolerance: float,
    locality_threshold: float,
    record_id_match_rate: float,
    edit_id: Optional[str] = None,
    beta: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    old_before = _strip(before["old_raw"])
    old_after = _strip(after["old_raw"])
    new_before = _strip(before["new_raw"])
    new_after = _strip(after["new_raw"])
    ref_before = _strip(before["reference_raw"])
    ref_after = _strip(after["reference_raw"])
    old_inc = _delta(_metric_value(after["old_raw"], "nll"), _metric_value(before["old_raw"], "nll"))
    new_delta = _delta(_metric_value(after["new_raw"], "nll"), _metric_value(before["new_raw"], "nll"))
    new_decrease = None if new_delta is None else -new_delta
    ref_delta = _delta(_metric_value(after["reference_raw"], "nll"), _metric_value(before["reference_raw"], "nll"))
    ref_abs = None if ref_delta is None else abs(ref_delta)
    row = {
        "method": method,
        "record_id": str(record.get("id")),
        "case_index": int(case_index),
        "edit_id": edit_id,
        "old_answer": _old_answer(record),
        "new_answer": _new_answer(record),
        "beta": beta,
        "old_answer_nll_before": old_before.get("nll") if old_before else None,
        "old_answer_nll_after": old_after.get("nll") if old_after else None,
        "old_answer_nll_increase": old_inc,
        "new_answer_nll_before": new_before.get("nll") if new_before else None,
        "new_answer_nll_after": new_after.get("nll") if new_after else None,
        "new_answer_nll_decrease": new_decrease,
        "reference_nll_before": ref_before.get("nll") if ref_before else None,
        "reference_nll_after": ref_after.get("nll") if ref_after else None,
        "reference_delta": ref_delta,
        "reference_delta_abs": ref_abs,
        "target_success": bool(new_decrease is not None and new_decrease > 0.0),
        "erase_success": bool(old_inc is not None and old_inc > 0.0),
        "locality_damage": bool(ref_abs is not None and ref_abs > float(locality_threshold)),
        "generation_before": before["generation"],
        "generation_after": after["generation"],
        "generation_empty": (
            after["generation"].get("generation_empty")
            if isinstance(after.get("generation"), dict)
            else None
        ),
        "rollback_max_abs_diff": rollback_diff,
        "rollback_pass": bool(rollback_diff <= float(rollback_tolerance)),
        "record_id_match_rate": float(record_id_match_rate),
        "old_raw_before": before["old_raw"],
        "old_raw_after": after["old_raw"],
        "new_raw_before": before["new_raw"],
        "new_raw_after": after["new_raw"],
        "reference_raw_before": before["reference_raw"],
        "reference_raw_after": after["reference_raw"],
    }
    if extra:
        row.update(extra)
    row["nan_inf_detected"] = not _finite(row)
    return row


def _aggregate_method(rows: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    metric_rows = [row for row in rows if row.get("method") == method]
    new_decreases = [float(row["new_answer_nll_decrease"]) for row in metric_rows if row.get("new_answer_nll_decrease") is not None]
    old_increases = [float(row["old_answer_nll_increase"]) for row in metric_rows if row.get("old_answer_nll_increase") is not None]
    ref_abs = [float(row["reference_delta_abs"]) for row in metric_rows if row.get("reference_delta_abs") is not None]
    mean_new = _mean(new_decreases)
    mean_ref = _mean(ref_abs)
    return {
        "method": method,
        "status": "complete" if metric_rows else "skipped",
        "record_count": len(metric_rows),
        "mean_new_answer_nll_decrease": mean_new,
        "mean_old_answer_nll_increase": _mean(old_increases),
        "mean_ref_delta": mean_ref,
        "positive_new_answer_edits": sum(1 for value in new_decreases if value > 0.0),
        "positive_old_answer_erasure_edits": sum(1 for value in old_increases if value > 0.0),
        "locality_damage_edits": sum(1 for row in metric_rows if row.get("locality_damage")),
        "target_to_reference_ratio": _safe_div(mean_new, mean_ref),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in metric_rows]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in metric_rows]),
        "nan_inf_count": sum(1 for row in metric_rows if row.get("nan_inf_detected")),
        "empty_generation_count": sum(1 for row in metric_rows if row.get("generation_empty")),
    }


def _evaluate_no_edit(
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    module_names: List[str],
    *,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> List[Dict[str, Any]]:
    snapshots = _snapshot_modules(model, module_names)
    rows = []
    for idx, record in enumerate(records):
        after = _evaluate_current(
            model,
            record,
            image_root,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            skip_generation=skip_generation,
        )
        rows.append(
            _make_eval_row(
                method="A_no_edit_baseline",
                record=record,
                case_index=idx,
                before=baselines[str(record["id"])],
                after=after,
                rollback_diff=_max_snapshot_diff(model, snapshots),
                rollback_tolerance=rollback_tolerance,
                locality_threshold=locality_threshold,
                record_id_match_rate=1.0,
                beta=0.0,
            )
        )
    _restore_modules(model, snapshots)
    return rows


def _evaluate_direct_erase(
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    module_names: List[str],
    bank_dir: Path,
    *,
    alpha: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not bank_dir.exists():
        return [], {"status": "skipped", "reason": f"direct erase bank not found: {bank_dir}", "bank_dir": str(bank_dir)}
    bank = EngramBank(bank_dir)
    try:
        edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    except Exception as exc:
        return [], {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "bank_dir": str(bank_dir)}
    snapshots = _snapshot_modules(model, module_names)
    rows: List[Dict[str, Any]] = []
    try:
        for idx, (record, edit_id) in enumerate(zip(records, edit_ids)):
            _restore_modules(model, snapshots)
            edit = bank.load_edit(edit_id)
            _apply_add_alpha(model, edit["updates"], alpha)
            after = _evaluate_current(
                model,
                record,
                image_root,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                skip_generation=skip_generation,
            )
            _restore_modules(model, snapshots)
            rows.append(
                _make_eval_row(
                    method="B_direct_engram_erase",
                    record=record,
                    case_index=idx,
                    before=baselines[str(record["id"])],
                    after=after,
                    rollback_diff=_max_snapshot_diff(model, snapshots),
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    record_id_match_rate=1.0 if matching.get("mode") == "record_id" else 0.0,
                    edit_id=edit_id,
                    beta=alpha,
                    extra={"engram_update_direction": "add", "direct_erase_alpha": alpha},
                )
            )
    finally:
        _restore_modules(model, snapshots)
    return rows, {"status": "complete", "bank_dir": str(bank_dir), "edit_record_matching": matching, "edit_ids": edit_ids}


def _extract_projector_bank(
    editor: MultimodalEditor,
    hparams: EngramMultimodalHparams,
    data_file: Path,
    records: List[Dict[str, Any]],
    bank_dir: Path,
) -> Dict[str, Any]:
    if bank_dir.exists():
        shutil.rmtree(bank_dir)
    ds = CaptionDataset(str(data_file), config=hparams)
    if len(ds) != len(records):
        raise RuntimeError(f"CaptionDataset length mismatch: {len(ds)} vs raw {len(records)}")
    extracted: List[Dict[str, Any]] = []
    for idx, request in enumerate(ds):
        record_id = str(records[idx]["id"])
        request["id"] = record_id
        request["record_id"] = record_id
        request["source_record_id"] = record_id
        hparams.edit_id = f"projector__{record_id}"
        weights_copy = None
        try:
            _, weights_copy = editor.apply_algo(
                editor.model,
                editor.tok,
                [request],
                hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=True,
                train_ds=None,
            )
        finally:
            if weights_copy:
                _restore_weight_copy(editor.model, weights_copy, hparams.device)
        extracted.append({"record_id": record_id, "edit_id": hparams.edit_id})
    bank = EngramBank(bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    return {"status": "complete", "bank_dir": str(bank_dir), "extracted": extracted, "edit_ids": edit_ids, "edit_record_matching": matching}


def _metadata_from_projection(
    *,
    record: Dict[str, Any],
    edit_id: str,
    lora_train_summary: Dict[str, Any],
    projection_summary: Dict[str, Any],
    hparams: EngramMultimodalHparams,
) -> Dict[str, Any]:
    return {
        "record_id": str(record["id"]),
        "edit_id": edit_id,
        "edit_mode": "replacement",
        "candidate_delta_source": "tiny_lora",
        "project_delta_with_engram": True,
        "replacement_beta": float(hparams.replacement_beta),
        "replacement_lambda_ref": float(hparams.replacement_lambda_ref),
        "lora_rank": int(hparams.lora_rank),
        "lora_steps": int(hparams.lora_steps),
        "lora_lr": float(hparams.lora_lr),
        "selected_modules": EXPECTED_MODULES,
        "projection": projection_summary,
        "lora_train": lora_train_summary,
    }


def _evaluate_lora_methods(
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    projector_extract: Dict[str, Any],
    module_names: List[str],
    hparams: EngramMultimodalHparams,
    out_dir: Path,
    *,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    bank = EngramBank(projector_bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    snapshots = _snapshot_modules(model, module_names)
    beta = float(hparams.replacement_beta)
    scale = float(hparams.lora_scale if hparams.lora_scale is not None else 1.0)
    per_record_metadata: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for idx, (record, edit_id) in enumerate(zip(records, edit_ids)):
        _restore_modules(model, snapshots)
        factors, train_summary = _train_tiny_lora(
            model,
            record,
            image_root,
            module_names,
            rank=int(hparams.lora_rank),
            steps=int(hparams.lora_steps),
            lr=float(hparams.lora_lr),
            scale=scale,
            lambda_ref=float(hparams.replacement_lambda_ref),
        )
        projector_edit = bank.load_edit(edit_id)
        safe_factors, projection_summary = _project_factors(factors, projector_edit)
        metadata = _metadata_from_projection(
            record=record,
            edit_id=edit_id,
            lora_train_summary=train_summary,
            projection_summary=projection_summary,
            hparams=hparams,
        )
        per_record_metadata.append(metadata)

        patch = EvalLoraPatch(model, factors, beta=beta)
        patch.install()
        try:
            after_unprojected = _evaluate_current(
                model,
                record,
                image_root,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                skip_generation=skip_generation,
            )
        finally:
            patch.remove()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(
            _make_eval_row(
                method="C_unprojected_tiny_lora_replacement",
                record=record,
                case_index=idx,
                before=baselines[str(record["id"])],
                after=after_unprojected,
                rollback_diff=_max_snapshot_diff(model, snapshots),
                rollback_tolerance=rollback_tolerance,
                locality_threshold=locality_threshold,
                record_id_match_rate=1.0 if matching.get("mode") == "record_id" else 0.0,
                edit_id=edit_id,
                beta=beta,
                extra={
                    "lora_rank": int(hparams.lora_rank),
                    "lora_steps": int(hparams.lora_steps),
                    "lora_lr": float(hparams.lora_lr),
                    "project_delta_with_engram": False,
                    "candidate_delta_norm_total": projection_summary["candidate_delta_norm_total"],
                },
            )
        )

        patch = EvalLoraPatch(model, safe_factors, beta=beta)
        patch.install()
        try:
            after_projected = _evaluate_current(
                model,
                record,
                image_root,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                skip_generation=skip_generation,
            )
        finally:
            patch.remove()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(
            _make_eval_row(
                method="D_engram_projected_tiny_lora_replacement",
                record=record,
                case_index=idx,
                before=baselines[str(record["id"])],
                after=after_projected,
                rollback_diff=_max_snapshot_diff(model, snapshots),
                rollback_tolerance=rollback_tolerance,
                locality_threshold=locality_threshold,
                record_id_match_rate=1.0 if matching.get("mode") == "record_id" else 0.0,
                edit_id=edit_id,
                beta=beta,
                extra={
                    "lora_rank": int(hparams.lora_rank),
                    "lora_steps": int(hparams.lora_steps),
                    "lora_lr": float(hparams.lora_lr),
                    "project_delta_with_engram": True,
                    "projector_available": bool(projection_summary["projector_available"]),
                    "candidate_delta_norm_total": projection_summary["candidate_delta_norm_total"],
                    "projected_delta_norm_total": projection_summary["projected_delta_norm_total"],
                    "projection_norm_ratio_total": projection_summary["projection_norm_ratio_total"],
                },
            )
        )
        _restore_modules(model, snapshots)
        _json_dump(out_dir / "replacement_bank_metadata" / f"{record['id']}.json", metadata)
    _restore_modules(model, snapshots)
    return rows, per_record_metadata


def _projected_acceptance(aggregates: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method = {row["method"]: row for row in aggregates}
    projected = by_method.get("D_engram_projected_tiny_lora_replacement", {})
    unprojected = by_method.get("C_unprojected_tiny_lora_replacement", {})
    checks = {
        "positive_new_at_least_4_of_5": int(projected.get("positive_new_answer_edits") or 0) >= 4,
        "mean_new_decrease_positive": (projected.get("mean_new_answer_nll_decrease") is not None)
        and float(projected["mean_new_answer_nll_decrease"]) > 0.0,
        "mean_ref_delta_less_than_mean_new_decrease": (
            projected.get("mean_ref_delta") is not None
            and projected.get("mean_new_answer_nll_decrease") is not None
            and float(projected["mean_ref_delta"]) < float(projected["mean_new_answer_nll_decrease"])
        ),
        "locality_damage_less_than_unprojected_lora": int(projected.get("locality_damage_edits") or 0)
        < int(unprojected.get("locality_damage_edits") or 0),
        "rollback_rate_is_1": float(projected.get("rollback_pass_rate") or 0.0) == 1.0,
        "record_id_match_rate_is_1": float(projected.get("record_id_match_rate") or 0.0) == 1.0,
        "no_nan": int(projected.get("nan_inf_count") or 0) == 0,
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks, "projected": projected, "unprojected": unprojected}


def _run_sequential_if_pass(
    model: nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    module_names: List[str],
    hparams: EngramMultimodalHparams,
    out_dir: Path,
    acceptance: Dict[str, Any],
    *,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Dict[str, Any]:
    seq_dir = out_dir / "sequential"
    seq_dir.mkdir(parents=True, exist_ok=True)
    if acceptance.get("status") != "pass":
        payload = {"status": "skipped", "reason": "non-sequential projected replacement did not pass acceptance", "acceptance": acceptance}
        _json_dump(seq_dir / "sequential_replacement_skipped.json", payload)
        return payload

    beta = float(hparams.replacement_beta) / 2.0
    scale = float(hparams.lora_scale if hparams.lora_scale is not None else 1.0)
    snapshots = _snapshot_modules(model, module_names)
    bank = EngramBank(projector_bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    active_patches: List[EvalLoraPatch] = []
    rows: List[Dict[str, Any]] = []
    try:
        for step, (record, edit_id) in enumerate(zip(records, edit_ids), start=1):
            factors, _ = _train_tiny_lora(
                model,
                record,
                image_root,
                module_names,
                rank=int(hparams.lora_rank),
                steps=int(hparams.lora_steps),
                lr=float(hparams.lora_lr),
                scale=scale,
                lambda_ref=float(hparams.replacement_lambda_ref),
            )
            safe_factors, projection_summary = _project_factors(factors, bank.load_edit(edit_id))
            patch = EvalLoraPatch(model, safe_factors, beta=beta)
            patch.install()
            active_patches.append(patch)
            for idx, eval_record in enumerate(records):
                after = _evaluate_current(
                    model,
                    eval_record,
                    image_root,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    skip_generation=skip_generation,
                )
                rows.append(
                    _make_eval_row(
                        method="sequential_D_engram_projected_tiny_lora_replacement",
                        record=eval_record,
                        case_index=idx,
                        before=baselines[str(eval_record["id"])],
                        after=after,
                        rollback_diff=0.0,
                        rollback_tolerance=rollback_tolerance,
                        locality_threshold=locality_threshold,
                        record_id_match_rate=1.0 if matching.get("mode") == "record_id" else 0.0,
                        edit_id=edit_id,
                        beta=beta,
                        extra={
                            "step": step,
                            "applied_record_id": str(record["id"]),
                            "projector_available": bool(projection_summary["projector_available"]),
                        },
                    )
                )
    finally:
        for patch in reversed(active_patches):
            patch.remove()
    rollback_diff = _max_snapshot_diff(model, snapshots)
    _restore_modules(model, snapshots)
    final_rows = [row for row in rows if row.get("step") == len(records)]
    final_aggregate = _aggregate_method(final_rows, "sequential_D_engram_projected_tiny_lora_replacement")
    payload = {
        "status": "complete",
        "beta": beta,
        "edit_record_matching": matching,
        "rollback_max_abs_diff_after_all_patches_removed": rollback_diff,
        "rollback_pass": rollback_diff <= rollback_tolerance,
        "per_record": rows,
        "final_aggregate": final_aggregate,
    }
    _json_dump(seq_dir / "sequential_replacement_results.json", payload)
    _write_csv(seq_dir / "sequential_replacement_results.csv", rows)
    return payload


def _write_final_report(
    out_dir: Path,
    data_summary: Dict[str, Any],
    projector_extract: Optional[Dict[str, Any]],
    direct_status: Dict[str, Any],
    aggregates: List[Dict[str, Any]],
    acceptance: Dict[str, Any],
    sequential: Dict[str, Any],
) -> None:
    by_method = {row["method"]: row for row in aggregates}
    decision = "C. Do not scale. Projected replacement did not pass the 5-edit non-sequential gate."
    if acceptance.get("status") == "pass" and sequential.get("status") == "complete":
        decision = "A. Non-sequential projected replacement passed and the conservative sequential smoke ran."
    elif acceptance.get("status") == "pass":
        decision = "B. Non-sequential projected replacement passed, but sequential validation did not complete."
    lines = [
        "# Final Localized Replacement 5-Edit Report",
        "",
        "## Motivation",
        "",
        "Direct ENGRAM erase showed non-sequential signal but failed the saved sequential smoke locality gate. This run tests an ENGRAM-localized replacement candidate before any 20-edit scaling.",
        "",
        "## Data",
        "",
        f"- Replacement data: `{data_summary.get('replacement_data_file')}`",
        f"- Records: `{data_summary.get('record_count')}`",
        f"- Private or patient data used: `{data_summary.get('private_or_patient_data_used')}`",
        f"- Original data modified: `{data_summary.get('original_data_modified')}`",
        "",
        "## Method",
        "",
        "- Candidate delta source: `tiny_lora`",
        "- Replacement mode: `lora_projected`",
        "- Projected delta: `Delta_safe = Delta_candidate @ P`, implemented as low-rank `B @ (A @ P)`",
        f"- Selected modules: `{EXPECTED_MODULES}`",
        f"- Projector extraction: `{(projector_extract or {}).get('status')}`",
        "",
        "## Non-Sequential Comparison",
        "",
        "| Method | mean new NLL decrease | mean old NLL increase | mean ref delta | positive new | positive old erase | locality damage | rollback | match | nan/inf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in [
        "A_no_edit_baseline",
        "B_direct_engram_erase",
        "C_unprojected_tiny_lora_replacement",
        "D_engram_projected_tiny_lora_replacement",
    ]:
        row = by_method.get(method, {"method": method, "status": "skipped"})
        lines.append(
            "| {method} | {new} | {old} | {ref} | {pos_new} | {pos_old} | {loc} | {roll} | {match} | {nan} |".format(
                method=method,
                new=_format(row.get("mean_new_answer_nll_decrease")),
                old=_format(row.get("mean_old_answer_nll_increase")),
                ref=_format(row.get("mean_ref_delta")),
                pos_new=row.get("positive_new_answer_edits"),
                pos_old=row.get("positive_old_answer_erasure_edits"),
                loc=row.get("locality_damage_edits"),
                roll=_format(row.get("rollback_pass_rate")),
                match=_format(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Direct Erase Bank",
            "",
            f"- Status: `{direct_status.get('status')}`",
            f"- Reason: `{direct_status.get('reason')}`",
            "",
            "## Acceptance",
            "",
            f"- Projected replacement gate: `{acceptance.get('status')}`",
        ]
    )
    for key, value in (acceptance.get("checks") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Sequential",
            "",
            f"- Status: `{sequential.get('status')}`",
            f"- Reason: `{sequential.get('reason')}`",
            "",
            "## Decision",
            "",
            decision,
            "",
        ]
    )
    (out_dir / "FINAL_LOCALIZED_REPLACEMENT_5EDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ENGRAM localized replacement 5-edit prototype with tiny LoRA candidate deltas.")
    parser.add_argument("--hparams", default="hparams/ENGRAM/llava_med_5edit_localized_replacement_tiny_lora.yaml")
    parser.add_argument("--source-data", default="outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/raw/engram_smoke_5edit.json")
    parser.add_argument("--source-image-root", default="outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/images")
    parser.add_argument("--output-dir", default="outputs/engram_localized_replacement_5edit")
    parser.add_argument("--sequential-report", default="outputs/engram_sequential_5edit_smoke/FINAL_SEQUENTIAL_5EDIT_SMOKE_REPORT.md")
    parser.add_argument("--best-direct-config", default="outputs/engram_token_module_ablation_5edit/best_overall_config.json")
    parser.add_argument("--direct-erase-bank", default="outputs/engram_token_module_ablation_5edit/module_scope_ablation/banks/module__all__qk_gate_sampled_depths")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direct-erase-alpha", type=float, default=0.075)
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_git_outputs(out_dir)
    _write_failure_summary(out_dir, Path(args.sequential_report), Path(args.best_direct_config))
    replacement_data, image_root, data_summary = _prepare_replacement_data(Path(args.source_data), Path(args.source_image_root), out_dir)
    shutil.copyfile(args.hparams, out_dir / "base_hparams.used.yaml")

    if args.prepare_only:
        payload = {
            "status": "prepared",
            "replacement_data_file": str(replacement_data),
            "image_root": str(image_root),
            "next_step": "Run without --prepare-only on the LLaVA-Med environment.",
        }
        _json_dump(out_dir / "prepare_only_status.json", payload)
        _write_final_report(out_dir, data_summary, None, {"status": "skipped", "reason": "prepare_only"}, [], {"status": "skipped", "checks": {}}, payload)
        return 0

    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    _configure_hparams(
        hparams,
        image_root=image_root,
        bank_dir=out_dir / "projector_bank",
        device=args.device,
        edit_mode="erase",
    )
    hparams.replacement_mode = "lora_projected"
    hparams.candidate_delta_source = "tiny_lora"
    hparams.project_delta_with_engram = True
    hparams.replacement_beta = float(getattr(hparams, "replacement_beta", 1.0))
    hparams.replacement_lambda_ref = float(getattr(hparams, "replacement_lambda_ref", 0.0))
    hparams.lora_rank = int(getattr(hparams, "lora_rank", 4))
    hparams.lora_steps = int(getattr(hparams, "lora_steps", 20))
    hparams.lora_lr = float(getattr(hparams, "lora_lr", 1.0e-4))
    _json_dump(
        out_dir / "effective_replacement_config.json",
        {
            "edit_mode": "replacement",
            "projector_extraction_edit_mode": hparams.edit_mode,
            "replacement_mode": hparams.replacement_mode,
            "candidate_delta_source": hparams.candidate_delta_source,
            "project_delta_with_engram": hparams.project_delta_with_engram,
            "replacement_beta": hparams.replacement_beta,
            "replacement_lambda_ref": hparams.replacement_lambda_ref,
            "lora_rank": hparams.lora_rank,
            "lora_steps": hparams.lora_steps,
            "lora_lr": hparams.lora_lr,
            "selected_modules": EXPECTED_MODULES,
            "token_scope": hparams.resolved_token_scope(),
            "covariance_device": hparams.resolved_covariance_device(),
            "solve_device": hparams.resolved_solve_device(),
            "max_cov_dim": hparams.max_cov_dim,
            "skip_if_dim_larger_than": hparams.skip_if_dim_larger_than,
        },
    )

    records = _load_records(replacement_data)
    editor = MultimodalEditor.from_hparams(hparams)
    selected = [layer.name for layer in select_linear_layers(editor.model, hparams)]
    selected_status = {
        "status": "pass" if set(selected) == set(EXPECTED_MODULES) and len(selected) == len(EXPECTED_MODULES) else "fail",
        "selected_module_names": selected,
        "expected_module_names": EXPECTED_MODULES,
    }
    _json_dump(out_dir / "selected_modules_preflight.json", selected_status)
    if selected_status["status"] != "pass":
        raise RuntimeError(f"Selected modules do not match locked q/k/gate set: {selected_status}")

    baselines = {
        str(record["id"]): _evaluate_current(
            editor.model,
            record,
            image_root,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            skip_generation=args.skip_generation,
        )
        for record in records
    }
    _json_dump(out_dir / "baseline_metrics.json", baselines)

    projector_extract = _extract_projector_bank(editor, hparams, replacement_data, records, out_dir / "projector_bank")
    _json_dump(out_dir / "projector_extraction_summary.json", projector_extract)

    rows: List[Dict[str, Any]] = []
    rows.extend(
        _evaluate_no_edit(
            editor.model,
            records,
            image_root,
            baselines,
            EXPECTED_MODULES,
            rollback_tolerance=args.rollback_tolerance,
            locality_threshold=args.locality_damage_threshold,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            skip_generation=args.skip_generation,
        )
    )
    direct_rows, direct_status = _evaluate_direct_erase(
        editor.model,
        records,
        image_root,
        baselines,
        EXPECTED_MODULES,
        Path(args.direct_erase_bank),
        alpha=float(args.direct_erase_alpha),
        rollback_tolerance=args.rollback_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=args.skip_generation,
    )
    rows.extend(direct_rows)
    lora_rows, bank_metadata = _evaluate_lora_methods(
        editor.model,
        records,
        image_root,
        baselines,
        out_dir / "projector_bank",
        projector_extract,
        EXPECTED_MODULES,
        hparams,
        out_dir,
        rollback_tolerance=args.rollback_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=args.skip_generation,
    )
    rows.extend(lora_rows)
    _json_dump(out_dir / "replacement_bank_metadata.json", bank_metadata)

    methods = [
        "A_no_edit_baseline",
        "B_direct_engram_erase",
        "C_unprojected_tiny_lora_replacement",
        "D_engram_projected_tiny_lora_replacement",
    ]
    aggregates = [_aggregate_method(rows, method) for method in methods]
    if not direct_rows:
        for row in aggregates:
            if row["method"] == "B_direct_engram_erase":
                row.update({"status": "skipped", "skip_reason": direct_status.get("reason")})
    acceptance = _projected_acceptance(aggregates)
    nonseq_payload = {
        "status": "complete",
        "data_summary": data_summary,
        "projector_extract": projector_extract,
        "direct_erase_status": direct_status,
        "aggregate_rows": aggregates,
        "per_record": rows,
        "acceptance": acceptance,
    }
    nonseq_dir = out_dir / "nonseq"
    _json_dump(nonseq_dir / "nonseq_replacement_results.json", nonseq_payload)
    _write_csv(nonseq_dir / "nonseq_replacement_results.csv", rows)
    _write_csv(nonseq_dir / "nonseq_replacement_aggregates.csv", aggregates)

    sequential = _run_sequential_if_pass(
        editor.model,
        records,
        image_root,
        baselines,
        out_dir / "projector_bank",
        EXPECTED_MODULES,
        hparams,
        out_dir,
        acceptance,
        rollback_tolerance=args.rollback_tolerance,
        locality_threshold=args.locality_damage_threshold,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        skip_generation=args.skip_generation,
    )
    _write_final_report(out_dir, data_summary, projector_extract, direct_status, aggregates, acceptance, sequential)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
