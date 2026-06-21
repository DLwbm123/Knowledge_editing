#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.erasure_metrics import erasure_delta_metrics, safe_model_answer_nll_and_logprob  # noqa: E402


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Expected non-empty JSON list: {path}")
    return records


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


def _sample(prompt: str, answer: str, image_path: str) -> Dict[str, Any]:
    return {"text_input": prompt, "prompt": prompt, "target": answer, "image_path": image_path}


def _target_sample(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    old_answer = record.get("pred")
    if not old_answer:
        raise RuntimeError(f"Record {record.get('id')} missing old target answer `pred`.")
    return _sample(record["src"], old_answer, _resolve_image(image_root, record["image"]))


def _reference_sample(record: Dict[str, Any], image_root: Path) -> Optional[Dict[str, Any]]:
    if not (record.get("m_loc_q") and record.get("m_loc_a") and record.get("m_loc")):
        return None
    return _sample(record["m_loc_q"], record["m_loc_a"], _resolve_image(image_root, record["m_loc"]))


def _edit_ids_for_records(
    bank: EngramBank,
    records: List[Dict[str, Any]],
    *,
    allow_positional_matching: bool = False,
) -> tuple[List[str], Dict[str, Any]]:
    return bank.match_edit_ids_to_records(records, allow_positional_matching=allow_positional_matching)


def _module_map(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def _snapshot_modules(model: torch.nn.Module, module_names: List[str]) -> Dict[str, Dict[str, torch.Tensor | None]]:
    modules = _module_map(model)
    snapshots: Dict[str, Dict[str, torch.Tensor | None]] = {}
    for name in module_names:
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"Bank module not found or not Linear: {name}")
        snapshots[name] = {
            "weight": module.weight.detach().clone().cpu(),
            "bias": module.bias.detach().clone().cpu() if module.bias is not None else None,
        }
    return snapshots


def _restore_modules(model: torch.nn.Module, snapshots: Dict[str, Dict[str, torch.Tensor | None]]) -> None:
    modules = _module_map(model)
    with torch.no_grad():
        for name, tensors in snapshots.items():
            module = modules[name]
            module.weight.copy_(tensors["weight"].to(module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None and tensors["bias"] is not None:
                module.bias.copy_(tensors["bias"].to(module.bias.device, dtype=module.bias.dtype))


def _max_snapshot_diff(model: torch.nn.Module, snapshots: Dict[str, Dict[str, torch.Tensor | None]]) -> float:
    modules = _module_map(model)
    diffs = []
    for name, tensors in snapshots.items():
        module = modules[name]
        diffs.append((module.weight.detach().cpu() - tensors["weight"]).abs().max().item())
        if module.bias is not None and tensors["bias"] is not None:
            diffs.append((module.bias.detach().cpu() - tensors["bias"]).abs().max().item())
    return float(max(diffs) if diffs else 0.0)


def _apply_alpha(model: torch.nn.Module, raw_updates: Dict[str, Dict[str, Any]], alpha: float) -> None:
    modules = _module_map(model)
    with torch.no_grad():
        for name, raw in raw_updates.items():
            module = modules[name]
            module.weight.add_((-float(alpha) * raw["weight"]).to(module.weight.device, dtype=module.weight.dtype))
            bias = raw.get("bias")
            if module.bias is not None and bias is not None:
                module.bias.add_((-float(alpha) * bias).to(module.bias.device, dtype=module.bias.dtype))


def _strip(metrics: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if not metrics.get("available"):
        return None
    return {"nll": float(metrics["nll"]), "logprob": float(metrics["logprob"]), "num_tokens": int(metrics["num_tokens"])}


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _aggregate(rows: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    available = [row for row in rows if row.get("erase_logprob_metrics_available")]
    target_nll = [float(row["erase_success_nll_increase"]) for row in available]
    target_logprob = [float(row["erase_success_logprob_drop"]) for row in available]
    reference_delta = [float(row["reference_delta_abs"]) for row in available if row.get("reference_delta_abs") is not None]
    mean_target = _mean(target_nll)
    mean_ref = _mean(reference_delta)
    return {
        "num_edits": len(rows),
        "num_metric_available": len(available),
        "num_metric_unavailable": len(rows) - len(available),
        "mean_target_nll_increase": mean_target,
        "mean_target_logprob_drop": _mean(target_logprob),
        "mean_reference_nll_delta_abs": mean_ref,
        "target_to_reference_delta_ratio": None if mean_target is None or mean_ref in (None, 0.0) else mean_target / mean_ref,
        "num_edits_with_positive_erasure_signal": sum(1 for value in target_nll if value > 0),
        "num_edits_with_locality_damage": sum(1 for value in reference_delta if value > tolerance),
        "rollback_all_within_tolerance": all(row.get("rollback_within_tolerance") for row in rows),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a signed alpha sweep by re-scaling stored ENGRAM E tensors.")
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", default="outputs/engram_erase_failure_diagnosis/signed_alpha_sweep")
    parser.add_argument("--alphas", default="-0.1,-0.05,-0.01,0,0.01,0.05,0.1")
    parser.add_argument("--device", default="0")
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-positional-matching",
        action="store_true",
        help="Allow legacy bank positional edit/record matching when record_id metadata is missing.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = _load_records(Path(args.data_file))
    image_root = Path(args.image_root)
    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    hparams.device = int(args.device) if str(args.device).isdigit() else args.device
    editor = MultimodalEditor.from_hparams(hparams)
    wrapper = editor.model
    wrapper.eval()
    bank = EngramBank(args.bank)
    edit_ids, edit_record_matching = _edit_ids_for_records(
        bank,
        records,
        allow_positional_matching=args.allow_positional_matching,
    )
    module_names: List[str] = []
    for edit_id in edit_ids:
        for name in bank.load_edit(edit_id)["updates"]:
            if name not in module_names:
                module_names.append(name)
    snapshots = _snapshot_modules(wrapper, module_names)
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    per_alpha_rows = []
    per_edit_rows = []
    helpful_by_record_plus: Dict[str, bool] = {}
    helpful_by_record_minus: Dict[str, bool] = {}
    for alpha in alphas:
        rows = []
        for case_index, (record, edit_id) in enumerate(zip(records, edit_ids)):
            edit = bank.load_edit(edit_id)
            _restore_modules(wrapper, snapshots)
            target = _target_sample(record, image_root)
            reference = _reference_sample(record, image_root)
            target_before_raw = safe_model_answer_nll_and_logprob(wrapper, dict(target))
            reference_before_raw = safe_model_answer_nll_and_logprob(wrapper, dict(reference)) if reference else None
            if alpha != 0.0:
                _apply_alpha(wrapper, edit["updates"], alpha)
            target_after_raw = safe_model_answer_nll_and_logprob(wrapper, dict(target))
            reference_after_raw = safe_model_answer_nll_and_logprob(wrapper, dict(reference)) if reference else None
            _restore_modules(wrapper, snapshots)
            rollback_diff = _max_snapshot_diff(wrapper, snapshots)
            target_before = _strip(target_before_raw)
            target_after = _strip(target_after_raw)
            reference_before = _strip(reference_before_raw or {})
            reference_after = _strip(reference_after_raw or {})
            unavailable = None
            if target_before is None or target_after is None:
                unavailable = {"target_before": target_before_raw, "target_after": target_after_raw}
            metrics = erasure_delta_metrics(
                target_before=target_before,
                target_after=target_after,
                reference_before=reference_before,
                reference_after=reference_after,
                unavailable_reason=json.dumps(unavailable, sort_keys=True) if unavailable else None,
            )
            row = {
                "alpha": alpha,
                "case_index": case_index,
                "record_id": record.get("id"),
                "edit_id": edit_id,
                "rollback_max_abs_diff": rollback_diff,
                "rollback_within_tolerance": rollback_diff <= args.rollback_tolerance,
                "target_before_raw": target_before_raw,
                "target_after_raw": target_after_raw,
                "reference_before_raw": reference_before_raw,
                "reference_after_raw": reference_after_raw,
                **metrics,
            }
            rows.append(row)
            per_edit_rows.append(row)
            if row.get("erase_success_nll_increase") is not None and float(row["erase_success_nll_increase"]) > 0:
                if alpha > 0:
                    helpful_by_record_plus[str(record.get("id"))] = True
                if alpha < 0:
                    helpful_by_record_minus[str(record.get("id"))] = True
        aggregate = _aggregate(rows, args.locality_damage_threshold)
        per_alpha_rows.append({"alpha": alpha, **aggregate})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "status": "complete",
        "hparams": args.hparams,
        "data_file": args.data_file,
        "image_root": args.image_root,
        "bank_dir": args.bank,
        "edit_record_matching": edit_record_matching,
        "alphas": alphas,
        "note": "This sweep reuses stored ENGRAM E tensors and applies W <- W - alpha * E after restoring original weights before every edit.",
        "aggregate_rows": per_alpha_rows,
        "per_edit": per_edit_rows,
        "summary": {
            "num_edits_where_plus_alpha_helps": len(helpful_by_record_plus),
            "num_edits_where_minus_alpha_helps": len(helpful_by_record_minus),
        },
    }
    (out_dir / "signed_alpha_sweep.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_csv(out_dir / "signed_alpha_sweep.csv", per_alpha_rows)
    _write_csv(
        out_dir / "signed_alpha_sweep_per_edit.csv",
        [
            {
                "alpha": row["alpha"],
                "record_id": row["record_id"],
                "edit_id": row["edit_id"],
                "erase_success_nll_increase": row.get("erase_success_nll_increase"),
                "erase_success_logprob_drop": row.get("erase_success_logprob_drop"),
                "reference_delta_abs": row.get("reference_delta_abs"),
                "rollback_max_abs_diff": row.get("rollback_max_abs_diff"),
                "rollback_within_tolerance": row.get("rollback_within_tolerance"),
            }
            for row in per_edit_rows
        ],
    )
    print(json.dumps({"json": str(out_dir / "signed_alpha_sweep.json"), "csv": str(out_dir / "signed_alpha_sweep.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
