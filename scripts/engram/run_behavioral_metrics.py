#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from easyeditor.models.engram.erasure_metrics import (  # noqa: E402
    erasure_delta_metrics,
    safe_model_answer_nll_and_logprob,
)


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
    return {
        "text_input": prompt,
        "prompt": prompt,
        "target": answer,
        "image_path": image_path,
    }


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


def _snapshot_modules(model: torch.nn.Module, module_names: List[str]) -> Dict[str, Dict[str, torch.Tensor | None]]:
    modules = dict(model.named_modules())
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
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, tensors in snapshots.items():
            module = modules[name]
            module.weight.copy_(tensors["weight"].to(module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None and tensors["bias"] is not None:
                module.bias.copy_(tensors["bias"].to(module.bias.device, dtype=module.bias.dtype))


def _max_snapshot_diff(model: torch.nn.Module, snapshots: Dict[str, Dict[str, torch.Tensor | None]]) -> float:
    modules = dict(model.named_modules())
    diffs = []
    for name, tensors in snapshots.items():
        module = modules[name]
        diffs.append((module.weight.detach().cpu() - tensors["weight"]).abs().max().item())
        if module.bias is not None and tensors["bias"] is not None:
            diffs.append((module.bias.detach().cpu() - tensors["bias"]).abs().max().item())
    return float(max(diffs) if diffs else 0.0)


def _all_bank_modules(bank: EngramBank, edit_ids: List[str]) -> List[str]:
    names: List[str] = []
    for edit_id in edit_ids:
        for name in bank.load_edit(edit_id)["updates"].keys():
            if name not in names:
                names.append(name)
    return names


def _strip_availability(metrics: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if not metrics.get("available"):
        return None
    return {
        "nll": float(metrics["nll"]),
        "logprob": float(metrics["logprob"]),
        "num_tokens": int(metrics["num_tokens"]),
    }


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _aggregate(rows: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    available = [row for row in rows if row.get("erase_logprob_metrics_available")]
    target_nll = [float(row["erase_success_nll_increase"]) for row in available]
    target_logprob = [float(row["erase_success_logprob_drop"]) for row in available]
    reference_delta = [
        float(row["reference_delta_abs"])
        for row in available
        if row.get("reference_delta_abs") is not None
    ]
    mean_target_nll = _mean(target_nll)
    mean_reference = _mean(reference_delta)
    return {
        "num_edits": len(rows),
        "num_metric_available": len(available),
        "num_metric_unavailable": len(rows) - len(available),
        "mean_target_nll_increase": mean_target_nll,
        "mean_target_logprob_drop": _mean(target_logprob),
        "mean_reference_nll_delta_abs": mean_reference,
        "target_to_reference_delta_ratio": (
            None
            if mean_target_nll is None or mean_reference in (None, 0.0)
            else mean_target_nll / mean_reference
        ),
        "num_edits_with_positive_erasure_signal": sum(1 for value in target_nll if value > 0),
        "num_edits_with_locality_damage": sum(1 for value in reference_delta if value > tolerance),
        "num_edits_with_metric_unavailable": len(rows) - len(available),
        "rollback_all_within_tolerance": all(row.get("rollback_within_tolerance") for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute ENGRAM erase-mode old-answer logprob/NLL diagnostics.")
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollback-tolerance", type=float, default=1e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
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

    bank = EngramBank(args.bank)
    edit_ids, edit_record_matching = _edit_ids_for_records(
        bank,
        records,
        allow_positional_matching=args.allow_positional_matching,
    )
    snapshots = _snapshot_modules(wrapper, _all_bank_modules(bank, edit_ids))

    rows = []
    for case_index, (record, edit_id) in enumerate(zip(records, edit_ids)):
        _restore_modules(wrapper, snapshots)
        target = _target_sample(record, image_root)
        reference = _reference_sample(record, image_root)
        target_before_raw = safe_model_answer_nll_and_logprob(wrapper, target)
        reference_before_raw = safe_model_answer_nll_and_logprob(wrapper, reference) if reference else None

        bank.apply_edit(wrapper, edit_id)
        target_after_raw = safe_model_answer_nll_and_logprob(wrapper, target)
        reference_after_raw = safe_model_answer_nll_and_logprob(wrapper, reference) if reference else None
        bank.rollback_edit(wrapper, edit_id)
        rollback_max_abs_diff = _max_snapshot_diff(wrapper, snapshots)

        target_before = _strip_availability(target_before_raw)
        target_after = _strip_availability(target_after_raw)
        reference_before = _strip_availability(reference_before_raw or {})
        reference_after = _strip_availability(reference_after_raw or {})
        unavailable = None
        if target_before is None or target_after is None:
            unavailable = {
                "target_before": target_before_raw,
                "target_after": target_after_raw,
            }
        metrics = erasure_delta_metrics(
            target_before=target_before,
            target_after=target_after,
            reference_before=reference_before,
            reference_after=reference_after,
            unavailable_reason=json.dumps(unavailable, sort_keys=True) if unavailable else None,
        )
        rows.append(
            {
                "case_index": case_index,
                "record_id": record.get("id"),
                "edit_id": edit_id,
                "bank_dir": str(bank.root),
                "target_prompt": record.get("src"),
                "target_old_answer": record.get("pred"),
                "reference_prompt": record.get("m_loc_q"),
                "reference_answer": record.get("m_loc_a"),
                "target_before_raw": target_before_raw,
                "target_after_raw": target_after_raw,
                "reference_before_raw": reference_before_raw,
                "reference_after_raw": reference_after_raw,
                "rollback_max_abs_diff": rollback_max_abs_diff,
                "rollback_within_tolerance": rollback_max_abs_diff <= args.rollback_tolerance,
                **metrics,
            }
        )
    _restore_modules(wrapper, snapshots)

    output = {
        "status": "pass" if all(row.get("rollback_within_tolerance") for row in rows) else "fail",
        "hparams": args.hparams,
        "data_file": args.data_file,
        "image_root": args.image_root,
        "bank_dir": args.bank,
        "edit_record_matching": edit_record_matching,
        "rollback_tolerance": args.rollback_tolerance,
        "locality_damage_threshold": args.locality_damage_threshold,
        "metric_definition": {
            "logprob": "sum of causal answer-token log probabilities",
            "nll": "mean causal answer-token negative log likelihood",
            "alignment": "logits[:, :-1] scored against labels[:, 1:] with -100 ignored",
        },
        "aggregate": _aggregate(rows, args.locality_damage_threshold),
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "status": output["status"], "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
