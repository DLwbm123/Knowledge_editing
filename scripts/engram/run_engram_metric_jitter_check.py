#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.erasure_metrics import safe_model_answer_nll_and_logprob  # noqa: E402


def _load_records(path: Path, max_edits: Optional[int]) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Expected non-empty JSON list: {path}")
    return records[:max_edits] if max_edits is not None else records


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _nll(metrics: Dict[str, Any]) -> Optional[float]:
    return float(metrics["nll"]) if metrics.get("available") and metrics.get("nll") is not None else None


def _std(values: List[float]) -> Optional[float]:
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


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
    parser = argparse.ArgumentParser(description="Measure alpha=0 ENGRAM metric jitter on the 5-edit dataset.")
    parser.add_argument("--hparams", default="hparams/ENGRAM/llava_med_5edit_alpha0.yaml")
    parser.add_argument(
        "--data-file",
        default="outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/raw/engram_smoke_5edit.json",
    )
    parser.add_argument(
        "--image-root",
        default="outputs/engram_5edit_behavioral_smoke/synthetic_root/data/medmkeb/images",
    )
    parser.add_argument("--output-dir", default="outputs/engram_sign_calibrated_5edit/jitter_check")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-edits", type=int, default=5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    _set_seed(args.seed)
    records = _load_records(Path(args.data_file), args.max_edits)
    image_root = Path(args.image_root)
    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    hparams.alpha = 0.0
    hparams.engram_update_direction = "subtract"
    hparams.device = int(args.device) if str(args.device).isdigit() else args.device

    editor = MultimodalEditor.from_hparams(hparams)
    wrapper = editor.model
    wrapper.eval()

    rows: List[Dict[str, Any]] = []
    for repeat in range(args.repeats):
        _set_seed(args.seed + repeat)
        with torch.no_grad():
            for case_index, record in enumerate(records):
                target = _target_sample(record, image_root)
                reference = _reference_sample(record, image_root)
                target_raw = safe_model_answer_nll_and_logprob(wrapper, dict(target))
                reference_raw = safe_model_answer_nll_and_logprob(wrapper, dict(reference)) if reference else {}
                rows.append(
                    {
                        "repeat": repeat,
                        "case_index": case_index,
                        "record_id": record.get("id"),
                        "alpha": 0.0,
                        "engram_update_direction": "subtract",
                        "effective_update_applied": False,
                        "target_nll": _nll(target_raw),
                        "reference_nll": _nll(reference_raw),
                        "target_available": bool(target_raw.get("available")),
                        "reference_available": bool(reference_raw.get("available")),
                        "target_raw": json.dumps(target_raw, sort_keys=True),
                        "reference_raw": json.dumps(reference_raw, sort_keys=True),
                    }
                )

    per_record: List[Dict[str, Any]] = []
    for record in records:
        record_id = record.get("id")
        record_rows = [row for row in rows if row["record_id"] == record_id]
        target_values = [float(row["target_nll"]) for row in record_rows if row["target_nll"] is not None]
        reference_values = [float(row["reference_nll"]) for row in record_rows if row["reference_nll"] is not None]
        per_record.append(
            {
                "record_id": record_id,
                "target_nll_std": _std(target_values),
                "reference_nll_std": _std(reference_values),
                "target_available_repeats": len(target_values),
                "reference_available_repeats": len(reference_values),
            }
        )

    target_stds = [float(row["target_nll_std"]) for row in per_record if row["target_nll_std"] is not None]
    reference_stds = [float(row["reference_nll_std"]) for row in per_record if row["reference_nll_std"] is not None]
    aggregate = {
        "status": "complete",
        "hparams": args.hparams,
        "data_file": args.data_file,
        "image_root": args.image_root,
        "repeats": args.repeats,
        "num_records": len(records),
        "alpha": 0.0,
        "engram_update_direction": "subtract",
        "effective_update_applied": False,
        "mean_target_nll_std": sum(target_stds) / len(target_stds) if target_stds else None,
        "max_target_nll_std": max(target_stds) if target_stds else None,
        "mean_reference_nll_std": sum(reference_stds) / len(reference_stds) if reference_stds else None,
        "max_reference_nll_std": max(reference_stds) if reference_stds else None,
        "per_record": per_record,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"aggregate": aggregate, "rows": rows}
    (out_dir / "jitter_check.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(out_dir / "jitter_check.csv", rows)
    _write_csv(out_dir / "jitter_check_per_record.csv", per_record)
    print(json.dumps({"json": str(out_dir / "jitter_check.json"), "csv": str(out_dir / "jitter_check.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
