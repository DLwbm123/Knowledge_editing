#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


EDIT_SPECS = [
    ("cardiomegaly", "lung", (210, 82, 82)),
    ("pneumonia", "heart", (82, 145, 210)),
    ("edema", "rib", (97, 170, 108)),
    ("nodule", "diaphragm", (188, 141, 70)),
    ("effusion", "trachea", (142, 95, 186)),
]


def _write_image(path: Path, color: tuple[int, int, int], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 208, 208), outline=(255, 255, 255), width=3)
    draw.text((24, 96), label, fill=(255, 255, 255))
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a synthetic 5-edit MedMKEB engineering-smoke dataset.")
    parser.add_argument("--root", default="outputs/engram_5edit_behavioral_smoke/synthetic_root")
    parser.add_argument("--summary", default="outputs/engram_5edit_behavioral_smoke/data_summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    raw_dir = root / "data" / "medmkeb" / "raw"
    image_dir = root / "data" / "medmkeb" / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    image_paths_resolved = []
    for idx, (old_answer, locality_answer, color) in enumerate(EDIT_SPECS, start=1):
        edit_name = f"edit_{idx}.png"
        rephrase_name = f"rephrase_{idx}.png"
        locality_name = f"locality_{idx}.png"
        _write_image(image_dir / edit_name, color, f"E{idx}")
        _write_image(image_dir / rephrase_name, tuple(min(255, c + 20) for c in color), f"R{idx}")
        _write_image(image_dir / locality_name, tuple(max(0, c - 35) for c in color), f"L{idx}")
        image_paths_resolved.extend(
            [
                str(image_dir / edit_name),
                str(image_dir / rephrase_name),
                str(image_dir / locality_name),
            ]
        )
        prompt = f"Question: What condition is shown in synthetic panel {idx}? Answer with one short phrase."
        record = {
            "id": f"synthetic-5edit-{idx}",
            "src": prompt,
            "alt": old_answer,
            "pred": old_answer,
            "erase_answer": old_answer,
            "rephrase": f"Question: Name the finding in synthetic panel {idx}. Answer with one short phrase.",
            "image": f"images/{edit_name}",
            "image_rephrase": f"images/{rephrase_name}",
            "loc": f"Question: What reference word should remain stable for synthetic panel {idx}? Answer with one word.",
            "loc_ans": locality_answer,
            "text_locality_unavailable_reason": "text_only_locality_requires_image_for_llava_med",
            "m_loc": f"images/{locality_name}",
            "m_loc_q": f"Question: What reference word is shown in locality panel {idx}? Answer with one word.",
            "m_loc_a": locality_answer,
            "modality": "synthetic",
            "department": "runtime_validation",
            "clinical_VQA_task": "mock_behavioral_smoke",
            "perceptual_granularity": "none",
            "original_task": "mock",
            "non_phi_statement": "synthetic color-block image for engineering smoke only",
        }
        records.append(record)

    data_file = raw_dir / "engram_smoke_5edit.json"
    data_file.write_text(json.dumps(records, indent=2), encoding="utf-8")

    summary = {
        "num_edits": len(records),
        "data_file": str(data_file),
        "target_variants_found": ["edit", "rephrase", "image_rephrase"],
        "reference_variants_found": ["locality_multimodal"],
        "text_locality_status": "unavailable_for_llava_med",
        "image_paths_resolved": image_paths_resolved,
        "missing_fields": [],
        "no_private_or_patient_data": True,
        "per_edit": [
            {
                "id": record["id"],
                "x_plus_nonempty": all(record.get(key) for key in ("src", "rephrase", "image_rephrase", "alt")),
                "x_minus_nonempty": all(record.get(key) for key in ("m_loc", "m_loc_q", "m_loc_a")),
                "expected_old_target_answer": record["pred"],
                "expected_locality_answer": record["m_loc_a"],
                "text_locality_unavailable_reason": record["text_locality_unavailable_reason"],
            }
            for record in records
        ],
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"data_file": str(data_file), "summary": str(summary_path), "num_edits": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
