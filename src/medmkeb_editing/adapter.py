from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .asset_resolver import IMAGE_FIELDS, default_candidate_roots, ensure_placeholder_image, resolve_image
from .paths import MedMKEBPaths
from .schema import extract_portability_qa, load_records, normalize_answer


VQA_TEMPLATE = "Question: {} Short answer:"


@dataclass
class AdaptedRecord:
    original: Dict[str, Any]
    prompt: str
    target: str
    image: Optional[str]
    file_type: str
    rephrase_prompt: Optional[str]
    rephrase_image: Optional[str]
    locality_text_prompt: Optional[str]
    locality_text_ground_truth: Optional[str]
    locality_multimodal_prompt: Optional[str]
    locality_multimodal_ground_truth: Optional[str]
    locality_multimodal_image: Optional[str]
    metadata: Dict[str, Any]
    missing_fields: List[str]
    missing_images: Dict[str, Any]


def _optional_text(record: Dict[str, Any], key: str) -> Optional[str]:
    if key not in record or record.get(key) in (None, ""):
        return None
    return normalize_answer(record.get(key)) if key.endswith(("ans", "_a", "alt", "pred")) else str(record.get(key))


def _format_prompt(text: Optional[str], template: str) -> str:
    text = text or ""
    if "{}" in template:
        return template.format(text)
    if "{prompt}" in template:
        return template.format(prompt=text)
    return text


def _resolve_field(record: Dict[str, Any], field: str, paths: MedMKEBPaths) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    roots = default_candidate_roots(paths)
    if field not in record or record.get(field) in (None, ""):
        return None, {"field": field, "reason": "missing_field"}
    resolution = resolve_image(record.get(field), roots)
    if resolution.exists and resolution.resolved is not None:
        return str(resolution.resolved), None
    return None, {
        "field": field,
        "reason": "unresolved_path",
        "value": record.get(field),
        "checked_paths": [str(path) for path in resolution.checked_paths[:10]],
    }


def _apply_missing_image_policy(
    paths: MedMKEBPaths,
    field: str,
    value: Optional[str],
    m_loc_image: Optional[str],
    policy: str,
) -> Optional[str]:
    if value:
        return value
    if policy == "blank":
        return str(ensure_placeholder_image(paths, f"{field}_blank_rgb_224.png"))
    if policy == "m_loc" and m_loc_image:
        return m_loc_image
    return value


def adapt_record(
    record: Dict[str, Any],
    paths: MedMKEBPaths,
    prompt_template: str = VQA_TEMPLATE,
    missing_image_policy: str = "fail",
) -> AdaptedRecord:
    missing_fields: List[str] = []
    missing_images: Dict[str, Any] = {}

    def required_text(key: str) -> str:
        value = _optional_text(record, key)
        if value is None:
            missing_fields.append(key)
            return ""
        return value

    m_loc_image, mloc_error = _resolve_field(record, "m_loc", paths)
    image, image_error = _resolve_field(record, "image", paths)
    rephrase_image, rephrase_error = _resolve_field(record, "image_rephrase", paths)
    image = _apply_missing_image_policy(paths, "image", image, m_loc_image, missing_image_policy)
    rephrase_image = _apply_missing_image_policy(
        paths, "image_rephrase", rephrase_image, m_loc_image, missing_image_policy
    )
    for error in (image_error, rephrase_error, mloc_error):
        if error:
            missing_images[error["field"]] = error

    metadata = {
        "id": record.get("id"),
        "modality": record.get("modality"),
        "department": record.get("department"),
        "clinical_VQA_task": record.get("clinical_VQA_task"),
        "perceptual_granularity": record.get("perceptual_granularity"),
        "original_task": record.get("original_task"),
        "pred_original": record.get("pred"),
        "alt_original": record.get("alt"),
        "port_new": record.get("port_new"),
        "portability_qa": extract_portability_qa(record),
    }

    return AdaptedRecord(
        original=record,
        prompt=_format_prompt(required_text("src"), prompt_template),
        target=normalize_answer(record.get("alt")),
        image=image,
        file_type="image",
        rephrase_prompt=_format_prompt(_optional_text(record, "rephrase"), prompt_template)
        if _optional_text(record, "rephrase")
        else None,
        rephrase_image=rephrase_image,
        locality_text_prompt=_optional_text(record, "loc"),
        locality_text_ground_truth=normalize_answer(record.get("loc_ans")) if record.get("loc_ans") else None,
        locality_multimodal_prompt=_format_prompt(_optional_text(record, "m_loc_q"), prompt_template)
        if _optional_text(record, "m_loc_q")
        else None,
        locality_multimodal_ground_truth=normalize_answer(record.get("m_loc_a")) if record.get("m_loc_a") else None,
        locality_multimodal_image=m_loc_image,
        metadata=metadata,
        missing_fields=missing_fields,
        missing_images=missing_images,
    )


def filter_records(
    records: Sequence[Dict[str, Any]],
    max_edits: Optional[int] = None,
    seed: int = 0,
    modality: Optional[str] = None,
    department: Optional[str] = None,
    task: Optional[str] = None,
    shuffle: bool = False,
    stream_type: str = "random",
) -> List[Dict[str, Any]]:
    selected = list(records)
    if modality:
        selected = [r for r in selected if str(r.get("modality")) == modality]
    if department:
        selected = [r for r in selected if str(r.get("department")) == department]
    if task:
        selected = [r for r in selected if str(r.get("clinical_VQA_task")) == task]

    rng = random.Random(seed)
    if stream_type == "modality_blocked":
        selected.sort(key=lambda r: (str(r.get("modality")), str(r.get("id"))))
    elif stream_type == "task_blocked":
        selected.sort(key=lambda r: (str(r.get("clinical_VQA_task")), str(r.get("id"))))
    elif shuffle or stream_type == "random":
        rng.shuffle(selected)
    else:
        raise ValueError(f"Unknown stream_type: {stream_type}")

    if max_edits is not None and max_edits > 0:
        selected = selected[:max_edits]
    return selected


def adapted_to_batch(adapted: Sequence[AdaptedRecord], file_type: str = "image") -> Dict[str, Any]:
    prompts = [r.prompt for r in adapted]
    targets = [r.target for r in adapted]
    images = [r.image for r in adapted]
    file_types = [file_type for _ in adapted]

    text_prompt = [r.locality_text_prompt for r in adapted]
    text_truth = [r.locality_text_ground_truth for r in adapted]
    mm_prompt = [r.locality_multimodal_prompt for r in adapted]
    mm_truth = [r.locality_multimodal_ground_truth for r in adapted]
    mm_image = [r.locality_multimodal_image for r in adapted]

    locality_inputs = {
        "text": {"prompt": text_prompt, "ground_truth": text_truth},
        "multimodal": {"prompt": mm_prompt, "ground_truth": mm_truth, "image": mm_image},
    }
    return {
        "prompts": prompts,
        "targets": targets,
        "image": images,
        "file_type": file_types,
        "rephrase_prompts": [r.rephrase_prompt for r in adapted],
        "rephrase_image": [r.rephrase_image for r in adapted],
        "locality_inputs": locality_inputs,
        "metadata": [r.metadata for r in adapted],
        "missing_fields": [r.missing_fields for r in adapted],
        "missing_images": [r.missing_images for r in adapted],
    }


def build_edit_batch(
    paths: MedMKEBPaths,
    data_file: Path,
    max_edits: Optional[int] = None,
    seed: int = 0,
    modality: Optional[str] = None,
    department: Optional[str] = None,
    task: Optional[str] = None,
    shuffle: bool = False,
    stream_type: str = "random",
    prompt_template: str = VQA_TEMPLATE,
    file_type: str = "image",
    missing_image_policy: str = "fail",
) -> Dict[str, Any]:
    records = load_records(data_file)
    selected = filter_records(
        records,
        max_edits=max_edits,
        seed=seed,
        modality=modality,
        department=department,
        task=task,
        shuffle=shuffle,
        stream_type=stream_type,
    )
    adapted = [
        adapt_record(
            record,
            paths,
            prompt_template=prompt_template,
            missing_image_policy=missing_image_policy,
        )
        for record in selected
    ]
    batch = adapted_to_batch(adapted, file_type=file_type)
    batch["source_file"] = str(data_file)
    batch["selected_records"] = len(selected)
    batch["total_records_in_file"] = len(records)
    batch["required_image_fields"] = list(IMAGE_FIELDS)
    batch["missing_image_policy"] = missing_image_policy
    return batch


def unresolved_image_count(batch: Dict[str, Any]) -> int:
    return sum(1 for row in batch.get("missing_images", []) if row)


def build_vlkeb_records(
    paths: MedMKEBPaths,
    data_file: Path,
    max_edits: Optional[int] = None,
    seed: int = 0,
    modality: Optional[str] = None,
    department: Optional[str] = None,
    task: Optional[str] = None,
    shuffle: bool = False,
    stream_type: str = "random",
    missing_image_policy: str = "fail",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build records that local VLKEB `CaptionDataset` can read directly.

    VLKEB already expects MedMKEB-like keys (`src`, `alt`, `image`,
    `image_rephrase`, `loc`, `m_loc`, ...). This function only filters records
    and resolves image fields to absolute paths. Absolute paths are intentional:
    VLKEB joins them with `coco_image`, and `os.path.join(root, absolute)` keeps
    the absolute path unchanged.
    """
    records = load_records(data_file)
    selected = filter_records(
        records,
        max_edits=max_edits,
        seed=seed,
        modality=modality,
        department=department,
        task=task,
        shuffle=shuffle,
        stream_type=stream_type,
    )

    vlkeb_records: List[Dict[str, Any]] = []
    missing_images: List[Dict[str, Any]] = []
    missing_fields: List[Dict[str, Any]] = []
    for idx, record in enumerate(selected):
        copied = dict(record)
        adapted = adapt_record(record, paths, prompt_template="{}", missing_image_policy=missing_image_policy)
        field_values = {
            "image": adapted.image,
            "image_rephrase": adapted.rephrase_image,
            "m_loc": adapted.locality_multimodal_image,
        }
        for field, value in field_values.items():
            if value:
                copied[field] = value
        for field in ("src", "alt", "pred", "rephrase", "loc", "loc_ans", "m_loc_q", "m_loc_a"):
            if copied.get(field) is None:
                copied[field] = ""
                missing_fields.append({"record_index": idx, "record_id": copied.get("id"), "field": field})
        if adapted.missing_images:
            missing_images.append(
                {
                    "record_index": idx,
                    "record_id": copied.get("id"),
                    "missing_images": adapted.missing_images,
                }
            )
        vlkeb_records.append(copied)

    report = {
        "source_file": str(data_file),
        "total_records_in_file": len(records),
        "selected_records": len(vlkeb_records),
        "missing_image_policy": missing_image_policy,
        "unresolved_records_with_images": len(missing_images),
        "missing_images_example": missing_images[:10],
        "missing_fields_example": missing_fields[:10],
        "format": "VLKEB CaptionDataset JSON",
    }
    return vlkeb_records, report
