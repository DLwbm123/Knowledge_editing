"""Pure helpers for routed-LoRA v1.1 model-visible input equivalence."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch

from scripts.engram.routed_banked_lora_utils import l2_normalize


POSITIVE_COLLISION = "EXCLUDED_POSITIVE_EQUIVALENCE_COLLISION"
NEGATIVE_DEDUP = "DEDUPLICATED_NEGATIVE_EQUIVALENCE_CLASS"
CROSS_SPLIT_DUPLICATE = "EXCLUDED_CROSS_SPLIT_EQUIVALENCE_DUPLICATE"


def router_input_equivalence_key(
    processed_pixel_tensor_sha256: str,
    image_sizes: Sequence[int],
    routing_input_ids: Sequence[int],
    attention_mask: Sequence[int],
) -> str:
    payload = {
        "processed_pixel_tensor_sha256": str(processed_pixel_tensor_sha256),
        "image_sizes": [int(value) for value in image_sizes],
        "routing_input_ids": [int(value) for value in routing_input_ids],
        "attention_mask": [int(value) for value in attention_mask],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def provenance(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_name": str(entry["input_id"]),
        "source_record": str(entry.get("record_id_audit", "")),
        "pair_type": str(entry.get("pair_type", "")),
        "group": str(entry.get("group", "")),
    }


def unique_negative_equivalence_classes(
    candidates: Sequence[Mapping[str, Any]],
    positive_keys: set[str],
    *,
    prior_split_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prior = set(prior_split_keys or set())
    classes: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for candidate_raw in candidates:
        candidate = dict(candidate_raw)
        key = str(candidate["router_input_equivalence_key"])
        prov = provenance(candidate)
        if key in positive_keys:
            exclusions.append({**prov, "router_input_equivalence_key": key, "status": POSITIVE_COLLISION})
            continue
        if key in prior:
            exclusions.append({**prov, "router_input_equivalence_key": key, "status": CROSS_SPLIT_DUPLICATE})
            continue
        if key in by_key:
            row = by_key[key]
            row["provenance"].append(prov)
            row["candidate_names"].append(prov["candidate_name"])
            row["source_records"] = sorted(set(row["source_records"] + [prov["source_record"]]))
            row["pair_types"] = sorted(set(row["pair_types"] + [prov["pair_type"]]))
            exclusions.append({**prov, "router_input_equivalence_key": key, "status": NEGATIVE_DEDUP, "representative_input_id": row["input_id"]})
            continue
        row = candidate
        row["provenance"] = [prov]
        row["candidate_names"] = [prov["candidate_name"]]
        row["source_records"] = [prov["source_record"]]
        row["pair_types"] = [prov["pair_type"]]
        by_key[key] = row
        classes.append(row)
    return classes, exclusions


def clamped_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    value = float(torch.dot(l2_normalize(left), l2_normalize(right)))
    if not math.isfinite(value):
        raise ValueError("Non-finite router cosine")
    return max(-1.0, min(1.0, value))


def clamped_router_scores(keys: Mapping[str, torch.Tensor], prototype: Mapping[str, torch.Tensor]) -> dict[str, float]:
    image = clamped_cosine(keys["img"], prototype["img"])
    text = clamped_cosine(keys["text"], prototype["text"])
    fused = clamped_cosine(keys["fused"], prototype["fused"])
    return {
        "s_img": image,
        "s_text": text,
        "s_fused": fused,
        "s_min": min(image, text),
        "s_joint": 0.30 * image + 0.30 * text + 0.40 * fused,
    }


def corrected_thresholds(
    calibration_scores: Sequence[Mapping[str, float]],
    positive_scores: Mapping[str, float],
    tolerance: float = 1e-6,
) -> dict[str, float]:
    if not calibration_scores:
        raise ValueError("ROUTER_INSUFFICIENT_UNIQUE_CALIBRATION_NEGATIVES")
    maxima = {name: max(float(row[name]) for row in calibration_scores) for name in ("s_fused", "s_min", "s_joint")}
    for name, maximum in maxima.items():
        positive = float(positive_scores[name])
        if not math.isfinite(maximum) or maximum >= positive - tolerance:
            raise ValueError("ROUTER_DISTINCT_INPUTS_NOT_SEPARABLE")
    return {
        "positive_fused": float(positive_scores["s_fused"]),
        "positive_min": float(positive_scores["s_min"]),
        "positive_joint": float(positive_scores["s_joint"]),
        "max_neg_fused": maxima["s_fused"],
        "max_neg_min": maxima["s_min"],
        "max_neg_joint": maxima["s_joint"],
        "tau_fused": 0.5 * (float(positive_scores["s_fused"]) + maxima["s_fused"]),
        "tau_min": 0.5 * (float(positive_scores["s_min"]) + maxima["s_min"]),
        "tau_joint": 0.5 * (float(positive_scores["s_joint"]) + maxima["s_joint"]),
        "comparison_tolerance": float(tolerance),
    }


def calibration_sufficiency(classes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    native = sum("native" in set(row.get("pair_types", [])) for row in classes)
    image_mismatch = sum(any("prototype_question" in value for value in row.get("pair_types", [])) for row in classes)
    question_mismatch = sum(any("prototype_image" in value for value in row.get("pair_types", [])) for row in classes)
    result = {
        "unique_count": len(classes),
        "unique_native_count": native,
        "valid_image_mismatch_count": image_mismatch,
        "valid_question_mismatch_count": question_mismatch,
    }
    result["passed"] = bool(len(classes) >= 8 and native >= 3 and image_mismatch >= 1 and question_mismatch >= 1)
    return result

