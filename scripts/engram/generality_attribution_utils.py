"""Pure helpers for the record-953 generality attribution audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CATEGORIES = ("TEXTUAL_GENERALITY", "VISUAL_GENERALITY", "PAIRED_GENERALITY")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_generality(*, image_differs: bool, question_differs: bool, source_field: str) -> str:
    if not image_differs and question_differs and source_field != "port_new":
        return "TEXTUAL_GENERALITY"
    if image_differs and not question_differs:
        return "VISUAL_GENERALITY"
    if question_differs and source_field == "port_new":
        # MedMKEB portability Q&A is the benchmark's paired multimodal field;
        # its image is inherited from the owning record rather than repeated.
        return "PAIRED_GENERALITY"
    if image_differs and question_differs:
        return "PAIRED_GENERALITY"
    raise ValueError("Input is not a source-grounded generality variation")


def failed_router_conjuncts(scores: Mapping[str, float], thresholds: Mapping[str, float], tolerance: float = 1e-6) -> list[str]:
    return [
        name for name, score_name, threshold_name in (
            ("fused", "s_fused", "tau_fused"),
            ("min", "s_min", "tau_min"),
            ("joint", "s_joint", "tau_joint"),
        )
        if float(scores[score_name]) + tolerance < float(thresholds[threshold_name])
    ]


def primary_failed_side(scores: Mapping[str, float], thresholds: Mapping[str, float], tolerance: float = 1e-6) -> str:
    failed = failed_router_conjuncts(scores, thresholds, tolerance)
    sides = []
    if float(scores["s_img"]) + tolerance < float(thresholds["tau_min"]):
        sides.append("IMAGE_SIDE")
    if float(scores["s_text"]) + tolerance < float(thresholds["tau_min"]):
        sides.append("QUESTION_SIDE")
    if "fused" in failed:
        sides.append("FUSED")
    if "joint" in failed:
        sides.append("JOINT")
    unique = list(dict.fromkeys(sides))
    return unique[0] if len(unique) == 1 else ("MULTIPLE" if unique else "NONE")


def nearest_by_components(query: Mapping[str, float], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for component in ("s_img", "s_text", "s_fused", "s_min", "s_joint"):
        if not rows:
            result[component] = None
            continue
        nearest = min(rows, key=lambda row: (abs(float(row[component]) - float(query[component])), str(row["input_id"])))
        result[component] = {
            "input_id": str(nearest["input_id"]),
            "score": float(nearest[component]),
            "absolute_distance": abs(float(nearest[component]) - float(query[component])),
        }
    return result


def diagnostic_separability(category: str, query: Mapping[str, float], calibration: Sequence[Mapping[str, Any]], tolerance: float = 1e-6) -> dict[str, Any]:
    if category == "TEXTUAL_GENERALITY":
        comparable = [row for row in calibration if float(row["s_img"]) >= 1.0 - tolerance]
        components = ("s_text", "s_fused")
        regime = "exact_image"
    elif category == "VISUAL_GENERALITY":
        comparable = [row for row in calibration if "prototype_question" in str(row.get("pair_type", "")) or "native" in str(row.get("pair_type", "")) and str(row.get("record_id_audit")) == "1592"]
        components = ("s_img", "s_fused")
        regime = "exact_question"
    elif category == "PAIRED_GENERALITY":
        comparable = list(calibration)
        components = ("s_fused", "s_joint")
        regime = "all_calibration"
    else:
        raise ValueError(category)
    if not comparable:
        return {"status": "INSUFFICIENT_COMPARABLE_NEGATIVES", "regime": regime, "comparable_count": 0, "components": list(components)}
    maxima = {name: max(float(row[name]) for row in comparable) for name in components}
    gap = all(float(query[name]) > maxima[name] + tolerance for name in components)
    return {
        "status": "DIAGNOSTIC_GAP_EXISTS" if gap else "DIAGNOSTIC_OVERLAP_WITH_NEGATIVES",
        "regime": regime,
        "comparable_count": len(comparable),
        "components": list(components),
        "negative_maxima": maxima,
        "query_scores": {name: float(query[name]) for name in components},
        "gaps": {name: float(query[name]) - maxima[name] for name in components},
    }


def attribution_label(unrestricted_successes: int, total: int, short_successes: int) -> str:
    if total <= 0:
        return "GENERALITY_ATTRIBUTION_INVALID_ENGINEERING_RUN"
    if unrestricted_successes == total:
        return "GENERALITY_ROUTER_RECALL_BOTTLENECK"
    if unrestricted_successes == 0 and short_successes > 0:
        return "GENERALITY_FORMAT_CONDITIONED_ONLY"
    if unrestricted_successes == 0:
        return "GENERALITY_ADAPTER_MEMORIZATION_BOTTLENECK"
    return "GENERALITY_MIXED_ROUTER_AND_ADAPTER_BOTTLENECK"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

