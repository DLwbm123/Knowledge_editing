"""Pure router and routed-bank helpers for the record-953 one-edit gate."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def normalize_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def membership_hash(record_id: str, image_sha256: str, question: str) -> str:
    raw = f"{record_id}||{image_sha256}||{normalize_question(question)}".encode()
    return hashlib.sha256(raw).hexdigest()


def l2_normalize(value: torch.Tensor) -> torch.Tensor:
    vector = value.detach().float().reshape(-1)
    norm = vector.norm()
    if not torch.isfinite(vector).all() or not torch.isfinite(norm) or float(norm) <= 0:
        raise ValueError("Router key is zero or non-finite")
    return vector / norm


def find_unique_subsequence(source: Sequence[int], candidates: Sequence[Sequence[int]]) -> list[int]:
    matches = set()
    source = list(map(int, source))
    for candidate in candidates:
        needle = list(map(int, candidate))
        if not needle:
            continue
        for start in range(max(0, len(source) - len(needle) + 1)):
            if source[start : start + len(needle)] == needle:
                matches.add((start, start + len(needle)))
    if len(matches) != 1:
        raise ValueError(f"Question span is ambiguous: {sorted(matches)}")
    start, end = next(iter(matches))
    return list(range(start, end))


def expanded_positions(original_positions: Sequence[int], image_position: int, image_token_count: int) -> list[int]:
    if image_token_count <= 0:
        raise ValueError("image_token_count must be positive")
    return [int(position) if int(position) < image_position else int(position) + image_token_count - 1 for position in original_positions]


def router_scores(keys: Mapping[str, torch.Tensor], prototype: Mapping[str, torch.Tensor]) -> dict[str, float]:
    cosine = lambda name: float(torch.dot(l2_normalize(keys[name]), l2_normalize(prototype[name])))
    image, text, fused = cosine("img"), cosine("text"), cosine("fused")
    return {
        "s_img": image,
        "s_text": text,
        "s_fused": fused,
        "s_min": min(image, text),
        "s_joint": 0.30 * image + 0.30 * text + 0.40 * fused,
    }


def calibrate_thresholds(calibration_scores: Sequence[Mapping[str, float]], prototype_scores: Mapping[str, float], tolerance: float = 1e-6) -> dict[str, float]:
    if not calibration_scores:
        raise ValueError("Calibration negatives are required")
    maxima = {name: max(float(row[name]) for row in calibration_scores) for name in ("s_fused", "s_min", "s_joint")}
    for name, maximum in maxima.items():
        if not math.isfinite(maximum) or maximum >= float(prototype_scores[name]) - tolerance:
            raise ValueError("ROUTER_CALIBRATION_NOT_SEPARABLE")
    return {
        "max_neg_fused": maxima["s_fused"],
        "max_neg_min": maxima["s_min"],
        "max_neg_joint": maxima["s_joint"],
        "tau_fused": 0.5 * (1.0 + maxima["s_fused"]),
        "tau_min": 0.5 * (1.0 + maxima["s_min"]),
        "tau_joint": 0.5 * (1.0 + maxima["s_joint"]),
        "comparison_tolerance": tolerance,
    }


def route_on(scores: Mapping[str, float], thresholds: Mapping[str, float], tolerance: float = 1e-6) -> bool:
    return bool(
        float(scores["s_fused"]) + tolerance >= float(thresholds["tau_fused"])
        and float(scores["s_min"]) + tolerance >= float(thresholds["tau_min"])
        and float(scores["s_joint"]) + tolerance >= float(thresholds["tau_joint"])
    )


def split_negative_records(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def record_id(row: Mapping[str, Any]) -> str:
        value = row.get("record_id", row.get("record_id_audit"))
        if value is None:
            raise ValueError("Negative source is missing its audit record ID")
        return str(value)

    ordered = sorted((dict(row) for row in rows), key=lambda row: (str(row["membership_hash"]), record_id(row)))
    if len(ordered) != 9:
        raise ValueError("Exactly nine non-target fixed-ten records are required")
    return ordered[:5], ordered[5:]


def routing_input_audit(entry: Mapping[str, Any], target: str, old_answer: str) -> bool:
    material = " ".join(str(entry.get(name, "")) for name in ("question", "router_prompt"))
    normalized = normalize_question(material)
    return normalize_question(target) not in normalized and normalize_question(old_answer) not in normalized and not entry.get("record_id_used_as_feature", False) and not entry.get("image_hash_used_as_feature", False)


def save_router_bank(path: Path, tensors: Mapping[str, torch.Tensor], manifest: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=False)
    payload = {name: value.detach().cpu().float().contiguous() for name, value in tensors.items()}
    save_file(payload, str(path / "router_keys.safetensors"))
    tensor_hash = hashlib.sha256((path / "router_keys.safetensors").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps({**dict(manifest), "router_keys_sha256": tensor_hash}, indent=2, sort_keys=True) + "\n")


def load_router_bank(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from safetensors.torch import load_file

    manifest = json.loads((path / "manifest.json").read_text())
    observed = hashlib.sha256((path / "router_keys.safetensors").read_bytes()).hexdigest()
    if observed != manifest["router_keys_sha256"]:
        raise RuntimeError("Router-key checksum mismatch")
    return load_file(str(path / "router_keys.safetensors")), manifest


def ensure_new_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path
