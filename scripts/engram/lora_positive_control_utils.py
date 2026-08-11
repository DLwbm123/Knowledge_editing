"""Focused utilities for the record-953 LoRA positive control."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


FIXED_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def resolve_target_modules(named_modules: Iterable[tuple[str, torch.nn.Module]]) -> list[str]:
    language: dict[tuple[int, str], str] = {}
    projector: list[str] = []
    for name, module in named_modules:
        if not isinstance(module, torch.nn.Linear):
            continue
        match = re.search(r"(?:^|\.)model\.layers\.(\d+)\.(.+)$", name)
        if match:
            layer, suffix = int(match.group(1)), match.group(2)
            if 16 <= layer <= 31 and suffix in FIXED_SUFFIXES:
                language[(layer, suffix)] = name
        if "mm_projector" in name:
            projector.append(name)
    expected = {(layer, suffix) for layer in range(16, 32) for suffix in FIXED_SUFFIXES}
    missing = sorted(expected - set(language))
    if missing:
        raise ValueError(f"Missing fixed language LoRA modules: {missing}")
    if not projector:
        raise ValueError("No multimodal-projector Linear modules resolved")
    ordered_language = [language[(layer, suffix)] for layer in range(16, 32) for suffix in FIXED_SUFFIXES]
    return ordered_language + sorted(set(projector))


def audit_trainable_parameters(named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
    rows = []
    invalid = []
    for name, parameter in named_parameters:
        if parameter.requires_grad:
            row = {"name": name, "numel": parameter.numel(), "dtype": str(parameter.dtype)}
            rows.append(row)
            if "lora_A" not in name and "lora_B" not in name:
                invalid.append(name)
            if parameter.dtype != torch.float32:
                invalid.append(name)
    return {"trainable": rows, "trainable_numel": sum(row["numel"] for row in rows), "invalid": sorted(set(invalid)), "passed": bool(rows) and not invalid}


def weighted_loss(primary: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
    return primary + 0.25 * auxiliary


def shifted_label_audit(prompt_length: int, response_ids: torch.Tensor, expansion: int, positions: Sequence[int]) -> dict[str, Any]:
    expected = [prompt_length - 1 + expansion + index for index in range(response_ids.numel())]
    return {
        "prompt_tokens_masked": True,
        "supervised_response_tokens": int(response_ids.numel()),
        "expected_predictor_positions": expected,
        "observed_predictor_positions": list(map(int, positions)),
        "passed": expected == list(map(int, positions)),
    }


def positive_control_match(output: str, target: str, *, eos: bool, cap_hit: bool, aliases: Sequence[str] = ()) -> dict[str, Any]:
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))
    normalized, target_norm = normalize(output), normalize(target)
    words, needle = normalized.split(), target_norm.split()
    span = any(words[index : index + len(needle)] == needle for index in range(max(0, len(words) - len(needle) + 1))) if needle else False
    alias_rows = []
    for alias in aliases:
        alias_norm = normalize(alias)
        alias_words = alias_norm.split()
        matched = any(words[index : index + len(alias_words)] == alias_words for index in range(max(0, len(words) - len(alias_words) + 1))) if alias_words else False
        alias_rows.append({"alias": alias, "matched": matched})
    contradiction = bool(target_norm and (f"not {target_norm}" in normalized or f"but {target_norm}" in normalized and normalized.rfind("but") > normalized.find(target_norm)))
    return {
        "raw_exact_match": str(output).strip() == str(target).strip(),
        "normalized_exact_match": normalized == target_norm,
        "canonical_target_span_match": span,
        "aliases": alias_rows,
        "alias_match": any(row["matched"] for row in alias_rows),
        "contradiction": contradiction,
        "eos_normal": bool(eos and not cap_hit),
        "success": bool(span and not contradiction and eos and not cap_hit),
    }


def adapter_state(named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().float().clone() for name, value in named_parameters if "lora_A" in name or "lora_B" in name}


def load_adapter_state(named_parameters: Iterable[tuple[str, torch.nn.Parameter]], state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(named_parameters)
    if set(state) != {name for name in parameters if "lora_A" in name or "lora_B" in name}:
        raise ValueError("Adapter state key mismatch")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))


def adapter_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].contiguous()
        digest.update(name.encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_adapter_payload(path: Path, state: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    torch.save(dict(state), path / "adapter.pt")
    manifest = {**dict(metadata), "adapter_sha256": adapter_hash(state)}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_adapter_payload(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((path / "manifest.json").read_text())
    if adapter_hash(state) != manifest["adapter_sha256"]:
        raise RuntimeError("Adapter checksum mismatch")
    return state, manifest


def ensure_new_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path
