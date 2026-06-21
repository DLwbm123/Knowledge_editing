#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from easyeditor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.engram_main import select_linear_layers  # noqa: E402


def _compile(patterns: Sequence[str]) -> List[re.Pattern]:
    return [re.compile(pattern) for pattern in patterns or []]


def _layer_number(name: str) -> Optional[int]:
    for pattern in (r"\.layers\.(\d+)\.", r"\.decoder\.layers\.(\d+)\.", r"layer\.(\d+)\."):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def _uneditable_reason(module: nn.Linear) -> Optional[str]:
    class_name = module.__class__.__name__.lower()
    if "4bit" in class_name or "8bit" in class_name or "quant" in class_name:
        return f"module class {module.__class__.__name__} looks quantized"
    if not isinstance(module.weight, torch.nn.Parameter):
        return "weight is not a torch Parameter"
    if not module.weight.dtype.is_floating_point:
        return f"weight dtype {module.weight.dtype} is not floating point"
    return None


def _shape(tensor: torch.Tensor) -> List[int]:
    return [int(dim) for dim in tensor.shape]


def _module_record(
    name: str,
    module: nn.Linear,
    hparams: EngramMultimodalHparams,
    selected_names: Set[str],
    skip_reason: Optional[str],
) -> Dict[str, Any]:
    absorb_bias = bool(hparams.resolved_absorb_bias() and module.bias is not None)
    covariance_dim = int(module.in_features) + (1 if absorb_bias else 0)
    return {
        "module_name": name,
        "weight_shape": _shape(module.weight),
        "bias_exists": module.bias is not None,
        "parameter_dtype": str(module.weight.dtype),
        "parameter_device": str(module.weight.device),
        "covariance_dim": covariance_dim,
        "editable_or_skipped": "selected" if name in selected_names else "skipped",
        "skip_reason": None if name in selected_names else (skip_reason or "not selected after priority/max_modules"),
    }


def inspect_modules(model: nn.Module, hparams: EngramMultimodalHparams) -> Dict[str, Any]:
    selected = select_linear_layers(model, hparams)
    selected_names = {layer.name for layer in selected}
    include = _compile(hparams.resolved_module_patterns())
    exclude = _compile(hparams.resolved_exclude_patterns())
    layer_filter = set(int(x) for x in hparams.engram_layers) if hparams.engram_layers is not None else None
    skip_dim = hparams.resolved_skip_dim()

    records: List[Dict[str, Any]] = []
    matched_before_max = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if include and not any(pattern.search(name) for pattern in include):
            continue
        matched_before_max += 1

        reason = None
        if exclude and any(pattern.search(name) for pattern in exclude):
            reason = "excluded by regex"
        layer_no = _layer_number(name)
        if reason is None and layer_filter is not None and layer_no is not None and layer_no not in layer_filter:
            reason = f"layer {layer_no} not in engram_layers"
        if reason is None:
            reason = _uneditable_reason(module)
        if reason is None:
            cov_dim = int(module.in_features) + (
                1 if hparams.resolved_absorb_bias() and module.bias is not None else 0
            )
            if skip_dim is not None and cov_dim > int(skip_dim):
                reason = f"cov_dim={cov_dim} exceeds limit={skip_dim}"

        records.append(_module_record(name, module, hparams, selected_names, reason))

    selected_records = [record for record in records if record["editable_or_skipped"] == "selected"]
    skipped_records = [record for record in records if record["editable_or_skipped"] == "skipped"]
    selected_names_list = [record["module_name"] for record in selected_records]
    return {
        "status": "ok" if selected_records else "no_selected_modules",
        "matched_linear_modules_before_max": matched_before_max,
        "selected_count": len(selected_records),
        "skipped_count": len(skipped_records),
        "selected_modules": selected_names_list,
        "selected": selected_records,
        "skipped": skipped_records,
        "coverage": {
            "mm_projector": any("mm_projector" in name for name in selected_names_list),
            "gate_proj": any(name.endswith("gate_proj") for name in selected_names_list),
            "q_proj": any(name.endswith("q_proj") for name in selected_names_list),
            "k_proj": any(name.endswith("k_proj") for name in selected_names_list),
        },
        "config": {
            "module_patterns": hparams.resolved_module_patterns(),
            "exclude_module_patterns": hparams.resolved_exclude_patterns(),
            "prioritize_module_selection": hparams.prioritize_module_selection,
            "module_priority_patterns": hparams.module_priority_patterns,
            "engram_max_modules": hparams.engram_max_modules,
            "skip_if_dim_larger_than": hparams.resolved_skip_dim(),
            "absorb_bias": hparams.resolved_absorb_bias(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight ENGRAM selected nn.Linear modules for a real model.")
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    if args.device is not None:
        hparams.device = args.device

    editor = MultimodalEditor.from_hparams(hparams)
    report = inspect_modules(editor.model, hparams)
    report["hparams"] = str(Path(args.hparams).resolve())
    report["device"] = hparams.device
    report["model_name"] = hparams.model_name
    report["model_path"] = hparams.name

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "selected_count": report["selected_count"]}))
    return 0 if report["selected_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
