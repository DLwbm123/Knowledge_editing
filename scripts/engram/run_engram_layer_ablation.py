#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

with redirect_stdout(sys.stderr):
    from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
    from easyeditor.models.engram.engram_main import apply_engram_to_linear  # noqa: E402


GROUPS: Dict[str, List[str]] = {
    "qk_only": [r"(q_proj|k_proj)$"],
    "qk_gate": [r"(q_proj|k_proj|gate_proj)$"],
    "projector_only": [r"(mm_projector|llama_proj|opt_proj)$"],
    "qk_gate_projector": [r"(q_proj|k_proj|gate_proj|mm_projector|llama_proj|opt_proj)$"],
    "all_configured": [r".*"],
    "no_qk_gate": [r"(mm_projector|llama_proj|opt_proj)$"],
}


def _toy_data():
    target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.9, -0.1, 0.0]])
    reference = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.9, 0.1], [0.0, 1.1, -0.1]])
    return target, reference


def run_mock(groups: List[str], out_dir: Path, alpha: float, direction: str) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target, reference = _toy_data()
    rows = []
    for group in groups:
        layer = torch.nn.Linear(3, 2, bias=True)
        torch.nn.init.constant_(layer.weight, 0.5)
        torch.nn.init.constant_(layer.bias, 0.1)
        before_target = layer(target).norm().item()
        before_reference = layer(reference).norm().item()
        hparams = EngramMultimodalHparams(
            alpha=alpha,
            engram_update_direction=direction,
            module_patterns=GROUPS[group],
            token_scope="all",
            absorb_bias=True,
        )
        update = apply_engram_to_linear(layer, target, reference, hparams=hparams, module_name=f"toy.{group}.q_proj")
        after_target = layer(target).norm().item()
        after_reference = layer(reference).norm().item()
        rows.append(
            {
                "group": group,
                "patterns": ";".join(GROUPS[group]),
                "alpha": alpha,
                "engram_update_direction": direction,
                "direction_sign": update.direction_sign,
                "target_norm_before": before_target,
                "target_norm_after": after_target,
                "reference_norm_before": before_reference,
                "reference_norm_after": after_reference,
                "target_delta": after_target - before_target,
                "reference_delta": after_reference - before_reference,
                "norm_ratio": update.stats["norm_ratio"],
            }
        )
    return rows


def write_outputs(rows: List[dict], out_dir: Path) -> None:
    with (out_dir / "layer_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    with (out_dir / "layer_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ENGRAM layer ablation.")
    parser.add_argument("--groups", default="qk_only,qk_gate,projector_only,qk_gate_projector,all_configured,no_qk_gate")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--direction", default="subtract", choices=["subtract", "add"])
    parser.add_argument("--out", default="outputs/engram_layer_ablation_mock")
    parser.add_argument("--mock", action="store_true", help="Use a tiny Linear layer; no model weights required.")
    args = parser.parse_args()
    if not args.mock:
        raise SystemExit("Only --mock mode is implemented in this lightweight runner. Use scripts/medmkeb/run_medmkeb_editing.py for real models.")
    groups = [item for item in args.groups.split(",") if item.strip()]
    unknown = [group for group in groups if group not in GROUPS]
    if unknown:
        raise SystemExit(f"Unknown groups: {unknown}")
    out_dir = Path(args.out)
    rows = run_mock(groups, out_dir, args.alpha, args.direction)
    write_outputs(rows, out_dir)
    print(json.dumps({"json": str(out_dir / "layer_ablation.json"), "csv": str(out_dir / "layer_ablation.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
