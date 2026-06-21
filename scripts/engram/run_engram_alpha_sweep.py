#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

with redirect_stdout(sys.stderr):
    from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
    from easyeditor.models.engram.engram_main import apply_engram_to_linear  # noqa: E402
    from easyeditor.models.engram.solver import apply_update_to_module  # noqa: E402


def _toy_data():
    target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.9, -0.1, 0.0]])
    reference = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.9, 0.1], [0.0, 1.1, -0.1]])
    return target, reference


def run_mock(alpha_values: List[float], out_dir: Path, direction: str) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target, reference = _toy_data()
    rows = []
    for alpha in alpha_values:
        layer = torch.nn.Linear(3, 2, bias=True)
        torch.nn.init.constant_(layer.weight, 0.5)
        torch.nn.init.constant_(layer.bias, 0.1)
        before_target = layer(target).norm().item()
        before_reference = layer(reference).norm().item()
        hparams = EngramMultimodalHparams(
            alpha=alpha,
            engram_update_direction=direction,
            module_patterns=[r".*"],
            token_scope="all",
            absorb_bias=True,
        )
        update = apply_engram_to_linear(layer, target, reference, hparams=hparams, module_name="toy.q_proj")
        after_target = layer(target).norm().item()
        after_reference = layer(reference).norm().item()
        apply_update_to_module(layer, update, direction=1)
        rollback_target = layer(target).norm().item()
        rows.append(
            {
                "alpha": alpha,
                "engram_update_direction": direction,
                "direction_sign": update.direction_sign,
                "target_norm_before": before_target,
                "target_norm_after": after_target,
                "reference_norm_before": before_reference,
                "reference_norm_after": after_reference,
                "target_delta": after_target - before_target,
                "reference_delta": after_reference - before_reference,
                "rollback_target_norm": rollback_target,
                "norm_ratio": update.stats["norm_ratio"],
            }
        )
    return rows


def write_outputs(rows: List[dict], out_dir: Path) -> None:
    with (out_dir / "alpha_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    with (out_dir / "alpha_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ENGRAM alpha sweep.")
    parser.add_argument("--alphas", default="0.05,0.1,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--direction", default="subtract", choices=["subtract", "add"])
    parser.add_argument("--out", default="outputs/engram_alpha_sweep_mock")
    parser.add_argument("--mock", action="store_true", help="Use a tiny Linear layer; no model weights required.")
    args = parser.parse_args()
    if not args.mock:
        raise SystemExit("Only --mock mode is implemented in this lightweight runner. Use scripts/medmkeb/run_medmkeb_editing.py for real models.")
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    out_dir = Path(args.out)
    rows = run_mock(alphas, out_dir, args.direction)
    write_outputs(rows, out_dir)
    print(json.dumps({"json": str(out_dir / "alpha_sweep.json"), "csv": str(out_dir / "alpha_sweep.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
