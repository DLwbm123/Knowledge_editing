#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

with redirect_stdout(sys.stderr):
    from easyeditor.models.engram.bank import EngramBank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or apply a saved ENGRAM bank edit.")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--edit-id", required=True, help="One edit id or a comma-separated list for mock composition.")
    parser.add_argument("--dry-run", action="store_true", help="Validate/load the edit without a model.")
    parser.add_argument("--mock", action="store_true", help="Apply edit(s) to a tiny Linear model; no real weights required.")
    args = parser.parse_args()

    bank = EngramBank(args.bank)
    edit_ids = [item.strip() for item in args.edit_id.split(",") if item.strip()]
    edit = bank.load_edit(edit_ids[0])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "loaded",
                    "edit_id": edit_ids[0],
                    "metadata": edit["metadata"],
                    "modules": sorted(edit["updates"].keys()),
                    "update_directions": {
                        name: {
                            "engram_update_direction": raw.get("engram_update_direction", "subtract"),
                            "direction_sign": raw.get("direction_sign", -1),
                            "alpha": raw.get("alpha", edit["metadata"].get("alpha")),
                        }
                        for name, raw in edit["updates"].items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.mock:
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(2, 1, bias=True)

        model = TinyModel()
        model.q_proj.weight.data[:] = torch.tensor([[1.0, 1.0]])
        model.q_proj.bias.data[:] = torch.tensor([0.25])
        original_weight = model.q_proj.weight.detach().clone()
        original_bias = model.q_proj.bias.detach().clone()
        for edit_id in edit_ids:
            bank.apply_edit(model, edit_id)
        composed = bank.compose_updates(edit_ids)
        print(
            json.dumps(
                {
                    "status": "mock_applied",
                    "edit_ids": edit_ids,
                    "weight_before": original_weight.tolist(),
                    "bias_before": original_bias.tolist(),
                    "weight_after": model.q_proj.weight.detach().tolist(),
                    "bias_after": model.q_proj.bias.detach().tolist(),
                    "compose_modules": sorted(composed.keys()),
                    "composed_weight_delta": composed.get("q_proj", {}).get("weight", torch.empty(0)).tolist(),
                    "composed_bias_delta": composed.get("q_proj", {}).get("bias", torch.empty(0)).tolist(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise SystemExit("Real model application is done through EngramBank.apply_edit(model, edit_id) after loading the model.")


if __name__ == "__main__":
    raise SystemExit(main())
