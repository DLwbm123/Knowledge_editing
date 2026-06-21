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
    parser = argparse.ArgumentParser(description="Inspect or rollback a saved ENGRAM edit.")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--edit-id", required=True, help="One edit id or a comma-separated list for mock rollback.")
    parser.add_argument("--dry-run", action="store_true", help="Validate/load rollback tensors without a model.")
    parser.add_argument("--mock", action="store_true", help="Apply then rollback edit(s) on a tiny Linear model.")
    args = parser.parse_args()

    bank = EngramBank(args.bank)
    edit_ids = [item.strip() for item in args.edit_id.split(",") if item.strip()]
    edit = bank.load_edit(edit_ids[0])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "rollback_ready",
                    "edit_id": edit_ids[0],
                    "modules": sorted(edit["updates"].keys()),
                    "note": "Call EngramBank.rollback_edit(model, edit_id) on the loaded model to undo this delta.",
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
        applied_weight = model.q_proj.weight.detach().clone()
        applied_bias = model.q_proj.bias.detach().clone()
        for edit_id in reversed(edit_ids):
            bank.rollback_edit(model, edit_id)
        restored = torch.allclose(model.q_proj.weight, original_weight, atol=1.0e-6) and torch.allclose(
            model.q_proj.bias, original_bias, atol=1.0e-6
        )
        print(
            json.dumps(
                {
                    "status": "mock_rolled_back",
                    "edit_ids": edit_ids,
                    "weight_after_apply": applied_weight.tolist(),
                    "bias_after_apply": applied_bias.tolist(),
                    "weight_after_rollback": model.q_proj.weight.detach().tolist(),
                    "bias_after_rollback": model.q_proj.bias.detach().tolist(),
                    "restored": bool(restored),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if restored else 1
    raise SystemExit("Real rollback is done through EngramBank.rollback_edit(model, edit_id) after loading the model.")


if __name__ == "__main__":
    raise SystemExit(main())
