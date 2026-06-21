#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.covariance import (  # noqa: E402
    flatten_activation_rows,
    token_scope_mask_from_batch,
)
from easyeditor.models.engram.engram_main import EngramMultimodalRewriteExecutor  # noqa: E402


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Expected non-empty JSON list: {path}")
    return records


def _resolve_image(root: Path, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.exists() and root.name == "images":
        rel = Path(value)
        if rel.parts and rel.parts[0] == "images":
            path = root / Path(*rel.parts[1:])
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def _request_from_record(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    return {
        "prompt": record.get("prompt", record.get("src")),
        "target": record.get("target", record.get("alt", record.get("pred"))),
        "image": _resolve_image(image_root, record.get("image")),
        "rephrase_prompt": record.get("rephrase_prompt", record.get("rephrase")),
        "image_rephrase": _resolve_image(image_root, record.get("image_rephrase")),
        "locality_prompt": record.get("loc"),
        "locality_ground_truth": record.get("loc_ans"),
        "multimodal_locality_prompt": record.get("multimodal_locality_prompt", record.get("m_loc_q")),
        "multimodal_locality_ground_truth": record.get("multimodal_locality_ground_truth", record.get("m_loc_a")),
        "multimodal_locality_image": _resolve_image(
            image_root,
            record.get("multimodal_locality_image", record.get("m_loc")),
        ),
    }


def _edit_ids_for_records(bank: EngramBank, num_records: int) -> List[str]:
    edits = bank.list_edits()
    if len(edits) < num_records:
        raise RuntimeError(f"Bank {bank.root} has {len(edits)} edits, expected at least {num_records}.")
    return [item["edit_id"] for item in edits[:num_records]]


class ActivationRowCollector:
    def __init__(
        self,
        model: torch.nn.Module,
        module_names: Sequence[str],
        *,
        token_scope: str,
        mask_fallback: str,
    ) -> None:
        modules = dict(model.named_modules())
        self.modules = {}
        for name in module_names:
            module = modules.get(name)
            if not isinstance(module, torch.nn.Linear):
                raise RuntimeError(f"Module is not nn.Linear or is missing: {name}")
            self.modules[name] = module
        self.token_scope = token_scope
        self.mask_fallback = mask_fallback
        self.rows: Dict[str, List[torch.Tensor]] = {name: [] for name in self.modules}
        self.warnings: Dict[str, List[str]] = {name: [] for name in self.modules}
        self.scope_logs: List[Dict[str, Any]] = []
        self._current_batch: Dict[str, Any] = {}
        self._handles: List[Any] = []

    def __enter__(self):
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_pre_hook(self._hook(name, module.in_features)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._current_batch = {}

    def set_batch(self, batch: Dict[str, Any]) -> None:
        self._current_batch = batch

    def clear_batch(self) -> None:
        self._current_batch = {}

    def _hook(self, name: str, input_dim: int):
        def hook(module, inputs):
            mask, diag = token_scope_mask_from_batch(
                self._current_batch,
                self.token_scope,
                mask_fallback=self.mask_fallback,
            )
            if not self.scope_logs or self.scope_logs[-1] != diag:
                self.scope_logs.append(dict(diag))
            rows, warning = flatten_activation_rows(inputs, input_dim, mask, mask_fallback=self.mask_fallback)
            if warning:
                self.warnings[name].append(warning)
            if rows.numel():
                self.rows[name].append(rows.detach().float().cpu())

        return hook

    def concatenated(self) -> Dict[str, torch.Tensor]:
        output: Dict[str, torch.Tensor] = {}
        for name, chunks in self.rows.items():
            module = self.modules[name]
            if chunks:
                output[name] = torch.cat(chunks, dim=0)
            else:
                output[name] = torch.empty(0, int(module.in_features))
        return output


def _collect_rows(
    model: torch.nn.Module,
    batches: Sequence[Dict[str, Any]],
    module_names: Sequence[str],
    *,
    token_scope: str,
    mask_fallback: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]], List[Dict[str, Any]]]:
    collector = ActivationRowCollector(model, module_names, token_scope=token_scope, mask_fallback=mask_fallback)
    with collector:
        with torch.no_grad():
            for batch in batches:
                collector.set_batch(batch)
                _ = model(batch)
                collector.clear_batch()
    return collector.concatenated(), collector.warnings, collector.scope_logs


def _matmul_norm(weight: torch.Tensor, rows: torch.Tensor) -> float:
    if rows.numel() == 0:
        return 0.0
    return float(rows.matmul(weight.detach().float().cpu().transpose(0, 1)).norm().item())


def _row_effect(
    module: torch.nn.Linear,
    raw_update: Dict[str, Any],
    rows: torch.Tensor,
    alpha: float,
) -> Dict[str, float]:
    weight = module.weight.detach().float().cpu()
    update_weight = raw_update["weight"].detach().float().cpu()
    new_weight = weight - float(alpha) * update_weight
    before = _matmul_norm(weight, rows)
    after = _matmul_norm(new_weight, rows)
    delta = _matmul_norm(-float(alpha) * update_weight, rows)
    return {
        "preactivation_norm_before": before,
        "preactivation_norm_after": after,
        "delta_norm": delta,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure direct layer-level ENGRAM erasure effects.")
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-json", default="outputs/engram_erase_failure_diagnosis/layer_effects.json")
    parser.add_argument("--output-csv", default="outputs/engram_erase_failure_diagnosis/layer_effects.csv")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-edits", type=int, default=None)
    parser.add_argument("--token-scope", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = _load_records(Path(args.data_file))
    if args.max_edits is not None:
        records = records[: args.max_edits]
    image_root = Path(args.image_root)
    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    hparams.device = int(args.device) if str(args.device).isdigit() else args.device
    if args.token_scope:
        hparams.token_scope = args.token_scope
    editor = MultimodalEditor.from_hparams(hparams)
    wrapper = editor.model
    wrapper.eval()
    executor = EngramMultimodalRewriteExecutor()

    bank = EngramBank(args.bank)
    edit_ids = _edit_ids_for_records(bank, len(records))
    modules = dict(wrapper.named_modules())
    rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []
    for case_index, (record, edit_id) in enumerate(zip(records, edit_ids)):
        edit = bank.load_edit(edit_id)
        module_names = list(edit["updates"].keys())
        request = _request_from_record(record, image_root)
        target_batches = executor._make_batches([request], editor.tok, hparams, hparams.resolved_target_variants(), executor._device_for_model(wrapper, hparams))
        reference_batches = executor._make_batches([request], editor.tok, hparams, hparams.resolved_reference_variants(), executor._device_for_model(wrapper, hparams))
        target_rows, target_warnings, target_scope_logs = _collect_rows(
            wrapper,
            target_batches,
            module_names,
            token_scope=hparams.resolved_token_scope(),
            mask_fallback=hparams.engram_mask_fallback,
        )
        reference_rows, reference_warnings, reference_scope_logs = _collect_rows(
            wrapper,
            reference_batches,
            module_names,
            token_scope=hparams.resolved_token_scope(),
            mask_fallback=hparams.engram_mask_fallback,
        )
        alpha = float(edit["metadata"].get("alpha", hparams.resolved_alpha()))
        for module_name in module_names:
            module = modules[module_name]
            target_effect = _row_effect(module, edit["updates"][module_name], target_rows[module_name], alpha)
            reference_effect = _row_effect(module, edit["updates"][module_name], reference_rows[module_name], alpha)
            row = {
                "case_index": case_index,
                "record_id": record.get("id"),
                "edit_id": edit_id,
                "module": module_name,
                "alpha": alpha,
                "token_scope": hparams.resolved_token_scope(),
                "target_activation_count": int(target_rows[module_name].shape[0]),
                "reference_activation_count": int(reference_rows[module_name].shape[0]),
                "target_preactivation_norm_before": target_effect["preactivation_norm_before"],
                "target_preactivation_norm_after": target_effect["preactivation_norm_after"],
                "reference_preactivation_norm_before": reference_effect["preactivation_norm_before"],
                "reference_preactivation_norm_after": reference_effect["preactivation_norm_after"],
                "target_delta_norm": target_effect["delta_norm"],
                "reference_delta_norm": reference_effect["delta_norm"],
                "reference_to_target_delta_ratio": None
                if target_effect["delta_norm"] == 0.0
                else reference_effect["delta_norm"] / target_effect["delta_norm"],
                "target_norm_reduced": target_effect["preactivation_norm_after"] < target_effect["preactivation_norm_before"],
            }
            rows.append(row)
            json_rows.append(
                {
                    **row,
                    "target_warnings": target_warnings.get(module_name, []),
                    "reference_warnings": reference_warnings.get(module_name, []),
                    "target_scope_logs": target_scope_logs,
                    "reference_scope_logs": reference_scope_logs,
                }
            )

    output = {
        "status": "complete",
        "hparams": args.hparams,
        "data_file": args.data_file,
        "image_root": args.image_root,
        "bank_dir": args.bank,
        "token_scope": hparams.resolved_token_scope(),
        "metric_note": "preactivation norms are weight-only W X; bias is not included in the requested W X fields.",
        "rows": json_rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_csv(Path(args.output_csv), rows)
    print(json.dumps({"json": str(output_json), "csv": str(args.output_csv), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
