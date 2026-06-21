#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.erasure_metrics import _extract_logits_labels, safe_model_answer_nll_and_logprob  # noqa: E402


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Expected non-empty JSON list: {path}")
    return records


def _resolve_image(root: Path, rel_path: str) -> str:
    path = Path(rel_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists() and root.name == "images":
        rel = Path(rel_path)
        if rel.parts and rel.parts[0] == "images":
            path = root / Path(*rel.parts[1:])
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def _sample(prompt: str, answer: str, image_path: str) -> Dict[str, Any]:
    return {
        "text_input": prompt,
        "prompt": prompt,
        "target": answer,
        "image_path": image_path,
    }


def _target_sample(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    old_answer = record.get("pred")
    if not old_answer:
        raise RuntimeError(f"Record {record.get('id')} missing old target answer `pred`.")
    return _sample(record["src"], old_answer, _resolve_image(image_root, record["image"]))


def _reference_sample(record: Dict[str, Any], image_root: Path) -> Optional[Dict[str, Any]]:
    if not (record.get("m_loc_q") and record.get("m_loc_a") and record.get("m_loc")):
        return None
    return _sample(record["m_loc_q"], record["m_loc_a"], _resolve_image(image_root, record["m_loc"]))


def _edit_ids_for_records(bank: EngramBank, num_records: int) -> List[str]:
    edits = bank.list_edits()
    if len(edits) < num_records:
        raise RuntimeError(f"Bank {bank.root} has {len(edits)} edits, expected at least {num_records}.")
    return [item["edit_id"] for item in edits[:num_records]]


def _module_map(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def _snapshot_modules(model: torch.nn.Module, module_names: List[str]) -> Dict[str, Dict[str, torch.Tensor | None]]:
    modules = _module_map(model)
    snapshots: Dict[str, Dict[str, torch.Tensor | None]] = {}
    for name in module_names:
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"Bank module not found or not Linear: {name}")
        snapshots[name] = {
            "weight": module.weight.detach().clone().cpu(),
            "bias": module.bias.detach().clone().cpu() if module.bias is not None else None,
        }
    return snapshots


def _restore_modules(model: torch.nn.Module, snapshots: Dict[str, Dict[str, torch.Tensor | None]]) -> None:
    modules = _module_map(model)
    with torch.no_grad():
        for name, tensors in snapshots.items():
            module = modules[name]
            module.weight.copy_(tensors["weight"].to(module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None and tensors["bias"] is not None:
                module.bias.copy_(tensors["bias"].to(module.bias.device, dtype=module.bias.dtype))


def _apply_scaled_update(module: torch.nn.Linear, raw_update: Dict[str, Any], scale: float) -> None:
    with torch.no_grad():
        module.weight.add_((float(scale) * raw_update["weight"]).to(module.weight.device, dtype=module.weight.dtype))
        bias = raw_update.get("bias")
        if module.bias is not None and bias is not None:
            module.bias.add_((float(scale) * bias).to(module.bias.device, dtype=module.bias.dtype))


def _answer_nll_tensor(model: torch.nn.Module, sample: Dict[str, Any], ignore_index: int = -100) -> torch.Tensor:
    output = model(dict(sample))
    logits, labels, _input_ids = _extract_logits_labels(output, sample)
    if logits.shape[:2] != labels.shape:
        min_len = min(logits.shape[1], labels.shape[1])
        logits = logits[:, :min_len]
        labels = labels[:, :min_len]
    if logits.shape[1] < 2:
        raise RuntimeError("sequence too short for causal shifted NLL")
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(ignore_index)
    if not valid.any():
        raise RuntimeError("no valid answer tokens after causal shift")
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return -token_log_probs[valid].mean()


def _metric_nll(model: torch.nn.Module, sample: Optional[Dict[str, Any]]) -> Optional[float]:
    if sample is None:
        return None
    metrics = safe_model_answer_nll_and_logprob(model, dict(sample))
    if not metrics.get("available"):
        return None
    return float(metrics["nll"])


def _flat_pair(weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]) -> torch.Tensor:
    parts = []
    if weight is not None:
        parts.append(weight.detach().float().reshape(-1).cpu())
    if bias is not None:
        parts.append(bias.detach().float().reshape(-1).cpu())
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def _grad_update_stats(
    model: torch.nn.Module,
    raw_updates: Dict[str, Dict[str, Any]],
    alpha: float,
) -> Dict[str, Any]:
    modules = _module_map(model)
    per_module: Dict[str, Any] = {}
    aggregate_minus = 0.0
    aggregate_plus = 0.0
    for name, raw in raw_updates.items():
        module = modules[name]
        grad_w = module.weight.grad.detach().float().cpu() if module.weight.grad is not None else torch.zeros_like(raw["weight"]).float()
        grad_b = None
        if module.bias is not None:
            if module.bias.grad is not None:
                grad_b = module.bias.grad.detach().float().cpu()
            elif raw.get("bias") is not None:
                grad_b = torch.zeros_like(raw["bias"]).float()
        update_w = raw["weight"].detach().float().cpu()
        update_b = raw.get("bias")
        update_b = update_b.detach().float().cpu() if update_b is not None else None
        delta_minus_w = -float(alpha) * update_w
        delta_plus_w = float(alpha) * update_w
        delta_minus_b = -float(alpha) * update_b if update_b is not None else None
        delta_plus_b = float(alpha) * update_b if update_b is not None else None
        grad_flat = _flat_pair(grad_w, grad_b)
        minus_flat = _flat_pair(delta_minus_w, delta_minus_b)
        plus_flat = _flat_pair(delta_plus_w, delta_plus_b)
        derivative_minus = float(torch.dot(grad_flat, minus_flat).item()) if grad_flat.numel() else 0.0
        derivative_plus = float(torch.dot(grad_flat, plus_flat).item()) if grad_flat.numel() else 0.0
        aggregate_minus += derivative_minus
        aggregate_plus += derivative_plus
        per_module[name] = {
            "directional_derivative_minus": derivative_minus,
            "directional_derivative_plus": derivative_plus,
            "cos_grad_NLL_minus_E": float(F.cosine_similarity(grad_flat, minus_flat, dim=0).item()) if grad_flat.norm() > 0 and minus_flat.norm() > 0 else None,
            "cos_grad_NLL_plus_E": float(F.cosine_similarity(grad_flat, plus_flat, dim=0).item()) if grad_flat.norm() > 0 and plus_flat.norm() > 0 else None,
            "grad_norm": float(grad_flat.norm().item()),
            "E_norm": float(_flat_pair(update_w, update_b).norm().item()),
        }
    return {
        "per_module": per_module,
        "aggregate_directional_derivative_minus": aggregate_minus,
        "aggregate_directional_derivative_plus": aggregate_plus,
    }


def _restrict_grads_to_modules(model: torch.nn.Module, module_names: List[str]) -> Dict[str, bool]:
    modules = _module_map(model)
    original: Dict[str, bool] = {}
    selected_params = set()
    for name in module_names:
        module = modules[name]
        selected_params.add(module.weight)
        if module.bias is not None:
            selected_params.add(module.bias)
    for name, param in model.named_parameters():
        original[name] = bool(param.requires_grad)
        param.requires_grad_(param in selected_params)
    return original


def _restore_requires_grad(model: torch.nn.Module, original: Dict[str, bool]) -> None:
    for name, param in model.named_parameters():
        if name in original:
            param.requires_grad_(original[name])


def _finite_difference(
    model: torch.nn.Module,
    snapshots: Dict[str, Dict[str, torch.Tensor | None]],
    raw_updates: Dict[str, Dict[str, Any]],
    target: Dict[str, Any],
    reference: Optional[Dict[str, Any]],
    eps_values: List[float],
    include_reference: bool,
) -> Dict[str, Any]:
    modules = _module_map(model)
    _restore_modules(model, snapshots)
    target_base = _metric_nll(model, target)
    reference_base = _metric_nll(model, reference) if include_reference else None
    aggregate = []
    per_module: Dict[str, List[Dict[str, Any]]] = {name: [] for name in raw_updates}
    for eps in eps_values:
        for sign_name, sign in (("minus", -1.0), ("plus", 1.0)):
            _restore_modules(model, snapshots)
            for name, raw in raw_updates.items():
                _apply_scaled_update(modules[name], raw, sign * eps)
            target_after = _metric_nll(model, target)
            reference_after = _metric_nll(model, reference) if include_reference else None
            aggregate.append(
                {
                    "eps": eps,
                    "direction": sign_name,
                    "target_nll_delta": None if target_base is None or target_after is None else target_after - target_base,
                    "reference_nll_delta": None if reference_base is None or reference_after is None else reference_after - reference_base,
                }
            )
            for name, raw in raw_updates.items():
                _restore_modules(model, snapshots)
                _apply_scaled_update(modules[name], raw, sign * eps)
                module_target_after = _metric_nll(model, target)
                module_reference_after = _metric_nll(model, reference) if include_reference else None
                per_module[name].append(
                    {
                        "eps": eps,
                        "direction": sign_name,
                        "target_nll_delta": None
                        if target_base is None or module_target_after is None
                        else module_target_after - target_base,
                        "reference_nll_delta": None
                        if reference_base is None or module_reference_after is None
                        else module_reference_after - reference_base,
                    }
                )
    _restore_modules(model, snapshots)
    return {
        "target_nll_base": target_base,
        "reference_nll_base": reference_base,
        "aggregate": aggregate,
        "per_module": per_module,
        "reference_finite_diff_status": "computed" if include_reference else "skipped_not_cheap_by_default",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose whether ENGRAM update sign locally increases target old-answer NLL.")
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", default="outputs/engram_erase_failure_diagnosis/sign_diagnostics")
    parser.add_argument("--device", default="0")
    parser.add_argument("--edit-index", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=None, help="Direction scale for derivative; defaults to bank metadata alpha.")
    parser.add_argument("--eps", default="1e-4,5e-4,1e-3,5e-3")
    parser.add_argument("--include-reference-finite-diff", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = _load_records(Path(args.data_file))
    if args.edit_index < 0 or args.edit_index >= len(records):
        raise RuntimeError(f"--edit-index {args.edit_index} out of range for {len(records)} records")
    image_root = Path(args.image_root)
    bank = EngramBank(args.bank)
    edit_ids = _edit_ids_for_records(bank, len(records))
    edit_id = edit_ids[args.edit_index]
    edit = bank.load_edit(edit_id)
    raw_updates = edit["updates"]
    alpha = float(args.alpha if args.alpha is not None else edit["metadata"].get("alpha", 1.0))

    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    hparams.device = int(args.device) if str(args.device).isdigit() else args.device
    editor = MultimodalEditor.from_hparams(hparams)
    wrapper = editor.model
    wrapper.eval()

    target = _target_sample(records[args.edit_index], image_root)
    reference = _reference_sample(records[args.edit_index], image_root)
    module_names = list(raw_updates.keys())
    snapshots = _snapshot_modules(wrapper, module_names)

    _restore_modules(wrapper, snapshots)
    original_requires_grad = _restrict_grads_to_modules(wrapper, module_names)
    wrapper.zero_grad(set_to_none=True)
    try:
        loss = _answer_nll_tensor(wrapper, target)
        target_nll_for_grad = float(loss.detach().cpu())
        loss.backward()
        grad_stats = _grad_update_stats(wrapper, raw_updates, alpha)
    finally:
        wrapper.zero_grad(set_to_none=True)
        _restore_requires_grad(wrapper, original_requires_grad)

    eps_values = [float(item) for item in args.eps.split(",") if item.strip()]
    finite_diff = _finite_difference(
        wrapper,
        snapshots,
        raw_updates,
        target,
        reference,
        eps_values,
        include_reference=args.include_reference_finite_diff,
    )
    _restore_modules(wrapper, snapshots)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output = {
        "edit_index": args.edit_index,
        "record_id": records[args.edit_index].get("id"),
        "edit_id": edit_id,
        "bank_dir": str(bank.root),
        "alpha": alpha,
        "selected_modules": module_names,
        "target_old_answer": records[args.edit_index].get("pred"),
        "target_nll_for_grad": target_nll_for_grad,
        "desired_erasure_condition": "directional_derivative_minus > 0",
        **grad_stats,
        "finite_difference": finite_diff,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"edit_{edit_id}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "edit_id": edit_id, "target_nll_for_grad": target_nll_for_grad}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
