#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from easyeditor.dataset.coco_caption import CaptionDataset  # noqa: E402
from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.engram_main import select_linear_layers  # noqa: E402
from easyeditor.models.engram.erasure_metrics import erasure_delta_metrics  # noqa: E402
from easyeditor.util import nethook  # noqa: E402


TOKEN_SCOPES = ["answer", "loss_predictor", "prompt_last", "all"]
ALPHAS = [0.0, 0.05, 0.075]
NEGATIVE_RECORD_ID = "synthetic-5edit-2"


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
    return {"text_input": prompt, "prompt": prompt, "target": answer, "image_path": image_path}


def _target_sample(record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    return _sample(record["src"], record.get("pred") or record.get("alt"), _resolve_image(image_root, record["image"]))


def _reference_sample(record: Dict[str, Any], image_root: Path) -> Optional[Dict[str, Any]]:
    if not (record.get("m_loc_q") and record.get("m_loc_a") and record.get("m_loc")):
        return None
    return _sample(record["m_loc_q"], record["m_loc_a"], _resolve_image(image_root, record["m_loc"]))


def _module_map(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def _snapshot_modules(model: torch.nn.Module, module_names: Iterable[str]) -> Dict[str, Dict[str, torch.Tensor | None]]:
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


def _restore_weight_copy(model: torch.nn.Module, weights_copy: Dict[str, torch.Tensor], device: Any) -> None:
    restore_device = device if str(device).startswith(("cuda", "cpu", "mps")) else f"cuda:{device}"
    with torch.no_grad():
        for key, value in weights_copy.items():
            nethook.get_parameter(model, key)[...] = value.to(restore_device)


def _max_snapshot_diff(model: torch.nn.Module, snapshots: Dict[str, Dict[str, torch.Tensor | None]]) -> float:
    modules = _module_map(model)
    diffs: List[float] = []
    for name, tensors in snapshots.items():
        module = modules[name]
        diffs.append(float((module.weight.detach().cpu() - tensors["weight"]).abs().max().item()))
        if module.bias is not None and tensors["bias"] is not None:
            diffs.append(float((module.bias.detach().cpu() - tensors["bias"]).abs().max().item()))
    return max(diffs) if diffs else 0.0


def _apply_add_alpha(model: torch.nn.Module, raw_updates: Dict[str, Dict[str, Any]], alpha: float) -> None:
    modules = _module_map(model)
    with torch.no_grad():
        for name, raw in raw_updates.items():
            module = modules[name]
            module.weight.add_((float(alpha) * raw["weight"]).to(module.weight.device, dtype=module.weight.dtype))
            bias = raw.get("bias")
            if module.bias is not None and bias is not None:
                module.bias.add_((float(alpha) * bias).to(module.bias.device, dtype=module.bias.dtype))


def _extract_logits_labels(output: Any, sample: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    logits = getattr(output, "logits", None)
    labels = getattr(output, "labels", None)
    input_ids = getattr(output, "input_ids", None)
    if isinstance(output, dict):
        logits = output.get("logits", logits)
        labels = output.get("labels", labels)
        input_ids = output.get("input_ids", input_ids)
    if labels is None:
        labels = sample.get("labels")
    if input_ids is None:
        input_ids = sample.get("input_ids")
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("model output did not include tensor logits")
    if not isinstance(labels, torch.Tensor):
        raise RuntimeError("model output/sample did not include tensor labels")
    if input_ids is not None and not isinstance(input_ids, torch.Tensor):
        input_ids = None
    return logits, labels, input_ids


def _token_strings(token_ids: List[int], tokenizer: Optional[Any]) -> List[str]:
    if tokenizer is None:
        return [str(token_id) for token_id in token_ids]
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return [str(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]
    return [str(tokenizer.decode([token_id], skip_special_tokens=False)) for token_id in token_ids]


def _answer_metrics(model: torch.nn.Module, sample: Dict[str, Any], ignore_index: int = -100) -> Dict[str, Any]:
    try:
        model.eval()
        with torch.no_grad():
            output = model(sample)
        logits, labels, _ = _extract_logits_labels(output, sample)
        if logits.shape[:2] != labels.shape:
            min_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :min_len]
            labels = labels[:, :min_len]
        if logits.shape[1] < 2:
            return {"available": False, "unavailable_reason": "sequence too short"}
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        valid = shift_labels.ne(ignore_index)
        if not valid.any():
            return {"available": False, "unavailable_reason": "no valid answer tokens after causal shift"}
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        safe_labels = shift_labels.masked_fill(~valid, 0)
        token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        selected = token_log_probs[valid]
        answer_token_ids = [int(token_id) for token_id in shift_labels[valid].detach().cpu().tolist()]
        per_token_logprob = [float(value) for value in selected.detach().cpu().tolist()]
        per_token_nll = [-value for value in per_token_logprob]
        tokenizer = getattr(model, "tokenizer", None)
        return {
            "available": True,
            "nll": float(-selected.mean().detach().cpu()),
            "logprob": float(selected.sum().detach().cpu()),
            "num_tokens": int(selected.numel()),
            "answer_token_count": int(selected.numel()),
            "answer_token_ids": answer_token_ids,
            "answer_tokens": _token_strings(answer_token_ids, tokenizer),
            "per_token_logprob": per_token_logprob,
            "per_token_nll": per_token_nll,
            "shift_applied": True,
        }
    except Exception as exc:
        return {"available": False, "unavailable_reason": f"{type(exc).__name__}: {exc}"}


def _strip(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not metrics or not metrics.get("available"):
        return None
    return {"nll": float(metrics["nll"]), "logprob": float(metrics["logprob"]), "num_tokens": int(metrics["num_tokens"])}


def _finite(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def _special_token_ids(tokenizer) -> set[int]:
    ids = {tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id, getattr(tokenizer, "unk_token_id", None)}
    return {int(value) for value in ids if value is not None}


def _generate_llava_med(wrapper, prompt: str, image_path: str, max_new_tokens: int, min_new_tokens: Optional[int]) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = wrapper.process_images([image], wrapper.image_processor, wrapper.llava_model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(wrapper.lm_device, dtype=wrapper.dtype)
    prompt_text = wrapper._conversation_prompt(prompt, None)
    input_ids = wrapper.tokenizer_image_token(
        prompt_text,
        wrapper.tokenizer,
        wrapper.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(wrapper.lm_device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=wrapper.lm_device)
    kwargs: Dict[str, Any] = {
        "images": image_tensor,
        "attention_mask": attention_mask,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": wrapper.tokenizer.eos_token_id,
    }
    if min_new_tokens is not None:
        kwargs["min_new_tokens"] = min_new_tokens
    with torch.inference_mode():
        output_ids = wrapper.llava_model.generate(input_ids, **kwargs)
    new_tokens = output_ids[0, input_ids.shape[1] :]
    generated_ids = [int(token_id) for token_id in new_tokens.detach().cpu().tolist()]
    decoded_raw = wrapper.tokenizer.decode(new_tokens, skip_special_tokens=False)
    decoded_skip_special = wrapper.tokenizer.decode(new_tokens, skip_special_tokens=True)
    decoded_stripped = decoded_skip_special.strip()
    special_ids = _special_token_ids(wrapper.tokenizer)
    eos_id = wrapper.tokenizer.eos_token_id
    return {
        "decoded_raw": decoded_raw,
        "decoded_skip_special": decoded_skip_special,
        "decoded_stripped": decoded_stripped,
        "generated_token_ids": generated_ids,
        "generation_empty": decoded_stripped == "",
        "generated_only_eos_or_special": bool(generated_ids) and all(token_id in special_ids for token_id in generated_ids),
        "stop_reason": "immediate_eos" if generated_ids and eos_id is not None and generated_ids[0] == int(eos_id) else "unknown",
    }


def _maybe_generate(wrapper, record: Dict[str, Any], image_root: Path, max_new_tokens: int, min_new_tokens: Optional[int], skip: bool) -> Optional[Dict[str, Any]]:
    if skip:
        return None
    return _generate_llava_med(wrapper, record["src"], _resolve_image(image_root, record["image"]), max_new_tokens, min_new_tokens)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys})


def _extract_layer_depth(name: str) -> Optional[int]:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _module_group_specs() -> Dict[str, Dict[str, Any]]:
    qkg = [r"llava_model\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj)|mlp\.gate_proj)$"]
    return {
        "smoke_4": {
            "module_patterns": [
                r"llava_model\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj)$",
                r"llava_model\.model\.layers\.\d+\.mlp\.gate_proj$",
                r"(mm_projector)(\.|$)",
            ],
            "priority": [r"(mm_projector)(\.|$)", r"gate_proj$", r"q_proj$", r"k_proj$"],
            "engram_layers": None,
            "engram_max_modules": 4,
        },
        "no_projector": {
            "module_patterns": qkg,
            "priority": [r"gate_proj$", r"q_proj$", r"k_proj$"],
            "engram_layers": None,
            "engram_max_modules": 3,
        },
        "projector_only": {
            "module_patterns": [r"(mm_projector)(\.|$)"],
            "priority": [r"(mm_projector)(\.|$)"],
            "engram_layers": None,
            "engram_max_modules": 1,
        },
        "qk_gate_sampled_depths": {
            "module_patterns": qkg,
            "priority": [r"gate_proj$", r"q_proj$", r"k_proj$"],
            "engram_layers": [0, 8, 16, 24],
            "engram_max_modules": None,
        },
        "late_qk_gate": {
            "module_patterns": qkg,
            "priority": [r"gate_proj$", r"q_proj$", r"k_proj$"],
            "engram_layers": list(range(24, 32)),
            "engram_max_modules": None,
        },
        "qk_gate_all_layers_budgeted": {
            "module_patterns": qkg,
            "priority": [r"gate_proj$", r"q_proj$", r"k_proj$"],
            "engram_layers": None,
            "engram_max_modules": None,
        },
    }


def _apply_config(hparams: EngramMultimodalHparams, config: Dict[str, Any], bank_dir: Path, alpha: float) -> None:
    hparams.edit_mode = "erase"
    hparams.engram_update_direction = "add"
    hparams.sequential_edit = False
    hparams.alpha = float(alpha)
    hparams.token_scope = config["token_scope"]
    hparams.module_patterns = list(config["module_patterns"])
    hparams.module_priority_patterns = list(config["priority"])
    hparams.prioritize_module_selection = True
    hparams.engram_layers = deepcopy(config.get("engram_layers"))
    hparams.engram_max_modules = config.get("engram_max_modules")
    hparams.bank_dir = str(bank_dir)
    hparams.engram_bank_path = str(bank_dir)
    hparams.edit_id = None
    hparams.engram_edit_id = None


def _make_config_id(*parts: str) -> str:
    return "__".join(str(part).replace("/", "_").replace(" ", "_") for part in parts)


def _request_dataset(data_file: Path, hparams: EngramMultimodalHparams) -> CaptionDataset:
    return CaptionDataset(str(data_file), config=hparams)


def _extract_bank_for_config(
    editor: MultimodalEditor,
    hparams: EngramMultimodalHparams,
    data_file: Path,
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
    bank_dir: Path,
    *,
    extraction_alpha: float,
) -> Dict[str, Any]:
    if bank_dir.exists():
        shutil.rmtree(bank_dir)
    _apply_config(hparams, config, bank_dir, extraction_alpha)
    selected = select_linear_layers(editor.model, hparams)
    selected_names = [layer.name for layer in selected]
    estimate = _module_estimate(selected)
    if config.get("skip_reason"):
        return {
            "status": "skipped",
            "skip_reason": config["skip_reason"],
            "selected_module_names": selected_names,
            **estimate,
        }
    ds = _request_dataset(data_file, hparams)
    if len(ds) != len(records):
        raise RuntimeError(f"Dataset length mismatch: CaptionDataset={len(ds)} raw={len(records)}")
    extracted = []
    for idx, request in enumerate(ds):
        record_id = str(records[idx].get("id"))
        request["id"] = record_id
        request["record_id"] = record_id
        request["source_record_id"] = record_id
        hparams.edit_id = f"{config['config_id']}__{record_id}"
        weights_copy = None
        try:
            _, weights_copy = editor.apply_algo(
                editor.model,
                editor.tok,
                [request],
                hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=True,
                train_ds=None,
            )
        except RuntimeError as exc:
            if "ENGRAM produced no layer updates after safety checks" not in str(exc):
                raise
            if bank_dir.exists():
                shutil.rmtree(bank_dir)
            return {
                "status": "skipped",
                "skip_reason": f"no layer updates after safety checks for record_id={record_id}: {exc}",
                "failed_record_id": record_id,
                "partial_extracted": extracted,
                "selected_module_names": selected_names,
                **estimate,
            }
        finally:
            if weights_copy:
                _restore_weight_copy(editor.model, weights_copy, hparams.device)
        extracted.append({"record_id": record_id, "edit_id": hparams.edit_id})
    bank = EngramBank(bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    rows = []
    for record, edit_id in zip(records, edit_ids):
        meta = bank.load_edit(edit_id)["metadata"]
        rows.append(
            {
                "raw_record_id": record.get("id"),
                "bank_record_id": meta.get("record_id") or meta.get("source_record_id"),
                "edit_id": edit_id,
                "matching_mode": matching.get("mode"),
                "record_id_match": str(record.get("id")) == str(meta.get("record_id") or meta.get("source_record_id")),
            }
        )
    return {
        "status": "complete",
        "bank_dir": str(bank_dir),
        "edit_ids": edit_ids,
        "edit_record_matching": matching,
        "record_id_rows": rows,
        "selected_module_names": selected_names,
        **estimate,
    }


def _module_estimate(layers: Sequence[Any]) -> Dict[str, Any]:
    cov_dims = [int(layer.cov_dim) for layer in layers]
    cov_bytes_one = sum(dim * dim * 4 for dim in cov_dims)
    depths = sorted({depth for depth in (_extract_layer_depth(layer.name) for layer in layers) if depth is not None})
    return {
        "module_count": len(layers),
        "selected_module_names": [layer.name for layer in layers],
        "selected_layer_depths": depths,
        "covariance_memory_estimate": {
            "single_covariance_bytes": cov_bytes_one,
            "target_plus_reference_bytes": cov_bytes_one * 2,
            "target_plus_reference_gib": cov_bytes_one * 2 / (1024**3),
            "max_cov_dim": max(cov_dims) if cov_dims else 0,
        },
    }


def _scope_fallback(metadata: Dict[str, Any]) -> Dict[str, Any]:
    logs = list(metadata.get("target_token_scope_logs", []) or []) + list(metadata.get("reference_token_scope_logs", []) or [])
    return {
        "fallback_used": any(bool(row.get("fallback_used")) for row in logs),
        "fallback_reasons": sorted({str(row.get("fallback_reason")) for row in logs if row.get("fallback_reason")}),
        "scope_logs": logs,
    }


def _layer_diagnostics(metadata: Dict[str, Any], alpha: float) -> List[Dict[str, Any]]:
    diagnostics = []
    for layer in metadata.get("layers", []) or []:
        norm_ratio = float(layer.get("norm_ratio", 0.0) or 0.0)
        diagnostics.append(
            {
                "module_name": layer.get("module_name"),
                "num_target_vectors": int(layer.get("num_target_vectors", 0) or 0),
                "num_reference_vectors": int(layer.get("num_reference_vectors", 0) or 0),
                "rank_plus": layer.get("rank_plus"),
                "rank_total": layer.get("rank_total"),
                "norm_ratio": norm_ratio,
                "effective_update_norm_ratio": abs(float(alpha)) * norm_ratio,
                "norm_W": layer.get("norm_W"),
                "norm_E": layer.get("norm_E"),
            }
        )
    return diagnostics


def _summarize_metadata(metadata: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    diagnostics = _layer_diagnostics(metadata, alpha)
    norm_values = [float(row["norm_ratio"]) for row in diagnostics]
    eff_values = [float(row["effective_update_norm_ratio"]) for row in diagnostics]
    return {
        "selected_modules": [row["module_name"] for row in diagnostics if row.get("module_name")],
        "target_activation_count": sum(int(row.get("num_target_vectors") or 0) for row in diagnostics),
        "reference_activation_count": sum(int(row.get("num_reference_vectors") or 0) for row in diagnostics),
        "module_diagnostics": diagnostics,
        "norm_ratio_distribution": _distribution(norm_values),
        "effective_norm_ratio_distribution": _distribution(eff_values),
        "norm_ratios": {str(row["module_name"]): row.get("norm_ratio") for row in diagnostics if row.get("module_name")},
        "effective_norm_ratios": {str(row["module_name"]): row.get("effective_update_norm_ratio") for row in diagnostics if row.get("module_name")},
        "skipped_modules": [],
        "skip_reasons": [],
        **_scope_fallback(metadata),
    }


def _distribution(values: List[float]) -> Dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": _mean(values),
    }


def _evaluate_bank(
    model: torch.nn.Module,
    records: List[Dict[str, Any]],
    image_root: Path,
    bank_dir: Path,
    config: Dict[str, Any],
    alphas: List[float],
    *,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Dict[str, Any]:
    bank = EngramBank(bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    module_names: List[str] = []
    for edit_id in edit_ids:
        for name in bank.load_edit(edit_id)["updates"].keys():
            if name not in module_names:
                module_names.append(name)
    snapshots = _snapshot_modules(model, module_names)
    baselines = {}
    for record in records:
        record_id = str(record.get("id"))
        target = _target_sample(record, image_root)
        reference = _reference_sample(record, image_root)
        baselines[record_id] = {
            "target_raw": _answer_metrics(model, dict(target)),
            "reference_raw": _answer_metrics(model, dict(reference)) if reference else None,
            "generation": _maybe_generate(model, record, image_root, max_new_tokens, min_new_tokens, skip_generation),
        }

    per_edit_rows = []
    aggregate_rows = []
    for alpha in alphas:
        alpha_rows = []
        for case_index, (record, edit_id) in enumerate(zip(records, edit_ids)):
            record_id = str(record.get("id"))
            edit = bank.load_edit(edit_id)
            metadata = edit["metadata"]
            summary = _summarize_metadata(metadata, alpha)
            _restore_modules(model, snapshots)
            if alpha != 0.0:
                _apply_add_alpha(model, edit["updates"], alpha)
            target = _target_sample(record, image_root)
            reference = _reference_sample(record, image_root)
            target_after_raw = _answer_metrics(model, dict(target))
            reference_after_raw = _answer_metrics(model, dict(reference)) if reference else None
            generation_after = _maybe_generate(model, record, image_root, max_new_tokens, min_new_tokens, skip_generation)
            _restore_modules(model, snapshots)
            generation_after_rollback = _maybe_generate(model, record, image_root, max_new_tokens, min_new_tokens, skip_generation)
            rollback_diff = _max_snapshot_diff(model, snapshots)
            baseline = baselines[record_id]
            target_before = _strip(baseline["target_raw"])
            target_after = _strip(target_after_raw)
            reference_before = _strip(baseline["reference_raw"])
            reference_after = _strip(reference_after_raw)
            unavailable = None
            if target_before is None or target_after is None:
                unavailable = {"target_before": baseline["target_raw"], "target_after": target_after_raw}
            metrics = erasure_delta_metrics(
                target_before=target_before,
                target_after=target_after,
                reference_before=reference_before,
                reference_after=reference_after,
                target_generation_before=(baseline["generation"] or {}).get("decoded_stripped") if baseline["generation"] else None,
                target_generation_after=(generation_after or {}).get("decoded_stripped") if generation_after else None,
                unavailable_reason=json.dumps(unavailable, sort_keys=True) if unavailable else None,
            )
            row = {
                "config_id": config["config_id"],
                "stage": config["stage"],
                "token_scope": config["token_scope"],
                "module_group": config.get("module_group"),
                "record_id": record_id,
                "case_index": case_index,
                "edit_id": edit_id,
                "alpha": alpha,
                "matching_mode": matching.get("mode"),
                "engram_update_direction": "add",
                "direction_sign": 1,
                **summary,
                "erase_target_nll_before": metrics.get("erase_target_nll_before"),
                "erase_target_nll_after": metrics.get("erase_target_nll_after"),
                "target_nll_increase": metrics.get("erase_success_nll_increase"),
                "target_logprob_drop": metrics.get("erase_success_logprob_drop"),
                "reference_nll_before": metrics.get("reference_nll_before"),
                "reference_nll_after": metrics.get("reference_nll_after"),
                "reference_delta_abs": metrics.get("reference_delta_abs"),
                "target_to_reference_delta_ratio": (
                    None
                    if metrics.get("reference_delta_abs") in (None, 0.0) or metrics.get("erase_success_nll_increase") is None
                    else metrics.get("erase_success_nll_increase") / metrics.get("reference_delta_abs")
                ),
                "generation_before": baseline["generation"],
                "generation_after": generation_after,
                "generation_after_rollback": generation_after_rollback,
                "rollback_max_abs_diff": rollback_diff,
                "rollback_pass": rollback_diff <= rollback_tolerance,
                "target_before_raw": baseline["target_raw"],
                "target_after_raw": target_after_raw,
                "reference_before_raw": baseline["reference_raw"],
                "reference_after_raw": reference_after_raw,
                "erase_logprob_metrics_available": metrics.get("erase_logprob_metrics_available"),
                "record_id_match_rate": 1.0 if matching.get("mode") == "record_id" else 0.0,
                "nan_inf_detected": not _finite(metrics) or not _finite(target_after_raw) or not _finite(reference_after_raw),
            }
            per_edit_rows.append(row)
            alpha_rows.append(row)
        aggregate_rows.append(_aggregate(alpha_rows, locality_threshold))
    _restore_modules(model, snapshots)
    return {
        "status": "complete",
        "config": config,
        "bank_dir": str(bank_dir),
        "edit_record_matching": matching,
        "aggregate_rows": aggregate_rows,
        "per_edit": per_edit_rows,
    }


def _aggregate(rows: List[Dict[str, Any]], locality_threshold: float) -> Dict[str, Any]:
    metric_rows = [row for row in rows if row.get("erase_logprob_metrics_available")]
    target_nll = [float(row["target_nll_increase"]) for row in metric_rows if row.get("target_nll_increase") is not None]
    target_drop = [float(row["target_logprob_drop"]) for row in metric_rows if row.get("target_logprob_drop") is not None]
    ref_delta = [float(row["reference_delta_abs"]) for row in metric_rows if row.get("reference_delta_abs") is not None]
    mean_target = _mean(target_nll)
    mean_ref = _mean(ref_delta)
    negative_rows = [row for row in rows if row["record_id"] == NEGATIVE_RECORD_ID]
    negative = negative_rows[0] if negative_rows else {}
    return {
        "config_id": rows[0]["config_id"] if rows else None,
        "stage": rows[0]["stage"] if rows else None,
        "token_scope": rows[0]["token_scope"] if rows else None,
        "module_group": rows[0].get("module_group") if rows else None,
        "alpha": rows[0]["alpha"] if rows else None,
        "module_count": len(rows[0].get("selected_modules") or []) if rows else 0,
        "selected_module_names": rows[0].get("selected_modules") if rows else [],
        "selected_layer_depths": sorted({depth for row in rows for depth in (_extract_layer_depth(name) for name in row.get("selected_modules") or []) if depth is not None}),
        "mean_target_nll_increase": mean_target,
        "mean_target_logprob_drop": _mean(target_drop),
        "mean_reference_delta_abs": mean_ref,
        "target_to_reference_delta_ratio": None if mean_target is None or mean_ref in (None, 0.0) else mean_target / mean_ref,
        "positive_target_edits": sum(1 for value in target_nll if value > 0),
        "locality_damage_edits": sum(1 for value in ref_delta if value > locality_threshold),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in rows]),
        "record_id_match_rate": _mean([1.0 if row.get("matching_mode") == "record_id" else 0.0 for row in rows]),
        "nan_inf_count": sum(1 for row in rows if row.get("nan_inf_detected")),
        "empty_generation_count": sum(1 for row in rows if isinstance(row.get("generation_after"), dict) and row["generation_after"].get("generation_empty")),
        "synthetic_5edit_2_target_nll_increase": negative.get("target_nll_increase"),
        "synthetic_5edit_2_reference_delta_abs": negative.get("reference_delta_abs"),
        "target_activation_count": _mean([float(row.get("target_activation_count") or 0.0) for row in rows]),
        "reference_activation_count": _mean([float(row.get("reference_activation_count") or 0.0) for row in rows]),
        "norm_ratio_distribution": _merge_distributions([row.get("norm_ratio_distribution") for row in rows]),
        "effective_norm_ratio_distribution": _merge_distributions([row.get("effective_norm_ratio_distribution") for row in rows]),
        "fallback_used": any(bool(row.get("fallback_used")) for row in rows),
        "fallback_reasons": sorted({reason for row in rows for reason in row.get("fallback_reasons", [])}),
        "score": None if mean_target is None or mean_ref is None else mean_target - mean_ref,
        "score_ratio": None if mean_target is None or mean_ref is None else mean_target / (mean_ref + 1.0e-12),
    }


def _merge_distributions(items: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    values = []
    for item in items:
        if not item:
            continue
        for key in ("min", "max", "mean"):
            if item.get(key) is not None:
                values.append(float(item[key]))
    return _distribution(values)


def _select_best(rows: List[Dict[str, Any]], *, prefer_fix_negative: bool = False) -> Optional[Dict[str, Any]]:
    candidates = []
    for row in rows:
        if float(row.get("alpha") or 0.0) == 0.0:
            continue
        mean_target = row.get("mean_target_nll_increase")
        mean_ref = row.get("mean_reference_delta_abs")
        if mean_target is None or mean_ref is None:
            continue
        ok = (
            float(mean_target) > 0
            and int(row.get("positive_target_edits") or 0) >= 4
            and float(mean_ref) < float(mean_target)
            and float(row.get("rollback_pass_rate") or 0.0) == 1.0
            and float(row.get("record_id_match_rate") or 0.0) == 1.0
            and int(row.get("nan_inf_count") or 0) == 0
        )
        if ok:
            candidates.append(row)
    preferred = [row for row in candidates if int(row.get("locality_damage_edits") or 0) == 0] or candidates
    if prefer_fix_negative:
        fixing = [row for row in preferred if (row.get("synthetic_5edit_2_target_nll_increase") or 0.0) > 0]
        if fixing:
            preferred = fixing
    return max(preferred, key=lambda row: float(row.get("score") or float("-inf"))) if preferred else None


def _format(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _report_table(rows: List[Dict[str, Any]], group_key: str) -> List[str]:
    lines = [
        f"| {group_key} | alpha | mean target NLL inc | mean ref delta | score | positive | locality | rollback | match | synthetic-5edit-2 | fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {alpha} | {target} | {ref} | {score} | {pos} | {loc} | {roll} | {match} | {neg} | {fallback} |".format(
                name=row.get(group_key),
                alpha=_format(row.get("alpha")),
                target=_format(row.get("mean_target_nll_increase")),
                ref=_format(row.get("mean_reference_delta_abs")),
                score=_format(row.get("score")),
                pos=row.get("positive_target_edits"),
                loc=row.get("locality_damage_edits"),
                roll=_format(row.get("rollback_pass_rate")),
                match=_format(row.get("record_id_match_rate")),
                neg=_format(row.get("synthetic_5edit_2_target_nll_increase")),
                fallback=row.get("fallback_used"),
            )
        )
    return lines


def _write_token_report(out_dir: Path, rows: List[Dict[str, Any]], best: Optional[Dict[str, Any]]) -> None:
    lines = ["# Token-Scope Ablation", "", *_report_table(rows, "token_scope"), ""]
    fixed = [row for row in rows if float(row.get("alpha") or 0.0) > 0 and (row.get("synthetic_5edit_2_target_nll_increase") or 0.0) > 0]
    lines.extend(
        [
            f"Best token scope: `{best.get('token_scope') if best else None}` at alpha `{best.get('alpha') if best else None}`.",
            f"`loss_predictor` or `prompt_last` fixes synthetic-5edit-2: `{bool(fixed and any(row.get('token_scope') in {'loss_predictor', 'prompt_last'} for row in fixed))}`.",
            "",
        ]
    )
    (out_dir / "token_scope_ablation" / "REPORT_TOKEN_SCOPE_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")


def _write_module_report(out_dir: Path, rows: List[Dict[str, Any]], best: Optional[Dict[str, Any]]) -> None:
    lines = ["# Module-Scope Ablation", "", *_report_table(rows, "module_group"), ""]
    no_projector = [row for row in rows if row.get("module_group") == "no_projector" and float(row.get("alpha") or 0.0) > 0]
    sampled = [row for row in rows if row.get("module_group") in {"qk_gate_sampled_depths", "late_qk_gate", "qk_gate_all_layers_budgeted"} and float(row.get("alpha") or 0.0) > 0]
    skipped = []
    seen_skips = set()
    for row in rows:
        if row.get("status") != "skipped":
            continue
        key = (row.get("module_group"), tuple(row.get("skip_reasons") or []))
        if key in seen_skips:
            continue
        seen_skips.add(key)
        skipped.append(row)
    lines.extend(
        [
            f"Best module group: `{best.get('module_group') if best else None}` at alpha `{best.get('alpha') if best else None}`.",
            f"Projector removal helps synthetic-5edit-2: `{any((row.get('synthetic_5edit_2_target_nll_increase') or 0.0) > 0 for row in no_projector)}`.",
            f"Mid/late/all qk_gate fixes synthetic-5edit-2: `{any((row.get('synthetic_5edit_2_target_nll_increase') or 0.0) > 0 for row in sampled)}`.",
            "",
        ]
    )
    if skipped:
        lines.extend(["## Skipped Module Groups", ""])
        for row in skipped:
            lines.append(f"- `{row.get('module_group')}`: `{'; '.join(row.get('skip_reasons') or [])}`")
        lines.append("")
    (out_dir / "module_scope_ablation" / "REPORT_MODULE_SCOPE_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")


def _write_best_token(out_dir: Path, current_answer_row: Optional[Dict[str, Any]], best: Optional[Dict[str, Any]]) -> str:
    selected = best
    if current_answer_row and best and (best.get("score") or 0.0) <= (current_answer_row.get("score") or 0.0):
        selected = current_answer_row
    if selected is None and current_answer_row:
        selected = current_answer_row
    token_scope = str(selected.get("token_scope") if selected else "answer")
    lines = [
        "# Best Token Scope Decision",
        "",
        f"Selected token_scope: `{token_scope}`",
        f"Selected alpha: `{selected.get('alpha') if selected else None}`",
        f"Score: `{_format(selected.get('score') if selected else None)}`",
        f"synthetic-5edit-2 target NLL increase: `{_format(selected.get('synthetic_5edit_2_target_nll_increase') if selected else None)}`",
        "",
    ]
    if current_answer_row and selected and selected.get("token_scope") == "answer":
        lines.append("No tested token scope beat the current `answer` scope under the stated constraints, so module ablation keeps `answer`.")
    (out_dir / "BEST_TOKEN_SCOPE_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return token_scope


def _write_forensic(out_dir: Path, records: List[Dict[str, Any]], all_rows: List[Dict[str, Any]]) -> None:
    record = next(row for row in records if row.get("id") == NEGATIVE_RECORD_ID)
    rows = [row for row in all_rows if row.get("record_id") == NEGATIVE_RECORD_ID and float(row.get("alpha") or 0.0) > 0]
    best = max(rows, key=lambda row: float(row.get("target_nll_increase") or float("-inf"))) if rows else None
    worst = min(rows, key=lambda row: float(row.get("target_nll_increase") or float("inf"))) if rows else None
    before = best.get("target_before_raw") if best else {}
    reference = best.get("reference_before_raw") if best else {}
    target_tokens = set(str(record.get("pred", "")).lower().split())
    reference_tokens = set((str(record.get("m_loc_q", "")) + " " + str(record.get("m_loc_a", ""))).lower().split())
    overlap = len(target_tokens & reference_tokens) / max(len(target_tokens | reference_tokens), 1)
    checks = {}
    for label, predicate in {
        "loss_predictor": lambda r: r.get("token_scope") == "loss_predictor",
        "prompt_last": lambda r: r.get("token_scope") == "prompt_last",
        "no_projector": lambda r: r.get("module_group") == "no_projector",
        "sampled_qk_gate": lambda r: r.get("module_group") == "qk_gate_sampled_depths",
        "late_qk_gate": lambda r: r.get("module_group") == "late_qk_gate",
        "all_qk_gate_budgeted": lambda r: r.get("module_group") == "qk_gate_all_layers_budgeted",
    }.items():
        subset = [row for row in rows if predicate(row)]
        checks[label] = any((row.get("target_nll_increase") or 0.0) > 0 for row in subset)
    lines = [
        "# synthetic-5edit-2 Forensic Analysis",
        "",
        f"- Prompt: `{record.get('src')}`",
        f"- Old target answer: `{record.get('pred')}`",
        f"- Image path: `{record.get('image')}`",
        f"- Pre-edit target NLL: `{_format(before.get('nll'))}`",
        f"- Pre-edit target logprob: `{_format(before.get('logprob'))}`",
        f"- Answer token ids: `{before.get('answer_token_ids')}`",
        f"- Answer token count: `{before.get('answer_token_count')}`",
        f"- Reference prompt: `{record.get('m_loc_q')}`",
        f"- Reference answer: `{record.get('m_loc_a')}`",
        f"- Reference token ids: `{reference.get('answer_token_ids')}`",
        f"- Target/reference token overlap: `{_format(overlap)}`",
        "",
        "## Best/Worst Per-Token NLL",
        "",
        f"- Best config: `{best.get('config_id') if best else None}` alpha `{best.get('alpha') if best else None}`, target NLL increase `{_format(best.get('target_nll_increase') if best else None)}`",
        f"- Best before per-token NLL: `{(best or {}).get('target_before_raw', {}).get('per_token_nll')}`",
        f"- Best after per-token NLL: `{(best or {}).get('target_after_raw', {}).get('per_token_nll')}`",
        f"- Worst config: `{worst.get('config_id') if worst else None}` alpha `{worst.get('alpha') if worst else None}`, target NLL increase `{_format(worst.get('target_nll_increase') if worst else None)}`",
        f"- Worst before per-token NLL: `{(worst or {}).get('target_before_raw', {}).get('per_token_nll')}`",
        f"- Worst after per-token NLL: `{(worst or {}).get('target_after_raw', {}).get('per_token_nll')}`",
        "",
        "## Config Checks",
    ]
    for key, value in checks.items():
        lines.append(f"- Remains fixed under {key}: `{value}`")
    if best:
        lines.extend(
            [
                "",
                "## Best Config Diagnostics",
                "",
                f"- Target activation count: `{best.get('target_activation_count')}`",
                f"- Reference activation count: `{best.get('reference_activation_count')}`",
                f"- Selected modules: `{best.get('selected_modules')}`",
                f"- Norm ratios: `{best.get('norm_ratios')}`",
                f"- Effective norm ratios: `{best.get('effective_norm_ratios')}`",
                f"- Generation before: `{best.get('generation_before')}`",
                f"- Generation after: `{best.get('generation_after')}`",
                f"- Generation after rollback: `{best.get('generation_after_rollback')}`",
                "- Layer-level preactivation norm change: `not instrumented in this ablation runner`",
                "- Reference preactivation perturbation: `not instrumented in this ablation runner`",
            ]
        )
    likely = (
        "The negative direction is configuration-sensitive if any check is true; otherwise it is a persistent "
        "edit-specific erase direction issue under the tested ENGRAM token/module surfaces."
    )
    lines.extend(["", f"Likely cause: {likely}", ""])
    (out_dir / "synthetic_5edit_2_FORENSIC.md").write_text("\n".join(lines), encoding="utf-8")
    _json_dump(out_dir / "synthetic_5edit_2_FORENSIC.json", {"best": best, "worst": worst, "checks": checks, "overlap": overlap})


def _make_plots(out_dir: Path, token_rows: List[Dict[str, Any]], module_rows: List[Dict[str, Any]], all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for filename, rows, label_key in [
            (
                "token_scope_score.png",
                [row for row in token_rows if float(row.get("alpha") or 0.0) > 0 and row.get("score") is not None],
                "token_scope",
            ),
            (
                "module_scope_score.png",
                [row for row in module_rows if float(row.get("alpha") or 0.0) > 0 and row.get("score") is not None],
                "module_group",
            ),
        ]:
            labels = [f"{row.get(label_key)}@{row.get('alpha')}" for row in rows]
            scores = [float(row["score"]) for row in rows]
            plt.figure(figsize=(max(7, len(labels) * 0.55), 4))
            plt.bar(range(len(labels)), scores)
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            plt.ylabel("score")
            plt.tight_layout()
            plt.savefig(plot_dir / filename, dpi=160)
            plt.close()
        neg_rows = [
            row
            for row in all_rows
            if row.get("record_id") == NEGATIVE_RECORD_ID
            and float(row.get("alpha") or 0.0) > 0
            and row.get("target_nll_increase") is not None
        ]
        labels = [f"{row.get('config_id')}@{row.get('alpha')}" for row in neg_rows]
        vals = [float(row["target_nll_increase"]) for row in neg_rows]
        plt.figure(figsize=(max(8, len(labels) * 0.35), 4))
        plt.bar(range(len(labels)), vals)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
        plt.ylabel("synthetic-5edit-2 target NLL increase")
        plt.tight_layout()
        plt.savefig(plot_dir / "synthetic_5edit_2_by_config.png", dpi=160)
        plt.close()
        return {"status": "pass", "plot_dir": str(plot_dir)}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "plot_dir": str(plot_dir)}


def _write_final(
    out_dir: Path,
    token_rows: List[Dict[str, Any]],
    module_rows: List[Dict[str, Any]],
    best_config: Optional[Dict[str, Any]],
    token_best: Optional[Dict[str, Any]],
    module_best: Optional[Dict[str, Any]],
    plot_status: Dict[str, Any],
) -> None:
    fixed = best_config and (best_config.get("synthetic_5edit_2_target_nll_increase") or 0.0) > 0
    if best_config is None:
        decision = "C. Direct erase remains unreliable. Pivot to Engram-localized replacement/LoRA."
    elif fixed and int(best_config.get("positive_target_edits") or 0) == 5 and int(best_config.get("locality_damage_edits") or 0) == 0:
        decision = "A. Safe to run 5-edit sequential smoke. Use best alpha / 2 for first sequential run."
    else:
        decision = "B. Partial signal remains. Run filtered model-known data or stronger prompt-generation dataset before sequential."
    lines = [
        "# Final Token/Module Ablation Report",
        "",
        "## Starting Point",
        "",
        "- Previous best alpha: `0.075`",
        "- Previous decision: partial signal, no sequential run yet.",
        "- Persistent negative edit: `synthetic-5edit-2`.",
        "",
        "## Token-Scope Ablation",
        "",
        *_report_table(token_rows, "token_scope"),
        "",
        f"Best token row: `{token_best}`",
        "",
        "## Module-Scope Ablation",
        "",
        *_report_table(module_rows, "module_group"),
        "",
        f"Best module row: `{module_best}`",
        "",
        "## Best Configuration",
        "",
    ]
    if best_config:
        lines.extend(
            [
                f"- token_scope: `{best_config.get('token_scope')}`",
                f"- module_group: `{best_config.get('module_group')}`",
                f"- alpha: `{best_config.get('alpha')}`",
                f"- mean_target_nll_increase: `{_format(best_config.get('mean_target_nll_increase'))}`",
                f"- mean_reference_delta_abs: `{_format(best_config.get('mean_reference_delta_abs'))}`",
                f"- positive_target_edits: `{best_config.get('positive_target_edits')}`",
                f"- locality_damage_edits: `{best_config.get('locality_damage_edits')}`",
                f"- rollback_pass_rate: `{_format(best_config.get('rollback_pass_rate'))}`",
                f"- record_id_match_rate: `{_format(best_config.get('record_id_match_rate'))}`",
                f"- synthetic-5edit-2 target NLL increase: `{_format(best_config.get('synthetic_5edit_2_target_nll_increase'))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## synthetic-5edit-2 Diagnosis",
            "",
            f"- Fixed by any selected best configuration: `{bool(fixed)}`",
            "- See `synthetic_5edit_2_FORENSIC.md` for token/module details.",
            "",
            "## Plots",
            "",
            f"- Plot status: `{plot_status.get('status')}`",
            "",
            "## Decision",
            "",
            decision,
            "",
            "No sequential editing, no 20-edit run, and no replacement mode were run in this task.",
        ]
    )
    (out_dir / "FINAL_TOKEN_MODULE_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _preflight(out_dir: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "status": "pass" if len(records) == 5 and all(record.get("id") for record in records) else "fail",
        "raw_records_with_record_id": sum(1 for record in records if record.get("id")),
        "record_id_match_rate": 1.0 if len(records) == 5 and all(record.get("id") for record in records) else 0.0,
        "positional_matching_allowed_by_default": False,
        "note": "New ablation banks are checked after extraction; EngramBank refuses positional matching unless allow_positional_matching=True.",
        "record_ids": [record.get("id") for record in records],
    }
    _json_dump(out_dir / "record_id_preflight.json", payload)
    if payload["status"] != "pass":
        raise RuntimeError(f"raw record_id preflight failed: {payload}")
    return payload


def _write_stage_outputs(base: Path, subdir: str, filename: str, payload: Dict[str, Any], aggregate_rows: List[Dict[str, Any]]) -> None:
    stage_dir = base / subdir
    _json_dump(stage_dir / f"{filename}.json", payload)
    _write_csv(stage_dir / f"{filename}.csv", aggregate_rows)
    _write_csv(stage_dir / f"{filename}_per_edit.csv", payload["per_edit"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ENGRAM 5-edit token/module ablations before sequential scaling.")
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default="outputs/engram_token_module_ablation_5edit")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alphas", default="0.0,0.05,0.075")
    parser.add_argument("--best-alpha", type=float, default=0.075)
    parser.add_argument("--rollback-tolerance", type=float, default=1e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--max-modules-run", type=int, default=24)
    parser.add_argument("--reuse-token-scope", action="store_true", help="Read existing token-scope JSON/CSV and run only module-scope/final reporting.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(Path(args.data_file))
    preflight = _preflight(out_dir, records)
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    hparams.device = int(args.device) if str(args.device).isdigit() else args.device
    image_root_arg = Path(args.image_root)
    dataset_image_root = image_root_arg.parent if image_root_arg.name == "images" else image_root_arg
    hparams.coco_image = str(dataset_image_root)
    hparams.rephrase_image = str(dataset_image_root)
    shutil.copyfile(args.hparams, out_dir / "base_hparams.used.yaml")

    editor = MultimodalEditor.from_hparams(hparams)
    image_root = Path(args.image_root)
    token_results = []
    token_aggregates = []
    token_per_edit = []
    extraction_reports = []
    base_group = _module_group_specs()["smoke_4"]

    token_json = out_dir / "token_scope_ablation" / "token_scope_ablation.json"
    if args.reuse_token_scope:
        if not token_json.exists():
            raise FileNotFoundError(f"--reuse-token-scope requested but missing {token_json}")
        token_payload = json.loads(token_json.read_text(encoding="utf-8"))
        if token_payload.get("status") != "complete":
            raise RuntimeError(f"Cannot reuse incomplete token payload: {token_json}")
        token_aggregates = list(token_payload.get("aggregate_rows") or [])
        token_per_edit = list(token_payload.get("per_edit") or [])
        extraction_reports = list(token_payload.get("extraction_reports") or [])
    else:
        for token_scope in TOKEN_SCOPES:
            config_id = _make_config_id("token", token_scope)
            config = {"stage": "token_scope", "config_id": config_id, "token_scope": token_scope, "module_group": "smoke_4", **base_group}
            bank_dir = out_dir / "token_scope_ablation" / "banks" / config_id
            extract = _extract_bank_for_config(editor, hparams, Path(args.data_file), records, config, bank_dir, extraction_alpha=args.best_alpha)
            extraction_reports.append({"config_id": config_id, **extract})
            result = _evaluate_bank(
                editor.model,
                records,
                image_root,
                bank_dir,
                config,
                alphas,
                rollback_tolerance=args.rollback_tolerance,
                locality_threshold=args.locality_damage_threshold,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                skip_generation=args.skip_generation,
            )
            token_results.append(result)
            token_aggregates.extend(result["aggregate_rows"])
            token_per_edit.extend(result["per_edit"])

        token_payload = {
            "status": "complete",
            "preflight": preflight,
            "extraction_reports": extraction_reports,
            "aggregate_rows": token_aggregates,
            "per_edit": token_per_edit,
        }
        _write_stage_outputs(out_dir, "token_scope_ablation", "token_scope_ablation", token_payload, token_aggregates)
    current_answer = next((row for row in token_aggregates if row.get("token_scope") == "answer" and float(row.get("alpha") or 0.0) == args.best_alpha), None)
    token_best = _select_best(token_aggregates, prefer_fix_negative=True)
    selected_token_scope = _write_best_token(out_dir, current_answer, token_best)
    _write_token_report(out_dir, token_aggregates, token_best)

    module_results = []
    module_aggregates = []
    module_per_edit = []
    module_extraction_reports = []
    for group_name, group in _module_group_specs().items():
        config_id = _make_config_id("module", selected_token_scope, group_name)
        config = {"stage": "module_scope", "config_id": config_id, "token_scope": selected_token_scope, "module_group": group_name, **deepcopy(group)}
        _apply_config(hparams, config, out_dir / "module_scope_ablation" / "banks" / config_id, args.best_alpha)
        selected_layers = select_linear_layers(editor.model, hparams)
        if len(selected_layers) > args.max_modules_run:
            config["skip_reason"] = f"selected module count {len(selected_layers)} exceeds max_modules_run={args.max_modules_run}"
        bank_dir = out_dir / "module_scope_ablation" / "banks" / config_id
        extract = _extract_bank_for_config(editor, hparams, Path(args.data_file), records, config, bank_dir, extraction_alpha=args.best_alpha)
        module_extraction_reports.append({"config_id": config_id, "module_group": group_name, **extract})
        if extract["status"] == "skipped":
            for alpha in alphas:
                module_aggregates.append(
                    {
                        "config_id": config_id,
                        "stage": "module_scope",
                        "token_scope": selected_token_scope,
                        "module_group": group_name,
                        "alpha": alpha,
                        "status": "skipped",
                        "skip_reasons": [extract["skip_reason"]],
                        "module_count": extract.get("module_count"),
                        "selected_module_names": extract.get("selected_module_names"),
                        "selected_layer_depths": extract.get("selected_layer_depths"),
                        "covariance_memory_estimate": extract.get("covariance_memory_estimate"),
                        "record_id_match_rate": None,
                        "rollback_pass_rate": None,
                        "nan_inf_count": None,
                    }
                )
            continue
        result = _evaluate_bank(
            editor.model,
            records,
            image_root,
            bank_dir,
            config,
            alphas,
            rollback_tolerance=args.rollback_tolerance,
            locality_threshold=args.locality_damage_threshold,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            skip_generation=args.skip_generation,
        )
        module_results.append(result)
        module_aggregates.extend(result["aggregate_rows"])
        module_per_edit.extend(result["per_edit"])

    module_payload = {
        "status": "complete",
        "selected_token_scope": selected_token_scope,
        "extraction_reports": module_extraction_reports,
        "aggregate_rows": module_aggregates,
        "per_edit": module_per_edit,
    }
    _write_stage_outputs(out_dir, "module_scope_ablation", "module_scope_ablation", module_payload, module_aggregates)
    module_best = _select_best([row for row in module_aggregates if row.get("status") != "skipped"], prefer_fix_negative=True)
    _write_module_report(out_dir, module_aggregates, module_best)

    all_per_edit = token_per_edit + module_per_edit
    all_aggregate = token_aggregates + [row for row in module_aggregates if row.get("status") != "skipped"]
    best_config = _select_best(all_aggregate, prefer_fix_negative=True) or _select_best(all_aggregate)
    _json_dump(out_dir / "best_overall_config.json", best_config)
    _write_forensic(out_dir, records, all_per_edit)
    plot_status = _make_plots(out_dir, token_aggregates, module_aggregates, all_per_edit)
    _json_dump(out_dir / "plot_status.json", plot_status)
    _write_final(out_dir, token_aggregates, module_aggregates, best_config, token_best, module_best, plot_status)

    print(
        json.dumps(
            {
                "status": "complete",
                "selected_token_scope": selected_token_scope,
                "best_overall": best_config,
                "token_json": str(out_dir / "token_scope_ablation" / "token_scope_ablation.json"),
                "module_json": str(out_dir / "module_scope_ablation" / "module_scope_ablation.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
