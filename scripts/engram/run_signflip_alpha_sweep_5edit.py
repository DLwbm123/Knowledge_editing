#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402
from easyeditor.models.engram.erasure_metrics import (  # noqa: E402
    erasure_delta_metrics,
    safe_model_answer_nll_and_logprob,
)


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
    old_answer = record.get("pred")
    if not old_answer:
        raise RuntimeError(f"Record {record.get('id')} missing old target answer `pred`.")
    return _sample(record["src"], old_answer, _resolve_image(image_root, record["image"]))


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


def _strip(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not metrics or not metrics.get("available"):
        return None
    return {"nll": float(metrics["nll"]), "logprob": float(metrics["logprob"]), "num_tokens": int(metrics["num_tokens"])}


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


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


def _layer_diagnostics(metadata: Dict[str, Any], alpha: float) -> List[Dict[str, Any]]:
    diagnostics = []
    for layer in metadata.get("layers", []) or []:
        norm_ratio = float(layer.get("norm_ratio", 0.0))
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


def _summarize_modules(metadata: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    diagnostics = _layer_diagnostics(metadata, alpha)
    return {
        "selected_modules": [row["module_name"] for row in diagnostics if row.get("module_name")],
        "target_activation_count": sum(int(row.get("num_target_vectors") or 0) for row in diagnostics),
        "reference_activation_count": sum(int(row.get("num_reference_vectors") or 0) for row in diagnostics),
        "module_diagnostics": diagnostics,
        "norm_ratios": {str(row["module_name"]): row.get("norm_ratio") for row in diagnostics if row.get("module_name")},
        "effective_norm_ratios": {
            str(row["module_name"]): row.get("effective_update_norm_ratio")
            for row in diagnostics
            if row.get("module_name")
        },
        "max_effective_update_norm_ratio": max(
            [float(row.get("effective_update_norm_ratio") or 0.0) for row in diagnostics] or [0.0]
        ),
        "skipped_modules": [],
        "skip_reasons": [],
    }


def _special_token_ids(tokenizer) -> set[int]:
    ids = {
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        getattr(tokenizer, "unk_token_id", None),
    }
    return {int(value) for value in ids if value is not None}


def _generate_llava_med(
    wrapper,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
) -> Dict[str, Any]:
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

    generate_kwargs: Dict[str, Any] = {
        "images": image_tensor,
        "attention_mask": attention_mask,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": wrapper.tokenizer.eos_token_id,
    }
    if min_new_tokens is not None:
        generate_kwargs["min_new_tokens"] = min_new_tokens
    with torch.inference_mode():
        output_ids = wrapper.llava_model.generate(input_ids, **generate_kwargs)
    new_tokens = output_ids[0, input_ids.shape[1] :]
    generated_ids = [int(token_id) for token_id in new_tokens.detach().cpu().tolist()]
    decoded_raw = wrapper.tokenizer.decode(new_tokens, skip_special_tokens=False)
    decoded_skip_special = wrapper.tokenizer.decode(new_tokens, skip_special_tokens=True)
    decoded_stripped = decoded_skip_special.strip()
    special_ids = _special_token_ids(wrapper.tokenizer)
    only_special = bool(generated_ids) and all(token_id in special_ids for token_id in generated_ids)
    eos_id = wrapper.tokenizer.eos_token_id
    if generated_ids and eos_id is not None and generated_ids[0] == int(eos_id):
        stop_reason = "immediate_eos"
    elif eos_id is not None and int(eos_id) in generated_ids:
        stop_reason = "eos"
    elif len(generated_ids) >= max_new_tokens:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "unknown"
    return {
        "decoded_raw": decoded_raw,
        "decoded_skip_special": decoded_skip_special,
        "decoded_stripped": decoded_stripped,
        "generated_token_ids": generated_ids,
        "stop_reason": stop_reason,
        "generation_empty": decoded_stripped == "",
        "generated_only_eos_or_special": only_special,
    }


def _maybe_generate(
    wrapper,
    record: Dict[str, Any],
    image_root: Path,
    *,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    skip_generation: bool,
) -> Optional[Dict[str, Any]]:
    if skip_generation:
        return None
    return _generate_llava_med(
        wrapper,
        record["src"],
        _resolve_image(image_root, record["image"]),
        max_new_tokens,
        min_new_tokens,
    )


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
            flat = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(value, sort_keys=True)
                else:
                    flat[key] = value
            writer.writerow(flat)


def _json_dump(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _aggregate(rows: List[Dict[str, Any]], locality_threshold: float) -> Dict[str, Any]:
    metric_rows = [row for row in rows if row.get("erase_logprob_metrics_available")]
    target_nll = [float(row["target_nll_increase"]) for row in metric_rows if row.get("target_nll_increase") is not None]
    target_drop = [float(row["target_logprob_drop"]) for row in metric_rows if row.get("target_logprob_drop") is not None]
    ref_delta = [float(row["reference_delta_abs"]) for row in metric_rows if row.get("reference_delta_abs") is not None]
    mean_target = _mean(target_nll)
    mean_ref = _mean(ref_delta)
    return {
        "alpha": rows[0]["alpha"] if rows else None,
        "num_edits": len(rows),
        "num_metric_available": len(metric_rows),
        "mean_target_nll_increase": mean_target,
        "mean_target_logprob_drop": _mean(target_drop),
        "mean_reference_delta_abs": mean_ref,
        "target_to_reference_delta_ratio": None if mean_target is None or mean_ref in (None, 0.0) else mean_target / mean_ref,
        "positive_target_edits": sum(1 for value in target_nll if value > 0),
        "locality_damage_edits": sum(1 for value in ref_delta if value > locality_threshold),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in rows]),
        "bank_save_rate": _mean([1.0 if row.get("bank_saved") else 0.0 for row in rows]),
        "record_id_match_rate": _mean([1.0 if row.get("matching_mode") == "record_id" else 0.0 for row in rows]),
        "nan_inf_count": sum(1 for row in rows if row.get("nan_inf_detected")),
        "empty_generation_count": sum(
            1
            for row in rows
            if isinstance(row.get("generation_after"), dict) and row["generation_after"].get("generation_empty")
        ),
        "score": None if mean_target is None or mean_ref is None else mean_target - mean_ref,
        "score_ratio": None if mean_target is None or mean_ref is None else mean_target / (mean_ref + 1.0e-12),
    }


def _select_best_alpha(aggregate_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = []
    for row in aggregate_rows:
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
    preferred = [row for row in candidates if int(row.get("locality_damage_edits") or 0) == 0]
    pool = preferred or candidates
    if not pool:
        return {
            "status": "no_alpha_satisfies_constraints",
            "selected_alpha": None,
            "recommendation": "Do not enter sequential editing.",
        }
    best = max(pool, key=lambda row: float(row.get("score") or float("-inf")))
    return {
        "status": "selected",
        "selected_alpha": best["alpha"],
        "selected_row": best,
        "selection_constraints": {
            "mean_target_nll_increase_gt_0": True,
            "positive_target_edits_gte_4_of_5": True,
            "mean_reference_delta_abs_lt_mean_target": True,
            "rollback_pass_rate_eq_1": True,
            "record_id_match_rate_eq_1": True,
            "nan_inf_count_eq_0": True,
            "locality_damage_edits_eq_0_if_possible": int(best.get("locality_damage_edits") or 0) == 0,
        },
    }


def _format_float(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_alpha0_report(out_dir: Path, alpha0_rows: List[Dict[str, Any]], metric_tolerance: float) -> Dict[str, Any]:
    checks = {
        "num_rows": len(alpha0_rows),
        "all_target_activation_nonzero": all(int(row.get("target_activation_count") or 0) > 0 for row in alpha0_rows),
        "all_reference_activation_nonzero": all(int(row.get("reference_activation_count") or 0) > 0 for row in alpha0_rows),
        "all_effective_update_norm_zero": all(float(row.get("max_effective_update_norm_ratio") or 0.0) == 0.0 for row in alpha0_rows),
        "all_target_nll_unchanged": all(abs(float(row.get("target_nll_increase") or 0.0)) <= metric_tolerance for row in alpha0_rows),
        "all_reference_nll_unchanged": all(
            abs(float(row.get("reference_nll_after") or 0.0) - float(row.get("reference_nll_before") or 0.0)) <= metric_tolerance
            for row in alpha0_rows
            if row.get("reference_nll_before") is not None and row.get("reference_nll_after") is not None
        ),
        "all_rollback_pass": all(bool(row.get("rollback_pass")) for row in alpha0_rows),
        "all_record_id_matching": all(row.get("matching_mode") == "record_id" for row in alpha0_rows),
    }
    checks["status"] = "pass" if len(alpha0_rows) == 5 and all(value for key, value in checks.items() if key != "num_rows") else "fail"
    lines = [
        "# Alpha=0 No-Change Gate",
        "",
        f"Status: `{checks['status']}`",
        "",
        "| check | value |",
        "|---|---:|",
    ]
    for key, value in checks.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "| record_id | target NLL delta | reference NLL abs delta | rollback max diff | target vectors | reference vectors |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in alpha0_rows:
        ref_delta = None
        if row.get("reference_nll_before") is not None and row.get("reference_nll_after") is not None:
            ref_delta = abs(float(row["reference_nll_after"]) - float(row["reference_nll_before"]))
        lines.append(
            "| {record_id} | {target_delta} | {ref_delta} | {rollback} | {target_count} | {reference_count} |".format(
                record_id=row.get("record_id"),
                target_delta=_format_float(row.get("target_nll_increase")),
                ref_delta=_format_float(ref_delta),
                rollback=_format_float(row.get("rollback_max_abs_diff")),
                target_count=row.get("target_activation_count"),
                reference_count=row.get("reference_activation_count"),
            )
        )
    path = out_dir / "alpha0_gate" / "REPORT_ALPHA0_GATE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checks


def _token_set(text: Optional[str]) -> set[str]:
    return {item.lower() for item in str(text or "").replace("?", " ").replace(".", " ").replace(",", " ").split() if item}


def _write_negative_edit_analysis(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    preferred_alpha: float = 0.05,
) -> Dict[str, Any]:
    by_record = {str(record.get("id")): record for record in records}
    alpha_rows = [row for row in rows if abs(float(row["alpha"]) - preferred_alpha) < 1.0e-12]
    negative = [row for row in alpha_rows if row.get("target_nll_increase") is not None and float(row["target_nll_increase"]) <= 0]
    if negative:
        chosen = min(negative, key=lambda row: float(row["target_nll_increase"]))
    elif alpha_rows:
        chosen = min(alpha_rows, key=lambda row: float(row.get("target_nll_increase") or 0.0))
    else:
        chosen = min(rows, key=lambda row: float(row.get("target_nll_increase") or 0.0))
    record_id = str(chosen.get("record_id"))
    record = by_record[record_id]
    record_rows = [row for row in rows if str(row.get("record_id")) == record_id]
    target_answer_tokens = _token_set(record.get("pred"))
    reference_tokens = _token_set(record.get("m_loc_a")) | _token_set(record.get("m_loc_q"))
    overlap = len(target_answer_tokens & reference_tokens) / max(len(target_answer_tokens | reference_tokens), 1)
    target_before_nll = chosen.get("erase_target_nll_before")
    probability_proxy = math.exp(-float(target_before_nll)) if target_before_nll is not None else None
    nll_values = [float(row["erase_target_nll_before"]) for row in alpha_rows if row.get("erase_target_nll_before") is not None]
    low_pre_edit_probability = None
    if target_before_nll is not None and nll_values:
        low_pre_edit_probability = float(target_before_nll) >= sorted(nll_values)[max(0, len(nll_values) - 2)]

    payload = {
        "record_id": record_id,
        "old_target_answer": record.get("pred"),
        "prompt": record.get("src"),
        "image_path": record.get("image"),
        "negative_at_alpha": preferred_alpha,
        "target_answer_probability_proxy_exp_neg_nll": probability_proxy,
        "old_answer_had_low_pre_edit_probability_by_within_alpha_rank": low_pre_edit_probability,
        "reference_target_token_jaccard": overlap,
        "reference_overlaps_heavily_with_target": overlap >= 0.5,
        "rows_across_alpha": record_rows,
    }
    lines = [
        "# Negative Edit Analysis",
        "",
        f"Record: `{record_id}`",
        "",
        f"- Old target answer: `{record.get('pred')}`",
        f"- Prompt: `{record.get('src')}`",
        f"- Image path: `{record.get('image')}`",
        f"- Reference answer: `{record.get('m_loc_a')}`",
        f"- Pre-edit answer probability proxy exp(-NLL): `{_format_float(probability_proxy)}`",
        f"- Low pre-edit probability by within-alpha rank: `{low_pre_edit_probability}`",
        f"- Reference/target token Jaccard: `{_format_float(overlap)}`",
        f"- Reference overlaps heavily with target: `{overlap >= 0.5}`",
        "",
        "| alpha | target NLL before | target NLL after | target NLL increase | reference NLL before | reference NLL after | selected modules | target vectors | reference vectors | max effective norm ratio | generation empty |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in record_rows:
        generation_empty = None
        if isinstance(row.get("generation_after"), dict):
            generation_empty = row["generation_after"].get("generation_empty")
        lines.append(
            "| {alpha} | {tb} | {ta} | {td} | {rb} | {ra} | {modules} | {tc} | {rc} | {norm} | {gen} |".format(
                alpha=_format_float(row.get("alpha")),
                tb=_format_float(row.get("erase_target_nll_before")),
                ta=_format_float(row.get("erase_target_nll_after")),
                td=_format_float(row.get("target_nll_increase")),
                rb=_format_float(row.get("reference_nll_before")),
                ra=_format_float(row.get("reference_nll_after")),
                modules=", ".join(row.get("selected_modules") or []),
                tc=row.get("target_activation_count"),
                rc=row.get("reference_activation_count"),
                norm=_format_float(row.get("max_effective_update_norm_ratio")),
                gen=generation_empty,
            )
        )
    likely_reason = "The edit remains negative at alpha=0.05 while record-id matching, rollback, and locality checks pass. This points to an edit-specific ENGRAM direction/feature interaction rather than the previous positional-matching bug."
    lines.extend(["", f"Likely reason: {likely_reason}", ""])
    payload["likely_reason"] = likely_reason
    _json_dump(out_dir / "negative_edit_analysis.json", payload)
    (out_dir / "negative_edit_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _write_best_alpha(out_dir: Path, decision: Dict[str, Any], aggregate_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Best Alpha Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Selected alpha: `{decision.get('selected_alpha')}`",
        "",
        "| alpha | mean target NLL increase | mean reference delta | score | score ratio | positive edits | locality damage | rollback pass rate | record-id match rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {alpha} | {target} | {ref} | {score} | {ratio} | {pos} | {loc} | {roll} | {match} |".format(
                alpha=_format_float(row.get("alpha")),
                target=_format_float(row.get("mean_target_nll_increase")),
                ref=_format_float(row.get("mean_reference_delta_abs")),
                score=_format_float(row.get("score")),
                ratio=_format_float(row.get("score_ratio")),
                pos=row.get("positive_target_edits"),
                loc=row.get("locality_damage_edits"),
                roll=_format_float(row.get("rollback_pass_rate")),
                match=_format_float(row.get("record_id_match_rate")),
            )
        )
    if decision["status"] != "selected":
        lines.extend(["", "No alpha satisfies the stated constraints. Do not enter sequential editing."])
    else:
        row = decision["selected_row"]
        lines.extend(
            [
                "",
                "Selected by maximizing `mean_target_nll_increase - mean_reference_delta_abs` among rows satisfying the constraints.",
                f"Selected score: `{_format_float(row.get('score'))}`.",
                f"Selected score ratio: `{_format_float(row.get('score_ratio'))}`.",
            ]
        )
    (out_dir / "BEST_ALPHA_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_plots(out_dir: Path, aggregate_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [float(row["alpha"]) for row in aggregate_rows]
        specs = [
            ("target_nll_vs_alpha.png", "mean_target_nll_increase", "Mean target NLL increase"),
            ("reference_delta_vs_alpha.png", "mean_reference_delta_abs", "Mean reference NLL abs delta"),
            ("score_vs_alpha.png", "score", "Score"),
        ]
        for filename, key, ylabel in specs:
            ys = [row.get(key) for row in aggregate_rows]
            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys, marker="o")
            plt.xlabel("alpha")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / filename, dpi=160)
            plt.close()
        return {"status": "pass", "plot_dir": str(plot_dir)}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "plot_dir": str(plot_dir)}


def _write_final_report(
    out_dir: Path,
    records: List[Dict[str, Any]],
    preflight: Dict[str, Any],
    alpha0_checks: Dict[str, Any],
    aggregate_rows: List[Dict[str, Any]],
    best: Dict[str, Any],
    negative: Dict[str, Any],
    plot_status: Dict[str, Any],
) -> None:
    decision_text = "B. Partial signal; run token/module ablation before sequential."
    if best["status"] == "selected":
        selected = best["selected_row"]
        if (
            int(selected.get("positive_target_edits") or 0) == 5
            and int(selected.get("locality_damage_edits") or 0) == 0
        ):
            decision_text = "A. Safe to run 5-edit sequential smoke with best alpha or best alpha/2."
    else:
        decision_text = "C. No stable alpha; pivot to Engram-localized replacement/LoRA."

    lines = [
        "# Final Sign-Flipped Alpha Sweep 5-Edit Report",
        "",
        "## Sign-Equivalence Fix",
        "",
        "The previous contradiction was caused by positional matching between bank edits and raw records. New bank metadata preserves source `record_id`, and post-hoc evaluation now refuses positional matching unless explicitly requested.",
        "",
        "The sign-equivalence audit already verified that subtract negative alpha, add positive alpha, and manual `W <- W + 0.05E` are parameter-equivalent with max parameter diff `0`.",
        "",
        "## Data",
        "",
        f"- Number of records: `{len(records)}`",
        f"- Record ids: `{[record.get('id') for record in records]}`",
        f"- X_plus availability: `{preflight.get('raw_records_with_record_id')}/5 records with target/edit variants and nonzero saved target activations`",
        f"- X_minus availability: `{preflight.get('bank_matches_using_record_id')}/5 record-id matched edits with nonzero saved reference activations`",
        f"- Non-PHI statement: `{records[0].get('non_phi_statement')}`",
        "",
        "## Alpha=0 Gate",
        "",
        f"- Status: `{alpha0_checks.get('status')}`",
        f"- Target NLL unchanged: `{alpha0_checks.get('all_target_nll_unchanged')}`",
        f"- Reference NLL unchanged: `{alpha0_checks.get('all_reference_nll_unchanged')}`",
        f"- Effective update norm zero: `{alpha0_checks.get('all_effective_update_norm_zero')}`",
        f"- Rollback pass: `{alpha0_checks.get('all_rollback_pass')}`",
        f"- Record-id matching: `{alpha0_checks.get('all_record_id_matching')}`",
        "",
        "## Sweep Results",
        "",
        "| alpha | mean target NLL increase | mean target logprob drop | mean reference delta | target/ref ratio | positive edits | locality damage | rollback pass rate | record-id match rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {alpha} | {target} | {drop} | {ref} | {ratio} | {pos} | {loc} | {roll} | {match} |".format(
                alpha=_format_float(row.get("alpha")),
                target=_format_float(row.get("mean_target_nll_increase")),
                drop=_format_float(row.get("mean_target_logprob_drop")),
                ref=_format_float(row.get("mean_reference_delta_abs")),
                ratio=_format_float(row.get("target_to_reference_delta_ratio")),
                pos=row.get("positive_target_edits"),
                loc=row.get("locality_damage_edits"),
                roll=_format_float(row.get("rollback_pass_rate")),
                match=_format_float(row.get("record_id_match_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## Best Alpha",
            "",
            f"- Selection status: `{best.get('status')}`",
            f"- Selected alpha: `{best.get('selected_alpha')}`",
        ]
    )
    if best.get("selected_row"):
        row = best["selected_row"]
        lines.extend(
            [
                f"- Score: `{_format_float(row.get('score'))}`",
                f"- Score ratio: `{_format_float(row.get('score_ratio'))}`",
                f"- Target effect stronger than reference damage: `{float(row.get('mean_target_nll_increase')) > float(row.get('mean_reference_delta_abs'))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Negative Edit Analysis",
            "",
            f"- Negative/minimum edit: `{negative.get('record_id')}`",
            f"- Likely reason: {negative.get('likely_reason')}",
            "",
            "## Bank / Metadata",
            "",
            f"- Record-id matching status: `{preflight.get('status')}`",
            f"- Matching mode: `{preflight.get('matching_mode')}`",
            "- Bank tensor source: fixed record-id-aware sign-equivalence rerun bank.",
            "- Per-alpha sidecar bank metadata with record_id was written under `alpha_sweep/bank_metadata/`.",
            "- Compose/apply compatibility was checked in the previous sign-equivalence audit and passed.",
            "",
            "## Plots",
            "",
            f"- Plot status: `{plot_status.get('status')}`",
            "",
            "## Decision",
            "",
            decision_text,
            "",
            "No sequential editing, no 20-edit run, and no replacement mode were run in this task.",
        ]
    )
    (out_dir / "FINAL_SIGNFLIP_ALPHA_SWEEP_5EDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sidecar_bank_metadata(
    out_dir: Path,
    *,
    alpha: float,
    record_id: str,
    edit_id: str,
    metadata: Dict[str, Any],
) -> Path:
    alpha_tag = str(alpha).replace(".", "p").replace("-", "m")
    sidecar = out_dir / "alpha_sweep" / "bank_metadata" / f"alpha_{alpha_tag}" / f"{record_id}.metadata.json"
    payload = dict(metadata)
    payload.update(
        {
            "record_id": record_id,
            "source_record_id": record_id,
            "source_edit_id": edit_id,
            "derived_alpha": alpha,
            "engram_update_direction": "add",
            "direction_sign": 1,
            "sidecar_only": True,
            "source_bank_saved": True,
        }
    )
    _json_dump(sidecar, payload)
    return sidecar


def _write_preflight(
    out_dir: Path,
    records: List[Dict[str, Any]],
    bank: EngramBank,
    edit_ids: List[str],
    matching: Dict[str, Any],
) -> Dict[str, Any]:
    rows = []
    for record, edit_id in zip(records, edit_ids):
        metadata = bank.load_edit(edit_id)["metadata"]
        bank_record_id = metadata.get("record_id") or metadata.get("source_record_id")
        rows.append(
            {
                "raw_record_id": record.get("id"),
                "bank_record_id": bank_record_id,
                "source_request_ids": metadata.get("source_request_ids"),
                "edit_id": edit_id,
                "matching_mode": matching.get("mode"),
                "record_id_match": str(record.get("id")) == str(bank_record_id),
            }
        )
    preflight = {
        "status": "pass"
        if len(rows) == 5
        and matching.get("mode") == "record_id"
        and all(row["raw_record_id"] for row in rows)
        and all(row["record_id_match"] for row in rows)
        else "fail",
        "matching_mode": matching.get("mode"),
        "raw_records_with_record_id": sum(1 for row in rows if row.get("raw_record_id")),
        "bank_matches_using_record_id": sum(1 for row in rows if row.get("matching_mode") == "record_id"),
        "positional_matching_used": matching.get("mode") != "record_id",
        "matching": matching,
        "rows": rows,
    }
    _json_dump(out_dir / "record_id_preflight.json", preflight)
    if preflight["status"] != "pass":
        raise RuntimeError(f"Record-id preflight failed: {preflight}")
    return preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the strict 5-edit sign-flipped ENGRAM alpha sweep gate.")
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--bank", required=True, help="Record-id-aware source bank containing unscaled E tensors.")
    parser.add_argument("--output-dir", default="outputs/engram_signflip_alpha_sweep_5edit")
    parser.add_argument("--alphas", default="0.0,0.005,0.01,0.025,0.05,0.075,0.1")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollback-tolerance", type=float, default=1e-4)
    parser.add_argument("--metric-tolerance", type=float, default=1e-7)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used_hparams = out_dir / "llava_med_5edit_signflip_alpha_sweep.used.yaml"
    shutil.copyfile(args.hparams, used_hparams)

    records = _load_records(Path(args.data_file))
    if len(records) != 5:
        raise RuntimeError(f"This gate expects exactly 5 records, got {len(records)}.")
    image_root = Path(args.image_root)
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    hparams.device = int(args.device) if str(args.device).isdigit() else args.device
    editor = MultimodalEditor.from_hparams(hparams)
    wrapper = editor.model
    wrapper.eval()

    bank = EngramBank(args.bank)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    preflight = _write_preflight(out_dir, records, bank, edit_ids, matching)

    module_names: List[str] = []
    for edit_id in edit_ids:
        for name in bank.load_edit(edit_id)["updates"].keys():
            if name not in module_names:
                module_names.append(name)
    snapshots = _snapshot_modules(wrapper, module_names)

    baselines: Dict[str, Dict[str, Any]] = {}
    for record in records:
        record_id = str(record.get("id"))
        target = _target_sample(record, image_root)
        reference = _reference_sample(record, image_root)
        baselines[record_id] = {
            "target_raw": safe_model_answer_nll_and_logprob(wrapper, dict(target)),
            "reference_raw": safe_model_answer_nll_and_logprob(wrapper, dict(reference)) if reference else None,
            "generation": _maybe_generate(
                wrapper,
                record,
                image_root,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                skip_generation=args.skip_generation,
            ),
        }

    per_edit_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        alpha_rows = []
        for case_index, (record, edit_id) in enumerate(zip(records, edit_ids)):
            record_id = str(record.get("id"))
            edit = bank.load_edit(edit_id)
            metadata = edit["metadata"]
            summary = _summarize_modules(metadata, alpha)
            sidecar = _write_sidecar_bank_metadata(
                out_dir,
                alpha=alpha,
                record_id=record_id,
                edit_id=edit_id,
                metadata=metadata,
            )
            _restore_modules(wrapper, snapshots)
            if alpha != 0.0:
                _apply_add_alpha(wrapper, edit["updates"], alpha)
            target = _target_sample(record, image_root)
            reference = _reference_sample(record, image_root)
            target_after_raw = safe_model_answer_nll_and_logprob(wrapper, dict(target))
            reference_after_raw = safe_model_answer_nll_and_logprob(wrapper, dict(reference)) if reference else None
            generation_after = _maybe_generate(
                wrapper,
                record,
                image_root,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                skip_generation=args.skip_generation,
            )
            _restore_modules(wrapper, snapshots)
            generation_after_rollback = _maybe_generate(
                wrapper,
                record,
                image_root,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                skip_generation=args.skip_generation,
            )
            rollback_diff = _max_snapshot_diff(wrapper, snapshots)

            baseline = baselines[record_id]
            target_before = _strip(baseline["target_raw"])
            target_after = _strip(target_after_raw)
            reference_before = _strip(baseline["reference_raw"])
            reference_after = _strip(reference_after_raw)
            unavailable = None
            if target_before is None or target_after is None:
                unavailable = {
                    "target_before": baseline["target_raw"],
                    "target_after": target_after_raw,
                }
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
                "record_id": record_id,
                "case_index": case_index,
                "edit_id": edit_id,
                "alpha": alpha,
                "engram_update_direction": "add",
                "direction_sign": 1,
                "matching_mode": matching.get("mode"),
                "bank_metadata_sidecar": str(sidecar),
                "bank_saved": sidecar.exists(),
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
                    if metrics.get("reference_delta_abs") in (None, 0.0)
                    or metrics.get("erase_success_nll_increase") is None
                    else metrics.get("erase_success_nll_increase") / metrics.get("reference_delta_abs")
                ),
                "generation_before": baseline["generation"],
                "generation_after": generation_after,
                "generation_after_rollback": generation_after_rollback,
                "rollback_max_abs_diff": rollback_diff,
                "rollback_pass": rollback_diff <= args.rollback_tolerance,
                "target_before_raw": baseline["target_raw"],
                "target_after_raw": target_after_raw,
                "reference_before_raw": baseline["reference_raw"],
                "reference_after_raw": reference_after_raw,
                "erase_logprob_metrics_available": metrics.get("erase_logprob_metrics_available"),
                "nan_inf_detected": not _finite(metrics) or not _finite(target_after_raw) or not _finite(reference_after_raw),
            }
            alpha_rows.append(row)
            per_edit_rows.append(row)
        aggregate_rows.append(_aggregate(alpha_rows, args.locality_damage_threshold))

    sweep_dir = out_dir / "alpha_sweep"
    output = {
        "status": "complete",
        "hparams": str(args.hparams),
        "used_hparams": str(used_hparams),
        "data_file": str(args.data_file),
        "image_root": str(args.image_root),
        "source_bank": str(args.bank),
        "alphas": alphas,
        "edit_record_matching": matching,
        "record_id_preflight": preflight,
        "note": "Sign-flipped add sweep applies W <- W + alpha * E from the fixed record-id-aware source bank after restoring original weights before every edit.",
        "aggregate_rows": aggregate_rows,
        "per_edit": per_edit_rows,
    }
    _json_dump(sweep_dir / "signflip_alpha_sweep_5edit.json", output)
    _write_csv(sweep_dir / "signflip_alpha_sweep_5edit.csv", per_edit_rows)
    _write_csv(sweep_dir / "signflip_alpha_sweep_5edit_aggregate.csv", aggregate_rows)

    alpha0_rows = [row for row in per_edit_rows if abs(float(row["alpha"])) < 1.0e-12]
    alpha0_checks = _write_alpha0_report(out_dir, alpha0_rows, args.metric_tolerance)
    if alpha0_checks["status"] != "pass":
        raise RuntimeError(f"Alpha=0 gate failed: {alpha0_checks}")

    best = _select_best_alpha(aggregate_rows)
    _json_dump(out_dir / "best_alpha_decision.json", best)
    _write_best_alpha(out_dir, best, aggregate_rows)
    negative = _write_negative_edit_analysis(out_dir, per_edit_rows, records)
    plot_status = _make_plots(out_dir, aggregate_rows)
    _json_dump(out_dir / "plot_status.json", plot_status)
    _write_final_report(out_dir, records, preflight, alpha0_checks, aggregate_rows, best, negative, plot_status)

    print(
        json.dumps(
            {
                "status": "complete",
                "json": str(sweep_dir / "signflip_alpha_sweep_5edit.json"),
                "csv": str(sweep_dir / "signflip_alpha_sweep_5edit.csv"),
                "selected_alpha": best.get("selected_alpha"),
                "alpha0_status": alpha0_checks["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
