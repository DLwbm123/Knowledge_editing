#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.engram.run_medmkeb_modelknown_editing import (  # noqa: E402
    DEFAULT_HPARAMS,
    DEFAULT_OUTPUT_DIR,
    EXPECTED_MODULES,
    _finite,
    _format,
    _heavy_imports,
    _json_dump,
    _mean,
    _package_hygiene,
    _plot_optional,
    _run_pytest,
    _safe_div,
    _write_csv,
    _write_env_report,
    _write_git_outputs,
    _write_preflight,
)
from scripts.engram.run_medmkeb_sequential_pareto_refine import (  # noqa: E402
    _configure_hparams_for_scope,
    _load_records,
    _read_json,
    _write_data_reuse_report,
)


ROUTED_DIRNAME = "routed_bank_20"
METHOD_ROUTED = "ENGRAM_routed_edit_bank"

QK_GATE_MODULES = list(EXPECTED_MODULES)

B_MERGED_NEW = 1.77491
B_MERGED_REF = 0.469512
B_MERGED_DAMAGE = 19
C_HIGH_NEW = 1.69794
C_HIGH_REF = 0.293396
C_HIGH_DAMAGE = 18
C_LOW_NEW = 0.302485
C_LOW_REF = 0.0387687
C_LOW_DAMAGE = 6
C_BOUNDED_NEW = 0.361779
C_BOUNDED_REF = 0.048950
C_BOUNDED_DAMAGE = 7


def _record_id(record: Dict[str, Any]) -> str:
    return str(record.get("record_id") or record.get("id"))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _ensure_layout(out_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": out_dir,
        "audit": out_dir / "audit",
        "tests": out_dir / "test_logs",
        "bank_metadata": out_dir / "bank_metadata",
        "runs": out_dir / "runs",
        "plots": out_dir / "plots",
        "generation": out_dir / "generation_diagnostics",
        "runtime_projector_banks": out_dir / "runtime_projector_banks",
        "runtime_edit_banks": out_dir / "runtime_edit_banks",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_tests(out_dir: Path, run_tests: bool) -> Dict[str, Any]:
    test_dir = out_dir / "test_logs"
    test_dir.mkdir(parents=True, exist_ok=True)
    if not run_tests:
        payload = {"status": "skipped", "reason": "--skip-tests"}
        _json_dump(test_dir / "test_status.json", payload)
        return payload
    engram_tests = sorted(str(path.relative_to(PROJECT_ROOT)) for path in PROJECT_ROOT.glob("tests/test_engram_*.py"))
    runs = [
        _run_pytest(test_dir / "test_engram_all.log", [*engram_tests, "-q"]),
        _run_pytest(test_dir / "test_routed_bank.log", ["tests/test_engram_routed_bank.py", "-q"]),
        _run_pytest(test_dir / "test_cure_crisp_projection.log", ["tests/test_cure_crisp_projection.py", "-q"]),
        _run_pytest(test_dir / "test_cure_kfac_collector_tiny_mllm.log", ["tests/test_cure_kfac_collector_tiny_mllm.py", "-q"]),
    ]
    payload = {
        "status": "pass" if runs[0]["returncode"] == 0 and runs[1]["returncode"] == 0 else "fail",
        "engram_tests_pass": runs[0]["returncode"] == 0,
        "routed_bank_tests_pass": runs[1]["returncode"] == 0,
        "cure_projection_tests_pass": runs[2]["returncode"] == 0,
        "cure_kfac_tests_pass": runs[3]["returncode"] == 0,
        "cure_tests_blocking": False,
        "runs": runs,
    }
    _json_dump(test_dir / "test_status.json", payload)
    return payload


def _bank_configs() -> List[Dict[str, Any]]:
    return [
        {
            "bank_config_id": "bank_high_strength",
            "beta": 0.5,
            "lora_steps": 20,
            "module_scope": "qk_gate_sampled_depths",
            "token_scope": "all",
        }
    ]


def _routed_configs() -> List[Dict[str, Any]]:
    return [
        {
            "config_id": "R_oracle_self_high",
            "method": METHOD_ROUTED,
            "bank_config_id": "bank_high_strength",
            "route_policy": "oracle_self",
            "prototype_type": "target",
            "threshold": None,
            "max_active_edits": 1,
        },
        {
            "config_id": "R_top1_target_no_threshold_high",
            "method": METHOD_ROUTED,
            "bank_config_id": "bank_high_strength",
            "route_policy": "top1_no_threshold",
            "prototype_type": "target",
            "threshold": None,
            "max_active_edits": 1,
        },
        {
            "config_id": "R_top1_target_threshold_0.5_high",
            "method": METHOD_ROUTED,
            "bank_config_id": "bank_high_strength",
            "route_policy": "top1_threshold",
            "prototype_type": "target",
            "threshold": 0.5,
            "max_active_edits": 1,
        },
        {
            "config_id": "R_top1_target_adaptive_high",
            "method": METHOD_ROUTED,
            "bank_config_id": "bank_high_strength",
            "route_policy": "top1_threshold",
            "prototype_type": "target",
            "threshold": "adaptive_p90_reference",
            "max_active_edits": 1,
        },
        {
            "config_id": "R_top3_target_adaptive_high",
            "method": METHOD_ROUTED,
            "bank_config_id": "bank_high_strength",
            "route_policy": "top3_threshold",
            "prototype_type": "target",
            "threshold": "adaptive_p90_reference",
            "max_active_edits": 3,
        },
        {
            "config_id": "R_top1_contrastive_adaptive_high",
            "method": METHOD_ROUTED,
            "bank_config_id": "bank_high_strength",
            "route_policy": "top1_threshold",
            "prototype_type": "contrastive",
            "threshold": "adaptive_p90_reference",
            "max_active_edits": 1,
        },
    ]


def _anchor_rows() -> List[Dict[str, Any]]:
    return [
        {
            "config_id": "B_merged_tiny_lora",
            "status": "anchor_reused",
            "final_new": B_MERGED_NEW,
            "final_ref": B_MERGED_REF,
            "positive_new": 20,
            "locality_damage": B_MERGED_DAMAGE,
            "source": "previous merged sequential tiny LoRA baseline",
        },
        {
            "config_id": "C_merged_high",
            "status": "anchor_reused",
            "final_new": C_HIGH_NEW,
            "final_ref": C_HIGH_REF,
            "positive_new": 20,
            "locality_damage": C_HIGH_DAMAGE,
            "source": "previous merged sequential ENGRAM-projected LoRA high-strength anchor",
        },
        {
            "config_id": "C_merged_low",
            "status": "anchor_reused",
            "final_new": C_LOW_NEW,
            "final_ref": C_LOW_REF,
            "positive_new": 20,
            "locality_damage": C_LOW_DAMAGE,
            "source": "previous merged sequential low-drift rescue anchor",
        },
        {
            "config_id": "C_merged_bounded",
            "status": "anchor_reused",
            "final_new": C_BOUNDED_NEW,
            "final_ref": C_BOUNDED_REF,
            "positive_new": 20,
            "locality_damage": C_BOUNDED_DAMAGE,
            "source": "previous merged sequential bounded refine anchor",
        },
    ]


def _module_scope_names(scope: str) -> List[str]:
    if scope != "qk_gate_sampled_depths":
        raise ValueError(f"unsupported module_scope: {scope}")
    return list(QK_GATE_MODULES)


def _tensor_summary(value: Any) -> Dict[str, Any]:
    detached = value.detach().float().cpu()
    return {
        "shape": list(detached.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "norm": float(detached.norm().item()),
        "max_abs": float(detached.abs().max().item()) if detached.numel() else 0.0,
    }


class RoutedLoraPatch:
    """Temporary additive LoRA-factor patch used for routed inference."""

    def __init__(self, model: Any, active_entries: Sequence[Dict[str, Any]], weights: Sequence[float]) -> None:
        self.model = model
        self.active_entries = list(active_entries)
        self.weights = [float(item) for item in weights]
        self.original_forwards: Dict[str, Any] = {}

    def install(self) -> None:
        from scripts.engram.run_token_module_ablation_5edit import _module_map

        modules = _module_map(self.model)
        module_names = sorted(
            {
                name
                for entry in self.active_entries
                for name in (entry.get("factors") or {}).keys()
            }
        )
        for module_name in module_names:
            module = modules.get(module_name)
            if module is None:
                raise RuntimeError(f"routed patch target missing: {module_name}")
            terms = []
            for entry, active_weight in zip(self.active_entries, self.weights):
                factor = (entry.get("factors") or {}).get(module_name)
                if not factor:
                    continue
                a = factor["A"].to(module.weight.device, dtype=module.weight.dtype)
                b = factor["B"].to(module.weight.device, dtype=module.weight.dtype)
                scale = float(factor.get("scale", 1.0)) * float(entry.get("beta", 1.0)) * float(active_weight)
                terms.append((a, b, scale))
            if not terms:
                continue
            self.original_forwards[module_name] = module.forward

            def patched_forward(x, *, _base=module.forward, _terms=tuple(terms)):
                base = _base(x)
                total = None
                x_float = x.to(dtype=base.dtype)
                for _a, _b, _scale in _terms:
                    low = torch_linear(x_float, _a)
                    delta = torch_linear(low, _b) * float(_scale)
                    total = delta if total is None else total + delta
                if total is None:
                    return base
                return base + total.to(dtype=base.dtype)

            module.forward = patched_forward  # type: ignore[method-assign]

    def remove(self) -> None:
        from scripts.engram.run_token_module_ablation_5edit import _module_map

        modules = _module_map(self.model)
        for name, forward in reversed(list(self.original_forwards.items())):
            modules[name].forward = forward  # type: ignore[method-assign]
        self.original_forwards.clear()


def torch_linear(x: Any, weight: Any) -> Any:
    import torch

    return torch.nn.functional.linear(x, weight)


def _cosine(query: Any, proto: Any) -> float:
    import torch

    return float(torch.dot(query.float().cpu(), proto.float().cpu()).item())


def _normalize(value: Any) -> Any:
    import torch

    flat = value.detach().float().cpu().reshape(-1)
    norm = flat.norm()
    if not torch.isfinite(norm) or float(norm.item()) <= 0.0:
        return torch.zeros_like(flat)
    return flat / norm


def route_edit_ids(
    *,
    query: Any,
    entries: Sequence[Dict[str, Any]],
    query_record_id: str,
    query_kind: str,
    route_policy: str,
    prototype_type: str,
    threshold: Optional[float],
    max_active_edits: int,
) -> Dict[str, Any]:
    if not entries:
        return {
            "active_edit_ids": [],
            "active_record_ids": [],
            "active_edit_similarities": [],
            "active_edit_count": 0,
            "self_edit_active": False,
            "top1_edit_id": None,
            "top1_record_id": None,
            "top1_similarity": None,
            "second_similarity": None,
            "max_similarity": None,
            "top1_margin": None,
            "threshold_value": threshold,
        }
    if route_policy == "oracle_self":
        match = [
            entry
            for entry in entries
            if str(entry["record_id"]) == str(query_record_id) and query_kind in {"new", "old", "target"}
        ]
        active = match[:1]
        sim_map = {entry["edit_id"]: _cosine(query, entry[prototype_type]) for entry in entries}
    else:
        scored = sorted(
            [(entry, _cosine(query, entry[prototype_type])) for entry in entries],
            key=lambda item: item[1],
            reverse=True,
        )
        if route_policy == "top1_no_threshold":
            active = [scored[0][0]]
        elif route_policy == "top1_threshold":
            active = [scored[0][0]] if threshold is not None and scored[0][1] >= float(threshold) else []
        elif route_policy == "top3_threshold":
            active = [
                entry
                for entry, sim in scored[: int(max_active_edits)]
                if threshold is not None and sim >= float(threshold)
            ]
        elif route_policy == "threshold_all":
            active = [
                entry
                for entry, sim in scored
                if threshold is not None and sim >= float(threshold)
            ][: int(max_active_edits)]
        else:
            raise ValueError(f"unsupported route_policy: {route_policy}")
        sim_map = {entry["edit_id"]: sim for entry, sim in scored}
    scored_all = sorted(
        [(entry, sim_map[entry["edit_id"]]) for entry in entries],
        key=lambda item: item[1],
        reverse=True,
    )
    top1 = scored_all[0] if scored_all else (None, None)
    second = scored_all[1][1] if len(scored_all) > 1 else None
    active_sims = [sim_map[entry["edit_id"]] for entry in active]
    active_ids = [str(entry["edit_id"]) for entry in active]
    active_record_ids = [str(entry["record_id"]) for entry in active]
    top1_sim = top1[1]
    return {
        "active_edit_ids": active_ids,
        "active_record_ids": active_record_ids,
        "active_edit_similarities": active_sims,
        "active_edit_count": len(active),
        "self_edit_active": str(query_record_id) in active_record_ids,
        "top1_edit_id": str(top1[0]["edit_id"]) if top1[0] is not None else None,
        "top1_record_id": str(top1[0]["record_id"]) if top1[0] is not None else None,
        "top1_similarity": top1_sim,
        "second_similarity": second,
        "max_similarity": top1_sim,
        "top1_margin": None if top1_sim is None or second is None else float(top1_sim) - float(second),
        "threshold_value": threshold,
    }


def _sample_for_kind(record: Dict[str, Any], image_root: Path, kind: str) -> Optional[Dict[str, Any]]:
    from scripts.engram.run_localized_replacement_5edit import _new_sample, _old_sample
    from scripts.engram.run_token_module_ablation_5edit import _reference_sample

    if kind in {"new", "target"}:
        return _new_sample(record, image_root)
    if kind == "old":
        return _old_sample(record, image_root)
    if kind == "reference":
        ref = _reference_sample(record, image_root)
        return dict(ref) if ref is not None else None
    raise ValueError(f"unsupported query kind: {kind}")


def _extract_prototype(model: Any, sample: Dict[str, Any], module_names: Sequence[str]) -> Any:
    import torch
    from scripts.engram.run_token_module_ablation_5edit import _module_map

    modules = _module_map(model)
    vectors: List[Any] = []
    handles = []

    def make_hook(name: str):
        def hook(_module: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
            if not inputs:
                return
            value = inputs[0]
            if not isinstance(value, torch.Tensor) or value.shape[-1] <= 0:
                return
            vectors.append(value.detach().float().reshape(-1, value.shape[-1]).mean(dim=0).cpu())

        return hook

    for name in module_names:
        module = modules.get(name)
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            _ = model(dict(sample))
    finally:
        for handle in handles:
            handle.remove()
    if not vectors:
        raise RuntimeError("prototype extraction collected no activation vectors")
    return _normalize(torch.cat(vectors, dim=0))


def _compute_adaptive_threshold(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    entries: Sequence[Dict[str, Any]],
    prototype_type: str,
    module_names: Sequence[str],
    query_cache: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    import torch

    sims: List[float] = []
    for record in records:
        rid = _record_id(record)
        if query_cache is not None:
            query = (query_cache.get((rid, "reference")) or {}).get("query")
            if query is None:
                continue
        else:
            sample = _sample_for_kind(record, image_root, "reference")
            if sample is None:
                continue
            query = _extract_prototype(model, sample, module_names)
        for entry in entries:
            sims.append(_cosine(query, entry[prototype_type]))
    if not sims:
        return {"status": "fallback", "threshold": 1.0, "similarity_count": 0, "reason": "no reference samples"}
    values = torch.tensor(sims, dtype=torch.float32)
    threshold = float(torch.quantile(values, 0.90).item())
    return {
        "status": "complete",
        "threshold": threshold,
        "similarity_count": len(sims),
        "mean_reference_similarity": float(values.mean().item()),
        "max_reference_similarity": float(values.max().item()),
        "p90_reference_similarity": threshold,
    }


def _build_query_cache(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Any],
    module_names: Sequence[str],
    out_dir: Path,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    kinds = [("old", "old_raw"), ("new", "new_raw"), ("reference", "reference_raw")]
    total = len(records) * len(kinds)
    done = 0
    progress_path = out_dir / "audit" / "query_cache_progress.json"
    _json_dump(progress_path, {"status": "running", "done": 0, "total": total})
    for record in records:
        rid = _record_id(record)
        base = baselines[str(rid)]
        for kind, baseline_key in kinds:
            sample = _sample_for_kind(record, image_root, kind)
            query = _extract_prototype(model, sample, module_names) if sample is not None else None
            cache[(rid, kind)] = {
                "sample": sample,
                "query": query,
                "baseline_raw": base.get(baseline_key),
            }
            done += 1
            _json_dump(
                progress_path,
                {
                    "status": "running",
                    "done": done,
                    "total": total,
                    "latest_record_id": rid,
                    "latest_kind": kind,
                },
            )
    _json_dump(progress_path, {"status": "complete", "done": done, "total": total})
    return cache


def _existing_projector_bank_matches(bank_cls: Any, bank_dir: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not bank_dir.exists():
        return {"status": "missing", "bank_dir": str(bank_dir)}
    try:
        bank = bank_cls(bank_dir)
        edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    except Exception as exc:
        return {"status": "invalid", "bank_dir": str(bank_dir), "reason": f"{type(exc).__name__}: {exc}"}
    if matching.get("mode") != "record_id" or len(edit_ids) != len(records):
        return {"status": "invalid", "bank_dir": str(bank_dir), "matching": matching}
    return {"status": "complete", "reused": True, "bank_dir": str(bank_dir), "edit_ids": edit_ids, "edit_record_matching": matching}


def _build_edit_bank(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    hparams: Any,
    projector_bank_dir: Path,
    out_dir: Path,
    bank_config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import torch

    heavy = _heavy_imports()
    EngramBank = heavy["EngramBank"]
    _project_factors = heavy["_project_factors"]
    _train_tiny_lora = heavy["_train_tiny_lora"]

    bank_id = str(bank_config["bank_config_id"])
    tensor_dir = out_dir / "runtime_edit_banks" / bank_id
    meta_dir = out_dir / "bank_metadata"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    module_names = _module_scope_names(str(bank_config["module_scope"]))
    expected_paths = [(record, tensor_dir / f"{_record_id(record)}.pt") for record in records]
    if expected_paths and all(path.exists() for _record, path in expected_paths):
        entries: List[Dict[str, Any]] = []
        metadata_rows: List[Dict[str, Any]] = []
        for idx, (record, tensor_path) in enumerate(expected_paths):
            rid = _record_id(record)
            saved = torch.load(tensor_path, map_location="cpu")
            entry = saved["entry"]
            if str(entry.get("record_id")) != str(rid):
                raise RuntimeError(
                    {
                        "reason": "cached edit bank record_id mismatch",
                        "expected_record_id": rid,
                        "cached_record_id": entry.get("record_id"),
                        "tensor_path": str(tensor_path),
                    }
                )
            entry["metadata"]["source_index"] = idx
            entries.append(entry)
            metadata_rows.append(entry["metadata"])
        matching = {
            "mode": "record_id",
            "allow_positional_matching": False,
            "matched_record_count": len(entries),
            "record_ids": [_record_id(record) for record in records],
            "source": "cached_runtime_edit_bank",
        }
        _json_dump(meta_dir / f"{bank_id}.json", {"bank_config": bank_config, "entries": metadata_rows, "record_id_matching": matching})
        _write_csv(meta_dir / f"{bank_id}.csv", metadata_rows)
        return entries, {
            "status": "complete",
            "reused": True,
            "bank_config": bank_config,
            "entry_count": len(entries),
            "record_id_matching": matching,
        }
    projector_bank = EngramBank(projector_bank_dir)
    edit_ids, matching = projector_bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    if matching.get("mode") != "record_id":
        raise RuntimeError(f"projector bank did not match by record_id: {matching}")

    entries: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    scale = float(hparams.lora_scale if getattr(hparams, "lora_scale", None) is not None else 1.0)
    for idx, (record, edit_id) in enumerate(zip(records, edit_ids)):
        rid = _record_id(record)
        tensor_path = tensor_dir / f"{rid}.pt"
        if tensor_path.exists():
            saved = torch.load(tensor_path, map_location="cpu")
            entry = saved["entry"]
            entries.append(entry)
            metadata_rows.append(entry["metadata"])
            continue
        factors, train_summary = _train_tiny_lora(
            model,
            record,
            image_root,
            module_names,
            rank=int(hparams.lora_rank),
            steps=int(bank_config["lora_steps"]),
            lr=float(hparams.lora_lr),
            scale=scale,
            lambda_ref=0.0,
        )
        safe_factors, projection_summary = _project_factors(factors, projector_bank.load_edit(str(edit_id)))
        target_sample = _sample_for_kind(record, image_root, "new")
        reference_sample = _sample_for_kind(record, image_root, "reference")
        assert target_sample is not None
        target_proto = _extract_prototype(model, target_sample, module_names)
        if reference_sample is not None:
            reference_proto = _extract_prototype(model, reference_sample, module_names)
            contrastive_proto = _normalize(target_proto - reference_proto)
        else:
            reference_proto = None
            contrastive_proto = target_proto
        delta_norms = {
            name: {
                "A": _tensor_summary(factor["A"]),
                "B": _tensor_summary(factor["B"]),
                "scale": float(factor.get("scale", 1.0)),
            }
            for name, factor in safe_factors.items()
        }
        metadata = {
            "record_id": rid,
            "edit_index": idx,
            "source_file": "outputs/medmkeb_engram_projected_lora/modelknown_20/medmkeb_modelknown_20.json",
            "source_index": idx,
            "old_answer": str(record.get("old_answer") or record.get("pred") or ""),
            "new_answer": str(record.get("new_answer") or record.get("target_new") or ""),
            "prompt": str(record.get("src") or record.get("prompt") or ""),
            "image_path": str(record.get("image") or record.get("image_path") or ""),
            "selected_modules": module_names,
            "beta": float(bank_config["beta"]),
            "lora_steps": int(bank_config["lora_steps"]),
            "module_scope": bank_config["module_scope"],
            "delta_norm_per_module": delta_norms,
            "engram_projection_norm_per_module": projection_summary.get("modules"),
            "target_activation_prototype_path_or_summary": {"norm": float(target_proto.norm().item()), "shape": list(target_proto.shape)},
            "reference_activation_prototype_path_or_summary": None
            if reference_proto is None
            else {"norm": float(reference_proto.norm().item()), "shape": list(reference_proto.shape)},
            "prototype_norm": float(target_proto.norm().item()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "lora_train": train_summary,
            "projection": projection_summary,
        }
        entry = {
            "edit_id": f"{bank_id}__{rid}",
            "record_id": rid,
            "beta": float(bank_config["beta"]),
            "factors": safe_factors,
            "target": target_proto,
            "reference": reference_proto,
            "contrastive": contrastive_proto,
            "metadata": metadata,
        }
        torch.save({"entry": entry}, tensor_path)
        entries.append(entry)
        metadata_rows.append(metadata)
    _json_dump(meta_dir / f"{bank_id}.json", {"bank_config": bank_config, "entries": metadata_rows, "record_id_matching": matching})
    _write_csv(meta_dir / f"{bank_id}.csv", metadata_rows)
    return entries, {"status": "complete", "bank_config": bank_config, "entry_count": len(entries), "record_id_matching": matching}


def _evaluate_one_sample_with_route(
    *,
    model: Any,
    record: Dict[str, Any],
    kind: str,
    sample: Optional[Dict[str, Any]],
    query: Optional[Any],
    baseline_raw: Optional[Dict[str, Any]],
    entries: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    threshold: Optional[float],
    module_names: Sequence[str],
    rollback_tolerance: float,
    eval_cache: Dict[Tuple[str, str, Tuple[str, ...]], Tuple[Optional[Dict[str, Any]], bool, float]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    import torch
    from scripts.engram.run_token_module_ablation_5edit import _answer_metrics, _max_snapshot_diff, _snapshot_modules

    if sample is None or query is None:
        return None, {
            "query_kind": kind,
            "active_edit_ids": [],
            "active_record_ids": [],
            "active_edit_similarities": [],
            "active_edit_count": 0,
            "self_edit_active": False,
            "top1_edit_id": None,
            "top1_record_id": None,
            "top1_similarity": None,
            "second_similarity": None,
            "max_similarity": None,
            "top1_margin": None,
            "threshold_value": threshold,
            "temporary_rollback_pass": True,
            "temporary_rollback_max_abs_diff": 0.0,
        }
    routing = route_edit_ids(
        query=query,
        entries=entries,
        query_record_id=_record_id(record),
        query_kind=kind,
        route_policy=str(config["route_policy"]),
        prototype_type=str(config["prototype_type"]),
        threshold=threshold,
        max_active_edits=int(config.get("max_active_edits") or 1),
    )
    active_by_id = {entry["edit_id"]: entry for entry in entries}
    active_entries = [active_by_id[edit_id] for edit_id in routing["active_edit_ids"]]
    weights = [1.0 for _ in active_entries]
    cache_key = (str(config["config_id"]), f"{_record_id(record)}::{kind}", tuple(routing["active_edit_ids"]))
    if not active_entries:
        raw = baseline_raw
        rollback_diff = 0.0
        rollback_pass = True
    elif cache_key in eval_cache:
        raw, rollback_pass, rollback_diff = eval_cache[cache_key]
    else:
        snapshots = _snapshot_modules(model, list(module_names))
        patch = RoutedLoraPatch(model, active_entries, weights)
        patch.install()
        try:
            raw = _answer_metrics(model, dict(sample))
        finally:
            patch.remove()
        rollback_diff = _max_snapshot_diff(model, snapshots)
        rollback_pass = rollback_diff <= float(rollback_tolerance)
        eval_cache[cache_key] = (raw, rollback_pass, rollback_diff)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    routing.update(
        {
            "query_kind": kind,
            "temporary_rollback_max_abs_diff": rollback_diff,
            "temporary_rollback_pass": rollback_pass,
        }
    )
    return raw, routing


def _raw_nll(raw: Optional[Dict[str, Any]]) -> Optional[float]:
    if not raw or not raw.get("available") or raw.get("nll") is None:
        return None
    return float(raw["nll"])


def _delta(after: Optional[float], before: Optional[float]) -> Optional[float]:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def _evaluate_routed_config(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Any],
    all_entries: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    run_dir: Path,
    module_names: Sequence[str],
    query_cache: Dict[Tuple[str, str], Dict[str, Any]],
    rollback_tolerance: float,
    locality_threshold: float,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    threshold_info = {"status": "not_required", "threshold": config.get("threshold")}
    if config.get("threshold") == "adaptive_p90_reference":
        threshold_info = _compute_adaptive_threshold(
            model=model,
            records=records,
            image_root=image_root,
            entries=all_entries,
            prototype_type=str(config["prototype_type"]),
            module_names=module_names,
            query_cache=query_cache,
        )
        threshold = float(threshold_info["threshold"])
    else:
        threshold = None if config.get("threshold") is None else float(config["threshold"])
    rows: List[Dict[str, Any]] = []
    routing_rows: List[Dict[str, Any]] = []
    record_ids = [_record_id(record) for record in records]
    record_id_match_rate = 1.0 if len(record_ids) == len(set(record_ids)) == len(records) else 0.0
    _json_dump(
        run_dir / "progress.json",
        {
            "status": "query_cache_ready",
            "config_id": config["config_id"],
            "cached_queries": len(query_cache),
            "steps_total": len(records),
            "step_completed": -1,
        },
    )
    eval_cache: Dict[Tuple[str, str, Tuple[str, ...]], Tuple[Optional[Dict[str, Any]], bool, float]] = {}
    for step in range(0, len(records) + 1):
        entries = list(all_entries[:step])
        applied_ids = [str(entry["record_id"]) for entry in entries]
        for idx, record in enumerate(records):
            rid = _record_id(record)
            old_cached = query_cache[(rid, "old")]
            new_cached = query_cache[(rid, "new")]
            ref_cached = query_cache[(rid, "reference")]
            old_raw, old_route = _evaluate_one_sample_with_route(
                model=model,
                record=record,
                kind="old",
                sample=old_cached["sample"],
                query=old_cached["query"],
                baseline_raw=old_cached["baseline_raw"],
                entries=entries,
                config=config,
                threshold=threshold,
                module_names=module_names,
                rollback_tolerance=rollback_tolerance,
                eval_cache=eval_cache,
            )
            new_raw, new_route = _evaluate_one_sample_with_route(
                model=model,
                record=record,
                kind="new",
                sample=new_cached["sample"],
                query=new_cached["query"],
                baseline_raw=new_cached["baseline_raw"],
                entries=entries,
                config=config,
                threshold=threshold,
                module_names=module_names,
                rollback_tolerance=rollback_tolerance,
                eval_cache=eval_cache,
            )
            ref_raw, ref_route = _evaluate_one_sample_with_route(
                model=model,
                record=record,
                kind="reference",
                sample=ref_cached["sample"],
                query=ref_cached["query"],
                baseline_raw=ref_cached["baseline_raw"],
                entries=entries,
                config=config,
                threshold=threshold,
                module_names=module_names,
                rollback_tolerance=rollback_tolerance,
                eval_cache=eval_cache,
            )
            base = baselines[str(rid)]
            old_delta = _delta(_raw_nll(old_raw), _raw_nll(base.get("old_raw")))
            new_delta = _delta(_raw_nll(new_raw), _raw_nll(base.get("new_raw")))
            ref_delta = _delta(_raw_nll(ref_raw), _raw_nll(base.get("reference_raw")))
            new_decrease = None if new_delta is None else -new_delta
            ref_abs = None if ref_delta is None else abs(ref_delta)
            is_edited = idx < step
            is_current = idx == step - 1
            is_previous = idx < max(step - 1, 0)
            is_future = idx >= step
            rollback_pass = bool(old_route["temporary_rollback_pass"] and new_route["temporary_rollback_pass"] and ref_route["temporary_rollback_pass"])
            rollback_max = max(
                float(old_route["temporary_rollback_max_abs_diff"] or 0.0),
                float(new_route["temporary_rollback_max_abs_diff"] or 0.0),
                float(ref_route["temporary_rollback_max_abs_diff"] or 0.0),
            )
            row = {
                "config_id": config["config_id"],
                "bank_config_id": config["bank_config_id"],
                "route_policy": config["route_policy"],
                "prototype_type": config["prototype_type"],
                "threshold": config.get("threshold"),
                "threshold_value": threshold,
                "step": step,
                "applied_record_ids": applied_ids,
                "record_id": rid,
                "is_edited_so_far": is_edited,
                "is_current_edit": is_current,
                "is_previous_edit": is_previous,
                "is_future_edit": is_future,
                "active_edit_ids": new_route["active_edit_ids"],
                "active_edit_count": new_route["active_edit_count"],
                "active_edit_similarities": new_route["active_edit_similarities"],
                "self_edit_active": new_route["self_edit_active"],
                "top1_edit_id": new_route["top1_edit_id"],
                "top1_record_id": new_route["top1_record_id"],
                "top1_similarity": new_route["top1_similarity"],
                "max_similarity": new_route["max_similarity"],
                "top1_margin": new_route["top1_margin"],
                "reference_active_edit_ids": ref_route["active_edit_ids"],
                "reference_active_edit_count": ref_route["active_edit_count"],
                "reference_self_edit_active": ref_route["self_edit_active"],
                "old_answer_nll": _raw_nll(old_raw),
                "old_answer_nll_delta_vs_step0": old_delta,
                "new_answer_nll": _raw_nll(new_raw),
                "new_answer_nll_decrease_vs_step0": new_decrease,
                "locality_reference_nll": _raw_nll(ref_raw),
                "locality_reference_delta_abs_vs_step0": ref_abs,
                "previous_edit_retention": new_decrease if is_previous else None,
                "previous_edit_forgetting": None if not is_previous or new_decrease is None else max(0.0, -float(new_decrease)),
                "retention_ratio": _safe_div(new_decrease, C_HIGH_NEW) if is_previous else None,
                "future_record_drift": abs(float(new_decrease or 0.0)) if is_future else None,
                "locality_damage": bool(ref_abs is not None and ref_abs > float(locality_threshold)),
                "temporary_rollback_pass": rollback_pass,
                "temporary_rollback_max_abs_diff": rollback_max,
                "record_id_match_rate": record_id_match_rate,
                "nan_inf_detected": not _finite({"old": old_raw, "new": new_raw, "ref": ref_raw, "rollback": rollback_max}),
            }
            rows.append(row)
            for kind, route in [("old", old_route), ("new", new_route), ("reference", ref_route)]:
                routing_rows.append(
                    {
                        "config_id": config["config_id"],
                        "step": step,
                        "record_id": rid,
                        "query_kind": kind,
                        "is_edited_so_far": is_edited,
                        "is_future_edit": is_future,
                        **route,
                }
            )
        step_summary = _aggregate_routed_step(rows, config, step, len(records))
        _json_dump(
            run_dir / "progress.json",
            {
                "status": "running",
                "config_id": config["config_id"],
                "steps_total": len(records),
                "step_completed": step,
                "matrix_rows": len(rows),
                "routing_rows": len(routing_rows),
                "eval_cache_entries": len(eval_cache),
                "latest_step_summary": step_summary,
            },
        )
        _write_csv(run_dir / "routed_step_matrix.partial.csv", rows)
        _write_csv(run_dir / "routing_trace.partial.csv", routing_rows)
    summary_rows = [_aggregate_routed_step(rows, config, step, len(records)) for step in range(0, len(records) + 1)]
    routing_metrics = _routing_metrics(routing_rows, rows, config, len(records))
    final = [row for row in summary_rows if row.get("final_step")]
    rollback = {
        "status": "pass" if _mean([1.0 if row.get("temporary_rollback_pass") else 0.0 for row in rows]) == 1.0 else "fail",
        "temporary_rollback_max_abs_diff": max(float(row.get("temporary_rollback_max_abs_diff") or 0.0) for row in rows) if rows else 0.0,
        "rollback_tolerance": rollback_tolerance,
    }
    payload = {
        "status": "complete",
        "config": config,
        "threshold_info": threshold_info,
        "per_record_step_rows": rows,
        "summary_rows": summary_rows,
        "final_summary": final[0] if final else {},
        "routing_metrics": routing_metrics,
        "rollback_check": rollback,
    }
    _json_dump(run_dir / "config.json", config)
    _json_dump(run_dir / "routed_step_matrix.json", rows)
    _write_csv(run_dir / "routed_step_matrix.csv", rows)
    _json_dump(run_dir / "routed_summary.json", payload)
    _write_csv(run_dir / "routed_summary.csv", summary_rows)
    _write_csv(run_dir / "routing_trace.csv", routing_rows)
    _json_dump(run_dir / "routing_metrics.json", routing_metrics)
    _json_dump(run_dir / "rollback_check.json", rollback)
    return payload


def _aggregate_routed_step(rows: List[Dict[str, Any]], config: Dict[str, Any], step: int, total: int) -> Dict[str, Any]:
    step_rows = [row for row in rows if int(row.get("step") or -1) == int(step)]
    edited = [row for row in step_rows if row.get("is_edited_so_far")]
    previous = [row for row in step_rows if row.get("is_previous_edit")]
    future = [row for row in step_rows if row.get("is_future_edit")]
    return {
        "config_id": config["config_id"],
        "bank_config_id": config["bank_config_id"],
        "route_policy": config["route_policy"],
        "prototype_type": config["prototype_type"],
        "threshold": config.get("threshold"),
        "step": step,
        "record_count": len(step_rows),
        "edited_record_count": len(edited),
        "mean_new_answer_nll_decrease": _mean([row.get("new_answer_nll_decrease_vs_step0") for row in edited]),
        "mean_new_answer_nll_decrease_all_records": _mean([row.get("new_answer_nll_decrease_vs_step0") for row in step_rows]),
        "mean_ref_abs": _mean([row.get("locality_reference_delta_abs_vs_step0") for row in step_rows]),
        "positive_new_answer_edits": sum(1 for row in edited if (row.get("new_answer_nll_decrease_vs_step0") or 0.0) > 0.0),
        "locality_damage_records": sum(1 for row in step_rows if row.get("locality_damage")),
        "previous_edit_retention": _mean([row.get("previous_edit_retention") for row in previous]),
        "mean_previous_edit_forgetting": _mean([row.get("previous_edit_forgetting") for row in previous]),
        "retention_ratio": _mean([row.get("retention_ratio") for row in previous]),
        "future_record_drift": _mean([row.get("future_record_drift") for row in future]),
        "temporary_rollback_pass_rate": _mean([1.0 if row.get("temporary_rollback_pass") else 0.0 for row in step_rows]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in step_rows]),
        "nan_inf_count": sum(1 for row in step_rows if row.get("nan_inf_detected")),
        "final_step": step == total,
    }


def _routing_metrics(routing_rows: List[Dict[str, Any]], matrix_rows: List[Dict[str, Any]], config: Dict[str, Any], total: int) -> Dict[str, Any]:
    final_targets = [
        row for row in routing_rows
        if int(row.get("step") or -1) == total and row.get("query_kind") == "new" and row.get("is_edited_so_far")
    ]
    final_refs = [
        row for row in routing_rows
        if int(row.get("step") or -1) == total and row.get("query_kind") == "reference"
    ]
    future_targets = [
        row for row in routing_rows
        if row.get("query_kind") == "new" and row.get("is_future_edit")
    ]
    return {
        "config_id": config["config_id"],
        "self_hit_rate": _mean([1.0 if row.get("self_edit_active") else 0.0 for row in final_targets]),
        "top1_self_rate": _mean([1.0 if str(row.get("top1_record_id")) == str(row.get("record_id")) else 0.0 for row in final_targets]),
        "mean_self_similarity": _mean([row.get("top1_similarity") for row in final_targets if str(row.get("top1_record_id")) == str(row.get("record_id"))]),
        "mean_top1_margin": _mean([row.get("top1_margin") for row in final_targets]),
        "reference_activation_rate": _mean([1.0 if (row.get("active_edit_count") or 0) > 0 else 0.0 for row in final_refs]),
        "reference_false_activation_rate": _mean([1.0 if (row.get("active_edit_count") or 0) > 0 else 0.0 for row in final_refs]),
        "mean_reference_active_count": _mean([row.get("active_edit_count") for row in final_refs]),
        "mean_reference_max_similarity": _mean([row.get("max_similarity") for row in final_refs]),
        "future_activation_rate": _mean([1.0 if (row.get("active_edit_count") or 0) > 0 else 0.0 for row in future_targets]),
        "final_target_count": len(final_targets),
        "final_reference_count": len(final_refs),
        "matrix_row_count": len(matrix_rows),
    }


def _score_routed(payload: Dict[str, Any]) -> Dict[str, Any]:
    final = payload.get("final_summary", {})
    metrics = payload.get("routing_metrics", {})
    rollback = payload.get("rollback_check", {})
    config = payload.get("config", {})
    new = final.get("mean_new_answer_nll_decrease")
    ref = final.get("mean_ref_abs")
    positive = int(final.get("positive_new_answer_edits") or 0)
    damage = int(final.get("locality_damage_records") or 0)
    rollback_rate = final.get("temporary_rollback_pass_rate")
    match = final.get("record_id_match_rate")
    nan = int(final.get("nan_inf_count") or 0)
    self_hit = metrics.get("self_hit_rate")
    ref_false = metrics.get("reference_false_activation_rate")
    basic = (
        positive >= 18
        and new is not None and float(new) > 0.0
        and ref is not None and float(ref) <= 0.10
        and damage <= 8
        and rollback_rate == 1.0
        and match == 1.0
        and nan == 0
    )
    strong = (
        positive >= 18
        and new is not None and float(new) >= 0.60
        and ref is not None and float(ref) <= 0.08
        and damage <= 8
        and self_hit is not None and float(self_hit) >= 0.70
        and ref_false is not None and float(ref_false) <= 0.30
        and rollback_rate == 1.0
        and match == 1.0
        and nan == 0
    )
    breakthrough = (
        positive >= 18
        and new is not None and float(new) >= 0.80
        and ref is not None and float(ref) <= 0.08
        and damage <= 5
        and self_hit is not None and float(self_hit) >= 0.80
        and ref_false is not None and float(ref_false) <= 0.20
        and (final.get("mean_previous_edit_forgetting") is None or float(final.get("mean_previous_edit_forgetting") or 0.0) <= C_HIGH_NEW)
        and rollback_rate == 1.0
        and match == 1.0
        and nan == 0
    )
    status = "breakthrough" if breakthrough else ("strong_pass" if strong else ("basic_pass" if basic else "fail"))
    return {
        "config_id": config.get("config_id"),
        "bank_config_id": config.get("bank_config_id"),
        "route_policy": config.get("route_policy"),
        "prototype_type": config.get("prototype_type"),
        "threshold": config.get("threshold"),
        "final_new": new,
        "final_ref": ref,
        "positive_new": positive,
        "locality_damage": damage,
        "previous_edit_retention": final.get("previous_edit_retention"),
        "mean_previous_edit_forgetting": final.get("mean_previous_edit_forgetting"),
        "retention_ratio": final.get("retention_ratio"),
        "future_record_drift": final.get("future_record_drift"),
        "self_hit_rate": self_hit,
        "top1_self_rate": metrics.get("top1_self_rate"),
        "reference_false_activation_rate": ref_false,
        "future_activation_rate": metrics.get("future_activation_rate"),
        "mean_reference_active_count": metrics.get("mean_reference_active_count"),
        "temporary_rollback_pass_rate": rollback_rate,
        "rollback_status": rollback.get("status"),
        "record_id_match_rate": match,
        "nan_inf_count": nan,
        "basic_pass": basic,
        "strong_pass": strong,
        "breakthrough": breakthrough,
        "status": status,
    }


def _choose_best(scores: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    routed = [row for row in scores if str(row.get("config_id", "")).startswith("R_")]
    if not routed:
        return None
    return sorted(
        routed,
        key=lambda row: (
            1 if row.get("breakthrough") else 0,
            1 if row.get("strong_pass") else 0,
            1 if row.get("basic_pass") else 0,
            float(row.get("final_new") or -999.0),
            -float(row.get("final_ref") or 999.0),
            -int(row.get("locality_damage") or 999),
        ),
        reverse=True,
    )[0]


def _plot_optional_routed(out_dir: Path, scores: List[Dict[str, Any]], payloads: Dict[str, Dict[str, Any]], best: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    made: List[str] = []
    try:
        routed = [row for row in scores if str(row.get("config_id", "")).startswith("R_") and row.get("final_new") is not None and row.get("final_ref") is not None]
        if routed:
            plt.figure(figsize=(7, 5))
            plt.scatter([float(row["final_ref"]) for row in routed], [float(row["final_new"]) for row in routed])
            for row in routed:
                plt.annotate(str(row["config_id"]).replace("_", "\n"), (float(row["final_ref"]), float(row["final_new"])), fontsize=5)
            plt.xlabel("final reference delta abs")
            plt.ylabel("final new-answer NLL decrease")
            plt.tight_layout()
            path = out_dir / "plots" / "ref_vs_new_scatter.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))
            plt.figure(figsize=(7, 5))
            plt.scatter([float(row.get("self_hit_rate") or 0.0) for row in routed], [float(row["final_new"]) for row in routed])
            plt.xlabel("self hit rate")
            plt.ylabel("final new-answer NLL decrease")
            plt.tight_layout()
            path = out_dir / "plots" / "routing_self_hit_vs_new.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))
            plt.figure(figsize=(7, 5))
            plt.scatter([float(row.get("reference_false_activation_rate") or 0.0) for row in routed], [float(row["final_ref"]) for row in routed])
            plt.xlabel("reference false activation rate")
            plt.ylabel("final reference delta abs")
            plt.tight_layout()
            path = out_dir / "plots" / "reference_false_activation_vs_ref.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))
        if best:
            payload = payloads.get(str(best["config_id"]))
            if payload:
                xs = [int(row.get("step") or 0) for row in payload.get("summary_rows", [])]
                for key, filename, ylabel in [
                    ("mean_ref_abs", "sequential_reference_curve_best.png", "reference delta abs"),
                    ("mean_new_answer_nll_decrease", "sequential_new_curve_best.png", "new-answer NLL decrease"),
                    ("mean_previous_edit_forgetting", "sequential_forgetting_curve_best.png", "previous-edit forgetting"),
                ]:
                    plt.figure(figsize=(7, 4))
                    plt.plot(xs, [float(row.get(key) or 0.0) for row in payload.get("summary_rows", [])], marker="o")
                    plt.xlabel("step")
                    plt.ylabel(ylabel)
                    plt.tight_layout()
                    path = out_dir / "plots" / filename
                    plt.savefig(path)
                    plt.close()
                    made.append(str(path))
        return {"status": "complete", "files": made}
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "files": made}


def _write_best_analysis(out_dir: Path, best: Optional[Dict[str, Any]], payloads: Dict[str, Dict[str, Any]]) -> None:
    if not best:
        (out_dir / "BEST_ROUTED_CONFIG_ANALYSIS.md").write_text("# Best Routed Config Analysis\n\nNo routed config completed.\n", encoding="utf-8")
        _write_csv(out_dir / "best_routed_per_record.csv", [])
        return
    payload = payloads[str(best["config_id"])]
    rows = [row for row in payload.get("per_record_step_rows", []) if int(row.get("step") or -1) == 20]
    failures = [row for row in rows if row.get("is_edited_so_far") and not row.get("self_edit_active")]
    false_refs = [row for row in rows if (row.get("reference_active_edit_count") or 0) > 0]
    _write_csv(out_dir / "best_routed_per_record.csv", rows)
    lines = [
        "# Best Routed Config Analysis",
        "",
        f"- Best config: `{best['config_id']}`",
        f"- Status: `{best['status']}`",
        f"- final_new: `{_format(best.get('final_new'))}`",
        f"- final_ref: `{_format(best.get('final_ref'))}`",
        f"- locality_damage: `{best.get('locality_damage')}/20`",
        f"- self_hit_rate: `{_format(best.get('self_hit_rate'))}`",
        f"- reference_false_activation_rate: `{_format(best.get('reference_false_activation_rate'))}`",
        "",
        "## Questions",
        "",
        "- Oracle routing breaks the strength-locality coupling only if `R_oracle_self_high` reaches strong or breakthrough in `routed_bank_summary.csv`.",
        "- Learned routing approaches oracle when self-hit and final_new are close while reference false activation stays low.",
        "- Contrastive prototype and adaptive threshold effects are visible in `routing_metrics.csv`.",
        "",
        "## Self-Hit Failures",
        "",
    ]
    if failures:
        for row in failures[:20]:
            lines.append(f"- `{row['record_id']}` top1={row.get('top1_record_id')} sim={_format(row.get('top1_similarity'))}")
    else:
        lines.append("- None at final step.")
    lines.extend(["", "## Reference False Activations", ""])
    if false_refs:
        for row in false_refs[:20]:
            lines.append(f"- `{row['record_id']}` active={row.get('reference_active_edit_ids')}")
    else:
        lines.append("- None at final step.")
    (out_dir / "BEST_ROUTED_CONFIG_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_report(
    *,
    out_dir: Path,
    scores: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    data_reuse: Dict[str, Any],
    generation: Dict[str, Any],
    plots: Dict[str, Any],
) -> None:
    routed = [row for row in scores if str(row.get("config_id", "")).startswith("R_")]
    oracle = next((row for row in routed if row.get("config_id") == "R_oracle_self_high"), None)
    strong = [row for row in routed if row.get("strong_pass")]
    breakthrough = [row for row in routed if row.get("breakthrough")]
    basic = [row for row in routed if row.get("basic_pass")]
    if breakthrough or strong:
        decision = "A. Routed bank reaches strong or breakthrough. Next: validate on 50 MedMKEB model-known edits or external Med-VQA."
    elif oracle and oracle.get("strong_pass") and not strong:
        decision = "B. Oracle routed works but learned routing does not. Next: improve routing/prototype learning."
    elif oracle and not oracle.get("basic_pass"):
        decision = "C. Oracle routed also fails. Next: improve per-edit delta/projection rather than routing."
    elif basic:
        decision = "D. Routed bank only basic/partial. Keep as promising but not solved; analyze failure mode."
    else:
        decision = "C. Oracle routed also fails. Next: improve per-edit delta/projection rather than routing."
    lines = [
        "# Final MedMKEB Routed Bank 20 Report",
        "",
        "## Starting Point",
        "",
        "- MedMKEB nonseq ENGRAM-projected tiny LoRA succeeds.",
        "- MedMKEB sequential global merge suffers strength-locality coupling.",
        "- Lower beta/fewer steps is only a partial rescue.",
        "- Crisp/CURE training-time pilot reached only a basic low-drift result, not strong or breakthrough.",
        "- This pilot tests routed edit-bank selective composition with temporary apply/rollback.",
        "",
        "## Data",
        "",
        f"- Exact reused records: `{data_reuse.get('selected_record_count')}`",
        f"- Record-id match rate: `{data_reuse.get('record_id_match_rate')}`",
        "- Positional matching used: `False`",
        "- Private or patient data used: `False`",
        "- No medical or clinical efficacy claim is made.",
        "",
        "## Method",
        "",
        "- Per-edit ENGRAM-projected LoRA factors are stored in a remote-only runtime bank.",
        "- Target/reference activation prototypes are extracted over q/k/gate sampled depths 0, 8, 16, and 24.",
        "- Each evaluation query routes over the currently available bank entries.",
        "- Selected deltas are applied through a temporary forward patch and removed immediately.",
        "- No global merge is performed in routed configs.",
        "",
        "## Main Results",
        "",
        "| config_id | status | final_new | final_ref | positive_new | damage | self_hit | ref_false | rollback | match | nan |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scores:
        lines.append(
            "| {config_id} | {status} | {new} | {ref} | {positive} | {damage} | {self_hit} | {ref_false} | {rollback} | {match} | {nan} |".format(
                config_id=row.get("config_id"),
                status=row.get("status"),
                new=_format(row.get("final_new")),
                ref=_format(row.get("final_ref")),
                positive=row.get("positive_new"),
                damage=row.get("locality_damage"),
                self_hit=_format(row.get("self_hit_rate")),
                ref_false=_format(row.get("reference_false_activation_rate")),
                rollback=_format(row.get("temporary_rollback_pass_rate")),
                match=_format(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Oracle Analysis",
            "",
            "- If `R_oracle_self_high` is strong or breakthrough, routing quality is the bottleneck.",
            "- If `R_oracle_self_high` is weak, per-edit deltas or projection are insufficient even with perfect routing.",
            "",
            "## Best Config",
            "",
        ]
    )
    if best:
        lines.extend(
            [
                f"- Best config: `{best['config_id']}`",
                f"- Status: `{best['status']}`",
                f"- final_new: `{_format(best.get('final_new'))}`",
                f"- final_ref: `{_format(best.get('final_ref'))}`",
                f"- locality_damage: `{best.get('locality_damage')}/20`",
                f"- self_hit_rate: `{_format(best.get('self_hit_rate'))}`",
                f"- reference_false_activation_rate: `{_format(best.get('reference_false_activation_rate'))}`",
            ]
        )
    else:
        lines.append("- No routed config completed.")
    lines.extend(
        [
            "",
            "## Generation Diagnostics",
            "",
            f"- Status: `{generation.get('status')}`",
            "- Generation diagnostics are secondary and not used as the pass/fail criterion.",
            "",
            "## Plots",
            "",
            f"- Status: `{plots.get('status')}`",
            "",
            "## Decision",
            "",
            decision,
            "",
            "## Limitations",
            "",
            "- Bounded 20-edit MedMKEB model-known subset.",
            "- NLL/logprob primary evidence.",
            "- Generation diagnostic only.",
            "- No medical or clinical efficacy claim.",
            "- Routed bank adds inference-time routing cost.",
        ]
    )
    (out_dir / "FINAL_MEDMKEB_ROUTED_BANK_20_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_package_hygiene_report(out_dir: Path) -> Dict[str, Any]:
    if (out_dir / "runtime_edit_banks").exists():
        shutil.rmtree(out_dir / "runtime_edit_banks")
    if (out_dir / "runtime_projector_banks").exists():
        shutil.rmtree(out_dir / "runtime_projector_banks")
    hygiene = _package_hygiene(out_dir, remove_runtime_bank=True)
    found: List[str] = []
    for path in out_dir.rglob("*"):
        name = path.name
        if path.is_dir() and name == "__pycache__":
            found.append(str(path))
        elif name == ".DS_Store" or name.startswith("._") or path.suffix in {".pt", ".pth", ".bin", ".pyc"}:
            found.append(str(path))
    payload = {**hygiene, "forbidden_artifacts_found": found, "forbidden_artifact_count": len(found)}
    lines = [
        "# Package Hygiene Report",
        "",
        f"- Forbidden artifacts found: `{len(found)}`",
        "- Checked exclusions: `.pt`, `.pth`, `.bin`, projector bank tensors, edit delta tensors, model weights, Hugging Face cache, CUDA cache, `__pycache__`, `.pyc`, `.DS_Store`, and `._*` AppleDouble files.",
        "",
    ]
    if found:
        lines.extend(["## Found", ""])
        lines.extend(f"- `{item}`" for item in found[:100])
    else:
        lines.append("No forbidden artifacts were found under this routed-bank output directory.")
    (out_dir / "PACKAGE_HYGIENE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json_dump(out_dir / "PACKAGE_HYGIENE_REPORT.md.json", payload)
    return payload


def _prepare(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    records = _load_records(Path(args.selected_records))[: int(args.max_records)]
    data_reuse = _write_data_reuse_report(
        out_dir=out_dir,
        selected_records_path=Path(args.selected_records),
        previous_record_preflight=Path(args.previous_record_preflight),
        records=records,
    )
    _json_dump(out_dir / "routed_bank_config_grid.json", {"bank_configs": _bank_configs(), "routed_configs": _routed_configs(), "anchors": _anchor_rows()})
    _write_csv(out_dir / "routed_bank_summary.csv", _anchor_rows())
    _json_dump(out_dir / "routed_bank_summary.json", {"scores": _anchor_rows(), "runtime_status": "prepare_only"})
    _write_final_report(out_dir=out_dir, scores=_anchor_rows(), best=None, data_reuse=data_reuse, generation={"status": "skipped"}, plots={"status": "skipped"})
    _write_package_hygiene_report(out_dir)
    return 0


def _generation_text_metrics(method: str, record: Dict[str, Any], idx: int, generation: Dict[str, Any], route: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    decoded = str(generation.get("decoded_stripped") or generation.get("decoded_skip_special") or "")
    old = str(record.get("old_answer") or record.get("pred") or "")
    new = str(record.get("new_answer") or record.get("target_new") or "")
    route = route or {}
    return {
        "method": method,
        "record_id": _record_id(record),
        "case_index": idx,
        "prompt": record.get("src") or record.get("prompt"),
        "old_answer": old,
        "new_answer": new,
        "generation": decoded,
        "generation_empty": bool(generation.get("generation_empty")),
        "contains_old_answer": old.casefold() in decoded.casefold() if old else False,
        "contains_new_answer": new.casefold() in decoded.casefold() if new else False,
        "exact_new_answer": decoded.strip().casefold() == new.strip().casefold() if new else False,
        "simple_casefold_contains": new.casefold() in decoded.casefold() if new else False,
        "active_edit_ids": route.get("active_edit_ids", []),
        "self_edit_active": route.get("self_edit_active", False),
        "notes": "generation diagnostic only; not primary gate",
    }


def _run_generation(args: argparse.Namespace) -> int:
    import torch

    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generation_diagnostics"
    gen_dir.mkdir(parents=True, exist_ok=True)
    _write_env_report(out_dir)
    all_records = _load_records(Path(args.selected_records))[: int(args.max_records)]
    records = all_records[: int(args.generation_records)]
    if not records:
        payload = {"status": "skipped", "reason": "no records"}
        _json_dump(gen_dir / "generation_5records.json", payload)
        return 2
    summary_path = out_dir / "routed_bank_summary.json"
    if not summary_path.exists():
        payload = {"status": "skipped", "reason": "missing routed_bank_summary.json"}
        _json_dump(gen_dir / "generation_5records.json", payload)
        return 2
    scores = _read_json(summary_path)["scores"]
    best = _choose_best(scores)
    if not best or not best.get("basic_pass"):
        payload = {"status": "skipped", "reason": "no routed config reached basic pass"}
        _json_dump(gen_dir / "generation_5records.json", payload)
        return 0
    config = next(item for item in _routed_configs() if item["config_id"] == best["config_id"])

    heavy = _heavy_imports()
    MultimodalEditor = heavy["MultimodalEditor"]
    EngramMultimodalHparams = heavy["EngramMultimodalHparams"]
    EngramBank = heavy["EngramBank"]
    select_linear_layers = heavy["select_linear_layers"]
    _extract_projector_bank = heavy["_extract_projector_bank"]
    _evaluate_current = heavy["_evaluate_current"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hparams = EngramMultimodalHparams.from_hparams(str(args.hparams))
    module_names = _module_scope_names("qk_gate_sampled_depths")
    projector_bank_dir = gen_dir / "runtime_projector_banks" / "qk_gate_sampled_depths_5records"
    selected_records_5_path = gen_dir / "generation_selected_records_5.json"
    _json_dump(selected_records_5_path, records)
    _configure_hparams_for_scope(
        hparams=hparams,
        image_root=Path(args.image_root),
        bank_dir=projector_bank_dir,
        device=str(args.device),
        module_names=module_names,
        lora_steps=20,
        lora_ref_loss_weight=0.0,
    )
    editor = MultimodalEditor.from_hparams(hparams)
    selected = [layer.name for layer in select_linear_layers(editor.model, hparams)]
    if set(selected) != set(module_names):
        raise RuntimeError({"reason": "selected module mismatch", "selected": selected, "expected": module_names})
    bank_status = _existing_projector_bank_matches(EngramBank, projector_bank_dir, records)
    if bank_status.get("status") != "complete":
        bank_status = _extract_projector_bank(editor, hparams, selected_records_5_path, records, projector_bank_dir)
        bank_status["reused"] = False
    _json_dump(gen_dir / "generation_projector_bank_status.json", bank_status)
    entries, edit_status = _build_edit_bank(
        model=editor.model,
        records=records,
        image_root=Path(args.image_root),
        hparams=hparams,
        projector_bank_dir=projector_bank_dir,
        out_dir=gen_dir,
        bank_config=_bank_configs()[0],
    )
    _json_dump(gen_dir / "generation_edit_bank_status.json", edit_status)
    query_cache = _build_query_cache(
        model=editor.model,
        records=records,
        image_root=Path(args.image_root),
        baselines={_record_id(record): {} for record in records},
        module_names=module_names,
        out_dir=gen_dir,
    )
    if config.get("threshold") == "adaptive_p90_reference":
        threshold_info = _compute_adaptive_threshold(
            model=editor.model,
            records=records,
            image_root=Path(args.image_root),
            entries=entries,
            prototype_type=str(config["prototype_type"]),
            module_names=module_names,
            query_cache=query_cache,
        )
        threshold = float(threshold_info["threshold"])
    else:
        threshold_info = {"status": "not_required", "threshold": config.get("threshold")}
        threshold = None if config.get("threshold") is None else float(config["threshold"])

    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        baseline = _evaluate_current(
            editor.model,
            record,
            Path(args.image_root),
            max_new_tokens=int(args.generation_max_new_tokens),
            min_new_tokens=None,
            skip_generation=False,
        )
        rows.append(_generation_text_metrics("baseline", record, idx, baseline.get("generation") or {}))

        query = query_cache[(_record_id(record), "new")]["query"]
        routed = route_edit_ids(
            query=query,
            entries=entries,
            query_record_id=_record_id(record),
            query_kind="new",
            route_policy=str(config["route_policy"]),
            prototype_type=str(config["prototype_type"]),
            threshold=threshold,
            max_active_edits=int(config.get("max_active_edits") or 1),
        )
        active_by_id = {entry["edit_id"]: entry for entry in entries}
        active_entries = [active_by_id[edit_id] for edit_id in routed["active_edit_ids"]]
        patch = RoutedLoraPatch(editor.model, active_entries, [1.0 for _ in active_entries])
        patch.install()
        try:
            routed_result = _evaluate_current(
                editor.model,
                record,
                Path(args.image_root),
                max_new_tokens=int(args.generation_max_new_tokens),
                min_new_tokens=None,
                skip_generation=False,
            )
        finally:
            patch.remove()
        rows.append(_generation_text_metrics(str(best["config_id"]), record, idx, routed_result.get("generation") or {}, routed))

    payload = {
        "status": "complete",
        "primary_gate": False,
        "best_config_id": best["config_id"],
        "threshold_info": threshold_info,
        "records": len(records),
        "rows": rows,
        "skipped_methods": [
            {"method": "C_high_merged", "reason": "not cheaply reproducible without rerunning merged sequential state"},
            {"method": "C_low_merged", "reason": "not cheaply reproducible without rerunning merged sequential state"},
        ],
    }
    _json_dump(gen_dir / "generation_5records.json", payload)
    _write_csv(gen_dir / "generation_5records.csv", rows)

    for path in [gen_dir / "runtime_projector_banks", gen_dir / "runtime_edit_banks"]:
        if path.exists():
            shutil.rmtree(path)
    data_reuse = {"selected_record_count": len(all_records), "record_id_match_rate": 1.0}
    _write_final_report(out_dir=out_dir, scores=scores, best=best, data_reuse=data_reuse, generation=payload, plots={"status": "complete"})
    _write_package_hygiene_report(out_dir)
    return 0


def _run_gpu(args: argparse.Namespace) -> int:
    import torch

    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    test_status = _write_tests(out_dir, run_tests=not args.skip_tests)
    preflight = _write_preflight(
        out_dir,
        hparams_path=Path(args.hparams),
        input_records=Path(args.selected_records),
        image_root=Path(args.image_root),
        test_status=test_status,
    )
    records = _load_records(Path(args.selected_records))[: int(args.max_records)]
    data_reuse = _write_data_reuse_report(
        out_dir=out_dir,
        selected_records_path=Path(args.selected_records),
        previous_record_preflight=Path(args.previous_record_preflight),
        records=records,
    )
    _json_dump(out_dir / "routed_bank_config_grid.json", {"bank_configs": _bank_configs(), "routed_configs": _routed_configs(), "anchors": _anchor_rows()})
    if test_status.get("status") != "pass" or preflight.get("status") != "pass":
        _json_dump(out_dir / "runtime.json", {"status": "blocked_preflight", "test_status": test_status, "preflight": preflight})
        return 2

    heavy = _heavy_imports()
    MultimodalEditor = heavy["MultimodalEditor"]
    EngramMultimodalHparams = heavy["EngramMultimodalHparams"]
    EngramBank = heavy["EngramBank"]
    select_linear_layers = heavy["select_linear_layers"]
    _extract_projector_bank = heavy["_extract_projector_bank"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hparams = EngramMultimodalHparams.from_hparams(str(args.hparams))
    module_names = _module_scope_names("qk_gate_sampled_depths")
    projector_bank_dir = out_dir / "runtime_projector_banks" / "qk_gate_sampled_depths"
    _configure_hparams_for_scope(
        hparams=hparams,
        image_root=Path(args.image_root),
        bank_dir=projector_bank_dir,
        device=str(args.device),
        module_names=module_names,
        lora_steps=20,
        lora_ref_loss_weight=0.0,
    )
    editor = MultimodalEditor.from_hparams(hparams)
    selected = [layer.name for layer in select_linear_layers(editor.model, hparams)]
    if set(selected) != set(module_names):
        raise RuntimeError({"reason": "selected module mismatch", "selected": selected, "expected": module_names})
    _json_dump(out_dir / "audit" / "selected_modules_qk_gate_sampled_depths.json", {"status": "pass", "selected_modules": selected})
    bank_status = _existing_projector_bank_matches(EngramBank, projector_bank_dir, records)
    if bank_status.get("status") != "complete":
        bank_status = _extract_projector_bank(editor, hparams, Path(args.selected_records), records, projector_bank_dir)
        bank_status["reused"] = False
    _json_dump(out_dir / "audit" / "runtime_projector_bank_status.json", bank_status)

    baselines = _read_json(Path(args.baseline_metrics))
    scores: List[Dict[str, Any]] = list(_anchor_rows())
    payloads: Dict[str, Dict[str, Any]] = {}
    routing_metric_rows: List[Dict[str, Any]] = []
    bank_entries_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for bank_config in _bank_configs():
        entries, status = _build_edit_bank(
            model=editor.model,
            records=records,
            image_root=Path(args.image_root),
            hparams=hparams,
            projector_bank_dir=projector_bank_dir,
            out_dir=out_dir,
            bank_config=bank_config,
        )
        bank_entries_by_id[str(bank_config["bank_config_id"])] = entries
        _json_dump(out_dir / "bank_metadata" / f"{bank_config['bank_config_id']}_status.json", status)
    query_cache = _build_query_cache(
        model=editor.model,
        records=records,
        image_root=Path(args.image_root),
        baselines=baselines,
        module_names=module_names,
        out_dir=out_dir,
    )
    for config in _routed_configs():
        entries = bank_entries_by_id[str(config["bank_config_id"])]
        run_dir = out_dir / "routed_runs" / str(config["config_id"])
        payload = _evaluate_routed_config(
            model=editor.model,
            records=records,
            image_root=Path(args.image_root),
            baselines=baselines,
            all_entries=entries,
            config=config,
            run_dir=run_dir,
            module_names=module_names,
            query_cache=query_cache,
            rollback_tolerance=float(args.rollback_tolerance),
            locality_threshold=float(args.locality_damage_threshold),
        )
        payloads[str(config["config_id"])] = payload
        score = _score_routed(payload)
        scores.append(score)
        routing_metric_rows.append(payload["routing_metrics"])
        _json_dump(run_dir / "score.json", score)
    best = _choose_best(scores)
    _write_csv(out_dir / "routed_bank_summary.csv", scores)
    _json_dump(out_dir / "routed_bank_summary.json", {"scores": scores})
    _write_csv(out_dir / "routing_metrics.csv", routing_metric_rows)
    _json_dump(out_dir / "routing_metrics.json", {"metrics": routing_metric_rows})
    _write_best_analysis(out_dir, best, payloads)
    plots = _plot_optional_routed(out_dir, scores, payloads, best)
    generation = {"status": "skipped", "reason": "generation diagnostics not implemented in this bounded NLL/logprob gate"}
    _json_dump(out_dir / "generation_diagnostics" / "generation_5records.json", generation)
    _write_final_report(out_dir=out_dir, scores=scores, best=best, data_reuse=data_reuse, generation=generation, plots=plots)
    _json_dump(out_dir / "runtime.json", {"status": "complete", "best": best})
    _write_package_hygiene_report(out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded MedMKEB 20-edit ENGRAM routed edit-bank pilot.")
    parser.add_argument("--mode", choices=["prepare", "run-gpu", "generation"], default="prepare")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / ROUTED_DIRNAME))
    parser.add_argument("--selected-records", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "medmkeb_modelknown_20.json"))
    parser.add_argument("--baseline-metrics", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "baseline_metrics.json"))
    parser.add_argument("--previous-record-preflight", default=str(DEFAULT_OUTPUT_DIR / "modelknown_20" / "record_id_preflight.json"))
    parser.add_argument("--image-root", default="/Volumes/DataP/knowledge_editing/data/medmkeb/images")
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--generation-records", type=int, default=5)
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "prepare":
        return _prepare(args)
    if args.mode == "generation":
        return _run_generation(args)
    return _run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
