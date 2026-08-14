#!/usr/bin/env python3
"""Bounded SR-TR ENGRAM V2 one-edit and two-order continual-editing gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dsca_medmkeb_diag_common import clone_batch, ensure_offline_env, to_jsonable
from easyeditor.models.engram import EngramMultimodalHparams
from easyeditor.models.engram_v2 import (
    SRTRMultimodalEditor,
    SRTRSolverConfig,
    SequentialEngramBankV2,
)
from easyeditor.trainer.models import get_model
from scripts.engram.engram_eval_utils import full_state_sha256, shifted_teacher_forced_metrics, tensor_sha256

EXPECTED_RECORD_IDS = ["953", "1293"]
ONE_EDIT_RECORD_ID = "953"
UNRELATED_RECORD_ID = "1293"
SUCCESS_EPS = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("one", "two"))
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--config", default="hparams/ENGRAM/llava_med_continual_v2.yaml")
    parser.add_argument("--model-config", default="hparams/ENGRAM/llava_med_continual_v1.yaml")
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_code_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite V2 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n")


def resolve_image(root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return root.parent / path if root.name == "images" and str(value).startswith("images/") else root / path


def prompt(value: Any) -> str:
    return f"Question: {str(value or '')} Short answer: "


def make_sample(model: Any, question: Any, answer: Any, image: Path) -> Dict[str, Any]:
    text = prompt(question)
    target = str(answer or "")
    labels = model.llava_tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return {
        "image_path": [str(image)],
        "prompt": [text],
        "target": [target],
        "text_input": [text + target],
        "labels": labels,
        "prompts_len": [len(model.llava_tokenizer(text, add_special_tokens=False).input_ids)],
    }


def build_views(model: Any, record: Dict[str, Any], image_root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "target": make_sample(model, record["src"], record["alt"], resolve_image(image_root, record["image"])),
        "generalization": make_sample(
            model, record["rephrase"], record["alt"], resolve_image(image_root, record["image_rephrase"])
        ),
        "locality": make_sample(
            model, record["m_loc_q"], record["m_loc_a"], resolve_image(image_root, record["m_loc"])
        ),
    }


def build_unrelated(model: Any, record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    return make_sample(model, record["src"], record["pred"], resolve_image(image_root, record["image"]))


def evaluate(model: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
    with torch.inference_mode():
        outputs = model(clone_batch(sample))
    return shifted_teacher_forced_metrics(outputs.logits, outputs.labels, ignore_index=model.IGNORE_INDEX)


def evaluate_views(model: Any, views: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {name: evaluate(model, sample) for name, sample in views.items()}


def metric_comparison(
    baseline: Dict[str, Dict[str, Any]], current: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for name, metric in current.items():
        base = baseline[name]
        improvement = float(base["target_nll"] - metric["target_nll"])
        rows[name] = {
            "baseline": base,
            "current": metric,
            "target_nll_improvement": improvement,
            "target_nll_abs_drift": abs(improvement),
        }
    return rows


def module_weight(model: torch.nn.Module, module_name: str) -> torch.Tensor:
    module = dict(model.named_modules()).get(module_name)
    if not isinstance(module, torch.nn.Linear):
        raise KeyError(f"Target module is not Linear: {module_name}")
    return module.weight


def restore_anchor(bank: SequentialEngramBankV2, model: torch.nn.Module) -> None:
    bank.rollback_to_prefix(model, 0)
    torch.cuda.empty_cache()


def delta_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    l = left.detach().double().reshape(-1)
    r = right.detach().double().reshape(-1)
    denom = float(l.norm() * r.norm())
    return float(l.dot(r) / denom) if denom > 0 else 0.0


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    ql = torch.linalg.qr(left.detach().double(), mode="reduced").Q
    qr = torch.linalg.qr(right.detach().double(), mode="reduced").Q
    singular = torch.linalg.svdvals(ql.transpose(0, 1).matmul(qr))
    return {
        "principal_cosines": singular.cpu().tolist(),
        "mean_squared_principal_cosine": float(singular.square().mean().cpu()) if singular.numel() else 0.0,
        "max_principal_cosine": float(singular.max().cpu()) if singular.numel() else 0.0,
    }


def max_metric_diff(left: Dict[str, Dict[str, Any]], right: Dict[str, Dict[str, Any]]) -> float:
    return max(
        (abs(float(left[name]["target_nll"]) - float(right[name]["target_nll"])) for name in left),
        default=0.0,
    )


def runtime_metadata() -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_cuda_device": 0,
        "physical_gpu_claim": "GPU0 (enforced by CUDA_VISIBLE_DEVICES=0)",
        "gpu_name": props.name,
        "gpu_total_memory": props.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled() if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None,
    }


def make_editor(config: Dict[str, Any]) -> SRTRMultimodalEditor:
    solver = SRTRSolverConfig(
        beta_ref=float(config["beta_ref"]),
        beta_old=float(config["beta_old"]),
        ridge_relative=float(config["ridge_relative"]),
        max_relative_weight_norm=float(config["max_relative_weight_norm"]),
        solve_dtype=torch.float64,
    )
    return SRTRMultimodalEditor(str(config["target_module"]), solver)


def init_bank(
    root: Path, model: torch.nn.Module, module_name: str, metadata: Dict[str, Any]
) -> SequentialEngramBankV2:
    if root.exists():
        raise FileExistsError(f"Refusing to reuse V2 bank directory: {root}")
    bank = SequentialEngramBankV2(root)
    key = f"{module_name}.weight"
    bank.initialize_anchor({key: module_weight(model, module_name)}, metadata=metadata)
    return bank


def one_edit_gate(
    comparison: Dict[str, Dict[str, Any]],
    stats: Dict[str, Any],
    bank_equivalence: Dict[str, Any],
    locality_limit: float,
    replay_tolerance: float,
) -> Dict[str, Any]:
    checks = {
        "target_improves": comparison["target"]["target_nll_improvement"] > SUCCESS_EPS,
        "generalization_improves": comparison["generalization"]["target_nll_improvement"] > SUCCESS_EPS,
        "locality_within_limit": comparison["locality"]["target_nll_abs_drift"] <= locality_limit,
        "unrelated_within_limit": comparison["unrelated"]["target_nll_abs_drift"] <= locality_limit,
        "trust_region_respected": float(stats["delta_relative_norm"]) <= 0.010001,
        "direct_reload_state_equal": bool(bank_equivalence["direct_reload_full_state_equal"]),
        "direct_reload_metrics_equal": float(bank_equivalence["direct_reload_max_nll_abs_diff"]) <= replay_tolerance,
        "rollback_anchor_state_equal": bool(bank_equivalence["rollback_anchor_full_state_equal"]),
        "rollback_baseline_metrics_equal": float(bank_equivalence["rollback_baseline_max_nll_abs_diff"]) <= replay_tolerance,
    }
    return {"passed": all(checks.values()), "checks": checks}


def run_one(
    *,
    model: Any,
    config: Dict[str, Any],
    records: Dict[str, Dict[str, Any]],
    image_root: Path,
    work_dir: Path,
    code_hash: str,
    config_hash: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    views = build_views(model, records[ONE_EDIT_RECORD_ID], image_root)
    views["unrelated"] = build_unrelated(model, records[UNRELATED_RECORD_ID], image_root)
    baseline = evaluate_views(model, views)
    anchor_full_hash, anchor_full_meta = full_state_sha256(model)
    module_name = str(config["target_module"])
    bank = init_bank(
        work_dir / "bank",
        model,
        module_name,
        {"record_ids": [ONE_EDIT_RECORD_ID], "config_hash": config_hash, "code_hash": code_hash},
    )
    editor = make_editor(config)
    edit_report = editor.solve_and_store(
        model=model,
        target_sample=views["target"],
        reference_sample=views["locality"],
        bank=bank,
        edit_id=f"edit_01_{ONE_EDIT_RECORD_ID}",
        source_example_ids=[ONE_EDIT_RECORD_ID],
        code_hash=code_hash,
        config_hash=config_hash,
        ignore_index=model.IGNORE_INDEX,
    )
    torch.cuda.empty_cache()
    direct = evaluate_views(model, views)
    comparison = metric_comparison(baseline, direct)
    direct_full_hash, direct_full_meta = full_state_sha256(model)

    restore_anchor(bank, model)
    rollback = evaluate_views(model, views)
    rollback_full_hash, rollback_full_meta = full_state_sha256(model)

    fresh_bank = SequentialEngramBankV2(work_dir / "bank")
    fresh_bank.assemble_state_into_model(model)
    reload_metrics = evaluate_views(model, views)
    reload_full_hash, reload_full_meta = full_state_sha256(model)
    bank_equivalence = {
        "mode": "one_edit",
        "anchor_full_state_sha256": anchor_full_hash,
        "direct_full_state_sha256": direct_full_hash,
        "reload_full_state_sha256": reload_full_hash,
        "rollback_full_state_sha256": rollback_full_hash,
        "direct_reload_full_state_equal": direct_full_hash == reload_full_hash,
        "rollback_anchor_full_state_equal": rollback_full_hash == anchor_full_hash,
        "direct_reload_max_nll_abs_diff": max_metric_diff(direct, reload_metrics),
        "rollback_baseline_max_nll_abs_diff": max_metric_diff(baseline, rollback),
        "bank_current_state_hash": fresh_bank.current_state_hash(),
        "full_state_metadata": {
            "anchor": anchor_full_meta,
            "direct": direct_full_meta,
            "reload": reload_full_meta,
            "rollback": rollback_full_meta,
        },
    }
    gate = one_edit_gate(
        comparison,
        edit_report["solver_stats"],
        bank_equivalence,
        float(config["locality_nll_abs_drift_limit"]),
        float(config["replay_nll_tolerance"]),
    )
    v1_path = ROOT / "outputs/engram_v2_method_development_20260710/phase_a/v1_replay_fixed/v1_replay_fixed.json"
    payload = {
        "method": "Sequential Residual Target-Representation ENGRAM",
        "method_version": config["method_version"],
        "mode": "one_edit",
        "record_id": ONE_EDIT_RECORD_ID,
        "runtime": runtime_metadata(),
        "config_hash": config_hash,
        "code_hash": code_hash,
        "baseline": baseline,
        "direct": direct,
        "comparison": comparison,
        "edit_report": edit_report,
        "bank_equivalence": bank_equivalence,
        "gate": gate,
        "v1_corrected_reference": json.loads(v1_path.read_text()) if v1_path.exists() else None,
    }
    return payload, bank_equivalence


def order_metrics(
    model: Any, views: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {record_id: evaluate_views(model, record_views) for record_id, record_views in views.items()}


def metric_tree_diff(
    left: Dict[str, Dict[str, Dict[str, Any]]], right: Dict[str, Dict[str, Dict[str, Any]]]
) -> float:
    values = []
    for record_id in left:
        values.append(max_metric_diff(left[record_id], right[record_id]))
    return max(values, default=0.0)


def run_order(
    *,
    model: Any,
    config: Dict[str, Any],
    order: Sequence[str],
    records: Dict[str, Dict[str, Any]],
    views: Dict[str, Dict[str, Dict[str, Any]]],
    baseline: Dict[str, Dict[str, Dict[str, Any]]],
    order_dir: Path,
    code_hash: str,
    config_hash: str,
    anchor_full_hash: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    module_name = str(config["target_module"])
    bank = init_bank(
        order_dir / "bank",
        model,
        module_name,
        {"record_ids": list(order), "config_hash": config_hash, "code_hash": code_hash},
    )
    editor = make_editor(config)
    steps = []
    for step, record_id in enumerate(order, start=1):
        report = editor.solve_and_store(
            model=model,
            target_sample=views[record_id]["target"],
            reference_sample=views[record_id]["locality"],
            bank=bank,
            edit_id=f"edit_{step:02d}_{record_id}",
            source_example_ids=[record_id],
            code_hash=code_hash,
            config_hash=config_hash,
            ignore_index=model.IGNORE_INDEX,
        )
        torch.cuda.empty_cache()
        current = order_metrics(model, views)
        comparisons = {
            rid: metric_comparison(baseline[rid], current[rid]) for rid in EXPECTED_RECORD_IDS
        }
        steps.append({"step": step, "record_id": record_id, "edit_report": report, "metrics": current, "comparison": comparisons})

    direct = order_metrics(model, views)
    direct_full_hash, direct_meta = full_state_sha256(model)
    payloads = [bank.load_edit(f"edit_{step:02d}_{record_id}") for step, record_id in enumerate(order, start=1)]
    key = f"{module_name}.weight"
    diagnostics = {
        "delta_cosine": delta_cosine(payloads[0]["deltas"][key], payloads[1]["deltas"][key]),
        "activation_subspace_overlap": subspace_overlap(
            payloads[0]["target_factors"][key], payloads[1]["target_factors"][key]
        ),
        "delta_relative_norms": [
            float(step["edit_report"]["solver_stats"]["delta_relative_norm"]) for step in steps
        ],
    }

    restore_anchor(bank, model)
    rollback = order_metrics(model, views)
    rollback_full_hash, rollback_meta = full_state_sha256(model)
    fresh_bank = SequentialEngramBankV2(order_dir / "bank")
    fresh_bank.assemble_state_into_model(model)
    reload_metrics = order_metrics(model, views)
    reload_full_hash, reload_meta = full_state_sha256(model)
    equivalence = {
        "order": list(order),
        "direct_full_state_sha256": direct_full_hash,
        "reload_full_state_sha256": reload_full_hash,
        "rollback_full_state_sha256": rollback_full_hash,
        "direct_reload_full_state_equal": direct_full_hash == reload_full_hash,
        "rollback_anchor_full_state_equal": rollback_full_hash == anchor_full_hash,
        "direct_reload_max_nll_abs_diff": metric_tree_diff(direct, reload_metrics),
        "rollback_baseline_max_nll_abs_diff": metric_tree_diff(baseline, rollback),
        "full_state_metadata": {"direct": direct_meta, "reload": reload_meta, "rollback": rollback_meta},
    }
    restore_anchor(bank, model)
    order_result = {
        "order": list(order),
        "steps": steps,
        "final_metrics": direct,
        "final_comparison": {
            rid: metric_comparison(baseline[rid], direct[rid]) for rid in EXPECTED_RECORD_IDS
        },
        "diagnostics": diagnostics,
        "bank_equivalence": equivalence,
    }
    return order_result, equivalence


def two_edit_gate(results: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    locality_limit = float(config["locality_nll_abs_drift_limit"])
    replay_tolerance = float(config["replay_nll_tolerance"])
    checks: Dict[str, bool] = {}
    for result in results:
        label = "_then_".join(result["order"])
        steps = result["steps"]
        checks[f"{label}_each_current_target_improves"] = all(
            step["comparison"][step["record_id"]]["target"]["target_nll_improvement"] > SUCCESS_EPS
            for step in steps
        )
        checks[f"{label}_both_final_targets_retained"] = all(
            result["final_comparison"][rid]["target"]["target_nll_improvement"] > SUCCESS_EPS
            for rid in EXPECTED_RECORD_IDS
        )
        checks[f"{label}_locality_within_limit"] = all(
            result["final_comparison"][rid]["locality"]["target_nll_abs_drift"] <= locality_limit
            for rid in EXPECTED_RECORD_IDS
        )
        checks[f"{label}_trust_regions_respected"] = all(
            value <= 0.010001 for value in result["diagnostics"]["delta_relative_norms"]
        )
        eq = result["bank_equivalence"]
        checks[f"{label}_bank_reload_equal"] = bool(eq["direct_reload_full_state_equal"]) and float(
            eq["direct_reload_max_nll_abs_diff"]
        ) <= replay_tolerance
        checks[f"{label}_rollback_equal"] = bool(eq["rollback_anchor_full_state_equal"]) and float(
            eq["rollback_baseline_max_nll_abs_diff"]
        ) <= replay_tolerance
    return {"passed": all(checks.values()), "checks": checks}


def run_two(
    *,
    model: Any,
    config: Dict[str, Any],
    records: Dict[str, Dict[str, Any]],
    image_root: Path,
    work_dir: Path,
    code_hash: str,
    config_hash: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    views = {rid: build_views(model, records[rid], image_root) for rid in EXPECTED_RECORD_IDS}
    baseline = {rid: evaluate_views(model, views[rid]) for rid in EXPECTED_RECORD_IDS}
    anchor_full_hash, anchor_meta = full_state_sha256(model)
    orders = (["953", "1293"], ["1293", "953"])
    results = []
    equivalence_rows = []
    for order in orders:
        label = "_then_".join(order)
        result, equivalence = run_order(
            model=model,
            config=config,
            order=order,
            records=records,
            views=views,
            baseline=baseline,
            order_dir=work_dir / label,
            code_hash=code_hash,
            config_hash=config_hash,
            anchor_full_hash=anchor_full_hash,
        )
        results.append(result)
        equivalence_rows.append(equivalence)

    order_analysis = {
        "orders": [result["order"] for result in results],
        "final_target_improvement_abs_differences": {
            rid: abs(
                float(results[0]["final_comparison"][rid]["target"]["target_nll_improvement"])
                - float(results[1]["final_comparison"][rid]["target"]["target_nll_improvement"])
            )
            for rid in EXPECTED_RECORD_IDS
        },
        "within_order_delta_cosines": {
            "_then_".join(result["order"]): result["diagnostics"]["delta_cosine"] for result in results
        },
        "within_order_activation_subspace_overlap": {
            "_then_".join(result["order"]): result["diagnostics"]["activation_subspace_overlap"]
            for result in results
        },
        "interpretation": "SR-TR V2 is sequential and state-dependent; order equality is measured, not assumed.",
    }
    gate = two_edit_gate(results, config)
    payload = {
        "method": "Sequential Residual Target-Representation ENGRAM",
        "method_version": config["method_version"],
        "mode": "two_edit_two_order",
        "runtime": runtime_metadata(),
        "config_hash": config_hash,
        "code_hash": code_hash,
        "anchor_full_state_sha256": anchor_full_hash,
        "anchor_full_state_metadata": anchor_meta,
        "baseline": baseline,
        "order_results": results,
        "order_analysis": order_analysis,
        "gate": gate,
    }
    bank_equivalence = {"mode": "two_edit_two_order", "orders": equivalence_rows}
    return payload, order_analysis, bank_equivalence


def main() -> None:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    result_path = args.work_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed V2 result: {result_path}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    ensure_offline_env()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("This bounded experiment must run on physical GPU0 with CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")

    config_path = (ROOT / args.config).resolve()
    model_config_path = (ROOT / args.model_config).resolve()
    config = yaml.safe_load(config_path.read_text())
    set_determinism(int(config["seed"]))
    model_config = EngramMultimodalHparams.from_hparams(str(model_config_path))
    model_config.dropout = 0.0
    model_config.no_grad_layers = None
    model_config.device = "cuda"
    model = get_model(model_config).to(torch.device("cuda")).eval()
    records_raw = json.loads((ROOT / config["dataset"]).read_text())
    records = {str(record["id"]): record for record in records_raw}
    if any(rid not in records for rid in EXPECTED_RECORD_IDS):
        raise RuntimeError(f"Missing frozen V2 record IDs: {EXPECTED_RECORD_IDS}")
    image_root = Path(model_config.coco_image)
    if not image_root.is_absolute():
        image_root = ROOT / image_root
    code_paths = [
        ROOT / "easyeditor/models/engram_v2/solver.py",
        ROOT / "easyeditor/models/engram_v2/bank.py",
        ROOT / "easyeditor/models/engram_v2/editor.py",
        Path(__file__).resolve(),
    ]
    code_hash = aggregate_code_hash(code_paths)
    config_hash = sha256_file(config_path)
    if args.mode == "one":
        result, bank_equivalence = run_one(
            model=model,
            config=config,
            records=records,
            image_root=image_root,
            work_dir=args.work_dir,
            code_hash=code_hash,
            config_hash=config_hash,
        )
        write_json(result_path, result)
        write_json(args.work_dir / "bank_equivalence.json", bank_equivalence)
        print(json.dumps({"mode": "one", "gate": result["gate"], "result": str(result_path)}, indent=2))
    else:
        result, order_analysis, bank_equivalence = run_two(
            model=model,
            config=config,
            records=records,
            image_root=image_root,
            work_dir=args.work_dir,
            code_hash=code_hash,
            config_hash=config_hash,
        )
        write_json(result_path, result)
        write_json(args.work_dir / "order_analysis.json", order_analysis)
        write_json(args.work_dir / "bank_equivalence.json", bank_equivalence)
        print(json.dumps({"mode": "two", "gate": result["gate"], "result": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
