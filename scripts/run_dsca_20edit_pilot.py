#!/usr/bin/env python3
"""Run a small sequential DSCA pilot on real BLIP2 multimodal editing data."""

import argparse
import csv
import faulthandler
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml


METRIC_COLUMNS = ["rel", "t_gen", "m_gen", "t_loc", "m_loc"]
PILOT_DATASET_ALIASES = {"e-vqa", "evqa", "vqa", "vlkeb", "medmkeb"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BLIP2-OPT DSCA 20-edit pilot on real multimodal editing data.")
    parser.add_argument("--alg", default="DSCA", choices=["DSCA"], help="Editing algorithm. Only DSCA is run here.")
    parser.add_argument("--model", default="blip2", choices=["blip2"], help="Backbone to validate.")
    parser.add_argument(
        "--dataset",
        default="VLKEB",
        choices=["VLKEB", "MEDMKEB", "MedMKEB", "medmkeb", "E-VQA", "EVQA", "vqa", "vlkeb", "e-vqa", "evqa"],
        help="Dataset name. Use VLKEB for HymanH/VLKEB-data, MEDMKEB for local medical editing data, and E-VQA only for explicit E-VQA paths.",
    )
    parser.add_argument("--dataset-path", default=None, help="Optional explicit VLKEB/E-VQA JSON path.")
    parser.add_argument("--image-root", default=None, help="Optional image root override for dataset records.")
    parser.add_argument("--rephrase-image-root", default=None, help="Optional rephrase image root override; defaults to --image-root.")
    parser.add_argument("--hparams", default="hparams/DSCA/blip2_20edit_pilot.yaml")
    parser.add_argument("--training-hparams", default="hparams/TRAINING/DSCA/blip2_20edit_pilot.yaml")
    parser.add_argument("--num-edits", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--refine-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and dataset paths without loading BLIP2.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Reserved for generation-based extensions.")
    parser.add_argument("--limit-loc-samples", type=int, default=None, help="Reserved for large locality sets.")
    parser.add_argument(
        "--base-signature-check",
        default="cheap",
        choices=["none", "cheap", "full", "final", "every-step", "off"],
        help="Base VLM mutation check mode. `full`/`final` scan before and after the run; `every-step` is diagnostic only.",
    )
    parser.add_argument("--phase-timing", action="store_true", help="Print per-step phase timing diagnostics.")
    parser.add_argument("--profile-edit-step", action="store_true", help="Enable DSCA.edit_step internal JSONL profiling.")
    parser.add_argument("--profile-start-step", type=int, default=1)
    parser.add_argument("--profile-end-step", type=int, default=10**9)
    parser.add_argument("--edit-step-timeout-sec", type=int, default=0)
    parser.add_argument("--dump-traceback-on-timeout", action="store_true")
    parser.add_argument("--disable-pca-refine", action="store_true", help="Diagnostic only: skip scheduled refine_subspaces.")
    parser.add_argument("--disable-basis-initialization", action="store_true", help="Diagnostic only: keep DSAM bases inactive.")
    parser.add_argument("--disable-cdistill", action="store_true", help="Diagnostic only: zero contrastive distillation loss.")
    parser.add_argument("--disable-align", action="store_true", help="Diagnostic only: zero alignment loss.")
    parser.add_argument("--disable-sparse", action="store_true", help="Diagnostic only: zero sparse routing loss.")
    parser.add_argument("--disable-task-loss", action="store_true", help="Diagnostic only: zero edit task loss.")
    parser.add_argument("--disable-repository-save-load-validation", action="store_true", help="Diagnostic only.")
    parser.add_argument("--disable-gradient-diagnostics", action="store_true", help="Diagnostic only.")
    parser.add_argument("--max-profile-steps", type=int, default=None, help="Stop after this many completed steps.")
    return parser.parse_args()


def ensure_offline_env() -> None:
    os.environ.setdefault("HF_HOME", str(Path("hugging_cache").resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def as_plain_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return dict(obj)


def normalize_device(device: str) -> Any:
    if device == "cuda":
        return "cuda"
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        return int(suffix) if suffix.isdigit() else device
    if device.isdigit():
        return int(device)
    return device


def torch_device(device: Any) -> torch.device:
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def apply_cli_overrides(config: Any, args: argparse.Namespace) -> Any:
    config.device = normalize_device(args.device)
    config.batch_size = args.batch_size
    config.val_batch_size = args.batch_size
    config.seed = args.seed
    config.dsca_rank = args.rank
    config.dsca_min_samples = args.min_samples
    config.dsca_refine_interval = args.refine_interval
    config.dsca_route_temperature = 0.07
    config.dsca_distill_temperature = 0.07
    config.dsca_lambda_align = 0.5
    config.dsca_lambda_distill = 1.0
    config.dsca_lambda_sparse = 1.0e-2
    config.dsca_task_weight = 1.0
    config.dsca_update_clusters_during_training = True
    config.dsca_update_clusters_during_inference = False
    config.dsca_freeze_vlm = True
    config.dsca_require_masks = True
    if args.image_root:
        config.coco_image = args.image_root
    if args.rephrase_image_root or args.image_root:
        config.rephrase_image = args.rephrase_image_root or args.image_root
    config.max_iters = args.num_edits
    return config


def apply_profile_overrides(config: Any, args: argparse.Namespace, out_dir: Path) -> Any:
    config.dsca_profile_edit_step = bool(args.profile_edit_step)
    config.dsca_profile_log_path = str(out_dir / "dsca_edit_step_profile.jsonl")
    config.dsca_profile_start_step = int(args.profile_start_step)
    config.dsca_profile_end_step = int(args.profile_end_step)
    config.dsca_disable_pca_refine = bool(args.disable_pca_refine)
    config.dsca_disable_basis_initialization = bool(args.disable_basis_initialization)
    config.dsca_disable_cdistill = bool(args.disable_cdistill)
    config.dsca_disable_align = bool(args.disable_align)
    config.dsca_disable_sparse = bool(args.disable_sparse)
    config.dsca_disable_task_loss = bool(args.disable_task_loss)
    return config


def normalize_base_signature_mode(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized in {"off", "none"}:
        return "none"
    if normalized in {"final", "full"}:
        return "full"
    if normalized == "cheap":
        return "cheap"
    if normalized == "every-step":
        return "every-step"
    raise ValueError(f"Unsupported base signature check mode: {mode}")


def dataset_source_name(dataset: str) -> str:
    name = str(dataset).lower()
    if name == "vlkeb":
        return "HymanH/VLKEB-data"
    if name == "medmkeb":
        return "local MedMKEB medical editing data"
    return dataset


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def log_phase(enabled: bool, step: int, phase: str, event: str, elapsed: Optional[float] = None) -> None:
    if not enabled:
        return
    payload: Dict[str, Any] = {"step": step, "phase": phase, "event": event}
    if elapsed is not None:
        payload["elapsed_sec"] = elapsed
    print(json.dumps(payload, sort_keys=True), flush=True)


def available_dataset_files(root: Path) -> List[str]:
    candidates: List[str] = []
    for base in (root / "datasets", root / "data", root / "VLKEB", root / "E-VQA"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv"}:
                try:
                    candidates.append(str(path.relative_to(root)))
                except ValueError:
                    candidates.append(str(path))
            if len(candidates) >= 200:
                return candidates
    for name in ("eval.json", "eval_multihop.json", "train.json"):
        path = root / name
        if path.is_file():
            candidates.append(str(path.relative_to(root)))
    return candidates


def resolve_dataset_path(args: argparse.Namespace, config: Any, root: Path) -> Tuple[Optional[Path], List[Path]]:
    explicit = Path(args.dataset_path) if args.dataset_path else None
    if explicit is not None:
        explicit = explicit if explicit.is_absolute() else root / explicit
        return (explicit if explicit.is_file() else None), [explicit]

    dataset_name = args.dataset.lower()
    if dataset_name not in PILOT_DATASET_ALIASES:
        return None, []

    if dataset_name == "vlkeb":
        candidates = [
            root / "datasets" / "eval.json",
            root / "datasets" / "eval_multihop.json",
            root / "datasets" / "VLKEB" / "eval.json",
            root / "data" / "VLKEB" / "eval.json",
        ]
    elif dataset_name == "medmkeb":
        candidates = [
            root / "datasets" / "MedMKEB" / "eval.json",
            root / "datasets" / "MEDMKEB" / "eval.json",
            root / "datasets" / "medmkeb" / "eval.json",
            root / "data" / "MedMKEB" / "eval.json",
            root / "data" / "medmkeb" / "eval.json",
        ]
    else:
        candidates = [
            root / "datasets" / "E-VQA" / "eval.json",
            root / "datasets" / "EVQA" / "eval.json",
            root / "data" / "E-VQA" / "eval.json",
            root / "data" / "EVQA" / "eval.json",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, candidates
    return None, candidates


def validate_dataset_available(args: argparse.Namespace, config: Any, out_dir: Path) -> Path:
    root = Path.cwd()
    dataset_path, checked = resolve_dataset_path(args, config, root)
    available = available_dataset_files(root)
    expected_images = list(dict.fromkeys([Path(config.coco_image), Path(config.rephrase_image)]))
    missing_images = [str(p) for p in expected_images if not (p if p.is_absolute() else root / p).exists()]

    report = {
        "dataset": args.dataset,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "checked_paths": [str(p) for p in checked],
        "available_dataset_files": available,
        "expected_image_roots": [str(p) for p in expected_images],
        "missing_image_roots": missing_images,
    }
    write_json(out_dir / "dataset_discovery.json", report)

    errors = []
    if dataset_path is None:
        errors.append(
            f"{args.dataset} JSON was not found. Expected one of: "
            + ", ".join(str(p) for p in checked)
        )
    if missing_images:
        errors.append("Required image root(s) missing: " + ", ".join(missing_images))
    if errors:
        message = "\n".join(errors) + "\nAvailable dataset files:\n" + "\n".join(available or ["<none>"])
        (out_dir / "dataset_error.txt").write_text(message + "\n")
        raise FileNotFoundError(message)
    return dataset_path


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_or_raise(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise RuntimeError(f"{name} became non-finite: {value}")


def duplicate_optimizer_params(optimizer: torch.optim.Optimizer) -> bool:
    seen = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            ident = id(param)
            if ident in seen:
                return True
            seen.add(ident)
    return False


def base_param_signature(model: torch.nn.Module) -> Tuple[float, float]:
    total_sum = 0.0
    total_norm = 0.0
    with torch.no_grad():
        for param in model.parameters():
            data = param.detach().float()
            total_sum += float(data.sum().cpu())
            total_norm += float(data.norm().cpu())
    return total_sum, total_norm


def sampled_base_param_signature(model: torch.nn.Module, max_tensors: int = 16, values_per_tensor: int = 16) -> Tuple[float, float]:
    total_sum = 0.0
    total_norm = 0.0
    seen = 0
    with torch.no_grad():
        for param in model.parameters():
            if param.numel() == 0:
                continue
            data = param.detach().flatten()[:values_per_tensor].float().cpu()
            total_sum += float(data.sum())
            total_norm += float(data.norm())
            seen += 1
            if seen >= max_tensors:
                break
    return total_sum, total_norm


def signature_delta(lhs: Tuple[float, float], rhs: Tuple[float, float]) -> float:
    return abs(lhs[0] - rhs[0]) + abs(lhs[1] - rhs[1])


def move_batch_to_device(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    from easyeditor.trainer.utils import dict_to

    return dict_to(batch, device)


def clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, dict):
            cloned[key] = clone_batch(value)
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def logits_and_labels(model: torch.nn.Module, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        outputs = model(batch)
    logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
    labels = batch["labels"]
    if logits.dim() != 3:
        raise RuntimeError(f"Expected logits [B,T,V], got {tuple(logits.shape)}.")
    logits = logits[:, :-1]
    logits = logits[:, -labels.shape[1] :]
    return logits.detach().cpu(), labels.detach().cpu()


def target_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, torch.Tensor]:
    mask = labels != -100
    safe_labels = labels.masked_fill(~mask, 0)
    pred = logits.argmax(dim=-1).masked_fill(~mask, 0)
    denom = mask.sum().item()
    if denom == 0:
        return float("nan"), pred
    correct = ((pred == safe_labels) & mask).sum().item()
    return float(correct / denom), pred


def locality_preservation(
    base_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    labels: torch.Tensor,
    topk: int,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    mask = labels != -100
    k = min(topk, base_logits.shape[-1])
    base_top = torch.topk(torch.softmax(base_logits.float(), dim=-1), k=k, dim=-1).indices
    edited_top = torch.topk(torch.softmax(edited_logits.float(), dim=-1), k=k, dim=-1).indices
    if k == 1:
        preserved = base_top.squeeze(-1) == edited_top.squeeze(-1)
        pred = edited_top.squeeze(-1).masked_fill(~mask, 0)
        base_pred = base_top.squeeze(-1).masked_fill(~mask, 0)
    else:
        preserved = torch.zeros(mask.shape, dtype=torch.bool)
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                preserved[row, col] = any(int(item) in set(base_top[row, col].tolist()) for item in edited_top[row, col].tolist())
        pred = edited_top[:, :, 0].masked_fill(~mask, 0)
        base_pred = base_top[:, :, 0].masked_fill(~mask, 0)
    denom = mask.sum().item()
    if denom == 0:
        return float("nan"), base_pred, pred
    return float((preserved & mask).sum().item() / denom), base_pred, pred


def decode_prediction(tokenizer: Any, ids: torch.Tensor) -> str:
    flat = [int(x) for x in ids.view(-1).tolist() if int(x) >= 0]
    try:
        return tokenizer.decode(flat, skip_special_tokens=True).strip()
    except Exception:
        return " ".join(str(x) for x in flat)


def normalize_medical_answer(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"^(the\s+answer\s+is|answer\s*:|it\s+is)\s+", "", text)
    text = re.sub(r"[^\w\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _alias_list(aliases: Any) -> List[str]:
    if aliases is None:
        return []
    if isinstance(aliases, str):
        return [aliases]
    if isinstance(aliases, (list, tuple)):
        return [str(item) for item in aliases if item is not None]
    return [str(aliases)]


def answer_match_fields(base_prediction: Any, edited_prediction: Any, target: Any, aliases: Any = None) -> Dict[str, Any]:
    normalized_target = normalize_medical_answer(target)
    normalized_base = normalize_medical_answer(base_prediction)
    normalized_edited = normalize_medical_answer(edited_prediction)
    alias_norms = [normalize_medical_answer(item) for item in _alias_list(aliases)]
    alias_norms = [item for item in alias_norms if item]
    all_targets = [normalized_target] + alias_norms
    all_targets = [item for item in all_targets if item]
    exact_raw = str(edited_prediction).strip() == str(target).strip() if edited_prediction is not None and target is not None else False
    exact_norm = bool(normalized_edited and normalized_target and normalized_edited == normalized_target)
    contains = bool(normalized_edited and any(candidate in normalized_edited for candidate in all_targets))
    base_contains = bool(normalized_base and any(candidate in normalized_base for candidate in all_targets))
    alias_exact = bool(normalized_edited and any(normalized_edited == candidate for candidate in alias_norms))
    alias_contains = bool(normalized_edited and any(candidate in normalized_edited for candidate in alias_norms))
    return {
        "exact_match_raw": exact_raw,
        "exact_match_normalized": exact_norm,
        "contains_target": contains,
        "base_contains_target": base_contains,
        "edited_contains_target": contains,
        "normalized_target": normalized_target,
        "normalized_base_prediction": normalized_base,
        "normalized_edited_prediction": normalized_edited,
        "aliases": _alias_list(aliases),
        "alias_exact_match": alias_exact,
        "alias_contains_match": alias_contains,
        "edited_equals_base": normalized_edited == normalized_base if normalized_edited or normalized_base else False,
    }


def token_count(tokenizer: Any, text: Any) -> int:
    if text is None:
        return 0
    try:
        return int(len(tokenizer.encode(str(text), add_special_tokens=False)))
    except Exception:
        return int(len(str(text).split()))


def validate_masks_from_forward(model: torch.nn.Module, batch: Dict[str, Any]) -> None:
    with torch.no_grad():
        outputs = model(batch)
    for name in ("attention_mask", "vision_mask", "prompt_mask", "answer_mask"):
        value = getattr(outputs, name, None)
        if value is None:
            raise RuntimeError(f"Forward output did not provide {name}.")
    attention = outputs.attention_mask.bool()
    vision = outputs.vision_mask.bool()
    prompt = outputs.prompt_mask.bool()
    answer = outputs.answer_mask.bool()
    if not (attention.shape == vision.shape == prompt.shape == answer.shape):
        raise RuntimeError("Mask shapes do not align with each other.")
    if (vision & prompt).any() or (vision & answer).any() or (prompt & answer).any():
        raise RuntimeError("DSCA masks overlap.")
    if ((vision | prompt | answer) & ~attention).any():
        raise RuntimeError("DSCA masks include padding positions.")
    if (vision | answer).any() and (prompt | answer).any():
        pass
    if (vision & answer).any() or (prompt & answer).any():
        raise RuntimeError("Answer tokens leaked into routing masks.")


def sample_batch(dataset: Any, record: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return dataset.collate_fn([record])


def save_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "edit_id",
                "rel",
                "t_gen",
                "m_gen",
                "t_loc",
                "m_loc",
                "avg",
                "num_clusters",
                "num_active_dsams",
                "mean_subspace_overlap",
                "avg_candidates",
                "residual_norm_mean",
                "route_weight_l1_replay",
                "base_param_delta_norm",
                "peak_gpu_memory_mb",
                "time_per_edit_sec",
            ],
        )
        writer.writeheader()


def append_metrics(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def mean_available(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def evaluate_case(
    model: torch.nn.Module,
    edited_model: torch.nn.Module,
    tokenizer: Any,
    batch_part: Dict[str, Any],
    sample_type: str,
    prompt: str,
    target: str,
    image_path: Optional[str],
    predictions_path: Path,
    step: int,
    aliases: Any = None,
) -> float:
    try:
        base_logits, labels = logits_and_labels(model, clone_batch(batch_part))
        edited_logits, _ = logits_and_labels(edited_model, clone_batch(batch_part))
        if sample_type in {"t_loc", "m_loc"}:
            topk = 1 if sample_type == "t_loc" else 10
            value, base_pred, edited_pred = locality_preservation(base_logits, edited_logits, labels, topk=topk)
            correct_or_preserved = bool(math.isfinite(value) and value >= 1.0)
        else:
            value, edited_pred = target_accuracy(edited_logits, labels)
            _, base_pred = target_accuracy(base_logits, labels)
            correct_or_preserved = bool(math.isfinite(value) and value >= 1.0)
        base_text = decode_prediction(tokenizer, base_pred)
        edited_text = decode_prediction(tokenizer, edited_pred)
        match_fields = answer_match_fields(base_text, edited_text, target, aliases)
        append_jsonl(
            predictions_path,
            {
                "step": step,
                "sample_type": sample_type,
                "prompt": prompt,
                "target": target,
                "base_prediction": base_text,
                "edited_prediction": edited_text,
                "correct_or_preserved": correct_or_preserved,
                "score": value,
                "image_path": image_path,
                "target_token_count": token_count(tokenizer, target),
                "generated_token_count": token_count(tokenizer, edited_text),
                "generation_config": {"mode": "teacher_forced_argmax"},
                **match_fields,
            },
        )
        return value
    except RuntimeError as exc:
        match_fields = answer_match_fields(None, None, target, aliases)
        append_jsonl(
            predictions_path,
            {
                "step": step,
                "sample_type": sample_type,
                "prompt": prompt,
                "target": target,
                "base_prediction": None,
                "edited_prediction": None,
                "correct_or_preserved": None,
                "score": None,
                "image_path": image_path,
                "warning": str(exc),
                "target_token_count": token_count(tokenizer, target),
                "generated_token_count": 0,
                "generation_config": {"mode": "teacher_forced_argmax"},
                **match_fields,
            },
        )
        return float("nan")


def run_pilot(args: argparse.Namespace) -> int:
    ensure_offline_env()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from easyeditor.dataset import VQADataset
    from easyeditor.models.dsca.dsca_hparams import DSCAMultimodalHparams
    from easyeditor.trainer.algs.dsca import DSCA
    from easyeditor.trainer.models import get_model
    from easyeditor.trainer.training_hparams.dsca_multimodal_training_hparams import DSCAMultimodalTrainingHparams

    eval_config = apply_cli_overrides(DSCAMultimodalHparams.from_hparams(args.hparams), args)
    train_config = apply_cli_overrides(DSCAMultimodalTrainingHparams.from_hparams(args.training_hparams), args)
    eval_config = apply_profile_overrides(eval_config, args, out_dir)
    train_config = apply_profile_overrides(train_config, args, out_dir)
    base_signature_mode = normalize_base_signature_mode(args.base_signature_check)

    resolved = {
        "args": vars(args),
        "eval_hparams": as_plain_dict(eval_config),
        "training_hparams": as_plain_dict(train_config),
        "environment": {
            key: os.environ.get(key)
            for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONPATH")
        },
        "dataset_source": dataset_source_name(args.dataset),
    }
    write_yaml(out_dir / "config_resolved.yaml", resolved)

    try:
        dataset_path = validate_dataset_available(args, train_config, out_dir)
    except FileNotFoundError as exc:
        summary = {
            "pilot_ran": False,
            "dataset_found": False,
            "dataset": args.dataset,
            "dataset_source": dataset_source_name(args.dataset),
            "error": str(exc),
            "output_dir": str(out_dir),
        }
        write_json(out_dir / "final_summary.json", summary)
        print(str(exc))
        return 2

    if args.dry_run:
        write_json(
            out_dir / "final_summary.json",
            {"pilot_ran": False, "dry_run": True, "dataset_found": True, "dataset_path": str(dataset_path)},
        )
        print(f"Dry run passed. dataset_path={dataset_path}")
        return 0

    set_seeds(args.seed)
    device = torch_device(train_config.device)
    if device.type == "cuda":
        cuda_device = device if device.index is not None else torch.cuda.current_device()
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

    dataset = VQADataset(str(dataset_path), size=args.num_edits, config=train_config)
    if len(dataset) < args.num_edits:
        message = f"Dataset has only {len(dataset)} usable edits after filtering; requested {args.num_edits}."
        (out_dir / "dataset_error.txt").write_text(message + "\n")
        write_json(out_dir / "final_summary.json", {"pilot_ran": False, "dataset_found": True, "error": message})
        print(message)
        return 3

    print("Loading BLIP2 model...")
    model = get_model(train_config).to(device).eval()
    alg = DSCA(model, train_config, lambda: None).to(device).eval()
    if any(param.requires_grad for param in alg.model.parameters()):
        raise RuntimeError("Base VLM has trainable parameters after DSCA initialization.")
    optimizer = torch.optim.Adam(alg.outer_parameters(), lr=train_config.lr)
    tokenizer = alg.model.opt_tokenizer

    metrics_path = out_dir / "metrics_per_step.csv"
    predictions_path = out_dir / "predictions.jsonl"
    diagnostics_path = out_dir / "dsca_diagnostics.jsonl"
    save_metrics_header(metrics_path)
    (out_dir / "predictions.jsonl").write_text("")
    (out_dir / "dsca_diagnostics.jsonl").write_text("")

    first_batch = sample_batch(dataset, dataset[0], train_config.device)
    validate_masks_from_forward(alg.model, clone_batch(first_batch["edit_inner"]))
    base_logits, _ = logits_and_labels(alg.model, clone_batch(first_batch["edit_inner"]))
    edited_logits, _ = logits_and_labels(alg, clone_batch(first_batch["edit_inner"]))
    if not torch.allclose(base_logits, edited_logits):
        raise RuntimeError("Empty repository identity failed before first edit.")

    if base_signature_mode in {"full", "every-step"}:
        base_signature = base_param_signature(alg.model)
    elif base_signature_mode == "cheap":
        base_signature = sampled_base_param_signature(alg.model)
    else:
        base_signature = None
    alg.repository.save(str(out_dir / "repository_step_000.pt"))

    start_time = time.time()
    final_rows: List[Dict[str, Any]] = []
    pass_flags = {
        "base_vlm_params_changed": False,
        "R_k_requires_grad_any": False,
        "duplicate_optimizer_param_groups": False,
        "repository_save_load": True,
        "loss_finite": True,
    }

    for index in range(args.num_edits):
        step = index + 1
        record = dataset[index]
        log_phase(args.phase_timing, step, "sample_batch", "start")
        phase_start = time.time()
        batch = sample_batch(dataset, record, train_config.device)
        log_phase(args.phase_timing, step, "sample_batch", "done", time.time() - phase_start)
        log_phase(args.phase_timing, step, "validate_masks", "start")
        phase_start = time.time()
        validate_masks_from_forward(alg.model, clone_batch(batch["edit_inner"]))
        log_phase(args.phase_timing, step, "validate_masks", "done", time.time() - phase_start)

        step_start = time.time()
        log_phase(args.phase_timing, step, "edit_step", "start")
        phase_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        timeout_handle = None
        profile_this_step = args.profile_edit_step and args.profile_start_step <= step <= args.profile_end_step
        if profile_this_step and args.dump_traceback_on_timeout and args.edit_step_timeout_sec > 0:
            timeout_handle = (out_dir / "dsca_step_timeout_traceback.log").open("a", encoding="utf-8")
            timeout_handle.write(f"\n===== step {step} timeout after {args.edit_step_timeout_sec}s =====\n")
            timeout_handle.flush()
            faulthandler.dump_traceback_later(args.edit_step_timeout_sec, repeat=False, file=timeout_handle)
        try:
            loss_total, _, _, _, info = alg.edit_step(batch, training=True, optimizer=optimizer)
        finally:
            if timeout_handle is not None:
                faulthandler.cancel_dump_traceback_later()
                timeout_handle.close()
        log_phase(args.phase_timing, step, "edit_step", "done", time.time() - phase_start)
        finite_or_raise("loss/dsca_total", float(loss_total.detach().cpu()))
        pass_flags["loss_finite"] = pass_flags["loss_finite"] and math.isfinite(float(loss_total.detach().cpu()))

        r_requires_grad = any(bool(dsam.R.requires_grad) for dsam in alg.repository.dsams)
        duplicate_params = False if args.disable_gradient_diagnostics else duplicate_optimizer_params(optimizer)
        if r_requires_grad:
            raise RuntimeError("At least one DSCA basis R_k requires grad.")
        if duplicate_params:
            raise RuntimeError("Duplicate optimizer parameter groups detected.")

        active_grad_norms = []
        if not args.disable_gradient_diagnostics:
            for dsam in alg.repository.dsams:
                if not dsam.active:
                    continue
                grads = [param.grad.detach().float().norm().item() for param in dsam.parameters() if param.grad is not None]
                if grads:
                    active_grad_norms.append(sum(grads))
        active_grad_mean = float(sum(active_grad_norms) / len(active_grad_norms)) if active_grad_norms else 0.0

        log_phase(args.phase_timing, step, "optimizer_step", "start")
        phase_start = time.time()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        log_phase(args.phase_timing, step, "optimizer_step", "done", time.time() - phase_start)

        if base_signature_mode == "every-step" and base_signature is not None:
            log_phase(args.phase_timing, step, "base_signature", "start")
            phase_start = time.time()
            delta = signature_delta(base_signature, base_param_signature(alg.model))
            log_phase(args.phase_timing, step, "base_signature", "done", time.time() - phase_start)
            if delta != 0.0:
                raise RuntimeError(f"Base VLM parameters changed; signature delta={delta}.")
        elif base_signature_mode == "cheap" and base_signature is not None:
            delta = signature_delta(base_signature, sampled_base_param_signature(alg.model))
            if delta != 0.0:
                raise RuntimeError(f"Sampled base VLM parameters changed; signature delta={delta}.")
        else:
            delta = 0.0

        if step in {10, args.num_edits} and not args.disable_repository_save_load_validation:
            log_phase(args.phase_timing, step, "repository_save_load", "start")
            phase_start = time.time()
            checkpoint_path = out_dir / f"repository_step_{step:03d}.pt"
            alg.repository.save(str(checkpoint_path))
            loaded = alg.repository.load(str(checkpoint_path))
            pass_flags["repository_save_load"] = pass_flags["repository_save_load"] and len(loaded) == len(alg.repository)
            log_phase(args.phase_timing, step, "repository_save_load", "done", time.time() - phase_start)

        sample_scores: Dict[str, float] = {}
        for sample_type, batch_part, prompt, target, image_path in [
            ("rel", batch["edit_inner"], record["prompt"], record["target"], record.get("image_path")),
            (
                "t_gen",
                batch["edit_outer"],
                record.get("rephrase_prompt", record["prompt"]),
                record["target"],
                record.get("image_path"),
            ),
            (
                "m_gen",
                batch["edit_outer_image"],
                record["prompt"],
                record["target"],
                record.get("image_rephrase_path"),
            ),
            (
                "t_loc",
                batch["loc"],
                record.get("locality_prompt"),
                record.get("locality_ground_truth"),
                None,
            ),
            (
                "m_loc",
                batch["loc_image"],
                record.get("multimodal_locality_prompt"),
                record.get("multimodal_locality_ground_truth"),
                record.get("multimodal_locality_image_path"),
            ),
        ]:
            log_phase(args.phase_timing, step, f"evaluate_{sample_type}", "start")
            phase_start = time.time()
            sample_scores[sample_type] = evaluate_case(
                alg.model,
                alg,
                tokenizer,
                batch_part,
                sample_type,
                prompt,
                target,
                image_path,
                predictions_path,
                step,
                record.get("aliases", record.get("target_aliases")),
            )
            log_phase(args.phase_timing, step, f"evaluate_{sample_type}", "done", time.time() - phase_start)
        avg = mean_available(sample_scores.values())

        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        else:
            peak_mb = 0.0
        time_per_edit = time.time() - step_start

        route_weights = getattr(alg, "_last_route_weights", None)
        if route_weights is not None and route_weights.numel():
            candidate_counts = (route_weights.detach().cpu() > 0).sum(dim=1).tolist()
            route_weight_summary = {
                "mean": float(route_weights.detach().float().mean().cpu()),
                "max": float(route_weights.detach().float().max().cpu()),
                "l1": float(route_weights.detach().float().abs().mean().cpu()),
            }
        else:
            candidate_counts = []
            route_weight_summary = {"mean": 0.0, "max": 0.0, "l1": 0.0}

        diagnostics = {
            "step": step,
            "num_clusters": len(alg.repository),
            "num_active_dsams": alg.repository.num_active(),
            "candidate_counts": candidate_counts,
            "route_weights_summary": route_weight_summary,
            "residual_norm": float(info.get("dsca/residual_norm_mean", 0.0)),
            "new_clusters_created": float(info.get("dsca/new_clusters_created", 0.0)),
            "new_dsams_activated": float(info.get("dsca/new_dsams_activated", 0.0)),
            "optimizer_param_groups": len(optimizer.param_groups),
            "duplicate_param_group_detected": duplicate_params,
            "R_k_requires_grad_any": r_requires_grad,
            "active_dsam_grad_norm_mean": active_grad_mean,
        }
        append_jsonl(diagnostics_path, diagnostics)

        row = {
            "step": step,
            "edit_id": str(record.get("instance_id", index)),
            "rel": sample_scores["rel"],
            "t_gen": sample_scores["t_gen"],
            "m_gen": sample_scores["m_gen"],
            "t_loc": sample_scores["t_loc"],
            "m_loc": sample_scores["m_loc"],
            "avg": avg,
            "num_clusters": len(alg.repository),
            "num_active_dsams": alg.repository.num_active(),
            "mean_subspace_overlap": float(alg.repository.mean_subspace_overlap().detach().cpu()),
            "avg_candidates": float(info.get("dsca/num_candidates_mean", 0.0)),
            "residual_norm_mean": float(info.get("dsca/residual_norm_mean", 0.0)),
            "route_weight_l1_replay": float(info.get("loss/dsca_sparse", 0.0)),
            "base_param_delta_norm": delta,
            "peak_gpu_memory_mb": peak_mb,
            "time_per_edit_sec": time_per_edit,
        }
        append_metrics(metrics_path, row)
        final_rows.append(row)
        print(row, flush=True)
        if args.max_profile_steps is not None and step >= args.max_profile_steps:
            break

    total_runtime = time.time() - start_time
    final_base_delta = 0.0
    if base_signature_mode == "full" and base_signature is not None:
        final_base_delta = signature_delta(base_signature, base_param_signature(alg.model))
        if final_base_delta != 0.0:
            pass_flags["base_vlm_params_changed"] = True
            raise RuntimeError(f"Base VLM parameters changed; signature delta={final_base_delta}.")
    elif base_signature_mode == "cheap" and base_signature is not None:
        final_base_delta = signature_delta(base_signature, sampled_base_param_signature(alg.model))
        if final_base_delta != 0.0:
            pass_flags["base_vlm_params_changed"] = True
            raise RuntimeError(f"Sampled base VLM parameters changed; signature delta={final_base_delta}.")

    final = final_rows[-1]
    final_summary = {
        "pilot_ran": True,
        "dataset_found": True,
        "dataset": args.dataset,
        "dataset_source": dataset_source_name(args.dataset),
        "dataset_path": str(dataset_path),
        "num_edits": args.num_edits,
        "completed_edits": len(final_rows),
        "base_signature_mode": base_signature_mode,
        "diagnostic_flags": {
            "disable_pca_refine": args.disable_pca_refine,
            "disable_basis_initialization": args.disable_basis_initialization,
            "disable_cdistill": args.disable_cdistill,
            "disable_align": args.disable_align,
            "disable_sparse": args.disable_sparse,
            "disable_task_loss": args.disable_task_loss,
            "disable_repository_save_load_validation": args.disable_repository_save_load_validation,
            "disable_gradient_diagnostics": args.disable_gradient_diagnostics,
        },
        "final_metrics": {key: final[key] for key in METRIC_COLUMNS + ["avg"]},
        "mean_metrics": {key: mean_available(row[key] for row in final_rows) for key in METRIC_COLUMNS + ["avg"]},
        "final_repository_size": len(alg.repository),
        "final_active_dsam_count": alg.repository.num_active(),
        "mean_subspace_overlap": float(alg.repository.mean_subspace_overlap().detach().cpu()),
        "total_runtime_sec": total_runtime,
        "time_per_edit_sec": total_runtime / max(args.num_edits, 1),
        "peak_gpu_memory_mb": final["peak_gpu_memory_mb"],
        "final_base_param_delta_norm": final_base_delta,
        "pass_fail_flags": pass_flags,
    }
    write_json(out_dir / "final_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    if args.model != "blip2":
        raise NotImplementedError("This pilot supports BLIP2-OPT only.")
    return run_pilot(args)


if __name__ == "__main__":
    raise SystemExit(main())
