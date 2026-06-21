#!/usr/bin/env python3
"""Shared helpers for DSCA MedMKEB failure diagnostics."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def ensure_offline_env() -> None:
    os.environ.setdefault("HF_HOME", "/remote-home/wangbomin/hugging_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
    return torch.device(str(device))


def normalize_medical_answer(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"^(the\s+answer\s+is|answer\s*:|it\s+is)\s+", "", text)
    text = re.sub(r"[^\w\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def alias_list(aliases: Any) -> List[str]:
    if aliases is None:
        return []
    if isinstance(aliases, str):
        return [aliases]
    if isinstance(aliases, (list, tuple)):
        return [str(item) for item in aliases if item is not None]
    return [str(aliases)]


def answer_fields(base_prediction: Any, edited_prediction: Any, target: Any, aliases: Any = None) -> Dict[str, Any]:
    normalized_target = normalize_medical_answer(target)
    normalized_base = normalize_medical_answer(base_prediction)
    normalized_edited = normalize_medical_answer(edited_prediction)
    alias_norms = [normalize_medical_answer(item) for item in alias_list(aliases)]
    alias_norms = [item for item in alias_norms if item]
    target_norms = [item for item in [normalized_target] + alias_norms if item]
    return {
        "normalized_target": normalized_target,
        "normalized_base_prediction": normalized_base,
        "normalized_edited_prediction": normalized_edited,
        "base_contains_target": bool(normalized_base and any(item in normalized_base for item in target_norms)),
        "edited_contains_target": bool(normalized_edited and any(item in normalized_edited for item in target_norms)),
        "exact_match_raw": bool(edited_prediction is not None and target is not None and str(edited_prediction).strip() == str(target).strip()),
        "exact_match_normalized": bool(normalized_edited and normalized_target and normalized_edited == normalized_target),
        "contains_target": bool(normalized_edited and any(item in normalized_edited for item in target_norms)),
        "alias_exact_match": bool(normalized_edited and any(normalized_edited == item for item in alias_norms)),
        "alias_contains_match": bool(normalized_edited and any(item in normalized_edited for item in alias_norms)),
        "edited_equals_base": bool((normalized_edited or normalized_base) and normalized_edited == normalized_base),
    }


def token_count(tokenizer: Any, text: Any) -> int:
    if text is None:
        return 0
    try:
        return int(len(tokenizer.encode(str(text), add_special_tokens=False)))
    except Exception:
        return int(len(str(text).split()))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return to_jsonable(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def resolve_dataset_path(dataset: str, root: Path, explicit_path: Optional[Path] = None) -> Path:
    if explicit_path is not None:
        path = explicit_path if explicit_path.is_absolute() else root / explicit_path
        if path.is_file():
            return path
        raise FileNotFoundError(f"Explicit dataset JSON not found: {path}")
    name = dataset.lower()
    candidates = []
    if name == "medmkeb":
        candidates = [
            root / "datasets" / "MedMKEB" / "eval.json",
            root / "datasets" / "MEDMKEB" / "eval.json",
            root / "datasets" / "medmkeb" / "eval.json",
        ]
    elif name == "vlkeb":
        candidates = [root / "datasets" / "eval.json", root / "datasets" / "VLKEB" / "eval.json"]
    else:
        candidates = [root / "datasets" / "E-VQA" / "eval.json", root / "datasets" / "EVQA" / "eval.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Dataset JSON not found; checked: " + ", ".join(str(item) for item in candidates))


def effective_image_root(image_root: Path, dataset_path: Path) -> Path:
    image_root = image_root.resolve()
    try:
        records = json.loads(dataset_path.read_text(errors="replace"))
        first = records[0] if isinstance(records, list) and records else {}
    except Exception:
        first = {}
    image_values = [
        str(first.get(key, ""))
        for key in (
            "image_path",
            "image_rephrase_path",
            "multimodal_locality_image_path",
            "image",
            "image_rephrase",
            "m_loc",
        )
        if first.get(key)
    ]
    if image_root.name == "images" and any(value.startswith("images/") for value in image_values):
        return image_root.parent
    return image_root


def configure_hparams(config: Any, args: Any, dataset_path: Path) -> Any:
    config.device = normalize_device(args.device)
    if hasattr(args, "rank") and args.rank is not None:
        config.dsca_rank = int(args.rank)
    if hasattr(args, "min_samples") and args.min_samples is not None:
        config.dsca_min_samples = int(args.min_samples)
    if hasattr(args, "refine_interval") and args.refine_interval is not None:
        config.dsca_refine_interval = int(args.refine_interval)
    if hasattr(args, "learning_rate") and args.learning_rate is not None:
        config.lr = float(args.learning_rate)
    if hasattr(args, "residual_scale") and args.residual_scale is not None:
        config.dsca_residual_scale = float(args.residual_scale)
    if hasattr(args, "dsca_generation_mode") and args.dsca_generation_mode:
        config.dsca_generation_mode = str(args.dsca_generation_mode)
    if hasattr(args, "residual_apply_mask") and args.residual_apply_mask:
        config.dsca_generation_residual_apply_mask = str(args.residual_apply_mask)
        config.dsca_generation_reuse_prefill_route = str(args.dsca_generation_mode) == "cache_reuse_route"
    image_root = effective_image_root(Path(args.image_root), dataset_path)
    config.coco_image = str(image_root)
    config.rephrase_image = str(image_root)
    config.dsca_freeze_vlm = True
    config.dsca_require_masks = True
    return config


def load_dataset_and_model(args: Any, num_samples: Optional[int] = None, load_repo: bool = False):
    ensure_offline_env()
    from easyeditor.dataset import VQADataset
    from easyeditor.models.dsca.dsca_hparams import DSCAMultimodalHparams
    from easyeditor.trainer.algs.dsca import DSCA
    from easyeditor.trainer.models import get_model

    root = Path.cwd()
    dataset_path = resolve_dataset_path(args.dataset, root, getattr(args, "dataset_path", None))
    config = DSCAMultimodalHparams.from_hparams(args.hparams)
    config = configure_hparams(config, args, dataset_path)
    if hasattr(args, "task_only") and args.task_only:
        config.dsca_lambda_align = 0.0
        config.dsca_lambda_distill = 0.0
        config.dsca_lambda_sparse = 0.0
        config.dsca_task_weight = 1.0
    device = torch_device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    dataset = VQADataset(str(dataset_path), size=num_samples, config=config)
    model = get_model(config).to(device).eval()
    alg = DSCA(model, config, lambda: None).to(device).eval()
    if load_repo:
        run_dir = Path(args.run_dir)
        checkpoint = run_dir / f"repository_step_{getattr(args, 'repository_step', 20):03d}.pt"
        if not checkpoint.exists():
            checkpoint = run_dir / "repository_step_020.pt"
        from easyeditor.trainer.algs.dsca_utils import DSCAConceptRepository

        loaded = DSCAConceptRepository.load(str(checkpoint), map_location=str(device))
        alg.repository.load_state_dict(loaded.state_dict())
        alg.repository.to(device)
    tokenizer = (
        getattr(alg.model, "opt_tokenizer", None)
        or getattr(alg.model, "llama_tokenizer", None)
        or getattr(alg.model, "llava_tokenizer", None)
    )
    return dataset, model, alg, tokenizer, config, dataset_path


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


def collate_record(dataset: Any, record: Dict[str, Any]) -> Dict[str, Any]:
    return dataset.collate_fn([record])


def tensor_logits(outputs: Any) -> torch.Tensor:
    return outputs if isinstance(outputs, torch.Tensor) else outputs.logits


def aligned_logits_and_labels(outputs: Any, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = tensor_logits(outputs)
    labels = batch["labels"]
    shift_logits = logits[:, :-1]
    shift_logits = shift_logits[:, -labels.shape[1] :]
    if shift_logits.shape[:2] != labels.shape:
        raise RuntimeError(f"Logits/labels shape mismatch: {tuple(shift_logits.shape)} vs {tuple(labels.shape)}")
    return shift_logits, labels


def target_nll_from_outputs(outputs: Any, batch: Dict[str, Any]) -> Dict[str, Any]:
    logits, labels = aligned_logits_and_labels(outputs, batch)
    mask = labels != -100
    count = int(mask.sum().item())
    if count == 0:
        return {"target_nll": float("nan"), "avg_target_logprob": float("nan"), "target_token_count": 0}
    safe_labels = labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    total_logprob = gathered.masked_select(mask).sum()
    nll = -total_logprob / count
    first_positions = mask.nonzero(as_tuple=False)
    first_rank = None
    if first_positions.numel() > 0:
        b, t = first_positions[0].tolist()
        target_id = int(labels[b, t].item())
        order = torch.argsort(logits[b, t].float(), descending=True)
        rank_pos = (order == target_id).nonzero(as_tuple=False)
        first_rank = int(rank_pos[0].item() + 1) if rank_pos.numel() else None
    return {
        "target_nll": float(nll.detach().cpu()),
        "avg_target_logprob": float((total_logprob / count).detach().cpu()),
        "target_token_count": count,
        "first_target_token_rank": first_rank,
    }


def decode_argmax_on_labels(tokenizer: Any, outputs: Any, batch: Dict[str, Any]) -> str:
    logits, labels = aligned_logits_and_labels(outputs, batch)
    mask = labels != -100
    pred = logits.argmax(dim=-1).masked_fill(~mask, -100)
    ids = [int(x) for x in pred.view(-1).detach().cpu().tolist() if int(x) >= 0]
    try:
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except Exception:
        return " ".join(str(x) for x in ids)


def labels_mask_report(batch: Dict[str, Any], outputs: Any, tokenizer: Any, target: Any) -> Dict[str, Any]:
    labels = batch["labels"]
    result: Dict[str, Any] = {
        "target_token_count": token_count(tokenizer, target),
        "labels_not_ignore_count": int((labels != -100).sum().item()),
    }
    for name in ("attention_mask", "vision_mask", "prompt_mask", "answer_mask"):
        value = getattr(outputs, name, None)
        result[f"{name}_sum"] = int(value.bool().sum().item()) if value is not None else None
    answer = getattr(outputs, "answer_mask", None)
    if answer is not None:
        label_mask = labels != -100
        if answer.shape[1] >= label_mask.shape[1]:
            answer_tail = answer[:, -label_mask.shape[1] :].bool()
            result["labels_align_answer_mask"] = bool(torch.equal(answer_tail.cpu(), label_mask.cpu()))
            result["answer_mask_tail_sum"] = int(answer_tail.sum().item())
        else:
            result["labels_align_answer_mask"] = False
            result["answer_mask_tail_sum"] = None
    else:
        result["labels_align_answer_mask"] = False
        result["answer_mask_tail_sum"] = None
    return result


def active_ids(repository: Any) -> List[int]:
    return [idx for idx in range(len(repository)) if bool(repository.active[idx].item())]


def temporarily_force_route(alg: Any, cluster_id: Optional[int]):
    class _Ctx:
        def __enter__(self_inner):
            self_inner.old_tau = alg.tau_visual
            self_inner.old_active = alg.repository.active.clone()
            alg.tau_visual = -1.0e9
            if cluster_id is not None and 0 <= cluster_id < len(alg.repository):
                forced = torch.zeros_like(alg.repository.active)
                forced[cluster_id] = bool(self_inner.old_active[cluster_id].item())
                alg.repository.active.copy_(forced)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            alg.tau_visual = self_inner.old_tau
            if alg.repository.active.shape == self_inner.old_active.shape:
                alg.repository.active.copy_(self_inner.old_active)
            else:
                restored = alg.repository.active.clone()
                limit = min(restored.numel(), self_inner.old_active.numel())
                if limit:
                    restored[:limit] = self_inner.old_active[:limit].to(restored.device)
                alg.repository.active.copy_(restored)

    return _Ctx()
