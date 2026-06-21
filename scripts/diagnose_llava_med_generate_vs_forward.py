#!/usr/bin/env python3
"""Strict LLaVA-Med generate-vs-forward diagnostic for DSCA overfit states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from dsca_medmkeb_diag_common import answer_fields, append_jsonl, to_jsonable, write_json
from easyeditor.trainer.algs.dsca_utils import DSCAConceptRepository, DSCAContext, dsca_intervention_context
from smoke_llava_med_dsca_generation import (
    prepare_generation_inputs,
    reset_generation_debug_state,
    resolve_image_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llava-med", choices=["llava-med"])
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--hparams", default="hparams/DSCA/llava_med.yaml")
    parser.add_argument("--training-hparams", default="hparams/TRAINING/DSCA/llava_med_stage1_smoke.yaml")
    parser.add_argument("--overfit-dir", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_env() -> None:
    os.environ.setdefault("HF_HOME", "/remote-home/wangbomin/hugging_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_arg(text: str) -> Any:
    if text == "cuda":
        return "cuda"
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        return int(suffix) if suffix.isdigit() else text
    return int(text) if text.isdigit() else text


def torch_device(value: Any) -> torch.device:
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    return torch.device(str(value))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(errors="replace")) if path.is_file() else {}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
    else:
        keys = list(fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key)) for key in keys})


def load_records(dataset_path: Path) -> List[Dict[str, Any]]:
    records = json.loads(dataset_path.read_text(errors="replace"))
    if not isinstance(records, list):
        raise RuntimeError(f"Dataset JSON root must be a list: {dataset_path}")
    return [row for row in records if isinstance(row, dict)]


def prompt_current_pilot(record: Dict[str, Any]) -> str:
    return "Question: {} Short answer: ".format(record.get("src", ""))


def target_for(record: Dict[str, Any]) -> str:
    return str(record.get("alt", record.get("target", "")))


def make_sample(model: Any, record: Dict[str, Any], image_root: Path, prompt: str, target: str) -> Dict[str, Any]:
    tokenizer = model.llava_tokenizer
    labels = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return {
        "image_path": [str(resolve_image_path(image_root, str(record["image"])))],
        "prompt": [prompt],
        "target": [target],
        "text_input": [prompt + target],
        "labels": labels,
        "prompts_len": [len(tokenizer(prompt, add_special_tokens=False).input_ids)],
    }


def load_model_alg(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    ensure_env()
    from easyeditor.trainer.algs.dsca import DSCA
    from easyeditor.trainer.models import get_model
    from easyeditor.trainer.training_hparams.dsca_multimodal_training_hparams import (
        DSCAMultimodalTrainingHparams,
    )

    config = DSCAMultimodalTrainingHparams.from_hparams(args.training_hparams or args.hparams)
    config.device = device_arg(args.device)
    config.coco_image = str(args.image_root)
    config.rephrase_image = str(args.image_root)
    config.dsca_generation_mode = "cache_reuse_route"
    config.dsca_generation_residual_apply_mask = "current_token"
    config.dsca_generation_reuse_prefill_route = True
    device = torch_device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    model = get_model(config).to(device).eval()
    alg = DSCA(model, config, lambda: None).to(device).eval()
    return model, alg, config


def load_repository(alg: Any, overfit_dir: Path, device: torch.device) -> Tuple[Path, Dict[str, Any]]:
    candidates = [
        overfit_dir / "final_repository.pt",
        overfit_dir / "repository_after_overfit.pt",
    ]
    repo_path = next((path for path in candidates if path.is_file()), None)
    if repo_path is None:
        raise FileNotFoundError("No final DSCA repository found in " + str(overfit_dir))
    loaded = DSCAConceptRepository.load(str(repo_path), map_location=str(device))
    alg.repository = loaded.to(device)
    summary = read_json(overfit_dir / "final_summary.json") or read_json(overfit_dir / "overfit_summary.json")
    return repo_path, summary


def decode_ids(tokenizer: Any, ids: Sequence[int], skip_special_tokens: bool = True, clean: bool = True) -> str:
    try:
        return tokenizer.decode(
            [int(item) for item in ids],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean,
        ).strip()
    except TypeError:
        return tokenizer.decode([int(item) for item in ids], skip_special_tokens=skip_special_tokens).strip()
    except Exception:
        return " ".join(str(int(item)) for item in ids)


def batch_decode_one(tokenizer: Any, tensor: torch.Tensor, skip_special_tokens: bool = True, clean: bool = True) -> str:
    ids = [int(item) for item in tensor.detach().view(-1).cpu().tolist()]
    return decode_ids(tokenizer, ids, skip_special_tokens=skip_special_tokens, clean=clean)


def target_token_ids_for_prompt(model: Any, prompt: str, target: str) -> List[int]:
    prompt_text = model._conversation_prompt(prompt, None)
    full_text = model._conversation_prompt(prompt, target)
    prompt_ids = model.tokenizer_image_token(
        prompt_text, model.llava_tokenizer, model.IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    full_ids = model.tokenizer_image_token(
        full_text, model.llava_tokenizer, model.IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    target_ids = full_ids[int(prompt_ids.numel()) :].detach().cpu().tolist()
    return [int(item) for item in target_ids if int(item) != int(model.IMAGE_TOKEN_INDEX)]


def rank_of_token(logits: torch.Tensor, token_id: int) -> int:
    order = torch.argsort(logits.float(), descending=True)
    found = (order == int(token_id)).nonzero(as_tuple=False)
    return int(found[0].item() + 1) if found.numel() else int(logits.numel() + 1)


def token_logprob(logits: torch.Tensor, token_id: int) -> float:
    return float(torch.log_softmax(logits.float(), dim=-1)[int(token_id)].detach().cpu())


def topk_tokens(tokenizer: Any, logits: torch.Tensor, k: int = 20) -> List[Dict[str, Any]]:
    probs, ids = torch.topk(torch.log_softmax(logits.float(), dim=-1), k=min(k, int(logits.numel())))
    rows: List[Dict[str, Any]] = []
    for value, idx in zip(probs.detach().cpu().tolist(), ids.detach().cpu().tolist()):
        rows.append(
            {
                "id": int(idx),
                "token": decode_ids(tokenizer, [int(idx)], skip_special_tokens=False),
                "logprob": float(value),
            }
        )
    return rows


def temporary_dsam_residual_scale(alg: Any, value: Optional[float]):
    if value is None:
        return nullcontext()

    class _Ctx:
        def __enter__(self_inner):
            self_inner.old_repo_scale = float(getattr(alg.repository, "residual_scale", 1.0))
            self_inner.old_values = [float(dsam.residual_scale) for dsam in alg.repository.dsams]
            alg.repository.residual_scale = float(value)
            for dsam in alg.repository.dsams:
                dsam.residual_scale = float(value)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            alg.repository.residual_scale = self_inner.old_repo_scale
            for dsam, old in zip(alg.repository.dsams, self_inner.old_values):
                dsam.residual_scale = old
            return False

    return _Ctx()


def annotate_events(
    events: List[Dict[str, Any]],
    input_len: int,
    sequence_ids: Optional[Sequence[int]] = None,
    score_step0_phase: str = "prefill",
) -> None:
    for event in events:
        phase = event.get("phase")
        event["sequence_length"] = event.get("hidden_shape", [None, None])[1] if event.get("hidden_shape") else None
        event["corresponds_to_generate_scores_step_0"] = bool(phase == score_step0_phase)
        event["residual_applied_before_logits_computation"] = bool(not event.get("skipped") and event.get("residual_norm") is not None)
        if sequence_ids:
            if phase == "prefill":
                event["current_token_id"] = int(sequence_ids[min(input_len - 1, len(sequence_ids) - 1)])
            else:
                idx = input_len + max(int(event.get("decode_call_index", 1)) - 2, 0)
                event["current_token_id"] = int(sequence_ids[idx]) if 0 <= idx < len(sequence_ids) else None
        else:
            event["current_token_id"] = None


def dsca_context(
    alg: Any,
    masks: Dict[str, torch.Tensor],
    events: List[Dict[str, Any]],
    call_label: str,
    use_cache: bool,
    force_route_ids: Optional[List[int]],
) -> DSCAContext:
    return DSCAContext(
        batch=masks,
        is_generation=True,
        debug_events=events,
        sample_id=0,
        call_label=call_label,
        generation_mode="cache_reuse_route",
        generation_reuse_prefill_route=True,
        generation_use_cache=use_cache,
        force_route_ids=force_route_ids,
        residual_apply_mask_mode="current_token",
        extend_generation_masks=(not use_cache) or True,
    )


def run_forward_logits(
    model: Any,
    alg: Any,
    sample: Dict[str, Any],
    call_label: str,
    dsca_enabled: bool,
    force_route_ids: Optional[List[int]],
    residual_scale: Optional[float],
) -> Dict[str, Any]:
    input_ids, image_tensor, masks = prepare_generation_inputs(model, sample)
    events: List[Dict[str, Any]] = []
    managers = []
    if dsca_enabled:
        reset_generation_debug_state(alg)
        managers.append(dsca_intervention_context(alg, dsca_context(alg, masks, events, call_label, False, force_route_ids)))
        managers.append(temporary_dsam_residual_scale(alg, residual_scale))
    with ExitStack() as stack:
        for manager in managers:
            stack.enter_context(manager)
        with torch.inference_mode():
            outputs = model.llava_model(
                input_ids=input_ids,
                images=image_tensor,
                attention_mask=torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device),
                return_dict=True,
                use_cache=False,
            )
    sequence_ids = [int(item) for item in input_ids[0].detach().cpu().tolist()]
    annotate_events(events, int(input_ids.shape[1]), sequence_ids=sequence_ids)
    return {
        "logits": outputs.logits[0, -1].detach().float().cpu(),
        "input_ids": input_ids.detach().cpu(),
        "events": events,
        "masks": {key: value.detach().cpu() for key, value in masks.items()},
    }


def generation_config_payload(model: Any) -> Dict[str, Any]:
    config = getattr(model.llava_model, "generation_config", None)
    if config is None:
        return {}
    try:
        return config.to_dict()
    except Exception:
        return dict(getattr(config, "__dict__", {}))


def run_generate(
    model: Any,
    alg: Any,
    sample: Dict[str, Any],
    call_label: str,
    force_route_ids: Optional[List[int]],
    residual_scale: Optional[float],
    max_new_tokens: int,
    use_cache: bool,
) -> Dict[str, Any]:
    input_ids, image_tensor, masks = prepare_generation_inputs(model, sample)
    events: List[Dict[str, Any]] = []
    reset_generation_debug_state(alg)
    with ExitStack() as stack:
        stack.enter_context(
            dsca_intervention_context(alg, dsca_context(alg, masks, events, call_label, use_cache, force_route_ids))
        )
        stack.enter_context(temporary_dsam_residual_scale(alg, residual_scale))
        with torch.inference_mode():
            output = model.llava_model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                pad_token_id=model.llava_tokenizer.pad_token_id,
                eos_token_id=model.llava_tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
    sequences = output.sequences.detach().cpu()
    input_len = int(input_ids.shape[1])
    sequence_includes_prompt = bool(
        sequences.shape[1] >= input_len and torch.equal(sequences[0, :input_len], input_ids[0].detach().cpu())
    )
    generated_ids = sequences[:, input_len:] if sequence_includes_prompt else sequences
    sequence_ids = [int(item) for item in sequences[0].detach().cpu().tolist()]
    annotate_events(events, input_len, sequence_ids=sequence_ids)
    scores0 = output.scores[0][0].detach().float().cpu() if output.scores else None
    return {
        "scores0": scores0,
        "sequences": sequences,
        "input_ids": input_ids.detach().cpu(),
        "generated_ids": generated_ids,
        "input_len": input_len,
        "sequence_includes_prompt": sequence_includes_prompt,
        "full_decoded_text": batch_decode_one(model.llava_tokenizer, sequences[0], skip_special_tokens=True),
        "generated_suffix_text": batch_decode_one(model.llava_tokenizer, generated_ids[0], skip_special_tokens=True),
        "full_decoded_text_keep_special": batch_decode_one(model.llava_tokenizer, sequences[0], skip_special_tokens=False),
        "generated_suffix_keep_special": batch_decode_one(model.llava_tokenizer, generated_ids[0], skip_special_tokens=False),
        "generated_suffix_no_cleanup": batch_decode_one(
            model.llava_tokenizer, generated_ids[0], skip_special_tokens=True, clean=False
        ),
        "events": events,
        "generation_config": generation_config_payload(model),
        "logits_processors": [],
        "stopping_criteria": [f"max_new_tokens={max_new_tokens}"],
    }


def compare_logits(
    tokenizer: Any,
    name: str,
    lhs: torch.Tensor,
    rhs: Optional[torch.Tensor],
    target_first_token_id: int,
) -> Dict[str, Any]:
    if rhs is None:
        return {
            "comparison": name,
            "allclose": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
    diff = (lhs.float() - rhs.float()).abs()
    return {
        "comparison": name,
        "allclose": bool(torch.allclose(lhs.float(), rhs.float(), rtol=1.0e-4, atol=1.0e-4)),
        "max_abs_diff": float(diff.max().detach().cpu()),
        "mean_abs_diff": float(diff.mean().detach().cpu()),
        "direct_argmax_token_id": int(lhs.argmax().item()),
        "direct_argmax_token": decode_ids(tokenizer, [int(lhs.argmax().item())], skip_special_tokens=False),
        "generate_scores_argmax_token_id": int(rhs.argmax().item()),
        "generate_scores_argmax_token": decode_ids(tokenizer, [int(rhs.argmax().item())], skip_special_tokens=False),
        "target_first_token_id": int(target_first_token_id),
        "target_first_token": decode_ids(tokenizer, [int(target_first_token_id)], skip_special_tokens=False),
        "direct_target_rank": rank_of_token(lhs, target_first_token_id),
        "generate_scores_target_rank": rank_of_token(rhs, target_first_token_id),
        "direct_target_logprob": token_logprob(lhs, target_first_token_id),
        "generate_scores_target_logprob": token_logprob(rhs, target_first_token_id),
    }


def append_events(path: Path, events: Iterable[Dict[str, Any]]) -> None:
    for event in events:
        append_jsonl(path, event)


def generated_stop_reason(tokenizer: Any, ids: Sequence[int]) -> str:
    eos = tokenizer.eos_token_id
    if eos is not None and any(int(item) == int(eos) for item in ids):
        return "eos"
    return "max_new_tokens"


def run_manual_greedy(
    model: Any,
    alg: Any,
    sample: Dict[str, Any],
    force_route_ids: Optional[List[int]],
    residual_scale: Optional[float],
    target_first_token_id: int,
    max_new_tokens: int,
    event_path: Path,
) -> Tuple[List[Dict[str, Any]], str]:
    input_ids, image_tensor, masks = prepare_generation_inputs(model, sample)
    original_len = int(input_ids.shape[1])
    cur_ids = input_ids.clone()
    rows: List[Dict[str, Any]] = []
    for step in range(max_new_tokens):
        events: List[Dict[str, Any]] = []
        reset_generation_debug_state(alg)
        with ExitStack() as stack:
            stack.enter_context(
                dsca_intervention_context(
                    alg,
                    dsca_context(alg, masks, events, f"manual_greedy_step_{step:02d}", False, force_route_ids),
                )
            )
            stack.enter_context(temporary_dsam_residual_scale(alg, residual_scale))
            with torch.inference_mode():
                outputs = model.llava_model(
                    input_ids=cur_ids,
                    images=image_tensor,
                    attention_mask=torch.ones_like(cur_ids, dtype=torch.long, device=cur_ids.device),
                    return_dict=True,
                    use_cache=False,
                )
        logits = outputs.logits[0, -1].detach().float().cpu()
        next_id = int(logits.argmax().item())
        cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]], device=cur_ids.device, dtype=cur_ids.dtype)], dim=1)
        sequence_ids = [int(item) for item in cur_ids[0].detach().cpu().tolist()]
        annotate_events(events, original_len, sequence_ids=sequence_ids)
        append_events(event_path, events)
        generated_ids = [int(item) for item in cur_ids[0, original_len:].detach().cpu().tolist()]
        rows.append(
            {
                "step": step,
                "next_token_id": next_id,
                "next_token": decode_ids(model.llava_tokenizer, [next_id], skip_special_tokens=False),
                "target_first_token_rank": rank_of_token(logits, target_first_token_id),
                "target_first_token_logprob": token_logprob(logits, target_first_token_id),
                "generated_ids": json.dumps(generated_ids),
                "generated_text_so_far": decode_ids(model.llava_tokenizer, generated_ids, skip_special_tokens=True),
                "hook_event_count": len(events),
                "residual_norm": max(
                    [float(event.get("residual_norm") or 0.0) for event in events] or [0.0]
                ),
                "active_route_ids": json.dumps(
                    sorted(
                        {
                            int(idx)
                            for event in events
                            for idx in (event.get("active_candidate_ids") or [])
                        }
                    )
                ),
            }
        )
        eos = model.llava_tokenizer.eos_token_id
        if eos is not None and next_id == int(eos):
            break
    final_ids = [int(item) for item in cur_ids[0, original_len:].detach().cpu().tolist()]
    return rows, decode_ids(model.llava_tokenizer, final_ids, skip_special_tokens=True)


def prompt_variants(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    question = str(record.get("src", record.get("prompt", record.get("question", ""))))
    return [
        ("current_pilot", prompt_current_pilot(record)),
        ("official_llava_med_conversation", question),
        ("short_answer_instruction", question + "\nAnswer with a short medical phrase only."),
        ("target_prefix", question + "\nThe answer is"),
        ("question_only", question),
    ]


def template_rows(
    model: Any,
    alg: Any,
    record: Dict[str, Any],
    image_root: Path,
    target: str,
    assigned_cluster_id: int,
    residual_scale: Optional[float],
    max_new_tokens: int,
    event_path: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    force_route_ids = [int(assigned_cluster_id)] if assigned_cluster_id is not None else None
    for name, prompt in prompt_variants(record):
        sample = make_sample(model, record, image_root, prompt, target)
        target_ids = target_token_ids_for_prompt(model, prompt, target)
        first_id = target_ids[0] if target_ids else 0
        direct = run_forward_logits(
            model,
            alg,
            sample,
            f"template_direct_{name}",
            dsca_enabled=True,
            force_route_ids=force_route_ids,
            residual_scale=residual_scale,
        )
        gen = run_generate(
            model,
            alg,
            sample,
            f"template_generate_{name}",
            force_route_ids=force_route_ids,
            residual_scale=residual_scale,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        append_events(event_path, direct["events"])
        append_events(event_path, gen["events"])
        fields = answer_fields(None, gen["generated_suffix_text"], target)
        generated_ids = [int(item) for item in gen["generated_ids"][0].detach().cpu().tolist()]
        rows.append(
            {
                "template": name,
                "prompt": prompt,
                "direct_target_rank": rank_of_token(direct["logits"], first_id),
                "direct_argmax_token": decode_ids(model.llava_tokenizer, [int(direct["logits"].argmax().item())], False),
                "generate_first_token": decode_ids(
                    model.llava_tokenizer,
                    [int(gen["scores0"].argmax().item())] if gen["scores0"] is not None else [],
                    False,
                ),
                "generate_scores_target_rank": rank_of_token(gen["scores0"], first_id) if gen["scores0"] is not None else None,
                "generated_suffix_text": gen["generated_suffix_text"],
                "full_decoded_text": gen["full_decoded_text"],
                "contains_target": bool(fields["contains_target"]),
                "exact_match": bool(fields["exact_match_normalized"]),
                "generated_length": len(generated_ids),
                "stop_reason": generated_stop_reason(model.llava_tokenizer, generated_ids),
            }
        )
    return rows


def markdown_top20(
    tokenizer: Any,
    direct_logits: torch.Tensor,
    generate_logits: Optional[torch.Tensor],
    target_first_token_id: int,
) -> str:
    lines = [
        "# Top-20 Direct vs Generate Tokens",
        "",
        f"- target first token id: `{target_first_token_id}`",
        f"- target first token: `{decode_ids(tokenizer, [target_first_token_id], skip_special_tokens=False)}`",
        "",
        "## Direct Edited Forward",
        "",
        "| rank | token id | token | logprob |",
        "|---:|---:|---|---:|",
    ]
    for idx, row in enumerate(topk_tokens(tokenizer, direct_logits, 20), start=1):
        token = str(row["token"]).replace("|", "\\|")
        lines.append(f"| {idx} | {row['id']} | `{token}` | {row['logprob']:.6f} |")
    lines.extend(["", "## Generate Scores[0]", "", "| rank | token id | token | logprob |", "|---:|---:|---|---:|"])
    if generate_logits is not None:
        for idx, row in enumerate(topk_tokens(tokenizer, generate_logits, 20), start=1):
            token = str(row["token"]).replace("|", "\\|")
            lines.append(f"| {idx} | {row['id']} | `{token}` | {row['logprob']:.6f} |")
    return "\n".join(lines) + "\n"


def root_cause(summary: Dict[str, Any]) -> str:
    if summary.get("slicing_bug_detected"):
        return "decoding/slicing bug"
    direct_rank = summary.get("direct_target_rank")
    generate_rank = summary.get("generate_scores_target_rank")
    rank_gap = abs(int(direct_rank or 0) - int(generate_rank or 0))
    same_argmax = summary.get("direct_argmax_token_id") == summary.get("generate_scores_argmax_token_id")
    if same_argmax and direct_rank and generate_rank and int(direct_rank) > 1000 and int(generate_rank) > 1000:
        return "direct rank diagnostic used different context"
    if summary.get("direct_target_rank") == 1 and summary.get("generate_scores_target_rank") != 1:
        return "generation logits mismatch"
    if summary.get("template_bug_detected"):
        return "prompt template mismatch"
    if summary.get("use_cache_mismatch_detected"):
        return "logits processor/stopping issue"
    if summary.get("direct_vs_generate_scores_allclose") is False and (not same_argmax or rank_gap > 1000):
        return "generation logits mismatch"
    return "other"


def write_report(path: Path, summary: Dict[str, Any], template_hits: Sequence[Dict[str, Any]]) -> None:
    hit_text = ", ".join(row["template"] for row in template_hits) if template_hits else "none"
    lines = [
        "# LLaVA-Med Generate-vs-Forward Diagnostic",
        "",
        f"- overfit dir: `{summary.get('overfit_dir')}`",
        f"- repository loaded: `{summary.get('repository_path')}`",
        f"- target: `{summary.get('target')}`",
        f"- direct target rank: `{summary.get('direct_target_rank')}`",
        f"- generate scores target rank: `{summary.get('generate_scores_target_rank')}`",
        f"- direct argmax token: `{summary.get('direct_argmax_token')}`",
        f"- generate argmax token: `{summary.get('generate_scores_argmax_token')}`",
        f"- allclose: `{summary.get('direct_vs_generate_scores_allclose')}`",
        f"- max abs diff: `{summary.get('direct_vs_generate_max_abs_diff')}`",
        f"- generated suffix text: `{summary.get('generated_suffix_text')}`",
        f"- full decoded text: `{summary.get('full_decoded_text')}`",
        f"- slicing bug detected: `{summary.get('slicing_bug_detected')}`",
        f"- template target hits: `{hit_text}`",
        f"- manual greedy contains target: `{summary.get('manual_greedy_contains_target')}`",
        f"- root cause: `{summary.get('root_cause_classification')}`",
        f"- recommended patch: `{summary.get('recommended_patch')}`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    ensure_env()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.output_dir / "generation_hook_events.jsonl"
    if event_path.exists():
        event_path.unlink()

    records = load_records(args.dataset_path)
    record = records[args.sample_index]
    target = target_for(record)
    model, alg, _config = load_model_alg(args)
    repo_path, overfit_summary = load_repository(alg, args.overfit_dir, model.lm_device)
    assigned_cluster_id = int(overfit_summary.get("assigned_cluster_id", 0))
    overfit_residual_scale = overfit_summary.get("residual_scale")
    residual_scale = float(overfit_residual_scale) if overfit_residual_scale is not None else None
    force_route_ids = [assigned_cluster_id]

    sample = make_sample(model, record, args.image_root, prompt_current_pilot(record), target)
    target_ids = target_token_ids_for_prompt(model, sample["prompt"][0], target)
    if not target_ids:
        raise RuntimeError("No target ids resolved for the generation prompt.")
    target_first_token_id = int(target_ids[0])

    direct_modes = [
        ("A_direct_base_disabled", False, None, None),
        ("B_direct_dsca_normal", True, None, None),
        ("C_direct_dsca_force_route", True, force_route_ids, None),
        ("D_direct_dsca_force_route_scale2", True, force_route_ids, 2.0),
    ]
    direct_results: Dict[str, Dict[str, Any]] = {}
    comparison_rows: List[Dict[str, Any]] = []
    for name, enabled, forced, scale in direct_modes:
        result = run_forward_logits(model, alg, sample, name, enabled, forced, scale)
        append_events(event_path, result["events"])
        direct_results[name] = result
        logits = result["logits"]
        comparison_rows.append(
            {
                "mode": name,
                "source": "direct_forward",
                "target_first_token_id": target_first_token_id,
                "target_first_token": decode_ids(model.llava_tokenizer, [target_first_token_id], False),
                "argmax_token_id": int(logits.argmax().item()),
                "argmax_token": decode_ids(model.llava_tokenizer, [int(logits.argmax().item())], False),
                "target_rank": rank_of_token(logits, target_first_token_id),
                "target_logprob": token_logprob(logits, target_first_token_id),
                "hook_event_count": len(result["events"]),
                "residual_scale": scale,
            }
        )

    primary_mode = "D_direct_dsca_force_route_scale2" if abs(float(residual_scale or 1.0) - 2.0) < 1.0e-8 else "C_direct_dsca_force_route"
    primary_direct = direct_results[primary_mode]["logits"]

    generate_true = run_generate(
        model,
        alg,
        sample,
        "E_generate_use_cache_true",
        force_route_ids=force_route_ids,
        residual_scale=residual_scale,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    append_events(event_path, generate_true["events"])
    generate_false = run_generate(
        model,
        alg,
        sample,
        "E_generate_use_cache_false",
        force_route_ids=force_route_ids,
        residual_scale=residual_scale,
        max_new_tokens=args.max_new_tokens,
        use_cache=False,
    )
    append_events(event_path, generate_false["events"])

    compare_true = compare_logits(
        model.llava_tokenizer,
        "primary_direct_vs_generate_use_cache_true",
        primary_direct,
        generate_true["scores0"],
        target_first_token_id,
    )
    compare_false = compare_logits(
        model.llava_tokenizer,
        "primary_direct_vs_generate_use_cache_false",
        primary_direct,
        generate_false["scores0"],
        target_first_token_id,
    )
    comparison_rows.extend(
        [
            {
                "mode": "E_generate_use_cache_true",
                "source": "generate_scores0",
                "target_first_token_id": target_first_token_id,
                "target_first_token": decode_ids(model.llava_tokenizer, [target_first_token_id], False),
                "argmax_token_id": compare_true.get("generate_scores_argmax_token_id"),
                "argmax_token": compare_true.get("generate_scores_argmax_token"),
                "target_rank": compare_true.get("generate_scores_target_rank"),
                "target_logprob": compare_true.get("generate_scores_target_logprob"),
                "allclose_to_primary_direct": compare_true.get("allclose"),
                "max_abs_diff_to_primary_direct": compare_true.get("max_abs_diff"),
                "mean_abs_diff_to_primary_direct": compare_true.get("mean_abs_diff"),
                "hook_event_count": len(generate_true["events"]),
                "residual_scale": residual_scale,
            },
            {
                "mode": "E_generate_use_cache_false",
                "source": "generate_scores0",
                "target_first_token_id": target_first_token_id,
                "target_first_token": decode_ids(model.llava_tokenizer, [target_first_token_id], False),
                "argmax_token_id": compare_false.get("generate_scores_argmax_token_id"),
                "argmax_token": compare_false.get("generate_scores_argmax_token"),
                "target_rank": compare_false.get("generate_scores_target_rank"),
                "target_logprob": compare_false.get("generate_scores_target_logprob"),
                "allclose_to_primary_direct": compare_false.get("allclose"),
                "max_abs_diff_to_primary_direct": compare_false.get("max_abs_diff"),
                "mean_abs_diff_to_primary_direct": compare_false.get("mean_abs_diff"),
                "hook_event_count": len(generate_false["events"]),
                "residual_scale": residual_scale,
            },
        ]
    )

    manual_rows, manual_text = run_manual_greedy(
        model,
        alg,
        sample,
        force_route_ids=force_route_ids,
        residual_scale=residual_scale,
        target_first_token_id=target_first_token_id,
        max_new_tokens=args.max_new_tokens,
        event_path=event_path,
    )
    template_result_rows = template_rows(
        model,
        alg,
        record,
        args.image_root,
        target,
        assigned_cluster_id,
        residual_scale,
        args.max_new_tokens,
        event_path,
    )

    existing_prediction_text = overfit_summary.get("final_free_generation")
    generated_suffix_text = generate_true["generated_suffix_text"]
    full_decoded_text = generate_true["full_decoded_text"]
    slicing_bug_detected = bool(
        generate_true["sequence_includes_prompt"]
        and generated_suffix_text != full_decoded_text
        and existing_prediction_text
        and str(existing_prediction_text).strip() == str(full_decoded_text).strip()
    )
    fields_suffix = answer_fields(None, generated_suffix_text, target)
    fields_manual = answer_fields(None, manual_text, target)
    current_template = next(row for row in template_result_rows if row["template"] == "current_pilot")
    template_hits = [
        row
        for row in template_result_rows
        if row["template"] != "current_pilot" and (bool(row["contains_target"]) or bool(row["exact_match"]))
    ]
    template_bug_detected = bool(template_hits and not (current_template["contains_target"] or current_template["exact_match"]))
    use_cache_mismatch_detected = bool(
        compare_true.get("generate_scores_argmax_token_id") != compare_false.get("generate_scores_argmax_token_id")
        or compare_true.get("generate_scores_target_rank") != compare_false.get("generate_scores_target_rank")
    )
    logits_processor_effect_detected = bool(
        compare_true.get("allclose") is False
        and (
            compare_true.get("direct_argmax_token_id") != compare_true.get("generate_scores_argmax_token_id")
            or abs(int(compare_true.get("direct_target_rank") or 0) - int(compare_true.get("generate_scores_target_rank") or 0))
            > 1000
        )
    )
    generate_bypasses_dsca_detected = bool(
        compare_true.get("direct_target_rank") == 1 and compare_true.get("generate_scores_target_rank") != 1
    )
    recommended_patch = "Fix prediction slicing to decode generated_ids suffix only."
    if generate_bypasses_dsca_detected:
        recommended_patch = "Inspect generate/prepare_inputs_for_generation path; scores[0] is not using the same edited logits."
    elif template_bug_detected:
        recommended_patch = "Switch evaluation prompt to the best matching official/short-answer template and rerun one-edit."
    elif (
        compare_true.get("direct_argmax_token_id") == compare_true.get("generate_scores_argmax_token_id")
        and int(compare_true.get("direct_target_rank") or 0) > 1000
        and int(compare_true.get("generate_scores_target_rank") or 0) > 1000
    ):
        recommended_patch = (
            "No generation-path patch; add a generation-aligned prompt-only next-token overfit diagnostic before rerunning one-edit."
        )
    elif not slicing_bug_detected:
        recommended_patch = "No concrete patch beyond reporting unresolved mismatch rows."

    generated_ids_list = [int(item) for item in generate_true["generated_ids"][0].detach().cpu().tolist()]
    generated_debug = {
        "raw_sequences_tensor": generate_true["sequences"].tolist(),
        "input_ids": generate_true["input_ids"].tolist(),
        "input_ids_length": generate_true["input_len"],
        "sequence_includes_prompt": generate_true["sequence_includes_prompt"],
        "generated_ids": generate_true["generated_ids"].tolist(),
        "full_decoded_text": full_decoded_text,
        "generated_suffix_text": generated_suffix_text,
        "full_decoded_text_keep_special": generate_true["full_decoded_text_keep_special"],
        "generated_suffix_keep_special": generate_true["generated_suffix_keep_special"],
        "generated_suffix_no_cleanup": generate_true["generated_suffix_no_cleanup"],
        "current_code_would_decode_full_sequence": False,
        "answer_should_decode_generated_ids_only": True,
        "tokenizer_special_tokens": {
            "bos_token": model.llava_tokenizer.bos_token,
            "eos_token": model.llava_tokenizer.eos_token,
            "pad_token": model.llava_tokenizer.pad_token,
            "unk_token": model.llava_tokenizer.unk_token,
            "additional_special_tokens": list(getattr(model.llava_tokenizer, "additional_special_tokens", []) or []),
        },
        "skip_special_tokens_true": generated_suffix_text,
        "skip_special_tokens_false": generate_true["generated_suffix_keep_special"],
        "clean_up_tokenization_spaces_true": generated_suffix_text,
        "clean_up_tokenization_spaces_false": generate_true["generated_suffix_no_cleanup"],
    }

    summary = {
        **compare_true,
        "direct_vs_generate_scores_allclose": compare_true.get("allclose"),
        "direct_vs_generate_max_abs_diff": compare_true.get("max_abs_diff"),
        "direct_vs_generate_mean_abs_diff": compare_true.get("mean_abs_diff"),
        "direct_argmax_token": compare_true.get("direct_argmax_token"),
        "generate_scores_argmax_token": compare_true.get("generate_scores_argmax_token"),
        "target_first_token": compare_true.get("target_first_token"),
        "direct_target_rank": compare_true.get("direct_target_rank"),
        "generate_scores_target_rank": compare_true.get("generate_scores_target_rank"),
        "generated_suffix_text": generated_suffix_text,
        "generated_suffix_contains_target": bool(fields_suffix["contains_target"]),
        "generated_suffix_exact_match": bool(fields_suffix["exact_match_normalized"]),
        "full_decoded_text": full_decoded_text,
        "existing_prediction_text_if_available": existing_prediction_text,
        "slicing_bug_detected": slicing_bug_detected,
        "template_bug_detected": template_bug_detected,
        "generate_bypasses_dsca_detected": generate_bypasses_dsca_detected,
        "logits_processor_effect_detected": logits_processor_effect_detected,
        "use_cache_mismatch_detected": use_cache_mismatch_detected,
        "manual_greedy_text": manual_text,
        "manual_greedy_contains_target": bool(fields_manual["contains_target"]),
        "manual_greedy_exact_match": bool(fields_manual["exact_match_normalized"]),
        "recommended_patch": recommended_patch,
        "overfit_dir": str(args.overfit_dir),
        "repository_path": str(repo_path),
        "output_dir": str(args.output_dir),
        "target": target,
        "prompt": sample["prompt"][0],
        "assigned_cluster_id": assigned_cluster_id,
        "residual_scale": residual_scale,
        "generation_config": generate_true["generation_config"],
        "logits_processor_list": generate_true["logits_processors"],
        "stopping_criteria_list": generate_true["stopping_criteria"],
        "eos_token_id": model.llava_tokenizer.eos_token_id,
        "pad_token_id": model.llava_tokenizer.pad_token_id,
        "bos_token_id": model.llava_tokenizer.bos_token_id,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "num_beams": 1,
    }
    summary["root_cause_classification"] = root_cause(summary)

    write_csv(args.output_dir / "first_step_logits_comparison.csv", comparison_rows)
    write_csv(args.output_dir / "template_generate_vs_forward.csv", template_result_rows)
    write_csv(args.output_dir / "manual_greedy_trace.csv", manual_rows)
    write_json(args.output_dir / "generated_ids_debug.json", generated_debug)
    write_json(args.output_dir / "generate_vs_forward_summary.json", summary)
    (args.output_dir / "top20_tokens_direct_vs_generate.md").write_text(
        markdown_top20(model.llava_tokenizer, primary_direct, generate_true["scores0"], target_first_token_id)
    )
    write_report(args.output_dir / "diagnosis_report.md", summary, template_hits)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
