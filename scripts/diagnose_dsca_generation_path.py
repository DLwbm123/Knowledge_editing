#!/usr/bin/env python3
"""Diagnose why DSCA teacher-forced gains do not become decoded MedMKEB answers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from dsca_medmkeb_diag_common import (
    active_ids,
    alias_list,
    answer_fields,
    append_jsonl,
    clone_batch,
    collate_record,
    decode_argmax_on_labels,
    load_dataset_and_model,
    target_nll_from_outputs,
    temporarily_force_route,
    to_jsonable,
    write_json,
)
from easyeditor.trainer.algs.dsca_utils import DSCAConceptRepository, DSCAContext, dsca_intervention_context, dsca_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, type=Path)
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--model", default="blip2", choices=["blip2", "llava-med"])
    parser.add_argument("--hparams", default="hparams/DSCA/blip2_20edit_pilot.yaml")
    parser.add_argument("--training-hparams", default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--repository-step", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generation-use-cache", default=None, choices=["true", "false"])
    parser.add_argument("--residual-apply-mask", default=None, choices=["attention", "vision_prompt", "all_nonpad", "current_token"])
    parser.add_argument("--dsca-generation-mode", default=None, choices=["normal", "prefill_only", "cache_reuse_route"])
    return parser.parse_args()


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(item) for item in values if item is not None and math.isfinite(float(item))]
    return sum(finite) / len(finite) if finite else None


def rank_of_token(logits: torch.Tensor, token_id: int) -> int:
    order = torch.argsort(logits.float(), descending=True)
    found = (order == int(token_id)).nonzero(as_tuple=False)
    return int(found[0].item() + 1) if found.numel() else int(logits.numel() + 1)


def token_logprob(logits: torch.Tensor, token_id: int) -> float:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(log_probs[int(token_id)].detach().cpu())


def topk_tokens(tokenizer: Any, logits: torch.Tensor, k: int = 10) -> List[Dict[str, Any]]:
    vals, ids = torch.topk(torch.log_softmax(logits.float(), dim=-1), k=min(k, logits.numel()))
    rows: List[Dict[str, Any]] = []
    for value, idx in zip(vals.detach().cpu().tolist(), ids.detach().cpu().tolist()):
        try:
            text = tokenizer.decode([int(idx)], skip_special_tokens=False)
        except Exception:
            text = str(idx)
        rows.append({"id": int(idx), "token": text, "logprob": float(value)})
    return rows


def decode_ids(tokenizer: Any, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
    try:
        return tokenizer.decode([int(item) for item in ids], skip_special_tokens=skip_special_tokens).strip()
    except Exception:
        return " ".join(str(int(item)) for item in ids)


def tensor_to_ids(tensor: torch.Tensor) -> List[int]:
    return [int(item) for item in tensor.detach().view(-1).cpu().tolist()]


def temporary_dsam_residual_scale(alg: Any, value: float):
    class _Ctx:
        def __enter__(self_inner):
            self_inner.old_values = [float(dsam.residual_scale) for dsam in alg.repository.dsams]
            for dsam in alg.repository.dsams:
                dsam.residual_scale = float(value)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            for dsam, old_value in zip(alg.repository.dsams, self_inner.old_values):
                dsam.residual_scale = old_value
            return False

    return _Ctx()


def residual_region_norms_from_tensors(
    residual: Optional[torch.Tensor],
    masks: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    keys = {
        "vision": "vision_mask",
        "prompt": "prompt_mask",
        "answer": "answer_mask",
        "padding": "attention_mask",
    }
    if residual is None or masks is None:
        return {
            "total_residual_norm": 0.0,
            "vision_residual_norm": 0.0,
            "prompt_residual_norm": 0.0,
            "answer_residual_norm": 0.0,
            "padding_residual_norm": 0.0,
        }
    residual = residual.detach()
    result = {"total_residual_norm": float(residual.norm().detach().cpu())}
    for label, mask_name in keys.items():
        if mask_name == "attention_mask":
            mask = ~masks.get(mask_name, torch.zeros(residual.shape[:2], device=residual.device, dtype=torch.bool)).bool()
        else:
            mask = masks.get(mask_name, torch.zeros(residual.shape[:2], device=residual.device, dtype=torch.bool)).bool()
        mask = mask.to(residual.device)
        if mask.shape != residual.shape[:2] or int(mask.sum().item()) == 0:
            value = 0.0
        else:
            token_norms = residual.float().norm(dim=-1).masked_select(mask)
            value = float(token_norms.mean().detach().cpu()) if token_norms.numel() else 0.0
        result[f"{label}_residual_norm"] = value
    return result


def residual_region_norms(alg: Any) -> Dict[str, float]:
    return residual_region_norms_from_tensors(getattr(alg, "_last_residual", None), getattr(alg, "_last_masks", None))


def prompt_text(dataset: Any, record: Dict[str, Any]) -> str:
    template = getattr(dataset, "prompt", "Question: {} Short answer: ")
    return template.format(record.get("prompt", ""))


def build_blip2_inputs(
    model: Any,
    image: Optional[torch.Tensor],
    token_ids: torch.Tensor,
    prompt_token_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    device = next(model.parameters()).device
    token_ids = token_ids.to(device)
    text_attention = torch.ones(token_ids.shape, device=device, dtype=torch.long)
    inputs_text = model.opt_model.model.decoder.embed_tokens(token_ids)
    if image is not None:
        image = image.to(device)
        with model.maybe_autocast():
            image_embeds = model.ln_vision(model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_opt = model.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long, device=device)
        inputs_embeds = torch.cat([inputs_opt, inputs_text], dim=1)
        attention_mask = torch.cat([atts_opt, text_attention], dim=1)
        vision_len = int(atts_opt.shape[1])
    else:
        inputs_embeds = inputs_text
        attention_mask = text_attention
        vision_len = 0
    vision_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    prompt_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    answer_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    if vision_len:
        vision_mask[:, :vision_len] = True
    text_start = vision_len
    prompt_end = text_start + int(prompt_token_count)
    prompt_mask[:, text_start:prompt_end] = True
    if token_ids.shape[1] > prompt_token_count:
        answer_mask[:, prompt_end : text_start + token_ids.shape[1]] = True
    masks = {
        "attention_mask": attention_mask,
        "vision_mask": vision_mask,
        "prompt_mask": prompt_mask & attention_mask.bool(),
        "answer_mask": answer_mask & attention_mask.bool(),
    }
    return inputs_embeds, attention_mask, masks


def opt_forward(
    model: Any,
    alg: Optional[Any],
    image: Optional[torch.Tensor],
    token_ids: torch.Tensor,
    prompt_token_count: int,
) -> Any:
    inputs_embeds, attention_mask, masks = build_blip2_inputs(model, image, token_ids, prompt_token_count)
    if alg is None:
        with model.maybe_autocast():
            return model.opt_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
    with dsca_intervention_context(alg, DSCAContext(batch=masks)):
        with model.maybe_autocast():
            return model.opt_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)


def route_snapshot(alg: Any, batch: Dict[str, Any], assigned: Optional[int]) -> Dict[str, Any]:
    reps = alg.capture_representations(clone_batch(batch))
    weights, selected, aux = dsca_route(
        reps["h_v"],
        reps["h_f"],
        alg.repository,
        tau_visual=alg.tau_visual,
        route_temperature=alg.route_temperature,
        candidate_topk=alg.candidate_topk,
    )
    candidate_ids = selected[0].nonzero(as_tuple=False).flatten().detach().cpu().tolist() if selected.numel() else []
    return {
        "normal_candidate_count": len(candidate_ids),
        "normal_candidate_ids": candidate_ids,
        "assigned_cluster_in_candidates": bool(assigned in candidate_ids) if assigned is not None else False,
        "active_dsam_available": bool(assigned is not None and assigned < len(alg.repository) and alg.repository.active[assigned].item()),
        "route_weight_assigned": float(weights[0, assigned].detach().cpu())
        if assigned is not None and weights.numel() and assigned < weights.shape[1]
        else 0.0,
        "visual_similarity_assigned": float(aux["visual_sim"][0, assigned].detach().cpu())
        if assigned is not None and aux["visual_sim"].numel() and assigned < aux["visual_sim"].shape[1]
        else None,
        "fused_similarity_assigned": float(aux["fused_sim"][0, assigned].detach().cpu())
        if assigned is not None and aux["fused_sim"].numel() and assigned < aux["fused_sim"].shape[1]
        else None,
        "active_dsam_ids": active_ids(alg.repository),
    }


def replay_assignments(dataset: Any, alg: Any, config: Any, limit: int) -> List[int]:
    repo = DSCAConceptRepository(
        hidden_size=alg.hidden_size,
        rank=alg.rank,
        gate_bottleneck=alg.gate_bottleneck,
        cluster_alpha=float(getattr(config, "dsca_cluster_alpha", 2.0)),
        proto_ema=float(getattr(config, "dsca_proto_ema", 0.95)),
        min_samples=int(getattr(config, "dsca_min_samples", 4)),
        refine_interval=int(getattr(config, "dsca_refine_interval", 10)),
        dsam_init_std=float(getattr(config, "dsca_dsam_init_std", 0.02)),
        residual_scale=float(getattr(config, "dsca_residual_scale", 1.0)),
        device=alg.repository.p_f.device,
        dtype=alg.repository.p_f.dtype,
    )
    ids: List[int] = []
    for idx in range(limit):
        batch = collate_record(dataset, dataset[idx])
        reps = alg.capture_representations(clone_batch(batch["edit_inner"]))
        assigned, _created = repo.assign_batch(reps["h_f"], reps["h_v"], initialize_basis=False)
        ids.extend(assigned)
    return ids


def greedy_generate(
    model: Any,
    alg: Optional[Any],
    tokenizer: Any,
    image: Optional[torch.Tensor],
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    prompt_token_count: int,
    cluster_id: Optional[int] = None,
    scale0: bool = False,
) -> Dict[str, Any]:
    generated: List[int] = []
    step_debug: List[Dict[str, Any]] = []
    context_managers = []
    if alg is not None and cluster_id is not None:
        context_managers.append(temporarily_force_route(alg, cluster_id))
    if alg is not None and scale0:
        context_managers.append(temporary_dsam_residual_scale(alg, 0.0))
    for manager in context_managers:
        manager.__enter__()
    try:
        for step in range(max_new_tokens):
            token_ids = torch.cat(
                [prompt_ids, torch.tensor([generated], device=prompt_ids.device, dtype=prompt_ids.dtype)],
                dim=1,
            )
            outputs = opt_forward(model, alg, image, token_ids, prompt_token_count)
            logits = outputs.logits[0, -1].float()
            next_id = int(torch.argmax(logits).detach().cpu())
            generated.append(next_id)
            weights = getattr(alg, "_last_route_weights", None) if alg is not None else None
            residual = residual_region_norms(alg) if alg is not None else {}
            step_debug.append(
                {
                    "decode_step": step,
                    "next_token_id": next_id,
                    "next_token": decode_ids(tokenizer, [next_id], skip_special_tokens=False),
                    "residual_norm": residual.get("total_residual_norm", 0.0),
                    "candidate_count": int(torch.count_nonzero(weights[0] > 0).item()) if weights is not None and weights.numel() else 0,
                    "route_weights": weights[0].detach().cpu().tolist() if weights is not None and weights.numel() else [],
                    "active_dsam_count": alg.repository.num_active() if alg is not None else 0,
                }
            )
            if next_id == getattr(model, "eos_token_id", None):
                break
    finally:
        for manager in reversed(context_managers):
            manager.__exit__(None, None, None)
    text = decode_ids(tokenizer, generated)
    return {
        "text": text,
        "tokens": generated,
        "normalized": answer_fields(None, text, "")["normalized_edited_prediction"],
        "steps": step_debug,
        "residual_norm_mean": mean([item["residual_norm"] for item in step_debug]) or 0.0,
        "candidate_count_mean": mean([item["candidate_count"] for item in step_debug]) or 0.0,
    }


def rank_diagnostic(
    model: Any,
    alg: Optional[Any],
    image: Optional[torch.Tensor],
    prompt_ids: torch.Tensor,
    target_ids: Sequence[int],
    prompt_token_count: int,
    cluster_id: Optional[int] = None,
) -> Dict[str, Any]:
    ranks: List[int] = []
    logprobs: List[float] = []
    top10: Optional[List[Dict[str, Any]]] = None
    context = temporarily_force_route(alg, cluster_id) if alg is not None and cluster_id is not None else None
    if context is not None:
        context.__enter__()
    try:
        for idx, target_id in enumerate(target_ids):
            prefix = torch.tensor([list(target_ids[:idx])], device=prompt_ids.device, dtype=prompt_ids.dtype)
            token_ids = torch.cat([prompt_ids, prefix], dim=1)
            outputs = opt_forward(model, alg, image, token_ids, prompt_token_count)
            logits = outputs.logits[0, -1].float()
            ranks.append(rank_of_token(logits, int(target_id)))
            logprobs.append(token_logprob(logits, int(target_id)))
            if idx == 0:
                top10 = topk_tokens(getattr(model, "opt_tokenizer", None), logits, k=10)
    finally:
        if context is not None:
            context.__exit__(None, None, None)
    return {
        "ranks": ranks,
        "logprobs": logprobs,
        "first_rank": ranks[0] if ranks else None,
        "first_logprob": logprobs[0] if logprobs else None,
        "mean_rank": mean([float(item) for item in ranks]),
        "top1_count": sum(1 for item in ranks if item == 1),
        "top10_first": top10 or [],
        "residual": residual_region_norms(alg) if alg is not None else {},
    }


def diagnosis_label(row: Dict[str, Any]) -> str:
    if row["generation_residual_norm_mean"] <= 1.0e-8 and row["teacher_forced_total_residual_norm"] > 1.0e-8:
        return "generation hook bypass or generation masks inactive"
    if row["teacher_forced_answer_residual_norm"] > 2.0 * max(row["teacher_forced_prompt_residual_norm"], row["teacher_forced_vision_residual_norm"], 1.0e-8):
        return "teacher-forcing answer-token leakage"
    if row["first_target_edited_rank"] is not None and row["first_target_base_rank"] is not None:
        if row["first_target_edited_rank"] < row["first_target_base_rank"] and row["first_target_edited_rank"] > 1:
            return "insufficient logit shift"
    if row["first_target_force_route_rank"] is not None and row["first_target_edited_rank"] is not None and row["first_target_force_route_rank"] < row["first_target_edited_rank"]:
        return "routing failure"
    if row["edited_equals_base"]:
        return "decoding/evaluation issue"
    return "BLIP2 medical-domain limitation or weak edit strength"


def make_report(output_dir: Path, summary: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# DSCA MedMKEB Generation Path Diagnostic",
        "",
        f"- output directory: `{output_dir}`",
        f"- generation path uses DSCA hook: {summary.get('generation_path_uses_dsca_hook')}",
        f"- edited free generation equals base rate: {summary.get('edited_equals_base_rate')}",
        f"- generation residual zero rate: {summary.get('generation_residual_zero_rate')}",
        f"- teacher-forced NLL improved count: {summary.get('teacher_forced_nll_improved_count')}",
        f"- prompt-only first target rank improved count: {summary.get('prompt_only_first_token_rank_improved_count')}",
        f"- force-route improvement count: {summary.get('force_route_improved_count')}",
        f"- answer-residual dominance rate: {summary.get('answer_residual_dominance_rate')}",
        f"- assigned cluster routed rate: {summary.get('assigned_cluster_routed_rate')}",
        f"- active DSAM available rate: {summary.get('active_dsam_available_rate')}",
        f"- top root-cause label: `{summary.get('top_root_cause_label')}`",
        "",
        "## Examples",
        "",
    ]
    for row in rows[:5]:
        lines.extend(
            [
                f"- step {row['step']} target `{row['target']}`",
                f"  base: `{row['base_free_text']}`",
                f"  edited: `{row['edited_free_text']}`",
                f"  label-window argmax: `{row['teacher_forced_label_window_argmax_text']}`",
                f"  first target ranks base/edited/force: {row['first_target_base_rank']} / {row['first_target_edited_rank']} / {row['first_target_force_route_rank']}",
                f"  diagnosis: `{row['diagnosis_label']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommended Next Experiment",
            "",
            str(summary.get("recommended_next_run")),
            "",
        ]
    )
    (output_dir / "generation_path_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.model == "llava-med":
        from smoke_llava_med_dsca_generation import run_llava_med_generation_path_diagnostic

        run_llava_med_generation_path_diagnostic(args)
        return
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / "generation_path_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()
    dataset, model, alg, tokenizer, config, dataset_path = load_dataset_and_model(args, num_samples=args.num_samples, load_repo=True)
    limit = min(args.num_samples, len(dataset))
    assigned_ids = replay_assignments(dataset, alg, config, limit)
    rows: List[Dict[str, Any]] = []

    for idx in range(limit):
        record = dataset[idx]
        batch = collate_record(dataset, record)
        edit_batch = clone_batch(batch["edit_inner"])
        image = edit_batch.get("image")
        prompt = prompt_text(dataset, record)
        prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(next(model.parameters()).device)
        prompt_ids = prompt_tokens.input_ids
        target_ids = tensor_to_ids(edit_batch["labels"])
        target_text = record.get("target")
        aliases = record.get("aliases", record.get("target_aliases"))
        assigned = assigned_ids[idx] if idx < len(assigned_ids) else None
        route = route_snapshot(alg, clone_batch(edit_batch), assigned)
        force_cluster = assigned if route["active_dsam_available"] else None

        with torch.no_grad():
            base_free = greedy_generate(model, None, tokenizer, image, prompt_ids, args.max_new_tokens, prompt_ids.shape[1])
            edited_free = greedy_generate(model, alg, tokenizer, image, prompt_ids, args.max_new_tokens, prompt_ids.shape[1])
            with temporary_dsam_residual_scale(alg, 0.0):
                scale0_free = greedy_generate(model, alg, tokenizer, image, prompt_ids, args.max_new_tokens, prompt_ids.shape[1])
            force_free = (
                greedy_generate(model, alg, tokenizer, image, prompt_ids, args.max_new_tokens, prompt_ids.shape[1], cluster_id=force_cluster)
                if force_cluster is not None
                else {"text": "", "tokens": [], "steps": [], "residual_norm_mean": 0.0}
            )
            base_outputs = model(clone_batch(edit_batch))
            edited_outputs = alg(clone_batch(edit_batch))
            tf_residual = residual_region_norms(alg)

            base_rank = rank_diagnostic(model, None, image, prompt_ids, target_ids, prompt_ids.shape[1])
            edited_rank = rank_diagnostic(model, alg, image, prompt_ids, target_ids, prompt_ids.shape[1])
            prompt_only_residual = residual_region_norms(alg)
            force_rank = (
                rank_diagnostic(model, alg, image, prompt_ids, target_ids, prompt_ids.shape[1], cluster_id=force_cluster)
                if force_cluster is not None
                else {"ranks": [], "logprobs": [], "first_rank": None, "first_logprob": None, "mean_rank": None, "top1_count": 0, "top10_first": [], "residual": {}}
            )

        base_nll = target_nll_from_outputs(base_outputs, edit_batch)
        edited_nll = target_nll_from_outputs(edited_outputs, edit_batch)
        teacher_argmax = decode_argmax_on_labels(tokenizer, edited_outputs, edit_batch)
        base_fields = answer_fields(None, base_free["text"], target_text, aliases)
        edited_fields = answer_fields(base_free["text"], edited_free["text"], target_text, aliases)
        force_fields = answer_fields(None, force_free["text"], target_text, aliases)
        tf_fields = answer_fields(None, teacher_argmax, target_text, aliases)
        first_base = base_rank["first_logprob"]
        first_edited = edited_rank["first_logprob"]
        first_force = force_rank["first_logprob"]
        row: Dict[str, Any] = {
            "step": idx + 1,
            "sample_id": idx,
            "prompt": record.get("prompt"),
            "target": target_text,
            "aliases": "; ".join(alias_list(aliases)),
            "base_free_text": base_free["text"],
            "base_free_tokens": ";".join(str(item) for item in base_free["tokens"]),
            "base_free_normalized": base_fields["normalized_edited_prediction"],
            "base_contains_target": base_fields["contains_target"],
            "base_exact_target": base_fields["exact_match_normalized"],
            "edited_free_text": edited_free["text"],
            "edited_free_tokens": ";".join(str(item) for item in edited_free["tokens"]),
            "edited_free_normalized": edited_fields["normalized_edited_prediction"],
            "edited_contains_target": edited_fields["contains_target"],
            "edited_exact_target": edited_fields["exact_match_normalized"],
            "edited_equals_base": edited_fields["edited_equals_base"],
            "edited_scale0_text": scale0_free["text"],
            "edited_scale0_equals_base": answer_fields(base_free["text"], scale0_free["text"], target_text)["edited_equals_base"],
            "scale0_residual_norm": scale0_free["residual_norm_mean"],
            "force_route_free_text": force_free["text"],
            "force_route_contains_target": force_fields["contains_target"],
            "base_target_nll": base_nll["target_nll"],
            "edited_target_nll": edited_nll["target_nll"],
            "delta_nll": edited_nll["target_nll"] - base_nll["target_nll"],
            "teacher_forced_label_window_argmax_text": teacher_argmax,
            "teacher_forced_argmax_exact_target": tf_fields["exact_match_normalized"],
            "teacher_forced_argmax_contains_target": tf_fields["contains_target"],
            "target_token_count": len(target_ids),
            "first_target_token": decode_ids(tokenizer, [target_ids[0]], skip_special_tokens=False) if target_ids else "",
            "first_target_base_rank": base_rank["first_rank"],
            "first_target_edited_rank": edited_rank["first_rank"],
            "first_target_force_route_rank": force_rank["first_rank"],
            "first_target_base_logprob": first_base,
            "first_target_edited_logprob": first_edited,
            "first_target_force_route_logprob": first_force,
            "first_target_delta_logprob": (first_edited - first_base) if first_edited is not None and first_base is not None else None,
            "mean_target_base_rank": base_rank["mean_rank"],
            "mean_target_edited_rank": edited_rank["mean_rank"],
            "mean_target_delta_logprob": mean(
                [
                    edited - base
                    for edited, base in zip(edited_rank["logprobs"], base_rank["logprobs"])
                ]
            ),
            "num_target_positions_improved": sum(
                1
                for edited, base in zip(edited_rank["logprobs"], base_rank["logprobs"])
                if edited > base
            ),
            "num_target_positions_top1_base": base_rank["top1_count"],
            "num_target_positions_top1_edited": edited_rank["top1_count"],
            "generation_residual_norm_mean": edited_free["residual_norm_mean"],
            "generation_candidate_count_by_decode_step": ";".join(str(item["candidate_count"]) for item in edited_free["steps"]),
            "generation_residual_norm_by_decode_step": ";".join(f"{item['residual_norm']:.6g}" for item in edited_free["steps"]),
            "generation_route_weights_by_decode_step": json.dumps([item["route_weights"] for item in edited_free["steps"]]),
            "generation_active_dsam_count_by_decode_step": ";".join(str(item["active_dsam_count"]) for item in edited_free["steps"]),
            "teacher_forced_total_residual_norm": tf_residual["total_residual_norm"],
            "teacher_forced_vision_residual_norm": tf_residual["vision_residual_norm"],
            "teacher_forced_prompt_residual_norm": tf_residual["prompt_residual_norm"],
            "teacher_forced_answer_residual_norm": tf_residual["answer_residual_norm"],
            "teacher_forced_padding_residual_norm": tf_residual["padding_residual_norm"],
            "prompt_only_residual_norm": prompt_only_residual["total_residual_norm"],
            "prompt_only_vision_residual_norm": prompt_only_residual["vision_residual_norm"],
            "prompt_only_prompt_residual_norm": prompt_only_residual["prompt_residual_norm"],
            "assigned_cluster_id": assigned,
            "assigned_cluster_in_candidates": route["assigned_cluster_in_candidates"],
            "active_dsam_available": route["active_dsam_available"],
            "route_weight_assigned": route["route_weight_assigned"],
            "normal_candidate_count": route["normal_candidate_count"],
            "force_route_used": force_cluster is not None,
            "force_route_delta_nll": None,
            "force_route_residual_norm": force_free["residual_norm_mean"],
        }
        row["diagnosis_label"] = diagnosis_label(row)
        rows.append(row)
        append_jsonl(
            debug_path,
            {
                **row,
                "base_decode_steps": base_free["steps"],
                "edited_decode_steps": edited_free["steps"],
                "scale0_decode_steps": scale0_free["steps"],
                "force_decode_steps": force_free["steps"],
                "first_target_top10_base": base_rank["top10_first"],
                "first_target_top10_edited": edited_rank["top10_first"],
                "first_target_top10_force_route": force_rank["top10_first"],
                "masks_sums": {
                    key: int(value.sum().detach().cpu())
                    for key, value in (getattr(alg, "_last_masks", {}) or {}).items()
                    if torch.is_tensor(value)
                },
                "sequence_lengths": {
                    "prompt_tokens": int(prompt_ids.shape[1]),
                    "target_tokens": len(target_ids),
                    "max_new_tokens": args.max_new_tokens,
                },
            },
        )

    csv_path = output_dir / "generation_path_per_sample.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    labels = Counter(row["diagnosis_label"] for row in rows)
    top_label = labels.most_common(1)[0][0] if labels else "inconclusive"
    summary = {
        "dataset_path": str(dataset_path),
        "num_rows": len(rows),
        "generation_path_uses_dsca_hook": any(row["generation_residual_norm_mean"] > 1.0e-8 for row in rows),
        "edited_equals_base_rate": sum(1 for row in rows if row["edited_equals_base"]) / len(rows) if rows else None,
        "generation_residual_zero_rate": sum(1 for row in rows if row["generation_residual_norm_mean"] <= 1.0e-8) / len(rows) if rows else None,
        "teacher_forced_nll_improved_count": sum(1 for row in rows if row["delta_nll"] < 0.0),
        "prompt_only_first_token_rank_improved_count": sum(
            1
            for row in rows
            if row["first_target_base_rank"] is not None
            and row["first_target_edited_rank"] is not None
            and row["first_target_edited_rank"] < row["first_target_base_rank"]
        ),
        "force_route_improved_count": sum(
            1
            for row in rows
            if row["first_target_force_route_rank"] is not None
            and row["first_target_edited_rank"] is not None
            and row["first_target_force_route_rank"] < row["first_target_edited_rank"]
        ),
        "answer_residual_dominance_rate": sum(
            1
            for row in rows
            if row["teacher_forced_answer_residual_norm"]
            > 2.0 * max(row["teacher_forced_prompt_residual_norm"], row["teacher_forced_vision_residual_norm"], 1.0e-8)
        )
        / len(rows)
        if rows
        else None,
        "assigned_cluster_routed_rate": sum(1 for row in rows if row["assigned_cluster_in_candidates"]) / len(rows) if rows else None,
        "active_dsam_available_rate": sum(1 for row in rows if row["active_dsam_available"]) / len(rows) if rows else None,
        "root_cause_counts": dict(labels),
        "top_root_cause_label": top_label,
        "recommended_next_run": {
            "generation hook bypass or generation masks inactive": "A. patch generation hook and rerun 20-edit evaluation",
            "teacher-forcing answer-token leakage": "B. rerun one-edit overfit with residual_apply_mask=vision_prompt",
            "routing failure": "D. run force-route ablation",
            "insufficient logit shift": "C. rerun 20-edit with lower min_samples / more train steps",
            "decoding/evaluation issue": "A. patch generation hook and rerun 20-edit evaluation",
        }.get(top_label, "F. use a medical VLM backbone if BLIP2 domain limitation dominates"),
    }
    write_json(output_dir / "generation_path_summary.json", summary)
    make_report(output_dir, summary, rows)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
