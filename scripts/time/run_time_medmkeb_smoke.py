#!/usr/bin/env python3
"""Bounded TIME MedMKEB smoke gates for LLaVA-Med."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dsca_medmkeb_diag_common import (  # noqa: E402
    append_jsonl,
    clone_batch,
    ensure_offline_env,
    resolve_dataset_path,
    target_nll_from_outputs,
    to_jsonable,
    torch_device,
    write_json,
)
from easyeditor.models.time_edit import TIMEEditMultimodalHparams  # noqa: E402
from easyeditor.trainer.algs.time_edit import TIMEEdit  # noqa: E402
from easyeditor.trainer.algs.time_edit_modules import time_memory_estimate  # noqa: E402
from easyeditor.trainer.models import get_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alg", default="TIME", choices=["TIME", "TIME_EDIT"])
    parser.add_argument("--mode", default="one", choices=["one", "nonseq", "sequential"])
    parser.add_argument("--dataset", default="MEDMKEB")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=Path, default=None)
    parser.add_argument("--image-root", "--image_root", dest="image_root", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-edits", "--max_edits", dest="max_edits", type=int, default=1)
    parser.add_argument("--hparams", "--config", default="hparams/TRAINING/TIME/llava_med_smoke.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-layer", "--target_layer", dest="target_layer", type=int, default=21)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scale-mode", "--scale_mode", dest="scale_mode", choices=["lora_like", "paper_inverse", "none"], default="lora_like")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--edit-iters", "--edit_iters", dest="edit_iters", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--init-std", "--init_std", dest="init_std", type=float, default=1.0e-3)
    parser.add_argument("--time-disable-selection", action="store_true")
    parser.add_argument("--time-disable-score-mixing", action="store_true")
    parser.add_argument("--time-disable-align-loss", action="store_true")
    parser.add_argument("--time-topk", type=int, default=0)
    parser.add_argument("--time-routing-mode", default="threshold", choices=["threshold", "topk", "threshold_topk", "force_current"])
    parser.add_argument("--time-force-current-train", dest="time_force_current_train", action="store_true", default=None)
    parser.add_argument("--time-gamma-sweep", default="")
    parser.add_argument("--time-scale-init-grid", default="")
    parser.add_argument("--time-token-scope", default="all", choices=["all", "last", "answer_mask"])
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=5)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_device_arg(text: str) -> Any:
    if text == "cuda":
        return "cuda"
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        return int(suffix) if suffix.isdigit() else text
    return int(text) if text.isdigit() else text


def load_records(dataset_path: Path, sample_index: int, max_edits: int) -> List[Dict[str, Any]]:
    records = json.loads(dataset_path.read_text(errors="replace"))
    if not isinstance(records, list):
        raise RuntimeError(f"Dataset JSON root must be a list: {dataset_path}")
    end = min(len(records), sample_index + max_edits)
    if sample_index < 0 or sample_index >= len(records) or sample_index >= end:
        raise IndexError(f"sample_index={sample_index} outside dataset of size {len(records)}")
    selected = records[sample_index:end]
    if len(selected) < max_edits:
        raise RuntimeError(f"Requested {max_edits} edits but only {len(selected)} records are available from index {sample_index}.")
    return selected


def resolve_image_path(image_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = image_root.resolve()
    if root.name == "images" and str(value).startswith("images/"):
        return root.parent / path
    return root / path


def record_id(record: Dict[str, Any], fallback: int) -> str:
    return str(record.get("id", record.get("record_id", fallback)))


def record_prompt(record: Dict[str, Any]) -> str:
    return "Question: {} Short answer: ".format(record.get("src") or record.get("prompt") or record.get("question") or "")


def record_target(record: Dict[str, Any]) -> str:
    return str(record.get("alt") or record.get("target") or record.get("answer") or "")


def make_sample(model: Any, record: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    prompt = record_prompt(record)
    target = record_target(record)
    labels = model.llava_tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return {
        "image_path": [str(resolve_image_path(image_root, record["image"]))],
        "prompt": [prompt],
        "target": [target],
        "text_input": [prompt + target],
        "labels": labels,
        "prompts_len": [len(model.llava_tokenizer(prompt, add_special_tokens=False).input_ids)],
    }


def configure(args: argparse.Namespace, dataset_path: Path) -> TIMEEditMultimodalHparams:
    config = TIMEEditMultimodalHparams.from_hparams(args.hparams)
    config.alg = args.alg
    config.alg_name = args.alg
    config.device = normalize_device_arg(args.device)
    config.lr = float(args.lr)
    config.edit_lr = float(args.lr)
    config.time_target_layer = int(args.target_layer)
    config.time_rank = int(args.rank)
    config.time_alpha = float(args.alpha)
    config.time_gamma = float(args.gamma)
    config.time_tau = float(args.tau)
    config.time_scale_mode = str(args.scale_mode)
    config.time_init_std = float(args.init_std)
    config.time_edit_iters = int(args.edit_iters)
    config.time_disable_selection = bool(args.time_disable_selection)
    config.time_disable_score_mixing = bool(args.time_disable_score_mixing)
    config.time_disable_align_loss = bool(args.time_disable_align_loss)
    config.time_topk = int(args.time_topk)
    config.time_routing_mode = str(args.time_routing_mode)
    if args.time_force_current_train is not None:
        config.time_force_current_during_training = bool(args.time_force_current_train)
    config.time_token_scope = str(args.time_token_scope)
    if args.image_root is not None:
        config.coco_image = str(args.image_root)
        config.rephrase_image = str(args.image_root)
    if not config.coco_image:
        config.coco_image = str(dataset_path.parent / "images")
        config.rephrase_image = config.coco_image
    return config


def logits_delta(outputs_a: Any, outputs_b: Any, batch: Dict[str, Any]) -> float:
    logits_a = outputs_a if isinstance(outputs_a, torch.Tensor) else outputs_a.logits
    logits_b = outputs_b if isinstance(outputs_b, torch.Tensor) else outputs_b.logits
    labels = batch["labels"]
    seq = min(logits_a.shape[1], logits_b.shape[1], labels.shape[1] + 1)
    return float((logits_b[:, -seq:].detach().float() - logits_a[:, -seq:].detach().float()).norm().cpu())


def evaluate_sample(
    alg: TIMEEdit,
    sample: Dict[str, Any],
    record: Dict[str, Any],
    sample_pos: int,
    phase: str,
    expected_expert: Optional[int],
    routing_debug_path: Path,
) -> Dict[str, Any]:
    debug_events: List[Dict[str, Any]] = []
    start = time.perf_counter()
    with torch.no_grad(), alg.time_disabled():
        base_batch = clone_batch(sample)
        base_outputs = alg.model(base_batch)
    with torch.no_grad():
        edited_batch = clone_batch(sample)
        edited_outputs = alg._forward_with_time(edited_batch, call_label=phase, debug_events=debug_events)
    elapsed = time.perf_counter() - start
    base_nll = target_nll_from_outputs(base_outputs, base_batch)
    edited_nll = target_nll_from_outputs(edited_outputs, edited_batch)
    routing = alg.routing_summary()
    rid = record_id(record, sample_pos)
    row = {
        "phase": phase,
        "sample_pos": sample_pos,
        "record_id": rid,
        "target": record_target(record),
        "expected_expert": expected_expert,
        "base_target_nll": base_nll.get("target_nll"),
        "target_nll": edited_nll.get("target_nll"),
        "target_nll_delta": (
            (base_nll.get("target_nll") or 0.0) - (edited_nll.get("target_nll") or 0.0)
            if base_nll.get("target_nll") is not None and edited_nll.get("target_nll") is not None
            else None
        ),
        "base_first_target_token_rank": base_nll.get("first_target_token_rank"),
        "first_target_token_rank": edited_nll.get("first_target_token_rank"),
        "target_rank_delta": (
            (base_nll.get("first_target_token_rank") or 0) - (edited_nll.get("first_target_token_rank") or 0)
            if base_nll.get("first_target_token_rank") is not None and edited_nll.get("first_target_token_rank") is not None
            else None
        ),
        "reference_delta": logits_delta(base_outputs, edited_outputs, edited_batch),
        "top_expert_id": routing.get("top_expert_id"),
        "top_score": routing.get("top_score"),
        "selected_expert_ids": routing.get("selected_expert_ids"),
        "selected_expert_set_size": routing.get("selected_expert_set_size"),
        "routing_top1_correct": bool(expected_expert is not None and routing.get("top_expert_id") == expected_expert),
        "residual_norm": routing.get("residual_norm"),
        "target_layer_hidden_delta_norm": routing.get("target_layer_hidden_delta_norm"),
        "target_layer_hidden_changed": routing.get("target_layer_hidden_changed"),
        "routing_mode": routing.get("routing_mode"),
        "gamma": routing.get("gamma"),
        "topk": routing.get("topk"),
        "elapsed_sec": elapsed,
        "generation_skipped": True,
    }
    append_jsonl(routing_debug_path, row)
    for event in debug_events:
        event = dict(event)
        event.update({"phase": f"{phase}_hook_event", "record_id": rid, "expected_expert": expected_expert})
        append_jsonl(routing_debug_path, event)
    return row


def train_one_edit(
    alg: TIMEEdit,
    sample: Dict[str, Any],
    record: Dict[str, Any],
    sample_pos: int,
    args: argparse.Namespace,
    loss_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rid = record_id(record, sample_pos)
    expert_index = alg.add_expert(
        rid,
        metadata={
            "sample_pos": sample_pos,
            "prompt": record_prompt(record),
            "target": record_target(record),
            "mode": args.mode,
        },
    )
    optimizer = torch.optim.AdamW(alg.outer_parameters(), lr=float(args.lr))
    batch = {
        "edit_inner": clone_batch(sample),
        "edit_outer": clone_batch(sample),
        "loc": clone_batch(sample),
        "loc_image": clone_batch(sample),
        "record_id": rid,
    }
    start = time.perf_counter()
    last_info: Dict[str, Any] = {}
    for step in range(1, int(args.edit_iters) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total, loss_rel, loss_loc, _loss_base, info = alg.edit_step(batch, training=True, optimizer=optimizer)
        torch.nn.utils.clip_grad_norm_(alg.outer_parameters(), float(alg.config.grad_clip), error_if_nonfinite=True)
        optimizer.step()
        last_info = dict(info)
        if step == 1 or step == args.edit_iters or step % max(1, args.log_every) == 0:
            loss_rows.append(
                {
                    "mode": args.mode,
                    "sample_pos": sample_pos,
                    "record_id": rid,
                    "expert_index": expert_index,
                    "step": step,
                    "loss_total": float(loss_total.detach().cpu()),
                    "loss_rel": float(loss_rel.detach().cpu()),
                    "loss_loc": float(loss_loc.detach().cpu()),
                    **{key.replace("/", "_"): value for key, value in last_info.items()},
                }
            )
    return {
        "record_id": rid,
        "sample_pos": sample_pos,
        "expert_index": expert_index,
        "train_elapsed_sec": time.perf_counter() - start,
        "last_info": last_info,
    }


def write_loss_trace(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(to_jsonable(row))


def summarize_evals(mode: str, eval_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]], alg: TIMEEdit, out_dir: Path) -> Dict[str, Any]:
    deltas = [row.get("target_nll_delta") for row in eval_rows if row.get("target_nll_delta") is not None]
    ref = [row.get("reference_delta") for row in eval_rows if row.get("reference_delta") is not None]
    selected_sizes = [row.get("selected_expert_set_size") for row in eval_rows if row.get("selected_expert_set_size") is not None]
    improved = [row for row in eval_rows if (row.get("target_nll_delta") or 0.0) > 0.0 or (row.get("target_rank_delta") or 0.0) > 0.0]
    positive_new_count = sum(1 for row in eval_rows if row.get("routing_top1_correct"))
    confusion: Dict[str, Dict[str, int]] = {}
    for row in eval_rows:
        expected = str(row.get("expected_expert"))
        observed = str(row.get("top_expert_id"))
        confusion.setdefault(expected, {})
        confusion[expected][observed] = confusion[expected].get(observed, 0) + 1
    repo_path = out_dir / "expert_repository.pt"
    alg.repository.save(repo_path)
    memory = time_memory_estimate(alg.repository.hidden_size, alg.repository.rank, alg.repository.s1, alg.repository.s2)
    memory["repository_size_bytes"] = float(repo_path.stat().st_size if repo_path.exists() else 0)
    return {
        "mode": mode,
        "num_eval_records": len(eval_rows),
        "num_experts": alg.repository.num_experts,
        "mean_target_nll_delta": float(sum(deltas) / len(deltas)) if deltas else None,
        "improved_count": len(improved),
        "positive_new_count": positive_new_count,
        "routing_top1_accuracy": float(positive_new_count / len(eval_rows)) if eval_rows else None,
        "mean_reference_delta": float(sum(ref) / len(ref)) if ref else None,
        "mean_selected_expert_set_size": float(sum(selected_sizes) / len(selected_sizes)) if selected_sizes else None,
        "routing_confusion_matrix": confusion,
        "eval_rows": eval_rows,
        "train_records": train_rows,
        "memory_estimate": memory,
        "expert_repository": str(repo_path),
        "acceptance": {
            "target_improvement_gate": len(improved) >= max(1, math.ceil(0.8 * len(eval_rows))) if eval_rows else False,
            "routing_gate": positive_new_count >= max(1, math.ceil(0.8 * len(eval_rows))) if eval_rows else False,
            "locality_reported": bool(ref),
        },
    }


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    config: TIMEEditMultimodalHparams,
    dataset_path: Path,
    summary: Dict[str, Any],
) -> None:
    memory = summary.get("memory_estimate", {})
    lines = [
        "# TIME MedMKEB Reproduction Report",
        "",
        "## Scope",
        "- Implementation: direct-factor TIME reproduction from the paper equations; no official code is used.",
        "- Cross-attention factor generator: not implemented in this first smoke; each edit optimizes trainable CP factors directly.",
        "- Long/full experiments started: no.",
        "",
        "## Files",
        "- Added `easyeditor/models/time_edit/`.",
        "- Added `easyeditor/trainer/algs/time_edit.py`.",
        "- Added `scripts/time/test_time_modules.py` and `scripts/time/run_time_medmkeb_smoke.py`.",
        "- Added `hparams/TIME/llava_med.yaml` and `hparams/TRAINING/TIME/llava_med_smoke.yaml`.",
        "- Modified EasyEdit registries and multimodal trainer dispatch.",
        "",
        "## Reused Utilities",
        "- Model loading: `easyeditor/trainer/models.py` and `LlavaMedForEditing`.",
        "- MedMKEB helpers: `scripts/dsca_medmkeb_diag_common.py`.",
        "- Logging style: JSON, JSONL, CSV, and Markdown artifacts under the requested output directory.",
        "",
        "## Mathematical Choices",
        f"- Hidden size/factors: H={summary.get('hidden_size')}, s1={summary.get('s1')}, s2={summary.get('s2')}.",
        f"- CP rank: R={config.time_rank}.",
        f"- Activation: {config.time_activation}.",
        f"- Scale mode: {config.time_scale_mode}, alpha={config.time_alpha}.",
        f"- Token scope: {config.time_token_scope}.",
        f"- Routing mode/threshold/top-k: mode={config.time_routing_mode}, gamma={config.time_gamma}, topk={config.time_topk}.",
        f"- Mixing: {'average selected experts' if config.time_disable_score_mixing else 'softmax over selected intrinsic scores'} with tau={config.time_tau}.",
        f"- Train-time force-current expert: {config.time_force_current_during_training}.",
        "",
        "## Run",
        f"- Dataset: `{dataset_path}`.",
        f"- Mode: {args.mode}.",
        f"- Max edits: {args.max_edits}.",
        f"- Edit iterations: {args.edit_iters}.",
        f"- Learning rate: {args.lr}.",
        f"- Skip generation: {args.skip_generation}.",
        "",
        "## Results",
        f"- Mean target NLL delta: {summary.get('mean_target_nll_delta')}.",
        f"- Improved count: {summary.get('improved_count')} / {summary.get('num_eval_records')}.",
        f"- Positive-new/top-1 count: {summary.get('positive_new_count')} / {summary.get('num_eval_records')}.",
        f"- Routing top-1 accuracy: {summary.get('routing_top1_accuracy')}.",
        f"- Mean reference/locality delta: {summary.get('mean_reference_delta')}.",
        f"- Mean selected expert set size: {summary.get('mean_selected_expert_set_size')}.",
        f"- Routing confusion matrix: `{json.dumps(summary.get('routing_confusion_matrix'), sort_keys=True)}`.",
        "",
        "## Memory",
        f"- TIME CP parameters per expert: {memory.get('time_params_per_expert')}.",
        f"- Equivalent LoRA parameters per expert at same rank: {memory.get('lora_params_per_expert')}.",
        f"- Repository size on disk: {memory.get('repository_size_bytes')} bytes.",
        "",
        "## Deviations",
        "- Direct random CP-factor initialization is used instead of the paper's cross-attention factor generator.",
        "- During training only, the current expert is forced active by default so randomly initialized factors can receive task gradients; inference uses intrinsic threshold/top-k routing.",
        "- Generality loss is logged as skipped when no separate paraphrase/generalization batch is provided by the smoke sample.",
        "- Generation diagnostics are skipped when `--skip-generation` is set.",
        "",
        "## Recommendation",
        "- Proceed to 20-edit nonseq only if the one-edit and 5-edit gates improve target NLL/rank and routing top-1 is interpretable.",
        "- Run `--time-disable-align-loss` as the first ablation after these gates.",
        "- Compare against SAME / LoRA-MoE after TIME passes the same bounded MedMKEB gates.",
        "",
    ]
    (out_dir / "TIME_MEDMKEB_REPRO_REPORT.md").write_text("\n".join(lines))


def run_smoke(
    args: argparse.Namespace,
    config: TIMEEditMultimodalHparams,
    dataset_path: Path,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    alg: TIMEEdit,
    out_dir: Path,
    print_summary: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    routing_debug_path = out_dir / "routing_debug.jsonl"
    if routing_debug_path.exists():
        routing_debug_path.unlink()
    write_json(
        out_dir / "time_hparams.json",
        {
            "args": vars(args),
            "config": dict(config.__dict__),
            "scale_mode": config.time_scale_mode,
            "hidden_size": alg.repository.hidden_size,
            "s1": alg.repository.s1,
            "s2": alg.repository.s2,
        },
    )

    loss_rows: List[Dict[str, Any]] = []
    train_records: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    immediate_after_edit: Dict[str, float] = {}

    for pos, (record, sample) in enumerate(zip(records, samples)):
        train_records.append(train_one_edit(alg, sample, record, pos, args, loss_rows))
        if args.mode == "sequential":
            for eval_pos in range(pos + 1):
                eval_row = evaluate_sample(
                    alg,
                    samples[eval_pos],
                    records[eval_pos],
                    eval_pos,
                    phase=f"after_edit_{pos}_eval_{eval_pos}",
                    expected_expert=eval_pos,
                    routing_debug_path=routing_debug_path,
                )
                eval_rows.append(eval_row)
                if eval_pos == pos and eval_row.get("target_nll") is not None:
                    immediate_after_edit[record_id(records[eval_pos], eval_pos)] = float(eval_row["target_nll"])

    if args.mode != "sequential":
        for eval_pos, (record, sample) in enumerate(zip(records, samples)):
            eval_rows.append(
                evaluate_sample(
                    alg,
                    sample,
                    record,
                    eval_pos,
                    phase=f"final_{args.mode}_eval_{eval_pos}",
                    expected_expert=eval_pos,
                    routing_debug_path=routing_debug_path,
                )
            )
    else:
        final_rows: List[Dict[str, Any]] = []
        for eval_pos, (record, sample) in enumerate(zip(records, samples)):
            row = evaluate_sample(
                alg,
                sample,
                record,
                eval_pos,
                phase=f"final_sequential_eval_{eval_pos}",
                expected_expert=eval_pos,
                routing_debug_path=routing_debug_path,
            )
            rid = record_id(record, eval_pos)
            if rid in immediate_after_edit and row.get("target_nll") is not None:
                row["retention_target_nll_delta_vs_immediate"] = immediate_after_edit[rid] - float(row["target_nll"])
            final_rows.append(row)
        eval_rows = final_rows

    write_loss_trace(out_dir / "loss_trace.csv", loss_rows)
    summary = summarize_evals(args.mode, eval_rows, train_records, alg, out_dir)
    summary.update({"hidden_size": alg.repository.hidden_size, "s1": alg.repository.s1, "s2": alg.repository.s2})
    if args.mode == "one":
        summary_path = out_dir / "one_edit_summary.json"
    elif args.mode == "nonseq":
        summary_path = out_dir / "five_edit_nonseq_summary.json"
    else:
        summary_path = out_dir / "five_edit_seq_summary.json"
    write_json(summary_path, summary)
    write_report(out_dir, args, config, dataset_path, summary)
    if print_summary:
        print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


@contextmanager
def temporary_time_routing(
    alg: TIMEEdit,
    routing_mode: Optional[str] = None,
    gamma: Optional[float] = None,
    topk: Optional[int] = None,
):
    old_gamma = alg.repository.gamma
    old_mode = alg.time_residual.routing_mode
    old_topk = alg.time_residual.topk
    old_config_gamma = getattr(alg.config, "time_gamma", None)
    old_config_mode = getattr(alg.config, "time_routing_mode", None)
    old_config_topk = getattr(alg.config, "time_topk", None)
    try:
        if gamma is not None:
            alg.repository.gamma = float(gamma)
            setattr(alg.config, "time_gamma", float(gamma))
        if routing_mode is not None:
            alg.time_residual.routing_mode = str(routing_mode)
            setattr(alg.config, "time_routing_mode", str(routing_mode))
        if topk is not None:
            alg.time_residual.topk = int(topk)
            setattr(alg.config, "time_topk", int(topk))
        yield
    finally:
        alg.repository.gamma = old_gamma
        alg.time_residual.routing_mode = old_mode
        alg.time_residual.topk = old_topk
        if old_config_gamma is not None:
            setattr(alg.config, "time_gamma", old_config_gamma)
        if old_config_mode is not None:
            setattr(alg.config, "time_routing_mode", old_config_mode)
        if old_config_topk is not None:
            setattr(alg.config, "time_topk", old_config_topk)


def parse_float_csv(text: str) -> List[float]:
    return [float(part.strip()) for part in str(text or "").split(",") if part.strip()]


def run_gamma_sweep(
    args: argparse.Namespace,
    alg: TIMEEdit,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    gammas = parse_float_csv(args.time_gamma_sweep)
    rows: List[Dict[str, Any]] = []
    routing_debug_path = out_dir / "routing_debug.jsonl"
    for gamma in gammas:
        with temporary_time_routing(alg, routing_mode="threshold", gamma=gamma, topk=0):
            row = evaluate_sample(
                alg,
                samples[0],
                records[0],
                0,
                phase=f"gamma_sweep_{gamma:g}",
                expected_expert=0,
                routing_debug_path=routing_debug_path,
            )
        row["sweep_gamma"] = gamma
        rows.append(row)
    if rows:
        write_loss_trace(out_dir / "time_gamma_sweep.csv", rows)
        write_json(out_dir / "time_gamma_sweep.json", {"rows": rows})
    return rows


def parse_scale_init_grid(text: str) -> Tuple[List[float], List[float]]:
    values = {"init_std": [], "alpha": []}
    for item in str(text or "").split(";"):
        if not item.strip():
            continue
        key, raw_values = item.split("=", 1)
        key = key.strip()
        if key not in values:
            raise ValueError(f"Unsupported TIME scale/init grid key: {key}")
        values[key] = parse_float_csv(raw_values)
    if not values["init_std"] or not values["alpha"]:
        raise ValueError("--time-scale-init-grid must include init_std=... and alpha=...")
    return values["init_std"], values["alpha"]


def format_float_for_path(value: float) -> str:
    return f"{float(value):g}".replace("-", "neg").replace(".", "p")


def run_scale_init_grid(
    args: argparse.Namespace,
    base_config: TIMEEditMultimodalHparams,
    dataset_path: Path,
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    model: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:
    init_values, alpha_values = parse_scale_init_grid(args.time_scale_init_grid)
    rows: List[Dict[str, Any]] = []
    for init_std in init_values:
        for alpha in alpha_values:
            combo_args = deepcopy(args)
            combo_args.init_std = float(init_std)
            combo_args.alpha = float(alpha)
            combo_args.time_routing_mode = "force_current"
            combo_args.time_topk = int(args.time_topk or 1)
            combo_config = deepcopy(base_config)
            combo_config.time_init_std = float(init_std)
            combo_config.time_alpha = float(alpha)
            combo_config.time_routing_mode = "force_current"
            combo_config.time_topk = int(combo_args.time_topk)
            combo_config.time_force_current_during_training = True
            subdir = args.out_dir / f"init_std_{format_float_for_path(init_std)}_alpha_{format_float_for_path(alpha)}"
            alg = TIMEEdit(model, combo_config, lambda: None).to(device)
            try:
                summary = run_smoke(combo_args, combo_config, dataset_path, records, samples, alg, subdir, print_summary=False)
                force_row = (summary.get("eval_rows") or [{}])[0]
                with temporary_time_routing(alg, routing_mode="topk", gamma=float(args.gamma), topk=1):
                    topk_row = evaluate_sample(
                        alg,
                        samples[0],
                        records[0],
                        0,
                        phase="grid_topk_eval",
                        expected_expert=0,
                        routing_debug_path=subdir / "routing_debug.jsonl",
                    )
                with temporary_time_routing(alg, routing_mode="threshold", gamma=0.5, topk=0):
                    threshold_row = evaluate_sample(
                        alg,
                        samples[0],
                        records[0],
                        0,
                        phase="grid_threshold_gamma_0_5_eval",
                        expected_expert=0,
                        routing_debug_path=subdir / "routing_debug.jsonl",
                    )
                rows.append(
                    {
                        "init_std": float(init_std),
                        "alpha": float(alpha),
                        "scale_mode": args.scale_mode,
                        "final_target_nll_delta": force_row.get("target_nll_delta"),
                        "final_target_rank_delta": force_row.get("target_rank_delta"),
                        "final_score": force_row.get("top_score"),
                        "force_current_residual_norm": force_row.get("residual_norm"),
                        "topk_residual_norm": topk_row.get("residual_norm"),
                        "topk_target_nll_delta": topk_row.get("target_nll_delta"),
                        "threshold_gamma_0_5_selected": bool((threshold_row.get("selected_expert_set_size") or 0) > 0),
                        "threshold_gamma_0_5_selected_expert_ids": threshold_row.get("selected_expert_ids"),
                        "threshold_gamma_0_5_top_score": threshold_row.get("top_score"),
                        "threshold_gamma_0_5_residual_norm": threshold_row.get("residual_norm"),
                        "subdir": str(subdir),
                    }
                )
            finally:
                alg.remove_hook()
                del alg
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    write_loss_trace(args.out_dir / "time_scale_init_grid.csv", rows)
    write_json(args.out_dir / "time_scale_init_grid.json", {"rows": rows})
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(errors="replace"))


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_eval(summary: Dict[str, Any]) -> Dict[str, Any]:
    rows = summary.get("eval_rows") or []
    return rows[0] if rows else {}


def _result_lines(title: str, row: Dict[str, Any]) -> List[str]:
    if not row:
        return [f"## {title}", "- Not run or not available.", ""]
    return [
        f"## {title}",
        f"- Target NLL before/after: {row.get('base_target_nll')} -> {row.get('target_nll')}.",
        f"- Target NLL delta: {row.get('target_nll_delta')}.",
        f"- Target rank delta: {row.get('target_rank_delta')}.",
        f"- Residual norm: {row.get('residual_norm')}.",
        f"- Target-layer hidden delta norm: {row.get('target_layer_hidden_delta_norm')}.",
        f"- Top score: {row.get('top_score')}.",
        f"- Selected expert ids: {row.get('selected_expert_ids')}.",
        f"- Locality/reference delta: {row.get('reference_delta')}.",
        "",
    ]


def calibrated_report_dir(out_dir: Path) -> Path:
    if out_dir.name == "one_edit_calibrated":
        return out_dir
    if out_dir.name.startswith("one_edit_"):
        return out_dir.parent / "one_edit_calibrated"
    return out_dir / "one_edit_calibrated"


def write_calibrated_report(out_dir: Path) -> Path:
    report_dir = calibrated_report_dir(out_dir)
    base_dir = report_dir.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    force_summary = _read_json(base_dir / "one_edit_calibrated_force" / "one_edit_summary.json")
    topk_summary = _read_json(base_dir / "one_edit_calibrated_topk" / "one_edit_summary.json")
    gamma_rows = _read_csv_rows(base_dir / "one_edit_calibrated_force" / "time_gamma_sweep.csv")
    grid_rows = _read_csv_rows(base_dir / "one_edit_scale_init_grid" / "time_scale_init_grid.csv")
    force_row = _first_eval(force_summary)
    topk_row = _first_eval(topk_summary)

    selected_gamma_rows = [row for row in gamma_rows if (_float_or_none(row.get("selected_expert_set_size")) or 0.0) > 0.0]
    selected_gammas = [_float_or_none(row.get("sweep_gamma")) for row in selected_gamma_rows]
    selected_gammas = [value for value in selected_gammas if value is not None]
    threshold_05_rows = [row for row in gamma_rows if _float_or_none(row.get("sweep_gamma")) == 0.5]
    threshold_05_selected = any((_float_or_none(row.get("selected_expert_set_size")) or 0.0) > 0.0 for row in threshold_05_rows)

    force_residual = _float_or_none(force_row.get("residual_norm"))
    force_delta = _float_or_none(force_row.get("target_nll_delta"))
    force_rank_delta = _float_or_none(force_row.get("target_rank_delta"))
    topk_residual = _float_or_none(topk_row.get("residual_norm"))
    topk_delta = _float_or_none(topk_row.get("target_nll_delta"))
    topk_rank_delta = _float_or_none(topk_row.get("target_rank_delta"))
    force_pass = (force_residual or 0.0) > 0.0 and ((force_delta or 0.0) > 0.0 or (force_rank_delta or 0.0) > 0.0)
    topk_pass = (topk_residual or 0.0) > 0.0 and ((topk_delta or 0.0) > 0.0 or (topk_rank_delta or 0.0) > 0.0)
    if force_residual is not None and force_residual <= 0.0:
        diagnosis = "hook_execution_bug"
    elif (force_pass or topk_pass) and not threshold_05_selected:
        diagnosis = "threshold_calibration_issue"
    elif (force_residual or 0.0) > 0.0 and not (force_pass or topk_pass):
        diagnosis = "scale_factor_issue"
    else:
        diagnosis = "mixed"

    lines = [
        "# TIME Calibrated 1-Edit Diagnostic Report",
        "",
        "## Files Changed",
        "- `easyeditor/trainer/algs/time_edit.py`",
        "- `easyeditor/trainer/algs/time_edit_modules.py`",
        "- `easyeditor/models/time_edit/time_edit_hparams.py`",
        "- `scripts/time/run_time_medmkeb_smoke.py`",
        "",
        "## Diagnostic Switches",
        "- `--time-force-current-train`: added and logged; it forces the current expert into the selected set only during training.",
        "- `--time-routing-mode`: added with `threshold`, `topk`, `threshold_topk`, and `force_current`.",
        "- `--time-gamma-sweep`: added for post-training threshold sweeps without retraining.",
        "- `--time-scale-init-grid`: added for the bounded init/alpha diagnostic grid.",
        "",
    ]
    lines.extend(_result_lines("Force-Current Result", force_row))
    lines.extend(_result_lines("Topk=1 Result", topk_row))
    lines.extend(
        [
            "## Gamma Sweep Summary",
            f"- Sweep rows: {len(gamma_rows)}.",
            f"- Smallest gamma that selects expert: {min(selected_gammas) if selected_gammas else None}.",
            f"- Largest gamma that still selects expert: {max(selected_gammas) if selected_gammas else None}.",
            f"- Paper gamma 0.5 selects expert: {threshold_05_selected}.",
            "",
        ]
    )
    if selected_gamma_rows:
        lines.append("| gamma | selected ids | residual norm | target NLL | target NLL delta | rank delta | top score |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for row in selected_gamma_rows:
            lines.append(
                "| {gamma} | {ids} | {residual} | {nll} | {delta} | {rank_delta} | {score} |".format(
                    gamma=row.get("sweep_gamma"),
                    ids=row.get("selected_expert_ids"),
                    residual=row.get("residual_norm"),
                    nll=row.get("target_nll"),
                    delta=row.get("target_nll_delta"),
                    rank_delta=row.get("target_rank_delta"),
                    score=row.get("top_score"),
                )
            )
        lines.append("")
    if grid_rows:
        lines.extend(["## Scale/Init Grid Summary", ""])
        lines.append("| init_std | alpha | force residual | topk residual | NLL delta | topk NLL delta | threshold 0.5 selected | top score |")
        lines.append("|---:|---:|---:|---:|---:|---:|---|---:|")
        for row in grid_rows:
            lines.append(
                "| {init_std} | {alpha} | {force_residual} | {topk_residual} | {delta} | {topk_delta} | {selected} | {score} |".format(
                    init_std=row.get("init_std"),
                    alpha=row.get("alpha"),
                    force_residual=row.get("force_current_residual_norm"),
                    topk_residual=row.get("topk_residual_norm"),
                    delta=row.get("final_target_nll_delta"),
                    topk_delta=row.get("topk_target_nll_delta"),
                    selected=row.get("threshold_gamma_0_5_selected"),
                    score=row.get("final_score"),
                )
            )
        lines.append("")
    else:
        lines.extend(["## Scale/Init Grid Summary", "- Not run or not available.", ""])
    lines.extend(["## Diagnosis", f"- Label: `{diagnosis}`.", ""])
    path = report_dir / "TIME_CALIBRATED_1EDIT_REPORT.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    args = parse_args()
    if args.mode in {"nonseq", "sequential"} and args.max_edits < 5:
        args.max_edits = 5
    if args.mode == "one":
        args.max_edits = 1
    set_seeds(args.seed)
    ensure_offline_env()

    dataset_path = resolve_dataset_path(args.dataset, Path.cwd(), args.dataset_path)
    records = load_records(dataset_path, args.sample_index, args.max_edits)
    config = configure(args, dataset_path)
    image_root = Path(config.coco_image).expanduser()
    device = torch_device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    model = get_model(config).to(device).eval()
    samples = [make_sample(model, record, image_root) for record in records]
    if args.time_scale_init_grid:
        grid_rows = run_scale_init_grid(args, config, dataset_path, records, samples, model, device)
        report_path = write_calibrated_report(args.out_dir)
        print(json.dumps(to_jsonable({"grid_rows": grid_rows, "calibrated_report": str(report_path)}), indent=2, sort_keys=True))
    else:
        alg = TIMEEdit(model, config, lambda: None).to(device)
        try:
            summary = run_smoke(args, config, dataset_path, records, samples, alg, args.out_dir, print_summary=False)
            gamma_rows = run_gamma_sweep(args, alg, records, samples, args.out_dir) if args.time_gamma_sweep else []
            report_path = write_calibrated_report(args.out_dir)
            payload = dict(summary)
            if gamma_rows:
                payload["gamma_sweep_rows"] = gamma_rows
            payload["calibrated_report"] = str(report_path)
            print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
        finally:
            alg.remove_hook()
            del alg

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
