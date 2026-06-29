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
        f"- Routing threshold/top-k: gamma={config.time_gamma}, topk={config.time_topk}.",
        f"- Mixing: {'average selected experts' if config.time_disable_score_mixing else 'softmax over selected intrinsic scores'} with tau={config.time_tau}.",
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


def main() -> None:
    args = parse_args()
    if args.mode in {"nonseq", "sequential"} and args.max_edits < 5:
        args.max_edits = 5
    if args.mode == "one":
        args.max_edits = 1
    set_seeds(args.seed)
    ensure_offline_env()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    routing_debug_path = args.out_dir / "routing_debug.jsonl"
    if routing_debug_path.exists():
        routing_debug_path.unlink()

    dataset_path = resolve_dataset_path(args.dataset, Path.cwd(), args.dataset_path)
    records = load_records(dataset_path, args.sample_index, args.max_edits)
    config = configure(args, dataset_path)
    image_root = Path(config.coco_image).expanduser()
    device = torch_device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    write_json(
        args.out_dir / "time_hparams.json",
        {
            "args": vars(args),
            "config": dict(config.__dict__),
            "scale_mode": config.time_scale_mode,
            "H_s1_s2_logged_after_model_load": True,
        },
    )

    model = get_model(config).to(device).eval()
    alg = TIMEEdit(model, config, lambda: None).to(device)
    samples = [make_sample(model, record, image_root) for record in records]
    hparams_payload = json.loads((args.out_dir / "time_hparams.json").read_text())
    hparams_payload.update({"hidden_size": alg.repository.hidden_size, "s1": alg.repository.s1, "s2": alg.repository.s2})
    write_json(args.out_dir / "time_hparams.json", hparams_payload)

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

    write_loss_trace(args.out_dir / "loss_trace.csv", loss_rows)
    summary = summarize_evals(args.mode, eval_rows, train_records, alg, args.out_dir)
    summary.update({"hidden_size": alg.repository.hidden_size, "s1": alg.repository.s1, "s2": alg.repository.s2})
    if args.mode == "one":
        summary_path = args.out_dir / "one_edit_summary.json"
    elif args.mode == "nonseq":
        summary_path = args.out_dir / "five_edit_nonseq_summary.json"
    else:
        summary_path = args.out_dir / "five_edit_seq_summary.json"
    write_json(summary_path, summary)
    write_report(args.out_dir, args, config, dataset_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))

    del alg
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
