#!/usr/bin/env python3
"""Evaluation-only Stage-0 loss/generation audit for the frozen ENGRAM V2 bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import ensure_offline_env, to_jsonable  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram_v2 import SequentialEngramBankV2  # noqa: E402
from easyeditor.trainer.models import get_model  # noqa: E402
from scripts.engram.run_engram_continual_v2 import build_views, set_determinism  # noqa: E402
from scripts.engram.run_engram_v2_10edit_generation_gate import generate as production_generate  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import (  # noqa: E402
    CanonicalInputs,
    assert_no_gold_leakage,
    build_canonical_inputs,
    first_supervised_position,
    ids_sha256,
    incremental_mean_nll,
    manual_greedy_trace,
    medical_answer_match,
    model_next_logits,
    normalized_candidate_score,
    score_target_incrementally,
    shared_generation_budget,
    tensor_sha256,
)

ORDER = ["953", "1293", "1592", "2174", "1628", "942", "1382", "1333", "671", "1343"]
BANK_ROOT = ROOT / "outputs/engram_v2_10edit_gate_20260710/run/bank"
DATASET_PATH = ROOT / "datasets/MedMKEB/eval.json"
MODEL_CONFIG = ROOT / "hparams/ENGRAM/llava_med_continual_v1.yaml"
MODULE_NAME = "llava_model.model.layers.21.self_attn.q_proj"
MODULE_KEY = MODULE_NAME + ".weight"
SEED = 42
TOP_K = 5
REPEATED_FORWARD_COUNT = 5
PROTOCOL_VERSION = "ENGRAM_V2_STAGE0_GENERATION_AUDIT_V1"
REQUIRED_OUTPUTS = (
    "manifest.json",
    "records.jsonl",
    "per_token_teacher_forced.jsonl",
    "free_running_trajectories.jsonl",
    "generation_outputs.csv",
    "candidate_scores.csv",
    "locality_metrics.csv",
    "retention_triangle.csv",
    "reproducibility_checks.json",
    "rollback_checks.json",
    "stage0_summary.json",
    "STAGE0_AUDIT_REPORT.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("primary", "fresh", "finalize"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--physical-gpu", default=0, type=int)
    parser.add_argument(
        "--uniform-cap",
        default=None,
        type=int,
        help="Optional predeclared uniform cap for every record/view; never target-dependent.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode) as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(to_jsonable(dict(payload)), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bank_manifest() -> Dict[str, Any]:
    rows = []
    for path in sorted(item for item in BANK_ROOT.rglob("*") if item.is_file()):
        rows.append({"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": rows, "sha256": canonical_hash(rows)}


def source_manifest() -> Dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/engram/stage0_generation_audit_utils.py",
        ROOT / "tests/test_engram_v2_stage0_generation_audit.py",
        ROOT / "scripts/engram/run_engram_v2_10edit_generation_gate.py",
        ROOT / "scripts/engram/run_engram_continual_v2.py",
        ROOT / "easyeditor/models/engram_v2/bank.py",
    ]
    rows = [{"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]
    return {"files": rows, "sha256": canonical_hash(rows)}


def eos_ids(model: Any) -> List[int]:
    value = model.llava_tokenizer.eos_token_id
    if value is None:
        return []
    return [int(item) for item in value] if isinstance(value, (list, tuple)) else [int(value)]


def load_model_views_bank(physical_gpu: int) -> Tuple[Any, Dict[str, Dict[str, Dict[str, Any]]], SequentialEngramBankV2, Dict[str, Dict[str, Any]]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly {physical_gpu}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    ensure_offline_env()
    set_determinism(SEED)
    config = EngramMultimodalHparams.from_hparams(str(MODEL_CONFIG))
    config.dropout, config.no_grad_layers, config.device = 0.0, None, "cuda"
    model = get_model(config).to(torch.device("cuda")).eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("Model is not fully in eval mode")
    records = {str(row["id"]): row for row in json.loads(DATASET_PATH.read_text())}
    if any(record_id not in records for record_id in ORDER):
        raise RuntimeError("Fixed MedMKEB record order is incomplete")
    image_root = Path(config.coco_image)
    if not image_root.is_absolute():
        image_root = ROOT / image_root
    views = {record_id: build_views(model, records[record_id], image_root) for record_id in ORDER}
    bank = SequentialEngramBankV2(BANK_ROOT)
    edits = bank.list_edits()
    if [str(item["source_example_ids"][0]) for item in edits] != ORDER:
        raise RuntimeError("Frozen bank order differs from Stage-0 fixed order")
    module = dict(model.named_modules()).get(MODULE_NAME)
    if not isinstance(module, torch.nn.Linear):
        raise RuntimeError(f"Frozen ENGRAM module is missing: {MODULE_NAME}")
    expected_anchor = bank.anchor_state()[MODULE_KEY].to(dtype=module.weight.dtype)
    if not torch.equal(module.weight.detach().cpu(), expected_anchor):
        raise RuntimeError("Loaded model does not equal the frozen S0 anchor")
    return model, views, bank, records


def state_weight_hash(model: Any) -> str:
    module = dict(model.named_modules())[MODULE_NAME]
    return tensor_sha256(module.weight)


def apply_prefix(model: Any, bank: SequentialEngramBankV2, prefix: int) -> Dict[str, Any]:
    hashes = bank.rollback_to_prefix(model, int(prefix))
    expected = bank.assemble_state([item["edit_id"] for item in bank.list_edits()[:prefix]])[MODULE_KEY]
    module = dict(model.named_modules())[MODULE_NAME]
    observed = module.weight.detach().cpu()
    expected_quantized = expected.to(dtype=observed.dtype)
    equal = bool(torch.equal(observed, expected_quantized))
    if not equal:
        raise RuntimeError(f"State prefix S{prefix} did not reconstruct exactly")
    return {"state_id": f"S{prefix}", "prefix": prefix, "module_weight_hash": state_weight_hash(model), "bank_hashes": hashes, "exact": equal}


@torch.inference_mode()
def loss_from_canonical(model: Any, canonical: CanonicalInputs) -> Tuple[float, torch.Tensor, int]:
    labels = canonical.full_ids.clone()
    labels[:, : canonical.answer_start] = model.IGNORE_INDEX
    labels[canonical.full_ids.eq(model.IMAGE_TOKEN_INDEX)] = model.IGNORE_INDEX
    supervised = first_supervised_position(labels, model.IGNORE_INDEX)
    attention = torch.ones_like(canonical.full_ids, dtype=torch.long, device=canonical.full_ids.device)
    output = model.llava_model(
        input_ids=canonical.full_ids,
        images=canonical.image,
        attention_mask=attention,
        labels=labels,
        return_dict=True,
        use_cache=False,
    )
    return float(output.loss.item()), labels, supervised if supervised is not None else -1


@torch.inference_mode()
def strict_generate(model: Any, canonical: CanonicalInputs, max_new_tokens: int) -> Dict[str, Any]:
    assert_no_gold_leakage(canonical.prompt_ids, canonical)
    attention = torch.ones_like(canonical.prompt_ids, dtype=torch.long, device=canonical.prompt_ids.device)
    kwargs = {
        "images": canonical.image,
        "attention_mask": attention,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": False,
        "min_new_tokens": 0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "bad_words_ids": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": model.llava_tokenizer.pad_token_id,
        "eos_token_id": model.llava_tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    output = model.llava_model.generate(canonical.prompt_ids, **kwargs)
    sequences = output.sequences
    prompt_length = int(canonical.prompt_ids.shape[1])
    if sequences.shape[1] >= prompt_length and torch.equal(sequences[:, :prompt_length], canonical.prompt_ids):
        generated = sequences[:, prompt_length:]
    else:
        generated = sequences
    token_ids = [int(item) for item in generated[0].detach().cpu().tolist()]
    first_scores = output.scores[0][0].float() if getattr(output, "scores", None) else None
    raw_first_logits = model_next_logits(model, canonical.prompt_ids, canonical.image)
    return {
        "token_ids": token_ids,
        "raw_output": model.llava_tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
        "first_token": token_ids[0] if token_ids else None,
        "first_score_top1": int(first_scores.argmax().item()) if first_scores is not None else None,
        "first_scores_hash": tensor_sha256(first_scores) if first_scores is not None else None,
        "first_scores_raw_max_abs_diff": float((first_scores - raw_first_logits).abs().max().item()) if first_scores is not None else None,
        "score_fallback": "output_scores=True; output_logits unavailable/not requested",
        "config": kwargs | {"images": "<canonical_pixel_tensor>", "attention_mask": "<all_ones>"},
    }


@torch.inference_mode()
def beam_generate(model: Any, canonical: CanonicalInputs, max_new_tokens: int) -> List[Dict[str, Any]]:
    assert_no_gold_leakage(canonical.prompt_ids, canonical)
    attention = torch.ones_like(canonical.prompt_ids, dtype=torch.long, device=canonical.prompt_ids.device)
    output = model.llava_model.generate(
        canonical.prompt_ids,
        images=canonical.image,
        attention_mask=attention,
        do_sample=False,
        num_beams=4,
        num_return_sequences=4,
        use_cache=False,
        min_new_tokens=0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        bad_words_ids=None,
        suppress_tokens=None,
        begin_suppress_tokens=None,
        length_penalty=1.0,
        early_stopping=True,
        max_new_tokens=int(max_new_tokens),
        pad_token_id=model.llava_tokenizer.pad_token_id,
        eos_token_id=model.llava_tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    rows = []
    prompt_length = int(canonical.prompt_ids.shape[1])
    scores = getattr(output, "sequences_scores", None)
    for index, sequence in enumerate(output.sequences):
        tokens = sequence[prompt_length:] if sequence.shape[0] >= prompt_length and torch.equal(sequence[:prompt_length], canonical.prompt_ids[0]) else sequence
        token_ids = [int(item) for item in tokens.detach().cpu().tolist()]
        rows.append({
            "beam_index": index,
            "token_ids": token_ids,
            "raw_output": model.llava_tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
            "sequence_score": float(scores[index].item()) if scores is not None else None,
        })
    return rows


def repeated_loss_and_margin(model: Any, canonical: CanonicalInputs, target_id: int) -> Dict[str, Any]:
    losses, margins, top1 = [], [], []
    for _ in range(REPEATED_FORWARD_COUNT):
        loss, _labels, _supervised = loss_from_canonical(model, canonical)
        logits = model_next_logits(model, canonical.prompt_ids, canonical.image)
        target_logit = logits[int(target_id)]
        competitor = torch.topk(logits, 2).values
        best = competitor[1] if int(logits.argmax().item()) == int(target_id) else competitor[0]
        losses.append(loss)
        margins.append(float((target_logit - best).item()))
        top1.append(int(logits.argmax().item()))
    return {
        "losses": losses,
        "loss_jitter": max(losses) - min(losses),
        "margins": margins,
        "margin_jitter": max(margins) - min(margins),
        "top1_ids": top1,
        "top1_stable": len(set(top1)) == 1,
    }


def clone_sample_with_target(sample: Mapping[str, Any], target: str, model: Any) -> Dict[str, Any]:
    copied = {key: value for key, value in sample.items()}
    prompt = str(sample["prompt"][0])
    copied["target"] = [str(target)]
    copied["text_input"] = [prompt + str(target)]
    copied["labels"] = model.llava_tokenizer(str(target), add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return copied


def candidate_set(record: Mapping[str, Any], target: str, aliases: Sequence[str], pre_edit_greedy: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = [("canonical_target", target)]
    rows.extend((f"accepted_alias_{index}", value) for index, value in enumerate(aliases))
    old = str(record.get("pred") or "").strip()
    if old:
        rows.append(("old_answer", old))
    if pre_edit_greedy.strip():
        rows.append(("pre_edit_greedy", pre_edit_greedy.strip()))
    for index, value in enumerate(record.get("hard_negative_answers") or []):
        rows.append((f"hard_negative_{index}", str(value)))
    deduplicated: List[Tuple[str, str]] = []
    seen = set()
    for name, value in rows:
        key = value.strip()
        if key and key not in seen:
            seen.add(key)
            deduplicated.append((name, key))
    return deduplicated


def evaluate_one(
    model: Any,
    sample: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    record_id: str,
    view: str,
    state_id: str,
    state_role: str,
    max_new_tokens: int,
    pre_edit_greedy: str = "",
    run_secondary: bool = True,
) -> Dict[str, Any]:
    canonical = build_canonical_inputs(model, sample)
    target = str(sample["target"][0])
    aliases = [str(item) for item in (record.get("accepted_answers") or [])]
    eos = eos_ids(model)
    per_token = score_target_incrementally(model, canonical, eos, top_k=TOP_K)
    model_loss, labels, first_supervised = loss_from_canonical(model, canonical)
    repeated = repeated_loss_and_margin(model, canonical, int(canonical.target_ids[0]))
    incremental_nll = incremental_mean_nll(per_token)
    tolerance = max(5e-4, 10.0 * float(repeated["loss_jitter"]))
    raw_logits = model_next_logits(model, canonical.prompt_ids, canonical.image)
    raw_first = int(raw_logits.argmax().item())
    raw_hash = tensor_sha256(raw_logits)
    manual = manual_greedy_trace(model, canonical, max_new_tokens, eos, top_k=TOP_K)
    strict = strict_generate(model, canonical, max_new_tokens)
    parity = bool(
        manual["token_ids"]
        and strict["first_token"] is not None
        and raw_first == int(manual["token_ids"][0]) == int(strict["first_token"])
        and strict["first_score_top1"] in (None, raw_first)
    )
    match = medical_answer_match(
        manual["raw_output"],
        target,
        aliases=aliases,
        required_terms=record.get("required_terms") or [],
        forbidden_terms=record.get("forbidden_terms") or [],
        required_polarity=record.get("required_polarity"),
        required_laterality=record.get("required_laterality"),
    )
    first_divergence = manual["first_divergence"]
    first_decisive = first_divergence
    decisive_row = per_token[first_decisive] if first_decisive is not None and first_decisive < len(per_token) else None
    beam_rows: List[Dict[str, Any]] = []
    production: Dict[str, Any] = {"skipped": True}
    candidates: List[Dict[str, Any]] = []
    if run_secondary and not manual["cap_hit"] and parity:
        beam_rows = beam_generate(model, canonical, max_new_tokens)
        production = production_generate(model, dict(sample))
        for name, value in candidate_set(record, target, aliases, pre_edit_greedy):
            candidate_sample = clone_sample_with_target(sample, value, model)
            candidate_canonical = build_canonical_inputs(model, candidate_sample)
            token_rows = score_target_incrementally(model, candidate_canonical, eos, top_k=TOP_K)
            candidates.append({"candidate_type": name, "candidate": value, "score": normalized_candidate_score(token_rows)})
    beam_matches = [medical_answer_match(item["raw_output"], target, aliases=aliases) for item in beam_rows]
    return {
        "record": {
            "edit_id": record_id,
            "record_id": record_id,
            "view": view,
            "state_id": state_id,
            "state_role": state_role,
            "prompt_hash": canonical.prompt_hash,
            "full_hash": canonical.full_hash,
            "pixel_hash": canonical.pixel_hash,
            "prompt_prefix_match": True,
            "answer_start": canonical.answer_start,
            "first_supervised_position": first_supervised,
            "first_target_token_supervised": first_supervised == canonical.answer_start,
            "labels_hash": ids_sha256(labels),
            "model_loss": model_loss,
            "incremental_mean_nll": incremental_nll,
            "loss_abs_diff": abs(model_loss - incremental_nll),
            "loss_tolerance": tolerance,
            "loss_matches": abs(model_loss - incremental_nll) <= tolerance,
            "repeated_forward": repeated,
            "raw_logits_hash": raw_hash,
            "raw_first_token": raw_first,
            "manual_first_token": manual["token_ids"][0] if manual["token_ids"] else None,
            "generate_first_token": strict["first_token"],
            "raw_manual_generate_first_token_parity": parity,
            "first_target_rank": per_token[0]["target_rank"],
            "first_target_margin": per_token[0]["margin"],
            "first_decisive_index": first_decisive,
            "first_decisive_rank": decisive_row["target_rank"] if decisive_row else None,
            "first_decisive_margin": decisive_row["margin"] if decisive_row else None,
            "teacher_forced_top1_fraction": sum(row["target_rank"] == 1 for row in per_token) / len(per_token),
            "minimum_target_margin": min(float(row["margin"]) for row in per_token),
            "first_non_top1_target_position": next((row["token_index"] for row in per_token if row["target_rank"] > 1), None),
            "first_free_generation_divergence": first_divergence,
            "greedy_output": manual["raw_output"],
            **match,
            "beam_top1_match": bool(beam_matches and beam_matches[0]["clinical_constraint_match"]),
            "beam_any_match": any(item["clinical_constraint_match"] for item in beam_matches),
            "candidate_winner": max(candidates, key=lambda item: item["score"])["candidate_type"] if candidates else None,
            "target_vs_old_candidate_margin": candidate_margin(candidates, "canonical_target", "old_answer"),
            "eos_step": manual["eos_step"],
            "stop_reason": manual["stop_reason"],
            "cap_hit": manual["cap_hit"],
            "early_eos_failure": manual["early_eos_failure"],
            "termination_failure": manual["termination_failure"],
            "repeated_trigram_count": manual["repeated_trigram_count"],
        },
        "per_token": per_token,
        "trajectory": manual,
        "strict": strict,
        "beam": beam_rows,
        "production": production,
        "candidates": candidates,
        "canonical": canonical,
    }


def candidate_margin(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> Optional[float]:
    keyed = {str(row["candidate_type"]): float(row["score"]) for row in rows}
    return keyed[left] - keyed[right] if left in keyed and right in keyed else None


def critical_alignment_pass(row: Mapping[str, Any]) -> bool:
    return bool(
        row["prompt_prefix_match"]
        and row["first_target_token_supervised"]
        and row["loss_matches"]
        and row["repeated_forward"]["top1_stable"]
        and row["raw_manual_generate_first_token_parity"]
        and not row["cap_hit"]
    )


def save_evaluation(out_dir: Path, result: Mapping[str, Any]) -> None:
    record = dict(result["record"])
    append_jsonl(out_dir / "records.jsonl", record)
    for row in result["per_token"]:
        append_jsonl(out_dir / "per_token_teacher_forced.jsonl", {"record_id": record["record_id"], "view": record["view"], "state_id": record["state_id"], **row})
    append_jsonl(out_dir / "free_running_trajectories.jsonl", {
        "record_id": record["record_id"], "view": record["view"], "state_id": record["state_id"], **result["trajectory"],
    })
    for candidate in result["candidates"]:
        append_jsonl(out_dir / "candidate_scores.jsonl", {"record_id": record["record_id"], "view": record["view"], "state_id": record["state_id"], **candidate})
    append_jsonl(out_dir / "generation_outputs.jsonl", {
        "record_id": record["record_id"], "view": record["view"], "state_id": record["state_id"],
        "strict": result["strict"], "manual": {key: value for key, value in result["trajectory"].items() if key != "trajectory"},
        "beam": result["beam"], "production": result["production"],
    })


def initialize_manifest(out_dir: Path, physical_gpu: int, budget: int, records: Mapping[str, Mapping[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    bank = SequentialEngramBankV2(BANK_ROOT)
    record_manifest = []
    for record_id in ORDER:
        record = records[record_id]
        record_manifest.append({
            "record_id": record_id,
            "question": str(record["src"]),
            "target": str(record["alt"]),
            "old_answer": str(record.get("pred") or ""),
            "accepted_answers": record.get("accepted_answers") or [],
            "annotation_coverage_incomplete": not bool(record.get("accepted_answers")),
        })
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_before_model_evaluation": True,
        "fixed_order": ORDER,
        "seed": SEED,
        "physical_gpu": physical_gpu,
        "max_new_tokens": budget,
        "bank_root": str(BANK_ROOT),
        "bank_manifest": bank_manifest(),
        "anchor_hash": json.loads((BANK_ROOT / "index.json").read_text())["anchor_hash"],
        "final_state_hash": bank.list_edits()[-1]["resulting_state_hash"],
        "source_manifest": source_manifest(),
        "model_config": str(MODEL_CONFIG),
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "records": record_manifest,
        "strict_generation": {
            "do_sample": False, "num_beams": 1, "use_cache": False, "min_new_tokens": 0,
            "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "custom_stopping_criteria": None,
        },
        "production_generation_differences": {
            "max_new_tokens": 16, "use_cache": True, "temperature_argument": 0.0,
            "note": "Production path is reported separately and never counted as strict Stage-0 generation.",
        },
        "constraints": {"training": False, "editing_rerun": False, "hyperparameter_change": False, "stage1_launched": False},
    }
    write_json(out_dir / "manifest.json", manifest)


def primary(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to reuse Stage-0 output directory: {args.out_dir}")
    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    aliases_and_targets = []
    for record_id in ORDER:
        aliases_and_targets.append(str(records[record_id]["alt"]))
        aliases_and_targets.extend(str(item) for item in (records[record_id].get("accepted_answers") or []))
    computed_budget = shared_generation_budget(model.llava_tokenizer, aliases_and_targets)
    budget = int(args.uniform_cap) if args.uniform_cap is not None else computed_budget
    if budget < computed_budget:
        raise ValueError(f"Uniform cap {budget} is below the preregistered minimum {computed_budget}")
    initialize_manifest(args.out_dir, args.physical_gpu, budget, records)

    checks: List[Dict[str, Any]] = []
    state_checks: List[Dict[str, Any]] = []
    first_id = ORDER[0]
    state_checks.append(apply_prefix(model, bank, 0))
    s0 = evaluate_one(
        model, views[first_id]["target"], records[first_id], record_id=first_id, view="target",
        state_id="S0", state_role="base_smoke_pre_edit", max_new_tokens=budget, run_secondary=True,
    )
    save_evaluation(args.out_dir, s0)
    state_checks.append(apply_prefix(model, bank, 1))
    s1 = evaluate_one(
        model, views[first_id]["target"], records[first_id], record_id=first_id, view="target",
        state_id="S1", state_role="first_edit_post_edit", max_new_tokens=budget,
        pre_edit_greedy=s0["record"]["greedy_output"], run_secondary=True,
    )
    save_evaluation(args.out_dir, s1)
    checks.extend([s0["record"], s1["record"]])

    smoke_pass = all(critical_alignment_pass(row) for row in checks)
    hard_stop_reasons = collect_hard_stop_reasons(checks)
    full_audit_ran = False
    if smoke_pass:
        full_audit_ran = True
        existing = {(row["record_id"], row["state_id"], row["view"]) for row in checks}
        pre_outputs: Dict[str, str] = {first_id: s0["record"]["greedy_output"]}
        for edit_index, record_id in enumerate(ORDER, start=1):
            roles = ((edit_index - 1, "pre_edit"), (edit_index, "post_edit"), (10, "final_retention"))
            for prefix, role in roles:
                state_checks.append(apply_prefix(model, bank, prefix))
                key = (record_id, f"S{prefix}", "target")
                if key not in existing:
                    result = evaluate_one(
                        model, views[record_id]["target"], records[record_id], record_id=record_id, view="target",
                        state_id=f"S{prefix}", state_role=role, max_new_tokens=budget,
                        pre_edit_greedy=pre_outputs.get(record_id, ""), run_secondary=True,
                    )
                    save_evaluation(args.out_dir, result)
                    checks.append(result["record"])
                    existing.add(key)
                    if role == "pre_edit":
                        pre_outputs[record_id] = result["record"]["greedy_output"]
                    if not critical_alignment_pass(result["record"]):
                        hard_stop_reasons = collect_hard_stop_reasons([result["record"]])
                        full_audit_ran = False
                        break
                locality_key = (record_id, f"S{prefix}", "locality")
                if locality_key not in existing:
                    locality = evaluate_one(
                        model, views[record_id]["locality"], records[record_id], record_id=record_id, view="locality",
                        state_id=f"S{prefix}", state_role=role, max_new_tokens=budget,
                        pre_edit_greedy="", run_secondary=False,
                    )
                    save_evaluation(args.out_dir, locality)
                    checks.append(locality["record"])
                    existing.add(locality_key)
                    if not critical_alignment_pass(locality["record"]):
                        hard_stop_reasons = collect_hard_stop_reasons([locality["record"]])
                        full_audit_ran = False
                        break
            if hard_stop_reasons:
                break

    anchor_check = apply_prefix(model, bank, 0)
    rollback_match = anchor_check["module_weight_hash"] == state_checks[0]["module_weight_hash"]
    rollback = {
        "anchor_module_hash_before": state_checks[0]["module_weight_hash"],
        "anchor_module_hash_after": anchor_check["module_weight_hash"],
        "anchor_restored": rollback_match,
        "state_checks": state_checks,
    }
    write_json(args.out_dir / "rollback_checks.json", rollback)
    write_json(args.out_dir / "reproducibility_checks.json", {
        "completed": False,
        "reason": "Fresh-process validation is a separate gated mode and was not yet run.",
    })
    summary = summarize_primary(checks, smoke_pass, full_audit_ran, hard_stop_reasons, rollback_match)
    write_json(args.out_dir / "stage0_summary.json", summary)
    materialize_csv_outputs(args.out_dir)
    write_json(args.out_dir / "primary_complete.json", {"summary_sha256": sha256_file(args.out_dir / "stage0_summary.json")})
    render_report(args.out_dir)
    finalize_hashes(args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


def collect_hard_stop_reasons(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    reasons = []
    for row in rows:
        key = f"{row['record_id']}:{row['state_id']}:{row['view']}"
        if not row["prompt_prefix_match"]:
            reasons.append(f"{key}:prompt_prefix_mismatch")
        if not row["first_target_token_supervised"]:
            reasons.append(f"{key}:first_target_not_supervised")
        if not row["loss_matches"]:
            reasons.append(f"{key}:incremental_nll_model_loss_mismatch")
        if not row["repeated_forward"]["top1_stable"]:
            reasons.append(f"{key}:repeated_forward_top1_changed")
        if not row["raw_manual_generate_first_token_parity"]:
            reasons.append(f"{key}:raw_manual_generate_mismatch")
        if row["cap_hit"]:
            reasons.append(f"{key}:strict_generation_cap_hit")
    return sorted(set(reasons))


def diagnosis_labels(rows: Sequence[Mapping[str, Any]], hard_stops: Sequence[str]) -> List[str]:
    labels = []
    if any(any(token in reason for token in ("prompt_prefix", "supervised", "loss_mismatch")) for reason in hard_stops):
        labels.append("EVALUATOR_ALIGNMENT_FAILURE")
    if any("raw_manual_generate" in reason for reason in hard_stops):
        labels.append("GENERATION_PATH_FAILURE")
    if any("cap_hit" in reason for reason in hard_stops):
        labels.append("INCONCLUSIVE_DUE_TO_INVALID_RUN")
    target_rows = [row for row in rows if row.get("view") == "target" and not row.get("cap_hit")]
    if target_rows and any((row.get("first_decisive_rank") or 1) > 1 for row in target_rows):
        labels.append("INSUFFICIENT_TARGET_MARGIN")
    if target_rows and any((row.get("first_non_top1_target_position") or 0) > 0 for row in target_rows):
        labels.append("PREFIX_TOKEN_MARGIN_FAILURE")
    if len(target_rows) >= 2:
        s0 = next((row for row in target_rows if row["record_id"] == ORDER[0] and row["state_id"] == "S0"), None)
        s1 = next((row for row in target_rows if row["record_id"] == ORDER[0] and row["state_id"] == "S1"), None)
        if s0 and s1 and float(s1["first_target_margin"]) <= float(s0["first_target_margin"]):
            labels.append("UPDATE_DIRECTION_MISMATCH")
    return list(dict.fromkeys(labels or ["INCONCLUSIVE_DUE_TO_INVALID_RUN"]))


def summarize_primary(
    rows: Sequence[Mapping[str, Any]], smoke_pass: bool, full_audit_ran: bool,
    hard_stops: Sequence[str], rollback_pass: bool,
) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "records_evaluated": len(rows),
        "base_and_first_edit_smoke_pass": smoke_pass,
        "fixed_ten_edit_audit_completed": full_audit_ran and not hard_stops,
        "hard_stop_reasons": list(hard_stops),
        "diagnosis_labels": diagnosis_labels(rows, hard_stops),
        "alignment_all_pass": all(bool(row["prompt_prefix_match"] and row["first_target_token_supervised"] and row["loss_matches"]) for row in rows),
        "first_token_parity_all_pass": all(bool(row["raw_manual_generate_first_token_parity"]) for row in rows),
        "strict_cap_hit_count": sum(bool(row["cap_hit"]) for row in rows),
        "unrestricted_generation_clinical_match_count": sum(bool(row["clinical_constraint_match"]) for row in rows if row["view"] == "target" and not row["cap_hit"]),
        "rollback_pass": rollback_pass,
        "fresh_process_completed": False,
        "training": False,
        "editing_rerun": False,
        "stage1_launched": False,
    }


def materialize_csv_outputs(out_dir: Path) -> None:
    records = read_jsonl(out_dir / "records.jsonl")
    generations = read_jsonl(out_dir / "generation_outputs.jsonl")
    candidates = read_jsonl(out_dir / "candidate_scores.jsonl")
    generation_rows = []
    for item in generations:
        manual = item.get("manual", {})
        generation_rows.append({
            "record_id": item["record_id"], "view": item["view"], "state_id": item["state_id"],
            "strict_output": item.get("strict", {}).get("raw_output", ""),
            "manual_output": manual.get("raw_output", ""), "stop_reason": manual.get("stop_reason"),
            "cap_hit": manual.get("cap_hit"), "production_output": item.get("production", {}).get("raw_output", ""),
        })
    write_csv(out_dir / "generation_outputs.csv", generation_rows, ["record_id", "view", "state_id", "strict_output", "manual_output", "stop_reason", "cap_hit", "production_output"])
    write_csv(out_dir / "candidate_scores.csv", candidates, ["record_id", "view", "state_id", "candidate_type", "candidate", "score"])
    locality = [row for row in records if row.get("view") == "locality"]
    write_csv(out_dir / "locality_metrics.csv", locality, ["record_id", "state_id", "state_role", "model_loss", "incremental_mean_nll", "greedy_output", "clinical_constraint_match", "cap_hit"])
    target = [row for row in records if row.get("view") == "target"]
    write_csv(out_dir / "retention_triangle.csv", target, ["record_id", "state_id", "state_role", "model_loss", "incremental_mean_nll", "first_target_rank", "first_target_margin", "clinical_constraint_match", "cap_hit"])


def fresh(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out_dir / "manifest.json").read_text())
    if manifest["bank_manifest"]["sha256"] != bank_manifest()["sha256"]:
        raise RuntimeError("Frozen bank drifted before fresh-process validation")
    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    primary_rows = read_jsonl(args.out_dir / "records.jsonl")
    checks = []
    for row in primary_rows:
        prefix = int(str(row["state_id"])[1:])
        apply_prefix(model, bank, prefix)
        sample = views[row["record_id"]][row["view"]]
        canonical = build_canonical_inputs(model, sample)
        token_rows = score_target_incrementally(model, canonical, eos_ids(model), top_k=TOP_K)
        manual = manual_greedy_trace(model, canonical, int(manifest["max_new_tokens"]), eos_ids(model), top_k=TOP_K)
        checks.append({
            "record_id": row["record_id"], "view": row["view"], "state_id": row["state_id"],
            "prompt_hash_equal": canonical.prompt_hash == row["prompt_hash"],
            "pixel_hash_equal": canonical.pixel_hash == row["pixel_hash"],
            "incremental_nll_abs_diff": abs(incremental_mean_nll(token_rows) - float(row["incremental_mean_nll"])),
            "token_ids_equal": manual["token_ids"] == next(
                item["manual"]["token_ids"] for item in read_jsonl(args.out_dir / "generation_outputs.jsonl")
                if item["record_id"] == row["record_id"] and item["view"] == row["view"] and item["state_id"] == row["state_id"]
            ),
        })
    tolerance = max(float(row["loss_tolerance"]) for row in primary_rows)
    passed = all(
        row["prompt_hash_equal"] and row["pixel_hash_equal"] and row["token_ids_equal"] and row["incremental_nll_abs_diff"] <= tolerance
        for row in checks
    )
    write_json(
        args.out_dir / "reproducibility_checks.json",
        {"completed": True, "passed": passed, "tolerance": tolerance, "rows": checks},
        exclusive=False,
    )
    summary = json.loads((args.out_dir / "stage0_summary.json").read_text())
    summary["fresh_process_completed"] = True
    summary["fresh_process_pass"] = passed
    write_json(args.out_dir / "stage0_summary.json", summary, exclusive=False)
    render_report(args.out_dir, overwrite=True)
    finalize_hashes(args.out_dir, overwrite=True)
    print(json.dumps({"fresh_process_pass": passed, "rows": len(checks)}, indent=2))


def render_report(out_dir: Path, *, overwrite: bool = False) -> None:
    summary = json.loads((out_dir / "stage0_summary.json").read_text())
    records = read_jsonl(out_dir / "records.jsonl")
    lines = [
        "# ENGRAM V2 Stage-0 Loss–Generation Audit",
        "",
        "## Diagnosis",
        "",
        *[f"- `{label}`" for label in summary["diagnosis_labels"]],
        "",
        "## Gate status",
        "",
        f"- Base/S0–S1 smoke: `{summary['base_and_first_edit_smoke_pass']}`",
        f"- Fixed ten-edit audit completed: `{summary['fixed_ten_edit_audit_completed']}`",
        f"- Fresh-process completed: `{summary.get('fresh_process_completed', False)}`",
        f"- Rollback passed: `{summary['rollback_pass']}`",
        f"- Strict cap hits: `{summary['strict_cap_hit_count']}`",
        "",
        "## Hard stops",
        "",
        *([f"- `{item}`" for item in summary["hard_stop_reasons"]] or ["- None"]),
        "",
        "## Record evidence",
        "",
        "| record | state | view | prefix | supervised | loss match | parity | first rank | first margin | stop | cap | output |",
        "|---|---|---|:---:|:---:|:---:|:---:|---:|---:|---|:---:|---|",
    ]
    for row in records:
        output = str(row["greedy_output"]).replace("|", "\\|").replace("\n", " ")[:100]
        lines.append(
            f"| {row['record_id']} | {row['state_id']} | {row['view']} | {row['prompt_prefix_match']} | "
            f"{row['first_target_token_supervised']} | {row['loss_matches']} | {row['raw_manual_generate_first_token_parity']} | "
            f"{row['first_target_rank']} | {row['first_target_margin']:.6f} | {row['stop_reason']} | {row['cap_hit']} | {output} |"
        )
    lines.extend([
        "",
        "## Scope confirmation",
        "",
        "- Existing ENGRAM V2 bank states were only reconstructed and evaluated; they were not recreated or overwritten.",
        "- No editor training, alpha/layer/config change, CURE/LoRA run, sweep, or Stage-1 training was launched.",
        "- Beam and closed-set candidate results, when present, are secondary diagnostics and are not counted as unrestricted generation success.",
        "",
    ])
    path = out_dir / "STAGE0_AUDIT_REPORT.md"
    mode = "w" if overwrite else "x"
    with path.open(mode) as handle:
        handle.write("\n".join(lines))


def finalize_hashes(out_dir: Path, *, overwrite: bool = False) -> None:
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    outputs = []
    for path in sorted(item for item in out_dir.iterdir() if item.is_file() and item.name != "manifest.json"):
        outputs.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest["output_files"] = outputs
    manifest["output_manifest_sha256"] = canonical_hash(outputs)
    write_json(manifest_path, manifest, exclusive=False)


def finalize(args: argparse.Namespace) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if not (args.out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Stage-0 outputs are incomplete: {missing}")
    render_report(args.out_dir, overwrite=True)
    finalize_hashes(args.out_dir, overwrite=True)
    print(json.dumps({"status": "STAGE0_FINALIZED", "output_dir": str(args.out_dir)}, indent=2))


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.mode == "primary":
        primary(args)
    elif args.mode == "fresh":
        fresh(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
