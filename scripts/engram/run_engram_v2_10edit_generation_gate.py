#!/usr/bin/env python3
"""Evaluation-only free-generation gate for an existing ENGRAM V2 bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import (  # noqa: E402
    answer_fields,
    ensure_offline_env,
    normalize_medical_answer,
    to_jsonable,
)
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram_v2 import SequentialEngramBankV2  # noqa: E402
from easyeditor.trainer.models import get_model  # noqa: E402
from scripts.engram.run_engram_continual_v2 import build_views, set_determinism  # noqa: E402

ORDER = ["953", "1293", "1592", "2174", "1628", "942", "1382", "1333", "671", "1343"]
SEED = 42
PHYSICAL_GPU = 1
MAX_NEW_TOKENS = 16
OUT = ROOT / "outputs/engram_v2_10edit_generation_gate_20260710"
BANK_ROOT = ROOT / "outputs/engram_v2_10edit_gate_20260710/run/bank"
SOURCE_GATE = ROOT / "outputs/engram_v2_10edit_gate_20260710"
DATASET_PATH = ROOT / "datasets/MedMKEB/eval.json"
MODEL_CONFIG = ROOT / "hparams/ENGRAM/llava_med_continual_v1.yaml"
MODEL_CHECKPOINT = Path("/remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b")
VISION_CHECKPOINT = Path("/remote-home/wangbomin/hugging_cache/openai/clip-vit-large-patch14-336")
MODULE_NAME = "llava_model.model.layers.21.self_attn.q_proj"
MODULE_KEY = MODULE_NAME + ".weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "primary", "fresh", "finalize"))
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


def exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def exclusive_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(dict(row)), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tree_manifest(root: Path) -> Dict[str, Any]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        print(f"hashing {path}", flush=True)
        entries.append({
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {"root": str(root), "files": entries, "manifest_sha256": canonical_hash(entries)}


def file_manifest(paths: Iterable[Path], relative_to: Path = ROOT) -> Dict[str, Any]:
    entries = []
    for path in sorted(paths):
        entries.append({
            "path": str(path.relative_to(relative_to)) if path.is_relative_to(relative_to) else str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {"files": entries, "manifest_sha256": canonical_hash(entries)}


def prompt_text(question: Any) -> str:
    return f"Question: {str(question or '')} Short answer: "


def resolve_image(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / "datasets/MedMKEB" / path


def prompt_target_manifest() -> Dict[str, Any]:
    rows = {str(row["id"]): row for row in json.loads(DATASET_PATH.read_text())}
    records = []
    for record_id in ORDER:
        row = rows[record_id]
        for view, question_key, target_key, image_key in (
            ("target", "src", "alt", "image"),
            ("locality", "m_loc_q", "m_loc_a", "m_loc"),
        ):
            image = resolve_image(row[image_key])
            records.append({
                "record_id": record_id,
                "view": view,
                "prompt": prompt_text(row[question_key]),
                "target": str(row[target_key]),
                "image_path": str(image),
                "image_size": image.stat().st_size,
                "image_sha256": sha256_file(image),
            })
    return {"ordered_record_ids": ORDER, "records": records, "manifest_sha256": canonical_hash(records)}


def bank_files() -> List[Path]:
    return sorted(path for path in BANK_ROOT.rglob("*") if path.is_file())


def prepare_protocol() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to reuse generation gate directory: {OUT}")
    prompt_manifest = prompt_target_manifest()
    bank = SequentialEngramBankV2(BANK_ROOT)
    edits = bank.list_edits()
    if [str(item["source_example_ids"][0]) for item in edits] != ORDER:
        raise RuntimeError("Existing bank order does not match the fixed sequence")
    source_files = [
        Path(__file__).resolve(),
        ROOT / "scripts/same_edit/overfit_same_edit_one_medmkeb_edit.py",
        ROOT / "scripts/dsca_medmkeb_diag_common.py",
        ROOT / "scripts/engram/run_engram_continual_v2.py",
        ROOT / "easyeditor/models/engram_v2/bank.py",
        ROOT / "tests/test_engram_v2_generation_gate.py",
    ]
    protocol = {
        "protocol_version": "ENGRAM_V2_10EDIT_FREE_GENERATION_GATE_V1",
        "created_before_generation": True,
        "fixed_sequence": ORDER,
        "seed": SEED,
        "physical_gpu": PHYSICAL_GPU,
        "visible_device_inside_process": 0,
        "artifact_paths": {
            "source_gate": str(SOURCE_GATE),
            "bank": str(BANK_ROOT),
            "dataset": str(DATASET_PATH),
            "model_checkpoint": str(MODEL_CHECKPOINT),
            "vision_checkpoint": str(VISION_CHECKPOINT),
        },
        "state_hashes": {
            "anchor_state_hash": json.loads((BANK_ROOT / "index.json").read_text())["anchor_hash"],
            "final_state_hash": edits[-1]["resulting_state_hash"],
            "bank_file_manifest": file_manifest(bank_files()),
        },
        "model_hashes": {
            "model_checkpoint_byte_manifest": tree_manifest(MODEL_CHECKPOINT),
            "vision_checkpoint_byte_manifest": tree_manifest(VISION_CHECKPOINT),
            "model_config_sha256": sha256_file(MODEL_CONFIG),
        },
        "prompt_target_manifest_sha256": prompt_manifest["manifest_sha256"],
        "decoding": {
            "algorithm": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "temperature_note": "Passed by the reused SAME-Edit evaluator and inactive because do_sample=false.",
            "max_new_tokens": MAX_NEW_TOKENS,
            "use_cache": True,
            "batch_size": 1,
            "prompt_format": "Question: {question} Short answer: ",
            "conversation_template": "mistral_instruct",
        },
        "normalization": {
            "source": "scripts/dsca_medmkeb_diag_common.py:normalize_medical_answer/answer_fields",
            "rules": [
                "strip and lowercase",
                "remove leading the answer is, answer:, or it is",
                "replace punctuation with spaces",
                "collapse whitespace",
                "exact means normalized output equals normalized target",
                "contains means normalized target is a substring of normalized output",
            ],
        },
        "frozen_decision_thresholds": {
            "free_generation_transfer": "ENGRAM exact or contains count must strictly improve over base, with toward_count >= 1 and toward_count > away_count",
            "toward_away_score": "exact=2, contains=1, neither=0",
            "locality_min_token_sequence_agreement": "9/10",
            "locality_min_normalized_output_agreement": "9/10",
            "locality_target_correctness_losses_max": 0,
            "fresh_process_target_token_and_output_agreement": "10/10",
            "fresh_process_locality_token_and_output_agreement": "10/10",
        },
        "evaluator_sources": file_manifest(source_files),
        "existing_gate_report_sha256": sha256_file(SOURCE_GATE / "ENGRAM_V2_10EDIT_GATE_REPORT.md"),
        "prohibitions": [
            "editing rerun", "training", "20-edit", "multiple configurations", "decode sweep",
            "prompt or target tuning", "changed-setting retry", "algorithm modification", "GPU2", "commit", "push",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=False)
    exclusive_json(OUT / "PROMPT_TARGET_MANIFEST.json", prompt_manifest)
    exclusive_json(OUT / "GENERATION_PROTOCOL.json", protocol)
    print(json.dumps({
        "status": "PROTOCOL_FROZEN",
        "protocol_sha256": sha256_file(OUT / "GENERATION_PROTOCOL.json"),
        "prompt_target_manifest_sha256": prompt_manifest["manifest_sha256"],
        "physical_gpu": PHYSICAL_GPU,
        "max_new_tokens": MAX_NEW_TOKENS,
    }, indent=2), flush=True)


def require_frozen_protocol() -> Dict[str, Any]:
    protocol = json.loads((OUT / "GENERATION_PROTOCOL.json").read_text())
    if not protocol.get("created_before_generation"):
        raise RuntimeError("Protocol was not frozen before generation")
    if protocol["physical_gpu"] != PHYSICAL_GPU or protocol["decoding"]["max_new_tokens"] != MAX_NEW_TOKENS:
        raise RuntimeError("Frozen protocol constants changed")
    current_manifest = prompt_target_manifest()
    if current_manifest["manifest_sha256"] != protocol["prompt_target_manifest_sha256"]:
        raise RuntimeError("Prompt/target manifest drift detected")
    current_bank = file_manifest(bank_files())
    if current_bank["manifest_sha256"] != protocol["state_hashes"]["bank_file_manifest"]["manifest_sha256"]:
        raise RuntimeError("Existing bank artifact drift detected")
    return protocol


def load_model_and_views() -> tuple[Any, Dict[str, Dict[str, Dict[str, Any]]], SequentialEngramBankV2, Dict[str, Any]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly {PHYSICAL_GPU}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")
    ensure_offline_env()
    set_determinism(SEED)
    protocol = require_frozen_protocol()
    config = EngramMultimodalHparams.from_hparams(str(MODEL_CONFIG))
    config.dropout, config.no_grad_layers, config.device = 0.0, None, "cuda"
    model = get_model(config).to(torch.device("cuda")).eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("Model is not fully in evaluation mode")
    records = {str(row["id"]): row for row in json.loads(DATASET_PATH.read_text())}
    image_root = Path(config.coco_image)
    if not image_root.is_absolute():
        image_root = ROOT / image_root
    views = {record_id: build_views(model, records[record_id], image_root) for record_id in ORDER}
    bank = SequentialEngramBankV2(BANK_ROOT)
    module = dict(model.named_modules()).get(MODULE_NAME)
    if not isinstance(module, torch.nn.Linear):
        raise RuntimeError(f"Expected linear module missing: {MODULE_NAME}")
    anchor = bank.anchor_state()[MODULE_KEY]
    observed = module.weight.detach().cpu().float()
    if not torch.equal(observed, anchor):
        raise RuntimeError("Loaded base model does not equal the saved ENGRAM anchor")
    return model, views, bank, protocol


def prepare_generation_inputs(model: Any, sample: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = sample["prompt"][0]
    prompt_with_template = model._conversation_prompt(prompt, None)
    input_ids = model.tokenizer_image_token(
        prompt_with_template,
        model.llava_tokenizer,
        model.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.lm_device)
    image = model._image_for_row(sample, 0)
    return input_ids, image


def repeated_ngram(text: str, n: int = 2) -> bool:
    words = normalize_medical_answer(text).split()
    grams = [tuple(words[index:index + n]) for index in range(max(0, len(words) - n + 1))]
    return len(grams) != len(set(grams))


def generate(model: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
    input_ids, image = prepare_generation_inputs(model, sample)
    attention = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    with torch.inference_mode():
        output_ids = model.llava_model.generate(
            input_ids,
            images=image,
            attention_mask=attention,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            pad_token_id=model.llava_tokenizer.pad_token_id,
            eos_token_id=model.llava_tokenizer.eos_token_id,
        )
    prompt_length = int(input_ids.shape[1])
    generated = output_ids[:, prompt_length:] if output_ids.shape[1] >= prompt_length else output_ids
    token_ids = [int(value) for value in generated[0].detach().cpu().tolist()]
    raw = model.llava_tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    eos_id = model.llava_tokenizer.eos_token_id
    ended_with_eos = bool(token_ids and eos_id is not None and token_ids[-1] == int(eos_id))
    normalized = normalize_medical_answer(raw)
    return {
        "raw_output": raw,
        "normalized_output": normalized,
        "token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "ended_with_eos": ended_with_eos,
        "failure_flags": {
            "empty": not bool(raw.strip()),
            "truncation": len(token_ids) >= MAX_NEW_TOKENS and not ended_with_eos,
            "repetition": repeated_ngram(raw),
            "refusal": bool(re.search(r"\b(i cannot|i can't|unable to|cannot determine|not enough information)\b", raw.lower())),
            "formatting": bool(raw and not normalized),
        },
    }


def generation_rows(model: Any, views: Mapping[str, Mapping[str, Dict[str, Any]]], state: str) -> List[Dict[str, Any]]:
    rows = []
    for record_id in ORDER:
        for view in ("target", "locality"):
            sample = views[record_id][view]
            result = generate(model, sample)
            row = {
                "record_id": record_id,
                "view": view,
                "state": state,
                "prompt": sample["prompt"][0],
                "target": sample["target"][0],
                "generation_settings": {
                    "do_sample": False, "temperature": 0.0,
                    "max_new_tokens": MAX_NEW_TOKENS, "use_cache": True,
                },
                **result,
            }
            rows.append(row)
            print(json.dumps({
                "state": state, "record_id": record_id, "view": view,
                "raw_output": result["raw_output"], "token_count": result["generated_token_count"],
            }, ensure_ascii=False), flush=True)
    return rows


def verify_final_model(model: Any, bank: SequentialEngramBankV2) -> Dict[str, Any]:
    module = dict(model.named_modules())[MODULE_NAME]
    assembled = bank.assemble_state()[MODULE_KEY]
    expected = assembled.to(dtype=module.weight.dtype).cpu().float()
    observed = module.weight.detach().cpu().float()
    return {
        "model_matches_quantized_assembled_state": bool(torch.equal(observed, expected)),
        "bank_current_state_hash": bank.current_state_hash(),
        "final_index_state_hash": bank.list_edits()[-1]["resulting_state_hash"],
        "all_values_finite": bool(torch.isfinite(module.weight).all().item()),
    }


def run_primary() -> None:
    if (OUT / "primary_generation_records.jsonl").exists():
        raise FileExistsError("Primary generation already exists; retries are prohibited")
    model, views, bank, protocol = load_model_and_views()
    base_rows = generation_rows(model, views, "base")
    applied = bank.assemble_state_into_model(model)
    final_check = verify_final_model(model, bank)
    if not final_check["model_matches_quantized_assembled_state"] or not final_check["all_values_finite"]:
        raise RuntimeError(f"Final state verification failed: {final_check}")
    final_rows = generation_rows(model, views, "engram_v2_final_10edit")
    exclusive_jsonl(OUT / "primary_generation_records.jsonl", [*base_rows, *final_rows])
    exclusive_json(OUT / "primary_summary.json", {
        "physical_gpu": PHYSICAL_GPU,
        "protocol_sha256": sha256_file(OUT / "GENERATION_PROTOCOL.json"),
        "base_state_verified_against_anchor": True,
        "final_state_verification": final_check,
        "applied_module_hashes": applied,
        "editing_rerun": False,
        "generation_record_count": len(base_rows) + len(final_rows),
        "model_checkpoint_manifest_sha256": protocol["model_hashes"]["model_checkpoint_byte_manifest"]["manifest_sha256"],
    })


def run_fresh() -> None:
    if (OUT / "fresh_generation_records.jsonl").exists():
        raise FileExistsError("Fresh generation already exists; retries are prohibited")
    model, views, bank, _protocol = load_model_and_views()
    bank.assemble_state_into_model(model)
    final_check = verify_final_model(model, bank)
    if not final_check["model_matches_quantized_assembled_state"] or not final_check["all_values_finite"]:
        raise RuntimeError(f"Fresh final state verification failed: {final_check}")
    rows = generation_rows(model, views, "engram_v2_final_10edit_fresh_process")
    exclusive_jsonl(OUT / "fresh_generation_records.jsonl", rows)
    exclusive_json(OUT / "fresh_summary.json", {
        "physical_gpu": PHYSICAL_GPU,
        "protocol_sha256": sha256_file(OUT / "GENERATION_PROTOCOL.json"),
        "final_state_verification": final_check,
        "editing_rerun": False,
        "generation_record_count": len(rows),
    })


def exact_and_contains(output: str, target: str) -> tuple[bool, bool]:
    normalized_output = normalize_medical_answer(output)
    normalized_target = normalize_medical_answer(target)
    return (
        bool(normalized_output and normalized_target and normalized_output == normalized_target),
        bool(normalized_output and normalized_target and normalized_target in normalized_output),
    )


def correctness_score(exact: bool, contains: bool) -> int:
    return 2 if exact else (1 if contains else 0)


def metadata_prefix_equal(full_metadata: Sequence[Any], historical_metadata: Sequence[Any], prefix: int) -> bool:
    """Checker-only rule: compare the full list truncated to the historical prefix."""
    return list(full_metadata[:prefix]) == list(historical_metadata)


def classify_decision(summary: Mapping[str, Any]) -> str:
    transfer = bool(
        (summary["engram_exact"] > summary["base_exact"] or summary["engram_contains"] > summary["base_contains"])
        and summary["toward"] >= 1
        and summary["toward"] > summary["away"]
    )
    locality = bool(
        summary["locality_token_agreement"] >= 9
        and summary["locality_normalized_agreement"] >= 9
        and summary["locality_correctness_losses"] == 0
    )
    fresh = bool(summary["fresh_target_agreement"] == 10 and summary["fresh_locality_agreement"] == 10)
    implementation = bool(summary["state_verified"] and summary["finite"])
    return "ENGRAM_V2_10EDIT_GENERATION_PASS" if transfer and locality and fresh and implementation else "ENGRAM_V2_10EDIT_GENERATION_FAIL"


def render_report(decision: str, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    recommendation = "PROMOTE_TO_MULTI_SEQUENCE_10EDIT" if decision.endswith("PASS") else "NO MULTI-SEED PROMOTION"
    per_record = [
        f"| {row['record_id']} | {row['base_exact']} | {row['base_contains']} | {row['engram_exact']} | {row['engram_contains']} | {row['change_direction']} |"
        for row in rows
    ]
    failure_counts = summary["failure_counts"]
    return "\n".join([
        "# ENGRAM V2 10-Edit Free-Generation Transfer Gate",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        f"Recommendation: **{recommendation}**.",
        "",
        "This is a single-protocol evaluation of already-created states. It is not an editing rerun and makes no ENGRAM-over-TIME claim.",
        "",
        "## Free-generation effectiveness",
        "",
        f"- Base exact: `{summary['base_exact']}/10`",
        f"- Base contains: `{summary['base_contains']}/10`",
        f"- ENGRAM exact: `{summary['engram_exact']}/10`",
        f"- ENGRAM contains: `{summary['engram_contains']}/10`",
        f"- Changed toward target: `{summary['toward']}`",
        f"- Changed away from target: `{summary['away']}`",
        f"- Changed with unchanged correctness score: `{summary['neutral_changed']}`",
        "",
        "| ID | Base exact | Base contains | ENGRAM exact | ENGRAM contains | Direction |",
        "|---:|:---:|:---:|:---:|:---:|:---|",
        *per_record,
        "",
        "## Locality/reference generation behavior",
        "",
        f"- Exact token-sequence agreement: `{summary['locality_token_agreement']}/10`",
        f"- Exact decoded-output agreement: `{summary['locality_output_agreement']}/10`",
        f"- Normalized-output agreement: `{summary['locality_normalized_agreement']}/10`",
        f"- Locality target correctness losses: `{summary['locality_correctness_losses']}`",
        "",
        "## Fresh-process reproducibility",
        "",
        f"- Target token/output agreement: `{summary['fresh_target_agreement']}/10`",
        f"- Locality token/output agreement: `{summary['fresh_locality_agreement']}/10`",
        "",
        "## Output-pathology audit",
        "",
        f"- Empty: `{failure_counts['empty']}`",
        f"- Truncation at 16 tokens: `{failure_counts['truncation']}`",
        f"- Repetition: `{failure_counts['repetition']}`",
        f"- Refusal: `{failure_counts['refusal']}`",
        f"- Formatting failure: `{failure_counts['formatting']}`",
        "",
        "## Separation of evidence",
        "",
        "- Implementation correctness: bank reconstruction and fresh-process checks are reported independently above.",
        "- State/NLL/logit effectiveness: the existing frozen gate retained 10/10 edits; worst target NLL improvement was 0.020393848 and worst locality NLL drift was 0.008853912.",
        "- Free-generation effectiveness: determined only by the exact/contains and toward/away results in this report.",
        "- Locality/reference behavior: determined only by base-versus-final generated token/text agreement in this report.",
        "- NLL improvement is not counted as free-generation success.",
        "",
        "## Checker metadata-prefix discrepancy",
        "",
        "The preserved raw checker compared the complete 10-entry final metadata list against each historical k-entry list at prefixes 1-9. The valid comparison is `final_metadata[:k] == historical_metadata`; the checker-only regression test now covers this truncation rule. Original raw evidence was not replaced or altered.",
        "",
        "## Constraint confirmation",
        "",
        "- No editing rerun; no training; no 20-edit; no sweep; no changed-setting retry.",
        "- No ENGRAM algorithm, checkpoint, dataset, state, or existing raw-artifact modification.",
        f"- Physical GPU `{PHYSICAL_GPU}` only; GPU2 was not used.",
        "- No commit and no push.",
        "",
    ])


def finalize() -> None:
    if (OUT / "engram_generation_records.jsonl").exists() or (OUT / "ENGRAM_V2_10EDIT_FREE_GENERATION_GATE_REPORT.md").exists():
        raise FileExistsError("Final generation artifacts already exist")
    primary = read_jsonl(OUT / "primary_generation_records.jsonl")
    fresh = read_jsonl(OUT / "fresh_generation_records.jsonl")
    keyed = {(row["state"], row["view"], row["record_id"]): row for row in primary}
    fresh_keyed = {(row["view"], row["record_id"]): row for row in fresh}
    combined = []
    locality_token = locality_output = locality_normalized = locality_losses = 0
    fresh_target = fresh_locality = 0
    base_exact_count = base_contains_count = final_exact_count = final_contains_count = 0
    toward = away = neutral_changed = 0
    failure_counts = {name: 0 for name in ("empty", "truncation", "repetition", "refusal", "formatting")}
    for record_id in ORDER:
        base = keyed[("base", "target", record_id)]
        final = keyed[("engram_v2_final_10edit", "target", record_id)]
        fresh_row = fresh_keyed[("target", record_id)]
        base_exact, base_contains = exact_and_contains(base["raw_output"], base["target"])
        final_exact, final_contains = exact_and_contains(final["raw_output"], final["target"])
        base_exact_count += int(base_exact)
        base_contains_count += int(base_contains)
        final_exact_count += int(final_exact)
        final_contains_count += int(final_contains)
        base_score = correctness_score(base_exact, base_contains)
        final_score = correctness_score(final_exact, final_contains)
        changed = base["token_ids"] != final["token_ids"] or base["raw_output"] != final["raw_output"]
        direction = "toward" if final_score > base_score else ("away" if final_score < base_score else ("neutral_changed" if changed else "unchanged"))
        toward += int(direction == "toward")
        away += int(direction == "away")
        neutral_changed += int(direction == "neutral_changed")
        fresh_consistent = final["token_ids"] == fresh_row["token_ids"] and final["raw_output"] == fresh_row["raw_output"]
        fresh_target += int(fresh_consistent)

        base_loc = keyed[("base", "locality", record_id)]
        final_loc = keyed[("engram_v2_final_10edit", "locality", record_id)]
        fresh_loc = fresh_keyed[("locality", record_id)]
        loc_token_equal = base_loc["token_ids"] == final_loc["token_ids"]
        loc_output_equal = base_loc["raw_output"] == final_loc["raw_output"]
        loc_normalized_equal = base_loc["normalized_output"] == final_loc["normalized_output"]
        locality_token += int(loc_token_equal)
        locality_output += int(loc_output_equal)
        locality_normalized += int(loc_normalized_equal)
        base_loc_exact, base_loc_contains = exact_and_contains(base_loc["raw_output"], base_loc["target"])
        final_loc_exact, final_loc_contains = exact_and_contains(final_loc["raw_output"], final_loc["target"])
        locality_losses += int(correctness_score(final_loc_exact, final_loc_contains) < correctness_score(base_loc_exact, base_loc_contains))
        fresh_loc_consistent = final_loc["token_ids"] == fresh_loc["token_ids"] and final_loc["raw_output"] == fresh_loc["raw_output"]
        fresh_locality += int(fresh_loc_consistent)
        for item in (base, final, base_loc, final_loc, fresh_row, fresh_loc):
            for name, value in item["failure_flags"].items():
                failure_counts[name] += int(bool(value))

        fields = answer_fields(base["raw_output"], final["raw_output"], final["target"])
        combined.append({
            "record_id": record_id,
            "prompt": final["prompt"],
            "target": final["target"],
            "base_raw_output": base["raw_output"],
            "engram_raw_output": final["raw_output"],
            "base_token_ids": base["token_ids"],
            "engram_token_ids": final["token_ids"],
            "normalized_target": fields["normalized_target"],
            "normalized_base_output": fields["normalized_base_prediction"],
            "normalized_engram_output": fields["normalized_edited_prediction"],
            "base_exact": base_exact,
            "base_contains": base_contains,
            "engram_exact": final_exact,
            "engram_contains": final_contains,
            "generation_changed": changed,
            "change_direction": direction,
            "state_type": "base_vs_final_engram_v2_10edit",
            "generation_settings": final["generation_settings"],
            "fresh_process_consistency": {
                "token_ids_equal": final["token_ids"] == fresh_row["token_ids"],
                "decoded_output_equal": final["raw_output"] == fresh_row["raw_output"],
            },
            "failure_flags": {"base": base["failure_flags"], "engram": final["failure_flags"]},
            "locality": {
                "prompt": final_loc["prompt"], "target": final_loc["target"],
                "base_raw_output": base_loc["raw_output"], "engram_raw_output": final_loc["raw_output"],
                "base_token_ids": base_loc["token_ids"], "engram_token_ids": final_loc["token_ids"],
                "token_ids_equal": loc_token_equal, "decoded_output_equal": loc_output_equal,
                "normalized_output_equal": loc_normalized_equal,
                "base_exact": base_loc_exact, "base_contains": base_loc_contains,
                "engram_exact": final_loc_exact, "engram_contains": final_loc_contains,
                "fresh_process_consistency": {
                    "token_ids_equal": final_loc["token_ids"] == fresh_loc["token_ids"],
                    "decoded_output_equal": final_loc["raw_output"] == fresh_loc["raw_output"],
                },
            },
        })

    primary_summary = json.loads((OUT / "primary_summary.json").read_text())
    fresh_summary = json.loads((OUT / "fresh_summary.json").read_text())
    state_verified = bool(
        primary_summary["base_state_verified_against_anchor"]
        and primary_summary["final_state_verification"]["model_matches_quantized_assembled_state"]
        and fresh_summary["final_state_verification"]["model_matches_quantized_assembled_state"]
    )
    finite = bool(
        primary_summary["final_state_verification"]["all_values_finite"]
        and fresh_summary["final_state_verification"]["all_values_finite"]
    )
    summary = {
        "base_exact": base_exact_count, "base_contains": base_contains_count,
        "engram_exact": final_exact_count, "engram_contains": final_contains_count,
        "toward": toward, "away": away, "neutral_changed": neutral_changed,
        "locality_token_agreement": locality_token,
        "locality_output_agreement": locality_output,
        "locality_normalized_agreement": locality_normalized,
        "locality_correctness_losses": locality_losses,
        "fresh_target_agreement": fresh_target, "fresh_locality_agreement": fresh_locality,
        "state_verified": state_verified, "finite": finite, "failure_counts": failure_counts,
    }
    decision = classify_decision(summary)
    summary["decision"] = decision
    summary["recommendation"] = "PROMOTE_TO_MULTI_SEQUENCE_10EDIT" if decision.endswith("PASS") else "NO MULTI-SEED PROMOTION"
    exclusive_jsonl(OUT / "engram_generation_records.jsonl", combined)
    exclusive_json(OUT / "ENGRAM_V2_10EDIT_FREE_GENERATION_GATE_SUMMARY.json", summary)
    report = render_report(decision, summary, combined)
    report_path = OUT / "ENGRAM_V2_10EDIT_FREE_GENERATION_GATE_REPORT.md"
    with report_path.open("x") as handle:
        handle.write(report)
    exclusive_json(OUT / "FINAL_ARTIFACT_HASHES.json", {
        "generation_protocol_sha256": sha256_file(OUT / "GENERATION_PROTOCOL.json"),
        "generation_records_sha256": sha256_file(OUT / "engram_generation_records.jsonl"),
        "summary_sha256": sha256_file(OUT / "ENGRAM_V2_10EDIT_FREE_GENERATION_GATE_SUMMARY.json"),
        "report_sha256": sha256_file(report_path),
    })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    mode = parse_args().mode
    if mode == "prepare":
        prepare_protocol()
    elif mode == "primary":
        run_primary()
    elif mode == "fresh":
        run_fresh()
    else:
        finalize()


if __name__ == "__main__":
    main()
