#!/usr/bin/env python3
"""Read-only ENGRAM V2 Stage-0A/B/C diagnostics; never launches Stage 1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import to_jsonable  # noqa: E402
from scripts.engram.run_engram_v2_stage0_generation_audit import (  # noqa: E402
    BANK_ROOT,
    MODULE_KEY,
    MODULE_NAME,
    ORDER,
    apply_prefix,
    bank_manifest,
    clone_sample_with_target,
    eos_ids,
    load_model_views_bank,
    state_weight_hash,
)
from scripts.engram.stage0_generation_audit_utils import (  # noqa: E402
    CanonicalInputs,
    build_canonical_inputs,
    eos_diagnostics,
    first_repeated_ngram_step,
    ids_sha256,
    manual_cached_greedy_trace,
    manual_greedy_trace,
    medical_answer_match,
    model_next_logits,
    score_target_incrementally,
    tensor_sha256,
)
from scripts.engram.stage0abc_diagnostic_utils import (  # noqa: E402
    create_new_output_dir,
    first_prefix_parity,
    temporary_parameter_delta,
)

ORIGINAL_STAGE0 = ROOT / "outputs/engram_v2_stage0_generation_audit/20260810_stage0_v2"
SHORT_INSTRUCTION = "Answer with only the final medical answer. Do not provide an explanation."
PROTOCOL = "ENGRAM_V2_STAGE0ABC_DIAGNOSTICS_V1"
TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("stage0a", "stage0bc", "finalize"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--physical-gpu", default=2, type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(to_jsonable(dict(payload)), sort_keys=True) + "\n")


def write_text(path: Path, text: str, *, exclusive: bool = True) -> None:
    with path.open("x" if exclusive else "w") as handle:
        handle.write(text.rstrip() + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_manifest() -> Dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/engram/stage0abc_diagnostic_utils.py",
        ROOT / "scripts/engram/stage0_generation_audit_utils.py",
        ROOT / "scripts/engram/run_engram_v2_stage0_generation_audit.py",
        ROOT / "tests/test_engram_v2_stage0abc_diagnostics.py",
    ]
    rows = [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "size": path.stat().st_size} for path in paths]
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"files": rows, "sha256": hashlib.sha256(raw).hexdigest()}


def state_fingerprint(model: Any, bank: Any, prefix: int) -> Dict[str, Any]:
    edits = bank.list_edits()
    bank_hash = json.loads((BANK_ROOT / "index.json").read_text())["anchor_hash"] if prefix == 0 else edits[prefix - 1]["resulting_state_hash"]
    return {
        "state_id": f"S{prefix}",
        "bank_prefix_hash": bank_hash,
        "editable_parameter_hash": state_weight_hash(model),
        "editable_parameter_dtype": str(dict(model.named_modules())[MODULE_NAME].weight.dtype),
        "editable_parameter_shape": list(dict(model.named_modules())[MODULE_NAME].weight.shape),
    }


def ledger(out_dir: Path, stage: str, diagnostic: str, before: Mapping[str, Any], after: Mapping[str, Any], **extra: Any) -> None:
    append_jsonl(out_dir / "state_hash_rollback_ledger.jsonl", {
        "stage": stage,
        "diagnostic": diagnostic,
        "before": dict(before),
        "after": dict(after),
        "state_hash_equal": before == after,
        **extra,
    })


@torch.inference_mode()
def hf_cached_greedy_trace(model: Any, canonical: CanonicalInputs, max_new_tokens: int) -> Dict[str, Any]:
    attention = torch.ones_like(canonical.prompt_ids, dtype=torch.long, device=canonical.prompt_ids.device)
    base_kwargs = {
        "images": canonical.image,
        "attention_mask": attention,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
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
    raw_logits_available = True
    try:
        output = model.llava_model.generate(canonical.prompt_ids, output_logits=True, **base_kwargs)
    except (TypeError, ValueError):
        raw_logits_available = False
        output = model.llava_model.generate(canonical.prompt_ids, **base_kwargs)
    prompt_len = canonical.answer_start
    sequence = output.sequences[0]
    generated = sequence[prompt_len:] if sequence.numel() >= prompt_len and torch.equal(sequence[:prompt_len], canonical.prompt_ids[0]) else sequence
    token_ids = [int(item) for item in generated.detach().cpu().tolist()]
    raw_values = list(getattr(output, "logits", ()) or ())
    processed_values = list(getattr(output, "scores", ()) or ())
    trace = []
    eos = eos_ids(model)
    for step, selected_id in enumerate(token_ids):
        processed = processed_values[step][0].float() if step < len(processed_values) else None
        raw = raw_values[step][0].float() if step < len(raw_values) else None
        basis = raw if raw is not None else processed
        row = {
            "step": step,
            "selected_id": selected_id,
            "selected_text": model.llava_tokenizer.decode([selected_id], skip_special_tokens=False),
            "raw_logits_available": raw is not None,
            "raw_logits_hash": tensor_sha256(raw) if raw is not None else None,
            "raw_top1_id": int(raw.argmax().item()) if raw is not None else None,
            "processed_scores_hash": tensor_sha256(processed) if processed is not None else None,
            "processed_top1_id": int(processed.argmax().item()) if processed is not None else None,
            "raw_processed_max_abs_diff": float((raw - processed).abs().max().item()) if raw is not None and processed is not None else None,
        }
        if basis is not None:
            row.update(eos_diagnostics(basis, eos))
        trace.append(row)
    eos_set = set(eos)
    eos_step = next((index for index, item in enumerate(token_ids) if item in eos_set), None)
    return {
        "token_ids": token_ids,
        "raw_output": model.llava_tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
        "trajectory": trace,
        "raw_logits_available": raw_logits_available,
        "stop_reason": "eos" if eos_step is not None else "max_new_tokens",
        "eos_step": eos_step,
        "cap_hit": eos_step is None and len(token_ids) >= int(max_new_tokens),
        "first_repeated_bigram_step": first_repeated_ngram_step(token_ids, 2),
        "first_repeated_trigram_step": first_repeated_ngram_step(token_ids, 3),
    }


def first_difference(*sequences: Sequence[int]) -> Optional[int]:
    minimum = min(len(item) for item in sequences)
    for index in range(minimum):
        if len({int(item[index]) for item in sequences}) != 1:
            return index
    return None if len({len(item) for item in sequences}) == 1 else minimum


def original_stage0_cell() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    record = next(
        row for row in read_jsonl(ORIGINAL_STAGE0 / "records.jsonl")
        if row["record_id"] == "1592" and row["state_id"] == "S2" and row["view"] == "locality"
    )
    trajectory = next(
        row for row in read_jsonl(ORIGINAL_STAGE0 / "free_running_trajectories.jsonl")
        if row["record_id"] == "1592" and row["state_id"] == "S2" and row["view"] == "locality"
    )
    return record, trajectory


def run_cap_paths(model: Any, canonical: CanonicalInputs, cap: int) -> Dict[str, Any]:
    no_cache = manual_greedy_trace(model, canonical, cap, eos_ids(model), top_k=TOP_K)
    cached = manual_cached_greedy_trace(model, canonical, cap, eos_ids(model), top_k=TOP_K)
    hf = hf_cached_greedy_trace(model, canonical, cap)
    return {
        "cap": cap,
        "manual_no_cache": no_cache,
        "manual_cached": cached,
        "hf_generate": hf,
        "cached_no_cache_equal": cached["token_ids"] == no_cache["token_ids"],
        "cached_hf_equal": cached["token_ids"] == hf["token_ids"],
        "first_path_divergence": first_difference(no_cache["token_ids"], cached["token_ids"], hf["token_ids"]),
    }


def stage0a(args: argparse.Namespace) -> None:
    out_dir = create_new_output_dir(args.out_dir)
    write_text(out_dir / "commands.log", " ".join(sys.argv))
    write_json(out_dir / "run_manifest.json", {
        "protocol": PROTOCOL,
        "created_before_diagnostics": True,
        "physical_gpu": args.physical_gpu,
        "source_manifest": source_manifest(),
        "bank_manifest": bank_manifest(),
        "original_stage0": str(ORIGINAL_STAGE0),
        "constraints": {"training": False, "bank_write": False, "stage1": False, "sampling": False},
    })
    model, views, bank, _records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 2)
    before = state_fingerprint(model, bank, 2)
    original_record, original_trace = original_stage0_cell()
    canonical = build_canonical_inputs(model, views["1592"]["locality"])
    input_checks = {
        "prompt_hash_equal": canonical.prompt_hash == original_record["prompt_hash"],
        "pixel_hash_equal": canonical.pixel_hash == original_record["pixel_hash"],
        "full_hash_equal": canonical.full_hash == original_record["full_hash"],
        "state_hash_equal": before["bank_prefix_hash"] == bank.list_edits()[1]["resulting_state_hash"],
    }
    runs = [run_cap_paths(model, canonical, 128)]
    if all(item["cap_hit"] for item in (runs[0]["manual_no_cache"], runs[0]["manual_cached"], runs[0]["hf_generate"])):
        runs.append(run_cap_paths(model, canonical, 256))
    after = state_fingerprint(model, bank, 2)
    ledger(out_dir, "stage0a", "cap_resolution_microcheck", before, after)
    original64 = [int(item) for item in original_trace["token_ids"]]
    parity64 = {}
    for run in runs:
        for name in ("manual_no_cache", "manual_cached", "hf_generate"):
            parity64[f"cap{run['cap']}:{name}"] = first_prefix_parity(original64, run[name]["token_ids"], 64)
    all_paths_equal = all(run["cached_no_cache_equal"] and run["cached_hf_equal"] for run in runs)
    all_input = all(input_checks.values()) and before == after
    all_first64 = all(parity64.values())
    final_run = runs[-1]
    eos_steps = [final_run[name]["eos_step"] for name in ("manual_no_cache", "manual_cached", "hf_generate")]
    common_eos = eos_steps[0] if len(set(eos_steps)) == 1 else None
    if not all_input or not all_first64:
        classification = "NONDETERMINISTIC_RECONSTRUCTION"
        passed = False
        uniform_cap = None
    elif not all_paths_equal:
        classification = "LONG_HORIZON_GENERATION_PATH_MISMATCH"
        passed = False
        uniform_cap = None
    elif common_eos is not None and common_eos < 128:
        classification = "CAP_PROTOCOL_TOO_SHORT"
        passed = True
        uniform_cap = 128
    elif common_eos is not None and common_eos < 256:
        classification = "CAP_PROTOCOL_TOO_SHORT"
        passed = True
        uniform_cap = 256
    else:
        classification = "BASELINE_NONTERMINATING_GENERATION"
        passed = False
        uniform_cap = None
    result = {
        "classification": classification,
        "microcheck_passed": passed,
        "proposed_uniform_cap": uniform_cap,
        "input_checks": input_checks,
        "state_before": before,
        "state_after": after,
        "state_unchanged": before == after,
        "first64_parity": parity64,
        "all_path_parity": all_paths_equal,
        "common_eos_step": common_eos,
        "runs": runs,
        "original_first64": original64,
    }
    write_json(out_dir / "stage0a_cap_resolution.json", result)
    write_text(out_dir / "STAGE0A_CAP_RESOLUTION_REPORT.md", render_stage0a(result))
    if not passed:
        write_not_run_reports(out_dir, f"Stage-0A hard stop: {classification}")
        write_next_decision(out_dir, stage0a=result, full_matrix=None, stage0b=None, stage0c=None)
        return

    del model
    torch.cuda.empty_cache()
    matrix_dir = out_dir / f"fixed_matrix_uniform_{uniform_cap}"
    command = [
        "/root/anaconda3/bin/python",
        str(ROOT / "scripts/engram/run_engram_v2_stage0_generation_audit.py"),
        "--mode", "primary",
        "--physical-gpu", str(args.physical_gpu),
        "--uniform-cap", str(uniform_cap),
        "--out-dir", str(matrix_dir),
    ]
    with (out_dir / "commands.log").open("a") as handle:
        handle.write("\n" + " ".join(command) + "\n")
    completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), text=True, capture_output=True)
    write_text(out_dir / "fixed_matrix_stdout.log", completed.stdout)
    write_text(out_dir / "fixed_matrix_stderr.log", completed.stderr)
    matrix_summary = json.loads((matrix_dir / "stage0_summary.json").read_text()) if (matrix_dir / "stage0_summary.json").exists() else {
        "fixed_ten_edit_audit_completed": False,
        "runner_exit_code": completed.returncode,
    }
    matrix_valid = bool(completed.returncode == 0 and matrix_summary.get("fixed_ten_edit_audit_completed"))
    fresh_result = {"completed": False, "passed": False}
    if matrix_valid:
        fresh_command = command.copy()
        fresh_command[fresh_command.index("primary")] = "fresh"
        with (out_dir / "commands.log").open("a") as handle:
            handle.write(" ".join(fresh_command) + "\n")
        fresh_completed = subprocess.run(fresh_command, cwd=ROOT, env=os.environ.copy(), text=True, capture_output=True)
        write_text(out_dir / "fresh_matrix_stdout.log", fresh_completed.stdout)
        write_text(out_dir / "fresh_matrix_stderr.log", fresh_completed.stderr)
        refreshed = json.loads((matrix_dir / "stage0_summary.json").read_text())
        fresh_result = {
            "completed": True,
            "exit_code": fresh_completed.returncode,
            "passed": bool(fresh_completed.returncode == 0 and refreshed.get("fresh_process_pass")),
        }
        matrix_valid = matrix_valid and fresh_result["passed"]
    gate = {
        "matrix_dir": str(matrix_dir),
        "matrix_exit_code": completed.returncode,
        "matrix_summary": matrix_summary,
        "fresh": fresh_result,
        "full_stage0_valid": matrix_valid,
    }
    write_json(out_dir / "stage0a_full_matrix_gate.json", gate)
    if not matrix_valid:
        write_not_run_reports(out_dir, "Stage-0A uniform-cap full matrix or fresh reconstruction did not pass")
        write_next_decision(out_dir, stage0a=result, full_matrix=gate, stage0b=None, stage0c=None)


def render_stage0a(result: Mapping[str, Any]) -> str:
    lines = [
        "# Stage-0A Cap Resolution",
        "",
        f"**Classification:** `{result['classification']}`",
        "",
        f"- Microcheck passed: `{result['microcheck_passed']}`",
        f"- Proposed uniform cap: `{result['proposed_uniform_cap']}`",
        f"- Prompt/image/state checks: `{all(result['input_checks'].values())}`",
        f"- First-64 parity: `{all(result['first64_parity'].values())}`",
        f"- Cached/no-cache/HF parity: `{result['all_path_parity']}`",
        f"- Common EOS step: `{result['common_eos_step']}`",
        "",
        "No target-dependent stopping, sampling, repetition penalty, bank write, editor training, or Stage-1 training was used.",
    ]
    return "\n".join(lines)


def short_answer_sample(model: Any, sample: Mapping[str, Any], record: Mapping[str, Any]) -> Dict[str, Any]:
    prompt = f"Question: {str(record['src'])} {SHORT_INSTRUCTION} Short answer: "
    target = str(sample["target"][0])
    copied = dict(sample)
    copied["prompt"] = [prompt]
    copied["target"] = [target]
    copied["text_input"] = [prompt + target]
    copied["labels"] = model.llava_tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return copied


def new_old_margin(model: Any, sample: Mapping[str, Any], record: Mapping[str, Any]) -> Dict[str, Any]:
    old = record.get("pred")
    if old is None or not str(old).strip():
        return {"available": False}
    new_canonical = build_canonical_inputs(model, sample)
    old_sample = clone_sample_with_target(sample, str(old), model)
    old_canonical = build_canonical_inputs(model, old_sample)
    new_ids, old_ids = new_canonical.target_ids.tolist(), old_canonical.target_ids.tolist()
    index = next((idx for idx, (new, prior) in enumerate(zip(new_ids, old_ids)) if int(new) != int(prior)), min(len(new_ids), len(old_ids)))
    if index >= len(new_ids) or index >= len(old_ids):
        return {"available": False, "reason": "No distinct comparable token"}
    prefix = new_canonical.full_ids[:, : new_canonical.answer_start + index]
    logits = model_next_logits(model, prefix, new_canonical.image)
    return {
        "available": True,
        "decision_index": index,
        "new_token_id": int(new_ids[index]),
        "old_token_id": int(old_ids[index]),
        "new_token": model.llava_tokenizer.decode([int(new_ids[index])], skip_special_tokens=False),
        "old_token": model.llava_tokenizer.decode([int(old_ids[index])], skip_special_tokens=False),
        "margin": float((logits[int(new_ids[index])] - logits[int(old_ids[index])]).item()),
    }


def surface_row(model: Any, sample: Mapping[str, Any], record: Mapping[str, Any], record_id: str, state_id: str, view_name: str, cap: int) -> Dict[str, Any]:
    canonical = build_canonical_inputs(model, sample)
    token_rows = score_target_incrementally(model, canonical, eos_ids(model), top_k=TOP_K)
    greedy = manual_greedy_trace(model, canonical, cap, eos_ids(model), top_k=TOP_K)
    aliases = [str(item) for item in (record.get("accepted_answers") or [])]
    match = medical_answer_match(greedy["raw_output"], sample["target"][0], aliases=aliases)
    return {
        "record_id": record_id,
        "state_id": state_id,
        "view": view_name,
        "target_repr": repr(str(sample["target"][0])),
        "old_answer_repr": repr(str(record["pred"])) if record.get("pred") is not None else None,
        "rendered_prompt_hash": canonical.prompt_hash,
        "prompt_token_ids": canonical.prompt_ids[0].detach().cpu().tolist(),
        "first_eight_target_tokens": [
            {"id": int(item), "text": model.llava_tokenizer.decode([int(item)], skip_special_tokens=False)}
            for item in canonical.target_ids[:8].tolist()
        ],
        "first_generated_token": greedy["token_ids"][0] if greedy["token_ids"] else None,
        "first_target_rank": token_rows[0]["target_rank"],
        "first_target_probability": token_rows[0]["target_probability"],
        "first_target_margin": token_rows[0]["margin"],
        "target_vs_current_top1_margin": token_rows[0]["margin"],
        "new_vs_old": new_old_margin(model, sample, record),
        "all_target_tokens": token_rows,
        "greedy_output": greedy["raw_output"],
        "normalized_exact_match": match["normalized_exact_match"],
        "preregistered_alias_match": match["preregistered_alias_match"],
        "stop_reason": greedy["stop_reason"],
        "cap_hit": greedy["cap_hit"],
        "annotation_coverage_incomplete": match["annotation_coverage_incomplete"],
    }


def run_stage0b(model: Any, views: Mapping[str, Any], bank: Any, records: Mapping[str, Any], cap: int, out_dir: Path) -> Dict[str, Any]:
    rows = []
    for record_id, pre, post in (("953", 0, 1), ("1293", 1, 2)):
        for prefix, role in ((pre, "pre"), (post, "post")):
            apply_prefix(model, bank, prefix)
            before = state_fingerprint(model, bank, prefix)
            original = views[record_id]["target"]
            short = short_answer_sample(model, original, records[record_id])
            rows.append(surface_row(model, original, records[record_id], record_id, f"S{prefix}", "unrestricted", cap))
            rows.append(surface_row(model, short, records[record_id], record_id, f"S{prefix}", "fixed_short_answer", cap))
            after = state_fingerprint(model, bank, prefix)
            ledger(out_dir, "stage0b", f"{record_id}:S{prefix}:{role}", before, after)
            if before != after:
                raise RuntimeError("Stage-0B changed model state")
    evidence = []
    surface = False
    knowledge = False
    for record_id in ("953", "1293"):
        post = "S1" if record_id == "953" else "S2"
        original = next(row for row in rows if row["record_id"] == record_id and row["state_id"] == post and row["view"] == "unrestricted")
        short = next(row for row in rows if row["record_id"] == record_id and row["state_id"] == post and row["view"] == "fixed_short_answer")
        delta = float(short["first_target_margin"] - original["first_target_margin"])
        surface = surface or abs(delta) > 1e-3 or short["first_target_rank"] != original["first_target_rank"]
        knowledge = knowledge or (original["first_target_margin"] < 0 and short["first_target_margin"] < 0)
        evidence.append({"record_id": record_id, "short_minus_unrestricted_first_margin": delta})
    classification = "BOTH_SURFACE_AND_KNOWLEDGE_EFFECTS" if surface and knowledge else ("TARGET_SURFACE_FORM_EFFECT" if surface else "TRUE_KNOWLEDGE_MARGIN_DEFICIT")
    result = {"classification": classification, "rows": rows, "evidence": evidence, "short_answer_is_diagnostic_only": True}
    write_json(out_dir / "stage0b_target_surface_audit.json", result)
    write_text(out_dir / "STAGE0B_TARGET_SURFACE_AUDIT.md", render_stage0b(result))
    return result


def render_stage0b(result: Mapping[str, Any]) -> str:
    lines = ["# Stage-0B Target Surface-Form Audit", "", f"**Classification:** `{result['classification']}`", ""]
    lines.extend([f"- {item['record_id']}: short-minus-unrestricted first-token margin `{item['short_minus_unrestricted_first_margin']:.6f}`" for item in result["evidence"]])
    lines.extend(["", "Short-answer-only behavior is diagnostic and is not counted as unrestricted generation success."])
    return "\n".join(lines)


def fixed_competitor_margin(model: Any, canonical: CanonicalInputs, competitor_id: int) -> Dict[str, Any]:
    logits = model_next_logits(model, canonical.prompt_ids, canonical.image)
    target_id = int(canonical.target_ids[0].item())
    return {
        "target_id": target_id,
        "competitor_id": int(competitor_id),
        "margin": float((logits[target_id] - logits[int(competitor_id)]).item()),
        "target_logit": float(logits[target_id].item()),
        "competitor_logit": float(logits[int(competitor_id)].item()),
        "logits_hash": tensor_sha256(logits),
        "top1_id": int(logits.argmax().item()),
    }


def raw_margin_gradient(model: Any, canonical: CanonicalInputs) -> Tuple[torch.Tensor, Dict[str, Any]]:
    module = dict(model.named_modules())[MODULE_NAME]
    previous_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module.weight.requires_grad_(True)
    attention = torch.ones_like(canonical.prompt_ids, dtype=torch.long, device=canonical.prompt_ids.device)
    output = model.llava_model(
        input_ids=canonical.prompt_ids,
        images=canonical.image,
        attention_mask=attention,
        return_dict=True,
        use_cache=False,
    )
    logits = output.logits[0, -1].float()
    target_id = int(canonical.target_ids[0].item())
    competitor_id = int(logits.detach().argmax().item())
    margin = logits[target_id] - logits[competitor_id]
    gradient = torch.autograd.grad(margin, module.weight)[0].detach().float().cpu()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(previous_flags[name])
    return gradient, {"target_id": target_id, "competitor_id": competitor_id, "margin": float(margin.detach().item())}


@torch.inference_mode()
def float32_lm_head_margin(model: Any, canonical: CanonicalInputs, target_id: int, competitor_id: int) -> Dict[str, Any]:
    attention = torch.ones_like(canonical.prompt_ids, dtype=torch.long, device=canonical.prompt_ids.device)
    output = model.llava_model(
        input_ids=canonical.prompt_ids,
        images=canonical.image,
        attention_mask=attention,
        return_dict=True,
        use_cache=False,
        output_hidden_states=True,
    )
    hidden = output.hidden_states[-1][0, -1].float()
    head = model.llava_model.get_output_embeddings().weight.float()
    values = torch.stack([head[int(target_id)].dot(hidden), head[int(competitor_id)].dot(hidden)])
    return {"target_logit": float(values[0].item()), "competitor_logit": float(values[1].item()), "margin": float((values[0] - values[1]).item())}


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    l, r = left.double().reshape(-1), right.double().reshape(-1)
    denom = float(l.norm() * r.norm())
    return float(l.dot(r) / denom) if denom > 0 else 0.0


def finite_probe(
    model: Any, bank: Any, prefix: int, target: CanonicalInputs, locality: CanonicalInputs,
    competitor_id: int, direction: torch.Tensor, scale: float, out_dir: Path, record_id: str,
) -> Dict[str, Any]:
    module = dict(model.named_modules())[MODULE_NAME]
    before_state = state_fingerprint(model, bank, prefix)
    baseline_target = fixed_competitor_margin(model, target, competitor_id)
    baseline_locality_logits = model_next_logits(model, locality.prompt_ids, locality.image)
    baseline_locality_top1 = int(baseline_locality_logits.argmax().item())
    baseline_locality_generation = manual_greedy_trace(model, locality, 16, eos_ids(model), top_k=TOP_K)["token_ids"]
    with temporary_parameter_delta(module.weight, direction * float(scale)) as delta_ledger:
        temporary_target = fixed_competitor_margin(model, target, competitor_id)
        locality_logits = model_next_logits(model, locality.prompt_ids, locality.image)
        locality_top1 = int(locality_logits.argmax().item())
        locality_generation = manual_greedy_trace(model, locality, 16, eos_ids(model), top_k=TOP_K)["token_ids"]
    restored_target = fixed_competitor_margin(model, target, competitor_id)
    restored_locality_generation = manual_greedy_trace(model, locality, 16, eos_ids(model), top_k=TOP_K)["token_ids"]
    after_state = state_fingerprint(model, bank, prefix)
    rollback = {
        "state_hash_equal": before_state == after_state,
        "target_logits_equal": baseline_target["logits_hash"] == restored_target["logits_hash"],
        "locality_generation_equal": baseline_locality_generation == restored_locality_generation,
        "parameter_rollback": delta_ledger,
    }
    ledger(out_dir, "stage0c", f"{record_id}:finite_difference:{scale}", before_state, after_state, rollback=rollback)
    if not all([rollback["state_hash_equal"], rollback["target_logits_equal"], rollback["locality_generation_equal"], delta_ledger["rollback_exact"]]):
        raise RuntimeError("Finite-difference rollback verification failed")
    return {
        "scale": scale,
        "target_margin_before": baseline_target["margin"],
        "target_margin_temporary": temporary_target["margin"],
        "target_margin_gain": temporary_target["margin"] - baseline_target["margin"],
        "locality_top1_before": baseline_locality_top1,
        "locality_top1_temporary": locality_top1,
        "locality_top1_changed": baseline_locality_top1 != locality_top1,
        "locality_prefix_before": baseline_locality_generation,
        "locality_prefix_temporary": locality_generation,
        "locality_prefix_changed": baseline_locality_generation != locality_generation,
        "rollback": rollback,
    }


def projection_availability() -> Dict[str, Any]:
    return {
        "current_engram_v2_editable_weight_space": {"evaluated": True, "module": MODULE_NAME},
        "engram_projected_tiny_lora": {
            "evaluated": False,
            "reason": "Frozen ENGRAM V2 bank stores additive q_proj deltas and target factors, not the exact legacy ENGRAM projector/tiny-LoRA operator for layer 21; mixing legacy projector banks would change the protocol.",
        },
        "engram_cure_projected_delta": {
            "evaluated": False,
            "reason": "No exact CURE K-FAC projection cache is registered for the frozen V2 layer-21 q_proj state; no projector was fabricated.",
        },
    }


def run_stage0c(model: Any, views: Mapping[str, Any], bank: Any, records: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    results = []
    hard_stop = None
    for record_id, pre, post in (("953", 0, 1), ("1293", 1, 2)):
        try:
            apply_prefix(model, bank, pre)
            before = state_fingerprint(model, bank, pre)
            target = build_canonical_inputs(model, views[record_id]["target"])
            locality = build_canonical_inputs(model, views[record_id]["locality"])
            gradient, base = raw_margin_gradient(model, target)
            gradient_norm = float(gradient.norm().item())
            if not torch.isfinite(gradient).all() or gradient_norm <= 0:
                raise RuntimeError("Margin gradient is non-finite or zero")
            pre_state = bank.assemble_state([item["edit_id"] for item in bank.list_edits()[:pre]])[MODULE_KEY]
            post_state = bank.assemble_state([item["edit_id"] for item in bank.list_edits()[:post]])[MODULE_KEY]
            actual_delta = post_state - pre_state
            actual_norm = float(actual_delta.norm().item())
            pre_margin = fixed_competitor_margin(model, target, base["competitor_id"])
            apply_prefix(model, bank, post)
            post_margin = fixed_competitor_margin(model, target, base["competitor_id"])
            apply_prefix(model, bank, pre)
            direction = gradient / gradient_norm
            probes = [finite_probe(model, bank, pre, target, locality, base["competitor_id"], direction.to(model.lm_device), factor * actual_norm, out_dir, record_id) for factor in (0.0, 0.25, 0.5, 1.0)]
            float32_margin = float32_lm_head_margin(model, target, base["target_id"], base["competitor_id"])
            after = state_fingerprint(model, bank, pre)
            ledger(out_dir, "stage0c", f"{record_id}:gradient_probe", before, after)
            result = {
                "record_id": record_id,
                "pre_state": f"S{pre}",
                "post_state": f"S{post}",
                "raw_margin_gradient_norm": gradient_norm,
                "effective_projected_margin_gradient_norm": gradient_norm,
                "gradient_retention_ratio": 1.0,
                "projector_symmetric": True,
                "projector_idempotent": True,
                "parameterization": "current_engram_v2_editable_weight_space_identity",
                "actual_delta_norm": actual_norm,
                "actual_delta_gradient_cosine": cosine(actual_delta, gradient),
                "first_order_predicted_margin_change": float((actual_delta.double() * gradient.double()).sum().item()),
                "observed_margin_change": float(post_margin["margin"] - pre_margin["margin"]),
                "prediction_error": float((post_margin["margin"] - pre_margin["margin"]) - (actual_delta.double() * gradient.double()).sum().item()),
                "actual_generation_dtype_margin": pre_margin,
                "float32_hidden_lm_head_margin": float32_margin,
                "finite_difference_probes": probes,
            }
            results.append(result)
        except Exception as error:
            hard_stop = {"record_id": record_id, "type": type(error).__name__, "message": str(error)}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
    labels = []
    for result in results:
        same_norm = next((probe for probe in result["finite_difference_probes"] if abs(probe["scale"] - result["actual_delta_norm"]) < 1e-8), None)
        if result["actual_delta_gradient_cosine"] < 0.1:
            labels.append("CURRENT_UPDATE_MISALIGNED_WITH_MARGIN")
        if same_norm and same_norm["target_margin_gain"] > max(1e-6, 5 * abs(result["observed_margin_change"])):
            labels.append("CURRENT_UPDATE_MISALIGNED_WITH_MARGIN")
        if same_norm and same_norm["target_margin_gain"] > 0:
            labels.append("MARGIN_DIRECTION_FEASIBLE")
        if same_norm and same_norm["target_margin_gain"] < 0.1 and result["observed_margin_change"] < 0.1:
            labels.append("CURRENT_DISPLACEMENT_BUDGET_INSUFFICIENT")
        if same_norm and same_norm["locality_top1_changed"]:
            labels.append("EFFECTIVENESS_LOCALITY_DIRECTION_CONFLICT")
    result = {
        "projection_parameterizations": projection_availability(),
        "records": results,
        "labels": list(dict.fromkeys(labels)),
        "hard_stop": hard_stop,
        "training": False,
        "bank_commit": False,
    }
    write_json(out_dir / "stage0c_margin_feasibility.json", result)
    write_text(out_dir / "STAGE0C_MARGIN_FEASIBILITY_REPORT.md", render_stage0c(result))
    return result


def render_stage0c(result: Mapping[str, Any]) -> str:
    lines = ["# Stage-0C Projected Margin Feasibility", "", "## Labels", ""]
    lines.extend([f"- `{label}`" for label in result["labels"]] or ["- No label forced"])
    if result["hard_stop"]:
        lines.extend(["", f"**Hard stop:** `{result['hard_stop']}`"])
    lines.extend(["", "Legacy tiny-LoRA and CURE projection paths were not evaluated unless an exact V2-compatible operator was available; none was fabricated."])
    return "\n".join(lines)


def stage0bc(args: argparse.Namespace) -> None:
    stage0a_result = json.loads((args.out_dir / "stage0a_cap_resolution.json").read_text())
    matrix_gate = json.loads((args.out_dir / "stage0a_full_matrix_gate.json").read_text())
    if not stage0a_result["microcheck_passed"] or not matrix_gate["full_stage0_valid"]:
        raise RuntimeError("Stage-0B/C are forbidden because Stage-0A/full-matrix gate did not pass")
    if (args.out_dir / "stage0b_target_surface_audit.json").exists() or (args.out_dir / "stage0c_margin_feasibility.json").exists():
        raise FileExistsError("Refusing to overwrite Stage-0B/C outputs")
    with (args.out_dir / "commands.log").open("a") as handle:
        handle.write(" ".join(sys.argv) + "\n")
    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    cap = int(stage0a_result["proposed_uniform_cap"])
    result_b = run_stage0b(model, views, bank, records, cap, args.out_dir)
    result_c = run_stage0c(model, views, bank, records, args.out_dir)
    write_next_decision(args.out_dir, stage0a=stage0a_result, full_matrix=matrix_gate, stage0b=result_b, stage0c=result_c)


def write_not_run_reports(out_dir: Path, reason: str) -> None:
    payload = {"status": "NOT_RUN", "reason": reason}
    write_json(out_dir / "stage0b_target_surface_audit.json", payload)
    write_text(out_dir / "STAGE0B_TARGET_SURFACE_AUDIT.md", f"# Stage-0B Target Surface Audit\n\n**NOT RUN:** {reason}")
    write_json(out_dir / "stage0c_margin_feasibility.json", payload)
    write_text(out_dir / "STAGE0C_MARGIN_FEASIBILITY_REPORT.md", f"# Stage-0C Margin Feasibility\n\n**NOT RUN:** {reason}")


def write_next_decision(out_dir: Path, *, stage0a: Mapping[str, Any], full_matrix: Optional[Mapping[str, Any]], stage0b: Optional[Mapping[str, Any]], stage0c: Optional[Mapping[str, Any]]) -> None:
    full_valid = bool(full_matrix and full_matrix.get("full_stage0_valid"))
    stage1_permitted = bool(full_valid and stage0b and stage0c and not stage0c.get("hard_stop"))
    facts = [
        f"Stage-0A classification: {stage0a['classification']}",
        f"Stage-0A microcheck passed: {stage0a['microcheck_passed']}",
        f"Full Stage-0 audit valid: {full_valid}",
        f"Stage-1 permitted by diagnostics: {stage1_permitted}",
        "Stage-1 was not launched.",
    ]
    inferences = []
    if stage0b:
        inferences.append(f"Stage-0B: {stage0b.get('classification')}")
    if stage0c:
        inferences.extend(f"Stage-0C: {label}" for label in stage0c.get("labels", []))
    unresolved = [] if full_valid else ["The fixed ten-edit Stage-0 audit is not yet valid under the declared cap/reconstruction gate."]
    text = ["# Stage-0 Next Decision", "", "## Verified facts", "", *[f"- {item}" for item in facts], "", "## Diagnostic inferences", "", *([f"- {item}" for item in inferences] or ["- None"]), "", "## Unresolved questions", "", *([f"- {item}" for item in unresolved] or ["- None"]), ""]
    write_text(out_dir / "STAGE0_NEXT_DECISION.md", "\n".join(text))


def finalize(args: argparse.Namespace) -> None:
    required = [
        "STAGE0A_CAP_RESOLUTION_REPORT.md", "stage0a_cap_resolution.json",
        "STAGE0B_TARGET_SURFACE_AUDIT.md", "stage0b_target_surface_audit.json",
        "STAGE0C_MARGIN_FEASIBILITY_REPORT.md", "stage0c_margin_feasibility.json",
        "STAGE0_NEXT_DECISION.md", "run_manifest.json", "commands.log", "state_hash_rollback_ledger.jsonl",
    ]
    missing = [name for name in required if not (args.out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing Stage-0ABC outputs: {missing}")
    manifest = json.loads((args.out_dir / "run_manifest.json").read_text())
    outputs = []
    for path in sorted(item for item in args.out_dir.rglob("*") if item.is_file() and item.name != "run_manifest.json"):
        outputs.append({"path": str(path.relative_to(args.out_dir)), "sha256": sha256_file(path), "size": path.stat().st_size})
    manifest["outputs"] = outputs
    manifest["stage1_launched"] = False
    manifest["bank_manifest_after"] = bank_manifest()
    manifest["bank_unchanged"] = manifest["bank_manifest"]["sha256"] == manifest["bank_manifest_after"]["sha256"]
    write_json(args.out_dir / "run_manifest.json", manifest, exclusive=False)
    print(json.dumps({"status": "FINALIZED", "bank_unchanged": manifest["bank_unchanged"], "output_count": len(outputs)}, indent=2))


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.mode == "stage0a":
        stage0a(args)
    elif args.mode == "stage0bc":
        stage0bc(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
