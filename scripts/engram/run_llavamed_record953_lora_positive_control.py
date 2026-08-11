#!/usr/bin/env python3
"""Fixed multi-layer LoRA positive control for LLaVA-Med record 953."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import to_jsonable  # noqa: E402
from scripts.engram.lora_positive_control_utils import (  # noqa: E402
    adapter_hash,
    adapter_state,
    audit_trainable_parameters,
    load_adapter_payload,
    load_adapter_state,
    positive_control_match,
    resolve_target_modules,
    save_adapter_payload,
    shifted_label_audit,
)
from scripts.engram.natural_generation_recovery_utils import (  # noqa: E402
    assert_no_target_leakage,
    canonical_natural_response,
    expanded_predictor_positions,
    target_only_response,
)
from scripts.engram.run_engram_natural_generation_recovery import (  # noqa: E402
    effect_objective,
    response_layout,
    response_objective,
    selected_logits,
)
from scripts.engram.run_engram_v2_one_shot_natural_generation_rescue import (  # noqa: E402
    bank_anchor_hash,
    content_indices,
    full_generation_parity,
    hf_cached_greedy_trace,
)
from scripts.engram.run_engram_v2_stage0_generation_audit import (  # noqa: E402
    MODEL_CONFIG,
    apply_prefix,
    bank_manifest,
    clone_sample_with_target,
    eos_ids,
    load_model_views_bank,
    state_weight_hash,
)
from scripts.engram.run_engram_v2_stage0abc_diagnostics import short_answer_sample  # noqa: E402
from scripts.engram.run_engram_v3_1_locality_corrected import full_locality_gate  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_greedy_trace  # noqa: E402


RECORD_ID = "953"
TARGET = "completely ectocervical and fully visible"
PRIMARY_RESPONSE = "The answer is completely ectocervical and fully visible."
SHORT_RESPONSE = "completely ectocervical and fully visible."
CAP = 128
SEED = 42
MAX_STEPS = 500
LR = 2e-4
RANK = 16
ALPHA = 32
PROTOCOL = "LLAVAMED_RECORD953_LORA_POSITIVE_CONTROL"
EXPECTED_BANK_HASH = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"
EXPECTED_ANCHOR_HASH = "791ba2d19c7549608ddd21a0a92f5da6a762401d9f95380d8e1a4a70e17688c7"
V3 = ROOT / "outputs/engram_v3_1_locality_corrected/20260811_v3_1_v4"
V3_HASHES = {
    "run_manifest.json": "885ee64794eb1aa0e1e3d2abf2d2e483a913cc0bda9ba7b3da7793ffdd41dc28",
    "v3_1_summary.json": "0eb6947a8562355ae53e81156189ff9db7a831ce4973c495da4e2eb60056a4f2",
    "v3_1_locality_basis_report.json": "6caa11cdbc693a6f24b4722743554d14fa179ddc44977dc9659ac17205f0e71a",
    "v3_1_final_locality_report.json": "e5f8eb502925a87c6cd31bc57f6149d9c43bbdad78e55f365fa862b6764f327a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "fresh"), default="run")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--adapter-bank", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, value: Any, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        handle.write(value.rstrip() + "\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(to_jsonable(dict(value)), sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_diff() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/engram/lora_positive_control_utils.py",
        ROOT / "tests/test_llavamed_record953_lora_positive_control.py",
    )
    result: list[str] = []
    for path in paths:
        result.extend(difflib.unified_diff([], path.read_text().splitlines(True), fromfile="/dev/null", tofile=f"b/{path.relative_to(ROOT)}"))
    return "".join(result)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def insert_lora(base_model: Any, resolved: Sequence[str]) -> Any:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=list(resolved),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, config)
    for name, parameter in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(False)
    return peft_model


def response_nll(model: Any, canonical: Any) -> torch.Tensor:
    logits = selected_logits(model, canonical)
    return -logits.log_softmax(dim=-1).gather(1, canonical.target_ids.long().unsqueeze(1)).mean()


def target_metrics(model: Any, natural: Any, short: Any, natural_layout: Mapping[str, Any], short_positions: Sequence[int]) -> dict[str, Any]:
    with torch.no_grad():
        _nr, _nn, natural_info = response_objective(model, natural, natural_layout["target_positions"], natural_layout["scaffold_positions"])
        _sr, _sn, short_info = response_objective(model, short, short_positions, [])
        primary_nll = float(response_nll(model, natural).item())
        short_nll = float(response_nll(model, short).item())
    return {"primary_response_nll": primary_nll, "auxiliary_response_nll": short_nll, "primary": natural_info, "auxiliary": short_info}


def compact_hf_generation(model: Any, canonical: Any, aliases: Sequence[str]) -> dict[str, Any]:
    trace = hf_cached_greedy_trace(model, canonical, CAP)
    matched = positive_control_match(
        trace["raw_output"],
        TARGET,
        eos=trace["stop_reason"] == "eos",
        cap_hit=trace["cap_hit"],
        aliases=aliases,
    )
    return {
        "raw_output": trace["raw_output"],
        "token_ids": trace["token_ids"],
        "stop_reason": trace["stop_reason"],
        "cap_hit": trace["cap_hit"],
        "eos_step": trace.get("eos_step"),
        "match": matched,
    }


def disable_adapter(peft_model: Any) -> None:
    peft_model.disable_adapter_layers()


def enable_adapter(peft_model: Any) -> None:
    peft_model.enable_adapter_layers()


def final_report(summary: Mapping[str, Any]) -> str:
    implication = (
        "Record 953 is learnable by the backbone; constrained ENGRAM editable space and optimization/locality are the primary bottleneck. Freeze the current ENGRAM parameter-editing core and move to a banked LoRA/routed-adapter method."
        if summary["unrestricted_success"]
        else (
            "The target was learnable only under the explicit short-answer format; this is not unrestricted success."
            if summary["short_answer_success"]
            else "Record 953 should not be the sole feasibility sample. Build a preregistered model-known or near-margin subset and do not resume ENGRAM V3.x tuning on this record."
        )
    )
    return f"""# LLaVA-Med Record 953 LoRA Positive-Control Decision

- Did unrestricted natural generation succeed? **{summary['unrestricted_success']}**
- At which optimizer step? **{summary['success_step']}**
- Exact unrestricted output: `{summary['exact_unrestricted_output']}`
- Did short-answer generation succeed? **{summary['short_answer_success']}**
- Strict-locality outputs changed: **{summary['strict_locality_damage']}/10**
- Clinical/canonical locality failures: **{summary['clinical_locality_failures']}/10**
- Adapter reload / fresh / rollback: **{summary['reload']} / {summary['fresh']} / {summary['rollback']}**
- What does the result imply for ENGRAM? **{implication}**
- Is Stage-2 permitted? **No**

## Primary label

`{summary['primary_label']}`

Locality: `{summary['strict_locality_label']}` / `{summary['clinical_locality_label']}`.
"""


def validate_v3() -> dict[str, str]:
    observed = {}
    for name, expected in V3_HASHES.items():
        path = V3 / name
        observed[name] = sha256(path)
        if observed[name] != expected:
            raise RuntimeError(f"V3.1 artifact hash mismatch: {name}")
    summary = json.loads((V3 / "v3_1_summary.json").read_text())
    if summary["primary_label"] != "V3_1_OPTIMIZER_STALLED_BEFORE_BUDGET" or summary["clinical_locality_failures"] != 1:
        raise RuntimeError("V3.1 starting result mismatch")
    return observed


def run(args: argparse.Namespace) -> None:
    seed_everything()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in ("training_trajectory.jsonl", "state_and_bank_hash_ledger.jsonl"):
        write_text(out_dir / name, "")
    exact_command = f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} {sys.executable} " + " ".join(sys.argv)
    write_text(out_dir / "exact_command_log.txt", exact_command)
    write_text(out_dir / "source_diff.patch", source_diff())
    v3_hashes = validate_v3()
    bank_before = bank_manifest()
    if bank_before["sha256"] != EXPECTED_BANK_HASH or bank_anchor_hash() != EXPECTED_ANCHOR_HASH:
        raise RuntimeError("Canonical ENGRAM bank mismatch")
    manifest = {
        "protocol": PROTOCOL,
        "record_id": RECORD_ID,
        "seed": SEED,
        "rank": RANK,
        "lora_alpha": ALPHA,
        "lora_dropout": 0.0,
        "bias": "none",
        "layers": [16, 31],
        "learning_rate": LR,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_steps": 10,
        "maximum_steps": MAX_STEPS,
        "primary_response": PRIMARY_RESPONSE,
        "auxiliary_response": SHORT_RESPONSE,
        "v3_1_path": str(V3),
        "v3_1_hashes": v3_hashes,
        "clinical_rule": "ADDED_UNSUPPORTED_SUBTYPE_OR_GEOGRAPHIC_SPECIFICITY_IS_FAILURE_RELATIVE_TO_S0",
        "cwd": str(ROOT),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": args.physical_gpu,
        "model_config": str(MODEL_CONFIG),
        "canonical_bank_before": bank_before,
        "stage2_launched": False,
        "ten_edit_launched": False,
        "engram_training_launched": False,
    }
    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    model.llava_model.gradient_checkpointing_enable()
    if hasattr(model.llava_model, "enable_input_require_grads"):
        model.llava_model.enable_input_require_grads()
    apply_prefix(model, bank, 0)
    clean_hash = state_weight_hash(model)
    record = records[RECORD_ID]
    aliases = [str(item) for item in (record.get("accepted_answers") or [])]
    if str(record["alt"]) != TARGET:
        raise RuntimeError("Record-953 target changed")
    original = build_canonical_inputs(model, views[RECORD_ID]["target"])
    natural = build_canonical_inputs(model, clone_sample_with_target(views[RECORD_ID]["target"], PRIMARY_RESPONSE, model))
    short_sample = short_answer_sample(model, views[RECORD_ID]["target"], record)
    short = build_canonical_inputs(model, clone_sample_with_target(short_sample, SHORT_RESPONSE, model))
    assert_no_target_leakage([original.prompt_text, short.prompt_text], TARGET)
    baseline_unrestricted = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)
    baseline_short = manual_greedy_trace(model, short, CAP, eos_ids(model), top_k=1)
    append_jsonl(out_dir / "state_and_bank_hash_ledger.jsonl", {"event": "CLEAN_S0", "weight_hash": clean_hash, "bank_hash": bank_before["sha256"], "unrestricted_token_ids": baseline_unrestricted["token_ids"]})
    resolved = resolve_target_modules(model.llava_model.named_modules())
    write_json(out_dir / "resolved_lora_modules.json", {"count": len(resolved), "modules": resolved, "language_count": 112, "projector_count": len(resolved) - 112})
    model.llava_model = insert_lora(model.llava_model, resolved)
    peft_model = model.llava_model
    audit = audit_trainable_parameters(peft_model.named_parameters())
    write_json(out_dir / "trainable_parameter_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("PC_INVALID_ENGINEERING_RUN: trainable parameter audit")
    zero_ids = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    if zero_ids != baseline_unrestricted["token_ids"]:
        raise RuntimeError("PC_INVALID_ENGINEERING_RUN: zero-adapter parity")
    with torch.no_grad():
        inputs = torch.cat([natural.prompt_ids, natural.target_ids[:-1].unsqueeze(0)], dim=1)
        output = model.llava_model(input_ids=inputs, images=natural.image, attention_mask=torch.ones_like(inputs), return_dict=True, use_cache=False)
        expansion = int(output.logits.shape[1] - inputs.shape[1])
    positions = expanded_predictor_positions(natural.answer_start, natural.target_ids.numel(), expansion)
    label_audit = shifted_label_audit(natural.answer_start, natural.target_ids, expansion, positions)
    if not label_audit["passed"]:
        raise RuntimeError("PC_INVALID_ENGINEERING_RUN: shifted labels")
    manifest["shifted_label_audit"] = label_audit
    manifest["resolved_lora_modules"] = resolved
    manifest["target_hash"] = hashlib.sha256(TARGET.encode()).hexdigest()
    manifest["question_hash"] = hashlib.sha256(original.prompt_text.encode()).hexdigest()
    manifest["image_hash"] = original.pixel_hash
    natural_layout = response_layout(model, natural, TARGET)
    short_positions = content_indices(model, short.target_ids.detach().cpu().tolist())
    initial_metrics = target_metrics(model, natural, short, natural_layout, short_positions)
    trainable = [parameter for parameter in peft_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    success_step = None
    last_primary_generation = compact_hf_generation(model, original, aliases)
    last_short_generation = compact_hf_generation(model, short, aliases)
    for step in range(1, MAX_STEPS + 1):
        peft_model.train()
        optimizer.zero_grad(set_to_none=True)
        primary_loss = response_nll(model, natural)
        if not torch.isfinite(primary_loss):
            raise RuntimeError("PC_INVALID_ENGINEERING_RUN: nonfinite primary loss")
        primary_loss.backward()
        auxiliary_loss = response_nll(model, short)
        if not torch.isfinite(auxiliary_loss):
            raise RuntimeError("PC_INVALID_ENGINEERING_RUN: nonfinite auxiliary loss")
        (0.25 * auxiliary_loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        current_lr = LR * min(step / 10.0, 1.0)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        row = {
            "step": step,
            "primary_loss": float(primary_loss.detach().item()),
            "auxiliary_loss": float(auxiliary_loss.detach().item()),
            "total_loss": float(primary_loss.detach().item() + 0.25 * auxiliary_loss.detach().item()),
            "learning_rate": current_lr,
            "gradient_norm_before_clip": float(grad_norm),
        }
        del primary_loss, auxiliary_loss
        if step % 10 == 0:
            peft_model.eval()
            metrics = target_metrics(model, natural, short, natural_layout, short_positions)
            last_primary_generation = compact_hf_generation(model, original, aliases)
            last_short_generation = compact_hf_generation(model, short, aliases)
            row.update({"target_metrics": metrics, "unrestricted_generation": last_primary_generation, "short_generation": last_short_generation})
            if last_primary_generation["match"]["success"]:
                success_step = step
        append_jsonl(out_dir / "training_trajectory.jsonl", row)
        if success_step is not None:
            break
        if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
            raise RuntimeError("PC_INVALID_ENGINEERING_RUN: canonical bank mutation")
    terminal_step = success_step or MAX_STEPS
    peft_model.eval()
    final_metrics = target_metrics(model, natural, short, natural_layout, short_positions)
    final_state = adapter_state(peft_model.named_parameters())
    diagnostic_root = out_dir / ("successful_adapter_bank_item" if success_step else "terminal_diagnostic_adapter")
    save_adapter_payload(diagnostic_root, final_state, {
        "protocol": PROTOCOL,
        "record_id": RECORD_ID,
        "target": TARGET,
        "target_hash": manifest["target_hash"],
        "question_hash": manifest["question_hash"],
        "image_hash": manifest["image_hash"],
        "base_s0_hash": clean_hash,
        "resolved_lora_modules": resolved,
        "rank": RANK,
        "alpha": ALPHA,
        "training_step": terminal_step,
        "canonical_bank_hash": EXPECTED_BANK_HASH,
    })
    locality = full_locality_gate(model, views, lambda: disable_adapter(peft_model), lambda: enable_adapter(peft_model))
    enable_adapter(peft_model)
    parity = full_generation_parity(model, original)
    final_primary = {
        "raw_output": parity["no_cache"]["raw_output"],
        "token_ids": parity["no_cache"]["token_ids"],
        "stop_reason": parity["no_cache"]["stop_reason"],
        "cap_hit": parity["no_cache"]["cap_hit"],
        "eos_step": parity["no_cache"].get("eos_step"),
        "match": positive_control_match(parity["no_cache"]["raw_output"], TARGET, eos=parity["no_cache"]["stop_reason"] == "eos", cap_hit=parity["no_cache"]["cap_hit"], aliases=aliases),
    }
    final_short = compact_hf_generation(model, short, aliases)
    unrestricted_success = bool(final_primary["match"]["success"] and parity["passed"])
    short_success = bool(final_short["match"]["success"])
    if unrestricted_success:
        adapter_manifest = json.loads((diagnostic_root / "manifest.json").read_text())
        adapter_manifest["edited_unrestricted_token_ids"] = final_primary["token_ids"]
        write_json(diagnostic_root / "manifest.json", adapter_manifest, exclusive=False)
    disable_adapter(peft_model)
    rollback_ids = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    rollback = "PASS" if rollback_ids == baseline_unrestricted["token_ids"] else "FAIL"
    enable_adapter(peft_model)
    zero_state = {name: torch.zeros_like(value) for name, value in final_state.items()}
    load_adapter_state(peft_model.named_parameters(), zero_state)
    load_adapter_state(peft_model.named_parameters(), final_state)
    reload_ids = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    reload_status = "PASS" if reload_ids == final_primary["token_ids"] else "FAIL"
    disable_adapter(peft_model)
    second_rollback_ids = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    rollback = "PASS" if rollback == "PASS" and second_rollback_ids == baseline_unrestricted["token_ids"] else "FAIL"
    base_model = peft_model.unload()
    model.llava_model = base_model
    apply_prefix(model, bank, 0)
    rollback = "PASS" if rollback == "PASS" and state_weight_hash(model) == clean_hash else "FAIL"
    fresh_status = "NOT_RUN"
    if unrestricted_success:
        del peft_model, base_model, model
        torch.cuda.empty_cache()
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", "fresh", "--out-dir", str(out_dir), "--adapter-bank", str(diagnostic_root), "--physical-gpu", str(args.physical_gpu)]
        environment = dict(os.environ)
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        fresh_result = json.loads((diagnostic_root / "fresh_result.json").read_text()) if (diagnostic_root / "fresh_result.json").exists() else {"passed": False}
        fresh_status = "PASS" if completed.returncode == 0 and fresh_result["passed"] else "FAIL"
    gain = (
        final_metrics["primary_response_nll"] < initial_metrics["primary_response_nll"] - 0.1
        or final_metrics["primary"]["first_target_rank"] < initial_metrics["primary"]["first_target_rank"]
        or final_metrics["primary"]["first_target_margin"] > initial_metrics["primary"]["first_target_margin"] + 0.1
    )
    if rollback != "PASS" or reload_status != "PASS" or (unrestricted_success and fresh_status != "PASS"):
        label = "PC_INVALID_ENGINEERING_RUN"
    elif unrestricted_success:
        label = "PC_PASS_UNRESTRICTED_NATURAL_GENERATION"
    elif short_success:
        label = "PC_FORMAT_CONDITIONED_SUCCESS_ONLY"
    elif gain:
        label = "PC_DIRECTIONAL_GAIN_WITHOUT_NATURAL_GENERATION"
    else:
        label = "PC_NO_LEARNABILITY_UNDER_FIXED_LORA"
    strict_label = "PC_LOCALITY_STRICT_PRESERVED" if locality["strict_damage_count"] == 0 else "PC_LOCALITY_STRICT_DAMAGED"
    clinical_label = "PC_LOCALITY_CLINICAL_PRESERVED" if locality["clinical_failure_count"] == 0 else "PC_LOCALITY_CLINICAL_DAMAGED"
    summary = {
        "primary_label": label,
        "unrestricted_success": unrestricted_success,
        "success_step": success_step,
        "terminal_step": terminal_step,
        "exact_unrestricted_output": final_primary["raw_output"],
        "short_answer_success": short_success,
        "exact_short_output": final_short["raw_output"],
        "strict_locality_damage": locality["strict_damage_count"],
        "clinical_locality_failures": locality["clinical_failure_count"],
        "maximum_locality_nll_drift": locality["maximum_nll_drift"],
        "strict_locality_label": strict_label,
        "clinical_locality_label": clinical_label,
        "reload": reload_status,
        "fresh": fresh_status,
        "rollback": rollback,
        "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH,
        "stage2_permitted": False,
    }
    write_json(out_dir / "positive_control_summary.json", summary)
    write_json(out_dir / "final_target_metrics.json", {"initial": initial_metrics, "final": final_metrics})
    write_json(out_dir / "final_generation_parity.json", {"unrestricted": final_primary, "short": final_short, "three_path_parity": parity["passed"], "manual_cached_token_ids": parity["cached"]["token_ids"], "hf_token_ids": parity["hf"]["token_ids"]})
    write_json(out_dir / "final_locality_report.json", locality)
    write_json(out_dir / "adapter_reload_fresh_rollback.json", {"adapter_path": str(diagnostic_root), "adapter_sha256": adapter_hash(final_state), "reload": reload_status, "fresh": fresh_status, "rollback": rollback})
    bank_after = bank_manifest()
    append_jsonl(out_dir / "state_and_bank_hash_ledger.jsonl", {"event": "FINAL_ROLLBACK", "weight_hash": clean_hash, "bank_hash": bank_after["sha256"], "rollback": rollback})
    manifest.update({"primary_label": label, "canonical_bank_after": bank_after, "resolved_lora_module_count": len(resolved), "trainable_parameter_count": audit["trainable_numel"], "terminal_adapter_sha256": adapter_hash(final_state)})
    write_json(out_dir / "run_manifest.json", manifest)
    write_text(out_dir / "POSITIVE_CONTROL_FINAL_DECISION.md", final_report(summary))


def fresh(args: argparse.Namespace) -> None:
    if args.adapter_bank is None:
        raise ValueError("--adapter-bank required")
    seed_everything()
    state, metadata = load_adapter_payload(args.adapter_bank)
    model, views, bank, _records = load_model_views_bank(args.physical_gpu)
    apply_prefix(model, bank, 0)
    original = build_canonical_inputs(model, views[RECORD_ID]["target"])
    baseline = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"]
    resolved = resolve_target_modules(model.llava_model.named_modules())
    if resolved != metadata["resolved_lora_modules"]:
        raise RuntimeError("Fresh resolved-module mismatch")
    model.llava_model = insert_lora(model.llava_model, resolved)
    load_adapter_state(model.llava_model.named_parameters(), state)
    edited = full_generation_parity(model, original)["no_cache"]["token_ids"]
    expected = metadata["edited_unrestricted_token_ids"]
    reconstruction = edited == expected
    peft_model = model.llava_model
    peft_model.disable_adapter_layers()
    rollback = manual_greedy_trace(model, original, CAP, eos_ids(model), top_k=1)["token_ids"] == baseline
    passed = reconstruction and rollback and bank_manifest()["sha256"] == EXPECTED_BANK_HASH
    write_json(args.adapter_bank / "fresh_result.json", {"reconstruction": reconstruction, "rollback": rollback, "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH, "passed": passed})
    if not passed:
        raise RuntimeError("Fresh adapter proof failed")


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must equal {args.physical_gpu}")
    if args.mode == "fresh":
        fresh(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
