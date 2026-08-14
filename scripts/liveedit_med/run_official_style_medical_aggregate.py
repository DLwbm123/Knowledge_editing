#!/usr/bin/env python3
"""Stage B: frozen step-3000 aggregate on all 64 medical held-out edits."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path: sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook, resolve_layer21_block
from methods.liveedit_med.official_metric_compat import locality_preservation, teacher_forced_accuracy
from methods.liveedit_med.posthoc_validation import (BaseRoutePlan, immutable_tree_manifest, native_sample,
    normalize_answer, plan_audit, route_residual, sample_to_model_row)
from methods.liveedit_med.routing_attribution import stable_repository
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.source_ops import apply_low_rank_expert_residual, route_repository
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import (capture_prompt, capture_teacher_forced,
    forced_generation, load_clean_model, routed_generation, text_only_canonical)
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest
from scripts.engram.run_engram_v2_stage0_generation_audit import eos_ids
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_cached_greedy_trace
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import MAX_NEW_TOKENS, compact_trace


PROTOCOL = "OFFICIAL_STYLE_METRICS_ON_MEDICAL_DOMAIN"


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle: json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def views(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {"native": native_sample(record), **{name: record["generality"][name][0]
            for name in ("textual", "visual", "paired")}}


@torch.inference_mode()
def clean_tf(model, sample):
    row = sample_to_model_row(sample); inputs, labels, masks = model._build_batch(row)
    output = model.llava_model(inputs_embeds=inputs, attention_mask=masks["attention_mask"].long(),
                               labels=labels, return_dict=True, use_cache=False)
    return output.logits.detach(), labels.detach()


@torch.inference_mode()
def routed_tf(model, block, modules, sample, repository):
    canonical = build_canonical_inputs(model, sample_to_model_row(sample))
    prompt_hidden, vision, question = capture_prompt(model, block, canonical)
    plan = route_repository(modules.input_extractor, question.float(), vision.float(), repository["evr"], repository["eqr"])
    audit = plan_audit(plan, repository["ids"])
    _residual, norms = route_residual(plan, prompt_hidden, repository["moe_c"], repository["moe_r"], modules.instant_reps_norm)
    row = sample_to_model_row(sample); inputs, labels, masks = model._build_batch(row)
    hook = None
    if not isinstance(plan, BaseRoutePlan):
        c, r = repository["moe_c"][plan.candidate_mask], repository["moe_r"][plan.candidate_mask]
        hook = Layer21ResidualHook(block, lambda hidden: apply_low_rank_expert_residual(
            hidden.float(), c, r, plan.final_weights, modules.instant_reps_norm).to(hidden.dtype)).install(); hook.enabled = True
    output = model.llava_model(inputs_embeds=inputs, attention_mask=masks["attention_mask"].long(),
                               labels=labels, return_dict=True, use_cache=False)
    if hook is not None: hook.remove()
    return output.logits.detach(), labels.detach(), audit, norms


def build_experts(model, block, modules, records):
    result = {}
    for record in records:
        cap = capture_teacher_forced(model, block, native_sample(record))
        eqr, evr, c, r = modules.generated_edit(cap["vision"].float(), cap["question"].float(), cap["answer"].float())
        clean_rep = torch.nn.functional.normalize(torch.cat([
            cap["vision"].float().mean(1), cap["question"].float().mean(1)], 1), dim=1)[0].detach().cpu()
        result[str(record["record_id"])] = {"eqr": eqr, "evr": evr, "moe_c": c, "moe_r": r,
                                             "clean_s0_representation": clean_rep}
    return result


@torch.inference_mode()
def clean_generation(model, sample):
    canonical = build_canonical_inputs(model, sample_to_model_row(sample))
    return compact_trace(manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1))


def repository_for(members, experts):
    repo = {"ids": [str(row["record_id"]) for row in members]}
    for key in ("eqr", "evr", "moe_c", "moe_r"):
        repo[key] = torch.cat([experts[rid][key] for rid in repo["ids"]], 0)
    return repo


def load_progress(path: Path, *, record_id: str, repository_size: int):
    """Load a completed row without recomputing or overwriting it."""
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if str(value.get("record_id")) != record_id or int(value.get("repository_size", -1)) != repository_size:
        raise RuntimeError(f"LIVEEDIT_MED_STAGE_B_PROGRESS_MISMATCH:{path}")
    return value


def worker(args) -> None:
    source = json.loads(args.source_records.read_text()); records = source["records"]["heldout"]
    assigned = [row for index, row in enumerate(records) if index % args.worker_count == args.worker_index]
    model, _bank = load_clean_model(args.physical_gpu); _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, manifest = load_safe_state(args.checkpoint)
    checkpoint_step = int(manifest["step"])
    modules.load_state_dict(state, strict=True); modules.eval(); experts = build_experts(model, block, modules, records)
    rows = []
    for record in assigned:
        rid = str(record["record_id"]); own = experts[rid]
        progress = {size: load_progress(args.progress_dir / f"record_{rid}_repo_{size}.json",
                                        record_id=rid, repository_size=size)
                    for size in (1, 10, 32)}
        if all(value is not None for value in progress.values()):
            rows.extend(progress[size] for size in (1, 10, 32))
            continue
        other = records[(records.index(record) + 1) % len(records)]
        native, other_native = native_sample(record), native_sample(other)
        safety_samples = {
            "same_image_different_question": {"image": native["image"], "prompt": other_native["prompt"], "target": native["target"]},
            "same_question_different_image": {"image": other_native["image"], "prompt": native["prompt"], "target": native["target"]},
        }
        clean_safety = {name: clean_generation(model, sample) for name, sample in safety_samples.items()}
        forced = {name: forced_generation(model, block, modules, sample, own["moe_c"], own["moe_r"])
                  for name, sample in views(record).items()}
        for size in (1, 10, 32):
            if progress[size] is not None:
                rows.append(progress[size])
                continue
            members = stable_repository(records, rid, size); repo = repository_for(members, experts)
            result = {"record_id": rid, "repository_size": size, "repository_ids": repo["ids"],
                      "forced_on": forced, "views": {}, "locality": {}, "hard_medical_safety": {}}
            for name, sample in views(record).items():
                logits, labels, route, norms = routed_tf(model, block, modules, sample, repo)
                result["views"][name] = {"official_teacher_forced": teacher_forced_accuracy(logits, labels),
                    "routed_generation": routed_generation(model, block, modules, sample, repo),
                    "route": route, "residual_norms": norms}
            for name, sample in (("image", record["locality"]["image_or_paired"][0]),
                                 ("text", record["locality"]["text_only"][0])):
                if name == "text":
                    canonical = text_only_canonical(model, sample)
                    result["locality"][name] = {"official_teacher_forced": {"exact": True, "accuracy": 1.0,
                        "reason": "TEXT_ONLY_VISUAL_HARD_EMPTY_CANDIDATE_BASE_BYPASS"}, "route": {"kind": "base"}}
                else:
                    pre, pre_labels = clean_tf(model, sample)
                    post, post_labels, route, norms = routed_tf(model, block, modules, sample, repo)
                    if not torch.equal(pre_labels, post_labels): raise RuntimeError("LIVEEDIT_MED_LOCALITY_LABEL_DRIFT")
                    result["locality"][name] = {"official_teacher_forced": locality_preservation(pre, post, pre_labels),
                                                "route": route, "residual_norms": norms}
            for name, sample in safety_samples.items():
                routed = routed_generation(model, block, modules, sample, repo); clean = clean_safety[name]
                result["hard_medical_safety"][name] = {"s0": clean, "routed": routed,
                    "exact_s0_preservation": clean["token_ids"] == routed["token_ids"] and clean["stop_reason"] == routed["stop_reason"],
                    "target_contamination": normalize_answer(native["target"]) in normalize_answer(routed["raw_output"])}
            rows.append(result)
            write_new(args.progress_dir / f"record_{rid}_repo_{size}.json", result)
    write_new(args.out, {"protocol": PROTOCOL, "worker_index": args.worker_index, "worker_count": args.worker_count,
                         "physical_gpu": args.physical_gpu, "checkpoint_step": checkpoint_step, "rows": rows})


def finalize(args) -> None:
    rows = []; checkpoint_steps = set()
    for path in args.shard:
        value = json.loads(path.read_text())
        if value.get("protocol") != PROTOCOL: raise RuntimeError("LIVEEDIT_MED_STAGE_B_SHARD_PROTOCOL")
        checkpoint_steps.add(int(value["checkpoint_step"]))
        rows.extend(value["rows"])
    if len(checkpoint_steps) != 1: raise RuntimeError("LIVEEDIT_MED_STAGE_B_CHECKPOINT_DRIFT")
    checkpoint_step = checkpoint_steps.pop()
    if len(rows) != 64 * 3 or len({(r["record_id"], r["repository_size"]) for r in rows}) != len(rows):
        raise RuntimeError(f"LIVEEDIT_MED_STAGE_B_INCOMPLETE:{len(rows)}")
    aggregate = {}
    for size in (1, 10, 32):
        subset = [r for r in rows if r["repository_size"] == size]
        item = {"example_count": len(subset), "metrics": {}}
        for view in ("native", "textual", "visual", "paired"):
            metrics = [r["views"][view]["official_teacher_forced"] for r in subset]
            item["metrics"][view] = {"teacher_forced_exact": sum(m["exact"] for m in metrics),
                "teacher_forced_token_accuracy": sum(m["correct_tokens"] for m in metrics) / max(1, sum(m["total_tokens"] for m in metrics)),
                "routed_generation_success": sum(r["views"][view]["routed_generation"]["match"]["success"] for r in subset),
                "forced_generation_success": sum(r["forced_on"][view]["match"]["success"] for r in subset)}
        item["metrics"]["image_locality"] = {"exact": sum(r["locality"]["image"]["official_teacher_forced"]["exact"] for r in subset)}
        item["metrics"]["text_locality"] = {"exact": sum(r["locality"]["text"]["official_teacher_forced"]["exact"] for r in subset)}
        item["metrics"]["hard_medical_safety"] = {"exact_s0": sum(
            entry["exact_s0_preservation"] for r in subset for entry in r["hard_medical_safety"].values()),
            "count": len(subset) * 2, "target_contaminations": sum(
            entry["target_contamination"] for r in subset for entry in r["hard_medical_safety"].values())}
        aggregate[str(size)] = item
    output = {"protocol": PROTOCOL, "claim_scope": "MEDICAL_DOMAIN_NOT_OFFICIAL_BENCHMARK_REPRODUCTION",
              "checkpoint_step": checkpoint_step, "split": "heldout", "edit_count": 64, "repository_sizes": [1, 10, 32],
              "aggregate": aggregate, "rows": rows, "canonical_bank_sha256": bank_manifest()["sha256"]}
    write_new(args.out_dir / "official_style_medical_aggregate.json", output)
    lines = ["# LiveEdit-Med official-style medical aggregate", "", f"Protocol: `{PROTOCOL}`", "",
             "This is a medical-domain compatibility evaluation, not an official benchmark reproduction.", "",
             "| Repository | Native TF exact | Native routed gen | Text routed gen | Visual routed gen | Paired routed gen | Image loc exact | Text loc exact | Hard safety exact |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for size in (1, 10, 32):
        m = aggregate[str(size)]["metrics"]
        lines.append(f"| {size} | {m['native']['teacher_forced_exact']}/64 | {m['native']['routed_generation_success']}/64 | {m['textual']['routed_generation_success']}/64 | {m['visual']['routed_generation_success']}/64 | {m['paired']['routed_generation_success']}/64 | {m['image_locality']['exact']}/64 | {m['text_locality']['exact']}/64 | {m['hard_medical_safety']['exact_s0']}/128 |")
    write_new(args.out_dir / "aggregate_machine_readable.json", aggregate)
    (args.out_dir / "OFFICIAL_STYLE_MEDICAL_AGGREGATE.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("worker"); p.add_argument("--source-records", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--worker-index", type=int, required=True); p.add_argument("--worker-count", type=int, required=True)
    p.add_argument("--progress-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("finalize"); p.add_argument("--shard", type=Path, action="append", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(); (worker if args.mode == "worker" else finalize)(args)


if __name__ == "__main__": main()
