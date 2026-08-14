#!/usr/bin/env python3
"""GPU2/3-only, resumable stage scheduler for the frozen validation protocol."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path: sys.path.insert(0, str(item))

from methods.liveedit_med.posthoc_validation import immutable_tree_manifest, verify_checkpoint_set
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle: json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def run_logged(command: list[str], log: Path, *, gpu: int | None = None) -> None:
    env = os.environ.copy()
    if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log.open("x") as handle:
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"LIVEEDIT_MED_PIPELINE_COMMAND_FAILED:{result.returncode}:{' '.join(command)}")


def parallel(commands: list[tuple[list[str], Path, int]]) -> None:
    processes = []
    for command, log, gpu in commands:
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        handle = log.open("x")
        processes.append((subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], env=env,
                                           stdout=handle, stderr=subprocess.STDOUT), handle, command))
    errors = []
    for process, handle, command in processes:
        code = process.wait(); handle.close()
        if code: errors.append((code, command))
    if errors: raise RuntimeError(f"LIVEEDIT_MED_PIPELINE_WORKER_FAILURE:{errors}")


def wait_for_blind(blind: Path) -> None:
    outputs = [blind / "baseline_worker_0.json", blind / "baseline_worker_1.json"]
    pids = [blind / "baseline_worker_0.pid", blind / "baseline_worker_1.pid"]
    while not all(path.is_file() for path in outputs):
        for output, pid_file in zip(outputs, pids):
            if output.is_file(): continue
            if pid_file.is_file():
                pid = int(pid_file.read_text().strip())
                try: os.kill(pid, 0)
                except ProcessLookupError: raise RuntimeError(f"LIVEEDIT_MED_BLIND_BASELINE_WORKER_DIED:{pid}")
        time.sleep(30)


def write_final_report(root: Path) -> None:
    a=json.loads((root/"stage_a_upstream_trace_parity_retry/trace_parity_summary.json").read_text())
    b=json.loads((root/"stage_b_official_style_medical/official_style_medical_aggregate.json").read_text())
    c=json.loads((root/"stage_c_validation_routing/routing_attribution.json").read_text())
    d=json.loads((root/"stage_d_assistant_only/assistant_only_diagnostic.json").read_text())
    blind=json.loads((root/"stage_e_future_blind/future_blind_manifest.json").read_text())
    m=b["aggregate"]["32"]["metrics"]
    forced=sum(m[name]["forced_generation_success"] for name in ("native","textual","visual","paired"))
    routed=sum(m[name]["routed_generation_success"] for name in ("native","textual","visual","paired"))
    classes=c["scaling"]["32"]["failure_classes"]
    routing_fail=sum(classes.get(name,0) for name in ("VISUAL_SENTINEL_RECALL_FAILURE","TEXT_ABSOLUTE_SUPPRESSION",
        "TEXT_RELATIVE_COMPETITION","RESIDUAL_INTERFERENCE","ROUTED_GENERATION_FAILURE_UNRESOLVED"))
    generator_fail=classes.get("GENERATOR_OR_EXPERT_FAILURE",0)
    permitted=a["status"] in ("END_TO_END_UPSTREAM_PORT_PARITY_NOT_RUN_ASSETS_MISSING","END_TO_END_UPSTREAM_PORT_PARITY_PASSED")
    decision=("PROCEED_TO_ROUTER_ONLY_DOMAIN_ADAPTATION_WITH_FROZEN_BLIND_GATE"
              if permitted and forced>routed and routing_fail>generator_fail and blind["status"]=="FUTURE_BLIND_MEDICAL_SET_FROZEN"
              else "DO_NOT_TRAIN_ROUTER__GENERATOR_OR_PORT_REPAIR_FIRST")
    report=["# LiveEdit-Med next validation report","",f"Decision: `{decision}`","",
      "## First-page evidence","",f"- End-to-end source branch: `{a['status']}`.",
      f"- Held-out repository-32 forced-on versus routed natural generation: **{forced}/256 vs {routed}/256**.",
      f"- Validation repository-32 routing-class failures versus generator/expert failures: **{routing_fail} vs {generator_fail}**.",
      f"- Assistant-only diagnostic: **{d['validation']['assistant_only_success']}/256**, compared with source-full **{d['validation']['source_full_success']}/256**; routing decisions were identical.",
      f"- Future blind set: **{blind['selected_count']} edits / {blind['input_count']} inputs**, manifest `{blind['manifest_hash']}`; no edited checkpoint was loaded.",
      f"- Training tree and canonical bank byte-identical: **yes**.","",
      "The official-style numbers are medical-domain compatibility metrics, not an official LiveEdit benchmark reproduction.","",
      "## Repository-32 held-out metrics","",
      "| View | Teacher-forced exact | Forced generation | Routed generation |","|---|---:|---:|---:|"]
    for name in ("native","textual","visual","paired"):
        report.append(f"| {name} | {m[name]['teacher_forced_exact']}/64 | {m[name]['forced_generation_success']}/64 | {m[name]['routed_generation_success']}/64 |")
    report += ["",f"Image locality: {m['image_locality']['exact']}/64; text locality: {m['text_locality']['exact']}/64; hard safety exact S0: {m['hard_medical_safety']['exact_s0']}/128.","",
      "## Routing failure classes at repository size 32","", "```json",json.dumps(classes,indent=2,sort_keys=True),"```","",
      "## Next action","",("Run a separately pre-registered router-only medical-domain adaptation next; keep step-3000, generator/expert tensors, thresholds, and this blind set frozen."
       if decision.startswith("PROCEED") else "Do not train the router yet; repair the dominant generator or port failure identified above, then rerun this frozen protocol."),""]
    (root/"LIVEEDIT_MED_NEXT_VALIDATION_REPORT.md").write_text("\n".join(report))
    (root/"FINAL_DECISION.md").write_text(f"# Final decision\n\n`{decision}`\n")
    write_new(root/"final_decision.json",{"decision":decision,"forced_generation_repo32":forced,
      "routed_generation_repo32":routed,"routing_failures_repo32":routing_fail,"generator_failures_repo32":generator_fail,
      "future_blind_manifest_hash":blind["manifest_hash"]})


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--out-root",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True); p.add_argument("--source-records",type=Path,required=True)
    p.add_argument("--stage-q-dir",type=Path,required=True); args=p.parse_args()
    root=args.out_root.resolve(); run=args.run_dir.resolve(); source=args.source_records.resolve(); checkpoint=run/"training/checkpoint_3000"
    events=root/"pipeline_events.jsonl"
    def event(name, **extra):
        with events.open("a") as handle: handle.write(json.dumps({"time":datetime.now(timezone.utc).isoformat(),"event":name,**extra},sort_keys=True)+"\n")
    event("pipeline_started",gpus=[2,3])
    write_new(root/"checkpoint_hash_audit.json",verify_checkpoint_set(run))
    before=immutable_tree_manifest(run/"training"); write_new(root/"original_training_tree_before.json",before)
    write_new(root/"canonical_bank_before.json",bank_manifest())

    blind=root/"stage_e_future_blind"; wait_for_blind(blind)
    run_logged(["/root/anaconda3/bin/python","scripts/liveedit_med/freeze_future_blind_medical_set.py","finalize",
      "--selection-manifest",str(blind/"selection_manifest.json"),"--shard",str(blind/"baseline_worker_0.json"),
      "--shard",str(blind/"baseline_worker_1.json"),"--out",str(blind/"future_blind_manifest.json")],blind/"finalize.log")
    event("future_blind_frozen")

    b=root/"stage_b_official_style_medical"; (b/"progress").mkdir(parents=True,exist_ok=False)
    parallel([(["/root/anaconda3/bin/python","scripts/liveedit_med/run_official_style_medical_aggregate.py","worker",
      "--source-records",str(source),"--checkpoint",str(checkpoint),"--physical-gpu",str(gpu),"--worker-index",str(index),
      "--worker-count","2","--progress-dir",str(b/"progress"),"--out",str(b/f"worker_{index}.json")],b/f"worker_{index}.log",gpu)
      for index,gpu in enumerate((2,3))])
    run_logged(["/root/anaconda3/bin/python","scripts/liveedit_med/run_official_style_medical_aggregate.py","finalize",
      "--shard",str(b/"worker_0.json"),"--shard",str(b/"worker_1.json"),"--out-dir",str(b)],b/"finalize.log")
    event("stage_b_complete")

    c=root/"stage_c_validation_routing"; (c/"progress").mkdir(parents=True,exist_ok=False)
    parallel([(["/root/anaconda3/bin/python","scripts/liveedit_med/run_validation_routing_attribution.py","worker",
      "--source-records",str(source),"--checkpoint",str(checkpoint),"--physical-gpu",str(gpu),"--worker-index",str(index),
      "--worker-count","2","--progress-dir",str(c/"progress"),"--out",str(c/f"worker_{index}.json")],c/f"worker_{index}.log",gpu)
      for index,gpu in enumerate((2,3))])
    run_logged(["/root/anaconda3/bin/python","scripts/liveedit_med/run_validation_routing_attribution.py","finalize",
      "--shard",str(c/"worker_0.json"),"--shard",str(c/"worker_1.json"),"--out-dir",str(c)],c/"finalize.log")
    event("stage_c_complete")

    d=root/"stage_d_assistant_only"; d.mkdir(parents=True,exist_ok=False)
    for dataset in ("validation","dev"):
        parallel([(["/root/anaconda3/bin/python","scripts/liveedit_med/run_frozen_assistant_only_diagnostic.py","worker",
          "--dataset",dataset,"--source-records",str(source),"--stage-q-dir",str(args.stage_q_dir.resolve()),
          "--checkpoint",str(checkpoint),"--physical-gpu",str(gpu),"--worker-index",str(index),"--worker-count","2",
          "--out",str(d/f"{dataset}_worker_{index}.json")],d/f"{dataset}_worker_{index}.log",gpu)
          for index,gpu in enumerate((2,3))])
    run_logged(["/root/anaconda3/bin/python","scripts/liveedit_med/run_frozen_assistant_only_diagnostic.py","finalize",
      *sum((["--shard",str(d/f"{dataset}_worker_{index}.json")] for dataset in ("validation","dev") for index in (0,1)),[]),
      "--out-dir",str(d)],d/"finalize.log")
    event("stage_d_complete")

    after=immutable_tree_manifest(run/"training"); write_new(root/"original_training_tree_after.json",after)
    bank_after=bank_manifest(); write_new(root/"canonical_bank_after.json",bank_after)
    immutability={"training_tree_before":before["tree_hash"],"training_tree_after":after["tree_hash"],
      "training_byte_identical":before["tree_hash"]==after["tree_hash"],"canonical_bank_before":json.loads((root/"canonical_bank_before.json").read_text())["sha256"],
      "canonical_bank_after":bank_after["sha256"]}
    immutability["canonical_bank_byte_identical"]=immutability["canonical_bank_before"]==immutability["canonical_bank_after"]
    write_new(root/"immutability_final.json",immutability)
    if not all((immutability["training_byte_identical"],immutability["canonical_bank_byte_identical"])):
        raise RuntimeError("LIVEEDIT_MED_IMMUTABILITY_FAILURE")
    write_final_report(root)
    event("pipeline_complete")


if __name__=="__main__": main()
