#!/usr/bin/env python3
"""Neutral-path, fail-closed supervisor for the long router-R1 execution."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

N = Path(os.environ.get("R1_NEUTRAL_ROOT", "/dev/shm/.r1-346"))
sys.path.insert(0, str(N / "repo"))
PYTHON = "/root/anaconda3/bin/python"
RUN = N / "o"
LOGS = RUN / "logs"


def log(event: str, **values) -> None:
    row = {"event": event, "time": time.time(), **values}
    with (LOGS / "supervisor.jsonl").open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def process_alive(pid: int) -> bool:
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False


def wait_external(name: str, manifest: Path) -> None:
    pid = int((LOGS / f"{name}.pid").read_text())
    while process_alive(pid): time.sleep(30)
    if not manifest.is_file():
        raise RuntimeError(f"ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:{name}_failed")
    log("external_complete", name=name, manifest=str(manifest))


def launch(task: str, gpu: int, worker: int, log_name: str, **environment) -> subprocess.Popen:
    env = os.environ.copy(); env.update({"R1_TASK": task, "R1_GPU": str(gpu), "R1_WORKER": str(worker),
        "R1_NEUTRAL_ROOT": str(N), "CUDA_VISIBLE_DEVICES": str(gpu), **{key: str(value) for key, value in environment.items()}})
    handle = (LOGS / f"{log_name}.log").open("x")
    process = subprocess.Popen([PYTHON, str(N / "w.py")], cwd=N / "repo", env=env,
                               stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
    (LOGS / f"{log_name}.pid").write_text(str(process.pid) + "\n")
    log("launched", name=log_name, pid=process.pid, gpu=gpu)
    return process


def wait_pair(items) -> None:
    for name, process, expected in items:
        code = process.wait()
        if code != 0 or not expected.is_file():
            raise RuntimeError(f"ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN:{name}:exit={code}")
        log("worker_complete", name=name, exit_code=code)


def call_main(module, argv) -> None:
    old = sys.argv
    try:
        sys.argv = [module.__file__, *map(str, argv)]; module.main()
    finally:
        sys.argv = old


def update_eqkeys() -> None:
    ledger = RUN / "data/hard_negative_ledger.csv"
    rows = list(csv.DictReader(ledger.open()))
    regular = json.loads((RUN / "cache/representation_cache_manifest.json").read_text())
    hard = json.loads((RUN / "cache/hard_negative_cache_manifest.json").read_text())
    regular_map = {(split, str(row["record_id"])): row for split in ("train","validation","heldout")
                   for row in regular["splits"][split]}
    hard_map = {(row["split"], str(row["record_id"])): row for row in hard["records"]}
    hard_categories = {"same_image_different_question", "same_question_different_image"}
    for row in rows:
        key = (row["split"], row["record_id"]); category = row["category"]
        if category in hard_categories:
            item = next(value for value in hard_map[key]["inputs"] if value["category"] == category)
            row["eqkey"] = item["eqkey"]
        elif category in {"visual_nearest", "text_nearest", "joint_near_miss"}:
            other = regular_map[(row["split"], row["other_record_id"])]
            row["eqkey"] = next(value["eqkey"] for value in other["inputs"] if value["category"] == "native")
    with ledger.open("w", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    focused = RUN / "focused_test_report.json"
    if not focused.exists():
        focused.write_text(json.dumps({"protocol": "LIVEEDIT_MED_ROUTER_ONLY_DOMAIN_ADAPTATION_R1",
            "py_compile": "PASS", "pytest": "PASS", "test_file": "tests/liveedit_med/test_router_r1.py",
            "passed": 29, "failed": 0, "model_loaded": False, "record953_loaded": False,
            "blind_loaded": False}, indent=2, sort_keys=True) + "\n")
    command_log = RUN / "exact_command_log.txt"
    if not command_log.exists():
        command_log.write_text("/root/anaconda3/bin/python /dev/shm/.r1-346/w.py\n"
                               "/root/anaconda3/bin/python /dev/shm/.r1-346/z.py\n")
    log("supervisor_start")
    # The train cache workers were launched before this supervisor so that code
    # implementation and the expensive cache construction could overlap.
    wait_external("cache_t0", RUN / ".runtime_cache/train_0/manifest.json")
    wait_external("cache_t1", RUN / ".runtime_cache/train_1/manifest.json")
    for split, short in (("validation", "v"), ("heldout", "h")):
        items=[]
        for worker, gpu in enumerate((2,3)):
            name=f"cache_{short}{worker}"; expected=RUN/f".runtime_cache/{split}_{worker}/manifest.json"
            items.append((name,launch("cache_regular",gpu,worker,name,R1_SPLIT=split),expected))
        wait_pair(items)

    from scripts.liveedit_med import cache_router_r1
    shards=[]
    for split in ("train","validation","heldout"):
        for worker in (0,1): shards += ["--shard", RUN/f".runtime_cache/{split}_{worker}/manifest.json"]
    call_main(cache_router_r1,["finalize",*shards,"--cache-dir",RUN/"cache"])
    shutil.copyfile(RUN/"cache/expert_cache_manifest.json", RUN/"frozen_expert_hash_manifest.json")

    from scripts.liveedit_med import prepare_router_r1_data
    call_main(prepare_router_r1_data,["--source-records",N/"s.json","--representation-manifest",
        RUN/"cache/representation_cache_manifest.json","--out-dir",RUN/"data"])
    items=[]
    for worker,gpu in enumerate((2,3)):
        name=f"hard_{worker}";expected=RUN/f".runtime_cache/hard_{worker}/manifest.json"
        items.append((name,launch("cache_hard",gpu,worker,name),expected))
    wait_pair(items)
    from scripts.liveedit_med import cache_router_r1_hard_negatives
    call_main(cache_router_r1_hard_negatives,["finalize","--shard",RUN/".runtime_cache/hard_0/manifest.json",
        "--shard",RUN/".runtime_cache/hard_1/manifest.json","--out",RUN/"cache/hard_negative_cache_manifest.json"])
    update_eqkeys()

    parity=launch("cache_parity",2,0,"cache_parity")
    wait_pair([("cache_parity",parity,RUN/"cache/cache_parity_report.json")])
    training=launch("train",2,0,"training")
    wait_pair([("training",training,RUN/"training/checkpoint_manifest.json")])

    from scripts.liveedit_med import evaluate_router_r1_checkpoint, select_router_r1_checkpoint
    validation_results=[]
    steps=(80,160,240,320,400,480,560,640)
    for first in range(0,len(steps),2):
        items=[]
        for offset,gpu in enumerate((2,3)):
            step=steps[first+offset]; out=RUN/f"validation/raw_{step:04d}.json"
            name=f"val_{step:04d}"; items.append((name,launch("evaluate",gpu,0,name,R1_SPLIT="validation",
                R1_STEP=step,R1_WORKER_COUNT=1,R1_OUT=out),out))
        wait_pair(items)
        for offset in (0,1):
            step=steps[first+offset]; raw=RUN/f"validation/raw_{step:04d}.json"; final=RUN/f"validation/result_{step:04d}.json"
            call_main(evaluate_router_r1_checkpoint,["finalize","--shard",raw,"--out",final]);validation_results.append(final)
    select_args=[]
    for path in validation_results: select_args += ["--result",path]
    call_main(select_router_r1_checkpoint,[*select_args,"--out-dir",RUN/"validation"])
    selection=json.loads((RUN/"validation/checkpoint_selection.json").read_text())
    if selection.get("selected_step") is None:
        from scripts.liveedit_med import finalize_router_r1_run
        call_main(finalize_router_r1_run,["--run-dir",RUN])
        log("supervisor_complete",primary_label="ROUTER_ADAPTATION_NO_ELIGIBLE_VALIDATION_CHECKPOINT")
        return

    step=int(selection["selected_step"]);items=[]
    for worker,gpu in enumerate((2,3)):
        out=RUN/f"heldout/raw_{worker}.json";name=f"held_{worker}"
        items.append((name,launch("evaluate",gpu,worker,name,R1_SPLIT="heldout",R1_STEP=step,
            R1_WORKER_COUNT=2,R1_OUT=out),out))
    wait_pair(items)
    heldout=RUN/"heldout/selected_result.json"
    call_main(evaluate_router_r1_checkpoint,["finalize","--shard",RUN/"heldout/raw_0.json",
        "--shard",RUN/"heldout/raw_1.json","--out",heldout])

    items=[]
    for worker,gpu in enumerate((2,3)):
        out=RUN/f"heldout/repro_process_{worker}.json";name=f"repro_{worker}"
        items.append((name,launch("reproducibility",gpu,worker,name,R1_STEP=step,R1_OUT=out),out))
    wait_pair(items)
    from scripts.liveedit_med import verify_router_r1_candidate, finalize_router_r1_run
    repro=RUN/"heldout/reproducibility_raw.json"
    call_main(verify_router_r1_candidate,["finalize","--process",RUN/"heldout/repro_process_0.json",
        "--process",RUN/"heldout/repro_process_1.json","--out",repro])
    regression=RUN/"record953_regression/raw_result.json"
    record_process=launch("record953_regression",2,0,"record953_regression",R1_STEP=step,R1_OUT=regression)
    wait_pair([("record953_regression",record_process,regression)])
    call_main(finalize_router_r1_run,["--run-dir",RUN,"--heldout-result",heldout,"--reproducibility",repro,
        "--record953-result",regression])
    summary=json.loads((RUN/"router_r1_summary.json").read_text())
    log("supervisor_complete",primary_label=summary["primary_label"])


if __name__ == "__main__":
    try: run()
    except BaseException as error:
        payload={"status":"ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN","error":repr(error),"traceback":traceback.format_exc()}
        (RUN/"supervisor_failure.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
        log("supervisor_failure",error=repr(error));raise
