#!/usr/bin/env python3
"""Bounded coordinator for the existing visual-verifier CLI; exits with this attempt."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.run_frozen_expert_visual_verifier import GPU_UUIDS, _task_results, _verdicts, atomic_json, validate_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    args = parser.parse_args()
    run, public = args.run_root.resolve(), args.public_dir.resolve()
    config = json.loads((run / "private/CAMPAIGN_CONFIG.json").read_text())
    old, execution = Path(config["old_run"]), Path(config["execution_run"])
    runtime = json.loads((old / "private/CAMPAIGN_RUNTIME_CONFIG.json").read_text())
    v4 = Path(runtime["cpu_gate"]).parent
    llava = "/remote-home/wangbomin/worktrees/llava-official-30697ca-20260904T090859Z"
    model = "/remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b"
    vision = "/remote-home/wangbomin/hugging_cache/openai/clip-vit-large-patch14-336"
    judge_python = "/remote-home/wangbomin/evoclinician/venvs/vllm-0.9.2-py312/bin/python"
    judge_model = "/remote-home/wangbomin/.cache/huggingface/hub/models--Qwen--Qwen3-32B-AWQ/snapshots/0499c3ac83fdef8810b907a23894ba91e95eddd8"
    runner = [sys.executable, str(ROOT / "scripts/medtrace/run_frozen_expert_visual_verifier.py")]
    commands, processes, exits, phase_times = {}, {}, {}, {}
    start = None

    def environment(gpu):
        identity, memory = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True).strip().split(", ")
        if identity != GPU_UUIDS[gpu] or int(memory) > 1000:
            raise RuntimeError(f"GPU{gpu} unavailable or UUID mismatch; no other process was stopped")
        return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=gpu,
                    M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES="2,3", M3BENCH_FORMAL_EXPECTED_GPU_UUID=identity,
                    M3BENCH_EXPECTED_LLAVA_SOURCE=llava, M3BENCH_MODEL_PATH=model, M3BENCH_VISION_PATH=vision,
                    PYTHONPATH=f"{llava}:{ROOT}")

    def launch(name, command, env=None):
        commands[name] = command
        atomic_json(run / "COMMANDS_PRIVATE.json", commands)
        with (run / f"{name}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env or dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        atomic_json(run / "PIDS.json", {key: value.pid for key, value in processes.items()})
        return process

    def remaining():
        return max(0, 12 * 3600 - (time.time() - start["epoch"]))

    def wait(name):
        began = time.monotonic()
        exits[name] = processes[name].wait(timeout=remaining())
        phase_times[name] = time.monotonic() - began
        atomic_json(run / "PROCESS_EXIT_CODES.json", exits)
        if exits[name]:
            raise RuntimeError(f"{name} exited {exits[name]}")

    completion = {"initialization": "FAIL", "tasks": {"completed": 0, "expected": 21, "failed": 0},
                  "raw_closure": "PARTIAL", "judge": "NOT_RUN", "evaluation": "NOT_RUN", "publication": "RETRY_REQUIRED",
                  "scientific_gain": "not_evaluated", "gpu1_used": False}
    try:
        env2 = environment("2")
        subprocess.run([*runner, "start", "--run-root", str(run)], check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        start = validate_start(run)
        completion["initialization"] = "PASS"
        worker = [*runner, "worker", "--run-root", str(run), "--old-run", str(old), "--execution-run", str(execution), "--expected-code-commit", config["code_commit"]]
        launch("worker_gpu2", worker, env2)
        first = json.loads((run / "private/TASK_QUEUE.json").read_text())["tasks"][0]["task_id"]
        while remaining() > 1800:
            queue = json.loads((run / "private/TASK_QUEUE.json").read_text())["tasks"]
            task = next(row for row in queue if row["task_id"] == first)
            if task["status"] == "COMPLETE":
                result = next(result for result in _task_results(run) if result["task"]["task_id"] == first)
                atomic_json(run / "FIRST_TASK_PASS.json", {"task_id": first, "result_sha256": task["result_sha256"], "timing": result["timing"], "at": time.time()})
                break
            if task["status"] == "FAILED" or processes["worker_gpu2"].poll() is not None:
                raise RuntimeError("first original task failed integration; see worker log")
            time.sleep(2)
        else:
            raise RuntimeError("budget ended before first task completed")
        try:
            env3 = environment("3")
        except RuntimeError as error:
            atomic_json(run / "GPU3_UNAVAILABLE.json", {"reason": str(error), "fallback": "GPU2 continues alone"})
        else:
            launch("worker_gpu3", worker, env3)
        for name in tuple(processes):
            wait(name)
        results = _task_results(run)
        completion["tasks"]["completed"] = len(results)
        if len(results) != 21:
            raise RuntimeError(f"only {len(results)}/21 task manifests complete")
        completion["raw_closure"] = "COMPLETE"
        judge = run / "private/judge"
        packet, sidecar, output = [judge / name for name in ("JUDGE_PACKET_PRIVATE.jsonl", "JUDGE_SIDECAR_PRIVATE.json", "JUDGE_OUTPUT_PRIVATE.jsonl")]
        launch("prepare_judge", [*runner, "prepare-judge", "--run-root", str(run), "--execution-run", str(execution), "--packet", str(packet), "--sidecar", str(sidecar)])
        wait("prepare_judge")
        if packet.stat().st_size:
            launch("judge", [judge_python, str(ROOT / "scripts/medtrace/run_fixed_judge_vllm.py"), "--model-path", judge_model,
                             "--packet", str(packet), "--lock", str(v4 / "private/JUDGE_LOCK_V4.json"), "--output", str(output),
                             "--execution-lock", str(judge / "JUDGE_EXECUTION_LOCK_PRIVATE.json"),
                             "--preflight-output", str(judge / "JUDGE_LENGTH_PREFLIGHT_PRIVATE.json"), "--max-model-len", "2048"], environment("2"))
            wait("judge")
            completion["judge"] = "COMPLETE"
        else:
            output.touch(exist_ok=False)
            _verdicts(sidecar, output)
            completion["judge"] = "REUSED_COMPLETE"
        launch("finalize", [*runner, "finalize", "--run-root", str(run), "--old-run", str(old), "--judge-output", str(output), "--sidecar", str(sidecar), "--public-dir", str(public)])
        wait("finalize")
        completion["evaluation"] = "COMPLETE"
        summary = json.loads((public / "GPU_USAGE_AND_COMPLETION.json").read_text())
        completion["scientific_gain"] = summary["selected_simplest_condition"] is not None
    except Exception as error:
        completion["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for name, process in processes.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            exits[name] = process.returncode
        atomic_json(run / "PROCESS_EXIT_CODES.json", exits)
        tasks = json.loads((run / "private/TASK_QUEUE.json").read_text())["tasks"]
        completion["tasks"] = {"completed": sum(t["status"] == "COMPLETE" for t in tasks), "failed": sum(t["status"] == "FAILED" for t in tasks), "expected": 21,
                               "unfinished": sum(t["status"] not in {"COMPLETE", "FAILED"} for t in tasks)}
        completion.update(attempt_id=run.name, research_execution_commit=config["code_commit"], process_exit_codes=exits,
                          wall_seconds=time.time() - start["epoch"] if start else None, end_epoch=time.time(), phase_wait_seconds=phase_times)
        atomic_json(public / "RUN_COMPLETION.json", completion)
        atomic_json(run / "RUN_COMPLETION.json", completion)
        (run / "GPU_CLEANUP_STATUS").write_text(subprocess.check_output(["nvidia-smi", "-i", "2,3", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv"], text=True))


if __name__ == "__main__":
    main()
