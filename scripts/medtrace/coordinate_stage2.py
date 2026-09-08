#!/usr/bin/env python3
"""One detached bounded Stage2 pipeline, then Judge and aggregate closeout."""
import argparse
from collections import Counter
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.run_stage2 import read, vf, sw
from scripts.medtrace.coordinate_selective_write import environment, JUDGE, JUDGE_PYTHON
from scripts.medtrace.neutral_entrypoint import neutral_command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--public-dir", type=Path, required=True)
    args = p.parse_args()
    run, public = args.run_root, args.public_dir
    config = read(run / "private/CAMPAIGN_CONFIG.json")
    if (run / "PIDS.json").exists() or (run / "STOP").exists():
        raise RuntimeError("existing attempt/stop requires explicit recovery")
    manifest = read(public / "NEW_EPISODE_MANIFEST_PUBLIC.json")
    if manifest.get("status") in {"PENDING", "NOT_SCANNED"}:
        raise RuntimeError("complete the authorized source scan before frozen queue launch")
    started = config.get('campaign_epoch', time.time())
    processes, exits, resources = {}, {}, {}
    status = dict(status="RUNNING", publication="PENDING_PUBLIC_PUSH")
    vf.atomic_json(run / "COORDINATOR_START.json", dict(epoch=started, pid=os.getpid(), wall_limit_seconds=24*3600))
    cpu = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    runner = [sys.executable, str(ROOT / "scripts/medtrace/run_stage2.py")]
    finalizer = [sys.executable, str(ROOT / "scripts/medtrace/finalize_stage2.py")]

    def elapsed():
        return time.time()-started

    def launch(name, command, env):
        visible, env = neutral_command(command, env, "run" if name.startswith("worker") else "job")
        with (run / f"{name}.log").open("x") as log:
            processes[name] = subprocess.Popen(visible, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        vf.atomic_json(run / "PIDS.json", dict(coordinator=os.getpid(), **{k: v.pid for k, v in processes.items()}))

    def stop(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def wait(names, limit):
        while any(processes[n].poll() is None for n in names):
            if elapsed() >= limit or (run / "STOP").exists():
                for name in names:
                    stop(processes[name])
                break
            time.sleep(2)
        for name in names:
            exits[name] = processes[name].returncode
        return all(exits[n] == 0 for n in names)

    def cpu_stage(name, action):
        launch(name, [*finalizer, action, '--run-root', str(run), '--public-dir', str(public)], cpu)
        if not wait([name], 24*3600):
            raise RuntimeError(name+' failed')

    def judge_stage(name, directory):
        if not read(directory / 'JUDGE_SIDECAR_PRIVATE.json')['new']:
            return
        env = None
        for gpu in sw.GPUS:
            try:
                env = environment(gpu, judge=True)
                break
            except RuntimeError:
                continue
        if env is None:
            raise RuntimeError('Judge memory unavailable; no unrelated jobs stopped')
        launch(name, [JUDGE_PYTHON, str(ROOT / 'scripts/medtrace/run_fixed_judge_vllm.py'),
            '--model-path', JUDGE, '--packet', str(directory / 'JUDGE_PACKET_PRIVATE.jsonl'),
            '--lock', str(Path(config['runtime']['cpu_gate']).parent / 'private/JUDGE_LOCK_V4.json'),
            '--output', str(directory / 'JUDGE_OUTPUT_PRIVATE.jsonl'),
            '--execution-lock', str(directory / 'JUDGE_EXECUTION_LOCK_PRIVATE.json'),
            '--preflight-output', str(directory / 'JUDGE_LENGTH_PREFLIGHT_PRIVATE.json'), '--max-model-len', '2048'], env)
        if not wait([name], 24*3600):
            raise RuntimeError(name+' failed')

    def next_workers(prefix, base_only=False):
        names = []
        for gpu in sw.GPUS:
            try:
                if not base_only and sw.gpu_check(gpu)['free_mib'] < 24576:
                    raise RuntimeError('new BalancEdit worker needs 24 GiB free')
                env = environment(gpu)
                name = prefix+gpu
                launch(name, [*runner, 'worker', '--run-root', str(run), *(['--base-only'] if base_only else [])], env)
                names.append(name)
            except RuntimeError:
                continue
        if not names:
            raise RuntimeError(prefix+' has no sufficiently free authorized device')
        wait(names, 20*3600)
        return names

    try:
        for gpu in sw.GPUS:
            try:
                resources[gpu] = sw.gpu_check(gpu)
                free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu], text=True))
                # Frozen 7B backbone + FP32 full transform, gradient and Adam states + peak margin.
                if free < 24*1024:
                    raise RuntimeError("BalancEdit stage requires 24 GiB free; unrelated jobs remain untouched")
                launch("worker_gpu"+gpu, [*runner, "worker", "--run-root", str(run)], environment(gpu))
            except RuntimeError as error:
                resources[gpu] = dict(unavailable=str(error))
        vf.atomic_json(public / "GPU_AND_TIMING.json", dict(preflight=resources, worker_free_mib=24576,
            judge_free_policy="ceil(.8*total)+2048 MiB", sharing_allowed=True, wall_hours_limit=24, gpu_hours_limit=48))
        workers = list(processes)
        if not workers:
            raise RuntimeError("no authorized GPU has sufficient current free memory")
        wait(workers, 20*3600)
        queue = vf.TaskQueue(run / "private/TASK_QUEUE.json", run)
        for task in queue.snapshot()["tasks"]:
            if task["status"] == "RUNNING":
                queue.update(task["task_id"], "FAILED", last_error="worker exited before atomic endpoint closure")
        if (run / "STOP").exists():
            raise RuntimeError("explicit STOP: no further model calls")
        if (run / 'private/BASE_TASK_QUEUE.json').exists() and elapsed() < 20*3600:
            # Old DEV16, then new Base-before generation/Judge, then new students.
            # No held-out student output exists when H_keep/U_keep are frozen.
            next_workers('worker_base_gpu', base_only=True)
            base_tasks = read(run / 'private/BASE_TASK_QUEUE.json')['tasks']
            if not all(t['status'] == 'COMPLETE' for t in base_tasks):
                raise RuntimeError('new Base-before generation incomplete; new students remain blocked')
            cpu_stage('prepare_base_judge', 'prepare-base-judge')
            judge_stage('base_judge', run / 'private/base_judge')
            cpu_stage('lock_base_before', 'lock-base-before')
            next_workers('worker_new_gpu')
        for task in queue.snapshot()['tasks']:
            if task['status'] == 'RUNNING':
                queue.update(task['task_id'], 'FAILED', last_error='worker exited before atomic endpoint closure')
        queue.cancel_pending()
        launch("prepare_judge", [*finalizer, "prepare-judge", "--run-root", str(run), "--public-dir", str(public)], cpu)
        if not wait(["prepare_judge"], 24*3600):
            raise RuntimeError("Judge packet preparation failed")
        directory = run / "private/judge"
        if read(directory / "JUDGE_SIDECAR_PRIVATE.json")["new"]:
            judge_env = None
            for gpu in sw.GPUS:
                try:
                    judge_env = environment(gpu, judge=True)
                    break
                except RuntimeError:
                    continue
            if judge_env is None:
                raise RuntimeError("Judge memory unavailable: raw endpoints preserved")
            launch("judge", [JUDGE_PYTHON, str(ROOT / "scripts/medtrace/run_fixed_judge_vllm.py"),
                "--model-path", JUDGE, "--packet", str(directory / "JUDGE_PACKET_PRIVATE.jsonl"),
                "--lock", str(Path(config["runtime"]["cpu_gate"]).parent / "private/JUDGE_LOCK_V4.json"),
                "--output", str(directory / "JUDGE_OUTPUT_PRIVATE.jsonl"),
                "--execution-lock", str(directory / "JUDGE_EXECUTION_LOCK_PRIVATE.json"),
                "--preflight-output", str(directory / "JUDGE_LENGTH_PREFLIGHT_PRIVATE.json"),
                "--max-model-len", "2048"], judge_env)
            if not wait(["judge"], 24*3600):
                raise RuntimeError("Judge failed; no incomplete verdicts scored")
        launch("finalize", [*finalizer, "finalize", "--run-root", str(run), "--public-dir", str(public)], cpu)
        if not wait(["finalize"], 24*3600):
            raise RuntimeError("finalizer failed")
        status.update(read(public / "RUN_COMPLETION.json"))
    except Exception as error:
        status.update(status="INCOMPLETE", error_type=type(error).__name__)
        vf.atomic_json(run / "FAILURE_PRIVATE.json", dict(error=f"{type(error).__name__}: {error}"))
    finally:
        for name, process in processes.items():
            stop(process)
            exits[name] = process.returncode
        status.update(process_exit_codes=exits, wall_seconds=elapsed(),
            counts=dict(Counter(t["status"] for t in read(run / "private/TASK_QUEUE.json")["tasks"])))
        vf.atomic_json(public / "RUN_COMPLETION.json", status)
        vf.atomic_json(public / "GPU_AND_TIMING.json", dict(preflight=resources, sharing_allowed=True,
            worker_free_mib=24576, judge_free_policy="ceil(.8*total)+2048 MiB", wall_seconds=elapsed(),
            gpu_hours_upper_bound=2*elapsed()/3600, wall_hours_limit=24, gpu_hours_limit=48, process_exit_codes=exits))
        vf.atomic_json(run / "RUN_COMPLETION.json", status)
    if status["status"] == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
