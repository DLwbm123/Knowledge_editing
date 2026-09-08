#!/usr/bin/env python3
"""Detached bounded Stage3 queue, full-answer Judge, and terminal aggregate closure."""
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
from scripts.medtrace.run_stage3 import preflight, public_manifest, read, sw, vf
from scripts.medtrace.coordinate_selective_write import environment, JUDGE, JUDGE_PYTHON
from scripts.medtrace.neutral_entrypoint import neutral_command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    args = p.parse_args()
    run, public = args.run_root, args.run_root/'public'
    config = read(run/'private/CAMPAIGN_CONFIG.json')
    if (run/'PIDS.json').exists() or (run/'STOP').exists():
        raise ValueError('existing run requires explicit recovery, not a new timer')
    public_manifest(run)
    start_path = run/'private/CAMPAIGN_START.json'
    if not start_path.exists():
        vf.atomic_json(start_path, dict(epoch=time.time(), code_commit=config['code_commit'], coordinator_pid=os.getpid()))
    config = preflight(run)
    started = read(start_path)['epoch']
    vf.atomic_json(run/'COORDINATOR_START.json', dict(epoch=started, pid=os.getpid(), wall_limit_seconds=86400))
    queue = vf.TaskQueue(run/'private/TASK_QUEUE.json', run)
    children, exits = {}, {}
    cpu = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    finalizer = [sys.executable, str(ROOT/'scripts/medtrace/finalize_stage3.py')]
    status = dict(status='RUNNING', compute='RUNNING', judge='NOT_RUN', publication='PENDING')

    def elapsed():
        return time.time()-started

    def launch(name, command, env):
        command, env = neutral_command(command, env, 'run' if name.startswith('worker') else 'job')
        with (run/(name+'.log')).open('x') as log:
            children[name] = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        vf.atomic_json(run/'PIDS.json', dict(coordinator=os.getpid(), **{n:p.pid for n,p in children.items()}))

    def stop(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def wait(name):
        proc = children[name]
        while proc.poll() is None and elapsed() < 86400 and not (run/'STOP').exists():
            time.sleep(2)
        stop(proc)
        exits[name] = proc.returncode
        if proc.returncode:
            raise RuntimeError(name+' failed; private log retained')

    def finalize_action(name, action, cap=2048):
        launch(name, [*finalizer, action, '--run-root', str(run), '--public-dir', str(public),
                      '--max-model-len', str(cap)], cpu)
        wait(name)

    def judge_command(directory, cap, preflight_only=False):
        return [JUDGE_PYTHON, str(ROOT/'scripts/medtrace/run_fixed_judge_vllm.py'),
            '--model-path', JUDGE, '--packet', str(directory/'JUDGE_PACKET_PRIVATE.jsonl'),
            '--lock', str(Path(config['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json'),
            '--output', str(directory/'JUDGE_OUTPUT_PRIVATE.jsonl'),
            '--execution-lock', str(directory/'JUDGE_EXECUTION_LOCK_PRIVATE.json'),
            '--preflight-output', str(directory/('CPU_LENGTH_PRIVATE.json' if preflight_only else 'JUDGE_LENGTH_PREFLIGHT_PRIVATE.json')),
            '--max-model-len', str(cap), *(['--preflight-only'] if preflight_only else [])]

    try:
        rounds = Counter()
        live = {}
        while elapsed() < 72000 and not (run/'STOP').exists():
            for gpu, name in list(live.items()):
                proc = children[name]
                if proc.poll() is None:
                    continue
                exits[name] = proc.returncode
                for t in queue.snapshot()['tasks']:
                    if t['status'] == 'RUNNING' and t.get('pid') == proc.pid:
                        queue.update(t['task_id'], 'FAILED', last_error='worker exited before endpoint closure')
                del live[gpu]
            pending = [t for t in queue.snapshot()['tasks'] if t['status'] == 'PENDING']
            if not pending and not live:
                break
            # Repeated same integration fault pauses its track, not unrelated work.
            failed = [t for t in queue.snapshot()['tasks'] if t['status'] == 'FAILED']
            faults = Counter((t['kind'], t.get('last_error')) for t in failed)
            paused = {kind for (kind, error), count in faults.items() if count >= 2}
            for t in pending:
                if t['kind'] in paused:
                    queue.update(t['task_id'], 'PAUSED_INTEGRATION', pause_reason='repeated same track integration fault')
            pending = [t for t in queue.snapshot()['tasks'] if t['status'] == 'PENDING']
            for gpu in sw.GPUS:
                if gpu in live or not pending:
                    continue
                try:
                    if sw.gpu_check(gpu)['free_mib'] < 24576:
                        continue
                    env = environment(gpu)
                except RuntimeError:
                    continue
                rounds[gpu] += 1
                name = f'worker_gpu{gpu}_{rounds[gpu]:03d}'
                launch(name, [sys.executable, str(ROOT/'scripts/medtrace/run_stage3.py'), '--run-root', str(run)], env)
                live[gpu] = name
            if not pending and not live:
                break
            time.sleep(5)
        for name in live.values():
            stop(children[name]); exits[name] = children[name].returncode
        for t in queue.snapshot()['tasks']:
            if t['status'] in {'RUNNING','PENDING'}:
                queue.update(t['task_id'], 'PENDING_RESUME', reason='bounded generation window ended')
        status['compute'] = 'CLOSED'
        if (run/'STOP').exists():
            raise RuntimeError('explicit STOP prevents additional model calls')
        finalize_action('prepare_judge', 'prepare-judge')
        directory = run/'private/judge'
        cap = read(directory/'JUDGE_SIDECAR_PRIVATE.json')['max_model_len']
        if read(directory/'JUDGE_SIDECAR_PRIVATE.json')['new']:
            env = None
            while elapsed() < 86400 and env is None:
                for gpu in sw.GPUS:
                    try:
                        env = environment(gpu, judge=True)
                        break
                    except RuntimeError:
                        continue
                if env is None:
                    time.sleep(30)
            if env is None:
                raise RuntimeError('Judge GPU memory unavailable within budget')
            launch('judge', judge_command(directory, cap), env)
            wait('judge')
        finalize_action('finalize', 'finalize', cap)
        status = read(public/'EXECUTION_STATUS.json')
    except Exception as error:
        status.update(status='PARTIAL', error_type=type(error).__name__)
        vf.atomic_json(run/'FAILURE_PRIVATE.json', dict(error=str(error)))
    finally:
        for name, proc in children.items():
            stop(proc); exits[name] = proc.returncode
        status.update(process_exit_codes=exits, wall_seconds=elapsed(), gpu_hours_upper_bound=2*elapsed()/3600,
                      queue_counts=dict(Counter(t['status'] for t in queue.snapshot()['tasks'])))
        vf.atomic_json(public/'EXECUTION_STATUS.json', status)
        vf.atomic_json(run/'RUN_COMPLETION.json', status)


if __name__ == '__main__':
    main()
