#!/usr/bin/env python3
"""Continue the existing Base phase through locked Base Judge and new writers."""
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
from scripts.medtrace.run_stage2 import vf, read, sw
from scripts.medtrace.coordinate_selective_write import environment, JUDGE, JUDGE_PYTHON
from scripts.medtrace.neutral_entrypoint import neutral_command


def main(args):
    run = args.run_root
    if (run/'TRAINING_COORDINATOR_PIDS.json').exists():
        raise FileExistsError('one-shot training coordinator already exists')
    config = read(run/'private/CAMPAIGN_CONFIG.json')
    clock = read(run/'private/CAMPAIGN_START.json')['epoch']
    children, exits = {}, {}
    def elapsed():
        return time.time()-clock
    def status(phase, **extra):
        old = read(run/'public/RUN_STATUS.json')
        vf.atomic_json(run/'public/RUN_STATUS.json', dict(old, status=phase, **extra))
    def launch(name, command, env):
        env['JOB_ENTRYPOINT'] = '/tmp/job.rff6tfx2/main.py'
        visible, env = neutral_command(command, env, 'run' if name.startswith('worker') else 'job')
        with (run/(name+'.log')).open('x') as log:
            proc = subprocess.Popen(visible, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children[name] = proc
        vf.atomic_json(run/'TRAINING_COORDINATOR_PIDS.json', dict(coordinator=os.getpid(), **{n:p.pid for n,p in children.items()}))
        return proc
    def wait(name, limit):
        proc = children[name]
        while proc.poll() is None:
            if elapsed() >= limit or (run/'STOP').exists():
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                break
            time.sleep(5)
        exits[name] = proc.returncode
        return proc.returncode == 0
    cpu = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    vf.atomic_json(run/'TRAINING_COORDINATOR_PIDS.json', dict(coordinator=os.getpid()))
    try:
        base_pid = read(run/'BASE_PHASE_PIDS.json')['base_worker']
        while True:
            queue = read(run/'private/BASE_TASK_QUEUE.json')['tasks']
            if any(t['status'] == 'FAILED' for t in queue):
                raise RuntimeError('Base worker failed; preserve partial outputs and inspect')
            proc_stat = Path(f'/proc/{base_pid}/stat')
            live = proc_stat.exists() and proc_stat.read_text().split(') ', 1)[1][0] != 'Z'
            if not live:
                if not all(t['status'] == 'COMPLETE' for t in queue):
                    raise RuntimeError('Base worker exited before completion')
                break
            if elapsed() >= 10*3600 or (run/'STOP').exists():
                raise RuntimeError('Base phase budget/STOP; no new model calls')
            time.sleep(10)
        status('BASE_JUDGE_PREPARING', compute='BASE_BEFORE_RAW_COMPLETE')
        judge_script = ROOT/'scripts/medtrace/stage5_base_judge.py'
        launch('prepare_base_judge', [sys.executable, str(judge_script), 'prepare', '--run-root', str(run)], cpu)
        if not wait('prepare_base_judge', 10*3600):
            raise RuntimeError('Base Judge preparation failed')
        directory = run/'private/base_judge'
        side = read(directory/'JUDGE_SIDECAR_PRIVATE.json')
        if side['new']:
            launch('base_judge', [JUDGE_PYTHON, str(ROOT/'scripts/medtrace/run_fixed_judge_vllm.py'),
                '--model-path', JUDGE, '--packet', str(directory/'JUDGE_PACKET_PRIVATE.jsonl'),
                '--lock', str(Path(config['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json'),
                '--output', str(directory/'JUDGE_OUTPUT_PRIVATE.jsonl'),
                '--execution-lock', str(directory/'JUDGE_EXECUTION_LOCK_PRIVATE.json'),
                '--preflight-output', str(directory/'JUDGE_LENGTH_PREFLIGHT_PRIVATE.json'),
                '--max-model-len', '2048'], environment('2', judge=True))
            if not wait('base_judge', 10*3600):
                raise RuntimeError('Base Judge failed; no students authorized by incomplete scores')
        launch('lock_base_before', [sys.executable, str(judge_script), 'lock', '--run-root', str(run)], cpu)
        if not wait('lock_base_before', 10*3600):
            raise RuntimeError('Base-before membership lock failed')
        status('STUDENT_TRAINING_RUNNING', compute='STUDENT_TRAINING_RUNNING', judge='BASE_BEFORE_COMPLETE')
        queue = vf.TaskQueue(run/'private/TASK_QUEUE.json', run)
        attempt = 0
        while any(t['status'] == 'PENDING' for t in queue.snapshot()['tasks']) and elapsed() < 10*3600 and not (run/'STOP').exists():
            if sw.gpu_check('2')['free_mib'] < 24576:
                raise RuntimeError('GPU2 resource margin unavailable; do not disturb other jobs')
            attempt += 1; name = f'worker_students_{attempt:02d}'
            launch(name, [sys.executable, str(ROOT/'scripts/medtrace/run_stage2.py'), 'worker', '--run-root', str(run)], environment('2'))
            if wait(name, 10*3600):
                break
            snapshot = queue.snapshot()['tasks']
            for t in snapshot:
                if t['status'] == 'RUNNING':
                    queue.update(t['task_id'], 'FAILED', last_error='worker exited before atomic closure')
            faults = Counter((t['kind'],t.get('last_error')) for t in queue.snapshot()['tasks'] if t['status'] == 'FAILED')
            paused = {kind for (kind, error), count in faults.items() if count >= 2}
            for t in queue.snapshot()['tasks']:
                if t['status'] == 'PENDING' and t['kind'] in paused:
                    queue.update(t['task_id'], 'PAUSED_INTEGRATION', reason='repeated same engineering fault; retain unaffected branches')
        status('WRITER_PHASE_CLOSED_EVALUATION_PENDING', compute='WRITER_PHASE_CLOSED',
            counts=dict(Counter(t['status'] for t in queue.snapshot()['tasks'])),
            remaining='New single/bank evaluation, final Judge, natural branch replay and publication remain pending',
            process_exit_codes=exits, elapsed_seconds=elapsed())
    except Exception as exc:
        status('EXECUTION_ATTENTION_REQUIRED', error_type=type(exc).__name__, error=str(exc), process_exit_codes=exits,
               elapsed_seconds=elapsed())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True, type=Path)
    main(p.parse_args())
