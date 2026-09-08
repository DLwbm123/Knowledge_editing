#!/usr/bin/env python3
"""Detach the one-shot Base-before phase using the existing Stage2 worker."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.run_stage2 import queue_task, vf, read, sw
from scripts.medtrace.coordinate_selective_write import environment
from scripts.medtrace.neutral_entrypoint import neutral_command


def main(args):
    run = args.run_root
    if (run/'BASE_PHASE_PIDS.json').exists():
        raise FileExistsError('Base-before already launched; inspect and resume only missing work')
    episodes = read(run/'private/NEW_EPISODE_MANIFEST_PRIVATE.json')['episodes']
    tasks, bases = [], []
    for e in episodes:
        i = e['event_index']
        bases.append(queue_task(i, 'BASE', i))
        be = queue_task(i, 'BE', i*10)
        init = queue_task(i, 'INIT', i*10+1, be['task_id'])
        w0 = queue_task(i, 'CP', i*10+2, init['task_id'], 'P4', 'W0_TASK_ONLY')
        new = [be, init, w0]
        if e['method_support']['W1'] == 'SUPPORTED':
            new.append(queue_task(i, 'CP', i*10+3, init['task_id'], 'P4', 'W1_KL_0.1'))
        tasks.extend(dict(t, status='WAITING_BASE_BEFORE') for t in new)
    vf.atomic_json(run/'private/BASE_TASK_QUEUE.json', dict(tasks=bases))
    vf.atomic_json(run/'private/TASK_QUEUE.json', dict(tasks=tasks))
    env = environment('2')
    assert sw.gpu_check('2')['free_mib'] >= 24576
    env['JOB_ENTRYPOINT'] = '/tmp/job.rff6tfx2/main.py'
    command = [sys.executable, str(ROOT/'scripts/medtrace/run_stage2.py'), 'worker', '--run-root', str(run), '--base-only']
    visible, env = neutral_command(command, env, 'run')
    with (run/'base_worker.log').open('x') as log:
        proc = subprocess.Popen(visible, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    vf.atomic_json(run/'BASE_PHASE_PIDS.json', dict(base_worker=proc.pid, visible=visible))
    status = read(run/'public/RUN_STATUS.json')
    status.update(status='BASE_BEFORE_RUNNING', compute='BASE_BEFORE_RUNNING', source_review='SOURCE_PACKETS_FROZEN',
        legal_new_edit_count=32, supported_writer_endpoints=dict(W0=32, W1=1, BE=32),
        existing_factorial='DERIVED_COMPLETE_REPLAY_PENDING', base_worker_pid=proc.pid)
    vf.atomic_json(run/'public/RUN_STATUS.json', status)
    print('BASE_BEFORE_STARTED', proc.pid, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True, type=Path)
    main(p.parse_args())
