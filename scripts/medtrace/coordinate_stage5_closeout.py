#!/usr/bin/env python3
"""One detached continuation: final banks, old replay, final Judge and reports."""
import argparse
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
    if (run/'CLOSEOUT_COORDINATOR_PIDS.json').exists():
        raise FileExistsError('closeout already launched; inspect before any recovery')
    config = read(run/'private/CAMPAIGN_CONFIG.json')
    start = read(run/'private/CAMPAIGN_START.json')['epoch']
    processes, exits, intervals = {}, {}, {}
    def elapsed():return time.time()-start
    def status(phase, **extra):
        old = read(run/'public/RUN_STATUS.json')
        vf.atomic_json(run/'public/RUN_STATUS.json', dict(old, status=phase, **extra))
    def launch(name, command, gpu=False, judge=False):
        env = environment('2', judge=judge) if gpu else dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
        if gpu and not judge and sw.gpu_check('2')['free_mib'] < 24576:
            raise RuntimeError('GPU2 memory insufficient; no other task terminated')
        env['JOB_ENTRYPOINT'] = '/tmp/job.rff6tfx2/main.py'
        visible, env = neutral_command(command, env, 'run' if gpu and not judge else 'job')
        with (run/(name+'.log')).open('x') as log:
            processes[name] = subprocess.Popen(visible, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        intervals[name] = dict(start_epoch=time.time(), gpu=2 if gpu else None)
        vf.atomic_json(run/'CLOSEOUT_COORDINATOR_PIDS.json', dict(coordinator=os.getpid(), **{n:p.pid for n,p in processes.items()}))
    def wait(name, limit):
        proc = processes[name]
        while proc.poll() is None:
            if elapsed() >= limit or (run/'STOP').exists():
                os.killpg(proc.pid, signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                break
            time.sleep(5)
        exits[name] = proc.returncode
        intervals[name]['end_epoch'] = time.time()
        vf.atomic_json(run/'private/CLOSEOUT_PROCESS_INTERVALS.json', intervals)
        return proc.returncode == 0
    vf.atomic_json(run/'CLOSEOUT_COORDINATOR_PIDS.json', dict(coordinator=os.getpid()))
    try:
        training_pid = read(run/'TRAINING_COORDINATOR_PIDS.json')['coordinator']
        while True:
            current = read(run/'public/RUN_STATUS.json')['status']
            if current == 'EXECUTION_ATTENTION_REQUIRED':
                raise RuntimeError('training phase needs attention; no whole-run restart')
            proc = Path(f'/proc/{training_pid}/stat')
            live = proc.exists() and proc.read_text().split(') ',1)[1][0] != 'Z'
            if not live:
                if current != 'WRITER_PHASE_CLOSED_EVALUATION_PENDING':
                    raise RuntimeError('training coordinator exited without phase closure')
                break
            if elapsed() >= 10*3600 or (run/'STOP').exists():
                raise RuntimeError('training wait reached budget or STOP')
            time.sleep(15)
        status('FINAL_BANK_EVALUATION_RUNNING', compute='FINAL_BANK_EVALUATION_RUNNING')
        for method in ('W0','W1','BE'):
            if elapsed() >= 10*3600:break
            name = 'bank_'+method
            launch(name, [sys.executable, str(ROOT/'scripts/medtrace/stage5_bank.py'),
                         '--run-root', str(run), '--method', method], gpu=True)
            wait(name, 10*3600)  # An affected bank failure does not erase other method outputs.
        if elapsed() < 10*3600:
            launch('existing_replay', [sys.executable, str(ROOT/'scripts/medtrace/stage5_existing_replay.py'),
                '--run-root', str(run)], gpu=True)
            wait('existing_replay', 10*3600)
        if (run/'STOP').exists():raise RuntimeError('explicit STOP')
        status('FINAL_JUDGE_PREPARING', compute='RAW_EVALUATION_CLOSED')
        finalizer = ROOT/'scripts/medtrace/finalize_stage5.py'
        launch('prepare_final_judge', [sys.executable, str(finalizer), 'prepare-judge', '--run-root', str(run)])
        if not wait('prepare_final_judge', 12*3600):raise RuntimeError('final Judge preparation failed')
        directory = run/'private/judge'
        side = read(directory/'JUDGE_SIDECAR_PRIVATE.json')
        if side['new']:
            launch('final_judge', [JUDGE_PYTHON, str(ROOT/'scripts/medtrace/run_fixed_judge_vllm.py'),
                '--model-path', JUDGE, '--packet', str(directory/'JUDGE_PACKET_PRIVATE.jsonl'),
                '--lock', str(Path(config['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json'),
                '--output', str(directory/'JUDGE_OUTPUT_PRIVATE.jsonl'),
                '--execution-lock', str(directory/'JUDGE_EXECUTION_LOCK_PRIVATE.json'),
                '--preflight-output', str(directory/'JUDGE_LENGTH_PREFLIGHT_PRIVATE.json'),
                '--max-model-len', '2048'], gpu=True, judge=True)
            if not wait('final_judge', 12*3600):raise RuntimeError('final Judge incomplete; raw outputs preserved')
        launch('final_reports', [sys.executable, str(finalizer), 'finalize', '--run-root', str(run)])
        if not wait('final_reports', 12*3600):raise RuntimeError('final reports failed')
        current = read(run/'public/RUN_STATUS.json')
        current.update(closeout_exit_codes=exits, wall_seconds=elapsed(), gpu_hours_upper_bound=elapsed()/3600,
                       gpu_hours_bound_basis='only GPU2 authorized; wall time is a conservative bound, not measured utilization',
                       publication='PUBLICATION_PENDING')
        vf.atomic_json(run/'public/RUN_STATUS.json', current)
        vf.atomic_json(run/'RUN_COMPLETION.json', current)
    except Exception as exc:
        status('CLOSEOUT_ATTENTION_REQUIRED', closeout_error=str(exc), closeout_exit_codes=exits, wall_seconds=elapsed())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    main(p.parse_args())
