#!/usr/bin/env python3
"""One bounded run; no recurring monitor, no management of unrelated processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.medtrace.run_selective_write import GPUS, gpu_check, read, vf

LLAVA = "/remote-home/wangbomin/worktrees/llava-official-30697ca-20260904T090859Z"
MODEL = "/remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b"
VISION = "/remote-home/wangbomin/hugging_cache/openai/clip-vit-large-patch14-336"
JUDGE = "/remote-home/wangbomin/.cache/huggingface/hub/models--Qwen--Qwen3-32B-AWQ/snapshots/0499c3ac83fdef8810b907a23894ba91e95eddd8"
JUDGE_PYTHON = "/remote-home/wangbomin/evoclinician/venvs/vllm-0.9.2-py312/bin/python"


def environment(gpu):
    gpu_check(gpu)
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1",
        M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=gpu, M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES="0,1",
        M3BENCH_FORMAL_EXPECTED_GPU_UUID=GPUS[gpu], M3BENCH_EXPECTED_LLAVA_SOURCE=LLAVA,
        M3BENCH_MODEL_PATH=MODEL, M3BENCH_VISION_PATH=VISION, PYTHONPATH=f"{LLAVA}:{ROOT}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--public-dir',type=Path,required=True)
    args=parser.parse_args()
    run,public=args.run_root,args.public_dir
    config=read(run/'private/CAMPAIGN_CONFIG.json')
    commands,processes,exits,resources={},{},{},{}
    started=time.time()
    runner=[sys.executable,str(ROOT/'scripts/medtrace/run_selective_write.py')]
    finalizer=[sys.executable,str(ROOT/'scripts/medtrace/finalize_selective_write.py')]

    def elapsed():
        path=run/'private/CAMPAIGN_START.json'
        return time.time()-read(path)['epoch'] if path.exists() else time.time()-started

    def launch(name,command,env):
        commands[name]=command
        vf.atomic_json(run/'COMMANDS_PRIVATE.json',commands)
        with (run/f'{name}.log').open('x') as log:
            processes[name]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        vf.atomic_json(run/'PIDS.json',{k:p.pid for k,p in processes.items()})

    def wait(name,limit):
        process=processes[name]
        while process.poll() is None:
            if elapsed()>=limit:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                exits[name]=process.returncode
                raise TimeoutError(f'{name} reached campaign budget')
            time.sleep(2)
        exits[name]=process.returncode
        if process.returncode:
            raise RuntimeError(f'{name} exit={process.returncode}; inspect its own log')

    cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
    status=dict(status='RUNNING',scientific_gain='NOT_EVALUATED',judge='NOT_RUN',error=None)
    try:
        available=[]
        for gpu in GPUS:
            try:
                resources[gpu]=gpu_check(gpu);available.append(gpu)
            except RuntimeError as error:
                resources[gpu]=dict(unavailable=str(error))
        vf.atomic_json(public/'GPU_AND_TIMING.json',dict(preflight=resources,authorization=config['authorization'],started_epoch=None))
        if not available:
            raise RuntimeError('GPU0/1 both unavailable; no wait automation created')
        first_gpu=available[0]
        launch('first_integration',[*runner,'worker','--run-root',str(run),'--first-only'],environment(first_gpu))
        wait('first_integration',20*3600)
        first=read(run/'private/TASK_QUEUE.json')['tasks'][0]
        if first['status']!='RAW_READY':
            raise RuntimeError('first original task did not reach RAW_READY')
        vf.atomic_json(run/'FIRST_TASK_PASS.json',dict(task_id=first['task_id'],epoch=time.time(),semantic_gate=False))
        for gpu in available:
            try:
                env=environment(gpu)
            except RuntimeError as error:
                resources[gpu]['continuation_unavailable']=str(error)
                continue
            launch(f'worker_gpu{gpu}',[*runner,'worker','--run-root',str(run)],env)
        worker_errors=[]
        for name in tuple(processes):
            if not name.startswith('worker_'):
                continue
            try: wait(name,20*3600)
            except (RuntimeError,TimeoutError) as error: worker_errors.append(str(error))
        if worker_errors:
            status['worker_errors']=worker_errors
        # All 7B workers have exited before any Judge is loaded.
        vf.TaskQueue(run/'private/TASK_QUEUE.json',run).cancel_pending()
        launch('prepare_judge',[*finalizer,'prepare-judge','--run-root',str(run),'--public-dir',str(public)],cpu)
        wait('prepare_judge',24*3600)
        judge=run/'private/judge'
        packet=judge/'JUDGE_PACKET_PRIVATE.jsonl'
        if packet.stat().st_size:
            judge_env=None
            for gpu in GPUS:
                try: judge_env=environment(gpu);break
                except RuntimeError: continue
            if judge_env is None:
                raise RuntimeError('no idle authorized GPU for Judge; no co-hosting')
            lock=Path(config['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json'
            launch('judge',[JUDGE_PYTHON,str(ROOT/'scripts/medtrace/run_fixed_judge_vllm.py'),
                '--model-path',JUDGE,'--packet',str(packet),'--lock',str(lock),
                '--output',str(judge/'JUDGE_OUTPUT_PRIVATE.jsonl'),'--execution-lock',str(judge/'JUDGE_EXECUTION_LOCK_PRIVATE.json'),
                '--preflight-output',str(judge/'JUDGE_LENGTH_PREFLIGHT_PRIVATE.json'),'--max-model-len','auto'],judge_env)
            wait('judge',24*3600)
        status['judge']='COMPLETE'
        launch('finalize',[*finalizer,'finalize','--run-root',str(run),'--public-dir',str(public)],cpu)
        wait('finalize',24*3600)
        status.update(read(public/'RUN_COMPLETION.json'))
    except Exception as error:
        status.update(status='INCOMPLETE',error=f'{type(error).__name__}: {error}')
    finally:
        for name,process in processes.items():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            exits[name]=process.returncode
        vf.atomic_json(run/'PROCESS_EXIT_CODES.json',exits)
        queue=vf.TaskQueue(run/'private/TASK_QUEUE.json',run)
        for task in queue.snapshot()['tasks']:
            if task['status']=='RUNNING':
                queue.update(task['task_id'],'FAILED',last_error='worker exited before atomic result closure')
        tasks=queue.snapshot()['tasks']
        status.update(counts={s:sum(t['status']==s for t in tasks) for s in sorted({t['status'] for t in tasks})},expected=70,
                      process_exit_codes=exits,publication='PENDING_LOCAL_PUBLICATION')
        vf.atomic_json(public/'RUN_COMPLETION.json',status)
        vf.atomic_json(run/'RUN_COMPLETION.json',status)
        vf.atomic_json(public/'GPU_AND_TIMING.json',dict(preflight=resources,authorization=config['authorization'],
            wall_seconds=elapsed() if (run/'private/CAMPAIGN_START.json').exists() else None,
            gpu_hours_upper_bound=2*elapsed()/3600 if (run/'private/CAMPAIGN_START.json').exists() else 0,
            epoch_end=time.time(),process_exit_codes=exits,gpu2_3_used=False))
    if status['status']=='INCOMPLETE':
        raise SystemExit(1)


if __name__=='__main__':
    main()
