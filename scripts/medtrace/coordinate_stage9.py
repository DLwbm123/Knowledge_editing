#!/usr/bin/env python3
"""Detached bounded Stage9 inference/Judge chain on explicitly authorized GPUs."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.medtrace import stage9 as s
from scripts.medtrace import coordinate_selective_write as common
from scripts.medtrace.neutral_entrypoint import neutral_command


def launch_chain(args):
    run=args.run_root
    if (run/'PIPELINE_PIDS.json').exists():raise FileExistsError('Stage9 already launched')
    temp=Path(tempfile.mkdtemp(prefix='job.'))
    shutil.copyfile(ROOT/'scripts/medtrace/neutral_entrypoint.py',temp/'main.py')
    env=dict(os.environ,JOB_ENTRYPOINT=str(temp/'main.py'))
    command,env=neutral_command([sys.executable,str(Path(__file__).resolve()),'coordinate','--run-root',str(run)],env,'main')
    with (run/'pipeline.log').open('x') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    s.vf.atomic_json(run/'PIPELINE_PIDS.json',dict(coordinator=p.pid,argv=command))
    print('DETACHED',p.pid,flush=True)


def coordinate(args):
    run=args.run_root;config=s.read(run/'private/CAMPAIGN_CONFIG.json');started=config['campaign_epoch']
    common.GPUS.update({str(i):config['gpu_uuids'][str(i)] for i in (0,2,3)})
    children={};exits={};intervals={}
    def launch(name,command,gpu=None,judge=False):
        env=common.environment(str(gpu),judge=judge) if gpu is not None else dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
        env['M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES']='0,2,3'
        if gpu is not None and not judge and common.gpu_check(str(gpu))['free_mib']<24576:raise RuntimeError('insufficient inference memory margin')
        env['JOB_ENTRYPOINT']=os.environ['JOB_ENTRYPOINT']
        visible,env=neutral_command(command,env,'run' if gpu is not None and not judge else 'job')
        with (run/(name+'.log')).open('x') as log:
            children[name]=subprocess.Popen(visible,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        intervals[name]=dict(start=time.time(),gpu=gpu)
        s.vf.atomic_json(run/'PIPELINE_PIDS.json',dict(coordinator=os.getpid(),children={n:p.pid for n,p in children.items()}))
    def wait(names,limit):
        while any(children[n].poll() is None for n in names):
            used=sum(v.get('end',time.time())-v['start'] for v in intervals.values() if v['gpu'] is not None)
            gpu_limit=6.5*3600 if limit==6.5*3600 else 8*3600
            if time.time()-started>=limit or used>=gpu_limit or (run/'STOP').exists():
                for n in names:
                    if children[n].poll() is None:
                        os.killpg(children[n].pid,signal.SIGTERM)
                        try:children[n].wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(children[n].pid,signal.SIGKILL);children[n].wait()
                break
            time.sleep(5)
        for n in names:exits[n]=children[n].returncode;intervals[n]['end']=time.time()
        s.vf.atomic_json(run/'private/PROCESS_INTERVALS.json',intervals)
        return all(exits[n]==0 for n in names)
    try:
        script=str(ROOT/'scripts/medtrace/stage9.py')
        launch('first',[sys.executable,script,'worker','--run-root',str(run),'--part','first'],2)
        s.vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='FIRST_REAL_EDIT_RUNNING',additional_steps_per_condition=160,gpus=[2],publication='PENDING'))
        if not wait(['first'],6.5*3600):raise RuntimeError('first edit interface/constraint check needs attention; retain formal development result')
        first=next(t for t in s.read(run/'private/TASKS.json') if t['stage9_status']=='PENDING')
        for condition in ('F0','F1','F2'):
            if s.read(run/f"private/edits/e{first['order']:02d}/{condition}/result.json")['status']!='COMPLETE':raise RuntimeError('first edit incomplete')
        for part in ('0','1'):
            launch('part'+part,[sys.executable,script,'worker','--run-root',str(run),'--part',part],2+int(part))
        s.vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='REMAINING_EDITS_RUNNING',additional_steps_per_condition=160,gpus=[2,3],publication='PENDING'))
        wait(['part0','part1'],6.5*3600)
        # Replay mismatch invalidates only that writer's derived results. Do not
        # silently score a failed replay as verified; preserve other raw outputs.
        if any(exits[m]!=0 for m in ('part0','part1')):raise RuntimeError('affected partition failed; preserve completed edits and inspect')
        launch('prepare_judge',[sys.executable,script,'prepare-judge','--run-root',str(run)])
        if not wait(['prepare_judge'],8*3600):raise RuntimeError('Judge preparation failed')
        side=s.read(run/'private/judge/JUDGE_SIDECAR_PRIVATE.json')
        if side['new']:
            d=run/'private/judge'
            launch('judge',[common.JUDGE_PYTHON,str(ROOT/'scripts/medtrace/run_fixed_judge_vllm.py'),
                '--model-path',common.JUDGE,'--packet',str(d/'JUDGE_PACKET_PRIVATE.jsonl'),
                '--lock',str(Path(config['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json'),
                '--output',str(d/'JUDGE_OUTPUT_PRIVATE.jsonl'),'--execution-lock',str(d/'JUDGE_EXECUTION_LOCK_PRIVATE.json'),
                '--preflight-output',str(d/'JUDGE_LENGTH_PREFLIGHT_PRIVATE.json'),'--max-model-len','2048'],gpu=0,judge=True)
            if not wait(['judge'],8*3600):raise RuntimeError('Judge incomplete; raw outputs preserved')
        launch('report',[sys.executable,script,'report','--run-root',str(run)])
        if not wait(['report'],8*3600):raise RuntimeError('report phase failed')
        status=s.read(run/'public/RUN_STATUS.json');status.update(exit_codes=exits,
            wall_seconds=time.time()-started,gpu_hours_upper_bound=sum(v['end']-v['start'] for v in intervals.values() if v['gpu'] is not None)/3600)
        s.vf.atomic_json(run/'public/RUN_STATUS.json',status);s.vf.atomic_json(run/'RUN_COMPLETION.json',status)
    except Exception as exc:
        # Only own children are terminated if orchestration cannot proceed.
        for p in children.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        s.vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='ATTENTION_REQUIRED',error=str(exc),exit_codes=exits,publication='PENDING',new_writer_training=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('launch','coordinate'));p.add_argument('--run-root',type=Path,required=True)
    a=p.parse_args()
    (launch_chain if a.action=='launch' else coordinate)(a)
