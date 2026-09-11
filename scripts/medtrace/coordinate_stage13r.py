#!/usr/bin/env python3
"""One detached GPU0 chain, no recurring monitoring or unrelated process control."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace import coordinate_selective_write as common
from scripts.medtrace.neutral_entrypoint import neutral_command
vf,read=common.vf,common.read


def launch(args):
    run=args.run_root
    if (run/'PIPELINE_PIDS.json').exists():raise FileExistsError('Already launched; preserve existing run')
    cfg=read(run/'private/CAMPAIGN_CONFIG.json')
    assert cfg['allowed_physical_gpus']==[0] and read(run/'private/TASKS.json')
    cfg.update(source_preparation_commit=cfg['code_commit'],code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        campaign_epoch=time.time(),coordinator_start_required=False)
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json',cfg)
    vf.atomic_json(run/'private/CAMPAIGN_START.json',dict(epoch=cfg['campaign_epoch']))
    common.GPUS.clear();common.GPUS['0']=cfg['gpu_uuids']['0']
    check=common.gpu_check('0');vf.atomic_json(run/'private/GPU_START_CHECK.json',check)
    tmp=Path(tempfile.mkdtemp(prefix='job.'));shutil.copyfile(ROOT/'scripts/medtrace/neutral_entrypoint.py',tmp/'main.py')
    cmd,env=neutral_command([sys.executable,str(Path(__file__).resolve()),'coordinate','--run-root',str(run)],
        dict(os.environ,JOB_ENTRYPOINT=str(tmp/'main.py'),CUDA_VISIBLE_DEVICES=''),'main')
    with (run/'pipeline.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    vf.atomic_json(run/'PIPELINE_PIDS.json',dict(coordinator=proc.pid,argv=cmd))
    print('DETACHED',proc.pid,flush=True)


def coordinate(args):
    run=args.run_root;cfg=read(run/'private/CAMPAIGN_CONFIG.json');start=cfg['campaign_epoch']
    common.GPUS.clear();common.GPUS['0']=cfg['gpu_uuids']['0']
    children={};exits={};intervals={}
    def execute(name,command,gpu=False,judge=False):
        env=common.environment('0',judge=judge) if gpu else dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
        env.update(JOB_ENTRYPOINT=os.environ['JOB_ENTRYPOINT'],M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES='0')
        visible,env=neutral_command(command,env,'run' if gpu and not judge else 'job')
        with (run/(name+'.log')).open('x') as log:
            proc=subprocess.Popen(visible,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[name]=proc;intervals[name]=dict(start=time.time(),gpu=0 if gpu else None)
        vf.atomic_json(run/'PIPELINE_PIDS.json',dict(coordinator=os.getpid(),children={k:p.pid for k,p in children.items()}))
        vf.atomic_json(run/'public/RUN_STATUS.json',dict(status=name.upper()+'_RUNNING',gpu=0,actual_N=len(read(run/'private/TASKS.json')),publication='PENDING'))
        limit=cfg['train_seconds'] if name in ('train','generate') else cfg['wall_hours']*3600
        while proc.poll() is None:
            if time.time()-start>=limit or (run/'STOP').exists():
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(5)
        exits[name]=proc.returncode;intervals[name]['end']=time.time()
        vf.atomic_json(run/'private/PROCESS_INTERVALS.json',intervals)
        if proc.returncode:raise RuntimeError(name+' failed; completed artifacts preserved')
    try:
        script=str(ROOT/'scripts/medtrace/stage13r.py')
        execute('train',[sys.executable,script,'train','--run-root',str(run)],gpu=True)
        execute('generate',[sys.executable,script,'generate','--run-root',str(run)],gpu=True)
        execute('prepare_judge',[sys.executable,script,'prepare-judge','--run-root',str(run)])
        side=read(run/'private/judge/JUDGE_SIDECAR_PRIVATE.json')
        if side['new']:
            d=run/'private/judge'
            execute('judge',[common.JUDGE_PYTHON,str(ROOT/'scripts/medtrace/run_fixed_judge_vllm.py'),
                '--model-path',common.JUDGE,'--packet',str(d/'JUDGE_PACKET_PRIVATE.jsonl'),
                '--lock',str(Path(cfg['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json'),
                '--output',str(d/'JUDGE_OUTPUT_PRIVATE.jsonl'),'--execution-lock',str(d/'JUDGE_EXECUTION_LOCK_PRIVATE.json'),
                '--preflight-output',str(d/'JUDGE_LENGTH_PREFLIGHT_PRIVATE.json'),'--max-model-len','2048'],gpu=True,judge=True)
        execute('report',[sys.executable,script,'report','--run-root',str(run)])
        status=read(run/'public/RUN_STATUS.json');status.update(exit_codes=exits,wall_seconds=time.time()-start,
            gpu_hours_upper_bound=sum(v['end']-v['start'] for v in intervals.values() if v['gpu'] is not None)/3600)
        vf.atomic_json(run/'public/RUN_STATUS.json',status);vf.atomic_json(run/'RUN_COMPLETION.json',status)
    except Exception as error:
        for proc in children.values():
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='ATTENTION_REQUIRED',error=str(error),exit_codes=exits,publication='PENDING'))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','coordinate'));p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args();globals()[args.action](args)
