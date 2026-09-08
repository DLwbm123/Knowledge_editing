#!/usr/bin/env python3
"""Bounded Stage4: readonly cohort reuse, one new KL writer, rejection-only bank replay."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.medtrace import run_stage3 as s3, run_stage3_bank as bank
from scripts.medtrace import prepare_stage2_sources as source
from scripts.medtrace.stage4_scope import calibrate, accepted
from scripts.medtrace.coordinate_selective_write import environment, JUDGE, JUDGE_PYTHON
from scripts.medtrace.neutral_entrypoint import neutral_command
from scripts.medtrace.finalize_selective_write import csv_write
from m3bench_repro.editors.routing import MemoryRouter

vf,sw,read=s3.vf,s3.sw,s3.read
torch=s3.s2.torch
OLD='OLD_STAGE3_COMMON7'
BANK='SLAKE_STAGE2_BANK16'
GPUS=dict(s3.ORIGINAL_GPUS, **{'0':'GPU-688ed629-7a46-b62c-e3e4-0940e1649b5f',
                              '1':'GPU-5fac6011-5432-229f-bb11-9ce9f104a334'})
CONDITIONS={'W0':'W0_TASK_ONLY','W01':'W1_KL_0.01','W1':'W1_KL_0.1','BE':'BALANCEDIT'}


def task_for(cohort,index,method='W01'):
    return dict(task_id=f'S4_{cohort}_e{index:04d}_{method}',kind='CP',event_index=index,
                parameterization='P4',condition=CONDITIONS[method],seed=20260910)


def source_task(directory,cohort,index,method):
    name=(f'S3_A_e{index:04d}_'+{'W0':'S0','W1':'S1','BE':'BE'}[method] if cohort==OLD else
          f'S2_e{index:03d}_'+('BE_BE_BALANCEDIT' if method=='BE' else 'CP_P4_'+CONDITIONS[method]))
    return directory/'private/tasks'/name


def configure(run):
    config=read(run/'private/CAMPAIGN_CONFIG.json')
    if config['kind']!='MEDTRACE_STAGE4' or config['gpu_uuids']!=GPUS:
        raise ValueError('Stage4 resource binding mismatch')
    sw.GPUS.update(GPUS)
    sw.SEED=s3.s2.SEED=20260910
    return config


def prepare(args):
    run=args.run_root
    if run.exists(): raise FileExistsError(run)
    run.mkdir(parents=True)
    # Real target NFS write/read probe; fail here instead of falling back to root/home.
    probe=run/'.write_probe'
    with probe.open('x') as f: f.write('stage4')
    if probe.read_text()!='stage4': raise OSError('storage probe failed')
    probe.unlink()
    start=(read(args.prior_preparation/'private/CAMPAIGN_START.json')['epoch']
           if args.prior_preparation else time.time())
    prior=read(args.stage3_run/'private/CAMPAIGN_CONFIG.json')
    terminal=read(args.stage3_run/'RUN_COMPLETION.json')
    if terminal['compute']['complete_method_endpoints']!=109 or terminal['judge']['missing_tuples']:
        raise ValueError('Stage3 anchor is not closed')
    config=dict(prior,kind='MEDTRACE_STAGE4',stage3_run=str(args.stage3_run),gpu_uuids=GPUS,
        allowed_physical_gpus=[0,1,2,3],forbidden_physical_gpus=[],authorization='USER_STAGE4_FOUR_GPUS',
        wall_hours=8,gpu_hours=32,train_seconds=6.5*3600,code_commit=subprocess.check_output(
            ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),stage4_started_epoch=start)
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json',config)
    vf.atomic_json(run/'private/CAMPAIGN_START.json',dict(epoch=start,code_commit=config['code_commit']))
    if args.prior_preparation:
        vf.atomic_json(run/'private/PREPARATION_RECOVERY.json',dict(previous=str(args.prior_preparation),
            reason='source review metadata may be null; no GPU tasks launched in previous preparation',
            original_clock_preserved=True,failed_preparation_preserved=True))
    q=read(args.stage3_run/'private/TASK_QUEUE.json')['tasks']
    indices=[t['event_index'] for t in q if t['kind']=='SINGLE_GROUP' and 'S1' in t['methods']]
    if len(indices)!=7: raise ValueError('old common cohort changed')
    stage2=Path(config['stage2_run'])
    bank_episodes=bank.load_manifest(stage2)
    cohorts=[(OLD,args.stage3_run,indices),(BANK,stage2,[e['event_index'] for e in bank_episodes])]
    manifest,queue,locks,exposure=[],[],[],[]
    for cohort,origin,ids in cohorts:
        child=run/'private/cohorts'/cohort
        old_config=read(origin/'private/CAMPAIGN_CONFIG.json')
        if old_config['runtime']!=config['runtime']: raise ValueError('cohort runtime mismatch')
        child_config=dict(old_config,teacher_reuse_root=str(origin),stage4_diagnostics=True,
                          train_seconds=config['train_seconds'])
        vf.atomic_json(child/'private/CAMPAIGN_CONFIG.json',child_config)
        vf.atomic_json(child/'private/CAMPAIGN_START.json',dict(epoch=start))
        for index in ids:
            data=read(origin/f'private/edits/e{index:02d}.json')
            if data['extra_fit'] or vf.sha256_file(Path(data['a2']))!=data['a2_sha256']:
                raise ValueError('cohort A2 or extra-fit binding mismatch')
            references={m:str(source_task(origin,cohort,index,m)) for m in ('W0','W1','BE')}
            for method,path in references.items():
                result=read(Path(path)/'result_private.json')
                if (not result['base_guard']['unchanged'] or result['step']!=(50 if method=='BE' else 320)
                        or (method!='BE' and result['a2_sha256']!=data['a2_sha256'])):
                    raise ValueError('historical endpoint binding mismatch')
                if any(result['outputs'][r['logical_id']]['row']!=r for r in data['rows']):
                    raise ValueError('historical endpoint rows mismatch')
            vf.atomic_json(child/f'private/edits/e{index:02d}.json',data)
            initial_dir=child/f'private/initial/e{index:02d}'
            initial_dir.mkdir(parents=True)
            for name in ('initial.json','reference_private.json'):
                shutil.copyfile(origin/f'private/initial/e{index:02d}'/name,initial_dir/name)
            # Reuse small exact-bound Base answers, not the full Base or teacher/model files.
            cached_base=child/f'private/base_generation/e{index:02d}'
            old_result=read(Path(references['W1'])/'result_private.json')
            for row in data['rows']:
                vf.atomic_json(cached_base/(vf.sha256_json(row)+'.json'),old_result['outputs'][row['logical_id']]['base'])
            item=dict(cohort=cohort,event_index=index,origin=str(origin),child=str(child),references=references,
                      a2_sha256=data['a2_sha256'],data_sha256=vf.sha256_json(data),record_id=data['record_id'])
            manifest.append(item)
            task=task_for(cohort,index)
            queue.append(dict(task,kind='WRITER',cohort=cohort,priority=len(queue),status='PENDING',attempts=0,depends_on=None))
            initial=read(initial_dir/'initial.json')
            locks.append(dict(cohort=cohort,edit=index,a2_sha256=data['a2_sha256'],
                config_sha256=vf.sha256_file(origin/'private/CAMPAIGN_CONFIG.json'),
                data_sha256=item['data_sha256'],initial_kl=initial['initial'],
                scale={g:max(v,.001) for g,v in initial['initial'].items()},
                teacher=dict(direction='Base||ON',temperature=1,cap=128,full_vocab=True,
                             fit_only=True,code=old_config['code_commit']),
                native_and_fit_target='inherited exact cohort binding'))
            exposure.append(dict(cohort=cohort,index=index,exposure='VIEWED_DEVELOPMENT',
                independent_case_count='UNKNOWN',status='EXECUTABLE',missing=''))
    # One bounded existing-source scan. Explicitly retain stricter legacy exposure exclusions.
    all_edits=bank_episodes+[read(args.stage3_run/f"private/edits/e{t['event_index']:02d}.json")
        for t in q if t['kind']=='SINGLE_GROUP' and t['status']=='RAW_READY']
    groups=set().union(*(source.image_groups(e) for e in all_edits))
    additions=run/'private/STAGE4_EXPOSURE_ADDITIONS.json'
    vf.atomic_json(additions,dict(images=[dict(dataset=d,image_id=i,reason='Stage2/3 all student-exposed roles') for d,i in sorted(groups)]))
    try:
        pool,screen,screen_ledger=source.scan(dict(config,additional_exclusions=str(additions)))
    except Exception as error:
        pool,screen,screen_ledger=[],dict(status='SOURCE_SCAN_FAILED',error_type=type(error).__name__),[]
        vf.atomic_json(run/'private/CONFIRMATION_SCAN_FAILURE.json',dict(traceback=traceback.format_exc()))
    choices={}
    for row in sorted(pool,key=lambda r:(r['dataset'],r['image_id'],str(r['source_qid']))):
        if row['base_correct'] is False:
            choices.setdefault(row['source_group'],row)
    candidates=list(choices.values())[:32]
    # Historical reviewed text is fact/image-bound. This scan does not authorize new template instantiations.
    approved={(e['event']['edit_record']['image_path'],e['event']['edit_record']['question']) for e in all_edits
              if (e.get('positive_review') or {}).get('approved_equivalent')}
    for index,row in enumerate(candidates,1):
        if (row['image_path'],row['question']) in approved:
            raise ValueError('candidate survived exclusion despite historical fact review; inspect exposure closure')
        exposure.append(dict(cohort='TRACK_C_CANDIDATE',index=index,exposure='QUERY_HOLDOUT_CASE_INDEPENDENCE_UNKNOWN',
            independent_case_count='UNKNOWN',status='UNSUPPORTED',
            missing='No authorized fact-specific fit/calibration paraphrase review; no new derived text approval'))
    vf.atomic_json(run/'private/CONFIRMATION_SCREEN.json',dict(candidates=candidates,screen=screen,
        ledger=screen_ledger,additional_groups=len(groups),cpu_seconds=time.time()-start,
        complete_legal_edits=0,status='CONFIRMATION_UNAVAILABLE',no_new_review_authorized=True))
    if time.time()-start>5400:
        screen['status']='SOURCE_TIME_LIMIT_EXCEEDED_NO_COHORT_ADMITTED'
    vf.atomic_json(run/'private/COHORTS.json',dict(edits=manifest))
    for prefix in bank.PREFIXES:
        queue.append(dict(task_id=f'S4_BANK_{prefix:02d}',kind='BANK',prefix=prefix,
                          priority=1000+prefix,status='PENDING',attempts=0,depends_on=None))
    vf.atomic_json(run/'private/TASK_QUEUE.json',dict(tasks=queue))
    public=run/'public'
    public.mkdir(parents=True,exist_ok=True)
    csv_write(public/'SOURCE_AND_EXPOSURE_MANIFEST.csv',exposure)
    vf.atomic_json(public/'METHOD_AND_ROUTER_LOCKS.json',dict(cohort_locks=locks,lambdas=[0,.01,.1],
        writer_steps=320,diagnostic_steps=[0,80,160,320],stage3_anchor='bab6cda735c4bd115e266871108cf492e524a8a3',
        stage2_anchor='74d2a337f7d2b830d58819f76c87058cef0c5f3b',source_commit=config['code_commit'],
        router=dict(rule='R0 off stays off; otherwise retain same expert iff (r-d)/max(r,1e-12)>=kappa',
            distance='unchanged FP32 euclidean d and r',ranking='MemoryRouter argmin, stable first entry tie',
            calibration='native and original calibration only; no future expert; positive family micro/macro loss <=.05',
            objective='equal H/U edit-macro activation on STRICT_BASE',thresholds='PENDING_CALIBRATION'),
        gpu_uuids=GPUS,wall_hours=8,gpu_hours=32,judge_reserve_minutes=90))
    vf.atomic_text(public/'UNSEEN_CONFIRMATION_REPORT.md',f'# Independent confirmation\n\nCONFIRMATION_UNAVAILABLE. '
        f'One inherited source screen after all Stage2/3-role exclusions yielded {len(pool)} training-side rows and '
        f'{len(candidates)} capped candidate edits with frozen Base-wrong eligibility. No candidate has an already '
        'authorized fact/image-specific positive and calibration text review. Template reuse on a new fact is not '
        'silently approved. No additional students run; no existing edit is renamed as unseen. Patient identity is '
        'unknown. Minimum additional permission, if desired in a later round, is source-only review of fit/calibration '
        'and held-out text for the listed private candidate facts; this run does not wait for or assume that permission.\n')
    vf.atomic_json(public/'EXECUTION_STATUS.json',dict(status='PREPARED',new_writer_endpoints=23,reused_writer_endpoints=69,
        bank_prefixes=[1,4,8,16],confirmation='CONFIRMATION_UNAVAILABLE',unsupported_confirmation=len(candidates),
        judge='NOT_RUN',publication='PENDING'))
    print(json.dumps(dict(run=str(run),writers=len(manifest),candidate_edits=len(candidates),source_seconds=time.time()-start)))


def writer(runtime,run,task):
    config=configure(run)
    item=next(e for e in read(run/'private/COHORTS.json')['edits'] if (e['cohort'],e['event_index'])==(task['cohort'],task['event_index']))
    child=Path(item['child']); sub=task_for(item['cohort'],item['event_index'])
    path=child/'private/tasks'/sub['task_id']/'result_private.json'
    if path.exists(): raise FileExistsError('explicit endpoint recovery required; no silent retraining')
    result=sw.train_task(runtime,argparse.Namespace(run_root=child),sub)
    if not result['base_guard']['unchanged']: raise RuntimeError('Base changed')
    # Frozen intermediate checkpoints are diagnostic only. No result feeds back to optimization.
    diagnostics=[]
    data=read(child/f"private/edits/e{item['event_index']:02d}.json")
    for method in ('W0','W01','W1'):
        directory=path.parent if method=='W01' else Path(item['references'][method])
        ck_dir=next(directory.glob('attempt_chunk*'))
        baseline=torch.load(ck_dir/'step0000.pt',map_location='cpu',weights_only=True)['expert']
        for step in (0,80,160,320):
            checkpoint=torch.load(ck_dir/f'step{step:04d}.pt',map_location='cpu',weights_only=True)
            delta=sum(float((v-baseline[k]).square().sum()) for k,v in checkpoint['expert'].items())**.5
            curve=read(ck_dir/'training_private.json')
            diagnostic=next(d for d in curve['diagnostics'] if d['step']==step)
            expert=vf.AsymmetricCPExpert(14336,4096,4).to(runtime.device)
            expert.load_state_dict(checkpoint['expert']);expert.requires_grad_(False)
            hook=vf.MedTraceLayerHook(runtime.get_module(vf.LAYER),expert);hook.attach()
            record=vf.EditorRecord.from_dict(data['event']['edit_record'])
            losses=[]
            try:
                from dataclasses import replace
                with torch.no_grad():
                    for question in [record.question,*[p['question'] for p in data['fit_paraphrases']]]:
                        batch=runtime.build_edit_batch(replace(record,question=question))
                        hook.set_teacher_routing(batch.labels);losses.append(float(runtime.compute_loss(batch)))
            finally: hook.detach()
            behaviors=[]
            if item['cohort']==OLD:
                for row in data['rows']:
                    if row.get('task')=='T2G' and row['role']=='formal_development':
                        behaviors.append(dict(row=row,output=sw.generated(runtime,row,expert)))
            diagnostics.append(dict(cohort=item['cohort'],edit=item['event_index'],method=method,step=step,
                native_ce=losses[0],fit_positive_ce=sum(losses[1:])/len(losses[1:]),update_norm_from_a2=delta,
                full_fit_kl=diagnostic['full_fit_kl'],scale=diagnostic['scale'],
                normalized_fit_kl={g:k/diagnostic['scale'][g] for g,k in diagnostic['full_fit_kl'].items()},
                behaviors=behaviors,endpoint_selector=False))
            del expert,checkpoint;gc.collect();torch.cuda.empty_cache()
    vf.atomic_json(path.parent/'STAGE4_DIAGNOSTICS_PRIVATE.json',dict(rows=diagnostics))


def bank_task(runtime,run,task):
    config=configure(run);prefix=task['prefix'];out=run/f'private/bank/prefix{prefix}'
    stage2=Path(config['stage2_run']);stage3=Path(config['stage3_run'])
    episodes=bank.load_manifest(stage2)[:prefix]
    guard=runtime.base_guard.verify()
    artifacts,entries=bank.load_artifacts(stage2,episodes,guard)
    router=MemoryRouter.from_state(dict(distance='euclidean',entries=entries),device=runtime.device)
    roles,_=bank.freeze_roles(episodes)
    target=runtime.target_lock['balancedit']['targets'][0]
    calibration=[]
    for e in episodes:
        for row in e['rows']:
            if row['role'] not in ('native','calibration'): continue
            batch=bank.input_batch(runtime,row)
            with torch.no_grad(): key=runtime.extract_layer_input_key(batch,module_path=target,pooling='mean')
            route=asdict(router.route(key))
            calibration.append(dict(edit=e['event_index'],role=row['role'],label=row['label'],
                family=row.get('family',row.get('rewrite_family','native' if row['role']=='native' else 'V1_context_wrapper')),
                negative_group=row.get('negative_group'),strict_role=roles[e['record_id'],row['logical_id']]['strict_role'],
                eqkey=row['eqkey'],route=route))
    lock=calibrate(calibration)
    lock.update(prefix=prefix,calibration_sha256=vf.sha256_json(calibration),
                frozen_epoch=time.time(),scope='prefix-only complete original bank competition')
    vf.atomic_json(out/'CALIBRATION_PRIVATE.json',dict(rows=calibration,lock=lock))
    vf.atomic_json(out/'THRESHOLD_LOCK.json',dict(lock,threshold_sha256=vf.sha256_json(lock)))
    # Evaluate only after threshold freezing. Earlier artifact reads validate bindings, not calibration outcomes.
    old=read(stage3/f'private/bank/prefix{prefix}/result_private.json')
    if old['base_guard']['after_sha256']!=guard['after_sha256'] or not old['base_guard']['unchanged']:
        raise ValueError('bank Base binding mismatch')
    old_items={(x['method'],x['source_expert'],x['row']['logical_id']):x for x in old['outputs']}
    child=run/'private/cohorts'/BANK
    new_results={}
    for e in episodes:
        t=task_for(BANK,e['event_index'])
        path=child/'private/tasks'/t['task_id']/'result_private.json'
        if path.exists():
            result=read(path)
            if result['a2_sha256']!=e['a2_sha256'] or result['step']!=320: raise ValueError('W01 bank checkpoint binding')
            new_results[e['record_id']]=(result,path.parent/'attempt_chunk16/step0320.pt')
    caches={};outputs=[];missing=[];replay_checked=False
    def w01(selected,row,base):
        if selected is None:return base
        if selected not in new_results: return None
        result,path=new_results[selected]
        key=(selected,row['eqkey'])
        if key not in caches:
            matches=[v for v in result['outputs'].values() if v['row']['eqkey']==row['eqkey']]
            if matches:
                bank.input_batch(runtime,row)
                caches[key]=matches[0]['forced']
            else:
                state=torch.load(path,map_location='cpu',weights_only=True)
                expert=vf.AsymmetricCPExpert(14336,4096,4).to(runtime.device)
                expert.load_state_dict(state['expert']);expert.requires_grad_(False)
                caches[key]=sw.generated(runtime,row,expert)
                del expert
        return caches[key]
    for e in episodes:
        for row in bank.evaluation_rows(e):
            old_item=old_items['S1',e['record_id'],row['logical_id']]
            batch=bank.input_batch(runtime,row)
            with torch.no_grad(): key=runtime.extract_layer_input_key(batch,module_path=target,pooling='mean')
            decision=asdict(router.route(key))
            if decision!=old_item['route']: raise ValueError('R0 route parity failed')
            if accepted(decision,0.)!=decision['activated']: raise ValueError('kappa0 evaluation parity failed')
            if not replay_checked:
                if not bank.same_output(sw.generated(runtime,row,None),old_item['base']): raise ValueError('OFF/Base parity failed')
                replay_checked=True
            on=accepted(decision,lock['kappa']) if lock['status']!='CALIBRATION_UNSUPPORTED' else decision['activated']
            for method in ('W0','W1','W01','BE'):
                if method=='W01':
                    actual=w01(decision['logical_edit_id'],row,old_item['base'])
                    own=w01(e['record_id'],row,old_item['base'])
                    if actual is None or own is None:
                        missing.append(dict(method=method,edit=e['event_index'],logical_id=row['logical_id']));continue
                    item=dict(old_item,actual=actual,own_forced=own,method=method)
                else:
                    key_method={'W0':'S0','W1':'S1','BE':'B'}[method]
                    item=dict(old_items[key_method,e['record_id'],row['logical_id']],method=method)
                    if item['route']!=decision: raise ValueError('writer-dependent R0 decision')
                outputs.append(dict(item,route_mode='R0'))
                if method!='BE':
                    outputs.append(dict(item,route_mode='RC',actual=item['actual'] if on else item['base'],on=on,
                        selected_expert=decision['logical_edit_id'] if on else None,
                        r0_selected_expert=decision['logical_edit_id'],kappa=lock['kappa']))
    if not runtime.base_guard.verify()['unchanged']:raise RuntimeError('bank changed Base')
    vf.atomic_json(out/'result_private.json',dict(status='RAW_READY' if not missing else 'PARTIAL',outputs=outputs,
        prefix=prefix,threshold=lock,missing=missing,base_guard=runtime.base_guard.verify(),off_base_parity=replay_checked,
        expected_inputs=sum(len(bank.evaluation_rows(e)) for e in episodes)*7))


class Queue(vf.TaskQueue):
    def ready(self):
        tasks=self.snapshot()['tasks']
        writers_closed=all(t['status'] not in ('PENDING','RUNNING') for t in tasks if t['kind']=='WRITER')
        return [t for t in tasks if t['status']=='PENDING' and (t['kind']=='WRITER' or writers_closed)]

    def claim(self,worker):
        def operation(data):
            tasks=data['tasks'];writers_closed=all(t['status'] not in ('PENDING','RUNNING') for t in tasks if t['kind']=='WRITER')
            for t in sorted(tasks,key=lambda t:t['priority']):
                if t['status']=='PENDING' and (t['kind']=='WRITER' or writers_closed):
                    t.update(status='RUNNING',worker=worker,pid=os.getpid(),started_at=time.time(),attempts=t['attempts']+1)
                    return dict(t)
        return self._locked(operation)


def worker(args):
    run=args.run_root;config=configure(run);gpu=os.environ['CUDA_VISIBLE_DEVICES']
    if gpu not in GPUS or sw.gpu_check(gpu)['free_mib']<24576:raise RuntimeError('worker GPU guard')
    runtime=vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(config['runtime']['cpu_gate'])))
    queue=Queue(run/'private/TASK_QUEUE.json',run)
    with vf.Telemetry(run/'private/GPU_TELEMETRY.jsonl',gpu,'gpu'+gpu):
        while sw.active_elapsed(run)<config['train_seconds'] and not (run/'STOP').exists():
            task=queue.claim('gpu'+gpu)
            if task is None:break
            print('START',task['task_id'],flush=True);started=time.time()
            try:
                (writer if task['kind']=='WRITER' else bank_task)(runtime,run,task)
                queue.update(task['task_id'],'RAW_READY',elapsed_seconds=time.time()-started)
                print('DONE',task['task_id'],flush=True)
            except Exception as error:
                queue.update(task['task_id'],'FAILED',last_error=f'{type(error).__name__}: {error}')
                vf.append_jsonl(run/'private/FAILURES.jsonl',dict(task=task,traceback=traceback.format_exc()))
                raise
            gc.collect();torch.cuda.empty_cache()


def coordinator(args):
    run=args.run_root;config=configure(run)
    if (run/'PIDS.json').exists():raise FileExistsError('coordinator already launched')
    queue=Queue(run/'private/TASK_QUEUE.json',run);children={};live={};rounds=Counter();exits={}
    def elapsed():return sw.active_elapsed(run)
    def launch(name,command,env):
        command,env=neutral_command(command,env,'run' if name.startswith('worker') else 'job')
        with (run/(name+'.log')).open('x') as log:
            children[name]=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        vf.atomic_json(run/'PIDS.json',dict(coordinator=os.getpid(),**{n:p.pid for n,p in children.items()}))
    def stop(p):
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    def wait(name):
        p=children[name]
        while p.poll() is None and elapsed()<8*3600 and not (run/'STOP').exists():time.sleep(2)
        stop(p);exits[name]=p.returncode
        if p.returncode:raise RuntimeError(name+' failed')
    finalizer=[sys.executable,str(ROOT/'scripts/medtrace/finalize_stage4.py')]
    cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
    status=dict(status='RUNNING',compute='RUNNING',judge='NOT_RUN',publication='PENDING')
    initial_status=read(run/'public/EXECUTION_STATUS.json')
    vf.atomic_json(run/'public/EXECUTION_STATUS.json',dict(initial_status,status='RUNNING',compute='RUNNING'))
    try:
        while elapsed()<config['train_seconds'] and not (run/'STOP').exists():
            for gpu,name in list(live.items()):
                proc=children[name]
                if proc.poll() is not None:
                    exits[name]=proc.returncode
                    for t in queue.snapshot()['tasks']:
                        if t['status']=='RUNNING' and t.get('pid')==proc.pid:
                            queue.update(t['task_id'],'FAILED',last_error='worker exited before closure')
                    del live[gpu]
            faults=Counter((t['kind'],t.get('last_error')) for t in queue.snapshot()['tasks'] if t['status']=='FAILED')
            paused={kind for (kind,error),count in faults.items() if count>=2}
            for t in queue.snapshot()['tasks']:
                if t['status']=='PENDING' and t['kind'] in paused:
                    queue.update(t['task_id'],'PAUSED_INTEGRATION',reason='repeated same integration fault in this branch')
            if not queue.ready() and not live:break
            for gpu in GPUS:
                if gpu in live or not queue.ready():continue
                try:
                    if sw.gpu_check(gpu)['free_mib']<24576:continue
                    env=environment(gpu)
                except RuntimeError:continue
                rounds[gpu]+=1;name=f'worker_gpu{gpu}_{rounds[gpu]:03d}'
                launch(name,[sys.executable,str(Path(__file__)), 'worker','--run-root',str(run)],env);live[gpu]=name
            time.sleep(5)
        for name in live.values():stop(children[name])
        for t in queue.snapshot()['tasks']:
            if t['status'] in ('PENDING','RUNNING'):queue.update(t['task_id'],'PENDING_RESUME')
        if (run/'STOP').exists():raise RuntimeError('STOP prevents Judge calls')
        launch('prepare_judge',[*finalizer,'prepare-judge','--run-root',str(run)],cpu);wait('prepare_judge')
        directory=run/'private/judge';side=read(directory/'JUDGE_SIDECAR_PRIVATE.json')
        if side['new']:
            env=None
            while elapsed()<8*3600 and env is None:
                for gpu in GPUS:
                    try:env=environment(gpu,judge=True);break
                    except RuntimeError:continue
                if env is None:time.sleep(15)
            if env is None:raise RuntimeError('Judge GPU unavailable within budget')
            launch('judge',[JUDGE_PYTHON,str(ROOT/'scripts/medtrace/run_fixed_judge_vllm.py'),
                '--model-path',JUDGE,'--packet',str(directory/'JUDGE_PACKET_PRIVATE.jsonl'),
                '--lock',str(Path(config['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json'),
                '--output',str(directory/'JUDGE_OUTPUT_PRIVATE.jsonl'),
                '--execution-lock',str(directory/'JUDGE_EXECUTION_LOCK_PRIVATE.json'),
                '--preflight-output',str(directory/'JUDGE_LENGTH_PREFLIGHT_PRIVATE.json'),
                '--max-model-len',str(side['max_model_len'])],env);wait('judge')
        launch('finalize',[*finalizer,'finalize','--run-root',str(run)],cpu);wait('finalize')
        status=read(run/'public/EXECUTION_STATUS.json')
    except Exception as error:
        status.update(status='PARTIAL',error_type=type(error).__name__,private_failure_record='FAILURE_PRIVATE.json')
        vf.atomic_json(run/'FAILURE_PRIVATE.json',dict(error=str(error),traceback=traceback.format_exc()))
    finally:
        for name,p in children.items():stop(p);exits[name]=p.returncode
        timing=dict(wall_seconds=elapsed(),gpu_hours_upper_bound=4*elapsed()/3600,
                    wall_limit_hours=8,gpu_limit_hours=32,process_exit_codes=exits)
        status.update(timing=timing,queue_counts=dict(Counter(t['status'] for t in queue.snapshot()['tasks'])))
        vf.atomic_json(run/'public/GPU_AND_TIMING.json',timing)
        vf.atomic_json(run/'public/EXECUTION_STATUS.json',status)
        vf.atomic_json(run/'RUN_COMPLETION.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','worker','coordinator'])
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--stage3-run',type=Path)
    p.add_argument('--prior-preparation',type=Path)
    a=p.parse_args();globals()[a.action](a)
