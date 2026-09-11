#!/usr/bin/env python3
"""Stage13R adapter: training-only original lineage, then separate scoring process."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gc
from pathlib import Path
from statistics import mean
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from scripts.medtrace import run_stage2 as old, stage12
from scripts.medtrace.run_stage3_bank import input_batch
from scripts.medtrace.stage4_scope import accepted
from methods.medtrace.selective_write import LowRankExpert
from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
vf,sw,read,f4=old.vf,old.sw,old.read,stage12.f4
WRITERS=('C_FACT','C_NO_H','BE')


def strict_training(data):
    assert not data['event']['probes']
    assert all(r['role'] in ('native','fit') for r in data['rows'])
    assert len([r for r in data['rows'] if r['role']=='native'])==1
    assert all(any(r.get('negative_group')==g and r['role']=='fit' for r in data['rows']) for g in ('H','U'))


def train(args):
    run=args.run_root;cfg=read(run/'private/CAMPAIGN_CONFIG.json');tasks=read(run/'private/TASKS.json')
    runtime=vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['runtime']['cpu_gate'])))
    runtime.stage2_base={};sw.SEED=old.SEED=20260910
    paired=run/'private/paired';compat=run/'private/interface';ptasks=[]
    pcfg=dict(cfg,stage8_run=str(compat),stage9_run=str(compat),coordinator_start_required=False)
    vf.atomic_json(paired/'private/CAMPAIGN_CONFIG.json',pcfg)
    vf.atomic_json(paired/'private/CAMPAIGN_START.json',dict(epoch=cfg['campaign_epoch']))
    for task in tasks:
        i=task['order'];data=read(run/task['training']);strict_training(data)
        p=run/('private/edits/e%02d.json'%i)
        if p.exists():data=read(p);strict_training(data)
        old.bind_rows(runtime,data)
        # Native eligibility reuse requires identical actual prompt tokens/image and a fresh Base replay.
        n=next(r for r in data['rows'] if r['role']=='native');known=data['base_before']
        prepared=runtime.adapter.prepare_inputs(n['image_path'],n['question'],None)
        assert prepared['input_ids'][0].tolist()==known['prompt_token_ids'] and prepared['image_sha256']==known['image_sha256'] and not known.get('error')
        replay=vf.scope_generate(runtime,n,None)
        assert replay['raw_answer']==known['model_answer_raw'] and replay['raw_token_ids']==known['raw_generated_token_ids'], 'native Base binding drift'
        vf.atomic_json(p,data)
        be=old.queue_task(i,'BE',i);init=old.queue_task(i,'INIT',i)
        w0=old.queue_task(i,'CP',i,parameterization='P4',condition='W0_TASK_ONLY');w0['seed']=task['seed']
        print('INITIALIZE',i,'BE',flush=True)
        bdir=run/'private/tasks'/be['task_id']
        if not (bdir/'result_private.json').exists():old.be_task(runtime,args,be)
        data=read(p)
        if not data.get('a2'):
            # Original initializer: native-only early stopping, then original A2/80. No new performance gate.
            print('INITIALIZE',i,'NATIVE_A2',flush=True);old.initialize_episode(runtime,args,init)
        wdir=run/'private/tasks'/w0['task_id']
        if not (wdir/'result_private.json').exists():
            endpoint=wdir/'attempt_chunk16/step0320.pt'
            if endpoint.exists():w0['resume_checkpoint']=str(endpoint)
            print('INITIALIZE',i,'CP_W0',flush=True);sw.train_task(runtime,args,w0)
        data=read(p);strict_training(data);be_result=read(bdir/'result_private.json')
        entries=[dict(method='B0',item=item) for item in be_result['outputs'].values()]
        assert len(entries)==len(data['rows'])
        vf.atomic_json(compat/('private/edits/e%02d/result.json'%i),dict(entries=entries,base_guard=be_result['base_guard']))
        ptasks.append(dict(order=i,event_index=i,cohort='NEW_SOURCE',stage11_status='PENDING',data=data,native_id='native',
            fit_positive_ids=[r['logical_id'] for r in data['rows'] if r['role']=='fit' and r['label']=='positive'],
            h_ids=[r['logical_id'] for r in data['rows'] if r.get('negative_group')=='H'],
            u_ids=[r['logical_id'] for r in data['rows'] if r.get('negative_group')=='U'],quarantined_ids=[],
            checkpoint=str(wdir/'attempt_chunk16/step0320.pt'),references=dict(BE=str(bdir))))
        print('INITIALIZED',i,flush=True)
    vf.atomic_json(paired/'private/TASKS.json',ptasks)
    del runtime;gc.collect();torch.cuda.empty_cache()
    from scripts.medtrace.stage11_worker import worker
    worker(argparse.Namespace(run_root=paired,parts=1,part='0'))
    vf.atomic_json(run/'private/TRAINING_COMPLETE.json',dict(status='COMPLETE',edits=len(tasks),training_reference_roles=['native','fit'],evaluation_reference_files_read=False))


def generate(args):
    run=args.run_root;cfg=read(run/'private/CAMPAIGN_CONFIG.json');assert read(run/'private/TRAINING_COMPLETE.json')['status']=='COMPLETE'
    runtime=vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['runtime']['cpu_gate'])))
    tasks=read(run/'private/TASKS.json');replays=[];k=cfg['fixed_kappa']
    assert k==0.7696741135364367
    for task in tasks:
        i=task['order'];out=run/('private/scored/e%02d.json'%i)
        if out.exists():continue
        data=read(run/('private/edits/e%02d.json'%i));data['rows']+=read(run/task['evaluation']);old.bind_rows(runtime,data)
        olditems=read(run/'private/tasks'/old.queue_task(i,'BE',i)['task_id']/'result_private.json')['outputs']
        editor=BalanceEditPaperSpecEditor(runtime);base_module=editor.wrapper.base;target=editor.target
        entries=[];start=time.time()
        try:
            editor.load_editor_state(run/'private/tasks'/old.queue_task(i,'BE',i)['task_id']/'editor_state.pt')
            record=vf.EditorRecord.from_dict(data['event']['edit_record'])
            assert editor.router.logical_ids==[record.record_id]
            experts={}
            for name in WRITERS[:2]:
                point=run/('private/paired/private/edits/e%02d'%i)/name/'step0320.pt'
                saved=torch.load(point,map_location=runtime.device,weights_only=True);assert saved['step']==320
                cp=vf.AsymmetricCPExpert(14336,4096,4).to(runtime.device)
                e=LowRankExpert(cp,task['seed'],rank=4).to(runtime.device);e.load_state_dict(saved['expert']);e.requires_grad_(False);experts[name]=e
            replay_seen=set()
            for row in data['rows']:
                if time.time()-cfg['campaign_epoch']>cfg['train_seconds']:raise TimeoutError('training/generation limit')
                batch=input_batch(runtime,row)
                with editor.disabled(),torch.no_grad():
                    key=runtime.extract_layer_input_key(batch,module_path=editor.target,pooling='mean')
                    route=asdict(editor.router.route(key))
                    base=olditems[row['logical_id']]['base'] if row['logical_id'] in olditems else vf.scope_generate(runtime,row,None)
                on=accepted(route,k)
                for name in WRITERS:
                    if name=='BE':
                        with editor._activated(record.record_id):forced=vf.scope_generate(runtime,row,None)
                    else:
                        with editor.disabled():forced=sw.generated(runtime,row,experts[name])
                    item=dict(row=row,base=base,forced=forced,fixed=forced if on else base,fixed_on=on,route=route,
                        disabled_parity=True,disabled_evidence='exact input binding and Base disabled native replay; fresh Base key per query')
                    entries.append(dict(track='A',prefix=0,edit=i,method=name,item=item,target=record.target,
                        cohort_name='STAGE13R_NEW_SOURCE',common_support=False,system_valid=True,diagnostic_step=None))
                    # At most eight actual natural R0/RC branch replays in the campaign.
                    if i==1 and name!='BE':
                        for mode in stage12.MODES:
                            active=bool(route['activated']) if mode=='BE_ROUTE_R0' else on
                            identity_=(name,mode,active)
                            if identity_ not in replay_seen and len(replays)<8:
                                with editor.disabled():actual=sw.generated(runtime,row,experts[name] if active else None)
                                good=old.same_output(actual,forced if active else base)
                                replays.append(dict(writer=name,mode=mode,on=active,passed=good));replay_seen.add(identity_)
                                assert good,'actual deployment branch mismatch'
            vf.atomic_json(out,dict(status='COMPLETE',entries=entries,generation_seconds=time.time()-start,
                base_guard=runtime.base_guard.verify(),new_base_routes=len(data['rows'])))
        finally:
            editor.reset_editor_state();runtime.replace_module(target,base_module)
        assert runtime.base_guard.verify()['unchanged']
        vf.atomic_json(run/'public/SYSTEM_REPLAY.json',dict(rows=replays,maximum=8,status='PASSED',route_keys='fresh Base input features; original BE native anchors and fixed old RC'))
        print('GENERATED',i,len(entries),flush=True)


def inventory(run):
    entries=[];ledger=[]
    for task in read(run/'private/TASKS.json'):
        path=run/('private/scored/e%02d.json'%task['order'])
        if path.exists():entries.extend(read(path)['entries'])
        ledger.append(dict(edit=task['order'],status='COMPLETE' if path.exists() else 'PENDING'))
    return entries,ledger,[]


def install(run):
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');oldrun=Path(cfg['stage12_run']);stage12.install(oldrun)
    verdicts,side=f4.f3.current_verdicts(oldrun);assert not set(side['all_expected'])-verdicts.keys()
    protocol=read(Path(cfg['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json')
    pool,execution=f4.f3.historical_judge(cfg,protocol);pool=pool.copy()
    identity=read(oldrun/'private/judge/REUSE_EXECUTION_IDENTITY_PRIVATE.json')
    tuples=f4.f3.tuples_for(stage12.inventory(oldrun)[0],protocol['config_sha256'])
    for key,value in verdicts.items():pool[key]=(tuples[key],value,identity)
    f4.f3.inventory=inventory;f4.f3.historical_judge=lambda *_:(pool,execution)


def report(args):
    run=args.run_root;install(run);entries,ledger,_=inventory(run);verdicts,side=f4.f3.current_verdicts(run)
    assert set(side['all_expected'])<=verdicts.keys()
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');allrows=[]
    for mode in stage12.MODES:
        detail=f4.details([stage12.deploy(e,mode,cfg['fixed_kappa']) for e in entries],verdicts,side['protocol_sha256'])
        for r in detail:
            if r['mode']=='R0':r['mode']=mode;allrows.append(r)
            elif mode==stage12.MODES[0]:allrows.append(r)
    vf.atomic_json(run/'private/DETAILS.json',allrows)
    tables=[r for r in f4.summarize(allrows) if r['cohort']=='STAGE13R_NEW_SOURCE']
    stage12.mixed_csv(run/'public/NEW_EDIT_WRITER_RESULTS.csv',[r for r in tables if r['mode'] in ('FORCED_ON','BASE')])
    stage12.mixed_csv(run/'public/NEW_EDIT_SYSTEM_RESULTS.csv',[r for r in tables if r['mode'] in stage12.MODES])
    lookup={(r['method'],r['mode'],r['edit'],r['eqkey']):r for r in allrows};pairs=[]
    for task in read(run/'private/TASKS.json'):
        i=task['order'];es=[e for e in entries if e['edit']==i and e['method']=='C_FACT']
        native=next(e['item']['row'] for e in es if e['item']['row']['role']=='native')
        for method in WRITERS:
            for mode in ('FORCED_ON',*stage12.MODES):
                for role in ('fit','evaluation'):
                    hs={e['item']['row']['eqkey']:e['item']['row'] for e in es if e['item']['row']['role']==role and e['item']['row'].get('negative_group')=='H'}
                    counts=Counter()
                    for key,h in hs.items():
                        a=lookup[method,mode,i,native['eqkey']]['semantic'];b=lookup[method,mode,i,key]['semantic']
                        assert a is not None and b is not None
                        counts['both_correct' if a and b else 'native_only' if a else 'H_only' if b else 'both_wrong']+=1
                    pairs.append(dict(edit=i,method=method,mode=mode,role=role,pairs=len(hs),pair_correct=counts['both_correct']/len(hs) if hs else None,
                        **{k:counts[k] for k in ('both_correct','native_only','H_only','both_wrong')}))
    stage12.mixed_csv(run/'public/PAIR_OUTCOMES.csv',pairs)
    effects=[]
    for mode in ('FORCED_ON',*stage12.MODES):
        for role in ('fit','evaluation'):
            a={p['edit']:p for p in pairs if p['method']=='C_FACT' and p['mode']==mode and p['role']==role and p['pairs']}
            b={p['edit']:p for p in pairs if p['method']=='C_NO_H' and p['mode']==mode and p['role']==role and p['pairs']}
            ds=[a[i]['pair_correct']-b[i]['pair_correct'] for i in sorted(a.keys()&b.keys())];lo,hi=f4.bootstrap(ds)
            effects.append(dict(mode=mode,role=role,metric='PairCorrect_edit_macro',paired_edits=len(ds),delta=mean(ds) if ds else None,ci_low=lo,ci_high=hi,
                C_FACT_micro=sum(p['both_correct'] for p in a.values())/sum(p['pairs'] for p in a.values()) if a else None,
                C_NO_H_micro=sum(p['both_correct'] for p in b.values())/sum(p['pairs'] for p in b.values()) if b else None,patients='UNKNOWN'))
    stage12.mixed_csv(run/'public/PAIR_EFFECTS.csv',effects)
    costs=[]
    for task in read(run/'private/TASKS.json'):
        i=task['order']
        native=read(run/('private/initial/e%02d/native/result.json'%i))
        costs.append(dict(edit=i,phase='NATIVE_CP',seconds=native['timing']['end_to_end_seconds'],steps=native['selected']['step'] if 'selected' in native else None))
        a2=read(run/('private/initial/e%02d/A2_INITIALIZATION_PRIVATE.json'%i));costs.append(dict(edit=i,phase='A2',seconds=a2['training_seconds'],steps=80))
        for name,kind,condition in (('CP_W0','CP','W0_TASK_ONLY'),('BE','BE','BALANCEDIT')):
            t=old.queue_task(i,kind,i,parameterization='P4' if name=='CP_W0' else 'BE',condition=condition)
            r=read(run/'private/tasks'/t['task_id']/'result_private.json')
            costs.append(dict(edit=i,phase=name,seconds=r['elapsed_seconds'],steps=r['step'],forwards=r.get('forward_count'),parameters=r.get('parameters')))
        for name in WRITERS[:2]:
            r=read(run/('private/paired/private/edits/e%02d'%i)/name/'result.json')
            costs.append(dict(edit=i,phase=name,seconds=r['wall_seconds'],steps=r['steps'],forwards=r['forwards'],training_tokens=r['training_tokens'],parameters=r['parameters']))
    stage12.mixed_csv(run/'public/COSTS.csv',costs)
    status=dict(status='COMPUTE_COMPLETE' if all(r['status']=='COMPLETE' for r in ledger) else 'PARTIAL',actual_N=len(ledger),
        completed_writers=len(ledger)*3,judge_required=len(side['all_expected']),judge_scored=len(verdicts),judge_reused=side['reused'],judge_new=side['new'],judge_missing=0,publication='PENDING')
    vf.atomic_json(run/'public/RUN_STATUS.json',status)
    lines=['# Stage13R frozen new-source comparison','',str(status),'',
        'Full original train expansion, not a repeat scan of the old875-row pool. See SOURCE_EXPANSION_SUMMARY.json for source revision and prospective roles.',
        'Unexecuted draft queues are not actual exposure. All existing reservations, evaluation-only identities, real development inputs and quality exclusions remain protected.',
        'Native and principal H/U evaluation images are outside previous MedTRACE development; training/evaluation image sets are globally disjoint. Patients and pretraining exposure remain UNKNOWN. Same-image paraphrases are text generalization only.',
        'Original native CP -> A2 -> CP-W0 initialization is counted. C_FACT/C_NO_H use the same own-edit initial function and320-step schedules, differing only H supervision. BalancEdit is the frozen adaptation, not paper-exact or supervision-budget-matched.',
        'FORCED_ON is writer behavior; R0/old fixed RC is deployment behavior. No threshold recalibration or algorithm selection.',
        '', '## Primary paired effects','',*['- '+str(r) for r in effects if r['role']=='evaluation'],
        '', 'See writer/system tables for native, all-source H/U correctness, Base-correct damage and positive/challenge panels; pair four-cells/micro/edit-macro have explicit support. Small shared-image support is not independent patient evidence.',
        'Source curation is agent source consistency, not human clinical certification. No sealed answers entered training; training finished in a separate process before evaluation references were read. No automatic next stage.']
    vf.atomic_text(run/'public/GPT_PRO_REVIEW.md','\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','generate','prepare-judge','report'));p.add_argument('--run-root',type=Path,required=True);args=p.parse_args()
    if args.action=='prepare-judge':install(args.run_root);side=f4.f3.prepare_judge(args);assert not side['execution_version_changed']
    else:globals()[args.action](args)
