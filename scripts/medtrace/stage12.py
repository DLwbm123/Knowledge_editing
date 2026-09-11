#!/usr/bin/env python3
"""Stage12 V2 only: one no-H ablation and fixed single-edit deployment."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import shutil
from statistics import mean
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from scripts.medtrace import stage11 as previous
from scripts.medtrace.stage11_worker import worker
from scripts.medtrace.stage4_scope import accepted
from scripts.medtrace.stage5_existing import pairs as paired_stats
vf,read,f4=previous.vf,previous.read,previous.f4
WRITERS=('C_FACT','C_NO_H','BE')
MODES=('BE_ROUTE_R0','RC_FIXED_OLD16')


def mixed_csv(path,rows):
    fields=dict.fromkeys(k for row in rows for k in row)
    f4.csv_write(path,[{k:row.get(k) for k in fields} for row in rows])


def deploy(entry,mode,kappa):
    item=dict(entry['item']);on=bool(item['route']['activated']) if mode==MODES[0] else accepted(item['route'],kappa)
    item.update(fixed_on=on,fixed=item['forced'] if on else item['base'])
    return dict(entry,item=item,route_mode=mode,provenance='DERIVED_FROM_BOUND_FROZEN_FORCED_AND_BASE')


def prepare(args):
    run=args.run_root;old=args.stage11_run;cfg=read(old/'private/CAMPAIGN_CONFIG.json')
    assert read(old/'RUN_COMPLETION.json')['completed_final_endpoints']==45
    assert read(old/'RUN_COMPLETION.json')['judge_missing']==0
    assert cfg['code_commit']=='25698311ad8f735304bf5c4cb117049ea6e24002'
    if run.exists():raise FileExistsError('preserve run; no second Stage12 protocol')
    assert shutil.disk_usage(run.parent).free>20*1024**3
    run.mkdir();p=run/'.probe';p.write_text('run');assert p.read_text()=='run';p.unlink()
    cfg.update(kind='MEDTRACE_STAGE12_V2',stage11_run=str(old),campaign_epoch=time.time(),code_commit=args.commit,
        worker_gpus=[0],judge_gpu=0,allowed_physical_gpus=[0],wall_hours=4,gpu_hours=8,authorization='USER_STAGE12_GPU0')
    tasks=read(old/'private/TASKS.json');assert len(tasks)==15 and all(t['stage11_status']=='PENDING' for t in tasks)
    k=read(Path(cfg['stage4_run'])/'private/bank/prefix16/THRESHOLD_LOCK.json')['kappa'];assert k==0.7696741135364367
    cfg['fixed_kappa']=k
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json',cfg);vf.atomic_json(run/'private/CAMPAIGN_START.json',dict(epoch=cfg['campaign_epoch']))
    vf.atomic_text(run/'private/PROTOCOL.md',args.protocol.read_text());vf.atomic_json(run/'private/TASKS.json',tasks)
    entries=[];coverage=[]
    for t in tasks:
        d=old/('private/edits/e%02d/J1_FREE_R4'%t['order'])
        assert read(d/'result.json')['status']=='COMPLETE' and (d/'step0320.pt').exists()
        fact=read(d/'endpoint0320.json')['entries'];be=read(Path(t['references']['BE'])/'result_private.json')
        assert be['status']=='RAW_READY'
        matched=0
        for e in fact:
            item=e['item'];row=item['row'];lid=row['logical_id'];b=be['outputs'].get(lid)
            assert b is not None and all(b['row'][key]==row[key] for key in ('question','image_path','reference','eqkey'))
            assert b['route']==item['route'] and b['base']['raw_token_ids']==item['base']['raw_token_ids']
            assert item['fixed_on']==accepted(item['route'],k)
            base=dict(e,cohort_name='CLOSEOUT_'+t['cohort'],method='C_FACT',diagnostic_step=None)
            entries.append(base)
            entries.append(dict(base,method='BE',item=dict(b,row=row)));matched+=1
        coverage.append(dict(edit=t['event_index'],candidate_inputs=len(fact),BE_matched_inputs=matched,
            selected_H_fit=len(t['h_ids']),medical_scope=t['medical_scope'],patients='UNKNOWN'))
    vf.atomic_json(run/'private/REUSED_ENTRIES.json',entries)
    vf.atomic_json(run/'public/RUN_COVERAGE_COST.json',dict(status='PREPARED',execution_sha=args.commit,
        stage11_public_sha='0f5c11cd75780be25f03ec7aff47d00c085f17e8',stage11_execution_sha=cfg['code_commit'] if cfg['kind']=='MEDTRACE_STAGE11' else '25698311ad8f735304bf5c4cb117049ea6e24002',
        fixed_candidate='C_FACT+BE_ROUTE+RC_FIXED_OLD16',kappa=k,coverage=coverage,new_training_trajectories=15,new_optimizer_steps=4800,
        steps=320,rank=4,parameters=73728,loss='0.5 native CE +0.5 fit CE +0.01 token-mean full-vocabulary Base||student KL; no H loss',
        optimizer='unchanged Stage11 Adam 1e-4 A/1e-3 B, clip1',seed='original W0 per-edit seed',schedule_seed=20260910,
        wall_hours=4,gpu_hours=8,training_generation_hours=3,gpu=[0],RC='read-only old16; no calibration',
        exposure='viewed development; not patient-independent confirmation',auxiliary_diagnostics='not added',publication='PENDING'))
    vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='PREPARED',training='PENDING',judge='PENDING',publication='PENDING'))
    print('PREPARED 15, BE matched',sum(x['BE_matched_inputs'] for x in coverage),flush=True)


def replay(runtime,run,t,expert,new_entries):
    """A few actual new-adapter branch replays; unchanged Base-route bindings reused."""
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');k=cfg['fixed_kappa'];out=[]
    cp=vf.AsymmetricCPExpert(14336,4096,4).to(runtime.device)
    from methods.medtrace.selective_write import LowRankExpert
    from scripts.medtrace import stage7 as s
    fact=LowRankExpert(cp,0,rank=4).to(runtime.device)
    point=Path(cfg['stage11_run'])/('private/edits/e%02d/J1_FREE_R4/step0320.pt'%t['order'])
    fact.load_state_dict(torch.load(point,map_location=runtime.device,weights_only=True)['expert']);fact.requires_grad_(False)
    originals=[e for e in read(run/'private/REUSED_ENTRIES.json') if e['edit']==t['event_index'] and e['method']=='C_FACT']
    for name,writer,entries in (('C_FACT',fact,originals),('C_NO_H',expert,new_entries)):
        for mode in MODES:
            seen=set()
            for e in entries:
                item=deploy(e,mode,k)['item'];on=item['fixed_on']
                if on in seen:continue
                s.input_batch(runtime,item['row']);actual=s.sw.generated(runtime,item['row'],writer if on else None)
                good=s.same_output(actual,item['fixed']);out.append(dict(writer=name,mode=mode,on=on,status='PASSED' if good else 'MISMATCH'))
                seen.add(on)
            for missing in {True,False}-seen:out.append(dict(writer=name,mode=mode,on=missing,status='NOT_NATURALLY_OBSERVED'))
    vf.atomic_json(run/'private/SYSTEM_REPLAY.json',dict(status='PASSED' if all(r['status']!='MISMATCH' for r in out) else 'SYSTEM_PENDING',rows=out,
        routing='unchanged historical Base-derived BE keys/routes, bound during prepare; not re-extracted or recalibrated'))
    del fact,cp


def inventory(run):
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');entries=read(run/'private/REUSED_ENTRIES.json');ledger=[];costs=[]
    for t in read(run/'private/TASKS.json'):
        d=run/('private/edits/e%02d/C_NO_H'%t['order']);p=d/'endpoint0320.json'
        if p.exists():entries.extend(dict(e,cohort_name='CLOSEOUT_'+t['cohort']) for e in read(p)['entries'])
        done=(d/'result.json').exists();ledger.append(dict(edit=t['event_index'],condition='C_NO_H',status='COMPLETE' if done else 'PENDING'))
        if done:costs.append(dict(edit=t['event_index'],condition='C_NO_H',**read(d/'result.json')))
        old=Path(cfg['stage11_run'])/('private/edits/e%02d/J1_FREE_R4/result.json'%t['order'])
        costs.append(dict(edit=t['event_index'],condition='C_FACT',reused=True,**{k:v for k,v in read(old).items() if k!='curve'}))
    return [deploy(e,m,cfg['fixed_kappa']) for e in entries for m in MODES],ledger,costs


def install(run):
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');old=Path(cfg['stage11_run']);previous.install(old)
    verdicts,side=f4.f3.current_verdicts(old);assert not set(side['all_expected'])-verdicts.keys()
    protocol=read(Path(cfg['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json');pool,execution=f4.f3.historical_judge(cfg,protocol);pool=pool.copy()
    identity=read(old/'private/judge/REUSE_EXECUTION_IDENTITY_PRIVATE.json');tuples=f4.f3.tuples_for(previous.inventory(old)[0],protocol['config_sha256'])
    for key,value in verdicts.items():pool[key]=(tuples[key],value,identity)
    f4.f3.inventory=inventory;f4.f3.historical_judge=lambda *_:(pool,execution)


def report(args):
    run=args.run_root;install(run);entries,ledger,costs=inventory(run);verdicts,side=f4.f3.current_verdicts(run)
    details=[]
    for mode in MODES:
        rows=f4.details([e for e in entries if e['route_mode']==mode],verdicts,side['protocol_sha256'])
        for row in rows:
            if row['method'] not in (*WRITERS,'BASE'):continue
            if row['mode']=='R0':details.append(dict(row,mode=mode))
            elif mode==MODES[0]:details.append(row)
    forced=[r for r in details if r['mode']=='FORCED_ON' and r['method'] in ('C_FACT','C_NO_H')]
    writer=[dict(r,row_kind='METRIC') for r in f4.summarize(forced) if r['method'] in ('C_FACT','C_NO_H') and r['cohort'].startswith('CLOSEOUT_')]
    mapped=[dict(r,method='W1' if r['method']=='C_FACT' else 'W0',mode='DEV_THRESHOLD_TRANSFER_DIAGNOSTIC') for r in forced]
    effects=paired_stats(mapped)
    writer.extend(dict(r,row_kind='PAIRED_EFFECT',candidate='C_FACT',control='C_NO_H',mode='FORCED_ON',control_mode='FORCED_ON') for r in effects if r['candidate']=='W1' and r['control']=='W0')
    tasks=read(run/'private/TASKS.json');mapping={'C_FACT':'J1_FREE_R4','C_NO_H':'J0_CP_R4'};inverse={v:k for k,v in mapping.items()}
    pairs=previous.pair_outcomes([dict(r,method=mapping[r['method']]) for r in forced],tasks)
    pairs=[dict(r,condition=inverse[r['condition']]) for r in pairs if r['step']==320]
    writer.extend(dict(r,row_kind='PAIR_COUNTS') for r in pairs)
    for panel in ('FIT_PairCorrect','EVAL_PairCorrect'):
        groups={m:{r['edit']:r for r in pairs if r['condition']==m and r['panel']==panel} for m in mapping}
        common=groups['C_FACT'].keys()&groups['C_NO_H'].keys()
        delta=[groups['C_FACT'][e]['pair_correct']-groups['C_NO_H'][e]['pair_correct'] for e in sorted(common)]
        if delta:
            lo,hi=f4.bootstrap(delta);writer.append(dict(row_kind='PAIR_EDIT_EFFECT',panel=panel,candidate='C_FACT',control='C_NO_H',delta=mean(delta),ci_low=lo,ci_high=hi,paired_edits=len(delta)))
    mixed_csv(run/'public/WRITER_ABLATION.csv',writer)
    system=[r for r in details if r['mode'] in MODES or r['method']=='BASE']
    # Baselines are compared on their actual common input support, no substituted targets.
    supports={m:{(r['edit'],r['eqkey']) for r in system if r['method']==m} for m in WRITERS}
    common=set.intersection(*supports.values())
    system=[r for r in system if (r['edit'],r['eqkey']) in common]
    base_tables=[r for r in f4.summarize(system) if r['method'] in (*WRITERS,'BASE') and r['cohort'].startswith('CLOSEOUT_')]
    system_tables=[dict(r,row_kind='METRIC',provenance='frozen output derivation') for r in base_tables]
    for method in WRITERS:
        for mode in MODES:
            a={(r['edit'],r['eqkey']):r for r in details if r['method']==method and r['mode']=='FORCED_ON'}
            groups=defaultdict(list)
            for r in system:
                if r['method']==method and r['mode']==mode and r['semantic'] is not None:groups[(r['role'],r['stratum'])].append(r)
            for (role,panel),rs in groups.items():
                paired=[(a[r['edit'],r['eqkey']],r) for r in rs if (r['edit'],r['eqkey']) in a and a[r['edit'],r['eqkey']]['semantic'] is not None]
                avoided=sum(x['semantic']==0 and y['semantic']==1 for x,y in paired);lost=sum(x['semantic']==1 and y['semantic']==0 for x,y in paired)
                system_tables.append(dict(row_kind='REJECTION_ACCOUNTING',method=method,mode=mode,role=role,stratum=panel,inputs=len(paired),
                    on_count=sum(y['on']==1 for _,y in paired),returned_base=sum(y['on']==0 for _,y in paired),avoided_errors=avoided,lost_corrections=lost,
                    net_correct_change=avoided-lost,positive_wrong_rejection=sum(y['on']==0 and x['semantic']==1 and y['semantic']==0 and y['strict_role']=='EDIT_TARGET' for x,y in paired)))
    replay_path=run/'private/SYSTEM_REPLAY.json';replays=read(replay_path) if replay_path.exists() else dict(status='PENDING')
    mixed_csv(run/'public/SINGLE_SYSTEM_RESULTS.csv',[dict(r,replay_status=replays['status']) for r in system_tables])
    missing=len(set(side['all_expected'])-verdicts.keys());completed=sum(r['status']=='COMPLETE' for r in ledger)
    status=dict(status='COMPUTE_COMPLETE' if completed==15 and not missing and replays['status']=='PASSED' else 'PARTIAL',training_completed=completed,
        planned_training=15,judge_required=len(side['all_expected']),judge_missing=missing,judge_new=side['new'],judge_reused=side['reused'],system_replay=replays['status'],publication='PENDING')
    for cost in costs:
        if 'curve' in cost:cost['curve']=[{k:v for k,v in row.items() if not k.endswith('_id')} for row in cost['curve']]
    record=read(run/'public/RUN_COVERAGE_COST.json');record.update(status=status,ledger=ledger,costs=costs,replays=replays,system_common_inputs=len(common))
    vf.atomic_json(run/'public/RUN_COVERAGE_COST.json',record);vf.atomic_json(run/'public/RUN_STATUS.json',status);vf.atomic_json(run/'private/DETAILS.json',details)
    lines=['# Stage12 V2 focused closeout','',str(status),'','Fixed candidate C_FACT + BE_ROUTE + RC_FIXED_OLD16. Viewed development only; patients UNKNOWN. No new sidecar/bank/algorithm or automatic next experiment.',
        'C_FACT reused Stage11 step320. Only C_NO_H trained320; no step160 evaluation. Same parameterization/optimizer/native-fit-U schedule; the ablation removes H supervision and its compute, not a comparison to all equally supervised generic methods.',
        'System rows are frozen-output derivations, with a few actual adapter branch replays; unchanged historical Base-derived routing is reused. Unsupported official M3Bench metrics remain NA, not constructed T2G.','',
        '|Writer|Pair panel|Both correct / pairs|Pair micro|Edit macro|','|---|---|---:|---:|---:|']
    for method in mapping:
        for panel in ('FIT_PairCorrect','EVAL_PairCorrect'):
            rs=[r for r in pairs if r['condition']==method and r['panel']==panel];n=sum(r['pairs'] for r in rs);yes=sum(r['both_correct'] for r in rs)
            lines.append(f'|{method}|{panel}|{yes}/{n}|{yes/n if n else None}|{mean(r["pair_correct"] for r in rs) if rs else None}|')
    lines+=['','|Writer|Mode|Role|Panel|Correct macro|Base-correct damage|','|---|---|---|---|---:|---:|']
    for r in writer+system_tables:
        if r.get('row_kind')=='METRIC' and r.get('average')=='macro':lines.append('|'+ '|'.join(str(r.get(k)) for k in ('method','mode','role','stratum','semantic','base_correct_damage'))+'|')
    lines+=['','Interpretation pending closeout: assess native and EVAL pairs jointly with U damage, cross-family behavior and cost; zero routed damage may come from returning Base, not writer protection. No new-source or scale validation claimed.']
    vf.atomic_text(run/'public/GPT_PRO_REVIEW.md','\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','worker','prepare-judge','report'));p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--stage11-run',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--commit');p.add_argument('--part',default='0');p.add_argument('--parts',type=int,default=1)
    a=p.parse_args()
    if a.action=='prepare':prepare(a)
    elif a.action=='worker':worker(a)
    elif a.action=='report':report(a)
    else:
        install(a.run_root);side=f4.f3.prepare_judge(a);assert not side['execution_version_changed']
