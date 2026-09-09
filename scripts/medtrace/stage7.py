#!/usr/bin/env python3
"""One APR development experiment using frozen original fit/evaluation roles."""
import argparse
from collections import Counter,defaultdict
import copy
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from methods.medtrace.anchor_repair import ExpandedExpert,CaptureHook,select_fit,repair
from scripts.medtrace import finalize_stage4 as f4
from scripts.medtrace.run_stage2 import vf,read,sw,same_output
from scripts.medtrace.run_stage3_bank import input_batch,file_identity
from scripts.medtrace.stage4_scope import accepted
from scripts.medtrace.stage5_base_judge import stage4_pool


def prepare(args):
    run=args.run_root;old=args.stage4_run
    if run.exists():raise FileExistsError('one-shot Stage7 exists')
    assert shutil.disk_usage(run.parent).free>20*1024**3
    run.mkdir();probe=run/'.probe';probe.write_text('APR');assert probe.read_text()=='APR';probe.unlink()
    cfg=read(old/'private/CAMPAIGN_CONFIG.json')
    cfg.update(kind='MEDTRACE_STAGE7',stage4_run=str(old),campaign_epoch=time.time(),wall_hours=8,
        gpu_hours=16,allowed_physical_gpus=[0,1],new_sgd_training=0,code_commit=args.commit)
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json',cfg)
    vf.atomic_json(run/'private/CAMPAIGN_START.json',dict(epoch=cfg['campaign_epoch']))
    vf.atomic_text(run/'private/PROTOCOL.md',args.protocol.read_text())
    cohorts=read(old/'private/COHORTS.json')['edits'];assert len(cohorts)==23
    tasks=[]
    for order,c in enumerate(cohorts):
        data=read(Path(c['child'])/f"private/edits/e{c['event_index']:02d}.json")
        assert vf.sha256_json(data)==c['data_sha256']
        directory=Path(c['references']['W0']);result=read(directory/'result_private.json')
        points=list(directory.glob('attempt_chunk*/step0320.pt'));assert len(points)==1
        assert result['status']=='RAW_READY' and result['step']==320 and result['task']['condition']=='W0_TASK_ONLY'
        assert result['a2_sha256']==data['a2_sha256'] and result['base_guard']['unchanged']
        positive,negatives=select_fit([r for r in data['rows'] if r['role'] in ('native','fit')])
        tasks.append(dict(order=order,cohort=c['cohort'],event_index=c['event_index'],data=data,
            references=c['references'],checkpoint=str(points[0]),checkpoint_identity=file_identity(points[0]),
            positive_ids=[r['logical_id'] for r in positive],negative_ids={g:[r['logical_id'] for r in rows] for g,rows in negatives.items()},
            status='PENDING' if any(negatives.values()) else 'UNSUPPORTED_NO_NEGATIVE_FIT'))
    vf.atomic_json(run/'private/TASKS.json',tasks)
    lock=read(old/'private/bank/prefix16/THRESHOLD_LOCK.json')
    vf.atomic_json(run/'public/METHOD_AND_RUN_LOCK.json',dict(status='FROZEN_BEFORE_FITTING',
        preparation_commit=args.commit,stage6_public_commit='f0d2fa95b7475a0630043a3ec4483c4002e3c3f3',
        old16_kappa=lock['kappa'],old16_threshold_lock=lock,layer=vf.LAYER,rank=4,
        conditions=['C0_W0_EXPANDED','C1_UNCONSTRAINED_REPAIR','C2_APR'],sgd_steps=0,
        fit_dtype='float64',deploy_dtype='float32',backbone_unchanged=True,
        svd_relative_tolerance=1e-10,fp64_constraint_tolerance=1e-8,fp32_constraint_tolerance=5e-5,
        ridge='0.01 * max(weighted_N_squared_frobenius / n_columns, 1e-12)',
        negative_sampling='at most4 H and4 U source groups sorted by frozen source_group; first EqKey QA; first4 Base predictors',
        positive_sampling='all native and original fit-positive complete W0 trajectories including EOS prediction',
        no_stage6_sidecar_fit=True,known_patients='UNKNOWN',cohorts=dict(Counter(t['cohort'] for t in tasks)),
        maximum_closed_form_solves=46,gpu_indices=[0,1],gpu_uuids={str(i):cfg['gpu_uuids'][str(i)] for i in (0,1)},
        wall_hours_limit=8,gpu_hours_limit=16,all_exposed_development=True))
    vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='PREPARED',edits=len(tasks),publication='PENDING'))
    print('PREPARED',len(tasks),flush=True)


def collect(runtime,row,cp,base):
    batch=input_batch(runtime,row)
    hook=CaptureHook(runtime.get_module(vf.LAYER),cp,base=base);hook.attach()
    try:output=vf.scope_generate(runtime,row,hook)
    finally:hook.detach()
    assert len(hook.columns)==len(output['raw_token_ids']),'incomplete actual predictor trajectory'
    values=torch.stack([hook.columns[i] for i in range(len(hook.columns))],dim=1)
    binding=dict(row=row,eqkey=row['eqkey'],prompt_token_ids=batch.raw_input_ids.tolist(),
        image_identity=batch.image_sha256,attention=batch.attention_mask.tolist() if batch.attention_mask is not None else None,
        module=vf.LAYER,generation=runtime.generation_config,normalization_epsilon=cp.epsilon,
        prefix_source='BASE' if base else 'ORIGINAL_W0',generated_tokens=output['raw_token_ids'],
        predictor_prefix_lengths=list(range(len(output['raw_token_ids']))),trace=output['request_lifecycle'])
    return values,binding,output


def worker(args):
    run=args.run_root;cfg=read(run/'private/CAMPAIGN_CONFIG.json');tasks=read(run/'private/TASKS.json')
    runtime=vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['runtime']['cpu_gate'])))
    def budget():
        if time.time()-cfg['campaign_epoch']>=6*3600 or (run/'STOP').exists():raise TimeoutError('compute deadline/STOP')
    for task in tasks:
        order=task['order']
        if not (order==0 if args.part=='first' else order>0 and order%2==int(args.part)):continue
        directory=run/f'private/edits/e{order:02d}'
        if (directory/'result.json').exists():continue
        budget();started=time.time();data=task['data'];rid=data['record_id']
        print('START',order,task['cohort'],task['event_index'],flush=True)
        if task['status']!='PENDING':
            vf.atomic_json(directory/'result.json',dict(status=task['status'],entries=[]));continue
        assert file_identity(task['checkpoint'])==task['checkpoint_identity'],'W0 checkpoint changed'
        state=torch.load(task['checkpoint'],map_location='cpu',weights_only=True)
        old=read(Path(task['references']['W0'])/'result_private.json')
        assert state['step']==320 and state['task']['seed']==old['task']['seed']
        cp=vf.AsymmetricCPExpert(14336,4096,4).to(runtime.device);cp.load_state_dict(state['expert']);cp.requires_grad_(False)
        assert runtime.base_guard.verify()['after_sha256']==old['base_guard']['after_sha256']
        rows={r['logical_id']:r for r in data['rows']}
        bindings=[];anchors=[];neg={};original_fit={}
        for lid in task['positive_ids']:
            budget();x,b,o=collect(runtime,rows[lid],cp,False)
            assert same_output(o,old['outputs'][lid]['forced']),'original W0 fit replay changed'
            anchors.append(x);bindings.append(b);original_fit[lid]=o
        for group,ids in task['negative_ids'].items():
            neg[group]=[]
            for lid in ids:
                budget();x,b,o=collect(runtime,rows[lid],cp,True)
                assert same_output(o,old['outputs'][lid]['base']),'Base negative prefix replay changed'
                neg[group].append(x[:,:4]);b['selected_predictors']=list(range(min(4,x.shape[1])));bindings.append(b)
        active=[g for g in neg if neg[g]]
        weighted=[x*math.sqrt(1/len(active)/len(neg[g])/x.shape[1]) for g in active for x in neg[g]]
        X=torch.cat(anchors,1);N=torch.cat(weighted,1);assert N.shape[1]<=32
        A=cp.input_basis().T.detach().cpu();B=(cp.output_basis()*cp.rho*(cp.beta/math.sqrt(cp.rank))).detach().cpu()
        fit_start=time.time();matrices,geometry=repair(A,B,X,N);fit_seconds=time.time()-fit_start
        c0=ExpandedExpert(cp).to(runtime.device)
        real=X[:,:min(32,X.shape[1])].T.to(runtime.device)
        torch.testing.assert_close(c0.residual(real),cp.residual(real),rtol=1e-5,atol=1e-6)
        sw.save(directory/'ACTIVATIONS_PRIVATE.pt',dict(X=X,N=N,bindings=bindings,checkpoint=task['checkpoint_identity']))
        experts={name:ExpandedExpert(cp,matrix.to(runtime.device)).to(runtime.device) for name,matrix in zip(('C0','C1','C2'),matrices)}
        payload=dict(experts={n:e.state_dict() for n,e in experts.items()},beta=cp.beta,epsilon=cp.epsilon,rank=cp.rank)
        sw.save(directory/'REPAIRS_PRIVATE.pt',payload)
        saved=torch.load(directory/'REPAIRS_PRIVATE.pt',map_location=runtime.device,weights_only=True)
        for name,expert in experts.items():expert.load_state_dict(saved['experts'][name])
        entries=[];replays=[];be=read(Path(task['references']['BE'])/'result_private.json')
        k=read(run/'public/METHOD_AND_RUN_LOCK.json')['old16_kappa']
        evalrows=[r for r in data['rows'] if r['role']!='calibration']
        # Fit panels are explicit in-sample replay, never renamed evaluation.
        for row in evalrows:
            budget();input_batch(runtime,row);base=old['outputs'][row['logical_id']]['base']
            route=be['outputs'][row['logical_id']]['route'];on=accepted(route,k)
            assert route['logical_edit_id'] in (rid,None),'unexpected expert in single-edit router'
            for name,expert in experts.items():
                forced=sw.generated(runtime,row,expert)
                fixed=forced if on else base
                if order==0 and len(replays)<8 and not any(r['condition']==name and r['on']==on for r in replays):
                    observed=sw.generated(runtime,row,expert if on else None)
                    assert same_output(observed,fixed),'actual fixed-gate branch replay mismatch'
                    replays.append(dict(condition=name,on=on,status='PASSED'))
                item=dict(row=row,base=base,forced=forced,fixed=fixed,fixed_on=on,route=route)
                entries.append(dict(track='A',prefix=0,edit=task['event_index'],method=name,item=item,
                    target=data['event']['edit_record']['gold_answer'],cohort_name='APR_'+task['cohort'],
                    common_support=False,system_valid=True))
            # Reuse W1 only after checking full row, seed and A2 checkpoint bindings.
            w1=read(Path(task['references']['W1'])/'result_private.json')
            if w1['status']=='RAW_READY' and w1['step']==320 and w1['task']['seed']==old['task']['seed'] and w1['a2_sha256']==old['a2_sha256']:
                item=w1['outputs'][row['logical_id']];assert item['row']==row and same_output(item['base'],base)
                item=dict(item,fixed=item['forced'] if on else base,fixed_on=on,route=route)
                entries.append(dict(track='A',prefix=0,edit=task['event_index'],method='HISTORICAL_W1',item=item,
                    target=data['event']['edit_record']['gold_answer'],cohort_name='APR_'+task['cohort'],common_support=False,system_valid=True))
        geometry.update(cohort=task['cohort'],edit=task['event_index'],fit_seconds=fit_seconds,
            support='H_U_REPAIR' if len(active)==2 else 'U_ONLY_REPAIR' if active==['U'] else 'H_ONLY_REPAIR',
            original_parameters=sum(p.numel() for p in cp.parameters()),
            expanded_parameters=sum(b.numel() for b in c0.buffers()),
            expanded_tensor_bytes=sum(b.numel()*b.element_size() for b in c0.buffers()),
            repairs_container_bytes=(directory/'REPAIRS_PRIVATE.pt').stat().st_size,
            elapsed_seconds=time.time()-started)
        guard=runtime.base_guard.verify();assert guard['unchanged']
        vf.atomic_json(directory/'result.json',dict(status='COMPLETE',entries=entries,geometry=geometry,replays=replays,base_guard=guard))
        print('DONE',order,'FP64_ANCHOR',geometry['fp64_anchor_error'],flush=True)


def inventory(run):
    entries=[];ledger=[];cost=[]
    for t in read(run/'private/TASKS.json'):
        p=run/f"private/edits/e{t['order']:02d}/result.json"
        r=read(p) if p.exists() else dict(status='PENDING',entries=[])
        ledger.append(dict(cohort=t['cohort'],edit=t['event_index'],status=r['status']))
        entries+=r['entries']
        if r.get('geometry'):cost.append(r['geometry'])
    return entries,ledger,cost


def install(run):
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');protocol=read(Path(cfg['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json')
    pool,execution=stage4_pool(cfg,protocol)
    f4.f3.inventory=inventory;f4.f3.historical_judge=lambda *_:(pool,execution)


def report(args):
    run=args.run_root;install(run);entries,ledger,geometry=inventory(run)
    verdicts,side=f4.f3.current_verdicts(run);assert side
    rows=f4.details(entries,verdicts,side['protocol_sha256'])
    for r in rows:
        if r['mode']=='R0':r['mode']='RC_FIXED_OLD16'
    tables=[r for r in f4.summarize(rows) if r['cohort'].startswith('APR_')]
    for t in tables:
        g=[r for r in rows if all(r[k]==t[k] for k in ('cohort','method','mode','role','stratum','strict_role'))]
        t['unique_eqkeys']=len({r['eqkey'] for r in g})
    f4.csv_write(run/'public/CORE_REPAIR_RESULTS.csv',tables)
    costs=[]
    for g in geometry:
        for name in ('C0','C1','C2'):
            costs.append(dict(**{k:v for k,v in g.items() if k not in ('C0','C1','C2','anchor_singular_values','R_singular_values')},
                condition=name,**g[name],anchor_singular_min=min(g['anchor_singular_values']),anchor_singular_max=max(g['anchor_singular_values']),
                R_singular_min=min(g['R_singular_values']),R_singular_max=max(g['R_singular_values'])))
    f4.csv_write(run/'public/REPAIR_GEOMETRY_AND_COST.csv',costs)
    # Reuse fixed-seed edit/image-cluster sensitivity; compare the actual conditions.
    from scripts.medtrace.stage5_existing import pairs
    aliases={'C0':'W0','C1':'W1','C2':'BE'};effects=[]
    for mode in ('FORCED_ON','RC_FIXED_OLD16'):
        subset=[dict(r,method=aliases[r['method']],mode='DEV_THRESHOLD_TRANSFER_DIAGNOSTIC') for r in rows if r['method'] in aliases and r['mode']==mode]
        for e in pairs(subset):
            e.update(candidate={v:k for k,v in aliases.items()}[e['candidate']],control={v:k for k,v in aliases.items()}[e['control']],mode=mode,control_mode=mode)
            effects.append(e)
    f4.csv_write(run/'public/PAIRED_EFFECTS.csv',effects)
    baseline={(r['cohort'],r['mode'],r['edit'],r['eqkey']):r for r in rows if r['method']=='C0'}
    changes=defaultdict(list)
    for r in rows:
        if r['method'] not in ('C1','C2'):continue
        b=baseline.get((r['cohort'],r['mode'],r['edit'],r['eqkey']))
        if not b or r['semantic'] is None or b['semantic'] is None:continue
        changes[r['cohort'],r['method'],r['mode'],r['role'],r['stratum']].append(dict(
            lost_correction=int(b['base_correct']==0 and b['semantic']==1 and r['semantic']==0),
            avoided_damage=int(b['base_correct']==1 and b['semantic']==0 and r['semantic']==1),
            introduced_damage=int(b['base_correct']==1 and b['semantic']==1 and r['semantic']==0),
            gained_correction=int(b['base_correct']==0 and b['semantic']==0 and r['semantic']==1)))
    f4.csv_write(run/'public/CORRECTION_TRADEOFF_COUNTS.csv',[dict(zip(('cohort','condition','mode','role','panel'),key),
        inputs=len(g),**{k:sum(r[k] for r in g) for k in g[0]}) for key,g in changes.items()])
    vf.atomic_json(run/'private/DETAILS.json',rows)
    status=dict(status='COMPUTE_COMPLETE' if all(t['status'] in ('COMPLETE','UNSUPPORTED_NO_NEGATIVE_FIT') for t in ledger) and len(verdicts)==len(side['all_expected']) else 'PARTIAL',
        coverage=ledger,judge_required=len(side['all_expected']),judge_scored=len(verdicts),judge_missing=len(set(side['all_expected'])-verdicts.keys()),
        judge_new=side['new'],judge_reused=side['reused'],closed_form_solves=2*len(geometry),new_sgd_training=0,publication='PUBLICATION_PENDING')
    vf.atomic_json(run/'public/RUN_STATUS.json',status)
    lines=['# Stage7 APR development results','','Status: '+status['status'],
        'Finite activation constraints are not unseen-query semantic guarantees. All data are exposed development cohorts. No SGD or threshold retuning.',
        'C0/C1/C2 share rank4 and the expanded input interface; input CP structure is no longer imposed. Storage is not the original1476 parameters.',
        '', '| Cohort | Condition | Mode | Role | Panel | Inputs | Images | Correct | Base-correct damage |',
        '|---|---|---|---|---|---:|---:|---:|---:|']
    def pct(v):return 'NA' if v is None else f'{100*v:.2f}%'
    for r in tables:
        if r['average']=='macro' and r['mode']=='FORCED_ON':lines.append(f"| {r['cohort']} | {r['method']} | {r['mode']} | {r['role']} | {r['stratum']} | {r['inputs']} | {r['source_images']} | {pct(r['semantic'])} | {pct(r['base_correct_damage'])} |")
    lines+=['','Full numerical geometry, FP32/FP64 anchor errors and paired sensitivity are separate. Fit/native are in-sample, not generality evidence. Historical W1 had gradient training, not the same fitting budget. System all-OFF masks writer differences. No clinical safety, originality, full TIME/AlphaEdit/M-ORE or M3Bench reproduction claim.',
        '',f"Judge missing: {status['judge_missing']}. Publication is tracked separately."]
    vf.atomic_text(run/'public/GPT_PRO_REVIEW.md','\n'.join(lines)+'\n')
    print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','worker','prepare-judge','report'))
    p.add_argument('--run-root',required=True,type=Path);p.add_argument('--stage4-run',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--commit');p.add_argument('--part',choices=('first','0','1'))
    a=p.parse_args()
    if a.action=='prepare':prepare(a)
    elif a.action=='worker':worker(a)
    elif a.action=='report':report(a)
    else:
        install(a.run_root);side=f4.f3.prepare_judge(a)
        assert not side['execution_version_changed'],'Judge numerical execution change forbidden'
