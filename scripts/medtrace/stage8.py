#!/usr/bin/env python3
"""Stage8 bounded closeout, reusing frozen Stage7 activations and controls."""
import argparse
import copy
import math
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from scripts.medtrace import stage7 as s
from methods.medtrace.bounded_repair import bounded_repair
from scripts.medtrace.stage5_existing import pairs
vf,read,f4=s.vf,s.read,s.f4


def prepare(args):
    run=args.run_root;old=args.stage7_run
    if run.exists():raise FileExistsError('preserve existing one-shot run')
    assert read(old/'RUN_COMPLETION.json')['status']=='COMPUTE_COMPLETE'
    assert shutil.disk_usage(run.parent).free>20*1024**3
    run.mkdir();probe=run/'.probe';probe.write_text('APR8');assert probe.read_text()=='APR8';probe.unlink()
    cfg=read(old/'private/CAMPAIGN_CONFIG.json')
    cfg.update(kind='MEDTRACE_STAGE8',stage7_run=str(old),campaign_epoch=time.time(),wall_hours=4,
        gpu_hours=8,allowed_physical_gpus=[0],code_commit=args.commit)
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json',cfg)
    vf.atomic_json(run/'private/CAMPAIGN_START.json',dict(epoch=cfg['campaign_epoch']))
    tasks=read(old/'private/TASKS.json');assert len(tasks)==23
    vf.atomic_json(run/'private/TASKS.json',tasks)
    vf.atomic_text(run/'private/PROTOCOL.md',args.protocol.read_text())
    lock=read(old/'public/METHOD_AND_RUN_LOCK.json')
    lock.update(stage7_public_commit='c1be54e590d0a6c7179256d41006e3d497c71522',execution_commit=args.commit,
        conditions=['B0_W0','B1_APR_FULL','B2_APR_DAMPED','B3_APR_TR'],eta=.10,
        multiplier_bracket_limit=64,multiplier_bisection_limit=64,norm_relative_slack=1e-5,
        gpu_indices=[0],gpu_uuids={'0':cfg['gpu_uuids']['0']},wall_hours_limit=4,gpu_hours_limit=8,
        maximum_new_endpoints=46,target_propagation='NA; no dedicated scoring protocol',
        stage8_formula='delta(mu)=-(A0 N) solve(RtR+(lambda0+mu)I,Rt); bound effective B0 delta Frobenius norm',
        historical_fields='Other numerical/checkpoint settings inherited from Stage7; preparation_commit refers to Stage7')
    vf.atomic_json(run/'public/RUN_LOCK.json',lock)
    vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='PREPARED',publication='PENDING'))


def worker(args):
    run=args.run_root;cfg=read(run/'private/CAMPAIGN_CONFIG.json');oldrun=Path(cfg['stage7_run'])
    runtime=vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['runtime']['cpu_gate'])))
    def budget():
        if time.time()-cfg['campaign_epoch']>=3*3600 or (run/'STOP').exists():raise TimeoutError('generation deadline')
    for task in read(run/'private/TASKS.json'):
        order=task['order']
        if (args.part=='first')!=(order==0):continue
        directory=run/f'private/edits/e{order:02d}';source=oldrun/f'private/edits/e{order:02d}'
        if (directory/'result.json').exists():continue
        budget();start=time.time();old=read(source/'result.json');assert old['status']=='COMPLETE'
        assert s.file_identity(task['checkpoint'])==task['checkpoint_identity']
        state=torch.load(task['checkpoint'],map_location='cpu',weights_only=True)
        cp=vf.AsymmetricCPExpert(14336,4096,4).to(runtime.device);cp.load_state_dict(state['expert']);cp.requires_grad_(False)
        assert runtime.base_guard.verify()['after_sha256']==old['base_guard']['after_sha256']
        cache=torch.load(source/'ACTIVATIONS_PRIVATE.pt',map_location='cpu',weights_only=True)
        saved=torch.load(source/'REPAIRS_PRIVATE.pt',map_location='cpu',weights_only=True)
        assert cache['checkpoint']==task['checkpoint_identity']
        rows={r['logical_id']:r for r in task['data']['rows']}
        ids=task['positive_ids']+[i for group in task['negative_ids'].values() for i in group]
        assert len(cache['bindings'])==len(ids)
        for binding,lid in zip(cache['bindings'],ids):
            assert binding['row']==rows[lid] and binding['module']==vf.LAYER
            assert binding['generation']==runtime.generation_config and binding['normalization_epsilon']==cp.epsilon
            batch=s.input_batch(runtime,rows[lid])
            assert binding['prompt_token_ids']==batch.raw_input_ids.tolist() and binding['image_identity']==batch.image_sha256
            assert binding['attention']==(batch.attention_mask.tolist() if batch.attention_mask is not None else None)
        A=cp.input_basis().T.detach().cpu();B=(cp.output_basis()*cp.rho*(cp.beta/math.sqrt(cp.rank))).detach().cpu()
        torch.testing.assert_close(A,saved['experts']['C0']['A'],rtol=0,atol=0)
        matrices,geometry=bounded_repair(A,B,cache['X'],cache['N'],saved['experts']['C2']['A'],[rows[i] for i in ids])
        print('FIT',order,{n:g['relative_repair_norm'] for n,g in geometry.items()},flush=True)
        experts={n:s.ExpandedExpert(cp,V.to(runtime.device)).to(runtime.device) for n,V in matrices.items()}
        s.sw.save(directory/'REPAIRS_PRIVATE.pt',{n:e.state_dict() for n,e in experts.items()})
        reload=torch.load(directory/'REPAIRS_PRIVATE.pt',map_location=runtime.device,weights_only=True)
        for n,e in experts.items():e.load_state_dict(reload[n])
        entries=[];replays=[]
        for entry in old['entries']:
            if entry['method'] not in ('C0','C2','HISTORICAL_W1','C1'):continue
            item=entry['item'];row=item['row'];assert row==rows[row['logical_id']]
            reused=copy.deepcopy(entry);reused['method']={'C0':'B0','C2':'B1','C1':'HISTORICAL_C1'}.get(entry['method'],entry['method'])
            entries.append(reused)
            if entry['method']!='C0':continue
            budget();s.input_batch(runtime,row)
            for name,expert in experts.items():
                # Only bitwise identical deployed state permits full-output reuse.
                identical=all(torch.equal(v.detach().cpu(),saved['experts']['C2'][k]) for k,v in expert.state_dict().items())
                control=next(e for e in old['entries'] if e['method']=='C2' and e['item']['row']['logical_id']==row['logical_id'])
                forced=control['item']['forced'] if identical else s.sw.generated(runtime,row,expert)
                fixed=forced if item['fixed_on'] else item['base']
                if order==0 and not any(r['method']==name and r['on']==item['fixed_on'] for r in replays):
                    observed=s.sw.generated(runtime,row,expert if item['fixed_on'] else None)
                    assert s.same_output(observed,fixed)
                    replays.append(dict(method=name,on=item['fixed_on'],status='PASSED'))
                new=copy.deepcopy(entry);new.update(method=name,item=dict(item,forced=forced,fixed=fixed),exact_apr_reuse=identical)
                entries.append(new)
        guard=runtime.base_guard.verify();assert guard['unchanged']
        vf.atomic_json(directory/'result.json',dict(status='COMPLETE',entries=entries,geometry=geometry,
            elapsed_seconds=time.time()-start,replays=replays,base_guard=guard))
        print('DONE',order,flush=True)


def inventory(run):
    return s.inventory(run)


def install(run):
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');old=Path(cfg['stage7_run']);s.install(old)
    verdicts,side=f4.f3.current_verdicts(old);assert not set(side['all_expected'])-verdicts.keys()
    protocol=read(Path(cfg['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json')
    pool,execution=s.stage4_pool(cfg,protocol)
    identity=read(old/'private/judge/REUSE_EXECUTION_IDENTITY_PRIVATE.json')
    tuples=f4.f3.tuples_for(s.inventory(old)[0],protocol['config_sha256'])
    for key,value in verdicts.items():pool[key]=(tuples[key],value,identity)
    f4.f3.inventory=inventory;f4.f3.historical_judge=lambda *_:(pool,execution)


def report(args):
    run=args.run_root;install(run);entries,ledger,cost=inventory(run)
    verdicts,side=f4.f3.current_verdicts(run);rows=f4.details(entries,verdicts,side['protocol_sha256'])
    for r in rows:
        if r['mode']=='R0':r['mode']='RC_FIXED_OLD16'
    tables=[r for r in f4.summarize(rows) if r['cohort'].startswith('APR_')]
    for t in tables:
        group=[r for r in rows if all(r[k]==t[k] for k in ('cohort','method','mode','role','stratum','strict_role'))]
        t['unique_eqkeys']=len({r['eqkey'] for r in group});t['patient']='UNKNOWN';t['target_propagation']='NA'
    f4.csv_write(run/'public/BOUNDED_APR_RESULTS.csv',tables)
    costs=[]
    for task in read(run/'private/TASKS.json'):
        path=run/f"private/edits/e{task['order']:02d}/result.json"
        if path.exists():
            for name,g in read(path).get('geometry',{}).items():costs.append(dict(cohort=task['cohort'],edit=task['event_index'],condition=name,**g))
    f4.csv_write(run/'public/REPAIR_GEOMETRY.csv',costs)
    effects=[]
    for candidate,control in (('B2','B1'),('B3','B2'),('B3','B0')):
        for mode in ('FORCED_ON','RC_FIXED_OLD16'):
            sub=[dict(r,method='W1' if r['method']==candidate else 'W0',mode='DEV_THRESHOLD_TRANSFER_DIAGNOSTIC') for r in rows if r['method'] in (candidate,control) and r['mode']==mode]
            effects.extend(dict(e,candidate=candidate,control=control,mode=mode,control_mode=mode) for e in pairs(sub))
    f4.csv_write(run/'public/PAIRED_EFFECTS.csv',effects)
    from collections import defaultdict
    baseline={(r['cohort'],r['mode'],r['edit'],r['eqkey']):r for r in rows if r['method']=='B0'}
    changes=defaultdict(list)
    for r in rows:
        if r['method'] not in ('B1','B2','B3'):continue
        b=baseline.get((r['cohort'],r['mode'],r['edit'],r['eqkey']))
        if not b or r['semantic'] is None or b['semantic'] is None:continue
        changes[r['cohort'],r['method'],r['mode'],r['role'],r['stratum']].append(dict(
            lost_correction=int(b['base_correct']==0 and b['semantic']==1 and r['semantic']==0),
            gained_correction=int(b['base_correct']==0 and b['semantic']==0 and r['semantic']==1),
            avoided_damage=int(b['base_correct']==1 and b['semantic']==0 and r['semantic']==1),
            introduced_damage=int(b['base_correct']==1 and b['semantic']==1 and r['semantic']==0)))
    f4.csv_write(run/'public/CORRECTION_TRADEOFF_COUNTS.csv',[dict(zip(('cohort','condition','mode','role','panel'),key),inputs=len(g),**{k:sum(r[k] for r in g) for k in g[0]}) for key,g in changes.items()])
    diagnostics=[]
    selected=[r for r in rows if r['method']=='B0' and r['mode']=='FORCED_ON' and r['stratum']=='T2G' and r['role']=='formal_development']
    for index,b in enumerate(selected):
        group=[r for r in rows if r['edit']==b['edit'] and r['cohort']==b['cohort'] and r['eqkey']==b['eqkey'] and r['mode']=='FORCED_ON']
        found={e['method']:e['item']['forced']['raw_token_ids'] for e in entries if e['edit']==b['edit'] and e['item']['row']['eqkey']==b['eqkey']}
        for method in ('B0','B1','B2','B3'):
            tokens=found[method];base=found['B0'];div=next((i for i,(x,y) in enumerate(zip(base,tokens)) if x!=y),min(len(base),len(tokens)) if len(base)!=len(tokens) else None)
            diagnostics.append(dict(probe=index,edit=b['edit'],condition=method,semantic=next(r['semantic'] for r in group if r['method']==method),
                first_token_divergence=div,activation_perp_ratio='NA',effective_perturbation='NA',logit_margin='NA',
                na_reason='Stage7 cached fit trajectories only; no common-prefix evaluation activations/logits; no extra diagnostic GPU replay'))
    assert len(selected)==23 and len(diagnostics)==92,'T2G diagnostic coverage'
    f4.csv_write(run/'public/T2G_READONLY_DIAGNOSTICS.csv',diagnostics)
    vf.atomic_json(run/'private/DETAILS.json',rows)
    missing=len(set(side['all_expected'])-verdicts.keys())
    status=dict(status='COMPUTE_COMPLETE' if len(ledger)==23 and all(t['status']=='COMPLETE' for t in ledger) and not missing else 'PARTIAL',
        coverage=ledger,judge_required=len(side['all_expected']),judge_scored=len(verdicts),judge_missing=missing,
        judge_new=side['new'],judge_reused=side['reused'],new_sgd_training=0,publication='PENDING')
    vf.atomic_json(run/'public/RUN_STATUS.json',status)
    lines=['# Stage8 bounded APR closeout','',str(status),'','Viewed development only. Fixed eta=.10; no new SGD or router. Full source/role-separated results and paired sensitivity are attached.',
        'Target propagation NA. T2G token divergence is observed; missing activation/logit diagnostics are NA, not inferred causes.',
        'Historical C1/W1 are reused, not equal-budget new controls. Fixed-RC all-OFF results cannot establish writer benefit.',
        'Rank4 expanded input:57860 scalars/231440 FP32 bytes per expert, not original1476 CP parameters.',
        '','|Cohort|Method|Role|Panel|Correct macro|Damage macro|','|---|---|---|---|---:|---:|']
    for t in tables:
        if t['average']=='macro' and t['mode']=='FORCED_ON':lines.append('|'+ '|'.join(str(t[k]) for k in ('cohort','method','role','stratum','semantic','base_correct_damage'))+'|')
    lines+=['','No automatic next experiment. Comparative research decision requires reading the three locked paired comparisons; no SOTA, clinical, independent-test or originality claim. Standard trust-region/constraint algebra is not itself novel; acknowledge TIME, M-ORE, AlphaEdit and ROME.']
    vf.atomic_text(run/'public/GPT_PRO_REVIEW.md','\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','worker','prepare-judge','report'))
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--stage7-run',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--commit');p.add_argument('--part',choices=('first','rest'))
    a=p.parse_args()
    if a.action=='prepare':prepare(a)
    elif a.action=='worker':worker(a)
    elif a.action=='report':report(a)
    else:
        install(a.run_root);side=f4.f3.prepare_judge(a);assert not side['execution_version_changed']
