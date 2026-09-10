#!/usr/bin/env python3
"""Joint fact writer comparison: frozen development support and paired outcomes."""
import argparse
from collections import Counter,defaultdict
import copy
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace import stage10 as previous,stage9
from scripts.medtrace.stage11_worker import worker,CONDITIONS
vf,read,f4=previous.vf,previous.read,previous.f4


def scope(row):
    q=row['question'].lower()
    if 'weighting' in q:return 'ACQUISITION_WEIGHTING'
    if 'part of the body' in q:return 'BODY_REGION'
    if previous.anatomy(row):return 'ORGAN_VISIBILITY'
    return 'OTHER_SUPPORTED_ATTRIBUTE'


def prepare(args):
    run=args.run_root;old=args.stage10_run;cfg=read(old/'private/CAMPAIGN_CONFIG.json')
    assert read(old/'RUN_COMPLETION.json')['judge_missing']==0
    if run.exists():raise FileExistsError('preserve existing Stage11 run')
    assert shutil.disk_usage(run.parent).free>30*1024**3
    run.mkdir();p=run/'.probe';p.write_text('joint');assert p.read_text()=='joint';p.unlink()
    cfg.update(kind='MEDTRACE_STAGE11',stage10_run=str(old),anatomical_evidence=False,campaign_epoch=time.time(),code_commit=args.commit,
        worker_gpus=[0,1],judge_gpu=0,allowed_physical_gpus=[0,1],wall_hours=8,gpu_hours=16,authorization='USER_STAGE11_GPU0_GPU1')
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json',cfg);vf.atomic_json(run/'private/CAMPAIGN_START.json',dict(epoch=cfg['campaign_epoch']))
    vf.atomic_text(run/'private/PROTOCOL.md',args.protocol.read_text())
    tasks=[t for t in read(Path(cfg['stage9_run'])/'private/TASKS.json') if t['stage9_status']=='PENDING'];assert len(tasks)==15
    sources={};audit=[]
    for t in tasks:
        rows=t['data']['rows'];byid={r['logical_id']:r for r in rows};bad=set();inputs=defaultdict(list)
        record=t['data']['event']['edit_record'];target=vf.EditorRecord.from_dict(record).target
        for row in rows:
            inputs[(str(Path(row['image_path'])), ' '.join(row['question'].lower().split()))].append(row)
        overlap=0
        for group in inputs.values():
            overlap+=len({r['role'] for r in group})>1
            labels={(' '.join((target if r['label']=='positive' else r['reference']).lower().split())) for r in group}
            if len(labels)>1:bad.update(r['logical_id'] for r in group)
        for lid in [t['native_id']]+t['h_ids']:
            row=byid[lid];path=row['source_file']
            if path not in sources:sources[path]=read(Path(path))
            ok=any(str(r.get('qid'))==str(row['source_qid']) and r.get('question')==row['question'] and r.get('answer')==row['reference'] and row['image_path'].endswith('/'+r.get('img_name','INVALID')) for r in sources[path])
            if not ok:bad.add(lid)
        t['quarantined_ids']=sorted(bad)
        for key in ('h_ids','u_ids','fit_positive_ids'):t[key]=[i for i in t[key] if i not in bad]
        t['stage11_status']='PENDING' if t['native_id'] not in bad and t['h_ids'] and t['u_ids'] and t['fit_positive_ids'] else 'QUARANTINED_SOURCE_CONTRADICTION'
        native=byid[t['native_id']];t['medical_scope']=scope(native)
        audit.append(dict(edit=t['event_index'],status=t['stage11_status'],medical_scope=t['medical_scope'],rows=len(rows),
            unique_inputs=len(inputs),unique_questions=len({r['question'] for r in rows}),images=len({r['image_path'] for r in rows}),
            source_groups=len({r.get('source_group') for r in rows}),cross_role_inputs=overlap,quarantined_records=len(bad),
            selected_H=len(t['h_ids']),selected_U=len(t['u_ids']),patients='UNKNOWN',roi_subset=t['event_index'] in (102,107,109,110,113)))
    vf.atomic_json(run/'private/TASKS.json',tasks);vf.atomic_json(run/'public/COVERAGE_LEDGER.json',audit)
    vf.atomic_json(run/'public/WRITER_AND_RUN_LOCK.json',dict(execution_sha=args.commit,stage10_public_sha='b84a0f35c6d8b6b4f8f8c475e48ef1e3ab3d2ef3',
        conditions=CONDITIONS,initial_function='original per-edit CP W0 step320; lossless free-rank conversion',steps=[160,320],primary_step=320,
        J0_reuse='NO strict continuation: historical checkpoint has optimizer but no RNG states; restart320 from W0',maximum_new_optimizer_steps=14400,
        loss='0.5 native CE + 0.5 original fit CE + H source CE + 0.01 full-vocabulary token-mean Base||student KL',
        schedule_seed=20260910,seed='original per-edit W0 seed',optimizer=dict(name='Adam',input_lr=1e-4,output_lr=1e-3,betas=[.9,.999],eps=1e-8,weight_decay=0,clip_grad_norm=1),
        normalization='CP factor norms absorbed by rho; free A row norms absorbed by B; not equal optimizer geometry',
        anticipated_parameters=[1476,73728,294912],actual_parameters='per-endpoint COSTS.csv',layer=vf.LAYER,worker_gpus=[0,1],judge_gpu=0,
        wall_hours=8,gpu_hours=16,training_generation_deadline_hours=6.5,publication='PENDING',research_result_sha='PENDING',
        exposure='viewed development; patient UNKNOWN; held-out H within trained edits, not independent confirmation',
        official_T2G='NA unless original bound row present; constructed SLAKE paraphrases are not official T2G',
        first_answer_token_rank='NA; do not rerun for auxiliary diagnostic',RC='read-only inherited fixed routing; no calibration',
        scope='No ROI loss, contrast, APR, new facts, new ranks or automatic next stage'))
    for name in ('EVIDENCE_PAIRED_RESULTS.csv','PAIR_CORRECT_RESULTS.csv'):
        shutil.copyfile(old/'public'/name,run/'public'/('STAGE10_'+name))
    vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='PREPARED',planned_edits=15,supported_edits=sum(t['stage11_status']=='PENDING' for t in tasks),publication='PENDING'))
    print('SUPPORTED',sum(t['stage11_status']=='PENDING' for t in tasks),flush=True)


def inventory(run):
    entries=[];ledger=[];curves=[]
    for t in read(run/'private/TASKS.json'):
        for condition in CONDITIONS:
            d=run/('private/edits/e%02d'%t['order'])/condition
            final=read(d/'result.json') if (d/'result.json').exists() else {}
            for step in (160,320):
                path=d/('endpoint%04d.json'%step);result=read(path) if path.exists() else dict(status=t['stage11_status'],entries=[])
                entries+=result['entries'];ledger.append(dict(edit=t['event_index'],condition=condition,step=step,status=result['status'],final_metadata=bool(final)))
            curves.extend(dict(edit=t['event_index'],condition=condition,**{k:v for k,v in c.items() if not k.endswith('_id')}) for c in final.get('curve',[]))
    return entries,ledger,curves


def install(run):
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');old=Path(cfg['stage10_run']);previous.install(old)
    verdicts,side=f4.f3.current_verdicts(old);assert not set(side['all_expected'])-verdicts.keys()
    protocol=read(Path(cfg['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json');pool,execution=f4.f3.historical_judge(cfg,protocol);pool=pool.copy()
    identity=read(old/'private/judge/REUSE_EXECUTION_IDENTITY_PRIVATE.json');tuples=f4.f3.tuples_for(previous.inventory(old)[0],protocol['config_sha256'])
    for key,value in verdicts.items():pool[key]=(tuples[key],value,identity)
    f4.f3.inventory=inventory;f4.f3.historical_judge=lambda *_:(pool,execution)


def pair_outcomes(details,tasks):
    values={(r['method'],r['diagnostic_step'],r['edit'],r['eqkey']):r['semantic'] for r in details if r['mode']=='FORCED_ON'}
    outcomes=[]
    for t in tasks:
        native=next(r for r in t['data']['rows'] if r['role']=='native')
        for condition in CONDITIONS:
            for step in (160,320):
                for role in ('fit','evaluation'):
                    hs=[r for r in t['data']['rows'] if r['role']==role and r.get('negative_group')=='H' and r['question']==native['question'] and r.get('conflict_verified') and r.get('relation_evidence') and r['logical_id'] not in t['quarantined_ids']]
                    if role=='fit':hs=[r for r in hs if r['logical_id'] in t['h_ids']]
                    hs=list({r['eqkey']:r for r in hs}.values());counts=Counter();used=[]
                    for h in hs:
                        a=values.get((condition,step,t['event_index'],native['eqkey']));b=values.get((condition,step,t['event_index'],h['eqkey']))
                        if a is None or b is None:continue
                        counts['both_correct' if a==1 and b==1 else 'native_only' if a==1 else 'H_only' if b==1 else 'both_wrong']+=1;used.append(h)
                    if used:
                        outcomes.append(dict(edit=t['event_index'],condition=condition,step=step,panel='FIT_PairCorrect' if role=='fit' else 'EVAL_PairCorrect',
                            medical_scope=t['medical_scope'],roi_subset=t['event_index'] in (102,107,109,110,113),pairs=len(used),
                            **{k:counts[k] for k in ('both_correct','native_only','H_only','both_wrong')},pair_correct=counts['both_correct']/len(used),
                            unique_inputs=len({native['eqkey']}|{r['eqkey'] for r in used}),unique_images=len({native['image_path']}|{r['image_path'] for r in used}),patients='UNKNOWN'))
    return outcomes


def report(args):
    run=args.run_root;install(run);entries,ledger,curves=inventory(run);verdicts,side=f4.f3.current_verdicts(run)
    details=f4.details(entries,verdicts,side['protocol_sha256']);tasks=read(run/'private/TASKS.json');byedit={t['event_index']:t for t in tasks}
    tables=[]
    for stratum in ['ALL','ROI5']+sorted({t['medical_scope'] for t in tasks}):
        selected=[r for r in details if stratum=='ALL' or (stratum=='ROI5' and r['edit'] in (102,107,109,110,113)) or byedit[r['edit']]['medical_scope']==stratum]
        tables.extend(dict(r,medical_stratum=stratum) for r in f4.summarize(selected))
    pairs=pair_outcomes(details,tasks);f4.csv_write(run/'public/PAIR_OUTCOMES.csv',pairs)
    f4.csv_write(run/'public/JOINT_FACT_RESULTS.csv',tables);f4.csv_write(run/'public/TRAINING_CURVES.csv',curves)
    vf.atomic_json(run/'private/DETAILS.json',details);vf.atomic_json(run/'public/COMPLETION_LEDGER.json',ledger)
    costs=[]
    for t in tasks:
        for condition in CONDITIONS:
            p=run/('private/edits/e%02d'%t['order'])/condition/'result.json'
            if p.exists():costs.append(dict(edit=t['event_index'],condition=condition,**{k:v for k,v in read(p).items() if k not in ('curve','initial_check')}))
    f4.csv_write(run/'public/COSTS.csv',costs)
    missing=len(set(side['all_expected'])-verdicts.keys());complete=sum(r['status']=='COMPLETE' for r in ledger)
    supported=sum(t['stage11_status']=='PENDING' for t in tasks)
    status=dict(status='COMPUTE_COMPLETE_SUPPORTED_SUBSET' if complete==supported*6 and len(costs)==supported*3 and not missing else 'PARTIAL',
        planned_final_endpoints=45,supported_final_endpoints=supported*3,completed_final_endpoints=len(costs),completed_step_endpoints=complete,
        judge_required=len(side['all_expected']),judge_scored=len(verdicts),judge_missing=missing,judge_reused=side['reused'],judge_new=side['new'],publication='PENDING')
    vf.atomic_json(run/'public/RUN_STATUS.json',status)
    lines=['# Stage11 joint fact writer comparison','',str(status),'',
        'Viewed development only. Compare step320 as primary and step160 as fixed interim. J0 reran from W0 because historical RNG state was absent; all objectives and schedules match. No automatic next experiment.',
        'FIT pairs use actually selected H-fit; EVAL pairs use held-out H within trained edits, not unseen patients. Missing exact-question pairs only restrict pair denominators. Patient UNKNOWN. First-answer-token rank NA. See stratified tables and costs.',
        '', '|Writer|Step|Pair panel|Both correct / pairs|Pair micro|Edit macro|','|---|---:|---|---:|---:|---:|']
    for condition in CONDITIONS:
        for step in (160,320):
            for panel in ('FIT_PairCorrect','EVAL_PairCorrect'):
                rs=[r for r in pairs if r['condition']==condition and r['step']==step and r['panel']==panel]
                n=sum(r['pairs'] for r in rs);yes=sum(r['both_correct'] for r in rs)
                lines.append(f'|{condition}|{step}|{panel}|{yes}/{n}|{yes/n if n else None}|{sum(r["pair_correct"] for r in rs)/len(rs) if rs else None}|')
    lines+=['','|Writer|Step|Role|Panel|Correct macro|Base-correct damage|','|---|---:|---|---|---:|---:|']
    for r in tables:
        if r['medical_stratum']=='ALL' and r['average']=='macro' and r['mode']=='FORCED_ON':lines.append('|'+ '|'.join(str(r[k]) for k in ('method','diagnostic_step','role','stratum','semantic','base_correct_damage'))+'|')
    lines+=['','Interpretation and fixed-writer recommendation pending joint review of FIT/EVAL pairs, native/cross-family preservation and U damage; completion alone is not effectiveness. No clinical or original-mechanism claim.']
    vf.atomic_text(run/'public/GPT_PRO_REVIEW.md','\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','worker','prepare-judge','report'));p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--stage10-run',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--commit');p.add_argument('--part',choices=('0','1'));p.add_argument('--parts',type=int,default=2)
    a=p.parse_args()
    if a.action=='prepare':prepare(a)
    elif a.action=='worker':worker(a)
    elif a.action=='report':report(a)
    else:
        install(a.run_root);side=f4.f3.prepare_judge(a);assert not side['execution_version_changed']
