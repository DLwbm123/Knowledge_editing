#!/usr/bin/env python3
"""Minimal official-anatomy evidence comparison and existing paired closeout."""
import argparse
from collections import Counter,defaultdict
import copy
from pathlib import Path
import random
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from transformers import CLIPImageProcessor
from scripts.medtrace import stage9 as previous,stage7 as s
from methods.medtrace.anatomical_evidence import prepare_roi,anatomy,OFFICIAL_MAPPING_URL,suppressed,pixel_view
from methods.medtrace.fact_contrast import answer_score
from dataclasses import replace
from scripts.medtrace.stage5_existing import pairs
vf,read,f4=s.vf,s.read,s.f4


def scope(row):
    if anatomy(row):return 'LOCAL_ANATOMICAL_VISIBILITY'
    if 'largest organ' in row['question']:return 'COMPARATIVE_ANATOMY'
    if 'weighting' in row['question'] or 'part of the body' in row['question']:return 'GLOBAL_ACQUISITION_OR_REGION'
    return 'OTHER'


def pair_rows(entries,details,tasks,methods):
    values={(r['method'],r['edit'],r['eqkey']):r['semantic'] for r in details if r['mode']=='FORCED_ON'}
    byentry={(e['method'],e['edit'],e['item']['row']['logical_id']):e for e in entries}
    out=[]
    for t in tasks:
        native=next(r for r in t['data']['rows'] if r['role']=='native')
        h=[r for r in t['data']['rows'] if r['role']=='evaluation' and r.get('negative_group')=='H' and r['question']==native['question'] and r.get('conflict_verified') and r.get('relation_evidence')]
        for method in methods:
            groups=defaultdict(list)
            for r in h:
                a=values.get((method,t['event_index'],native['eqkey']));b=values.get((method,t['event_index'],r['eqkey']))
                if a is None or b is None:continue
                groups[r['source_group']].append(int(a==1 and b==1))
            if groups:out.append(dict(edit=t['event_index'],condition=method,scope=scope(native),pairs=sum(map(len,groups.values())),
                correct_pairs=sum(sum(g) for g in groups.values()),source_groups=len(groups),
                pair_correct=sum(sum(g)/len(g) for g in groups.values())/len(groups),patients='UNKNOWN'))
    return out


def prepare(args):
    run=args.run_root;old=args.stage9_run
    if run.exists():raise FileExistsError('preserve Stage10 run')
    assert read(old/'RUN_COMPLETION.json')['complete_trajectories']==45
    assert shutil.disk_usage(run.parent).free>20*1024**3
    run.mkdir();p=run/'.probe';p.write_text('evidence');assert p.read_text()=='evidence';p.unlink()
    cfg=read(old/'private/CAMPAIGN_CONFIG.json');cfg.update(kind='MEDTRACE_STAGE10',stage9_run=str(old),
        anatomical_evidence=True,campaign_epoch=time.time(),code_commit=args.commit,wall_hours=8,gpu_hours=12)
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json',cfg);vf.atomic_json(run/'private/CAMPAIGN_START.json',dict(epoch=cfg['campaign_epoch']))
    vf.atomic_text(run/'private/PROTOCOL.md',args.protocol.read_text())
    tasks=[t for t in read(old/'private/TASKS.json') if t['stage9_status']=='PENDING'];assert len(tasks)==15
    processor=CLIPImageProcessor.from_pretrained(args.processor,local_files_only=True)
    support=[]
    for t in tasks:
        rows={r['logical_id']:r for r in t['data']['rows']};instances=[];audit=[]
        for lid in [t['native_id']]+t['h_ids']:
            row=rows[lid];assert row['role'] in ('native','fit')
            instance,reason=prepare_roi(row,processor)
            item=dict(role=row['role'],scope=scope(row),anatomy=anatomy(row),state=row['reference'] if anatomy(row) else 'SOURCE_BOUND_PRIVATE',
                laterality='UNKNOWN',time='UNKNOWN',status=reason or 'SUPPORTED',source_identity=row['eqkey'])
            if instance is not None:
                if Path(row['image_path']).parent.name not in ('xmlab26','xmlab8','xmlab16'):raise ValueError('source image text-overlay inspection missing')
                instance['image_sha256']=vf.sha256_file(Path(row['image_path']));instance['annotation_sha256']=vf.sha256_file(Path(instance['annotation']))
                item.update(annotation_sha256=instance['annotation_sha256'],image_sha256=instance['image_sha256'],
                    region_type='OFFICIAL_SEGMENTATION',class_mapping=OFFICIAL_MAPPING_URL,pixels_R=instance['pixels_R'],pixels_C=instance['pixels_C'],
                    tissue_R=instance['tissue_R'],tissue_C=instance['tissue_C'],nonpadding_fraction=1.,geometry=instance['geometry'],
                    diagnostic_control_new=instance['diagnostic_control_new'])
                instances.append(instance)
            audit.append(item)
        t['stage9_status']='PENDING' if instances else 'EVIDENCE_UNSUPPORTED'
        t['evidence_path']=str(run/f"private/evidence/e{t['order']:02d}.pt")
        diagnostic=[];diagnostic_missing=Counter()
        for row in t['data']['rows']:
            if row['role']!='evaluation' or not anatomy(row):continue
            if Path(row['image_path']).parent.name not in ('xmlab26','xmlab8','xmlab16'):
                diagnostic_missing['NO_SELECTED_IMAGE_OVERLAY_CHECK']+=1;continue
            instance,reason=prepare_roi(row,processor)
            if instance is None:diagnostic_missing[reason]+=1
            else:diagnostic.append(instance)
        t['diagnostic_path']=str(run/f"private/score_only/e{t['order']:02d}.pt")
        s.sw.save(Path(t['diagnostic_path']),diagnostic)
        if instances:
            s.sw.save(Path(t['evidence_path']),instances)
            rng=random.Random(20260911);sequence=[i%len(instances) for i in range(160)];rng.shuffle(sequence)
            signs={i:[1,-1]*80 for i in range(len(instances))}
            # Exact balance over each instance's actual occurrence count.
            for i in signs:
                count=sequence.count(i);signs[i]=([1,-1]*(count//2)+([1] if count%2 else []));rng.shuffle(signs[i])
            cursors=Counter();schedule=[]
            for i in sequence:
                j=cursors[i];schedule.append([i,j%len(instances[i]['controls']),signs[i][j]]);cursors[i]+=1
            t['evidence_schedule']=schedule
        support.append(dict(edit=t['event_index'],status=t['stage9_status'],evidence_instances=len(instances),records=audit,
            evaluation_diagnostic_instances=len(diagnostic),evaluation_diagnostic_missing=dict(diagnostic_missing)))
    vf.atomic_json(run/'private/TASKS.json',tasks)
    vf.atomic_json(run/'public/ANATOMICAL_EVIDENCE_SUPPORT_PUBLIC.json',dict(planned_edits=15,supported_edits=sum(t['stage9_status']=='PENDING' for t in tasks),tasks=support,
        official_url='https://www.med-vqa.com/slake/',mapping_url=OFFICIAL_MAPPING_URL,license='CC-BY-4.0 per author HF dataset card; no raw assets published',
        version='existing frozen image/QA unchanged; author mapping used only for existing local mask IDs',patient='UNKNOWN'))
    vf.atomic_json(run/'public/METHOD_AND_RUN_LOCK.json',dict(source_commit=args.commit,stage9_public_commit='4cfe33ecc81d91127b96aeeeef4c0b6d6ed15edd',
        conditions=['E0_SOURCE_CE_REUSED','E1_TRUE_ANATOMICAL_EVIDENCE','E2_RANDOMIZED_RELATION_CONTROL'],steps=160,
        common_objective='unchanged Stage9 F1; independent original W0 clones',loss='Lsup+.1*(CE(without_C)+relu(.2-sigma*(score_C-score_R)))',
        E1_sigma=1,E2_sigma='per-instance balanced presaved random sequence',schedule_seed=20260911,base_schedule_seed=20260910,
        fill='full preprocessed image per-channel spatial mean; same all-channel pixel locations',
        ROI_geometry='same square-pad CLIP resize/center crop; nearest-neighbor masks; non-square images rejected',
        control='same-shape translations, grid16, disjoint, >=80 percent nonblack (>0.05) pixels; no observed text overlays in inspected sources; <=4 training controls',
        inference='unchanged full image, mask-free FORCED_ON',rank=4,parameters=1476,worker_gpus=[2,3],judge_gpu=0,wall_hours=8,gpu_hours=12,
        source_assets='private images/QA/masks/checkpoints; only permitted hashes and counts public',publication='PENDING'))
    # Frozen cached Stage9 outputs: no new inference or selection by success.
    entries,_,_=previous.inventory(old);details=read(old/'private/DETAILS.json')
    table=pair_rows(entries,details,tasks,('F0','F1','F2'))
    f4.csv_write(run/'public/STAGE9_PAIR_CORRECT_RESULTS.csv',table)
    failures=[]
    for t in tasks:
        native=next(r for r in t['data']['rows'] if r['role']=='native')
        for method in ('F1','F2'):
            score=next(r['semantic'] for r in details if r['method']==method and r['mode']=='FORCED_ON' and r['edit']==t['event_index'] and r['role']=='native')
            if score==1:continue
            answer=next(e['item']['forced']['raw_answer'] for e in entries if e['method']==method and e['edit']==t['event_index'] and e['item']['row']['role']=='native')
            hanswers={r['reference'].strip().lower() for r in t['data']['rows'] if r['logical_id'] in t['h_ids']}
            failures.append(dict(edit=t['event_index'],condition=method,scope=scope(native),classification='EXACT_H_SOURCE_ANSWER_MATCH' if answer.strip().lower() in hanswers else 'UNKNOWN_NOT_EXACT_H_SOURCE_MATCH',
                interpretation='output-level match only; not a causal explanation',source_groups=len(t['h_ids'])))
    f4.csv_write(run/'public/STAGE9_NATIVE_FAILURE_BREAKDOWN.csv',failures)
    vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='PREPARED',supported_edits=sum(t['stage9_status']=='PENDING' for t in tasks),publication='PENDING'))
    print('E_SUPPORTED',sum(t['stage9_status']=='PENDING' for t in tasks),flush=True)


def worker(args):previous.worker(args)


def diagnostics(runtime,record,task,expert,directory):
    """Frozen endpoint only; evaluation pixels cannot enter the fitter."""
    if any(p.requires_grad for p in expert.parameters()):raise ValueError('diagnostics require frozen parameters')
    private=[];public=[]
    for index,entry in enumerate(torch.load(task['diagnostic_path'],map_location='cpu',weights_only=True)):
        row=entry['row'];assert row['role']=='evaluation';scores={};outputs={}
        for name,mask in (('full',torch.zeros_like(entry['region'])),('without_C',entry['diagnostic_control']),('without_R',entry['region'])):
            hook=vf.MedTraceLayerHook(runtime.get_module(vf.LAYER),expert);hook.attach()
            try:
                with torch.no_grad(),pixel_view(runtime.adapter,row,suppressed(entry['pixels'],mask),entry['pixels']):
                    batch=runtime.build_edit_batch(replace(record,question=row['question'],image_path=Path(row['image_path']),target=row['reference']))
                    hook.set_teacher_routing(batch.labels);out=runtime.model(**batch.forward_kwargs())
                    scores[name]=float(answer_score(out.logits,batch.labels,batch.attention_mask,runtime.adapter.tokenizer.eos_token_id,runtime.adapter.tokenizer.bos_token_id,runtime.adapter.tokenizer.pad_token_id))
                    hook.clear_request_routing();outputs[name]=vf.scope_generate(runtime,row,hook)
            finally:hook.detach()
        public.append(dict(edit=task['event_index'],diagnostic=index,anatomy=entry['anatomy'],role='evaluation',**scores,
            delta_R=scores['full']-scores['without_R'],delta_C=scores['full']-scores['without_C'],difference=scores['without_C']-scores['without_R'],
            control_position_new=entry['diagnostic_control_new'],without_C_token_changed=not s.same_output(outputs['full'],outputs['without_C']),
            without_R_token_changed=not s.same_output(outputs['full'],outputs['without_R'])))
        private.append(dict(row=row,scores=scores,outputs=outputs))
    vf.atomic_json(directory/'EVIDENCE_DIAGNOSTICS_PRIVATE.json',private)
    vf.atomic_json(directory/'EVIDENCE_DIAGNOSTICS_AGGREGATE.json',public)


def inventory(run):
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');entries=[];ledger=[];curves=[]
    for t in read(run/'private/TASKS.json'):
        for name in ('E1','E2'):
            path=run/f"private/edits/e{t['order']:02d}/{name}/result.json"
            r=read(path) if path.exists() else dict(status=t['stage9_status'],entries=[])
            entries+=r['entries'];ledger.append(dict(edit=t['event_index'],condition=name,status=r['status']))
            curves.extend(dict(edit=t['event_index'],condition=name,**{k:v for k,v in c.items() if not k.endswith('_id')}) for c in r.get('curve',[]))
        if t['stage9_status']=='PENDING':
            r=read(Path(cfg['stage9_run'])/f"private/edits/e{t['order']:02d}/F1/result.json")
            entries.extend(dict(e,method='E0') for e in r['entries'])
            old=read(Path(cfg['stage8_run'])/f"private/edits/e{t['order']:02d}/result.json")
            for e in old['entries']:
                if e['method'] in ('B0','B2'):
                    e=copy.deepcopy(e);e['cohort_name']='FACT_'+t['cohort'];entries.append(e)
    return entries,ledger,curves


def install(run):
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');old=Path(cfg['stage9_run']);previous.install(old)
    verdicts,side=f4.f3.current_verdicts(old);assert not set(side['all_expected'])-verdicts.keys()
    protocol=read(Path(cfg['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json');pool,execution=f4.f3.historical_judge(cfg,protocol);pool=pool.copy()
    identity=read(old/'private/judge/REUSE_EXECUTION_IDENTITY_PRIVATE.json');tuples=f4.f3.tuples_for(previous.inventory(old)[0],protocol['config_sha256'])
    for key,value in verdicts.items():pool[key]=(tuples[key],value,identity)
    f4.f3.inventory=inventory;f4.f3.historical_judge=lambda *_:(pool,execution)


def report(args):
    run=args.run_root;install(run);entries,ledger,curves=inventory(run);verdicts,side=f4.f3.current_verdicts(run)
    rows=f4.details(entries,verdicts,side['protocol_sha256']);tables=[r for r in f4.summarize(rows) if r['cohort'].startswith('FACT_')]
    f4.csv_write(run/'public/EVIDENCE_PAIRED_RESULTS.csv',tables);f4.csv_write(run/'public/TRAINING_CURVES.csv',curves)
    tasks=[t for t in read(run/'private/TASKS.json') if t['stage9_status']=='PENDING']
    f4.csv_write(run/'public/PAIR_CORRECT_RESULTS.csv',pair_rows(entries,rows,tasks,('E0','E1','E2','B0','B2')))
    diagnostic=[]
    for t in tasks:
        for name in ('E1','E2'):
            path=run/f"private/edits/e{t['order']:02d}/{name}/EVIDENCE_DIAGNOSTICS_AGGREGATE.json"
            if path.exists():diagnostic.extend(dict(r,condition=name) for r in read(path))
    f4.csv_write(run/'public/EVIDENCE_DEPENDENCE_DIAGNOSTICS.csv',diagnostic)
    effects=[]
    for candidate,control in (('E1','E0'),('E1','E2'),('E2','E0'),('E1','B0')):
        sub=[dict(r,method='W1' if r['method']==candidate else 'W0',mode='DEV_THRESHOLD_TRANSFER_DIAGNOSTIC') for r in rows if r['method'] in (candidate,control) and r['mode']=='FORCED_ON']
        effects.extend(dict(e,candidate=candidate,control=control,mode='FORCED_ON',control_mode='FORCED_ON') for e in pairs(sub))
    f4.csv_write(run/'public/PAIRED_EFFECTS.csv',effects);vf.atomic_json(run/'private/DETAILS.json',rows)
    missing=len(set(side['all_expected'])-verdicts.keys());complete=sum(t['status']=='COMPLETE' for t in ledger);unsupported=sum(t['status']=='EVIDENCE_UNSUPPORTED' for t in ledger)
    status=dict(status='COMPUTE_COMPLETE_SUPPORTED_SUBSET' if complete+unsupported==30 and not missing else 'PARTIAL',
        completed_trajectories=complete,evidence_unsupported_trajectories=unsupported,judge_required=len(side['all_expected']),judge_scored=len(verdicts),judge_missing=missing,judge_new=side['new'],judge_reused=side['reused'],publication='PENDING')
    vf.atomic_json(run/'public/RUN_STATUS.json',status);vf.atomic_json(run/'public/COMPLETION_LEDGER.json',ledger)
    lines=['# Stage10 anatomical evidence development results','',str(status),'',
        'Same E support; E0 reuses original F1. Evidence does not enter ordinary inference. Official organ masks are not lesion labels; occluded R is not given a no/other-organ gold.',
        'Stage9 PairCorrect and observed native-failure classification are separate cached-output supplements; not new algorithm evidence.',
        'Source exposure is viewed development. Old7 original T2G unsupported. Missing organ ROI does not mean the organ is absent. No further method automatically launched.',
        '','|Method|Role|Panel|Correct macro|Damage macro|','|---|---|---|---:|---:|']
    for t in tables:
        if t['average']=='macro' and t['mode']=='FORCED_ON':lines.append('|'+ '|'.join(str(t[k]) for k in ('method','role','stratum','semantic','base_correct_damage'))+'|')
    vf.atomic_text(run/'public/GPT_PRO_REVIEW.md','\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','worker','prepare-judge','report'));p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--stage9-run',type=Path);p.add_argument('--processor',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--commit');p.add_argument('--part',choices=('first','0','1'))
    a=p.parse_args()
    if a.action=='prepare':prepare(a)
    elif a.action=='worker':worker(a)
    elif a.action=='report':report(a)
    else:
        install(a.run_root);side=f4.f3.prepare_judge(a);assert not side['execution_version_changed']
