#!/usr/bin/env python3
"""Fixed old16 threshold transfer and evaluation-only sidecar; never trains."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace import finalize_stage5 as f5
from scripts.medtrace import run_stage3_bank as bank
from scripts.medtrace.stage5_bank import Generator, available, METHODS, CONDITIONS
from scripts.medtrace.stage4_scope import accepted
from scripts.medtrace.run_stage2 import vf, read, bind_rows

FIXED = 'RC_FIXED_OLD16'
OLD = 'STAGE5_NEW32_FOLLOWUP_TRANSFER_NOT_BLIND'
NEW = 'STAGE6_FROZEN_SOURCE_EVALUATION_SIDECAR'


def threshold(run):
    lock = read(run/'private/TRANSFER_LOCK.json')
    # Explicit protocol-required integrity check of this one immutable source.
    raw = Path(lock['source_path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == lock['source_sha256'], 'old16 threshold source changed'
    assert json.loads(raw)['kappa'] == lock['kappa']
    return lock['kappa']


def transfer(entry, kappa):
    item = dict(entry['item']); on = accepted(item['route'], kappa)
    if entry['track'] == 'B':
        item.update(actual=item['actual'] if on else item['base'], on=on,
                    selected_expert=item['selected_expert'] if on else None)
    else:
        item.update(fixed=item['fixed'] if on else item['base'], fixed_on=on)
    return dict(entry, route_mode=FIXED, item=item,
                provenance='DERIVED_FROM_FROZEN_OUTPUTS')


def prepare(args):
    run, old = args.run_root, args.stage5_run
    if run.exists():raise FileExistsError('one-shot run exists; do not reset clock')
    assert shutil.disk_usage(run.parent).free > 10*1024**3
    run.mkdir(); probe = run/'.probe'
    with probe.open('x') as f:f.write('stage6')
    assert probe.read_text() == 'stage6'; probe.unlink()
    config = read(old/'private/CAMPAIGN_CONFIG.json')
    config.update(kind='MEDTRACE_STAGE6', stage5_run=str(old), allowed_physical_gpus=[0,1],
                  wall_hours=6, gpu_hours=12, train_seconds=0, campaign_epoch=time.time(),
                  authorization='USER_STAGE6_GPU0_GPU1', code_commit=args.commit)
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json', config)
    vf.atomic_json(run/'private/CAMPAIGN_START.json', dict(epoch=config['campaign_epoch']))
    source = Path(config['stage4_run'])/'private/bank/prefix16/THRESHOLD_LOCK.json'
    raw = source.read_bytes(); old_lock = json.loads(raw)
    lock = dict(source_path=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                kappa=old_lock['kappa'], original_lock=old_lock, condition=FIXED,
                source_cohort='STAGE4_SLAKE16_PREFIX16', new_calibration_used=False,
                calibration_status='NOT_PERFORMED_FIXED_NUMERICAL_TRANSFER',
                score_implementation='scripts/medtrace/stage4_scope.py:margin,accepted')
    vf.atomic_json(run/'private/TRANSFER_LOCK.json', lock)
    vf.atomic_text(run/'private/PROTOCOL.md', args.protocol.read_text())
    vf.atomic_json(run/'public/RUN_AND_TRANSFER_LOCK.json', dict(
        **{k:v for k,v in lock.items() if k!='source_path'}, threshold_source='Stage4 private/bank/prefix16/THRESHOLD_LOCK.json',
        source_public_commit='8c6c84eaf8c040c1408ebf956c76d1e3ae246839',
        stage5_writer_execution='09cd564', stage5_closeout_execution='cb22fab',
        stage5_numerical_reporting='17e4376', stage5_result_commit='abf004385f3932602216c16667330071f0d205ad',
        stage6_preparation_commit=args.commit, methods=['W0','BE'], new_writer_training=0,
        gpu_indices=[0,1], gpu_uuids={str(i):config['gpu_uuids'][str(i)] for i in (0,1)},
        wall_hours_limit=6, gpu_hours_limit=12, old_results_read_only=True))
    vf.atomic_json(run/'public/RUN_STATUS.json', dict(status='PROTOCOL_AND_THRESHOLD_FROZEN',
        compute='STARTED', publication='PENDING', new_writer_training=0))
    print('PROTOCOL_FROZEN', lock['kappa'], flush=True)


def freeze_sidecar(args):
    run=args.run_root; config=read(run/'private/CAMPAIGN_CONFIG.json'); old=Path(config['stage5_run'])
    destination=run/'private/EVAL_SIDECAR.json'
    if destination.exists():raise FileExistsError('evaluation sidecar already frozen')
    review=read(args.review)
    assert review['source_consistency_reviewed'] and not review['clinical_human_reviewed']
    pool=read(old/'private/INHERITED_SOURCE_POOL.json')['rows']
    byid={str(r['qid']):r for r in pool}
    episodes=available(old,'B'); targets={e['rows'][0]['image_id'] for e in episodes}
    norm=lambda s:' '.join(str(s).lower().split())
    known={(r['image_id'],norm(r['question'])):r['role'] for e in episodes for r in e['rows']}
    prior_images=defaultdict(set)
    for e in episodes:
        for r in e['rows']:prior_images[r['image_id']].add(r['role'])
    rows=[]; ledger=[]; counts=Counter(); image_usage=Counter()
    for e in episodes:
        i=e['event_index']; native=e['rows'][0]; attr=review['target_attributes'][str(i)]
        selected=[]
        for qid in review['positive_pairs'].get(str(i),[]):
            r=byid[str(qid)]
            assert r['image_name'].lower()==native['image_id'] and norm(r['answer'])==norm(native['reference'])
            selected.append((r,'positive',None,'Existing source alternative QA; reviewed same referent, attribute, scope and answer.'))
        for pair in review['hard_pairs'].get(str(i),[]):
            selected.append((byid[str(pair['qid'])],'negative','H',pair['evidence']))
        eligible=[]
        for qid,attribute in review['negative_attributes'].items():
            r=byid[qid]; identity=(r['image_name'].lower(),norm(r['question']))
            reason=None
            if attr in review['no_u_target_attributes']:reason='Broad target scope: unrelated relation not established'
            elif attribute==attr or [attr,attribute] in review['uncertain_attribute_pairs']:reason='Same or potentially overlapping fact scope; no U claim'
            elif identity in known:reason='Existing Stage5 input retained in original role, not new evaluation'
            if reason:ledger.append(dict(edit=i,qid=qid,status='EXCLUDED_U',reason=reason));continue
            assert r['image_name'].lower() not in targets
            eligible.append((r,attribute))
        # Deterministic coverage-first selection, no route/answer scores available.
        used=set()
        for r,attribute in sorted(eligible,key=lambda v:(image_usage[v[0]['image_name'].lower()],v[0]['image_name'].lower(),int(v[0]['qid']))):
            image=r['image_name'].lower()
            if image in used or len(used)>=4:
                ledger.append(dict(edit=i,qid=r['qid'],status='NOT_SELECTED',reason='Frozen four-U/source-diversity cap'));continue
            selected.append((r,'negative','U','Reviewed distinct explicit source attribute: '+attr+' versus '+attribute))
            used.add(image);image_usage[image]+=1
        for r,label,group,evidence in selected:
            image=r['image_name'].lower(); identity=image,norm(r['question'])
            if identity in known:
                ledger.append(dict(edit=i,qid=r['qid'],status='EXCLUDED_EXISTING_INPUT',reason=known[identity]));continue
            assert label=='positive' or image not in targets
            row=dict(dataset='VQA-RAD',source_qid=r['qid'],image_id=image,source_group='VQA-RAD:'+image,
                image_path=str(Path(native['image_path']).parent/r['image_name']),question=r['question'],reference=str(r['answer']),
                role='evaluation',label=label,negative_group=group,logical_id=f's6-{i}-{r["qid"]}',
                fact_relation='reviewed_same_fact_text_augmentation' if label=='positive' else 'stage6_reviewed_source_relation',
                relation_evidence=evidence,conflict_verified=group=='H',
                probe_kind='existing_source_alternative' if label=='positive' else group,
                confirmation_panel='existing_source_alternative' if label=='positive' else group,
                prior_source_image_roles=sorted(prior_images[image]),patient_id='UNKNOWN')
            rows.append(dict(edit=i,row=row));counts[i,label,group]+=1
            ledger.append(dict(edit=i,qid=r['qid'],status='SELECTED',label=label,group=group,evidence=evidence))
    assert all(r['row']['role']=='evaluation' for r in rows)
    vf.atomic_json(destination,dict(status='FROZEN_BEFORE_NEW_INPUT_ROUTES_AND_OUTPUTS',rows=rows,
        review=review,ledger=ledger,source_rows=len(pool),source_images=55,frozen_epoch=time.time(),
        no_new_medical_facts=True,clinical_human_reviewed=False))
    public=[]
    for e in episodes:
        i=e['event_index']; selected=[r['row'] for r in rows if r['edit']==i]
        public.append(dict(edit=i,positive=sum(r['label']=='positive' for r in selected),
            H=sum(r['negative_group']=='H' for r in selected),U=sum(r['negative_group']=='U' for r in selected),
            new_input_count=len(selected),prior_fit_image_observations=sum('fit' in r['prior_source_image_roles'] for r in selected),
            H_missing_reason=None if counts[i,'negative','H'] else 'No explicit conflicting same-object/attribute source pair in this bounded review',
            image_generality='NA',patient_identity='UNKNOWN'))
    vf.atomic_json(run/'public/EVAL_SUPPORT_MANIFEST_PUBLIC.json',dict(records=public,
        source_pool_rows=len(pool),source_pool_images=55,negative_images=len({r['row']['image_id'] for r in rows if r['row']['label']=='negative'}),
        H_images=len({r['row']['image_id'] for r in rows if r['row']['negative_group']=='H'}),
        H_edits=sum(r['H']>0 for r in public),ledger_states=dict(Counter(r['status'] for r in ledger)),
        source_use_authorized=True,source_consistency_reviewed=True,clinical_human_reviewed=False,
        reused_source_images_not_independent_unexposed_patients=True,review_needed=review['public_review_needed'],
        old_roles_unchanged=True,new_inputs_evaluation_only=True,source_review_seconds=time.time()-config['campaign_epoch']))
    print('SIDECAR_FROZEN',len(rows),flush=True)


def derive(args):
    run=args.run_root; old=Path(read(run/'private/CAMPAIGN_CONFIG.json')['stage5_run'])
    path=run/'private/TRANSFER_ENTRIES.json'
    if path.exists():raise FileExistsError('frozen transfer already derived')
    k=threshold(run); originals,_,_=f5.inventory(old)
    originals=[e for e in originals if e['route_mode']=='R0' and
        (e['track']=='A' and e['item']['row']['role'] in ('native','evaluation') or e['track']=='B' and e['method'] in ('W0','BE'))]
    be={(e['track'],e['prefix'],e['edit'],e['item']['row']['eqkey']):e for e in originals if e['method']=='BE'}
    entries=[]
    for e in originals:
        item=dict(e['item']);r=item['row'];index=e['track'],e['prefix'],e['edit'],r['eqkey']
        if e['track']=='A':item['route']=be[index]['item']['route']
        else:assert item['route']==be[index]['item']['route'] and item['base']==be[index]['item']['base']
        e=dict(e,item=item,cohort_name=OLD,common_support=False,provenance='DERIVED_FROM_FROZEN_OUTPUTS')
        entries.extend([e,transfer(e,k)])
    for label in ('BE','W0'):
        b=read(old/'private/final_bank'/label/'result_private.json')
        assert b['bank_size']==32 and b['manifest_order']==list(range(201,233))
    vf.atomic_json(path,entries)
    print('TRANSFER_DERIVED',len(entries),flush=True)


def inventory(run):
    entries=read(run/'private/TRANSFER_ENTRIES.json')
    for label in ('W0','BE'):
        p=run/f'private/sidecar_{label}.json'
        if p.exists():entries+=read(p)['entries']
    return entries,[],[]


def install(run):
    old=Path(read(run/'private/CAMPAIGN_CONFIG.json')['stage5_run'])
    f5.install(old)
    config=read(old/'private/CAMPAIGN_CONFIG.json')
    protocol=read(Path(config['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json')
    pool,execution=f5.f4.f3.historical_judge(config,protocol)
    verdicts,side=f5.f4.f3.current_verdicts(old)
    assert side and set(side['all_expected'])<=verdicts.keys()
    identity=read(old/'private/judge/REUSE_EXECUTION_IDENTITY_PRIVATE.json')
    for packet in vf.read_jsonl(old/'private/judge/JUDGE_PACKET_PRIVATE.jsonl'):
        key=packet['opaque_query_id'];assert packet==f5.f4.f3.packet_row(packet['question'],packet['gold_answer'],packet['raw_base_answer'],protocol['config_sha256'])
        pool[key]=(packet,verdicts[key],identity)
    f5.f4.f3.inventory=inventory
    f5.f4.f3.historical_judge=lambda *_:(pool,execution)
    return pool,execution,protocol


def score_rows(entries,verdicts,protocol):
    rows=[]
    for mode in ('R0',FIXED):
        subset=[e for e in entries if e['route_mode']==mode]
        for r in f5.f4.details(subset,verdicts,protocol):
            if r['track']=='A' and r['mode']!='R0':continue
            r['mode']=mode;r['v4_primary']=None
            r['stratum']={'T0':'native_source_edit','source_style_confirmation':'derived_source_style','cross_family_confirmation':'derived_imperative_style'}.get(r['stratum'],r['stratum'])
            if r['cohort']==NEW:
                for key in ('own_forced_correct','rejection','wrong_writer','writer_error','positive_wrong_selection','positive_owner_error'):
                    r[key]=None  # No owner-forced sidecar generation was requested.
            rows.append(r)
    return rows


def report(args):
    run=args.run_root; pool,_,protocol=install(run);entries,_,_=inventory(run)
    verdicts={k:v[1] for k,v in pool.items()}
    side=None
    if (run/'private/judge/JUDGE_SIDECAR_PRIVATE.json').exists():
        current,side=f5.f4.f3.current_verdicts(run);verdicts.update(current)
    rows=score_rows(entries,verdicts,protocol['config_sha256'])
    expected=f5.f4.f3.tuples_for(entries,protocol['config_sha256']);missing=set(expected)-verdicts.keys()
    tables=[r for r in f5.f4.summarize(rows) if r['cohort'] in (OLD,NEW)]
    for t in tables:
        group=[r for r in rows if all(r[k]==t[k] for k in ('cohort','track','prefix','method','mode','role','stratum','strict_role'))]
        t['unique_eqkeys']=len({r['eqkey'] for r in group})
    f5.f4.csv_write(run/'public/FIXED_RC_TRANSFER_RESULTS.csv',tables)
    bykey={(r['cohort'],r['track'],r['prefix'],r['method'],r['edit'],r['eqkey']):r for r in rows if r['mode']=='R0'}
    accounts=defaultdict(list)
    for r in rows:
        if r['mode']!=FIXED:continue
        before=bykey[r['cohort'],r['track'],r['prefix'],r['method'],r['edit'],r['eqkey']]
        a,b,base=before['semantic'],r['semantic'],r['base_correct']
        if None in (a,b,base):continue
        avoided=int(base==1 and a==0 and b==1);lost=int(base==0 and a==1 and b==0)
        assert b-a==avoided-lost, 'RC causal accounting mismatch'
        accounts[r['cohort'],r['track'],r['prefix'],r['method'],r['stratum']].append(dict(
            edit=r['edit'],image=r['source_image'],eqkey=r['eqkey'],delta=b-a,
            avoided_damage=avoided,lost_correction=lost,
            rejected_correct_positive=int(r['strict_role']=='EDIT_TARGET' and a==1 and before['on']==1 and r['on']==0)))
    accounting=[]
    for key,g in accounts.items():
        edits=sorted({r['edit'] for r in g});images=sorted({r['image'] for r in g})
        de=[mean(r['delta'] for r in g if r['edit']==e) for e in edits]
        di=[mean(r['delta'] for r in g if r['image']==i) for i in images]
        lo,hi=f5.f4.bootstrap(de);ilo,ihi=f5.f4.bootstrap(di)
        accounting.append(dict(zip(('cohort','track','prefix','method','panel'),key),inputs=len(g),
            unique_eqkeys=len({r['eqkey'] for r in g}),edits=len(edits),source_images=len(images),
            **{k:sum(r[k] for r in g) for k in ('avoided_damage','lost_correction','rejected_correct_positive')},
            net_correct_change=sum(r['delta'] for r in g),edit_macro_change=mean(de),
            ci_low=lo,ci_high=hi,image_cluster_ci_low=ilo,image_cluster_ci_high=ihi,patients='UNKNOWN'))
    f5.f4.csv_write(run/'public/RC_CAUSAL_ACCOUNTING.csv',accounting)
    pair_input=[dict(r,mode=('DEV_THRESHOLD_TRANSFER_DIAGNOSTIC' if r['track']=='A' else 'RC')) if r['mode']==FIXED else r for r in rows]
    effects=f5.pairs(pair_input)
    for r in effects:
        for field in ('mode','control_mode'):
            if r[field] in ('RC','DEV_THRESHOLD_TRANSFER_DIAGNOSTIC'):r[field]=FIXED
    f5.f4.csv_write(run/'public/PAIRED_EFFECTS.csv',effects)
    vf.atomic_json(run/'private/DETAILS.json',rows)
    complete=all((run/f'private/sidecar_{m}.json').exists() and read(run/f'private/sidecar_{m}.json')['status']=='RAW_READY' for m in ('W0','BE')) and not missing
    replay=[read(run/f'private/replay_{m}.json') if (run/f'private/replay_{m}.json').exists() else {'status':'PENDING'} for m in ('W0','BE')]
    complete=complete and all(r['status']=='COMPLETE' for r in replay)
    status=dict(status='COMPUTE_COMPLETE' if complete else 'PARTIAL',new_writer_training=0,
        judge_required=len(expected),judge_missing=len(missing),judge_new=side['new'] if side else 0,
        judge_reused=side['reused'] if side else len(expected)-len(missing),publication='PUBLICATION_PENDING',
        replays=replay,wall_seconds=time.time()-read(run/'private/CAMPAIGN_START.json')['epoch'])
    vf.atomic_json(run/'public/RUN_STATUS.json',status)
    lines=['# MedTRACE Stage6 fixed threshold transfer','','Status: '+status['status'],
        '',f'Unique old16 kappa = {threshold(run)!r}. No new calibration and no writer training.',
        'Stage5 New32 is a follow-up transfer evaluation, not an untouched blind test. R0 rankings are unchanged; rejection returns the frozen Base.',
        'New sidecar reuses the cleared 340-row/55-image source pool. Prior source-image fit exposure is disclosed; new QA inputs do not alter old roles. Clinical review is not claimed. Official image generality is NA.',
        'W1 is a single-edit appendix only. Missing H is not a passing hard-rejection result. All-OFF is Base return, not writer protection. Source-image clusters, not repeated edit-input counts, bound independent support.',
        '', '## Final-bank transfer (edit macro)', '',
        '| Cohort | Writer | Route | Panel | Inputs | Images | Correct | ON | Base-correct damage |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    def pct(x):return 'NA' if x is None else f'{x*100:.2f}%'
    for r in tables:
        if r['track']=='B' and r['average']=='macro':
            lines.append(f"| {r['cohort']} | {r['method']} | {r['mode']} | {r['stratum']} | {r['inputs']} | {r['source_images']} | {pct(r['semantic'])} | {pct(r['on'])} | {pct(r['base_correct_damage'])} |")
    lines+=['','RC_CAUSAL_ACCOUNTING.csv exactly decomposes accuracy change into avoided damage minus lost corrections. PAIRED_EFFECTS.csv includes edit and image-cluster uncertainty with fixed seed20260908, 10000 draws. No noninferiority or clinical-safety claim.',
        '',f'Judge required={len(expected)}, missing={len(missing)}. Publication remains separate until verified.']
    vf.atomic_text(run/'public/GPT_PRO_REVIEW.md','\n'.join(lines)+'\n')
    print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','freeze-sidecar','derive','prepare-judge','report'))
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--stage5-run',type=Path)
    p.add_argument('--review',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--commit')
    a=p.parse_args()
    if a.action=='prepare':prepare(a)
    elif a.action=='freeze-sidecar':freeze_sidecar(a)
    elif a.action=='derive':derive(a)
    elif a.action=='report':report(a)
    else:
        install(a.run_root);side=f5.f4.f3.prepare_judge(a)
        assert not side['execution_version_changed'],'Judge runtime must stay unchanged'
        print(side['new'],side['reused'],flush=True)
