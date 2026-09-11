#!/usr/bin/env python3
"""Fixed C_EXTRA_QA only; reuse historical writers and context-bound judging."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import shutil
from statistics import mean
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from scripts.medtrace import stage13r as prior
from scripts.medtrace.stage11_worker import worker,make_expert
from scripts.medtrace.stage14_sources import image_key,answer_type
from scripts.medtrace.stage7 import input_batch
vf,sw,read,f4,stage12=prior.vf,prior.sw,prior.read,prior.f4,prior.stage12
WRITERS=('C_FACT','C_NO_H','C_EXTRA_QA','BE')
MODES=('FORCED_ON',*stage12.MODES)
COHORTS=('OLD15','NEW7')
SUBSETS=('FULL_COHORT','COMMON_SUPPORTED','SAME_H_IMAGE','DIFFERENT_TRAIN_IMAGE')


def train(args):
    run=args.run_root;tasks=read(run/'private/TASKS.json')
    assert len(tasks)<=22
    for t in tasks:
        assert not t['data']['event']['probes'] and all(r['role'] in ('native','fit') for r in t['data']['rows'])
    worker(argparse.Namespace(run_root=run,parts=1,part='0'))
    checks=[]
    for t in tasks:
        result=read(run/('private/edits/e%02d/C_EXTRA_QA/result.json'%t['order']))
        assert result['steps']==320 and result['parameters']==73728 and result['forwards']==1280
        assert [{k:r[k] for k in ('step','fit_id','H_id','U_id')} for r in result['curve']]==t['schedule']
        assert all(r['G_id']==t['H_to_G'][r['H_id']] for r in result['curve'])
        checks.append(dict(cohort=t['cohort'],edit=t['event_index'],steps=320,parameters=73728,
            parent_native_fit_U_schedule_identical=True,G_forwards=320,student_forwards=1280))
    vf.atomic_json(run/'public/TRAINING_VALIDATION.json',dict(status='PASSED',rows=checks,training_only_reference_roles=True))
    vf.atomic_json(run/'private/TRAINING_COMPLETE.json',dict(status='COMPLETE',trajectories=len(tasks)))


def generate(args):
    run=args.run_root;cfg=read(run/'private/CAMPAIGN_CONFIG.json')
    assert read(run/'private/TRAINING_COMPLETE.json')['status']=='COMPLETE'
    runtime=vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['runtime']['cpu_gate'])))
    originals=read(run/'private/REUSED_ENTRIES.json');layer=runtime.get_module(vf.LAYER);dout,din=layer.weight.shape
    replay_path=run/'public/SYSTEM_REPLAY.json';replays=read(replay_path)['rows'] if replay_path.exists() else []
    for t in read(run/'private/TASKS.json'):
        out=run/('private/scored/e%02d.json'%t['order'])
        if out.exists():continue
        point=run/('private/edits/e%02d/C_EXTRA_QA/step0320.pt'%t['order'])
        state=torch.load(point,map_location=runtime.device,weights_only=True)
        assert state['condition']=='C_EXTRA_QA' and state['step']==320 and state['order']==t['order']
        cp=vf.AsymmetricCPExpert(din,dout,4).to(runtime.device)
        expert=make_expert(cp,'C_EXTRA_QA',t['seed']).to(runtime.device);expert.load_state_dict(state['expert']);expert.requires_grad_(False)
        old=[e for e in originals if e['cohort_name']==t['cohort'] and e['edit']==t['event_index'] and e['method']=='C_FACT']
        assert old and len({e['item']['row']['logical_id'] for e in old})==len(old)
        entries=[];began=time.time()
        for e in old:
            if time.time()-cfg['campaign_epoch']>cfg['train_seconds']:raise TimeoutError('training/generation budget')
            item=e['item'];row=item['row'];input_batch(runtime,row)
            forced=sw.generated(runtime,row,expert)
            new=dict(e,method='C_EXTRA_QA',item=dict(item,forced=forced,fixed=forced if item['fixed_on'] else item['base']))
            entries.append(new)
            for mode in stage12.MODES:
                routed=stage12.deploy(new,mode,cfg['fixed_kappa']);on=routed['item']['fixed_on']
                if len(replays)<4 and not any(r['mode']==mode and r['on']==on for r in replays):
                    actual=sw.generated(runtime,row,expert if on else None);good=prior.old.same_output(actual,routed['item']['fixed'])
                    replays.append(dict(writer='C_EXTRA_QA',mode=mode,on=on,passed=good));assert good,'natural deployment replay differs'
                    vf.atomic_json(replay_path,dict(status='PASSED',rows=replays,maximum_actual_replays=4,outputs='DERIVED from frozen Base/route and new forced output'))
        guard=runtime.base_guard.verify();assert guard['unchanged']
        vf.atomic_json(out,dict(status='COMPLETE',entries=entries,generation_seconds=time.time()-began,base_guard=guard,
            Base_outputs_and_routes='REUSED_EXACT_INPUT_BINDINGS',new_writer_only=True))
        print('GENERATED',t['order'],len(entries),flush=True)
        del cp,expert,state


def inventory(run):
    entries=read(run/'private/REUSED_ENTRIES.json');ledger=[]
    for t in read(run/'private/TASKS.json'):
        path=run/('private/scored/e%02d.json'%t['order'])
        if path.exists():entries.extend(read(path)['entries'])
        ledger.append(dict(cohort=t['cohort'],edit=t['event_index'],status='COMPLETE' if path.exists() else 'PENDING'))
    return entries,ledger,[]


def install(run):
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');old=Path(cfg['stage13r_run']);prior.install(old)
    verdicts,side=f4.f3.current_verdicts(old);assert set(side['all_expected'])<=verdicts.keys()
    protocol=read(Path(cfg['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json')
    pool,execution=f4.f3.historical_judge(cfg,protocol);pool=pool.copy()
    identity=read(old/'private/judge/REUSE_EXECUTION_IDENTITY_PRIVATE.json')
    tuples=f4.f3.tuples_for(prior.inventory(old)[0],protocol['config_sha256'])
    for key,value in verdicts.items():pool[key]=(tuples[key],value,identity)
    f4.f3.inventory=inventory;f4.f3.historical_judge=lambda *_:(pool,execution)


def pair_rows(entries,details):
    lookup={(r['cohort'],r['method'],r['mode'],r['edit'],r['eqkey']):r for r in details}
    assert len(lookup)==len(details),'duplicate analysis input binding'
    groups=defaultdict(list)
    for e in entries:groups[e['cohort_name'],e['method'],e['edit']].append(e['item']['row'])
    pairs=[];units=[]
    for (cohort,method,edit),rows in sorted(groups.items()):
        native=next(r for r in rows if r['role']=='native')
        for mode in MODES:
            a=lookup[cohort,method,mode,edit,native['eqkey']]['semantic'];assert a is not None
            for role in ('fit','evaluation'):
                hs={r['eqkey']:r for r in rows if r['role']==role and r.get('negative_group')=='H'};counts=Counter()
                for key,h in hs.items():
                    b=lookup[cohort,method,mode,edit,key]['semantic'];assert b is not None
                    cell='both_correct' if a and b else 'native_only' if a else 'H_only' if b else 'both_wrong';counts[cell]+=1
                    units.append(dict(cohort=cohort,method=method,mode=mode,edit=edit,role=role,image=image_key(h),both_correct=int(a and b)))
                pairs.append(dict(cohort=cohort,method=method,mode=mode,edit=edit,role=role,native_correct=a,pairs=len(hs),
                    pair_correct=counts['both_correct']/len(hs) if hs else None,**{k:counts[k] for k in ('both_correct','native_only','H_only','both_wrong')}))
    return pairs,units


def support_ids(freeze,cohort,subset):
    return {r['edit'] for r in freeze['rows'] if r['cohort']==cohort and
        (subset=='FULL_COHORT' or r['status']=='SUPPORTED' and (subset=='COMMON_SUPPORTED' or r['tier']==subset))}


def pair_tables(pairs,freeze):
    tables=[];effects=[]
    for cohort in COHORTS:
        for subset in SUBSETS:
            ids=support_ids(freeze,cohort,subset)
            for mode in MODES:
                for role in ('fit','evaluation'):
                    selected={m:[r for r in pairs if r['cohort']==cohort and r['edit'] in ids and r['method']==m and r['mode']==mode and r['role']==role] for m in WRITERS}
                    for method,rs in selected.items():
                        good=[r for r in rs if r['pairs']];n=sum(r['pairs'] for r in good);cells={k:sum(r[k] for r in good) for k in ('both_correct','native_only','H_only','both_wrong')}
                        tables.append(dict(row_kind='PAIR_CORRECT',cohort=cohort,subset=subset,method=method,mode=mode,role=role,
                            requested_edits=len(ids),native_edit_support=len(rs),native_correct=mean(r['native_correct'] for r in rs) if rs else None,
                            pair_edit_support=len(good),pairs=n,pair_macro=mean(r['pair_correct'] for r in good) if good else None,
                            pair_micro=cells['both_correct']/n if n else None,status='AVAILABLE' if rs else 'UNSUPPORTED',**cells))
                    a={r['edit']:r['pair_correct'] for r in selected['C_FACT'] if r['pairs']}
                    for control in ('C_EXTRA_QA','C_NO_H'):
                        b={r['edit']:r['pair_correct'] for r in selected[control] if r['pairs']};common=sorted(a.keys()&b.keys());ds=[a[i]-b[i] for i in common];lo,hi=f4.bootstrap(ds)
                        effects.append(dict(cohort=cohort,subset=subset,mode=mode,role=role,metric='PairCorrect_edit_macro',candidate='C_FACT',control=control,
                            paired_edits=len(ds),delta=mean(ds) if ds else None,ci_low=lo,ci_high=hi,
                            comparison_identity='post-publication supplemental' if control=='C_EXTRA_QA' else 'original fixed comparison reused',patients='UNKNOWN'))
    return tables,effects


def leave_one_image_out(units,freeze):
    out=[];images=sorted({r['image'] for r in units if r['cohort']=='NEW7' and r['role']=='evaluation'})
    for number,image in enumerate(images,1):
        for subset in SUBSETS:
            ids=support_ids(freeze,'NEW7',subset)
            for mode in MODES:
                for control in ('C_EXTRA_QA','C_NO_H'):
                    by=defaultdict(list)
                    for r in units:
                        if r['cohort']=='NEW7' and r['role']=='evaluation' and r['mode']==mode and r['edit'] in ids and r['image']!=image:
                            by[r['method'],r['edit']].append(r['both_correct'])
                    common=sorted({e for m,e in by if m=='C_FACT'}&{e for m,e in by if m==control})
                    ds=[mean(by['C_FACT',e])-mean(by[control,e]) for e in common]
                    out.append(dict(cohort='NEW7',subset=subset,mode=mode,omitted_H_image='H_SOURCE_%02d'%number,
                        candidate='C_FACT',control=control,remaining_edits=len(common),remaining_pairs=sum(len(by['C_FACT',e]) for e in common),
                        delta=mean(ds) if ds else None,interpretation='source-image sensitivity, not a patient-level CI'))
    return out


def report(args):
    run=args.run_root;install(run);cfg=read(run/'private/CAMPAIGN_CONFIG.json');entries,ledger,_=inventory(run)
    verdicts,side=f4.f3.current_verdicts(run);assert set(side['all_expected'])<=verdicts.keys()
    freeze=read(run/'public/SUPPORT_FREEZE.json');details=[]
    for mode in stage12.MODES:
        produced=f4.details([stage12.deploy(e,mode,cfg['fixed_kappa']) for e in entries],verdicts,side['protocol_sha256'])
        for r in produced:
            if r['mode']=='R0':details.append(dict(r,mode=mode))
            elif mode==stage12.MODES[0]:details.append(r)
    vf.atomic_json(run/'private/DETAILS.json',details)
    pairs,units=pair_rows(entries,details);ptable,effects=pair_tables(pairs,freeze)
    stage12.mixed_csv(run/'public/PAIR_OUTCOMES.csv',pairs)
    stage12.mixed_csv(run/'public/PAIR_EFFECTS.csv',effects)
    stage12.mixed_csv(run/'public/LEAVE_ONE_H_IMAGE_OUT.csv',leave_one_image_out(units,freeze))
    tables=[];tradeoffs=[]
    for cohort in COHORTS:
        for subset in SUBSETS:
            ids=support_ids(freeze,cohort,subset);selected=[r for r in details if r['cohort']==cohort and r['edit'] in ids]
            tables.extend(dict(r,subset=subset,row_kind='METRIC') for r in f4.summarize(selected) if r['cohort']==cohort)
            forced={(r['method'],r['edit'],r['eqkey']):r for r in selected if r['mode']=='FORCED_ON'}
            groups=defaultdict(list)
            for r in selected:
                if r['mode'] in stage12.MODES:groups[r['method'],r['mode'],r['role'],r['stratum']].append(r)
            for (method,mode,role,panel),rs in groups.items():
                paired=[(forced[r['method'],r['edit'],r['eqkey']],r) for r in rs]
                tradeoffs.append(dict(row_kind='REJECTION_ACCOUNTING',cohort=cohort,subset=subset,method=method,mode=mode,role=role,stratum=panel,inputs=len(paired),
                    on_count=sum(y['on']==1 for _,y in paired),returned_base=sum(y['on']==0 for _,y in paired),
                    errors_rejected=sum(x['semantic']==0 and y['on']==0 for x,y in paired),
                    avoided_errors=sum(x['semantic']==0 and y['semantic']==1 for x,y in paired),
                    damage_avoided=sum(x['base_correct']==1 and x['semantic']==0 and y['semantic']==1 for x,y in paired),
                    corrections_lost=sum(x['base_correct']==0 and x['semantic']==1 and y['semantic']==0 for x,y in paired)))
    stage12.mixed_csv(run/'public/MATCHED_SUPERVISION_RESULTS.csv',[r for r in tables+ptable if r['mode'] in ('FORCED_ON','BASE')])
    stage12.mixed_csv(run/'public/SYSTEM_TRADEOFF_RESULTS.csv',[r for r in tables+ptable if r['mode'] in stage12.MODES]+tradeoffs)
    costs=[];grows=[];trainrows=[]
    keys=('steps','parameters','fp32_bytes','optimizer_tensor_bytes','checkpoint_bytes','wall_seconds','generation_seconds','forwards','training_tokens','peak_allocated_bytes','peak_reserved_bytes')
    for t in read(run/'private/TASKS.json'):
        result=read(run/('private/edits/e%02d/C_EXTRA_QA/result.json'%t['order']));parent=read(t['parent_fact_result'])
        costs.append(dict(cohort=t['cohort'],edit=t['event_index'],method='C_EXTRA_QA',reused=False,G_CE_forwards=320,**{k:result[k] for k in keys}))
        costs.append(dict(cohort=t['cohort'],edit=t['event_index'],method='C_FACT',reused=True,H_CE_forwards=320,**{k:parent[k] for k in keys}))
        grows.extend(r for r in t['data']['rows'] if r['logical_id'] in t['g_ids']);trainrows.extend(t['data']['rows'])
    stage12.mixed_csv(run/'public/NEW_CONTROL_COSTS.csv',costs)
    newcost=[r for r in costs if not r['reused']]
    original_costs=[]
    for cohort,source,file in [('OLD15',Path(cfg['stage12_run']),'private/TASKS.json'),('NEW7',Path(cfg['stage13r_run']),'private/paired/private/TASKS.json')]:
        for t in read(source/file):
            paths={'C_FACT':Path(cfg['stage11_run'])/('private/edits/e%02d/J1_FREE_R4/result.json'%t['order']) if cohort=='OLD15' else source/('private/paired/private/edits/e%02d/C_FACT/result.json'%t['order']),
                'C_NO_H':source/(('private/edits/e%02d/C_NO_H/result.json' if cohort=='OLD15' else 'private/paired/private/edits/e%02d/C_NO_H/result.json')%t['order']),
                'BE':Path(t['references']['BE'])/'result_private.json'}
            rows={r['logical_id']:r for r in t['data']['rows']};train=[rows[i] for i in [t['native_id'],*t['fit_positive_ids'],*t['h_ids'],*t['u_ids']]]
            for method,path in paths.items():
                r=read(path);original_costs.append(dict(cohort=cohort,edit=t['event_index'],method=method,reused=True,
                    **{k:r.get(k) for k in keys},original_elapsed_seconds=r.get('elapsed_seconds'),original_forward_count=r.get('forward_count'),
                    original_step=r.get('step'),native_fit_QAs=1+len(t['fit_positive_ids']),selected_H_QAs=len(t['h_ids']),selected_U_QAs=len(t['u_ids']),
                    source_images=len({image_key(x) for x in train}),H_answer_types=dict(Counter(answer_type(rows[i]) for i in t['h_ids'])),
                    scope='H support is diagnostic only for C_NO_H and BE; BE uses its original native recipe'))
    stage12.mixed_csv(run/'public/ORIGINAL_WRITER_COSTS.csv',original_costs)
    support=dict(freeze=freeze,execution_sha=cfg['code_commit'],source_preparation_sha=cfg['source_preparation_commit'],public_baseline=cfg['public_baseline'],
        new_trajectories=len(newcost),unique_G_QA=len({(r['dataset'],r['source_qid']) for r in grows}),unique_G_images=len({image_key(r) for r in grows}),
        unique_training_inputs=len({(image_key(r),r['question']) for r in trainrows}),unique_training_images=len({image_key(r) for r in trainrows}),
        G_answer_types_by_edit_support=dict(Counter(answer_type(r) for r in grows)),costs=costs,original_writer_costs=original_costs,
        total_new_training_forwards=sum(r['forwards'] for r in newcost),total_new_training_tokens=sum(r['training_tokens'] for r in newcost),
        all_source_QA_answers_preserved=True,equal_CE_calls_not_equal_information_or_FLOPs=True,
        scope='OLD15 development; NEW7 previously published cohort with supplemental control; no independent patient claim',
        historical_costs='See original Stage12 RUN_COVERAGE_COST and Stage13R COSTS; old writer computation not repeated',
        preparation_timing='source_review_seconds covers curator execution, not uninstrumented interactive engineering')
    vf.atomic_json(run/'public/SUPPORT_AND_COST.json',support)
    # Preserve complete original panels separately, without restricting them to new-control support.
    for name,source,files in [('stage12',Path(cfg['stage12_run']),('WRITER_ABLATION.csv','SINGLE_SYSTEM_RESULTS.csv')),
                              ('stage13r',Path(cfg['stage13r_run']),('PAIR_OUTCOMES.csv','PAIR_EFFECTS.csv','NEW_EDIT_WRITER_RESULTS.csv','NEW_EDIT_SYSTEM_RESULTS.csv','COSTS.csv'))]:
        dest=run/'public/original_full_panels'/name;dest.mkdir(parents=True,exist_ok=True)
        for file in files:shutil.copyfile(source/'public'/file,dest/file)
    replay=read(run/'public/SYSTEM_REPLAY.json');assert replay['status']=='PASSED' and len(replay['rows'])<=4
    status=dict(status='COMPUTE_COMPLETE' if all(r['status']=='COMPLETE' for r in ledger) else 'PARTIAL',planned_new_trajectories=freeze['actual_new_trajectories'],
        completed_new_trajectories=len(newcost),unsupported_old=15-len([r for r in newcost if r['cohort']=='OLD15']),
        judge_required=len(side['all_expected']),judge_scored=len(verdicts),judge_missing=0,judge_reused=side['reused'],judge_new=side['new'],publication='PENDING')
    vf.atomic_json(run/'public/RUN_STATUS.json',status)
    lines=['# Stage14 fixed supervision-matched closeout','',str(status),'',
        'Only C_EXTRA_QA is new. OLD15 and NEW7 remain separate; the original Stage13R C_FACT/C_NO_H confirmation is unchanged. This extra-control comparison is a post-publication paired supplement, not an unseen prospective confirmation.',
        'All historical panels are retained verbatim under original_full_panels. The old15 have no new G support because their training images also occur in the current combined evaluation image set. No role was relaxed, no historical sample was deleted or retrained.',
        'The frozen source overlay uses original authorized correct QA and source-text-only different-proposition review. SAME_H_IMAGE does not imply equal information, statistical independence, equal answer tokens or equal FLOPs. CE slot count is matched; actual token/forward costs are reported.',
        '', '|Cohort|Subset|Writer|Native|H pair edit-macro|H pair micro|Pairs|Edit support|', '|---|---|---|---:|---:|---:|---:|---:|']
    for r in ptable:
        if r['mode']=='FORCED_ON' and r['role']=='evaluation' and r['subset'] in ('FULL_COHORT','COMMON_SUPPORTED'):
            lines.append('|'+ '|'.join(str(r.get(k)) for k in ('cohort','subset','method','native_correct','pair_macro','pair_micro','pairs','pair_edit_support'))+'|')
    lines+=['','## Paired effects','']
    lines += ['- '+str(r) for r in effects if r['subset']=='COMMON_SUPPORTED' and r['role']=='evaluation']
    lines += ['', 'Interpret H selection using C_FACT minus C_EXTRA_QA jointly with native/H/U damage, source-style, cross-family and same-answer challenge tables. Confidence intervals crossing zero, native failures and unfavorable control comparisons do not trigger rescue or filtering.',
        'Edit bootstrap uses the inherited 10,000 draws and seed20260908. NEW7 shares only three H images; LEAVE_ONE_H_IMAGE_OUT.csv reports reduced-support sensitivity, not a reliable patient-level confidence interval.',
        'R0 and fixed old RC are parallel frozen deployment tradeoffs. Derived route outputs reuse exact bound Base/routes. RC returning Base on all H/U cannot establish individual writer protection; inspect ON counts and corrections lost. Original RC remains Stage13R predeclared primary system, R0 secondary.',
        'BalancEdit is an external adaptation with different capacity and original supervision budget, not a matched-H or paper-exact baseline. Constructed questions are not official T2G, and same-answer different images are not official T1G.',
        'Source consistency review is not human clinical signoff. Patients and pretraining exposure are UNKNOWN. No clinical, broad task or general deployment superiority claim; no additional method/seed/native/threshold or automatic next experiment.']
    vf.atomic_text(run/'public/GPT_PRO_REVIEW.md','\n'.join(lines)+'\n')
    vf.atomic_text(run/'public/METHOD_V1_AND_CLAIMS.md','''# Frozen method V1 and claim boundary

LLaVA-Med v1.5 Mistral7B and original tokenizer, visual encoder, prompts, generation/EOS masks and Judge execution are fixed by the inherited runtime locks. Each own-edit native CP -> A2 -> CP-W0 checkpoint precedes H supervision; its function is expanded into free rank4 at model.layers.21.mlp.down_proj (73,728 FP32 parameters).

For 320 steps, C_FACT minimizes 0.5 native CE +0.5 fit-paraphrase CE +H-source CE +0.01 full-vocabulary token-mean KL(Base||student) on U under the exact Base prefix. Adam A1e-4/B1e-3, betas(0.9,0.999), eps1e-8, weight decay0, clip1 and original normalization are unchanged. Native/fit/U schedules and per-edit seeds are inherited exactly. H is the previously selected same proposition/different image/conflicting verified source answer; no new H selection is performed.

C_NO_H removes H CE. C_EXTRA_QA replaces each H CE slot with another authorized original source QA, prioritized on the same H image, then an existing legal episode training image, with answer-type/token proximity selected before training. G is a different reviewed proposition, excludes protected/evaluation images and the fixed U input, and uses no more unique QA than selected H. This changes auxiliary fact content intentionally; it is not equal information or exact equal FLOPs. Missing legal G gives unsupported, never an invented source answer.

FORCED_ON measures writer behavior. The existing Base-derived BalancEdit router supplies R0; RC uses the original kappa0.7696741135364367 without recalibration. OFF returns exact Base. Outputs may be derived from bound forced/Base generations, with at most four naturally encountered actual branch replays. Judge reuse requires the full question/reference/raw-output/protocol tuple and execution ancestry, never bare answer-string matching.

Validated by this run: explicit R4 mapping; same own-edit W0 and native/fit/U schedules; 320 G calls and1280 student forwards per new trajectory; declared finite cohort results and replay/coverage records. Source review is agent consistency, not human clinical review. The prior new7 FACT/NO_H analysis retains its original predeclared identity; new EXTRA comparisons are post-publication supplements. Old15 is development, not a second independent clinical confirmation.

Not validated: independent-patient effects, new unseen-dataset generality, unrestricted factual/clinical correctness, equal-information superiority, full official M3Bench reproduction, unseen threshold optimality, pretraining independence or broad robustness. Only three new-cohort H images are available. Same-answer challenge costs and all failed natives remain visible. Basic low-rank adapters, CE, KL and routing are not claimed as new inventions. No additional writer/module/loss/seed/threshold or next stage is authorized by this closeout.
''')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','generate','prepare-judge','report'));p.add_argument('--run-root',type=Path,required=True);args=p.parse_args()
    if args.action=='prepare-judge':install(args.run_root);side=f4.f3.prepare_judge(args);assert not side['execution_version_changed']
    else:globals()[args.action](args)
