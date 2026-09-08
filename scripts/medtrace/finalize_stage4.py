#!/usr/bin/env python3
"""Stage4 exact-tuple Judge reuse and fixed-support descriptive analysis."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.medtrace import finalize_stage3 as f3
from scripts.medtrace.run_stage4 import OLD,BANK,task_for,read,vf
from scripts.medtrace.finalize_selective_write import csv_write,hierarchical,opaque

ORIGINAL_HISTORY=f3.historical_judge
METRICS=(*f3.METRICS,'wrong_given_on_base_correct','wrong_given_on_strict_base',
         'positive_wrong_rejection','positive_wrong_selection','positive_owner_error')


def historical(config,protocol):
    pool,execution=ORIGINAL_HISTORY(config,protocol)
    directory=Path(config['stage3_run'])/'private/judge'
    side=read(directory/'JUDGE_SIDECAR_PRIVATE.json')
    if side['protocol_sha256']!=protocol['config_sha256']:raise ValueError('Stage3 Judge protocol changed')
    packets={r['opaque_query_id']:r for r in vf.read_jsonl(directory/'JUDGE_PACKET_PRIVATE.jsonl')}
    if vf.sha256_file(directory/'JUDGE_PACKET_PRIVATE.jsonl')!=side['packet_sha256']:raise ValueError('Stage3 Judge packet drift')
    reused=read(directory/'REUSED_VERDICTS_PRIVATE.json');identity=read(directory/'REUSE_EXECUTION_IDENTITY_PRIVATE.json')
    for key,value in reused.items():
        if key not in pool or pool[key][1:]!=(value,identity):raise ValueError('Stage3 reuse ancestry mismatch')
    if side['new']:
        verdicts,execution,_=f3.validated_judge(directory)
        if f3.execution_identity(execution)!=identity:raise ValueError('Stage3 execution mismatch')
        for key,value in verdicts.items():
            r=packets[key]
            if r!=f3.packet_row(r['question'],r['gold_answer'],r['raw_base_answer'],protocol['config_sha256']):
                raise ValueError('Stage3 full tuple mismatch')
            pool[key]=(r,value,identity)
    if set(packets)|set(reused)!=set(side['all_expected']):raise ValueError('Stage3 required Judge coverage')
    return pool,execution


def inventory(run):
    config=read(run/'private/CAMPAIGN_CONFIG.json')
    if config['kind']!='MEDTRACE_STAGE4':raise ValueError('wrong stage')
    entries=[];ledger=[];costs=[]
    for cohort in read(run/'private/COHORTS.json')['edits']:
        child=Path(cohort['child']);i=cohort['event_index']
        data=read(child/f'private/edits/e{i:02d}.json')
        if vf.sha256_json(data)!=cohort['data_sha256']:raise ValueError('frozen Stage4 input changed')
        for method in ('W0','W01','W1','BE'):
            directory=(child/'private/tasks'/task_for(cohort['cohort'],i)['task_id'] if method=='W01'
                       else Path(cohort['references'][method]))
            path=directory/'result_private.json'
            complete=False
            if path.exists():
                result=read(path);f3.finite(result)
                if (not result['base_guard']['unchanged'] or result['step']!=(50 if method=='BE' else 320)
                        or (method!='BE' and result['a2_sha256']!=data['a2_sha256'])):
                    raise ValueError('method endpoint identity changed')
                for row in data['rows']:
                    item=result['outputs'][row['logical_id']];f3.validate_row(item['row'],row)
                    entries.append(dict(track='A',prefix=0,edit=i,method=method,item=item,common_support=False,
                        cohort_name=cohort['cohort'],system_valid=item.get('system_replay_valid',item.get('route_branch_parity',True)),
                        target=data['event']['edit_record']['gold_answer'],task_id=directory.name))
                complete=len(result['outputs'])==len(data['rows'])
                costs.append(dict(cohort=cohort['cohort'],edit=i,method=method,new_training=method=='W01',
                    **{k:result.get(k) for k in f3.COSTS}))
            ledger.append(dict(cohort=cohort['cohort'],edit=i,method=method,track='A',complete=complete,
                               planned_inputs=len(data['rows']),complete_inputs=len(data['rows']) if complete else 0))
        diagpath=child/'private/tasks'/task_for(cohort['cohort'],i)['task_id']/'STAGE4_DIAGNOSTICS_PRIVATE.json'
        if diagpath.exists():
            control=read(Path(cohort['references']['W1'])/'result_private.json')
            for diagnostic in read(diagpath)['rows']:
                for b in diagnostic['behaviors']:
                    row=b['row'];base=control['outputs'][row['logical_id']]['base']
                    entries.append(dict(track='A',prefix=0,edit=i,method=diagnostic['method'],
                        item=dict(row=row,base=base,forced=b['output'],fixed=b['output'],fixed_on=True),
                        target=data['event']['edit_record']['gold_answer'],system_valid=True,common_support=False,
                        cohort_name=cohort['cohort'],diagnostic_step=diagnostic['step']))
    for prefix in (1,4,8,16):
        path=run/f'private/bank/prefix{prefix}/result_private.json'
        complete=False
        if path.exists():
            result=read(path);f3.finite(result)
            lock=read(path.parent/'THRESHOLD_LOCK.json');plain={k:v for k,v in lock.items() if k!='threshold_sha256'}
            if vf.sha256_json(plain)!=lock['threshold_sha256']:raise ValueError('threshold changed after freeze')
            if not result['base_guard']['unchanged'] or not result['off_base_parity']:raise ValueError('bank Base parity missing')
            if result['threshold']!=plain:raise ValueError('bank threshold mismatch')
            for item in result['outputs']:
                entries.append(dict(track='B',prefix=prefix,edit=item['source_edit_index'],method=item['method'],
                    item=item,cohort_name=BANK,route_mode=item['route_mode'],common_support=False,system_valid=True))
            complete=not result['missing'] and len(result['outputs'])==result['expected_inputs']
        ledger.append(dict(track='B',prefix=prefix,complete=complete))
    return entries,ledger,costs


def install():
    f3.historical_judge=historical
    f3.inventory=inventory


def details(entries,verdicts,protocol):
    output=[]
    cells=defaultdict(list)
    for entry in entries:
        cells[entry['cohort_name'],entry['track'],entry.get('route_mode'),entry.get('diagnostic_step')].append(entry)
    for (cohort,track,route,step),group in cells.items():
        produced=f3.details_for(group,verdicts,protocol)
        for row in produced:
            if track=='B' and row['mode']!='ROUTED':continue
            if step is not None and row['mode']!='FORCED_ON':continue
            row.update(cohort=cohort,mode=route if track=='B' else 'R0' if row['mode']=='ROUTED' else row['mode'],
                       diagnostic_step=320 if step is None else step,is_diagnostic=step is not None)
            correct=row['semantic'];on=row['on']==1;local=row['strict_role']=='STRICT_BASE'
            positive=row['strict_role'] in ('EDIT_TARGET','NOW_EDITED_CONTEXT')
            row.update(wrong_given_on_base_correct=1-correct if local and on and row['base_correct']==1 and correct is not None else None,
                wrong_given_on_strict_base=1-correct if local and on and correct is not None else None,
                positive_wrong_rejection=float(not on and correct==0) if positive and correct is not None else None,
                positive_wrong_selection=row['wrong_writer'],positive_owner_error=row['writer_error'])
            output.append(row)
    return output


def macro(rows,metric):
    eligible=[r[metric] for r in rows if r[metric] is not None]
    return (mean(eligible) if eligible else None) if metric=='v4_primary' else hierarchical(rows,metric)


def summarize(rows):
    keys=('cohort','track','prefix','method','mode','role','stratum','strict_role','diagnostic_step','is_diagnostic')
    cells=defaultdict(list)
    for row in rows:cells[tuple(row[k] for k in keys)].append(row)
    for cohort in (OLD,BANK):
        for method in ('W0','W01','W1','BE'):
            for mode in ('R0','FORCED_ON'):
                for task in f3.FORMAL:
                    if not any(k[0]==cohort and k[1]=='A' and k[3:5]==(method,mode) and k[6]==task for k in cells):
                        cells[cohort,'A',0,method,mode,'native' if task=='T0' else 'formal_development',task,'UNKNOWN',320,False]=[]
    result=[]
    for key,group in sorted(cells.items()):
        for average in ('macro','micro'):
            item=dict(zip(keys,key));item.update(average=average,inputs=len(group),edits=len({r['edit'] for r in group}),
                source_images=len({r['source_image'] for r in group}),patients='UNKNOWN')
            for metric in METRICS:
                values=[r[metric] for r in group if r[metric] is not None]
                edit_values=[macro([r for r in group if r['edit']==i],metric) for i in {r['edit'] for r in group}]
                edit_values=[v for v in edit_values if v is not None]
                item[metric]=(mean(values) if average=='micro' else mean(edit_values)) if values and edit_values else None
                item[metric+'_numerator']=sum(values) if values else None
                item[metric+'_denominator']=len(values)
                item[metric+'_edit_support']=len(edit_values)
            result.append(item)
    return result


def bootstrap(values):
    if len(values)<2:return None,None
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(20260908)
    draws=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    return tuple(map(float,np.quantile(draws,[.025,.975])))


def paired(rows):
    keys=('cohort','track','prefix','role','stratum','strict_role')
    cells=defaultdict(list)
    for r in rows:
        if not r['is_diagnostic']:cells[tuple(r[k] for k in keys)].append(r)
    output=[]
    for key,group in sorted(cells.items()):
        comparisons=([(c,m,'W01',m) for c in ('W0','W1') for m in ('R0','FORCED_ON')] if key[1]=='A' else
                     [(m,'R0',m,'RC') for m in ('W0','W01','W1')])
        for control,control_mode,candidate,candidate_mode in comparisons:
            for metric in ('semantic','v4_primary','base_correct_damage','base_wrong_became_correct','base_wrong_changed','fpr'):
                a={(r['edit'],r['eqkey']):r for r in group if (r['method'],r['mode'])==(candidate,candidate_mode) and r[metric] is not None}
                b={(r['edit'],r['eqkey']):r for r in group if (r['method'],r['mode'])==(control,control_mode) and r[metric] is not None}
                common=sorted(set(a)&set(b));edits=sorted({k[0] for k in common})
                effects=[macro([a[k] for k in common if k[0]==i],metric)-macro([b[k] for k in common if k[0]==i],metric) for i in edits]
                lo,hi=bootstrap(effects)
                images=defaultdict(list)
                for k in common:images[a[k]['source_image']].append(a[k][metric]-b[k][metric])
                cluster=[mean(v) for v in images.values()];cl,ch=bootstrap(cluster)
                loso=[]
                if len(images)>1:
                    for image in images:
                        kept=[k for k in common if a[k]['source_image']!=image]
                        remaining=[macro([a[k] for k in kept if k[0]==i],metric)-macro([b[k] for k in kept if k[0]==i],metric)
                                   for i in edits if any(k[0]==i for k in kept)]
                        loso.append(mean(remaining))
                output.append(dict(zip(keys,key),candidate=candidate,control=control,mode=candidate_mode,
                    control_mode=control_mode,metric=metric,paired_inputs=len(common),paired_edits=len(edits),
                    source_images=len(images),patients='UNKNOWN',delta=mean(effects) if effects else None,ci_low=lo,ci_high=hi,
                    input_delta_micro=mean(a[k][metric]-b[k][metric] for k in common) if common else None,
                    image_cluster_delta=mean(cluster) if cluster else None,image_cluster_ci_low=cl,image_cluster_ci_high=ch,
                    leave_one_image_out_min=min(loso) if loso else None,leave_one_image_out_max=max(loso) if loso else None,
                    sensitivity_scope='shared probe-source image; not independent patient proof',bootstrap_draws=10000,bootstrap_seed=20260908))
    return output


def history(rows):
    cells=defaultdict(list)
    for r in rows:
        if r['track']=='B':cells[r['method'],r['mode'],r['edit'],r['eqkey']].append(r)
    result=[]
    for (method,mode,edit,_),group in cells.items():
        first=min(group,key=lambda r:r['prefix']);last=next((r for r in group if r['prefix']==16),None)
        if last is None:continue
        for metric in ('semantic','base_correct_damage','fpr','other_expert_correct','wrong_given_on_base_correct'):
            a,b=first[metric],last[metric]
            result.append(dict(method=method,mode=mode,edit=edit,first_prefix=first['prefix'],last_prefix=16,
                stratum=last['stratum'],role_stable=first['strict_role']==last['strict_role'],metric=metric,
                retention_eligible=first['prefix']<16 and first['strict_role']==last['strict_role'],
                first=a,last=b,change=b-a if a is not None and b is not None and first['strict_role']==last['strict_role'] else None))
    return result


def finalize(args):
    run=args.run_root;public=run/'public';entries,ledger,costs=inventory(run)
    verdicts,side=f3.current_verdicts(run)
    if side is None:raise ValueError('Judge packet absent')
    protocol=side['protocol_sha256'];rows=details(entries,verdicts,protocol);tables=summarize(rows);effects=paired(rows)
    missing=len(set(side['all_expected'])-set(verdicts))
    csv_write(public/'WRITER_TRADEOFF_RESULTS.csv',[r for r in tables if r['track']=='A' and not r['is_diagnostic']])
    csv_write(public/'WRITER_PAIRED_EFFECTS.csv',[r for r in effects if r['track']=='A'])
    csv_write(public/'BANK_SCOPE_RESULTS.csv',[r for r in tables if r['track']=='B'])
    csv_write(public/'BANK_PAIRED_EFFECTS.csv',[r for r in effects if r['track']=='B'])
    csv_write(public/'BANK_MATCHED_HISTORY.csv',history(rows))
    csv_write(public/'ROUTE_FAILURE_DECOMPOSITION.csv',[r for r in tables if r['track']=='B'])
    csv_write(public/'INTERMEDIATE_T2G_DIAGNOSTICS.csv',[r for r in tables if r['is_diagnostic']])
    diagnostics=[]
    for e in read(run/'private/COHORTS.json')['edits']:
        path=Path(e['child'])/'private/tasks'/task_for(e['cohort'],e['event_index'])['task_id']/'STAGE4_DIAGNOSTICS_PRIVATE.json'
        if path.exists():
            for r in read(path)['rows']:
                row={k:v for k,v in r.items() if k not in ('behaviors','full_fit_kl','scale','normalized_fit_kl')}
                for g in ('H','U'):
                    row[g+'_raw_kl']=r['full_fit_kl'].get(g);row[g+'_normalized_kl']=r['normalized_fit_kl'].get(g)
                diagnostics.append(row)
    csv_write(public/'TRAINING_DIAGNOSTICS.csv',diagnostics);csv_write(public/'METHOD_COSTS.csv',costs)
    csv_write(public/'FIT_BASE_CORRECT_SUPPORT.csv',[r for r in tables if r['track']=='A' and r['method']=='BASE' and r['role']=='fit' and r['stratum'] in ('H','U')])
    thresholds=[]
    for prefix in (1,4,8,16):
        path=run/f'private/bank/prefix{prefix}/THRESHOLD_LOCK.json'
        if path.exists():thresholds.append(read(path))
    locks=read(public/'METHOD_AND_ROUTER_LOCKS.json');locks['router']['thresholds']=thresholds
    vf.atomic_json(public/'METHOD_AND_ROUTER_LOCKS.json',locks)
    closure=dict(status='COMPLETE_CURRENT_TUPLES' if not missing else 'PARTIAL',required=len(side['all_expected']),
        scored=len(verdicts),missing=missing,reused=side['reused'],new=side['new'],max_model_len=side['max_model_len'],
        execution_version_changed=side['execution_version_changed'],no_truncation=True)
    vf.atomic_json(public/'JUDGE_CLOSURE.json',closure)
    new=[r for r in ledger if r['track']=='A' and r['method']=='W01']
    complete=sum(r['complete'] for r in new);bank_complete=sum(r['complete'] for r in ledger if r['track']=='B')
    screen=read(run/'private/CONFIRMATION_SCREEN.json')
    status=dict(status='COMPLETE_EXECUTABLE_CONFIRMATION_UNAVAILABLE' if complete==23 and bank_complete==4 and not missing else 'PARTIAL',
        compute=dict(new_writer_complete=complete,new_writer_planned=23,reused_writer_complete=sum(r['complete'] for r in ledger if r['track']=='A' and r['method']!='W01'),
                     bank_prefix_complete=bank_complete,bank_prefix_planned=4),judge=closure,
        confirmation='CONFIRMATION_UNAVAILABLE',unsupported_confirmation=len(screen['candidates']),
        publication='PENDING_PUBLIC_PUSH_AND_ANONYMOUS_VERIFICATION',ledger=ledger)
    vf.atomic_json(public/'EXECUTION_STATUS.json',status)
    def effect(cohort,track,prefix,panel,candidate,control,mode,metric,role='evaluation'):
        return next((r for r in effects if (r['cohort'],r['track'],r['prefix'],r['stratum'],r['candidate'],r['control'],r['mode'],r['metric'],r['role'])==
            (cohort,track,prefix,panel,candidate,control,mode,metric,role)),None)
    def show(r):
        if not r or r['delta'] is None:return 'NA (no paired support)'
        ci='NA' if r['ci_low'] is None else f"[{100*r['ci_low']:.2f}, {100*r['ci_high']:.2f}] pp"
        return f"{100*r['delta']:+.2f} pp; CI {ci}; {r['paired_edits']} edits / {r['paired_inputs']} inputs / {r['source_images']} probe images"
    lines=['# MedTRACE Stage4 factual review','',f"Status: {status['status']}. Judge missing={missing}. Publication state here is the generator-time snapshot; verify the Git commit for later public delivery.",'',
        '1. Did weaker KL recover original T2G, and what protection was lost?','']
    for mode in ('R0','FORCED_ON'):
        for control in ('W0','W1'):
            lines.append(f"- OLD_STAGE3_COMMON7 {mode} original T2G W01-{control}: "+show(effect(OLD,'A',0,'T2G','W01',control,mode,'v4_primary','formal_development')))
    for panel in ('H','U'):
        for control in ('W0','W1'):
            lines.append(f"- SLAKE16 FORCED_ON {panel} damage W01-{control}: "+show(effect(BANK,'A',0,panel,'W01',control,'FORCED_ON','base_correct_damage')))
    lines.extend(['','2. Is the loss still present under FORCED_ON?','',
        'The separate FORCED_ON effects above isolate the writer from rejection; negative values remain a writer-side observed tradeoff, not a closed route. Intermediate checkpoints are diagnostic only.','',
        '3. Does RC reject non-target inputs with fixed support?',''])
    for panel in ('H','U','T0','source_style_confirmation','cross_family_confirmation'):
        for metric in (('fpr','base_correct_damage','semantic','base_wrong_became_correct') if panel in ('H','U') else ('semantic',)):
            r=effect(BANK,'B',16,panel,'W1','W1','RC',metric,'native' if panel=='T0' else 'evaluation')
            lines.append(f'- Fixed W1 RC-R0, prefix16 {panel} {metric}: '+show(r))
    lines.extend(['','4. Writer, rejection, and other-expert contributions','',
        'WRITER_PAIRED_EFFECTS isolates W01 at fixed R0/FORCED_ON. BANK_PAIRED_EFFECTS isolates rejection at fixed W1 and supplies W0/W01 interactions. ROUTE_FAILURE_DECOMPOSITION reports P(ON|strict Base), P(wrong|ON,Base-correct,strict Base), final damage, all-source accuracy and Base-wrong correction/change with separate denominators. Other-expert correctness is a descriptive attribution, not an independent causal contribution. Owner mismatch alone is not scored as wrong.','',
        '5. Independent confirmation','',f"CONFIRMATION_UNAVAILABLE: {len(screen['candidates'])} capped candidates, zero complete authorized new edits. Existing-source/fact-specific positive support and patient independence are not invented; see UNSEEN_CONFIRMATION_REPORT.md. The 7 and 16 development cohorts remain separate.",'',
        '6. Preregistered descriptive decisions',''])
    joint=[]
    t2=effect(OLD,'A',0,'T2G','W01','W0','R0','v4_primary','formal_development')
    joint.append(t2 is not None and t2['delta'] is not None and t2['delta']>=-.05)
    native_old=effect(OLD,'A',0,'T0','W01','W0','R0','semantic','native')
    joint.append(native_old is not None and native_old['delta'] is not None and native_old['delta']>=-.05)
    for panel in ('T0','source_style_confirmation','cross_family_confirmation'):
        r=effect(BANK,'A',0,panel,'W01','W0','FORCED_ON','semantic','native' if panel=='T0' else 'evaluation')
        joint.append(r is not None and r['delta'] is not None and r['delta']>=-.05)
    for panel,limit in (('U',-.10),('H',0.)):
        r=effect(BANK,'A',0,panel,'W01','W0','FORCED_ON','base_correct_damage')
        joint.append(r is not None and r['delta'] is not None and r['delta']<=limit)
    lines.append('DESCRIPTIVE_JOINT_CANDIDATE' if all(joint) else 'NO_DESCRIPTIVE_JOINT_WRITER_CANDIDATE: one or more fixed criteria failed or lacked support.')
    scope_checks=[]
    for panel in ('T0','source_style_confirmation','cross_family_confirmation'):
        r=effect(BANK,'B',16,panel,'W1','W1','RC','semantic','native' if panel=='T0' else 'evaluation')
        scope_checks.append(r is not None and r['delta'] is not None and r['delta']>=-.05)
    for panel in ('H','U'):
        for metric in ('fpr','base_correct_damage'):
            r=effect(BANK,'B',16,panel,'W1','W1','RC',metric)
            scope_checks.append(r is not None and r['delta'] is not None and r['delta']<0)
    lines.extend(['DESCRIPTIVE_SCOPE_CANDIDATE' if all(scope_checks) else 'NO_JOINT_SCOPE_IMPROVEMENT: see separate positive/activation/damage effects.',
        '', 'No independent confirmation supports a resolved-generalization, noninferiority, clinical-safety, intrinsic-routing or sequential-200 claim. No automatic Stage5. Threshold constraints are engineering constraints, not 95% guarantees.',
        '', 'Statistics: fixed 10,000 paired-edit bootstrap, seed 20260908; image-cluster and leave-one-image-out sensitivities accompany shared-source dependence. Cases/patients UNKNOWN. Zero denominators NA. Native diagnostics and constructed confirmation are not official T0/T2G substitutes. All-source correctness, Base-wrong correction and Base-correct damage remain side by side.',
        '', 'Stage3 continuity: 7 common groups include six formal T0 anchors; original W1 T2G macro 30.56%, paired W1-W0 -52.78 pp also under FORCED_ON, and T1L damage 100% on six Base-correct inputs. Stage3 was complete before this run; its static pending-publication wording did not cause a rerun.'])
    vf.atomic_text(public/'GPT_PRO_REVIEW.md','\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in status.items() if k!='ledger'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare-judge','finalize'])
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--max-model-len',type=int,default=2048)
    a=p.parse_args();install()
    f3.prepare_judge(a) if a.action=='prepare-judge' else finalize(a)
