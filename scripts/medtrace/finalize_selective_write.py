#!/usr/bin/env python3
"""Complete-answer fixed Judge, calibration-only lambda lock, edit-cluster summaries."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.medtrace.run_selective_write import read, vf, CONDITIONS, RELATIONS, summarize
from methods.medtrace.selective_write import select_global_lambda


def csv_write(path,rows):
    if not rows:
        vf.atomic_text(path, "status\nNO_COMPLETE_RESULTS\n")
        return
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def results(run):
    tasks=read(run/'private/TASK_QUEUE.json')['tasks']
    out=[]
    for task in tasks:
        if task['status'] not in {'RAW_READY','JUDGED'}:
            continue
        path=run/'private/tasks'/task['task_id']/'result_private.json'
        value=read(path)
        if value['status']!='RAW_READY' or value['step']!=320 or value['task']['task_id']!=task['task_id']:
            raise ValueError('incomplete task cannot enter semantic closure')
        if not value['base_guard']['unchanged'] or not all(x['disabled_parity'] for x in value['outputs'].values()):
            raise ValueError('failed Base guard or disabled parity')
        out.append(value)
    for i in sorted({r['task']['event_index'] for r in out}):
        reference=read(run/f'private/initial/e{i:02d}/reference_private.json')
        reference['task'].update(task_id=f'A2_e{i:02d}',parameterization='REFERENCE')
        out.append(reference)
    return out


def opaque(question,reference,raw,protocol):
    return vf.sha256_json((question,reference,raw,protocol))


def prepare_judge(args):
    values=results(args.run_root)
    if not values:
        raise RuntimeError('no complete 320-step results to Judge')
    config=read(args.run_root/'private/CAMPAIGN_CONFIG.json')
    protocol=read(Path(config['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json')
    tuples={}
    for result in values:
        data=read(args.run_root/f"private/edits/e{result['task']['event_index']:02d}.json")
        target=data['event']['edit_record']['gold_answer']
        for item in result['outputs'].values():
            row=item['row']
            for reference in {row['reference'],target}:
                for branch in ('base','forced','fixed'):
                    raw=item[branch]['raw_answer']
                    key=opaque(row['question'],reference,raw,protocol['config_sha256'])
                    tuples[key]=dict(opaque_query_id=key,question=row['question'],gold_answer=reference,
                                     raw_base_answer=raw,adjudication_pass=1)
    judge=args.run_root/'private/judge'
    judge.mkdir(parents=True,exist_ok=True)
    packet=judge/'JUDGE_PACKET_PRIVATE.jsonl'
    if packet.exists():
        raise FileExistsError('one-shot packet already exists; do not silently re-judge')
    vf.atomic_text(packet,''.join(json.dumps(row,sort_keys=True)+'\n' for row in tuples.values()))
    vf.atomic_json(judge/'JUDGE_SIDECAR_PRIVATE.json',dict(protocol_sha256=protocol['config_sha256'],
        snapshot=protocol['judge_snapshot_sha'],expected=sorted(tuples),complete_answers=True,
        reuse='no historical execution-lock substitution; exact tuples deduplicated within this run only',
        packet_sha256=vf.sha256_file(packet),task_ids=[v['task']['task_id'] for v in values]))


def stratum(row):
    if row['fact_relation']==RELATIONS['H']:
        return 'H'
    if row['fact_relation']==RELATIONS['U']:
        return 'U'
    if row['role']=='challenge':
        return 'same_image_other_fact_challenge'
    if row['role']=='formal_development':
        return row.get('task',row['fact_relation'])
    if row.get('fit_positive_source')=='HISTORICAL_SCOPE_FIT_ONLY':
        return 'scope_fit_positive_not_used_for_positive_CE'
    return 'positive' if row['label']=='positive' else 'unknown_relation'


def hierarchical(rows,key):
    groups=defaultdict(dict)
    for row in rows:
        if row[key] is not None:
            groups[row['source_group']][row['eqkey']]=row[key]
    return mean(mean(values.values()) for values in groups.values()) if groups else None


def snapshot(args):
    """Report real checkpoint progress without pretending endpoint/Judge closure."""
    summarize(args)
    tasks=read(args.run_root/'private/TASK_QUEUE.json')['tasks']
    rows=[]
    for task in tasks:
        directory=args.run_root/'private/tasks'/task['task_id']
        row={k:task[k] for k in ('task_id','event_index','parameterization','condition','status')}
        row.update(observed_optimizer_step=None,H_fit_kl=None,U_fit_kl=None,raw_endpoint_ready=False,semantic_status='NOT_JUDGED')
        histories=list(directory.glob('attempt*/training_private.json'))
        if histories:
            history=read(max(histories,key=lambda p:p.stat().st_mtime))
            diagnostic=history['diagnostics'][-1]
            row.update(observed_optimizer_step=diagnostic['step'],H_fit_kl=diagnostic['full_fit_kl']['H'],U_fit_kl=diagnostic['full_fit_kl']['U'])
        row['raw_endpoint_ready']=(directory/'result_private.json').is_file() and task['status'] in {'RAW_READY','JUDGED'}
        if task['status']=='JUDGED':row['semantic_status']='JUDGED'
        rows.append(row)
    csv_write(args.public_dir/'SELECTIVE_WRITE_BY_EDIT.csv',rows)
    counts={s:sum(t['status']==s for t in tasks) for s in sorted({t['status'] for t in tasks})}
    vf.atomic_json(args.public_dir/'LIVE_PROGRESS.json',dict(counts=counts,expected=70,
        trained_320=sum(r['observed_optimizer_step']==320 for r in rows),raw_ready=sum(r['raw_endpoint_ready'] for r in rows),
        scientific_gain='NOT_EVALUATED',publication_scope='IN_PROGRESS_SNAPSHOT'))


def finalize(args):
    run,public=args.run_root,args.public_dir
    values=results(run)
    sidecar=read(run/'private/judge/JUDGE_SIDECAR_PRIVATE.json')
    execution=read(run/'private/judge/JUDGE_EXECUTION_LOCK_PRIVATE.json')
    if execution['legacy_semantic_protocol_sha256']!=sidecar['protocol_sha256'] or execution['model']['snapshot']!=sidecar['snapshot']:
        raise ValueError('Judge execution protocol drift')
    if execution['packet']['sha256']!=sidecar['packet_sha256']:
        raise ValueError('Judge packet identity drift')
    judgements=vf.read_jsonl(run/'private/judge/JUDGE_OUTPUT_PRIVATE.jsonl')
    verdicts={}
    for row in judgements:
        if (not row.get('parse_valid') or type(row.get('is_correct')) is not bool
                or row.get('legacy_semantic_protocol_sha256')!=sidecar['protocol_sha256']
                or row.get('judge_snapshot_sha')!=sidecar['snapshot'] or row['opaque_query_id'] in verdicts):
            raise ValueError('invalid, duplicated or foreign Judge verdict')
        verdicts[row['opaque_query_id']]=row['is_correct']
    if set(verdicts)!=set(sidecar['expected']):
        raise ValueError('incomplete full-answer Judge coverage')
    semantic=lambda row,reference,raw: verdicts[opaque(row['question'],reference,raw,sidecar['protocol_sha256'])]
    detailed=[]
    frozen_gates={}
    for result in values:
        task=result['task'];edit=task['event_index']
        data=read(run/f'private/edits/e{edit:02d}.json')
        target=data['event']['edit_record']['gold_answer']
        gates={lid:item['fixed_on'] for lid,item in result['outputs'].items()}
        if edit in frozen_gates and frozen_gates[edit]!=gates:
            raise ValueError('fixed-router decisions/FPR changed across write conditions')
        frozen_gates[edit]=gates
        for item in result['outputs'].values():
            row=item['row']
            base_correct=semantic(row,row['reference'],item['base']['raw_answer'])
            for mode,branch in (('FORCED_ON','forced'),('FIXED_ROUTER','fixed'),('DISABLED','base')):
                raw=item[branch]['raw_answer']
                correct=semantic(row,row['reference'],raw)
                kl=item['kl'] if mode=='FORCED_ON' or mode=='FIXED_ROUTER' and item['fixed_on'] else (0. if item['kl'] is not None else None)
                detailed.append(dict(edit=edit,parameterization=task['parameterization'],condition=task['condition'],
                    mode=mode,role=row['role'],stratum=stratum(row),eqkey=row['eqkey'],source_group=row['source_group'],
                    semantic=float(correct),exact=float(vf.normalize_medical_answer(raw)==vf.normalize_medical_answer(row['reference'])),
                    token_parity=float(item[branch]['raw_token_ids']==item['base']['raw_token_ids']),
                    kl=kl,base_correct=float(base_correct),base_correct_preserved=float(correct) if base_correct else None,
                    base_correct_damage=float(not correct) if base_correct else None,
                    base_wrong_became_correct=float(correct) if not base_correct else None,
                    base_wrong_changed=float(item[branch]['raw_token_ids']!=item['base']['raw_token_ids']) if not base_correct else None,
                    target_consistency=float(semantic(row,target,raw)),fixed_on=float(item['fixed_on']),
                    source_unknown=float(stratum(row)=='unknown_relation')))
    keys=('edit','parameterization','condition','mode','role','stratum')
    metrics=('semantic','exact','token_parity','kl','base_correct','base_correct_preserved','base_correct_damage',
             'base_wrong_became_correct','base_wrong_changed','target_consistency','fixed_on','source_unknown')
    cells=defaultdict(list)
    for row in detailed:
        cells[tuple(row[k] for k in keys)].append(row)
    by_edit=[]
    for key,rows in sorted(cells.items()):
        cell=dict(zip(keys,key));cell.update(n=len(rows),source_groups=len({r['source_group'] for r in rows}))
        for metric in metrics:
            cell[metric+'_group_mean']=hierarchical(rows,metric)
            present=[r[metric] for r in rows if r[metric] is not None]
            cell[metric+'_micro']=mean(present) if present else None
            cell[metric+'_den']=len(present)
        by_edit.append(cell)
    csv_write(public/'SELECTIVE_WRITE_BEHAVIOR_BY_EDIT.csv',by_edit)
    macro_cells=defaultdict(list)
    for row in by_edit:
        macro_cells[tuple(row[k] for k in keys[1:])].append(row)
    macros=[]
    for key,rows in sorted(macro_cells.items()):
        item=dict(zip(keys[1:],key));item.update(edits=len(rows),n=sum(r['n'] for r in rows))
        for metric in metrics:
            present=[r[metric+'_group_mean'] for r in rows if r[metric+'_group_mean'] is not None]
            item[metric+'_macro']=mean(present) if present else None
        macros.append(item)
    csv_write(public/'SELECTIVE_WRITE_BEHAVIOR_MACRO.csv',macros)

    # This view is created from calibration rows only. Evaluation never reaches the selector.
    cal=[r for r in by_edit if r['role']=='calibration' and r['mode']=='FORCED_ON']
    selections={}
    expected=[1,2,3,7,13,15,16]
    for parameterization in ('P4','L16'):
        table=[]
        for edit in expected:
            row=dict(edit=edit,role='calibration')
            for condition in ('A2',*CONDITIONS[:4]):
                p='REFERENCE' if condition=='A2' else parameterization
                cells={v['stratum']:v for v in cal if v['edit']==edit and v['condition']==condition and v['parameterization']==p}
                if not {'positive','H','U'}<=set(cells):
                    break
                row[condition]=dict(positive_semantic=cells['positive']['semantic_group_mean'],
                                    H_kl=cells['H']['kl_group_mean'],U_kl=cells['U']['kl_group_mean'])
            else:
                table.append(row)
        selections[parameterization]=select_global_lambda(table) if len(table)==7 else dict(lambda_value=None,qualified=False,status='INCOMPLETE_CALIBRATION',edits=len(table))
    selection_path=run/'private/SELECTED_W1.json'
    if selection_path.exists() and read(selection_path)!=selections:
        raise ValueError('cannot retune an existing global lambda lock')
    vf.atomic_json(selection_path,selections)
    vf.atomic_json(public/'GLOBAL_W1_SELECTION.json',selections)

    comparisons=[]
    for p in ('P4','L16'):
        selected=selections[p].get('lambda')
        controls=[CONDITIONS[0]]+([f'W1_KL_{selected:g}'] if selected is not None else [])
        for control in controls:
            for candidate in (CONDITIONS[-1],*([f'W1_KL_{selected:g}'] if selected is not None and control==CONDITIONS[0] else [])):
                for role,group,metric in (('evaluation','positive','semantic'),('evaluation','H','kl'),('evaluation','U','kl'),
                                           ('evaluation','H','base_correct_damage'),('evaluation','U','base_correct_damage')):
                    def mapping(condition):
                        return {r['edit']:r[metric+'_group_mean'] for r in by_edit if r['parameterization']==p and r['condition']==condition
                                and r['mode']=='FORCED_ON' and r['role']==role and r['stratum']==group and r[metric+'_group_mean'] is not None}
                    a,b=mapping(candidate),mapping(control);shared=set(a)&set(b)
                    ci=vf._paired_ci(a,b) if len(shared)>=2 else (None,None)
                    comparisons.append(dict(parameterization=p,candidate=candidate,control=control,role=role,stratum=group,metric=metric,
                        paired_edits=len(shared),delta=mean(a[e]-b[e] for e in shared) if shared else None,ci_low=ci[0],ci_high=ci[1],
                        W1_qualified=selections[p]['qualified']))
    csv_write(public/'PAIRED_EDIT_EFFECTS.csv',comparisons)
    diagnostics=[]
    for result in values:
        if result['task']['condition']=='A2':continue
        for point in result['diagnostics']:
            for group in ('H','U'):
                diagnostics.append(dict(task_id=result['task']['task_id'],step=point['step'],group=group,
                    kl=point['full_fit_kl'][group],epsilon=point['epsilon'][group],scale=point['scale'][group],
                    residual=point['constraint_residual'][group],dual=point['dual'][group],saturation=point['saturation'][group]))
    csv_write(public/'CONSTRAINT_TRAJECTORIES.csv',diagnostics)
    queue=vf.TaskQueue(run/'private/TASK_QUEUE.json',run)
    for result in values:
        if result['task']['condition']!='A2':
            queue.update(result['task']['task_id'],'JUDGED')
    tasks=queue.snapshot()['tasks'];completed=sum(t['status']=='JUDGED' for t in tasks)
    summarize(args)
    status=dict(status='STAGE1_COMPLETE' if completed==70 else 'PARTIAL_RESULTS',expected=70,judged=completed,
        judge='COMPLETE_FOR_AVAILABLE_RESULTS',evaluation='COMPLETE_FOR_AVAILABLE_RESULTS',novel_n=0,novel_confirmation='NOT_RUN_NO_AUTHORIZED_UNSEEN_COHORT',
        scientific_gain='SEE_PAIRED_EFFECTS_NOT_A_PASS_COUNT',old_scientific_gain=False,global_W1=selections,
        fixed_router_identity='PASS',publication='PENDING_LOCAL_PUBLICATION')
    vf.atomic_json(public/'RUN_COMPLETION.json',status)
    lines=['# Selective-write results','',f'Judged trajectories: {completed}/70. New-edit confirmation N=0; no new-edit replication claim.',
        '',f'Calibration-only W1 selection: `{json.dumps(selections,sort_keys=True)}`.',
        '', 'FORCED_ON is primary. FIXED_ROUTER decisions are invariant across all write conditions; its FPR cannot improve here. DISABLED was replayed against Base. Full raw answers, not excerpts, were sent to the locked Judge.',
        '', 'See SELECTIVE_WRITE_BEHAVIOR_MACRO.csv and SELECTIVE_WRITE_BEHAVIOR_BY_EDIT.csv for separate native/fit/calibration/evaluation, T1G/T2G/T1L, H/U and challenge strata. Source-image macro and row micro denominators are explicit. See PAIRED_EDIT_EFFECTS.csv for W1/W2 minus W0 and W2 minus selected W1, with edit-cluster bootstrap intervals; missing paired edits are reported, not imputed.',
        '', 'W1 ordinary KL benefit and W2 incremental benefit must be read separately. A low fit KL alone is insufficient; check held-out positive preservation and negative behavior together. Cross-P4/L16 differences are joint capacity/parameterization/optimization evidence, not capacity-matched proof. Multipliers at cap or positive constraint residuals mean the prescribed fit constraints were not reached.',
        '', 'Full-answer target consistency is reported separately from source-reference correctness. Base-correct damage and Base-wrong changes have distinct denominators. Maintaining an already-wrong Base answer is not clinical correction. No novelty, full TIME/LiveEdit/M-ORE reproduction, SOTA or clinical safety claim.']
    vf.atomic_text(public/'SELECTIVE_WRITE_REPORT.md','\n'.join(lines)+'\n')
    vf.atomic_text(public/'GPT_PRO_REVIEW.md','\n'.join(['# GPT Pro review','',*lines[2:], '',
        'Review questions: (1) Does calibrated ordinary KL improve forced-on behavior without losing positive semantics? (2) Does W2 add a paired benefit over that W1? (3) Is any effect confined to L16, with the capacity confound disclosed? (4) New edits have NOT been tested; do not treat seven viewed facts as replication.'])+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare-judge','finalize','snapshot'))
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--public-dir',type=Path,required=True)
    args=p.parse_args()
    {'prepare-judge':prepare_judge,'finalize':finalize,'snapshot':snapshot}[args.action](args)


if __name__=='__main__':main()
