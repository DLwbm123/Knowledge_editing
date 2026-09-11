"""One bounded support-only review on existing episode training images."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace import stage13r as prior
from scripts.medtrace.stage13r_sources import canonical
from scripts.medtrace.prepare_stage2_sources import normalized, reviewed_attribute
vf,read=prior.vf,prior.read
PUBLIC_BASE='f22b56d6b9117d06784c4f355e30278c1397ba2e'


def image_key(row):
    return canonical(row['dataset'],row['image_path'])


def input_key(row):
    return image_key(row),normalized(row['question'])


def answer_type(row):
    return 'YES_NO' if normalized(row['reference']) in ('yes','no') else 'OPEN'


def select_support(task,pool,blocked_images,blocked_qa,u_inputs,token_length,seed):
    rows={r['logical_id']:r for r in task['data']['rows']}
    train_images={image_key(r) for r in rows.values() if r['role'] in ('native','fit')}-blocked_images
    forbidden={reviewed_attribute(rows[i]['question']) for i in [task['native_id'],*task['h_ids']]}
    if None in forbidden:raise ValueError('unknown native/H proposition')
    candidates=[r for r in pool if image_key(r) in train_images
        and (r['dataset'],str(r['source_qid'])) not in blocked_qa and input_key(r) not in u_inputs
        and reviewed_attribute(r['question']) is not None and reviewed_attribute(r['question']) not in forbidden]
    candidates.sort(key=lambda r:(r['dataset'],r['image_id'],str(r['source_qid'])))
    random.Random(seed).shuffle(candidates)
    if not candidates:return [],{},'NO_LEGAL_TRAIN_IMAGE' if not train_images else 'NO_LEGAL_OTHER_PROPOSITION_QA'
    selected=[];mapping={};metadata=[]
    for hid in task['h_ids']:
        h=rows[hid]
        g=min(candidates,key=lambda r:(image_key(r)!=image_key(h),answer_type(r)!=answer_type(h),abs(token_length(r)-token_length(h))))
        identity=(g['dataset'],str(g['source_qid']))
        existing=next((r for r in selected if (r['dataset'],str(r['source_qid']))==identity),None)
        if existing is None:
            existing=dict(g,logical_id='extra-fit-'+str(len(selected)),role='fit',label='negative',negative_group='G',fact_relation='other_source_proposition')
            selected.append(existing)
        mapping[hid]=existing['logical_id']
        metadata.append(dict(H_id=hid,G_id=existing['logical_id'],tier='SAME_H_IMAGE' if image_key(g)==image_key(h) else 'DIFFERENT_TRAIN_IMAGE',
            H_answer_type=answer_type(h),G_answer_type=answer_type(g),H_answer_tokens=token_length(h),G_answer_tokens=token_length(g),
            H_attribute=reviewed_attribute(h['question']),G_attribute=reviewed_attribute(g['question'])))
    assert len(selected)<=len(task['h_ids'])
    return selected,dict(H_to_G=mapping,slots=metadata),None


def prepare(args):
    began=time.time();run=args.run_root;old=args.stage13r_run;cfg=read(old/'private/CAMPAIGN_CONFIG.json');r12=Path(cfg['stage12_run'])
    if run.exists():raise FileExistsError('one frozen support overlay; preserve existing run')
    assert read(old/'RUN_COMPLETION.json')['completed_writers']==21
    assert read(old/'public/PUBLICATION_RECEIPT.json')['public_sha']==PUBLIC_BASE
    assert read(r12/'RUN_COMPLETION.json')['training_completed']==15
    oldtasks=read(r12/'private/TASKS.json');newtasks=read(old/'private/paired/private/TASKS.json')
    assert len(oldtasks)==15 and len(newtasks)==7
    # Role metadata only during selection; no student responses or Judge outcomes.
    role_rows=[r for t in oldtasks for r in t['data']['rows']]
    role_rows += [r for t in newtasks for r in t['data']['rows']]
    for t in newtasks:role_rows+=read(old/('private/evaluation_complete/e%02d.json'%t['order']))
    evaluation=[r for r in role_rows if r['role'] not in ('native','fit')]
    blocked_images={image_key(r) for r in evaluation}
    blocked_qa={(r['dataset'],str(r['source_qid'])) for r in evaluation if 'source_qid' in r}
    u_inputs={input_key(r) for t in oldtasks+newtasks for r in t['data']['rows'] if r['logical_id'] in t['u_ids']}
    overlay=read(old/'private/SOURCE_OVERLAY.json')
    permitted={image_key(r) for t in oldtasks+newtasks for r in t['data']['rows'] if r['role'] in ('native','fit')}-blocked_images
    # Existing Stage13R overlay already carries original reserved/quality/identity exclusions.
    pool=[r for r in overlay['rows'] if image_key(r) in permitted and overlay['image_roles'][r['source_group']]=='adaptation']
    assert {image_key(r) for r in pool}==permitted, 'unsupported source identity requires review, not an expanded search'
    raw_path=Path(pool[0]['source_file']);raw={str(r['qid']):r for r in read(raw_path) if r.get('q_lang')=='en' and canonical('SLAKE',r['img_name']) in permitted}
    for row in pool:
        source=raw[str(row['source_qid'])]
        assert (source['question'],source['answer'])==(row['question'],row['reference']) and Path(row['image_path']).is_file()
    from transformers import AutoTokenizer
    from scripts.medtrace.coordinate_selective_write import MODEL
    import torch
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    token_length=lambda r:len(tokenizer.encode(r['reference'],add_special_tokens=False))+1
    tasks=[];support=[];private_support=[]
    for cohort,ts in (('OLD15',oldtasks),('NEW7',newtasks)):
        for source in ts:
            state=torch.load(source['checkpoint'],map_location='cpu',weights_only=True);assert state['step']==320
            g,selection,reason=select_support(source,pool,blocked_images,blocked_qa,u_inputs,token_length,state['task']['seed'])
            order=source['order']+(1000 if cohort=='NEW7' else 0)
            row=dict(cohort=cohort,edit=source['event_index'],order=order,status='SUPPORTED' if g else 'UNSUPPORTED',reason=reason,
                selected_H_QA=len(source['h_ids']),selected_G_QA=len(g),tier='SAME_H_IMAGE' if g and all(s['tier']=='SAME_H_IMAGE' for s in selection['slots']) else 'DIFFERENT_TRAIN_IMAGE' if g else 'UNSUPPORTED',
                slots=[{k:v for k,v in s.items() if not k.endswith('_id')} for s in selection.get('slots',[])])
            support.append(row)
            private_support.append(dict(row,record=source['data']['event']['edit_record']['record_id'],checkpoint=source['checkpoint'],selection=selection,G=g,
                excluded_training_images=sorted({image_key(r) for r in source['data']['rows'] if r['role'] in ('native','fit')}&blocked_images)))
            if not g:continue
            parent=old/('private/paired/private/edits/e%02d/C_FACT'%source['order']) if cohort=='NEW7' else Path(cfg['stage11_run'])/('private/edits/e%02d/J1_FREE_R4'%source['order'])
            parent_result=read(parent/'result.json')
            inherited_schedule=[{k:r[k] for k in ('step','fit_id','H_id','U_id')} for r in parent_result['curve']]
            t=deepcopy(source);training_ids={t['native_id'],*t['fit_positive_ids'],*t['h_ids'],*t['u_ids']}
            t['data']['rows']=[r for r in t['data']['rows'] if r['logical_id'] in training_ids]+g;t['data']['event']['probes']=[]
            t.update(order=order,cohort=cohort,source_order=source['order'],source_task_file=str(old/'private/paired/private/TASKS.json' if cohort=='NEW7' else r12/'private/TASKS.json'),
                stage11_status='PENDING',g_ids=[r['logical_id'] for r in g],H_to_G=selection['H_to_G'],schedule=inherited_schedule,
                teacher_root=str(old/'private/paired' if cohort=='NEW7' else Path(cfg['stage9_run'])),teacher_order=source['order'],seed=state['task']['seed'],
                parent_fact_result=str(parent/'result.json'),source_interface=str(old/('private/interface/private/edits/e%02d/result.json'%source['order']) if cohort=='NEW7' else Path(cfg['stage8_run'])/('private/edits/e%02d/result.json'%source['order'])))
            assert all(r['role'] in ('native','fit') for r in t['data']['rows']) and len(inherited_schedule)==320
            tasks.append(t)
    assert time.time()-began<=45*60
    assert shutil.disk_usage(run.parent).free>20*1024**3
    run.mkdir();probe=run/'.probe';probe.write_text('run');assert probe.read_text()=='run';probe.unlink()
    cfg.update(kind='MEDTRACE_STAGE14',stage13r_run=str(old),campaign_epoch=time.time(),code_commit=args.commit,
        public_baseline=PUBLIC_BASE,allowed_physical_gpus=[1],worker_gpus=[1],judge_gpu=1,forbidden_physical_gpus=[0,2,3],
        authorization='USER_STAGE14_GPU1',wall_hours=4,gpu_hours=8,train_seconds=3*3600,stage8_run=str(run/'private/interface'))
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json',cfg);vf.atomic_json(run/'private/TASKS.json',tasks)
    vf.atomic_json(run/'private/SUPPORT_OVERLAY.json',dict(frozen_before_training=True,source_provenance=overlay['provenance'],rows=private_support,
        all_cohort_evaluation_images=sorted(blocked_images),fixed_U_inputs=sorted(u_inputs),source_review_seconds=time.time()-began,
        reviewed_only_existing_training_images=sorted(permitted),selection_used_student_outputs=False))
    vf.atomic_json(run/'public/SUPPORT_FREEZE.json',dict(cohorts={'OLD15':15,'NEW7':7},actual_new_trajectories=len(tasks),rows=support,
        source_review_seconds=time.time()-began,review_level='agent original-source consistency; no human clinical certification',
        source_images_reviewed=len(permitted),source_QAs_reviewed=len(pool),evaluation_image_exclusion_includes_calibration_and_positive_paraphrase_images=True,
        old_results_retained=True,selection='same H image, answer type, token distance, deterministic source order with inherited seed tie break',
        token_match_measure='original answer tokenizer length plus EOS; actual training token counts reported separately'))
    vf.atomic_text(run/'private/PROTOCOL.md',args.protocol.read_text())
    # Selection is frozen. Import bound old outputs only now; never regenerate them.
    reused=[e for e in prior.stage12.inventory(r12)[0] if e['route_mode']==prior.stage12.MODES[0]]
    reused=[dict(e,cohort_name='OLD15',common_support=False,diagnostic_step=None) for e in reused]
    reused += [dict(e,cohort_name='NEW7',common_support=False,diagnostic_step=None) for e in prior.inventory(old)[0]]
    vf.atomic_json(run/'private/REUSED_ENTRIES.json',reused)
    for t in tasks:
        olddata=read(t['source_interface']);ids={r['logical_id'] for r in t['data']['rows']}-set(t['g_ids'])
        olddata['entries']=[e for e in olddata['entries'] if e['method']=='B0' and e['item']['row']['logical_id'] in ids]
        assert {e['item']['row']['logical_id'] for e in olddata['entries']}==ids
        vf.atomic_json(run/('private/interface/private/edits/e%02d/result.json'%t['order']),olddata)
    vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='PREPARED',actual_N=len(tasks),publication='PENDING'))
    print('FROZEN',len(tasks),dict(Counter(r['cohort']+':'+r['status'] for r in support)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--stage13r-run',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--protocol',type=Path,required=True);prepare(p.parse_args())
