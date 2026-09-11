#!/usr/bin/env python3
"""Full raw-train overlay; old manifests and unused draft queues remain read-only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
from scripts.medtrace.prepare_stage2_sources import read, image_groups, identity, normalized, source_review, reviewed_attribute, write_new

RELEASE = 'a9083ce6c34ac3ffb17671a605962924d8a8f9e9'
TRAIN_BLOB = 'afb9b3d634173aaccdb974f6b3d2754fb105c203'
SEED = 20260910


def canonical(dataset, name):
    p = Path(name)
    if dataset == 'SLAKE':
        # Original and project-created source variants share the verified xmlab ID.
        if p.parent.name.startswith('xmlab') and p.name.startswith('source'):
            return dataset, p.parent.name.lower()
        if p.name.startswith('xmlab'):
            return dataset, p.name.split('_')[0].split('::')[0].lower()
        raise ValueError('UNKNOWN SLAKE image identity: '+name)
    if dataset == 'VQA-RAD' and p.name.lower().startswith('synpic'):
        return dataset, p.name.lower()
    raise ValueError('UNKNOWN source identity')


def partition(groups, strata):
    parts = {}
    for category in sorted(set(strata.values())):
        ordered = sorted(g for g in groups if strata[g] == category)
        random.Random(str(SEED)+'\0'+category).shuffle(ordered)
        for index, group in enumerate(ordered):
            parts[group] = 'evaluation' if index % 3 == 0 else 'adaptation'
    return parts


def conflict(a, b):
    if a['source_group'] == b['source_group'] or normalized(a['question']) != normalized(b['question']):
        return False
    attr = reviewed_attribute(a['question'])
    x, y = normalized(a['reference']), normalized(b['reference'])
    if not attr or x == y:
        return False
    if attr.startswith('presence:'):
        return {x, y} == {'yes', 'no'}
    domains = {'body_region': {'head', 'chest', 'abdomen', 'neck', 'pelvic cavity'},
               'mr_weighting': {'t1', 't2'},
               'largest_visible_organ': {'lung', 'liver', 'small bowel', 'kidney', 'brain'}}
    return attr in domains and {x, y} <= domains[attr]


def collect(root):
    outputs = root/'Knowledge_editing/outputs'
    a = read(outputs/'medtrace_stage2_20260908_r02/private/NEW_EPISODE_MANIFEST_PRIVATE.json')
    b = read(outputs/'medtrace_stage5_20260909_r01/private/NEW_EPISODE_MANIFEST_PRIVATE.json')
    queue = read(root/'medtrace_runs/20260906T030312Z/private/TASK_QUEUE.json')['tasks']
    assert len(queue) == 120 and all(t['status'] == 'COMPLETE' for t in queue)
    assert read(outputs/'medtrace_stage2_20260908_r02/RUN_COMPLETION.json')['new_n'] == 16
    assert read(outputs/'medtrace_stage5_20260909_r01/RUN_COMPLETION.json')['completed_writers']['BE'] == 32
    actual, evidence = set(), []
    valid = ['20260906T030312Z/private/frozen_data.json',
             '20260905T134727Z/private/frozen_data.json',
             '20260905T125112Z/track_b/roles_private.json']
    for rel in valid:
        p = root/'medtrace_runs'/rel
        groups = image_groups(read(p)); actual |= groups
        evidence.append(dict(path=str(p), kind='COMPLETED_STUDENT_OR_ROUTER_DEVELOPMENT', images=len(groups)))
    for name in ('invalid_pre_hardfix', 'invalid_pre_role_balance'):
        p = root/'medtrace_runs/20260906T030312Z/private'/(name+'_TASK_QUEUE.json')
        tasks = read(p)['tasks']
        assert len(tasks) == 120 and all(t['status'] == 'PENDING' and t['attempts'] == 0 for t in tasks)
        evidence.append(dict(path=str(p), kind='UNEXECUTED_DRAFT_NOT_ACTUAL_EXPOSURE', tasks=len(tasks)))
    for p in (root/'medtrace_runs').glob('*/private/INPUT_MANIFEST.json'):
        actual |= {('VQA-RAD' if 'VQA-RAD' in i or identity(i).startswith('synpic') else 'SLAKE', identity(i))
                   for i in read(p).get('images', [])}
    for episodes in (a['episodes'], b['episodes']):
        actual |= image_groups(episodes)
    # Later stages' actual edit payloads; not source pools, prediction-only caches, or draft native lists.
    later = list(outputs.glob('medtrace_stage*/private/edits/e*.json'))
    for p in later:
        actual |= image_groups(read(p))
    evidence.append(dict(kind='STAGE2_TO12_EDIT_PAYLOADS', files=len(later)))
    ignored = {'historical_medtrace_all_roles', 'historical_medtrace_input_images', 'stage1_all_roles', 'existing_image_hash_identity_closure'}
    reserved = {(r['dataset'], r['image_id']): set(r['reasons'])-ignored
                for r in a['exposure_exclusion_ledger'] if set(r['reasons'])-ignored}
    catalog = outputs/'m3bench_data_runtime_finalization_v3/20260904T014138Z/data_static/STATIC_QUERY_INVENTORY.jsonl'
    hashes, query_index = defaultdict(set), defaultdict(list)
    # Existing identity index: no new full image hash scan, no sealed answer file opened.
    for line in catalog.open():
        r = json.loads(line); group = r['dataset'], identity(r['image_id'])
        if r.get('image_sha256'): hashes[r['image_sha256']].add(group)
        query_index[(group, normalized(r['question']), normalized(r['gold_answer']))].append(r['query_id'])
    blocked_hashes = {h for h, groups in hashes.items() if groups & (actual | reserved.keys())}
    closure = set().union(*(hashes[h] for h in blocked_hashes))
    digest_by_group = {g: h for h, groups in hashes.items() for g in groups}
    source = root/'DataP/knowledge_editing/data/m3bench'
    slake_file = source/'SLAKE/train.json'; raw_bytes = slake_file.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(raw_bytes)).encode()+b'\0'+raw_bytes).hexdigest()
    assert blob == TRAIN_BLOB, 'Local train differs from author revision; do not silently revise answers'
    slake = json.loads(raw_bytes)
    vqa = read(source/'VQA-RAD/VQA_RAD Dataset Public.json')
    summaries, pool = {}, []
    for dataset, allrows in (('SLAKE', slake), ('VQA-RAD', vqa)):
        rows = allrows if dataset == 'SLAKE' else [r for r in allrows if r['phrase_type'] in ('freeform', 'para')]
        groups = {canonical(dataset, r['img_name'] if dataset == 'SLAKE' else r['image_name']) for r in rows}
        missing = 0
        for r in rows:
            name = r['img_name'] if dataset == 'SLAKE' else r['image_name']
            group = canonical(dataset, name)
            path = source/('SLAKE/imgs/'+name if dataset == 'SLAKE' else 'VQA-RAD/images/'+name)
            missing += not path.is_file()
            if group in actual | reserved.keys() | closure or (dataset == 'SLAKE' and r['q_lang'] != 'en'):
                continue
            if group not in digest_by_group: raise ValueError('Candidate lacks existing identity binding')
            pool.append(dict(dataset=dataset, source_qid=r['qid'], image_id=group[1], image_path=str(path),
                question=r['question'], reference=r['answer'], source_group=dataset+':'+RELEASE+':'+group[1],
                source_release=RELEASE if dataset == 'SLAKE' else 'existing_original_train', source_role='train',
                image_sha256=digest_by_group[group], patient_id='UNKNOWN', source_file=str(slake_file) if dataset == 'SLAKE' else str(source/'VQA-RAD/VQA_RAD Dataset Public.json'),
                source_location=str(r.get('location', 'unknown')), source_annotation={k:r[k] for k in ('triple', 'content_type', 'question_type') if k in r},
                base_query_ids=query_index[(group, normalized(r['question']), normalized(r['answer']))]))
        summaries[dataset] = dict(raw_rows=len(allrows), train_rows=len(rows), train_images=len(groups),
            languages=dict(Counter(r.get('q_lang', 'en') for r in rows)),
            actual_development_images=len(groups & actual), reserved_or_quality_images=len(groups & reserved.keys()),
            nonadditive_counts=True, missing_image_rows=missing,
            fresh_images=len(groups-actual-reserved.keys()-closure))
    return pool, summaries, evidence, dict(release=RELEASE, train_git_blob=blob,
        train_sha256=hashlib.sha256(raw_bytes).hexdigest(), license='CC-BY-4.0',
        author='https://www.med-vqa.com/slake/', published_data='https://huggingface.co/datasets/BoKelvin/SLAKE',
        local_train_matches_author_revision=True, downloaded=False), a


def assemble(pool, parts):
    fit = [r for r in pool if parts[r['source_group']] == 'adaptation']
    evaluation = [r for r in pool if parts[r['source_group']] == 'evaluation']
    packages = []
    for native in sorted(fit, key=lambda r: (r['image_id'], int(r['source_qid']))):
        try: review = source_review(native)
        except ValueError: continue
        hfit = [r for r in fit if conflict(native, r)]
        heval = [r for r in evaluation if conflict(native, r)]
        unrelated = lambda r: r['source_group'] != native['source_group'] and reviewed_attribute(r['question']) not in (None, reviewed_attribute(native['question']))
        ufit = [r for r in fit if unrelated(r)]
        if hfit and heval and ufit:
            packages.append(dict(native=native, review=review, H_fit=hfit[:1], H_evaluation=heval[:2],
                U_fit=ufit[:1], U_evaluation=[r for r in evaluation if unrelated(r)][:4],
                challenge=[r for r in evaluation if normalized(r['question']) == normalized(native['question']) and normalized(r['reference']) == normalized(native['reference'])]))
    return packages


def main(args):
    start = time.time(); run = args.run_root
    if run.exists(): raise FileExistsError('Preserve existing source overlay')
    pool, summaries, evidence, provenance, old = collect(args.storage_root)
    groups = {r['source_group'] for r in pool}
    strata = {r['source_group']: r['source_location'] for r in pool}
    parts = partition(groups, strata)
    packages = assemble(pool, parts)  # No Base verdicts consulted before complete source packages exist.
    write_new(run/'private/SOURCE_OVERLAY.json', dict(rows=pool, image_roles=parts, evidence=evidence, provenance=provenance))
    write_new(run/'private/SOURCE_PACKAGES.json', packages)
    cfg = read(args.stage12_run/'private/CAMPAIGN_CONFIG.json')
    verdict_file = Path(cfg['runtime']['cpu_gate']).parent/'private/BASE_VERDICTS_V4.jsonl'
    verdicts = {r['query_id']:r['is_correct'] for r in map(json.loads, verdict_file.open())}
    predictions = {r['query_id']:r for r in map(json.loads, Path(cfg['runtime']['base_predictions']).open())}
    selected, missing = {}, []
    for package in packages:
        n = package['native']; candidates = [q for q in n['base_query_ids'] if q in verdicts and q in predictions]
        flags = {verdicts[q] for q in candidates}
        if len(flags) != 1:
            missing.append(dict(image=n['image_id'], source_qid=n['source_qid'], reason='BASE_MEMBERSHIP_BINDING_PENDING')); continue
        if flags == {False}:
            package['base_before'] = predictions[candidates[0]]
            package['base_before_qid'] = candidates[0]
            selected.setdefault(n['source_group'], package)
    queue = list(selected.values())[:32]
    train_images, eval_images = set(), set()
    tasks = []
    for i, package in enumerate(queue, 1):
        native = dict(package['native'], logical_id='native', role='native', label='positive', fact_relation='native')
        rid = 'SLAKE:'+RELEASE+':'+str(native['source_qid'])
        positives = [dict(native, logical_id='positive-fit-'+str(j), role='fit', question=p['question'], fact_relation='reviewed_same_fact_text_augmentation', rewrite_family=p['family'])
                     for j, p in enumerate(package['review']['positives']['fit'])]
        evalrows = [dict(native, logical_id='positive-evaluation-'+str(j), role='evaluation', question=p['question'], fact_relation='reviewed_same_fact_text_augmentation', confirmation_panel=p['probe_kind'])
                    for j, p in enumerate(package['review']['positives']['evaluation'])]
        rows = [native, *positives]
        for key in ('H_fit', 'U_fit', 'H_evaluation', 'U_evaluation', 'challenge'):
            for j, original in enumerate(package[key]):
                group, role = key.split('_') if '_' in key else ('challenge', 'challenge')
                row = dict(original, logical_id=key+'-'+str(j), role=role, label='negative', negative_group=group,
                    support_origin='NEW_AUTHORIZED_TRAIN_OVERLAY', conflict_verified=group=='H',
                    fact_relation='same_question_different_image_conflicting_source_answer' if group=='H' else 'broad_unrelated_source_qa' if group=='U' else 'same_question_other_image_same_source_answer')
                (rows if role == 'fit' else evalrows).append(row)
        train_images |= {r['source_group'] for r in rows}
        eval_images |= {r['source_group'] for r in evalrows if r['label']=='negative'}
        event = dict(event_id=rid, event_position=i, probes=[], edit_record=dict(record_id=rid, dataset='SLAKE',
            image_path=native['image_path'], relative_image_path=native['image_id'], question=native['question'], gold_answer=native['reference'],
            official_rephrase=positives[0]['question'], question_type='SOURCE_CONFIRMATION', formal_sequence_position=i))
        training = dict(event=event, event_index=i, track='V4_STAGE3', source_eligibility_frozen=True, extra_fit=[], rows=rows,
            fit_paraphrases=package['review']['positives']['fit'], base_before=package['base_before'], base_before_qid=package['base_before_qid'])
        write_new(run/('private/training/e%02d.json'%i), training)
        write_new(run/('private/evaluation/e%02d.json'%i), evalrows)
        tasks.append(dict(order=i, event_index=i, record_id=rid, seed=int(hashlib.sha256((str(SEED)+'\0'+rid).encode()).hexdigest()[:8],16)&0x7fffffff,
            training='private/training/e%02d.json'%i, evaluation='private/evaluation/e%02d.json'%i, status='PENDING'))
    assert not train_images & eval_images, 'GLOBAL cross-edit training/evaluation image leakage'
    write_new(run/'private/TASKS.json', tasks)
    cfg.update(kind='MEDTRACE_STAGE13R', code_commit=args.commit, stage12_run=str(args.stage12_run),
        worker_gpus=[0], judge_gpu=0, allowed_physical_gpus=[0], forbidden_physical_gpus=[1,2,3],
        train_seconds=6.5*3600, wall_hours=8, gpu_hours=16, authorization='USER_STAGE13R_GPU0',
        fixed_kappa=0.7696741135364367, seed=SEED, source_revision=RELEASE)
    write_new(run/'private/CAMPAIGN_CONFIG.json', cfg)
    write_new(run/'private/MEMBERSHIP_PENDING.json', missing[:10])
    summary = dict(status='QUEUE_FROZEN' if tasks else 'NO_ELIGIBLE_EPISODES', actual_N=len(tasks),
        source_inventory=summaries, provenance=provenance, fresh_english_rows=len(pool), fresh_english_images=len(groups),
        prospective_image_roles=dict(Counter(parts.values())), complete_source_packages=len(packages),
        base_membership_pending=len(missing), new_training_images=len(train_images), new_main_evaluation_images=len(eval_images),
        source_preparation_seconds=time.time()-start, source_review='AGENT_SOURCE_CONSISTENCY_NOT_HUMAN_CLINICAL',
        support_reuse='Allowed but not needed for assembled fresh-source packages',
        protocol='Frozen Stage13/Stage12 methods; native CP->A2->CP-W0->paired freeR4; original BalancEdit',
        independence='New native/main evaluation source images; unknown patients/pretraining exposure; same-image paraphrases are not image generalization',
        draft_handling='Two never-executed PENDING/attempts0 draft queues are not actual development; actual earlier/later runs and all reserved/quality roles still excluded',
        raw_or_private_data_published=False, training='PENDING' if tasks else 'NOT_RUN', judge='PENDING' if tasks else 'NOT_RUN', publication='PENDING')
    write_new(run/'public/SOURCE_EXPANSION_SUMMARY.json', summary)
    print(json.dumps(summary), flush=True)


def complete_challenges(run):
    """Repair only the importer cap using already frozen evaluation roles, never outcomes.

    Preserve original evaluation files. Training files/selection/roles are immutable;
    expanded scoring files must be completed before any evaluation generation starts.
    """
    if (run/'private/scored').exists():raise RuntimeError('Evaluation has already begun; do not amend inputs')
    overlay=read(run/'private/SOURCE_OVERLAY.json');pool=overlay['rows'];roles=overlay['image_roles']
    changes=[]
    for task in read(run/'private/TASKS.json'):
        data=read(run/task['training']);native=data['rows'][0];rows=read(run/task['evaluation'])
        same=[r for r in pool if roles[r['source_group']]=='evaluation' and normalized(r['question'])==normalized(native['question']) and normalized(r['reference'])==normalized(native['reference'])]
        complete=[r for r in rows if r['role']!='challenge']
        complete.extend(dict(r,logical_id='challenge-'+str(j),role='challenge',label='negative',negative_group='challenge',
            support_origin='NEW_AUTHORIZED_TRAIN_OVERLAY',conflict_verified=False,fact_relation='same_question_other_image_same_source_answer') for j,r in enumerate(same))
        write_new(run/('private/evaluation_complete/e%02d.json'%task['order']),complete)
        changes.append(dict(edit=task['order'],original_challenges=sum(r['role']=='challenge' for r in rows),complete_challenges=len(same)))
    write_new(run/'public/EVALUATION_IMPORT_COMPLETENESS_FIX.json',dict(status='COMPLETE_BEFORE_EVALUATION_GENERATION',epoch=time.time(),
        reason='Importer incorrectly capped same-answer challenge at one; include all supported records within the pre-frozen evaluation partition',
        train_queue_changed=False,training_files_changed=False,image_role_assignments_changed=False,source_pool_changed=False,model_outputs_used=False,
        original_evaluation_files_preserved=True,rows=changes))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--storage-root', type=Path, required=True)
    p.add_argument('--stage12-run', type=Path, required=True); p.add_argument('--run-root', type=Path, required=True); p.add_argument('--commit', required=True)
    main(p.parse_args())
