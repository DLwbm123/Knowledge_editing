#!/usr/bin/env python3
"""CPU-only frozen V4 Track A preparation; --preview never writes artifacts.

Only the coordinator creates campaign start records, before loading workers.
Historical weights, outputs, and answer/Judge results are never opened here.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace import prepare_stage2_sources as source

SEED = 20260910
TASKS = ('T0', 'T1L', 'T1G', 'T2L', 'T2G', 'T3L', 'T3G', 'T4L', 'T4G')
BOUND_TASKS = ('T1L', 'T1G', 'T2G')
RELATIONS = dict(H='same_question_different_image_conflicting_source_answer',
                 U='broad_unrelated_source_qa')
METHODS = ['BE', 'S0', 'S1']


def official_overlap_counts(catalog):
    formal = {p['question'] for e in catalog for p in e['probes']}
    normalized = {source.normalized(q) for q in formal}
    result = {}
    for label, events in (('raw_catalog', catalog), ('merged_representatives',
            [e for e in catalog if e['task'] not in BOUND_TASKS])):
        result[label] = {}
        for kind in ('legacy_official_rephrase', 'identity_fallback_no_frozen_rephrase'):
            questions = [e['edit_record'].get('official_rephrase', '') for e in events
                         if e['edit_record'].get('router_positive_source') == kind]
            result[label][kind] = dict(total=len(questions), exact_global_formal_overlap=sum(q in formal for q in questions),
                exact_nonoverlap=sum(q not in formal for q in questions),
                normalized_global_formal_overlap=sum(source.normalized(q) in normalized for q in questions),
                normalized_nonoverlap=sum(source.normalized(q) not in normalized for q in questions))
    return result


def input_key(row):
    return str(row['image_path']), source.normalized(row['question'])


def binding(event):
    row = event['edit_record']
    return row['record_id'], row['dataset'], *input_key(row), source.normalized(row['gold_answer'])


def tokens(row):
    """Only explicit equivalence/family fields, never infer from answer or wording."""
    fields = ('equivalence_group', 'equivalence_group_id', 'eqkey', 'rewrite_family', 'question_family', 'family')
    return {('family' if key in ('rewrite_family', 'question_family', 'family') else key, str(row[key]))
            for key in fields if row.get(key) and row[key] != 'native'}


def group_catalog(catalog, manifest):
    counts = Counter(e['task'] for e in catalog)
    probes = Counter(p['task'] for e in catalog for p in e['probes'])
    if len({e['event_id'] for e in catalog}) != len(catalog):
        raise ValueError('duplicate authoritative event IDs')
    for task in TASKS:
        expected = manifest['tasks'][task]
        if counts[task] != expected['eligible_edit_count'] or probes[task] != expected['eligible_probe_count']:
            raise ValueError('authoritative catalog count mismatch: ' + task)
    anchors = {e['edit_record']['record_id']: deepcopy(e) for e in catalog if e['task'] == 'T0'}
    if len(anchors) != counts['T0']:
        raise ValueError('duplicate T0 anchor record')
    for event in anchors.values():
        event['catalog_event_ids'] = [event['event_id']]
        # Native is one evaluation row; do not duplicate the catalog's T0 probe.
        if (len(event['probes']) != 1 or input_key(event['probes'][0]) != input_key(event['edit_record'])
                or event['probes'][0]['reference'] != event['edit_record']['gold_answer']):
            raise ValueError('T0 native probe binding mismatch')
        event['probes'] = []
    extra = []
    for event in catalog:
        if event['task'] == 'T0':
            continue
        if event['task'] in BOUND_TASKS:
            target = anchors.get(event['edit_record']['record_id'])
            if target is None or binding(target) != binding(event):
                raise ValueError('bound probe is not the identical T0 edit: ' + event['event_id'])
            target['catalog_event_ids'].append(event['event_id'])
            target['probes'].extend(deepcopy(event['probes']))
        else:
            extra.append(dict(deepcopy(event), catalog_event_ids=[event['event_id']]))
    grouped = list(anchors.values()) + extra
    return grouped, dict(t0_anchors=counts['T0'], catalog_events=len(catalog),
        catalog_events_by_task=dict(counts), formal_probes=sum(probes.values()),
        formal_probes_by_task=dict(probes), planned_groups=len(grouped),
        task_specific_groups=len(extra), bound_catalog_events=sum(counts[t] for t in BOUND_TASKS),
        unique_native_inputs=len({input_key(e['edit_record']) for e in grouped}),
        planned_formal_outputs_per_system=sum(probes.values()))


def load_sources(stage2_run):
    config = source.read(stage2_run / 'private/CAMPAIGN_CONFIG.json')
    gate = Path(config['runtime']['cpu_gate'])
    catalog_path = gate / 'inputs/frozen/FORMAL_SINGLE_EVENT_CATALOG.jsonl'
    handoff = gate.parent / 'cohorts_v4/handoff_v4'
    manifest = source.read(handoff / 'FORMAL_CATALOG_MANIFEST.json')
    if manifest.get('scope') != 'public-release-aligned T0-T4' or manifest.get('method_outputs_used') is not False:
        raise ValueError('V4 catalog scope/selection provenance mismatch')
    method_path = gate / 'locks/FORMAL_METHOD_CONFIG_BUNDLE.json'
    method = source.read(method_path)['method_configs']['balancedit']
    preflight = source.read(gate / 'CPU_PREFLIGHT.json')
    official_authority = (method.get('positive') == 'same image plus frozen official question rephrase'
                          and preflight['checks'].get('t0_legacy_binding_exact') is True)
    catalog = source.jsonl(catalog_path)
    for event in catalog:
        if official_authority and event['edit_record'].get('router_positive_source') == 'legacy_official_rephrase':
            event['official_support_authority'] = dict(method_lock=str(method_path),
                positive=method['positive'], binding='V4 CPU bridge exact legacy record binding',
                permitted_role='editing_support', family_status='UNKNOWN')
    stage2_path = stage2_run / 'private/NEW_EPISODE_MANIFEST_PRIVATE.json'
    stage2 = source.read(stage2_path)
    if stage2['runtime'] != config['runtime']:
        raise ValueError('Stage2 source runtime binding differs from campaign')
    # The source inventory already authorizes the complete screened pool. Do not
    # rescan source datasets or silently expand its historical exclusion boundary.
    if stage2['source_inventory']['membership_boundary']['status'] != 'RESOLVED_IDENTITY_MEMBERSHIP__ENTIRE_SOURCE_POOL':
        raise ValueError('Stage2 source pool identity clearance missing')
    historical = source.read(Path(config['old_run']) / 'private/frozen_data.json')
    sidecars = []
    for path in sorted((stage2_run / 'private/edits').glob('e*.json')):
        item = source.read(path)
        item['support_sidecar'] = str(path)
        sidecars.append(item)
    # Static inventory is obtained from the existing scanner's recorded evidence.
    inventory_path = next((Path(e['path']) for e in stage2['source_inventory']['evidence']
                           if Path(e['path']).name == 'STATIC_QUERY_INVENTORY.jsonl'), None)
    if inventory_path is None:
        raise ValueError('Stage2 source evidence lacks static query inventory')
    return config, catalog, manifest, stage2, historical, sidecars, source.jsonl(inventory_path), dict(
        catalog=str(catalog_path), manifest=str(handoff / 'FORMAL_CATALOG_MANIFEST.json'),
        stage2_sources=str(stage2_path), static_inventory=str(inventory_path),
        official_support_method_lock=str(method_path),
        historical_fit=str(Path(config['old_run']) / 'private/frozen_data.json'))


def build_plan(catalog, manifest, stage2, historical, sidecars, inventory, check_files=True):
    events, counts = group_catalog(catalog, manifest)
    by_id = {r['query_id']: r for r in inventory}
    image_ids = defaultdict(set)
    for row in inventory:
        image_ids[str(row['image_path'])].add('source:' + row['dataset'] + ':' + source.identity(row['image_id']))
        if row.get('image_sha256'):
            image_ids[str(row['image_path'])].add('existing_hash:' + row['image_sha256'])

    def groups(row):
        return image_ids.get(str(row['image_path']), {'path:' + str(row['image_path'])})

    formal = [p for e in catalog for p in e['probes']]
    evaluation = [p for e in catalog if e['task'] != 'T0' for p in e['probes']]
    protected_text = {source.normalized(r['question']) for r in formal}
    protected_inputs = {input_key(r) for r in evaluation}
    protected_families = set().union(*(tokens(r) for r in formal))
    protected_images = set().union(*(groups(r) for r in formal), *(groups(e['edit_record']) for e in events))
    # Existing inventory aliases and explicit equivalence IDs close over protected
    # queries. relation_id is provenance, not a semantic equivalence assertion.
    protected_ids = {p['probe_id'] for p in evaluation}
    explicit = set().union(*(tokens(by_id[q]) for q in protected_ids if q in by_id))
    for row in inventory:
        if row['query_id'] in protected_ids or tokens(row) & explicit:
            protected_inputs.add(input_key(row))
            protected_text.add(source.normalized(row['question']))
            protected_families.update(tokens(row))
            protected_images.update(groups(row))
    sidecar_index = {binding(s['event']): s for s in sidecars}
    # Freeze all historical and new auxiliary held-out queries before considering
    # any event's fit inputs; no per-event filtering can miss another event here.
    for sidecar in sidecars:
        for row in sidecar.get('rows', []):
            if row['role'] in ('calibration', 'evaluation', 'formal_development', 'challenge'):
                protected_inputs.add(input_key(row))
                if row['label'] == 'positive':
                    protected_text.add(source.normalized(row['question']))
                    protected_families.update(tokens(row))
                else:
                    protected_images.update(groups(row))
    reviews = {}
    for event in events:
        try:
            review = source.source_review(event['edit_record'])
        except ValueError:
            continue
        reviews[event['event_id']] = review
        for role in ('calibration', 'evaluation'):
            for item in review['positives'][role]:
                protected_inputs.add(input_key(dict(event['edit_record'], question=item['question'])))
                protected_text.add(source.normalized(item['question']))
                protected_families.update(tokens(item))

    protected_aliases = {(g, question) for path, question in protected_inputs
                         for g in groups(dict(image_path=path))}

    def protected_input(row):
        return input_key(row) in protected_inputs or any(
            (g, source.normalized(row['question'])) in protected_aliases for g in groups(row))

    pool = stage2['source_pool']
    eligible_pool = [r for r in pool if not groups(r) & protected_images and not protected_input(r)]
    # One global image role prevents eA's H/U fit image becoming eB's eval image.
    role_groups = sorted({r['source_group'] for r in eligible_pool})
    roles = {g: source.ROLES[i % 3] for i, g in enumerate(role_groups)}
    historical_fit_images = set().union(*(groups(r) for s in sidecars for r in s.get('rows', [])
        if r['role'] == 'fit' and r['label'] == 'negative'))
    for row in eligible_pool:
        if groups(row) & historical_fit_images:
            roles[row['source_group']] = 'fit'
    episodes = []
    for index, original in enumerate(events, 1):
        event = deepcopy(original)
        edit = event['edit_record']
        native = dict(logical_id='native', role='native', label='positive', question=edit['question'],
            reference=edit['gold_answer'], image_path=edit['image_path'], dataset=edit['dataset'],
            source_group=sorted(groups(edit))[0], fact_relation='native', patient_id='UNKNOWN',
            task='T0' if event['task'] == 'T0' else 'NATIVE_DIAGNOSTIC', base_query_ids=[edit['record_id']])
        rows = [native]
        for p in event['probes']:
            rows.append(dict(p, logical_id='formal-' + p['probe_id'], role='formal_development',
                label='negative' if p['task'].endswith('L') else 'positive',
                fact_relation=p['task'], source_group=sorted(groups(p))[0], patient_id='UNKNOWN',
                base_query_ids=[p['probe_id']]))
        if len({r['logical_id'] for r in rows}) != len(rows):
            raise ValueError('duplicate probe IDs in grouped event: ' + event['event_id'])
        sidecar = sidecar_index.get(binding(event))
        candidates = []
        rejection = Counter()
        official = edit.get('official_rephrase', '')
        official_audit = dict(router_positive_source=edit.get('router_positive_source'),
            authority=event.get('official_support_authority'), family_status='UNKNOWN')
        if source.normalized(official) == source.normalized(edit['question']):
            official_audit['status'] = 'REJECTED_NATIVE_IDENTITY_DUPLICATE'
        elif source.normalized(official) in protected_text:
            official_audit['status'] = 'REJECTED_FORMAL_OR_HELDOUT_QUERY_OVERLAP'
        elif event.get('official_support_authority') and official:
            official_audit['status'] = 'FROZEN_OFFICIAL_EDITING_SUPPORT'
            candidates.append(dict(question=official, family=None, family_status='UNKNOWN',
                review_status='FROZEN_OFFICIAL_EDITING_SUPPORT',
                support_source=event['official_support_authority']['method_lock'],
                official_support_authority=event['official_support_authority']))
        else:
            official_audit['status'] = 'NO_EXPLICIT_OFFICIAL_SUPPORT_AUTHORITY'
        if sidecar:
            candidates.extend(dict(p, support_source=sidecar['support_sidecar']) for p in sidecar.get('fit_paraphrases', []))
        # Exact record binding in the frozen source sidecar, never transfer a
        # clinical rephrase merely because two records have similar questions.
        old_event = next((e for e in historical.get('dev', []) if binding(e) == binding(event)), None)
        if old_event:
            candidates.extend(dict(p, support_source='historical_frozen_generality_paraphrases')
                for p in historical.get('generality_paraphrases', {}).get(edit['record_id'], []))
        if event['event_id'] in reviews:
            positive, fit, _ = source.positive_rows(reviews[event['event_id']], native)
            rows.extend(dict(r, task='SOURCE_STYLE_CONFIRMATION' if r.get('probe_kind') == 'source_style_confirmation'
                             else 'CROSS_FAMILY_CONFIRMATION') for r in positive if r['role'] != 'fit')
            candidates.extend(dict(p, support_source='Stage2_source_review_finite_family') for p in fit)
        fit, seen = [], set()
        for item in candidates:
            q = source.normalized(item.get('question'))
            official_approved = (item.get('review_status') == 'FROZEN_OFFICIAL_EDITING_SUPPORT'
                                 and item.get('official_support_authority'))
            reason = ('FIT_EQUIVALENCE_NOT_APPROVED' if not official_approved and
                      (item.get('review_status') != 'APPROVED_EQUIVALENT' or not item.get('family')) else
                      'FIT_FORMAL_OR_HELDOUT_QUERY_OVERLAP' if q in protected_text else
                      'FIT_PROTECTED_EXPLICIT_FAMILY_OVERLAP' if tokens(item) & protected_families else None)
            if reason:
                rejection[reason] += 1
            elif q and q not in seen and q != source.normalized(edit['question']):
                seen.add(q)
                fit.append(item)
        rows.extend(dict(native, question=p['question'], role='fit', logical_id=f'positive-fit-{i}',
            rewrite_family=p['family'], fact_relation='reviewed_same_fact_text_augmentation',
            task='TRAINING_SUPPORT', fit_positive_source='A2_NATIVE_OR_PARAPHRASE',
            base_query_ids=[], support_source=p['support_source']) for i, p in enumerate(fit))
        attribute = source.reviewed_attribute(edit['question'])
        used = set()
        negatives = []
        if sidecar:
            negatives.extend(dict(r, support_source=sidecar['support_sidecar']) for r in sidecar.get('rows', [])
                if r['role'] == 'fit' and r['label'] == 'negative' and r.get('fact_relation') in RELATIONS.values())
        # One bounded supplement from the pre-authorized pool, following exactly
        # Stage2's finite attribute checks. Unknown native attributes remain unknown.
        if attribute:
            for other in eligible_pool:
                other_attribute = source.reviewed_attribute(other['question'])
                annotation = other.get('source_annotation', {}).get('question_type')
                if other_attribute is None and annotation in ('MODALITY', 'PLANE'):
                    other_attribute = annotation.lower()
                group = ('H' if source.normalized(other['question']) == source.normalized(edit['question'])
                         and other_attribute == attribute and source.normalized(other['reference']) != source.normalized(edit['gold_answer']) else
                         'U' if other_attribute and other_attribute != attribute else None)
                if group:
                    negatives.append(dict(other, role=roles[other['source_group']], label='negative', negative_group=group,
                        fact_relation=RELATIONS[group], conflict_verified=group == 'H',
                        support_source='Stage2_authorized_source_pool', relation_evidence='Stage2 finite attribute/source annotation rule'))
        support = {r: dict(H=0, U=0) for r in source.ROLES}
        for other in negatives:
            group = next(g for g, rel in RELATIONS.items() if rel == other['fact_relation'])
            image_group = groups(other)
            if image_group & protected_images or protected_input(other):
                rejection['HU_PROTECTED_QUERY_OR_IMAGE_GROUP'] += 1
                continue
            if image_group & used or support[other['role']][group] >= 20:
                continue
            if check_files and not Path(other['image_path']).is_file():
                rejection['HU_SOURCE_IMAGE_MISSING'] += 1
                continue
            used.update(image_group)
            support[other['role']][group] += 1
            rows.append(dict(other, logical_id=f'negative-{group}-{other["role"]}-{support[other["role"]][group]}',
                negative_group=group, source_group=sorted(image_group)[0], patient_id='UNKNOWN'))
        common_errors = []
        if protected_input(edit):
            common_errors.append('NATIVE_IS_PROTECTED_EVALUATION_QUERY')
        if check_files and any(not Path(r['image_path']).is_file() for r in rows):
            common_errors.append('REQUIRED_IMAGE_MISSING')
        fit_errors = [] if fit else ['NO_APPROVED_ROLE_DISJOINT_FIT_PARAPHRASE']
        hu_errors = [g + '_FIT_UNSUPPORTED_BY_AUTHORIZED_SOURCE_RELATION' for g in ('H', 'U') if not support['fit'][g]]
        unsupported = dict(BE=common_errors + fit_errors, S0=common_errors + fit_errors, S1=common_errors + fit_errors + hu_errors)
        methods = [m for m in METHODS if not unsupported[m]]
        original_anchor = edit.get('official_rephrase', '')
        edit['official_rephrase'] = fit[0]['question'] if fit else ''
        edit['router_positive_source'] = 'stage3_frozen_authorized_fit' if fit else 'UNSUPPORTED_NO_LEGAL_POSITIVE_ANCHOR'
        episodes.append(dict(event_index=index, event=event, record_id=edit['record_id'], track='V4_STAGE3', seed=SEED,
            rows=rows, fit_paraphrases=fit, extra_fit=[], negative_support=support,
            source_eligibility_frozen=True, inputs_frozen_before_student=True,
            trainable=bool(methods), executable_methods=methods, common_support=len(methods) == 3,
            unsupported_reasons=unsupported, pending=sorted(set(sum(unsupported.values(), []))),
            support_rejection_counts=dict(rejection), patient_id='UNKNOWN',
            original_official_rephrase=original_anchor,
            frozen_aliases=dict(native_query_id=edit['record_id'],
                image_identities=sorted(groups(edit)), catalog_event_ids=event['catalog_event_ids'],
                explicit_equivalence_metadata=sorted(tokens(by_id.get(edit['record_id'], {}))),
                patient_id='UNKNOWN', inferred_semantic_equivalence=False),
            official_rephrase_use=official_audit,
            exposure=dict(status='VIEWED_MEDTRACE' if sidecar or old_event else 'BASE_EXPOSED_NOT_IN_MEDTRACE_SELECTION_SIDECARS',
                historical_source=sidecar['support_sidecar'] if sidecar else None,
                blind_claim=False, broader_historical_exposure='UNKNOWN'),
            reuse=dict(checkpoints=False, reason='New global fit/evaluation exclusion and support schedule; exact historical bindings not established'),
            positive_review=reviews.get(event['event_id']), H_all_retained_before_Base_correct_filter=True))
    # Check the complete newly frozen ledger, including supplementary panels.
    # This catches cross-event leakage that a single-edit role check cannot see.
    heldout = [r for e in episodes for r in e['rows']
               if r['role'] in ('formal_development', 'calibration', 'evaluation', 'challenge')]
    heldout_keys = {input_key(r) for r in heldout}
    heldout_images = set().union(*(groups(r) for r in heldout if r['label'] == 'negative'))
    for episode in episodes:
        for row in episode['rows']:
            if row['role'] == 'fit' and (input_key(row) in heldout_keys or
                    (row['label'] == 'negative' and groups(row) & heldout_images)):
                raise ValueError('cross-event fit/evaluation identity leak in frozen inputs')
    executable = {m: sum(m in e['executable_methods'] for e in episodes) for m in METHODS}
    counts.update(executable_groups=sum(bool(e['executable_methods']) for e in episodes),
        official_rephrase_overlap_audit=official_overlap_counts(catalog),
        official_rephrase_admission=dict(Counter(e['official_rephrase_use']['status'] for e in episodes)),
        executable_by_method=executable, common_support_groups=sum(e['common_support'] for e in episodes),
        unsupported_by_method={m: dict(Counter(reason for e in episodes for reason in e['unsupported_reasons'][m])) for m in METHODS},
        support_by_task={t: dict(planned=sum(e['event']['task'] == t for e in episodes),
            **{m: sum(e['event']['task'] == t and m in e['executable_methods'] for e in episodes) for m in METHODS})
            for t in TASKS if t not in BOUND_TASKS},
        authorized_pool_rows=len(pool), pool_rows_after_protection=len(eligible_pool),
        executable_formal_probes={m: dict(Counter(r['task'] for e in episodes if m in e['executable_methods']
            for r in e['rows'] if r['role'] == 'formal_development' or (r['role'] == 'native' and r.get('task') == 'T0'))) for m in METHODS},
        planned_generated_rows_per_system=sum(len(e['rows']) for e in episodes),
        executable_generated_rows_per_system={m: sum(len(e['rows']) for e in episodes if m in e['executable_methods']) for m in METHODS})
    formal_support = {m: sum(counts['executable_formal_probes'][m].values()) for m in METHODS}
    counts['expected_formal_records_by_path'] = dict(BASE=formal_support['BE'],
        **{m+'_'+path: formal_support[m] for m in METHODS for path in ('ROUTED', 'FORCED_ON')})
    counts['expected_formal_records_all_paths'] = sum(counts['expected_formal_records_by_path'].values())
    queue = [dict(task_id=f'S3_A_e{e["event_index"]:04d}_GROUP', kind='SINGLE_GROUP', event_index=e['event_index'],
        priority=e['event_index'], status='PENDING' if e['executable_methods'] else 'UNSUPPORTED_TRAINING_INPUTS',
        depends_on=None, attempts=0, methods=e['executable_methods'], allplanned=METHODS[:],
        unsupported_reasons=e['unsupported_reasons'], seed=SEED) for e in episodes]
    return episodes, queue, counts


def prepare(args):
    run, old = Path(args.run_root), Path(args.stage2_run)
    if run.exists():
        raise FileExistsError(run)
    config, catalog, manifest, stage2, historical, sidecars, inventory, paths = load_sources(old)
    episodes, tasks, counts = build_plan(catalog, manifest, stage2, historical, sidecars, inventory)
    if getattr(args, 'preview', False):
        return counts
    inherited_epoch = config.pop('campaign_epoch', None)
    config.pop('reuse_stage2_run', None)
    config.update(kind='MEDTRACE_STAGE3', stage2_run=str(old),
        campaign_epoch=time.time(), seed=SEED, novel_seed=SEED,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        stage2_training_execution_commit=config['code_commit'], train_seconds=20*3600, wall_hours=24, gpu_hours=48,
        authorization='Stage3: physical GPU2/3 only, shared free-memory admission; GPU1 forbidden',
        allowed_physical_gpus=[2, 3], forbidden_physical_gpus=[1],
        coordinator_start_required=True, prior_campaign_epoch_not_reused=inherited_epoch,
        methods=['BE', 'P4-W0_TASK_ONLY', 'P4-W1_KL_0.1'])
    banks = [dict(task_id=f'S3_B_prefix{k:02d}', kind='BANK', prefix=k, priority=-100+k,
        status='PENDING', depends_on=None, attempts=0, methods=METHODS[:]) for k in (1, 4, 8, 16)]
    run.mkdir(parents=True)
    source.write_new(run / 'private/CAMPAIGN_CONFIG.json', config)
    for episode in episodes:
        source.write_new(run / f'private/edits/e{episode["event_index"]:02d}.json', episode)
    source.write_new(run / 'private/TASK_QUEUE.json', dict(schema_version='medtrace-stage3-group-queue-v1', tasks=banks+tasks))
    result = dict(schema_version='medtrace-stage3-preparation-v1', protocol='V4_RELEASE_ALIGNED_AUGMENTED_SUPPORT_EVALUATION',
        status='INPUTS_FROZEN_BEFORE_STUDENT', counts=counts, sources=paths,
        catalog=manifest, task_queue='private/TASK_QUEUE.json', track='V4_STAGE3',
        source_builder='scripts/medtrace/prepare_stage2_sources.py',
        input_policy='All events protected together; explicit families and existing image identities only; unknown patient/family remains UNKNOWN',
        checkpoint_reuse='Track A NONE: amended fit/H/U/role bindings; Track B original Stage2 artifacts only',
        timing='New campaign epoch; coordinator must atomically create starts before workers and preserve starts on resume',
        t5='NA_NO_LEGAL_MATERIAL', performance_gate=False)
    source.write_new(run / 'RUN_MANIFEST.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage2-run', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    print(json.dumps(prepare(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
