#!/usr/bin/env python3
"""Source-only Stage2 inventory and episode data builder; never loads a model.

python scripts/medtrace/prepare_stage2_sources.py --config CONFIG --run-root RUN
Optional CONFIG.positive_review points to source-question-only reviewed families.
Without that review, candidates are emitted honestly as NOT_TRAINABLE drafts.
"""
import argparse
import csv
import json
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/remote-home/wangbomin')
ROLES = ('fit', 'calibration', 'evaluation')
HISTORICAL = (
    '20260906T030312Z/private/frozen_data.json',
    '20260906T030312Z/private/invalid_pre_hardfix_frozen_data.json',
    '20260906T030312Z/private/invalid_pre_role_balance_frozen_data.json',
    '20260905T134727Z/private/frozen_data.json',
    '20260905T125112Z/track_b/roles_private.json',
)


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identity(value):
    path = Path(str(value))
    return (path.parent.name if path.name.lower() == 'source.jpg' else path.name).lower()


def normalized(value):
    return ' '.join(str(value or '').casefold().split())


def image_groups(value, dataset='SLAKE'):
    """Project identity fields only, including every historical negative role."""
    groups = set()
    if isinstance(value, dict):
        dataset = value.get('source_dataset', value.get('dataset', dataset))
        for key in ('image_name', 'image_path', 'relative_image_path'):
            if value.get(key):
                groups.add((dataset, identity(value[key])))
        for child in value.values():
            if isinstance(child, (dict, list)):
                groups.update(image_groups(child, dataset))
    elif isinstance(value, list):
        for child in value:
            groups.update(image_groups(child, dataset))
    return groups


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrites: a later review gets a new run/data directory.
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def scan(config):
    root = Path(config.get('storage_root', ROOT))
    sources = Path(config.get('source_root', root / 'DataP/knowledge_editing/data/m3bench'))
    v4 = root / 'Knowledge_editing/outputs/m3bench_current_stack_v4/20260904T102909Z'
    static = root / 'Knowledge_editing/outputs/m3bench_data_runtime_finalization_v3/20260904T014138Z/data_static'
    catalog = jsonl(static / 'STATIC_QUERY_INVENTORY.jsonl')
    by_query = {row['query_id']: (row['dataset'], identity(row['image_id'])) for row in catalog}
    reasons = defaultdict(set)
    evidence = []
    boundary_groups = {}
    boundary_counts = {}

    def exclude(groups, reason, path):
        for group in groups:
            reasons[group].add(reason)
        evidence.append({'path': str(path), 'reason': reason, 'image_groups': len(groups)})

    for relative in HISTORICAL:
        path = root / 'medtrace_runs' / relative
        exclude(image_groups(read(path)), 'historical_medtrace_all_roles', path)
    for path in sorted((root / 'medtrace_runs').glob('*/private/INPUT_MANIFEST.json')):
        value = read(path)
        groups = {('VQA-RAD' if 'VQA-RAD' in im or identity(im).startswith('synpic') else 'SLAKE', identity(im))
                  for im in value.get('images', [])}
        exclude(groups, 'historical_medtrace_input_images', path)
    stage1 = Path(config['stage1_run'])
    for path in sorted((stage1 / 'private/edits').glob('*.json')):
        exclude(image_groups(read(path)), 'stage1_all_roles', path)
    # Public-source structural catalog, NOT private formal/QUAL answer files.
    structural = {(row['dataset'], identity(row['image_id'])) for row in catalog
                  if any(item.get('role') != 'source_qa' for item in row['lineage'])}
    structural.update((row['dataset'], identity(row['image_id']))
                      for row in jsonl(static / 'STATIC_T0_CANDIDATES.jsonl'))
    exclude(structural, 'reserved_formal_or_structural_source_group', static)
    # Legacy heldout selector freeze(): amended-189 targets, corresponding T1L
    # probes only. Excluding both full source supersets avoids answer access.
    t0_path = static / 'STATIC_T0_CANDIDATES.jsonl'
    t1_path = static / 'STATIC_T1_RELATIONS.jsonl'
    t0_rows = jsonl(t0_path)
    t1_rows = [member for relation in jsonl(t1_path) if relation['task'] == 'T1L' for member in relation['members']]
    t0_groups = {(row['dataset'], identity(row['image_id'])) for row in t0_rows}
    t1_groups = {(row['dataset'], identity(row['image_id'])) for row in t1_rows}
    exclude(t0_groups, 'legacy_heldout_target_source_superset', t0_path)
    exclude(t1_groups, 'legacy_heldout_locality_source_superset', t1_path)
    boundary_groups['legacy_heldout_source_superset'] = t0_groups | t1_groups
    boundary_counts['legacy_heldout_source_superset'] = dict(target_rows=len(t0_rows), target_images=len(t0_groups),
        locality_rows=len(t1_rows), locality_images=len(t1_groups))
    evidence.append(dict(path=str(root / 'worktrees/m3bench-lora-strong-v1-20260902T134549Z/scripts/m3bench_lora_strong.py'),
                         reason='legacy_heldout_selector_freeze_amended189_and_T1L', image_groups=len(t0_groups | t1_groups)))
    # These handoff records are identity-only: query IDs, not private QUAL answers.
    task_events = {}
    for task in ('T2L', 'T3G', 'T4L', 'T4G'):
        path = v4 / 'cohorts_v4/handoff_v4' / (task + '_FORMAL_RECORDS.jsonl')
        for event in jsonl(path):
            task_events[event['event_id']] = event, path
    mapped_non_t0, non_t0_groups = 0, set()
    non_t0_preexcluded = True
    for name in ('LORA_DEV16', 'LORA_QUAL16', 'LORA_SEQ16', 'QUAL8'):
        path = v4 / 'manifests_v2' / (name + '_V2_MANIFEST.json')
        ids = read(path)['event_ids']
        groups = set()
        for event_id in ids:
            if event_id.removeprefix('T0:') in by_query:
                groups.add(by_query[event_id.removeprefix('T0:')])
            else:
                if event_id not in task_events:
                    raise ValueError(f'No identity-only source mapping for {name}: {event_id}')
                event, source_path = task_events[event_id]
                mapped = {by_query[q] for q in [event['edit_query_id'], *event['probe_query_ids']]}
                non_t0_preexcluded &= mapped <= structural
                non_t0_groups.update(mapped)
                groups.update(mapped)
                mapped_non_t0 += 1
                exclude(mapped, 'qualification_non_t0_identity_handoff', source_path)
        exclude(groups, name, path)
    boundary_groups['qualification_non_t0'] = non_t0_groups
    boundary_counts['qualification_non_t0'] = dict(mapped_events=mapped_non_t0, image_groups=len(non_t0_groups),
                                                  already_in_structural_exclusion=non_t0_preexcluded)
    exclude({('SLAKE', 'xmlab281'), ('SLAKE', 'xmlab281_3')}, 'disabled_source_group', 'historical_exclusion')
    live = root / 'Knowledge_editing/outputs/liveedit_med_eqkey_clean_fast_confirmation_v1/20260815T122907Z/eqkey_family_audit/positive_input_ledger.csv'
    with live.open() as handle:
        live_rows = [{key: row[key] for key in ('record_id', 'raw_image_sha256')} for row in csv.DictReader(handle)]
    split_path = root / 'Knowledge_editing/outputs/liveedit_med_strict_source_v1/20260814T101648Z/training/split_manifest.json'
    splits = read(split_path)
    expected_ids = set().union(*(set(map(str, splits[role + '_ids'])) for role in ('train', 'validation', 'heldout')))
    if {row['record_id'] for row in live_rows} != expected_ids or any(not row['raw_image_sha256'] for row in live_rows):
        raise ValueError('LiveEdit identity ledger does not completely cover its source split manifest')
    live_hashes = {row['raw_image_sha256'] for row in live_rows}
    live_groups = {(row['dataset'], identity(row['image_id'])) for row in catalog if row.get('image_sha256') in live_hashes}
    exclude(live_groups, 'liveedit_all640_source_image_superset', live)
    evidence.append(dict(path=str(split_path), reason='liveedit_identity_coverage_binding', image_groups=len(live_hashes)))
    boundary_groups['liveedit_source_superset'] = live_groups
    boundary_counts['liveedit_source_superset'] = dict(records=len(expected_ids), views=len(live_rows),
        image_hashes=len(live_hashes), catalog_image_matches=len(live_groups),
        split_records={role: len(splits[role + '_ids']) for role in ('train', 'validation', 'heldout')})
    # Optional identity-only additions from governance owner; no answer file required.
    if config.get('additional_exclusions'):
        path = Path(config['additional_exclusions'])
        for row in read(path)['images']:
            exclude({(row['dataset'], identity(row['image_id']))}, row['reason'], path)

    slake = {split: read(sources / 'SLAKE' / (split + '.json')) for split in ('train', 'validation', 'test')}
    vqa = read(sources / 'VQA-RAD/VQA_RAD Dataset Public.json')
    exclude({('SLAKE', identity(row['img_name'])) for split in ('validation', 'test') for row in slake[split]},
            'source_validation_or_test_image', sources / 'SLAKE')
    exclude({('VQA-RAD', identity(row['image_name'])) for row in vqa if row['phrase_type'].startswith('test')},
            'source_test_image', sources / 'VQA-RAD/VQA_RAD Dataset Public.json')
    # Close every exclusion (including optional and source-split exclusions) over
    # existing image hashes. No new image hashing or model preprocessing.
    blocked_hashes = live_hashes | {row['image_sha256'] for row in catalog
                                  if (row['dataset'], identity(row['image_id'])) in reasons and row.get('image_sha256')}
    exclude({(row['dataset'], identity(row['image_id'])) for row in catalog if row.get('image_sha256') in blocked_hashes},
            'existing_image_hash_identity_closure', static / 'STATIC_QUERY_INVENTORY.jsonl')
    query_index = defaultdict(list)
    image_hashes, hash_members = {}, defaultdict(set)
    for row in catalog:
        query_index[(row['dataset'], identity(row['image_id']), normalized(row['question']), normalized(row['gold_answer']))].append(row['query_id'])
        if row.get('image_sha256'):
            image_group = row['dataset'], identity(row['image_id'])
            image_hashes[image_group] = row['image_sha256']
            hash_members[row['image_sha256']].add(image_group)
    canonical_groups = {group: ':'.join(min(hash_members[digest])) for group, digest in image_hashes.items()}
    verdict_path = v4 / 'private/BASE_VERDICTS_V4.jsonl'
    verdicts = {row['query_id']: row['is_correct'] for row in jsonl(verdict_path)}
    pool, counts = [], Counter()
    for dataset, rows in [('SLAKE', slake['train']), ('VQA-RAD', vqa)]:
        for row in rows:
            if dataset == 'SLAKE' and row['q_lang'] != 'en':
                counts[dataset + ':non_english'] += 1
                continue
            if dataset == 'VQA-RAD' and row['phrase_type'] not in ('freeform', 'para'):
                counts[dataset + ':source_test_role'] += 1
                continue
            image_id = identity(row['img_name'] if dataset == 'SLAKE' else row['image_name'])
            group = dataset, image_id
            if group in reasons:
                for reason in reasons[group]:
                    counts[dataset + ':excluded:' + reason] += 1
                continue
            image = sources / ('SLAKE/imgs/' + row['img_name'] if dataset == 'SLAKE' else 'VQA-RAD/images/' + row['image_name'])
            if not str(row.get('answer', '')).strip():
                counts[dataset + ':missing_source_answer'] += 1
                continue
            key = dataset, image_id, normalized(row['question']), normalized(row['answer'])
            qids = query_index.get(key, [])
            values = {verdicts[q] for q in qids if q in verdicts}
            before = next(iter(values)) if len(values) == 1 else None
            pool.append({'dataset': dataset, 'source_qid': row['qid'], 'image_id': image_id,
                         'image_path': str(image), 'question': row['question'], 'reference': row['answer'],
                         'source_group': canonical_groups.get(group, dataset + ':' + image_id), 'patient_id': 'UNKNOWN',
                         'source_file': str(sources / ('SLAKE/train.json' if dataset == 'SLAKE' else 'VQA-RAD/VQA_RAD Dataset Public.json')),
                         'source_role': 'train' if dataset == 'SLAKE' else row['phrase_type'],
                         'source_annotation': {k: row[k] for k in ('img_id', 'triple', 'base_type', 'question_type', 'qid_linked_id', 'question_relation') if k in row},
                         'base_query_ids': qids, 'base_correct': before, 'base_verdict_path': str(verdict_path)})
    pool_groups = {(row['dataset'], row['image_id']) for row in pool}
    catalog_hash_groups = {(row['dataset'], identity(row['image_id'])) for row in catalog if row.get('image_sha256')}
    overlaps = {name: len(pool_groups & groups) for name, groups in boundary_groups.items()}
    if any(overlaps.values()) or pool_groups & reasons.keys() or pool_groups - catalog_hash_groups:
        raise ValueError('Whole source pool has a reserved identity overlap or lacks existing image identity coverage')
    boundary = dict(status='RESOLVED_IDENTITY_MEMBERSHIP__ENTIRE_SOURCE_POOL', rows_checked=len(pool),
                    image_groups_checked=len(pool_groups), existing_image_hash_coverage=len(pool_groups & catalog_hash_groups),
                    reserved_overlap_images=overlaps, evidence_counts=boundary_counts,
                    scope='Source-role and existing image-identity clearance for the entire pool, not H conflict verification, patient independence, or training-role approval.',
                    new_user_authority_required=False, sealed_heldout_answer_files_opened=False)
    summary = {'source_counts': {'SLAKE/train': len(slake['train']), 'SLAKE/validation': len(slake['validation']),
                                'SLAKE/test': len(slake['test']), 'VQA-RAD/all': len(vqa)},
               'screened_development_pool': {d: {'rows': sum(row['dataset'] == d for row in pool),
                    'images': len({row['source_group'] for row in pool if row['dataset'] == d}),
                    'base_incorrect_rows': sum(row['dataset'] == d and row['base_correct'] is False for row in pool)} for d in ('SLAKE', 'VQA-RAD')},
               'exclusion_counts_nonadditive': dict(counts), 'evidence': evidence,
               'membership_boundary': boundary,
               'permission_basis': 'User Stage2 V2 authorizes non-reserved development sources; not file presence. Old VQA-RAD audit-only is superseded only for this non-reserved scope.',
               'patient_id': 'UNKNOWN', 'all_static_base_inference_not_treated_as_reservation': True,
               'positive_review_required_before_student': True}
    ledger = [{'dataset': d, 'image_id': im, 'reasons': sorted(rs)} for (d, im), rs in sorted(reasons.items())]
    return pool, summary, ledger


def positive_rows(review, native):
    if not review:
        return [], [], ['SOURCE_QUESTION_ONLY_EQUIVALENCE_REVIEW_PENDING']
    if review.get('visibility') != 'SOURCE_QUESTION_ONLY__NO_STUDENT_OUTPUTS' or not review.get('approved_equivalent'):
        raise ValueError('positive review visibility/approval not established')
    output, fit, families, texts = [], [], {}, {normalized(native['question'])}
    for role in ROLES:
        items = review['positives'][role]
        if len(items) < 4:
            raise ValueError('four approved positives per role are required')
        families[role] = {item['family'] for item in items}
        for index, item in enumerate(items, 1):
            key = normalized(item['question'])
            if not key or key in texts:
                raise ValueError('positive text overlaps native or another role')
            texts.add(key)
            row = dict(native, question=item['question'], role=role, label='positive',
                       logical_id=f'positive-{role}-{index}', fact_relation='reviewed_same_fact_text_augmentation',
                       rewrite_family=item['family'], probe_kind=item.get('probe_kind', 'cross_family_confirmation'))
            output.append(row)
            if role == 'fit':
                fit.append({'question': item['question'], 'family': item['family'], 'review_status': 'APPROVED_EQUIVALENT'})
    if any(families[a] & families[b] for a, b in (('fit', 'calibration'), ('fit', 'evaluation'), ('calibration', 'evaluation'))):
        raise ValueError('rewrite families cross roles; different wrapper IDs alone are not independent families')
    if not any(row.get('probe_kind') == 'source_style_confirmation' for row in output if row['role'] == 'evaluation'):
        raise ValueError('evaluation lacks reviewed source-style confirmation probes')
    return output, fit, []


def reviewed_attribute(question):
    """Source-question-only review of the finite native attribute families in this cohort."""
    q = normalized(question)
    if q == 'which part of the body does this image belong to?':
        return 'body_region'
    match = re.fullmatch(r'does the picture contain (spleen|kidney|liver)\?', q)
    if match:
        return 'presence:' + match[1]
    if q == 'what is the mr weighting in this image?':
        return 'mr_weighting'
    if q == 'what is the largest organ in the picture?':
        return 'largest_visible_organ'
    if q in ('what imaging modality was used?', 'what modality is used to take this image?', 'is this image taken via ct?'):
        return 'modality'
    if q in ('what is the scanning plane of this image?', 'is this a transverse plane?'):
        return 'plane'
    return None


def source_review(source):
    """Prospective agent equivalence review, not a human/clinical signoff.

    Fit uses direct lexical interrogatives; calibration retains the approved V1
    source-question context wrappers. Evaluation is two distinct syntactic families:
    source-style interrogatives and imperative attribute requests, never random
    splits of one generated template batch. No new evaluation output is inspected.
    """
    attribute = reviewed_attribute(source['question'])
    if attribute == 'body_region':
        fit = ['What body part is shown in this image?', 'Which body region does this picture show?',
               'What part of the body is depicted here?', 'Which body part does the image depict?']
        evaluation = ['Which anatomical region is depicted in the image?', 'What is the anatomical region shown here?',
                      'Identify the region of the body represented in this image.', 'Name the body region depicted in this picture.']
    elif attribute and attribute.startswith('presence:'):
        organ = attribute.split(':')[1]
        fit = [f'Does this image contain the {organ}?', f'Does this picture show the {organ}?',
               f'Is the {organ} shown in this image?', f'Can the {organ} be seen in this picture?']
        evaluation = [f'Is the {organ} present in the displayed image?', f'Is there a {organ} visible in this image?',
                      f'Indicate whether the {organ} is visible in the image.', f'State whether this picture contains the {organ}.']
    elif attribute == 'mr_weighting':
        fit = ['What MR weighting is shown here?', 'Which MR weighting does this image use?',
               'What is the MRI weighting of this picture?', 'Which MRI weighting is displayed in this image?']
        evaluation = ['What magnetic resonance weighting is represented in this image?', 'Which magnetic resonance weighting does this image show?',
                      'Identify the MR weighting represented in the image.', 'Name the MRI weighting shown in this picture.']
    elif attribute == 'largest_visible_organ':
        fit = ['Which organ is largest in this image?', 'What is the biggest organ shown in the picture?',
               'Which of the organs in this image is the largest?', 'What organ is the largest one depicted here?']
        evaluation = ['Which visible organ is greatest in size in this picture?', 'What is the largest of the organs depicted in the image?',
                      'Identify the largest organ shown in this picture.', 'Name the biggest organ depicted in this image.']
    else:
        raise ValueError('source attribute is outside the reviewed finite cohort families')
    calibration = [prefix + source['question'] for prefix in ('Based on this image, answer: ',
        'Using the visual evidence shown, answer: ', 'Review the displayed image and answer: ', 'With reference to this image, answer: ')]
    positives = dict(fit=[dict(question=q, family='direct_lexical_question') for q in fit],
        calibration=[dict(question=q, family='V1_context_wrapper') for q in calibration],
        evaluation=[dict(question=q, family='source_style_interrogative' if i < 2 else 'imperative_attribute_request',
            probe_kind='source_style_confirmation' if i < 2 else 'cross_family_confirmation') for i, q in enumerate(evaluation)])
    return dict(visibility='SOURCE_QUESTION_ONLY__NO_STUDENT_OUTPUTS', approved_equivalent=True,
        reviewer='Codex source-only prospective linguistic review; not independent clinical or human signoff',
        clinical_attribute=attribute, preserved='attribute, polarity, referent, laterality, tense and answer task', positives=positives)


def build(config, pool):
    reviews = read(config['positive_review']).get('records', {}) if config.get('positive_review') else {}
    chosen = {}
    for row in sorted(pool, key=lambda row: (0 if row['dataset'] == 'SLAKE' else 1, row['image_id'], str(row['source_qid']))):
        # Frozen Base-before only; no student success filtering, no repeated seeds.
        if row['base_correct'] is False and Path(row['image_path']).is_file():
            chosen.setdefault(row['source_group'], row)
    # V1 config contains novel_n=0; it is historical evidence, not V2's target.
    selected = list(chosen.values())[:min(int(config.get('stage2_n', 16)), 16)]
    target_groups = {row['source_group'] for row in selected}
    negative_groups = sorted({row['source_group'] for row in pool} - target_groups)
    episodes = []
    for index, source in enumerate(selected, 101):
        record_id = f'medtrace-stage2-e{index}'
        native = dict(source, role='native', label='positive', logical_id='native', fact_relation='native')
        review = reviews.get(source['dataset'] + ':' + str(source['source_qid']))
        if config.get('stage2_source_review_v2'):
            review = source_review(source)
        positive, fit, pending = positive_rows(review, native)
        for row in positive:
            if row['role'] == 'evaluation':
                row['confirmation_panel'] = row['probe_kind']
        rows = [native, *positive]
        native_attribute = reviewed_attribute(source['question'])
        ordered = sorted(pool, key=lambda row: hashlib.sha256((record_id+'\0'+row['source_group']+'\0'+str(row['source_qid'])).encode()).hexdigest())
        hard, contexts, used_images = [], [], set()
        for other in ordered:
            # Match the reviewed clinical attribute AND original annotated question;
            # different gold/image alone never establishes conflict.
            if (other['source_group'] in target_groups or other['source_group'] in used_images or
                    normalized(other['question']) != normalized(source['question']) or
                    reviewed_attribute(other['question']) != native_attribute):
                continue
            used_images.add(other['source_group'])
            (hard if normalized(other['reference']) != normalized(source['reference']) else contexts).append(other)
        broad = []
        for other in ordered:
            if other['source_group'] in target_groups or other['source_group'] in used_images:
                continue
            attribute = reviewed_attribute(other['question'])
            source_type = other['source_annotation'].get('question_type')
            # These source-annotated acquisition attributes differ from all six
            # reviewed native attributes except their named same-attribute case.
            if attribute is None and source_type in ('MODALITY', 'PLANE'):
                attribute = source_type.lower()
            if attribute is None or attribute == native_attribute:
                continue
            used_images.add(other['source_group'])
            broad.append(other)
        for group, negatives in (('H', hard), ('U', broad)):
            for j, other in enumerate(negatives):
                role = ROLES[j % 3]
                if group == 'U' and j >= 60:
                    break
                rows.append(dict(other, role=role, label='negative', negative_group=group,
                    fact_relation='same_question_different_image_conflicting_source_answer' if group == 'H' else 'broad_unrelated_source_qa',
                    logical_id=f'negative-{group}-{role}-{j}', conflict_verified=group == 'H',
                    relation_evidence='source-annotated same reviewed clinical attribute and identical original question, distinct annotated value on another image' if group == 'H' else 'different reviewed or source-annotated clinical/acquisition attribute on another image',
                    H_keep=other['base_correct'] if group == 'H' else None))
        for j, other in enumerate(contexts):
            rows.append(dict(other, role='challenge', label='negative', logical_id=f'same-answer-context-{j}',
                fact_relation='same_question_other_image_same_source_answer',
                confirmation_panel='same_answer_context_not_conflict', conflict_verified=False))
        support = {role: {group: sum(row.get('negative_group') == group and row['role'] == role for row in rows)
                          for group in ('H', 'U')} for role in ROLES}
        if not all(support['fit'].values()):
            pending.append('SELECTIVE_WRITE_FIT_H_OR_U_UNSUPPORTED')
        event = {'event_id': record_id, 'event_position': index,
                 'edit_record': {'record_id': record_id, 'dataset': source['dataset'], 'question': source['question'],
                                 'gold_answer': source['reference'], 'official_rephrase': fit[0]['question'] if fit else '',
                                 'router_positive_source': 'reviewed_source_style_not_official',
                                 'image_path': source['image_path'], 'relative_image_path': source['image_id'],
                                 'formal_sequence_position': index, 'question_type': 'SOURCE_CONFIRMATION'},
                 'probes': [{'probe_id': row['logical_id'], 'task': 'SOURCE_STYLE_CONFIRMATION',
                             'question': row['question'], 'reference': row['reference'], 'image_path': row['image_path'],
                             'dataset': source['dataset'], 'variant_type': row.get('probe_kind', 'cross_family_confirmation')}
                            for row in positive if row['role'] == 'evaluation']}
        episodes.append({'event_index': index, 'record_id': record_id, 'seed': 20260910, 'event': event,
                         'track': 'NEW_CONFIRMATION', 'source_eligibility_frozen': not pending,
                         'rows': rows, 'fit_paraphrases': fit, 'negative_support': support,
                         'trainable': not pending, 'pending': pending, 'patient_id': 'UNKNOWN', 'positive_review': review,
                         'H_all_retained_before_Base_correct_filter': True,
                         'role_isolation': 'source-image disjoint within each independent edit; cross-edit shared negatives are reported, not independent samples',
                         'image_generality': 'NA; no authorized equivalent image variants found in source-only pool'})
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--public-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--review-v2', action='store_true')
    parser.add_argument('--install-queue', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert image_groups({'negatives': {'evaluation': [{'image_name': 'xmlab2/source.jpg'}]}, 'dataset': 'SLAKE'}) == {('SLAKE', 'xmlab2')}
        assert positive_rows(None, {'question': 'native'})[2]
        for question in ('Does the picture contain spleen?', 'Which part of the body does this image belong to?',
                         'What is the mr weighting in this image?', 'What is the largest organ in the picture?'):
            source = dict(question=question)
            positive, fit, pending = positive_rows(source_review(source), source)
            assert len(positive) == 12 and len(fit) == 4 and not pending
        review = {'visibility': 'SOURCE_QUESTION_ONLY__NO_STUDENT_OUTPUTS', 'approved_equivalent': True,
                  'positives': {role: [{'question': f'{role} {i}', 'family': 'same', 'probe_kind': 'source_style_confirmation'} for i in range(4)] for role in ROLES}}
        try:
            positive_rows(review, {'question': 'native'})
        except ValueError:
            pass
        else:
            raise AssertionError('cross-role families accepted')
        print('source scanner self-test PASS')
        return
    if not args.config or not args.run_root:
        parser.error('--config and --run-root are required')
    config = read(args.config)
    config['stage2_source_review_v2'] = args.review_v2
    if not config.get('stage1_run') or not config.get('runtime'):
        raise ValueError('config must bind stage1_run and frozen runtime')
    private = args.run_root / 'private/NEW_EPISODE_MANIFEST_PRIVATE.json'
    if private.exists():
        raise FileExistsError(private)
    pool, summary, ledger = scan(config)
    episodes = build(config, pool)
    summary.update(candidate_episode_count=len(episodes), actual_n=sum(e['trainable'] for e in episodes), ready_episode_count=sum(e['trainable'] for e in episodes),
                   status='SOURCE_SCREEN_COMPLETE__REVIEW_READINESS_EXPLICIT',
                   negative_relation_contract='H requires same reviewed clinical attribute, identical original annotated question and distinct annotated value on another source image; U requires a different reviewed or source-annotated attribute. Same-answer contexts are separate, not conflicts.',
                   positive_review='prospective source-only agent linguistic review, not human/clinical signoff' if args.review_v2 else 'PENDING',
                   evaluation_families=['source_style_confirmation', 'cross_family_confirmation'],
                   official_image_generality='NA; no authorized equivalent image variants in scanned source pool')
    write_new(private, {'schema_version': 'medtrace-stage2-source-episodes-v2', 'stage1_run': config['stage1_run'],
                        'runtime': config['runtime'], 'episodes': episodes, 'source_inventory': summary,
                        'exposure_exclusion_ledger': ledger, 'source_pool': pool})
    public = args.public_dir or args.run_root / 'public'
    public_summary = {k: value for k, value in summary.items() if k not in ('evidence', 'patient_id')}
    public_summary['patient_identity'] = 'UNKNOWN; no patient-disjoint claim'
    public_summary['evidence'] = [dict(file=Path(item['path']).name, reason=item['reason'], image_groups=item['image_groups']) for item in summary['evidence']]
    write_new(public / 'NEW_EPISODE_MANIFEST_PUBLIC.json', public_summary)
    write_new(public / 'SOURCE_EXPOSURE_SUMMARY.json', {k: public_summary[k] for k in ('source_counts', 'screened_development_pool', 'exclusion_counts_nonadditive', 'evidence', 'membership_boundary')})
    if args.install_queue:
        if not args.review_v2 or not all(e['trainable'] for e in episodes):
            raise ValueError('queue admission requires reviewed families and actual fit H/U support; candidates remain recorded')
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.medtrace.run_stage2 import queue_task, NEW_METHODS, vf
        base_tasks, new_tasks = [], []
        for episode in episodes:
            i = episode['event_index']
            write_new(args.run_root / f'private/edits/e{i:02d}.json', episode)
            base_tasks.append(queue_task(i, 'BASE', i))
            be = queue_task(i, 'BE', 1000+(i-101)*10)
            init = queue_task(i, 'INIT', be['priority']+1, be['task_id'])
            new_tasks.extend([be, init])
            new_tasks.extend(queue_task(i, 'CP', be['priority']+2+j, init['task_id'], p, c) for j, (p, c) in enumerate(NEW_METHODS))
        for task in new_tasks:
            task['status'] = 'WAITING_BASE_BEFORE'
        write_new(args.run_root / 'private/BASE_TASK_QUEUE.json', dict(tasks=base_tasks))
        queue = vf.TaskQueue(args.run_root / 'private/TASK_QUEUE.json', args.run_root)
        def append(data):
            if any(t['event_index'] >= 100 for t in data['tasks']):
                raise ValueError('new queue already installed')
            data['tasks'].extend(new_tasks)
        queue._locked(append)
    print(json.dumps({'private_manifest': str(private), 'candidate_episodes': len(episodes),
                      'ready_episodes': summary['ready_episode_count'], 'public_dir': str(public)}))


if __name__ == '__main__':
    main()
