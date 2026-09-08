#!/usr/bin/env python3
"""Source-only Stage2 inventory and episode data builder; never loads a model.

python scripts/medtrace/prepare_stage2_sources.py --config CONFIG --run-root RUN
Optional CONFIG.positive_review points to source-question-only reviewed families.
Without that review, candidates are emitted honestly as NOT_TRAINABLE drafts.
"""
import argparse
import csv
import json
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
    unmapped = []
    for name in ('LORA_DEV16', 'LORA_QUAL16', 'LORA_SEQ16', 'QUAL8'):
        path = v4 / 'manifests_v2' / (name + '_V2_MANIFEST.json')
        ids = read(path)['event_ids']
        exclude({by_query[q.removeprefix('T0:')] for q in ids if q.removeprefix('T0:') in by_query}, name, path)
        unmapped.extend({'manifest': str(path), 'event_id': q} for q in ids if q.removeprefix('T0:') not in by_query)
    exclude({('SLAKE', 'xmlab281'), ('SLAKE', 'xmlab281_3')}, 'disabled_source_group', 'historical_exclusion')
    live = root / 'Knowledge_editing/outputs/liveedit_med_eqkey_clean_fast_confirmation_v1/20260815T122907Z/eqkey_family_audit/positive_input_ledger.csv'
    with live.open() as handle:
        live_hashes = {row['raw_image_sha256'] for row in csv.DictReader(handle) if row.get('raw_image_sha256')}
    blocked_hashes = live_hashes | {row['image_sha256'] for row in catalog
                                  if (row['dataset'], identity(row['image_id'])) in reasons and row.get('image_sha256')}
    exclude({(row['dataset'], identity(row['image_id'])) for row in catalog if row.get('image_sha256') in blocked_hashes},
            'existing_image_hash_identity_closure', live)
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
    query_index = defaultdict(list)
    for row in catalog:
        query_index[(row['dataset'], identity(row['image_id']), normalized(row['question']), normalized(row['gold_answer']))].append(row['query_id'])
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
                         'source_group': dataset + ':' + image_id, 'patient_id': 'UNKNOWN',
                         'source_file': str(sources / ('SLAKE/train.json' if dataset == 'SLAKE' else 'VQA-RAD/VQA_RAD Dataset Public.json')),
                         'source_role': 'train' if dataset == 'SLAKE' else row['phrase_type'],
                         'source_annotation': {k: row[k] for k in ('img_id', 'triple', 'base_type', 'question_type', 'qid_linked_id', 'question_relation') if k in row},
                         'base_query_ids': qids, 'base_correct': before, 'base_verdict_path': str(verdict_path)})
    summary = {'source_counts': {'SLAKE/train': len(slake['train']), 'SLAKE/validation': len(slake['validation']),
                                'SLAKE/test': len(slake['test']), 'VQA-RAD/all': len(vqa)},
               'screened_development_pool': {d: {'rows': sum(row['dataset'] == d for row in pool),
                    'images': len({row['source_group'] for row in pool if row['dataset'] == d}),
                    'base_incorrect_rows': sum(row['dataset'] == d and row['base_correct'] is False for row in pool)} for d in ('SLAKE', 'VQA-RAD')},
               'exclusion_counts_nonadditive': dict(counts), 'evidence': evidence,
               'unresolved_direct_event_id_mappings': unmapped,
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
    global_roles = {group: ROLES[index % 3] for index, group in enumerate(negative_groups)}
    episodes = []
    for index, source in enumerate(selected, 101):
        record_id = f'medtrace-stage2-e{index}'
        native = dict(source, role='native', label='positive', logical_id='native', fact_relation='native')
        positive, fit, pending = positive_rows(reviews.get(source['dataset'] + ':' + str(source['source_qid'])), native)
        rows = [native, *positive]
        used = set()
        for other in sorted(pool, key=lambda row: (row['source_group'], str(row['source_qid']))):
            if other['source_group'] not in global_roles:
                continue
            group = 'H' if normalized(other['question']) == normalized(source['question']) else 'U'
            role = global_roles[other['source_group']]
            key = group, role, other['source_group']
            if key in used or sum(row.get('negative_group') == group and row['role'] == role for row in rows) >= (4 if group == 'H' else 20):
                continue
            used.add(key)
            # Explicitly NOT a semantic conflict label based on gold disagreement.
            rows.append(dict(other, role=role, label='negative', negative_group=group,
                             fact_relation='same_question_different_image_source_qa' if group == 'H' else 'broad_source_qa',
                             logical_id=f'negative-{group}-{role}-{len(used)}', conflict_verified=False,
                             relation_evidence='original_source_image_and_question_identity',
                             H_keep=other['base_correct'] if group == 'H' else None))
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
                         'trainable': not pending, 'pending': pending, 'patient_id': 'UNKNOWN'})
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--public-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert image_groups({'negatives': {'evaluation': [{'image_name': 'xmlab2/source.jpg'}]}, 'dataset': 'SLAKE'}) == {('SLAKE', 'xmlab2')}
        assert positive_rows(None, {'question': 'native'})[2]
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
    if not config.get('stage1_run') or not config.get('runtime'):
        raise ValueError('config must bind stage1_run and frozen runtime')
    private = args.run_root / 'private/NEW_EPISODE_MANIFEST_PRIVATE.json'
    if private.exists():
        raise FileExistsError(private)
    pool, summary, ledger = scan(config)
    episodes = build(config, pool)
    summary.update(candidate_episode_count=len(episodes), actual_n=sum(e['trainable'] for e in episodes), ready_episode_count=sum(e['trainable'] for e in episodes),
                   status='SOURCE_SCREEN_COMPLETE__REVIEW_READINESS_EXPLICIT',
                   negative_relation_contract='Use negative_group H/U; do not relabel as verified conflicts or silently pass legacy RELATIONS guards.',
                   unresolved_permissions='Direct non-T0 QUAL8 IDs and older heldout identity closure require owner verification against structural exclusion; no sealed/heldout answers were read by this scanner.')
    write_new(private, {'schema_version': 'medtrace-stage2-source-episodes-v2', 'stage1_run': config['stage1_run'],
                        'runtime': config['runtime'], 'episodes': episodes, 'source_inventory': summary,
                        'exposure_exclusion_ledger': ledger, 'source_pool': pool})
    public = args.public_dir or args.run_root / 'public'
    public_summary = {k: value for k, value in summary.items() if k not in ('evidence', 'patient_id', 'unresolved_direct_event_id_mappings')}
    public_summary['patient_identity'] = 'UNKNOWN; no patient-disjoint claim'
    public_summary['identity_mapping_pending_count'] = len(summary['unresolved_direct_event_id_mappings'])
    public_summary['evidence'] = [dict(file=Path(item['path']).name, reason=item['reason'], image_groups=item['image_groups']) for item in summary['evidence']]
    write_new(public / 'NEW_EPISODE_MANIFEST_PUBLIC.json', public_summary)
    write_new(public / 'SOURCE_EXPOSURE_SUMMARY.json', {k: public_summary[k] for k in ('source_counts', 'screened_development_pool', 'exclusion_counts_nonadditive', 'evidence', 'identity_mapping_pending_count', 'unresolved_permissions')})
    print(json.dumps({'private_manifest': str(private), 'candidate_episodes': len(episodes),
                      'ready_episodes': summary['ready_episode_count'], 'public_dir': str(public)}))


if __name__ == '__main__':
    main()
