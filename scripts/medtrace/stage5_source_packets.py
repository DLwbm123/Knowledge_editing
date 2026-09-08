#!/usr/bin/env python3
"""Materialize a finite operator-reviewed source packet, never generate medical facts."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.run_stage4 import read, vf
from scripts.medtrace import prepare_stage2_sources as s


def main(args):
    run = args.run_root
    if (run/'private/NEW_SOURCE_REVIEW_MANIFEST.json').exists():
        raise FileExistsError('source roles already frozen')
    config = read(run/'private/CAMPAIGN_CONFIG.json')
    candidates = read(run/'private/INHERITED_CANDIDATES.json')['candidates']
    bounded = read(run/'private/INHERITED_SOURCE_POOL.json')['rows']
    review = read(args.review)
    assert review['status'] == 'SOURCE_CONSISTENCY_REVIEWED_DERIVED_TEXT'
    assert set(review['records']) == {str(c['source_qid']) for c in candidates}
    # The reviewed input contains explicit qid-to-attribute assignments. Never
    # infer a medical relation from answer inequality or the broad PRES label.
    attrs = review['source_attributes']
    negative_ids = review['negative_source_qids']
    by_id = {str(r['qid']): r for r in bounded}
    catalog = Path(config['runtime']['cpu_gate']).parents[2]/'unused'
    static = Path('/remote-home/wangbomin/Knowledge_editing/outputs/m3bench_data_runtime_finalization_v3/20260904T014138Z/data_static/STATIC_QUERY_INVENTORY.jsonl')
    allowed_images = {r['image_name'].lower() for r in bounded}
    # Only previously cleared source images contribute content to this index.
    identities = {}
    for r in s.jsonl(static):
        if r['dataset'] == 'VQA-RAD' and s.identity(r['image_id']) in allowed_images:
            key = (s.identity(r['image_id']), s.normalized(r['question']), s.normalized(r['gold_answer']))
            identities.setdefault(key, []).append(r['query_id'])
    before = {r['query_id']: r['is_correct'] for r in s.jsonl(Path(candidates[0]['base_verdict_path']))}
    native_groups = {c['image_id'] for c in candidates}
    def source_row(qid):
        r = by_id[str(qid)]; image = r['image_name'].lower()
        assert image not in native_groups, 'negative image is a new target'
        qids = identities.get((image, s.normalized(r['question']), s.normalized(r['answer'])), [])
        values = {before[q] for q in qids if q in before}
        assert qids and len(values) == 1, 'source Base membership unavailable'
        return dict(dataset='VQA-RAD', source_qid=r['qid'], image_id=image,
            source_group='VQA-RAD:'+image, image_path=str(Path(candidates[0]['image_path']).parent/r['image_name']),
            question=r['question'], reference=str(r['answer']), source_role=r['phrase_type'],
            source_file=candidates[0]['source_file'], base_query_ids=qids,
            base_correct=next(iter(values)), patient_id='UNKNOWN',
            source_annotation={'question_type':r['question_type']})
    negatives = [source_row(qid) for qid in negative_ids]
    assert len({r['source_group'] for r in negatives}) == len(negatives), 'one source question per negative image'
    # Frozen global image roles avoid cross-edit fit/calibration/evaluation leakage.
    hard_image = source_row(141)['source_group']
    groups = sorted({r['source_group'] for r in negatives} - {hard_image})
    group_role = {g:s.ROLES[i % 3] for i,g in enumerate(groups)}
    group_role[hard_image] = 'fit'
    episodes, packets, public = [], [], []
    for index, c in enumerate(candidates, 201):
        qid = str(c['source_qid']); texts = review['records'][qid]
        assert len(texts) == 8
        normalized = [s.normalized(c['question']), *map(s.normalized, texts)]
        assert len(set(normalized)) == len(normalized), 'native duplicate or cross-role text collision'
        rows = [dict(c, reference=str(c['reference']), role='native', label='positive', logical_id='native', fact_relation='native')]
        text_audits, fit = [], []
        for j, text in enumerate(texts):
            role = 'fit' if j < 2 else 'calibration' if j < 4 else 'evaluation'
            style = 'source_style_interrogative' if j < 6 else 'imperative_attribute_request'
            # All paraphrases of one native fact remain one semantic lineage.
            family = 'native_fact_derived_text'
            rows.append(dict(c, reference=str(c['reference']), question=text, role=role, label='positive',
                logical_id=f'positive-{role}-{j}', fact_relation='reviewed_same_fact_text_augmentation',
                rewrite_family=family, surface_style=style,
                probe_kind='source_style_confirmation' if j < 6 else 'cross_family_confirmation',
                confirmation_panel='source_style_confirmation' if j < 6 else 'cross_family_confirmation'))
            if role == 'fit':
                fit.append(dict(question=text, family=family, review_status=review['status']))
            text_audits.append(dict(original_question=c['question'], candidate=text, role=role,
                status=review['status'], review_version=review['review_version'],
                semantic_lineage=f'VQA-RAD:{qid}', surface_style=style,
                equivalent_reason='Same explicitly sourced referent and requested attribute; no answer/diagnosis generated.',
                slots=dict(object='preserved', attribute=attrs[qid], negation='preserved', laterality='preserved',
                    numeric='preserved', temporal_condition='preserved', answer_granularity='preserved'),
                clinical_human_reviewed=False))
        for other in negatives:
            other_qid = str(other['source_qid'])
            if attrs[other_qid] == attrs[qid]:
                # Only this finite yes/no predicate conflict was reviewed.
                if (qid, other_qid) != ('1715', '141'):
                    continue
                group, relation = 'H', 'same_question_different_image_conflicting_source_answer'
                evidence = 'Source asks gallbladder enlargement on each image; gall bladder spelling normalized; native No versus other Yes. Same binary predicate, no new image interpretation.'
            else:
                group, relation = 'U', 'broad_unrelated_source_qa'
                evidence = 'Distinct explicitly reviewed source attribute on another source image: '+attrs[qid]+' versus '+attrs[other_qid]
            role = group_role[other['source_group']]
            rows.append(dict(other, role=role, label='negative', negative_group=group,
                logical_id=f'negative-{group}-{role}-{other_qid}', fact_relation=relation,
                conflict_verified=group == 'H', relation_evidence=evidence,
                H_keep=other['base_correct'] if group == 'H' else None))
        support = {role:{g:sum(r['role'] == role and r.get('negative_group') == g for r in rows)
                         for g in ('H','U')} for role in s.ROLES}
        rid = f'medtrace-stage5-e{index}'
        event = dict(event_id=rid, event_position=index, edit_record=dict(record_id=rid,
            dataset=c['dataset'], question=c['question'], gold_answer=str(c['reference']),
            official_rephrase=fit[0]['question'], router_positive_source='SOURCE_CONSISTENCY_REVIEWED_DERIVED_TEXT_NOT_OFFICIAL',
            image_path=c['image_path'], relative_image_path=c['image_id'], formal_sequence_position=index,
            question_type='SOURCE_CONFIRMATION'), probes=[])
        coverage = dict(W0='SUPPORTED', BE='SUPPORTED',
            W1='SUPPORTED' if all(support['fit'].values()) else 'UNSUPPORTED_MISSING_FIT_H_OR_U')
        packet = dict(candidate_index=index-200, source_qid=c['source_qid'], native=c, event_index=index,
            source_use_authorized=True, text_equivalence_reviewed=True, clinical_human_reviewed=False,
            clinical_review_requirement='No mandatory clinical signature identified in inherited source-use locks; no clinical validity endorsement.',
            text_audits=text_audits, negative_support=support, method_support=coverage,
            original_image_generality='NA', original_official_text_generality='NA',
            independent_source_image=True, patient_id='UNKNOWN', status='SOURCE_PACKET_FROZEN',
            review_needed=review.get('review_needed', {}).get(qid, []))
        assert not packet['review_needed'], 'unresolved text cannot enter training'
        episode = dict(event_index=index, record_id=rid, seed=20260910, event=event,
            track='NEW_CONFIRMATION', source_eligibility_frozen=True, rows=rows, fit_paraphrases=fit,
            extra_fit=[], negative_support=support, method_support=coverage, patient_id='UNKNOWN',
            positive_review=dict(approved_equivalent=True, reviewer=review['review_type'],
                status=review['status'], semantic_families_independent=False),
            image_generality='NA', role_isolation='globally disjoint negative source-image roles; native image shared across positive roles',
            source_packet=packet)
        episodes.append(episode);packets.append(packet)
        public.append(dict(candidate_index=index-200, event_index=index, source_use_authorized=True,
            text_equivalence_reviewed=True, clinical_human_reviewed=False, method_support=coverage,
            positive_counts=dict(fit=2, calibration=2, evaluation=4), negative_support=support,
            status='SOURCE_PACKET_FROZEN', original_image_generality='NA', official_text_generality='NA',
            patient_id='UNKNOWN', semantic_family_count=1, distinct_surface_styles=2))
    for e in episodes:
        vf.atomic_json(run/f"private/edits/e{e['event_index']:02d}.json", e)
    vf.atomic_json(run/'private/NEW_EPISODE_MANIFEST_PRIVATE.json', dict(episodes=episodes))
    vf.atomic_json(run/'private/NEW_SOURCE_REVIEW_MANIFEST.json', dict(records=packets, global_negative_roles=group_role,
        source_review=review['review_type'], review_version=review['review_version']))
    vf.atomic_json(run/'public/NEW_SOURCE_REVIEW_MANIFEST.json', dict(records=public,
        new_source_images=32, patient_independence='UNKNOWN',
        source_text='Derived language, not official M3Bench rephrases or independent clinical annotation',
        family_limitation='All derivatives share native semantic lineage. Surface style is not semantic independence.',
        H_limitation='Only one finite binary source conflict supported; W1 missing-fit endpoints remain unsupported.',
        method_support_counts={m:sum(e['method_support'][m] == 'SUPPORTED' for e in episodes) for m in ('W0','W1','BE')}))
    print('SOURCE_PACKETS_FROZEN', len(episodes), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True, type=Path)
    p.add_argument('--review', required=True, type=Path)
    main(p.parse_args())
