"""Run with python -B tests/medtrace/test_stage3_sources.py (CPU; no writes)."""
from collections import Counter
import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.medtrace import prepare_stage3 as stage3


def event(rid, task='T0', question='Unknown clinical question?', probes=None):
    edit = dict(record_id=rid, dataset='SLAKE', image_path=f'/images/{rid}/source.jpg',
        relative_image_path=rid, question=question, gold_answer='yes', official_rephrase='Unapproved official?',
        formal_sequence_position=1, question_type=task)
    return dict(event_id=task+':'+rid, task=task, event_position=1, edit_record=edit,
        probes=probes if probes is not None else [dict(probe_id=rid, task=task, dataset='SLAKE',
            image_path=edit['image_path'], question=question, reference='yes')])


def manifest(catalog):
    es = Counter(e['task'] for e in catalog)
    ps = Counter(p['task'] for e in catalog for p in e['probes'])
    return dict(tasks={t: dict(eligible_edit_count=es[t], eligible_probe_count=ps[t]) for t in stage3.TASKS})


def plan(catalog, sidecars=(), pool=(), inventory=(), historical=None):
    return stage3.build_plan(catalog, manifest(catalog), dict(source_pool=list(pool)), historical or {},
                             list(sidecars), list(inventory), check_files=False)


class Stage3SourcesTest(unittest.TestCase):
    def test_writer_relation_constants_match_without_loading_torch(self):
        tree = ast.parse((stage3.ROOT / 'methods/medtrace/selective_write.py').read_text())
        relation = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'RELATIONS' for t in node.targets))
        self.assertEqual(stage3.RELATIONS, relation)

    def test_explicit_official_support_admitted_but_formal_overlap_rejected(self):
        anchor = event('a')
        anchor['edit_record']['router_positive_source'] = 'legacy_official_rephrase'
        anchor['official_support_authority'] = dict(method_lock='frozen-method-lock', permitted_role='editing_support')
        first = plan([anchor])[0][0]
        self.assertEqual(first['executable_methods'], ['BE', 'S0'])
        self.assertEqual(first['fit_paraphrases'][0]['family'], None)
        self.assertEqual(first['official_rephrase_use']['status'], 'FROZEN_OFFICIAL_EDITING_SUPPORT')
        other = event('b', 'T2L', probes=[dict(probe_id='p', task='T2L', dataset='SLAKE',
            image_path='/other.jpg', question=anchor['edit_record']['official_rephrase'], reference='yes')])
        first = plan([anchor, other])[0][0]
        self.assertEqual(first['executable_methods'], [])
        self.assertEqual(first['official_rephrase_use']['status'], 'REJECTED_FORMAL_OR_HELDOUT_QUERY_OVERLAP')

    def test_exact_catalog_grouping_preserves_task_specific_edit(self):
        anchor = event('a')
        bound = event('a', 'T2G', probes=[dict(probe_id='p', task='T2G', dataset='SLAKE',
            image_path='/images/a/source.jpg', question='Formal rephrase?', reference='yes')])
        other = event('b', 'T2L', probes=[])
        catalog = [anchor, bound, other]
        grouped, counts = stage3.group_catalog(catalog, manifest(catalog))
        self.assertEqual((counts['catalog_events'], counts['planned_groups'], counts['t0_anchors']), (3, 2, 1))
        self.assertEqual(grouped[0]['catalog_event_ids'], ['T0:a', 'T2G:a'])
        self.assertEqual(grouped[1]['edit_record']['record_id'], 'b')
        self.assertEqual(catalog[0]['probes'][0]['probe_id'], 'a')  # input immutable
        bad = deepcopy(catalog)
        bad[1]['edit_record']['gold_answer'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identical T0'):
            stage3.group_catalog(bad, manifest(bad))

    def test_global_query_family_and_image_alias_protection(self):
        anchor = event('a')
        probe = dict(probe_id='p', task='T2G', dataset='SLAKE', image_path='/eval/image.jpg',
                     question='Held out?', reference='yes', rewrite_family='protected')
        other = event('b', 'T2L', probes=[dict(probe, task='T2L')])
        fit = [dict(question=q, family=f, review_status='APPROVED_EQUIVALENT')
               for q, f in [('Held out?', 'f1'), ('Different text?', 'protected'), ('Safe approved?', 'safe')]]
        negative = dict(role='fit', label='negative', image_path='/alias/image.jpg', question='Negative?',
                        reference='no', fact_relation=stage3.RELATIONS['H'], source_group='alias')
        sidecar = dict(event=anchor, support_sidecar='approved-sidecar', fit_paraphrases=fit, rows=[negative])
        inventory = [dict(query_id=q, dataset='SLAKE', image_id=q, image_path=p, image_sha256='existing-digest',
                          question='Held out?') for q, p in [('p', '/eval/image.jpg'), ('alias', '/alias/image.jpg')]]
        episodes, queue, counts = plan([anchor, other], [sidecar], inventory=inventory)
        first = episodes[0]
        self.assertEqual([p['question'] for p in first['fit_paraphrases']], ['Safe approved?'])
        self.assertEqual(first['executable_methods'], ['BE', 'S0'])
        self.assertEqual(first['negative_support']['fit']['H'], 0)
        self.assertEqual(queue[0]['methods'], ['BE', 'S0'])
        self.assertEqual(queue[0]['allplanned'], ['BE', 'S0', 'S1'])
        self.assertTrue(queue[0]['unsupported_reasons']['S1'])
        self.assertEqual(counts['executable_formal_probes']['BE'], {'T0': 1})
        self.assertEqual(first['rows'][-1]['base_query_ids'], [])

    def test_no_invented_equivalence_or_identity_anchor(self):
        catalog = [event('a')]
        episodes, queue, counts = plan(catalog)
        self.assertTrue(episodes[0]['source_eligibility_frozen'])
        self.assertEqual(episodes[0]['event']['edit_record']['official_rephrase'], '')
        self.assertEqual(queue[0]['methods'], [])
        self.assertEqual(queue[0]['status'], 'UNSUPPORTED_TRAINING_INPUTS')
        self.assertEqual(counts['common_support_groups'], 0)
        # Approved support for a different image/fact is not portable by question.
        other = event('b')
        sidecar = dict(event=other, support_sidecar='elsewhere', rows=[],
            fit_paraphrases=[dict(question='Safe?', family='f', review_status='APPROVED_EQUIVALENT')])
        self.assertEqual(plan(catalog, [sidecar])[0][0]['fit_paraphrases'], [])

    def test_native_evaluation_collision_blocks_only_affected_group(self):
        anchor = event('a')
        other = event('b', 'T2L', probes=[dict(anchor['probes'][0], task='T2L')])
        fit = [dict(question='Safe?', family='safe', review_status='APPROVED_EQUIVALENT')]
        sidecar = dict(event=anchor, support_sidecar='old', rows=[], fit_paraphrases=fit)
        episodes, queue, _ = plan([anchor, other], [sidecar])
        self.assertIn('NATIVE_IS_PROTECTED_EVALUATION_QUERY', episodes[0]['unsupported_reasons']['BE'])
        self.assertEqual(len(queue), 2)

    def test_existing_source_builder_support_and_auxiliary_namespace(self):
        question = 'Which part of the body does this image belong to?'
        catalog = [event('a', question=question)]
        pool = [dict(dataset='SLAKE', source_qid=i, source_group=f'pool-{i}', source_annotation={},
            image_path=f'/pool/{i}/source.jpg', image_id=str(i), reference='other', base_correct=None,
            question=question if i < 3 else 'What imaging modality was used?') for i in range(6)]
        episodes, _, counts = plan(catalog, pool=pool)
        self.assertEqual(episodes[0]['executable_methods'], ['BE', 'S0', 'S1'])
        self.assertEqual(len(episodes[0]['fit_paraphrases']), 4)
        self.assertEqual(counts['common_support_groups'], 1)
        self.assertEqual(counts['executable_formal_probes']['S1'], {'T0': 1})
        for role in ('fit', 'calibration', 'evaluation'):
            self.assertEqual(episodes[0]['negative_support'][role], dict(H=1, U=1))

    def test_prepare_new_epoch_no_start_and_no_historical_overwrite(self):
        catalog = [event('a')]
        oldconfig = dict(stage1_run='/stage1', old_run='/history', runtime={}, code_commit='old-code',
                         campaign_epoch=10, reuse_stage2_run='/r01')
        loaded = (oldconfig, catalog, manifest(catalog), dict(source_pool=[]), {}, [], [], {})
        written = {}
        args = SimpleNamespace(run_root=Path('/new-run'), stage2_run=Path('/stage2'), preview=False)
        with patch.object(stage3, 'load_sources', return_value=deepcopy(loaded)), \
             patch.object(Path, 'exists', return_value=False), patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'mkdir'), patch.object(stage3.time, 'time', return_value=100), \
             patch.object(stage3.subprocess, 'check_output', return_value='new-code\n'), \
             patch.object(stage3.source, 'write_new', side_effect=lambda path, value: written.update({str(path): value})):
            stage3.prepare(args)
        config = written['/new-run/private/CAMPAIGN_CONFIG.json']
        self.assertEqual(config['campaign_epoch'], 100)
        self.assertEqual(config['kind'], 'MEDTRACE_STAGE3')
        self.assertEqual(config['stage1_run'], '/stage1')
        self.assertNotIn('reuse_stage2_run', config)
        self.assertIn('/new-run/private/edits/e01.json', written)
        self.assertFalse(any('START' in p for p in written))
        self.assertTrue(all(p.startswith('/new-run/') for p in written))
        tasks = written['/new-run/private/TASK_QUEUE.json']['tasks']
        self.assertEqual([t['prefix'] for t in tasks[:4]], [1, 4, 8, 16])
        self.assertEqual(oldconfig['campaign_epoch'], 10)


if __name__ == '__main__':
    unittest.main()
