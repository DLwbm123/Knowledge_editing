"""Focused CPU fixtures: frozen denominators, Judge ancestry/caps and bank roles."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.medtrace import finalize_stage3 as f


def write(path, value):
    f.vf.atomic_json(path, value)


def output(answer):
    return dict(raw_answer=answer, raw_token_ids=[ord(c) for c in answer])


def row(lid='native', task='T0', reference='yes', **extra):
    return dict(logical_id=lid, question='Private question '+lid, reference=reference,
        image_path='/private/image.png', role='native' if lid == 'native' else 'formal_development',
        label='negative' if task.endswith('L') else 'positive', fact_relation=task, task=task,
        eqkey=lid, source_group='image', **extra)


def campaign(tmp_path, rows=None, methods=('BE', 'S0', 'S1')):
    run, stage1, stage2 = [tmp_path / name for name in ('run', 'stage1', 'stage2')]
    cpu = tmp_path / 'runtime/cpu'
    protocol = dict(config_sha256='protocol', judge_snapshot_sha='snapshot')
    write(cpu.parent / 'private/JUDGE_LOCK_V4.json', protocol)
    write(run / 'private/CAMPAIGN_CONFIG.json', dict(kind='MEDTRACE_STAGE3', stage1_run=str(stage1),
        stage2_run=str(stage2), runtime=dict(cpu_gate=str(cpu))))
    task = dict(task_id='S3_A_e0001_GROUP', kind='SINGLE_GROUP', event_index=1, status='FAILED',
                methods=list(methods), unsupported_reasons={m: ['NO_H_FIT'] for m in f.METHODS if m not in methods})
    write(run / 'private/TASK_QUEUE.json', dict(tasks=[task]))
    rows = rows or [row()]
    write(run / 'private/edits/e01.json', dict(track='V4_STAGE3', rows=rows,
        event=dict(edit_record=dict(gold_answer='yes'))))
    return SimpleNamespace(run_root=run, public_dir=run/'public', max_model_len=2048), protocol


def single(args, method='BE', rows=None, answer='yes', on=True):
    rows = rows or [row()]
    task_id = 'S3_A_e0001_'+method
    task = dict(task_id=task_id, event_index=1, kind='BE' if method == 'BE' else 'CP',
        parameterization='BE' if method == 'BE' else 'P4',
        condition='BALANCEDIT' if method == 'BE' else 'W0_TASK_ONLY' if method == 'S0' else 'W1_KL_0.1', seed=20260910)
    value = dict(status='RAW_READY', task=task, step=50 if method == 'BE' else 320,
        base_guard=dict(unchanged=True), outputs={r['logical_id']: dict(row=r, base=output('no'),
            forced=output(answer), fixed=output(answer if on else 'no'), fixed_on=on, kl=None) for r in rows})
    write(args.run_root / f'private/tasks/{task_id}/result_private.json', value)
    return value


def execution(cap=2048):
    return dict(model=dict(path='/judge', snapshot='snapshot'), tokenizer=dict(file_sha256={}),
        legacy_semantic_protocol_sha256='protocol', runtime=dict(vllm='v1', gpu_name='gpu',
            gpu_uuid='uuid', physical_gpu='2'), generation=dict(max_model_len=cap, max_tokens=24,
            temperature=0, resolved_engine_config='volatile'))


def judge(directory, packets, verdicts, lock=None, reused=None):
    directory.mkdir(parents=True, exist_ok=True)
    packet = directory / 'JUDGE_PACKET_PRIVATE.jsonl'
    f.vf.atomic_text(packet, ''.join(json.dumps(r)+'\n' for r in packets))
    keys = [r['opaque_query_id'] for r in packets]
    lock = deepcopy(lock or execution())
    lock['packet'] = dict(sha256=f.vf.sha256_file(packet))
    write(directory / 'JUDGE_EXECUTION_LOCK_PRIVATE.json', lock)
    f.vf.atomic_text(directory / 'JUDGE_OUTPUT_PRIVATE.jsonl', ''.join(json.dumps(dict(
        opaque_query_id=k, is_correct=verdicts[k], parse_valid=True,
        legacy_semantic_protocol_sha256='protocol', judge_snapshot_sha='snapshot'))+'\n' for k in keys))
    side = dict(expected=keys, all_expected=keys+list(reused or {}), packet_sha256=lock['packet']['sha256'],
        protocol_sha256='protocol', snapshot='snapshot', new=len(keys), reused=len(reused or {}))
    write(directory / 'JUDGE_SIDECAR_PRIVATE.json', side)
    if reused is not None:
        write(directory / 'REUSED_VERDICTS_PRIVATE.json', reused)
        write(directory / 'REUSE_EXECUTION_IDENTITY_PRIVATE.json', f.execution_identity(lock))


def history(args):
    config = f.read(args.run_root / 'private/CAMPAIGN_CONFIG.json')
    first = f.packet_row('Private question native', 'yes', 'no', 'protocol')
    second = f.packet_row('unrelated historical question', 'ref', 'answer', 'protocol')
    judge(Path(config['stage1_run']) / 'private/judge', [first], {first['opaque_query_id']: False})
    judge(Path(config['stage2_run']) / 'private/judge', [second], {second['opaque_query_id']: True},
          reused={first['opaque_query_id']: False})
    return first, second


def preflight(cap):
    return dict(maximum_prompt_tokens=cap-24, required_model_length=cap, max_model_len=cap,
        requested_max_model_len=2048, no_truncation=True)


def test_partial_group_counts_unsupported_missing_and_scores_completed_files(tmp_path):
    args, _ = campaign(tmp_path, methods=('BE', 'S0'))
    single(args)
    entries, ledger, _ = f.inventory(args.run_root)
    assert len(entries) == 1 and len(ledger) == 3
    assert sum(r['complete'] for r in ledger) == 1
    assert sum(r['executable'] for r in ledger) == 2
    status = f.finalize(args)
    assert status['status'] == 'PARTIAL' and status['judge']['status'] == 'PENDING'
    assert status['compute']['planned_method_endpoints'] == 3
    assert status['compute']['unsupported_method_endpoints'] == 1
    assert status['compute']['pending_method_endpoints'] == 1
    report = (args.public_dir/'GPT_PRO_REVIEW.md').read_text()
    assert 'Private question' not in report and '/private/image.png' not in report
    assert 'NATIVE_DIAGNOSTIC' in report and 'T5 is NA' in report


def test_judge_reuse_chain_and_new_cap_all_current_only(tmp_path):
    args, protocol = campaign(tmp_path)
    single(args)
    old, unrelated = history(args)
    pool, lock = f.historical_judge(f.read(args.run_root/'private/CAMPAIGN_CONFIG.json'), protocol)
    assert pool[old['opaque_query_id']][1] is False
    with patch.object(f, 'length_preflight', return_value=preflight(4096)):
        side = f.prepare_judge(args)
        assert f.prepare_judge(args) == side  # unchanged resume is a no-op
    assert side['max_model_len'] == 4096 and side['execution_version_changed']
    assert side['new'] == 2 and side['reused'] == 0
    assert unrelated['opaque_query_id'] not in side['all_expected']
    current = args.run_root/'private/judge'
    packets = f.vf.read_jsonl(current/'JUDGE_PACKET_PRIVATE.jsonl')
    # Preserve Stage3 sidecar when simulating the external coordinator's Judge.
    judge(current, packets, {p['opaque_query_id']: p['raw_base_answer'] == 'yes' for p in packets}, execution(4096), {})
    write(current/'JUDGE_SIDECAR_PRIVATE.json', side)
    verdicts, _ = f.current_verdicts(args.run_root)
    assert len(verdicts) == 2
    actual = f.read(current/'JUDGE_EXECUTION_LOCK_PRIVATE.json')
    actual['runtime']['vllm'] = 'different'
    write(current/'JUDGE_EXECUTION_LOCK_PRIVATE.json', actual)
    with pytest.raises(ValueError, match='numerical execution drift'):
        f.current_verdicts(args.run_root)


def test_reused_stage1_chain_cannot_be_forged(tmp_path):
    args, protocol = campaign(tmp_path)
    first, _ = history(args)
    config = f.read(args.run_root/'private/CAMPAIGN_CONFIG.json')
    write(Path(config['stage2_run'])/'private/judge/REUSED_VERDICTS_PRIVATE.json', {first['opaque_query_id']: True})
    with pytest.raises(ValueError, match='unproven reused'):
        f.historical_judge(config, protocol)


def test_same_cap_reuses_full_tuple_but_changed_answer_does_not(tmp_path):
    args, _ = campaign(tmp_path)
    single(args, answer='no with a new full trailing sentence')
    history(args)
    with patch.object(f, 'length_preflight', return_value=preflight(2048)):
        side = f.prepare_judge(args)
    assert side['reused'] == 1 and side['new'] == 1 and not side['execution_version_changed']
    packets = f.vf.read_jsonl(args.run_root/'private/judge/JUDGE_PACKET_PRIVATE.jsonl')
    assert packets[0]['raw_base_answer'].endswith('new full trailing sentence')


def test_original_v4_eligibility_separate_from_all_source_correct(tmp_path):
    rows = [row(), row('g-wrong', 'T2G'), row('g-correct', 'T2G'),
            row('l-wrong', 'T1L'), row('l-correct', 'T1L'), row('special', 'NATIVE_DIAGNOSTIC')]
    args, _ = campaign(tmp_path, rows)
    value = single(args, rows=rows)
    for lid in ('g-correct', 'l-correct'):
        value['outputs'][lid]['base'] = output('yes')
    value['outputs']['g-correct']['fixed'] = output('no')
    value['outputs']['l-wrong']['fixed'] = output('no')
    write(args.run_root/'private/tasks/S3_A_e0001_BE/result_private.json', value)
    entries, ledger, _ = f.inventory(args.run_root)
    tuples = f.tuples_for(entries, 'protocol')
    verdicts = {k: v['raw_base_answer'] == v['gold_answer'] for k, v in tuples.items()}
    details = f.details_for(entries, verdicts, 'protocol')
    summary = f.aggregate(details, ledger)
    routed = {r['stratum']: r for r in summary if r['method'] == 'BE' and r['mode'] == 'ROUTED'
              and r['average'] == 'macro' and r['cohort'] == 'ALL_PLANNED'}
    assert routed['T2G']['semantic'] == .5 and routed['T2G']['v4_primary'] == 1
    assert routed['T1L']['semantic'] == .5 and routed['T1L']['v4_primary'] == 1
    assert routed['T2G']['v4_primary_denominator'] == 1
    assert routed['NATIVE_DIAGNOSTIC']['v4_primary'] is None
    assert routed['T5']['semantic'] is None and routed['T5']['inputs'] == 0
    assert routed['T0']['cap_hit'] is None and routed['T0']['ended_with_eos'] is None


def test_missing_rows_routes_and_finite_gates(tmp_path):
    rows = [row(), row('g', 'T1G')]
    args, _ = campaign(tmp_path, rows)
    single(args, 'S0', rows=[rows[0]])
    entries, ledger, _ = f.inventory(args.run_root)
    assert entries and not any(r['complete'] for r in ledger)
    value = single(args, 'S1', rows=[rows[0]], on=False)
    with pytest.raises(ValueError, match='route disagreement'):
        f.inventory(args.run_root)
    value['training_seconds'] = float('nan')
    write(args.run_root/'private/tasks/S3_A_e0001_S1/result_private.json', value)
    with pytest.raises(ValueError, match='nonfinite'):
        f.inventory(args.run_root)


def test_bank_role_targets_and_error_attribution():
    entries = []
    cases = [('EDIT_TARGET', 'yes', False, None, 'no'),
             ('EDIT_TARGET', 'yes', True, 'wrong', 'no'),
             ('EDIT_TARGET', 'yes', True, 'own', 'no'),
             ('EDIT_TARGET', 'yes', True, 'equivalent', 'yes'),
             ('NOW_EDITED_CONTEXT', 'updated', True, 'equivalent', 'updated'),
             ('TARGET_CONFLICT', None, True, 'wrong', 'yes'),
             ('UNKNOWN', None, False, None, 'no'),
             ('STRICT_BASE', 'yes', True, 'wrong', 'no')]
    for index, (role, ref, on, selected, answer) in enumerate(cases):
        r = row(str(index), 'H')
        r.update(role='evaluation', label='negative' if index >= 4 else 'positive', negative_group='H')
        item = dict(row=r, base=output('yes'), actual=output(answer), own_forced=output('yes'),
            method='S1', source_edit_index=101, source_expert='own', selected_expert=selected,
            prefix=16, on=on, strict_role=role, effective_reference=ref,
            legitimate_experts=['own', 'equivalent'], first_evaluable_prefix=1, history_key=str(index))
        entries.append(dict(track='B', prefix=16, edit=101, method='S1', item=item,
                            common_support=True, system_valid=True, target=None))
    tuples = f.tuples_for(entries, 'protocol')
    verdicts = {k: v['raw_base_answer'] == v['gold_answer'] for k, v in tuples.items()}
    details = [r for r in f.details_for(entries, verdicts, 'protocol') if r['mode'] == 'ROUTED']
    assert details[0]['rejection'] == 1
    assert details[1]['wrong_writer'] == 1
    assert details[2]['writer_error'] == 1
    assert details[3]['equivalent_correct'] == 1
    assert details[4]['semantic'] == 1 and details[4]['base_correct_damage'] is None
    assert details[4]['original_reference_correct'] == 0
    assert details[5]['semantic'] is None and details[5]['original_reference_correct'] == 1
    assert details[6]['semantic'] is None
    assert details[7]['base_correct_damage'] == 1


def test_preflight_lifts_whole_tuple_length_without_truncating(tmp_path):
    model = tmp_path/'model'
    model.mkdir()
    write(model/'config.json', dict(max_position_embeddings=4096))
    lock = execution()
    lock['model']['path'] = str(model)
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(range(len(text))))
    fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
    with patch.dict('sys.modules', transformers=fake_transformers), patch.object(f, 'render', side_effect=lambda _, r: r['raw_base_answer']):
        result = f.length_preflight({'a': dict(raw_base_answer='x'*2040)}, lock, 2048)
        assert result['max_model_len'] == 4096 and result['maximum_prompt_tokens'] == 2040
        with pytest.raises(ValueError, match='unsupported context'):
            f.length_preflight({'a': dict(raw_base_answer='x'*4096)}, lock, 2048)
