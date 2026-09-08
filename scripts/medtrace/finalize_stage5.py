#!/usr/bin/env python3
"""Stage5 exact Judge reuse, single/bank factorial and explicit missing coverage."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace import finalize_stage4 as f4
from scripts.medtrace import finalize_stage2 as f2
from scripts.medtrace.stage5_base_judge import stage4_pool
from scripts.medtrace.stage5_existing import reject, pairs
from scripts.medtrace.stage5_bank import METHODS, CONDITIONS

COHORT = 'STAGE5_SOURCE_CONSISTENCY_REVIEWED_NEW32'


def inventory(run):
    entries, coverage = [], []
    tasks = f4.read(run/'private/TASK_QUEUE.json')['tasks']
    by_method = {(t['event_index'], t['condition']):t for t in tasks if t['kind'] in ('BE','CP')}
    for original in f4.read(run/'private/NEW_EPISODE_MANIFEST_PRIVATE.json')['episodes']:
        i = original['event_index']; data = f4.read(run/f'private/edits/e{i:02d}.json')
        be_task = by_method.get((i, 'BALANCEDIT'))
        be_path = run/'private/tasks'/be_task['task_id']/'result_private.json' if be_task else None
        be = f4.read(be_path) if be_path and be_path.exists() else None
        scope_path = run/f'private/single_scope/e{i:02d}.json'
        scope = f4.read(scope_path) if scope_path.exists() else None
        for label, method in METHODS.items():
            t = by_method.get((i, CONDITIONS[method])); path = run/'private/tasks'/t['task_id']/'result_private.json' if t else None
            complete = bool(t and t['status'] == 'RAW_READY' and path.exists())
            coverage.append(dict(edit=i, method=label, status=t['status'] if t else data['method_support'][label],
                completed=complete, planned_native=1, planned_evaluation_positive=4,
                available_native=1 if complete else 0, source_images=1, patients='UNKNOWN'))
            if not complete:
                continue
            result = f4.read(path)
            assert result['status'] == 'RAW_READY' and result['base_guard']['unchanged']
            assert result['step'] == (50 if method == 'B' else 320) and result['task']['task_id'] == t['task_id']
            if method != 'B':
                assert result['a2_sha256'] == data['a2_sha256']
            assert be and scope and scope['lock']['frozen_before_evaluation']
            for row in data['rows']:
                item = result['outputs'][row['logical_id']]; baseline = be['outputs'][row['logical_id']]
                assert item['row'] == baseline['row'] == row
                assert item['fixed_on'] == baseline['route']['activated']
                assert item['base']['raw_token_ids'] == baseline['base']['raw_token_ids']
                assert item.get('system_replay_valid', item.get('route_branch_parity', True))
                entry = dict(track='A', prefix=0, edit=i, method=label, item=item, common_support=False,
                    system_valid=True, target=data['event']['edit_record']['gold_answer'], cohort_name=COHORT,
                    route_mode='R0')
                entries.append(entry)
                entries.append(dict(entry, route_mode='RC', item=reject(dict(item, route=baseline['route']), scope['lock']['kappa'], False)))
    for label in METHODS:
        path = run/'private/final_bank'/label/'result_private.json'
        if not path.exists():
            continue
        result = f4.read(path)
        if result['status'] == 'UNAVAILABLE':
            continue
        assert result['base_guard']['unchanged']
        for item in result['outputs']:
            entries.append(dict(track='B', prefix=result['bank_size'], edit=item['source_edit_index'], method=label,
                item=item, common_support=False, system_valid=True, cohort_name=COHORT, route_mode=item['route_mode']))
    return entries, coverage, []


def install(run):
    config = f4.read(run/'private/CAMPAIGN_CONFIG.json')
    protocol = f4.read(Path(config['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json')
    pool, execution = stage4_pool(config, protocol)
    directory = run/'private/base_judge'
    side = f4.read(directory/'JUDGE_SIDECAR_PRIVATE.json')
    identity = f4.read(directory/'REUSE_EXECUTION_IDENTITY_PRIVATE.json')
    if side['new']:
        verdicts, base_execution, _ = f2.validated_judge(directory)
        assert f2.execution_identity(base_execution) == identity == f2.execution_identity(execution)
        for row in f4.vf.read_jsonl(directory/'JUDGE_PACKET_PRIVATE.jsonl'):
            key = row['opaque_query_id']
            assert row == f4.f3.packet_row(row['question'], row['gold_answer'], row['raw_base_answer'], protocol['config_sha256'])
            pool[key] = row, verdicts[key], identity
    f4.f3.inventory = inventory
    f4.f3.historical_judge = lambda *_: (pool, execution)


def details(entries, verdicts, protocol):
    rows = []
    for mode in ('R0','RC'):
        subset = [e for e in entries if e['route_mode'] == mode]
        values = f4.details(subset, verdicts, protocol)
        for r in values:
            if r['track'] == 'A':
                if mode == 'RC' and r['mode'] != 'R0':
                    continue
                if r['mode'] == 'R0':
                    r['mode'] = mode
                    if r['strict_role'] == 'EDIT_TARGET':
                        r['rejection'] = float(r['on'] == 0 and r['own_forced_correct'] == 1)
            r['stratum'] = {'T0':'native_source_edit', 'source_style_confirmation':'derived_source_style',
                            'cross_family_confirmation':'derived_imperative_style'}.get(r['stratum'], r['stratum'])
            r['v4_primary'] = None  # These are source-derived tasks, not official M3Bench probes.
            rows.append(r)
    return rows


def finalize(args):
    run = args.run_root; public = run/'public'
    entries, coverage, _ = inventory(run)
    verdicts, side = f4.f3.current_verdicts(run)
    assert side
    rows = details(entries, verdicts, side['protocol_sha256'])
    tables = [r for r in f4.summarize(rows) if r['cohort'] == COHORT]
    f4.csv_write(public/'NEW_EDIT_SINGLE_RESULTS.csv', [r for r in tables if r['track'] == 'A'])
    f4.csv_write(public/'NEW_BANK_RESULTS.csv', [r for r in tables if r['track'] == 'B'])
    f4.csv_write(public/'NEW_WRITER_COVERAGE.csv', coverage)
    # Existing paired helper labels single RC as transfer. This is only an
    # internal comparison alias; published new single mode is episode RC.
    pair_rows = [dict(r, mode='DEV_THRESHOLD_TRANSFER_DIAGNOSTIC') if r['track'] == 'A' and r['mode'] == 'RC' else r for r in rows]
    effects = pairs(pair_rows)
    for r in effects:
        for key in ('mode','control_mode'):
            if r[key] == 'DEV_THRESHOLD_TRANSFER_DIAGNOSTIC':r[key] = 'RC'
    f4.csv_write(public/'NEW_PAIRED_EFFECTS.csv', effects)
    f4.vf.atomic_json(run/'private/NEW_DETAILS.json', rows)
    missing = len(set(side['all_expected'])-verdicts.keys())
    bank_status = {}; bank_sizes = {}; thresholds = []
    for method in METHODS:
        path = run/'private/final_bank'/method/'result_private.json'
        value = f4.read(path) if path.exists() else {}
        bank_status[method] = value.get('status', 'MISSING')
        bank_sizes[method] = value.get('bank_size', 0)
        if value.get('threshold'):thresholds.append(dict(method=method, **value['threshold']))
    completed = {m:sum(r['completed'] for r in coverage if r['method'] == m) for m in METHODS}
    counts = dict(Counter(r['status'] for r in coverage))
    config = f4.read(run/'private/CAMPAIGN_CONFIG.json')
    source = f4.read(public/'NEW_SOURCE_REVIEW_MANIFEST.json')
    old_replay = f4.read(public/'EXISTING_REPLAY_STATUS.json') if (public/'EXISTING_REPLAY_STATUS.json').exists() else {'status':'PENDING'}
    status = f4.read(public/'RUN_STATUS.json')
    unfinished = any(t['status'] in ('PENDING','RUNNING','WAITING_BASE_BEFORE')
                     for t in f4.read(run/'private/TASK_QUEUE.json')['tasks'])
    closed = not unfinished and not missing and all(v in ('RAW_READY','UNAVAILABLE') for v in bank_status.values()) and old_replay['status'] == 'COMPLETE'
    status.update(status='COMPUTE_COMPLETE' if closed else 'PARTIAL', compute='COMPLETE' if closed else 'PARTIAL',
        judge=dict(required=len(side['all_expected']), scored=len(verdicts), missing=missing, reused=side['reused'], new=side['new']),
        completed_writers=completed, coverage_counts=counts, final_bank_sizes=bank_sizes, final_bank_status=bank_status,
        original_image_generality='NA', official_text_generality='NA', publication='PUBLICATION_PENDING',
        independent_source_images=32, negative_source_images=23, patients='UNKNOWN')
    f4.vf.atomic_json(public/'RUN_STATUS.json', status)
    f4.vf.atomic_json(public/'METHOD_ROUTER_AND_EXPOSURE_LOCK.json', dict(source_commit=config['source_commit'],
        writer_execution=f4.read(run/'private/TRAINING_LAUNCH_PROVENANCE.json')['code_commit'],
        writer_methods=['W0','W1','BE'], kl_weight=.1, cp_steps=320, be_steps=50,
        source_review=source, new_bank_thresholds=thresholds,
        new_single_thresholds=[dict(edit=int(p.stem[1:]), **f4.read(p)['lock']) for p in sorted((run/'private/single_scope').glob('e*.json'))],
        original_stage4_bank_threshold=f4.read(Path(config['stage4_run'])/'private/bank/prefix16/THRESHOLD_LOCK.json'),
        no_new_algorithm=True, no_performance_gate=True, no_patient_independence_claim=True,
        positive_semantic_family_independence=False, same_method_bank_compositions=bank_sizes))
    lines = ['# MedTRACE Stage5 factual review', '', 'Status: '+status['status'], '',
        '## What the new confirmation can establish', '',
        'The original 32 source-image-distinct candidates were retained. Source-only derived texts have two surface styles but one native semantic lineage per fact; they are not official M3Bench rephrases or independently clinically reviewed facts.',
        'New calibration has no H support. RC therefore equals R0 under CALIBRATION_UNSUPPORTED, rather than providing an independent confirmation of calibrated rejection. No evaluation threshold was fitted.', '',
        '## Writer and routing contribution', '',
        'EXISTING_RC_WRITER_FACTORIAL.csv compares old W0/W1/BE at identical frozen routing; NEW_EDIT_SINGLE_RESULTS.csv gives per-episode RC and FORCED_ON. NEW_PAIRED_EFFECTS.csv uses paired common support only.',
        'W0/W1 comparisons on new edits are supported by at most one edit; do not claim KL unnecessary or generally effective from this sample. Final-bank sizes differ when writer support is missing; cross-size effects are not same-router writer ablations.',
        'BE_RC_ADAPTATION remains separate from native BalancEdit. An all-OFF result is Base return, not a writer improvement. Damage, full-source accuracy and Base-wrong improvement retain separate denominators.', '',
        '## Coverage and closure', '', f'Completed writers: {completed}. Coverage states: {counts}. Bank sizes: {bank_sizes}.',
        f'Judge: {status["judge"]}. Original image/text generality: NA where unavailable.',
        'No old writer training or historical Judge rerun was required by Track A. Old transfer is a development diagnostic, not new confirmation. Natural-branch replay evidence is separate from derived output provenance.',
        'Publication is a separate pending state until a later verified Git commit/public URL. Never rerun computation for a network failure.', '']
    def pct(value):return 'NA' if value in (None,'') else f'{100*float(value):.2f}%'
    lines += ['## Old fixed-routing factorial (edit macro)', '',
              '| Writer | Route | Panel | Full-source accuracy | Base-correct damage | Activation |',
              '|---|---|---|---:|---:|---:|']
    with (public/'EXISTING_RC_WRITER_FACTORIAL.csv').open() as f:
        old_tables = list(csv.DictReader(f))
    for r in old_tables:
        if r['track'] == 'B' and r['prefix'] == '16' and r['average'] == 'macro' and r['role'] == 'evaluation' and r['stratum'] in ('H','U'):
            lines.append(f"| {r['method']} | {r['mode']} | {r['stratum']} | {pct(r['semantic'])} | {pct(r['base_correct_damage'])} | {pct(r['fpr'])} |")
    lines += ['', '## New single-edit confirmation (edit macro)', '',
              '| Writer | Route | Panel | Inputs | Available edits | Full-source accuracy | Base-correct damage |',
              '|---|---|---|---:|---:|---:|---:|']
    for r in tables:
        if r['track'] == 'A' and r['average'] == 'macro' and r['mode'] in ('R0','RC') and r['role'] in ('native','evaluation'):
            lines.append(f"| {r['method']} | {r['mode']} | {r['stratum']} | {r['inputs']} | {r['edits']} | {pct(r['semantic'])} | {pct(r['base_correct_damage'])} |")
    lines += ['', 'Tables use available denominators, never count unavailable W1 endpoints as successes. '
              'The separate 96-row writer coverage manifest retains every planned edit/writer pair. '
              'For common-support inference consult paired_inputs/paired_edits; one edit has no bootstrap confidence interval.', '']
    f4.vf.atomic_text(public/'GPT_PRO_REVIEW.md', '\n'.join(lines))
    print(status['status'], status['judge'], completed, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare-judge','finalize'))
    p.add_argument('--run-root', type=Path, required=True)
    args = p.parse_args()
    install(args.run_root)
    if args.action == 'prepare-judge':
        side = f4.f3.prepare_judge(args)
        assert not side['execution_version_changed'], 'Judge numerical version must stay fixed'
    else:
        finalize(args)
