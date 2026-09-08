#!/usr/bin/env python3
"""Stage3 CPU packet preparation and partial-safe aggregate closeout; never run Judge."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.finalize_stage2 import execution_identity, validated_judge, read, vf
from scripts.medtrace.finalize_selective_write import csv_write, hierarchical, opaque, stratum
from scripts.medtrace.run_fixed_judge_vllm import OUTPUT_BUDGET, MODEL_LENGTH_CHOICES, tokenizer_files
from scripts.run_semantic_judge_v3 import render
from m3bench_repro.evaluation.metrics import ProbeOutcome, generality, locality as v4_locality, reliability

METHODS = ('BE', 'S0', 'S1')
PREFIXES = (1, 4, 8, 16)
FORMAL = ('T0', 'T1L', 'T1G', 'T2L', 'T2G', 'T3L', 'T3G', 'T4L', 'T4G', 'T5')
COSTS = ('parameters', 'training_seconds', 'elapsed_seconds', 'load_seconds', 'generation_seconds',
         'route_seconds', 'peak_vram_bytes', 'resident_vram_bytes', 'storage_bytes', 'disk_bytes',
         'forward_count', 'backward_count', 'teacher_forward_count', 'fit_samples',
         'peak_resident_vram_bytes', 'feature_seconds', 'request_seconds', 'replay_seconds',
         'generated', 'derived', 'optimizer_steps')
METRICS = ('semantic', 'v4_primary', 'original_reference_correct', 'base_correct', 'base_correct_damage',
           'base_correct_preserved', 'base_wrong_became_correct', 'base_wrong_changed',
           'on', 'joint_on_correct', 'fpr', 'misfire_damage', 'token_parity', 'kl',
           'own_forced_correct', 'rejection', 'wrong_writer', 'writer_error', 'equivalent_correct',
           'other_expert_correct', 'selected_writer_error', 'ended_with_eos', 'cap_hit')


def finite(value):
    """Reject invalid numerical artifacts before computing any public statistics."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite numerical artifact')
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('finite_losses', 'finite_gradients') and item is not True:
                raise ValueError('nonfinite training evidence')
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def edit_data(run, index):
    candidates = list(dict.fromkeys(run / f'private/edits/e{index:{width}d}.json'
                                  for width in ('', '02', '03', '04')))
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError(f'expected one frozen edit manifest for event {index}')
    return read(found[0])


def panel(row):
    if row['role'] in ('native', 'formal_development'):
        return row.get('task') or ('T0' if row['role'] == 'native' else row['fact_relation'])
    return row.get('confirmation_panel') or row.get('probe_kind') or row.get('negative_group') or stratum(row)


def complete_output(output):
    return (isinstance(output, dict) and isinstance(output.get('raw_answer'), str)
            and isinstance(output.get('raw_token_ids'), list)
            and all(type(t) is int for t in output['raw_token_ids']))


def supported(task, method):
    executable = task.get('executable', task.get('executable_methods', task.get('methods', METHODS)))
    if isinstance(executable, dict):
        value = executable.get(method, False)
        return value is True or isinstance(value, dict) and value.get('executable') is True
    return executable if isinstance(executable, bool) else method in executable


def validate_row(actual, frozen):
    for key in ('logical_id', 'question', 'reference', 'image_path', 'role', 'label', 'task',
                'fact_relation', 'eqkey', 'confirmation_panel', 'negative_group'):
        if key in frozen and actual.get(key) != frozen[key]:
            raise ValueError(f'frozen row binding mismatch: {key}')


def inventory(run):
    """Use frozen denominators, including missing/unsupported methods and partial groups."""
    config = read(run / 'private/CAMPAIGN_CONFIG.json')
    if config.get('kind') != 'MEDTRACE_STAGE3':
        raise ValueError('not a Stage3 campaign')
    tasks = read(run / 'private/TASK_QUEUE.json')['tasks']
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('duplicate queue task')
    entries, ledger, costs = [], [], []
    decisions, bases = {}, {}
    bank_episodes = None
    seen_groups = set()
    for task in tasks:
        kind = task['kind']
        if kind not in ('SINGLE_GROUP', 'BANK'):
            raise ValueError('foreign Stage3 task kind')
        index = task['event_index'] if kind == 'SINGLE_GROUP' else task['prefix']
        if (kind, index) in seen_groups:
            raise ValueError('duplicate group/prefix')
        seen_groups.add((kind, index))
        if kind == 'SINGLE_GROUP':
            data = edit_data(run, index)
            if data['track'] != 'V4_STAGE3':
                raise ValueError('historical event cannot become Stage3')
            expected = {(method, index, row['logical_id']): row for method in METHODS for row in data['rows']}
            paths = [(method, run / f'private/tasks/S3_A_e{index:04d}_{method}/result_private.json')
                     for method in METHODS]
        else:
            if index not in PREFIXES:
                raise ValueError('unplanned bank prefix')
            stage2 = Path(config['stage2_run'])
            if bank_episodes is None:
                bank_episodes = read(stage2 / 'private/NEW_EPISODE_MANIFEST_PRIVATE.json')['episodes']
                if len(bank_episodes) != 16 or len({e['event_index'] for e in bank_episodes}) != 16:
                    raise ValueError('bank requires frozen Stage2 sixteen-edit order')
            expected = {}
            for episode in bank_episodes[:index]:
                data = edit_data(stage2, episode['event_index'])
                for row in data['rows']:
                    if row['role'] == 'native' or (row['role'] == 'evaluation' and
                            (row['label'] == 'positive' or row.get('negative_group') in ('H', 'U'))):
                        for method in METHODS:
                            expected[method, episode['event_index'], row['logical_id']] = row
            paths = [(None, run / f'private/bank/prefix{index}/result_private.json')]
        ready, seen_rows = set(), set()
        result_complete = {}
        for method, path in paths:
            if not path.is_file():
                continue
            result = read(path)
            finite(result)
            if result.get('base_guard', {}).get('unchanged') is not True:
                raise ValueError('missing/failed Base guard')
            rt = result['task']
            if kind == 'SINGLE_GROUP':
                if (rt['task_id'] != path.parent.name or rt['event_index'] != index
                        or rt['kind'] != ('BE' if method == 'BE' else 'CP')
                        or result.get('step') != (50 if method == 'BE' else 320)):
                    raise ValueError('single endpoint identity/step mismatch')
                if method != 'BE' and (rt.get('parameterization') != 'P4' or
                        rt.get('condition') != ('W0_TASK_ONLY' if method == 'S0' else 'W1_KL_0.1')):
                    raise ValueError('frozen writer condition mismatch')
                if rt.get('seed', 20260910) != 20260910:
                    raise ValueError('Stage3 native seed mismatch')
                output_items = list(result['outputs'].values())
                result_complete[method] = result.get('status') == 'RAW_READY'
                costs.append(dict(track='A', prefix=0, edit=index, method=method,
                                  **{key: result.get(key) for key in COSTS}))
            else:
                if rt['task_id'] != task['task_id'] or rt['prefix'] != index:
                    raise ValueError('bank task binding mismatch')
                if result.get('roles_frozen_before_outputs') is not True:
                    raise ValueError('bank roles not frozen before outputs')
                if result.get('expected_items', len(expected)) != len(expected):
                    raise ValueError('bank expected coverage differs from frozen Stage2 manifest')
                output_items = result['outputs']
                result_complete.update({m: result.get('status') == 'RAW_READY' for m in METHODS})
                bank_costs = result.get('costs', [])
                if isinstance(bank_costs, dict):
                    bank_costs = [dict(value, method=key) for key, value in bank_costs.items()
                                  if isinstance(value, dict)]
                for cost in bank_costs:
                    costs.append(dict(track='B', prefix=index, edit=None, method='BE' if cost['method'] == 'B' else cost['method'],
                                      **{key: cost.get(key) for key in COSTS}))
            for item in output_items:
                actual_method = method if kind == 'SINGLE_GROUP' else ('BE' if item['method'] == 'B' else item['method'])
                edit = index if kind == 'SINGLE_GROUP' else item['source_edit_index']
                row = item['row']
                key = actual_method, edit, row['logical_id']
                if key not in expected or key in seen_rows:
                    raise ValueError('unexpected or duplicate endpoint row')
                seen_rows.add(key)
                validate_row(row, expected[key])
                if kind == 'SINGLE_GROUP' and not supported(task, actual_method):
                    raise ValueError('unsupported frozen method produced an endpoint')
                branches = ('base', 'forced', 'fixed') if kind == 'SINGLE_GROUP' else ('base', 'actual', 'own_forced')
                if not all(complete_output(item.get(branch)) for branch in branches):
                    continue
                on = item['fixed_on'] if kind == 'SINGLE_GROUP' else item['on']
                if type(on) is not bool:
                    raise ValueError('invalid routing boolean')
                if kind == 'BANK' and (item['prefix'] != index or bool(item['selected_expert']) != on):
                    raise ValueError('bank selected writer/ON mismatch')
                route_key = kind, index, edit, row['logical_id']
                if actual_method in ('S0', 'S1'):
                    decision = on if kind == 'SINGLE_GROUP' else (on, item['selected_expert'])
                    if route_key in decisions and decisions[route_key] != decision:
                        raise ValueError('S0/S1 route disagreement')
                    decisions[route_key] = decision
                if route_key in bases and bases[route_key] != item['base']:
                    # Only answer/tokens define output parity; timers can differ.
                    if any(bases[route_key][k] != item['base'][k] for k in ('raw_answer', 'raw_token_ids')):
                        raise ValueError('shared Base output mismatch')
                bases[route_key] = item['base']
                system_valid = item.get('system_replay_valid', item.get('route_branch_parity', True)) is True
                ready.add(key)
                entries.append(dict(track='A' if kind == 'SINGLE_GROUP' else 'B', prefix=0 if kind == 'SINGLE_GROUP' else index,
                    edit=edit, method=actual_method, item=item, system_valid=system_valid,
                    common_support=kind == 'BANK' or all(supported(task, m) for m in METHODS),
                    target=data['event']['edit_record']['gold_answer'] if kind == 'SINGLE_GROUP' else None,
                    task_id=task['task_id']))
        for method in METHODS:
            planned = sum(key[0] == method for key in expected)
            available = sum(key[0] == method for key in ready)
            executable = kind == 'BANK' or supported(task, method)
            ledger.append(dict(track='A' if kind == 'SINGLE_GROUP' else 'B', prefix=0 if kind == 'SINGLE_GROUP' else index,
                edit=index if kind == 'SINGLE_GROUP' else None, method=method, task_id=task['task_id'],
                queue_status=task['status'], executable=executable, planned_inputs=planned,
                common_support=kind == 'BANK' or all(supported(task, m) for m in METHODS),
                unsupported_reasons=task.get('unsupported_reasons', {}).get(method, []),
                panel_support=dict(Counter(row['role']+'|'+panel(row) for key, row in expected.items() if key[0] == method)),
                complete_inputs=available, complete=bool(executable and planned and available == planned and
                    result_complete.get(method) and all(e['system_valid'] for e in entries
                        if e['task_id'] == task['task_id'] and e['method'] == method))))
    return entries, ledger, costs


def packet_row(question, reference, answer, protocol):
    key = opaque(question, reference, answer, protocol)
    return dict(opaque_query_id=key, question=question, gold_answer=reference,
                raw_base_answer=answer, adjudication_pass=1)


def tuples_for(entries, protocol):
    tuples = {}
    for entry in entries:
        item, row = entry['item'], entry['item']['row']
        references = {row['reference']}
        if entry['track'] == 'A':
            references.add(entry['target'])
            branches = ('base', 'forced', 'fixed')
        else:
            if item.get('effective_reference') is not None:
                references.add(item['effective_reference'])
            branches = ('base', 'actual', 'own_forced')
        for reference in sorted(references):
            for branch in branches:
                packet = packet_row(row['question'], reference, item[branch]['raw_answer'], protocol)
                tuples[packet['opaque_query_id']] = packet
    return tuples


def historical_judge(config, protocol):
    """Prove reused verdict ancestry against the actual Stage1/Stage2 packets."""
    pool, latest = {}, None
    directories = [Path(config['stage1_run']) / 'private/judge',
                   Path(config['stage2_run']) / 'private/base_judge',
                   Path(config['stage2_run']) / 'private/judge']
    for directory in directories:
        if not directory.exists():
            if directory.name == 'base_judge':
                continue
            raise FileNotFoundError(directory)
        side = read(directory / 'JUDGE_SIDECAR_PRIVATE.json')
        if side['protocol_sha256'] != protocol['config_sha256'] or side['snapshot'] != protocol['judge_snapshot_sha']:
            raise ValueError('historical semantic protocol mismatch')
        packets = vf.read_jsonl(directory / 'JUDGE_PACKET_PRIVATE.jsonl')
        mapping = {r['opaque_query_id']: r for r in packets}
        if len(mapping) != len(packets) or set(mapping) != set(side['expected']):
            raise ValueError('historical packet coverage mismatch')
        if vf.sha256_file(directory / 'JUDGE_PACKET_PRIVATE.jsonl') != side['packet_sha256']:
            raise ValueError('historical packet bytes binding mismatch')
        for key, row in mapping.items():
            if row != packet_row(row['question'], row['gold_answer'], row['raw_base_answer'], protocol['config_sha256']):
                raise ValueError('historical full tuple binding mismatch')
        reused_path = directory / 'REUSED_VERDICTS_PRIVATE.json'
        reused = read(reused_path) if reused_path.exists() else {}
        reuse_identity = read(directory / 'REUSE_EXECUTION_IDENTITY_PRIVATE.json') if reused_path.exists() else None
        if set(reused) & set(mapping):
            raise ValueError('overlapping new/reused Judge tuples')
        for key, verdict in reused.items():
            if type(verdict) is not bool or key not in pool or pool[key][1:] != (verdict, reuse_identity):
                raise ValueError('unproven reused Stage1/Stage2 verdict chain')
        if set(mapping) | set(reused) != set(side.get('all_expected', side['expected'])):
            raise ValueError('historical all_expected closure mismatch')
        if packets:
            verdicts, execution, _ = validated_judge(directory)
            identity = execution_identity(execution)
            if reused and identity != reuse_identity:
                raise ValueError('historical mixed numerical execution')
            for key, verdict in verdicts.items():
                if key in pool and pool[key][2] == identity and pool[key][1] != verdict:
                    raise ValueError('same-execution conflicting verdict')
                pool[key] = mapping[key], verdict, identity
            latest = execution
    if latest is None:
        raise ValueError('no historical Judge execution identity')
    return pool, latest


def length_preflight(tuples, execution, requested):
    """Same render/tokenization as the existing Judge, without importing vLLM."""
    from transformers import AutoTokenizer
    model = Path(execution['model']['path'])
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if tokenizer_files(model) != execution['tokenizer']['file_sha256']:
        raise ValueError('Judge tokenizer file identity changed')
    maximum = max((len(tokenizer.encode(render(tokenizer, row), add_special_tokens=True))
                   for row in tuples.values()), default=0)
    required = maximum + OUTPUT_BUDGET
    selected = next((cap for cap in MODEL_LENGTH_CHOICES if cap >= max(requested, required)), None)
    supported_cap = read(model / 'config.json').get('max_position_embeddings', 0)
    if selected is None or selected > supported_cap:
        raise ValueError(f'complete Judge answers require unsupported context length {required}')
    return dict(maximum_prompt_tokens=maximum, required_model_length=required,
                max_model_len=selected, requested_max_model_len=requested, no_truncation=True)


def prepare_judge(args):
    run = args.run_root
    config = read(run / 'private/CAMPAIGN_CONFIG.json')
    protocol = read(Path(config['runtime']['cpu_gate']).parent / 'private/JUDGE_LOCK_V4.json')
    entries, _, _ = inventory(run)
    tuples = tuples_for(entries, protocol['config_sha256'])
    pool, execution = historical_judge(config, protocol)
    preflight = length_preflight(tuples, execution, getattr(args, 'max_model_len', 2048))
    identity = execution_identity(execution)
    requested_identity = deepcopy(identity)
    requested_identity['generation']['max_model_len'] = preflight['max_model_len']
    changed = requested_identity != identity
    reused = {key: pool[key][1] for key in tuples if key in pool and pool[key][2] == requested_identity}
    # A new cap is one new numerical execution for ALL current tuples, including Base.
    if changed:
        reused = {}
    new = {key: tuples[key] for key in sorted(tuples) if key not in reused}
    directory = run / 'private/judge'
    directory.mkdir(parents=True, exist_ok=True)
    sidecar_path = directory / 'JUDGE_SIDECAR_PRIVATE.json'
    if sidecar_path.exists():
        old = read(sidecar_path)
        if (old['all_expected'] == sorted(tuples) and old['max_model_len'] == preflight['max_model_len']
                and read(directory / 'REUSE_EXECUTION_IDENTITY_PRIVATE.json') == requested_identity
                and vf.sha256_file(directory / 'JUDGE_PACKET_PRIVATE.jsonl') == old['packet_sha256']):
            return old
        raise FileExistsError('prepared Judge packet changed; preserve this execution and start a new run snapshot')
    packet = directory / 'JUDGE_PACKET_PRIVATE.jsonl'
    vf.atomic_text(packet, ''.join(json.dumps(row, sort_keys=True) + '\n' for row in new.values()))
    vf.atomic_json(directory / 'REUSED_VERDICTS_PRIVATE.json', reused)
    vf.atomic_json(directory / 'REUSE_EXECUTION_IDENTITY_PRIVATE.json', requested_identity)
    sidecar = dict(protocol_sha256=protocol['config_sha256'], snapshot=protocol['judge_snapshot_sha'],
        expected=sorted(new), all_expected=sorted(tuples), reused=len(reused), new=len(new),
        packet_sha256=vf.sha256_file(packet), complete_answers=True, **preflight,
        previous_max_model_len=identity['generation']['max_model_len'], execution_version_changed=changed,
        version_policy='cap-only execution version; all current tuples rejudged; unrelated history untouched' if changed
                       else 'exact complete tuple and identical numerical execution reuse')
    vf.atomic_json(sidecar_path, sidecar)
    return sidecar


def current_verdicts(run):
    directory = run / 'private/judge'
    side_path = directory / 'JUDGE_SIDECAR_PRIVATE.json'
    if not side_path.exists():
        return {}, None
    side = read(side_path)
    if vf.sha256_file(directory / 'JUDGE_PACKET_PRIVATE.jsonl') != side['packet_sha256']:
        raise ValueError('current Judge packet bytes mismatch')
    identity = read(directory / 'REUSE_EXECUTION_IDENTITY_PRIVATE.json')
    verdicts = read(directory / 'REUSED_VERDICTS_PRIVATE.json')
    config = read(run / 'private/CAMPAIGN_CONFIG.json')
    protocol = read(Path(config['runtime']['cpu_gate']).parent / 'private/JUDGE_LOCK_V4.json')
    pool, prior_execution = historical_judge(config, protocol)
    prior_identity = execution_identity(prior_execution)
    expected_identity = deepcopy(prior_identity)
    expected_identity['generation']['max_model_len'] = side['max_model_len']
    if identity != expected_identity or (side['execution_version_changed'] and verdicts):
        raise ValueError('unapproved numerical Judge execution drift or mixed caps')
    for key, value in verdicts.items():
        if type(value) is not bool or key not in pool or pool[key][1:] != (value, identity):
            raise ValueError('current reused verdict ancestry mismatch')
    if side['new'] and all((directory / name).is_file() for name in
                          ('JUDGE_OUTPUT_PRIVATE.jsonl', 'JUDGE_EXECUTION_LOCK_PRIVATE.json')):
        new, execution, _ = validated_judge(directory)
        if execution_identity(execution) != identity:
            raise ValueError('Judge numerical execution drift beyond declared context-cap version')
        verdicts.update(new)
    if set(verdicts) - set(side['all_expected']):
        raise ValueError('foreign current verdict')
    return verdicts, side


def details_for(entries, verdicts, protocol):
    details, base_seen = [], set()
    def score(row, ref, output):
        if ref is None:
            return None
        value = verdicts.get(opaque(row['question'], ref, output['raw_answer'], protocol))
        return None if value is None else float(value)
    for entry in entries:
        item, row = entry['item'], entry['item']['row']
        bank = entry['track'] == 'B'
        role = item['strict_role'] if bank else ('STRICT_BASE' if row['label'] == 'negative' else 'EDIT_TARGET')
        if role not in ('EDIT_TARGET', 'STRICT_BASE', 'NOW_EDITED_CONTEXT', 'TARGET_CONFLICT', 'UNKNOWN'):
            raise ValueError('unknown strict bank role')
        reference = item['effective_reference'] if bank else row['reference']
        if bank and ((role in ('TARGET_CONFLICT', 'UNKNOWN')) != (reference is None)):
            raise ValueError('bank effective-reference role mismatch')
        strict = reference is not None
        base_correct = score(row, reference, item['base'])
        own = score(row, row['reference'], item['own_forced' if bank else 'forced'])
        modes = (('BASE', 'base'), ('FORCED_ON', 'own_forced' if bank else 'forced'),
                 ('ROUTED', 'actual' if bank else 'fixed'))
        for mode, branch in modes:
            if mode == 'ROUTED' and not entry['system_valid']:
                continue
            if mode == 'BASE':
                base_key = entry['track'], entry['prefix'], entry['edit'], row['logical_id'], reference
                if base_key in base_seen:
                    continue
                base_seen.add(base_key)
            output = item[branch]
            correct = score(row, reference, output)
            original = score(row, row['reference'], output)
            on = mode == 'FORCED_ON' or mode == 'ROUTED' and item['on' if bank else 'fixed_on']
            parity = float(output['raw_token_ids'] == item['base']['raw_token_ids'])
            locality = role == 'STRICT_BASE'
            routed_target = bank and mode == 'ROUTED' and role in ('EDIT_TARGET', 'NOW_EDITED_CONTEXT')
            selected = item.get('selected_expert')
            own_selected = selected == item.get('source_expert')
            equivalent = selected in item.get('legitimate_experts', []) and not own_selected
            task_panel = panel(row)
            primary = None
            if not bank and correct is not None and row['role'] in ('native', 'formal_development'):
                if task_panel == 'T0':
                    primary = reliability(bool(correct))
                elif base_correct is not None and task_panel in FORMAL:
                    outcome = [ProbeOutcome(bool(base_correct), bool(correct))]
                    primary = v4_locality(outcome) if task_panel.endswith('L') else generality(outcome) if task_panel.endswith('G') else None
            metric = dict(semantic=correct, v4_primary=primary, original_reference_correct=original, base_correct=base_correct,
                base_correct_damage=1-correct if locality and base_correct == 1 and correct is not None else None,
                base_correct_preserved=correct if locality and base_correct == 1 else None,
                base_wrong_became_correct=correct if base_correct == 0 else None,
                base_wrong_changed=1-parity if base_correct == 0 else None,
                on=float(on), joint_on_correct=float(on)*correct if correct is not None else None,
                fpr=float(on) if locality else None,
                misfire_damage=float(on)*(1-correct) if locality and base_correct == 1 and correct is not None else None,
                token_parity=parity, kl=(item.get('kl') if on else 0.) if item.get('kl') is not None else None,
                own_forced_correct=own,
                rejection=float(not on and own == 1) if routed_target and own is not None else None,
                wrong_writer=float(on and not own_selected and not equivalent and own == 1 and correct == 0) if routed_target and own is not None and correct is not None else None,
                writer_error=float(on and (own_selected or equivalent or own == 0) and correct == 0) if routed_target and correct is not None else None,
                equivalent_correct=float(on and equivalent and correct == 1) if routed_target and correct is not None else None,
                other_expert_correct=float(on and not own_selected and correct == 1) if routed_target and correct is not None else None,
                selected_writer_error=float(on and correct == 0) if routed_target and correct is not None else None,
                ended_with_eos=float(output['ended_with_eos']) if type(output.get('ended_with_eos')) is bool else None,
                cap_hit=float(output['cap_hit']) if type(output.get('cap_hit')) is bool else None)
            details.append(dict(track=entry['track'], prefix=entry['prefix'], edit=entry['edit'],
                method='BASE' if mode == 'BASE' else entry['method'], mode=mode, role=row['role'],
                stratum=panel(row), strict_role=role, strict_valid=strict,
                cohort='ALL_PLANNED', common_support=entry['common_support'],
                eqkey=row.get('eqkey', row['logical_id']), source_group=row.get('source_group', row['image_path']),
                source_image=row['image_path'], first_evaluable_prefix=item.get('first_evaluable_prefix'),
                history_key=item.get('history_key'), **metric))
    return details + [dict(row, cohort='COMMON_SUPPORTED') for row in details if row['common_support'] and row['track'] == 'A']


def aggregate(details, ledger):
    keys = ('track', 'prefix', 'method', 'mode', 'role', 'stratum', 'strict_role', 'cohort')
    cells = defaultdict(list)
    for row in details:
        cells[tuple(row[k] for k in keys)].append(row)
    # Explicit unavailable official tasks, including T5; never manufacture an Overall.
    for method in (*METHODS, 'BASE'):
        for mode in (('BASE',) if method == 'BASE' else ('ROUTED', 'FORCED_ON')):
            for task in FORMAL:
                if not any(k[0] == 'A' and k[2:4] == (method, mode) and k[5] == task for k in cells):
                    cells['A', 0, method, mode, 'native' if task == 'T0' else 'formal_development', task, 'UNKNOWN', 'ALL_PLANNED'] = []
    output = []
    for key, rows in sorted(cells.items()):
        metadata = dict(zip(keys, key))
        relevant = [r for r in ledger if r['track'] == key[0] and r['prefix'] == key[1]
                    and (key[2] == 'BASE' or r['method'] == key[2])
                    and (key[7] != 'COMMON_SUPPORTED' or r['common_support'])]
        if key[2] == 'BASE':
            relevant = list({r['task_id']: r for r in sorted(relevant, key=lambda r: r['executable'])}.values())
        panel_key = key[4]+'|'+key[5]
        for average in ('micro', 'macro'):
            summary = dict(metadata, average=average, inputs=len(rows), edits=len({r['edit'] for r in rows}),
                source_images=len({r['source_image'] for r in rows}), source_groups=len({r['source_group'] for r in rows}),
                eqkeys=len({r['eqkey'] for r in rows}), strict_valid_inputs=sum(r['strict_valid'] for r in rows),
                planned_method_endpoints=len(relevant), executable_method_endpoints=sum(r['executable'] for r in relevant),
                complete_method_endpoints=sum(r['complete'] for r in relevant))
            summary.update(planned_panel_inputs=sum(r['panel_support'].get(panel_key, 0) for r in relevant),
                executable_panel_inputs=sum(r['panel_support'].get(panel_key, 0) for r in relevant if r['executable']),
                official_primary_definition=('postcorrect' if key[5] == 'T0' else 'pre_correct_then_postcorrect' if key[5].endswith('L')
                    else 'pre_wrong_then_postcorrect' if key[5].endswith('G') else 'NA') if key[0] == 'A' and key[5] in FORMAL and key[5] != 'T5' else 'NA')
            for metric in METRICS:
                values = [r[metric] for r in rows if r[metric] is not None]
                by_edit = []
                for edit in sorted({r['edit'] for r in rows}):
                    group = [r for r in rows if r['edit'] == edit]
                    eligible = [r[metric] for r in group if r[metric] is not None]
                    value = (mean(eligible) if eligible else None) if metric == 'v4_primary' else hierarchical(group, metric)
                    if value is not None:
                        by_edit.append(value)
                summary[metric] = mean(values if average == 'micro' else by_edit) if values else None
                summary[metric + '_numerator'] = sum(values) if values else None
                summary[metric + '_denominator'] = len(values)
                summary[metric + '_edit_support'] = len(by_edit)
            output.append(summary)
    return output


def paired_effects(details):
    output = []
    cells = defaultdict(list)
    for row in details:
        if row['track'] == 'A' and row['method'] in METHODS and row['cohort'] == 'ALL_PLANNED':
            cells[row['mode'], row['role'], row['stratum'], row['strict_role']].append(row)
    for key, rows in sorted(cells.items()):
        for control in ('S0', 'BE'):
            for metric in ('semantic', 'v4_primary', 'base_correct_damage'):
                maps = []
                for method in ('S1', control):
                    values = {}
                    for edit in {r['edit'] for r in rows}:
                        group = [r for r in rows if r['method'] == method and r['edit'] == edit]
                        eligible = [r[metric] for r in group if r[metric] is not None]
                        values[edit] = ((mean(eligible) if eligible else None) if metric == 'v4_primary' else hierarchical(group, metric))
                    maps.append({edit: value for edit, value in values.items() if value is not None})
                a, b = maps
                common = set(a) & set(b)
                lo, hi = vf._paired_ci(a, b) if len(common) >= 2 else (None, None)
                output.append(dict(mode=key[0], role=key[1], stratum=key[2], strict_role=key[3],
                    candidate='S1', control=control, metric=metric, paired_edits=len(common),
                    delta=mean(a[i]-b[i] for i in common) if common else None, ci_low=lo, ci_high=hi))
    return output


def history_changes(details):
    grouped, cells = defaultdict(dict), defaultdict(list)
    for row in details:
        if row['track'] == 'B' and row['mode'] == 'ROUTED':
            grouped[row['method'], row['edit'], row['history_key']][row['prefix']] = row
    for (method, edit, _), points in grouped.items():
        last = points.get(16)
        first = points.get(next(iter(points.values()))['first_evaluable_prefix'])
        if first is None or last is None:
            continue
        for metric in ('semantic', 'original_reference_correct', 'joint_on_correct'):
            compatible = (metric == 'original_reference_correct' or first['strict_role'] == last['strict_role'])
            if compatible and first[metric] is not None and last[metric] is not None:
                cells[method, edit, first['prefix'], first['stratum'], metric].append(last[metric]-first[metric])
    return [dict(method=k[0], edit=k[1], first_prefix=k[2], last_prefix=16, stratum=k[3], metric=k[4],
                 paired_inputs=len(v), change=mean(v)) for k, v in sorted(cells.items())]


def finalize(args):
    run, public = args.run_root, args.public_dir
    public.mkdir(parents=True, exist_ok=True)
    entries, ledger, costs = inventory(run)
    verdicts, side = current_verdicts(run)
    config = read(run / 'private/CAMPAIGN_CONFIG.json')
    protocol = read(Path(config['runtime']['cpu_gate']).parent / 'private/JUDGE_LOCK_V4.json')['config_sha256']
    tuples = tuples_for(entries, protocol)
    details = details_for(entries, verdicts, protocol)
    summaries = aggregate(details, ledger)
    pairs = paired_effects(details)
    csv_write(public / 'SINGLE_EVAL_RESULTS.csv', [r for r in summaries if r['track'] == 'A'])
    csv_write(public / 'EXPERT_BANK16_RESULTS.csv', [r for r in summaries if r['track'] == 'B'])
    csv_write(public / 'SINGLE_PAIRED_EFFECTS.csv', pairs)
    csv_write(public / 'EXPERT_BANK16_HISTORY.csv', history_changes(details))
    csv_write(public / 'METHOD_COSTS.csv', costs)
    missing = set(tuples) - set(verdicts)
    complete = sum(r['complete'] for r in ledger)
    planned, executable = len(ledger), sum(r['executable'] for r in ledger)
    expected_prefixes = {r['prefix'] for r in ledger if r['track'] == 'B'}
    tracks_present = any(r['track'] == 'A' for r in ledger) and expected_prefixes == set(PREFIXES)
    compute_complete = bool(tracks_present and executable and complete == executable)
    judge_complete = bool(tuples and not missing and side and set(tuples) == set(side['all_expected']))
    status = dict(status='READY_FOR_PUBLICATION' if compute_complete and judge_complete and planned == executable else 'PARTIAL',
        compute=dict(status='COMPLETE_EXECUTABLE' if compute_complete else 'PARTIAL', planned_method_endpoints=planned,
            executable_method_endpoints=executable, unsupported_method_endpoints=planned-executable,
            complete_method_endpoints=complete, failed_method_endpoints=sum(r['queue_status'] in ('FAILED', 'ERROR') for r in ledger),
            pending_method_endpoints=sum(r['executable'] and not r['complete'] for r in ledger),
            planned_inputs=sum(r['planned_inputs'] for r in ledger), complete_inputs=sum(r['complete_inputs'] for r in ledger),
            full_planned_coverage=compute_complete and planned == executable, missing_bank_prefixes=sorted(set(PREFIXES)-expected_prefixes)),
        judge=dict(status='COMPLETE_CURRENT_TUPLES' if judge_complete else 'PARTIAL' if verdicts else 'PENDING',
            required_tuples=len(tuples), scored_tuples=len(set(tuples) & set(verdicts)), missing_tuples=len(missing),
            reused=side['reused'] if side else 0, max_model_len=side['max_model_len'] if side else None,
            execution_version_changed=side['execution_version_changed'] if side else None),
        publication=dict(status='PENDING_PUBLIC_PUSH_AND_ANONYMOUS_VERIFICATION', aggregate_files_written=True),
        queue=ledger, scientific_status='NO_PERFORMANCE_GATE', t5='NA_NO_LEGAL_MATERIAL', overall='NOT_COMPUTED')
    finite(status)
    vf.atomic_json(public / 'EXECUTION_STATUS.json', status)
    def selected(track, method, mode='ROUTED'):
        return [r for r in summaries if r['track'] == track and r['method'] == method and r['mode'] == mode
                and r['average'] == 'macro' and r['cohort'] == 'ALL_PLANNED' and r['role'] in ('native', 'formal_development', 'evaluation')]
    def table(rows):
        fmt = lambda v: 'NA' if v is None else f'{v:.4f}'
        header = '| Method | Prefix | Role / panel | Strict role | Inputs | V4 primary macro (eligible) | All-source macro | All-source micro | Damage macro |\n|---|---:|---|---|---:|---:|---:|---:|---:|'
        lines = []
        for row in rows:
            micro = row['semantic_numerator']/row['semantic_denominator'] if row['semantic_denominator'] else None
            lines.append(f"| {row['method']} | {row['prefix']} | {row['role']} / {row['stratum']} | {row['strict_role']} | {row['inputs']} | {fmt(row['v4_primary'])} ({row['v4_primary_denominator']}) | {fmt(row['semantic'])} | {fmt(micro)} | {fmt(row['base_correct_damage'])} |")
        return header + '\n' + '\n'.join(lines) if lines else 'NA: no complete endpoints with this role.'
    comparison = '\n'.join(f"- {r['stratum']} ({r['role']}), S1−{r['control']} {r['metric']}: "
        f"{r['delta'] if r['delta'] is not None else 'NA'}; paired edits={r['paired_edits']}."
        for r in pairs if r['mode'] == 'ROUTED' and r['role'] in ('native', 'formal_development', 'evaluation')) or 'NA: no paired judged edit support.'
    bank = selected('B', 'S1')
    report = f"""# MedTRACE Stage3 factual review

Status: {status['status']}; compute={status['compute']['status']}; Judge={status['judge']['status']}; publication=PENDING.
S1 = BE_ROUTE + P4-W1_KL_0.1; S0 = same route + P4-W0_TASK_ONLY; BE = native BalancEdit V4 adaptation.

1. How does S1 perform on original V4 generalization and locality?

{table(selected('A', 'S1'))}

Official task labels remain separate from fit and source-style/cross-family confirmation. T5 is NA; no official Overall is computed.
Primary metric mapping reuses m3bench_repro/evaluation/metrics.py: T0 reliability=postcorrect; T*G=postcorrect among Base-wrong; T*L=postcorrect among Base-correct. Primary macro averages eligible probes per edit then edits; semantic is the separate all-source correctness metric. Token parity is diagnostic. Task-specific native rows remain NATIVE_DIAGNOSTIC, not T0 anchors.

2. What protection does S1 retain against W0, and at what cost?

{comparison}

Paired edit intervals are in SINGLE_PAIRED_EFFECTS.csv. They resample edits; shared source images remain correlated. Missing judged pairs are NA, not zero effects.

3. How does the native BalancEdit system compare in behavior and cost?

{table(selected('A', 'BE'))}

Only ROUTED versus ROUTED and separately FORCED_ON versus FORCED_ON are compared. METHOD_COSTS.csv reports available training, teacher, generation, loading, routing, memory and storage measurements; missing measurements are NA. Layers, capacity and supervision differ; no matched-capacity claim.

4. Do benefits survive sixteen coexisting experts; where do failures come from?

{table(bank)}

EXPERT_BANK16_RESULTS.csv reports rejection with a correct own writer, wrong-writer failures, selected writer errors and equivalent-expert correct outputs, each with explicit denominators. EXPERT_BANK16_HISTORY.csv pairs each input's first evaluable prefix with prefix16; absent pairs are not counted. Conflicting/unknown roles retain original-reference descriptive scores but have no strict semantic/locality score. NOW_EDITED_CONTEXT is scored against its frozen effective target and excluded from strict Base damage. This is viewed Stage2 insertion replay, not causal online training or M3Bench 200-edit sequential.

5. What is covered, what is missing, and should the route be expanded?

Planned method endpoints={planned}; executable={executable}; unsupported={planned-executable}; complete={complete}; pending={status['compute']['pending_method_endpoints']}. Current Judge tuples={len(tuples)}; missing={len(missing)}. EXECUTION_STATUS.json retains every queued method and missing prefix. Publication is unverified.
No new experiment or performance permission gate is introduced. {'Finish the existing pending computation/Judge/publication only; no scale-up conclusion is supported by this partial snapshot.' if status['status'] == 'PARTIAL' else 'The frozen evaluation is ready for publication review; no automatic scale-up, new algorithm, or clinical claim follows.'}

Historical continuity: Stage2 GPT_PRO_REVIEW.md records a real original-T2G loss for P4-W1 and sparse old H support; a perfect constructed-text panel does not erase that finding. Historical scores are not relabeled as current execution. See the Stage2 publication at 74d2a337f7d2b830d58819f76c87058cef0c5f3b.
Micro averages count observed inputs; macro averages aggregate equivalent inputs within source groups, then edits. Image/group supports are not patient counts (patient identity UNKNOWN). Empty denominators are NA. Judge uses full answers; EOS/cap-hit flags are reported only when supplied by the generator.
"""
    vf.atomic_text(public / 'GPT_PRO_REVIEW.md', report)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-judge', 'finalize'))
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--public-dir', type=Path, required=True)
    parser.add_argument('--max-model-len', type=int, choices=MODEL_LENGTH_CHOICES, default=2048)
    args = parser.parse_args()
    result = {'prepare-judge': prepare_judge, 'finalize': finalize}[args.action](args)
    print(json.dumps(result if args.action == 'prepare-judge' else {k: result[k] for k in ('status', 'compute', 'judge', 'publication')}, sort_keys=True))


if __name__ == '__main__':
    main()
