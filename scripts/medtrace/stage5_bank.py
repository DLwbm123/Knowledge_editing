#!/usr/bin/env python3
"""Evaluate each available final writer bank; freeze RC before bank outputs."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace import run_stage3_bank as bank
from scripts.medtrace.stage4_scope import calibrate, accepted
from scripts.medtrace.run_stage2 import vf, sw, read
from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
from m3bench_repro.editors.routing import MemoryRouter
import torch

METHODS = {'BE': 'B', 'W0': 'S0', 'W1': 'S1'}
CONDITIONS = dict(bank.METHODS)


class Generator:
    """Keep one selected BE transform resident, as in the existing bank runner."""
    def __init__(self, runtime, artifacts):
        self.runtime, self.artifacts = runtime, artifacts
        self.editor = BalanceEditPaperSpecEditor(runtime)
        self.target = self.editor.target
        self.original = self.editor.wrapper.base
        self.loaded = None
        self.generated = 0
        self.load_seconds = self.generation_seconds = 0.

    def generate(self, method, selected, row):
        editor, runtime = self.editor, self.runtime
        editor.wrapper.set_active(None)
        if method != 'B' and self.loaded is not None:
            editor.wrapper.clear(); self.loaded = None
        started = time.monotonic(); expert = None
        if selected is not None:
            artifact = self.artifacts[method, selected]
            if method == 'B':
                if self.loaded != selected:
                    editor.wrapper.clear()
                    state = torch.load(artifact['path'], map_location='cpu', weights_only=True)
                    editor.wrapper.load_exported_state(state['wrapper'])
                    self.loaded = selected
            else:
                state = torch.load(artifact['path'], map_location='cpu', weights_only=True)
                expert = vf.AsymmetricCPExpert(14336, 4096, 4).to(runtime.device)
                expert.load_state_dict(state['expert']); expert.requires_grad_(False)
        self.load_seconds += time.monotonic()-started
        started = time.monotonic()
        try:
            if method == 'B':
                with editor._activated(selected):
                    value = vf.scope_generate(runtime, row, None)
            else:
                value = sw.generated(runtime, row, expert)
            self.generated += 1
            return value
        finally:
            editor.wrapper.set_active(None)
            self.generation_seconds += time.monotonic()-started

    def close(self):
        self.editor.reset_editor_state()
        self.runtime.replace_module(self.target, self.original)


def available(run, method):
    tasks = read(run/'private/TASK_QUEUE.json')['tasks']
    completed = {(t['event_index'], t['condition']) for t in tasks if t['status'] == 'RAW_READY'}
    episodes = []
    for original in read(run/'private/NEW_EPISODE_MANIFEST_PRIVATE.json')['episodes']:
        i = original['event_index']
        if (i, CONDITIONS[method]) not in completed or (i, 'BALANCEDIT') not in completed:
            continue
        e = read(run/f'private/edits/e{i:02d}.json')
        assert e['event'] == original['event'] and e['source_eligibility_frozen']
        for row, frozen in zip(e['rows'], original['rows'], strict=True):
            assert all(row.get(k) == v for k, v in frozen.items()) and row.get('eqkey')
        episodes.append(e)
    return episodes


def evaluate(runtime, run, label):
    method = METHODS[label]; out = run/'private/final_bank'/label
    if (out/'result_private.json').exists():
        raise FileExistsError('bank exists; reuse completed result or inspect partial cache')
    episodes = available(run, method)
    if not episodes:
        vf.atomic_json(out/'result_private.json', dict(status='UNAVAILABLE', outputs=[], method=label, bank_size=0))
        return
    # Native BE defines the corresponding Base key/radius even for CP writers.
    bank.METHODS = {m:CONDITIONS[m] for m in {'B', method}}
    guard = runtime.base_guard.verify()
    artifacts, entries = bank.load_artifacts(run, episodes, guard)
    cached = bank.index_legacy(artifacts, episodes)
    router = MemoryRouter.from_state(dict(distance='euclidean', entries=entries), device=runtime.device)
    roles, overlap = bank.freeze_roles(episodes)
    target = runtime.target_lock['balancedit']['targets'][0]
    start = time.monotonic(); feature_seconds = route_seconds = 0.
    def route(row):
        nonlocal feature_seconds, route_seconds
        t = time.monotonic(); batch = bank.input_batch(runtime, row)
        with torch.no_grad():
            key = runtime.extract_layer_input_key(batch, module_path=target, pooling='mean')
        feature_seconds += time.monotonic()-t
        t = time.monotonic(); decision = asdict(router.route(key))
        route_seconds += time.monotonic()-t
        return decision
    calibration = []
    for e in episodes:
        for row in e['rows']:
            if row['role'] in ('native', 'calibration'):
                calibration.append(dict(edit=e['event_index'], role=row['role'], label=row['label'],
                    family=row.get('rewrite_family', 'native'), negative_group=row.get('negative_group'),
                    strict_role=roles[e['record_id'], row['logical_id']]['strict_role'],
                    eqkey=row['eqkey'], route=route(row)))
    lock = calibrate(calibration)
    lock.update(calibration_sha256=vf.sha256_json(calibration), frozen_before_bank_evaluation=True,
                bank_size=len(episodes), adaptation='UNSEEN_EDIT_WITH_EPISODE_CALIBRATION')
    vf.atomic_json(out/'THRESHOLD_LOCK.json', dict(lock=lock, rows=calibration))
    vf.atomic_json(out/'ROLE_LOCK_PRIVATE.json', dict(roles=[dict(source=rid, logical_id=lid, **r)
        for (rid, lid), r in roles.items()], overlaps=overlap, manifest_order=[e['record_id'] for e in episodes]))
    # All calibration queries above ran against bare frozen Base, before any
    # writer was installed. Never use an output or reference to decide routing.
    generator = Generator(runtime, artifacts)
    result = dict(status='PARTIAL', outputs=[], method=label, bank_size=len(episodes),
                  threshold=lock, manifest_order=[e['event_index'] for e in episodes], replays={},
                  provenance='EXACT_SINGLE_OUTPUT_REUSE_OR_ACTUAL_SELECTED_WRITER_GENERATION')
    derived = 0
    def output(selected, row):
        nonlocal derived
        key = ((method, selected) if selected is not None else None, row['eqkey'])
        if key in cached:
            derived += 1
            return cached[key][0]
        identity = dict(eqkey=row['eqkey'], writer=artifacts[method, selected]['identity'] if selected else 'BASE')
        path = out/'cache'/(vf.sha256_json(identity)+'.json')
        if path.exists():
            value = read(path); assert value['identity'] == identity
            answer = value['output']; derived += 1
        else:
            answer = generator.generate(method, selected, row)
            vf.atomic_json(path, dict(identity=identity, output=answer))
        cached[key] = (answer, dict(derived=True))
        return answer
    try:
        for e in episodes:
            for row in bank.evaluation_rows(e):
                if sw.active_elapsed(run) >= 10*3600 or (run/'STOP').exists():
                    raise TimeoutError('bank generation budget/STOP')
                with generator.editor.disabled():
                    decision = route(row)
                selected = decision['logical_edit_id']
                base, own, actual = output(None, row), output(e['record_id'], row), output(selected, row)
                on = accepted(decision, lock['kappa'])
                rc = actual if on else base
                branch = 'ON' if on else 'OFF'
                if branch not in result['replays']:
                    replay = generator.generate(method, selected if on else None, row)
                    assert bank.same_output(replay, rc), 'actual accepted/disabled branch mismatch'
                    result['replays'][branch] = dict(status='PASSED', edit=e['event_index'], logical_id=row['logical_id'])
                item = dict(row=row, base=base, own_forced=own, method=label, source_edit_index=e['event_index'],
                    source_expert=e['record_id'], route=decision, prefix=len(episodes),
                    **roles[e['record_id'], row['logical_id']])
                result['outputs'].append(dict(item, actual=actual, route_mode='R0', on=decision['activated'], selected_expert=selected))
                result['outputs'].append(dict(item, actual=rc, route_mode='RC', on=on, selected_expert=selected if on else None))
        result['status'] = 'RAW_READY'
    finally:
        generator.close()
        result.update(base_guard=runtime.base_guard.verify(), elapsed_seconds=time.monotonic()-start,
            costs=dict(new_training=0, generated=generator.generated, derived=derived,
                load_seconds=generator.load_seconds, generation_seconds=generator.generation_seconds,
                feature_seconds=feature_seconds, route_seconds=route_seconds,
                storage_bytes=sum(a['identity']['size_bytes'] for (m, _),a in artifacts.items() if m == method)))
        vf.atomic_json(out/'result_private.json', result)
        assert result['base_guard']['unchanged']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True, type=Path)
    p.add_argument('--method', required=True, choices=METHODS)
    args = p.parse_args()
    config = read(args.run_root/'private/CAMPAIGN_CONFIG.json')
    runtime = vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(config['runtime']['cpu_gate'])))
    evaluate(runtime, args.run_root, args.method)
