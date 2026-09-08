#!/usr/bin/env python3
"""Frozen Stage3 event groups on the existing resident Stage2 runtime."""
import argparse
from collections import Counter
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace import run_stage2 as s2

vf, sw, read = s2.vf, s2.sw, s2.read


def public_manifest(run):
    private = read(run/'RUN_MANIFEST.json')
    config = read(run/'private/CAMPAIGN_CONFIG.json')
    locks = read(ROOT/'reports/medtrace_stage2_20260908/METHOD_CONFIG_LOCKS.json')
    cp = dict(locks['new_cp'], conditions=['P4-W0_TASK_ONLY','P4-W1_KL_0.1'])
    cp.pop('w2', None)
    value = dict(protocol='V4_RELEASE_ALIGNED_AUGMENTED_SUPPORT_EVALUATION',
        candidate='MEDTRACE_SW_P4_KL01_FROZEN_CANDIDATE_V1',
        methods=dict(S1='BE_ROUTE + P4-W1_KL_0.1', S0='same BE_ROUTE + P4-W0_TASK_ONLY',
                     B='BalancEdit V4 adaptation native routing'), cp=cp, balancedit=locks['balancedit'],
        counts=private['counts'], preparation_status=private['status'],
        stage2_public_commit='74d2a337f7d2b830d58819f76c87058cef0c5f3b',
        stage2_source_execution_commit='3cde3ed0652a2577d81262abd10e112f669b49e9',
        execution_commit=config['code_commit'], result_commit='assigned at result publication',
        historical_patch_provenance='Stage2 EXECUTION_PATCH_PROVENANCE.json; compatible source ancestry retained',
        sources={key: Path(value).name for key,value in private['sources'].items()},
        exposure='Stage2 bank and old MedTRACE edits viewed; other cohorts separately tagged, no whole-cohort blind claim',
        patient_identity='UNKNOWN', task_order='bank prefixes 1,4,8,16; T0 anchor groups; frozen task-specific groups',
        checkpoint_reuse=private['checkpoint_reuse'], base_reuse='Exact runtime, actual prompt tokens and image binding; no global Base rerun',
        resources=locks['resource_limits'], performance_gate=False, t5='NA_NO_LEGAL_MATERIAL',
        native_condition_policy='Stage3 retains nonfinite hard stop, not legacy finite condition-number performance threshold',
        unsupported_policy='S0 may run without H/U; S1 never drops missing protection loss',
        scientific_claim='Frozen augmented-support systems evaluation, not paper-exact M3Bench or clinical validation')
    vf.atomic_json(run/'public/RUN_MANIFEST.json', value)
    vf.atomic_json(run/'public/EXECUTION_STATUS.json', dict(status='PREPARED', compute='NOT_RUN', judge='NOT_RUN',
        publication='PENDING', queue_counts=dict(Counter(t['status'] for t in read(run/'private/TASK_QUEUE.json')['tasks']))))


def preflight(run):
    config = read(run / 'private/CAMPAIGN_CONFIG.json')
    start = read(run / 'private/CAMPAIGN_START.json')
    if config['kind'] != 'MEDTRACE_STAGE3' or config['gpu_uuids'] != sw.GPUS:
        raise ValueError('wrong campaign or GPU identity')
    if start['code_commit'] != config['code_commit'] or start['epoch'] <= 0:
        raise ValueError('coordinator start/config binding missing')
    for task in read(run / 'private/TASK_QUEUE.json')['tasks']:
        if task['kind'] == 'SINGLE_GROUP':
            data = read(run / f"private/edits/e{task['event_index']:02d}.json")
            if not data.get('source_eligibility_frozen'):
                raise ValueError('unfrozen training/evaluation inputs')
    return config


def method_task(group, method):
    return dict(task_id=group['task_id'].removesuffix('_GROUP')+'_'+method,
                event_index=group['event_index'], seed=s2.SEED,
                kind='BE' if method == 'BE' else 'CP',
                parameterization='BE' if method == 'BE' else 'P4',
                condition={'BE':'BALANCEDIT', 'S0':'W0_TASK_ONLY', 'S1':'W1_KL_0.1'}[method])


def single_group(runtime, args, task):
    start = time.monotonic()
    run = args.run_root
    done = []
    for method in ('BE', 'S0', 'S1'):
        if method not in task['methods']:
            continue
        sub = method_task(task, method)
        result_path = run / 'private/tasks' / sub['task_id'] / 'result_private.json'
        if result_path.exists():
            result = read(result_path)
            if result['task'] != sub or not result['base_guard']['unchanged']:
                raise ValueError('saved current-run endpoint identity mismatch')
        elif method == 'BE':
            result = s2.be_task(runtime, args, sub)
        else:
            data = read(run / f"private/edits/e{sub['event_index']:02d}.json")
            if 'frozen_gate' not in data:
                raise ValueError('missing shared Base-only BE route decisions')
            if 'a2' not in data:
                s2.initialize_episode(runtime, args, dict(sub, kind='INIT'))
            result = sw.train_task(runtime, args, sub)
        if any(not item.get('system_replay_valid', item.get('route_branch_parity', True))
               for item in result['outputs'].values()):
            raise ValueError('actual system generation disagrees with frozen route branch')
        done.append(method)
        print('METHOD_DONE', sub['task_id'], flush=True)
        gc.collect()
        s2.torch.cuda.empty_cache()
    if {'S0','S1'} <= set(done):
        a, b = [read(run / 'private/tasks' / method_task(task, m)['task_id'] / 'result_private.json') for m in ('S0','S1')]
        if a['frozen_gate_sha256'] != b['frozen_gate_sha256'] or a['a2_sha256'] != b['a2_sha256']:
            raise ValueError('paired S0/S1 route or common A2 drift')
    return dict(status='RAW_READY', task=task, methods=done, elapsed_seconds=time.monotonic()-start,
                base_guard=runtime.base_guard.verify())


def worker(args):
    # CPU prerequisites are checked before allocating a model.
    config = preflight(args.run_root)
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    if gpu not in sw.GPUS or os.environ.get('M3BENCH_FORMAL_EXPECTED_GPU_UUID') != sw.GPUS[gpu]:
        raise ValueError('unauthorized GPU')
    if sw.gpu_check(gpu)['free_mib'] < 24576:
        raise RuntimeError('worker requires inherited 24576 MiB free guard')
    runtime = vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(config['runtime']['cpu_gate'])))
    runtime.stage2_base = {r['query_id']: r for r in vf.read_jsonl(Path(config['runtime']['base_predictions']))}
    sw.SEED = s2.SEED
    queue = vf.TaskQueue(args.run_root / 'private/TASK_QUEUE.json', args.run_root)
    from scripts.medtrace.run_stage3_bank import run_bank
    with vf.Telemetry(args.run_root / 'private/GPU_TELEMETRY.jsonl', gpu, 'gpu'+gpu) as telemetry:
        while sw.active_elapsed(args.run_root) < config['train_seconds'] and not (args.run_root/'STOP').exists():
            task = queue.claim('gpu'+gpu)
            if task is None:
                break
            telemetry.task_id = task['task_id']
            print('START', task['task_id'], flush=True)
            started = time.monotonic()
            try:
                result = run_bank(runtime, args, task) if task['kind'] == 'BANK' else single_group(runtime, args, task)
                if not runtime.base_guard.verify()['unchanged']:
                    raise RuntimeError('Base changed')
                queue.update(task['task_id'], 'RAW_READY', elapsed_seconds=time.monotonic()-started, finished_epoch=time.time())
                with (args.run_root/'private/first_task.lock').open('a+') as lock:
                    s2.fcntl.flock(lock, s2.fcntl.LOCK_EX)
                    if not (args.run_root/'FIRST_TASK_COMPLETE.json').exists():
                        vf.atomic_json(args.run_root/'FIRST_TASK_COMPLETE.json', dict(task_id=task['task_id'],
                            elapsed_seconds=time.monotonic()-started, epoch=time.time(), performance_gate=False))
                print('DONE', task['task_id'], flush=True)
            except Exception as error:
                queue.update(task['task_id'], 'FAILED', last_error=f'{type(error).__name__}: {error}')
                vf.append_jsonl(args.run_root/'private/FAILURES.jsonl', dict(task_id=task['task_id'], kind=task['kind'],
                    error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(), epoch=time.time()))
                # Fresh-process recovery may run independent pending tasks; never reuse an uncertain runtime.
                raise
            gc.collect()
            s2.torch.cuda.empty_cache()
            if args.first_only:
                break


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--first-only', action='store_true')
    worker(p.parse_args())
