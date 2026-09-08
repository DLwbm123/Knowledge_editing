#!/usr/bin/env python3
"""Small natural ON/OFF replay of the old frozen bank; no threshold changes."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace import run_stage3_bank as bank
from scripts.medtrace.stage5_bank import Generator, METHODS, CONDITIONS
from scripts.medtrace.stage4_scope import accepted
from scripts.medtrace.run_stage2 import vf, read
from m3bench_repro.editors.routing import MemoryRouter
import torch


def main(args):
    run = args.run_root
    if (run/'public/EXISTING_REPLAY_STATUS.json').exists():
        raise FileExistsError('old natural replay already recorded')
    config = read(run/'private/CAMPAIGN_CONFIG.json')
    runtime = vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(config['runtime']['cpu_gate'])))
    old = Path(config['stage4_run']); stage2 = Path(config['stage2_run'])
    episodes = bank.load_manifest(stage2)
    bank.METHODS = CONDITIONS
    artifacts, entries = bank.load_artifacts(stage2, episodes, runtime.base_guard.verify())
    router = MemoryRouter.from_state(dict(distance='euclidean', entries=entries), device=runtime.device)
    result = read(old/'private/bank/prefix16/result_private.json')
    kappa = read(old/'private/bank/prefix16/THRESHOLD_LOCK.json')['kappa']
    generator = Generator(runtime, artifacts)
    replays = []
    try:
        for label, method in METHODS.items():
            seen = set()
            for item in result['outputs']:
                if item['method'] != label or item['route_mode'] != 'R0':
                    continue
                on = accepted(item['route'], kappa); branch = 'ON' if on else 'OFF'
                if branch in seen:
                    continue
                row = item['row']
                with generator.editor.disabled(), torch.no_grad():
                    batch = bank.input_batch(runtime, row)
                    key = runtime.extract_layer_input_key(batch, module_path=generator.target, pooling='mean')
                    observed = asdict(router.route(key))
                assert observed == item['route'], 'actual Base route changed'
                expected = item['actual'] if on else item['base']
                replay = generator.generate(method, observed['logical_edit_id'] if on else None, row)
                assert bank.same_output(replay, expected), 'natural RC branch replay mismatch'
                replays.append(dict(method=label, branch=branch, status='PASSED', edit=item['source_edit_index']))
                seen.add(branch)
                if len(seen) == 2:
                    break
            for branch in {'ON','OFF'}-seen:
                replays.append(dict(method=label, branch=branch, status='NOT_NATURALLY_OBSERVED'))
    finally:
        generator.close()
    assert runtime.base_guard.verify()['unchanged']
    vf.atomic_json(run/'public/EXISTING_REPLAY_STATUS.json', dict(status='COMPLETE', replays=replays,
        additional_writer_training=0, additional_judge_calls=0, generated=generator.generated,
        selection='first natural branch in frozen source order; no outcome selection or threshold adjustment'))
    print('EXISTING_REPLAY_COMPLETE', replays, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    main(p.parse_args())
