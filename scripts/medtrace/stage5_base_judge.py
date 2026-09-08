#!/usr/bin/env python3
"""Reuse the complete Stage4 Judge ancestry for Stage5 Base-before membership."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace import finalize_stage4 as f4
from scripts.medtrace import finalize_stage2 as f2


def stage4_pool(config, protocol):
    f4.install()
    pool, execution = f4.historical(config, protocol)
    old = Path(config['stage4_run'])
    verdicts, side = f4.f3.current_verdicts(old)
    assert side and set(side['all_expected']) <= verdicts.keys()
    directory = old/'private/judge'
    identity = f4.read(directory/'REUSE_EXECUTION_IDENTITY_PRIVATE.json')
    for packet in f4.vf.read_jsonl(directory/'JUDGE_PACKET_PRIVATE.jsonl'):
        key = packet['opaque_query_id']
        assert packet == f4.f3.packet_row(packet['question'], packet['gold_answer'], packet['raw_base_answer'], protocol['config_sha256'])
        pool[key] = (packet, verdicts[key], identity)
    assert f4.f3.execution_identity(execution) == identity
    return pool, execution


def main(args):
    run = args.run_root
    if args.action == 'lock':
        f2.lock_base_before(SimpleNamespace(run_root=run, public_dir=run/'public'))
        status = f4.read(run/'public/RUN_STATUS.json')
        status.update(status='BASE_BEFORE_LOCKED', compute='STUDENT_TRAINING_PENDING', judge='BASE_BEFORE_COMPLETE')
        f4.vf.atomic_json(run/'public/RUN_STATUS.json', status)
        return
    config = f4.read(run/'private/CAMPAIGN_CONFIG.json')
    protocol = f4.read(Path(config['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json')
    pool, execution = stage4_pool(config, protocol)
    entries = []
    for episode in f4.read(run/'private/NEW_EPISODE_MANIFEST_PRIVATE.json')['episodes']:
        i = episode['event_index']; data = f4.read(run/f'private/edits/e{i:02d}.json')
        for row in data['rows']:
            base = f4.read(run/f'private/base_generation/e{i:02d}'/(f4.vf.sha256_json(row)+'.json'))
            entries.append(dict(track='A', target=row['reference'], item=dict(row=row, base=base, forced=base, fixed=base)))
    # A distinct child snapshot lets the existing Judge engine keep immutable
    # packet files while later student/bank Judge work uses its own directory.
    child = run/'private/base_judge_snapshot'
    f4.vf.atomic_json(child/'private/CAMPAIGN_CONFIG.json', config)
    f4.f3.inventory = lambda _: (entries, [], [])
    f4.f3.historical_judge = lambda *_: (pool, execution)
    side = f4.f3.prepare_judge(SimpleNamespace(run_root=child, max_model_len=2048))
    assert not side['execution_version_changed'], 'no Judge runtime/cap change authorized'
    destination = run/'private/base_judge'
    if not destination.exists():
        destination.symlink_to(child/'private/judge', target_is_directory=True)
    else:
        assert destination.resolve() == (child/'private/judge').resolve()
    print('BASE_JUDGE_PREPARED', side['reused'], side['new'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'lock'))
    p.add_argument('--run-root', type=Path, required=True)
    main(p.parse_args())
