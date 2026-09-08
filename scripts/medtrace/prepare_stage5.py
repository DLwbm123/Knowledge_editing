#!/usr/bin/env python3
"""Freeze the inherited candidate list and bounded source pool; no new search."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace import prepare_stage2_sources as source
from scripts.medtrace.run_stage4 import read, vf


def main(args):
    run, old = args.run_root, args.stage4_run
    if run.exists():
        raise FileExistsError('one-shot preparation already exists')
    assert shutil_disk_free(run.parent) > 100 * 1024**3, 'insufficient storage margin'
    run.mkdir()
    probe = run/'.write_probe'
    with probe.open('x') as f:
        f.write('stage5')
    assert probe.read_text() == 'stage5'
    probe.unlink()
    config = read(old/'private/CAMPAIGN_CONFIG.json')
    config.update(kind='MEDTRACE_STAGE5', stage4_run=str(old), train_seconds=10*3600,
        wall_hours=12, gpu_hours=48, allowed_physical_gpus=[2],
        source_commit='65185fe0b2d4574819cbabb4b1c308c9d0c5900c',
        code_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        campaign_epoch=time.time())
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json', config)
    vf.atomic_json(run/'private/CAMPAIGN_START.json', dict(epoch=config['campaign_epoch']))
    screen = read(old/'private/CONFIRMATION_SCREEN.json')
    candidates = screen['candidates']
    assert len(candidates) == 32 and len({r['source_group'] for r in candidates}) == 32
    vf.atomic_json(run/'private/INHERITED_CANDIDATES.json', dict(candidates=candidates,
        provenance='Stage4 exact ordered 32; no new candidate search; no student filtering'))
    # Reconstruct only the previously cleared VQA-RAD train-side pool using
    # Stage4's saved identity exclusions. Never load sealed QUAL/heldout files.
    blocked = {(r['dataset'], r['image_id']) for r in screen['ledger']}
    rows = read(Path(candidates[0]['source_file']))
    bounded = [r for r in rows if r['phrase_type'] in ('freeform', 'para')
        and ('VQA-RAD', source.identity(r['image_name'])) not in blocked and str(r.get('answer', '')).strip()]
    expected = screen['screen']['screened_development_pool']['VQA-RAD']
    assert len(bounded) == expected['rows'] == 340
    assert len({source.identity(r['image_name']) for r in bounded}) == expected['images'] == 55
    for c in candidates:
        matches = [r for r in bounded if r['qid'] == c['source_qid'] and source.identity(r['image_name']) == c['image_id']]
        assert len(matches) == 1 and matches[0]['question'] == c['question'] and matches[0]['answer'] == c['reference']
    vf.atomic_json(run/'private/INHERITED_SOURCE_POOL.json', dict(rows=bounded,
        source_use_authorized=True, clinical_human_reviewed=False,
        status='EXACT_STAGE4_IDENTITY_BOUNDED_POOL_RECONSTRUCTED', source_rows=340, source_images=55))
    vf.atomic_json(run/'public/RUN_STATUS.json', dict(status='PREPARATION_IN_PROGRESS',
        compute='NOT_STARTED', source_review='IN_PROGRESS', existing_factorial='PENDING',
        candidate_count=32, legal_new_edit_count=None, judge='NOT_PREPARED', publication='PENDING',
        source_commit=config['source_commit'], code_commit=config['code_commit'],
        wall_hours_limit=12, gpu_hours_limit=48, gpu=2))
    print('PREPARED_INHERITED_POOL', run, flush=True)


def shutil_disk_free(path):
    import shutil
    return shutil.disk_usage(path).free


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--stage4-run', type=Path, required=True)
    main(p.parse_args())
