"""Freeze new single-edit RC before any evaluation generation."""
from dataclasses import asdict
from scripts.medtrace.stage4_scope import calibrate
from scripts.medtrace.run_stage3_bank import input_batch


def freeze(runtime, editor, data, run):
    from scripts.medtrace.run_stage2 import vf
    import torch
    rows = []
    for row in data['rows']:
        if row['role'] not in ('native', 'calibration'):
            continue
        batch = input_batch(runtime, row)
        with torch.no_grad(), editor.disabled():
            key = runtime.extract_layer_input_key(batch, module_path=editor.target, pooling='mean')
        rows.append(dict(edit=data['event_index'], role=row['role'], label=row['label'],
            family=row.get('rewrite_family', 'native'), negative_group=row.get('negative_group'),
            strict_role='EDIT_TARGET' if row['label'] == 'positive' else 'STRICT_BASE',
            eqkey=row['eqkey'], route=asdict(editor.router.route(key))))
    lock = calibrate(rows)
    lock.update(calibration_sha256=vf.sha256_json(rows), frozen_before_evaluation=True,
                adaptation='UNSEEN_EDIT_WITH_EPISODE_CALIBRATION', semantic_family_independence=False)
    path = run/f"private/single_scope/e{data['event_index']:02d}.json"
    if path.exists():
        raise FileExistsError('single threshold already frozen; no silent retraining')
    vf.atomic_json(path, dict(lock=lock, rows=rows))
