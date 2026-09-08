#!/usr/bin/env python3
"""Stage5 old-cohort factorial, derived from exact-bound Stage4 outputs only."""
import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace import finalize_stage4 as f4
from scripts.medtrace.stage4_scope import accepted, calibrate


def reject(item, kappa, bank):
    result = dict(item)
    on = accepted(item['route'], kappa)
    if bank:
        result.update(route_mode='RC', on=on,
                      actual=item['actual'] if on else item['base'],
                      selected_expert=item['route']['logical_edit_id'] if on else None,
                      r0_selected_expert=item['route']['logical_edit_id'], kappa=kappa)
    else:
        result.update(fixed_on=on, fixed=item['fixed'] if on else item['base'])
    return result


def pairs(rows):
    cells = defaultdict(list)
    keys = ('cohort', 'track', 'prefix', 'role', 'stratum', 'strict_role')
    for row in rows:
        cells[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, group in cells.items():
        rc = 'RC' if key[1] == 'B' else 'DEV_THRESHOLD_TRANSFER_DIAGNOSTIC'
        comparisons = [(m, rc, m, 'R0') for m in ('W0', 'W1', 'BE')]
        comparisons += [('W1', rc, 'W0', rc), ('BE', rc, 'W0', rc), ('BE', rc, 'W1', rc)]
        for candidate, mode, control, before in comparisons:
            for metric in ('semantic', 'v4_primary', 'base_correct_damage', 'base_wrong_became_correct', 'fpr', 'rejection'):
                a = {(r['edit'], r['eqkey']): r for r in group if (r['method'], r['mode']) == (candidate, mode) and r[metric] is not None}
                b = {(r['edit'], r['eqkey']): r for r in group if (r['method'], r['mode']) == (control, before) and r[metric] is not None}
                common = sorted(a.keys() & b.keys())
                if not common:
                    continue
                edits = sorted({k[0] for k in common})
                delta = [mean(a[k][metric]-b[k][metric] for k in common if k[0] == e) for e in edits]
                images = defaultdict(list)
                for k in common:
                    images[a[k]['source_image']].append(a[k][metric]-b[k][metric])
                lo, hi = f4.bootstrap(delta)
                ilo, ihi = f4.bootstrap([mean(v) for v in images.values()])
                loso = []
                for image in images:
                    keep = [k for k in common if a[k]['source_image'] != image]
                    if keep:
                        loso.append(mean(mean(a[k][metric]-b[k][metric] for k in keep if k[0] == e)
                                         for e in edits if any(k[0] == e for k in keep)))
                out.append(dict(zip(keys, key), candidate=candidate, mode=mode, control=control,
                    control_mode=before, metric=metric, delta=mean(delta), ci_low=lo, ci_high=hi,
                    paired_inputs=len(common), paired_edits=len(edits), source_images=len(images),
                    patients='UNKNOWN', image_cluster_ci_low=ilo, image_cluster_ci_high=ihi,
                    leave_one_image_out_min=min(loso) if loso else None,
                    leave_one_image_out_max=max(loso) if loso else None,
                    bootstrap_seed=20260908, bootstrap_draws=10000))
    return out


def main(args):
    run, old = args.run_root, args.stage4_run
    public = run/'public'
    if (public/'EXISTING_RC_WRITER_FACTORIAL.csv').exists():
        raise FileExistsError('existing factorial already computed; reuse it')
    f4.install()
    entries, ledger, costs = f4.inventory(old)
    verdicts, side = f4.f3.current_verdicts(old)
    assert side and set(side['all_expected']) <= verdicts.keys(), 'Stage4 Judge incomplete'
    lock = f4.read(old/'private/bank/prefix16/THRESHOLD_LOCK.json')
    calibration = f4.read(old/'private/bank/prefix16/CALIBRATION_PRIVATE.json')
    calculated = calibrate(calibration['rows'])
    assert all(lock[k] == v for k, v in calculated.items()), 'calibration contract mismatch'
    kappa = lock['kappa']
    entries = [e for e in entries if e.get('diagnostic_step') is None and e['method'] in ('W0', 'W1', 'BE')
               and (e['track'] == 'A' or e['prefix'] == 16)]
    # Every writer used precisely the same native BE router in the Stage4 bank.
    routed = {(e['edit'], e['item']['row']['logical_id']): e['item'] for e in entries
              if e['track'] == 'B' and e['method'] == 'W1' and e['route_mode'] == 'R0'}
    extra = []
    for e in entries:
        if e['track'] != 'B' or e['route_mode'] != 'R0':
            continue
        reference = routed[e['edit'], e['item']['row']['logical_id']]
        assert e['item']['route'] == reference['route']
        assert e['item']['base']['raw_token_ids'] == reference['base']['raw_token_ids']
        if e['method'] == 'BE':
            extra.append(dict(e, item=reject(e['item'], kappa, True), route_mode='RC'))
    entries += extra
    rows = f4.details(entries, verdicts, side['protocol_sha256'])
    single_be = {(e['cohort_name'], e['edit'], e['item']['row']['logical_id']): e['item']
                 for e in entries if e['track'] == 'A' and e['method'] == 'BE'}
    transfer = []
    for e in entries:
        if e['track'] != 'A' or e['cohort_name'] != f4.OLD:
            continue
        item = e['item']; native = single_be[e['cohort_name'], e['edit'], item['row']['logical_id']]
        assert native['row'] == item['row'] and native['fixed_on'] == item['fixed_on']
        assert native['base']['raw_token_ids'] == item['base']['raw_token_ids']
        # BE's flag records which request was actually replayed, not a failed
        # check: native was replayed; other requests use exact-bound Base cache.
        be_native = next(v for (c, i, _), v in single_be.items()
                         if c == e['cohort_name'] and i == e['edit'] and v['row']['role'] == 'native')
        base_verified = item.get('disabled_parity') or (e['method'] == 'BE'
            and be_native['disabled_parity'] and item.get('disabled_evidence') ==
            'actual native replay; other inputs use exact-bound frozen Base cache')
        assert base_verified and e['system_valid'], 'Base/ON parity unavailable'
        transfer.append(dict(e, item=reject(dict(item, route=native['route']), kappa, False)))
    derived = [r for r in f4.details(transfer, verdicts, side['protocol_sha256']) if r['mode'] == 'R0']
    for r in derived:
        r['mode'] = 'DEV_THRESHOLD_TRANSFER_DIAGNOSTIC'
        if r['strict_role'] == 'EDIT_TARGET':
            r['rejection'] = float(r['on'] == 0 and r['own_forced_correct'] == 1)
    rows += derived
    for r in rows:
        r['provenance'] = 'DERIVED_FROM_FROZEN_OUTPUTS'
    f4.vf.atomic_json(run/'private/EXISTING_DETAILS.json', rows)
    tables = [dict(r, provenance='DERIVED_FROM_FROZEN_OUTPUTS') for r in f4.summarize(rows)
              if r['method'] in ('BASE', 'W0', 'W1', 'BE')]
    f4.csv_write(public/'EXISTING_RC_WRITER_FACTORIAL.csv', tables)
    f4.csv_write(public/'EXISTING_RC_PAIRED_EFFECTS.csv', pairs(rows))
    f4.vf.atomic_json(public/'EXISTING_REUSE_STATUS.json', dict(status='DERIVED_COMPLETE_REPLAY_PENDING',
        new_writer_training=0, new_judge_calls=0, missing_judge_scores=sum(r['semantic'] is None and r['strict_valid'] for r in rows),
        historical_judge_closure=len(verdicts), bank_prefix=16, kappa=kappa,
        threshold_transfer_scope='OLD_STAGE3_COMMON7 single entries only; development diagnostic, not new confirmation',
        decision_contract='same original expert or Base; no rank change; frozen calibration-only threshold',
        replay='Stage4 actual OFF/Base and route parity inherited; new natural-branch replay pending'))
    print('EXISTING_DERIVED_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--stage4-run', type=Path, required=True)
    main(p.parse_args())
