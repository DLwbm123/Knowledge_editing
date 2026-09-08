"""One rejection-only calibration rule. Inputs are frozen calibration metadata only."""
import math
from collections import defaultdict
from statistics import mean


def margin(route):
    if not route['activated']:
        return None
    d, r = route['nearest_distance'], route['radius']
    if d is None or r is None or not math.isfinite(d) or not math.isfinite(r) or r <= 0 or d < 0:
        return None
    return (r-d)/max(r, 1e-12)


def accepted(route, kappa):
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError('kappa must be nonnegative finite')
    m = margin(route)
    return bool(route['activated'] and m is not None and m >= kappa)


def coverage(rows, kappa, macro):
    if not rows:
        return None
    groups = defaultdict(list)
    for row in rows:
        groups[row['edit']].append(float(accepted(row['route'], kappa)))
    return mean(mean(v) for v in groups.values()) if macro else mean(v for vs in groups.values() for v in vs)


def calibrate(rows):
    if any(r['role'] not in ('native','calibration') for r in rows):
        raise ValueError('fit/evaluation cannot calibrate rejection')
    if any(r['route']['activated'] != accepted(r['route'], 0.) for r in rows):
        raise ValueError('kappa=0 not equivalent to R0; invalid radius or mismatched distance units')
    positives = [r for r in rows if r['role']=='calibration' and r['label']=='positive']
    negative = {g:[r for r in rows if r['role']=='calibration' and r['negative_group']==g
                    and r['strict_role']=='STRICT_BASE'] for g in ('H','U')}
    if not positives or not all(negative.values()):
        return dict(kappa=0.,status='CALIBRATION_UNSUPPORTED',calibration_rows=len(rows))
    families = sorted({r['family'] for r in positives})
    native = [r for r in rows if r['role']=='native']
    # IEEE binary64 comparisons, inclusive acceptance, nextafter crosses each discontinuity.
    candidates = sorted({0., *[math.nextafter(m, math.inf) for r in rows
                               if (m:=margin(r['route'])) is not None and m >= 0]})
    scored=[]
    for k in candidates:
        if any(r['route']['activated'] and not accepted(r['route'],k) for r in native):
            continue
        constraints=[]
        for family in families:
            group=[r for r in positives if r['family']==family]
            for macro in (False,True):
                before,after=coverage(group,0.,macro),coverage(group,k,macro)
                constraints.append(dict(family=family,average='macro' if macro else 'micro',
                                        r0=before,rc=after,decrease=before-after))
        if any(v['decrease'] > .05+1e-12 for v in constraints):
            continue
        objective=mean(coverage(negative[g],k,True) for g in ('H','U'))
        scored.append((objective,k,constraints))
    best=min(scored,key=lambda t:(t[0],t[1]))
    initial=mean(coverage(negative[g],0.,True) for g in ('H','U'))
    return dict(kappa=best[1],status='CALIBRATED' if best[0] < initial else 'NO_SCOPE_SEPARATION',
                r0_objective=initial,rc_objective=best[0],constraints=best[2],
                candidates=len(candidates),feasible=len(scored),calibration_rows=len(rows),
                tie_rule='binary64 inclusive >=; nextafter breakpoints; minimum objective then smallest kappa',
                objective='equal H/U edit-macro strict-base activation; no outcome/gold scoring')
