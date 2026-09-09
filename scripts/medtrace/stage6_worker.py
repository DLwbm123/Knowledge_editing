#!/usr/bin/env python3
"""Frozen selected-writer replay and new evaluation-only bank inputs."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.medtrace import stage6 as s
from m3bench_repro.editors.routing import MemoryRouter
import torch


def main(args):
    run=args.run_root; config=s.read(run/'private/CAMPAIGN_CONFIG.json')
    old=Path(config['stage5_run']);label=args.method;method=s.METHODS[label]
    out=run/f'private/sidecar_{label}.json'
    if out.exists():raise FileExistsError('completed/partial result exists; inspect, do not restart whole worker')
    k=s.threshold(run)
    runtime=s.vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(config['runtime']['cpu_gate'])))
    episodes=s.available(old,method);assert len(episodes)==32
    s.bank.METHODS={m:s.CONDITIONS[m] for m in ('B',method)}
    artifacts,keys=s.bank.load_artifacts(old,episodes,runtime.base_guard.verify())
    router=MemoryRouter.from_state(dict(distance='euclidean',entries=keys),device=runtime.device)
    generator=s.Generator(runtime,artifacts)
    byedit={e['event_index']:e for e in episodes}
    keybyid={e['logical_edit_id']:e for e in keys}
    def guard_budget():
        if time.time()-config['campaign_epoch']>=4*3600 or (run/'STOP').exists():raise TimeoutError('Stage6 generation deadline/STOP')
    def route(row,selected_router=router):
        with generator.editor.disabled(),torch.no_grad():
            batch=s.bank.input_batch(runtime,row)
            key=runtime.extract_layer_input_key(batch,module_path=generator.target,pooling='mean')
            return asdict(selected_router.route(key))
    old_fit_eqkeys={r['eqkey'] for e in episodes for r in e['rows'] if r['role']=='fit'}
    replays=[];cache={};entries=[];started=time.time()
    def generate(selected,row):
        identity=dict(eqkey=row['eqkey'],writer=artifacts[method,selected]['identity'] if selected else 'BASE')
        path=run/'private/cache'/label/(s.vf.sha256_json(identity)+'.json')
        cachekey=selected,row['eqkey']
        if cachekey in cache:return cache[cachekey]
        if path.exists():
            value=s.read(path);assert value['identity']==identity;output=value['output']
        else:
            guard_budget();output=generator.generate(method,selected,row)
            s.vf.atomic_json(path,dict(identity=identity,output=output))
        cache[cachekey]=output;return output
    try:
        # First natural branch per writer/deployment, at most 2*2*2=8 requests.
        originals=s.read(run/'private/TRANSFER_ENTRIES.json')
        for track in ('A','B'):
            seen=set()
            for entry in originals:
                if entry['method']!=label or entry['track']!=track or entry['route_mode']!=s.FIXED:continue
                item=entry['item'];row=item['row'];on=s.accepted(item['route'],k);branch='ON' if on else 'OFF'
                if branch in seen:continue
                guard_budget()
                rr=router if track=='B' else MemoryRouter.from_state(dict(distance='euclidean',entries=[keybyid[byedit[entry['edit']]['record_id']]]),device=runtime.device)
                observed=route(row,rr)
                assert observed==item['route'],'Base route replay mismatch'
                expected=item['actual' if track=='B' else 'fixed']
                answer=generator.generate(method,observed['logical_edit_id'] if on else None,row)
                assert s.bank.same_output(answer,expected),'fixed-transfer actual token replay mismatch'
                replays.append(dict(track=track,branch=branch,status='PASSED',edit=entry['edit']))
                seen.add(branch)
            for b in sorted({'ON','OFF'}-seen):replays.append(dict(track=track,branch=b,status='NOT_NATURALLY_OBSERVED'))
        s.vf.atomic_json(run/f'private/replay_{label}.json',dict(status='COMPLETE',method=label,replays=replays))
        side=s.read(run/'private/EVAL_SIDECAR.json')
        assert side['status']=='FROZEN_BEFORE_NEW_INPUT_ROUTES_AND_OUTPUTS'
        for probe in side['rows']:
            guard_budget();row=dict(probe['row']);assert row['role']=='evaluation'
            owner=byedit[probe['edit']]
            # Only realizes EqKey; no routes or labels are chosen by bind_rows.
            s.bind_rows(runtime,dict(event=owner['event'],rows=[row],track='STAGE6_EVALUATION_ONLY'))
            assert row['eqkey'] not in old_fit_eqkeys, 'new sidecar realized input overlaps old fit'
            decision=route(row);selected=decision['logical_edit_id'] if decision['activated'] else None
            baseline=generate(None,row);actual=generate(selected,row)
            positive=row['label']=='positive'
            item=dict(row=row,base=baseline,actual=actual,own_forced=baseline,
                own_diagnostic_available=False,own_forced_placeholder='BASE_SCHEMA_ONLY_NOT_OWNER_EVIDENCE',
                route=decision,on=decision['activated'],selected_expert=selected,source_expert=owner['record_id'],
                strict_role='EDIT_TARGET' if positive else 'STRICT_BASE',effective_reference=row['reference'],
                legitimate_experts=[owner['record_id']] if positive else [],source_edit_index=probe['edit'])
            entry=dict(track='B',prefix=32,edit=probe['edit'],method=label,item=item,common_support=False,
                system_valid=True,cohort_name=s.NEW,route_mode='R0',provenance='ACTUAL_SELECTED_WRITER_OR_EXACT_CACHE')
            entries.extend([entry,s.transfer(entry,k)])
        status='RAW_READY'
    except Exception:
        status='PARTIAL';raise
    finally:
        generator.close();guard=runtime.base_guard.verify()
        s.vf.atomic_json(out,dict(status=status,entries=entries,base_guard=guard,
            actual_source_order=[e['event_index'] for e in episodes],method=label,
            generated=generator.generated,load_seconds=generator.load_seconds,generation_seconds=generator.generation_seconds,
            checkpoint_identities=[dict(writer=m,expert=rid,**a['identity']) for (m,rid),a in artifacts.items()],
            elapsed_seconds=time.time()-started))
        assert guard['unchanged']
    print('SIDECAR_RAW_READY',label,len(entries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--method',choices=('W0','BE'),required=True)
    main(p.parse_args())
