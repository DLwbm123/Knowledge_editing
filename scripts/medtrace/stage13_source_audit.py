#!/usr/bin/env python3
"""Bounded Stage13 source-first audit; no model, sealed data, or student selection."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import shutil
import time


def read(path):
    return json.loads(Path(path).read_text())


def image_key(row):
    path=Path(row.get('image_path',row.get('image_name','')))
    if not str(path) or str(path)=='.':raise ValueError('missing source image identity')
    return row['dataset'],(path.parent.name if path.name.lower()=='source.jpg' else path.name).lower()


def source_support(pool,stage_episodes):
    used=defaultdict(set)
    for stage,episodes in stage_episodes.items():
        for episode in episodes:
            for row in episode['rows']:used[image_key(row)].add(stage)
    grouped=defaultdict(list)
    for row in pool:grouped[image_key(row)].append(row)
    return grouped,used,[r for r in pool if image_key(r) not in used]


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:json.dump(value,handle,indent=2,sort_keys=True);handle.write('\n')


def main(args):
    started=time.time();cfg=read(args.stage12_run/'private/CAMPAIGN_CONFIG.json')
    assert read(args.stage12_run/'RUN_COMPLETION.json')['status']=='COMPUTE_COMPLETE'
    stage2=Path(cfg['stage2_run']);a=read(stage2/'private/NEW_EPISODE_MANIFEST_PRIVATE.json')
    b=read(args.stage5_run/'private/NEW_EPISODE_MANIFEST_PRIVATE.json')
    assert read(stage2/'RUN_COMPLETION.json')['new_n']==len(a['episodes'])==16
    done=read(args.stage5_run/'RUN_COMPLETION.json')
    assert done['completed_writers']['W0']==done['completed_writers']['BE']==len(b['episodes'])==32
    grouped,used,fresh=source_support(a['source_pool'],{'Stage2':a['episodes'],'Stage5':b['episodes']})
    # This closeout branch must not silently discard a genuinely fresh source.
    if fresh:raise RuntimeError('Fresh authorized images exist; complete episode assembly is required, do not emit N=0')
    run=args.run_root
    if run.exists():raise FileExistsError('existing Stage13 audit must not be overwritten')
    assert shutil.disk_usage(run.parent).free>1024**3
    run.mkdir();probe=run/'.probe';probe.write_text('run');assert probe.read_text()=='run';probe.unlink()
    needed=[
        'New authorized native image/source QA outside all Stage0-12 development images and native fact-equivalence groups; fixed target and original Base-wrong eligibility.',
        'Native fit paraphrases meeting original CP/A2/W0 initialization contract; use existing source alternatives or authorized answer-preserving derivation with review provenance.',
        'At least one same-proposition conflicting H-fit source QA on another image whose existing role permits training.',
        'At least one same-proposition conflicting H-evaluation QA on a different image, excluded from every method training image set.',
        'Original required U-fit support plus role-separated U-evaluation and locked distinct positive-evaluation text families; keep available same-answer contexts.',
        'Freeze explicit source IDs, per-edit deterministic seeds and separate training/evaluation-answer files before student execution; no sealed QUAL/test role reassignment.'
    ]
    remediation=[];private=[]
    for index,((dataset,image),rows) in enumerate(sorted(grouped.items()),1):
        opaque=f'source-group-{index:03d}'
        remediation.append(dict(source_group=opaque,dataset=dataset,qa_rows=len(rows),base_wrong_rows=sum(r.get('base_correct') is False for r in rows),
            excluded_reason='ALREADY_USED_MEDTRACE_DEVELOPMENT_IMAGE',development_stages=sorted(used[dataset,image]),
            action='REPLACE_WITH_GENUINELY_UNEXPOSED_AUTHORIZED_EPISODE; cannot repair independence by rewriting old questions',required_material_ids=list(range(1,len(needed)+1))))
        private.append(dict(source_group=opaque,dataset=dataset,image_id=image,source_qids=[r['source_qid'] for r in rows],
            source_files=sorted({r['source_file'] for r in rows}),development_stages=sorted(used[dataset,image])))
    write_new(run/'private/SOURCE_GROUP_BINDINGS.json',private)
    write_new(run/'private/CONFIG.json',dict(stage12_run=str(args.stage12_run),stage5_run=str(args.stage5_run),stage2_run=str(stage2),protocol=str(args.protocol),execution_sha=args.commit))
    shutil.copyfile(args.protocol,run/'private/PROTOCOL.md')
    record=dict(status='SOURCE_AUDIT_COMPLETE__NEW_EDIT_EVALUATION_NOT_IMPLEMENTED',actual_N=0,target_N=32,
        stage12_public_sha='607df8440ecd9fd08d7b8b5c263928957dac9ce6',execution_sha=args.commit,
        authorization='USER_STAGE13_GPU3',gpu_used=False,gpu_hours=0,training='NOT_RUN_NO_ELIGIBLE_EPISODES',generation='NOT_RUN',judge='NOT_RUN',publication='PENDING',
        candidate_scope='Previously authorized and role-cleared source pool from Stage2; preserve its explicit sealed/formal/source-test exclusions. Not an exhaustive claim about all public datasets.',
        source_rows=len(a['source_pool']),source_images=len(grouped),fresh_images=0,
        base_wrong_rows=sum(r.get('base_correct') is False for r in a['source_pool']),
        source_rows_by_dataset=dict(Counter(r['dataset'] for r in a['source_pool'])),source_images_by_dataset=dict(Counter(k[0] for k in grouped)),
        verified_completed_development=dict(Stage2_edits=16,Stage5_edits=32,Stage2_images=len({image_key(r) for e in a['episodes'] for r in e['rows']}),Stage5_images=len({image_key(r) for e in b['episodes'] for r in e['rows']})),
        exclusion_logic='Source image disjointness is necessary and fails for every admitted image before native selection; no arbitrary first32 sampling. Base-only exposure is not an exclusion.',
        later_stages='Stage2 and Stage5 alone exhaust the admitted pool; additional development exposure can only exclude more, so no full Stage0-12 artifact rescan is needed.',
        native_fact_group_selection='NOT_REACHED; no unexposed source image remains',patient_id='UNKNOWN',
        required_materials={str(i):text for i,text in enumerate(needed,1)},remediation=remediation,
        preparation_seconds=time.time()-started,source_budget_minutes=90,wall_limit_hours=8,gpu_limit_hours=16,
        primary_writer_metric='H-evaluation PairCorrect edit macro, with native correctness; all NA because N=0',
        fixed_writer='Stage12 C_FACT free rank4, original CP/A2/W0 initialization lineage,320 steps',
        fixed_system='C_FACT+BE_ROUTE+RC_FIXED_OLD16; kappa0.7696741135364367, not recalibrated',
        conclusion='Independent new-edit evaluation was not implemented; this is a source-availability boundary, not negative method evidence. Old15 not substituted; no automatic next stage.')
    write_new(run/'public/RUN_SUPPORT_COST.json',record)
    for name in ('NEW_EDIT_WRITER_RESULTS.csv','NEW_EDIT_SYSTEM_RESULTS.csv'):
        path=run/'public'/name
        with path.open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['status','actual_N','metric','value','reason']);writer.writeheader()
            writer.writerow(dict(status='NOT_EVALUATED',actual_N=0,metric='ALL_PREDECLARED',value='NA',reason='No new source image remains in authorized role-cleared pool'))
    lines=['# Stage13 frozen new-edit evaluation: source closeout','',
        '**Actual N=0. No student training, generation or Judge was run. This is not a negative scientific result.**','',
        f'The existing role-cleared pool has {len(a["source_pool"])} source QA rows across{len(grouped)} images (SLAKE25; VQA-RAD92). All117 images occur in completed Stage2 or Stage5 MedTRACE development episodes: Stage2 uses62 and Stage5 uses55. These are actual completed student-development episodes, not Base-only inference exclusions.',
        'The117-row remediation list in RUN_SUPPORT_COST.json identifies every source image group by a sanitized identifier, its development exposure, QA count and required replacement materials. Exact source paths/QIDs remain in private/SOURCE_GROUP_BINDINGS.json.',
        'No arbitrary32 natives were selected; the necessary source-image disjointness check eliminates all existing source groups before complete-episode selection. Additional source-only QA on these same images cannot fix independence. The previous bounded scope is not expanded to sealed QUAL, source test/evaluation training roles or unauthorized datasets.',
        '', '## Concrete materials needed to resume','']
    lines += [f'{i}. {text}' for i,text in enumerate(needed,1)]
    lines += ['', 'One complete legally supported new episode is enough to proceed;32 is a target, not an admission gate. Supply a separate authorized source package with explicit train/evaluation roles and source-image/fact exposure records. No request to unlock old heldout data is implied.',
        'Stage12 writer-level H-supervision benefits and fixed-RC system ties remain unchanged. Their continuation on independent new edits remains untested. No Stage14, old15 retraining, method/threshold search or GPU waiting automation was started.',
        'The resource ceiling is8h wall/16GPUh; it does not require consuming GPU time when no episode can legally enter training. Publication contains only the audit code, aggregate counts, sanitized remediation metadata and NA result markers.']
    (run/'public/GPT_PRO_REVIEW.md').write_text('\n'.join(lines)+'\n')
    write_new(run/'RUN_COMPLETION.json',dict(status=record['status'],actual_N=0,training='NOT_RUN',judge='NOT_RUN',publication='PENDING'))
    print(json.dumps({k:record[k] for k in ('status','actual_N','source_rows','source_images','fresh_images','base_wrong_rows')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage12-run',type=Path,required=True);p.add_argument('--stage5-run',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--commit',required=True)
    main(p.parse_args())
