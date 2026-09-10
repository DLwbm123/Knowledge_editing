#!/usr/bin/env python3
"""Frozen fact-contrast experiment; original fit data only, no APR search."""
import argparse
from collections import Counter,defaultdict
import copy
from dataclasses import replace
from pathlib import Path
import random
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from scripts.medtrace import stage7 as s,stage8 as previous
from methods.medtrace.fact_contrast import answer_score,pair_half,legal_pair
from methods.medtrace.selective_write import optimizer_for,balanced_schedule,full_vocab_kl
from scripts.medtrace.stage5_existing import pairs
vf,read,f4=s.vf,s.read,s.f4


def prepare(args):
    run=args.run_root;old=args.stage8_run
    if run.exists():raise FileExistsError('existing experiment must be resumed, not overwritten')
    assert read(old/'RUN_COMPLETION.json')['status']=='COMPUTE_COMPLETE'
    assert shutil.disk_usage(run.parent).free>30*1024**3
    run.mkdir();p=run/'.probe';p.write_text('stage9');assert p.read_text()=='stage9';p.unlink()
    config=read(old/'private/CAMPAIGN_CONFIG.json');config.update(kind='MEDTRACE_STAGE9',stage8_run=str(old),
        campaign_epoch=time.time(),allowed_physical_gpus=[0,2,3],worker_gpus=[2,3],judge_gpu=0,code_commit=args.commit,
        wall_hours=8,gpu_hours=8,new_sgd_training=True)
    vf.atomic_json(run/'private/CAMPAIGN_CONFIG.json',config)
    vf.atomic_json(run/'private/CAMPAIGN_START.json',dict(epoch=config['campaign_epoch']))
    vf.atomic_text(run/'private/PROTOCOL.md',args.protocol.read_text())
    sources={};tasks=read(old/'private/TASKS.json');public=[]
    def source_matches(row):
        path=row.get('source_file')
        if not path:return False
        if path not in sources:sources[path]=read(Path(path))
        return any(str(r.get('qid'))==str(row['source_qid']) and r.get('question')==row['question'] and r.get('answer')==row['reference']
            and Path(row['image_path']).as_posix().endswith('/'+r.get('img_name','INVALID')) for r in sources[path])
    for t in tasks:
        rows=t['data']['rows'];native=next(r for r in rows if r['role']=='native');h=[];u=[];rejected=Counter();relations=[]
        groups=defaultdict(list)
        for row in rows:
            if row['role']=='fit' and row.get('negative_group')=='H':groups[row['source_group']].append(row)
        for key,candidates in sorted(groups.items()):
            for row in sorted(candidates,key=lambda r:(r['eqkey'],r['logical_id'])):
                ok,reason=legal_pair(native,row)
                if ok and not (source_matches(native) and source_matches(row)):ok=False;reason='ORIGINAL_SOURCE_QA_MISMATCH'
                if ok:
                    h.append(row['logical_id']);relations.append(reason);break
                rejected[reason]+=1
            if len(h)==4:break
        ugroups=defaultdict(list)
        for row in rows:
            if row['role']=='fit' and row.get('negative_group')=='U':ugroups[row['source_group']].append(row)
        for key,candidates in sorted(ugroups.items())[:4]:u.append(sorted(candidates,key=lambda r:(r['eqkey'],r['logical_id']))[0]['logical_id'])
        positive=[r['logical_id'] for r in rows if r['role']=='fit' and r['label']=='positive']
        runnable=bool(h and u and positive)
        t.update(h_ids=h,u_ids=u,fit_positive_ids=positive,native_id=native['logical_id'],
            stage9_status='PENDING' if runnable else 'UNSUPPORTED_MATCHED_TRAINING_INPUT',rejected=dict(rejected))
        public.append(dict(order=t['order'],cohort=t['cohort'],edit=t['event_index'],status=t['stage9_status'],
            H_pairs=len(h),H_source_groups=len(h),U_source_groups=len(u),fit_positive_inputs=len(positive),rejected=dict(rejected),relations=relations,
            new_use='original fit H source-answer supervision; exact question, source records and explicit slot/proposition relation',patient='UNKNOWN'))
    vf.atomic_json(run/'private/TASKS.json',tasks)
    vf.atomic_json(run/'public/FIT_SOURCE_MANIFEST_PUBLIC.json',dict(planned=23,runnable=sum(t['stage9_status']=='PENDING' for t in tasks),tasks=public,
        exposure='viewed development',selection='frozen source-group then EqKey order; no outcome selection',audit='finite original fit packet audit; not new clinical signoff'))
    vf.atomic_json(run/'public/RUN_LOCK.json',dict(execution_commit=args.commit,stage8_public_commit='c88307eea7b0503f472c0039f0b5b9853f9d6825',
        conditions=['F0_POS_CONT','F1_SOURCE_CE','F2_BIDIRECTIONAL_FACT_CONTRAST'],steps=160,record_step=80,seed_order=20260910,
        layer=vf.LAYER,rank=4,parameters=['u_in','v_in','u_out','v_out','rho'],parameter_count=1476,
        optimizer=dict(name='Adam',input_lr=1e-4,output_rho_lr=1e-3,betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1),
        loss=dict(positive='0.5 native CE + 0.5 rotating fit CE',H='source CE including EOS',U='0.01 unnormalized token-mean Base||student KL',pair='0.1 times bidirectional mean hinge, margin0.5; content tokens only'),
        initialization='independent clones of original CP W0 step320, optimizer empty; original seed retained',
        gpus=[0,2,3],worker_gpus=[2,3],judge_gpu=0,gpu_uuids={k:config['gpu_uuids'][k] for k in ('0','2','3')},
        wall_hours=8,total_gpu_hours=8,training_gpu_hours_limit=6.5,publication='PENDING'))
    vf.atomic_json(run/'public/RUN_STATUS.json',dict(status='PREPARED',runnable=sum(t['stage9_status']=='PENDING' for t in tasks),publication='PENDING'))
    print('ELIGIBLE',sum(t['stage9_status']=='PENDING' for t in tasks),flush=True)


def worker(args):
    run=args.run_root;config=read(run/'private/CAMPAIGN_CONFIG.json');alltasks=read(run/'private/TASKS.json')
    tasks=[t for t in alltasks if t['stage9_status']=='PENDING'];first=tasks[0]['order']
    runtime=vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(config['runtime']['cpu_gate'])))
    def budget():
        if (run/'STOP').exists() or time.time()-config['campaign_epoch']>6.5*3600:raise TimeoutError('training/endpoint budget')
    for index,t in enumerate(tasks):
        if args.part=='first' and t['order']!=first:continue
        if args.part!='first' and (t['order']==first or index%2!=int(args.part)):continue
        data=t['data'];rows={r['logical_id']:r for r in data['rows']};native=rows[t['native_id']]
        assert s.file_identity(t['checkpoint'])==t['checkpoint_identity']
        state=torch.load(t['checkpoint'],map_location='cpu',weights_only=True);assert state['step']==320
        old=read(Path(config['stage8_run'])/f"private/edits/e{t['order']:02d}/result.json")
        olditems={e['item']['row']['logical_id']:e['item'] for e in old['entries'] if e['method']=='B0'}
        assert runtime.base_guard.verify()['after_sha256']==old['base_guard']['after_sha256']
        record=vf.EditorRecord.from_dict(data['event']['edit_record'])
        schedules={g:balanced_schedule({rows[i]['source_group']:[rows[i]] for i in t[key]},160,20260910) for g,key in (('H','h_ids'),('U','u_ids'))}
        order=t['fit_positive_ids'].copy();random.Random(20260910).shuffle(order)
        batches={}
        def batch(row,answer):
            key=(row['logical_id'],answer)
            if key not in batches:
                s.input_batch(runtime,row)
                b=runtime.build_edit_batch(replace(record,question=row['question'],image_path=Path(row['image_path']),target=answer))
                if runtime.adapter.tokenizer.eos_token_id not in b.target_token_ids:raise ValueError('CE must include EOS')
                batches[key]=b
            return batches[key]
        # Base logits use the original frozen Base tokens; no new U gold or Base generation.
        teachers={}
        def teacher(row,hook):
            hook.clear_request_routing();budget()
            tokens=olditems[row['logical_id']]['base']['raw_token_ids']
            kwargs,labels,mask,binding=s.sw.teacher_batch(runtime,row,tokens)
            path=run/f"private/teacher/e{t['order']:02d}/{row['logical_id']}.pt"
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True);assert saved['binding']==binding
            else:
                with torch.no_grad():logp=runtime.model(**kwargs).logits[mask].float().log_softmax(-1).cpu()
                saved=dict(binding=binding,logp=logp);s.sw.save(path,saved)
            assert not saved['logp'].requires_grad
            return kwargs,labels,mask,saved['logp']
        for condition in (('E1','E2') if config.get('anatomical_evidence') else ('F0','F1','F2')):
            budget();directory=run/f"private/edits/e{t['order']:02d}/{condition}"
            if (directory/'result.json').exists():continue
            vf.set_seed(state['task']['seed']) if hasattr(vf,'set_seed') else torch.manual_seed(state['task']['seed'])
            expert=vf.AsymmetricCPExpert(14336,4096,4).to(runtime.device);expert.load_state_dict(state['expert']);expert.requires_grad_(True)
            optimizer=optimizer_for(expert,runtime.model);hook=vf.MedTraceLayerHook(runtime.get_module(vf.LAYER),expert);hook.attach()
            start=time.time();curve=[];startstep=0;forwards=0;forward_tokens=0
            evidence=torch.load(t['evidence_path'],map_location='cpu',weights_only=True) if config.get('anatomical_evidence') else None
            checkpoint=directory/'latest.pt'
            if checkpoint.exists():
                resume=torch.load(checkpoint,map_location=runtime.device,weights_only=True)
                assert resume['condition']==condition and resume['order']==t['order']
                expert.load_state_dict(resume['expert']);optimizer.load_state_dict(resume['optimizer']);startstep=resume['step'];curve=resume['curve'];forwards=resume['forwards']
            def forward(b):
                nonlocal forwards,forward_tokens
                hook.set_teacher_routing(b.labels);kwargs=b.forward_kwargs();out=runtime.model(**kwargs);forwards+=1
                ce=out.loss;forward_tokens+=len(b.target_token_ids)
                score=answer_score(out.logits,b.labels,b.attention_mask,runtime.adapter.tokenizer.eos_token_id,runtime.adapter.tokenizer.bos_token_id,runtime.adapter.tokenizer.pad_token_id)
                if not torch.isfinite(ce+score):raise FloatingPointError('nonfinite answer loss')
                return ce,score
            try:
                for step in range(startstep+1,161):
                    budget();optimizer.zero_grad(set_to_none=True);h=schedules['H'][step-1];u=schedules['U'][step-1]
                    ce,correct=forward(batch(native,record.target));loss=.5*ce;pairval=0.
                    if condition=='F2':
                        _,wrong=forward(batch(native,h['reference']));term=pair_half(correct,wrong);pairval+=float(term.detach());loss=loss+.1*term
                    pos=float(ce.detach())*.5;loss.backward();del loss,ce,correct
                    ce,_=forward(batch(rows[order[(step-1)%len(order)]],record.target));pos+=.5*float(ce.detach());(.5*ce).backward();del ce
                    hval=0.;klval=0.
                    if condition!='F0':
                        ce,correct=forward(batch(h,h['reference']));hval=float(ce.detach());loss=ce
                        if condition=='F2':
                            _,wrong=forward(batch(h,record.target));term=pair_half(correct,wrong);pairval+=float(term.detach());loss=loss+.1*term
                        loss.backward();del loss,ce,correct
                        if u['logical_id'] not in teachers:teachers[u['logical_id']]=teacher(u,hook)
                        kwargs,labels,mask,logp=teachers[u['logical_id']];hook.set_teacher_routing(labels)
                        logits=runtime.model(**kwargs).logits[mask];forwards+=1;forward_tokens+=int(mask.sum())
                        kl=full_vocab_kl(logits,logp);klval=float(kl.detach());(.01*kl).backward();del logits,kl
                    extra={}
                    if evidence is not None:
                        from methods.medtrace.anatomical_evidence import evidence_loss
                        ei,ci,sign=t['evidence_schedule'][step-1];entry=evidence[ei]
                        aux,extra=evidence_loss(runtime,record,entry,entry['controls'][ci],hook,1 if condition=='E1' else sign)
                        (.1*aux).backward();del aux
                        forwards+=2;forward_tokens+=extra['evidence_tokens']
                    norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
                    if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
                    optimizer.step();expert.normalize_factors_(verify_dense=False)
                    if not all(torch.isfinite(p).all() for p in expert.parameters()):raise FloatingPointError('nonfinite parameters')
                    curve.append(dict(step=step,pos_ce=pos,H_ce=hval,U_kl=klval,pair_loss=pairval,grad_norm=float(norm),elapsed_seconds=time.time()-start,
                        H_id=h['logical_id'],U_id=u['logical_id'],fit_id=order[(step-1)%len(order)],forward_tokens=forward_tokens,**extra))
                    if step%20==0:
                        payload=dict(expert=expert.state_dict(),optimizer=optimizer.state_dict(),step=step,condition=condition,order=t['order'],curve=curve,forwards=forwards,seed=state['task']['seed'])
                        s.sw.save(checkpoint,payload)
                        if step in (80,160):s.sw.save(directory/f'step{step:04d}.pt',payload)
                        print('TRAIN',t['order'],condition,step,flush=True)
            finally:hook.detach()
            expert.requires_grad_(False);entries=[]
            for lid,item in olditems.items():
                budget();row=item['row'];s.input_batch(runtime,row);forced=s.sw.generated(runtime,row,expert)
                entries.append(dict(track='A',prefix=0,edit=t['event_index'],method=condition,
                    item=dict(item,forced=forced,fixed=forced if item['fixed_on'] else item['base']),target=record.target,
                    cohort_name='FACT_'+t['cohort'],common_support=False,system_valid=True))
            if config.get('anatomical_evidence'):
                from scripts.medtrace.stage10 import diagnostics
                diagnostics(runtime,record,t,expert,directory)
            guard=runtime.base_guard.verify();assert guard['unchanged']
            vf.atomic_json(directory/'result.json',dict(status='COMPLETE',entries=entries,curve=curve,steps=160,
                original_W0_seed=state['task']['seed'],forwards=forwards,base_guard=guard,parameters=sum(p.numel() for p in expert.parameters())))
            print('DONE',t['order'],condition,flush=True)
            del expert,optimizer


def inventory(run):
    entries=[];ledger=[];curves=[];cfg=read(run/'private/CAMPAIGN_CONFIG.json')
    for t in read(run/'private/TASKS.json'):
        for condition in ('F0','F1','F2'):
            path=run/f"private/edits/e{t['order']:02d}/{condition}/result.json"
            result=read(path) if path.exists() else dict(status=t['stage9_status'],entries=[])
            entries+=result['entries'];ledger.append(dict(cohort=t['cohort'],edit=t['event_index'],condition=condition,status=result['status']))
            for row in result.get('curve',[]):curves.append(dict(cohort=t['cohort'],edit=t['event_index'],condition=condition,**{k:v for k,v in row.items() if not k.endswith('_id')}))
        if t['stage9_status']=='PENDING':
            old=read(Path(cfg['stage8_run'])/f"private/edits/e{t['order']:02d}/result.json")
            for e in old['entries']:
                if e['method'] in ('B0','B1','B2','HISTORICAL_W1'):
                    e=copy.deepcopy(e);e['cohort_name']='FACT_'+t['cohort'];entries.append(e)
    return entries,ledger,curves


def install(run):
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');old=Path(cfg['stage8_run']);previous.install(old)
    verdicts,side=f4.f3.current_verdicts(old);assert not set(side['all_expected'])-verdicts.keys()
    protocol=read(Path(cfg['runtime']['cpu_gate']).parent/'private/JUDGE_LOCK_V4.json')
    pool,execution=f4.f3.historical_judge(cfg,protocol);pool=pool.copy()
    identity=read(old/'private/judge/REUSE_EXECUTION_IDENTITY_PRIVATE.json')
    tuples=f4.f3.tuples_for(previous.inventory(old)[0],protocol['config_sha256'])
    for key,value in verdicts.items():pool[key]=(tuples[key],value,identity)
    f4.f3.inventory=inventory;f4.f3.historical_judge=lambda *_:(pool,execution)


def report(args):
    run=args.run_root;install(run);entries,ledger,curves=inventory(run);verdicts,side=f4.f3.current_verdicts(run)
    rows=f4.details(entries,verdicts,side['protocol_sha256']);tables=[r for r in f4.summarize(rows) if r['cohort'].startswith('FACT_')]
    f4.csv_write(run/'public/FACT_CONTRAST_RESULTS.csv',tables);f4.csv_write(run/'public/TRAINING_CURVES.csv',curves)
    vf.atomic_json(run/'public/COMPLETION_LEDGER.json',ledger);vf.atomic_json(run/'private/DETAILS.json',rows)
    effects=[]
    for candidate,control in (('F1','F0'),('F2','F1'),('F2','B0'),('F2','B2')):
        sub=[dict(r,method='W1' if r['method']==candidate else 'W0',mode='DEV_THRESHOLD_TRANSFER_DIAGNOSTIC') for r in rows if r['method'] in (candidate,control) and r['mode']=='FORCED_ON']
        effects.extend(dict(e,candidate=candidate,control=control,mode='FORCED_ON',control_mode='FORCED_ON') for e in pairs(sub))
    f4.csv_write(run/'public/PAIRED_EFFECTS.csv',effects)
    missing=len(set(side['all_expected'])-verdicts.keys());complete=sum(t['status']=='COMPLETE' for t in ledger);unsupported=sum(t['status']=='UNSUPPORTED_MATCHED_TRAINING_INPUT' for t in ledger)
    status=dict(status='COMPUTE_COMPLETE_SUPPORTED_SUBSET' if complete+unsupported==69 and not missing else 'PARTIAL',planned_trajectories=69,
        complete_trajectories=complete,unsupported_trajectories=unsupported,judge_required=len(side['all_expected']),judge_scored=len(verdicts),judge_missing=missing,
        judge_new=side['new'],judge_reused=side['reused'],publication='PENDING')
    vf.atomic_json(run/'public/RUN_STATUS.json',status)
    lines=['# Stage9 fact-contrast development results','',str(status),'',
        'New H source-answer supervision is shared by F1/F2; only F2-F1 isolates contrast. Historical W0/APR/B2/W1 are not equal-training controls. Unsupported old tasks are not replaced. Viewed development only; patient UNKNOWN.',
        'No automatic F3 or new facts. See fit-source manifest for permission/provenance limits. No clinical safety, independent confirmation or originality claim.',
        '','|Cohort|Method|Role|Panel|Correct macro|Damage macro|','|---|---|---|---|---:|---:|']
    for t in tables:
        if t['average']=='macro' and t['mode']=='FORCED_ON':lines.append('|'+ '|'.join(str(t[k]) for k in ('cohort','method','role','stratum','semantic','base_correct_damage'))+'|')
    vf.atomic_text(run/'public/GPT_PRO_REVIEW.md','\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','worker','prepare-judge','report'))
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--stage8-run',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--commit');p.add_argument('--part',choices=('first','0','1'))
    a=p.parse_args()
    if a.action=='prepare':prepare(a)
    elif a.action=='worker':worker(a)
    elif a.action=='report':report(a)
    else:
        install(a.run_root);side=f4.f3.prepare_judge(a);assert not side['execution_version_changed']
