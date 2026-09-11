"""Fixed joint-fact objective; only CP/free rank parameterization differs."""
import argparse
from dataclasses import replace
from pathlib import Path
import random
import time
import torch
from scripts.medtrace import stage9 as prior
from methods.medtrace.selective_write import LowRankExpert,optimizer_for,balanced_schedule,full_vocab_kl
s,vf,read=prior.s,prior.vf,prior.read
CONDITIONS=('J0_CP_R4','J1_FREE_R4','J2_FREE_R16')


def make_expert(cp,condition,seed):
    return cp if condition==CONDITIONS[0] else LowRankExpert(cp,seed,rank=4 if condition in (CONDITIONS[1],'C_FACT','C_NO_H') else 16)


def worker(args):
    run=args.run_root;cfg=read(run/'private/CAMPAIGN_CONFIG.json')
    no_h=cfg.get('kind')=='MEDTRACE_STAGE12_V2'
    new_sources=cfg.get('kind')=='MEDTRACE_STAGE13R'
    endpoints=(320,) if no_h or new_sources else (160,320)
    tasks=[t for t in read(run/'private/TASKS.json') if t['stage11_status']=='PENDING']
    runtime=vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['runtime']['cpu_gate'])))
    layer=runtime.get_module(vf.LAYER);dout,din=layer.weight.shape
    assert not runtime.model.training and not any(p.requires_grad for p in runtime.model.parameters())
    def budget():
        if (run/'STOP').exists() or time.time()-cfg['campaign_epoch']>(3 if no_h and not new_sources else 6.5)*3600:raise TimeoutError('training and generation budget')
    for index,t in enumerate(tasks):
        if index%args.parts!=int(args.part):continue
        rows={r['logical_id']:r for r in t['data']['rows']};native=rows[t['native_id']]
        if new_sources:
            assert all(r['role'] in ('native','fit') for r in rows.values()), 'new-source training must not receive evaluation answers'
        state=torch.load(t['checkpoint'],map_location='cpu',weights_only=True);assert state['step']==320
        old=read(Path(cfg['stage8_run'])/('private/edits/e%02d/result.json'%t['order']))
        olditems={e['item']['row']['logical_id']:e['item'] for e in old['entries'] if e['method']=='B0' and e['item']['row']['logical_id'] not in t['quarantined_ids']}
        if new_sources:
            assert set(olditems)==set(rows) and all(v['row']['role'] in ('native','fit') for v in olditems.values())
        assert runtime.base_guard.verify()['after_sha256']==old['base_guard']['after_sha256']
        record=vf.EditorRecord.from_dict(t['data']['event']['edit_record'])
        schedule={g:balanced_schedule({rows[i]['source_group']:[rows[i]] for i in t[key]},320,20260910) for g,key in (('H','h_ids'),('U','u_ids'))}
        order=t['fit_positive_ids'].copy();random.Random(20260910).shuffle(order)
        batches={};teachers={}
        def batch(row,answer):
            key=(row['logical_id'],answer)
            if key not in batches:
                s.input_batch(runtime,row)
                batches[key]=runtime.build_edit_batch(replace(record,question=row['question'],image_path=Path(row['image_path']),target=answer))
                assert runtime.adapter.tokenizer.eos_token_id in batches[key].target_token_ids
            return batches[key]
        def teacher(row,hook):
            hook.clear_request_routing()
            kwargs,labels,mask,binding=s.sw.teacher_batch(runtime,row,olditems[row['logical_id']]['base']['raw_token_ids'])
            previous=Path(cfg['stage9_run'])/('private/teacher/e%02d'%t['order'])/(row['logical_id']+'.pt')
            path=previous if previous.exists() else run/('private/teacher/e%02d'%t['order'])/(row['logical_id']+'.pt')
            if path.exists():
                cached=torch.load(path,map_location='cpu',weights_only=True);assert cached['binding']==binding
            else:
                with torch.no_grad():logp=runtime.model(**kwargs).logits[mask].float().log_softmax(-1).cpu()
                cached=dict(binding=binding,logp=logp);s.sw.save(path,cached)
            return kwargs,labels,mask,cached['logp']
        for condition in (('C_FACT','C_NO_H') if new_sources else ('C_NO_H',) if no_h else CONDITIONS):
            if new_sources:no_h=condition=='C_NO_H'
            directory=run/('private/edits/e%02d'%t['order'])/condition
            if (directory/'result.json').exists():continue
            budget();seed=state['task']['seed']
            vf.set_seed(seed) if hasattr(vf,'set_seed') else torch.manual_seed(seed)
            cp=vf.AsymmetricCPExpert(din,dout,4).to(runtime.device);cp.load_state_dict(state['expert'])
            expert=make_expert(cp,condition,seed).to(runtime.device);expert.requires_grad_(True)
            optimizer=optimizer_for(expert,runtime.model)
            check={};start=time.time();curve=[];startstep=0;tokens=0;forwards=0;generation_seconds=0.
            # Check actual native and H activation transfer, and ordinary generation.
            if not no_h and condition!=CONDITIONS[0] and not (directory/'latest.pt').exists():
                errors=[]
                def compare(_module,inputs,_output):
                    with torch.no_grad():
                        a=cp.residual(inputs[0]);b=expert.residual(inputs[0]);errors.append(float((a-b).abs().max()))
                        if not torch.allclose(a,b,atol=5e-5,rtol=5e-5):raise ValueError('W0 conversion residual mismatch')
                handle=layer.register_forward_hook(compare)
                try:
                    for name,row in (('native',native),('H',rows[t['h_ids'][0]])):
                        s.input_batch(runtime,row);a=s.sw.generated(runtime,row,cp);b=s.sw.generated(runtime,row,expert)
                        check[name+'_token_parity']=s.same_output(a,b)
                finally:handle.remove()
                check['max_activation_residual_error']=max(errors)
                check['generation_difference_policy']='record floating-point sensitivity; no label or sample selection'
                vf.atomic_json(directory/'INITIAL_INTERFACE_CHECK.json',check)
            checkpoint=directory/'latest.pt'
            if checkpoint.exists():
                saved=torch.load(checkpoint,map_location=runtime.device,weights_only=True)
                assert saved['condition']==condition and saved['order']==t['order']
                expert.load_state_dict(saved['expert']);optimizer.load_state_dict(saved['optimizer'])
                startstep=saved['step'];curve=saved['curve'];tokens=saved['tokens'];forwards=saved['forwards']
                torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu());random.setstate(saved['python_rng'])
            hook=vf.MedTraceLayerHook(layer,expert);hook.attach()
            def forward(row,answer):
                nonlocal tokens,forwards
                b=batch(row,answer);hook.set_teacher_routing(b.labels)
                out=runtime.model(**b.forward_kwargs());tokens+=len(b.target_token_ids);forwards+=1
                assert torch.isfinite(out.loss)
                return out.loss
            def endpoint(step):
                nonlocal generation_seconds
                hook.detach();expert.requires_grad_(False);began=time.time();entries=[]
                try:
                    for item in olditems.values():
                        budget();row=item['row'];s.input_batch(runtime,row);forced=s.sw.generated(runtime,row,expert)
                        entries.append(dict(track='A',prefix=0,edit=t['event_index'],method=condition,diagnostic_step=160 if step==160 else None,
                            item=dict(item,forced=forced,fixed=forced if item['fixed_on'] else item['base']),target=record.target,
                            cohort_name='JOINT_'+t['cohort'],common_support=False,system_valid=True))
                    guard=runtime.base_guard.verify();assert guard['unchanged']
                    vf.atomic_json(directory/('endpoint%04d.json'%step),dict(status='COMPLETE',entries=entries,step=step,base_guard=guard))
                    if no_h and index==0 and not new_sources:
                        from scripts.medtrace.stage12 import replay
                        replay(runtime,run,t,expert,entries)
                finally:expert.requires_grad_(True);hook.attach()
                generation_seconds+=time.time()-began
            try:
                for step in range(startstep+1,321):
                    budget();optimizer.zero_grad(set_to_none=True);h=schedule['H'][step-1];u=schedule['U'][step-1]
                    ce=forward(native,record.target);nv=float(ce.detach());(.5*ce).backward();del ce
                    ce=forward(rows[order[(step-1)%len(order)]],record.target);pv=float(ce.detach());(.5*ce).backward();del ce
                    hv=0.
                    if not no_h:
                        ce=forward(h,h['reference']);hv=float(ce.detach());ce.backward();del ce
                    if u['logical_id'] not in teachers:teachers[u['logical_id']]=teacher(u,hook)
                    kwargs,labels,mask,logp=teachers[u['logical_id']];hook.set_teacher_routing(labels)
                    logits=runtime.model(**kwargs).logits[mask];forwards+=1;tokens+=int(mask.sum())
                    kl=full_vocab_kl(logits,logp);uv=float(kl.detach());(.01*kl).backward();del logits,kl
                    assert all(p.grad is not None for p in expert.parameters())
                    norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);assert torch.isfinite(norm)
                    optimizer.step();expert.normalize_factors_(verify_dense=False)
                    assert all(torch.isfinite(p).all() for p in expert.parameters())
                    curve.append(dict(step=step,native_ce=nv,fit_positive_ce=pv,H_ce=hv,U_kl=uv,grad_norm=float(norm),
                        elapsed_seconds=time.time()-start,cumulative_training_tokens=tokens,H_id=h['logical_id'],U_id=u['logical_id'],fit_id=order[(step-1)%len(order)]))
                    if step%20==0:
                        payload=dict(expert=expert.state_dict(),optimizer=optimizer.state_dict(),step=step,condition=condition,order=t['order'],curve=curve,tokens=tokens,forwards=forwards,
                            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate())
                        s.sw.save(checkpoint,payload)
                        print('TRAIN',t['order'],condition,step,flush=True)
                        if step in endpoints:
                            point=directory/('step%04d.pt'%step);s.sw.save(point,payload)
                            # Reload through the same deployed writer before generating.
                            expert.load_state_dict(torch.load(point,map_location=runtime.device,weights_only=True)['expert'])
                            endpoint(step)
                            torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state(payload['cuda_rng']);random.setstate(payload['python_rng'])
                # A stopped process can have saved step320 before endpoint generation.
                for step in endpoints:
                    if not (directory/('endpoint%04d.json'%step)).exists():
                        saved=torch.load(directory/('step%04d.pt'%step),map_location=runtime.device,weights_only=True);expert.load_state_dict(saved['expert']);endpoint(step)
                optbytes=sum(v.numel()*v.element_size() for state_ in optimizer.state.values() for v in state_.values() if torch.is_tensor(v))
                vf.atomic_json(directory/'result.json',dict(status='COMPLETE',steps=320,curve=curve,parameters=sum(p.numel() for p in expert.parameters()),
                    fp32_bytes=sum(p.numel()*p.element_size() for p in expert.parameters()),optimizer_tensor_bytes=optbytes,
                    checkpoint_bytes=(directory/'step0320.pt').stat().st_size,d_in=din,d_out=dout,wall_seconds=time.time()-start,
                    generation_seconds=generation_seconds,forwards=forwards,training_tokens=tokens,initial_check=check,
                    initialization='original W0; no H supervision' if no_h else 'original W0; J0 restart due to missing historical RNG',first_answer_token_rank='NA',
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            finally:hook.detach()
            print('DONE',t['order'],condition,flush=True)
            del expert,cp,optimizer
