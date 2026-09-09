import io
import torch
from methods.medtrace.core import AsymmetricCPExpert,MedTraceLayerHook
from methods.medtrace.anchor_repair import ExpandedExpert,repair,select_fit


def test_repair_dense_and_edges():
    torch.manual_seed(9)
    cp=AsymmetricCPExpert(12,8,2,beta=.7);cp.rho.data.fill_(.4)
    a=torch.randn(9,12);expanded=ExpandedExpert(cp)
    torch.testing.assert_close(cp.residual(a),expanded.residual(a))
    A=expanded.A.double();B=(cp.output_basis()*cp.rho*(cp.beta/2**.5)).double()
    X=torch.randn(12,3,dtype=torch.double);N=torch.randn(12,4,dtype=torch.double)
    matrices,g=repair(A,B,X,N);U=torch.linalg.svd(X,full_matrices=False)[0]
    R=N-U@(U.T@N);ridge=g['ridge'];D=B@A
    dense=D-D@N@torch.linalg.solve(R.T@R+ridge*torch.eye(4),R.T)
    torch.testing.assert_close(B@matrices[2].double(),dense,rtol=1e-5,atol=1e-6)
    assert g['C2']['anchor_relative_error']<1e-8 and g['C1']['anchor_relative_error']>1e-4
    delta=matrices[2].double()-A
    torch.testing.assert_close(delta@U,torch.zeros_like(delta@U),atol=1e-6,rtol=0)
    residual=(A@N+delta@N)@R.T+ridge*delta
    assert residual.norm()<1e-5
    # Repeated/near-collinear columns and exact zero-R case.
    repair(A,B,torch.cat([X,X,X+1e-13],1),N)
    _,z=repair(A,B,torch.eye(12,dtype=torch.double),N)
    assert z['R_rank']==0 and z['C2']['negative_energy_ratio']==1
    try:repair(A,B,X,N[:,:0])
    except ValueError:pass
    else:assert False
    row=dict(role='fit',label='negative',negative_group='U',source_group='g',eqkey='q',logical_id='q')
    _,groups=select_fit([row]);assert not groups['H'] and len(groups['U'])==1
    try:select_fit([dict(row,role='evaluation')])
    except ValueError:pass
    else:assert False
    buf=io.BytesIO();torch.save(expanded.state_dict(),buf);buf.seek(0)
    loaded=ExpandedExpert(cp);loaded.load_state_dict(torch.load(buf,weights_only=True))
    torch.testing.assert_close(loaded.residual(a),expanded.residual(a))
    layer=torch.nn.Linear(12,8);x=a[None];before=layer(x)
    hook=MedTraceLayerHook(layer,loaded);hook.attach()
    try:
        with hook.generation_request():
            layer(x);raise RuntimeError('cleanup probe')
    except RuntimeError:pass
    torch.testing.assert_close(layer(x),before);hook.detach()


def test_legacy_fit_binding(tmp_path):
    import json
    from types import SimpleNamespace
    from scripts.medtrace.stage7 import input_batch,vf
    batch=SimpleNamespace(image_sha256='image',raw_input_ids=torch.tensor([[1,2,3]]),
        attention_mask=torch.ones(1,3,dtype=torch.long),key_token_index=2,image_token_start=0,image_token_end=1)
    locks={'source_version':'frozen'}
    payload=dict(image_tensor_sha256='image',target_free_prompt_tokens=[1,2,3],attention_mask=[1,1,1],
        assistant_boundary_index=2,image_token_span=[0,1],**locks)
    row=dict(question='q',image_path='image',reference='a',role='fit',label='negative',
        eqkey=vf.sha256_json(payload),panel='matched',support_source=str(tmp_path/'source.json'))
    event=dict(edit_record=dict(record_id='id',dataset='test',question='q',image_path='image',gold_answer='a',
        official_rephrase='',relative_image_path='image',formal_sequence_position=0,question_type='test'))
    (tmp_path/'source.json').write_text(json.dumps(dict(rows=[row],event=event,cache_locks={'matched':locks})))
    runtime=SimpleNamespace(generation_config={},build_question_batch=lambda *a,**kw:batch)
    assert input_batch(runtime,row) is batch
    batch.raw_input_ids=torch.tensor([[1,2,4]])
    try:input_batch(runtime,row)
    except ValueError:pass
    else:raise AssertionError('legacy binding must reject changed actual tokens')
