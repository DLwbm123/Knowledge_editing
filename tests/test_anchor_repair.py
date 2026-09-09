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
