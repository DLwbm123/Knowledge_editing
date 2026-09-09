import torch
from methods.medtrace.anchor_repair import repair
from methods.medtrace.bounded_repair import bounded_repair


def test_bounded_dense_kkt_and_edges():
    torch.manual_seed(8)
    A=torch.randn(2,12).float().double();B=torch.randn(8,2).double()
    X=torch.randn(12,3).double();N=torch.randn(12,4).double()
    fit=[dict(role='native',label='positive')]
    for anchors,negative in ((X,N),(torch.cat([X,X+1e-13],1),N),(torch.eye(12).double(),N),(X,torch.zeros_like(N))):
        old,_=repair(A,B,anchors,negative)
        result,g=bounded_repair(A,B,anchors,negative,old[2],fit)
        for name,V in result.items():
            C=B@(V.double()-A)
            assert float(C.norm()/(B@A).norm())<=.1*(1+1e-5)
            assert g[name]['fp32_anchor_error']<5e-5
        assert g['B3']['objective']<=g['B2']['objective']+1e-5*max(1,g['B2']['objective'])
        if g['B3']['mu']==0:torch.testing.assert_close(result['B3'],old[2])
        U,s,_=torch.linalg.svd(anchors,full_matrices=False);U=U[:,s>s[0]*1e-10]
        R=negative-U@(U.T@negative);V=result['B3'].double()-A
        kkt=(A@negative+V@negative)@R.T+(g['B3']['ridge']+g['B3']['mu'])*V
        assert float(kkt.norm())<1e-4*max(1,float((A@negative@R.T).norm()))
    for bad_rows,bad_N in (([dict(role='evaluation',label='positive')],N),(fit,N[:,:0])):
        try:bounded_repair(A,B,X,bad_N,A,bad_rows)
        except ValueError:pass
        else:raise AssertionError('invalid fitting input accepted')
