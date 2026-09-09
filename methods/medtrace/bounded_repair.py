"""Fixed eta=.10 anchor-preserving trust-region repair; no evaluation inputs."""
import time
import torch
from .anchor_repair import select_fit


def bounded_repair(A,B,X,N,apr,fit_rows):
    select_fit(fit_rows)  # Reject evaluation/calibration at the fitting boundary.
    started=time.time()
    A,B,X,N,apr=[v.detach().cpu().double() for v in (A,B,X,N,apr)]
    if not X.shape[1] or not N.shape[1]:raise ValueError('missing original fit activations')
    if not all(torch.isfinite(v).all() for v in (A,B,X,N,apr)):raise ValueError('nonfinite input')
    H=B.T@B
    def norm(V):
        products=H*(V@V.T);value=products.sum()
        if not torch.isfinite(value) or value < -1e-12*products.abs().sum().clamp_min(1):raise ValueError('invalid effective norm')
        return value.clamp_min(0).sqrt()
    U,s,_=torch.linalg.svd(X,full_matrices=False)
    rank=int((s>s[0]*1e-10).sum()) if s[0]>0 else 0
    U=U[:,:rank];R=N-U@(U.T@N)
    ridge=.01*max(float(N.square().sum()/N.shape[1]),1e-12)
    radius=.10*float(norm(A));full=apr-A
    alpha=min(1.,radius/max(float(norm(full)),1e-12)) if norm(full)>0 else 1.
    damped=alpha*full
    gram=R.T@R;eye=torch.eye(N.shape[1],dtype=torch.double)
    def delta(mu):return -torch.linalg.solve(gram+(ridge+mu)*eye,(A@N).T).T@R.T
    zero=delta(0.)
    # Reconstruct the FP64 optimum from frozen activations; retain deployed APR if feasible.
    torch.testing.assert_close((A+zero).float(),apr.float(),rtol=1e-5,atol=1e-6)
    mu=0.;iterations=0
    if norm(zero)<=radius and norm(full)<=radius:
        tr=full
    else:
        hi=max(ridge,1e-12)
        for _ in range(64):
            if norm(delta(hi))<=radius:break
            hi*=2
        else:raise ValueError('could not bracket trust-region multiplier')
        lo=0.
        for iterations in range(1,65):
            mid=(lo+hi)/2
            if norm(delta(mid))<=radius:hi=mid
            else:lo=mid
        mu=hi;tr=delta(mu)
    den=(B@A@X).norm().clamp_min(1e-12)
    base_energy=(B@A@N).square().sum().clamp_min(1e-24)
    def objective(V):return float((B@(A+V)@N).square().sum()+ridge*norm(V).square())
    results={};geometry={}
    for name,V in (('B2',damped),('B3',tr)):
        deployed=(A+V).float();actual=deployed.double()-A
        err=float((B@V@X).norm()/den);fp32=float((B@actual@X).norm()/den)
        ratio=float(norm(actual)/norm(A).clamp_min(1e-12))
        if err>1e-8 or fp32>5e-5 or ratio>.10*(1+1e-5):raise ValueError('frozen anchor/norm tolerance exceeded')
        geometry[name]=dict(alpha=alpha if name=='B2' else None,mu=mu if name=='B3' else None,
            eta=.10,ridge=ridge,radius=radius,anchor_rank=rank,anchor_error=err,
            fp32_anchor_error=fp32,relative_repair_norm=ratio,
            negative_energy_ratio=float((B@deployed.double()@N).square().sum()/base_energy),
            objective=objective(V),input_matrix_bytes=deployed.numel()*deployed.element_size(),
            multiplier_iterations=iterations if name=='B3' else 0)
        results[name]=deployed
    if geometry['B3']['objective']>geometry['B2']['objective']+1e-5*max(1,geometry['B2']['objective']):raise ValueError('TR objective worse than feasible damping')
    for g in geometry.values():g['pair_fit_seconds']=time.time()-started
    return results,geometry
