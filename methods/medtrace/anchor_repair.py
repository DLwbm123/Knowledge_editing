"""One fixed FP64 residual repair, with FP32 rank-r deployment."""
import math
from collections import defaultdict
import torch
from torch import nn
from .core import MedTraceLayerHook


class ExpandedExpert(nn.Module):
    def __init__(self, cp, A=None):
        super().__init__()
        self.beta,self.epsilon,self.rank=cp.beta,cp.epsilon,cp.rank
        self.register_buffer('A',(cp.input_basis().T if A is None else A).detach().float().clone())
        for name in ('u_out','v_out','rho'):self.register_buffer(name,getattr(cp,name).detach().float().clone())

    def normalize_activation(self,x):
        x=x.to(self.A.dtype)
        return x/(x.square().mean(-1,keepdim=True).sqrt()+self.epsilon)

    def output_basis(self):
        return torch.einsum('ir,jr->ijr',self.u_out,self.v_out).reshape(-1,self.rank)

    def residual(self,x):
        return ((self.normalize_activation(x)@self.A.T)*self.rho)@self.output_basis().T*(self.beta/math.sqrt(self.rank))


def select_fit(rows):
    if any(r['role'] not in ('native','fit') for r in rows):raise ValueError('calibration/evaluation cannot fit APR')
    positive=[r for r in rows if r['label']=='positive']
    groups={g:defaultdict(list) for g in ('H','U')}
    for r in rows:
        if r['label']=='negative':
            if r['role']!='fit' or r.get('negative_group') not in groups:raise ValueError('invalid fit stratum')
            groups[r['negative_group']][r['source_group']].append(r)
    negative={g:[sorted(v,key=lambda r:(r['eqkey'],r['logical_id']))[0] for _,v in sorted(group.items())[:4]] for g,group in groups.items()}
    return positive,negative


def repair(A,B,X,N):
    A,B,X,N=[v.detach().cpu().double() for v in (A,B,X,N)]
    if N.shape[1]==0:raise ValueError('UNSUPPORTED_NO_NEGATIVE_FIT')
    if X.shape[1]==0:raise ValueError('missing positive anchors')
    if not all(torch.isfinite(v).all() for v in (A,B,X,N)):raise ValueError('nonfinite fitting input')
    U,s,_=torch.linalg.svd(X,full_matrices=False);rank=int((s>s[0]*1e-10).sum()) if len(s) and s[0]>0 else 0
    U=U[:,:rank];R=N-U@(U.T@N)
    ridge=.01*max(float(N.square().sum()/N.shape[1]),1e-12)
    gram1=N.T@N+ridge*torch.eye(N.shape[1],dtype=torch.float64)
    gram2=R.T@R+ridge*torch.eye(N.shape[1],dtype=torch.float64)
    A1=A-torch.linalg.solve(gram1,(A@N).T).T@N.T
    A2=A-torch.linalg.solve(gram2,(A@N).T).T@R.T
    def normmap(V):return ((B.T@B)*(V@V.T)).sum().clamp_min(0).sqrt()
    _,rs,Vh=torch.linalg.svd(R,full_matrices=False)
    rr=int((rs>rs[0]*1e-10).sum()) if len(rs) and rs[0]>0 else 0
    anchor_den=(B@A@X).norm().clamp_min(1e-12)
    constraints=float((B@(A2-A)@X).norm()/anchor_den)
    if constraints>1e-8:raise ValueError('FP64 anchor constraint violation')
    geometry=dict(anchor_columns=X.shape[1],negative_columns=N.shape[1],anchor_rank=rank,
        anchor_singular_values=s.tolist(),R_rank=rr,R_singular_values=rs.tolist(),ridge=ridge,
        gram1_condition=float(torch.linalg.cond(gram1)),gram2_condition=float(torch.linalg.cond(gram2)),
        irreducible_combination_energy=float((B@A@N@Vh[rr:].T).square().sum()),
        fp64_anchor_error=constraints)
    for name,V in (('C0',A),('C1',A1),('C2',A2)):
        geometry[name]=dict(anchor_relative_error=float((B@(V-A)@X).norm()/anchor_den),
            fp32_anchor_relative_error=float((B@(V.float().double()-A.float().double())@X).norm()/anchor_den),
            negative_energy_ratio=float((B@V@N).square().sum()/(B@A@N).square().sum().clamp_min(1e-24)),
            relative_repair_norm=float(normmap(V-A)/normmap(A).clamp_min(1e-12)))
    if geometry['C2']['fp32_anchor_relative_error']>5e-5:raise ValueError('FP32 anchor deployment error exceeds frozen tolerance')
    return (A.float(),A1.float(),A2.float()),geometry


class CaptureHook(MedTraceLayerHook):
    """Capture exact active predictor inputs; Base mode returns unmodified output."""
    def __init__(self,layer,expert,base=False):
        super().__init__(layer,expert);self.base=base;self.columns={}

    def _forward_hook(self,module,args,output):
        result=super()._forward_hook(module,args,output)
        if not self.enabled or not self.generation_routing:return result
        trace=self.generation_trace[-1]
        first=trace['first_active_predictor'];count=trace['active_predictor_count']
        values=self.expert.normalize_activation(args[0])[0,first:first+count].detach().cpu()
        start=len(self.columns) if trace['sequence_length']==1 else 0
        for j,v in enumerate(values):self.columns.setdefault(start+j,v)
        return output if self.base else result
