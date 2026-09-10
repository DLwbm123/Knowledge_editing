import io
import torch
from methods.medtrace.core import AsymmetricCPExpert,MedTraceLayerHook
from methods.medtrace.selective_write import LowRankExpert,optimizer_for


def test_free_writer_transfer_gradient_and_lifecycle():
    torch.manual_seed(7)
    cp=AsymmetricCPExpert(24,12,4)
    with torch.no_grad():cp.rho.fill_(.3)
    layer=torch.nn.Linear(24,12,bias=False).requires_grad_(False)
    a=torch.randn(1,3,24);base=layer(a).detach()
    for rank in (4,16):
        free=LowRankExpert(cp,7,rank=rank)
        assert torch.allclose(cp.residual(a),free.residual(a),atol=1e-6)
        optimizer=optimizer_for(free,layer)
        assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(free.A),id(free.B)}
        for step in range(2):
            optimizer.zero_grad();free.residual(a).square().sum().backward()
            assert free.A.grad is not None and free.B.grad is not None
            if rank==16:
                assert free.B.grad[:,4:].abs().sum()>0
                if step:assert free.A.grad[4:].abs().sum()>0
            optimizer.step();before=free.residual(a).detach();free.normalize_factors_()
            assert torch.allclose(before,free.residual(a),atol=1e-6)
        stream=io.BytesIO();torch.save(free.state_dict(),stream);stream.seek(0)
        clone=LowRankExpert(cp,7,rank=rank);clone.load_state_dict(torch.load(stream,weights_only=True))
        assert torch.equal(clone.residual(a),free.residual(a))
        hook=MedTraceLayerHook(layer,clone);hook.attach()
        assert torch.equal(layer(a),base)
        hook.set_teacher_routing(torch.tensor([[-100,1,2]]));assert not torch.equal(layer(a),base)
        hook.clear_request_routing();assert torch.equal(layer(a),base)
        hook.detach();assert torch.equal(layer(a),base)
