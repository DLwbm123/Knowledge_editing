from types import SimpleNamespace
import torch
from methods.medtrace.anatomical_evidence import suppressed,pixel_view


def test_equal_area_noop_cleanup_and_loss_direction():
    p=torch.arange(3*8*8).reshape(3,8,8).float();r=torch.zeros(8,8,dtype=torch.bool);r[1:3,1:3]=True;c=torch.roll(r,4,1)
    assert r.sum()==c.sum() and not (r&c).any()
    torch.testing.assert_close(suppressed(p,torch.zeros_like(r)),p)
    assert torch.equal(suppressed(p,r)[:,~r],p[:,~r])
    original=lambda *args:dict(images=p.unsqueeze(0))
    adapter=SimpleNamespace(prepare_inputs=original);row=dict(image_path='image',question='question')
    try:
        with pixel_view(adapter,row,suppressed(p,r)):
            assert not torch.equal(adapter.prepare_inputs('image','question')['images'],p.unsqueeze(0))
            raise RuntimeError('cleanup')
    except RuntimeError:pass
    assert adapter.prepare_inputs is original
    d=torch.tensor(0.,requires_grad=True);torch.relu(.2-d).backward();assert d.grad<0
