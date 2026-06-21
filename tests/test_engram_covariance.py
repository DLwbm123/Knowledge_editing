import torch

from easyeditor.models.engram.covariance import covariance_from_activations, flatten_activation_rows


def test_covariance_handles_2d_3d_and_nested_inputs():
    x2 = torch.eye(3)
    stat2 = covariance_from_activations(x2, input_dim=3)
    assert stat2.count == 3
    assert stat2.cov.shape == (3, 3)

    x3 = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
    rows, warning = flatten_activation_rows(((x3,),), input_dim=3, mask=mask)
    assert warning is None
    assert rows.shape == (4, 3)


def test_covariance_bias_absorption_adds_constant_dimension():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    stat = covariance_from_activations(x, input_dim=2, absorb_bias=True)
    assert stat.cov.shape == (3, 3)
    assert stat.count == 2
    assert torch.allclose(stat.cov[-1, -1], torch.tensor(2.0))


def test_covariance_mask_fallback_to_all_when_unaligned():
    x = torch.randn(2, 3, 4)
    bad_mask = torch.ones(5, dtype=torch.bool)
    rows, warning = flatten_activation_rows(x, input_dim=4, mask=bad_mask, mask_fallback="all")
    assert rows.shape == (6, 4)
    assert "falling back" in warning

