import torch

from easyeditor.models.engram.solver import SolverConfig, compute_projector


def test_pinv_projector_prefers_target_subspace():
    sigma_plus = torch.diag(torch.tensor([3.0, 0.0]))
    sigma_minus = torch.diag(torch.tensor([0.0, 3.0]))
    projector, stats = compute_projector(
        sigma_plus,
        sigma_minus,
        config=SolverConfig(method="pinv", rcond=1.0e-6),
        num_target_vectors=3,
        num_reference_vectors=3,
    )
    assert torch.allclose(projector, torch.diag(torch.tensor([1.0, 0.0])), atol=1.0e-5)
    assert stats["rank_plus"] == 1
    assert stats["rank_total"] == 2


def test_svd_projector_supports_rank_limit():
    sigma_plus = torch.diag(torch.tensor([4.0, 1.0]))
    sigma_minus = torch.zeros(2, 2)
    projector, stats = compute_projector(
        sigma_plus,
        sigma_minus,
        config=SolverConfig(method="svd", rcond=1.0e-6, svd_rank=1),
    )
    assert stats["solver"] == "svd"
    assert stats["rank_total"] == 1
    assert projector.shape == (2, 2)

