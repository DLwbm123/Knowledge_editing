import torch

from easyeditor.models.engram.solver import SolverConfig, compute_projector
from easyeditor.models.engram_v2 import SRTRSolverConfig, normalized_negative_gradient, solve_sr_tr_delta


def test_paper_origin_projector_matches_direct_formula_fp64():
    x_target = torch.tensor([[1.0, 0.8], [0.0, 0.1]], dtype=torch.float64)
    x_reference = torch.tensor([[0.0, 0.1], [1.0, 0.9]], dtype=torch.float64)
    plus = x_target @ x_target.T
    minus = x_reference @ x_reference.T
    projector, _ = compute_projector(plus, minus, config=SolverConfig(method="pinv", rcond=1.0e-12, solve_device="cpu"))
    expected = plus @ torch.linalg.pinv(plus + minus, rcond=1.0e-12)
    assert torch.allclose(projector.double(), expected, atol=1.0e-6, rtol=1.0e-6)


def test_sr_tr_closed_form_matches_direct_inverse_and_fits_target():
    torch.manual_seed(4)
    weight = torch.randn(3, 4, dtype=torch.float64)
    x_target = torch.tensor([[1.0, 0.8], [0.0, 0.1], [0.2, 0.0], [0.0, 0.0]], dtype=torch.float64)
    x_reference = torch.tensor([[0.0, 0.0], [1.0, 0.8], [0.0, 0.1], [0.2, 1.0]], dtype=torch.float64)
    residual = torch.tensor([[0.4, 0.3], [-0.2, -0.1], [0.1, 0.2]], dtype=torch.float64)
    config = SRTRSolverConfig(beta_ref=1.0, beta_old=0.0, ridge_relative=1.0e-3, max_relative_weight_norm=10.0)
    delta, stats = solve_sr_tr_delta(weight, x_target, residual, x_reference=x_reference, config=config)
    covariance = x_target @ x_target.T + x_reference @ x_reference.T + stats["ridge"] * torch.eye(4, dtype=torch.float64)
    expected = residual @ x_target.T @ torch.linalg.inv(covariance)
    assert torch.allclose(delta.double(), expected, atol=2.0e-5, rtol=2.0e-5)
    assert stats["target_residual_error_after"] < stats["target_residual_error_before"]
    assert stats["reference_effect_norm"] < stats["target_residual_error_before"]


def test_old_factor_penalty_reduces_new_delta_effect_on_old_subspace():
    weight = torch.eye(2, dtype=torch.float64)
    x_target = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    x_old = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    residual = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    unconstrained, _ = solve_sr_tr_delta(
        weight, x_target, residual,
        config=SRTRSolverConfig(beta_ref=0.0, beta_old=0.0, ridge_relative=1.0e-3, max_relative_weight_norm=10.0),
    )
    protected, stats = solve_sr_tr_delta(
        weight, x_target, residual, x_old=x_old,
        config=SRTRSolverConfig(beta_ref=0.0, beta_old=10.0, ridge_relative=1.0e-3, max_relative_weight_norm=10.0),
    )
    assert torch.linalg.norm(protected.double() @ x_old) < torch.linalg.norm(unconstrained.double() @ x_old)
    assert stats["old_effect_norm"] >= 0.0


def test_factor_concatenation_equals_streaming_covariance_accumulation():
    torch.manual_seed(7)
    batches = [torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)]
    streaming = sum((batch @ batch.T for batch in batches), torch.zeros(5, 5, dtype=torch.float64))
    concatenated = torch.cat(batches, dim=1)
    assert torch.allclose(streaming, concatenated @ concatenated.T, atol=1.0e-12, rtol=0.0)


def test_normalized_negative_gradient_is_descent_direction():
    gradient = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    residual, norm = normalized_negative_gradient(gradient)
    assert norm == 5.0
    assert torch.allclose(residual, torch.tensor([[-0.6, -0.8]], dtype=torch.float64))
    assert float((gradient * residual).sum()) < 0.0
