import torch

from easyeditor.models.engram.crisp_projection import (
    apply_crisp_projection_to_delta,
    apply_crisp_projection_to_grad,
    combine_crisp_kfac_caches,
    compute_crisp_kfac_projection_cache,
)


def _diag_cache(energy_threshold=0.7):
    A = torch.diag(torch.tensor([10.0, 1.0]))
    B = torch.diag(torch.tensor([8.0, 1.0]))
    return compute_crisp_kfac_projection_cache(A, B, energy_threshold=energy_threshold)


def _vec_col(matrix):
    return matrix.t().reshape(-1)


def _unvec_col(vector, rows, cols):
    return vector.reshape(cols, rows).t()


def test_projection_preserves_delta_shape():
    cache = _diag_cache()
    delta = torch.randn(2, 2)
    projected = apply_crisp_projection_to_delta(delta, cache)
    assert projected.shape == delta.shape


def test_high_curvature_components_are_suppressed():
    cache = _diag_cache(energy_threshold=0.7)
    delta = torch.zeros(2, 2)
    delta[0, 0] = 5.0
    projected = apply_crisp_projection_to_delta(delta, cache)
    assert torch.allclose(projected, torch.zeros_like(delta), atol=1.0e-6)


def test_low_curvature_components_are_retained():
    cache = _diag_cache(energy_threshold=0.7)
    delta = torch.zeros(2, 2)
    delta[1, 1] = 5.0
    projected = apply_crisp_projection_to_delta(delta, cache)
    assert torch.allclose(projected, delta, atol=1.0e-6)


def test_small_explicit_kronecker_projector_matches_matrix_free_result():
    A = torch.tensor([[3.0, 0.5], [0.5, 1.0]])
    B = torch.tensor([[2.0, 0.25], [0.25, 0.8]])
    cache = compute_crisp_kfac_projection_cache(A, B, energy_threshold=0.6)
    delta = torch.tensor([[1.0, -2.0], [3.0, 0.5]])

    matrix_free = apply_crisp_projection_to_delta(delta, cache)

    ua = cache["Ua"].float()
    ub = cache["Ub"].float()
    mask_vec = cache["M"].float().reshape(-1)
    basis = torch.kron(ua, ub)
    explicit = basis @ torch.diag(mask_vec) @ basis.t()
    explicit_projected = _unvec_col(explicit @ _vec_col(delta.float()), *delta.shape)

    assert torch.allclose(matrix_free, explicit_projected, atol=1.0e-5)


def test_dtype_conversion_works():
    cache = compute_crisp_kfac_projection_cache(
        torch.eye(3, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        energy_threshold=0.9,
        dtype=torch.float32,
    )
    delta = torch.randn(2, 3, dtype=torch.float64)
    projected = apply_crisp_projection_to_grad(delta, cache)
    assert projected.dtype == torch.float64
    assert projected.shape == delta.shape


def test_projection_cache_uses_fallback_when_eigh_backend_fails(monkeypatch):
    def failing_eigh(_matrix):
        raise RuntimeError("simulated eigh backend failure")

    monkeypatch.setattr(torch.linalg, "eigh", failing_eigh)

    cache = compute_crisp_kfac_projection_cache(torch.eye(3), torch.eye(2), energy_threshold=0.9)
    delta = torch.randn(2, 3)
    projected = apply_crisp_projection_to_delta(delta, cache)

    assert projected.shape == delta.shape
    assert torch.isfinite(projected).all()
    assert cache["metadata"]["A_decomposition_backend"] in {
        "scipy.linalg.eigh.evr_fallback",
        "torch.linalg.eigh.float64_fallback",
        "torch.linalg.svd.float32_fallback",
    }
    assert cache["metadata"]["B_decomposition_backend"] in {
        "scipy.linalg.eigh.evr_fallback",
        "torch.linalg.eigh.float64_fallback",
        "torch.linalg.svd.float32_fallback",
    }
    assert "simulated eigh backend failure" in cache["metadata"]["A_decomposition_error"]


def test_no_nan_inf_and_cache_combination():
    cache = _diag_cache()
    delta = torch.randn(2, 2)
    projected = apply_crisp_projection_to_delta(delta, cache)
    assert torch.isfinite(projected).all()
    assert torch.isfinite(cache["Ua"]).all()
    assert torch.isfinite(cache["Ub"]).all()

    combined = combine_crisp_kfac_caches(
        [
            {"A": torch.eye(2), "B": torch.eye(2) * 2.0, "num_samples": 2},
            {"A": torch.eye(2) * 3.0, "B": torch.eye(2) * 4.0, "num_samples": 1},
        ]
    )
    assert torch.allclose(combined["A"], torch.eye(2) * (5.0 / 3.0))
    assert torch.allclose(combined["B"], torch.eye(2) * (8.0 / 3.0))
    assert combined["num_samples"] == 3.0
