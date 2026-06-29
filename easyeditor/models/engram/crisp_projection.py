from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch


def _as_fp32_square_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square 2D tensor, got shape {tuple(value.shape)}")
    matrix = value.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or inf")
    return (0.5 * (matrix + matrix.t())).contiguous()


def _short_error(error: BaseException, max_chars: int = 500) -> str:
    message = str(error).replace("\n", " ")
    return message[:max_chars]


def _safe_symmetric_decomposition(matrix: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor, str, Optional[str]]:
    """Return symmetric eigenspaces, with SVD fallback for CPU LAPACK/MKL failures."""

    try:
        values, vectors = torch.linalg.eigh(matrix)
        return values.float(), vectors.float(), "torch.linalg.eigh.float32", None
    except RuntimeError as first_error:
        first_message = _short_error(first_error)

    try:
        import scipy.linalg  # type: ignore[import-not-found]

        values_np, vectors_np = scipy.linalg.eigh(
            matrix.numpy(),
            driver="evr",
            check_finite=False,
            overwrite_a=False,
        )
        values = torch.from_numpy(values_np).float()
        vectors = torch.from_numpy(vectors_np).float()
        return values, vectors, "scipy.linalg.eigh.evr_fallback", first_message
    except Exception as scipy_error:  # pragma: no cover - depends on optional scipy/LAPACK backend
        scipy_message = _short_error(scipy_error)

    try:
        values64, vectors64 = torch.linalg.eigh(matrix.double())
        return values64.float(), vectors64.float(), "torch.linalg.eigh.float64_fallback", first_message
    except RuntimeError as second_error:
        second_message = _short_error(second_error)

    try:
        vectors, values, _ = torch.linalg.svd(matrix, full_matrices=False)
        return (
            values.float(),
            vectors.float(),
            "torch.linalg.svd.float32_fallback",
            f"{first_message}; {scipy_message}; {second_message}",
        )
    except RuntimeError as third_error:
        raise RuntimeError(
            f"{name} symmetric decomposition failed with eigh and svd fallbacks: "
            f"{first_message}; {scipy_message}; {second_message}; {_short_error(third_error)}"
        ) from third_error


def _rank_threshold_by_energy(values: torch.Tensor, energy_threshold: float) -> tuple[int, torch.Tensor]:
    if not 0.0 < float(energy_threshold) <= 1.0:
        raise ValueError(f"energy_threshold must be in (0, 1], got {energy_threshold}")
    flat = values.reshape(-1).to(dtype=torch.float32)
    if flat.numel() == 0:
        raise ValueError("empty eigenvalue product vector")
    if not torch.isfinite(flat).all():
        raise ValueError("eigenvalue product vector contains NaN or inf")

    nonnegative = flat.clamp_min(0.0)
    total = nonnegative.sum()
    if total <= 0:
        return 0, torch.tensor(float("inf"), dtype=torch.float32)

    sorted_values, _ = torch.sort(nonnegative, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) / total
    rank = int(torch.searchsorted(cumulative, torch.tensor(float(energy_threshold))).item()) + 1
    rank = max(1, min(rank, int(sorted_values.numel())))
    return rank, sorted_values[rank - 1]


def _cache_metadata(
    *,
    A_shape: Iterable[int],
    B_shape: Iterable[int],
    Sa: torch.Tensor,
    Sb: torch.Tensor,
    M: torch.Tensor,
    energy_threshold: float,
    rank: int,
    threshold: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    A_backend: str,
    B_backend: str,
    A_decomposition_error: Optional[str],
    B_decomposition_error: Optional[str],
) -> Dict[str, Any]:
    return {
        "A_shape": list(A_shape),
        "B_shape": list(B_shape),
        "mask_shape": list(M.shape),
        "energy_threshold": float(energy_threshold),
        "keep_ratio": float(M.float().mean().item()),
        "rank": int(rank),
        "threshold": None if torch.isinf(threshold) else float(threshold.item()),
        "A_eig_min": float(Sa.min().item()),
        "A_eig_max": float(Sa.max().item()),
        "B_eig_min": float(Sb.min().item()),
        "B_eig_max": float(Sb.max().item()),
        "cache_device": str(device),
        "cache_dtype": str(dtype).replace("torch.", ""),
        "A_decomposition_backend": A_backend,
        "B_decomposition_backend": B_backend,
        "A_decomposition_error": A_decomposition_error,
        "B_decomposition_error": B_decomposition_error,
    }


def build_crisp_kfac_projection_cache_from_decomposition(
    *,
    A_shape: Iterable[int],
    B_shape: Iterable[int],
    Sa: torch.Tensor,
    Ua: torch.Tensor,
    Sb: torch.Tensor,
    Ub: torch.Tensor,
    energy_threshold: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    A_backend: str = "precomputed",
    B_backend: str = "precomputed",
    A_decomposition_error: Optional[str] = None,
    B_decomposition_error: Optional[str] = None,
) -> Dict[str, Any]:
    Sa = Sa.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    Sb = Sb.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    Ua = Ua.detach().to(device="cpu", dtype=torch.float32)
    Ub = Ub.detach().to(device="cpu", dtype=torch.float32)

    products = torch.outer(Sa.clamp_min(0.0), Sb.clamp_min(0.0))
    rank, threshold = _rank_threshold_by_energy(products, float(energy_threshold))
    M = products < threshold

    if not torch.isfinite(Ua).all() or not torch.isfinite(Ub).all():
        raise ValueError("projection eigenspaces contain NaN or inf")

    target_device = torch.device(device)
    cache: Dict[str, Any] = {
        "Ua": Ua.to(device=target_device, dtype=dtype),
        "Ub": Ub.to(device=target_device, dtype=dtype),
        "M": M.to(device=target_device),
        "Sa": Sa.cpu(),
        "Sb": Sb.cpu(),
        "metadata": _cache_metadata(
            A_shape=A_shape,
            B_shape=B_shape,
            Sa=Sa,
            Sb=Sb,
            M=M,
            energy_threshold=float(energy_threshold),
            rank=rank,
            threshold=threshold,
            device=str(device),
            dtype=dtype,
            A_backend=A_backend,
            B_backend=B_backend,
            A_decomposition_error=A_decomposition_error,
            B_decomposition_error=B_decomposition_error,
        ),
    }
    return cache


def compute_crisp_kfac_projection_cache(
    A: torch.Tensor,
    B: torch.Tensor,
    energy_threshold: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Build a CrispEdit-style low-curvature projection cache from K-FAC factors.

    Adapted from CrispEdit's `calculate_projection_cache_with_kfac`.
    The cache is model-wrapper-independent and stores eigenspaces plus the
    low-curvature mask used by the matrix-free projection formula.
    """

    A_fp32 = _as_fp32_square_matrix(A, "A")
    B_fp32 = _as_fp32_square_matrix(B, "B")
    Sa, Ua, A_backend, A_error = _safe_symmetric_decomposition(A_fp32, "A")
    Sb, Ub, B_backend, B_error = _safe_symmetric_decomposition(B_fp32, "B")
    return build_crisp_kfac_projection_cache_from_decomposition(
        A_shape=A_fp32.shape,
        B_shape=B_fp32.shape,
        Sa=Sa,
        Ua=Ua,
        Sb=Sb,
        Ub=Ub,
        energy_threshold=energy_threshold,
        device=device,
        dtype=dtype,
        A_backend=A_backend,
        B_backend=B_backend,
        A_decomposition_error=A_error,
        B_decomposition_error=B_error,
    )


def _apply_projection(matrix: torch.Tensor, projection_cache: Dict[str, Any], name: str) -> torch.Tensor:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got shape {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or inf")

    ua = projection_cache["Ua"].to(device=matrix.device, dtype=matrix.dtype)
    ub = projection_cache["Ub"].to(device=matrix.device, dtype=matrix.dtype)
    mask = projection_cache["M"].to(device=matrix.device, dtype=matrix.dtype)
    expected_shape = (ub.shape[0], ua.shape[0])
    if tuple(matrix.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} shape {tuple(matrix.shape)} does not match projection cache "
            f"shape {tuple(expected_shape)}"
        )

    projected = ub @ ((ub.t() @ matrix @ ua) * mask.t()) @ ua.t()
    if not torch.isfinite(projected).all():
        raise ValueError(f"projected {name} contains NaN or inf")
    return projected.to(dtype=matrix.dtype)


def apply_crisp_projection_to_delta(delta: torch.Tensor, projection_cache: Dict[str, Any]) -> torch.Tensor:
    """Apply CrispEdit's matrix-free low-curvature projection to a dense delta."""

    return _apply_projection(delta, projection_cache, "delta")


def apply_crisp_projection_to_grad(grad: torch.Tensor, projection_cache: Dict[str, Any]) -> torch.Tensor:
    """Apply CrispEdit's matrix-free low-curvature projection to a gradient."""

    return _apply_projection(grad, projection_cache, "grad")


def combine_crisp_kfac_caches(
    caches: List[Dict[str, Any]],
    normalize_trace_with_first: bool = False,
) -> Dict[str, Any]:
    """Combine K-FAC caches by sample-weighted average.

    This ports CrispEdit's `combine_layer_to_cov_caches` behavior for a single
    model-independent layer cache. If `normalize_trace_with_first` is enabled,
    later caches are trace-matched to the first cache before averaging.
    """

    if not caches:
        raise ValueError("caches must be non-empty")

    total_samples = 0.0
    combined_A: Optional[torch.Tensor] = None
    combined_B: Optional[torch.Tensor] = None
    first_A_trace: Optional[torch.Tensor] = None
    first_B_trace: Optional[torch.Tensor] = None
    metadata_rows: List[Dict[str, Any]] = []

    for idx, cache in enumerate(caches):
        if not isinstance(cache, dict):
            raise TypeError(f"cache {idx} must be a dict")
        A = _as_fp32_square_matrix(cache["A"], f"caches[{idx}]['A']")
        B = _as_fp32_square_matrix(cache["B"], f"caches[{idx}]['B']")
        if idx == 0:
            first_A_trace = torch.trace(A).abs().clamp_min(1.0e-12)
            first_B_trace = torch.trace(B).abs().clamp_min(1.0e-12)
        elif normalize_trace_with_first:
            assert first_A_trace is not None and first_B_trace is not None
            A_trace = torch.trace(A).abs().clamp_min(1.0e-12)
            B_trace = torch.trace(B).abs().clamp_min(1.0e-12)
            A = A * (first_A_trace / A_trace)
            B = B * (first_B_trace / B_trace)

        if combined_A is not None and tuple(A.shape) != tuple(combined_A.shape):
            raise ValueError(f"A shape mismatch at cache {idx}: {tuple(A.shape)} vs {tuple(combined_A.shape)}")
        if combined_B is not None and tuple(B.shape) != tuple(combined_B.shape):
            raise ValueError(f"B shape mismatch at cache {idx}: {tuple(B.shape)} vs {tuple(combined_B.shape)}")

        num_samples = float(cache.get("num_samples", 1.0))
        if num_samples <= 0:
            raise ValueError(f"cache {idx} num_samples must be positive, got {num_samples}")
        total_samples += num_samples
        combined_A = A * num_samples if combined_A is None else combined_A + A * num_samples
        combined_B = B * num_samples if combined_B is None else combined_B + B * num_samples
        metadata_rows.append(
            {
                "cache_index": idx,
                "num_samples": num_samples,
                "A_shape": list(A.shape),
                "B_shape": list(B.shape),
                "A_trace": float(torch.trace(A).item()),
                "B_trace": float(torch.trace(B).item()),
            }
        )

    assert combined_A is not None and combined_B is not None
    combined_A = combined_A / total_samples
    combined_B = combined_B / total_samples
    return {
        "A": combined_A.cpu(),
        "B": combined_B.cpu(),
        "num_samples": total_samples,
        "metadata": {
            "source_cache_count": len(caches),
            "normalize_trace_with_first": bool(normalize_trace_with_first),
            "sources": metadata_rows,
        },
    }


def combine_layer_to_crisp_kfac_caches(
    layer_to_caches: Iterable[Dict[str, Dict[str, Any]]],
    normalize_trace_with_first: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Combine multiple layer-name-to-cache mappings.

    This helper mirrors CrispEdit's layer-level cache combination while reusing
    the single-layer `combine_crisp_kfac_caches` implementation above.
    """

    cache_maps = list(layer_to_caches)
    if not cache_maps:
        raise ValueError("layer_to_caches must be non-empty")
    layer_names = set(cache_maps[0])
    for idx, cache_map in enumerate(cache_maps[1:], start=1):
        if set(cache_map) != layer_names:
            raise ValueError(f"layer set mismatch at cache map {idx}")
    return {
        layer_name: combine_crisp_kfac_caches(
            [cache_map[layer_name] for cache_map in cache_maps],
            normalize_trace_with_first=normalize_trace_with_first,
        )
        for layer_name in sorted(layer_names)
    }
