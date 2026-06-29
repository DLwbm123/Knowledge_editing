from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .crisp_projection import compute_crisp_kfac_projection_cache


LossFn = Callable[[nn.Module, Any], torch.Tensor]


@dataclass
class _ModuleAccumulator:
    module_name: str
    in_features: int
    out_features: int
    A_sum: torch.Tensor
    B_sum: torch.Tensor
    num_activation_vectors: int = 0
    num_gradient_vectors: int = 0
    forward_hook_calls: int = 0
    gradient_hook_calls: int = 0

    @classmethod
    def create(cls, module_name: str, module: nn.Linear) -> "_ModuleAccumulator":
        return cls(
            module_name=module_name,
            in_features=int(module.in_features),
            out_features=int(module.out_features),
            A_sum=torch.zeros(int(module.in_features), int(module.in_features), dtype=torch.float32),
            B_sum=torch.zeros(int(module.out_features), int(module.out_features), dtype=torch.float32),
        )

    def add_activation(self, value: torch.Tensor) -> None:
        flat = _flatten_last_dim(value.detach(), self.in_features, f"{self.module_name} input")
        self.A_sum += flat.t().matmul(flat).cpu()
        self.num_activation_vectors += int(flat.shape[0])
        self.forward_hook_calls += 1

    def add_gradient(self, value: torch.Tensor) -> None:
        flat = _flatten_last_dim(value.detach(), self.out_features, f"{self.module_name} output grad")
        self.B_sum += flat.t().matmul(flat).cpu()
        self.num_gradient_vectors += int(flat.shape[0])
        self.gradient_hook_calls += 1

    def to_cache(self) -> Dict[str, Any]:
        if self.num_activation_vectors <= 0:
            raise RuntimeError(f"No activation vectors collected for {self.module_name}")
        if self.num_gradient_vectors <= 0:
            raise RuntimeError(f"No gradient vectors collected for {self.module_name}")
        A = self.A_sum / float(self.num_activation_vectors)
        B = self.B_sum / float(self.num_gradient_vectors)
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise RuntimeError(f"Non-finite K-FAC cache for {self.module_name}")
        return {
            "A": A.cpu(),
            "B": B.cpu(),
            "num_samples": min(self.num_activation_vectors, self.num_gradient_vectors),
            "num_activation_vectors": self.num_activation_vectors,
            "num_gradient_vectors": self.num_gradient_vectors,
            "forward_hook_calls": self.forward_hook_calls,
            "gradient_hook_calls": self.gradient_hook_calls,
        }


def _flatten_last_dim(value: torch.Tensor, expected_dim: int, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if value.shape[-1] != expected_dim:
        raise ValueError(f"{label} last dim {value.shape[-1]} != expected {expected_dim}")
    flat = value.reshape(-1, expected_dim).to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(flat).all():
        raise ValueError(f"{label} contains NaN or inf")
    return flat


def _module_map(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def select_crisp_kfac_linear_modules(
    model: nn.Module,
    module_patterns: Iterable[str],
    exclude_module_patterns: Optional[Iterable[str]] = None,
) -> List[str]:
    patterns = [re.compile(pattern) for pattern in module_patterns]
    excludes = [re.compile(pattern) for pattern in (exclude_module_patterns or [])]
    selected: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pattern.search(name) for pattern in patterns) and not any(pattern.search(name) for pattern in excludes):
            selected.append(name)
    return selected


def _diagnostic_from_cache(
    module_name: str,
    cache: Optional[Dict[str, Any]],
    *,
    skipped: bool,
    skip_reason: Optional[str],
    energy_threshold: Optional[float],
    projection_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if cache is None:
        return {
            "module_name": module_name,
            "skipped": bool(skipped),
            "skip_reason": skip_reason,
            "energy_threshold": energy_threshold,
        }
    A = cache["A"]
    B = cache["B"]
    metadata = dict((projection_cache or {}).get("metadata") or {})
    return {
        "module_name": module_name,
        "A_shape": list(A.shape),
        "B_shape": list(B.shape),
        "A_rank": int(torch.linalg.matrix_rank(A).item()),
        "B_rank": int(torch.linalg.matrix_rank(B).item()),
        "A_trace": float(torch.trace(A).item()),
        "B_trace": float(torch.trace(B).item()),
        "num_activation_vectors": int(cache.get("num_activation_vectors", 0)),
        "num_gradient_vectors": int(cache.get("num_gradient_vectors", 0)),
        "forward_hook_calls": int(cache.get("forward_hook_calls", 0)),
        "gradient_hook_calls": int(cache.get("gradient_hook_calls", 0)),
        "energy_threshold": energy_threshold,
        "mask_keep_ratio": metadata.get("keep_ratio"),
        "cache_device": "cpu",
        "cache_dtype": "float32",
        "skipped": bool(skipped),
        "skip_reason": skip_reason,
    }


def collect_crisp_kfac_caches(
    model: nn.Module,
    module_names: Iterable[str],
    samples: Iterable[Any],
    loss_fn: LossFn,
    *,
    max_dim: int = 4097,
    energy_threshold: Optional[float] = None,
    build_projection_cache: bool = False,
    projection_device: str = "cpu",
    projection_dtype: torch.dtype = torch.float32,
    clear_cuda_cache: bool = False,
) -> Dict[str, Any]:
    """Collect MLLM-compatible K-FAC caches for selected Linear modules.

    The collector is intentionally wrapper-independent: callers provide samples
    and a loss function that runs the actual image-text model path. Parameters
    are not optimized. Module outputs are retained only to collect output
    gradients for the K-FAC B factor.
    """

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("samples must be non-empty")

    modules = _module_map(model)
    requested = list(module_names)
    accumulators: Dict[str, _ModuleAccumulator] = {}
    diagnostics: List[Dict[str, Any]] = []
    hooks: List[Any] = []
    skipped: Dict[str, str] = {}

    for name in requested:
        module = modules.get(name)
        if module is None:
            skipped[name] = "module_not_found"
            diagnostics.append(
                _diagnostic_from_cache(name, None, skipped=True, skip_reason=skipped[name], energy_threshold=energy_threshold)
            )
            continue
        if not isinstance(module, nn.Linear):
            skipped[name] = "module_not_linear"
            diagnostics.append(
                _diagnostic_from_cache(name, None, skipped=True, skip_reason=skipped[name], energy_threshold=energy_threshold)
            )
            continue
        if int(module.in_features) > int(max_dim) or int(module.out_features) > int(max_dim):
            skipped[name] = f"dim_larger_than_max_dim={max_dim}"
            diagnostics.append(
                _diagnostic_from_cache(name, None, skipped=True, skip_reason=skipped[name], energy_threshold=energy_threshold)
            )
            continue
        accumulators[name] = _ModuleAccumulator.create(name, module)

    if not accumulators:
        return {
            "layer_to_cache": {},
            "layer_to_projection_cache": {},
            "diagnostics": diagnostics,
            "sample_count": len(sample_list),
            "skipped_modules": skipped,
        }

    def make_hook(module_name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any):
            if not inputs:
                raise RuntimeError(f"No input tuple for {module_name}")
            accumulator = accumulators[module_name]
            accumulator.add_activation(inputs[0])
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"Expected tensor output for {module_name}, got {type(output)}")
            retained = output if output.requires_grad else output.detach().requires_grad_(True)

            def grad_hook(grad: torch.Tensor) -> torch.Tensor:
                accumulator.add_gradient(grad)
                return grad

            retained.register_hook(grad_hook)
            return retained

        return hook

    training_state = bool(model.training)
    requires_grad_state = [(param, bool(param.requires_grad)) for param in model.parameters()]
    try:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        for name in accumulators:
            hooks.append(modules[name].register_forward_hook(make_hook(name)))

        sample_count = 0
        for sample in sample_list:
            sample_count += 1
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss = loss_fn(model, sample)
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(f"loss_fn must return a tensor, got {type(loss)}")
                if loss.ndim != 0:
                    loss = loss.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite K-FAC loss at sample {sample_count}: {loss}")
                loss.backward()
    finally:
        for handle in hooks:
            handle.remove()
        for param, value in requires_grad_state:
            param.requires_grad_(value)
        model.train(training_state)
        model.zero_grad(set_to_none=True)
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_to_cache: Dict[str, Dict[str, Any]] = {}
    layer_to_projection_cache: Dict[str, Dict[str, Any]] = {}
    for name, accumulator in accumulators.items():
        cache = accumulator.to_cache()
        layer_to_cache[name] = cache
        projection_cache = None
        if build_projection_cache:
            if energy_threshold is None:
                raise ValueError("energy_threshold is required when build_projection_cache=True")
            projection_cache = compute_crisp_kfac_projection_cache(
                cache["A"],
                cache["B"],
                float(energy_threshold),
                device=projection_device,
                dtype=projection_dtype,
            )
            layer_to_projection_cache[name] = projection_cache
        diagnostics.append(
            _diagnostic_from_cache(
                name,
                cache,
                skipped=False,
                skip_reason=None,
                energy_threshold=energy_threshold,
                projection_cache=projection_cache,
            )
        )

    return {
        "layer_to_cache": layer_to_cache,
        "layer_to_projection_cache": layer_to_projection_cache,
        "diagnostics": diagnostics,
        "sample_count": sample_count,
        "skipped_modules": skipped,
    }
