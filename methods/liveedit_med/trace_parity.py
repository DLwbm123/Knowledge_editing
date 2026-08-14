"""Hash-based comparison helpers for end-to-end upstream trace parity."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import torch


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    payload = str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode() + tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def compare_tensor(name: str, upstream: torch.Tensor, port: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    if upstream.shape != port.shape:
        return {"name": name, "passed": False, "reason": "shape_mismatch", "upstream_shape": list(upstream.shape), "port_shape": list(port.shape)}
    delta = (upstream.detach().float().cpu() - port.detach().float().cpu()).abs()
    passed = bool(torch.allclose(upstream.detach().float().cpu(), port.detach().float().cpu(), atol=atol, rtol=rtol))
    return {
        "name": name, "passed": passed, "max_abs_error": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.mean().item()) if delta.numel() else 0.0,
        "upstream_sha256": tensor_sha256(upstream), "port_sha256": tensor_sha256(port),
        "shape": list(upstream.shape), "atol": atol, "rtol": rtol,
    }


def summarize_trace(rows: Sequence[Mapping[str, Any]], *, required_names: Sequence[str]) -> dict[str, Any]:
    by_name = {str(row["name"]): dict(row) for row in rows}
    missing = [name for name in required_names if name not in by_name]
    passed = sum(bool(by_name[name].get("passed")) for name in required_names if name in by_name)
    return {"required": len(required_names), "passed": passed, "missing": missing,
            "all_passed": not missing and passed == len(required_names), "comparisons": [by_name[name] for name in required_names if name in by_name]}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
