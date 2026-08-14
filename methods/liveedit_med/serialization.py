"""Safe JSON/safetensors serialization for LiveEdit medical modules and banks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def tensor_hashes(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    result = {}
    for name, value in state.items():
        tensor = value.detach().contiguous().cpu()
        digest = hashlib.sha256(str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode() + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
        result[name] = digest
    return result


def save_safe_state(directory: Path, state: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]):
    from safetensors.torch import save_file
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {name: value.detach().float().contiguous().cpu() for name, value in state.items()}
    save_file(tensors, str(directory / "model.safetensors"))
    manifest = {**dict(metadata), "tensor_hashes": tensor_hashes(tensors)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_safe_state(directory: Path):
    from safetensors.torch import load_file
    state = load_file(str(directory / "model.safetensors"), device="cpu")
    manifest = json.loads((directory / "manifest.json").read_text())
    if tensor_hashes(state) != manifest["tensor_hashes"]:
        raise RuntimeError("LIVEEDIT_MED_ARTIFACT_HASH_MISMATCH")
    return state, manifest
