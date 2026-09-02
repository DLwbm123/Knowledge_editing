"""Non-destructive routed layer wrappers for paper-spec memory editors."""

from __future__ import annotations

import copy
import hashlib
import math
import os
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_slot(logical_edit_id: str) -> str:
    return "edit_" + hashlib.sha256(logical_edit_id.encode("utf-8")).hexdigest()[:24]


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class GraceValueLinear(nn.Module):
    """Frozen base linear plus GRACE's routed value replacement."""

    def __init__(self, base: nn.Linear, *, replacement: str = "replace_prompt"):
        super().__init__()
        if replacement not in {"replace_prompt", "replace_last", "replace_all"}:
            raise ValueError(f"unsupported GRACE replacement: {replacement}")
        self.base = base
        freeze_module(self.base)
        self.replacement = replacement
        self.values = nn.ParameterDict()
        self.logical_to_slot: dict[str, str] = {}
        self.slot_to_logical: dict[str, str] = {}
        self.active_logical_id: str | None = None
        self.token_index = -1
        self.enabled = True

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def add_cold_value(self, logical_edit_id: str, *, seed: int) -> nn.Parameter:
        if logical_edit_id in self.logical_to_slot:
            return self.values[self.logical_to_slot[logical_edit_id]]
        slot = safe_slot(logical_edit_id)
        if slot in self.slot_to_logical and self.slot_to_logical[slot] != logical_edit_id:
            raise RuntimeError("logical-edit slot collision")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        value = torch.rand(self.out_features, generator=generator, dtype=torch.float32)
        parameter = nn.Parameter(value.to(self.base.weight.device), requires_grad=True)
        self.values[slot] = parameter
        self.logical_to_slot[logical_edit_id] = slot
        self.slot_to_logical[slot] = logical_edit_id
        return parameter

    def get_value(self, logical_edit_id: str) -> nn.Parameter:
        return self.values[self.logical_to_slot[logical_edit_id]]

    def set_active(self, logical_edit_id: str | None, *, token_index: int = -1) -> None:
        if logical_edit_id is not None and logical_edit_id not in self.logical_to_slot:
            raise KeyError(logical_edit_id)
        self.active_logical_id = logical_edit_id
        self.token_index = int(token_index)

    def disable(self) -> None:
        self.active_logical_id = None

    def train_only(self, logical_edit_id: str | None) -> list[nn.Parameter]:
        trainable = []
        for slot, parameter in self.values.items():
            active = logical_edit_id is not None and self.slot_to_logical[slot] == logical_edit_id
            parameter.requires_grad_(active)
            if active:
                trainable.append(parameter)
        return trainable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        logical_id = self.active_logical_id
        if not self.enabled or logical_id is None:
            return base_output
        value = self.get_value(logical_id).to(dtype=base_output.dtype)
        if base_output.ndim == 2:
            expanded = base_output.unsqueeze(0)
            squeeze = True
        elif base_output.ndim == 3:
            expanded = base_output
            squeeze = False
        else:
            raise ValueError(f"unexpected linear output shape: {tuple(base_output.shape)}")
        sequence_length = expanded.shape[1]
        token_index = self.token_index if self.token_index >= 0 else sequence_length + self.token_index
        token_index = min(max(token_index, 0), sequence_length - 1)
        replacement = value.view(1, 1, -1).expand(expanded.shape[0], sequence_length, -1)
        mask = torch.zeros(
            (expanded.shape[0], sequence_length, 1), dtype=torch.bool, device=expanded.device
        )
        if self.replacement == "replace_all":
            mask[:] = True
        elif self.replacement == "replace_last":
            mask[:, token_index, :] = True
        elif self.replacement == "replace_prompt":
            mask[:, : token_index + 1, :] = True
        output = torch.where(mask, replacement, expanded)
        return output.squeeze(0) if squeeze else output

    def export_state(self) -> dict:
        return {
            "replacement": self.replacement,
            "values": {
                logical_id: self.values[slot].detach().cpu()
                for logical_id, slot in self.logical_to_slot.items()
            },
        }

    def load_exported_state(self, state: dict) -> None:
        if state["replacement"] != self.replacement:
            raise ValueError("GRACE replacement contract mismatch")
        self.values = nn.ParameterDict()
        self.logical_to_slot.clear()
        self.slot_to_logical.clear()
        for logical_id, value in state["values"].items():
            slot = safe_slot(logical_id)
            parameter = nn.Parameter(
                torch.as_tensor(value, dtype=torch.float32, device=self.base.weight.device),
                requires_grad=False,
            )
            self.values[slot] = parameter
            self.logical_to_slot[logical_id] = slot
            self.slot_to_logical[slot] = logical_id


class RoutedFullLinear(nn.Module):
    """BalanceEdit wrapper: base linear plus per-edit full transformation copies."""

    def __init__(self, base: nn.Linear, *, inactive_store_dir: Path | None = None):
        super().__init__()
        self.base = base
        freeze_module(self.base)
        self.inactive_store_dir = Path(inactive_store_dir) if inactive_store_dir else None
        if self.inactive_store_dir is not None:
            self.inactive_store_dir.mkdir(parents=True, exist_ok=True)
        self.edited = nn.ModuleDict()
        self.logical_to_slot: dict[str, str] = {}
        self.slot_to_logical: dict[str, str] = {}
        self.archived: dict[str, dict[str, object]] = {}
        self.dirty: set[str] = set()
        self.active_logical_id: str | None = None
        self.enabled = True

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def add_edit(self, logical_edit_id: str) -> nn.Linear:
        if logical_edit_id in self.logical_to_slot:
            raise ValueError(f"duplicate BalanceEdit transformation: {logical_edit_id}")
        if self.inactive_store_dir is not None:
            for resident_id, resident_slot in list(self.logical_to_slot.items()):
                if resident_slot in self.edited:
                    self._archive_loaded(resident_id, evict=True)
        slot = safe_slot(logical_edit_id)
        edited = copy.deepcopy(self.base).to(device=self.base.weight.device, dtype=torch.float32)
        for parameter in edited.parameters():
            parameter.requires_grad_(True)
        self.edited[slot] = edited
        self.logical_to_slot[logical_edit_id] = slot
        self.slot_to_logical[slot] = logical_edit_id
        self.dirty.add(logical_edit_id)
        return edited

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _archive_path(self, logical_edit_id: str) -> Path:
        if self.inactive_store_dir is None:
            raise RuntimeError("disk backing is disabled")
        if logical_edit_id in self.archived:
            return Path(str(self.archived[logical_edit_id]["path"]))
        base = self.inactive_store_dir / f"{safe_slot(logical_edit_id)}.pt"
        if not base.exists():
            return base
        attempt = 1
        while True:
            candidate = self.inactive_store_dir / (
                f"{safe_slot(logical_edit_id)}.recovery_{attempt:03d}.pt"
            )
            if not candidate.exists():
                return candidate
            attempt += 1

    def _archive_loaded(self, logical_edit_id: str, *, evict: bool = True) -> None:
        if self.inactive_store_dir is None:
            return
        slot = self.logical_to_slot[logical_edit_id]
        if slot not in self.edited:
            return
        module = self.edited[slot]
        path = self._archive_path(logical_edit_id)
        if logical_edit_id in self.dirty or not path.exists():
            if logical_edit_id in self.archived:
                raise RuntimeError(
                    f"refusing to mutate an archived BalanceEdit state: {logical_edit_id}"
                )
            if path.exists():
                raise RuntimeError(f"refusing to overwrite BalanceEdit state: {path}")
            state = {
                name: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
                for name, value in module.state_dict().items()
            }
            temporary = path.with_suffix(".pt.tmp")
            torch.save(state, temporary)
            os.replace(temporary, path)
            os.chmod(path, 0o444)
            self.archived[logical_edit_id] = {
                "path": str(path),
                "sha256": self._file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            self.dirty.discard(logical_edit_id)
        elif logical_edit_id not in self.archived:
            self.archived[logical_edit_id] = {
                "path": str(path),
                "sha256": self._file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        if evict:
            del self.edited[slot]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_archived(self, logical_edit_id: str) -> nn.Linear:
        metadata = self.archived[logical_edit_id]
        path = Path(str(metadata["path"]))
        if self._file_sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"BalanceEdit archived state hash mismatch: {path}")
        edited = copy.deepcopy(self.base).to(device=self.base.weight.device, dtype=torch.float32)
        state = torch.load(path, map_location="cpu", weights_only=True)
        edited.load_state_dict(
            {name: value.to(device=self.base.weight.device, dtype=torch.float32) for name, value in state.items()}
        )
        freeze_module(edited)
        self.edited[self.logical_to_slot[logical_edit_id]] = edited
        return edited

    def get_edit(self, logical_edit_id: str) -> nn.Linear:
        slot = self.logical_to_slot[logical_edit_id]
        if slot not in self.edited:
            return self._load_archived(logical_edit_id)
        return self.edited[slot]

    def set_active(self, logical_edit_id: str | None) -> None:
        if logical_edit_id is not None and logical_edit_id not in self.logical_to_slot:
            raise KeyError(logical_edit_id)
        if self.inactive_store_dir is not None and logical_edit_id is not None:
            for resident_id, resident_slot in list(self.logical_to_slot.items()):
                if resident_id != logical_edit_id and resident_slot in self.edited:
                    self._archive_loaded(resident_id, evict=True)
        if logical_edit_id is not None:
            self.get_edit(logical_edit_id)
        elif self.active_logical_id is not None:
            # Keep one immutable inactive copy resident so consecutive probes routed
            # to the same edit do not re-read a ~1 GB state file.
            self._archive_loaded(self.active_logical_id, evict=False)
        self.active_logical_id = logical_edit_id

    def train_only(self, logical_edit_id: str | None) -> list[nn.Parameter]:
        trainable = []
        for slot, module in self.edited.items():
            active = logical_edit_id is not None and self.slot_to_logical[slot] == logical_edit_id
            for parameter in module.parameters():
                parameter.requires_grad_(active)
                if active:
                    self.dirty.add(self.slot_to_logical[slot])
                    trainable.append(parameter)
        return trainable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logical_id = self.active_logical_id
        if not self.enabled or logical_id is None:
            return self.base(inputs)
        edited = self.get_edit(logical_id)
        return edited(inputs.to(dtype=torch.float32)).to(dtype=inputs.dtype)

    def export_state(self) -> dict:
        if self.inactive_store_dir is not None:
            if self.active_logical_id is not None:
                self.set_active(None)
            for logical_id in list(self.logical_to_slot):
                self._archive_loaded(logical_id, evict=True)
            return {
                "storage_mode": "disk_backed_float32_exact",
                "inactive_store_dir": str(self.inactive_store_dir),
                "archived": dict(self.archived),
            }
        return {
            "storage_mode": "resident_float32",
            "edits": {
                logical_id: {
                    key: value.detach().cpu()
                    for key, value in self.edited[slot].state_dict().items()
                }
                for logical_id, slot in self.logical_to_slot.items()
            }
        }

    def load_exported_state(self, state: dict) -> None:
        if state.get("storage_mode") == "disk_backed_float32_exact":
            self.edited = nn.ModuleDict()
            self.logical_to_slot.clear()
            self.slot_to_logical.clear()
            self.archived = dict(state["archived"])
            self.dirty.clear()
            self.inactive_store_dir = Path(state["inactive_store_dir"])
            for logical_id, metadata in self.archived.items():
                path = Path(str(metadata["path"]))
                if not path.is_file() or self._file_sha256(path) != metadata["sha256"]:
                    raise RuntimeError(f"BalanceEdit archived state unavailable or changed: {path}")
                slot = safe_slot(logical_id)
                self.logical_to_slot[logical_id] = slot
                self.slot_to_logical[slot] = logical_id
            self.active_logical_id = None
            return
        self.edited = nn.ModuleDict()
        self.logical_to_slot.clear()
        self.slot_to_logical.clear()
        self.archived.clear()
        self.dirty.clear()
        for logical_id, weights in state["edits"].items():
            edited = self.add_edit(logical_id)
            edited.load_state_dict(
                {
                    name: torch.as_tensor(value, dtype=torch.float32, device=self.base.weight.device)
                    for name, value in weights.items()
                }
            )
            for parameter in edited.parameters():
                parameter.requires_grad_(False)
            self.dirty.discard(logical_id)

    def parameter_statistics(self) -> dict[str, object]:
        per_edit_count = sum(parameter.numel() for parameter in self.base.parameters())
        per_edit_bytes = per_edit_count * torch.tensor([], dtype=torch.float32).element_size()
        entry_count = len(self.logical_to_slot)
        return {
            "storage_mode": "disk_backed_float32_exact"
            if self.inactive_store_dir is not None
            else "resident_float32",
            "entry_count": entry_count,
            "archived_entry_count": len(self.archived),
            "parameter_count": entry_count * per_edit_count,
            "parameter_bytes": entry_count * per_edit_bytes,
        }

    def clear(self) -> None:
        if self.active_logical_id is not None:
            self.set_active(None)
        self.edited = nn.ModuleDict()
        self.logical_to_slot.clear()
        self.slot_to_logical.clear()
        self.archived.clear()
        self.dirty.clear()
        self.active_logical_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RoutedLoRALinear(nn.Module):
    """BELoRA linear wrapper with independent adapters keyed by logical edit ID."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if rank <= 0 or alpha <= 0 or dropout != 0.0:
            raise ValueError("BELoRA lock requires positive rank/alpha and dropout=0")
        self.base = base
        freeze_module(self.base)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = float(dropout)
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.logical_to_slot: dict[str, str] = {}
        self.slot_to_logical: dict[str, str] = {}
        self.active_logical_id: str | None = None
        self.enabled = True

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def add_adapter(self, logical_edit_id: str, *, seed: int) -> tuple[nn.Parameter, nn.Parameter]:
        if logical_edit_id in self.logical_to_slot:
            raise ValueError(f"duplicate BELoRA adapter: {logical_edit_id}")
        slot = safe_slot(logical_edit_id)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        a = torch.empty((self.rank, self.in_features), dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        b = torch.zeros((self.out_features, self.rank), dtype=torch.float32)
        self.lora_A[slot] = nn.Parameter(a.to(self.base.weight.device), requires_grad=True)
        self.lora_B[slot] = nn.Parameter(b.to(self.base.weight.device), requires_grad=True)
        self.logical_to_slot[logical_edit_id] = slot
        self.slot_to_logical[slot] = logical_edit_id
        return self.lora_A[slot], self.lora_B[slot]

    def set_active(self, logical_edit_id: str | None) -> None:
        if logical_edit_id is not None and logical_edit_id not in self.logical_to_slot:
            raise KeyError(logical_edit_id)
        self.active_logical_id = logical_edit_id

    def train_only(self, logical_edit_id: str | None) -> list[nn.Parameter]:
        trainable = []
        for slot in self.lora_A:
            active = logical_edit_id is not None and self.slot_to_logical[slot] == logical_edit_id
            self.lora_A[slot].requires_grad_(active)
            self.lora_B[slot].requires_grad_(active)
            if active:
                trainable.extend((self.lora_A[slot], self.lora_B[slot]))
        return trainable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        logical_id = self.active_logical_id
        if not self.enabled or logical_id is None:
            return base_output
        slot = self.logical_to_slot[logical_id]
        value = inputs.to(dtype=torch.float32)
        delta = F.linear(F.linear(value, self.lora_A[slot]), self.lora_B[slot]) * self.scaling
        return base_output + delta.to(dtype=base_output.dtype)

    def export_state(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "adapters": {
                logical_id: {
                    "A": self.lora_A[slot].detach().cpu(),
                    "B": self.lora_B[slot].detach().cpu(),
                }
                for logical_id, slot in self.logical_to_slot.items()
            },
        }

    def load_exported_state(self, state: dict) -> None:
        if (state["rank"], state["alpha"], state["dropout"]) != (
            self.rank,
            self.alpha,
            self.dropout,
        ):
            raise ValueError("BELoRA adapter config mismatch")
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.logical_to_slot.clear()
        self.slot_to_logical.clear()
        for logical_id, weights in state["adapters"].items():
            slot = safe_slot(logical_id)
            self.lora_A[slot] = nn.Parameter(
                torch.as_tensor(weights["A"], dtype=torch.float32, device=self.base.weight.device),
                requires_grad=False,
            )
            self.lora_B[slot] = nn.Parameter(
                torch.as_tensor(weights["B"], dtype=torch.float32, device=self.base.weight.device),
                requires_grad=False,
            )
            self.logical_to_slot[logical_id] = slot
            self.slot_to_logical[slot] = logical_id


def trainable_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    return [parameter for module in modules for parameter in module.parameters() if parameter.requires_grad]
