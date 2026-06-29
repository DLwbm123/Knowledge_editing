from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .solver import (
    EngramLayerUpdate,
    apply_update_to_module,
    normalize_update_direction,
    paper_direction_equivalent,
)

BANK_VERSION = 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _module_map(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def _direction_sign_for_raw(raw: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> int:
    if raw.get("direction_sign") is not None:
        return int(raw["direction_sign"])
    if metadata and metadata.get("direction_sign") is not None:
        return int(metadata["direction_sign"])
    direction = raw.get("engram_update_direction")
    if direction is None and metadata:
        direction = metadata.get("engram_update_direction")
    direction = normalize_update_direction(direction or "subtract")
    return 1 if direction == "add" else -1


def _update_direction_for_raw(raw: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
    direction = raw.get("engram_update_direction")
    if direction is None and metadata:
        direction = metadata.get("engram_update_direction")
    if direction is None:
        sign = _direction_sign_for_raw(raw, metadata)
        return "add" if sign > 0 else "subtract"
    return normalize_update_direction(direction)


def _normalize_raw_update(raw: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(raw)
    raw.setdefault("alpha", float(metadata.get("alpha", 1.0)))
    raw.setdefault("beta", float(metadata.get("beta", 0.0)))
    direction = _update_direction_for_raw(raw, metadata)
    sign = _direction_sign_for_raw(raw, metadata)
    raw["engram_update_direction"] = direction
    raw["direction_sign"] = sign
    raw.setdefault("behavior_objective", metadata.get("behavior_objective"))
    raw.setdefault("paper_direction_equivalent", metadata.get("paper_direction_equivalent", paper_direction_equivalent(direction)))
    stats = dict(raw.get("stats") or {})
    stats.setdefault("alpha", float(raw["alpha"]))
    stats.setdefault("engram_update_direction", direction)
    stats.setdefault("direction_sign", sign)
    stats.setdefault("behavior_objective", raw.get("behavior_objective"))
    stats.setdefault("paper_direction_equivalent", raw.get("paper_direction_equivalent"))
    if "effective_update_norm_ratio" not in stats and "effective_norm_ratio" in stats:
        stats["effective_update_norm_ratio"] = stats["effective_norm_ratio"]
    if "effective_norm_ratio" not in stats and "effective_update_norm_ratio" in stats:
        stats["effective_norm_ratio"] = stats["effective_update_norm_ratio"]
    raw["stats"] = stats
    return raw


class EngramBank:
    """Persistent directory-backed ENGRAM bank."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.edits_dir = self.root / "edits"
        self.index_path = self.root / "index.json"

    def ensure(self) -> None:
        self.edits_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index({"version": BANK_VERSION, "edits": []})

    def _read_index(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return {"version": BANK_VERSION, "edits": []}
        with self.index_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_index(self, index: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(index), f, indent=2, sort_keys=True)

    def save_edit(
        self,
        *,
        edit_id: str,
        metadata: Dict[str, Any],
        updates: Dict[str, EngramLayerUpdate],
        overwrite: bool = False,
    ) -> Path:
        self.ensure()
        edit_dir = self.edits_dir / edit_id
        if edit_dir.exists() and not overwrite:
            raise FileExistsError(f"ENGRAM bank edit already exists: {edit_id}")
        edit_dir.mkdir(parents=True, exist_ok=True)

        metadata = dict(metadata)
        metadata.setdefault("edit_id", edit_id)
        metadata.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        metadata.setdefault("bank_version", BANK_VERSION)
        metadata["selected_modules"] = list(updates.keys())
        metadata["layers"] = [dict(update.stats) for update in updates.values()]
        if updates:
            first_update = next(iter(updates.values()))
            metadata.setdefault("alpha", float(first_update.alpha))
            metadata.setdefault("beta", float(first_update.beta))
            metadata.setdefault("engram_update_direction", first_update.engram_update_direction)
            metadata.setdefault("direction_sign", int(first_update.direction_sign))
            metadata.setdefault("behavior_objective", first_update.behavior_objective)
            metadata.setdefault("paper_direction_equivalent", first_update.paper_direction_equivalent)
            effective = [
                float(update.stats.get("effective_update_norm_ratio", update.stats.get("effective_norm_ratio", 0.0)))
                for update in updates.values()
            ]
            metadata.setdefault("effective_update_norm_ratio", max(effective) if effective else 0.0)

        tensors: Dict[str, Dict[str, Any]] = {}
        for name, update in updates.items():
            tensors[name] = {
                "weight": update.weight.detach().cpu(),
                "bias": update.bias.detach().cpu() if update.bias is not None else None,
                "projector": update.projector.detach().cpu() if update.projector is not None else None,
                "delta_safe_weight": update.delta_safe_weight.detach().cpu() if update.delta_safe_weight is not None else None,
                "delta_safe_bias": update.delta_safe_bias.detach().cpu() if update.delta_safe_bias is not None else None,
                "alpha": float(update.alpha),
                "beta": float(update.beta),
                "engram_update_direction": update.engram_update_direction,
                "direction_sign": int(update.direction_sign),
                "behavior_objective": update.behavior_objective,
                "paper_direction_equivalent": update.paper_direction_equivalent,
                "stats": dict(update.stats),
            }

        with (edit_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(metadata), f, indent=2, sort_keys=True)
        torch.save({"version": BANK_VERSION, "updates": tensors}, edit_dir / "tensors.pt")

        index = self._read_index()
        index["edits"] = [item for item in index.get("edits", []) if item.get("edit_id") != edit_id]
        index["edits"].append(
            {
                "edit_id": edit_id,
                "timestamp": metadata.get("timestamp"),
                "concept_id": metadata.get("concept_id"),
                "modality": metadata.get("modality"),
                "metadata_path": str(edit_dir / "metadata.json"),
                "tensor_path": str(edit_dir / "tensors.pt"),
            }
        )
        self._write_index(index)
        return edit_dir

    def list_edits(self) -> List[Dict[str, Any]]:
        index = self._read_index()
        return list(index.get("edits", []))

    def match_edit_ids_to_records(
        self,
        records: List[Dict[str, Any]],
        *,
        allow_positional_matching: bool = False,
    ) -> tuple[List[str], Dict[str, Any]]:
        """Match saved edits to source records, preferring metadata record ids.

        Older ENGRAM banks did not save record ids, so callers must fall back to
        positional matching for those banks. New banks save record_id/source_record_id
        and avoid silent metric drift when run order differs from data order.
        """
        edits = self.list_edits()
        if not edits:
            raise RuntimeError(f"No edits in ENGRAM bank {self.root}.")

        id_to_edit: Dict[str, str] = {}
        missing_metadata_ids: List[str] = []
        duplicate_metadata_ids: List[str] = []
        for item in edits:
            edit_id = item["edit_id"]
            try:
                metadata = self.load_edit(edit_id)["metadata"]
            except FileNotFoundError:
                missing_metadata_ids.append(edit_id)
                continue
            record_ids: List[str] = []
            for key in ("record_id", "source_record_id"):
                value = metadata.get(key)
                if value is not None:
                    record_ids.append(str(value))
            for value in metadata.get("source_request_ids", []) or []:
                if value is not None:
                    record_ids.append(str(value))
            for record_id in dict.fromkeys(record_ids):
                if record_id in id_to_edit and id_to_edit[record_id] != edit_id:
                    duplicate_metadata_ids.append(record_id)
                else:
                    id_to_edit[record_id] = edit_id

        record_ids = [str(record.get("id") or record.get("record_id") or record.get("source_record_id") or "") for record in records]
        if id_to_edit and all(record_id and record_id in id_to_edit for record_id in record_ids):
            return [id_to_edit[record_id] for record_id in record_ids], {
                "mode": "record_id",
                "record_ids": record_ids,
                "missing_metadata_edit_ids": missing_metadata_ids,
                "duplicate_metadata_record_ids": duplicate_metadata_ids,
            }

        if len(edits) < len(records):
            raise RuntimeError(f"Bank {self.root} has {len(edits)} edits, expected at least {len(records)}.")
        fallback = {
            "mode": "positional_fallback",
            "reason": "bank metadata does not fully cover requested record ids",
            "record_ids": record_ids,
            "metadata_record_ids": sorted(id_to_edit),
            "missing_metadata_edit_ids": missing_metadata_ids,
            "duplicate_metadata_record_ids": duplicate_metadata_ids,
        }
        if not allow_positional_matching:
            raise RuntimeError(
                "ENGRAM bank record-id preflight failed: refusing positional matching because "
                "bank metadata does not fully cover requested raw record ids. "
                "Use --allow-positional-matching only for explicit legacy-bank audits. "
                f"Details: {fallback}"
            )
        return [item["edit_id"] for item in edits[: len(records)]], fallback

    def load_edit(self, edit_id: str) -> Dict[str, Any]:
        edit_dir = self.edits_dir / edit_id
        metadata_path = edit_dir / "metadata.json"
        tensor_path = edit_dir / "tensors.pt"
        if not metadata_path.exists() or not tensor_path.exists():
            raise FileNotFoundError(f"ENGRAM edit not found: {edit_id}")
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        tensors = torch.load(tensor_path, map_location="cpu")
        updates = {
            module_name: _normalize_raw_update(raw, metadata)
            for module_name, raw in tensors["updates"].items()
        }
        if "engram_update_direction" not in metadata:
            metadata["engram_update_direction"] = "subtract"
        if "direction_sign" not in metadata:
            metadata["direction_sign"] = -1
        if "paper_direction_equivalent" not in metadata:
            metadata["paper_direction_equivalent"] = paper_direction_equivalent(metadata["engram_update_direction"])
        return {"metadata": metadata, "updates": updates}

    def delete_edit(self, edit_id: str) -> None:
        edit_dir = self.edits_dir / edit_id
        if edit_dir.exists():
            for path in edit_dir.iterdir():
                path.unlink()
            edit_dir.rmdir()
        index = self._read_index()
        index["edits"] = [item for item in index.get("edits", []) if item.get("edit_id") != edit_id]
        self._write_index(index)

    def export_summary_csv(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        for item in self.list_edits():
            try:
                edit = self.load_edit(item["edit_id"])
            except FileNotFoundError:
                continue
            meta = edit["metadata"]
            layer_count = len(meta.get("layers", []))
            rows.append(
                {
                    "edit_id": meta.get("edit_id"),
                    "concept_id": meta.get("concept_id"),
                    "modality": meta.get("modality"),
                    "timestamp": meta.get("timestamp"),
                    "alpha": meta.get("alpha"),
                    "beta": meta.get("beta"),
                    "edit_mode": meta.get("edit_mode"),
                    "token_scope": meta.get("token_scope"),
                    "num_layers": layer_count,
                }
            )
        fieldnames = [
            "edit_id",
            "concept_id",
            "modality",
            "timestamp",
            "alpha",
            "beta",
            "edit_mode",
            "token_scope",
            "num_layers",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return out_path

    def compose_updates(self, edit_ids: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        ids = list(edit_ids) if edit_ids is not None else [item["edit_id"] for item in self.list_edits()]
        composed: Dict[str, Dict[str, Any]] = {}
        for edit_id in ids:
            edit = self.load_edit(edit_id)
            for module_name, update in edit["updates"].items():
                target = composed.setdefault(module_name, {})
                for key in ("weight", "bias", "delta_safe_weight", "delta_safe_bias"):
                    tensor = update.get(key)
                    if tensor is None:
                        continue
                    if key in {"weight", "bias"}:
                        scale = float(_direction_sign_for_raw(update, edit["metadata"])) * float(update.get("alpha", 1.0))
                    else:
                        scale = float(update.get("beta", 0.0))
                    value = scale * tensor.detach().cpu()
                    target[key] = value if key not in target else target[key] + value
        return composed

    def apply_edit(self, model: nn.Module, edit_id: str) -> None:
        self._apply_or_rollback(model, edit_id, direction=-1)

    def rollback_edit(self, model: nn.Module, edit_id: str) -> None:
        self._apply_or_rollback(model, edit_id, direction=1)

    def _apply_or_rollback(self, model: nn.Module, edit_id: str, *, direction: int) -> None:
        edit = self.load_edit(edit_id)
        modules = _module_map(model)
        for module_name, raw in edit["updates"].items():
            module = modules.get(module_name)
            if not isinstance(module, nn.Linear):
                raise KeyError(f"ENGRAM bank module not found or not Linear: {module_name}")
            update = EngramLayerUpdate(
                module_name=module_name,
                weight=raw["weight"],
                bias=raw.get("bias"),
                projector=raw.get("projector"),
                delta_safe_weight=raw.get("delta_safe_weight"),
                delta_safe_bias=raw.get("delta_safe_bias"),
                alpha=float(raw.get("alpha", edit["metadata"].get("alpha", 1.0))),
                beta=float(raw.get("beta", edit["metadata"].get("beta", 0.0))),
                engram_update_direction=_update_direction_for_raw(raw, edit["metadata"]),
                direction_sign=_direction_sign_for_raw(raw, edit["metadata"]),
                behavior_objective=raw.get("behavior_objective", edit["metadata"].get("behavior_objective")),
                paper_direction_equivalent=raw.get(
                    "paper_direction_equivalent",
                    edit["metadata"].get("paper_direction_equivalent", paper_direction_equivalent(_update_direction_for_raw(raw, edit["metadata"]))),
                ),
                stats=raw.get("stats", {}),
            )
            apply_update_to_module(module, update, direction=direction)
