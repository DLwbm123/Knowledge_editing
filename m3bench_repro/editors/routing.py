"""Deterministic routing primitives shared by GRACE, BalanceEdit, and BELoRA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F


DistanceName = Literal["cosine", "euclidean"]


def canonical_float32(value: float) -> float:
    """Return the deterministic IEEE-754 binary32 representation as a host float."""
    return float(torch.as_tensor(float(value), dtype=torch.float32).item())


def route_dict_equal(
    left: dict | None,
    right: dict | None,
    *,
    radius_mode: Literal["exact", "float32"] = "exact",
) -> bool:
    """Compare route reports exactly, optionally canonicalizing only ``radius``."""
    if radius_mode == "exact":
        return left == right
    if radius_mode != "float32":
        raise ValueError(f"unsupported radius comparison mode: {radius_mode}")
    if left is None or right is None:
        return left is right
    left_copy = dict(left)
    right_copy = dict(right)
    if ("radius" in left_copy) != ("radius" in right_copy):
        return False
    if "radius" in left_copy:
        left_radius = left_copy["radius"]
        right_radius = right_copy["radius"]
        if left_radius is None or right_radius is None:
            if left_radius is not right_radius:
                return False
        else:
            left_copy["radius"] = canonical_float32(left_radius)
            right_copy["radius"] = canonical_float32(right_radius)
    return left_copy == right_copy


def _vector(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().to(dtype=torch.float32)
    if value.ndim == 1:
        return value
    if value.numel() == value.shape[-1]:
        return value.reshape(-1)
    raise ValueError(f"expected one vector, got shape {tuple(value.shape)}")


def _matrix(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().to(dtype=torch.float32)
    if value.ndim == 1:
        return value.unsqueeze(0)
    if value.ndim == 2:
        return value
    if value.numel() % value.shape[-1] == 0:
        return value.reshape(-1, value.shape[-1])
    raise ValueError(f"expected vectors in final dimension, got shape {tuple(value.shape)}")


def cosine_distances(keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Return stable float32 cosine distances after explicit normalization."""
    key_matrix = _matrix(keys)
    query_vector = _vector(query).to(key_matrix.device)
    if key_matrix.shape[-1] != query_vector.shape[-1]:
        raise ValueError("key/query dimension mismatch")
    normalized_keys = F.normalize(key_matrix, p=2, dim=-1)
    normalized_query = F.normalize(query_vector, p=2, dim=-1)
    similarity = (normalized_keys * normalized_query.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - similarity).clamp_min(0.0)


def euclidean_distances(keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Return L2 distance from every key to one query."""
    key_matrix = _matrix(keys)
    query_vector = _vector(query).to(key_matrix.device)
    if key_matrix.shape[-1] != query_vector.shape[-1]:
        raise ValueError("key/query dimension mismatch")
    return torch.linalg.vector_norm(key_matrix - query_vector.unsqueeze(0), dim=-1)


def distances(keys: torch.Tensor, query: torch.Tensor, name: DistanceName) -> torch.Tensor:
    if name == "cosine":
        return cosine_distances(keys, query)
    if name == "euclidean":
        return euclidean_distances(keys, query)
    raise ValueError(f"unsupported distance: {name}")


def balanced_radius(
    key: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    alpha: float,
    distance: DistanceName,
) -> torch.Tensor:
    """Locked BalanceEdit radius formula.

    Source contract: ``(1-alpha) * locality_distance + alpha * rephrase_distance``.
    The negative/black anchor supplies locality distance and the positive
    rephrase anchor supplies rephrase distance.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    locality = distances(_vector(key), _vector(negative), distance)[0]
    rephrase = distances(_vector(key), _vector(positive), distance)[0]
    return (1.0 - alpha) * locality + alpha * rephrase


@dataclass(frozen=True)
class RouteDecision:
    logical_edit_id: str | None
    nearest_logical_edit_id: str | None
    nearest_distance: float | None
    radius: float | None
    activated: bool
    distance: DistanceName


class MemoryRouter:
    """One-key/one-radius-per-edit nearest-entry router."""

    def __init__(self, distance: DistanceName):
        self.distance = distance
        self.logical_ids: list[str] = []
        self.keys: list[torch.Tensor] = []
        self.radii: list[float] = []
        self.labels: list[tuple[int, ...]] = []

    def __len__(self) -> int:
        return len(self.logical_ids)

    def clear(self) -> None:
        self.logical_ids.clear()
        self.keys.clear()
        self.radii.clear()
        self.labels.clear()

    def add(
        self,
        logical_edit_id: str,
        key: torch.Tensor,
        radius: float | torch.Tensor,
        label: Iterable[int] = (),
    ) -> None:
        if logical_edit_id in self.logical_ids:
            raise ValueError(f"duplicate logical edit id: {logical_edit_id}")
        radius_value = float(torch.as_tensor(radius, dtype=torch.float32).item())
        if not torch.isfinite(torch.tensor(radius_value)) or radius_value < 0:
            raise ValueError(f"invalid route radius: {radius_value}")
        self.logical_ids.append(logical_edit_id)
        stored_key = _vector(key).clone()
        if self.distance == "cosine":
            stored_key = F.normalize(stored_key, p=2, dim=0)
        self.keys.append(stored_key)
        self.radii.append(radius_value)
        self.labels.append(tuple(int(x) for x in label))

    def _key_matrix(self, device: torch.device | None = None) -> torch.Tensor:
        if not self.keys:
            raise RuntimeError("router is empty")
        matrix = torch.stack(self.keys)
        return matrix if device is None else matrix.to(device)

    def route(self, query: torch.Tensor) -> RouteDecision:
        if not self.keys:
            return RouteDecision(None, None, None, None, False, self.distance)
        query_vector = _vector(query)
        dists = distances(self._key_matrix(query_vector.device), query_vector, self.distance)
        nearest_index = int(torch.argmin(dists).item())
        nearest_distance = float(dists[nearest_index].item())
        radius = self.radii[nearest_index]
        activated = nearest_distance <= radius
        nearest_id = self.logical_ids[nearest_index]
        return RouteDecision(
            nearest_id if activated else None,
            nearest_id,
            nearest_distance,
            radius,
            activated,
            self.distance,
        )

    def export_state(self) -> dict:
        return {
            "distance": self.distance,
            "entries": [
                {
                    "logical_edit_id": logical_id,
                    "key": key.detach().cpu(),
                    "radius": radius,
                    "label": list(label),
                }
                for logical_id, key, radius, label in zip(
                    self.logical_ids, self.keys, self.radii, self.labels, strict=True
                )
            ],
        }

    @classmethod
    def from_state(cls, state: dict, *, device: torch.device | str | None = None) -> "MemoryRouter":
        router = cls(state["distance"])
        for entry in state["entries"]:
            key = torch.as_tensor(entry["key"], dtype=torch.float32)
            if device is not None:
                key = key.to(device)
            router.add(entry["logical_edit_id"], key, entry["radius"], entry.get("label", ()))
        return router


@dataclass(frozen=True)
class GraceInsertResult:
    requested_logical_edit_id: str
    effective_logical_edit_id: str
    action: Literal["insert", "insert_far", "collision_split", "same_label_reuse"]
    nearest_distance: float | None


class GraceCodebook(MemoryRouter):
    """GRACE codebook with source-compatible insert/update/collision policy.

    The only primary adaptation is the authorized distance function.  Main
    M3Bench use is cosine; Euclidean remains available solely for source-parity
    diagnostics.
    """

    def __init__(self, *, distance: DistanceName = "cosine", eps_init: float = 1.0):
        super().__init__(distance)
        if eps_init <= 0:
            raise ValueError("eps_init must be positive")
        self.eps_init = float(eps_init)

    @staticmethod
    def source_label_match(left: Iterable[int], right: Iterable[int]) -> bool:
        return tuple(int(x) for x in left) == tuple(int(x) for x in right)

    def insert_with_source_semantics(
        self,
        logical_edit_id: str,
        key: torch.Tensor,
        label: Iterable[int],
    ) -> GraceInsertResult:
        label_tuple = tuple(int(x) for x in label)
        if not self.keys:
            self.add(logical_edit_id, key, self.eps_init, label_tuple)
            return GraceInsertResult(logical_edit_id, logical_edit_id, "insert", None)

        query = _vector(key)
        dists = distances(self._key_matrix(query.device), query, self.distance)
        nearest_index = int(torch.argmin(dists).item())
        nearest_distance = float(dists[nearest_index].item())
        nearest_id = self.logical_ids[nearest_index]
        if nearest_distance > self.eps_init + self.radii[nearest_index]:
            self.add(logical_edit_id, query, self.eps_init, label_tuple)
            return GraceInsertResult(logical_edit_id, logical_edit_id, "insert_far", nearest_distance)

        if not self.source_label_match(label_tuple, self.labels[nearest_index]):
            self.add(logical_edit_id, query, self.eps_init, label_tuple)
            new_index = len(self.radii) - 1
            self.radii[nearest_index] = max(0.0, nearest_distance / 2.0 - 1e-5)
            self.radii[new_index] = nearest_distance / 2.0
            return GraceInsertResult(logical_edit_id, logical_edit_id, "collision_split", nearest_distance)

        if nearest_distance > self.radii[nearest_index]:
            self.radii[nearest_index] = nearest_distance
        return GraceInsertResult(logical_edit_id, nearest_id, "same_label_reuse", nearest_distance)

    def export_state(self) -> dict:
        state = super().export_state()
        state["eps_init"] = self.eps_init
        return state

    @classmethod
    def from_state(cls, state: dict, *, device: torch.device | str | None = None) -> "GraceCodebook":
        codebook = cls(distance=state["distance"], eps_init=state["eps_init"])
        for entry in state["entries"]:
            key = torch.as_tensor(entry["key"], dtype=torch.float32)
            if device is not None:
                key = key.to(device)
            codebook.add(entry["logical_edit_id"], key, entry["radius"], entry.get("label", ()))
        return codebook


def decision_as_json(decision: RouteDecision) -> dict:
    return asdict(decision)
