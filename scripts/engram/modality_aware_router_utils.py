"""Pure utilities for the cross-edit modality-aware semantic router."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BRANCHES = ("textual", "visual", "paired")
MODALITIES = ("img", "text", "fused")
SCALAR_NAMES = ("cos_raw", "l2_raw", "cos_pca", "l2_pca")
SCORE_NAMES = ("s_img", "s_text", "s_fused", "s_min", "s_joint")


def normalize_question(value: str) -> str:
    return " ".join(str(value).casefold().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*values: Any) -> str:
    return hashlib.sha256("||".join(str(value) for value in values).encode()).hexdigest()


def source_sort_key(record_id: str, image_sha256: str, question: str) -> str:
    return stable_hash(record_id, image_sha256, normalize_question(question))


def edit_level_split(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = [dict(row) for row in records]
    n = len(rows)
    if n < 36:
        raise RuntimeError("SEMANTIC_ROUTER_INSUFFICIENT_CROSS_EDIT_DATA")
    if n >= 80:
        rows = rows[:96]
        train, calibration, heldout = rows[:64], rows[64:80], rows[80:96]
    elif n >= 48:
        n_train = int(n * .60)
        n_cal = int(n * .20)
        train, calibration, heldout = rows[:n_train], rows[n_train:n_train + n_cal], rows[n_train + n_cal:]
        if len(calibration) < 8 or len(heldout) < 8:
            raise RuntimeError("SEMANTIC_ROUTER_INSUFFICIENT_CROSS_EDIT_DATA")
    else:
        train, calibration, heldout = rows[:24], rows[24:30], rows[30:]
        if len(heldout) < 6:
            raise RuntimeError("SEMANTIC_ROUTER_INSUFFICIENT_CROSS_EDIT_DATA")
    ids = [{str(row["record_id"]) for row in part} for part in (train, calibration, heldout)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise RuntimeError("SEMANTIC_ROUTER_INVALID_ENGINEERING_RUN: edit split overlap")
    return {"train": train, "calibration": calibration, "heldout": heldout}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom else 0.0


def validated_scores(candidate: Mapping[str, np.ndarray], prototype: Mapping[str, np.ndarray]) -> dict[str, float]:
    result = {f"s_{name}": cosine(candidate[name], prototype[name]) for name in MODALITIES}
    result["s_min"] = min(result["s_img"], result["s_text"], result["s_fused"])
    result["s_joint"] = .30 * result["s_img"] + .30 * result["s_text"] + .40 * result["s_fused"]
    return result


def fit_pcas(keys: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from sklearn.decomposition import PCA

    fitted, report = {}, {}
    for name in MODALITIES:
        matrix = np.asarray(keys[name], dtype=np.float64)
        n_components = min(32, matrix.shape[0] - 1, matrix.shape[1])
        if n_components < 1:
            raise RuntimeError("SEMANTIC_ROUTER_INVALID_ENGINEERING_RUN: insufficient PCA keys")
        pca = PCA(n_components=n_components, whiten=True, svd_solver="full")
        transformed = pca.fit_transform(matrix)
        if not np.isfinite(transformed).all():
            raise RuntimeError("SEMANTIC_ROUTER_INVALID_ENGINEERING_RUN: non-finite PCA")
        fitted[name] = pca
        report[name] = {
            "n_components": n_components,
            "mean": pca.mean_.tolist(),
            "components": pca.components_.tolist(),
            "explained_variance": pca.explained_variance_.tolist(),
            "whiten": True,
            "svd_solver": "full",
            "fit_dtype": "float64",
            "inference_dtype": "float32",
            "fit_key_count": int(matrix.shape[0]),
        }
    return fitted, report


def pca_transform(pca: Any, value: np.ndarray) -> np.ndarray:
    return np.asarray(pca.transform(np.asarray(value, dtype=np.float64).reshape(1, -1))[0], dtype=np.float32)


def pca_transform_exported(mean: np.ndarray, components: np.ndarray, variance: np.ndarray, value: np.ndarray) -> np.ndarray:
    centered = np.asarray(value, dtype=np.float32) - np.asarray(mean, dtype=np.float32)
    projected = np.asarray(components, dtype=np.float32) @ centered
    return projected / np.sqrt(np.asarray(variance, dtype=np.float32))


def relation_features(branch: str, prototype: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray], pcas: Mapping[str, Any]) -> np.ndarray:
    if branch not in BRANCHES:
        raise ValueError(branch)
    z_proto = {name: pca_transform(pcas[name], prototype[name]) for name in MODALITIES}
    z_candidate = {name: pca_transform(pcas[name], candidate[name]) for name in MODALITIES}
    scalars: dict[str, dict[str, float]] = {}
    for name in MODALITIES:
        scalars[name] = {
            "cos_raw": cosine(candidate[name], prototype[name]),
            "l2_raw": float(np.linalg.norm(np.asarray(candidate[name]) - np.asarray(prototype[name]))),
            "cos_pca": cosine(z_candidate[name], z_proto[name]),
            "l2_pca": float(np.linalg.norm(z_candidate[name] - z_proto[name])),
        }
    full = {"textual": ("text", "fused"), "visual": ("img", "fused"), "paired": MODALITIES}[branch]
    scalar_modalities = {"textual": ("img",), "visual": ("text",), "paired": MODALITIES}[branch]
    parts = []
    for name in full:
        parts.extend((np.abs(z_candidate[name] - z_proto[name]), z_candidate[name] * z_proto[name]))
    for name in scalar_modalities:
        parts.append(np.asarray([scalars[name][field] for field in SCALAR_NAMES], dtype=np.float32))
    scores = validated_scores(candidate, prototype)
    parts.append(np.asarray([scores[field] for field in SCORE_NAMES], dtype=np.float32))
    result = np.concatenate(parts).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("SEMANTIC_ROUTER_INVALID_ENGINEERING_RUN: non-finite features")
    return result


def fixed_logistic_regression() -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=42)


def train_branch(features: np.ndarray, labels: np.ndarray) -> tuple[Any, Any, dict[str, Any]]:
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    scaler = StandardScaler().fit(x)
    model = fixed_logistic_regression().fit(scaler.transform(x), y)
    report = {
        "converged": bool(int(model.n_iter_[0]) < int(model.max_iter)),
        "n_iter": int(model.n_iter_[0]),
        "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
        "classes": model.classes_.tolist(),
        "configuration": {"penalty": "l2", "C": 1.0, "class_weight": "balanced", "solver": "liblinear", "max_iter": 2000, "random_state": 42},
    }
    return scaler, model, report


def exported_probability(feature: np.ndarray, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> float:
    standardized = (np.asarray(feature, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / np.asarray(scale, dtype=np.float64)
    logit = float(np.asarray(coef, dtype=np.float64).reshape(1, -1) @ standardized.reshape(-1, 1) + np.asarray(intercept).reshape(-1)[0])
    return float(1.0 / (1.0 + np.exp(-logit)))


def zero_fp_threshold(negative_probabilities: Sequence[float]) -> tuple[float, float]:
    if not negative_probabilities:
        raise RuntimeError("SEMANTIC_ROUTER_INVALID_ENGINEERING_RUN: empty calibration negatives")
    max_neg = float(max(negative_probabilities))
    return max_neg, float(np.nextafter(max_neg, 1.0))


def stable_negative_cap(rows: Sequence[Mapping[str, Any]], positive_count: int, ratio: int = 8) -> list[dict[str, Any]]:
    limit = int(positive_count) * int(ratio)
    return sorted((dict(row) for row in rows), key=lambda row: stable_hash(row.get("prototype_id"), row.get("candidate_id"), row.get("equivalence_key")))[:limit]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** .5) / denom
    return [center - half, center + half]


def save_model_npz(path: Path, branch: str, pcas: Mapping[str, Any], scaler: Any, model: Any) -> None:
    arrays: dict[str, np.ndarray] = {
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float32),
        "coef": np.asarray(model.coef_, dtype=np.float32),
        "intercept": np.asarray(model.intercept_, dtype=np.float32),
        "branch": np.asarray(branch),
    }
    for name in MODALITIES:
        arrays[f"pca_{name}_mean"] = np.asarray(pcas[name].mean_, dtype=np.float32)
        arrays[f"pca_{name}_components"] = np.asarray(pcas[name].components_, dtype=np.float32)
        arrays[f"pca_{name}_variance"] = np.asarray(pcas[name].explained_variance_, dtype=np.float32)
    np.savez(path, **arrays)


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
