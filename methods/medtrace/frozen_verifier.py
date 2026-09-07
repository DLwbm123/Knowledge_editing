"""Frozen MedTRACE routing diagnostics; no backbone or CP parameters are trained."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import AsymmetricCPExpert


def l2_normalize(value: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(epsilon)


@dataclass(frozen=True)
class VerifierFeatures:
    cp_prompt: torch.Tensor
    cp_visual: torch.Tensor
    pre_cp_prompt: torch.Tensor
    pre_cp_visual: torch.Tensor
    pooling_weights: torch.Tensor


def verifier_features(
    expert: AsymmetricCPExpert,
    prompt: torch.Tensor,
    visual: torch.Tensor,
) -> VerifierFeatures:
    """Return matched CP4/pre-CP features using one frozen-Q visual pooling."""
    if prompt.ndim != 1 or visual.ndim != 2 or prompt.shape[-1] != visual.shape[-1]:
        raise ValueError("verifier expects one prompt vector and a visual-token matrix")
    normalized_prompt = expert.normalize_activation(prompt)
    normalized_visual = expert.normalize_activation(visual)
    q = expert.input_basis()
    prompt_cp_raw = normalized_prompt @ q
    visual_cp_tokens = normalized_visual @ q
    weights = F.softmax((l2_normalize(visual_cp_tokens) @ l2_normalize(prompt_cp_raw)) / 0.1, dim=0)
    pre_cp_visual_raw = (weights[:, None] * normalized_visual).sum(dim=0)
    cp_visual_raw = (weights[:, None] * visual_cp_tokens).sum(dim=0)
    projected = pre_cp_visual_raw @ q
    if not torch.allclose(cp_visual_raw, projected, rtol=1e-5, atol=1e-5):
        raise RuntimeError("matched visual pooling no longer commutes with frozen Q")
    return VerifierFeatures(
        cp_prompt=l2_normalize(prompt_cp_raw),
        cp_visual=l2_normalize(cp_visual_raw),
        pre_cp_prompt=l2_normalize(normalized_prompt),
        pre_cp_visual=l2_normalize(pre_cp_visual_raw),
        pooling_weights=weights,
    )


class LinearApplicabilityVerifier(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.question = nn.Linear(width, 1)
        self.image = nn.Linear(width, 1)
        nn.init.zeros_(self.question.weight); nn.init.zeros_(self.question.bias)
        nn.init.zeros_(self.image.weight); nn.init.zeros_(self.image.bias)

    def logits(self, question: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.question(question).squeeze(-1), self.image(image).squeeze(-1)

    def decisions(
        self,
        question: torch.Tensor,
        image: torch.Tensor,
        threshold_question: float,
        threshold_image: float,
    ) -> torch.Tensor:
        q, v = self.logits(question, image)
        return (q > threshold_question) & (v > threshold_image)


def _group_mean(values: torch.Tensor, groups: Sequence[str]) -> torch.Tensor:
    if len(values) != len(groups) or not len(groups):
        raise ValueError("group-balanced loss requires aligned nonempty values and groups")
    unique = tuple(dict.fromkeys(groups))
    return torch.stack([values[[index for index, group in enumerate(groups) if group == name]].mean() for name in unique]).mean()


def verifier_loss(
    verifier: LinearApplicabilityVerifier,
    *,
    question_features: torch.Tensor,
    question_labels: torch.Tensor,
    question_groups: Sequence[str],
    image_features: torch.Tensor,
    image_labels: torch.Tensor,
    image_groups: Sequence[str],
    matched_pairs: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q_logits, v_logits = verifier.logits(question_features, image_features)
    q_pos = question_labels.bool(); q_neg = ~q_pos
    v_pos = image_labels.bool(); v_neg = ~v_pos
    if not all((q_pos.any(), q_neg.any(), v_pos.any(), v_neg.any(), bool(matched_pairs))):
        raise ValueError("verifier supervision is missing a required class or matched pair")
    lq = 0.5 * _group_mean(F.softplus(-q_logits[q_pos]), [g for g, keep in zip(question_groups, q_pos.tolist(), strict=True) if keep])
    lq += 0.5 * _group_mean(F.softplus(q_logits[q_neg]), [g for g, keep in zip(question_groups, q_neg.tolist(), strict=True) if keep])
    lv = 0.5 * _group_mean(F.softplus(-v_logits[v_pos]), [g for g, keep in zip(image_groups, v_pos.tolist(), strict=True) if keep])
    lv += 0.5 * _group_mean(F.softplus(v_logits[v_neg]), [g for g, keep in zip(image_groups, v_neg.tolist(), strict=True) if keep])
    pair_margin = torch.stack([F.softplus(1 - (v_logits[positive] - v_logits[negative])) for positive, negative in matched_pairs]).mean()
    lv += 0.5 * pair_margin
    regularization = 1e-3 * (verifier.question.weight.square().sum() + verifier.image.weight.square().sum())
    return lq + lv + regularization, {"question": lq, "image": lv, "pair_margin": pair_margin, "l2": regularization}


def train_verifier(
    *,
    question_features: torch.Tensor,
    question_labels: torch.Tensor,
    question_groups: Sequence[str],
    image_features: torch.Tensor,
    image_labels: torch.Tensor,
    image_groups: Sequence[str],
    matched_pairs: Sequence[tuple[int, int]],
    steps: int = 800,
) -> tuple[LinearApplicabilityVerifier, list[dict[str, float | int]]]:
    if steps != 800:
        raise ValueError("the frozen verifier budget is exactly 800 steps")
    verifier = LinearApplicabilityVerifier(question_features.shape[-1]).to(question_features.device)
    optimizer = torch.optim.Adam(verifier.parameters(), lr=1e-2, weight_decay=0)
    curve = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = verifier_loss(
            verifier,
            question_features=question_features, question_labels=question_labels,
            question_groups=question_groups, image_features=image_features,
            image_labels=image_labels, image_groups=image_groups, matched_pairs=matched_pairs,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite frozen-verifier optimization")
        optimizer.step()
        if step in {1, 80, 320, 800}:
            curve.append({"step": step, "loss": float(loss.item()), "gradient_norm": float(grad_norm.item()), **{name: float(value.item()) for name, value in parts.items()}})
    return verifier, curve


def mean_decision(prompt_score: torch.Tensor, visual_score: torch.Tensor, threshold: float) -> torch.Tensor:
    return (0.5 * prompt_score + 0.5 * visual_score) > threshold


def conjunction_decision(
    prompt_score: torch.Tensor,
    visual_score: torch.Tensor,
    threshold_prompt: float,
    threshold_visual: float,
) -> torch.Tensor:
    return (prompt_score > threshold_prompt) & (visual_score > threshold_visual)


def _thresholds(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("calibration requires scores")
    unique = sorted(set(float(value) for value in values))
    return [math.nextafter(unique[0], -math.inf), *unique, math.nextafter(unique[-1], math.inf)]


def calibrate_decisions(
    positive: Sequence[tuple[float, ...]],
    hard: Sequence[tuple[float, ...]],
    broad: Sequence[tuple[float, ...]],
) -> dict[str, dict[str, Any]]:
    """Calibrate one score or an AND pair with a frozen deterministic grid."""
    if not positive or not hard or not broad:
        raise ValueError("positive, hard and broad calibration groups are required")
    width = len(positive[0])
    if width not in {1, 2} or any(len(row) != width for row in (*positive, *hard, *broad)):
        raise ValueError("calibration accepts aligned one-dimensional or two-dimensional scores")
    axes = [_thresholds([row[index] for row in (*positive, *hard, *broad)]) for index in range(width)]
    grids = [(value,) for value in axes[0]] if width == 1 else [(left, right) for left in axes[0] for right in axes[1]]

    def decision(row: tuple[float, ...], threshold: tuple[float, ...]) -> bool:
        return all(value > boundary for value, boundary in zip(row, threshold, strict=True))

    def metrics(threshold: tuple[float, ...]) -> dict[str, Any]:
        rate = lambda rows: sum(decision(row, threshold) for row in rows) / len(rows)
        return {"thresholds": list(threshold), "positive_tpr": rate(positive), "hard_fpr": rate(hard), "broad_fpr": rate(broad)}

    rows = [metrics(grid) for grid in grids]
    safety = [row for row in rows if row["hard_fpr"] == row["broad_fpr"] == 0]
    primary = max(safety, key=lambda row: (row["positive_tpr"], *row["thresholds"]))
    covered = [row for row in rows if row["positive_tpr"] >= 0.90]
    if not covered:
        raise RuntimeError("COVERAGE90 has no feasible threshold pair")
    secondary = min(covered, key=lambda row: (row["hard_fpr"], row["broad_fpr"], -row["positive_tpr"], *[-value for value in row["thresholds"]]))
    return {
        "PRIMARY_SAFETY_FIRST": primary,
        "SECONDARY_COVERAGE90": secondary,
        "tie_break": "maximize coverage then higher thresholds for PRIMARY; hard FPR, broad FPR, coverage, higher thresholds for SECONDARY",
    }
