"""Paper-compatible macro metrics based exclusively on free-generation outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProbeOutcome:
    pre_correct: bool
    post_correct: bool


def reliability(target_post_correct: bool) -> float:
    """Reliability for one originally-wrong target, scored after free generation."""
    return float(target_post_correct)


def locality(probes: Iterable[ProbeOutcome]) -> float | None:
    """Only originally-correct probes count; return None when no eligible probe exists."""
    eligible = [p for p in probes if p.pre_correct]
    if not eligible:
        return None
    flips = sum(not p.post_correct for p in eligible)
    return 1.0 - flips / len(eligible)


def generality(probes: Iterable[ProbeOutcome]) -> float | None:
    """Only originally-wrong probes count; success is a free-generation fix."""
    eligible = [p for p in probes if not p.pre_correct]
    if not eligible:
        return None
    fixes = sum(p.post_correct for p in eligible)
    return fixes / len(eligible)


def macro_average(per_edit_scores: Iterable[float | None]) -> float | None:
    scores = [x for x in per_edit_scores if x is not None]
    return sum(scores) / len(scores) if scores else None


def harmonic_mean(scores: Sequence[float | None]) -> float | None:
    values = [x for x in scores if x is not None]
    if not values or any(x < 0 or x > 1 for x in values):
        raise ValueError("scores must be nonempty values in [0, 1]")
    if any(x == 0 for x in values):
        return 0.0
    return len(values) / sum(1.0 / x for x in values)
