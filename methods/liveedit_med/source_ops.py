"""Source-equation operations separated from model adaptation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


SIM_SCALE = 1 / (1024 ** 0.5)


@dataclass(frozen=True)
class BaseRoutePlan:
    reason: str = "EMPTY_CANDIDATE_BASE_BYPASS"
    candidate_mask: torch.Tensor | None = None
    visual_scores: torch.Tensor | None = None
    sentinel_score: torch.Tensor | None = None


@dataclass(frozen=True)
class RoutePlan:
    candidate_mask: torch.Tensor
    visual_scores: torch.Tensor
    sentinel_score: torch.Tensor
    text_scores: torch.Tensor
    relative_weights: torch.Tensor
    absolute_weights: torch.Tensor
    final_weights: torch.Tensor


def generate_expert_and_keys(edit_extractor: nn.Module, moegen_c: nn.Module, moegen_r: nn.Module,
                             vision_reps: torch.Tensor, question_reps: torch.Tensor,
                             answer_reps: torch.Tensor):
    evr = edit_extractor.extract_vision(question_reps, vision_reps)
    eqr = edit_extractor.extract_query(question_reps)
    edit_reps = torch.cat([vision_reps, question_reps, answer_reps], 1)
    return eqr, evr, moegen_c(edit_reps), moegen_r(edit_reps)


def apply_low_rank_expert_residual(hidden: torch.Tensor, moe_cs: torch.Tensor, moe_rs: torch.Tensor,
                                   weights: torch.Tensor, instant_norm: nn.Module) -> torch.Tensor:
    assert len(hidden) == len(weights) == 1
    value = instant_norm(hidden)[0]
    value = torch.relu(torch.einsum("ld,mrd->lmr", value, moe_cs))
    value = torch.einsum("lmr,mrd,m->ld", value, moe_rs, weights[0])
    return value.unsqueeze(0)


def compute_text_soft_weights(input_text_keys: torch.Tensor, edit_text_keys: torch.Tensor,
                              split: bool = False):
    score = torch.einsum("ned,med->nme", input_text_keys, edit_text_keys).mean(2) * SIM_SCALE
    relative = torch.softmax(score, 1)
    absolute = torch.sigmoid(score)
    return (relative, absolute) if split else relative * absolute


def route_repository(input_extractor: nn.Module, question_reps: torch.Tensor, vision_reps: torch.Tensor,
                     edit_visual_keys: torch.Tensor, edit_text_keys: torch.Tensor) -> BaseRoutePlan | RoutePlan:
    input_visual = input_extractor.extract_vision(question_reps, vision_reps)
    visual_score = torch.einsum("bed,med->bme", input_visual, edit_visual_keys).mean(2) * SIM_SCALE
    sentinel_visual = input_extractor.extract_from_visprot(question_reps)
    sentinel_score = torch.einsum("bed,bed->be", input_visual, sentinel_visual).mean(1, True) * SIM_SCALE
    candidate = (visual_score > sentinel_score)[0]
    if int(candidate.sum()) == 0:
        return BaseRoutePlan(candidate_mask=candidate, visual_scores=visual_score,
                             sentinel_score=sentinel_score)
    input_text = input_extractor.extract_query(question_reps)
    text_score = torch.einsum("ned,med->nme", input_text, edit_text_keys[candidate]).mean(2) * SIM_SCALE
    relative, absolute = torch.softmax(text_score, 1), torch.sigmoid(text_score)
    return RoutePlan(candidate, visual_score, sentinel_score, text_score, relative, absolute, relative * absolute)


def direct_expert_residual(hidden: torch.Tensor, raw_c: torch.Tensor, raw_r: torch.Tensor,
                           instant_norm: nn.Module, lora_scale: float = 5.0) -> torch.Tensor:
    rank = int(raw_c.shape[0]); scale = lora_scale * rank ** 0.5
    return apply_low_rank_expert_residual(hidden, (raw_c / scale).unsqueeze(0),
                                          (raw_r / scale).unsqueeze(0),
                                          torch.ones(1, 1, device=hidden.device, dtype=hidden.dtype), instant_norm)


def source_routing_losses(input_extractor: nn.Module, edit_extractor: nn.Module,
                          neighbor_inputs: Sequence[tuple[torch.Tensor, torch.Tensor]],
                          neighbor_edits: Sequence[tuple[torch.Tensor, torch.Tensor]],
                          prototype_inputs: Sequence[tuple[torch.Tensor, torch.Tensor]],
                          prototype_edits: Sequence[tuple[torch.Tensor, torch.Tensor]], eps: float = 1e-8):
    def hard(inputs, edits):
        ivr = torch.cat([input_extractor.extract_vision(q, v) for v, q in inputs], 0)
        evr = torch.cat([edit_extractor.extract_vision(q, v) for v, q in edits], 0)
        sim = torch.einsum("bed,med->bme", ivr, evr).mean(2) * SIM_SCALE
        prot = torch.cat([input_extractor.extract_from_visprot(q) for _v, q in inputs])
        sim_prot = torch.einsum("bed,bed->be", ivr, prot).mean(1, True) * SIM_SCALE
        return torch.softmax(torch.cat([sim, sim_prot], 1), 1)
    neighbor = hard(neighbor_inputs, neighbor_edits)
    prototype = hard(prototype_inputs, prototype_edits)
    return -torch.log(torch.diag(neighbor) + eps).mean(), -torch.log(prototype[:, -1] + eps).mean()


def source_soft_losses(input_keys: torch.Tensor, edit_keys: torch.Tensor, eps: float = 1e-8):
    relative, absolute = compute_text_soft_weights(input_keys, edit_keys, split=True)
    relative_loss = -torch.log(torch.diag(relative) + eps).mean()
    pos, neg = torch.diag(absolute), torch.diag(absolute.roll(1, 1))
    absolute_loss = -(torch.log(pos + eps) + torch.log(1 - neg + eps)).mean()
    return relative_loss, absolute_loss


def deterministic_source_masks(request_counts: Sequence[int], reliability_indices: Sequence[int], seed: int = 43):
    rng = np.random.default_rng(seed); counts = torch.tensor(request_counts)
    cols = int(counts.sum()); starts = torch.cumsum(torch.cat([torch.tensor([0]), counts[:-1]]), 0)
    rel = torch.zeros(len(counts), cols, dtype=torch.bool)
    rel[torch.arange(len(counts)), starts + torch.tensor(reliability_indices)] = True
    idx = torch.arange(cols).expand(len(counts), cols)
    gen = (idx >= starts[:, None]) & (idx < (starts + counts)[:, None])
    loc = torch.zeros_like(gen)
    prefixes = []
    for i in range(len(counts)):
        ns = rng.integers(0, cols + 1, 3); prefixes.append(ns.tolist())
        rel[i, :ns[0]] = True; gen[i, :ns[1]] = True; loc[i, :ns[2]] = True
    return rel, gen, loc, prefixes
