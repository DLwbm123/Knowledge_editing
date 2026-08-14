"""Source-faithful LiveEdit module container and loss helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .source_ops import (apply_low_rank_expert_residual, compute_text_soft_weights,
                         deterministic_source_masks, generate_expert_and_keys,
                         source_routing_losses, source_soft_losses)
from .upstream_modules import LowRankGenerator, QVExtractor, reset_layer_norm


@dataclass(frozen=True)
class LiveEditMedicalConfig:
    module_dim: int = 1024
    cross_att_head_n: int = 8
    lora_rank: int = 4
    eqe_n: int = 4
    lora_scale: float = 5.0
    llm_mid_dim: int = 4096
    edit_layer_i: int = 21
    learning_rate: float = 1e-4
    lr_cut_it: tuple[int, ...] = (10000,)
    lr_cut_rate: float = .1


class LiveEditMedicalModules(nn.Module):
    def __init__(self, config: LiveEditMedicalConfig, vision_tokens: int = 576):
        super().__init__(); c = config; self.config = c
        self.edit_extractor = QVExtractor(c.eqe_n, c.llm_mid_dim, c.module_dim, c.cross_att_head_n, vision_tokens, False)
        self.input_extractor = QVExtractor(c.eqe_n, c.llm_mid_dim, c.module_dim, c.cross_att_head_n, vision_tokens, True)
        self.moegen_c = LowRankGenerator(c.llm_mid_dim, c.lora_rank, c.lora_scale, c.llm_mid_dim, c.module_dim, c.cross_att_head_n)
        self.moegen_r = LowRankGenerator(c.llm_mid_dim, c.lora_rank, c.lora_scale, c.llm_mid_dim, c.module_dim, c.cross_att_head_n)
        self.instant_reps_norm = reset_layer_norm(nn.LayerNorm(c.llm_mid_dim))

    def generated_edit(self, vision, question, answer):
        return generate_expert_and_keys(self.edit_extractor, self.moegen_c, self.moegen_r, vision, question, answer)

    def optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.learning_rate)
        cuts = torch.tensor(self.config.lr_cut_it)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: self.config.lr_cut_rate ** int((step > cuts).sum()))
        return optimizer, scheduler

    def assert_trainable_boundary(self, base_model: nn.Module):
        if any(parameter.requires_grad for parameter in base_model.parameters()):
            raise RuntimeError("LIVEEDIT_MED_UNAUTHORIZED_BACKBONE_GRADIENT")
        expected = {"edit_extractor", "input_extractor", "moegen_c", "moegen_r", "instant_reps_norm"}
        if {name.split(".", 1)[0] for name, p in self.named_parameters() if p.requires_grad} != expected:
            raise RuntimeError("LIVEEDIT_MED_TRAINABLE_BOUNDARY_MISMATCH")


def source_total_loss(rel: torch.Tensor, generalities: Sequence[torch.Tensor], localities: Sequence[torch.Tensor],
                      soft_relative: torch.Tensor, soft_absolute: torch.Tensor,
                      hard_neighbor: torch.Tensor, hard_prototype: torch.Tensor):
    return rel + sum(generalities) + sum(localities) + soft_relative + soft_absolute + hard_neighbor + hard_prototype
