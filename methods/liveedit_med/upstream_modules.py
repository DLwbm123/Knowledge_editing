"""Exact tensor definitions ported from LiveEdit commit 3615a37.

Source: editor/vllm_editors/liveedit/modules.py.  Names are intentionally kept
compatible so state dictionaries can be loaded in both directions.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def reset_layer_norm(module: nn.LayerNorm) -> nn.LayerNorm:
    """Apply PyTorch defaults despite LLaVA's process-global reset monkeypatch."""
    if module.elementwise_affine:
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


def reset_linear(module: nn.Linear) -> nn.Linear:
    """Apply PyTorch Linear defaults despite LLaVA's process-global monkeypatch."""
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(module.bias, -bound, bound)
    return module


class Attention(nn.Module):
    def __init__(self, inp1_dim: int, inp2_dim: int, qk_dim: int, v_dim: int, head_n: int,
                 add_bias_q: bool = True, add_bias_k: bool = True, add_bias_v: bool = True) -> None:
        super().__init__()
        assert qk_dim % head_n == 0
        self.head_n = head_n
        self.qk_head_dim = qk_dim // head_n
        self.v_head_dim = v_dim // head_n
        self.scale_factor = 1 / (self.qk_head_dim ** 0.5)
        self.q_mlp = nn.Linear(inp1_dim, qk_dim, add_bias_q)
        self.k_mlp = nn.Linear(inp2_dim, qk_dim, add_bias_k)
        self.v_mlp = nn.Linear(inp2_dim, v_dim, add_bias_v)
        self.reset_parameters()

    def forward(self, inp1: torch.Tensor, inp2: torch.Tensor, rescale_with_score: bool = False):
        b, l1, _ = inp1.shape
        b, l2, _ = inp2.shape
        q = self.q_mlp(inp1).reshape(b, l1, self.head_n, self.qk_head_dim)
        k = self.k_mlp(inp2).reshape(b, l2, self.head_n, self.qk_head_dim)
        v = self.v_mlp(inp2).reshape(b, l2, self.head_n, self.v_head_dim)
        score = torch.softmax(torch.einsum("blhd,bmhd->blmh", q, k) * self.scale_factor, 2)
        result = torch.einsum("blmh,bmhd->blhd", score, v)
        if not rescale_with_score:
            return result.reshape(b, l1, -1)
        result = result / torch.sum(score ** 2, 2).unsqueeze(-1) ** 0.5
        return result.reshape(b, l1, -1)

    def reset_parameters(self):
        reset_linear(self.q_mlp); reset_linear(self.k_mlp); reset_linear(self.v_mlp)


class QVExtractor(nn.Module):
    def __init__(self, eqe_n: int, inpt_reps_dim: int, module_dim: int, cross_att_head_n: int,
                 vision_tok_n: int, vis_prot: bool = False) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(inpt_reps_dim)
        self.eqe1 = nn.Parameter(torch.zeros(1, eqe_n, module_dim))
        self.ca_query_info_ext1 = Attention(module_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        self.ca_vision_info_ext = Attention(module_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        self.layer_norm2 = nn.LayerNorm(inpt_reps_dim)
        self.eqe2 = nn.Parameter(torch.zeros(1, eqe_n, module_dim))
        self.ca_query_info_ext2 = Attention(module_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        if vis_prot:
            self.vis_rep_prot = nn.Parameter(torch.zeros(1, vision_tok_n, inpt_reps_dim))
        self.reset_parameters()

    def extract_vision(self, query_reps: torch.Tensor, vision_reps: torch.Tensor) -> torch.Tensor:
        assert len(query_reps) == len(vision_reps) == 1
        query_reps, vision_reps = self.layer_norm1(query_reps), self.layer_norm1(vision_reps)
        eqr = self.ca_query_info_ext1(self.eqe1, query_reps)
        return self.ca_vision_info_ext(eqr, vision_reps)

    def extract_query(self, query_reps: torch.Tensor) -> torch.Tensor:
        assert len(query_reps) == 1
        return self.ca_query_info_ext2(self.eqe2, self.layer_norm2(query_reps))

    def extract_from_visprot(self, query_reps: torch.Tensor):
        return self.extract_vision(query_reps, self.vis_rep_prot)

    def forward(self):
        raise RuntimeError("QVExtractor has no monolithic forward; call an extraction method")

    def reset_parameters(self):
        reset_layer_norm(self.layer_norm1); nn.init.kaiming_normal_(self.eqe1)
        self.ca_query_info_ext1.reset_parameters(); self.ca_vision_info_ext.reset_parameters()
        reset_layer_norm(self.layer_norm2); nn.init.kaiming_normal_(self.eqe2)
        self.ca_query_info_ext2.reset_parameters()
        if hasattr(self, "vis_rep_prot"):
            nn.init.kaiming_normal_(self.vis_rep_prot)


class LowRankGenerator(nn.Module):
    def __init__(self, lora_dim: int, lora_rank: int, lora_scale: float, inpt_reps_dim: int,
                 module_dim: int, cross_att_head_n: int) -> None:
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(1, lora_rank, module_dim))
        self.ca_lora = Attention(module_dim, inpt_reps_dim, module_dim, lora_dim, cross_att_head_n)
        self.layer_norm = nn.LayerNorm(inpt_reps_dim)
        self.scale = 1 / (lora_scale * lora_rank ** 0.5)
        self.reset_parameters()

    def forward(self, inpt_reps: torch.Tensor):
        assert len(inpt_reps) == 1
        return self.ca_lora(self.phi, self.layer_norm(inpt_reps)) * self.scale

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.phi); self.ca_lora.reset_parameters(); reset_layer_norm(self.layer_norm)
