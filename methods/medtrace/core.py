"""MedTRACE V0.2 zero-effect CP core; no benchmark claims."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def factor_pair(size: int) -> tuple[int, int]:
    if size < 1:
        raise ValueError("factorized dimension must be positive")
    pairs = [(divisor, size // divisor) for divisor in range(1, math.isqrt(size) + 1) if size % divisor == 0]
    return min(pairs, key=lambda pair: (abs(math.log(pair[0]) - math.log(pair[1])), pair[0]))


class AsymmetricCPExpert(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int, *, beta: float = 1.0, epsilon: float = 1e-6):
        super().__init__()
        if rank < 1:
            raise ValueError("CP rank must be positive")
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        self.p_in, self.q_in = factor_pair(d_in)
        self.p_out, self.q_out = factor_pair(d_out)
        self.beta, self.epsilon = float(beta), float(epsilon)
        self.u_in = nn.Parameter(torch.randn(self.p_in, rank) / math.sqrt(self.p_in))
        self.v_in = nn.Parameter(torch.randn(self.q_in, rank) / math.sqrt(self.q_in))
        self.u_out = nn.Parameter(torch.randn(self.p_out, rank) / math.sqrt(self.p_out))
        self.v_out = nn.Parameter(torch.randn(self.q_out, rank) / math.sqrt(self.q_out))
        self.rho = nn.Parameter(torch.zeros(rank))

    def normalize_activation(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.shape[-1] != self.d_in:
            raise ValueError(f"expected activation width {self.d_in}, got {activation.shape[-1]}")
        return activation / (activation.float().square().mean(dim=-1, keepdim=True).sqrt().to(activation.dtype) + self.epsilon)

    @staticmethod
    def _columns(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ir,jr->ijr", left, right).reshape(left.shape[0] * right.shape[0], left.shape[1])

    def input_basis(self) -> torch.Tensor:
        return self._columns(self.u_in, self.v_in)

    def output_basis(self) -> torch.Tensor:
        return self._columns(self.u_out, self.v_out)

    def component_contraction(self, activation: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_activation(activation)
        return normalized @ self.input_basis()

    def residual(self, activation: torch.Tensor) -> torch.Tensor:
        contraction = self.component_contraction(activation)
        return (contraction * self.rho) @ self.output_basis().T * (self.beta / math.sqrt(self.rank))

    def materialize_dense(self) -> torch.Tensor:
        return self.output_basis() @ torch.diag(self.rho) @ self.input_basis().T * (self.beta / math.sqrt(self.rank))

    def intrinsic_score(self, activation: torch.Tensor) -> torch.Tensor:
        return self.component_contraction(activation).abs().mean(dim=-1)

    @torch.no_grad()
    def normalize_factors_(self, *, tolerance: float = 1e-5) -> None:
        before = self.materialize_dense().float()
        for component in range(self.rank):
            scale = self.rho.new_tensor(1.0)
            for factor in (self.u_in, self.v_in, self.u_out, self.v_out):
                norm = factor[:, component].norm().clamp_min(self.epsilon)
                factor[:, component].div_(norm)
                scale.mul_(norm)
            self.rho[component].mul_(scale)
            q = torch.outer(self.u_in[:, component], self.v_in[:, component]).reshape(-1)
            if q[q.abs().argmax()] < 0:
                self.u_in[:, component].neg_()
                self.rho[component].neg_()
        if not torch.allclose(before, self.materialize_dense().float(), rtol=tolerance, atol=tolerance):
            raise RuntimeError("CP normalization changed the materialized residual map")


class MedTraceLayerHook:
    def __init__(self, layer: nn.Module, expert: AsymmetricCPExpert):
        self.layer, self.expert = layer, expert
        self.enabled = False
        self.token_mask: torch.Tensor | None = None
        self._handle = None

    def attach(self) -> None:
        if self._handle is not None:
            raise RuntimeError("MedTRACE hook is already attached")
        self._handle = self.layer.register_forward_hook(self._forward_hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def set_request_routing(self, token_mask: torch.Tensor) -> None:
        self.token_mask = token_mask.bool()
        self.enabled = True

    def clear_request_routing(self) -> None:
        self.enabled = False
        self.token_mask = None

    def _forward_hook(self, _module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return output
        if not args or self.token_mask is None or args[0].shape[:-1] != self.token_mask.shape:
            raise RuntimeError("assistant-only MedTRACE token mask does not match the layer activation")
        residual = self.expert.residual(args[0]).to(output.dtype)
        return output + residual * self.token_mask.to(output.device).unsqueeze(-1)


@dataclass(frozen=True)
class ScopeCalibration:
    threshold: float
    score_scale: float
    true_positive_rate: float
    false_positive_rate: float


def calibrate_threshold(positive: list[float], negative: list[float], *, target_fpr: float, scale_floor: float = 1e-3) -> ScopeCalibration:
    if not positive or not negative or not 0 <= target_fpr <= 1:
        raise ValueError("nonempty calibration scores and a valid target FPR are required")
    candidates = sorted(set(positive + negative))
    candidates.append(math.nextafter(max(candidates), math.inf))
    feasible = []
    for threshold in candidates:
        tpr = sum(score > threshold for score in positive) / len(positive)
        fpr = sum(score > threshold for score in negative) / len(negative)
        if fpr <= target_fpr:
            feasible.append((tpr, threshold, fpr))
    if not feasible:
        raise RuntimeError("scope calibration has no feasible threshold")
    tpr, threshold, fpr = max(feasible, key=lambda item: (item[0], item[1]))
    median = torch.tensor(negative).median().item()
    mad = torch.tensor([abs(score - median) for score in negative]).median().item()
    return ScopeCalibration(threshold, max(1.4826 * mad, scale_floor), tpr, fpr)
