from __future__ import annotations

import inspect
import json
import logging
import math
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger(__name__)

_CHECKPOINT_FRAME_NAMES = frozenset({"recompute_fn", "_run_fn_with_dynamo_disabled"})


def _inside_activation_checkpoint_recompute() -> bool:
    frame = inspect.currentframe()
    depth = 0
    try:
        current = frame
        while current is not None and depth < 256:
            depth += 1
            code = current.f_code
            if code.co_name in _CHECKPOINT_FRAME_NAMES:
                path = code.co_filename.replace("\\", "/")
                if "torch/utils/checkpoint" in path:
                    return True
            current = current.f_back
    finally:
        del frame
    return False


@dataclass
class SAMEEditConfig:
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    expert_num: int = 4
    top_k: int = 1
    current_edit: int = 0
    oracle_edit_routing: bool = True
    eval_oracle_edit_routing: bool = False
    learned_hidden_routing: bool = True
    adaptive_activation: bool = False
    tau_score: float = 0.1
    curvature_mode: str = "off"
    curvature_mu: float = 0.9
    curvature_max_grad_ratio: float = 10.0
    allow_missing_covariance: bool = True
    spectral_router: bool = False
    router_start_step: int = 10
    router_scaling_factor: float = 0.2
    window_size: int = 3
    max_components: int = 64
    cumulative_energy_ratio: float = 0.9
    target_modules: str = "last4_down_proj"
    target_module_patterns: List[str] = field(default_factory=list)
    target_last_n_layers: int = 4
    exclude_vision_tower: bool = True
    exclude_module_path_segments: List[str] = field(
        default_factory=lambda: [
            "vision_tower",
            "visual_encoder",
            "vision_model",
            "image_encoder",
            "clip",
            "mm_projector.vision",
        ]
    )
    update_covariance: bool = True
    route_loss_weight: float = 0.0
    adapter_dtype: str = "float32"
    log_rank_remainder: bool = True

    @classmethod
    def from_hparams(cls, hparams: Any) -> "SAMEEditConfig":
        config = cls()
        for field_name in asdict(config).keys():
            prefixed = f"same_edit_{field_name}"
            if hasattr(hparams, prefixed):
                value = getattr(hparams, prefixed)
                if field_name == "exclude_module_path_segments" and value in (None, []):
                    continue
                setattr(config, field_name, value)
            elif hasattr(hparams, field_name):
                value = getattr(hparams, field_name)
                if field_name == "exclude_module_path_segments" and value in (None, []):
                    continue
                setattr(config, field_name, value)
        if hasattr(hparams, "lora_r"):
            config.lora_r = int(getattr(hparams, "lora_r"))
        if hasattr(hparams, "lora_alpha"):
            config.lora_alpha = float(getattr(hparams, "lora_alpha"))
        if hasattr(hparams, "lora_dropout"):
            config.lora_dropout = float(getattr(hparams, "lora_dropout"))
        if hasattr(hparams, "expert_num"):
            config.expert_num = int(getattr(hparams, "expert_num"))
        if hasattr(hparams, "top_k"):
            config.top_k = int(getattr(hparams, "top_k"))
        if hasattr(hparams, "cur_edit"):
            config.current_edit = int(getattr(hparams, "cur_edit"))
        return config.normalized()

    def normalized(self) -> "SAMEEditConfig":
        self.lora_r = int(self.lora_r)
        self.expert_num = max(1, int(self.expert_num))
        self.top_k = max(1, int(self.top_k))
        self.current_edit = int(self.current_edit)
        self.max_components = max(1, int(self.max_components))
        self.window_size = max(1, int(self.window_size))
        self.target_last_n_layers = max(1, int(self.target_last_n_layers))
        if isinstance(self.curvature_mode, bool):
            self.curvature_mode = "prism" if self.curvature_mode else "off"
        self.curvature_mode = str(self.curvature_mode).lower()
        if self.curvature_mode in {"false", "none", "null", "0"}:
            self.curvature_mode = "off"
        if self.curvature_mode in {"true", "1"}:
            self.curvature_mode = "prism"
        if self.curvature_mode not in {"off", "prism", "safe"}:
            raise ValueError(f"Unsupported SAME-Edit curvature_mode={self.curvature_mode!r}")
        return self

    @property
    def per_expert_r(self) -> int:
        return max(1, int(self.lora_r) // max(1, int(self.expert_num)))

    @property
    def effective_rank(self) -> int:
        return self.per_expert_r * max(1, int(self.expert_num))

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["per_expert_r"] = self.per_expert_r
        payload["effective_rank"] = self.effective_rank
        return payload


class SAMEEditExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features, bias=False)
        self.weight = self.mlp.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SAMEEditLinearA(nn.Module):
    def __init__(self, in_features: int, per_expert_r: int, expert_num: int):
        super().__init__()
        self.expert_num = int(expert_num)
        self.per_expert_r = int(per_expert_r)
        self.loraA = nn.ModuleList(
            [SAMEEditExpert(in_features, self.per_expert_r) for _ in range(self.expert_num)]
        )

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        weights = routing_weights.to(device=x.device, dtype=x.dtype)
        output = self.loraA[0](x) * weights[0]
        for idx in range(1, self.expert_num):
            output = output + self.loraA[idx](x) * weights[idx]
        return output


class SAMEEditLinearB(nn.Module):
    def __init__(self, per_expert_r: int, out_features: int, expert_num: int):
        super().__init__()
        self.expert_num = int(expert_num)
        self.per_expert_r = int(per_expert_r)
        self.loraB = nn.ModuleList(
            [SAMEEditExpert(self.per_expert_r, out_features) for _ in range(self.expert_num)]
        )

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        weights = routing_weights.to(device=x.device, dtype=x.dtype)
        output = self.loraB[0](x) * weights[0]
        for idx in range(1, self.expert_num):
            output = output + self.loraB[idx](x) * weights[idx]
        return output


class SAMEEditLayer(nn.Module):
    def set_current_edit(self, edit_index: int) -> None:
        raise NotImplementedError

    def save_task_covariance_snapshot(self) -> None:
        raise NotImplementedError


class SAMEEditLinear(SAMEEditLayer):
    def __init__(
        self,
        base_linear: nn.Linear,
        config: SAMEEditConfig,
        module_name: str = "",
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"SAMEEditLinear can only wrap nn.Linear, got {type(base_linear)}")
        self.module_name = module_name
        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.expert_num = int(config.expert_num)
        self.per_expert_r = int(config.per_expert_r)
        self.lora_r = int(config.lora_r)
        self.lora_alpha = float(config.lora_alpha)
        self.scaling = float(config.lora_alpha) / max(1, int(config.lora_r))
        self.top_k = int(config.top_k)
        self.current_edit = int(config.current_edit)
        self.oracle_edit_routing = bool(config.oracle_edit_routing)
        self.eval_oracle_edit_routing = bool(config.eval_oracle_edit_routing)
        self.learned_hidden_routing = bool(config.learned_hidden_routing)
        self.adaptive_activation = bool(config.adaptive_activation)
        self.tau_score = float(config.tau_score)
        self.curvature_mode = str(config.curvature_mode)
        self.curvature_mu = float(config.curvature_mu)
        self.curvature_max_grad_ratio = float(config.curvature_max_grad_ratio)
        self.allow_missing_covariance = bool(config.allow_missing_covariance)
        self.spectral_router = bool(config.spectral_router)
        self.router_start_step = int(config.router_start_step)
        self.router_scaling_factor = float(config.router_scaling_factor)
        self.window_size = int(config.window_size)
        self.max_components = min(int(config.max_components), self.in_features)
        self.cumulative_energy_ratio = float(config.cumulative_energy_ratio)
        self.update_covariance = bool(config.update_covariance)
        self.adapter_dtype = str(config.adapter_dtype).lower()
        self.adapters_enabled = True
        self.current_step = 0
        self.last_routing: Optional[torch.Tensor] = None
        self.last_router_probs: Optional[torch.Tensor] = None
        self.last_router_logits_mean: Optional[torch.Tensor] = None
        self.last_curvature_info: Dict[str, Any] = {}
        self.last_router_hook_info: Dict[str, Any] = {}

        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad_(False)

        self.dropout = nn.Dropout(p=float(config.lora_dropout)) if config.lora_dropout > 0 else nn.Identity()
        self.router = nn.Linear(self.in_features, self.expert_num, bias=False)
        self.lora_A = SAMEEditLinearA(self.in_features, self.per_expert_r, self.expert_num)
        self.lora_B = SAMEEditLinearB(self.per_expert_r, self.out_features, self.expert_num)

        self.register_buffer("cov_U", torch.zeros(self.in_features, self.max_components))
        self.register_buffer("cov_S", torch.zeros(self.max_components))
        self.register_buffer("cov_alpha", torch.tensor(0.0))
        self.register_buffer("cov_U_prev", torch.zeros(self.in_features, self.max_components))
        self.register_buffer("cov_S_prev", torch.zeros(self.max_components))
        self.register_buffer("cov_prev_valid", torch.tensor(False))
        self.register_buffer("utilization", torch.ones(self.expert_num) / self.expert_num)
        self.register_buffer("importance", torch.ones(self.expert_num) / self.expert_num)
        self.register_buffer("expert_masks", torch.ones(self.expert_num))

        self.reset_same_parameters()
        self._align_adapter_dtype_and_device()
        self._register_hooks()

    @classmethod
    def from_linear(cls, base_linear: nn.Linear, config: SAMEEditConfig, module_name: str = "") -> "SAMEEditLinear":
        return cls(base_linear=base_linear, config=config, module_name=module_name)

    def reset_same_parameters(self) -> None:
        for expert in self.lora_A.loraA:
            nn.init.normal_(expert.mlp.weight, mean=0.0, std=0.01)
        for expert in self.lora_B.loraB:
            nn.init.zeros_(expert.mlp.weight)
        nn.init.zeros_(self.router.weight)

    def _adapter_torch_dtype(self) -> torch.dtype:
        weight = self.base_linear.weight
        base_dtype = weight.dtype if weight.dtype.is_floating_point else torch.float32
        if self.adapter_dtype in {"base", "model", "auto"}:
            return base_dtype
        if self.adapter_dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if self.adapter_dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        return torch.float32

    def _align_adapter_dtype_and_device(self) -> None:
        weight = self.base_linear.weight
        dtype = self._adapter_torch_dtype()
        device = weight.device
        self.router.to(device=device, dtype=dtype)
        self.lora_A.to(device=device, dtype=dtype)
        self.lora_B.to(device=device, dtype=dtype)
        for buffer_name in (
            "cov_U",
            "cov_S",
            "cov_alpha",
            "cov_U_prev",
            "cov_S_prev",
            "cov_prev_valid",
            "utilization",
            "importance",
            "expert_masks",
        ):
            buffer = getattr(self, buffer_name)
            target_dtype = torch.bool if buffer.dtype == torch.bool else dtype if buffer.is_floating_point() else buffer.dtype
            setattr(self, buffer_name, buffer.to(device=device, dtype=target_dtype))

    def _register_hooks(self) -> None:
        self.router.weight.register_hook(self._spectral_aware_router_hook)
        for expert_id, expert in enumerate(self.lora_A.loraA):
            expert.mlp.weight.register_hook(self._make_curvature_hook(expert_id))

    def set_current_edit(self, edit_index: int) -> None:
        self.current_edit = int(edit_index)

    def assigned_expert(self) -> int:
        return int(self.current_edit) % max(1, self.expert_num)

    def _one_hot_routing(self) -> torch.Tensor:
        routing = torch.zeros(self.expert_num, device=self.router.weight.device, dtype=self.router.weight.dtype)
        routing[self.assigned_expert()] = 1.0
        return routing

    def _minmax(self, value: torch.Tensor) -> torch.Tensor:
        if torch.allclose(value.max(), value.min()):
            return torch.zeros_like(value)
        return (value - value.min()) / (value.max() - value.min() + 1e-8)

    def _activation_scores(self) -> torch.Tensor:
        return self._minmax(self.utilization.float()) - self._minmax(self.importance.float())

    def _apply_adaptive_activation(self) -> None:
        scores = self._activation_scores().to(self.expert_masks.device, self.expert_masks.dtype)
        masks = torch.ones_like(self.expert_masks)
        low_score = scores < float(self.tau_score)
        low_score[self.assigned_expert()] = False
        masks[low_score] = 0.0
        if float(masks.sum().detach().cpu()) <= 0.0:
            masks[self.assigned_expert()] = 1.0
        self.expert_masks.copy_(masks.detach())

    def _masked_and_topk(self, routing: torch.Tensor, force_assigned: bool) -> torch.Tensor:
        routing = routing.to(device=self.expert_masks.device, dtype=self.expert_masks.dtype)
        masks = self.expert_masks.clone()
        if force_assigned:
            masks[self.assigned_expert()] = 1.0
        masked = routing * masks
        if float(masked.sum().detach().cpu()) <= 0.0:
            masked = torch.zeros_like(routing)
            masked[self.assigned_expert()] = 1.0
        else:
            masked = masked / (masked.sum() + 1e-8)

        k = min(max(1, int(self.top_k)), self.expert_num)
        if k < self.expert_num:
            sparse = torch.zeros_like(masked)
            if force_assigned:
                assigned = self.assigned_expert()
                sparse[assigned] = masked[assigned].clamp_min(1.0e-8)
                if k > 1:
                    candidate = masked.clone()
                    candidate[assigned] = 0.0
                    values, indices = torch.topk(candidate, k=k - 1)
                    sparse[indices] = values
            else:
                values, indices = torch.topk(masked, k=k)
                sparse[indices] = values
            sparse = sparse / (sparse.sum() + 1e-8)
            return sparse
        return masked

    def _router_probs(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_input = x_flat.to(device=self.router.weight.device, dtype=self.router.weight.dtype)
        logits = self.router(router_input)
        if not self.training:
            logits = logits / max(1, self.expert_num)
        self.last_router_logits_mean = logits.mean(dim=0)
        probs = F.softmax(logits, dim=-1)
        return probs.mean(dim=0), probs

    def _routing(self, x_flat: torch.Tensor) -> torch.Tensor:
        force_oracle = (self.training and self.oracle_edit_routing) or (
            (not self.training) and self.eval_oracle_edit_routing
        )
        mean_probs, all_probs = self._router_probs(x_flat)
        self.last_router_probs = all_probs.detach()
        if force_oracle:
            routing = self._one_hot_routing()
        elif self.learned_hidden_routing:
            routing = mean_probs
        else:
            routing = torch.ones_like(mean_probs) / max(1, mean_probs.numel())
        routing = self._masked_and_topk(routing, force_assigned=self.training or force_oracle)
        self.last_routing = routing.detach()
        return routing.to(device=self.lora_A.loraA[0].weight.device, dtype=self.lora_A.loraA[0].weight.dtype)

    def _update_covariance(self, x_flat: torch.Tensor) -> None:
        if not self.update_covariance:
            return
        if _inside_activation_checkpoint_recompute():
            return
        with torch.no_grad():
            x_work = x_flat.detach()
            if x_work.numel() == 0:
                return
            batch_size = int(x_work.shape[0])
            d = int(x_work.shape[-1])
            k = min(self.max_components, d)
            compute_dtype = torch.float32 if x_work.dtype in {torch.float16, torch.bfloat16} else x_work.dtype
            centered = x_work.to(compute_dtype) - x_work.to(compute_dtype).mean(dim=0, keepdim=True)
            if float(self.cov_alpha.detach().cpu()) == 0.0 or self.current_step % 20 == 0:
                random_basis = torch.randn(d, k, device=x_work.device, dtype=compute_dtype)
                sketch = centered.t() @ (centered @ random_basis)
                q, _ = torch.linalg.qr(sketch, mode="reduced")
                projected = centered @ q
                denom = max(batch_size - 1, 1)
                s = torch.sqrt(torch.sum(projected * projected, dim=0) / float(denom))
                self.cov_U.zero_()
                self.cov_S.zero_()
                self.cov_U[:, :k].copy_(q[:, :k].to(device=self.cov_U.device, dtype=self.cov_U.dtype))
                self.cov_S[:k].copy_(s[:k].to(device=self.cov_S.device, dtype=self.cov_S.dtype))
            self.cov_alpha.add_(float(batch_size))

    def _update_activation_metrics(self, router_probs: torch.Tensor, x_flat: torch.Tensor) -> None:
        if _inside_activation_checkpoint_recompute():
            return
        with torch.no_grad():
            probs = router_probs.detach().to(device=self.utilization.device, dtype=self.utilization.dtype)
            util = probs.mean(dim=0)
            self.utilization.mul_(0.95).add_(util, alpha=0.05)
            energy = torch.norm(x_flat.detach().to(probs.device, probs.dtype), dim=-1, keepdim=True) ** 2
            importance = (probs * energy).mean(dim=0)
            self.importance.mul_(0.95).add_(importance, alpha=0.05)
            if self.adaptive_activation:
                self._apply_adaptive_activation()

    def _energy_rank(self, singular_values: torch.Tensor) -> int:
        energy = singular_values.float() ** 2
        total = energy.sum()
        if float(total.detach().cpu()) <= 1.0e-12:
            return 0
        ratio = torch.cumsum(energy, dim=0) / (total + 1.0e-12)
        k = int((ratio <= float(self.cumulative_energy_ratio)).sum().item()) + 1
        return max(1, min(k, int(singular_values.numel())))

    def _spectral_aware_router_hook(self, grad: torch.Tensor) -> torch.Tensor:
        if (
            not self.spectral_router
            or not self.training
            or self.current_step < self.router_start_step
            or grad is None
        ):
            return grad
        u = self.cov_U.to(device=grad.device, dtype=grad.dtype)
        s = self.cov_S.to(device=grad.device, dtype=grad.dtype)
        k = self._energy_rank(s)
        if k <= 0:
            return grad
        v_parallel = u[:, :k]
        grad_parallel = grad @ v_parallel @ v_parallel.t()
        smooth = torch.zeros_like(s[:k])
        for idx in range(k):
            start = max(0, idx - self.window_size + 1)
            smooth[idx] = s[start : idx + 1].mean()
        scaling = 1.0 / (smooth + 1.0e-6)
        scaling = scaling / scaling.max().clamp_min(1.0e-6)
        scaling = torch.clamp(
            scaling,
            min=1.0 - float(self.router_scaling_factor),
            max=1.0 + float(self.router_scaling_factor),
        )
        grad_parallel_scaled = (grad_parallel @ v_parallel) * scaling.unsqueeze(0)
        grad_parallel_scaled = grad_parallel_scaled @ v_parallel.t()
        grad_perp = grad - grad_parallel
        out = grad_parallel_scaled + grad_perp
        self.last_router_hook_info = {
            "k": k,
            "covariance_energy": float((s.float() ** 2).sum().detach().cpu()),
            "grad_norm_before": float(grad.detach().float().norm().cpu()),
            "grad_norm_after": float(out.detach().float().norm().cpu()),
        }
        return out

    def _make_curvature_hook(self, expert_id: int):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if (
                grad is None
                or not self.training
                or self.curvature_mode == "off"
                or int(self.current_edit) <= 0
            ):
                return grad
            if not bool(self.cov_prev_valid.detach().cpu().item()):
                if self.allow_missing_covariance:
                    return grad
                raise RuntimeError(
                    f"SAME-Edit curvature requested for edit {self.current_edit}, "
                    f"but previous covariance is missing in {self.module_name}"
                )
            u_prev = self.cov_U_prev.to(device=grad.device, dtype=grad.dtype)
            s_prev = self.cov_S_prev.to(device=grad.device, dtype=grad.dtype)
            k = self._energy_rank(s_prev)
            if k <= 0:
                return grad
            u_k = u_prev[:, :k]
            grad_t = grad.t()
            mu = max(float(self.curvature_mu), 1.0e-8)
            inv_s = 1.0 / (s_prev[:k] + mu)
            parallel = u_k @ (torch.diag(inv_s) @ (u_k.t() @ grad_t))
            perp = grad_t - u_k @ (u_k.t() @ grad_t)
            scaled = (parallel + perp / mu).t()

            before = grad.detach().float().norm()
            after = scaled.detach().float().norm()
            ratio = float(after.cpu() / before.clamp_min(1.0e-12).cpu())
            nonfinite = int((~torch.isfinite(scaled)).sum().detach().cpu().item())
            if self.curvature_mode == "safe":
                max_ratio = max(float(self.curvature_max_grad_ratio), 1.0)
                if nonfinite or ratio > max_ratio:
                    target_norm = before * max_ratio
                    scaled = scaled * (target_norm / after.clamp_min(1.0e-12))
                    if not torch.isfinite(scaled).all():
                        scaled = grad
            self.last_curvature_info = {
                "expert_id": expert_id,
                "mode": self.curvature_mode,
                "k": k,
                "covariance_energy": float((s_prev.float() ** 2).sum().detach().cpu()),
                "cov_prev_valid": True,
                "grad_norm_before": float(before.cpu()),
                "grad_norm_after": float(scaled.detach().float().norm().cpu()),
                "gradient_ratio": ratio,
                "nan_inf_count": nonfinite,
            }
            return scaled

        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_result = self.base_linear(x)
        if not self.adapters_enabled or self.lora_r <= 0:
            return base_result
        previous_dtype = base_result.dtype
        x_adapter = self.dropout(x).to(device=self.lora_A.loraA[0].weight.device, dtype=self.lora_A.loraA[0].weight.dtype)
        x_flat = x_adapter.reshape(-1, x_adapter.shape[-1])
        if self.training:
            self._update_covariance(x_flat)
        routing = self._routing(x_flat)
        lora_a_output = self.lora_A(x_adapter, routing)
        lora_b_output = self.lora_B(lora_a_output, routing)
        output = base_result + lora_b_output.to(base_result.dtype) * self.scaling
        if self.training and self.last_router_probs is not None:
            self._update_activation_metrics(self.last_router_probs, x_flat)
            self.current_step += 1
        return output.to(previous_dtype)

    def save_task_covariance_snapshot(self) -> None:
        with torch.no_grad():
            k = int((self.cov_S.detach().abs() > 1.0e-10).sum().item())
            self.cov_U_prev.zero_()
            self.cov_S_prev.zero_()
            if k > 0:
                self.cov_U_prev[:, :k].copy_(self.cov_U[:, :k])
                self.cov_S_prev[:k].copy_(self.cov_S[:k])
                self.cov_prev_valid.fill_(True)
            else:
                self.cov_prev_valid.fill_(False)

    def reset_for_new_edit(self, edit_index: int, snapshot_previous: bool = True) -> None:
        if snapshot_previous:
            self.save_task_covariance_snapshot()
        self.set_current_edit(edit_index)
        with torch.no_grad():
            self.cov_alpha.zero_()
            self.utilization.fill_(1.0 / max(1, self.expert_num))
            self.importance.fill_(1.0 / max(1, self.expert_num))
            self.expert_masks.fill_(1.0)

    def same_state_dict(self) -> Dict[str, Any]:
        return {
            "router": self.router.state_dict(),
            "lora_A": self.lora_A.state_dict(),
            "lora_B": self.lora_B.state_dict(),
            "buffers": {
                name: getattr(self, name).detach().cpu().clone()
                for name in (
                    "cov_U",
                    "cov_S",
                    "cov_alpha",
                    "cov_U_prev",
                    "cov_S_prev",
                    "cov_prev_valid",
                    "utilization",
                    "importance",
                    "expert_masks",
                )
            },
            "current_edit": int(self.current_edit),
            "current_step": int(self.current_step),
        }

    def load_same_state_dict(self, state: Dict[str, Any]) -> None:
        self.router.load_state_dict(state.get("router", {}), strict=False)
        self.lora_A.load_state_dict(state.get("lora_A", {}), strict=False)
        self.lora_B.load_state_dict(state.get("lora_B", {}), strict=False)
        for name, value in state.get("buffers", {}).items():
            if hasattr(self, name) and torch.is_tensor(value):
                target = getattr(self, name)
                target.copy_(value.to(device=target.device, dtype=target.dtype))
        self.current_edit = int(state.get("current_edit", self.current_edit))
        self.current_step = int(state.get("current_step", self.current_step))

    def routing_summary(self) -> Dict[str, Any]:
        routing = self.last_routing.detach().cpu() if torch.is_tensor(self.last_routing) else None
        masks = self.expert_masks.detach().cpu()
        util = self.utilization.detach().cpu()
        imp = self.importance.detach().cpu()
        return {
            "module": self.module_name,
            "routing": routing.tolist() if routing is not None else None,
            "top_expert_id": int(routing.argmax().item()) if routing is not None and routing.numel() else None,
            "assigned_expert_id": self.assigned_expert(),
            "routing_entropy": float((-(routing.float() * (routing.float() + 1.0e-8).log()).sum()).item())
            if routing is not None
            else None,
            "expert_mask": masks.tolist(),
            "utilization": util.tolist(),
            "importance": imp.tolist(),
            "active_expert_count": int((masks > 0).sum().item()),
            "cov_prev_valid": bool(self.cov_prev_valid.detach().cpu().item()),
            "curvature": dict(self.last_curvature_info),
            "router_hook": dict(self.last_router_hook_info),
        }


def _extract_layer_number(name: str) -> Optional[int]:
    for pattern in (r"\.layers\.(\d+)\.", r"\.decoder\.layers\.(\d+)\.", r"layer\.(\d+)\."):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def _module_matches_explicit_patterns(name: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if name.endswith(pattern):
            return True
        try:
            if re.search(pattern, name):
                return True
        except re.error:
            if pattern in name:
                return True
    return False


def _is_vision_or_projector(name: str, config: SAMEEditConfig) -> bool:
    if not config.exclude_vision_tower:
        return False
    lowered = name.lower()
    return any(segment.lower() in lowered for segment in config.exclude_module_path_segments)


def select_same_edit_target_modules(model: nn.Module, config: SAMEEditConfig) -> List[str]:
    candidates: List[Tuple[str, nn.Linear, Optional[int]]] = []
    for name, module in model.named_modules():
        if isinstance(module, SAMEEditLinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_or_projector(name, config):
            continue
        candidates.append((name, module, _extract_layer_number(name)))

    if config.target_module_patterns:
        names = [name for name, _module, _layer in candidates if _module_matches_explicit_patterns(name, config.target_module_patterns)]
        if not names:
            raise ValueError(f"SAME-Edit explicit target_module_patterns matched no modules: {config.target_module_patterns}")
        return names

    mode = str(config.target_modules or "last4_down_proj").lower()
    ffn_suffixes = ("gate_proj", "up_proj", "down_proj")
    if mode == "last8_down_proj":
        last_n = 8
        suffixes = ("down_proj",)
    elif mode == "all_down_proj":
        return [name for name, _module, _layer in candidates if name.endswith("down_proj")]
    elif mode == "late_ffn_all":
        last_n = int(config.target_last_n_layers)
        suffixes = ffn_suffixes
    elif mode == "all_ffn":
        return [name for name, _module, _layer in candidates if name.endswith(ffn_suffixes)]
    elif mode == "mm_projector_plus_late_ffn":
        last_n = int(config.target_last_n_layers)
        suffixes = ffn_suffixes
    else:
        last_n = int(config.target_last_n_layers)
        suffixes = ("down_proj",)

    layer_ids = sorted({layer for _name, _module, layer in candidates if layer is not None})
    if layer_ids:
        selected_layers = set(layer_ids[-last_n:])
        selected = [
            name
            for name, _module, layer in candidates
            if layer in selected_layers and name.endswith(suffixes)
        ]
    else:
        selected = [name for name, _module, _layer in candidates if name.endswith(suffixes)]
        selected = selected[-last_n:]

    if mode == "mm_projector_plus_late_ffn":
        selected.extend(
            name
            for name, _module, _layer in candidates
            if "mm_projector" in name.lower() or "multi_modal_projector" in name.lower()
        )
    if not selected:
        raise ValueError(f"SAME-Edit target_modules={config.target_modules!r} matched no linear modules.")
    return selected


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent: Any = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _iter_same_edit_layers(model: nn.Module) -> List[Tuple[str, SAMEEditLinear]]:
    if hasattr(model, "same_model"):
        return _iter_same_edit_layers(getattr(model, "same_model"))
    if hasattr(model, "same_edit_layers"):
        return list(model.same_edit_layers())  # type: ignore[attr-defined]
    return [(name, module) for name, module in model.named_modules() if isinstance(module, SAMEEditLinear)]


def _param_numel(parameters: Iterator[nn.Parameter]) -> int:
    return sum(param.numel() for param in parameters)


def _grad_stats(parameters: Iterator[nn.Parameter]) -> Tuple[float, int]:
    squared_norm = 0.0
    nonfinite = 0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        nonfinite += int((~torch.isfinite(grad)).sum().cpu().item())
        finite_grad = torch.nan_to_num(grad.float(), nan=0.0, posinf=0.0, neginf=0.0)
        squared_norm += float((finite_grad * finite_grad).sum().cpu())
    return math.sqrt(squared_norm), nonfinite


def same_edit_trainable_summary(model: nn.Module) -> Dict[str, Any]:
    layers = _iter_same_edit_layers(model)
    same_param_ids = set()
    router_params: List[nn.Parameter] = []
    lora_a_params: List[nn.Parameter] = []
    lora_b_params: List[nn.Parameter] = []
    for _name, layer in layers:
        router_params.extend(list(layer.router.parameters()))
        lora_a_params.extend(list(layer.lora_A.parameters()))
        lora_b_params.extend(list(layer.lora_B.parameters()))
    for param in router_params + lora_a_params + lora_b_params:
        same_param_ids.add(id(param))
    all_params = list(model.parameters())
    base_trainable = [
        param for param in all_params if param.requires_grad and id(param) not in same_param_ids
    ]
    target_names = [
        layer.module_name or name
        for name, layer in layers
    ]
    return {
        "total_params": sum(param.numel() for param in all_params),
        "trainable_params": sum(param.numel() for param in all_params if param.requires_grad),
        "same_edit_linear_count": len(layers),
        "target_module_names": target_names,
        "router_param_count": _param_numel(iter(router_params)),
        "lora_A_param_count": _param_numel(iter(lora_a_params)),
        "lora_B_param_count": _param_numel(iter(lora_b_params)),
        "base_trainable_param_count": sum(param.numel() for param in base_trainable),
    }


def same_edit_gradient_summary(model: nn.Module) -> Dict[str, Any]:
    layers = _iter_same_edit_layers(model)
    same_param_ids = set()
    router_params: List[nn.Parameter] = []
    lora_a_params: List[nn.Parameter] = []
    lora_b_params: List[nn.Parameter] = []
    for _name, layer in layers:
        router_params.extend(list(layer.router.parameters()))
        lora_a_params.extend(list(layer.lora_A.parameters()))
        lora_b_params.extend(list(layer.lora_B.parameters()))
    for param in router_params + lora_a_params + lora_b_params:
        same_param_ids.add(id(param))
    base_params = [param for param in model.parameters() if id(param) not in same_param_ids]
    router_norm, router_bad = _grad_stats(iter(router_params))
    lora_a_norm, lora_a_bad = _grad_stats(iter(lora_a_params))
    lora_b_norm, lora_b_bad = _grad_stats(iter(lora_b_params))
    base_norm, base_bad = _grad_stats(iter(base_params))
    return {
        "router_grad_norm": router_norm,
        "lora_A_grad_norm": lora_a_norm,
        "lora_B_grad_norm": lora_b_norm,
        "base_grad_norm": base_norm,
        "nan_inf_grad_count": router_bad + lora_a_bad + lora_b_bad + base_bad,
    }


def print_same_edit_trainable_summary(model: nn.Module) -> Dict[str, Any]:
    summary = same_edit_trainable_summary(model)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


class SAMEEditModel(nn.Module):
    def __init__(self, model: nn.Module, config: SAMEEditConfig):
        super().__init__()
        self.model = model
        self.config = config.normalized()
        self.replaced_module_names: List[str] = []
        self._install()

    def _install(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        targets = select_same_edit_target_modules(self.model, self.config)
        if self.config.log_rank_remainder and self.config.lora_r % self.config.expert_num != 0:
            LOG.warning(
                "SAME-Edit lora_r=%s is not divisible by expert_num=%s; using per_expert_r=%s, effective_rank=%s.",
                self.config.lora_r,
                self.config.expert_num,
                self.config.per_expert_r,
                self.config.effective_rank,
            )
        for name in targets:
            parent, child_name = _get_parent_module(self.model, name)
            target = getattr(parent, child_name)
            if not isinstance(target, nn.Linear):
                raise TypeError(f"SAME-Edit target {name} is not nn.Linear: {type(target)}")
            wrapped = SAMEEditLinear.from_linear(target, self.config, module_name=name)
            setattr(parent, child_name, wrapped)
            self.replaced_module_names.append(name)
        LOG.info("SAME-Edit replaced modules: %s", self.replaced_module_names)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def same_edit_layers(self) -> Iterator[Tuple[str, SAMEEditLinear]]:
        for name, module in self.model.named_modules():
            if isinstance(module, SAMEEditLinear):
                yield name, module

    def set_current_edit(self, edit_index: int) -> None:
        self.config.current_edit = int(edit_index)
        for _name, module in self.same_edit_layers():
            module.set_current_edit(edit_index)

    def reset_for_new_edit(self, edit_index: int, snapshot_previous: bool = True) -> None:
        self.config.current_edit = int(edit_index)
        for _name, module in self.same_edit_layers():
            module.reset_for_new_edit(edit_index, snapshot_previous=snapshot_previous)

    def save_covariance_snapshot(self) -> None:
        for _name, module in self.same_edit_layers():
            module.save_task_covariance_snapshot()

    @contextmanager
    def adapters_disabled(self):
        old_values: List[Tuple[SAMEEditLinear, bool]] = []
        for _name, module in self.same_edit_layers():
            old_values.append((module, module.adapters_enabled))
            module.adapters_enabled = False
        try:
            yield
        finally:
            for module, old in old_values:
                module.adapters_enabled = old

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def routing_supervision_loss(self) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        fallback: Optional[torch.Tensor] = None
        for _name, module in self.same_edit_layers():
            logits = module.last_router_logits_mean
            if fallback is None:
                fallback = module.router.weight.sum() * 0.0
            if torch.is_tensor(logits):
                target = torch.tensor([module.assigned_expert()], device=logits.device, dtype=torch.long)
                losses.append(F.cross_entropy(logits.float().unsqueeze(0), target))
        if losses:
            return torch.stack(losses).mean()
        if fallback is not None:
            return fallback
        params = self.trainable_parameters()
        if params:
            return params[0].sum() * 0.0
        return torch.tensor(0.0)

    def validate_covariance_for_curvature(self) -> None:
        if self.config.curvature_mode == "off" or int(self.config.current_edit) <= 0:
            return
        valid = [
            bool(module.cov_prev_valid.detach().cpu().item())
            for _name, module in self.same_edit_layers()
        ]
        if valid and not any(valid) and not self.config.allow_missing_covariance:
            raise RuntimeError(
                "SAME-Edit curvature is enabled for edit_id > 0, but no adapted layer has cov_prev_valid=True."
            )

    def state_bundle(self) -> Dict[str, Any]:
        return {
            "method": "same_edit",
            "config": self.config.to_json_dict(),
            "replaced_module_names": list(self.replaced_module_names),
            "layers": {name: module.same_state_dict() for name, module in self.same_edit_layers()},
        }

    def load_state_bundle(self, bundle: Dict[str, Any]) -> None:
        layer_state = bundle.get("layers", {})
        by_name = dict(self.same_edit_layers())
        for name, state in layer_state.items():
            if name in by_name:
                by_name[name].load_same_state_dict(state)
        self.validate_covariance_for_curvature()

    def save_same_edit_state(self, output_dir: str | Path) -> Dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        bundle = self.state_bundle()
        torch.save(bundle, output / "same_edit_state.pt")
        summary = self.summary()
        (output / "same_edit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    def load_same_edit_state(self, path: str | Path) -> None:
        bundle = torch.load(path, map_location="cpu")
        self.load_state_bundle(bundle)

    def summary(self) -> Dict[str, Any]:
        layer_summaries = [module.routing_summary() for _name, module in self.same_edit_layers()]
        usage = torch.zeros(self.config.expert_num)
        active_counts = []
        cov_valid = 0
        for row in layer_summaries:
            routing = row.get("routing")
            if routing:
                usage += torch.tensor(routing, dtype=torch.float32)
            active_counts.append(int(row.get("active_expert_count") or 0))
            cov_valid += int(bool(row.get("cov_prev_valid")))
        if layer_summaries:
            usage = usage / len(layer_summaries)
        return {
            "method": "same_edit",
            "current_edit": int(self.config.current_edit),
            "assigned_expert_id": int(self.config.current_edit) % max(1, self.config.expert_num),
            "expert_num": int(self.config.expert_num),
            "top_k": int(self.config.top_k),
            "target_modules": list(self.replaced_module_names),
            "target_module_count": len(self.replaced_module_names),
            "per_expert_r": int(self.config.per_expert_r),
            "effective_rank": int(self.config.effective_rank),
            "expert_usage_histogram": usage.tolist(),
            "mean_active_expert_count": float(sum(active_counts) / len(active_counts)) if active_counts else 0.0,
            "covariance_valid_count": cov_valid,
            "layers": layer_summaries,
        }
