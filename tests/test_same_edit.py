from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from easyeditor.models.same_edit import (
    SAMEEditConfig,
    SAMEEditLinear,
    SAMEEditModel,
    same_edit_trainable_summary,
)


class TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.down_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(x)


class TinyModel(nn.Module):
    def __init__(self, dim: int = 6, layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(dim) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def first_same_layer(model: SAMEEditModel) -> SAMEEditLinear:
    return next(module for _name, module in model.same_edit_layers())


def make_same(**kwargs) -> SAMEEditModel:
    base = TinyModel()
    defaults = {
        "lora_r": 4,
        "expert_num": 2,
        "top_k": 1,
        "target_modules": "last4_down_proj",
        "curvature_mode": "off",
    }
    defaults.update(kwargs)
    config = SAMEEditConfig(**defaults)
    return SAMEEditModel(base, config)


def test_same_edit_identity_at_initialization():
    torch.manual_seed(1)
    base = TinyModel()
    reference = TinyModel()
    reference.load_state_dict(base.state_dict())
    same = SAMEEditModel(
        base,
        SAMEEditConfig(lora_r=4, expert_num=2, target_modules="last4_down_proj"),
    )
    x = torch.randn(2, 3, 6)
    same.train()
    assert torch.allclose(same(x), reference(x), atol=1.0e-6)
    assert all(not layer.base_linear.weight.requires_grad for _name, layer in same.same_edit_layers())


def test_same_edit_forward_shapes_and_routing_topk():
    same = make_same(oracle_edit_routing=False, learned_hidden_routing=True, top_k=1)
    same.train()
    out_seq = same(torch.randn(2, 3, 6))
    out_flat = same(torch.randn(5, 6))
    layer = first_same_layer(same)
    routing = layer.last_routing
    assert out_seq.shape == (2, 3, 6)
    assert out_flat.shape == (5, 6)
    assert routing is not None
    assert torch.isclose(routing.sum(), torch.tensor(1.0, dtype=routing.dtype), atol=1.0e-5)
    assert int((routing > 0).sum()) <= 1


def test_same_edit_trainable_summary_reports_frozen_base():
    same = make_same()
    summary = same_edit_trainable_summary(same)
    assert summary["same_edit_linear_count"] == 4
    assert summary["router_param_count"] > 0
    assert summary["lora_A_param_count"] > 0
    assert summary["lora_B_param_count"] > 0
    assert summary["base_trainable_param_count"] == 0


def test_same_edit_adapters_default_to_fp32_when_base_is_half():
    base = TinyModel().half()
    same = SAMEEditModel(base, SAMEEditConfig(lora_r=4, expert_num=2, target_modules="last4_down_proj"))
    layer = first_same_layer(same)
    assert layer.router.weight.dtype == torch.float32
    assert layer.lora_A.loraA[0].weight.dtype == torch.float32
    assert layer.lora_B.loraB[0].weight.dtype == torch.float32
    assert layer.base_linear.weight.dtype == torch.float16


def test_same_edit_route_supervision_trains_router_under_oracle_routing():
    torch.manual_seed(2)
    same = make_same(oracle_edit_routing=True)
    same.train()
    _ = same(torch.randn(2, 3, 6))
    loss = same.routing_supervision_loss()
    loss.backward()
    layer = first_same_layer(same)
    assert layer.router.weight.grad is not None
    assert float(layer.router.weight.grad.detach().norm()) > 0.0


def test_same_edit_assigned_expert_stays_active_under_masking():
    same = make_same(adaptive_activation=True, tau_score=2.0, oracle_edit_routing=True)
    same.reset_for_new_edit(1, snapshot_previous=False)
    same.train()
    _ = same(torch.randn(2, 3, 6))
    layer = first_same_layer(same)
    assert layer.expert_masks[layer.assigned_expert()].item() == pytest.approx(1.0)
    assert int((layer.expert_masks > 0).sum()) >= 1


def test_same_edit_covariance_update_and_snapshot():
    same = make_same()
    same.train()
    _ = same(torch.randn(2, 3, 6))
    layer = first_same_layer(same)
    assert float(layer.cov_alpha.item()) > 0.0
    assert not layer.cov_U.requires_grad
    layer.save_task_covariance_snapshot()
    assert bool(layer.cov_prev_valid.item())
    assert torch.count_nonzero(layer.cov_S_prev).item() > 0


def test_same_edit_save_load_restores_router_experts_and_buffers(tmp_path):
    same = make_same()
    same.train()
    _ = same(torch.randn(2, 3, 6))
    layer = first_same_layer(same)
    with torch.no_grad():
        layer.router.weight.fill_(0.25)
        layer.lora_B.loraB[0].mlp.weight.fill_(0.5)
    layer.save_task_covariance_snapshot()
    bundle = same.state_bundle()

    restored = make_same()
    restored.load_state_bundle(bundle)
    restored_layer = first_same_layer(restored)
    assert torch.allclose(restored_layer.router.weight, layer.router.weight)
    assert torch.allclose(restored_layer.lora_B.loraB[0].mlp.weight, layer.lora_B.loraB[0].mlp.weight)
    assert bool(restored_layer.cov_prev_valid.item())

    summary = restored.save_same_edit_state(tmp_path)
    assert (tmp_path / "same_edit_state.pt").exists()
    assert summary["method"] == "same_edit"


def test_same_edit_curvature_and_spectral_hooks_are_finite():
    same = make_same(
        current_edit=1,
        curvature_mode="safe",
        allow_missing_covariance=False,
        spectral_router=True,
        router_start_step=0,
        oracle_edit_routing=False,
        top_k=2,
    )
    layer = first_same_layer(same)
    with torch.no_grad():
        for _name, same_layer in same.same_edit_layers():
            same_layer.cov_U_prev.zero_()
            same_layer.cov_U_prev[:, : same_layer.in_features].copy_(torch.eye(same_layer.in_features))
            same_layer.cov_S_prev.fill_(1.0)
            same_layer.cov_prev_valid.fill_(True)
            same_layer.cov_U.zero_()
            same_layer.cov_U[:, : same_layer.in_features].copy_(torch.eye(same_layer.in_features))
            same_layer.cov_S.fill_(1.0)
            for expert in same_layer.lora_B.loraB:
                expert.mlp.weight.fill_(0.1)
    same.train()
    loss = same(torch.randn(2, 3, 6)).sum()
    loss.backward()
    for param in same.trainable_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
    router_hook_out = layer._spectral_aware_router_hook(torch.ones_like(layer.router.weight))
    assert torch.isfinite(router_hook_out).all()


def test_same_edit_files_do_not_import_engram():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "easyeditor" / "models" / "same_edit").glob("*.py"))
    files += [root / "easyeditor" / "trainer" / "algs" / "same_edit.py"]
    offenders = []
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip().lower()
            if (stripped.startswith("import ") or stripped.startswith("from ")) and "engram" in stripped:
                offenders.append(f"{path}:{lineno}:{line}")
    assert offenders == []
