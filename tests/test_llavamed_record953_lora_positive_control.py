from pathlib import Path

import pytest
import torch

from scripts.engram.lora_positive_control_utils import (
    adapter_state,
    audit_trainable_parameters,
    ensure_new_output_dir,
    load_adapter_payload,
    load_adapter_state,
    positive_control_match,
    resolve_target_modules,
    save_adapter_payload,
    shifted_label_audit,
    weighted_loss,
)
from scripts.engram.natural_generation_recovery_utils import assert_no_target_leakage


def tiny_module_tree():
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList()
    for _ in range(32):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, torch.nn.Linear(2, 2))
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, name, torch.nn.Linear(2, 2))
        root.model.layers.append(layer)
    root.model.mm_projector = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.GELU(), torch.nn.Linear(2, 2))
    return root


def test_resolved_target_modules_match_fixed_specification():
    names = resolve_target_modules(tiny_module_tree().named_modules())
    assert len(names) == 16 * 7 + 2
    assert names[0] == "model.layers.16.self_attn.q_proj"
    assert names[-1] == "model.mm_projector.2"


def test_only_fp32_lora_parameters_are_trainable():
    model = torch.nn.Module()
    model.base = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    model.lora_A = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    model.lora_B = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    assert audit_trainable_parameters(model.named_parameters())["passed"]


def test_zero_adapter_s0_token_parity():
    base = torch.tensor([1.0, 2.0])
    delta = torch.zeros_like(base)
    assert torch.equal((base + delta).argmax().reshape(1), base.argmax().reshape(1))


def test_no_target_leakage_into_generation_input():
    assert_no_target_leakage(["What is visible?"], "completely ectocervical and fully visible")
    with pytest.raises(AssertionError):
        assert_no_target_leakage(["Answer: completely ectocervical and fully visible"], "completely ectocervical and fully visible")


def test_shifted_label_correctness_after_expansion():
    result = shifted_label_audit(5, torch.tensor([7, 8, 9]), 3, [7, 8, 9])
    assert result["passed"] and result["supervised_response_tokens"] == 3


def test_weighted_primary_auxiliary_loss():
    assert weighted_loss(torch.tensor(2.0), torch.tensor(4.0)).item() == 3.0


def test_deterministic_generation_success_matcher():
    target = "completely ectocervical and fully visible"
    assert positive_control_match(f"The answer is {target}.", target, eos=True, cap_hit=False)["success"]
    assert not positive_control_match(f"The answer is not {target}.", target, eos=True, cap_hit=False)["success"]


def test_isolated_adapter_save_load(tmp_path: Path):
    model = torch.nn.Module()
    model.lora_A = torch.nn.Parameter(torch.ones(2))
    model.lora_B = torch.nn.Parameter(torch.full((2,), 2.0))
    state = adapter_state(model.named_parameters())
    save_adapter_payload(tmp_path / "bank", state, {"record_id": "953"})
    loaded, manifest = load_adapter_payload(tmp_path / "bank")
    assert manifest["record_id"] == "953" and set(loaded) == set(state)


def test_exact_adapter_unload_rollback():
    model = torch.nn.Module()
    model.lora_A = torch.nn.Parameter(torch.ones(2))
    model.lora_B = torch.nn.Parameter(torch.ones(2))
    initial = adapter_state(model.named_parameters())
    with torch.no_grad():
        model.lora_A.zero_(); model.lora_B.zero_()
    load_adapter_state(model.named_parameters(), initial)
    assert all(torch.equal(initial[name], value) for name, value in adapter_state(model.named_parameters()).items())


def test_output_directory_non_overwrite(tmp_path: Path):
    ensure_new_output_dir(tmp_path / "run")
    with pytest.raises(FileExistsError):
        ensure_new_output_dir(tmp_path / "run")
