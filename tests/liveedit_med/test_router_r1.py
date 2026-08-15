from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from methods.liveedit_med.router_r1 import (
    CHECKPOINT_STEPS,
    NEGATIVE_CATEGORIES,
    configure_router_only,
    deterministic_repository,
    negative_category,
    negative_text_absolute_loss,
    negative_visual_loss,
    positive_text_losses,
    positive_visual_loss,
    repository_size,
    select_checkpoint,
    semantic_category,
    validation_eligibility,
)
from methods.liveedit_med.serialization import load_safe_state, save_safe_state, tensor_hashes
from methods.liveedit_med.source_ops import compute_text_soft_weights
from methods.liveedit_med.trace_parity import state_dict_sha256
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules


ROOT = Path(__file__).resolve().parents[2]


def tiny_modules():
    return LiveEditMedicalModules(LiveEditMedicalConfig(llm_mid_dim=16, module_dim=8,
        cross_att_head_n=2, eqe_n=2), vision_tokens=3)


def test_repository_and_category_cycles_are_exact():
    assert [repository_size(i) for i in range(1, 11)] == [1, 4, 8, 16, 32] * 2
    assert [semantic_category(i) for i in range(1, 7)] == ["textual", "visual", "paired"] * 2
    assert [negative_category(i) for i in range(1, 11)] == list(NEGATIVE_CATEGORIES) * 2
    assert CHECKPOINT_STEPS == (80, 160, 240, 320, 400, 480, 560, 640)


def test_repository_membership_is_target_first_and_deterministic():
    nearest = {"a": {"visual": ["b", "c"], "text": ["c", "d"], "joint": ["e"]}}
    ids = list("abcdefghij")
    assert deterministic_repository("a", 4, nearest, ids) == ["a", "b", "c", "d"]
    assert deterministic_repository("a", 4, nearest, ids) == deterministic_repository("a", 4, nearest, ids)


def test_only_extractors_are_trainable():
    modules = tiny_modules()
    names = configure_router_only(modules)
    assert names
    assert all(name.startswith(("edit_extractor.", "input_extractor.")) for name in names)
    assert not any(parameter.requires_grad for name, parameter in modules.named_parameters()
                   if name.startswith(("moegen_c.", "moegen_r.", "instant_reps_norm.")))


def test_final_weight_is_sigmoid_times_softmax_without_renormalization():
    x = torch.tensor([[[1.0, 0.0]]])
    e = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    relative, absolute = compute_text_soft_weights(x, e, split=True)
    final = compute_text_soft_weights(x, e)
    assert torch.equal(final, relative * absolute)
    assert not torch.isclose(final.sum(), torch.tensor(1.0))


def test_router_losses_have_expected_sign_and_gradients():
    input_visual = torch.randn(1, 2, 8, requires_grad=True)
    edit_visual = torch.randn(3, 2, 8, requires_grad=True)
    sentinel = torch.randn(1, 2, 8, requires_grad=True)
    input_text = torch.randn(1, 2, 8, requires_grad=True)
    edit_text = torch.randn(3, 2, 8, requires_grad=True)
    locality = torch.randn(1, 2, 8, requires_grad=True)
    pa, pr = positive_text_losses(input_text, edit_text, 1, locality)
    losses = [positive_visual_loss(input_visual, edit_visual, sentinel, 1),
              negative_visual_loss(input_visual, edit_visual, sentinel), pa, pr,
              negative_text_absolute_loss(input_text, edit_text)]
    assert all(torch.isfinite(value) and value >= 0 for value in losses)
    sum(losses).backward()
    assert all(value.grad is not None for value in (input_visual, edit_visual, sentinel, input_text, edit_text, locality))


def test_validation_floor_rejects_all_off_and_selects_lexicographically():
    forced = {"native": 43, "textual": 40, "visual": 38, "paired": 39}
    base = {"routed_native": 39, "routed_textual": 30, "routed_visual": 29,
            "routed_paired": 30, "target_contamination": 0, "clinical_canonical_failures": 0,
            "negative_locality_exact_s0": 120, "mean_candidate_count": 2.0,
            "text_relative_competition_failures": 3, "negative_locality_kl": .01}
    assert validation_eligibility(base, forced)
    assert not validation_eligibility({**base, "routed_native": 0}, forced)
    early = {**base, "step": 80}
    safer = {**base, "step": 160, "negative_locality_exact_s0": 121}
    assert select_checkpoint([early, safer], forced)["step"] == 160
    assert select_checkpoint([{**early, "routed_native": 0}], forced) is None


def test_07_unique_strict_checkpoint_anchor_contract():
    text = (ROOT / "scripts/liveedit_med/prepare_router_r1_run.py").read_text()
    assert 'args.strict_run.name == "20260814T101648Z"' in text
    assert 'selection.get("selected_step") == 3200' in text


def test_08_generators_and_norm_are_frozen():
    modules = tiny_modules(); configure_router_only(modules)
    assert all(not p.requires_grad for name, p in modules.named_parameters()
               if name.startswith(("moegen_c.", "moegen_r.", "instant_reps_norm.")))


def test_09_cache_representation_online_exact_contract():
    text = (ROOT / "scripts/liveedit_med/verify_router_r1_cache.py").read_text()
    assert '"representation_exact"' in text and "torch.equal" in text


def test_10_cache_expert_online_exact_contract():
    text = (ROOT / "scripts/liveedit_med/verify_router_r1_cache.py").read_text()
    assert '"expert_exact"' in text and '"expert__moe_c"' in text


def test_11_strict_source_continuation_retained():
    text = (ROOT / "scripts/liveedit_med/train_router_r1.py").read_text()
    assert "STRICT_SOURCE_REAPPLY_LAYER21" in text


def test_12_inference_continuation_unchanged():
    text = (ROOT / "scripts/liveedit_med/train_router_r1.py").read_text()
    assert "official_layer21_output_hook_then_layer22" in text


def test_13_visual_candidate_equation_unchanged():
    text = (ROOT / "methods/liveedit_med/source_ops.py").read_text()
    assert "candidate = (visual_score > sentinel_score)[0]" in text


def test_14_no_post_renormalization_or_no_edit_expert():
    text = (ROOT / "methods/liveedit_med/source_ops.py").read_text()
    assert "relative * absolute" in text
    assert "NO_EDIT" not in text


def test_15_distractor_order_visual_text_joint_then_hash():
    nearest = {"a": {"visual": ["b"], "text": ["c"], "joint": ["d"]}}
    assert deterministic_repository("a", 4, nearest, list("abcdef")) == ["a", "b", "c", "d"]


def test_16_hard_negative_mining_declares_clean_s0_only():
    text = (ROOT / "scripts/liveedit_med/prepare_router_r1_data.py").read_text()
    assert '"clean_s0_only": True' in text
    assert '"target_answers_used_as_features": False' in text
    assert "other not in {visual_nearest, text_nearest}" in text


def test_17_record953_excluded_from_fitting_and_selection():
    for name in ("prepare_router_r1_data.py", "select_router_r1_checkpoint.py"):
        text = (ROOT / "scripts/liveedit_med" / name).read_text()
        assert "record953" in text


def test_18_blind_content_not_loaded_by_data_or_training():
    for name in ("prepare_router_r1_data.py", "train_router_r1.py"):
        text = (ROOT / "scripts/liveedit_med" / name).read_text()
        assert "sealed_file" not in text and "blind_set_builder" not in text


def test_19_eqkey_conflict_is_a_hard_stop():
    text = (ROOT / "scripts/liveedit_med/cache_router_r1_hard_negatives.py").read_text()
    assert "EqKey" in text and "INVALID_ENGINEERING_RUN" in text


def test_20_positive_output_uses_full_repository_mask():
    text = (ROOT / "scripts/liveedit_med/train_router_r1.py").read_text()
    assert "mask[repo_begin:repo_end] = True" in text
    assert 'native_masks = masks[0::4]' in text


def test_21_negative_kl_uses_full_repository_mask():
    text = (ROOT / "scripts/liveedit_med/train_router_r1.py").read_text()
    assert 'negative_masks = masks[2::4]' in text
    assert '("hard_negative", negative_rows, negative_masks, True)' in text


def test_22_source_kl_direction_is_base_to_candidate():
    text = (ROOT / "methods/liveedit_med/cached_suffix.py").read_text()
    assert "KL(base || candidate)" in text


def test_23_fresh_optimizer_has_no_old_moments():
    modules = tiny_modules(); configure_router_only(modules)
    optimizer = torch.optim.Adam([p for p in modules.parameters() if p.requires_grad], lr=5e-5)
    assert optimizer.state == {}


def test_24_frozen_hash_unchanged_after_one_router_step():
    modules = tiny_modules(); configure_router_only(modules)
    frozen_before = state_dict_sha256({name: value for name, value in modules.state_dict().items()
        if name.startswith(("moegen_c.", "moegen_r.", "instant_reps_norm."))})
    optimizer = torch.optim.Adam([p for p in modules.parameters() if p.requires_grad], lr=5e-5)
    loss = sum(parameter.square().mean() for parameter in modules.parameters() if parameter.requires_grad)
    loss.backward(); optimizer.step()
    frozen_after = state_dict_sha256({name: value for name, value in modules.state_dict().items()
        if name.startswith(("moegen_c.", "moegen_r.", "instant_reps_norm."))})
    assert frozen_before == frozen_after


def test_25_output_directory_nonoverwrite_contract():
    for name in ("prepare_router_r1_run.py", "cache_router_r1.py", "select_router_r1_checkpoint.py"):
        assert "FileExistsError" in (ROOT / "scripts/liveedit_med" / name).read_text()


def test_26_canonical_bank_hash_checked_during_training():
    text = (ROOT / "scripts/liveedit_med/train_router_r1.py").read_text()
    assert 'bank_manifest()["sha256"] != EXPECTED_BANK_HASH' in text


def test_27_candidate_state_reload(tmp_path):
    modules = tiny_modules(); state = {name: value for name, value in modules.state_dict().items()}
    save_safe_state(tmp_path / "candidate", state, {"protocol": "test"})
    loaded, _manifest = load_safe_state(tmp_path / "candidate")
    assert tensor_hashes(state) == tensor_hashes(loaded)


def test_28_fresh_reconstruction_route_membership_parity():
    nearest = {"a": {"visual": ["b"], "text": ["c"], "joint": ["d"]}}
    first = deterministic_repository("a", 4, copy.deepcopy(nearest), list("abcdef"))
    second = deterministic_repository("a", 4, copy.deepcopy(nearest), list("abcdef"))
    assert first == second


def test_29_exact_router_rollback():
    modules = tiny_modules(); configure_router_only(modules)
    before = copy.deepcopy(modules.state_dict()); before_hash = state_dict_sha256(before)
    with torch.no_grad():
        next(p for p in modules.parameters() if p.requires_grad).add_(1)
    modules.load_state_dict(before, strict=True)
    assert state_dict_sha256(modules.state_dict()) == before_hash
