#!/usr/bin/env python3
"""Static and unit checks for the TIME CP-factor implementation."""

from __future__ import annotations

import argparse
import json
import py_compile
import sys
import tempfile
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_time_medmkeb_smoke import (  # noqa: E402
    ADAPTIVE_MARGIN_CONFIRMATION_MODE,
    build_adaptive_margin_confirmation_plan,
)
from easyeditor.trainer.algs.time_edit_modules import (  # noqa: E402
    TIMECPResidual,
    TIMEExpertRepository,
    choose_factor_shape,
)


NEW_OR_MODIFIED = [
    "easyeditor/models/time_edit/__init__.py",
    "easyeditor/models/time_edit/time_edit_hparams.py",
    "easyeditor/models/time_edit/time_edit_main.py",
    "easyeditor/models/time_edit/time_edit_modules.py",
    "easyeditor/trainer/algs/time_edit_modules.py",
    "easyeditor/trainer/algs/time_edit.py",
    "easyeditor/models/__init__.py",
    "easyeditor/trainer/algs/__init__.py",
    "easyeditor/util/alg_dict.py",
    "easyeditor/util/alg_train_dict.py",
    "easyeditor/trainer/MultimodalTrainer.py",
    "scripts/time/test_time_modules.py",
    "scripts/time/run_time_medmkeb_smoke.py",
]


def assert_close(lhs: torch.Tensor, rhs: torch.Tensor, msg: str) -> None:
    if not torch.allclose(lhs, rhs, atol=0.0, rtol=0.0):
        raise AssertionError(msg)


def test_py_compile() -> None:
    existing = []
    for rel in NEW_OR_MODIFIED:
        path = PROJECT_ROOT / rel
        if path.exists() and path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)
            existing.append(rel)
    if not existing:
        raise AssertionError("No Python files were compiled.")


def make_repo(hidden_size: int = 16, rank: int = 2, num_experts: int = 3) -> TIMEExpertRepository:
    s1, s2 = choose_factor_shape(hidden_size)
    repo = TIMEExpertRepository(
        hidden_size=hidden_size,
        rank=rank,
        s1=s1,
        s2=s2,
        init_std=1.0e-2,
        gamma=0.5,
        activation="gelu",
    )
    for idx in range(num_experts):
        repo.add_expert(f"toy-{idx}")
    return repo


def test_shape() -> None:
    torch.manual_seed(1)
    repo = make_repo()
    module = TIMECPResidual(repo)
    x = torch.randn(2, 5, 16)
    residual, debug = module(x, return_debug=True)
    if residual.shape != x.shape:
        raise AssertionError(f"Residual shape {tuple(residual.shape)} != {tuple(x.shape)}")
    if debug.scores.shape != (2, 5, 3):
        raise AssertionError(f"Score shape mismatch: {tuple(debug.scores.shape)}")


def test_identity_cases() -> None:
    x = torch.randn(2, 3, 16)
    empty = TIMEExpertRepository(hidden_size=16, rank=2)
    empty_module = TIMECPResidual(empty)
    assert_close(empty_module(x), torch.zeros_like(x), "Empty repository residual must be exactly zero.")

    high = make_repo()
    high.gamma = 1.0e9
    high_module = TIMECPResidual(high)
    assert_close(high_module(x), torch.zeros_like(x), "High threshold residual must be exactly zero.")

    active = make_repo()
    active_module = TIMECPResidual(active)
    assert_close(active_module(x, disable_time=True), torch.zeros_like(x), "disable_time residual must be exactly zero.")


def test_save_load() -> None:
    torch.manual_seed(2)
    repo = make_repo(num_experts=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "expert_repository.pt"
        repo.save(path)
        loaded = TIMEExpertRepository.load(path)
    if loaded.num_experts != repo.num_experts:
        raise AssertionError("Loaded repository expert count changed.")
    if loaded.metadata != repo.metadata:
        raise AssertionError("Loaded repository metadata changed.")
    before = repo.get_factors(detach=True)
    after = loaded.get_factors(detach=True)
    for name in before:
        if not torch.allclose(before[name], after[name]):
            raise AssertionError(f"Factor {name} changed after save/load.")


def test_routing_synthetic() -> None:
    repo = make_repo(num_experts=3)
    repo.activation = "identity"
    repo.gamma = -1.0
    for expert in repo.experts:
        for param in expert.parameters():
            param.data.zero_()
    repo.experts[1].U_in.data[0, 0] = 1.0
    repo.experts[1].V_in.data[0, 0] = 1.0
    repo.experts[1].U_out.data[0, 0] = 1.0
    repo.experts[1].V_out.data[0, 0] = 1.0
    module = TIMECPResidual(repo, layer_norm=False)
    x = torch.zeros(1, 1, 16)
    x[0, 0, 0] = 3.0
    _residual, debug = module(x, return_debug=True)
    top = int(debug.scores[0, 0].argmax().item())
    if top != 1:
        raise AssertionError(f"Synthetic aligned expert should route top-1 to expert 1, got {top}.")


def test_score_variants_and_relative_threshold() -> None:
    repo = make_repo(hidden_size=16, rank=1, num_experts=2)
    repo.activation = "identity"
    for expert in repo.experts:
        for param in expert.parameters():
            param.data.zero_()
    repo.experts[0].U_in.data[0, 0] = 2.0
    repo.experts[0].V_in.data[0, 0] = 2.0
    repo.experts[0].U_out.data[0, 0] = 1.0
    repo.experts[0].V_out.data[0, 0] = 1.0
    repo.experts[1].U_in.data[0, 0] = 1.0
    repo.experts[1].V_in.data[0, 0] = 1.0
    repo.experts[1].U_out.data[0, 0] = 1.0
    repo.experts[1].V_out.data[0, 0] = 1.0

    x = torch.zeros(1, 1, 16)
    x[0, 0, 0] = 1.0
    module = TIMECPResidual(repo, routing_mode="relative_threshold", relative_threshold=0.5, layer_norm=False)
    _residual, debug = module(x, return_debug=True)
    if set(debug.score_variants) != {"none", "factor", "factor_z", "self_score", "factor_self_score"}:
        raise AssertionError(f"Unexpected score variants: {sorted(debug.score_variants)}")
    if not torch.isfinite(debug.score_variants["factor_z"]).all():
        raise AssertionError("factor_z scores must be finite.")
    selected = debug.selected[0, 0].tolist()
    if selected != [True, False]:
        raise AssertionError(f"Relative threshold should keep only the dominant expert, got {selected}.")


def test_post_retrain_calibration_controls() -> None:
    repo = make_repo(hidden_size=16, rank=1, num_experts=3)
    repo.activation = "identity"
    repo.gamma = -1.0
    for expert in repo.experts:
        for param in expert.parameters():
            param.data.zero_()
        expert.V_in.data[0, 0] = 1.0
        expert.U_out.data[0, 0] = 1.0
        expert.V_out.data[0, 0] = 1.0
    for idx, value in enumerate((3.0, 2.0, 1.0)):
        repo.experts[idx].U_in.data[0, 0] = value

    x = torch.zeros(1, 2, 16)
    x[:, :, 0] = 1.0
    capped = TIMECPResidual(repo, routing_mode="threshold", max_selected_experts=2, score_pool="mean", layer_norm=False)
    _residual, debug = capped(x, return_debug=True)
    selected = debug.selected[0, 0].tolist()
    if selected != [True, True, False]:
        raise AssertionError(f"Max-selected cap should keep the top two experts, got {selected}.")

    calibrated = TIMECPResidual(repo, routing_mode="topk", topk=1, score_norm="none", calibration_mode="zscore_neg", score_pool="mean", layer_norm=False)
    calibrated.calibration_stats = {"mu_neg": [2.9, 0.0, 0.0], "std_neg": [0.1, 1.0, 1.0]}
    _residual, debug = calibrated(x, return_debug=True)
    top = int(debug.scores[0, 0].argmax().item())
    if top != 1:
        raise AssertionError(f"zscore_neg calibration should rerank top expert to 1, got {top}.")


def test_adaptive_rank_margin_defaults_do_not_change_threshold() -> None:
    scores = torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32)
    repo = make_repo(num_experts=3)
    repo.gamma = 0.15
    base = TIMECPResidual(repo, routing_mode="threshold")
    experimental_defaults = TIMECPResidual(repo, routing_mode="threshold")
    expected = base._selection(scores, None)
    observed = experimental_defaults._selection(scores, None)
    if not torch.equal(expected, observed):
        raise AssertionError("Adaptive rank-margin defaults changed threshold routing.")


def test_adaptive_rank_margin_disabled_matches_topk() -> None:
    scores = torch.tensor([[[0.9, 0.8, 0.79, 0.1]]], dtype=torch.float32)
    repo = make_repo(num_experts=4)
    base = TIMECPResidual(repo, routing_mode="topk", topk=2)
    disabled = TIMECPResidual(
        repo,
        routing_mode="adaptive_rank_margin_topk2",
        topk=2,
        enable_adaptive_rank_margin_rescue=False,
    )
    if not torch.equal(base._selection(scores, None), disabled._selection(scores, None)):
        raise AssertionError("Disabled adaptive rank-margin mode must match existing top-k selection.")


def test_adaptive_rank_margin_rank3_gap_and_trigger() -> None:
    scores = torch.tensor([[[0.9, 0.8, 0.79, 0.1]]], dtype=torch.float32)
    repo = make_repo(num_experts=4)
    module = TIMECPResidual(
        repo,
        routing_mode="adaptive_rank_margin_topk2",
        enable_adaptive_rank_margin_rescue=True,
        adaptive_rank_margin=0.02,
        adaptive_rank_margin_debug=True,
    )
    selected = module._selection(scores, None)
    if selected[0, 0].nonzero(as_tuple=False).flatten().tolist() != [0, 1, 2]:
        raise AssertionError(f"Close rank2/rank3 gap should rescue rank3, got {selected[0, 0].tolist()}.")
    gap = module._last_adaptive_rank_margin_debug.get("gap")
    trigger = module._last_adaptive_rank_margin_debug.get("triggered")
    rank3_ids = module._last_adaptive_rank_margin_debug.get("rank3_ids")
    if gap is None or abs(float(gap[0, 0].item()) - 0.01) > 1.0e-6:
        raise AssertionError(f"Rank2-rank3 gap not computed correctly: {gap}.")
    if trigger is None or not bool(trigger[0, 0].item()):
        raise AssertionError("Close rank2/rank3 gap did not trigger adaptive rescue.")
    if rank3_ids is None or int(rank3_ids[0, 0].item()) != 2:
        raise AssertionError(f"Rank3 expert id not recorded correctly: {rank3_ids}.")


def test_adaptive_rank_margin_no_trigger_when_clear() -> None:
    scores = torch.tensor([[[0.9, 0.8, 0.7, 0.1]]], dtype=torch.float32)
    repo = make_repo(num_experts=4)
    module = TIMECPResidual(
        repo,
        routing_mode="adaptive_rank_margin_topk2",
        enable_adaptive_rank_margin_rescue=True,
        adaptive_rank_margin=0.02,
        adaptive_rank_margin_debug=True,
    )
    selected = module._selection(scores, None)
    if selected[0, 0].nonzero(as_tuple=False).flatten().tolist() != [0, 1]:
        raise AssertionError(f"Clear rank2/rank3 gap should keep top-2 only, got {selected[0, 0].tolist()}.")
    trigger = module._last_adaptive_rank_margin_debug.get("triggered")
    if trigger is None or bool(trigger[0, 0].item()):
        raise AssertionError("Clear rank2/rank3 gap unexpectedly triggered adaptive rescue.")


def test_adaptive_rank_margin_fallback_with_fewer_than_three_experts() -> None:
    scores = torch.tensor([[[0.9, 0.8]]], dtype=torch.float32)
    repo = make_repo(num_experts=2)
    module = TIMECPResidual(
        repo,
        routing_mode="adaptive_rank_margin_topk2",
        enable_adaptive_rank_margin_rescue=True,
        adaptive_rank_margin=0.02,
        adaptive_rank_margin_debug=True,
    )
    selected = module._selection(scores, None)
    if selected[0, 0].nonzero(as_tuple=False).flatten().tolist() != [0, 1]:
        raise AssertionError(f"Fewer than three experts should fall back to top-2, got {selected[0, 0].tolist()}.")
    if module._last_adaptive_rank_margin_debug:
        raise AssertionError("Fewer-than-three fallback should not emit rank3 diagnostics.")
    if selected.device != scores.device:
        raise AssertionError("Adaptive rank-margin selection changed tensor device.")


def test_gradient_isolation() -> None:
    torch.manual_seed(3)
    base = torch.nn.Linear(16, 16)
    for param in base.parameters():
        param.requires_grad_(False)
    repo = make_repo(num_experts=2)
    repo.freeze_all()
    repo.unfreeze_expert(1)
    module = TIMECPResidual(repo, layer_norm=False)
    x = torch.randn(2, 4, 16)
    hidden = base(x)
    residual = module(hidden, force_expert_ids=[1])
    loss = residual.pow(2).mean()
    loss.backward()
    current_grad = sum(float(param.grad.detach().abs().sum()) for param in repo.experts[1].parameters() if param.grad is not None)
    previous_grads = [param.grad for param in repo.experts[0].parameters()]
    base_grads = [param.grad for param in base.parameters()]
    if current_grad <= 0.0:
        raise AssertionError("Current expert factors did not receive gradients.")
    if any(grad is not None for grad in previous_grads):
        raise AssertionError("Previous expert factors received gradients.")
    if any(grad is not None for grad in base_grads):
        raise AssertionError("Base model parameters received gradients.")


def test_no_dense_tensor_saved() -> None:
    repo = make_repo(hidden_size=16, rank=2, num_experts=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "expert_repository.pt"
        repo.save(path)
        payload = torch.load(path, map_location="cpu")
    dense_shapes = []
    for expert in payload["experts"]:
        for name, value in expert.items():
            if tuple(value.shape) == (repo.hidden_size, repo.hidden_size):
                dense_shapes.append((name, tuple(value.shape)))
    if dense_shapes:
        raise AssertionError(f"Repository saved dense H x H tensors: {dense_shapes}")



def _confirmation_args(repo_path: Path | None, **overrides: object) -> argparse.Namespace:
    values = {
        "time_adaptive_margin_sequential_confirmation": True,
        "time_confirmation_plan_only": True,
        "eval_only": True,
        "max_edits": 10,
        "time_load_repository": repo_path,
        "confirmation_routing_mode": ADAPTIVE_MARGIN_CONFIRMATION_MODE,
        "out_dir": Path("outputs/time_medmkeb_smoke/plan_only"),
        "eval_routing_modes": "",
        "time_routing_calibration": False,
        "time_post_retrain_calibration": False,
        "time_10edit_routing_repair_eval": False,
        "time_fragile_routing_repair_eval": False,
        "time_adaptive_margin_microcheck": False,
        "time_routing_calibration_grid": "",
        "time_gamma_sweep": "",
        "time_scale_init_grid": "",
        "time_overfit_grid": "",
        "time_adaptive_topk_margin": None,
        "time_anti_collapse_loss": False,
        "time_routing_margin_loss": False,
        "time_force_current_train": None,
        "device": "cuda",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _assert_confirmation_plan_rejects(args: argparse.Namespace, expected: str, env: dict[str, str] | None = None) -> None:
    try:
        build_adaptive_margin_confirmation_plan(args, environ=env or {})
    except ValueError as exc:
        if expected not in str(exc):
            raise AssertionError(f"Expected error containing {expected!r}, got {exc!r}.") from exc
        return
    raise AssertionError(f"Plan unexpectedly accepted invalid args: {args}")


def test_adaptive_margin_confirmation_plan_guards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp) / "expert_repository.pt"
        repo_path.write_bytes(b"placeholder")
        args = _confirmation_args(repo_path)
        plan = build_adaptive_margin_confirmation_plan(args, environ={})
        if plan["mode"] != ADAPTIVE_MARGIN_CONFIRMATION_MODE:
            raise AssertionError(f"Wrong confirmation mode: {plan}")
        if plan["number_of_configs"] != 1:
            raise AssertionError(f"Plan must contain exactly one config: {plan}")
        if plan["actual_run"]:
            raise AssertionError(f"Plan-only validation must not mark an actual run: {plan}")
        if not plan["eval_only"] or plan["max_edits"] != 10:
            raise AssertionError(f"Plan lost eval-only/max-edits guards: {plan}")

        _assert_confirmation_plan_rejects(_confirmation_args(repo_path, eval_only=False), "--eval-only")
        _assert_confirmation_plan_rejects(_confirmation_args(repo_path, max_edits=20), "--max-edits")
        _assert_confirmation_plan_rejects(
            _confirmation_args(repo_path, confirmation_routing_mode="adaptive_topk2_to_3_margin0p28"),
            "--routing-mode must be exactly",
        )
        _assert_confirmation_plan_rejects(
            _confirmation_args(repo_path, confirmation_routing_mode="topk3"),
            "--routing-mode must be exactly",
        )
        _assert_confirmation_plan_rejects(
            _confirmation_args(repo_path, eval_routing_modes="adaptive_margin0p25,topk3"),
            "Do not pass --eval-routing-modes",
        )
        _assert_confirmation_plan_rejects(
            _confirmation_args(repo_path, time_10edit_routing_repair_eval=True),
            "Grid/sweep/multi-config flags",
        )
        _assert_confirmation_plan_rejects(
            _confirmation_args(None),
            "--existing-expert-repository/--time-load-repository",
        )
        _assert_confirmation_plan_rejects(
            _confirmation_args(repo_path, device="cuda:2"),
            "GPU 2",
        )
        _assert_confirmation_plan_rejects(
            _confirmation_args(repo_path),
            "GPU 2",
            env={"CUDA_VISIBLE_DEVICES": "2"},
        )


def main() -> None:
    tests = [
        test_py_compile,
        test_shape,
        test_identity_cases,
        test_save_load,
        test_routing_synthetic,
        test_score_variants_and_relative_threshold,
        test_post_retrain_calibration_controls,
        test_adaptive_rank_margin_defaults_do_not_change_threshold,
        test_adaptive_rank_margin_disabled_matches_topk,
        test_adaptive_rank_margin_rank3_gap_and_trigger,
        test_adaptive_rank_margin_no_trigger_when_clear,
        test_adaptive_rank_margin_fallback_with_fewer_than_three_experts,
        test_gradient_isolation,
        test_no_dense_tensor_saved,
        test_adaptive_margin_confirmation_plan_guards,
    ]
    results = {}
    for test in tests:
        test()
        results[test.__name__] = "passed"
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
