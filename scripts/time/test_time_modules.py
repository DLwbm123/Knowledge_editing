#!/usr/bin/env python3
"""Static and unit checks for the TIME CP-factor implementation."""

from __future__ import annotations

import json
import py_compile
import sys
import tempfile
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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


def main() -> None:
    tests = [
        test_py_compile,
        test_shape,
        test_identity_cases,
        test_save_load,
        test_routing_synthetic,
        test_score_variants_and_relative_threshold,
        test_post_retrain_calibration_controls,
        test_gradient_isolation,
        test_no_dense_tensor_saved,
    ]
    results = {}
    for test in tests:
        test()
        results[test.__name__] = "passed"
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
