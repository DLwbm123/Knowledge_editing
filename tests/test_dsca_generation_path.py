import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location("diagnose_dsca_generation_path", SCRIPTS / "diagnose_dsca_generation_path.py")
diag = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diag
SPEC.loader.exec_module(diag)

COMMON_SPEC = importlib.util.spec_from_file_location("dsca_medmkeb_diag_common", SCRIPTS / "dsca_medmkeb_diag_common.py")
common = importlib.util.module_from_spec(COMMON_SPEC)
sys.modules[COMMON_SPEC.name] = common
COMMON_SPEC.loader.exec_module(common)


def test_prompt_only_rank_diagnostic_helpers():
    logits = torch.tensor([0.0, 3.0, 1.0, 2.0])

    assert diag.rank_of_token(logits, 1) == 1
    assert diag.rank_of_token(logits, 2) == 3
    assert diag.token_logprob(logits, 1) > diag.token_logprob(logits, 2)


def test_residual_region_logging_separates_prompt_and_answer():
    residual = torch.zeros(1, 5, 3)
    residual[:, 0] = 1.0
    residual[:, 1:3] = 2.0
    residual[:, 3:4] = 4.0
    masks = {
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool),
        "vision_mask": torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool),
        "prompt_mask": torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.bool),
        "answer_mask": torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.bool),
    }

    norms = diag.residual_region_norms_from_tensors(residual, masks)

    assert norms["answer_residual_norm"] > norms["prompt_residual_norm"] > norms["vision_residual_norm"]
    assert norms["padding_residual_norm"] == 0.0
    assert norms["total_residual_norm"] > 0.0


def test_residual_scale_zero_context_restores_identity_setting():
    dsam_a = SimpleNamespace(residual_scale=1.5)
    dsam_b = SimpleNamespace(residual_scale=2.5)
    alg = SimpleNamespace(repository=SimpleNamespace(dsams=[dsam_a, dsam_b]))

    with diag.temporary_dsam_residual_scale(alg, 0.0):
        assert dsam_a.residual_scale == 0.0
        assert dsam_b.residual_scale == 0.0

    assert dsam_a.residual_scale == 1.5
    assert dsam_b.residual_scale == 2.5


def test_force_route_context_does_not_alter_normal_state():
    class DummyRepository:
        def __init__(self, active):
            self.active = active

        def __len__(self):
            return int(self.active.numel())

    active = torch.tensor([True, False, True])
    alg = SimpleNamespace(tau_visual=0.5, repository=DummyRepository(active.clone()))

    with common.temporarily_force_route(alg, 0):
        assert alg.tau_visual < -1.0e8
        assert alg.repository.active.tolist() == [True, False, False]

    assert alg.tau_visual == 0.5
    assert alg.repository.active.tolist() == active.tolist()


def test_diagnosis_labels_generation_hook_bypass_and_answer_leakage():
    row = {
        "generation_residual_norm_mean": 0.0,
        "teacher_forced_total_residual_norm": 1.0,
        "teacher_forced_answer_residual_norm": 0.0,
        "teacher_forced_prompt_residual_norm": 0.0,
        "teacher_forced_vision_residual_norm": 0.0,
        "first_target_edited_rank": None,
        "first_target_base_rank": None,
        "first_target_force_route_rank": None,
        "edited_equals_base": True,
    }
    assert "generation hook bypass" in diag.diagnosis_label(row)

    row.update(
        {
            "generation_residual_norm_mean": 1.0,
            "teacher_forced_total_residual_norm": 1.0,
            "teacher_forced_answer_residual_norm": 10.0,
            "teacher_forced_prompt_residual_norm": 1.0,
            "teacher_forced_vision_residual_norm": 1.0,
        }
    )
    assert diag.diagnosis_label(row) == "teacher-forcing answer-token leakage"
