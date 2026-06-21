import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

SMOKE_SPEC = importlib.util.spec_from_file_location(
    "smoke_llava_med_dsca_generation_for_tests",
    SCRIPTS / "smoke_llava_med_dsca_generation.py",
)
smoke = importlib.util.module_from_spec(SMOKE_SPEC)
sys.modules[SMOKE_SPEC.name] = smoke
SMOKE_SPEC.loader.exec_module(smoke)

from easyeditor.trainer.algs.dsca import DSCA
from easyeditor.trainer.algs.dsca_utils import DSCAContext, dsca_intervention_context, masked_mean


class IdentityLayer(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


class DummyLlavaStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([IdentityLayer()])


class DummyLlavaBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.model = DummyLlavaStack()


class DummyLlavaMedVLLM(nn.Module):
    def __init__(self, dim=4, vocab=7):
        super().__init__()
        self.llava_model = DummyLlavaBackbone(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, batch):
        hidden = batch["hidden"]
        for layer in self.llava_model.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.head(hidden))


def make_config(**overrides):
    values = {
        "model_name": "llava-med",
        "device": "cpu",
        "dsca_layer": 0,
        "dsca_layer_module": None,
        "dsca_freeze_vlm": True,
        "dsca_rank": 2,
        "dsca_gate_bottleneck": 3,
        "dsca_min_samples": 1,
        "dsca_refine_interval": 1,
        "dsca_cluster_alpha": 2.0,
        "dsca_proto_ema": 0.5,
        "dsca_tau_visual": -1.0,
        "dsca_route_temperature": 0.5,
        "dsca_distill_temperature": 0.5,
        "dsca_candidate_topk": None,
        "dsca_residual_scale": 1.0,
        "dsca_residual_apply_mask": "attention",
        "dsca_generation_mode": "cache_reuse_route",
        "dsca_generation_residual_apply_mask": "current_token",
        "dsca_generation_reuse_prefill_route": True,
        "dsca_generation_update_repository": False,
        "dsca_repository_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def full_hidden():
    return torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.0],
            ]
        ]
    )


def one_token_hidden():
    return torch.tensor([[[0.25, -0.15, 0.4, 0.1]]])


def full_masks():
    return {
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "vision_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        "prompt_mask": torch.tensor([[0, 0, 1, 1]], dtype=torch.bool),
        "answer_mask": torch.zeros(1, 4, dtype=torch.bool),
    }


def build_alg(**config_overrides):
    torch.manual_seed(7)
    return DSCA(DummyLlavaMedVLLM(), make_config(**config_overrides), lambda: None)


def activate_matching_cluster(alg):
    hidden = full_hidden()
    masks = full_masks()
    h_v = masked_mean(hidden, masks["vision_mask"])[0]
    h_f = masked_mean(hidden, masks["vision_mask"] | masks["prompt_mask"])[0]
    cid = alg.repository.create_cluster(h_f, h_v)
    dsam = alg.repository.dsams[cid]
    dsam.set_basis(torch.eye(alg.hidden_size)[: alg.rank])
    with torch.no_grad():
        dsam.W.zero_()
        dsam.b.fill_(0.2)
        dsam.gate_down.weight.zero_()
        dsam.gate_down.bias.zero_()
        dsam.gate_up.weight.zero_()
        dsam.gate_up.bias.fill_(2.0)
    alg.repository.active[cid] = True
    return cid


def run_prefill_and_cached_decode(alg, events=None):
    if events is None:
        events = []
    context = DSCAContext(batch=full_masks(), is_generation=True, debug_events=events)
    with dsca_intervention_context(alg, context):
        alg.model({"hidden": full_hidden()})
        cached = alg.model({"hidden": one_token_hidden()})
    return cached, events


def test_prefill_route_is_cached_and_cached_decode_reuses_it():
    alg = build_alg()
    activate_matching_cluster(alg)

    _cached, events = run_prefill_and_cached_decode(alg)

    assert alg._cached_generation_route_selected is not None
    assert alg._cached_generation_route_selected.tolist() == [[True]]
    cached_events = [item for item in events if item["phase"] == "cached_decode"]
    assert cached_events
    assert all(item["cached_decode_route_reused"] for item in cached_events)
    assert cached_events[-1]["active_candidate_ids"] == [0]


def test_current_token_residual_mask_applies_to_one_token_hidden():
    alg = build_alg()
    activate_matching_cluster(alg)

    _cached, events = run_prefill_and_cached_decode(alg)
    cached_events = [item for item in events if item["phase"] == "cached_decode"]

    assert cached_events[-1]["apply_mask_sum"] == 1
    assert cached_events[-1]["residual_norm"] > 0.0


def test_no_candidate_cached_decode_remains_identity():
    alg = build_alg(dsca_tau_visual=2.0)
    activate_matching_cluster(alg)
    base = alg.model({"hidden": one_token_hidden()}).logits.detach()

    cached, events = run_prefill_and_cached_decode(alg)

    assert torch.allclose(cached.logits.detach(), base)
    assert events[-1]["candidate_ids"] == []
    assert events[-1]["residual_norm"] == 0.0


def test_residual_scale_zero_cached_decode_remains_identity():
    alg = build_alg()
    activate_matching_cluster(alg)
    for dsam in alg.repository.dsams:
        dsam.residual_scale = 0.0
    base = alg.model({"hidden": one_token_hidden()}).logits.detach()

    cached, events = run_prefill_and_cached_decode(alg)

    assert torch.equal(cached.logits.detach(), base)
    assert events[-1]["cached_decode_route_reused"]
    assert events[-1]["residual_norm"] == 0.0


def test_hook_active_is_event_based_even_when_residual_is_zero():
    alg = build_alg(dsca_tau_visual=2.0)
    activate_matching_cluster(alg)

    _cached, events = run_prefill_and_cached_decode(alg)
    summary = smoke.hook_event_summary(events)

    assert summary["hook_entered"]
    assert summary["hook_event_count"] == 2
    assert summary["max_event_residual_norm"] == 0.0


def test_route_aware_acceptance_ignores_no_route_zero_residual_samples():
    base_row = {
        "empty_equals_base": True,
        "empty_repo_residual_norm": 0.0,
        "inactive_equals_base": True,
        "inactive_residual_norm": 0.0,
        "scale0_equals_base": True,
        "scale0_residual_norm": 0.0,
        "active_hidden_delta_norm": 1.0,
        "active_logits_delta_norm": 1.0,
        "logits_delta_nonzero": True,
        "masks_align": True,
        "hook_entered": True,
        "hook_error_count": 0,
        "nonfinite_event_residual_count": 0,
        "repository_unchanged_during_generation": True,
        "base_text": "base",
        "active_dsam_text": "edited",
    }
    active_row = {
        **base_row,
        "active_route_selected": True,
        "residual_nonzero": True,
        "generation_residual_norm_mean": 0.5,
    }
    no_route_row = {
        **base_row,
        "active_route_selected": False,
        "residual_nonzero": False,
        "generation_residual_norm_mean": 0.0,
    }

    summary = smoke.summarize_smoke_rows(Path("dummy.json"), [active_row, no_route_row])

    assert summary["active_route_case_count"] == 1
    assert summary["no_route_case_count"] == 1
    assert summary["active_route_nonzero_residual_rate"] == 1.0
    assert summary["active_routed_dsam_residual_nonzero"]


def test_force_route_flags_are_diagnostic_only_by_default():
    args = SimpleNamespace(force_route_all_active=False, force_route_assigned_cluster=False)
    alg = SimpleNamespace(repository=SimpleNamespace(active=torch.tensor([True, True])))

    assert smoke.force_route_ids_for(alg, assigned_cluster_id=0, args=args) is None
