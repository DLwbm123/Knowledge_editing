import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

DSCA_UTILS_PATH = Path(__file__).resolve().parents[1] / "easyeditor" / "trainer" / "algs" / "dsca_utils.py"
UTILS_SPEC = importlib.util.spec_from_file_location("dsca_utils_for_tests", DSCA_UTILS_PATH)
dsca_utils = importlib.util.module_from_spec(UTILS_SPEC)
sys.modules[UTILS_SPEC.name] = dsca_utils
UTILS_SPEC.loader.exec_module(dsca_utils)

DSAModule = dsca_utils.DSAModule
DSCAConceptRepository = dsca_utils.DSCAConceptRepository
dsca_contrastive_distill_loss = dsca_utils.dsca_contrastive_distill_loss
dsca_route = dsca_utils.dsca_route
dsca_sparse_loss = dsca_utils.dsca_sparse_loss
extract_dsca_region_representations = dsca_utils.extract_dsca_region_representations
masked_mean = dsca_utils.masked_mean
pca_basis = dsca_utils.pca_basis
residualized_pca_basis = dsca_utils.residualized_pca_basis

LLAVA_MED_PATH = Path(__file__).resolve().parents[1] / "easyeditor" / "trainer" / "llava_med_models" / "llava_med.py"
LLAVA_MED_SPEC = importlib.util.spec_from_file_location("llava_med_for_tests", LLAVA_MED_PATH)
llava_med = importlib.util.module_from_spec(LLAVA_MED_SPEC)
sys.modules[LLAVA_MED_SPEC.name] = llava_med
LLAVA_MED_SPEC.loader.exec_module(llava_med)
build_llava_med_masks = llava_med.build_llava_med_masks


class DummyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states, *args, **kwargs):
        return (torch.tanh(self.linear(hidden_states)),)


class DummyDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(dim)])


class DummyOPTInner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.decoder = DummyDecoder(dim)


class DummyOPT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.model = DummyOPTInner(dim)


class DummyVLLM(nn.Module):
    def __init__(self, dim=6, vocab=13):
        super().__init__()
        self.opt_model = DummyOPT(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, batch):
        hidden = batch["hidden"]
        for layer in self.opt_model.model.decoder.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(
            logits=self.head(hidden),
            attention_mask=batch["attention_mask"],
            vision_mask=batch["vision_mask"],
            prompt_mask=batch["prompt_mask"],
            answer_mask=batch.get("answer_mask"),
        )


class DummyLlavaStack(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(dim)])


class DummyLlavaBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.model = DummyLlavaStack(dim)


class DummyLlavaMedVLLM(nn.Module):
    def __init__(self, dim=6, vocab=13):
        super().__init__()
        self.llava_model = DummyLlavaBackbone(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, batch):
        hidden = batch["hidden"]
        for layer in self.llava_model.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(
            logits=self.head(hidden),
            attention_mask=batch["attention_mask"],
            vision_mask=batch["vision_mask"],
            prompt_mask=batch["prompt_mask"],
            answer_mask=batch.get("answer_mask"),
        )


def make_config(**overrides):
    values = {
        "model_name": "blip2",
        "model_class": "Dummy",
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
        "dsca_lambda_align": 0.5,
        "dsca_lambda_distill": 1.0,
        "dsca_lambda_sparse": 1.0e-2,
        "dsca_task_weight": 1.0,
        "dsca_update_clusters_during_training": True,
        "dsca_update_clusters_during_inference": False,
        "dsca_freeze_repository_at_eval": True,
        "dsca_require_masks": True,
        "dsca_candidate_topk": None,
        "dsca_residual_scale": 1.0,
        "dsca_residual_apply_mask": "attention",
        "dsca_repository_path": None,
        "dsca_dsam_init_std": 0.02,
        "dsca_debug": True,
        "accumulate_bs": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_batch(batch_size=2, seq_len=6, dim=6, vocab=13):
    hidden = torch.randn(batch_size, seq_len, dim)
    labels = torch.randint(0, vocab, (batch_size, 2))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    vision_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    prompt_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    answer_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    vision_mask[:, :2] = True
    prompt_mask[:, 2:4] = True
    answer_mask[:, 4:] = True
    return {
        "hidden": hidden,
        "labels": labels,
        "attention_mask": attention_mask,
        "vision_mask": vision_mask,
        "prompt_mask": prompt_mask,
        "answer_mask": answer_mask,
        "image": torch.randn(batch_size, 3, 4, 4),
        "prompts_len": [2] * batch_size,
        "text_input": ["q a"] * batch_size,
    }


def load_dsca_class():
    from easyeditor.trainer.algs.dsca import DSCA

    return DSCA


def activate_cluster(repo, h_f=None, h_v=None, basis=None):
    h_f = torch.randn(repo.hidden_size) if h_f is None else h_f
    h_v = torch.randn(repo.hidden_size) if h_v is None else h_v
    idx = repo.create_cluster(h_f, h_v)
    basis = torch.eye(repo.hidden_size)[: repo.rank] if basis is None else basis
    repo.dsams[idx].set_basis(basis)
    repo.active[idx] = True
    return idx


def test_empty_repository_identity():
    DSCA = load_dsca_class()
    torch.manual_seed(1)
    alg = DSCA(DummyVLLM(), make_config(), lambda: DummyVLLM())
    batch = make_batch()

    assert torch.allclose(alg.model(batch).logits, alg(batch).logits)


def test_no_candidate_identity():
    DSCA = load_dsca_class()
    torch.manual_seed(2)
    alg = DSCA(DummyVLLM(), make_config(dsca_tau_visual=0.99), lambda: DummyVLLM())
    batch = make_batch(batch_size=1)
    activate_cluster(alg.repository, h_f=torch.ones(6), h_v=-torch.ones(6))

    assert torch.allclose(alg.model(batch).logits, alg(batch).logits)


def test_inactive_dsam_identity():
    DSCA = load_dsca_class()
    torch.manual_seed(3)
    alg = DSCA(DummyVLLM(), make_config(), lambda: DummyVLLM())
    batch = make_batch(batch_size=1)
    alg.repository.create_cluster(torch.ones(6), torch.ones(6))

    assert torch.allclose(alg.model(batch).logits, alg(batch).logits)


def test_dsam_residual_shape_and_gate_range():
    torch.manual_seed(4)
    dsam = DSAModule(hidden_size=5, rank=2, gate_bottleneck=3)
    dsam.set_basis(torch.eye(5)[:2])
    hidden = torch.randn(2, 4, 5)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)

    residual = dsam(hidden, mask)

    assert residual.shape == hidden.shape
    assert torch.all(residual[~mask] == 0)
    assert dsam.last_gate.min() >= 0 and dsam.last_gate.max() <= 1


def test_dsam_fp16_residual_is_finite_and_scale_zero_is_identity():
    torch.manual_seed(44)
    dsam = DSAModule(hidden_size=16, rank=4, gate_bottleneck=4)
    dsam.set_basis(torch.eye(16)[:4])
    hidden = (torch.randn(2, 3, 16) * 2048.0).half()
    hidden[0, 0, 0] = float("inf")
    residual = dsam(hidden, torch.ones(2, 3, dtype=torch.bool))

    assert residual.dtype == hidden.dtype
    assert torch.isfinite(residual.float()).all()

    dsam.residual_scale = 0.0
    zero = dsam(hidden, torch.ones(2, 3, dtype=torch.bool))

    assert torch.equal(zero, torch.zeros_like(hidden))


def test_dsam_basis_not_trainable():
    dsam = DSAModule(hidden_size=5, rank=2, gate_bottleneck=3)

    assert "R" in dict(dsam.named_buffers())
    assert not dsam.R.requires_grad
    assert dsam.W.requires_grad and dsam.b.requires_grad
    assert all(param.requires_grad for param in dsam.gate_down.parameters())


def test_cluster_creation_and_ema_update():
    repo = DSCAConceptRepository(hidden_size=3, rank=1, gate_bottleneck=2, min_samples=10, proto_ema=0.5)

    first, created_first = repo.assign_or_create(torch.tensor([1.0, 0, 0]), torch.tensor([1.0, 0, 0]))
    old_proto = repo.p_f[first].clone()
    second, created_second = repo.assign_or_create(torch.tensor([1.1, 0, 0]), torch.tensor([1.0, 0.1, 0]))
    third, created_third = repo.assign_or_create(torch.tensor([10.0, 0, 0]), torch.tensor([0, 1.0, 0]))

    assert created_first and not created_second and created_third
    assert first == second and third != first
    assert not torch.allclose(repo.p_f[first], old_proto)
    assert len(repo) == 2


def test_pca_initializes_basis_after_min_samples():
    repo = DSCAConceptRepository(hidden_size=4, rank=2, gate_bottleneck=2, min_samples=2)
    cid = repo.create_cluster(torch.tensor([1.0, 0, 0, 0]), torch.tensor([1.0, 0, 0, 0]))
    repo.append_to_buffer(cid, torch.tensor([0.0, 1.0, 0, 0]))

    assert repo.initialize_basis_if_ready(cid)
    assert repo.dsams[cid].active
    assert torch.allclose(repo.dsams[cid].R @ repo.dsams[cid].R.t(), torch.eye(2), atol=1.0e-5)


def test_residualized_pca_reduces_overlap():
    features_a = torch.tensor([[3.0, 0.1, 0.0], [2.8, -0.1, 0.0], [3.2, 0.0, 0.0]])
    features_b = torch.tensor([[3.0, 0.0, 0.1], [2.8, 0.0, -0.1], [3.2, 0.0, 0.0]])
    basis_a = pca_basis(features_a, rank=1)
    naive_b = pca_basis(features_b, rank=1)
    resid_b = residualized_pca_basis(features_b, rank=1, prior_bases=[basis_a])

    naive_overlap = (basis_a @ naive_b.t()).pow(2).sum()
    resid_overlap = (basis_a @ resid_b.t()).pow(2).sum()

    assert resid_overlap < naive_overlap


def test_visual_routing_threshold():
    repo = DSCAConceptRepository(hidden_size=2, rank=1, gate_bottleneck=2, min_samples=1)
    activate_cluster(repo, h_f=torch.tensor([1.0, 0.0]), h_v=torch.tensor([1.0, 0.0]), basis=torch.tensor([[1.0, 0.0]]))
    activate_cluster(repo, h_f=torch.tensor([0.0, 1.0]), h_v=torch.tensor([-1.0, 0.0]), basis=torch.tensor([[0.0, 1.0]]))

    _, selected, _ = dsca_route(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), repo, 0.0, 1.0)

    assert selected.tolist() == [[True, False]]


def test_fused_soft_routing_over_candidates_only():
    repo = DSCAConceptRepository(hidden_size=2, rank=1, gate_bottleneck=2, min_samples=1)
    activate_cluster(repo, h_f=torch.tensor([1.0, 0.0]), h_v=torch.tensor([1.0, 0.0]), basis=torch.tensor([[1.0, 0.0]]))
    activate_cluster(repo, h_f=torch.tensor([0.0, 1.0]), h_v=torch.tensor([1.0, 0.0]), basis=torch.tensor([[0.0, 1.0]]))
    activate_cluster(repo, h_f=torch.tensor([-1.0, 0.0]), h_v=torch.tensor([-1.0, 0.0]), basis=torch.tensor([[1.0, 0.0]]))

    weights, selected, _ = dsca_route(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), repo, 0.0, 0.5)

    assert selected.tolist() == [[True, True, False]]
    assert weights[0, 2].item() == 0
    assert torch.allclose(weights[0, :2].sum(), torch.tensor(1.0))


def test_per_example_routing_no_batch_mixing():
    repo = DSCAConceptRepository(hidden_size=2, rank=1, gate_bottleneck=2, min_samples=1)
    activate_cluster(repo, h_f=torch.tensor([1.0, 0.0]), h_v=torch.tensor([1.0, 0.0]), basis=torch.tensor([[1.0, 0.0]]))
    activate_cluster(repo, h_f=torch.tensor([0.0, 1.0]), h_v=torch.tensor([0.0, 1.0]), basis=torch.tensor([[0.0, 1.0]]))

    weights, selected, _ = dsca_route(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        repo,
        0.5,
        1.0,
    )

    assert selected.tolist() == [[True, False], [False, True]]
    assert weights[0, 0] > 0 and weights[0, 1] == 0
    assert weights[1, 1] > 0 and weights[1, 0] == 0


def test_masks_exclude_answer_and_padding_from_features():
    hidden = torch.randn(1, 6, 4)
    masks = {
        "vision_mask": torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool),
        "prompt_mask": torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.bool),
        "answer_mask": torch.tensor([[0, 0, 0, 0, 1, 0]], dtype=torch.bool),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool),
    }
    changed = hidden.clone()
    changed[:, 4:] = 10000.0

    reps_a = extract_dsca_region_representations(hidden, masks)
    reps_b = extract_dsca_region_representations(changed, masks)

    assert torch.allclose(reps_a["h_v"], reps_b["h_v"])
    assert torch.allclose(reps_a["h_t"], reps_b["h_t"])
    assert torch.allclose(reps_a["h_f"], reps_b["h_f"])


def test_task_loss_masks_answer_only():
    DSCA = load_dsca_class()
    alg = DSCA(DummyVLLM(), make_config(), lambda: DummyVLLM())
    labels = torch.tensor([[3, 4]])
    logits = torch.zeros(1, 6, 13)
    changed = logits.clone()
    changed[:, :2] = 10000.0

    loss_a = alg.edit_loss_fn(alg.config, logits, labels)["nll"]
    loss_b = alg.edit_loss_fn(alg.config, changed, labels)["nll"]

    assert torch.allclose(loss_a, loss_b)


def test_cdistill_uses_teacher_with_dsca_disabled():
    edited = torch.randn(2, 4, requires_grad=True)
    teacher = torch.randn(2, 4, requires_grad=True)

    loss = dsca_contrastive_distill_loss(edited, teacher, temperature=0.5)
    loss.backward()

    assert edited.grad is not None
    assert teacher.grad is None


def test_sparse_loss_on_replay_weights():
    weights = torch.tensor([[0.0, 0.0], [0.25, 0.75]])

    assert dsca_sparse_loss(torch.zeros_like(weights)).item() == 0
    assert dsca_sparse_loss(weights).item() > 0


def test_base_vlm_frozen_training_step():
    DSCA = load_dsca_class()
    torch.manual_seed(5)
    alg = DSCA(DummyVLLM(), make_config(), lambda: DummyVLLM())
    batch = {"edit_inner": make_batch(), "loc_image": make_batch(), "loc": make_batch()}
    before = {name: param.detach().clone() for name, param in alg.model.named_parameters()}
    opt = torch.optim.SGD(alg.outer_parameters(), lr=0.01)

    opt.zero_grad()
    alg.edit_step(batch, training=True, optimizer=opt)
    opt.step()

    assert all(not param.requires_grad for param in alg.model.parameters())
    for name, param in alg.model.named_parameters():
        assert torch.allclose(param.detach(), before[name])
    active_grads = [
        float(param.grad.abs().sum())
        for dsam in alg.repository.dsams
        if dsam.active
        for param in dsam.parameters()
        if param.grad is not None
    ]
    assert sum(active_grads) > 0


def test_dynamic_optimizer_adds_new_dsam_params():
    DSCA = load_dsca_class()
    alg = DSCA(DummyVLLM(), make_config(), lambda: DummyVLLM())
    opt = torch.optim.SGD(alg.outer_parameters(), lr=0.01)
    batch = {"edit_inner": make_batch(batch_size=1), "loc_image": make_batch(batch_size=1), "loc": make_batch(batch_size=1)}

    before = sum(len(group["params"]) for group in opt.param_groups)
    alg.edit_step(batch, training=True, optimizer=opt)
    after_first = sum(len(group["params"]) for group in opt.param_groups)
    added_again = alg.dsca_register_new_params_with_optimizer(opt)

    assert after_first > before
    assert added_again == 0


def test_repository_save_load(tmp_path):
    repo = DSCAConceptRepository(hidden_size=4, rank=2, gate_bottleneck=3, min_samples=1)
    cid = activate_cluster(repo, h_f=torch.tensor([1.0, 0, 0, 0]), h_v=torch.tensor([0, 1.0, 0, 0]))
    repo.metadata[cid] = {"id": "cluster"}
    path = tmp_path / "dsca_repo.pt"

    repo.save(str(path))
    loaded = DSCAConceptRepository.load(str(path))

    assert torch.allclose(loaded.p_f, repo.p_f)
    assert torch.allclose(loaded.p_v, repo.p_v)
    assert loaded.metadata == repo.metadata
    assert loaded.dsams[0].active
    assert torch.allclose(loaded.dsams[0].R, repo.dsams[0].R)


def test_layer_not_found_raises():
    DSCA = load_dsca_class()
    with pytest.raises(ValueError, match="could not resolve"):
        DSCA(DummyVLLM(), make_config(dsca_layer_module="opt_model.model.decoder.layers.99"), lambda: DummyVLLM())


def test_unsupported_backbone_raises():
    DSCA = load_dsca_class()
    with pytest.raises(NotImplementedError, match="Stage 1 supports BLIP2 and MiniGPT-4"):
        DSCA(DummyVLLM(), make_config(model_name="llava"), lambda: DummyVLLM())


def test_llava_med_verified_backbone_uses_mistral_layer_path():
    DSCA = load_dsca_class()
    alg = DSCA(DummyLlavaMedVLLM(), make_config(model_name="llava-med"), lambda: DummyLlavaMedVLLM())

    assert alg.dsca_layer_path == "llava_model.model.layers.0"


def test_llava_med_masks_align_after_image_expansion():
    image_token = -200
    token_ids = torch.tensor([1, image_token, 2, 3, 4])
    labels = torch.tensor([-100, -100, -100, -100, -100, 99, -100])
    attention = torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=torch.bool)

    masks = build_llava_med_masks(
        token_ids=token_ids,
        labels=labels,
        expanded_attention_mask=attention,
        image_token_index=image_token,
        image_feature_len=3,
    )

    assert masks["vision_mask"].tolist() == [False, True, True, True, False, False, False]
    assert masks["answer_mask"].tolist() == [False, False, False, False, False, True, False]
    assert masks["prompt_mask"].tolist() == [True, False, False, False, True, False, False]
    assert not ((masks["vision_mask"] | masks["prompt_mask"] | masks["answer_mask"]) & ~masks["attention_mask"]).any()


def test_forward_without_dsca_matches_base():
    DSCA = load_dsca_class()
    alg = DSCA(DummyVLLM(), make_config(), lambda: DummyVLLM())
    batch = make_batch()

    assert torch.allclose(alg.model(batch).logits, alg(batch).logits)


def test_no_liveedit_or_asam_regression_imports():
    assert hasattr(dsca_utils, "DSCAConceptRepository")
    assert hasattr(dsca_utils, "DSAModule")
