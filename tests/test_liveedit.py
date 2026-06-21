import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

LIVEEDIT_UTILS_PATH = Path(__file__).resolve().parents[1] / "easyeditor" / "trainer" / "algs" / "liveedit_utils.py"
UTILS_SPEC = importlib.util.spec_from_file_location("liveedit_utils_for_tests", LIVEEDIT_UTILS_PATH)
liveedit_utils = importlib.util.module_from_spec(UTILS_SPEC)
sys.modules[UTILS_SPEC.name] = liveedit_utils
UTILS_SPEC.loader.exec_module(liveedit_utils)

CrossAttentionReadout = liveedit_utils.CrossAttentionReadout
ExpertGenerator = liveedit_utils.ExpertGenerator
ExpertRepository = liveedit_utils.ExpertRepository
LiveEditFeatureExtractor = liveedit_utils.LiveEditFeatureExtractor
apply_liveedit_residual_to_output = liveedit_utils.apply_liveedit_residual_to_output
first_hidden_from_output = liveedit_utils.first_hidden_from_output
get_liveedit_masks = liveedit_utils.get_liveedit_masks
get_liveedit_routing_masks = liveedit_utils.get_liveedit_routing_masks
hard_route = liveedit_utils.hard_route
liveedit_similarity = liveedit_utils.liveedit_similarity
liveedit_routing_losses = liveedit_utils.liveedit_routing_losses
low_rank_residual = liveedit_utils.low_rank_residual
soft_routing_weights = liveedit_utils.soft_routing_weights


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
    def __init__(self, dim=6, vocab=11):
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
            labels=batch["labels"],
            vision_mask=batch["vision_mask"],
            prompt_mask=batch["prompt_mask"],
            answer_mask=batch.get("answer_mask"),
        )


def make_batch(batch_size=2, seq_len=6, dim=6, vocab=11):
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


def make_config(**overrides):
    values = {
        "model_name": "blip2",
        "model_class": "Dummy",
        "device": "cpu",
        "liveedit_layer": 0,
        "liveedit_module_dim": 8,
        "liveedit_feature_k": 2,
        "liveedit_rank": 3,
        "liveedit_lora_scale": 5,
        "liveedit_cross_att_heads": 2,
        "liveedit_sentinel_tokens": 2,
        "liveedit_similarity": "inner_product",
        "liveedit_force_topk_when_empty": False,
        "liveedit_freeze_vllm": True,
        "liveedit_rel_weight": 1.0,
        "liveedit_gen_weight": 1.0,
        "liveedit_loc_weight": 1.0,
        "liveedit_route_weight": 1.0,
        "liveedit_hr_weight": 1.0,
        "liveedit_sr1_weight": 1.0,
        "liveedit_sr2_weight": 1.0,
        "accumulate_bs": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def load_liveedit_class():
    return load_liveedit_module().LiveEdit


def load_liveedit_module():
    import easyeditor.trainer.algs.liveedit as liveedit_module

    return liveedit_module


def test_empty_repository_identity():
    LiveEdit = load_liveedit_class()
    torch.manual_seed(1)
    base = DummyVLLM()
    alg = LiveEdit(base, make_config(), lambda: DummyVLLM())
    batch = make_batch()

    expected = base(batch).logits
    actual = alg(batch).logits

    assert torch.allclose(actual, expected)


def test_no_selected_experts_identity(monkeypatch):
    liveedit_module = load_liveedit_module()
    LiveEdit = liveedit_module.LiveEdit
    torch.manual_seed(11)
    alg = LiveEdit(DummyVLLM(), make_config(), lambda: DummyVLLM())
    batch = make_batch(batch_size=1)
    alg.repository.append(
        torch.randn(2, 3, 6),
        torch.randn(2, 3, 6),
        torch.randn(2, 2, 8),
        torch.randn(2, 2, 8),
    )

    def no_selected(phi_bar, phi_hat, phi_theta, **kwargs):
        selected = torch.zeros(phi_bar.shape[0], phi_hat.shape[0], dtype=torch.bool, device=phi_bar.device)
        sim = torch.zeros(phi_bar.shape[0], phi_hat.shape[0], dtype=phi_bar.dtype, device=phi_bar.device)
        threshold = torch.zeros(phi_bar.shape[0], 1, dtype=phi_bar.dtype, device=phi_bar.device)
        return selected, sim, threshold

    monkeypatch.setattr(liveedit_module, "hard_route", no_selected)
    expected = alg.model(batch).logits
    actual = alg(batch).logits

    assert torch.allclose(actual, expected)


def test_forward_without_liveedit_matches_base():
    LiveEdit = load_liveedit_class()
    torch.manual_seed(12)
    alg = LiveEdit(DummyVLLM(), make_config(), lambda: DummyVLLM())
    batch = make_batch()

    assert torch.allclose(alg.model(batch).logits, alg(batch).logits)


@dataclass
class DummyModelOutput:
    last_hidden_state: torch.Tensor
    extra: torch.Tensor


def test_hook_preserves_output_structure():
    hidden = torch.randn(2, 3, 4)
    residual = torch.ones_like(hidden)

    tensor_out = apply_liveedit_residual_to_output(hidden, residual)
    tuple_out = apply_liveedit_residual_to_output((hidden, "cache"), residual)
    list_out = apply_liveedit_residual_to_output([hidden, "cache"], residual)
    model_out = apply_liveedit_residual_to_output(DummyModelOutput(hidden, torch.tensor(1.0)), residual)

    assert torch.allclose(tensor_out, hidden + residual)
    assert isinstance(tuple_out, tuple) and torch.allclose(tuple_out[0], hidden + residual)
    assert isinstance(list_out, list) and torch.allclose(list_out[0], hidden + residual)
    assert isinstance(model_out, DummyModelOutput)
    assert torch.allclose(first_hidden_from_output(model_out), hidden + residual)
    assert model_out.extra.item() == 1.0


def test_liveedit_layer_not_found_raises():
    LiveEdit = load_liveedit_class()
    with pytest.raises(ValueError, match="could not resolve"):
        LiveEdit(DummyVLLM(), make_config(liveedit_layer=99), lambda: DummyVLLM())


def test_expert_residual_shape():
    hidden = torch.randn(2, 5, 4, dtype=torch.float32)
    u = torch.randn(3, 2, 4)
    v = torch.randn(3, 2, 4)
    weights = torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.25, 0.75]])

    residual = low_rank_residual(hidden, u, v, weights)

    assert residual.shape == hidden.shape
    assert residual.dtype == hidden.dtype
    assert residual.device == hidden.device


def test_low_rank_residual_orientation():
    hidden = torch.tensor([[[1.0, 2.0]]])
    u = torch.tensor([[[3.0, 5.0]]])
    v = torch.tensor([[[7.0, 11.0]]])
    weights = torch.tensor([[0.5]])

    residual = low_rank_residual(hidden, u, v, weights)

    expected_activation = torch.relu(torch.tensor(13.0))
    expected = torch.tensor([[[expected_activation * 7.0 * 0.5, expected_activation * 11.0 * 0.5]]])
    assert torch.allclose(residual, expected)


def test_soft_routing_weights():
    psi_bar = torch.tensor([[[1.0, 0.0]]])
    psi_hat = torch.tensor([[[2.0, 0.0]], [[1.0, 0.0]], [[-1.0, 0.0]]])
    selected = torch.tensor([[True, False, True]])

    weights, sims = soft_routing_weights(psi_bar, psi_hat, selected)
    selected_sims = sims[0, [0, 2]]
    expected = torch.sigmoid(selected_sims) * torch.softmax(selected_sims, dim=0)

    assert torch.allclose(weights[0, [0, 2]], expected)
    assert weights[0, 1] == 0


def test_unselected_experts_zero_weight():
    psi_bar = torch.randn(2, 2, 4)
    psi_hat = torch.randn(3, 2, 4)
    selected = torch.tensor([[False, False, False], [True, False, True]])

    weights, _ = soft_routing_weights(psi_bar, psi_hat, selected)

    assert torch.count_nonzero(weights[0]).item() == 0
    assert weights[1, 1].item() == 0


def test_hard_routing_sentinel_threshold():
    phi_bar = torch.tensor([[[1.0, 0.0]]])
    phi_hat = torch.tensor([[[2.0, 0.0]], [[0.5, 0.0]], [[-1.0, 0.0]]])
    phi_theta = torch.tensor([[[1.0, 0.0]]])

    selected, _, _ = hard_route(phi_bar, phi_hat, phi_theta)

    assert selected.tolist() == [[True, False, False]]


def test_hard_routing_no_forced_topk_by_default():
    phi_bar = torch.tensor([[[1.0, 0.0]]])
    phi_hat = torch.tensor([[[0.1, 0.0]], [[0.2, 0.0]]])
    phi_theta = torch.tensor([[[10.0, 0.0]]])

    selected, _, _ = hard_route(phi_bar, phi_hat, phi_theta, force_topk_when_empty=False)

    assert selected.tolist() == [[False, False]]


def test_hard_routing_per_input_threshold():
    phi_bar = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    phi_hat = torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]])
    phi_theta = torch.tensor([[[1.5, 0.0]], [[0.0, 3.0]]])

    selected, _, threshold = hard_route(phi_bar, phi_hat, phi_theta)

    assert selected.tolist() == [[True, False], [False, False]]
    assert threshold.shape == (2, 1)


def test_feature_pooling_uses_masks():
    torch.manual_seed(2)
    extractor = LiveEditFeatureExtractor(hidden_size=4, feature_k=2, module_dim=8, num_heads=2)
    hidden = torch.randn(1, 6, 4)
    vision_mask = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    prompt_mask = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.bool)
    changed = hidden.clone()
    changed[:, 4:] = 10000.0

    phi_1, psi_1 = extractor.extract(hidden, vision_mask, prompt_mask)
    phi_2, psi_2 = extractor.extract(changed, vision_mask, prompt_mask)

    assert torch.allclose(phi_1, phi_2)
    assert torch.allclose(psi_1, psi_2)


def test_masks_align_with_hidden_length():
    batch = make_batch()
    hidden = batch["hidden"][:, :-1]

    with pytest.raises(RuntimeError, match="does not match hidden sequence"):
        get_liveedit_masks(batch, hidden)


def test_routing_masks_allow_answer_free_inference():
    batch = make_batch(batch_size=1)
    hidden = batch["hidden"]
    batch.pop("answer_mask")

    vision_mask, prompt_mask, attention_mask = get_liveedit_routing_masks(batch, hidden)

    assert vision_mask.shape == prompt_mask.shape == attention_mask.shape == hidden.shape[:2]


def test_answer_free_inference_with_repository():
    LiveEdit = load_liveedit_class()
    torch.manual_seed(13)
    alg = LiveEdit(
        DummyVLLM(),
        make_config(liveedit_force_topk_when_empty=True),
        lambda: DummyVLLM(),
    )
    batch = make_batch(batch_size=1)
    batch.pop("answer_mask")
    u = torch.randn(1, 3, 6)
    v = torch.randn(1, 3, 6)
    phi = torch.randn(1, 2, 8)
    psi = torch.randn(1, 2, 8)

    outputs = alg.forward_with_experts(batch, u, v, phi, psi, force_all_experts=True)

    assert outputs.logits.shape[:2] == batch["hidden"].shape[:2]


def test_cross_attention_mask_blocks_padding():
    torch.manual_seed(14)
    readout = CrossAttentionReadout(query_dim=4, token_dim=5, module_dim=8, output_dim=6, num_heads=2)
    query = torch.randn(1, 2, 4)
    tokens = torch.randn(1, 4, 5)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    changed = tokens.clone()
    changed[:, 2:] = 10000.0

    assert torch.allclose(readout(query, tokens, mask), readout(query, changed, mask))


def test_cross_attention_shape_dtype_device():
    readout = CrossAttentionReadout(query_dim=4, token_dim=5, module_dim=8, output_dim=6, num_heads=2)
    query = torch.randn(2, 3, 4)
    tokens = torch.randn(2, 5, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)

    out = readout(query, tokens, mask)

    assert out.shape == (2, 3, 6)
    assert out.dtype == tokens.dtype
    assert out.device == tokens.device


def test_cross_attention_no_nan_fp16_if_supported():
    readout = CrossAttentionReadout(query_dim=4, token_dim=5, module_dim=8, output_dim=6, num_heads=2).half()
    query = torch.randn(1, 2, 4, dtype=torch.float16)
    tokens = torch.randn(1, 3, 5, dtype=torch.float16)
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)

    try:
        out = readout(query, tokens, mask)
    except RuntimeError as exc:
        pytest.skip(f"fp16 CPU attention not supported in this torch build: {exc}")
    assert torch.isfinite(out).all()


def test_expert_repository_save_load(tmp_path):
    repo = ExpertRepository(rank=2, hidden_size=4, feature_k=3, module_dim=5)
    u = torch.randn(2, 2, 4)
    v = torch.randn(2, 2, 4)
    phi = torch.randn(2, 3, 5)
    psi = torch.randn(2, 3, 5)
    metadata = [{"id": 1}, {"id": 2}]
    repo.append(u, v, phi, psi, metadata=metadata)
    path = tmp_path / "repo.pt"

    repo.save(str(path))
    loaded = ExpertRepository.load(str(path))

    assert torch.allclose(loaded.u, repo.u)
    assert torch.allclose(loaded.v, repo.v)
    assert torch.allclose(loaded.phi, repo.phi)
    assert torch.allclose(loaded.psi, repo.psi)
    assert loaded.metadata == metadata


def test_repository_append_batch_clear_dtype_and_detach():
    repo = ExpertRepository(rank=2, hidden_size=4, feature_k=3, module_dim=5)
    u = torch.randn(2, 2, 4, requires_grad=True)
    v = torch.randn(2, 2, 4, requires_grad=True)
    phi = torch.randn(2, 3, 5, requires_grad=True)
    psi = torch.randn(2, 3, 5, requires_grad=True)

    repo.append(u, v, phi, psi, metadata=[{"id": 1}, {"id": 2}])
    repo.to(dtype=torch.float64)

    assert len(repo) == 2
    assert repo.u.dtype == torch.float64
    assert repo.metadata == [{"id": 1}, {"id": 2}]
    assert not repo.u.requires_grad
    repo.clear()
    assert len(repo) == 0
    assert repo.metadata == []


def test_repository_invalid_load_errors(tmp_path):
    path = tmp_path / "bad_repo.pt"
    torch.save({"state_dict": {"u": torch.empty(0, 2, 4)}}, path)

    with pytest.raises(RuntimeError, match="missing tensors"):
        ExpertRepository.load(str(path))


def test_liveedit_training_freezes_base_vllm():
    LiveEdit = load_liveedit_class()
    torch.manual_seed(3)
    alg = LiveEdit(DummyVLLM(), make_config(), lambda: DummyVLLM())
    batch = {
        "edit_inner": make_batch(),
        "edit_outer": make_batch(),
        "edit_outer_image": make_batch(),
        "loc": make_batch(),
        "loc_image": make_batch(),
        "port": None,
    }
    batch["loc"]["image"] = None
    batch["loc"]["vision_mask"] = torch.zeros_like(batch["loc"]["vision_mask"])
    base_before = {name: param.detach().clone() for name, param in alg.model.named_parameters()}
    editor_before = [param.detach().clone() for param in alg.outer_parameters()]
    opt = torch.optim.SGD(alg.outer_parameters(), lr=0.01)

    opt.zero_grad()
    alg.edit_step(batch, training=True)
    opt.step()

    assert all(not param.requires_grad for param in alg.model.parameters())
    assert all(param.grad is None for param in alg.model.parameters())
    for name, param in alg.model.named_parameters():
        assert torch.allclose(param.detach(), base_before[name])
    editor_grad = sum(
        float(param.grad.abs().sum()) for param in alg.outer_parameters() if param.grad is not None
    )
    editor_changed = any(
        not torch.allclose(param.detach(), before) for param, before in zip(alg.outer_parameters(), editor_before)
    )
    assert editor_grad > 0
    assert editor_changed


def test_locality_target_mask_uses_answer_only():
    liveedit_module = load_liveedit_module()
    batch = make_batch()
    logits = torch.randn(batch["hidden"].shape[0], batch["hidden"].shape[1], 11)
    outputs = SimpleNamespace(answer_mask=batch["answer_mask"], attention_mask=batch["attention_mask"])

    mask = liveedit_module._target_mask(outputs, batch, logits)

    assert torch.equal(mask.bool(), batch["answer_mask"])


def test_routing_losses_are_finite():
    torch.manual_seed(4)
    tensors = [torch.randn(3, 2, 5, requires_grad=True) for _ in range(8)]

    losses = liveedit_routing_losses(*tensors)
    loss = losses["hr"] + losses["sr1"] + losses["sr2"]
    loss.backward()

    assert torch.isfinite(loss)
    assert all(t.grad is not None for t in tensors)


def test_routing_loss_uses_row_local_sentinel_terms():
    torch.manual_seed(15)
    phi_hat_g = torch.randn(2, 1, 3, requires_grad=True)
    psi_hat_g = torch.randn(2, 1, 3, requires_grad=True)
    phi_bar_g = torch.randn(2, 1, 3, requires_grad=True)
    psi_bar_g = torch.randn(2, 1, 3, requires_grad=True)
    phi_bar_l = torch.randn(2, 1, 3, requires_grad=True)
    psi_hat_l = torch.randn(2, 1, 3, requires_grad=True)
    phi_theta_g = torch.randn(2, 1, 3, requires_grad=True)
    phi_theta_l = torch.randn(2, 1, 3, requires_grad=True)

    losses = liveedit_routing_losses(
        phi_hat_g,
        psi_hat_g,
        phi_bar_g,
        psi_bar_g,
        phi_bar_l,
        psi_hat_l,
        phi_theta_g,
        phi_theta_l,
    )
    manual_l_edit = liveedit_similarity(phi_bar_l, phi_hat_g, "inner_product", phi_hat_g.shape[-1])
    manual_l_sentinel = liveedit_similarity(phi_bar_l, phi_theta_l, "inner_product", phi_hat_g.shape[-1]).diag().unsqueeze(1)
    manual_hard_l = F.cross_entropy(
        torch.cat([manual_l_edit, manual_l_sentinel], dim=1),
        torch.full((2,), 2, dtype=torch.long),
    )

    assert torch.allclose(losses["hard_l"], manual_hard_l)


def test_sr1_negative_excludes_positive_for_batch_one():
    tensors = [torch.randn(1, 2, 4, requires_grad=True) for _ in range(8)]

    losses = liveedit_routing_losses(*tensors)

    assert torch.isfinite(losses["sr1"])


def test_batch_temporary_experts_keep_grad():
    torch.manual_seed(5)
    generator = ExpertGenerator(hidden_size=4, rank=2, module_dim=8, num_heads=2, lora_scale=5)
    hidden = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    u, v = generator(hidden, mask)
    weights = torch.ones(2, 2)
    residual = low_rank_residual(hidden, u, v, weights)
    loss = residual.pow(2).mean()
    loss.backward()

    grad = sum(float(param.grad.abs().sum()) for param in generator.parameters() if param.grad is not None)
    assert grad > 0


def test_repository_experts_are_detached():
    repo = ExpertRepository(rank=2, hidden_size=4, feature_k=2, module_dim=3)
    tensors = [torch.randn(1, *shape, requires_grad=True) for shape in [(2, 4), (2, 4), (2, 3), (2, 3)]]

    repo.append(*tensors)

    assert not repo.u.requires_grad
    assert not repo.v.requires_grad
    assert not repo.phi.requires_grad
    assert not repo.psi.requires_grad


def test_each_edit_generates_independent_expert():
    torch.manual_seed(16)
    generator = ExpertGenerator(hidden_size=4, rank=2, module_dim=8, num_heads=2, lora_scale=5)
    hidden = torch.randn(2, 5, 4)
    hidden[1] += 5.0
    mask = torch.ones(2, 5, dtype=torch.bool)

    u, v = generator(hidden, mask)

    assert u.shape[0] == 2 and v.shape[0] == 2
    assert not torch.allclose(u[0], u[1])


def test_lifelong_append_and_inference():
    LiveEdit = load_liveedit_class()
    torch.manual_seed(6)
    alg = LiveEdit(
        DummyVLLM(),
        make_config(liveedit_force_topk_when_empty=True),
        lambda: DummyVLLM(),
    )
    batch = make_batch(batch_size=1)
    u = torch.randn(2, 3, 6)
    v = torch.randn(2, 3, 6)
    phi = torch.randn(2, 2, 8)
    psi = torch.randn(2, 2, 8)
    alg.repository.append(u, v, phi, psi, metadata=[{"id": 1}, {"id": 2}])

    base_logits = alg.model(batch).logits
    edited_logits = alg.forward_with_experts(
        batch,
        alg.repository.u,
        alg.repository.v,
        alg.repository.phi,
        alg.repository.psi,
        force_all_experts=True,
    ).logits

    assert len(alg.repository) == 2
    assert not torch.allclose(base_logits, edited_logits)
