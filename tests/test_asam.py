import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

ASAM_UTILS_PATH = Path(__file__).resolve().parents[1] / "easyeditor" / "trainer" / "algs" / "asam_utils.py"
ASAM_SPEC = importlib.util.spec_from_file_location("asam_utils_for_tests", ASAM_UTILS_PATH)
asam_utils = importlib.util.module_from_spec(ASAM_SPEC)
sys.modules[ASAM_SPEC.name] = asam_utils
ASAM_SPEC.loader.exec_module(asam_utils)

asam_enabled = asam_utils.asam_enabled
asam_latent_context = asam_utils.asam_latent_context
capture_asam_representations = asam_utils.capture_asam_representations
find_asam_capture_module = asam_utils.find_asam_capture_module
generate_lar_deltas = asam_utils.generate_lar_deltas
gradient_diagnostics_for_named_params = asam_utils.gradient_diagnostics_for_named_params
gradient_diagnostics_for_params = asam_utils.gradient_diagnostics_for_params
gradient_nonzero_fraction = asam_utils.gradient_nonzero_fraction
masked_delta_norm = asam_utils.masked_delta_norm
maybe_apply_asam_delta = asam_utils.maybe_apply_asam_delta
pool_representation = asam_utils.pool_representation
rank_constrained_subspace_loss = asam_utils.rank_constrained_subspace_loss
_forward_logits = asam_utils._forward_logits


class DummyOutput(SimpleNamespace):
    pass


class DummyEditable(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, batch):
        return self.model(batch)

    def edit_loss_fn(self, config, logits, labels):
        return {"nll": F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)}


class LogitsOnlyEditable(DummyEditable):
    def forward(self, batch):
        return self.model(batch).logits


class DummyJointModel(nn.Module):
    def __init__(self, dim=5, vocab=7):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, batch):
        latent = maybe_apply_asam_delta(self, batch["latent"], perturb_mask=batch["perturb_mask"])
        hidden = self.linear(latent)
        logits = self.head(hidden)
        return DummyOutput(
            logits=logits,
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            asam_perturb_mask=batch["perturb_mask"],
        )


class DummyBlock(nn.Module):
    def __init__(self, dim=5):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


class SameBlockModel(nn.Module):
    def __init__(self, dim=5, vocab=7):
        super().__init__()
        self.block = DummyBlock(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, batch):
        latent = maybe_apply_asam_delta(self, batch["latent"], perturb_mask=batch["perturb_mask"])
        hidden = self.block(latent)
        return DummyOutput(
            logits=self.head(hidden),
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            asam_perturb_mask=batch["perturb_mask"],
        )


class MultiBlockModel(nn.Module):
    def __init__(self, dim=5, vocab=7):
        super().__init__()
        self.blocks = nn.ModuleList([DummyBlock(dim), DummyBlock(dim), DummyBlock(dim)])
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, batch):
        hidden = maybe_apply_asam_delta(self, batch["latent"], perturb_mask=batch["perturb_mask"])
        for block in self.blocks:
            hidden = block(hidden)
        return DummyOutput(
            logits=self.head(hidden),
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            asam_perturb_mask=batch["perturb_mask"],
        )


class UsedUnusedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.used = nn.Linear(2, 2, bias=False)
        self.unused = nn.Linear(2, 2, bias=False)


class MaskPoolingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(1, 1, bias=False)
        self.head = nn.Linear(1, 3, bias=False)
        with torch.no_grad():
            self.block.weight.fill_(1.0)
            self.head.weight.fill_(1.0)

    def forward(self, batch):
        latent = maybe_apply_asam_delta(self, batch["latent"], perturb_mask=batch["perturb_mask"])
        hidden = self.block(latent)
        return DummyOutput(
            logits=self.head(hidden),
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            asam_perturb_mask=batch["perturb_mask"],
        )


def make_config(**overrides):
    values = {
        "alg": "ASAM_FT",
        "inner_params": ["linear.weight"],
        "asam_use_lar": True,
        "asam_epsilon": 1.0e-3,
        "asam_num_variants": 4,
        "asam_lar_step_size": 1.0e-3,
        "asam_tau": 4.0,
        "asam_beta": 10.0,
        "asam_pooling": "prompt_or_image_only",
        "asam_capture_module": None,
        "asam_alignment_params": None,
        "asam_debug_grad_check": False,
        "asam_debug_require_all_inner_grads": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_batch(batch_size=2, seq_len=5, dim=5, vocab=7, perturb_mask=None):
    labels = torch.randint(0, vocab, (batch_size, seq_len))
    labels[:, :3] = -100
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    if perturb_mask is None:
        perturb_mask = labels.eq(-100) & attention_mask.bool()
    return {
        "latent": torch.randn(batch_size, seq_len, dim),
        "labels": labels,
        "attention_mask": attention_mask,
        "perturb_mask": perturb_mask,
    }


def test_lar_delta_masks_answer_tokens():
    torch.manual_seed(1)
    perturb_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]], dtype=torch.bool)
    batch = make_batch(perturb_mask=perturb_mask)
    config = make_config(asam_num_variants=3)
    editable = DummyEditable(DummyJointModel(), config)

    deltas, _ = generate_lar_deltas(editable, config, batch)

    assert len(deltas) == 3
    for delta in deltas:
        assert torch.all(delta[~perturb_mask] == 0)
        assert delta[perturb_mask].abs().sum() > 0
        assert torch.all(masked_delta_norm(delta, perturb_mask) <= config.asam_epsilon + 1.0e-7)


def test_lar_does_not_perturb_padding():
    torch.manual_seed(2)
    perturb_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    batch = make_batch(perturb_mask=perturb_mask)
    batch["attention_mask"] = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    config = make_config()
    editable = DummyEditable(DummyJointModel(), config)

    deltas, _ = generate_lar_deltas(editable, config, batch)

    for delta in deltas:
        assert torch.all(delta[~perturb_mask] == 0)


def test_capture_module_covers_all_inner_params_same_block():
    torch.manual_seed(3)
    config = make_config(inner_params=["block.fc1.weight", "block.fc2.weight"])
    model = SameBlockModel()
    editable = DummyEditable(model, config)
    batch = make_batch()
    module, module_name = find_asam_capture_module(model, config)

    reps = capture_asam_representations(editable, config, batch, [None, torch.randn_like(batch["latent"]) * 1.0e-4])
    loss = rank_constrained_subspace_loss(reps, tau=4.0)
    loss.backward()

    assert module is model.block
    assert module_name == "block"
    assert model.block.fc1.weight.grad is not None and model.block.fc1.weight.grad.abs().sum() > 0
    assert model.block.fc2.weight.grad is not None and model.block.fc2.weight.grad.abs().sum() > 0


def test_capture_module_covers_multiple_edited_blocks():
    torch.manual_seed(4)
    config = make_config(
        inner_params=[
            "blocks.0.fc1.weight",
            "blocks.1.fc2.weight",
            "blocks.2.fc1.weight",
        ]
    )
    model = MultiBlockModel()
    editable = DummyEditable(model, config)
    batch = make_batch()
    module, module_name = find_asam_capture_module(model, config)

    reps = capture_asam_representations(editable, config, batch, [None, torch.randn_like(batch["latent"]) * 1.0e-4])
    loss = rank_constrained_subspace_loss(reps, tau=4.0)
    loss.backward()

    assert module is model.blocks[2]
    assert module_name == "blocks.2"
    for name, param in model.named_parameters():
        if name in config.inner_params:
            assert param.grad is not None and param.grad.abs().sum() > 0, name


def test_gradient_check_counts_none_as_zero():
    model = UsedUnusedModel()
    x = torch.ones(1, 2)
    loss = model.used(x).sum()

    info = gradient_diagnostics_for_named_params(
        loss,
        model.named_parameters(),
        ["used.weight", "unused.weight"],
    )

    assert info["asam/grad_num_unused_inner_params"] == 1.0
    assert info["asam/grad_norm/unused_weight"] == 0.0
    assert info["asam/grad_total_params_checked"] == float(model.used.weight.numel() + model.unused.weight.numel())
    assert 0.0 < info["asam/grad_nonzero_fraction_all_inner_params"] < 1.0


def test_gradient_diagnostics_counts_none_as_full_param_size():
    model = UsedUnusedModel()
    x = torch.ones(1, 2)
    loss = model.used(x).sum()

    fraction = gradient_nonzero_fraction(loss, [model.used.weight, model.unused.weight])
    info = gradient_diagnostics_for_params(loss, [model.used.weight, model.unused.weight], "asam/test")

    expected_denominator = model.used.weight.numel() + model.unused.weight.numel()
    assert fraction == model.used.weight.numel() / expected_denominator
    assert info["asam/test_grad_total_params_checked"] == float(expected_denominator)
    assert info["asam/test_grad_num_unused_params"] == 1.0
    assert info["asam/test_grad_nonzero_fraction"] == fraction


def test_asam_ft_preserves_raw_model_output_for_rcsl_pooling():
    config = make_config(inner_params=["block.weight"], asam_capture_module="block")
    model = MaskPoolingModel()
    editable = LogitsOnlyEditable(model, config)
    batch = {
        "latent": torch.tensor([[[1.0], [3.0], [1000.0], [2000.0]]]),
        "labels": torch.tensor([[-100, -100, 5, 6]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "perturb_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
    }

    output, logits = _forward_logits(editable, batch)
    reps = capture_asam_representations(editable, config, batch, [None])

    assert hasattr(output, "asam_perturb_mask")
    assert output.asam_perturb_mask.shape == batch["perturb_mask"].shape
    assert logits.shape[:2] == batch["latent"].shape[:2]
    assert torch.allclose(reps[0], torch.tensor([[2.0]]))


def test_rcsl_pooling_raises_when_lar_mask_missing():
    rep = torch.randn(1, 4, 3)
    labels = torch.tensor([[7, 8]])
    attention_mask = torch.ones(1, 2, dtype=torch.long)

    try:
        pool_representation(
            rep,
            labels=labels,
            attention_mask=attention_mask,
            perturb_mask=None,
            require_reliable_mask=True,
        )
    except RuntimeError as exc:
        assert "requires a reliable full-sequence prompt/image mask" in str(exc)
    else:
        raise AssertionError("ASAM RCSL pooling should fail when LAR mask/full labels are unavailable.")


def test_rcsl_pooling_excludes_answer_tokens():
    rep = torch.tensor([[[1.0, 1.0], [3.0, 5.0], [1000.0, 1000.0], [2000.0, 2000.0]]])
    perturb_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)

    pooled = pool_representation(rep, perturb_mask=perturb_mask, require_reliable_mask=True)

    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))


def test_rcsl_anchor_detached_variants_grad():
    anchor = torch.randn(2, 5, requires_grad=True)
    variant_1 = torch.randn(2, 5, requires_grad=True)
    variant_2 = torch.randn(2, 5, requires_grad=True)

    loss = rank_constrained_subspace_loss([anchor, variant_1, variant_2], tau=4.0)
    loss.backward()

    assert anchor.grad is None or torch.all(anchor.grad == 0)
    assert variant_1.grad is not None and variant_1.grad.abs().sum() > 0
    assert variant_2.grad is not None and variant_2.grad.abs().sum() > 0


def test_rcsl_svd_on_hidden_matrix_not_gram(monkeypatch):
    calls = []
    original_svdvals = torch.linalg.svdvals

    def recording_svdvals(matrix):
        calls.append(tuple(matrix.shape))
        return original_svdvals(matrix)

    monkeypatch.setattr(torch.linalg, "svdvals", recording_svdvals)
    reps = [torch.randn(2, 5, requires_grad=(idx > 0)) for idx in range(3)]

    rank_constrained_subspace_loss(reps, tau=4.0)

    assert calls == [(3, 5), (3, 5)]


def test_batch_grouping_no_cross_example_mixing():
    reps = [torch.randn(2, 6, requires_grad=(idx > 0)) for idx in range(3)]
    loss = rank_constrained_subspace_loss(reps, tau=4.0)

    manual_losses = []
    for batch_idx in range(2):
        h_s = torch.stack([reps[0][batch_idx].detach(), reps[1][batch_idx], reps[2][batch_idx]])
        h_s = F.normalize(h_s, p=2, dim=-1, eps=1e-8)
        singular_values = torch.linalg.svdvals(h_s)
        manual_losses.append(F.cross_entropy((singular_values / 4.0).unsqueeze(0), torch.zeros(1, dtype=torch.long)))

    assert torch.allclose(loss, torch.stack(manual_losses).mean())


def test_backward_compat_forward_without_asam_delta():
    model = DummyJointModel()
    batch = make_batch(batch_size=1)
    baseline = model(batch).logits
    without_context = model(batch).logits
    with asam_latent_context(model, delta=torch.zeros_like(batch["latent"])):
        zero_delta = model(batch).logits

    assert torch.allclose(baseline, without_context)
    assert torch.allclose(baseline, zero_delta)
    assert not asam_enabled(SimpleNamespace(alg="FT", asam_enabled=False))


def test_pooling_excludes_answer_tokens_when_masks_allow_it():
    rep = torch.tensor([[[1.0], [3.0], [100.0], [200.0]]])
    labels = torch.tensor([[-100, -100, 5, 6]])
    attention_mask = torch.ones(1, 4, dtype=torch.long)
    perturb_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)

    pooled = pool_representation(rep, labels=labels, attention_mask=attention_mask, perturb_mask=perturb_mask)

    assert torch.allclose(pooled, torch.tensor([[2.0]]))
