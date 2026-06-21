from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from easyeditor.models.engram.crisp_kfac_collector import (
    collect_crisp_kfac_caches,
    select_crisp_kfac_linear_modules,
)


class TinySelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return torch.tanh(self.q_proj(x) + self.k_proj(x))


class TinyMlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.gate_proj(x)) * x


class TinyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = TinySelfAttention(dim)
        self.mlp = TinyMlp(dim)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class TinyMultimodalModel(nn.Module):
    def __init__(self, vocab_size=11, dim=4):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.image_proj = nn.Linear(3, dim, bias=False)
        self.layers = nn.ModuleList([TinyLayer(dim)])
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, image, input_ids, labels):
        text = self.text_embed(input_ids)
        image_token = self.image_proj(image).unsqueeze(1)
        hidden = torch.cat([image_token, text], dim=1)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)


def _sample(batch_size=2, seq_len=3):
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])[:batch_size, :seq_len]
    labels = torch.full((batch_size, seq_len + 1), -100)
    labels[:, 1:] = input_ids
    image = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])[:batch_size]
    return {"image": image, "input_ids": input_ids, "labels": labels}


def _loss_fn(model, sample):
    return model(**sample).loss


def test_selects_qk_gate_and_excludes_lm_head():
    model = TinyMultimodalModel()
    selected = select_crisp_kfac_linear_modules(
        model,
        [r"q_proj$", r"k_proj$", r"gate_proj$"],
        [r"lm_head$", r"image_proj$"],
    )
    assert selected == [
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.mlp.gate_proj",
    ]


def test_collects_finite_kfac_caches_and_projection_cache():
    torch.manual_seed(7)
    model = TinyMultimodalModel()
    modules = [
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.mlp.gate_proj",
    ]
    result = collect_crisp_kfac_caches(
        model,
        modules,
        [_sample()],
        _loss_fn,
        energy_threshold=0.7,
        build_projection_cache=True,
    )

    assert result["sample_count"] == 1
    assert set(result["layer_to_cache"]) == set(modules)
    assert set(result["layer_to_projection_cache"]) == set(modules)

    expected_vectors = 2 * 4
    for module_name in modules:
        cache = result["layer_to_cache"][module_name]
        assert cache["A"].shape == (4, 4)
        assert cache["B"].shape == (4, 4)
        assert cache["num_activation_vectors"] == expected_vectors
        assert cache["num_gradient_vectors"] == expected_vectors
        assert cache["forward_hook_calls"] == 1
        assert cache["gradient_hook_calls"] == 1
        assert torch.isfinite(cache["A"]).all()
        assert torch.isfinite(cache["B"]).all()
        projection = result["layer_to_projection_cache"][module_name]
        assert projection["Ua"].shape == (4, 4)
        assert projection["Ub"].shape == (4, 4)
        assert projection["M"].shape == (4, 4)

    diagnostics = {row["module_name"]: row for row in result["diagnostics"]}
    for module_name in modules:
        row = diagnostics[module_name]
        assert row["skipped"] is False
        assert row["A_rank"] > 0
        assert row["B_rank"] > 0
        assert row["mask_keep_ratio"] is not None
        assert row["cache_device"] == "cpu"
        assert row["cache_dtype"] == "float32"


def test_missing_and_oversize_modules_are_reported_as_skipped():
    model = TinyMultimodalModel()
    result = collect_crisp_kfac_caches(
        model,
        ["missing.module", "layers.0.self_attn.q_proj"],
        [_sample()],
        _loss_fn,
        max_dim=2,
        energy_threshold=0.7,
        build_projection_cache=False,
    )
    assert result["layer_to_cache"] == {}
    assert result["skipped_modules"]["missing.module"] == "module_not_found"
    assert result["skipped_modules"]["layers.0.self_attn.q_proj"] == "dim_larger_than_max_dim=2"
