import torch

from easyeditor.models.engram import EngramMultimodalHparams
from easyeditor.models.engram.engram_main import EngramMultimodalRewriteExecutor, select_linear_layers


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return str(text).split()

    def __call__(self, texts, add_special_tokens=False, return_tensors=None, padding=False):
        del add_special_tokens, return_tensors
        if isinstance(texts, str):
            texts = [texts]
        lengths = [max(1, len(str(text).split())) for text in texts]
        width = max(lengths) if padding else lengths[0]
        input_ids = torch.zeros(len(texts), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TinyVisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.patch_proj(x)


class TinyProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_projector = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.mm_projector(x)


class TinyLLMBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.k_proj = torch.nn.Linear(4, 4)
        self.gate_proj = torch.nn.Linear(4, 4)
        self.up_proj = torch.nn.Linear(4, 4)
        self.down_proj = torch.nn.Linear(4, 4)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        gated = torch.sigmoid(self.gate_proj(x)) * x
        hidden = torch.relu(self.up_proj(x))
        return q + k + gated + 0.01 * self.down_proj(hidden)


class TinyMedicalMLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = TinyVisionEncoder()
        self.mm_projector = TinyProjector()
        self.layers = torch.nn.ModuleList([TinyLLMBlock()])
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.eye_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def _features_from_texts(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "target" in lowered or "edit" in lowered:
                rows.append(torch.tensor([1.0, 0.0, 0.0, 0.0]))
            elif "reference" in lowered or "locality" in lowered:
                rows.append(torch.tensor([0.0, 1.0, 0.0, 0.0]))
            else:
                rows.append(torch.tensor([0.0, 0.0, 1.0, 0.0]))
        return torch.stack(rows, dim=0)

    def forward(self, batch=None, input_ids=None, images=None, labels=None, attention_mask=None):
        del input_ids, images, labels, attention_mask
        if isinstance(batch, dict):
            texts = batch.get("text_input") or batch.get("prompt") or [""]
        else:
            texts = [""]
        x = self._features_from_texts(texts).unsqueeze(1).repeat(1, 2, 1)
        vision = self.vision_encoder(x)
        x = x + self.mm_projector(vision)
        for layer in self.layers:
            x = layer(x)
        return x.sum(dim=(1, 2))


def _batch(text):
    return {"text_input": [text]}


def test_engram_integration_tiny_mllm_selects_edits_and_rolls_back():
    model = TinyMedicalMLLM()
    tok = DummyTokenizer()
    hparams = EngramMultimodalHparams(
        model_name="blip2",
        module_patterns=[
            r"layers\.\d+\.(q_proj|k_proj|gate_proj)$",
            r"mm_projector\.mm_projector$",
        ],
        exclude_module_patterns=[r"down_proj$", r"up_proj$"],
        target_variants=["edit"],
        reference_variants=["locality_text"],
        token_scope="all",
        alpha=0.5,
        absorb_bias=True,
        covariance_device="cpu",
        solve_device="cpu",
        norm_ratio_warn_threshold=10.0,
    )
    expected = {
        "mm_projector.mm_projector",
        "layers.0.q_proj",
        "layers.0.k_proj",
        "layers.0.gate_proj",
    }
    selected = {layer.name for layer in select_linear_layers(model, hparams)}
    assert selected == expected

    request = {
        "prompt": "target edit prompt",
        "target": "target answer",
        "locality_prompt": "reference locality prompt",
        "locality_ground_truth": "reference answer",
        "image": None,
    }
    target_batch = _batch("target edit prompt target answer")
    reference_batch = _batch("reference locality prompt reference answer")
    before_target = model(target_batch).detach().clone()
    before_reference = model(reference_batch).detach().clone()
    original = {name: param.detach().clone() for name, param in model.named_parameters() if name.rsplit(".", 1)[0] in expected}

    executor = EngramMultimodalRewriteExecutor()
    _, weights_copy = executor.apply_to_model(model, tok, [request], hparams, return_orig_weights=True)

    assert set(executor.last_updates) == expected
    for module_name, update in executor.last_updates.items():
        module = dict(model.named_modules())[module_name]
        assert executor.last_target_stats[module_name].count > 0
        assert executor.last_reference_stats[module_name].count > 0
        assert update.weight.shape == module.weight.shape
        if module.bias is not None:
            assert update.bias.shape == module.bias.shape

    after_target = model(target_batch).detach()
    after_reference = model(reference_batch).detach()
    target_change = (after_target - before_target).abs().mean()
    reference_change = (after_reference - before_reference).abs().mean()
    assert target_change > reference_change

    with torch.no_grad():
        for name, value in weights_copy.items():
            module_name, param_name = name.rsplit(".", 1)
            module = dict(model.named_modules())[module_name]
            getattr(module, param_name).copy_(value)

    for name, value in original.items():
        assert torch.allclose(dict(model.named_parameters())[name], value, atol=1.0e-6)


def test_engram_priority_selection_keeps_projector_and_gate_before_qk_truncation():
    model = TinyMedicalMLLM()
    hparams = EngramMultimodalHparams(
        module_patterns=[
            r"layers\.\d+\.(q_proj|k_proj|gate_proj)$",
            r"mm_projector\.mm_projector$",
        ],
        prioritize_module_selection=True,
        module_priority_patterns=[
            r"mm_projector\.mm_projector$",
            r"gate_proj$",
            r"q_proj$",
            r"k_proj$",
        ],
        engram_max_modules=2,
    )
    selected = [layer.name for layer in select_linear_layers(model, hparams)]
    assert selected == ["mm_projector.mm_projector", "layers.0.gate_proj"]


def test_engram_priority_selection_balances_priority_groups_under_max_modules():
    model = TinyMedicalMLLM()
    model.mm_projector.extra_projector = torch.nn.Linear(4, 4)
    hparams = EngramMultimodalHparams(
        module_patterns=[
            r"layers\.\d+\.(q_proj|k_proj|gate_proj)$",
            r"mm_projector\.(mm_projector|extra_projector)$",
        ],
        prioritize_module_selection=True,
        module_priority_patterns=[
            r"mm_projector\.",
            r"gate_proj$",
            r"q_proj$",
            r"k_proj$",
        ],
        engram_max_modules=4,
    )
    selected = [layer.name for layer in select_linear_layers(model, hparams)]
    assert len(selected) == 4
    assert sum(name.startswith("mm_projector.") for name in selected) == 1
    assert "layers.0.gate_proj" in selected
    assert "layers.0.q_proj" in selected
    assert "layers.0.k_proj" in selected


def test_engram_norm_skip_uses_effective_alpha_scaled_ratio():
    model = TinyMedicalMLLM()
    tok = DummyTokenizer()
    hparams = EngramMultimodalHparams(
        model_name="blip2",
        module_patterns=[r"layers\.0\.q_proj$"],
        target_variants=["edit"],
        reference_variants=["locality_text"],
        token_scope="all",
        alpha=0.0,
        covariance_device="cpu",
        solve_device="cpu",
        skip_if_norm_ratio_larger_than=0.0,
    )
    request = {
        "prompt": "target edit prompt",
        "target": "target answer",
        "locality_prompt": "reference locality prompt",
        "locality_ground_truth": "reference answer",
        "image": None,
    }
    executor = EngramMultimodalRewriteExecutor()
    executor.apply_to_model(model, tok, [request], hparams, return_orig_weights=True)
    update = executor.last_updates["layers.0.q_proj"]
    assert update.stats["norm_ratio"] > 0.0
    assert update.stats["effective_norm_ratio"] == 0.0
