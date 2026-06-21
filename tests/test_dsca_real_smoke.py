import copy
import os
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DSCA_REAL_SMOKE") != "1",
    reason="Set RUN_DSCA_REAL_SMOKE=1 to load real VLM weights.",
)


def _require_blip2_assets():
    root = Path("hugging_cache")
    required = [
        root / "opt-2.7b" / "pytorch_model.bin",
        root / "bert-base-uncased" / "pytorch_model.bin",
        root / "blip2_pretrained_opt2.7b.pth",
        root / "eva_vit_g.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip("missing BLIP2 assets: " + ", ".join(missing))


def _load_blip2_dsca():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real BLIP2 DSCA smoke test.")
    _require_blip2_assets()
    from easyeditor.trainer.algs.dsca import DSCA
    from easyeditor.trainer.models import get_model
    from easyeditor.trainer.training_hparams.dsca_multimodal_training_hparams import (
        DSCAMultimodalTrainingHparams,
    )

    hparams_path = Path("hparams/TRAINING/DSCA/blip2_stage1_smoke.yaml")
    config = DSCAMultimodalTrainingHparams.from_hparams(str(hparams_path))
    config.device = "cuda"
    config.dsca_tau_visual = 0.0
    torch.manual_seed(1234)
    model = get_model(config).to(config.device).eval()
    alg = DSCA(model, config, lambda: None).to(config.device).eval()
    return alg


def _make_batch(model, *, prompt="Question: What color is the square? Answer:", target=" red", seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    tokenizer = model.opt_tokenizer
    text = prompt + target
    prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    labels = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    image = torch.randn(1, 3, 364, 364, generator=generator).to(device)
    return {
        "image": image,
        "text_input": [text],
        "labels": labels,
        "prompts_len": [prompt_len],
    }


def _clone_batch(batch):
    cloned = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _logits(outputs):
    return outputs if isinstance(outputs, torch.Tensor) else outputs.logits


def _assert_exact_noop(alg, batch):
    with torch.no_grad():
        base_logits = _logits(alg.model(_clone_batch(batch))).detach()
        edited_logits = _logits(alg(_clone_batch(batch))).detach()
    assert torch.allclose(base_logits, edited_logits, atol=0.0, rtol=0.0)


def _assert_masks(output):
    seq_len = output.logits.shape[1]
    for name in ("vision_mask", "prompt_mask", "answer_mask", "attention_mask"):
        mask = getattr(output, name)
        assert mask.shape == (1, seq_len)
    assert output.vision_mask.sum() > 0
    assert output.prompt_mask.sum() > 0
    assert output.answer_mask.sum() > 0
    assert not (output.vision_mask & output.prompt_mask).any()
    assert not (output.vision_mask & output.answer_mask).any()
    assert not (output.prompt_mask & output.answer_mask).any()
    assert not ((output.vision_mask | output.prompt_mask | output.answer_mask) & ~output.attention_mask.bool()).any()


def test_blip2_dsca_real_smoke():
    alg = _load_blip2_dsca()
    batch = _make_batch(alg.model)

    assert all(not param.requires_grad for param in alg.model.parameters())

    alg.repository.clear()
    _assert_exact_noop(alg, batch)

    with torch.no_grad():
        output = alg.model(_clone_batch(batch))
    _assert_masks(output)

    alg.repository.clear()
    reps = alg.capture_representations(_clone_batch(batch))
    inactive_id = alg.repository.create_cluster(reps["h_f"][0], reps["h_v"][0])
    assert inactive_id == 0
    assert alg.repository.num_active() == 0
    _assert_exact_noop(alg, batch)

    alg.repository.clear()
    active_id = alg.repository.create_cluster(reps["h_f"][0], -reps["h_v"][0])
    alg.repository.dsams[active_id].set_basis(torch.eye(alg.hidden_size, device=reps["h_f"].device)[: alg.rank])
    alg.repository.active[active_id] = True
    old_tau = alg.tau_visual
    alg.tau_visual = 0.99
    _assert_exact_noop(alg, batch)
    alg.tau_visual = old_tau

    alg.repository.clear()
    active_id = alg.repository.create_cluster(reps["h_f"][0], reps["h_v"][0])
    alg.repository.dsams[active_id].set_basis(torch.eye(alg.hidden_size, device=reps["h_f"].device)[: alg.rank])
    with torch.no_grad():
        alg.repository.dsams[active_id].b.fill_(0.25)
    alg.repository.active[active_id] = True
    alg.tau_visual = -1.0
    with torch.no_grad():
        base_logits = _logits(alg.model(_clone_batch(batch))).detach()
        edited_logits = _logits(alg(_clone_batch(batch))).detach()
    assert not torch.allclose(base_logits, edited_logits, atol=0.0, rtol=0.0)

    alg.remove_hook()
    with torch.no_grad():
        removed_hook_logits = _logits(alg(_clone_batch(batch))).detach()
    assert torch.allclose(base_logits, removed_hook_logits, atol=0.0, rtol=0.0)


def test_minigpt4_dependency_probe():
    try:
        import sentencepiece  # noqa: F401
    except Exception as exc:
        pytest.skip(f"sentencepiece unavailable: {exc!r}")
    root = Path("hugging_cache/vicuna-7b")
    if not (root / "tokenizer.model").exists():
        pytest.skip("vicuna tokenizer.model is missing")
    from transformers import LlamaTokenizer

    tok = LlamaTokenizer.from_pretrained(str(root), use_fast=False, local_files_only=True)
    assert tok.eos_token is not None
