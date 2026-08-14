#!/usr/bin/env python3
"""Stage A: execute official-backbone end-to-end LiveEdit trace parity.

The immutable upstream modules are loaded directly from the pinned source
snapshot.  A small compatibility adapter restores the source-era LLaVA
processor/merge boundary removed by newer Transformers releases.  No trained
checkpoint, medical record, or canonical repository is loaded.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.source_ops import (  # noqa: E402
    SIM_SCALE,
    apply_low_rank_expert_residual,
    compute_text_soft_weights,
    generate_expert_and_keys,
)
from methods.liveedit_med.trace_parity import (  # noqa: E402
    REQUIRED_TRACE_BOUNDARIES,
    compare_discrete,
    compare_tensor,
    compare_tensor_mapping,
    legacy_llava_merge_input_ids_with_image_features,
    seed_everything,
    source_era_llava_processor_outputs,
    state_dict_sha256,
    summarize_trace,
    tensor_sha256,
)
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules  # noqa: E402
from methods.liveedit_med.source_training_continuation import (  # noqa: E402
    SourceTrainingContinuationMode,
    coerce_source_training_mode,
    forward_source_training_logits,
)


PINNED_COMMIT = "3615a37b05294509f411df045621940f276a5e6b"
PASS = "END_TO_END_UPSTREAM_PORT_PARITY_PASS"
FAIL = "END_TO_END_UPSTREAM_PORT_PARITY_FAIL"
MISSING = "END_TO_END_UPSTREAM_PORT_PARITY_NOT_RUN_ASSETS_MISSING"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SOURCE_SAMPLE_FAMILIES = ("E-VQA", "E-IC", "VLKEB")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"STAGE_A_CANNOT_IMPORT_UPSTREAM:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialModules(nn.Module):
    """Container made only from classes imported from the immutable source."""

    def __init__(self, upstream: ModuleType, config: LiveEditMedicalConfig, vision_tokens: int):
        super().__init__()
        c = config
        self.edit_extractor = upstream.QVExtractor(
            c.eqe_n, c.llm_mid_dim, c.module_dim, c.cross_att_head_n, vision_tokens, False
        )
        self.input_extractor = upstream.QVExtractor(
            c.eqe_n, c.llm_mid_dim, c.module_dim, c.cross_att_head_n, vision_tokens, True
        )
        self.moegen_c = upstream.LowRankGenerator(
            c.llm_mid_dim, c.lora_rank, c.lora_scale, c.llm_mid_dim, c.module_dim, c.cross_att_head_n
        )
        self.moegen_r = upstream.LowRankGenerator(
            c.llm_mid_dim, c.lora_rank, c.lora_scale, c.llm_mid_dim, c.module_dim, c.cross_att_head_n
        )
        self.instant_reps_norm = nn.LayerNorm(c.llm_mid_dim)


def module_state(module: OfficialModules | LiveEditMedicalModules) -> dict[str, torch.Tensor]:
    return {name: value for name, value in module.state_dict().items()}


def gradient_state(module: OfficialModules | LiveEditMedicalModules) -> dict[str, torch.Tensor]:
    state = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"STAGE_A_MISSING_GRADIENT:{name}")
        state[name] = parameter.grad.detach().cpu().clone()
    return state


def state_difference(upstream: Mapping[str, torch.Tensor], port: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    rows = []
    for name in sorted(upstream):
        left, right = upstream[name].float(), port[name].float()
        delta = (left - right).abs()
        rows.append({
            "name": name,
            "exact": torch.equal(upstream[name], port[name]),
            "max_abs_error": float(delta.max()) if delta.numel() else 0.0,
            "mean_abs_error": float(delta.mean()) if delta.numel() else 0.0,
            "sign_mismatch_count": int(((left.sign() != right.sign()) & ((left != 0) | (right != 0))).sum()),
        })
    return {
        "exact": all(row["exact"] for row in rows),
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "mismatched_tensors": sum(not row["exact"] for row in rows),
        "sign_mismatch_count": sum(row["sign_mismatch_count"] for row in rows),
        "largest": sorted(rows, key=lambda row: row["max_abs_error"], reverse=True)[:20],
    }


def copy_official_state_to_port(official: OfficialModules, port: LiveEditMedicalModules) -> None:
    state = official.state_dict()
    translated = {
        ("input_extractor." + key[len("inpt_extractor."):]) if key.startswith("inpt_extractor.") else key: value
        for key, value in state.items()
    }
    # OfficialModules intentionally uses the port-facing name, but retaining
    # the translation makes the adapter robust to an upstream-style container.
    port.load_state_dict(translated, strict=True)


@dataclass
class Prepared:
    processor: dict[str, torch.Tensor]
    merged: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor
    visual_indices: tuple[int, ...]
    question_indices: tuple[int, ...]
    answer_indices: tuple[int, ...]
    clean_hidden: torch.Tensor
    clean_logits: torch.Tensor

    @property
    def visual(self) -> torch.Tensor:
        return self.clean_hidden[:, self.visual_indices]

    @property
    def question(self) -> torch.Tensor:
        return self.clean_hidden[:, self.question_indices]

    @property
    def answer(self) -> torch.Tensor:
        return self.clean_hidden[:, self.answer_indices]


def source_labels(tokenizer: Any, prompt: str, target: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, str]:
    target = " " + target if prompt[-1] not in (" ", "\n") and target[0] not in (" ", "\n") else target
    full = prompt + target
    label = tokenizer(full, return_tensors="pt", padding=True)["input_ids"][0]
    label = torch.roll(label, -1, 0)
    mask = torch.zeros_like(label)
    prompt_ids = tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"][0]
    mask[len(prompt_ids) - 1 : -1] = 1
    start = len(prompt_ids) - 1
    return label[start:].unsqueeze(0).to(device), mask[start:].unsqueeze(0).to(device), full


def source_merge(
    model: Any, processor_outputs: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = processor_outputs["input_ids"].to(device)
    mask = processor_outputs["attention_mask"].to(device)
    pixels = processor_outputs["pixel_values"].to(device=device, dtype=model.dtype)
    embeds = model.get_input_embeddings()(ids)
    vision = model.vision_tower(pixels, output_hidden_states=True)
    selected = vision.hidden_states[model.config.vision_feature_layer]
    strategy = model.config.vision_feature_select_strategy
    if strategy == "default":
        selected = selected[:, 1:]
    elif strategy != "full":
        raise RuntimeError(f"STAGE_A_UNSUPPORTED_VISION_FEATURE_STRATEGY:{strategy}")
    image_features = model.multi_modal_projector(selected)
    merged, merged_mask, _labels, position_ids = legacy_llava_merge_input_ids_with_image_features(
        model, image_features, embeds, ids, mask, None
    )
    return merged, merged_mask, position_ids


def decoder_context(language_model: Any, hidden: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor):
    core = language_model.model
    length = hidden.shape[1]
    cache_position = torch.arange(length, device=hidden.device)
    causal = core._update_causal_mask(attention_mask, hidden, cache_position, None, False)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    return cache_position, causal, position_embeddings


def run_layers(
    language_model: Any,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    start: int,
    residual_after: int | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    core = language_model.model
    cache_position, causal, position_embeddings = decoder_context(language_model, hidden, attention_mask, position_ids)
    post_layer = None
    for index in range(start, core.config.num_hidden_layers):
        hidden = core.layers[index](
            hidden,
            attention_mask=causal,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
        if residual_after == index:
            if residual is None:
                raise RuntimeError("STAGE_A_MISSING_RESIDUAL")
            hidden = hidden + residual.to(hidden.dtype)
            post_layer = hidden
    hidden = core.norm(hidden)
    return language_model.lm_head(hidden), post_layer


def clean_forward(
    model: Any, merged: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _args, output):
        captured["hidden"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = model.language_model.model.layers[21].register_forward_hook(hook)
    try:
        with torch.no_grad():
            output = model.language_model(
                inputs_embeds=merged,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    return captured["hidden"].detach(), output.logits.detach()


def prepare(
    model: Any,
    processor: Any,
    prompt: str,
    target: str,
    image: Image.Image,
    device: torch.device,
) -> Prepared:
    labels, label_mask, full = source_labels(processor.tokenizer, prompt, target, device)
    text = "<image>\n" + full
    proc = source_era_llava_processor_outputs(processor, text, image)
    image_token = int(model.config.image_token_index)
    positions = torch.where(proc["input_ids"][0] == image_token)[0]
    if positions.numel() != 1:
        raise RuntimeError(f"STAGE_A_SOURCE_ERA_IMAGE_PLACEHOLDER_COUNT:{positions.numel()}")
    merged, attention_mask, position_ids = source_merge(model, proc, device)
    image_start = int(positions[0])
    image_count = (model.config.vision_config.image_size // model.config.vision_config.patch_size) ** 2
    visual = tuple(range(image_start, image_start + image_count))
    answer_count = labels.shape[1] - 1
    answer_start = merged.shape[1] - answer_count
    question = tuple(range(visual[-1] + 1, answer_start))
    answer = tuple(range(answer_start, merged.shape[1]))
    if not question or not answer:
        raise RuntimeError("STAGE_A_EMPTY_SOURCE_SPAN")
    hidden, logits = clean_forward(model, merged, attention_mask, position_ids)
    return Prepared(proc, merged.detach(), attention_mask.detach(), position_ids.detach(), labels, label_mask,
                    visual, question, answer, hidden, logits)


def prepare_prompt_only(
    model: Any, processor: Any, prompt: str, image: Image.Image, device: torch.device
) -> Prepared:
    text = "<image>\n" + prompt
    proc = source_era_llava_processor_outputs(processor, text, image)
    positions = torch.where(proc["input_ids"][0] == int(model.config.image_token_index))[0]
    if positions.numel() != 1:
        raise RuntimeError("STAGE_A_PROMPT_IMAGE_PLACEHOLDER_COUNT")
    merged, attention_mask, position_ids = source_merge(model, proc, device)
    image_start = int(positions[0])
    image_count = (model.config.vision_config.image_size // model.config.vision_config.patch_size) ** 2
    visual = tuple(range(image_start, image_start + image_count))
    question = tuple(range(visual[-1] + 1, merged.shape[1]))
    hidden, logits = clean_forward(model, merged, attention_mask, position_ids)
    empty = torch.empty((1, 0), dtype=torch.long, device=device)
    return Prepared(proc, merged.detach(), attention_mask.detach(), position_ids.detach(), empty, empty,
                    visual, question, (), hidden, logits)


def official_generated(modules: OfficialModules, prepared: Prepared):
    vision, question, answer = prepared.visual.float(), prepared.question.float(), prepared.answer.float()
    evr = modules.edit_extractor.extract_vision(question, vision)
    eqr = modules.edit_extractor.extract_query(question)
    edit = torch.cat([vision, question, answer], 1)
    return eqr, evr, modules.moegen_c(edit), modules.moegen_r(edit)


def official_route(
    modules: OfficialModules, prepared: Prepared, evr: torch.Tensor, eqr: torch.Tensor
) -> dict[str, torch.Tensor]:
    vision, question = prepared.visual.float(), prepared.question.float()
    ivr = modules.input_extractor.extract_vision(question, vision)
    visual_score = torch.einsum("bed,med->bme", ivr, evr).mean(2) * SIM_SCALE
    sentinel = modules.input_extractor.extract_from_visprot(question)
    sentinel_score = torch.einsum("bed,bed->be", ivr, sentinel).mean(1, True) * SIM_SCALE
    candidate = (visual_score > sentinel_score)[0]
    iqr = modules.input_extractor.extract_query(question)
    text_score = torch.einsum("ned,med->nme", iqr, eqr[candidate]).mean(2) * SIM_SCALE
    relative = torch.softmax(text_score, 1) if text_score.shape[1] else text_score
    absolute = torch.sigmoid(text_score)
    return {
        "input_visual": ivr,
        "input_text": iqr,
        "sentinel_key": sentinel,
        "visual_score": visual_score,
        "sentinel_score": sentinel_score,
        "candidate": candidate,
        "text_score": text_score,
        "relative": relative,
        "absolute": absolute,
        "final": relative * absolute,
    }


def port_route(
    modules: LiveEditMedicalModules, prepared: Prepared, evr: torch.Tensor, eqr: torch.Tensor
) -> dict[str, torch.Tensor]:
    vision, question = prepared.visual.float(), prepared.question.float()
    ivr = modules.input_extractor.extract_vision(question, vision)
    visual_score = torch.einsum("bed,med->bme", ivr, evr).mean(2) * SIM_SCALE
    sentinel = modules.input_extractor.extract_from_visprot(question)
    sentinel_score = torch.einsum("bed,bed->be", ivr, sentinel).mean(1, True) * SIM_SCALE
    candidate = (visual_score > sentinel_score)[0]
    iqr = modules.input_extractor.extract_query(question)
    text_score = torch.einsum("ned,med->nme", iqr, eqr[candidate]).mean(2) * SIM_SCALE
    relative = torch.softmax(text_score, 1) if text_score.shape[1] else text_score
    absolute = torch.sigmoid(text_score)
    return {"input_visual": ivr, "input_text": iqr, "sentinel_key": sentinel,
            "visual_score": visual_score, "sentinel_score": sentinel_score, "candidate": candidate,
            "text_score": text_score, "relative": relative, "absolute": absolute, "final": relative * absolute}


def residual_or_zero(
    hidden: torch.Tensor,
    moe_c: torch.Tensor,
    moe_r: torch.Tensor,
    weights: torch.Tensor,
    norm: nn.Module,
) -> torch.Tensor:
    if weights.shape[1] == 0:
        return torch.zeros_like(hidden)
    return apply_low_rank_expert_residual(hidden.float(), moe_c, moe_r, weights, norm)


def label_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = logits[:, -labels.shape[1] :].float()
    gathered = torch.log_softmax(selected, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return -(gathered * mask).sum() / mask.sum()


def label_prediction(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return logits[:, -labels.shape[1] :].argmax(-1)


def kl_loss(candidate: torch.Tensor, base: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    candidate = candidate[:, -mask.shape[1] :].float()
    base = base[:, -mask.shape[1] :].float()
    value = (base.softmax(-1) * (base.log_softmax(-1) - candidate.log_softmax(-1))).sum(-1)
    return (value * mask).sum() / mask.sum()


def training_logits(
    model: Any,
    prepared: Prepared,
    residual: torch.Tensor,
    *,
    official_source_semantics: bool,
    port_continuation_mode: SourceTrainingContinuationMode,
) -> torch.Tensor:
    if official_source_semantics:
        # Pinned BaseVLLMForEdit.forward_from_mid_layer injects the captured
        # layer-21 *output* as layer-21 input, then the official hook adds the
        # precomputed residual to that block's new output.
        logits, _ = run_layers(
            model.language_model, prepared.clean_hidden, prepared.attention_mask, prepared.position_ids,
            start=21, residual_after=21, residual=residual,
        )
        return logits
    return forward_source_training_logits(
        model.language_model,
        prepared.clean_hidden,
        prepared.attention_mask,
        residual=residual,
        mode=port_continuation_mode,
        position_ids=prepared.position_ids,
    )


def soft_losses(input_key: torch.Tensor, edit_key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    relative, absolute = compute_text_soft_weights(input_key, edit_key, split=True)
    eps = 1e-8
    relative_loss = -torch.log(torch.diag(relative) + eps).mean()
    pos = torch.diag(absolute)
    neg = torch.diag(absolute.roll(1, 1))
    absolute_loss = -(torch.log(pos + eps) + torch.log(1 - neg + eps)).mean()
    return relative_loss, absolute_loss


def hard_losses(
    input_extractor: nn.Module,
    edit_extractor: nn.Module,
    positive: Prepared,
    negative: Prepared,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8

    def distribution(inp: Prepared, edit: Prepared) -> torch.Tensor:
        ivr = input_extractor.extract_vision(inp.question.float(), inp.visual.float())
        evr = edit_extractor.extract_vision(edit.question.float(), edit.visual.float())
        score = torch.einsum("bed,med->bme", ivr, evr).mean(2) * SIM_SCALE
        sentinel = input_extractor.extract_from_visprot(inp.question.float())
        sentinel_score = torch.einsum("bed,bed->be", ivr, sentinel).mean(1, True) * SIM_SCALE
        return torch.softmax(torch.cat([score, sentinel_score], 1), 1)

    neighbor = distribution(positive, positive)
    prototype = distribution(negative, positive)
    return -torch.log(neighbor[0, 0] + eps), -torch.log(prototype[0, -1] + eps)


def source_objective(
    model: Any,
    modules: OfficialModules | LiveEditMedicalModules,
    views: Mapping[str, Prepared],
    *,
    official_source_semantics: bool,
    port_continuation_mode: SourceTrainingContinuationMode,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(modules, OfficialModules):
        eqr, _evr, moe_c, moe_r = official_generated(modules, views["native"])
        input_extractor = modules.input_extractor
    else:
        eqr, _evr, moe_c, moe_r = generate_expert_and_keys(
            modules.edit_extractor, modules.moegen_c, modules.moegen_r,
            views["native"].visual.float(), views["native"].question.float(), views["native"].answer.float(),
        )
        input_extractor = modules.input_extractor
    components: dict[str, torch.Tensor] = {}
    for name in ("native", "text_generality", "image_generality"):
        item = views[name]
        input_key = input_extractor.extract_query(item.question.float())
        weights = compute_text_soft_weights(input_key, eqr)
        residual = apply_low_rank_expert_residual(item.clean_hidden.float(), moe_c, moe_r, weights, modules.instant_reps_norm)
        logits = training_logits(
            model, item, residual, official_source_semantics=official_source_semantics,
            port_continuation_mode=port_continuation_mode,
        )
        key = "reliability" if name == "native" else name
        components[key] = label_loss(logits, item.labels, item.label_mask)
    locality = views["image_locality"]
    locality_key = input_extractor.extract_query(locality.question.float())
    locality_weights = compute_text_soft_weights(locality_key, eqr)
    locality_residual = apply_low_rank_expert_residual(
        locality.clean_hidden.float(), moe_c, moe_r, locality_weights, modules.instant_reps_norm
    )
    locality_logits = training_logits(
        model, locality, locality_residual, official_source_semantics=official_source_semantics,
        port_continuation_mode=port_continuation_mode,
    )
    components["image_locality"] = kl_loss(locality_logits, locality.clean_logits, locality.label_mask)
    native_input_key = input_extractor.extract_query(views["native"].question.float())
    components["soft_relative"], components["soft_absolute"] = soft_losses(native_input_key, eqr)
    components["hard_neighbor"], components["hard_prototype"] = hard_losses(
        input_extractor, modules.edit_extractor, views["native"], views["image_locality"]
    )
    total = sum(components.values())
    components["total"] = total
    return total, components


def backward_source_objective(
    model: Any,
    modules: OfficialModules | LiveEditMedicalModules,
    views: Mapping[str, Prepared],
    *,
    official_source_semantics: bool,
    port_continuation_mode: SourceTrainingContinuationMode,
) -> None:
    """Accumulate the complete source objective one component at a time.

    Recomputing the small editor graph for every term is mathematically the
    same summed gradient while preventing four frozen-backbone suffix graphs
    from residing on a 40 GiB device simultaneously.
    """

    def generated():
        if isinstance(modules, OfficialModules):
            return official_generated(modules, views["native"])
        return generate_expert_and_keys(
            modules.edit_extractor, modules.moegen_c, modules.moegen_r,
            views["native"].visual.float(), views["native"].question.float(), views["native"].answer.float(),
        )

    for name in ("native", "text_generality", "image_generality"):
        eqr, _evr, moe_c, moe_r = generated()
        item = views[name]
        input_key = modules.input_extractor.extract_query(item.question.float())
        weights = compute_text_soft_weights(input_key, eqr)
        residual = apply_low_rank_expert_residual(
            item.clean_hidden.float(), moe_c, moe_r, weights, modules.instant_reps_norm
        )
        logits = training_logits(
            model, item, residual, official_source_semantics=official_source_semantics,
            port_continuation_mode=port_continuation_mode,
        )
        label_loss(logits, item.labels, item.label_mask).backward()
        del logits, residual, weights, input_key, eqr, moe_c, moe_r
        torch.cuda.empty_cache()

    eqr, _evr, moe_c, moe_r = generated()
    locality = views["image_locality"]
    locality_key = modules.input_extractor.extract_query(locality.question.float())
    locality_weights = compute_text_soft_weights(locality_key, eqr)
    locality_residual = apply_low_rank_expert_residual(
        locality.clean_hidden.float(), moe_c, moe_r, locality_weights, modules.instant_reps_norm
    )
    locality_logits = training_logits(
        model, locality, locality_residual, official_source_semantics=official_source_semantics,
        port_continuation_mode=port_continuation_mode,
    )
    kl_loss(locality_logits, locality.clean_logits, locality.label_mask).backward()
    del locality_logits, locality_residual, locality_weights, locality_key, eqr, moe_c, moe_r
    torch.cuda.empty_cache()

    eqr, _evr, _moe_c, _moe_r = generated()
    native_key = modules.input_extractor.extract_query(views["native"].question.float())
    relative_loss, absolute_loss = soft_losses(native_key, eqr)
    (relative_loss + absolute_loss).backward()
    del relative_loss, absolute_loss, native_key, eqr, _moe_c, _moe_r

    neighbor, prototype = hard_losses(
        modules.input_extractor, modules.edit_extractor, views["native"], views["image_locality"]
    )
    (neighbor + prototype).backward()


def tensor_row(name: str, upstream: torch.Tensor, port: torch.Tensor, *, discrete: bool = False):
    if discrete:
        return compare_discrete(name, upstream, port)
    compute_dtype = upstream.dtype in (torch.float16, torch.bfloat16) or port.dtype in (torch.float16, torch.bfloat16)
    return compare_tensor(name, upstream, port, atol=5e-4 if compute_dtype else 1e-6,
                          rtol=1e-5 if not compute_dtype else 0.0)


def asset_audit(args: argparse.Namespace) -> dict[str, Any]:
    missing = []
    for label, path, kind in (
        ("official_llava_v1_5_7b_hf_model", args.official_model, "dir"),
        ("one_source_valid_E-VQA_or_E-IC_or_VLKEB_sample", args.official_sample, "file"),
        ("source_sample_image_root", args.image_root, "dir"),
        ("pinned_upstream_snapshot", args.upstream_root, "dir"),
    ):
        if path is None or (kind == "dir" and not path.is_dir()) or (kind == "file" and not path.is_file()):
            missing.append(label)
    if missing:
        return {"passed": False, "missing_assets": missing}
    upstream_manifest = json.loads((args.upstream_root / "UPSTREAM_MANIFEST.json").read_text())
    source_files = []
    for row in upstream_manifest["files"]:
        path = args.upstream_root / row["path"]
        observed = git_blob_sha1(path) if path.is_file() else None
        source_files.append({"path": row["path"], "expected_blob_sha": row["expected_blob_sha"],
                             "observed_blob_sha": observed, "passed": observed == row["expected_blob_sha"]})
    sample_manifest_path = args.official_sample.parent / "manifest.json"
    sample_manifest = json.loads(sample_manifest_path.read_text())
    images = []
    for row in sample_manifest["images"]:
        path = Path(row["path"])
        observed = file_sha256(path) if path.is_file() else None
        images.append({"path": str(path), "expected_sha256": row["sha256"], "observed_sha256": observed,
                       "passed": observed == row["sha256"]})
    download_manifest = json.loads((args.official_model / "download_manifest.json").read_text())
    model_files = []
    for row in download_manifest["files"]:
        path = args.official_model / row["path"]
        # Hash every file, including the three model shards, before loading.
        observed = file_sha256(path) if path.is_file() else None
        model_files.append({"path": row["path"], "size": path.stat().st_size if path.is_file() else None,
                            "expected_sha256": row["sha256"], "observed_sha256": observed,
                            "passed": observed == row["sha256"]})
    passed = (
        upstream_manifest.get("commit") == PINNED_COMMIT
        and all(row["passed"] for row in source_files)
        and sample_manifest.get("dataset_family") in SOURCE_SAMPLE_FAMILIES
        and file_sha256(args.official_sample) == sample_manifest["sample_sha256"]
        and all(row["passed"] for row in images)
        and download_manifest.get("repo_id") == MODEL_ID
        and all(row["passed"] for row in model_files)
    )
    return {"passed": passed, "pinned_commit": upstream_manifest.get("commit"),
            "upstream_files": source_files, "sample_manifest": sample_manifest,
            "model_revision": download_manifest.get("revision"), "model_files": model_files}


def missing_outcome(args: argparse.Namespace, audit: Mapping[str, Any]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=False)
    result = {"stage": "A", "status": MISSING, "asset_audit": audit, "edited_checkpoint_loaded": False,
              "timestamp_utc": utc_now()}
    write_json(args.out_dir / "asset_audit.json", audit)
    write_json(args.out_dir / "trace_parity_summary.json", result)
    (args.out_dir / "END_TO_END_PARITY_REPORT.md").write_text(
        f"# Stage A end-to-end parity\n\nStatus: `{MISSING}`\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="JSON file supplying all path/device arguments")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--official-model", type=Path)
    parser.add_argument("--official-sample", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--upstream-root", type=Path, default=ROOT / "third_party/liveedit_official_3615a37")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--port-continuation-mode",
        choices=[item.value for item in SourceTrainingContinuationMode],
        default=SourceTrainingContinuationMode.CORRECTED_SEMANTICS_CONTINUE_LAYER22.value,
    )
    args = parser.parse_args()
    if args.config:
        config = json.loads(args.config.read_text())
        for key, value in config.items():
            setattr(args, key, Path(value) if key.endswith(("_dir", "_root", "_model", "_sample")) else value)
    if args.image_root is None and args.official_sample is not None:
        args.image_root = args.official_sample.parent / "images"
    port_continuation_mode = coerce_source_training_mode(args.port_continuation_mode)
    audit = asset_audit(args)
    if not audit.get("passed"):
        missing_outcome(args, audit)
        return
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "asset_audit.json", audit)
    seed_everything(42)
    device = torch.device(args.device)
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    model = LlavaForConditionalGeneration.from_pretrained(
        args.official_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval().requires_grad_(False)
    processor = LlavaProcessor.from_pretrained(args.official_model, use_fast=False)
    if int(model.config.image_token_index) != int(processor.tokenizer.convert_tokens_to_ids("<image>")):
        raise RuntimeError("STAGE_A_IMAGE_TOKEN_ID_MISMATCH")
    row = json.loads(args.official_sample.read_text())[0]
    native_image = Image.open(args.image_root / row["image"]).convert("RGB")
    rephrase_image = Image.open(args.image_root / row["image_rephrase"]).convert("RGB")
    locality_image = Image.open(args.image_root / row["m_loc"]).convert("RGB")

    # Each execution path independently runs preprocessing, merge, and the clean
    # backbone trace.  This is intentionally not a shared cached tensor.
    upstream_views = {
        "native": prepare(model, processor, row["src"], row["alt"], native_image, device),
        "text_generality": prepare(model, processor, row["rephrase"], row["alt"], native_image, device),
        "image_generality": prepare(model, processor, row["src"], row["alt"], rephrase_image, device),
        "image_locality": prepare(model, processor, row["m_loc_q"] + " The answer is:", row["m_loc_a"], locality_image, device),
        "route": prepare_prompt_only(model, processor, row["src"], native_image, device),
    }
    port_views = {
        "native": prepare(model, processor, row["src"], row["alt"], native_image, device),
        "text_generality": prepare(model, processor, row["rephrase"], row["alt"], native_image, device),
        "image_generality": prepare(model, processor, row["src"], row["alt"], rephrase_image, device),
        "image_locality": prepare(model, processor, row["m_loc_q"] + " The answer is:", row["m_loc_a"], locality_image, device),
        "route": prepare_prompt_only(model, processor, row["src"], native_image, device),
    }

    upstream_module = load_module(
        args.upstream_root / "editor/vllm_editors/liveedit/modules.py", "liveedit_stage_a_upstream_modules"
    )
    config = LiveEditMedicalConfig(
        llm_mid_dim=model.config.text_config.hidden_size,
        source_training_continuation_mode=port_continuation_mode,
    )
    vision_tokens = len(upstream_views["native"].visual_indices)
    seed_everything(42)
    official = OfficialModules(upstream_module, config, vision_tokens).to(device)
    port = LiveEditMedicalModules(config, vision_tokens).to(device)
    copy_official_state_to_port(official, port)
    initial_official_hash = state_dict_sha256(module_state(official))
    initial_port_hash = state_dict_sha256(module_state(port))
    if initial_official_hash != initial_port_hash:
        raise RuntimeError("STAGE_A_INITIAL_MODULE_STATE_MISMATCH")

    up_eqr, up_evr, up_c, up_r = official_generated(official, upstream_views["native"])
    po_eqr, po_evr, po_c, po_r = generate_expert_and_keys(
        port.edit_extractor, port.moegen_c, port.moegen_r,
        port_views["native"].visual.float(), port_views["native"].question.float(), port_views["native"].answer.float(),
    )
    up_route = official_route(official, upstream_views["route"], up_evr, up_eqr)
    po_route = port_route(port, port_views["route"], po_evr, po_eqr)
    up_selected_c, up_selected_r = up_c[up_route["candidate"]], up_r[up_route["candidate"]]
    po_selected_c, po_selected_r = po_c[po_route["candidate"]], po_r[po_route["candidate"]]
    up_residual = residual_or_zero(
        upstream_views["route"].clean_hidden, up_selected_c, up_selected_r, up_route["final"], official.instant_reps_norm
    )
    po_residual = residual_or_zero(
        port_views["route"].clean_hidden, po_selected_c, po_selected_r, po_route["final"], port.instant_reps_norm
    )
    up_post = upstream_views["route"].clean_hidden + up_residual.to(upstream_views["route"].clean_hidden.dtype)
    po_post = port_views["route"].clean_hidden + po_residual.to(port_views["route"].clean_hidden.dtype)
    with torch.no_grad():
        up_logits, _ = run_layers(model.language_model, up_post, upstream_views["route"].attention_mask,
                                  upstream_views["route"].position_ids, start=22)
        po_logits, _ = run_layers(model.language_model, po_post, port_views["route"].attention_mask,
                                  port_views["route"].position_ids, start=22)

    # Source teacher-forced loss components deliberately use each path's actual
    # continuation semantics.  This is the boundary that can reveal a full-run
    # port mismatch hidden by isolated module tests.
    with torch.no_grad():
        _up_total_eval, up_components = source_objective(
            model, official, upstream_views, official_source_semantics=True,
            port_continuation_mode=port_continuation_mode,
        )
        _po_total_eval, po_components = source_objective(
            model, port, port_views, official_source_semantics=False,
            port_continuation_mode=port_continuation_mode,
        )
    up_component_values = {key: value.detach().reshape(1) for key, value in up_components.items()}
    po_component_values = {key: value.detach().reshape(1) for key, value in po_components.items()}
    with torch.no_grad():
        up_tf_logits = training_logits(
            model, upstream_views["native"],
            apply_low_rank_expert_residual(
                upstream_views["native"].clean_hidden.float(), up_c, up_r,
                compute_text_soft_weights(official.input_extractor.extract_query(upstream_views["native"].question.float()), up_eqr),
                official.instant_reps_norm,
            ),
            official_source_semantics=True,
            port_continuation_mode=port_continuation_mode,
        )
        po_tf_logits = training_logits(
            model, port_views["native"],
            apply_low_rank_expert_residual(
                port_views["native"].clean_hidden.float(), po_c, po_r,
                compute_text_soft_weights(port.input_extractor.extract_query(port_views["native"].question.float()), po_eqr),
                port.instant_reps_norm,
            ),
            official_source_semantics=False,
            port_continuation_mode=port_continuation_mode,
        )
        up_prediction = label_prediction(up_tf_logits, upstream_views["native"].labels)
        po_prediction = label_prediction(po_tf_logits, port_views["native"].labels)

    rows = [
        compare_tensor_mapping("processor_outputs", upstream_views["native"].processor,
                               port_views["native"].processor, atol=0, rtol=0),
        tensor_row("merged_multimodal_input_embeddings", upstream_views["native"].merged,
                   port_views["native"].merged),
        compare_tensor_mapping("attention_mask_and_position_ids",
                               {"attention_mask": upstream_views["native"].attention_mask,
                                "position_ids": upstream_views["native"].position_ids},
                               {"attention_mask": port_views["native"].attention_mask,
                                "position_ids": port_views["native"].position_ids}, atol=0, rtol=0),
        tensor_row("layer21_full_block_clean_hidden_state", upstream_views["native"].clean_hidden,
                   port_views["native"].clean_hidden),
        compare_tensor_mapping("visual_span_indices_and_tensor",
                               {"indices": torch.tensor(upstream_views["native"].visual_indices),
                                "tensor": upstream_views["native"].visual},
                               {"indices": torch.tensor(port_views["native"].visual_indices),
                                "tensor": port_views["native"].visual}, atol=5e-4, rtol=0),
        compare_tensor_mapping("question_span_indices_and_tensor",
                               {"indices": torch.tensor(upstream_views["native"].question_indices),
                                "tensor": upstream_views["native"].question},
                               {"indices": torch.tensor(port_views["native"].question_indices),
                                "tensor": port_views["native"].question}, atol=5e-4, rtol=0),
        compare_tensor_mapping("target_answer_span_indices_and_tensor",
                               {"indices": torch.tensor(upstream_views["native"].answer_indices),
                                "tensor": upstream_views["native"].answer},
                               {"indices": torch.tensor(port_views["native"].answer_indices),
                                "tensor": port_views["native"].answer}, atol=5e-4, rtol=0),
        tensor_row("edit_end_visual_routing_key", up_evr, po_evr),
        tensor_row("edit_end_text_routing_key", up_eqr, po_eqr),
        tensor_row("input_end_visual_routing_key", up_route["input_visual"], po_route["input_visual"]),
        tensor_row("input_end_text_routing_key", up_route["input_text"], po_route["input_text"]),
        tensor_row("question_conditioned_visual_sentinel_key", up_route["sentinel_key"], po_route["sentinel_key"]),
        tensor_row("generated_moe_c", up_c, po_c),
        tensor_row("generated_moe_r", up_r, po_r),
        tensor_row("visual_similarity_scores", up_route["visual_score"], po_route["visual_score"]),
        tensor_row("sentinel_score", up_route["sentinel_score"], po_route["sentinel_score"]),
        tensor_row("hard_candidate_mask", up_route["candidate"], po_route["candidate"], discrete=True),
        tensor_row("text_raw_score", up_route["text_score"], po_route["text_score"]),
        tensor_row("sigmoid_absolute_weight", up_route["absolute"], po_route["absolute"]),
        tensor_row("softmax_relative_weight", up_route["relative"], po_route["relative"]),
        tensor_row("final_weight", up_route["final"], po_route["final"]),
        tensor_row("expert_residual", up_residual, po_residual),
        tensor_row("post_layer_hidden_state", up_post, po_post),
        tensor_row("final_logits", up_logits, po_logits),
        compare_discrete("source_teacher_forced_token_prediction", up_prediction, po_prediction),
        compare_tensor_mapping("source_loss_components", up_component_values, po_component_values,
                               atol=1e-6, rtol=1e-5),
    ]

    # Complete Adam step on cloned initial states.  Only port-owned modules are
    # trainable; the official backbone remains frozen and byte-identical.
    official_optimizer = torch.optim.Adam(official.parameters(), lr=1e-4)
    official_optimizer.zero_grad(set_to_none=True)
    backward_source_objective(
        model, official, upstream_views, official_source_semantics=True,
        port_continuation_mode=port_continuation_mode,
    )
    official_gradients = gradient_state(official)
    official_gradient_hash = state_dict_sha256(official_gradients)
    official_optimizer.step()
    official_after = state_dict_sha256(module_state(official))
    official_after_state = {name: value.detach().cpu().clone() for name, value in official.state_dict().items()}
    del official_optimizer
    gc.collect(); torch.cuda.empty_cache()
    port_optimizer = torch.optim.Adam(port.parameters(), lr=1e-4)
    port_optimizer.zero_grad(set_to_none=True)
    backward_source_objective(
        model, port, port_views, official_source_semantics=False,
        port_continuation_mode=port_continuation_mode,
    )
    port_gradients = gradient_state(port)
    port_gradient_hash = state_dict_sha256(port_gradients)
    port_optimizer.step()
    port_after = state_dict_sha256(module_state(port))
    port_after_state = {name: value.detach().cpu().clone() for name, value in port.state_dict().items()}
    optimizer_row = compare_discrete(
        "optimizer_step_parameter_hashes",
        {"before": initial_official_hash, "after": official_after},
        {"before": initial_port_hash, "after": port_after},
    )
    rows.append(optimizer_row)
    summary = summarize_trace(rows, required_names=REQUIRED_TRACE_BOUNDARIES)
    status = PASS if summary["all_passed"] else FAIL
    tensor_manifest = {
        "protocol": "LIVEEDIT_STAGE_A_OFFICIAL_BACKBONE_E2E_TRACE_V1",
        "seed": 42,
        "model_compute_dtype": str(model.dtype),
        "model_id": MODEL_ID,
        "model_revision": audit["model_revision"],
        "official_sample_sha256": file_sha256(args.official_sample),
        "official_image_sha256": file_sha256(args.image_root / row["image"]),
        "upstream_commit": PINNED_COMMIT,
        "source_era_compatibility": [
            "single_image_placeholder_processor_boundary",
            "legacy__merge_input_ids_with_image_features",
        ],
        "training_continuation": {
            "upstream": "pinned_forward_from_mid_layer_reinjects_layer21_output_at_layer21_input",
            "port": port_continuation_mode.value,
        },
        "trace_tensor_hashes": {
            item["name"]: {
                "upstream": item.get("upstream_sha256"),
                "port": item.get("port_sha256"),
            }
            for item in rows if "upstream_sha256" in item
        },
    }
    optimizer_artifact = {
        "passed": optimizer_row["passed"],
        "initial_official": initial_official_hash,
        "initial_port": initial_port_hash,
        "after_official": official_after,
        "after_port": port_after,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "official_gradient_hash": official_gradient_hash,
        "port_gradient_hash": port_gradient_hash,
        "gradient_difference": state_difference(official_gradients, port_gradients),
        "parameter_difference_after_step": state_difference(official_after_state, port_after_state),
    }
    final = {
        "stage": "A",
        "status": status,
        "required_boundaries": len(REQUIRED_TRACE_BOUNDARIES),
        "passed_boundaries": summary["passed"],
        "failed_boundaries": [item["name"] for item in rows if not item.get("passed")],
        "asset_audit_passed": audit["passed"],
        "edited_checkpoint_loaded": False,
        "base_model_modified": False,
        "timestamp_utc": utc_now(),
        "port_continuation_mode": port_continuation_mode.value,
    }
    write_json(args.out_dir / "tensor_trace_manifest.json", tensor_manifest)
    write_json(args.out_dir / "tensor_parity_metrics.json", summary)
    write_json(args.out_dir / "optimizer_step_parity.json", optimizer_artifact)
    write_json(args.out_dir / "trace_parity_summary.json", final)
    report = [
        "# Stage A: official-backbone end-to-end trace parity", "",
        f"Status: `{status}`", "",
        f"- Required boundaries: {len(REQUIRED_TRACE_BOUNDARIES)}",
        f"- Passed boundaries: {summary['passed']}",
        f"- Failed boundaries: {', '.join(final['failed_boundaries']) or 'none'}",
        f"- Optimizer-step parameter hash parity: {optimizer_row['passed']}",
        "- Edited checkpoint loaded: no",
        "- Base model modified: no", "",
        "The source-era processor and removed private merge method were restored only as an execution compatibility adapter.",
        f"The source-loss boundary preserves the pinned upstream continuation semantics and compares them with port mode `{port_continuation_mode.value}`.", "",
    ]
    (args.out_dir / "END_TO_END_PARITY_REPORT.md").write_text("\n".join(report))
    (args.out_dir / "END_TO_END_TRACE_PARITY.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
