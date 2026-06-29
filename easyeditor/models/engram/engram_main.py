from __future__ import annotations

import json
import logging
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .bank import EngramBank
from .covariance import (
    CovarianceStat,
    LayerCovarianceCollector,
    SelectedLayer,
    as_device,
    covariance_from_activations,
    dtype_from_name,
)
from .engram_hparams import EngramMultimodalHparams
from .solver import (
    EngramLayerUpdate,
    SolverConfig,
    apply_update_to_module,
    compute_linear_engram_update,
    compute_projector,
)

LOG = logging.getLogger(__name__)


def _safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _join_prompt_target(prompt: str, target: str) -> str:
    prompt = "" if prompt is None else str(prompt)
    target = "" if target is None else str(target)
    if not prompt:
        return target
    if not target:
        return prompt
    if prompt.endswith(" ") or target.startswith(" "):
        return prompt + target
    return prompt + " " + target


def _compile_patterns(patterns: Sequence[str]) -> List[re.Pattern]:
    return [re.compile(pattern) for pattern in patterns]


def _extract_layer_number(name: str) -> Optional[int]:
    for pattern in (r"\.layers\.(\d+)\.", r"\.decoder\.layers\.(\d+)\.", r"layer\.(\d+)\."):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def _is_quantized_or_uneditable(module: nn.Linear) -> Optional[str]:
    class_name = module.__class__.__name__.lower()
    if "4bit" in class_name or "8bit" in class_name or "quant" in class_name:
        return f"module class {module.__class__.__name__} looks quantized"
    if not isinstance(module.weight, torch.nn.Parameter):
        return "weight is not a torch Parameter"
    if not module.weight.dtype.is_floating_point:
        return f"weight dtype {module.weight.dtype} is not floating point"
    return None


def select_linear_layers(model: nn.Module, hparams: EngramMultimodalHparams) -> List[SelectedLayer]:
    include = _compile_patterns(hparams.resolved_module_patterns())
    exclude = _compile_patterns(hparams.resolved_exclude_patterns())
    priority = _compile_patterns(hparams.module_priority_patterns)
    skip_dim = hparams.resolved_skip_dim()
    selected: List[SelectedLayer] = []
    skipped: List[str] = []
    layer_filter = set(int(x) for x in hparams.engram_layers) if hparams.engram_layers is not None else None

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if include and not any(pattern.search(name) for pattern in include):
            continue
        if exclude and any(pattern.search(name) for pattern in exclude):
            skipped.append(f"{name}: excluded by regex")
            continue
        layer_no = _extract_layer_number(name)
        if layer_filter is not None and layer_no is not None and layer_no not in layer_filter:
            continue
        reason = _is_quantized_or_uneditable(module)
        if reason:
            LOG.warning("[ENGRAM] skipping selected non-editable module %s: %s", name, reason)
            continue
        input_dim = int(module.in_features)
        cov_dim = input_dim + (1 if hparams.resolved_absorb_bias() and module.bias is not None else 0)
        if skip_dim is not None and cov_dim > int(skip_dim):
            skipped.append(f"{name}: cov_dim={cov_dim} exceeds limit={skip_dim}")
            continue
        selected.append(
            SelectedLayer(
                name=name,
                module=module,
                input_dim=input_dim,
                output_dim=int(module.out_features),
                absorb_bias=bool(hparams.resolved_absorb_bias() and module.bias is not None),
            )
        )

    if hparams.prioritize_module_selection and priority:
        def priority_key(item: SelectedLayer) -> Tuple[int, str]:
            for idx, pattern in enumerate(priority):
                if pattern.search(item.name):
                    return idx, item.name
            return len(priority), item.name

        ordered = sorted(selected, key=priority_key)
        seeded: List[SelectedLayer] = []
        seen = set()
        for pattern in priority:
            for item in ordered:
                if item.name in seen:
                    continue
                if pattern.search(item.name):
                    seeded.append(item)
                    seen.add(item.name)
                    break
        seeded.extend(item for item in ordered if item.name not in seen)
        selected = seeded
    if hparams.engram_max_modules is not None:
        selected = selected[: int(hparams.engram_max_modules)]

    LOG.info("[ENGRAM] selected modules: %s", [layer.name for layer in selected])
    if skipped:
        LOG.info("[ENGRAM] skipped modules: %s", skipped)
    return selected


def _empty_stat(layer: SelectedLayer, hparams: EngramMultimodalHparams) -> CovarianceStat:
    dtype = dtype_from_name(hparams.resolved_covariance_dtype())
    device = as_device(hparams.resolved_covariance_device())
    return CovarianceStat(
        cov=torch.zeros(layer.cov_dim, layer.cov_dim, device=device, dtype=dtype),
        count=0,
        input_dim=layer.input_dim,
        absorb_bias=layer.absorb_bias,
    )


def _merge_stats(target: Dict[str, CovarianceStat], reference: Dict[str, CovarianceStat]) -> Dict[str, CovarianceStat]:
    merged: Dict[str, CovarianceStat] = {}
    for name, plus in target.items():
        minus = reference.get(name)
        cov = plus.cov.clone()
        count = int(plus.count)
        if minus is not None:
            cov.add_(minus.cov.to(device=cov.device, dtype=cov.dtype))
            count += int(minus.count)
        merged[name] = CovarianceStat(
            cov=cov,
            count=count,
            input_dim=plus.input_dim,
            absorb_bias=plus.absorb_bias,
            warnings=list(plus.warnings) + (list(minus.warnings) if minus is not None else []),
        )
    return merged


def _solver_config(hparams: EngramMultimodalHparams) -> SolverConfig:
    return SolverConfig(
        method=hparams.solver,
        rcond=hparams.resolved_rcond(),
        svd_rank=hparams.svd_rank,
        energy_threshold=hparams.energy_threshold,
        solve_device=hparams.resolved_solve_device(),
        normalize_covariance=hparams.resolved_normalize_covariance(),
        jitter=hparams.resolved_jitter(),
    )


def solve_layer_update(
    *,
    layer: SelectedLayer,
    target_stat: CovarianceStat,
    reference_stat: CovarianceStat,
    hparams: EngramMultimodalHparams,
    candidate_weight_delta: Optional[torch.Tensor] = None,
    candidate_bias_delta: Optional[torch.Tensor] = None,
) -> EngramLayerUpdate:
    if target_stat.count <= 0:
        raise RuntimeError(f"ENGRAM collected no target activations for selected layer {layer.name}")
    if reference_stat.count < int(hparams.min_reference_examples):
        message = (
            f"ENGRAM reference count {reference_stat.count} below min_reference_examples="
            f"{hparams.min_reference_examples} for {layer.name}"
        )
        if hparams.skip_if_insufficient_reference:
            raise RuntimeError(message)
        LOG.warning("[ENGRAM] %s", message)

    projector, stats = compute_projector(
        target_stat.cov,
        reference_stat.cov,
        config=_solver_config(hparams),
        num_target_vectors=target_stat.count,
        num_reference_vectors=reference_stat.count,
    )
    return compute_linear_engram_update(
        layer.name,
        layer.module,
        projector,
        input_dim=layer.input_dim,
        absorb_bias=layer.absorb_bias,
        alpha=hparams.resolved_alpha(),
        beta=float(hparams.beta if str(hparams.edit_mode).lower() == "replacement" else 0.0),
        engram_update_direction=hparams.resolved_update_direction(),
        direction_sign=hparams.resolved_direction_sign(),
        behavior_objective=hparams.behavior_objective,
        candidate_weight_delta=candidate_weight_delta,
        candidate_bias_delta=candidate_bias_delta,
        store_projector=bool(hparams.store_projector),
        stats=stats,
    )


def apply_engram_to_linear(
    module: nn.Linear,
    target_activations: Any,
    reference_activations: Any,
    *,
    hparams: Optional[EngramMultimodalHparams] = None,
    module_name: str = "linear",
    candidate_module: Optional[nn.Linear] = None,
) -> EngramLayerUpdate:
    hparams = hparams or EngramMultimodalHparams(
        module_patterns=[r".*"],
        token_scope="all",
        covariance_device="cpu",
        solve_device="cpu",
    )
    layer = SelectedLayer(
        name=module_name,
        module=module,
        input_dim=int(module.in_features),
        output_dim=int(module.out_features),
        absorb_bias=bool(hparams.resolved_absorb_bias() and module.bias is not None),
    )
    dtype = dtype_from_name(hparams.resolved_covariance_dtype())
    target_stat = covariance_from_activations(
        target_activations,
        input_dim=layer.input_dim,
        absorb_bias=layer.absorb_bias,
        device=hparams.resolved_covariance_device(),
        dtype=dtype,
    )
    reference_stat = covariance_from_activations(
        reference_activations,
        input_dim=layer.input_dim,
        absorb_bias=layer.absorb_bias,
        device=hparams.resolved_covariance_device(),
        dtype=dtype,
    )
    cand_w = None
    cand_b = None
    if candidate_module is not None:
        cand_w = candidate_module.weight.detach() - module.weight.detach()
        if candidate_module.bias is not None and module.bias is not None:
            cand_b = candidate_module.bias.detach() - module.bias.detach()
    update = solve_layer_update(
        layer=layer,
        target_stat=target_stat,
        reference_stat=reference_stat,
        hparams=hparams,
        candidate_weight_delta=cand_w,
        candidate_bias_delta=cand_b,
    )
    apply_update_to_module(module, update, direction=-1)
    return update


class EngramMultimodalRewriteExecutor:
    """Forward-only ENGRAM editor for EasyEdit multimodal models."""

    def __init__(self) -> None:
        self.last_report: Dict[str, Any] = {}
        self.last_updates: Dict[str, EngramLayerUpdate] = {}
        self.last_target_stats: Dict[str, CovarianceStat] = {}
        self.last_reference_stats: Dict[str, CovarianceStat] = {}
        self.last_target_scope_logs: List[Dict[str, Any]] = []
        self.last_reference_scope_logs: List[Dict[str, Any]] = []

    def _device_for_model(self, model: nn.Module, hparams: EngramMultimodalHparams) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return as_device(hparams.device)

    def _supports_text_only(self, hparams: EngramMultimodalHparams) -> bool:
        return str(hparams.model_name).lower() not in {"llava-med", "llava_med", "owl-2"}

    def _stack_images(self, images: Sequence[Any], device: torch.device) -> Any:
        if not images or all(image is None for image in images):
            return None
        if any(image is None for image in images):
            raise ValueError("ENGRAM cannot batch mixed image and text-only examples; use batch_size=1.")
        if all(isinstance(image, torch.Tensor) for image in images):
            return torch.stack([image.to(device) for image in images], dim=0)
        return list(images)

    def _token_masks(self, tok: Any, prompts: List[str], targets: List[str], text_input: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        encoded = tok(text_input, add_special_tokens=False, return_tensors="pt", padding=True)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(encoded["input_ids"])
        attention_mask = attention_mask.to(device)
        prompt_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        answer_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for row, (prompt, target) in enumerate(zip(prompts, targets)):
            prompt_len = len(tok.encode(prompt, add_special_tokens=False))
            target_len = len(tok.encode(target, add_special_tokens=False))
            prompt_mask[row, : min(prompt_len, prompt_mask.shape[1])] = True
            start = min(prompt_len, answer_mask.shape[1])
            end = min(prompt_len + target_len, answer_mask.shape[1])
            answer_mask[row, start:end] = True
        return {
            "attention_mask": attention_mask.bool(),
            "prompt_mask": prompt_mask & attention_mask.bool(),
            "answer_mask": answer_mask & attention_mask.bool(),
        }

    def _variant_fields(self, request: Dict[str, Any], variant: str, hparams: EngramMultimodalHparams) -> Tuple[Optional[str], Optional[str], Any]:
        if variant == "edit":
            return _safe_text(request.get("prompt")), _safe_text(request.get("target")), request.get("image")
        if variant == "rephrase":
            return _safe_text(request.get("rephrase_prompt")), _safe_text(request.get("target")), request.get("image")
        if variant == "image_rephrase":
            return _safe_text(request.get("prompt")), _safe_text(request.get("target")), request.get("image_rephrase")
        if variant == "locality_text":
            if not self._supports_text_only(hparams):
                return None, None, None
            return _safe_text(request.get("locality_prompt")), _safe_text(request.get("locality_ground_truth")), None
        if variant == "locality_multimodal":
            prompt = request.get("multimodal_locality_prompt", request.get("locality_image_prompt"))
            target = request.get("multimodal_locality_ground_truth", request.get("locality_image_ground_truth"))
            image = request.get("multimodal_locality_image", request.get("locality_image"))
            return _safe_text(prompt), _safe_text(target), image
        if variant == "portability":
            prompts = request.get("portability_prompt") or []
            targets = request.get("portability_ground_truth") or []
            if prompts and targets:
                return _safe_text(prompts[0]), _safe_text(targets[0]), request.get("image")
            return None, None, None
        raise ValueError(f"Unknown ENGRAM request variant: {variant}")

    def _load_retain_pool(self, hparams: EngramMultimodalHparams) -> List[Dict[str, Any]]:
        if not hparams.retain_pool_path:
            return []
        path = Path(hparams.retain_pool_path)
        if not path.exists():
            raise FileNotFoundError(f"ENGRAM retain_pool_path does not exist: {path}")
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _make_batch(
        self,
        requests: Sequence[Dict[str, Any]],
        tok: Any,
        hparams: EngramMultimodalHparams,
        variant: str,
        device: torch.device,
    ) -> Optional[Dict[str, Any]]:
        prompts: List[str] = []
        targets: List[str] = []
        images: List[Any] = []
        source_requests: Iterable[Dict[str, Any]] = requests
        if variant == "retain_pool":
            source_requests = self._load_retain_pool(hparams)
        for request in source_requests:
            prompt, target, image = self._variant_fields(request, variant, hparams) if variant != "retain_pool" else (
                _safe_text(request.get("prompt")),
                _safe_text(request.get("target", request.get("ground_truth", ""))),
                request.get("image"),
            )
            if prompt is None or target is None:
                continue
            prompts.append(prompt)
            targets.append(target)
            images.append(image)

        if not prompts:
            return None

        text_input = [_join_prompt_target(prompt, target) for prompt, target in zip(prompts, targets)]
        labels = tok(targets, add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"].to(device)
        batch: Dict[str, Any] = {
            "image": self._stack_images(images, device),
            "prompt": prompts,
            "target": targets,
            "text_input": text_input,
            "labels": labels,
            "prompts_len": [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts],
            "variant": variant,
        }
        batch.update(self._token_masks(tok, prompts, targets, text_input, device))
        if all(isinstance(image, (str, os.PathLike)) for image in images if image is not None):
            batch["image_paths"] = [str(image) for image in images]
        return batch

    def _make_batches(
        self,
        requests: Sequence[Dict[str, Any]],
        tok: Any,
        hparams: EngramMultimodalHparams,
        variants: Sequence[str],
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        batches: List[Dict[str, Any]] = []
        for variant in variants:
            batch = self._make_batch(requests, tok, hparams, variant, device)
            if batch is not None:
                batches.append(batch)
            else:
                LOG.warning("[ENGRAM] no examples for variant=%s", variant)
        return batches

    def _collect(
        self,
        model: nn.Module,
        batches: Sequence[Dict[str, Any]],
        layers: Sequence[SelectedLayer],
        hparams: EngramMultimodalHparams,
        *,
        collection_name: str,
    ) -> Dict[str, CovarianceStat]:
        collector = LayerCovarianceCollector(
            layers,
            covariance_device=hparams.resolved_covariance_device(),
            covariance_dtype=dtype_from_name(hparams.resolved_covariance_dtype()),
            token_scope=hparams.resolved_token_scope(),
            mask_fallback=hparams.engram_mask_fallback,
        )
        with collector:
            with torch.no_grad():
                for batch in batches:
                    collector.set_batch(batch)
                    _ = model(batch)
                    collector.clear_batch()
        if collection_name == "target":
            self.last_target_scope_logs = list(collector.selection_logs)
        elif collection_name == "reference":
            self.last_reference_scope_logs = list(collector.selection_logs)
        return collector.stats

    def _candidate_deltas(
        self,
        layers: Sequence[SelectedLayer],
        hparams: EngramMultimodalHparams,
    ) -> Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        if str(hparams.edit_mode).lower() != "replacement":
            return {layer.name: (None, None) for layer in layers}
        source = str(hparams.candidate_delta_source or "none").lower()
        if source == "none":
            raise RuntimeError("ENGRAM replacement mode is experimental and requires candidate_delta_source.")
        if source == "lora_adapter":
            raise NotImplementedError("ENGRAM replacement mode with lora_adapter is not wired here; merge adapter to state_dict_pair first.")
        if source != "state_dict_pair":
            raise ValueError(f"Unknown candidate_delta_source={hparams.candidate_delta_source!r}")
        if not hparams.base_state_dict_path or not hparams.candidate_state_dict_path:
            raise RuntimeError("state_dict_pair replacement requires base_state_dict_path and candidate_state_dict_path.")
        base = torch.load(hparams.base_state_dict_path, map_location="cpu")
        candidate = torch.load(hparams.candidate_state_dict_path, map_location="cpu")
        base_sd = base.get("state_dict", base) if isinstance(base, dict) else base
        cand_sd = candidate.get("state_dict", candidate) if isinstance(candidate, dict) else candidate
        deltas: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
        for layer in layers:
            w_key = f"{layer.name}.weight"
            b_key = f"{layer.name}.bias"
            if w_key not in base_sd or w_key not in cand_sd:
                raise KeyError(f"Replacement delta missing key {w_key}")
            weight_delta = cand_sd[w_key].detach().cpu() - base_sd[w_key].detach().cpu()
            bias_delta = None
            if b_key in base_sd and b_key in cand_sd:
                bias_delta = cand_sd[b_key].detach().cpu() - base_sd[b_key].detach().cpu()
            deltas[layer.name] = (weight_delta, bias_delta)
        return deltas

    def _metadata(
        self,
        hparams: EngramMultimodalHparams,
        layers: Sequence[SelectedLayer],
        collect_time: float,
        requests: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        source_request_ids: List[str] = []
        if requests:
            for request in requests:
                request_id = (
                    request.get("record_id")
                    or request.get("source_record_id")
                    or request.get("id")
                    or request.get("case_id")
                )
                if request_id is not None:
                    source_request_ids.append(str(request_id))
        metadata = {
            "edit_id": hparams.resolved_edit_id(),
            "concept_id": hparams.concept_id,
            "modality": hparams.modality,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_name_or_path": hparams.name,
            "selected_modules": [layer.name for layer in layers],
            "target_variant_names": hparams.resolved_target_variants(),
            "reference_variant_names": hparams.resolved_reference_variants(),
            "alpha": hparams.resolved_alpha(),
            "beta": hparams.beta if str(hparams.edit_mode).lower() == "replacement" else 0.0,
            "engram_update_direction": hparams.resolved_update_direction(),
            "direction_sign": hparams.resolved_direction_sign(),
            "behavior_objective": hparams.behavior_objective,
            "paper_direction_equivalent": "paper_style_W_minus_alpha_E"
            if hparams.resolved_update_direction() == "subtract"
            else "equivalent_to_paper_subtract_with_signed_alpha_negative",
            "edit_mode": hparams.edit_mode,
            "replacement_experimental": str(hparams.edit_mode).lower() == "replacement",
            "token_scope": hparams.resolved_token_scope(),
            "covariance_dtype": hparams.resolved_covariance_dtype(),
            "covariance_device": hparams.resolved_covariance_device(),
            "rcond": hparams.resolved_rcond(),
            "svd_rank": hparams.svd_rank,
            "energy_threshold": hparams.energy_threshold,
            "collect_time_sec": collect_time,
        }
        if source_request_ids:
            metadata["source_request_ids"] = source_request_ids
            if len(source_request_ids) == 1:
                metadata["record_id"] = source_request_ids[0]
                metadata["source_record_id"] = source_request_ids[0]
        return metadata

    def apply_to_model(
        self,
        model: nn.Module,
        tok: Any,
        requests: List[Dict[str, Any]],
        hparams: EngramMultimodalHparams,
        copy: bool = False,
        return_orig_weights: bool = False,
        keep_original_weight: bool = False,
        **kwargs: Any,
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        del kwargs
        if copy:
            model = deepcopy(model)
        device = self._device_for_model(model, hparams)
        was_training = model.training
        model.eval()

        layers = select_linear_layers(model, hparams)
        if not layers:
            raise RuntimeError("ENGRAM selected no editable nn.Linear modules. Check YAML regex config.")
        hparams.resolved_direction_sign()

        target_batches = self._make_batches(requests, tok, hparams, hparams.resolved_target_variants(), device)
        reference_batches = self._make_batches(requests, tok, hparams, hparams.resolved_reference_variants(), device)
        if not target_batches:
            raise RuntimeError("ENGRAM found no target/edit variants. Check prompt/target/image request keys.")
        if not reference_batches:
            LOG.warning("[ENGRAM] no reference variants found; Sigma_minus will be zero.")

        start = time.time()
        target_stats = self._collect(model, target_batches, layers, hparams, collection_name="target")
        if reference_batches:
            reference_stats = self._collect(model, reference_batches, layers, hparams, collection_name="reference")
        else:
            reference_stats = {layer.name: _empty_stat(layer, hparams) for layer in layers}
            self.last_reference_scope_logs = []
        collect_time = time.time() - start
        self.last_target_stats = target_stats
        self.last_reference_stats = reference_stats

        candidate_deltas = self._candidate_deltas(layers, hparams)
        weights_copy: Dict[str, torch.Tensor] = {}
        updates: Dict[str, EngramLayerUpdate] = {}
        for layer in layers:
            weights_copy[f"{layer.name}.weight"] = layer.module.weight.detach().clone()
            if layer.module.bias is not None:
                weights_copy[f"{layer.name}.bias"] = layer.module.bias.detach().clone()
            cand_w, cand_b = candidate_deltas[layer.name]
            target_count = int(target_stats[layer.name].count)
            reference_count = int(reference_stats[layer.name].count)
            LOG.info(
                "[ENGRAM] module=%s dtype=%s device=%s weight_shape=%s bias=%s absorb_bias=%s "
                "target_vectors=%s reference_vectors=%s cov_dim=%s cov_device=%s solver=%s "
                "rcond=%s svd_rank=%s energy_threshold=%s rollback_snapshot=true",
                layer.name,
                layer.module.weight.dtype,
                layer.module.weight.device,
                tuple(layer.module.weight.shape),
                layer.module.bias is not None,
                layer.absorb_bias,
                target_count,
                reference_count,
                layer.cov_dim,
                hparams.resolved_covariance_device(),
                hparams.solver,
                hparams.resolved_rcond(),
                hparams.svd_rank,
                hparams.energy_threshold,
            )
            if target_count <= 0:
                LOG.warning("[ENGRAM] skipping %s because target activation count is zero", layer.name)
                continue
            if reference_count <= 0:
                LOG.warning("[ENGRAM] WARNING: %s has zero reference activation vectors", layer.name)
            update = solve_layer_update(
                layer=layer,
                target_stat=target_stats[layer.name],
                reference_stat=reference_stats[layer.name],
                hparams=hparams,
                candidate_weight_delta=cand_w,
                candidate_bias_delta=cand_b,
            )
            norm_ratio = float(update.stats.get("norm_ratio", 0.0))
            effective_norm_ratio = abs(float(update.alpha)) * norm_ratio
            update.stats["effective_norm_ratio"] = effective_norm_ratio
            update.stats["effective_update_norm_ratio"] = effective_norm_ratio
            update.stats["engram_update_direction"] = update.engram_update_direction
            update.stats["direction_sign"] = update.direction_sign
            update.stats["behavior_objective"] = update.behavior_objective
            update.stats["paper_direction_equivalent"] = update.paper_direction_equivalent
            if norm_ratio > float(hparams.norm_ratio_warn_threshold):
                LOG.warning(
                    "[ENGRAM] %s norm_ratio %.6f exceeds warning threshold %.6f (effective %.6f after alpha)",
                    layer.name,
                    norm_ratio,
                    hparams.norm_ratio_warn_threshold,
                    effective_norm_ratio,
                )
            if (
                hparams.skip_if_norm_ratio_larger_than is not None
                and effective_norm_ratio > float(hparams.skip_if_norm_ratio_larger_than)
            ):
                LOG.warning(
                    "[ENGRAM] skipping %s because effective_norm_ratio %.6f exceeds skip threshold %.6f",
                    layer.name,
                    effective_norm_ratio,
                    hparams.skip_if_norm_ratio_larger_than,
                )
                continue
            LOG.info(
                "[ENGRAM] solved module=%s rank_plus=%s rank_total=%s norm_W=%.6g norm_E=%.6g "
                "norm_ratio=%.6g alpha=%.6g direction=%s sign=%s update_dtype=%s",
                layer.name,
                update.stats.get("rank_plus"),
                update.stats.get("rank_total"),
                update.stats.get("norm_W", 0.0),
                update.stats.get("norm_E", 0.0),
                update.stats.get("norm_ratio", 0.0),
                update.alpha,
                update.engram_update_direction,
                update.direction_sign,
                update.weight.dtype,
            )
            updates[layer.name] = update
        if not updates:
            raise RuntimeError("ENGRAM produced no layer updates after safety checks.")

        applied: List[SelectedLayer] = []
        with torch.no_grad():
            try:
                for layer in layers:
                    if layer.name not in updates:
                        continue
                    apply_update_to_module(layer.module, updates[layer.name], direction=-1)
                    applied.append(layer)
                if hparams.clear_cuda_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                LOG.exception("[ENGRAM] update application failed; rolling back %s already-applied modules", len(applied))
                for layer in reversed(applied):
                    layer.module.weight.copy_(
                        weights_copy[f"{layer.name}.weight"].to(layer.module.weight.device, dtype=layer.module.weight.dtype)
                    )
                    bias_key = f"{layer.name}.bias"
                    if layer.module.bias is not None and bias_key in weights_copy:
                        layer.module.bias.copy_(weights_copy[bias_key].to(layer.module.bias.device, dtype=layer.module.bias.dtype))
                raise

        metadata = self._metadata(hparams, layers, collect_time, requests=requests)
        metadata["target_token_scope_logs"] = self.last_target_scope_logs
        metadata["reference_token_scope_logs"] = self.last_reference_scope_logs
        metadata["layers"] = [dict(update.stats) for update in updates.values()]
        metadata["selected_modules"] = list(updates.keys())
        layer_effective = [
            float(update.stats.get("effective_update_norm_ratio", update.stats.get("effective_norm_ratio", 0.0)))
            for update in updates.values()
        ]
        metadata["effective_update_norm_ratio"] = max(layer_effective) if layer_effective else 0.0
        self.last_report = {"metadata": metadata, "updates": list(updates.keys())}
        self.last_updates = updates
        bank_dir = hparams.resolved_bank_dir()
        if bank_dir:
            edit_id = hparams.resolved_edit_id() or f"engram_{int(time.time())}"
            metadata["edit_id"] = edit_id
            EngramBank(bank_dir).save_edit(edit_id=edit_id, metadata=metadata, updates=updates, overwrite=True)
            LOG.info("[ENGRAM] saved edit %s to bank %s", edit_id, bank_dir)

        if was_training:
            model.train()
        if not return_orig_weights:
            return model, {}
        return model, weights_copy


def apply_engram_to_multimodal_model(*args: Any, **kwargs: Any):
    return EngramMultimodalRewriteExecutor().apply_to_model(*args, **kwargs)
