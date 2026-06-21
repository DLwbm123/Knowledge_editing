import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsca_utils import (
    DSCAConceptRepository,
    DSCAContext,
    DSCAPhaseTimer,
    dsca_contrastive_distill_loss,
    dsca_intervention_context,
    dsca_route,
    dsca_sparse_loss,
    extract_dsca_region_representations,
    extract_tensor_from_module_output,
    find_module_by_path,
    get_dsca_masks_from_output_or_batch,
    replace_tensor_in_module_output,
)
from .editable_model import EditableModel
from ..utils import safe_backward

LOG = logging.getLogger(__name__)


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _tensor_logits(outputs: Any) -> torch.Tensor:
    return outputs if isinstance(outputs, torch.Tensor) else outputs.logits


def _answer_labels_for_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] > labels.shape[1]:
        return labels
    return labels[:, -logits.shape[1] - 1 :]


class DSCA(EditableModel):
    """Stage 1 DSCA activation-space multimodal editor."""

    def __init__(self, model, config, model_constructor):
        super().__init__(model, config, model_constructor)
        model_name = str(self.config.model_name).lower()
        if ("llava" in model_name and model_name not in {"llava-med", "llava_med"}) or "paligemma" in model_name:
            raise NotImplementedError(
                "DSCA Stage 1 supports BLIP2 and MiniGPT-4, plus verified LLaVA-Med only; "
                "this backbone lacks verified full-sequence masks."
            )
        self.hidden_size = self._infer_hidden_size()
        self.dsca_layer = int(_cfg(self.config, "dsca_layer", 21))
        self.dsca_layer_module = _cfg(self.config, "dsca_layer_module", None)
        self.rank = int(_cfg(self.config, "dsca_rank", 128))
        self.gate_bottleneck = int(_cfg(self.config, "dsca_gate_bottleneck", 64))
        self.tau_visual = float(_cfg(self.config, "dsca_tau_visual", 0.0))
        self.route_temperature = float(_cfg(self.config, "dsca_route_temperature", 0.07))
        self.distill_temperature = float(_cfg(self.config, "dsca_distill_temperature", 0.07))
        self.candidate_topk = _cfg(self.config, "dsca_candidate_topk", None)
        self.require_masks = bool(_cfg(self.config, "dsca_require_masks", True))
        self.residual_apply_mask = str(_cfg(self.config, "dsca_residual_apply_mask", "attention"))
        self.generation_mode = str(_cfg(self.config, "dsca_generation_mode", "normal"))
        self.generation_residual_apply_mask = str(
            _cfg(self.config, "dsca_generation_residual_apply_mask", self.residual_apply_mask)
        )
        self.generation_reuse_prefill_route = bool(_cfg(self.config, "dsca_generation_reuse_prefill_route", False))
        self.generation_update_repository = bool(_cfg(self.config, "dsca_generation_update_repository", False))
        self.update_clusters_during_training = bool(_cfg(self.config, "dsca_update_clusters_during_training", True))
        self.update_clusters_during_inference = bool(_cfg(self.config, "dsca_update_clusters_during_inference", False))
        self.freeze_repository_at_eval = bool(_cfg(self.config, "dsca_freeze_repository_at_eval", True))
        self.disable_pca_refine = bool(_cfg(self.config, "dsca_disable_pca_refine", False))
        self.disable_basis_initialization = bool(_cfg(self.config, "dsca_disable_basis_initialization", False))
        self.disable_task_loss = bool(_cfg(self.config, "dsca_disable_task_loss", False))
        self.disable_align = bool(_cfg(self.config, "dsca_disable_align", False))
        self.disable_cdistill = bool(_cfg(self.config, "dsca_disable_cdistill", False))
        self.disable_sparse = bool(_cfg(self.config, "dsca_disable_sparse", False))
        self.repository = DSCAConceptRepository(
            hidden_size=self.hidden_size,
            rank=self.rank,
            gate_bottleneck=self.gate_bottleneck,
            cluster_alpha=float(_cfg(self.config, "dsca_cluster_alpha", 2.0)),
            proto_ema=float(_cfg(self.config, "dsca_proto_ema", 0.95)),
            min_samples=int(_cfg(self.config, "dsca_min_samples", 32)),
            refine_interval=int(_cfg(self.config, "dsca_refine_interval", 500)),
            max_buffer_size=_cfg(self.config, "dsca_max_buffer_size", None),
            dsam_init_std=float(_cfg(self.config, "dsca_dsam_init_std", 0.02)),
            residual_scale=float(_cfg(self.config, "dsca_residual_scale", 1.0)),
        )
        self._optimizer_anchor = nn.Parameter(torch.zeros(1))
        self._dsca_context: Optional[DSCAContext] = None
        self._last_capture_hidden: Optional[torch.Tensor] = None
        self._last_reps: Optional[Dict[str, torch.Tensor]] = None
        self._last_route_weights: Optional[torch.Tensor] = None
        self._last_route_selected: Optional[torch.Tensor] = None
        self._last_residual: Optional[torch.Tensor] = None
        self._last_apply_mask: Optional[torch.Tensor] = None
        self._last_masks: Optional[Dict[str, torch.Tensor]] = None
        self._last_info: Dict[str, float] = {}
        self._generation_hook_call_index = 0
        self._cached_generation_route_weights: Optional[torch.Tensor] = None
        self._cached_generation_route_selected: Optional[torch.Tensor] = None
        self.global_step = 0
        self._registered_param_ids = set()
        self._install_hook()
        if bool(_cfg(self.config, "dsca_freeze_vlm", True)):
            self.freeze_base_vlm()
        repo_path = _cfg(self.config, "dsca_repository_path", None)
        if repo_path:
            loaded = DSCAConceptRepository.load(repo_path)
            self.repository.load_state_dict(loaded.state_dict())

    def _tensor_device_from_batch(self, batch: Dict[str, Any]) -> Optional[torch.device]:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.device
            if isinstance(value, dict):
                found = self._tensor_device_from_batch(value)
                if found is not None:
                    return found
        return None

    def _make_phase_timer(self, batch: Dict[str, Any]) -> DSCAPhaseTimer:
        step = int(self.global_step)
        enabled = bool(_cfg(self.config, "dsca_profile_edit_step", False))
        start_step = int(_cfg(self.config, "dsca_profile_start_step", 0))
        end_step = int(_cfg(self.config, "dsca_profile_end_step", 10**9))
        enabled = enabled and start_step <= step <= end_step
        return DSCAPhaseTimer(
            enabled=enabled,
            log_path=_cfg(self.config, "dsca_profile_log_path", None),
            device=self._tensor_device_from_batch(batch),
            step=step,
        )

    def _infer_hidden_size(self) -> int:
        if hasattr(self.model, "opt_model"):
            return int(self.model.opt_model.config.hidden_size)
        if hasattr(self.model, "llama_model"):
            return int(self.model.llama_model.config.hidden_size)
        if hasattr(self.model, "llava_model"):
            return int(self.model.llava_model.config.hidden_size)
        raise NotImplementedError(f"DSCA does not know how to infer hidden size for {type(self.model)}.")

    def _layer_path(self) -> str:
        if self.dsca_layer_module:
            return str(self.dsca_layer_module)
        model_name = str(self.config.model_name).lower()
        if "blip2" in model_name:
            return f"opt_model.model.decoder.layers.{self.dsca_layer}"
        if "minigpt4" in model_name:
            return f"llama_model.model.layers.{self.dsca_layer}"
        if model_name in {"llava-med", "llava_med"}:
            return f"llava_model.model.layers.{self.dsca_layer}"
        raise NotImplementedError(
            f"DSCA Stage 1 supports BLIP2 and MiniGPT-4, plus verified LLaVA-Med only, got {self.config.model_name}."
        )

    def _install_hook(self) -> None:
        layer_path = self._layer_path()
        layer = find_module_by_path(self.model, layer_path)
        self._hook_handle = layer.register_forward_hook(self._dsca_hook)
        self.dsca_layer_path = layer_path
        LOG.info("DSCA hooked %s", layer_path)

    def remove_hook(self) -> None:
        if hasattr(self, "_hook_handle") and self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def freeze_base_vlm(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    def set_editor_train(self, training: bool) -> None:
        self.model.eval()
        self.repository.train(training)
        for param in self.repository.parameters():
            param.requires_grad_(training)
        self._optimizer_anchor.requires_grad_(training)

    def outer_parameters(self):
        return [self._optimizer_anchor] + list(self.repository.parameters())

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in list(state.keys()):
            if key.startswith(f"{prefix}model."):
                del state[key]
        state[f"{prefix}dsca_config"] = deepcopy(self.config)
        state[f"{prefix}global_step"] = torch.tensor(self.global_step)
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = dict(state_dict)
        state_dict.pop("dsca_config", None)
        global_step = state_dict.pop("global_step", None)
        result = super().load_state_dict(state_dict, strict=False)
        if global_step is not None:
            self.global_step = int(global_step.item() if torch.is_tensor(global_step) else global_step)
        unexpected = [key for key in result.unexpected_keys if not key.startswith("model.")]
        missing = [key for key in result.missing_keys if not key.startswith("model.")]
        if unexpected or (strict and missing):
            raise RuntimeError(f"DSCA checkpoint mismatch. missing={missing}, unexpected={unexpected}")
        return result

    def _apply_mask_for_residual(
        self,
        masks: Dict[str, torch.Tensor],
        mode: Optional[str] = None,
        hidden: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
    ) -> torch.Tensor:
        mode = mode or self.residual_apply_mask
        if mode in {"attention", "all_nonpad"}:
            return masks["attention_mask"]
        if mode == "vision_prompt":
            return masks["vision_mask"] | masks["prompt_mask"]
        if mode == "current_token":
            if hidden is None:
                return masks["attention_mask"]
            current = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
            current[:, -1] = True
            return current & masks["attention_mask"].to(hidden.device).bool()
        raise ValueError(f"Unsupported dsca_residual_apply_mask: {mode}")

    def _generation_mode_for_context(self, context: Optional[DSCAContext]) -> str:
        if context is not None and context.generation_mode:
            return str(context.generation_mode)
        if context is not None and context.is_generation:
            return self.generation_mode
        return "normal"

    def _generation_reuse_prefill_route_for_context(self, context: Optional[DSCAContext], generation_mode: str) -> bool:
        if generation_mode != "cache_reuse_route":
            return False
        if context is not None and context.generation_reuse_prefill_route is not None:
            return bool(context.generation_reuse_prefill_route)
        if context is not None and context.is_generation:
            return bool(self.generation_reuse_prefill_route)
        return True

    def _generation_residual_mask_for_context(self, context: Optional[DSCAContext]) -> Optional[str]:
        if context is not None and context.residual_apply_mask_mode:
            return str(context.residual_apply_mask_mode)
        if context is not None and context.is_generation:
            return self.generation_residual_apply_mask
        return None

    def _zero_info(self, hidden: torch.Tensor, weights: Optional[torch.Tensor] = None, selected: Optional[torch.Tensor] = None):
        if weights is None:
            weights = torch.zeros(hidden.shape[0], len(self.repository), device=hidden.device, dtype=hidden.dtype)
        if selected is None:
            selected = torch.zeros(hidden.shape[0], len(self.repository), device=hidden.device, dtype=torch.bool)
        counts = selected.sum(dim=1).float() if selected.numel() else torch.zeros(hidden.shape[0], device=hidden.device)
        return {
            "dsca/num_candidates_mean": float(counts.mean().detach().cpu()) if counts.numel() else 0.0,
            "dsca/num_candidates_max": float(counts.max().detach().cpu()) if counts.numel() else 0.0,
            "dsca/route_weight_mean": float(weights[weights > 0].mean().detach().cpu()) if (weights > 0).any() else 0.0,
            "dsca/route_weight_max": float(weights.max().detach().cpu()) if weights.numel() else 0.0,
            "dsca/residual_norm_mean": 0.0,
        }

    def _route_and_residual(
        self,
        hidden: torch.Tensor,
        masks: Dict[str, torch.Tensor],
        timer: Optional[DSCAPhaseTimer] = None,
        context: Optional[DSCAContext] = None,
        precomputed_weights: Optional[torch.Tensor] = None,
        precomputed_selected: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        forced_ids = [] if context is None or context.force_route_ids is None else list(context.force_route_ids)
        using_precomputed_route = precomputed_weights is not None and precomputed_selected is not None
        generation_mode = self._generation_mode_for_context(context)
        cached_route_decode = context is not None and generation_mode == "cache_reuse_route" and hidden.dim() == 3 and hidden.shape[1] == 1
        needs_reps = not using_precomputed_route and not cached_route_decode
        reps: Optional[Dict[str, torch.Tensor]] = None
        if needs_reps:
            if timer:
                timer.start("extract_region_representations", {"hidden_shape": list(hidden.shape)})
            reps = extract_dsca_region_representations(hidden, masks)
            if timer:
                timer.stop(
                    "extract_region_representations",
                    {
                        "vision_tokens": masks["vision_mask"].sum(dim=1),
                        "prompt_tokens": masks["prompt_mask"].sum(dim=1),
                        "attention_tokens": masks["attention_mask"].sum(dim=1),
                    },
                )
            self._last_reps = reps
        if len(self.repository) == 0 or self.repository.num_active() == 0:
            weights = torch.zeros(hidden.shape[0], len(self.repository), device=hidden.device, dtype=hidden.dtype)
            self._last_route_weights = weights
            self._last_route_selected = torch.zeros(hidden.shape[0], len(self.repository), device=hidden.device, dtype=torch.bool)
            return torch.zeros_like(hidden), self._zero_info(hidden, weights)
        if using_precomputed_route:
            weights = precomputed_weights.to(hidden.device, hidden.dtype)
            selected = precomputed_selected.to(hidden.device).bool()
        elif context is not None and context.disable_normal_routing:
            weights = torch.zeros(hidden.shape[0], len(self.repository), device=hidden.device, dtype=hidden.dtype)
            selected = torch.zeros(hidden.shape[0], len(self.repository), device=hidden.device, dtype=torch.bool)
        elif cached_route_decode:
            weights = torch.zeros(hidden.shape[0], len(self.repository), device=hidden.device, dtype=hidden.dtype)
            selected = torch.zeros(hidden.shape[0], len(self.repository), device=hidden.device, dtype=torch.bool)
        else:
            if reps is None:
                raise RuntimeError("DSCA route computation requires region representations.")
            weights, selected, _ = dsca_route(
                reps["h_v"],
                reps["h_f"],
                self.repository,
                tau_visual=self.tau_visual,
                route_temperature=self.route_temperature,
                candidate_topk=self.candidate_topk,
                timer=timer,
            )
        if forced_ids:
            forced = torch.zeros(hidden.shape[0], len(self.repository), device=hidden.device, dtype=torch.bool)
            for idx in forced_ids:
                if 0 <= int(idx) < len(self.repository) and bool(self.repository.active[int(idx)].item()):
                    forced[:, int(idx)] = True
            selected = forced
            weights = torch.zeros(hidden.shape[0], len(self.repository), device=hidden.device, dtype=hidden.dtype)
            counts = forced.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
            weights = torch.where(forced, torch.ones_like(weights) / counts, weights)
        self._last_route_weights = weights
        self._last_route_selected = selected
        if weights.numel() == 0 or torch.count_nonzero(weights).item() == 0:
            return torch.zeros_like(hidden), self._zero_info(hidden, weights, selected)
        apply_mask = self._apply_mask_for_residual(
            masks,
            mode=self._generation_residual_mask_for_context(context),
            hidden=hidden,
        )
        self._last_apply_mask = apply_mask.detach()
        residual = torch.zeros_like(hidden)
        if timer:
            timer.start(
                "compute_dsam_residual",
                {
                    "cluster_count": len(self.repository),
                    "active_count": self.repository.num_active(),
                    "hidden_shape": list(hidden.shape),
                    "apply_tokens": apply_mask.sum(dim=1),
                },
            )
        for idx, dsam in enumerate(self.repository.dsams):
            if idx >= weights.shape[1] or not dsam.active:
                continue
            if torch.count_nonzero(weights[:, idx]).item() == 0:
                continue
            residual = residual + dsam(hidden, apply_mask) * weights[:, idx].view(-1, 1, 1).to(hidden.dtype)
        if timer:
            timer.stop("compute_dsam_residual", {"residual_shape": list(residual.shape)})
        info = self._zero_info(hidden, weights, selected)
        info["dsca/residual_norm_mean"] = float(residual.norm(dim=-1).mean().detach().cpu())
        return residual, info

    def _dsca_hook(self, module: nn.Module, args: Tuple[Any, ...], output: Any) -> Any:
        context = self._dsca_context
        if context is None or not context.enabled:
            return output
        hidden = extract_tensor_from_module_output(output)
        self._generation_hook_call_index += 1
        self._last_route_weights = None
        self._last_route_selected = None
        self._last_apply_mask = None
        event = self._make_hook_event(context, hidden, output, args)
        try:
            masks = get_dsca_masks_from_output_or_batch(output, context.batch or {}, hidden, require_answer=False)
        except RuntimeError as exc:
            masks = self._maybe_generation_masks_for_mismatch(context, hidden, str(exc))
            if masks is None:
                self._finish_hook_event(event, context, hidden, None, None, None, error=str(exc), skipped=True)
                if hidden.dim() == 3 and hidden.shape[1] == 1 and "shape" in str(exc):
                    return output
                raise
        event.update(self._mask_event_fields(masks))
        generation_mode = self._generation_mode_for_context(context)
        reuse_prefill_route = self._generation_reuse_prefill_route_for_context(context, generation_mode)
        if hidden.dim() == 3 and hidden.shape[1] > 1:
            self._cached_generation_route_weights = None
            self._cached_generation_route_selected = None
        if hidden.dim() == 3 and hidden.shape[1] == 1 and generation_mode == "prefill_only":
            self._finish_hook_event(event, context, hidden, masks, None, None, skipped=True, skip_reason="prefill_only_cached_decode")
            return output
        precomputed_weights = None
        precomputed_selected = None
        if hidden.dim() == 3 and hidden.shape[1] == 1 and generation_mode == "cache_reuse_route" and reuse_prefill_route:
            precomputed_weights = self._cached_generation_route_weights
            precomputed_selected = self._cached_generation_route_selected
            event["cached_decode_route_reused"] = precomputed_weights is not None and precomputed_selected is not None
        else:
            event["cached_decode_route_reused"] = False
        self._last_masks = {key: value.detach() for key, value in masks.items()}
        self._last_capture_hidden = hidden
        skip_region_reps = hidden.dim() == 3 and hidden.shape[1] == 1 and generation_mode == "cache_reuse_route" and reuse_prefill_route
        if not skip_region_reps:
            reps = extract_dsca_region_representations(hidden, masks)
            self._last_reps = reps
        if context.capture_only:
            self._last_residual = None
            self._finish_hook_event(event, context, hidden, masks, None, None, skipped=True, skip_reason="capture_only")
            return output
        residual, info = self._route_and_residual(
            hidden,
            masks,
            timer=context.timer,
            context=context,
            precomputed_weights=precomputed_weights,
            precomputed_selected=precomputed_selected,
        )
        self._last_residual = residual.detach()
        self._last_info = info
        if hidden.dim() == 3 and hidden.shape[1] > 1 and generation_mode in {"normal", "prefill_only", "cache_reuse_route"}:
            self._cached_generation_route_weights = self._last_route_weights.detach().clone() if self._last_route_weights is not None else None
            self._cached_generation_route_selected = self._last_route_selected.detach().clone() if self._last_route_selected is not None else None
        self._finish_hook_event(event, context, hidden, masks, residual, info)
        if torch.count_nonzero(residual).item() == 0:
            return output
        new_hidden = hidden + residual
        if not skip_region_reps:
            self._last_reps = extract_dsca_region_representations(new_hidden, masks)
        return replace_tensor_in_module_output(output, new_hidden)

    def _maybe_generation_masks_for_mismatch(
        self,
        context: DSCAContext,
        hidden: torch.Tensor,
        error: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if hidden.dim() != 3 or "shape" not in error:
            return None
        generation_mode = self._generation_mode_for_context(context)
        reuse_prefill_route = self._generation_reuse_prefill_route_for_context(context, generation_mode)
        if not context.extend_generation_masks and not reuse_prefill_route:
            return None
        batch = context.batch or {}
        attention = batch.get("attention_mask")
        vision = batch.get("vision_mask")
        prompt = batch.get("prompt_mask")
        if not torch.is_tensor(attention) or not torch.is_tensor(vision) or not torch.is_tensor(prompt):
            return None
        target_shape = hidden.shape[:2]
        source_len = int(attention.shape[1])
        target_len = int(target_shape[1])
        if target_len == source_len:
            return None
        if target_len == 1 and generation_mode != "cache_reuse_route":
            return None
        masks: Dict[str, torch.Tensor] = {}
        for name in ("attention_mask", "vision_mask", "prompt_mask", "answer_mask"):
            value = batch.get(name)
            if not torch.is_tensor(value):
                value = torch.zeros_like(attention)
            value = value.to(hidden.device).bool()
            if target_len < source_len:
                masks[name] = value[:, -target_len:]
            else:
                pad = torch.zeros(value.shape[0], target_len - source_len, device=hidden.device, dtype=torch.bool)
                if name == "attention_mask":
                    pad.fill_(True)
                masks[name] = torch.cat([value, pad], dim=1)
        if generation_mode == "cache_reuse_route" and target_len == 1 and reuse_prefill_route:
            masks["attention_mask"] = torch.ones(target_shape, device=hidden.device, dtype=torch.bool)
            masks["vision_mask"] = torch.zeros(target_shape, device=hidden.device, dtype=torch.bool)
            masks["prompt_mask"] = torch.ones(target_shape, device=hidden.device, dtype=torch.bool)
            masks["answer_mask"] = torch.zeros(target_shape, device=hidden.device, dtype=torch.bool)
            return masks
        return get_dsca_masks_from_output_or_batch(None, masks, hidden, require_answer=False)

    def _make_hook_event(self, context: DSCAContext, hidden: torch.Tensor, output: Any, args: Tuple[Any, ...]) -> Dict[str, Any]:
        phase = "prefill" if hidden.dim() == 3 and hidden.shape[1] > 1 else "cached_decode" if hidden.dim() == 3 and hidden.shape[1] == 1 else "unknown"
        return {
            "sample_id": context.sample_id,
            "call_label": context.call_label,
            "decode_call_index": self._generation_hook_call_index,
            "phase": phase,
            "layer_path": getattr(self, "dsca_layer_path", None),
            "hook_entered": True,
            "hidden_shape": list(hidden.shape),
            "hidden_dtype": str(hidden.dtype),
            "hidden_device": str(hidden.device),
            "output_structure_type": type(output).__name__,
            "use_cache": context.generation_use_cache,
            "past_key_values_present": any("Cache" in type(arg).__name__ or "past" in type(arg).__name__.lower() for arg in args),
            "dsca_context_active": context.enabled,
            "repository_update_enabled": context.update_clusters,
            "repository_num_clusters": len(self.repository),
            "repository_num_active_dsams": self.repository.num_active(),
            "residual_scale": self.repository.residual_scale,
            "residual_apply_mask_mode": self._generation_residual_mask_for_context(context) or self.residual_apply_mask,
            "generation_mode": self._generation_mode_for_context(context),
            "generation_reuse_prefill_route": self._generation_reuse_prefill_route_for_context(
                context, self._generation_mode_for_context(context)
            ),
            "force_route_ids": list(context.force_route_ids or []),
            "disable_normal_routing": bool(context.disable_normal_routing),
        }

    def _mask_event_fields(self, masks: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        def shape(name: str) -> Optional[List[int]]:
            value = masks.get(name)
            return list(value.shape) if torch.is_tensor(value) else None

        def mask_sum(name: str) -> Optional[int]:
            value = masks.get(name)
            return int(value.sum().detach().cpu()) if torch.is_tensor(value) else None

        return {
            "attention_mask_shape": shape("attention_mask"),
            "vision_mask_shape": shape("vision_mask"),
            "prompt_mask_shape": shape("prompt_mask"),
            "answer_mask_shape": shape("answer_mask"),
            "attention_mask_sum": mask_sum("attention_mask"),
            "vision_mask_sum": mask_sum("vision_mask"),
            "prompt_mask_sum": mask_sum("prompt_mask"),
            "answer_mask_sum": mask_sum("answer_mask"),
        }

    def _finish_hook_event(
        self,
        event: Dict[str, Any],
        context: DSCAContext,
        hidden: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]],
        residual: Optional[torch.Tensor],
        info: Optional[Dict[str, float]],
        error: Optional[str] = None,
        skipped: bool = False,
        skip_reason: Optional[str] = None,
    ) -> None:
        if context.debug_events is None:
            return
        weights = self._last_route_weights
        selected = self._last_route_selected
        apply_mask = self._last_apply_mask
        candidate_ids: List[int] = []
        active_candidate_ids: List[int] = []
        route_weights: List[float] = []
        if torch.is_tensor(weights) and weights.numel():
            row = weights[0].detach().float().cpu()
            route_weights = row.tolist()
            candidate_ids = [idx for idx, value in enumerate(route_weights) if value > 0.0]
            active_candidate_ids = [
                idx for idx in candidate_ids if idx < len(self.repository) and bool(self.repository.active[idx].item())
            ]
        event.update(
            {
                "candidate_ids": candidate_ids,
                "active_candidate_ids": active_candidate_ids,
                "route_weights": route_weights,
                "residual_norm": float(residual.detach().float().norm().cpu()) if torch.is_tensor(residual) else None,
                "hidden_delta_norm": float(residual.detach().float().norm().cpu()) if torch.is_tensor(residual) else None,
                "apply_mask_shape": list(apply_mask.shape) if torch.is_tensor(apply_mask) else None,
                "apply_mask_sum": int(apply_mask.sum().detach().cpu()) if torch.is_tensor(apply_mask) else None,
                "apply_mask_all_false_active_route": bool(active_candidate_ids and torch.is_tensor(apply_mask) and int(apply_mask.sum().detach().cpu()) == 0),
                "skipped": skipped,
                "skip_reason": skip_reason,
                "error": error,
            }
        )
        if masks is not None:
            event.update(self._mask_event_fields(masks))
        context.debug_events.append(event)

    def capture_representations(
        self,
        batch: Dict[str, Any],
        timer: Optional[DSCAPhaseTimer] = None,
        phase: str = "hidden_capture",
    ) -> Dict[str, torch.Tensor]:
        self._last_capture_hidden = None
        self._last_reps = None
        if timer:
            timer.start(phase)
        with torch.no_grad(), dsca_intervention_context(self, DSCAContext(batch=batch, capture_only=True, timer=timer)):
            self.model(batch)
        if self._last_reps is None:
            raise RuntimeError("DSCA failed to capture layer representations.")
        if timer:
            timer.stop(
                phase,
                {
                    "h_v_shape": list(self._last_reps["h_v"].shape),
                    "h_t_shape": list(self._last_reps["h_t"].shape),
                    "h_f_shape": list(self._last_reps["h_f"].shape),
                },
            )
        return {key: value.detach() for key, value in self._last_reps.items()}

    def _forward_with_dsca(
        self,
        batch: Dict[str, Any],
        timer: Optional[DSCAPhaseTimer] = None,
        phase: str = "forward_dsca_enabled",
    ) -> Any:
        if timer:
            timer.start(phase)
        try:
            with dsca_intervention_context(self, DSCAContext(batch=batch, timer=timer)):
                return self.model(batch)
        finally:
            if timer:
                timer.stop(phase)

    def forward(self, *inputs, **kwargs):
        if len(inputs) != 1 or not isinstance(inputs[0], dict) or kwargs:
            return self.model(*inputs, **kwargs)
        return self._forward_with_dsca(inputs[0])

    def _nll(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        logits = _tensor_logits(outputs)
        labels = batch["labels"]
        return self.edit_loss_fn(self.config, logits, _answer_labels_for_logits(logits, labels))["nll"]

    def _update_clusters_from_batch(
        self,
        batch: Dict[str, Any],
        metadata: Optional[Iterable[Dict[str, Any]]] = None,
        timer: Optional[DSCAPhaseTimer] = None,
    ) -> Tuple[int, int]:
        reps = self.capture_representations(batch, timer=timer, phase="mask_capture_forward")
        if timer:
            timer.start(
                "update_concepts_and_buffers",
                {
                    "num_clusters": len(self.repository),
                    "num_active_dsams": self.repository.num_active(),
                    "pca_buffer_sizes": [int(buf.shape[0]) for buf in self.repository.pca_buffers],
                },
            )
        ids, created = self.repository.assign_batch(
            reps["h_f"],
            reps["h_v"],
            metadata=metadata,
            timer=timer,
            initialize_basis=not self.disable_basis_initialization,
        )
        if timer:
            timer.stop(
                "update_concepts_and_buffers",
                {
                    "created": created,
                    "assigned_ids": ids,
                    "num_clusters": len(self.repository),
                    "num_active_dsams": self.repository.num_active(),
                    "pca_buffer_sizes": [int(buf.shape[0]) for buf in self.repository.pca_buffers],
                },
            )
        if self.disable_pca_refine:
            activated = 0
        else:
            activated = self.repository.refine_subspaces_if_due(max(self.global_step, 1), timer=timer)
        return created, activated

    def edit(self, batch, condition=None, detach_history=False):
        created, activated = self._update_clusters_from_batch(batch)
        return self, {
            "dsca/num_clusters": float(len(self.repository)),
            "dsca/new_clusters_created": float(created),
            "dsca/new_dsams_activated": float(activated),
        }

    def dsca_collect_new_trainable_params(self):
        params = []
        for param in self.repository.parameters():
            if param.requires_grad and id(param) not in self._registered_param_ids:
                params.append(param)
                self._registered_param_ids.add(id(param))
        return params

    def dsca_register_new_params_with_optimizer(self, optimizer) -> int:
        if optimizer is None:
            self.dsca_collect_new_trainable_params()
            return 0
        existing = {id(param) for group in optimizer.param_groups for param in group["params"]}
        new_params = []
        for param in self.repository.parameters():
            if param.requires_grad and id(param) not in existing:
                new_params.append(param)
        if new_params:
            optimizer.add_param_group({"params": new_params})
        return len(new_params)

    def _base_trainable_count(self) -> float:
        return float(sum(param.numel() for param in self.model.parameters() if param.requires_grad))

    def _editor_trainable_count(self) -> float:
        return float(sum(param.numel() for param in self.outer_parameters() if param.requires_grad))

    def edit_step(self, batch: Dict[str, Any], training: bool, optimizer=None):
        self.global_step += int(training)
        timer = self._make_phase_timer(batch)
        timer.start(
            "edit_step_total",
            {
                "training": training,
                "global_step": self.global_step,
                "num_clusters": len(self.repository),
                "num_active_dsams": self.repository.num_active(),
                "dsca_min_samples": self.repository.min_samples,
                "dsca_refine_interval": self.repository.refine_interval,
                "disable_pca_refine": self.disable_pca_refine,
                "disable_basis_initialization": self.disable_basis_initialization,
            },
        )
        created = activated = new_opt_params = 0
        try:
            timer.start("set_editor_train")
            self.set_editor_train(training)
            timer.stop("set_editor_train", {"editor_trainable_params": self._editor_trainable_count()})

            if training and self.update_clusters_during_training:
                timer.start(
                    "update_clusters_from_batch",
                    {
                        "num_clusters": len(self.repository),
                        "num_active_dsams": self.repository.num_active(),
                        "whether_refine_triggered": (
                            (not self.disable_pca_refine)
                            and self.repository.refine_interval > 0
                            and max(self.global_step, 1) % self.repository.refine_interval == 0
                        ),
                    },
                )
                created, activated = self._update_clusters_from_batch(batch["edit_inner"], timer=timer)
                timer.stop(
                    "update_clusters_from_batch",
                    {
                        "created": created,
                        "activated": activated,
                        "num_clusters": len(self.repository),
                        "num_active_dsams": self.repository.num_active(),
                        "pca_buffer_sizes": [int(buf.shape[0]) for buf in self.repository.pca_buffers],
                    },
                )
                timer.start("dynamic_optimizer_add_params")
                new_opt_params = self.dsca_register_new_params_with_optimizer(optimizer)
                timer.stop(
                    "dynamic_optimizer_add_params",
                    {
                        "new_optimizer_params": new_opt_params,
                        "num_optimizer_param_groups": 0 if optimizer is None else len(optimizer.param_groups),
                    },
                )

            with torch.no_grad():
                teacher_edit = self.capture_representations(
                    batch["edit_inner"],
                    timer=timer,
                    phase="edit_teacher_forward_dsca_disabled",
                )
                replay_batch = batch.get("loc_image", batch.get("loc"))
                teacher_replay = (
                    self.capture_representations(
                        replay_batch,
                        timer=timer,
                        phase="replay_teacher_forward_dsca_disabled",
                    )
                    if replay_batch is not None
                    else None
                )

            with torch.set_grad_enabled(training):
                edit_outputs = self._forward_with_dsca(
                    batch["edit_inner"],
                    timer=timer,
                    phase="edit_forward_dsca_enabled",
                )
                timer.start("loss_task")
                raw_l_task = self._nll(edit_outputs, batch["edit_inner"])
                l_task = raw_l_task * 0.0 if self.disable_task_loss else raw_l_task
                timer.stop("loss_task", {"disabled": self.disable_task_loss})

                edited_edit_reps = self._last_reps
                timer.start("loss_align")
                raw_l_align = 1.0 - F.cosine_similarity(edited_edit_reps["h_f"], teacher_edit["h_t"].detach(), dim=-1).mean()
                l_align = raw_l_align * 0.0 if self.disable_align else raw_l_align
                timer.stop("loss_align", {"disabled": self.disable_align})

                if replay_batch is not None and teacher_replay is not None:
                    replay_outputs = self._forward_with_dsca(
                        replay_batch,
                        timer=timer,
                        phase="replay_forward_dsca_enabled",
                    )
                    replay_reps = self._last_reps
                    replay_weights = self._last_route_weights
                    timer.start("loss_cdistill")
                    raw_l_cdistill = dsca_contrastive_distill_loss(
                        replay_reps["h_f"], teacher_replay["h_f"].detach(), self.distill_temperature
                    )
                    l_cdistill = raw_l_cdistill * 0.0 if self.disable_cdistill else raw_l_cdistill
                    timer.stop("loss_cdistill", {"disabled": self.disable_cdistill})
                    timer.start("loss_sparse")
                    raw_l_sparse = dsca_sparse_loss(
                        replay_weights if replay_weights is not None else torch.empty(0, device=l_task.device)
                    )
                    l_sparse = raw_l_sparse * 0.0 if self.disable_sparse else raw_l_sparse
                    timer.stop("loss_sparse", {"disabled": self.disable_sparse})
                else:
                    replay_outputs = None
                    l_cdistill = l_task * 0.0
                    l_sparse = l_task * 0.0

                timer.start("loss_total_assemble")
                l_total = (
                    float(_cfg(self.config, "dsca_task_weight", 1.0)) * l_task
                    + float(_cfg(self.config, "dsca_lambda_align", 0.5)) * l_align
                    + float(_cfg(self.config, "dsca_lambda_distill", 1.0)) * l_cdistill
                    + float(_cfg(self.config, "dsca_lambda_sparse", 1.0e-2)) * l_sparse
                    + self._optimizer_anchor.sum() * 0.0
                )
                timer.stop("loss_total_assemble")

            if training:
                timer.start("backward", {"outer_parameter_count": len(list(self.outer_parameters()))})
                safe_backward(l_total, self.outer_parameters(), int(_cfg(self.config, "accumulate_bs", 1)), allow_unused=True)
                timer.stop("backward")
                if bool(_cfg(self.config, "dsca_debug", False)):
                    timer.start("base_freeze_assert")
                    if bool(_cfg(self.config, "dsca_freeze_vlm", True)) and self._base_trainable_count() != 0.0:
                        raise RuntimeError("DSCA expected frozen base VLM parameters but found trainable base parameters.")
                    timer.stop("base_freeze_assert")
                    timer.start("R_k_requires_grad_scan")
                    for idx, dsam in enumerate(self.repository.dsams):
                        if dsam.active and dsam.R.requires_grad:
                            raise RuntimeError(f"DSCA basis R_{idx} must be non-trainable.")
                    timer.stop("R_k_requires_grad_scan")

            timer.start("scalar_item_logging")
            info = dict(self._last_info)
            info.update(
                {
                    "loss/dsca_total": float(l_total.detach().cpu()),
                    "loss/dsca_task": float(l_task.detach().cpu()),
                    "loss/dsca_align": float(l_align.detach().cpu()),
                    "loss/dsca_cdistill": float(l_cdistill.detach().cpu()),
                    "loss/dsca_sparse": float(l_sparse.detach().cpu()),
                    "loss/total": float(l_total.detach().cpu()),
                    "loss/edit": float(l_task.detach().cpu()),
                    "loss/loc": float(l_cdistill.detach().cpu()),
                    "dsca/num_clusters": float(len(self.repository)),
                    "dsca/num_active_dsams": float(self.repository.num_active()),
                    "dsca/mean_subspace_overlap": float(self.repository.mean_subspace_overlap().detach().cpu()),
                    "dsca/base_vlm_trainable_params": self._base_trainable_count()
                    if bool(_cfg(self.config, "dsca_debug", False))
                    else 0.0,
                    "dsca/editor_trainable_params": self._editor_trainable_count(),
                    "dsca/new_clusters_created": float(created),
                    "dsca/new_dsams_activated": float(activated),
                    "dsca/new_optimizer_params": float(new_opt_params),
                }
            )
            timer.stop("scalar_item_logging")
            return l_total, l_task, l_cdistill, l_sparse, info
        finally:
            timer.stop(
                "edit_step_total",
                {
                    "num_clusters": len(self.repository),
                    "num_active_dsams": self.repository.num_active(),
                    "pca_buffer_sizes": [int(buf.shape[0]) for buf in self.repository.pca_buffers],
                },
            )


def sanitize_dsca_metadata(requests: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for request in requests:
        yield {
            "prompt": request.get("prompt"),
            "target": request.get("target", request.get("target_new")),
            "rephrase_prompt": request.get("rephrase_prompt"),
            "locality_prompt": request.get("locality_prompt"),
        }
