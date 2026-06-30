from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_edit_modules import (
    TIMECPResidual,
    TIMEExpertRepository,
    TIMEForwardDebug,
    extract_first_tensor,
    find_module_by_path,
    replace_first_tensor,
    time_memory_estimate,
)
from .editable_model import EditableModel
from ..losses import kl_loc_loss
from ..utils import safe_backward


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _tensor_logits(outputs: Any) -> torch.Tensor:
    return outputs if isinstance(outputs, torch.Tensor) else outputs.logits


def _answer_labels_for_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] > labels.shape[1]:
        return labels
    return labels[:, -logits.shape[1] - 1 :]


def _to_float(value: Optional[torch.Tensor]) -> float:
    if value is None:
        return 0.0
    return float(value.detach().float().cpu())


@dataclass
class TIMEInterventionContext:
    batch: Optional[Dict[str, Any]] = None
    enabled: bool = True
    disable_time: bool = False
    force_expert_ids: List[int] = field(default_factory=list)
    call_label: str = "forward"
    debug_events: Optional[List[Dict[str, Any]]] = None


class TIMEEdit(EditableModel):
    def __init__(self, model, config, model_constructor):
        super().__init__(model, config, model_constructor)
        self.hidden_size = self._infer_hidden_size()
        self.target_layer = int(_cfg(self.config, "time_target_layer", 21))
        self.time_layer_module = _cfg(self.config, "time_layer_module", None)
        self.repository = TIMEExpertRepository(
            hidden_size=self.hidden_size,
            rank=int(_cfg(self.config, "time_rank", 4)),
            init_std=float(_cfg(self.config, "time_init_std", 1.0e-3)),
            target_layer=self.target_layer,
            alpha=float(_cfg(self.config, "time_alpha", 0.1)),
            gamma=float(_cfg(self.config, "time_gamma", 0.5)),
            tau=float(_cfg(self.config, "time_tau", 1.0)),
            scale_mode=str(_cfg(self.config, "time_scale_mode", "lora_like")),
            activation=str(_cfg(self.config, "time_activation", "gelu")),
        )
        self.time_residual = TIMECPResidual(
            self.repository,
            disable_selection=bool(_cfg(self.config, "time_disable_selection", False)),
            disable_score_mixing=bool(_cfg(self.config, "time_disable_score_mixing", False)),
            topk=int(_cfg(self.config, "time_topk", 0) or 0),
            routing_mode=str(_cfg(self.config, "time_routing_mode", "threshold")),
            residual_sign=str(_cfg(self.config, "time_residual_sign", "plus")),
            expert_gain=float(_cfg(self.config, "time_expert_gain", 1.0)),
            score_norm=str(_cfg(self.config, "time_score_norm", "none")),
            relative_threshold=_cfg(self.config, "time_relative_threshold", None),
            mixing_mode=str(_cfg(self.config, "time_mixing_mode", "softmax")),
        )
        self._optimizer_anchor = nn.Parameter(torch.zeros(1))
        self._time_context: Optional[TIMEInterventionContext] = None
        self._last_debug: Optional[TIMEForwardDebug] = None
        self._last_token_mask: Optional[torch.Tensor] = None
        self._last_info: Dict[str, float] = {}
        self.global_step = 0
        self.current_expert_index: Optional[int] = None
        self._install_hook()
        if bool(_cfg(self.config, "time_freeze_vlm", True)):
            self.freeze_base_vlm()
        repo_path = _cfg(self.config, "time_repository_path", None)
        if repo_path:
            loaded = TIMEExpertRepository.load(repo_path)
            self.repository.load_state_bundle(loaded.state_bundle())

    def _infer_hidden_size(self) -> int:
        if hasattr(self.model, "opt_model"):
            return int(self.model.opt_model.config.hidden_size)
        if hasattr(self.model, "llama_model"):
            return int(self.model.llama_model.config.hidden_size)
        if hasattr(self.model, "llava_model"):
            return int(self.model.llava_model.config.hidden_size)
        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)
        raise NotImplementedError(f"TIME does not know how to infer hidden size for {type(self.model)}.")

    def _layer_path(self) -> str:
        if self.time_layer_module:
            return str(self.time_layer_module)
        model_name = str(self.config.model_name).lower()
        if "blip2" in model_name:
            return f"opt_model.model.decoder.layers.{self.target_layer}"
        if "minigpt4" in model_name:
            return f"llama_model.model.layers.{self.target_layer}"
        if model_name in {"llava-med", "llava_med"}:
            return f"llava_model.model.layers.{self.target_layer}"
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return f"model.layers.{self.target_layer}"
        raise NotImplementedError(
            f"TIME supports BLIP2, MiniGPT-4, and verified LLaVA-Med layer paths by default, got {self.config.model_name}."
        )

    def _install_hook(self) -> None:
        layer_path = self._layer_path()
        layer = find_module_by_path(self.model, layer_path)
        self._hook_handle = layer.register_forward_hook(self._time_hook)
        self.time_layer_path = layer_path

    def remove_hook(self) -> None:
        if hasattr(self, "_hook_handle") and self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def freeze_base_vlm(self) -> None:
        for name, param in self.model.named_parameters():
            if not name.startswith("repository.") and "time" not in name:
                param.requires_grad_(False)
        self.model.eval()

    def set_editor_train(self, training: bool) -> None:
        self.model.eval()
        self.repository.freeze_all()
        if training and self.current_expert_index is not None:
            self.repository.unfreeze_expert(int(self.current_expert_index))
        self._optimizer_anchor.requires_grad_(False)

    def outer_parameters(self):
        params = self.repository.trainable_parameters()
        return params if params else [self._optimizer_anchor]

    def add_expert(self, record_id: Any, metadata: Optional[Dict[str, Any]] = None) -> int:
        device = next(self.parameters()).device
        self.repository.freeze_all()
        index = self.repository.add_expert(record_id=record_id, metadata=metadata, device=device)
        self.current_expert_index = index
        self.repository.unfreeze_expert(index)
        return index

    def delete_last_expert(self) -> Optional[Dict[str, Any]]:
        deleted = self.repository.delete_last_expert()
        self.current_expert_index = self.repository.num_experts - 1 if self.repository.num_experts else None
        return deleted

    @contextmanager
    def time_disabled(self):
        old = self._time_context
        self._time_context = TIMEInterventionContext(enabled=False, disable_time=True)
        try:
            yield
        finally:
            self._time_context = old

    @contextmanager
    def time_intervention(self, context: TIMEInterventionContext):
        old = self._time_context
        self._time_context = context
        try:
            yield
        finally:
            self._time_context = old

    def _token_mask(self, hidden: torch.Tensor, batch: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        scope = str(_cfg(self.config, "time_token_scope", "all")).lower()
        if scope == "all":
            attention = None if batch is None else batch.get("attention_mask")
            if torch.is_tensor(attention) and tuple(attention.shape) == tuple(hidden.shape[:2]):
                return attention.to(hidden.device).bool()
            return torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
        if scope == "last":
            mask = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
            mask[:, -1] = True
            return mask
        if scope == "answer_mask":
            answer = None if batch is None else batch.get("answer_mask")
            if torch.is_tensor(answer) and tuple(answer.shape) == tuple(hidden.shape[:2]) and bool(answer.any()):
                return answer.to(hidden.device).bool()
            return torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
        raise ValueError(f"Unsupported TIME token_scope: {scope}")

    def _time_hook(self, module: nn.Module, args: Tuple[Any, ...], output: Any) -> Any:
        context = self._time_context
        if context is None or not context.enabled:
            return output
        hidden = extract_first_tensor(output)
        token_mask = self._token_mask(hidden, context.batch)
        residual, debug = self.time_residual(
            hidden,
            token_mask=token_mask,
            disable_time=context.disable_time,
            force_expert_ids=context.force_expert_ids,
            return_debug=True,
        )
        self._last_debug = debug
        self._last_token_mask = token_mask.detach() if torch.is_tensor(token_mask) else None
        if context.debug_events is not None:
            context.debug_events.append(self._debug_event(context, hidden, debug))
        if torch.count_nonzero(residual).item() == 0:
            return output
        return replace_first_tensor(output, hidden + residual)

    def _debug_event(self, context: TIMEInterventionContext, hidden: torch.Tensor, debug: TIMEForwardDebug) -> Dict[str, Any]:
        pooled_scores = self._pooled_scores(debug.scores, self._last_token_mask).detach().float().cpu()
        pooled_raw_scores = self._pooled_scores(debug.raw_scores, self._last_token_mask).detach().float().cpu()
        pooled_variants = {
            name: self._pooled_scores(value, self._last_token_mask).detach().float().cpu()
            for name, value in debug.score_variants.items()
        }
        pooled_selected = self._pooled_selected(debug.selected, self._last_token_mask).detach().cpu()
        route_weights = self._pooled_scores(debug.weights, self._last_token_mask).detach().float().cpu()
        top_score, top_id = (pooled_scores.max(dim=-1) if pooled_scores.numel() else (torch.tensor([]), torch.tensor([])))
        row_scores = pooled_scores[0].tolist() if pooled_scores.numel() else []
        row_raw_scores = pooled_raw_scores[0].tolist() if pooled_raw_scores.numel() else []
        row_variants = {
            name: values[0].tolist() if values.numel() else []
            for name, values in pooled_variants.items()
        }
        row_selected = pooled_selected[0].tolist() if pooled_selected.numel() else []
        row_weights = route_weights[0].tolist() if route_weights.numel() else []
        return {
            "call_label": context.call_label,
            "layer_path": getattr(self, "time_layer_path", None),
            "hidden_shape": list(hidden.shape),
            "repository_num_experts": self.repository.num_experts,
            "current_expert_index": self.current_expert_index,
            "top_expert_id": int(top_id[0].item()) if top_id.numel() else None,
            "top_score": float(top_score[0].item()) if top_score.numel() else None,
            "selected_expert_ids": [idx for idx, flag in enumerate(row_selected) if bool(flag)],
            "selected_expert_set_size": int(sum(1 for flag in row_selected if bool(flag))),
            "pooled_scores": row_scores,
            "raw_pooled_scores": row_raw_scores,
            "score_variant_pooled_scores": row_variants,
            "pooled_weights": row_weights,
            "residual_norm": float(debug.residual.detach().float().norm().cpu()),
            "target_layer_hidden_delta_norm": float(debug.residual.detach().float().norm().cpu()),
            "target_layer_hidden_changed": bool(torch.count_nonzero(debug.residual).item() > 0),
            "token_scope": str(_cfg(self.config, "time_token_scope", "all")),
            "scale_mode": self.repository.scale_mode,
            "gamma": float(self.repository.gamma),
            "topk": int(self.time_residual.topk),
            "routing_mode": str(self.time_residual.routing_mode),
            "score_norm": str(self.time_residual.score_norm),
            "relative_threshold": self.time_residual.relative_threshold,
            "mixing_mode": str(self.time_residual.mixing_mode),
            "residual_sign": str(self.time_residual.residual_sign),
            "expert_gain": float(self.time_residual.expert_gain),
            "force_expert_ids": list(context.force_expert_ids),
        }

    @staticmethod
    def _pooled_scores(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if scores.numel() == 0:
            return scores.reshape(scores.shape[0], 0)
        if mask is None:
            return scores.mean(dim=1)
        mask = mask.to(scores.device).bool()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(scores.dtype)
        return (scores * mask.unsqueeze(-1).to(scores.dtype)).sum(dim=1) / denom

    @staticmethod
    def _pooled_selected(selected: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if selected.numel() == 0:
            return selected.reshape(selected.shape[0], 0)
        if mask is None:
            return selected.any(dim=1)
        mask = mask.to(selected.device).bool()
        return (selected & mask.unsqueeze(-1)).any(dim=1)

    def _forward_with_time(
        self,
        batch: Dict[str, Any],
        call_label: str = "forward",
        force_current: bool = False,
        debug_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        force_ids = []
        force_current = force_current or str(_cfg(self.config, "time_routing_mode", "threshold")).lower() == "force_current"
        if force_current and self.current_expert_index is not None:
            force_ids = [int(self.current_expert_index)]
        context = TIMEInterventionContext(batch=batch, force_expert_ids=force_ids, call_label=call_label, debug_events=debug_events)
        with self.time_intervention(context):
            return self.model(batch)

    def forward(self, *inputs, **kwargs):
        if len(inputs) != 1 or not isinstance(inputs[0], dict) or kwargs:
            return self.model(*inputs, **kwargs)
        return self._forward_with_time(inputs[0])

    def _nll(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        logits = _tensor_logits(outputs)
        labels = batch["labels"]
        return self.edit_loss_fn(self.config, logits, _answer_labels_for_logits(logits, labels))["nll"]

    @staticmethod
    def _same_batch_content(lhs: Optional[Dict[str, Any]], rhs: Optional[Dict[str, Any]]) -> bool:
        if lhs is None or rhs is None:
            return False
        for key in ("text_input", "prompt", "target"):
            left = lhs.get(key)
            right = rhs.get(key)
            if left is not None and right is not None and left != right:
                return False
        return True

    def _reference_kl(self, reference_batch: Optional[Dict[str, Any]]) -> torch.Tensor:
        if reference_batch is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        with torch.no_grad(), self.time_disabled():
            base_outputs = self.model(reference_batch)
            base_logits = _tensor_logits(base_outputs)
        post_outputs = self._forward_with_time(reference_batch, call_label="locality")
        post_logits = _tensor_logits(post_outputs)
        mask = getattr(post_outputs, "attention_mask", None)
        if mask is None and isinstance(reference_batch, dict):
            mask = reference_batch.get("attention_mask")
        if mask is None:
            mask = torch.ones(post_logits.shape[:2], device=post_logits.device)
        return kl_loc_loss(base_logits.detach(), post_logits, mask=mask)

    def _alignment_loss(self, current_index: Optional[int]) -> torch.Tensor:
        if bool(_cfg(self.config, "time_disable_align_loss", False)) or current_index is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        if self._last_debug is None or self.repository.num_experts <= 1:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        previous_ids = [idx for idx in range(self.repository.num_experts) if idx != int(current_index)]
        if not previous_ids:
            return torch.tensor(0.0, device=self._last_debug.scores.device)
        max_neg = int(_cfg(self.config, "time_negative_experts", 8))
        if max_neg > 0:
            previous_ids = previous_ids[-max_neg:]
        ids = [int(current_index)] + previous_ids
        pooled = self._pooled_scores(self._last_debug.scores, self._last_token_mask)
        logits = pooled[:, ids] / max(float(_cfg(self.config, "time_tau", 1.0)), 1.0e-8)
        targets = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits, targets)

    def edit(self, batch, condition=None, detach_history=False):
        record_id = "unknown"
        if isinstance(condition, dict):
            record_id = condition.get("record_id", condition.get("id", record_id))
        elif isinstance(batch, dict):
            record_id = batch.get("record_id", record_id)
        index = self.add_expert(record_id=record_id)
        return self, {"time/current_expert_index": float(index), "time/num_experts": float(self.repository.num_experts)}

    def edit_step(self, batch: Dict[str, Any], training: bool, optimizer=None):
        self.global_step += int(training)
        if self.current_expert_index is None:
            self.add_expert(record_id=batch.get("record_id", "unknown") if isinstance(batch, dict) else "unknown")
        self.set_editor_train(training)
        edit_batch = batch.get("edit_inner") or batch.get("edit_outer") or batch
        gen_batch = batch.get("edit_outer")
        reference_batch = batch.get("loc") or batch.get("loc_image")
        force_current = bool(training and _cfg(self.config, "time_force_current_during_training", True))

        with torch.set_grad_enabled(training):
            edit_outputs = self._forward_with_time(edit_batch, call_label="edit", force_current=force_current)
            l_rel = self._nll(edit_outputs, edit_batch)

            l_gen = l_rel * 0.0
            gen_skipped = True
            if (
                gen_batch is not None
                and gen_batch is not edit_batch
                and not self._same_batch_content(gen_batch, edit_batch)
                and float(_cfg(self.config, "time_lambda_gen", 1.0)) != 0.0
            ):
                gen_outputs = self._forward_with_time(gen_batch, call_label="generality", force_current=force_current)
                l_gen = self._nll(gen_outputs, gen_batch)
                gen_skipped = False

            l_loc = l_rel * 0.0
            if reference_batch is not None and float(_cfg(self.config, "time_lambda_loc", 1.0)) != 0.0:
                l_loc = self._reference_kl(reference_batch)

            l_align = self._alignment_loss(self.current_expert_index)
            l_total = (
                float(_cfg(self.config, "time_lambda_rel", 1.0)) * l_rel
                + float(_cfg(self.config, "time_lambda_gen", 1.0)) * l_gen
                + float(_cfg(self.config, "time_lambda_loc", 1.0)) * l_loc
                + float(_cfg(self.config, "time_lambda_align", 0.5)) * l_align
            )
            l_total = l_total + sum((param.sum() * 0.0 for param in self.outer_parameters()), l_total * 0.0)

        if training:
            safe_backward(l_total, self.outer_parameters(), int(_cfg(self.config, "accumulate_bs", 1)), allow_unused=True)

        routing = self.routing_summary()
        nonfinite = 0
        for param in self.outer_parameters():
            if param.grad is not None:
                nonfinite += int((~torch.isfinite(param.grad)).sum().detach().cpu().item())
        info = {
            "loss/time_total": _to_float(l_total),
            "loss/time_rel": _to_float(l_rel),
            "loss/time_gen": _to_float(l_gen),
            "loss/time_loc": _to_float(l_loc),
            "loss/time_align": _to_float(l_align),
            "loss/total": _to_float(l_total),
            "loss/edit": _to_float(l_rel),
            "loss/loc": _to_float(l_loc),
            "time/current_expert_index": float(self.current_expert_index if self.current_expert_index is not None else -1),
            "time/num_experts": float(self.repository.num_experts),
            "time/gen_loss_skipped": float(gen_skipped),
            "time/top_expert_id": float(routing.get("top_expert_id", -1) if routing.get("top_expert_id") is not None else -1),
            "time/top_score": float(routing.get("top_score", 0.0) or 0.0),
            "time/selected_expert_set_size": float(routing.get("selected_expert_set_size", 0) or 0),
            "time/nan_inf_grad_count": float(nonfinite),
            "time/trainable_params": float(sum(param.numel() for param in self.outer_parameters() if param.requires_grad)),
            "time/force_current_train": float(force_current),
            "time/routing_mode_is_force_current": float(str(self.time_residual.routing_mode) == "force_current"),
            "time/score_norm_is_none": float(str(self.time_residual.score_norm) == "none"),
            "time/residual_sign_is_minus": float(str(self.time_residual.residual_sign) == "minus"),
            "time/expert_gain": float(self.time_residual.expert_gain),
            "time/reliability_only": float(bool(_cfg(self.config, "time_reliability_only", False))),
        }
        self._last_info = info
        return l_total, l_rel, l_loc, torch.tensor(0.0, device=l_total.device), info

    def routing_summary(self) -> Dict[str, Any]:
        if self._last_debug is None:
            return {
                "top_expert_id": None,
                "top_score": None,
                "selected_expert_ids": [],
                "selected_expert_set_size": 0,
                "pooled_scores": [],
                "pooled_weights": [],
                "residual_norm": 0.0,
            }
        debug = self._last_debug
        pooled_scores = self._pooled_scores(debug.scores, self._last_token_mask).detach().float().cpu()
        pooled_raw_scores = self._pooled_scores(debug.raw_scores, self._last_token_mask).detach().float().cpu()
        pooled_variants = {
            name: self._pooled_scores(value, self._last_token_mask).detach().float().cpu()
            for name, value in debug.score_variants.items()
        }
        pooled_selected = self._pooled_selected(debug.selected, self._last_token_mask).detach().cpu()
        pooled_weights = self._pooled_scores(debug.weights, self._last_token_mask).detach().float().cpu()
        if pooled_scores.numel() == 0:
            top_id = None
            top_score = None
            scores = []
            raw_scores = []
            score_variants = {}
            weights = []
            selected_ids: List[int] = []
        else:
            top = pooled_scores[0].max(dim=-1)
            top_score = float(top.values.item())
            top_id = int(top.indices.item())
            scores = pooled_scores[0].tolist()
            raw_scores = pooled_raw_scores[0].tolist()
            score_variants = {
                name: values[0].tolist() if values.numel() else []
                for name, values in pooled_variants.items()
            }
            weights = pooled_weights[0].tolist()
            selected_ids = [idx for idx, flag in enumerate(pooled_selected[0].tolist()) if bool(flag)]
        return {
            "top_expert_id": top_id,
            "top_score": top_score,
            "selected_expert_ids": selected_ids,
            "selected_expert_set_size": len(selected_ids),
            "pooled_scores": scores,
            "raw_pooled_scores": raw_scores,
            "score_variant_pooled_scores": score_variants,
            "pooled_weights": weights,
            "residual_norm": float(debug.residual.detach().float().norm().cpu()),
            "target_layer_hidden_delta_norm": float(debug.residual.detach().float().norm().cpu()),
            "target_layer_hidden_changed": bool(torch.count_nonzero(debug.residual).item() > 0),
            "gamma": float(self.repository.gamma),
            "topk": int(self.time_residual.topk),
            "routing_mode": str(self.time_residual.routing_mode),
            "score_norm": str(self.time_residual.score_norm),
            "relative_threshold": self.time_residual.relative_threshold,
            "mixing_mode": str(self.time_residual.mixing_mode),
            "residual_sign": str(self.time_residual.residual_sign),
            "expert_gain": float(self.time_residual.expert_gain),
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return {
            f"{prefix}time_repository": self.repository.state_bundle(),
            f"{prefix}time_global_step": torch.tensor(self.global_step),
            f"{prefix}time_config": deepcopy(self.config),
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        state = dict(state_dict)
        bundle = state.get("time_repository", state.get("model.time_repository"))
        if bundle is not None:
            device = next(self.parameters()).device
            self.repository.load_state_bundle(bundle, device=device)
            self.current_expert_index = self.repository.num_experts - 1 if self.repository.num_experts else None
        step = state.get("time_global_step", state.get("model.time_global_step"))
        if step is not None:
            self.global_step = int(step.item() if torch.is_tensor(step) else step)
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def save_time_state(self, output_dir: str | Path) -> Dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        repo_path = output / "expert_repository.pt"
        self.repository.save(repo_path)
        summary = {
            "repository_path": str(repo_path),
            "num_experts": self.repository.num_experts,
            "hidden_size": self.repository.hidden_size,
            "s1": self.repository.s1,
            "s2": self.repository.s2,
            "rank": self.repository.rank,
            "scale_mode": self.repository.scale_mode,
            "activation": self.repository.activation,
            "metadata": list(self.repository.metadata),
            "memory_estimate": time_memory_estimate(
                self.repository.hidden_size,
                self.repository.rank,
                self.repository.s1,
                self.repository.s2,
            ),
        }
        (output / "time_repository_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary
