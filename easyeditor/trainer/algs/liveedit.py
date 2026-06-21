import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .editable_model import EditableModel
from .liveedit_utils import (
    ExpertGenerator,
    ExpertRepository,
    LiveEditContext,
    LiveEditFeatureExtractor,
    apply_liveedit_residual_to_output,
    find_module_by_path,
    first_hidden_from_output,
    get_liveedit_masks,
    get_liveedit_routing_masks,
    hard_route,
    liveedit_routing_losses,
    low_rank_residual,
    soft_routing_weights,
    temporary_context,
)
from ..losses import kl_loc_loss
from ..utils import safe_backward

LOG = logging.getLogger(__name__)


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _tensor_logits(outputs: Any) -> torch.Tensor:
    return outputs if isinstance(outputs, torch.Tensor) else outputs.logits


def _attention_mask(outputs: Any, logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(outputs, torch.Tensor) and getattr(outputs, "attention_mask", None) is not None:
        return outputs.attention_mask
    return torch.ones(logits.shape[:2], device=logits.device, dtype=logits.dtype)


def _target_mask(outputs: Any, batch: Dict[str, Any], logits: torch.Tensor) -> torch.Tensor:
    mask = None
    if not isinstance(outputs, torch.Tensor):
        mask = getattr(outputs, "answer_mask", None)
    if mask is None:
        mask = batch.get("answer_mask")
    if mask is None and "labels" in batch:
        labels = batch["labels"]
        if labels.dim() == 2:
            mask = labels.ne(-100)
    if mask is None:
        raise RuntimeError("LiveEdit locality KL requires an answer/target mask.")
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    if mask.dim() != 2:
        raise RuntimeError(f"LiveEdit target mask must be [batch, seq], got {tuple(mask.shape)}.")
    if mask.shape[1] != logits.shape[1]:
        if mask.shape[1] > logits.shape[1]:
            mask = mask[:, -logits.shape[1] :]
        else:
            raise RuntimeError(
                f"LiveEdit target mask length {mask.shape[1]} is shorter than logits length {logits.shape[1]}."
            )
    if mask.sum() == 0:
        raise RuntimeError("LiveEdit locality KL target mask has no selected tokens.")
    return mask


class LiveEdit(EditableModel):
    """LiveEdit low-rank mixture-of-experts editor for multimodal VLLMs."""

    def __init__(self, model, config, model_constructor):
        super().__init__(model, config, model_constructor)
        model_name = str(self.config.model_name).lower()
        if model_name == "llava":
            raise NotImplementedError("LiveEdit LLaVA support is deferred until reliable region masks are implemented.")
        self.hidden_size = self._infer_hidden_size()
        self.liveedit_layer = int(_cfg(self.config, "liveedit_layer", 21))
        self.module_dim = int(_cfg(self.config, "liveedit_module_dim", 1024))
        self.feature_k = int(_cfg(self.config, "liveedit_feature_k", 4))
        self.rank = int(_cfg(self.config, "liveedit_rank", 4))
        self.lora_scale = float(_cfg(self.config, "liveedit_lora_scale", 5.0))
        self.cross_att_heads = int(_cfg(self.config, "liveedit_cross_att_heads", 8))
        self.similarity = str(_cfg(self.config, "liveedit_similarity", "inner_product"))
        self.hard_topk = _cfg(self.config, "liveedit_hard_topk", None)
        self.force_topk_when_empty = bool(_cfg(self.config, "liveedit_force_topk_when_empty", False))
        sentinel_tokens = int(_cfg(self.config, "liveedit_sentinel_tokens", 32))

        self.expert_generator = ExpertGenerator(
            self.hidden_size,
            self.rank,
            self.module_dim,
            self.cross_att_heads,
            self.lora_scale,
        )
        self.edit_feature_extractor = LiveEditFeatureExtractor(
            self.hidden_size,
            self.feature_k,
            self.module_dim,
            self.cross_att_heads,
        )
        self.input_feature_extractor = LiveEditFeatureExtractor(
            self.hidden_size,
            self.feature_k,
            self.module_dim,
            self.cross_att_heads,
            sentinel_tokens=sentinel_tokens,
            with_sentinel=True,
        )
        self.instant_reps_norm = nn.LayerNorm(self.hidden_size)
        self.repository = ExpertRepository(self.rank, self.hidden_size, self.feature_k, self.module_dim)
        self._liveedit_context: Optional[LiveEditContext] = None
        self._last_capture: Optional[torch.Tensor] = None
        self._last_info: Dict[str, float] = {}
        self._install_hook()
        if bool(_cfg(self.config, "liveedit_freeze_vllm", True)):
            self.freeze_base_vllm()
        repo_path = _cfg(self.config, "liveedit_repository_path", None)
        if repo_path:
            loaded = ExpertRepository.load(repo_path)
            self.repository.load_state_dict(loaded.state_dict())

    def _infer_hidden_size(self) -> int:
        if hasattr(self.model, "opt_model"):
            return int(self.model.opt_model.config.hidden_size)
        if hasattr(self.model, "llama_model"):
            return int(self.model.llama_model.config.hidden_size)
        raise NotImplementedError(f"LiveEdit does not know how to infer hidden size for {type(self.model)}.")

    def _layer_path(self) -> str:
        model_name = str(self.config.model_name).lower()
        if "blip2" in model_name:
            return f"opt_model.model.decoder.layers.{self.liveedit_layer}"
        if "minigpt4" in model_name:
            return f"llama_model.model.layers.{self.liveedit_layer}"
        raise NotImplementedError(f"LiveEdit supports BLIP2 and MiniGPT-4 only, got {self.config.model_name}.")

    def _install_hook(self) -> None:
        layer_path = self._layer_path()
        layer = find_module_by_path(self.model, layer_path)
        if not hasattr(self, "_hook_handle"):
            self._hook_handle = layer.register_forward_hook(self._liveedit_hook)
        self.liveedit_layer_path = layer_path
        LOG.info("LiveEdit hooked %s", layer_path)

    def freeze_base_vllm(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    def set_editor_train(self, training: bool) -> None:
        self.model.eval()
        self.expert_generator.train(training)
        self.edit_feature_extractor.train(training)
        self.input_feature_extractor.train(training)
        self.instant_reps_norm.train(training)
        for module in (
            self.expert_generator,
            self.edit_feature_extractor,
            self.input_feature_extractor,
            self.instant_reps_norm,
        ):
            for param in module.parameters():
                param.requires_grad_(training)

    def outer_parameters(self):
        return list(self.expert_generator.parameters()) + list(self.edit_feature_extractor.parameters()) + list(
            self.input_feature_extractor.parameters()
        ) + list(self.instant_reps_norm.parameters())

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in list(state.keys()):
            if key.startswith(f"{prefix}model."):
                del state[key]
        state[f"{prefix}model_config"] = getattr(self.model, "config", None)
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = dict(state_dict)
        state_dict.pop("model_config", None)
        result = super().load_state_dict(state_dict, strict=False)
        unexpected = [key for key in result.unexpected_keys if not key.startswith("model.")]
        missing = [key for key in result.missing_keys if not key.startswith("model.")]
        if unexpected or (strict and missing):
            raise RuntimeError(f"LiveEdit checkpoint mismatch. missing={missing}, unexpected={unexpected}")
        return result

    def _liveedit_hook(self, module: nn.Module, args: Tuple[Any, ...], output: Any) -> Any:
        context = self._liveedit_context
        if context is None or not context.enabled:
            return output
        hidden = first_hidden_from_output(output)
        if context.capture_only:
            self._last_capture = hidden
            return output
        if context.u is None or context.u.shape[0] == 0:
            return output
        residual, info = self._compute_residual(hidden, context)
        self._last_info = info
        if residual is None:
            return output
        return apply_liveedit_residual_to_output(output, residual)

    def _compute_residual(self, hidden: torch.Tensor, context: LiveEditContext) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        vision_mask, prompt_mask, _ = get_liveedit_routing_masks(context.batch, hidden)
        if vision_mask.sum() == 0:
            return None, {"liveedit/num_selected_experts_mean": 0.0}
        u = context.u.to(hidden.device, hidden.dtype)
        v = context.v.to(hidden.device, hidden.dtype)
        phi_hat = context.phi.to(hidden.device, hidden.dtype)
        psi_hat = context.psi.to(hidden.device, hidden.dtype)
        if context.force_all_experts:
            selected = torch.ones(hidden.shape[0], u.shape[0], dtype=torch.bool, device=hidden.device)
            visual_sim = torch.zeros(hidden.shape[0], u.shape[0], dtype=hidden.dtype, device=hidden.device)
            threshold = torch.zeros(hidden.shape[0], 1, dtype=hidden.dtype, device=hidden.device)
        else:
            phi_bar, psi_bar = self.input_feature_extractor.extract(hidden, vision_mask, prompt_mask)
            phi_theta = self.input_feature_extractor.extract_sentinel(hidden, prompt_mask)
            selected, visual_sim, threshold = hard_route(
                phi_bar,
                phi_hat,
                phi_theta,
                similarity=self.similarity,
                hard_topk=self.hard_topk,
                force_topk_when_empty=self.force_topk_when_empty,
            )
        if context.force_all_experts:
            _, psi_bar = self.input_feature_extractor.extract(hidden, vision_mask, prompt_mask)
        weights, prompt_sim = soft_routing_weights(psi_bar, psi_hat, selected, self.similarity)
        residual = low_rank_residual(self.instant_reps_norm(hidden), u, v, weights)
        selected_counts = selected.sum(dim=1).float()
        nonzero_weights = weights[weights > 0]
        info = {
            "liveedit/num_selected_experts_mean": float(selected_counts.mean().detach().cpu()),
            "liveedit/num_selected_experts_max": float(selected_counts.max().detach().cpu()) if selected_counts.numel() else 0.0,
            "liveedit/hard_visual_sim_mean": float(visual_sim.mean().detach().cpu()) if visual_sim.numel() else 0.0,
            "liveedit/sentinel_threshold_mean": float(threshold.mean().detach().cpu()) if threshold.numel() else 0.0,
            "liveedit/soft_weight_mean": float(nonzero_weights.mean().detach().cpu()) if nonzero_weights.numel() else 0.0,
            "liveedit/soft_weight_max": float(nonzero_weights.max().detach().cpu()) if nonzero_weights.numel() else 0.0,
            "liveedit/residual_norm_mean": float(residual.norm(dim=-1).mean().detach().cpu()),
        }
        return residual, info

    def capture_hidden(self, batch: Dict[str, Any]) -> torch.Tensor:
        self._last_capture = None
        with torch.no_grad(), temporary_context(self, LiveEditContext(batch=batch, capture_only=True)):
            self.model(batch)
        if self._last_capture is None:
            raise RuntimeError("LiveEdit failed to capture layer representation.")
        return self._last_capture.detach()

    def generate_experts(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.capture_hidden(batch)
        vision_mask, prompt_mask, answer_mask, attention_mask = get_liveedit_masks(batch, hidden)
        edit_signal_mask = (vision_mask | prompt_mask | answer_mask) & attention_mask
        u, v = self.expert_generator(hidden, edit_signal_mask)
        phi, psi = self.edit_feature_extractor.extract(hidden, vision_mask, prompt_mask)
        return u, v, phi, psi, hidden

    def append_from_batch(self, batch: Dict[str, Any], metadata: Optional[Iterable[Dict[str, Any]]] = None) -> None:
        with torch.no_grad():
            u, v, phi, psi, _ = self.generate_experts(batch)
        before = len(self.repository)
        self.repository.append(u, v, phi, psi, metadata=metadata, detach=True)
        after = len(self.repository)
        expected = before + u.shape[0]
        if after != expected:
            raise RuntimeError(f"LiveEdit repository size mismatch: expected {expected}, got {after}.")

    def forward_with_experts(
        self,
        batch: Dict[str, Any],
        u: torch.Tensor,
        v: torch.Tensor,
        phi: torch.Tensor,
        psi: torch.Tensor,
        force_all_experts: bool = False,
    ) -> Any:
        if u.shape[0] == 0:
            return self.model(batch)
        context = LiveEditContext(batch=batch, u=u, v=v, phi=phi, psi=psi, force_all_experts=force_all_experts)
        with temporary_context(self, context):
            return self.model(batch)

    def forward(self, *inputs, **kwargs):
        if len(inputs) != 1 or not isinstance(inputs[0], dict) or kwargs:
            return self.model(*inputs, **kwargs)
        batch = inputs[0]
        return self.forward_with_experts(batch, self.repository.u, self.repository.v, self.repository.phi, self.repository.psi)

    def edit(self, batch, condition=None, detach_history=False):
        self.append_from_batch(batch)
        return self, {"liveedit/num_experts": float(len(self.repository))}

    def _nll(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        logits = _tensor_logits(outputs)
        labels = batch["labels"]
        if logits.shape[1] > labels.shape[1]:
            return self.edit_loss_fn(self.config, logits, labels)["nll"]
        return self.edit_loss_fn(self.config, logits, labels[:, -logits.shape[1] - 1 :])["nll"]

    def _edit_metrics(self, outputs: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        logits = _tensor_logits(outputs)
        labels = batch["labels"]
        if logits.shape[1] > labels.shape[1]:
            return self.edit_loss_fn(self.config, logits, labels)
        return self.edit_loss_fn(self.config, logits, labels[:, -logits.shape[1] - 1 :])

    def _capture_feature_sets(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        hidden = self.capture_hidden(batch)
        vision_mask, prompt_mask, _, _ = get_liveedit_masks(batch, hidden)
        phi_hat, psi_hat = self.edit_feature_extractor.extract(hidden, vision_mask, prompt_mask)
        phi_bar, psi_bar = self.input_feature_extractor.extract(hidden, vision_mask, prompt_mask)
        phi_theta = self.input_feature_extractor.extract_sentinel(hidden, prompt_mask)
        return {
            "phi_hat": phi_hat,
            "psi_hat": psi_hat,
            "phi_bar": phi_bar,
            "psi_bar": psi_bar,
            "phi_theta": phi_theta,
        }

    def _route_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        edit_features = self._capture_feature_sets(batch["edit_inner"])
        gen_features = self._capture_feature_sets(batch["edit_outer"])
        loc_features = self._capture_feature_sets(batch["loc_image"])
        choose_hat = (torch.rand(edit_features["phi_hat"].shape[0], device=edit_features["phi_hat"].device) < 0.5)[
            :, None, None
        ]
        choose_bar = (torch.rand(edit_features["phi_bar"].shape[0], device=edit_features["phi_bar"].device) < 0.5)[
            :, None, None
        ]
        phi_hat_g = torch.where(choose_hat, edit_features["phi_hat"], gen_features["phi_hat"])
        psi_hat_g = torch.where(choose_hat, edit_features["psi_hat"], gen_features["psi_hat"])
        phi_bar_g = torch.where(choose_bar, edit_features["phi_bar"], gen_features["phi_bar"])
        psi_bar_g = torch.where(choose_bar, edit_features["psi_bar"], gen_features["psi_bar"])
        phi_theta_g = torch.where(choose_bar, edit_features["phi_theta"], gen_features["phi_theta"])
        return liveedit_routing_losses(
            phi_hat_g,
            psi_hat_g,
            phi_bar_g,
            psi_bar_g,
            loc_features["phi_bar"],
            loc_features["psi_hat"],
            phi_theta_g,
            loc_features["phi_theta"],
            similarity=self.similarity,
        )

    def _base_trainable_count(self) -> float:
        return float(sum(param.numel() for param in self.model.parameters() if param.requires_grad))

    def _editor_trainable_count(self) -> float:
        return float(sum(param.numel() for param in self.outer_parameters() if param.requires_grad))

    def edit_step(self, batch: Dict[str, Any], training: bool):
        self.set_editor_train(training)
        with torch.no_grad():
            base_loc_outputs = self.model(batch["loc"])
            base_loc_logits = _tensor_logits(base_loc_outputs)
            base_image_loc_outputs = self.model(batch["loc_image"])
            base_image_loc_logits = _tensor_logits(base_image_loc_outputs)

        u, v, phi, psi, _ = self.generate_experts(batch["edit_inner"])
        with torch.set_grad_enabled(training):
            rel_outputs = self.forward_with_experts(batch["edit_inner"], u, v, phi, psi, force_all_experts=True)
            gen_text_outputs = self.forward_with_experts(batch["edit_outer"], u, v, phi, psi, force_all_experts=True)
            gen_image_outputs = self.forward_with_experts(batch["edit_outer_image"], u, v, phi, psi, force_all_experts=True)
            loc_outputs = self.forward_with_experts(batch["loc"], u, v, phi, psi, force_all_experts=True)
            loc_image_outputs = self.forward_with_experts(batch["loc_image"], u, v, phi, psi, force_all_experts=True)

            l_rel = self._nll(rel_outputs, batch["edit_inner"])
            l_gen_text = self._nll(gen_text_outputs, batch["edit_outer"])
            l_gen_image = self._nll(gen_image_outputs, batch["edit_outer_image"])
            l_gen = 0.5 * (l_gen_text + l_gen_image)

            loc_logits = _tensor_logits(loc_outputs)
            loc_mask = _target_mask(loc_outputs, batch["loc"], loc_logits)
            image_loc_logits = _tensor_logits(loc_image_outputs)
            image_loc_mask = _target_mask(loc_image_outputs, batch["loc_image"], image_loc_logits)
            l_loc_text = kl_loc_loss(base_loc_logits.detach(), loc_logits, mask=loc_mask)
            l_loc_image = kl_loc_loss(base_image_loc_logits.detach(), image_loc_logits, mask=image_loc_mask)
            l_loc = l_loc_text + l_loc_image

            route_losses = self._route_loss(batch)
            l_hr = route_losses["hr"]
            l_sr1 = route_losses["sr1"]
            l_sr2 = route_losses["sr2"]
            l_route = (
                float(_cfg(self.config, "liveedit_hr_weight", 1.0)) * l_hr
                + float(_cfg(self.config, "liveedit_sr1_weight", 1.0)) * l_sr1
                + float(_cfg(self.config, "liveedit_sr2_weight", 1.0)) * l_sr2
            )
            l_edit = (
                float(_cfg(self.config, "liveedit_rel_weight", 1.0)) * l_rel
                + float(_cfg(self.config, "liveedit_gen_weight", 1.0)) * l_gen
                + float(_cfg(self.config, "liveedit_loc_weight", 1.0)) * l_loc
            )
            l_total = l_edit + float(_cfg(self.config, "liveedit_route_weight", 1.0)) * l_route

        if training:
            safe_backward(l_total, self.outer_parameters(), int(_cfg(self.config, "accumulate_bs", 1)), allow_unused=True)
            if bool(_cfg(self.config, "liveedit_debug", False)) and self._base_trainable_count() != 0.0:
                raise RuntimeError("LiveEdit expected frozen base VLLM parameters but found trainable base parameters.")

        with torch.no_grad():
            rel_metrics = self._edit_metrics(rel_outputs, batch["edit_inner"])
        info_dict = {
            "loss/liveedit_total": float(l_total.detach().cpu()),
            "loss/liveedit_edit": float(l_edit.detach().cpu()),
            "loss/liveedit_rel": float(l_rel.detach().cpu()),
            "loss/liveedit_gen": float(l_gen.detach().cpu()),
            "loss/liveedit_loc": float(l_loc.detach().cpu()),
            "loss/liveedit_route": float(l_route.detach().cpu()),
            "loss/liveedit_hr": float(l_hr.detach().cpu()),
            "loss/liveedit_sr1": float(l_sr1.detach().cpu()),
            "loss/liveedit_sr2": float(l_sr2.detach().cpu()),
            "loss/total": float(l_total.detach().cpu()),
            "loss/total_edit": float(l_edit.detach().cpu()),
            "loss/edit": float(l_rel.detach().cpu()),
            "loss/loc": float(l_loc.detach().cpu()),
            "edit/acc": float(rel_metrics["acc"].detach().cpu()),
            "inner/acc": float(rel_metrics["acc"].detach().cpu()),
            "image_rephrase/acc": 0.0,
            "loc/acc": 0.0,
            "image_loc/acc": 0.0,
            "liveedit/num_experts": float(len(self.repository)),
            "liveedit/base_vllm_trainable_params": self._base_trainable_count(),
            "liveedit/editor_trainable_params": self._editor_trainable_count(),
        }
        info_dict.update(self._last_info)
        if torch.cuda.is_available():
            info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
            info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        return l_total, l_rel, l_loc, torch.tensor(0.0, device=l_total.device), info_dict


def sanitize_liveedit_metadata(requests: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for request in requests:
        yield {
            "prompt": request.get("prompt"),
            "target": request.get("target", request.get("target_new")),
            "rephrase_prompt": request.get("rephrase_prompt"),
            "locality_prompt": request.get("locality_prompt"),
        }
