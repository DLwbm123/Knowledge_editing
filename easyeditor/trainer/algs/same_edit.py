from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from ...models.same_edit.same_edit_modules import SAMEEditConfig, SAMEEditModel
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


class SAMEEdit(EditableModel):
    def __init__(self, model, config, model_constructor):
        super().__init__(model, config, model_constructor)
        self.same_config = SAMEEditConfig.from_hparams(config)
        self.same_model = SAMEEditModel(self.model, self.same_config)
        self.model = self.same_model
        self.global_step = 0
        self._last_info: Dict[str, Any] = {}

    def set_editor_train(self, training: bool) -> None:
        self.train(training)
        self.same_model.model.eval()
        for _name, layer in self.same_model.same_edit_layers():
            layer.train(training)
            layer.base_linear.eval()

    def outer_parameters(self):
        return self.same_model.trainable_parameters()

    def edit(self, batch, condition=None, detach_history=False):
        edit_index = int(_cfg(self.config, "same_edit_current_edit", self.same_config.current_edit))
        if isinstance(condition, dict) and "same_edit_current_edit" in condition:
            edit_index = int(condition["same_edit_current_edit"])
        self.same_model.set_current_edit(edit_index)
        info = self.same_model.summary()
        self._last_info = info
        return self, {"same_edit/current_edit": float(edit_index)}

    def _nll(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        logits = _tensor_logits(outputs)
        labels = batch["labels"]
        return self.edit_loss_fn(self.config, logits, _answer_labels_for_logits(logits, labels))["nll"]

    def _base_forward(self, batch: Dict[str, Any]) -> Any:
        with self.same_model.adapters_disabled():
            return self.model(batch)

    def _reference_kl(self, reference_batch: Optional[Dict[str, Any]]) -> torch.Tensor:
        if reference_batch is None:
            return torch.tensor(0.0, device=next(self.outer_parameters()).device)
        with torch.no_grad():
            base_outputs = self._base_forward(reference_batch)
            base_logits = _tensor_logits(base_outputs)
        post_outputs = self.model(reference_batch)
        post_logits = _tensor_logits(post_outputs)
        mask = getattr(post_outputs, "attention_mask", None)
        if mask is None:
            mask = torch.ones(post_logits.shape[:2], device=post_logits.device)
        return kl_loc_loss(base_logits.detach(), post_logits, mask=mask)

    def edit_step(self, batch: Dict[str, Any], training: bool, optimizer=None):
        self.global_step += int(training)
        edit_index = int(_cfg(self.config, "same_edit_current_edit", self.same_config.current_edit))
        self.same_config.current_edit = edit_index
        self.same_model.set_current_edit(edit_index)
        self.same_model.validate_covariance_for_curvature()
        self.set_editor_train(training)
        edit_batch = batch.get("edit_outer") or batch.get("edit_inner") or batch
        inner_batch = batch.get("edit_inner") or edit_batch
        image_batch = batch.get("edit_outer_image")
        reference_batch = batch.get("loc_image") or batch.get("loc")

        with torch.set_grad_enabled(training):
            edit_outputs = self.model(edit_batch)
            l_edit = self._nll(edit_outputs, edit_batch)

            l_image = l_edit * 0.0
            if bool(_cfg(self.config, "same_edit_use_rephrase_loss", False)) and image_batch is not None:
                image_outputs = self.model(image_batch)
                l_image = self._nll(image_outputs, image_batch)

            l_loc = l_edit * 0.0
            if bool(_cfg(self.config, "same_edit_use_locality_kl", False)):
                l_loc = self._reference_kl(reference_batch)

            l_route = l_edit * 0.0
            route_loss_weight = float(
                _cfg(self.config, "same_edit_route_loss_weight", _cfg(self.config, "route_loss_weight", 0.0))
            )
            if route_loss_weight:
                l_route = self.same_model.routing_supervision_loss()

            l_total = (
                float(_cfg(self.config, "cedit", 1.0)) * l_edit
                + float(_cfg(self.config, "iedit", 0.0)) * l_image
                + float(_cfg(self.config, "cloc", 0.0)) * l_loc
                + route_loss_weight * l_route
            )
            l_total = l_total + sum((param.sum() * 0.0 for param in self.outer_parameters()), l_total * 0.0)

        if training:
            safe_backward(
                l_total,
                self.outer_parameters(),
                int(_cfg(self.config, "accumulate_bs", 1)),
                allow_unused=True,
            )

        with torch.no_grad():
            inner_outputs = self.model(inner_batch)
            inner_nll = self._nll(inner_outputs, inner_batch)
            summary = self.same_model.summary()
            first_layer = summary.get("layers", [{}])[0] if summary.get("layers") else {}
            routing = first_layer.get("routing") or []
            assigned = summary.get("assigned_expert_id")
            top_expert = first_layer.get("top_expert_id")
            overlap = float(routing[int(assigned)]) if routing and assigned is not None and int(assigned) < len(routing) else 0.0
            nonfinite = 0
            for param in self.outer_parameters():
                if param.grad is not None:
                    nonfinite += int((~torch.isfinite(param.grad)).sum().detach().cpu().item())

        info = {
            "loss/same_edit_total": float(l_total.detach().cpu()),
            "loss/same_edit_edit": float(l_edit.detach().cpu()),
            "loss/same_edit_inner": float(inner_nll.detach().cpu()),
            "loss/same_edit_image": float(l_image.detach().cpu()),
            "loss/same_edit_loc": float(l_loc.detach().cpu()),
            "loss/same_edit_route": float(l_route.detach().cpu()),
            "loss/total": float(l_total.detach().cpu()),
            "loss/edit": float(l_edit.detach().cpu()),
            "loss/loc": float(l_loc.detach().cpu()),
            "same_edit/current_edit": float(self.same_config.current_edit),
            "same_edit/assigned_expert_id": float(assigned if assigned is not None else -1),
            "same_edit/top_expert_id": float(top_expert if top_expert is not None else -1),
            "same_edit/routing_overlap": overlap,
            "same_edit/routing_entropy": float(first_layer.get("routing_entropy") or 0.0),
            "same_edit/active_expert_count": float(summary.get("mean_active_expert_count") or 0.0),
            "same_edit/covariance_valid_count": float(summary.get("covariance_valid_count") or 0),
            "same_edit/nan_inf_grad_count": float(nonfinite),
            "same_edit/trainable_params": float(sum(param.numel() for param in self.outer_parameters())),
        }
        self._last_info = info
        return l_total, l_edit, l_loc, torch.tensor(0.0, device=l_total.device), info

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return {
            f"{prefix}same_edit_bundle": self.same_model.state_bundle(),
            f"{prefix}same_edit_global_step": torch.tensor(self.global_step),
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        state = dict(state_dict)
        bundle = state.get("same_edit_bundle")
        if bundle is None:
            bundle = state.get("model.same_edit_bundle")
        if bundle is not None:
            self.same_model.load_state_bundle(bundle)
        step = state.get("same_edit_global_step", state.get("model.same_edit_global_step"))
        if step is not None:
            self.global_step = int(step.item() if torch.is_tensor(step) else step)
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def save_same_edit_state(self, output_dir: str | Path) -> Dict[str, Any]:
        return self.same_model.save_same_edit_state(output_dir)

    def write_summary(self, output_dir: str | Path) -> Dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        summary = self.same_model.save_same_edit_state(output)
        (output / "same_edit_runtime_info.json").write_text(json.dumps(self._last_info, indent=2, sort_keys=True) + "\n")
        return summary
