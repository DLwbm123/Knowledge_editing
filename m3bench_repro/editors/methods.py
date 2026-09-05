"""Four authorized M3Bench paper-spec editor implementations."""

from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn

from .llava_runtime import (
    EditorRecord,
    LlavaMedEditorRuntime,
    PreparedBatch,
    canonical_sha256,
    seed_everything,
    write_json_atomic,
)
from .routed_layers import GraceValueLinear, RoutedFullLinear, RoutedLoRALinear, safe_slot
from .routing import (
    GraceCodebook,
    MemoryRouter,
    balanced_radius,
    decision_as_json,
    euclidean_distances,
)


CLASSIFICATION = "M3BENCH_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V2_EFFECT_REPAIRED"
SEED = 20260828


@dataclass(frozen=True)
class LoraRuntimeConfig:
    profile_name: str = "LoRA-paper-spec-5"
    learning_rate: float = 5e-5
    steps_per_edit: int = 5
    rank: int = 16
    alpha: int = 16
    layer_scope: str = "all"
    target_modules: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
    stop_rule: str = "fixed"
    min_steps: int = 1
    check_interval: int = 1
    target_nll_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.steps_per_edit < 1 or self.rank < 1 or self.alpha < 1:
            raise ValueError("LoRA learning rate and steps must be positive")
        if self.layer_scope not in {"all", "last_16", "last_8"}:
            raise ValueError(f"unsupported LoRA layer scope: {self.layer_scope}")
        if not self.target_modules or not set(self.target_modules) <= {
            "gate_proj", "up_proj", "down_proj", "q_proj", "v_proj"
        }:
            raise ValueError(f"unsupported LoRA target modules: {self.target_modules}")
        if self.stop_rule not in {"fixed", "first_success"}:
            raise ValueError(f"unsupported LoRA stop rule: {self.stop_rule}")
        if self.min_steps < 1 or self.check_interval < 1 or self.min_steps > self.steps_per_edit:
            raise ValueError("invalid LoRA adaptive-stop interval")

    @property
    def paper_spec_default(self) -> bool:
        return self == LoraRuntimeConfig()

    def select_targets(self, available: list[str]) -> list[str]:
        parsed = []
        for target in available:
            match = re.search(r"\.layers\.(\d+)\..*\.(\w+_proj)$", target)
            if match and match.group(2) in self.target_modules:
                parsed.append((int(match.group(1)), target))
        if not parsed:
            raise ValueError("LoRA profile selected no available target modules")
        final_layer = max(layer for layer, _ in parsed)
        keep = {"all": final_layer + 1, "last_16": 16, "last_8": 8}[self.layer_scope]
        selected = [target for layer, target in parsed if layer >= final_layer - keep + 1]
        if not selected:
            raise ValueError("LoRA profile selected no layers")
        return selected


def record_seed(record_id: str, namespace: str) -> int:
    payload = f"{SEED}:{namespace}:{record_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def finite_gradients(parameters: list[nn.Parameter]) -> bool:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(gradient).all().item() for gradient in gradients)


def parameter_bytes(parameters: list[nn.Parameter]) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in parameters)


class PaperSpecEditor(ABC):
    method: str

    def __init__(self, runtime: LlavaMedEditorRuntime):
        if runtime.target_lock is None:
            raise RuntimeError("resolve and freeze module inventory before installing an editor")
        self.runtime = runtime
        self.edit_history: list[str] = []

    @abstractmethod
    def apply_edit(self, record: EditorRecord) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def score_target_nll(self, record: EditorRecord) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate(self, record: EditorRecord, *, use_cache: bool = True) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save_editor_state(self, path: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_editor_state(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_editor_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def state_summary(self) -> dict[str, Any]:
        raise NotImplementedError

    def base_integrity(self) -> dict[str, Any]:
        if self.runtime.base_guard is None:
            raise RuntimeError("base guard was not captured")
        return self.runtime.base_guard.verify()

    def write_config_lock(self, output_dir: Path) -> None:
        config = self.config_lock()
        config["config_sha256"] = canonical_sha256(config)
        write_json_atomic(output_dir / "METHOD_CONFIG_LOCK_V2.json", config)

    @abstractmethod
    def config_lock(self) -> dict[str, Any]:
        raise NotImplementedError


class LoraPaperSpecEditor(PaperSpecEditor):
    method = "lora"

    def __init__(self, runtime: LlavaMedEditorRuntime, config: LoraRuntimeConfig | None = None):
        super().__init__(runtime)
        from peft import LoraConfig, get_peft_model

        self.runtime_config = config or LoraRuntimeConfig()
        targets = self.runtime_config.select_targets(list(runtime.target_lock["lora"]["targets"]))
        config = LoraConfig(
            r=self.runtime_config.rank,
            lora_alpha=self.runtime_config.alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(runtime.adapter.model, config)
        runtime.adapter.model = peft_model
        self.peft_model = peft_model
        self.adapter_name = "default"
        self.targets = targets
        self._set_enabled(True)
        self._freeze_non_lora()
        self._initial_adapter_state = self._capture_adapter_state()
        self.initial_adapter_sha256 = self.adapter_state_sha256()

    def _freeze_non_lora(self) -> None:
        for name, parameter in self.peft_model.named_parameters():
            parameter.requires_grad_("lora_" in name)

    def _set_enabled(self, enabled: bool) -> None:
        if enabled:
            self.peft_model.set_adapter(self.adapter_name)
            self.peft_model.enable_adapter_layers()
        else:
            self.peft_model.disable_adapter_layers()

    @contextmanager
    def disabled(self) -> Iterator[None]:
        self._set_enabled(False)
        try:
            yield
        finally:
            self._set_enabled(True)

    def trainable(self) -> list[nn.Parameter]:
        self._freeze_non_lora()
        return [parameter for parameter in self.peft_model.parameters() if parameter.requires_grad]

    def _capture_adapter_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, parameter in self.peft_model.named_parameters()
            if "lora_" in name
        }

    def adapter_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.peft_model.named_parameters()):
            if "lora_" not in name:
                continue
            value = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def apply_edit(self, record: EditorRecord) -> dict[str, Any]:
        seed = record_seed(record.record_id, self.method)
        seed_everything(seed)
        self._set_enabled(True)
        batch = self.runtime.build_edit_batch(record)
        parameters = self.trainable()
        if not parameters:
            raise RuntimeError("LoRA has no trainable parameters")
        optimizer = torch.optim.AdamW(parameters, lr=self.runtime_config.learning_rate)
        losses = []
        gradients_finite = []
        stop_checks = []
        self.peft_model.eval()
        for step in range(1, self.runtime_config.steps_per_edit + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = self.runtime.compute_loss(batch)
            loss.backward()
            gradients_finite.append(finite_gradients(parameters))
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            if (
                self.runtime_config.stop_rule == "first_success"
                and step >= self.runtime_config.min_steps
                and step % self.runtime_config.check_interval == 0
            ):
                from scripts.m3bench_base_correctness_v3 import exact_correct, public_fuzzy_correct

                score = self.score_target_nll(record)
                answer = self.generate(record)["generation"]["decoded_text"]
                passed = (
                    score["nll"] <= self.runtime_config.target_nll_threshold
                    and score["first_target_token_rank"] == 1
                    and (exact_correct(answer, record.target) or public_fuzzy_correct(answer, record.target))
                )
                stop_checks.append({"step": step, "passed": passed, **score})
                if passed:
                    break
        self.edit_history.append(record.record_id)
        return {
            "record_id": record.record_id,
            "seed": seed,
            "losses": losses,
            "finite_losses": all(torch.isfinite(torch.tensor(losses)).tolist()),
            "finite_gradients": all(gradients_finite),
            "epochs": len(losses),
            "stop_rule": self.runtime_config.stop_rule,
            "stop_checks": stop_checks,
            "trainable_parameter_count": sum(p.numel() for p in parameters),
            "trainable_parameter_bytes": parameter_bytes(parameters),
            "target_mask": batch.mask_report(),
        }

    def score_target_nll(self, record: EditorRecord) -> dict[str, Any]:
        batch = self.runtime.build_edit_batch(record)
        with torch.no_grad():
            score = self.runtime.score_target(batch)
        return {**score, "route": None, "target_mask": batch.mask_report()}

    def generate(self, record: EditorRecord, *, use_cache: bool = True) -> dict[str, Any]:
        self._set_enabled(True)
        return {"route": None, "generation": self.runtime.generate(record, use_cache=use_cache)}

    def save_editor_state(self, path: Path) -> dict[str, Any]:
        path.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(path, safe_serialization=True)
        metadata = {"method": self.method, "edit_history": self.edit_history, "targets": self.targets}
        write_json_atomic(path / "m3bench_editor_state.json", metadata)
        files = sorted(p for p in path.rglob("*") if p.is_file())
        digest = hashlib.sha256()
        for file in files:
            digest.update(str(file.relative_to(path)).encode("utf-8"))
            digest.update(file.read_bytes())
        return {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "size_bytes": sum(file.stat().st_size for file in files),
            "file_count": len(files),
        }

    def load_editor_state(self, path: Path) -> None:
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        weights = load_peft_weights(path, device=str(self.runtime.device))
        result = set_peft_model_state_dict(
            self.peft_model, weights, adapter_name=self.adapter_name
        )
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"unexpected LoRA state keys: {result.unexpected_keys}")
        metadata = json.loads((path / "m3bench_editor_state.json").read_text(encoding="utf-8"))
        self.edit_history = list(metadata["edit_history"])
        self._set_enabled(True)

    def reset_editor_state(self) -> None:
        current = dict(self.peft_model.named_parameters())
        with torch.no_grad():
            for name, initial in self._initial_adapter_state.items():
                if name not in current:
                    raise RuntimeError(f"LoRA reset target disappeared: {name}")
                current[name].copy_(initial.to(device=current[name].device, dtype=current[name].dtype))
        if self.adapter_state_sha256() != self.initial_adapter_sha256:
            raise RuntimeError("LoRA reset did not restore the exact initial adapter state")
        self._set_enabled(False)
        self.edit_history.clear()

    def state_summary(self) -> dict[str, Any]:
        adapters = [
            (name, parameter)
            for name, parameter in self.peft_model.named_parameters()
            if "lora_" in name
        ]
        return {
            "method": self.method,
            "edit_history": list(self.edit_history),
            "active_adapter": self.adapter_name,
            "adapter_parameter_count": sum(parameter.numel() for _, parameter in adapters),
            "adapter_parameter_bytes": sum(
                parameter.numel() * parameter.element_size() for _, parameter in adapters
            ),
            "target_count": len(self.targets),
        }

    def config_lock(self) -> dict[str, Any]:
        lock = {
            "schema_version": "m3bench-editor-method-config-v2",
            "method": "LoRA",
            "classification": CLASSIFICATION,
            "scope": "all language-model MLP blocks" if self.runtime_config.layer_scope == "all" else self.runtime_config.layer_scope.replace("_", " ") + " language-model blocks",
            "targets": self.targets,
            "rank": self.runtime_config.rank,
            "lora_alpha": self.runtime_config.alpha,
            "dropout": 0.0,
            "optimizer": "AdamW",
            "learning_rate": self.runtime_config.learning_rate,
            "batch_size": 1,
            "gradient_clip": 1.0,
            "epochs_per_edit": self.runtime_config.steps_per_edit,
            "projector": "excluded",
            "vision_encoder": "excluded",
            "source": "PEFT 0.19.1 @ ba6a19060d6ab54a87538a6e77e3e4d5a907375b",
        }
        if not self.runtime_config.paper_spec_default:
            lock.update({
                "profile_name": self.runtime_config.profile_name,
                "paper_spec_deviation": True,
                "deviation_scope": "explicit training and adapter structure",
                "layer_scope": self.runtime_config.layer_scope,
                "target_modules": list(self.runtime_config.target_modules),
                "stop_rule": self.runtime_config.stop_rule,
                "min_steps": self.runtime_config.min_steps,
                "check_interval": self.runtime_config.check_interval,
                "target_nll_threshold": self.runtime_config.target_nll_threshold,
            })
        return lock


class GracePaperSpecEditor(PaperSpecEditor):
    method = "grace"

    def __init__(self, runtime: LlavaMedEditorRuntime):
        super().__init__(runtime)
        self.target = runtime.target_lock["grace"]["targets"][0]
        base = runtime.get_module(self.target)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"GRACE target must resolve to nn.Linear, got {type(base)}")
        self.wrapper = GraceValueLinear(base, replacement="replace_prompt")
        runtime.replace_module(self.target, self.wrapper)
        self.codebook = GraceCodebook(distance="cosine", eps_init=1.0)
        self.requested_to_effective: dict[str, str] = {}

    @contextmanager
    def disabled(self) -> Iterator[None]:
        previous = self.wrapper.active_logical_id
        previous_token_index = self.wrapper.token_index
        self.wrapper.disable()
        try:
            yield
        finally:
            if previous is None:
                self.wrapper.disable()
            else:
                self.wrapper.set_active(previous, token_index=previous_token_index)

    @contextmanager
    def _activated(self, logical_id: str | None, *, token_index: int) -> Iterator[None]:
        self.wrapper.set_active(logical_id, token_index=token_index)
        try:
            yield
        finally:
            self.wrapper.disable()

    @contextmanager
    def route_generation(self, record: EditorRecord) -> Iterator[Any]:
        decision = self._route(record)
        with self._activated(decision.logical_edit_id, token_index=-1):
            yield decision

    def _question_key(self, record: EditorRecord) -> torch.Tensor:
        with self.disabled():
            question_batch = self.runtime.build_question_batch(record)
            return self.runtime.extract_layer_input_key(
                question_batch, module_path=self.target, pooling="last_prompt"
            )

    def apply_edit(self, record: EditorRecord) -> dict[str, Any]:
        seed = record_seed(record.record_id, self.method)
        seed_everything(seed)
        key = self._question_key(record)
        batch = self.runtime.build_edit_batch(record)
        insertion = self.codebook.insert_with_source_semantics(
            record.record_id, key, batch.target_token_ids
        )
        effective_id = insertion.effective_logical_edit_id
        if effective_id not in self.wrapper.logical_to_slot:
            self.wrapper.add_cold_value(effective_id, seed=seed)
        parameters = self.wrapper.train_only(effective_id)
        if len(parameters) != 1:
            raise RuntimeError("GRACE must train exactly one entry value")
        optimizer = torch.optim.Adam(parameters, lr=1.0)
        losses, gradient_checks = [], []
        try:
            with self._activated(effective_id, token_index=batch.key_token_index):
                for _ in range(100):
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.runtime.compute_loss(batch)
                    loss.backward()
                    gradient_checks.append(finite_gradients(parameters))
                    optimizer.step()
                    losses.append(float(loss.detach().cpu().item()))
        finally:
            self.wrapper.train_only(None)
            self.wrapper.disable()
        self.requested_to_effective[record.record_id] = effective_id
        self.edit_history.append(record.record_id)
        return {
            "record_id": record.record_id,
            "seed": seed,
            "losses": losses,
            "finite_losses": all(torch.isfinite(torch.tensor(losses)).tolist()),
            "finite_gradients": all(gradient_checks),
            "steps": 100,
            "insert": asdict(insertion),
            "key_norm": float(torch.linalg.vector_norm(key).cpu().item()),
            "entry_count": len(self.codebook),
            "trainable_parameter_count": sum(p.numel() for p in parameters),
            "target_mask": batch.mask_report(),
        }

    def _route(self, record: EditorRecord):
        key = self._question_key(record)
        return self.codebook.route(key)

    def score_target_nll(self, record: EditorRecord) -> dict[str, Any]:
        decision = self._route(record)
        batch = self.runtime.build_edit_batch(record)
        with self._activated(decision.logical_edit_id, token_index=batch.key_token_index):
            with torch.no_grad():
                score = self.runtime.score_target(batch)
        return {
            **score,
            "route": decision_as_json(decision),
            "target_mask": batch.mask_report(),
        }

    def generate(self, record: EditorRecord, *, use_cache: bool = True) -> dict[str, Any]:
        with self.route_generation(record) as decision:
            output = self.runtime.generate(record, use_cache=use_cache)
        return {"route": decision_as_json(decision), "generation": output}

    def save_editor_state(self, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "method": self.method,
            "target": self.target,
            "codebook": self.codebook.export_state(),
            "wrapper": self.wrapper.export_state(),
            "requested_to_effective": self.requested_to_effective,
            "edit_history": self.edit_history,
        }
        torch.save(state, path)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}

    def load_editor_state(self, path: Path) -> None:
        state = torch.load(path, map_location=self.runtime.device, weights_only=False)
        if state["target"] != self.target:
            raise ValueError("GRACE target mismatch")
        self.codebook = GraceCodebook.from_state(state["codebook"], device=self.runtime.device)
        self.wrapper.load_exported_state(state["wrapper"])
        self.requested_to_effective = dict(state["requested_to_effective"])
        self.edit_history = list(state["edit_history"])

    def reset_editor_state(self) -> None:
        self.codebook.clear()
        self.wrapper.values = nn.ParameterDict()
        self.wrapper.logical_to_slot.clear()
        self.wrapper.slot_to_logical.clear()
        self.wrapper.disable()
        self.requested_to_effective.clear()
        self.edit_history.clear()

    def state_summary(self) -> dict[str, Any]:
        values = list(self.wrapper.values.values())
        logical_edit_ids = list(self.codebook.logical_ids)
        requested_ids = list(self.requested_to_effective)
        resident_ids = set(logical_edit_ids)
        return {
            "method": self.method,
            "entry_count": len(self.codebook),
            "logical_edit_ids": logical_edit_ids,
            "radii": list(self.codebook.radii),
            "value_entry_count": len(values),
            "value_parameter_count": sum(p.numel() for p in values),
            "value_parameter_bytes": parameter_bytes(values),
            "edit_history": list(self.edit_history),
            "requested_to_effective_count": len(self.requested_to_effective),
            "requested_mapping_keys_match_history": requested_ids == self.edit_history,
            "requested_mapping_values_resident": all(
                effective_id in resident_ids for effective_id in self.requested_to_effective.values()
            ),
        }

    def config_lock(self) -> dict[str, Any]:
        return {
            "schema_version": "m3bench-editor-method-config-v2",
            "method": "GRACE",
            "classification": "M3Bench-paper-spec adaptation of locked GRACE source",
            "scope": "final-layer LM up_projection adaptor",
            "target": self.target,
            "distance": "1 - cosine_similarity",
            "activation": "nearest_distance <= stored_radius",
            "eps_init": 1.0,
            "steps_per_edit": 100,
            "optimizer": "Adam",
            "learning_rate": 1.0,
            "trainable": "selected entry value only",
            "value_init": "cold uniform [0,1), source lock",
            "replacement": "replace_prompt inclusive through expanded key token",
            "collision_and_radius_update": "locked source semantics",
            "euclidean": "diagnostic only, not primary smoke",
            "source": "GRACE @ f674183f17a995d109e10ee6140d4c3e6d016115",
        }


class _BalanceRoutingMixin:
    target: str
    router: MemoryRouter

    def _question_key(self, record: EditorRecord) -> torch.Tensor:
        with self.disabled():
            batch = self.runtime.build_question_batch(record)
            return self.runtime.extract_layer_input_key(batch, module_path=self.target, pooling="mean")

    def _anchors(self, record: EditorRecord, black_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Path]:
        with self.disabled():
            original_batch = self.runtime.build_question_batch(record)
            positive_batch = self.runtime.build_question_batch(record, question=record.official_rephrase)
            black_path = self.runtime.make_black_image(record, black_dir)
            negative_batch = self.runtime.build_question_batch(record, image_path=black_path)
            key = self.runtime.extract_layer_input_key(original_batch, module_path=self.target, pooling="mean")
            positive = self.runtime.extract_layer_input_key(positive_batch, module_path=self.target, pooling="mean")
            negative = self.runtime.extract_layer_input_key(negative_batch, module_path=self.target, pooling="mean")
        return key, positive, negative, black_path

    def _route(self, record: EditorRecord):
        return self.router.route(self._question_key(record))


class BalanceEditPaperSpecEditor(_BalanceRoutingMixin, PaperSpecEditor):
    method = "balancedit"

    def __init__(self, runtime: LlavaMedEditorRuntime, *, inactive_store_dir: Path | None = None):
        super().__init__(runtime)
        self.target = runtime.target_lock["balancedit"]["targets"][0]
        base = runtime.get_module(self.target)
        if not isinstance(base, nn.Linear):
            raise TypeError("BalanceEdit target must be a linear up_projection")
        self.wrapper = RoutedFullLinear(base, inactive_store_dir=inactive_store_dir)
        runtime.replace_module(self.target, self.wrapper)
        self.router = MemoryRouter("euclidean")

    @contextmanager
    def disabled(self) -> Iterator[None]:
        previous = self.wrapper.active_logical_id
        self.wrapper.set_active(None)
        try:
            yield
        finally:
            self.wrapper.set_active(previous)

    @contextmanager
    def _activated(self, logical_id: str | None) -> Iterator[None]:
        self.wrapper.set_active(logical_id)
        try:
            yield
        finally:
            self.wrapper.set_active(None)

    @contextmanager
    def route_generation(self, record: EditorRecord) -> Iterator[Any]:
        decision = self._route(record)
        with self._activated(decision.logical_edit_id):
            yield decision

    def apply_edit(self, record: EditorRecord) -> dict[str, Any]:
        seed = record_seed(record.record_id, self.method)
        seed_everything(seed)
        key, positive, negative, black_path = self._anchors(
            record, self.runtime.run_root / "inputs/black_images"
        )
        radius = balanced_radius(
            key, positive, negative, alpha=0.2, distance="euclidean"
        )
        positive_distance = float(euclidean_distances(key, positive)[0].cpu().item())
        negative_distance = float(euclidean_distances(key, negative)[0].cpu().item())
        batch = self.runtime.build_edit_batch(record)
        self.router.add(record.record_id, key, radius, batch.target_token_ids)
        edited = self.wrapper.add_edit(record.record_id)
        parameters = self.wrapper.train_only(record.record_id)
        optimizer = torch.optim.Adam(parameters, lr=0.01)
        losses, gradient_checks = [], []
        try:
            with self._activated(record.record_id):
                for _ in range(50):
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.runtime.compute_loss(batch)
                    loss.backward()
                    gradient_checks.append(finite_gradients(parameters))
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu().item()))
        finally:
            self.wrapper.train_only(None)
            self.wrapper.set_active(None)
        self.edit_history.append(record.record_id)
        return {
            "record_id": record.record_id,
            "seed": seed,
            "losses": losses,
            "finite_losses": all(torch.isfinite(torch.tensor(losses)).tolist()),
            "finite_gradients": all(gradient_checks),
            "steps": 50,
            "radius": float(radius.cpu().item()),
            "positive_distance": positive_distance,
            "negative_distance": negative_distance,
            "distance": "euclidean",
            "black_image_path": str(black_path),
            "black_image_sha256": hashlib.sha256(black_path.read_bytes()).hexdigest(),
            "router_positive_source": record.router_positive_source,
            "entry_count": len(self.router),
            "trainable_parameter_count": sum(p.numel() for p in parameters),
            "trainable_parameter_bytes": parameter_bytes(parameters),
            "target_mask": batch.mask_report(),
        }

    def score_target_nll(self, record: EditorRecord) -> dict[str, Any]:
        decision = self._route(record)
        batch = self.runtime.build_edit_batch(record)
        with self._activated(decision.logical_edit_id):
            with torch.no_grad():
                score = self.runtime.score_target(batch)
        return {**score, "route": decision_as_json(decision), "target_mask": batch.mask_report()}

    def generate(self, record: EditorRecord, *, use_cache: bool = True) -> dict[str, Any]:
        with self.route_generation(record) as decision:
            output = self.runtime.generate(record, use_cache=use_cache)
        return {"route": decision_as_json(decision), "generation": output}

    def save_editor_state(self, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": self.method,
                "target": self.target,
                "router": self.router.export_state(),
                "wrapper": self.wrapper.export_state(),
                "edit_history": self.edit_history,
            },
            path,
        )
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}

    def load_editor_state(self, path: Path) -> None:
        state = torch.load(path, map_location=self.runtime.device, weights_only=False)
        if state["target"] != self.target:
            raise ValueError("BalanceEdit target mismatch")
        self.router = MemoryRouter.from_state(state["router"], device=self.runtime.device)
        self.wrapper.load_exported_state(state["wrapper"])
        self.edit_history = list(state["edit_history"])

    def reset_editor_state(self) -> None:
        self.router.clear()
        self.wrapper.clear()
        self.edit_history.clear()

    def state_summary(self) -> dict[str, Any]:
        statistics = self.wrapper.parameter_statistics()
        return {
            "method": self.method,
            "entry_count": len(self.router),
            "logical_edit_ids": list(self.router.logical_ids),
            "radii": list(self.router.radii),
            "edited_parameter_count": statistics["parameter_count"],
            "edited_parameter_bytes": statistics["parameter_bytes"],
            "full_weight_copies_per_edit": 1,
            "inactive_storage": statistics["storage_mode"],
            "archived_entry_count": statistics["archived_entry_count"],
            "edit_history": list(self.edit_history),
        }

    def config_lock(self) -> dict[str, Any]:
        return {
            "schema_version": "m3bench-editor-method-config-v2",
            "method": "BalanceEdit",
            "classification": "paper-spec LLaVA-Med adaptation of locked BalanceEdit source",
            "scope": "final-layer LM MLP up_projection",
            "target": self.target,
            "key": "mean input hidden representation at edited up_projection",
            "positive": "same image plus frozen official question rephrase",
            "negative": "same-size black image plus original question",
            "distance": "euclidean from minigpt4_euc source config",
            "radius_formula": "(1-alpha)*negative_distance + alpha*positive_distance",
            "alpha": 0.2,
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "gradient_clip": 1.0,
            "steps_per_edit": 50,
            "edited_state_precision": "float32; outputs cast to frozen backbone dtype",
            "source": "BalanceEdit @ 83749e52a1d27331d21cfec845b6089294730c2f; hparams/BalancEdit/minigpt4_euc.yaml",
        }


class BeloraPaperSpecEditor(_BalanceRoutingMixin, PaperSpecEditor):
    method = "belora"
    steps_per_edit = 50

    def __init__(self, runtime: LlavaMedEditorRuntime):
        super().__init__(runtime)
        self.targets = list(runtime.target_lock["belora"]["targets"])
        self.target = next(path for path in self.targets if path.endswith("up_proj"))
        self.wrappers: dict[str, RoutedLoRALinear] = {}
        for path in self.targets:
            base = runtime.get_module(path)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"BELoRA target is not linear: {path}")
            wrapper = RoutedLoRALinear(base, rank=16, alpha=16, dropout=0.0)
            runtime.replace_module(path, wrapper)
            self.wrappers[path] = wrapper
        self.router = MemoryRouter("euclidean")
        self.edit_to_adapter: dict[str, str] = {}

    @contextmanager
    def disabled(self) -> Iterator[None]:
        previous = {path: wrapper.active_logical_id for path, wrapper in self.wrappers.items()}
        for wrapper in self.wrappers.values():
            wrapper.set_active(None)
        try:
            yield
        finally:
            for path, wrapper in self.wrappers.items():
                wrapper.set_active(previous[path])

    def _set_active(self, logical_id: str | None) -> None:
        adapter_name = self.edit_to_adapter[logical_id] if logical_id is not None else None
        for wrapper in self.wrappers.values():
            wrapper.set_active(adapter_name)

    @contextmanager
    def _activated(self, logical_id: str | None) -> Iterator[None]:
        self._set_active(logical_id)
        try:
            yield
        finally:
            self._set_active(None)

    @contextmanager
    def route_generation(self, record: EditorRecord) -> Iterator[Any]:
        decision = self._route(record)
        with self._activated(decision.logical_edit_id):
            yield decision

    def adapter_state_sha256(self, logical_id: str) -> str:
        adapter_name = self.edit_to_adapter[logical_id]
        digest = hashlib.sha256()
        for path, wrapper in sorted(self.wrappers.items()):
            slot = wrapper.logical_to_slot[adapter_name]
            digest.update(path.encode("utf-8"))
            for label, value in (("A", wrapper.lora_A[slot]), ("B", wrapper.lora_B[slot])):
                tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
                digest.update(label.encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def apply_edit(self, record: EditorRecord) -> dict[str, Any]:
        seed = record_seed(record.record_id, self.method)
        seed_everything(seed)
        key, positive, negative, black_path = self._anchors(
            record, self.runtime.run_root / "inputs/black_images"
        )
        radius = balanced_radius(key, positive, negative, alpha=0.2, distance="euclidean")
        positive_distance = float(euclidean_distances(key, positive)[0].cpu().item())
        negative_distance = float(euclidean_distances(key, negative)[0].cpu().item())
        batch = self.runtime.build_edit_batch(record)
        self.router.add(record.record_id, key, radius, batch.target_token_ids)
        adapter_name = safe_slot(record.record_id)
        if adapter_name in self.edit_to_adapter.values():
            raise RuntimeError("BELoRA adapter-name collision")
        self.edit_to_adapter[record.record_id] = adapter_name
        parameters = []
        for index, wrapper in enumerate(self.wrappers.values()):
            wrapper.add_adapter(adapter_name, seed=seed + index)
            parameters.extend(wrapper.train_only(adapter_name))
        optimizer = torch.optim.AdamW(parameters, lr=5e-5)
        losses, gradient_checks = [], []
        try:
            with self._activated(record.record_id):
                for _ in range(self.steps_per_edit):
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.runtime.compute_loss(batch)
                    loss.backward()
                    gradient_checks.append(finite_gradients(parameters))
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu().item()))
        finally:
            for wrapper in self.wrappers.values():
                wrapper.train_only(None)
            self._set_active(None)
        self.edit_history.append(record.record_id)
        return {
            "record_id": record.record_id,
            "seed": seed,
            "losses": losses,
            "finite_losses": all(torch.isfinite(torch.tensor(losses)).tolist()),
            "finite_gradients": all(gradient_checks),
            "epochs": self.steps_per_edit,
            "radius": float(radius.cpu().item()),
            "positive_distance": positive_distance,
            "negative_distance": negative_distance,
            "distance": "euclidean",
            "black_image_path": str(black_path),
            "black_image_sha256": hashlib.sha256(black_path.read_bytes()).hexdigest(),
            "router_positive_source": record.router_positive_source,
            "entry_count": len(self.router),
            "logical_adapter_set_size": len(self.wrappers),
            "adapter_name": adapter_name,
            "adapter_state_sha256": self.adapter_state_sha256(record.record_id),
            "trainable_parameter_count": sum(p.numel() for p in parameters),
            "trainable_parameter_bytes": parameter_bytes(parameters),
            "target_mask": batch.mask_report(),
        }

    def score_target_nll(self, record: EditorRecord) -> dict[str, Any]:
        decision = self._route(record)
        batch = self.runtime.build_edit_batch(record)
        with self._activated(decision.logical_edit_id):
            with torch.no_grad():
                score = self.runtime.score_target(batch)
        return {**score, "route": decision_as_json(decision), "target_mask": batch.mask_report()}

    def generate(self, record: EditorRecord, *, use_cache: bool = True) -> dict[str, Any]:
        with self.route_generation(record) as decision:
            output = self.runtime.generate(record, use_cache=use_cache)
        return {"route": decision_as_json(decision), "generation": output}

    def save_editor_state(self, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": self.method,
                "targets": self.targets,
                "router": self.router.export_state(),
                "wrappers": {path: wrapper.export_state() for path, wrapper in self.wrappers.items()},
                "edit_to_adapter": self.edit_to_adapter,
                "edit_history": self.edit_history,
            },
            path,
        )
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}

    def load_editor_state(self, path: Path) -> None:
        state = torch.load(path, map_location=self.runtime.device, weights_only=False)
        if state["targets"] != self.targets:
            raise ValueError("BELoRA target set mismatch")
        self.router = MemoryRouter.from_state(state["router"], device=self.runtime.device)
        for target, wrapper_state in state["wrappers"].items():
            self.wrappers[target].load_exported_state(wrapper_state)
        self.edit_to_adapter = dict(state["edit_to_adapter"])
        self.edit_history = list(state["edit_history"])

    def reset_editor_state(self) -> None:
        self.router.clear()
        for wrapper in self.wrappers.values():
            wrapper.lora_A = nn.ParameterDict()
            wrapper.lora_B = nn.ParameterDict()
            wrapper.logical_to_slot.clear()
            wrapper.slot_to_logical.clear()
            wrapper.set_active(None)
        self.edit_to_adapter.clear()
        self.edit_history.clear()

    def state_summary(self) -> dict[str, Any]:
        parameters = [
            parameter
            for wrapper in self.wrappers.values()
            for parameter in list(wrapper.lora_A.values()) + list(wrapper.lora_B.values())
        ]
        return {
            "method": self.method,
            "implementation_label": "BELORA_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V2_EFFECT_REPAIRED",
            "entry_count": len(self.router),
            "logical_edit_ids": list(self.router.logical_ids),
            "radii": list(self.router.radii),
            "wrapped_linear_count": len(self.wrappers),
            "adapter_parameter_count": sum(p.numel() for p in parameters),
            "adapter_parameter_bytes": parameter_bytes(parameters),
            "full_weight_copies": 0,
            "edit_to_adapter": dict(self.edit_to_adapter),
            "adapter_hashes": {
                logical_id: self.adapter_state_sha256(logical_id)
                for logical_id in self.edit_history
            },
            "edit_history": list(self.edit_history),
        }

    def config_lock(self) -> dict[str, Any]:
        return {
            "schema_version": "m3bench-editor-method-config-v2",
            "method": "BELoRA",
            "implementation_label": "BELORA_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V2_EFFECT_REPAIRED",
            "classification": "independent paper-spec reimplementation; not author implementation",
            "scope": "final-layer LM MLP internal linears",
            "targets": self.targets,
            "routing": "BalanceEdit pooled-key nearest route",
            "distance": "euclidean from BalanceEdit minigpt4_euc source config",
            "radius_formula": "(1-alpha)*negative_distance + alpha*positive_distance",
            "alpha": 0.2,
            "rank": 16,
            "lora_alpha": 16,
            "dropout": 0.0,
            "optimizer": "AdamW",
            "learning_rate": 5e-5,
            "batch_size": 1,
            "gradient_clip": 1.0,
            "epochs_per_edit": self.steps_per_edit,
            "paper_spec_deviation": True,
            "paper_spec_epochs_per_edit": 5,
            "deviation_reason": "approved 8-record effect gate: 5-20 steps were generation no-op; 50 was the first tested checkpoint that changed generation",
            "update_storage": "per-edit LoRA parameters only; no full module copies",
            "projector": "excluded from primary smoke",
            "vision_encoder": "excluded",
            "author_runtime": "unavailable",
        }


def create_editor(
    method: str,
    runtime: LlavaMedEditorRuntime,
    *,
    balancedit_inactive_store_dir: Path | None = None,
    lora_config: LoraRuntimeConfig | None = None,
) -> PaperSpecEditor:
    normalized = method.lower()
    if normalized == "lora":
        return LoraPaperSpecEditor(runtime, config=lora_config)
    if normalized == "grace":
        return GracePaperSpecEditor(runtime)
    if normalized in {"balancedit", "be"}:
        return BalanceEditPaperSpecEditor(runtime, inactive_store_dir=balancedit_inactive_store_dir)
    if normalized == "belora":
        return BeloraPaperSpecEditor(runtime)
    raise ValueError(f"unknown editor method: {method}")
