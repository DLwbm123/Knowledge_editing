"""Shared Foundation-compatible LLaVA-Med runtime for all four editors."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from m3bench_repro.inference.llava_med import LlavaMedAdapter


LLAVA_SOURCE = Path(os.environ.get("M3BENCH_LLAVA_SOURCE", "external/LLaVA-Med"))
MODEL_PATH = Path(os.environ.get("M3BENCH_MODEL_PATH", "models/llava-med"))
VISION_PATH = Path(os.environ.get("M3BENCH_VISION_PATH", "models/clip-vit-large-patch14-336"))
RUN_ROOT = Path(os.environ.get("M3BENCH_RUN_ROOT", "outputs/m3bench-editor-runtime"))
FROZEN_GENERATION_CONFIG = RUN_ROOT / (
    "carry_forward/foundation_v4/runtime/llava_med_generation_frozen.json"
)
IGNORE_INDEX = -100
MODULE_PATTERN = re.compile(
    r"^model\.layers\.(?P<block>\d+)\.mlp\.(?P<projection>gate_proj|up_proj|down_proj)$"
)


def select_unique_outer_module(candidates: list[str], *, label: str) -> str:
    """Resolve one outer module when named_modules also exposes nested wrappers."""
    unique = sorted(set(candidates), key=lambda value: (value.count("."), value))
    roots = [
        candidate
        for candidate in unique
        if not any(
            candidate.startswith(other + ".")
            for other in unique
            if other != candidate
        )
    ]
    if len(roots) != 1:
        raise RuntimeError(f"ambiguous {label} module roots: candidates={unique}, roots={roots}")
    return roots[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path: Path, obj: object, *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    if read_only:
        os.chmod(path, 0o444)


def build_target_only_labels(
    full_input_ids: torch.Tensor,
    prefix_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    image_token_index: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Mask the exact question/image prefix and padding, leaving only target completion tokens."""
    if full_input_ids.ndim != 2 or prefix_input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input IDs and attention mask must be rank-2")
    if full_input_ids.shape != attention_mask.shape or full_input_ids.shape[0] != 1:
        raise ValueError("target-only runtime currently requires batch size 1")
    prefix_length = prefix_input_ids.shape[1]
    if full_input_ids.shape[1] <= prefix_length:
        raise RuntimeError("target answer produced no tokens")
    if not torch.equal(full_input_ids[:, :prefix_length], prefix_input_ids):
        raise RuntimeError("full prompt does not preserve the exact question-only token prefix")
    labels = full_input_ids.clone()
    labels[:, :prefix_length] = IGNORE_INDEX
    labels[attention_mask == 0] = IGNORE_INDEX
    if torch.any(labels == image_token_index):
        raise RuntimeError("image token leaked into target labels")
    target_ids = tuple(int(x) for x in labels[0][labels[0] != IGNORE_INDEX].tolist())
    if not target_ids:
        raise RuntimeError("target mask is empty")
    return labels, target_ids


@dataclass(frozen=True)
class EditorRecord:
    record_id: str
    dataset: str
    question: str
    target: str
    official_rephrase: str
    image_path: Path
    relative_image_path: str
    formal_sequence_position: int
    question_type: str
    router_positive_source: str = "frozen_official_rephrase"

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EditorRecord":
        return cls(
            record_id=row["record_id"],
            dataset=row["dataset"],
            question=row["question"],
            target=row["gold_answer"],
            official_rephrase=row["official_rephrase"],
            image_path=Path(row["image_path"]),
            relative_image_path=row["relative_image_path"],
            formal_sequence_position=int(row["formal_sequence_position"]),
            question_type=row["question_type"],
            router_positive_source=row.get("router_positive_source", "frozen_official_rephrase"),
        )


@dataclass
class PreparedBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor | None
    labels: torch.Tensor
    raw_input_ids: torch.Tensor
    raw_labels: torch.Tensor
    target_token_ids: tuple[int, ...]
    target_start_expanded: int | None
    key_token_index_text: int
    key_token_text: str
    key_token_index: int
    image_token_count: int
    image_tensor_shape: list[int] | list[list[int]]
    prompt: str
    image_sha256: str
    preprocessing_seed: int

    def forward_kwargs(self) -> dict[str, Any]:
        return {
            "inputs_embeds": self.inputs_embeds,
            "attention_mask": self.attention_mask,
            "position_ids": self.position_ids,
            "labels": self.labels,
            "use_cache": False,
            "return_dict": True,
        }

    def mask_report(self) -> dict[str, Any]:
        target_positions = torch.where(self.labels[0] != IGNORE_INDEX)[0]
        attention = (
            self.attention_mask[0].bool()
            if self.attention_mask is not None
            else torch.ones(self.labels.shape[1], dtype=torch.bool, device=self.labels.device)
        )
        padding_positions = ~attention
        first_target = int(target_positions[0].item()) if len(target_positions) else None
        return {
            "expanded_sequence_length": int(self.labels.shape[1]),
            "target_token_count": int(len(target_positions)),
            "target_token_ids": list(self.target_token_ids),
            "target_start_expanded": first_target,
            "key_token_index_text": self.key_token_index_text,
            "key_token_text": self.key_token_text,
            "key_token_role": "last_non_padding_prompt_text_token",
            "key_token_index": self.key_token_index,
            "text_space_sequence_length": int(self.raw_input_ids.shape[1]),
            "image_token_count": self.image_token_count,
            "image_tensor_shape": self.image_tensor_shape,
            "multimodal_expansion_verified": self.labels.shape[1] > self.raw_input_ids.shape[1],
            "prompt_and_image_positions_masked": bool(
                first_target is not None
                and torch.all(self.labels[0, :first_target] == IGNORE_INDEX).item()
            ),
            "padding_positions_ignored": bool(
                torch.all(self.labels[0, padding_positions] == IGNORE_INDEX).item()
            ),
            "target_positions_unmasked": bool(
                len(target_positions) > 0
                and torch.all(self.labels[0, target_positions] != IGNORE_INDEX).item()
            ),
        }


@dataclass
class BaseParameterGuard:
    names: list[str]
    parameters: list[nn.Parameter]
    samples: list[str]

    @staticmethod
    def _sample_hash(name: str, parameter: nn.Parameter) -> str:
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(str(parameter.dtype).encode("ascii"))
        flat = parameter.detach().reshape(-1)
        if flat.numel() <= 256:
            sample = flat
        else:
            indices = torch.linspace(
                0, flat.numel() - 1, steps=256, device=flat.device, dtype=torch.float64
            ).long()
            sample = flat.index_select(0, indices)
        digest.update(sample.to(dtype=torch.float32, device="cpu").numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def capture(cls, model: nn.Module) -> "BaseParameterGuard":
        names, parameters, samples = [], [], []
        for name, parameter in model.named_parameters():
            names.append(name)
            parameters.append(parameter)
            samples.append(cls._sample_hash(name, parameter))
        return cls(names, parameters, samples)

    @property
    def aggregate_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, sample in zip(self.names, self.samples, strict=True):
            digest.update(name.encode("utf-8"))
            digest.update(sample.encode("ascii"))
        return digest.hexdigest()

    def verify(self) -> dict[str, Any]:
        changed = []
        trainable = []
        current_samples = []
        for name, parameter, expected in zip(
            self.names, self.parameters, self.samples, strict=True
        ):
            actual = self._sample_hash(name, parameter)
            current_samples.append(actual)
            if actual != expected:
                changed.append(name)
            if parameter.requires_grad:
                trainable.append(name)
        digest = hashlib.sha256()
        for name, sample in zip(self.names, current_samples, strict=True):
            digest.update(name.encode("utf-8"))
            digest.update(sample.encode("ascii"))
        return {
            "unchanged": not changed,
            "changed_parameters": changed,
            "base_parameters_requiring_grad": trainable,
            "before_sha256": self.aggregate_sha256,
            "after_sha256": digest.hexdigest(),
            "parameter_count": len(self.parameters),
        }


class LlavaMedEditorRuntime:
    """Single shared multimodal runtime used by every editor implementation."""

    def __init__(
        self,
        *,
        device: str,
        run_root: Path = RUN_ROOT,
        model_path: Path = MODEL_PATH,
        vision_path: Path = VISION_PATH,
        generation_config_path: Path | None = None,
        loader_mode: str = "project",
    ):
        self.device = torch.device(device)
        self.run_root = Path(run_root)
        self.model_path = Path(model_path)
        self.vision_path = Path(vision_path)
        self.generation_config_path = Path(
            generation_config_path or FROZEN_GENERATION_CONFIG
        )
        self.adapter = LlavaMedAdapter(
            self.model_path,
            self.vision_path,
            device=device,
            load_mode=loader_mode,
        )
        self.generation_config = json.loads(self.generation_config_path.read_text(encoding="utf-8"))
        self.inventory: dict[str, Any] | None = None
        self.target_lock: dict[str, Any] | None = None
        self.base_guard: BaseParameterGuard | None = None

    @property
    def model(self) -> nn.Module:
        if self.adapter.model is None:
            raise RuntimeError("runtime is not loaded")
        return self.adapter.model

    def llava_model(self) -> nn.Module:
        candidate = self.model
        if hasattr(candidate, "get_base_model"):
            candidate = candidate.get_base_model()
        if hasattr(candidate, "prepare_inputs_labels_for_multimodal"):
            return candidate
        if hasattr(candidate, "base_model") and hasattr(
            candidate.base_model, "prepare_inputs_labels_for_multimodal"
        ):
            return candidate.base_model
        raise RuntimeError("cannot resolve underlying LLaVA-Med model")

    def load_frozen_backbone(self, *, seed: int) -> None:
        seed_everything(seed)
        source = str(LLAVA_SOURCE)
        if source not in sys.path:
            sys.path.insert(0, source)
        self.adapter.load()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.base_guard = BaseParameterGuard.capture(self.model)

    def capture_base_guard(self) -> BaseParameterGuard:
        self.base_guard = BaseParameterGuard.capture(self.llava_model())
        return self.base_guard

    def resolve_module_inventory(self, *, freeze: bool = True) -> tuple[dict, dict]:
        modules = dict(self.llava_model().named_modules())
        linears = []
        for name, module in modules.items():
            match = MODULE_PATTERN.fullmatch(name)
            if match and isinstance(module, nn.Linear):
                linears.append(
                    {
                        "path": name,
                        "block": int(match.group("block")),
                        "projection": match.group("projection"),
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "bias": module.bias is not None,
                        "parameter_count": sum(p.numel() for p in module.parameters()),
                        "dtype": str(module.weight.dtype),
                        "device": str(module.weight.device),
                    }
                )
        linears.sort(key=lambda item: (item["block"], item["projection"]))
        blocks = sorted({item["block"] for item in linears})
        if not blocks or blocks != list(range(max(blocks) + 1)):
            raise RuntimeError(f"non-contiguous language block inventory: {blocks}")
        by_block = {block: [] for block in blocks}
        for item in linears:
            by_block[item["block"]].append(item)
        for block, items in by_block.items():
            projections = {item["projection"] for item in items}
            if projections != {"gate_proj", "up_proj", "down_proj"}:
                raise RuntimeError(f"incomplete MLP inventory in block {block}: {projections}")
        final_block = max(blocks)
        final_prefix = f"model.layers.{final_block}.mlp"
        projector_candidates = [name for name in modules if name.endswith("mm_projector")]
        vision_candidates = [name for name in modules if name.endswith("vision_tower")]
        projector_path = select_unique_outer_module(projector_candidates, label="projector")
        vision_path = select_unique_outer_module(vision_candidates, label="vision encoder")
        inventory = {
            "schema_version": "m3bench-llava-med-module-inventory-v1",
            "classification": "M3BENCH_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V2_EFFECT_REPAIRED",
            "model_class": self.llava_model().__class__.__name__,
            "model_path": str(self.model_path),
            "vision_tower_cache": str(self.vision_path),
            "language_block_count": len(blocks),
            "language_blocks": blocks,
            "final_block_path": f"model.layers.{final_block}",
            "final_mlp_path": final_prefix,
            "candidate_internal_linears": linears,
            "projector_candidates": sorted(projector_candidates),
            "projector_path": projector_path,
            "vision_encoder_candidates": sorted(vision_candidates),
            "vision_encoder_path": vision_path,
            "model_dtype": str(next(self.llava_model().parameters()).dtype),
            "device": str(self.device),
            "total_model_parameters": sum(p.numel() for p in self.llava_model().parameters()),
        }
        all_mlp = [item["path"] for item in linears]
        final_mlp = [
            f"{final_prefix}.gate_proj",
            f"{final_prefix}.up_proj",
            f"{final_prefix}.down_proj",
        ]
        target_lock = {
            "schema_version": "m3bench-llava-med-edit-target-lock-v1",
            "inventory_sha256": canonical_sha256(inventory),
            "lora": {
                "scope": "all language-model MLP blocks",
                "targets": all_mlp,
                "target_count": len(all_mlp),
            },
            "grace": {
                "scope": "final-layer LM up_projection adaptor",
                "targets": [f"{final_prefix}.up_proj"],
            },
            "balancedit": {
                "scope": "final-layer LM MLP up_projection",
                "targets": [f"{final_prefix}.up_proj"],
            },
            "belora": {
                "scope": "final-layer LM MLP internal linears",
                "targets": final_mlp,
                "logical_edit_activates_coordinated_set": True,
            },
            "projector_excluded": projector_path,
            "vision_encoder_excluded": vision_path,
        }
        target_lock["target_lists_sha256"] = canonical_sha256(
            {method: value.get("targets", []) for method, value in target_lock.items() if isinstance(value, dict)}
        )
        self.inventory, self.target_lock = inventory, target_lock
        if freeze:
            runtime_dir = self.run_root / "runtime"
            write_json_atomic(runtime_dir / "LLAVA_MED_MODULE_INVENTORY.json", inventory)
            write_json_atomic(runtime_dir / "LLAVA_MED_EDIT_TARGET_LOCK.json", target_lock)
        return inventory, target_lock

    def get_module(self, path: str) -> nn.Module:
        modules = dict(self.llava_model().named_modules())
        if path not in modules:
            raise KeyError(f"module path not present: {path}")
        return modules[path]

    def replace_module(self, path: str, replacement: nn.Module) -> nn.Module:
        parent_path, child_name = path.rsplit(".", 1)
        parent = self.get_module(parent_path)
        original = getattr(parent, child_name)
        setattr(parent, child_name, replacement)
        return original

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        return tokenizer_image_token(
            prompt, self.adapter.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

    def _expand_multimodal(
        self,
        *,
        raw_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        images: torch.Tensor | list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        tensors = images if isinstance(images, list) else [images]
        if not tensors or any(not isinstance(item, torch.Tensor) or item.numel() == 0 for item in tensors):
            raise RuntimeError("multimodal edit requires a non-empty image tensor")
        core = self.llava_model()
        with torch.no_grad():
            _, position_ids, expanded_attention, _, inputs_embeds, expanded_labels = (
                core.prepare_inputs_labels_for_multimodal(
                    raw_input_ids,
                    None,
                    attention_mask,
                    None,
                    labels,
                    images,
                    image_sizes=None,
                )
            )
        if inputs_embeds is None or expanded_labels is None:
            raise RuntimeError("multimodal expansion failed")
        if inputs_embeds.shape[1] <= raw_input_ids.shape[1]:
            raise RuntimeError("image token was not expanded into multimodal embeddings")
        if expanded_labels.shape[:2] != inputs_embeds.shape[:2]:
            raise RuntimeError("expanded labels and embeddings are misaligned")
        return (
            inputs_embeds.detach(),
            expanded_attention,
            position_ids,
            expanded_labels.detach(),
        )

    def build_edit_batch(self, record: EditorRecord) -> PreparedBatch:
        full = self.adapter.prepare_inputs(record.image_path, record.question, record.target)
        prefix_prompt = self.adapter._prompt(record.question, None)
        prefix_ids = self._tokenize_prompt(prefix_prompt)
        raw_input_ids = full["input_ids"]
        from llava.constants import IMAGE_TOKEN_INDEX
        raw_labels, target_ids = build_target_only_labels(
            raw_input_ids,
            prefix_ids,
            full["attention_mask"],
            image_token_index=IMAGE_TOKEN_INDEX,
        )
        inputs_embeds, attention, position_ids, labels = self._expand_multimodal(
            raw_input_ids=raw_input_ids,
            attention_mask=full["attention_mask"],
            labels=raw_labels,
            images=full["images"],
        )
        target_positions = torch.where(labels[0] != IGNORE_INDEX)[0]
        if not len(target_positions):
            raise RuntimeError("expanded target mask is empty")
        target_start = int(target_positions[0].item())
        key_index = target_start - 1
        if key_index < 0:
            raise RuntimeError("expanded prompt is empty")
        expanded_target_ids = tuple(int(x) for x in labels[0, target_positions].tolist())
        if expanded_target_ids != target_ids:
            raise RuntimeError("multimodal expansion changed target token IDs")
        image_count = int((raw_input_ids == IMAGE_TOKEN_INDEX).sum().item())
        if image_count != 1:
            raise RuntimeError("edit batch must contain exactly one image token")
        images = full["images"] if isinstance(full["images"], list) else [full["images"]]
        image_shape: list[int] | list[list[int]] = (
            [list(item.shape) for item in images]
            if isinstance(full["images"], list)
            else list(full["images"].shape)
        )
        return PreparedBatch(
            inputs_embeds=inputs_embeds,
            attention_mask=attention,
            position_ids=position_ids,
            labels=labels,
            raw_input_ids=raw_input_ids,
            raw_labels=raw_labels,
            target_token_ids=target_ids,
            target_start_expanded=target_start,
            key_token_index_text=prefix_ids.shape[1] - 1,
            key_token_text=self.adapter.tokenizer.decode(
                [int(prefix_ids[0, -1].item())], skip_special_tokens=False
            ),
            key_token_index=key_index,
            image_token_count=image_count,
            image_tensor_shape=image_shape,
            prompt=full["prompt"],
            image_sha256=full["image_sha256"],
            preprocessing_seed=int(full["preprocessing_seed"]),
        )

    def build_question_batch(
        self,
        record: EditorRecord,
        *,
        question: str | None = None,
        image_path: Path | None = None,
    ) -> PreparedBatch:
        selected_question = question if question is not None else record.question
        selected_image = image_path if image_path is not None else record.image_path
        raw = self.adapter.prepare_inputs(selected_image, selected_question, None)
        raw_input_ids = raw["input_ids"]
        from llava.constants import IMAGE_TOKEN_INDEX
        raw_labels = torch.full_like(raw_input_ids, IGNORE_INDEX)
        inputs_embeds, attention, position_ids, labels = self._expand_multimodal(
            raw_input_ids=raw_input_ids,
            attention_mask=raw["attention_mask"],
            labels=raw_labels,
            images=raw["images"],
        )
        if attention is not None:
            valid_positions = torch.where(attention[0].bool())[0]
            key_index = int(valid_positions[-1].item())
        else:
            key_index = inputs_embeds.shape[1] - 1
        image_count = int((raw_input_ids == IMAGE_TOKEN_INDEX).sum().item())
        if image_count != 1:
            raise RuntimeError("question batch must contain exactly one image token")
        image_shape = (
            [list(item.shape) for item in raw["images"]]
            if isinstance(raw["images"], list)
            else list(raw["images"].shape)
        )
        return PreparedBatch(
            inputs_embeds=inputs_embeds,
            attention_mask=attention,
            position_ids=position_ids,
            labels=labels,
            raw_input_ids=raw_input_ids,
            raw_labels=raw_labels,
            target_token_ids=(),
            target_start_expanded=None,
            key_token_index_text=raw_input_ids.shape[1] - 1,
            key_token_text=self.adapter.tokenizer.decode(
                [int(raw_input_ids[0, -1].item())], skip_special_tokens=False
            ),
            key_token_index=key_index,
            image_token_count=image_count,
            image_tensor_shape=image_shape,
            prompt=raw["prompt"],
            image_sha256=raw["image_sha256"],
            preprocessing_seed=int(raw["preprocessing_seed"]),
        )

    def make_black_image(self, record: EditorRecord, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(record.record_id.encode("utf-8")).hexdigest()[:24] + ".png"
        path = output_dir / name
        if not path.exists():
            with Image.open(record.image_path) as image:
                size = image.convert("RGB").size
            Image.new("RGB", size, color=(0, 0, 0)).save(path, format="PNG", optimize=False)
        return path

    def compute_loss(self, batch: PreparedBatch) -> torch.Tensor:
        output = self.model(**batch.forward_kwargs())
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite edit loss: {loss}")
        return loss

    def extract_layer_input_key(
        self,
        batch: PreparedBatch,
        *,
        module_path: str,
        pooling: str,
    ) -> torch.Tensor:
        module = self.get_module(module_path)
        captured: list[torch.Tensor] = []

        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("editable module input was not a tensor")
            captured.append(args[0].detach())

        handle = module.register_forward_pre_hook(hook)
        try:
            kwargs = batch.forward_kwargs()
            kwargs["labels"] = None
            with torch.no_grad():
                self.model(**kwargs)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one module activation, captured {len(captured)}")
        activation = captured[0]
        if activation.ndim != 3 or activation.shape[0] != 1:
            raise RuntimeError(f"unexpected activation shape: {tuple(activation.shape)}")
        if pooling == "last_prompt":
            key = activation[0, batch.key_token_index]
        elif pooling == "mean":
            if batch.attention_mask is None:
                key = activation[0].mean(dim=0)
            else:
                key = activation[0, batch.attention_mask[0].bool()].mean(dim=0)
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        return key.to(dtype=torch.float32)

    def generate(
        self,
        record: EditorRecord,
        *,
        question: str | None = None,
        use_cache: bool | None = None,
    ) -> dict[str, Any]:
        config = dict(self.generation_config)
        if use_cache is not None:
            config["use_cache"] = bool(use_cache)
        result = self.adapter.generate_with_result(
            record.image_path,
            question if question is not None else record.question,
            config,
        )
        return {
            "decoded_text": result.decoded_text,
            "raw_token_ids": list(result.raw_token_ids),
            "sequence_contract": result.contract.value,
            "use_cache": config["use_cache"],
        }

    def verify_target_mask_determinism(self, record: EditorRecord) -> dict[str, Any]:
        first = self.build_edit_batch(record)
        second = self.build_edit_batch(record)
        first_report = first.mask_report()
        second_report = second.mask_report()
        result = {
            "same_raw_input_ids": torch.equal(first.raw_input_ids, second.raw_input_ids),
            "same_raw_labels": torch.equal(first.raw_labels, second.raw_labels),
            "same_expanded_labels": torch.equal(first.labels, second.labels),
            "same_target_token_ids": first.target_token_ids == second.target_token_ids,
            "first": first_report,
            "second": second_report,
        }
        result["pass"] = all(
            result[key]
            for key in (
                "same_raw_input_ids",
                "same_raw_labels",
                "same_expanded_labels",
                "same_target_token_ids",
            )
        ) and all(
            first_report[key]
            for key in (
                "prompt_and_image_positions_masked",
                "padding_positions_ignored",
                "target_positions_unmasked",
            )
        )
        return result

    @contextmanager
    def peak_memory(self) -> Iterator[dict[str, int]]:
        if self.device.type != "cuda":
            result = {"allocated_bytes": 0, "reserved_bytes": 0}
            yield result
            return
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        torch.cuda.set_device(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)
        result: dict[str, int] = {}
        try:
            yield result
        finally:
            torch.cuda.synchronize(device_index)
            result["allocated_bytes"] = int(torch.cuda.max_memory_allocated(device_index))
            result["reserved_bytes"] = int(torch.cuda.max_memory_reserved(device_index))


def load_frozen_smoke_records(path: Path | None = None) -> list[EditorRecord]:
    source = path or RUN_ROOT / "inputs/APPROVED_SMOKE_RECORDS_WITH_REPHRASES.jsonl"
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [EditorRecord.from_dict(row) for row in rows]
    if len(records) != 8 or len({record.record_id for record in records}) != 8:
        raise RuntimeError("frozen smoke record invariant failure")
    return records
