"""LLaVA-Med-v1.5-Mistral-7B adapter using the checkpoint's native conversation path."""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from m3bench_repro.models import VLMAdapter
from .preprocessing import deterministic_process_image


def install_llava_mistral_transformers_compat(model_class: type[Any]) -> bool:
    """Bridge the pinned legacy LLaVA-Med forward API to Transformers 4.51.

    Newer Transformers generation passes ``cache_position`` and
    ``logits_to_keep``.  The pinned LLaVA-Med class predates both keywords.
    They are optional optimization hints: the inherited Mistral model infers a
    dynamic cache position when omitted, and computing all logits preserves
    the legacy behavior.  Keeping this shim in the adapter leaves the pinned
    external LLaVA-Med checkout unchanged.
    """
    if "cache_position" in inspect.signature(model_class.forward).parameters:
        return False
    if getattr(model_class, "_m3bench_transformers_compat_installed", False):
        return False

    legacy_forward = model_class.forward

    def compatible_forward(
        self: Any,
        *args: Any,
        attention_mask: Any = None,
        cache_position: Any = None,
        logits_to_keep: Any = 0,
        **kwargs: Any,
    ) -> Any:
        del cache_position, logits_to_keep
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        return legacy_forward(self, *args, **kwargs)

    compatible_forward.__name__ = legacy_forward.__name__
    compatible_forward.__doc__ = legacy_forward.__doc__
    model_class.forward = compatible_forward
    model_class._m3bench_transformers_compat_installed = True
    return True


class GenerationSequenceContract(str, Enum):
    """How ``generate`` returns token sequences for a model family."""

    PROMPT_AND_CONTINUATION = "prompt_and_continuation"
    CONTINUATION_ONLY = "continuation_only"


@dataclass(frozen=True)
class GenerationResult:
    """Raw generated ids plus the contract-aware decoded answer."""

    raw_token_ids: tuple[int, ...]
    decoded_text: str
    contract: GenerationSequenceContract


def build_llava_med_query(question: str, *, mm_use_im_start_end: bool) -> str:
    """Build the native LLaVA-Med user turn without any target answer."""
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

    if not question.strip():
        raise ValueError("question must be non-empty")
    image_marker = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if mm_use_im_start_end
        else DEFAULT_IMAGE_TOKEN
    )
    return f"{image_marker}\n{question}"


def count_image_token_ids(token_ids: list[int] | tuple[int, ...], image_token_index: int) -> int:
    """Pure helper retained for source-only prompt-contract tests."""
    return sum(token_id == image_token_index for token_id in token_ids)


def decode_generation_sequence(
    tokenizer: Any,
    token_ids: list[int] | tuple[int, ...],
    *,
    contract: GenerationSequenceContract,
    prompt_token_count: int | None = None,
) -> GenerationResult:
    """Decode exactly the token sequence supplied by this model family.

    LLaVA-Med's multimodal ``generate`` path feeds ``inputs_embeds`` to
    Transformers and returns a continuation-only sequence.  Applying a text
    prompt-length slice to that sequence silently deletes the answer.
    """
    ids = tuple(int(token_id) for token_id in token_ids)
    if contract is GenerationSequenceContract.PROMPT_AND_CONTINUATION:
        if prompt_token_count is None:
            raise ValueError("prompt_token_count is required for a prefixed sequence")
        ids = ids[prompt_token_count:]
    return GenerationResult(
        raw_token_ids=ids,
        decoded_text=tokenizer.decode(ids, skip_special_tokens=True).strip(),
        contract=contract,
    )


class LlavaMedAdapter(VLMAdapter):
    conversation_mode = "mistral_instruct"
    generation_sequence_contract = GenerationSequenceContract.CONTINUATION_ONLY

    def __init__(self, model_path: str | Path, vision_tower_path: str | Path, device: str = "cuda:0"):
        self.model_path = Path(model_path)
        self.vision_tower_path = Path(vision_tower_path)
        self.device = torch.device(device)
        self.tokenizer = self.model = self.image_processor = None

    def load(self) -> None:
        # The cached vision tower is deliberately passed as an explicit local path:
        # no model component is implicitly fetched from the network during a run.
        from llava.model import LlavaMistralForCausalLM
        from llava.utils import disable_torch_init

        if not self.model_path.is_dir() or not self.vision_tower_path.is_dir():
            raise FileNotFoundError("LLaVA-Med checkpoint or local vision-tower cache is absent")
        install_llava_mistral_transformers_compat(LlavaMistralForCausalLM)
        disable_torch_init()
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        config.mm_vision_tower = str(self.vision_tower_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, local_files_only=True)
        self.model = LlavaMistralForCausalLM.from_pretrained(
            self.model_path, config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True
        )
        tower = self.model.get_vision_tower()
        if not tower.is_loaded:
            tower.load_model()
        tower.to(device=self.device, dtype=torch.float16)
        self.model.model.mm_projector.to(device=self.device, dtype=torch.float16)
        self.model.to(device=self.device, dtype=torch.float16).eval()
        self.image_processor = tower.image_processor

    def _prompt(self, question: str, answer: str | None = None) -> str:
        from llava.conversation import conv_templates

        conv = conv_templates[self.conversation_mode].copy()
        query = build_llava_med_query(
            question, mm_use_im_start_end=bool(self.model.config.mm_use_im_start_end)
        )
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], answer)
        return conv.get_prompt()

    def prepare_inputs(self, image_path: str | Path, question: str, answer: str | None = None) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("call load() first")
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        prompt = self._prompt(question, answer)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        token_ids = [int(token_id) for token_id in input_ids[0].tolist()]
        if count_image_token_ids(token_ids, IMAGE_TOKEN_INDEX) != 1:
            raise RuntimeError("native LLaVA-Med prompt must contain exactly one image token")
        image_tensor, seed, image_hash = deterministic_process_image(image_path, self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [x.to(self.device, dtype=torch.float16) for x in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        return {"prompt": prompt, "input_ids": input_ids, "attention_mask": attention_mask, "images": image_tensor,
                "preprocessing_seed": seed, "image_sha256": image_hash}

    def generate_with_result(
        self, image_path: str | Path, question: str, generation_config: dict[str, Any]
    ) -> GenerationResult:
        batch = self.prepare_inputs(image_path, question)
        kwargs = {
            key: value
            for key, value in generation_config.items()
            if value is not None
            and key in {"do_sample", "num_beams", "max_new_tokens", "use_cache", "eos_token_id", "pad_token_id"}
        }
        if kwargs.get("pad_token_id") is None:
            kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        with torch.inference_mode():
            output = self.model.generate(
                batch["input_ids"], attention_mask=batch["attention_mask"], images=batch["images"], **kwargs
            )
        return decode_generation_sequence(
            self.tokenizer,
            output[0].detach().cpu().tolist(),
            contract=self.generation_sequence_contract,
            prompt_token_count=int(batch["input_ids"].shape[1]),
        )

    def generate(self, image_path: str | Path, question: str, generation_config: dict[str, Any]) -> str:
        return self.generate_with_result(image_path, question, generation_config).decoded_text

    def teacher_forced_loss(self, image_path: str | Path, question: str, target: str):
        batch = self.prepare_inputs(image_path, question, target)
        prefix = self.prepare_inputs(image_path, question, None)["input_ids"].shape[1]
        labels = batch["input_ids"].clone()
        labels[:, :prefix] = -100
        return self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], images=batch["images"], labels=labels
        ).loss

    def get_named_modules(self) -> dict[str, Any]:
        return dict(self.model.named_modules())

    def get_editable_mlp_modules(self) -> dict[str, Any]:
        return {n: m for n, m in self.model.named_modules() if ".mlp." in n and n.endswith(("gate_proj", "up_proj", "down_proj"))}

    def base_checksum(self) -> str:
        """Stable sampled parameter checksum for reset evidence without a 15-GB GPU-to-CPU copy."""
        digest = hashlib.sha256()
        for name, value in self.model.named_parameters():
            if "lora_" not in name:
                digest.update(name.encode())
                flat = value.detach().reshape(-1)
                # First and last blocks catch changes throughout each matrix while keeping
                # the smoke gate practical on a shared GPU. Full checkpoint file SHA256s
                # are also retained in the run provenance for immutable-base evidence.
                sample = torch.cat((flat[:64], flat[-64:])).float().cpu().numpy().tobytes()
                digest.update(sample)
        return digest.hexdigest()

    def clone_or_reload_base(self) -> None:
        self.model = self.tokenizer = self.image_processor = None
        torch.cuda.empty_cache()
        self.load()
