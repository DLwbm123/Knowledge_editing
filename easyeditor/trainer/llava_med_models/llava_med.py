import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers.utils import ModelOutput


@dataclass
class LlavaMedOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.IntTensor = None
    attention_mask: torch.IntTensor = None
    vision_mask: torch.IntTensor = None
    prompt_mask: torch.IntTensor = None
    answer_mask: torch.IntTensor = None


def _normalize_device(device: Any) -> str:
    if isinstance(device, torch.device):
        return str(device)
    text = str(device)
    if text.isdigit():
        return f"cuda:{text}" if torch.cuda.is_available() else "cpu"
    if text == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return text


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower(), torch.float16)


def _as_list(value: Any, batch_size: int) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value for _ in range(batch_size)]


def _target_from_labels(tokenizer: Any, labels: torch.Tensor, row: int) -> str:
    ids = labels[row]
    ids = ids[ids.ne(-100)].detach().cpu().tolist()
    return tokenizer.decode(ids, skip_special_tokens=True) if ids else ""


def build_llava_med_masks(
    token_ids: torch.Tensor,
    labels: torch.Tensor,
    expanded_attention_mask: torch.Tensor,
    image_token_index: int,
    image_feature_len: int,
) -> Dict[str, torch.Tensor]:
    """Build full-sequence DSCA masks after LLaVA image-token expansion."""
    if token_ids.dim() != 1:
        raise RuntimeError(f"LLaVA-Med mask builder expects flat token ids, got {tuple(token_ids.shape)}.")
    if expanded_attention_mask.dim() != 1:
        raise RuntimeError("LLaVA-Med expanded attention mask must be one-dimensional.")
    image_positions = torch.where(token_ids == int(image_token_index))[0]
    if image_positions.numel() != 1:
        raise RuntimeError(f"LLaVA-Med expects exactly one image token, found {int(image_positions.numel())}.")
    if image_feature_len <= 0:
        raise RuntimeError(f"LLaVA-Med image feature length must be positive, got {image_feature_len}.")

    seq_len = int(expanded_attention_mask.shape[0])
    image_start = int(image_positions[0].item())
    image_end = image_start + int(image_feature_len)
    if image_end > seq_len:
        raise RuntimeError(
            f"LLaVA-Med image span [{image_start}, {image_end}) exceeds expanded sequence length {seq_len}."
        )

    attention_mask = expanded_attention_mask.bool()
    vision_mask = torch.zeros(seq_len, dtype=torch.bool, device=attention_mask.device)
    vision_mask[image_start:image_end] = True
    answer_mask = labels.ne(-100) & attention_mask
    prompt_mask = attention_mask & ~vision_mask & ~answer_mask
    if not vision_mask.any():
        raise RuntimeError("LLaVA-Med vision_mask is empty.")
    if not prompt_mask.any():
        raise RuntimeError("LLaVA-Med prompt_mask is empty.")
    if (vision_mask & prompt_mask).any() or (vision_mask & answer_mask).any() or (prompt_mask & answer_mask).any():
        raise RuntimeError("LLaVA-Med masks overlap.")
    if ((vision_mask | prompt_mask | answer_mask) & ~attention_mask).any():
        raise RuntimeError("LLaVA-Med masks include padding positions.")
    return {
        "attention_mask": attention_mask,
        "vision_mask": vision_mask,
        "prompt_mask": prompt_mask,
        "answer_mask": answer_mask,
    }


class LlavaMedForEditing(nn.Module):
    """Official LLaVA-Med v1.5 Mistral wrapper with EasyEdit full-sequence masks."""

    def __init__(
        self,
        model_path: str,
        vision_tower_path: str,
        model_name: str = "llava-med-v1.5-mistral-7b",
        official_loader_source: str = "third_party/LLaVA-Med",
        device: Any = "cuda",
        dtype: str = "float16",
        conversation_template: str = "mistral_instruct",
    ) -> None:
        super().__init__()
        self.model_path = str(model_path)
        self.vision_tower_path = str(vision_tower_path)
        self.model_name = str(model_name)
        self.official_loader_source = str(Path(official_loader_source).expanduser().resolve())
        self.device_name = _normalize_device(device)
        self.dtype = _torch_dtype(dtype)
        self.conversation_template = conversation_template

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ["LLAVA_MED_VISION_TOWER_PATH"] = self.vision_tower_path
        if self.official_loader_source not in sys.path:
            sys.path.insert(0, self.official_loader_source)

        from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IGNORE_INDEX = IGNORE_INDEX
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token

        disable_torch_init()
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_path,
            None,
            self.model_name,
            device=self.device_name,
        )
        self.tokenizer = tokenizer
        self.llava_tokenizer = tokenizer
        self.llava_model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.llava_model.eval()
        self.config = self.llava_model.config
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def model(self):
        return self.llava_model

    @property
    def hidden_size(self) -> int:
        return int(self.llava_model.config.hidden_size)

    @property
    def lm_device(self) -> torch.device:
        return next(self.llava_model.parameters()).device

    def _conversation_prompt(self, prompt: str, target: Optional[str]) -> str:
        question = prompt
        if self.DEFAULT_IMAGE_TOKEN not in question:
            question = self.DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = self.conv_templates[self.conversation_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], target)
        return conv.get_prompt()

    def _batch_size(self, samples: Dict[str, Any]) -> int:
        if isinstance(samples.get("text_input"), (list, tuple)):
            return len(samples["text_input"])
        if isinstance(samples.get("prompt"), (list, tuple)):
            return len(samples["prompt"])
        image = samples.get("image")
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            return int(image.shape[0])
        image_path = samples.get("image_path", samples.get("image_paths"))
        if isinstance(image_path, (list, tuple)):
            return len(image_path)
        labels = samples.get("labels")
        if isinstance(labels, torch.Tensor) and labels.dim() >= 2:
            return int(labels.shape[0])
        return 1

    def _prompt_target_lists(self, samples: Dict[str, Any], batch_size: int) -> Tuple[List[str], List[str]]:
        text_inputs = _as_list(samples.get("text_input", ""), batch_size)
        prompts_value = samples.get("prompt", samples.get("prompts", None))
        targets_value = samples.get("target", samples.get("targets", None))
        prompts = _as_list(prompts_value, batch_size) if prompts_value is not None else [None] * batch_size
        targets = _as_list(targets_value, batch_size) if targets_value is not None else [None] * batch_size
        labels = samples.get("labels")
        prompt_lens = samples.get("prompts_len")

        resolved_prompts: List[str] = []
        resolved_targets: List[str] = []
        for idx in range(batch_size):
            text = str(text_inputs[idx] or "")
            prompt = prompts[idx]
            target = targets[idx]
            if target is None and isinstance(labels, torch.Tensor):
                target = _target_from_labels(self.tokenizer, labels, idx)
            if prompt is None and prompt_lens is not None:
                token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                cut = int(prompt_lens[idx])
                prompt = self.tokenizer.decode(token_ids[:cut], skip_special_tokens=True)
            if prompt is None:
                prompt = text[: max(0, len(text) - len(str(target or "")))]
            if target is None:
                target = text[len(str(prompt)) :]
            resolved_prompts.append(str(prompt))
            resolved_targets.append(str(target))
        return resolved_prompts, resolved_targets

    def _image_for_row(self, samples: Dict[str, Any], row: int) -> torch.Tensor:
        image_paths = samples.get("image_path", samples.get("image_paths"))
        if image_paths is not None:
            path = _as_list(image_paths, self._batch_size(samples))[row]
            image = Image.open(path).convert("RGB")
            processed = self.process_images([image], self.image_processor, self.llava_model.config)
            if isinstance(processed, list):
                processed = torch.stack(processed, dim=0)
            return processed.to(self.lm_device, dtype=self.dtype)

        image = samples.get("image")
        if isinstance(image, Image.Image):
            processed = self.process_images([image.convert("RGB")], self.image_processor, self.llava_model.config)
            if isinstance(processed, list):
                processed = torch.stack(processed, dim=0)
            return processed.to(self.lm_device, dtype=self.dtype)
        if isinstance(image, (list, tuple)) and image and isinstance(image[row], Image.Image):
            processed = self.process_images([image[row].convert("RGB")], self.image_processor, self.llava_model.config)
            if isinstance(processed, list):
                processed = torch.stack(processed, dim=0)
            return processed.to(self.lm_device, dtype=self.dtype)
        if isinstance(image, torch.Tensor):
            tensor = image[row : row + 1] if image.dim() == 4 else image.unsqueeze(0)
            return tensor.to(self.lm_device, dtype=self.dtype)
        raise RuntimeError("LLaVA-Med samples must provide `image`, `image_path`, or `image_paths`.")

    def _build_one(self, prompt: str, target: str, image_tensor: torch.Tensor):
        prompt_text = self._conversation_prompt(prompt, None)
        full_text = self._conversation_prompt(prompt, target)
        prompt_ids = self.tokenizer_image_token(
            prompt_text, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        full_ids = self.tokenizer_image_token(
            full_text, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        if prompt_ids.numel() > full_ids.numel():
            raise RuntimeError("LLaVA-Med prompt tokenization is longer than prompt+target tokenization.")

        input_ids = full_ids.unsqueeze(0).to(self.lm_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.lm_device)
        labels = input_ids.clone()
        labels[:, : int(prompt_ids.numel())] = self.IGNORE_INDEX
        labels[input_ids.eq(self.IMAGE_TOKEN_INDEX)] = self.IGNORE_INDEX

        (
            _,
            _position_ids,
            expanded_attention_mask,
            _,
            inputs_embeds,
            expanded_labels,
        ) = self.llava_model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            images=image_tensor,
        )
        if inputs_embeds is None or expanded_labels is None or expanded_attention_mask is None:
            raise RuntimeError("LLaVA-Med multimodal preparation did not return expanded embeddings, labels, and masks.")

        image_feature_len = int(inputs_embeds.shape[1] - (input_ids.shape[1] - 1))
        masks = build_llava_med_masks(
            token_ids=input_ids[0],
            labels=expanded_labels[0],
            expanded_attention_mask=expanded_attention_mask[0],
            image_token_index=self.IMAGE_TOKEN_INDEX,
            image_feature_len=image_feature_len,
        )
        return inputs_embeds[0], expanded_labels[0], masks

    def _build_batch(self, samples: Dict[str, Any]):
        batch_size = self._batch_size(samples)
        prompts, targets = self._prompt_target_lists(samples, batch_size)
        rows = [self._build_one(prompts[idx], targets[idx], self._image_for_row(samples, idx)) for idx in range(batch_size)]
        max_len = max(row[0].shape[0] for row in rows)
        hidden_size = rows[0][0].shape[-1]
        inputs_embeds = torch.zeros(
            batch_size,
            max_len,
            hidden_size,
            device=self.lm_device,
            dtype=rows[0][0].dtype,
        )
        labels = torch.full(
            (batch_size, max_len),
            self.IGNORE_INDEX,
            device=self.lm_device,
            dtype=rows[0][1].dtype,
        )
        masks = {
            name: torch.zeros(batch_size, max_len, device=self.lm_device, dtype=torch.bool)
            for name in ("attention_mask", "vision_mask", "prompt_mask", "answer_mask")
        }
        for idx, (embeds, row_labels, row_masks) in enumerate(rows):
            length = embeds.shape[0]
            inputs_embeds[idx, :length] = embeds
            labels[idx, :length] = row_labels
            for name in masks:
                masks[name][idx, :length] = row_masks[name]
        return inputs_embeds, labels, masks

    def forward(self, samples: Dict[str, Any]) -> LlavaMedOutput:
        inputs_embeds, labels, masks = self._build_batch(samples)
        samples["attention_mask"] = masks["attention_mask"]
        samples["vision_mask"] = masks["vision_mask"]
        samples["prompt_mask"] = masks["prompt_mask"]
        samples["answer_mask"] = masks["answer_mask"]
        outputs = self.llava_model(
            inputs_embeds=inputs_embeds,
            attention_mask=masks["attention_mask"].long(),
            labels=labels,
            return_dict=True,
            use_cache=False,
        )
        return LlavaMedOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            labels=labels,
            attention_mask=masks["attention_mask"],
            vision_mask=masks["vision_mask"],
            prompt_mask=masks["prompt_mask"],
            answer_mask=masks["answer_mask"],
        )

