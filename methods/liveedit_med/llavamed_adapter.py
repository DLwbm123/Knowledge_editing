"""Audited layer-21 adaptation boundary for the project's LLaVA-Med model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from scripts.engram.routed_banked_lora_utils import expanded_positions, find_unique_subsequence
from scripts.engram.stage0_generation_audit_utils import CanonicalInputs


LAYER21_PATH = "model.layers.21"


@dataclass(frozen=True)
class ExpandedSpans:
    pre_image: tuple[int, ...]
    visual: tuple[int, ...]
    question: tuple[int, ...]
    boundary: int
    answer: tuple[int, ...]
    valid_length: int


def resolve_layer21_block(model: Any):
    matches = [(name, module) for name, module in model.llava_model.named_modules() if name == LAYER21_PATH]
    if len(matches) != 1 or matches[0][1].__class__.__name__ != "MistralDecoderLayer":
        raise RuntimeError("LIVEEDIT_MED_LAYER21_BLOCK_PATH_UNRESOLVED")
    return matches[0]


def build_spans(model: Any, canonical: CanonicalInputs, *, include_answer: bool) -> ExpandedSpans:
    ids = canonical.full_ids if include_answer else canonical.prompt_ids
    ids_list = [int(x) for x in ids[0].tolist()]
    image_positions = [i for i, value in enumerate(ids_list) if value == int(model.IMAGE_TOKEN_INDEX)]
    if len(image_positions) != 1: raise RuntimeError("LIVEEDIT_MED_AMBIGUOUS_TOKEN_SPAN")
    raw_prompt = str(canonical.prompt_text)
    # The canonical question is the text between image newline and closing instruction.
    question = raw_prompt.split("\n", 1)[-1].rsplit("[/INST]", 1)[0].strip()
    candidates = [[int(v) for v in model.llava_tokenizer(prefix + question, add_special_tokens=False).input_ids]
                  for prefix in ("", " ", "\n")]
    q_positions = find_unique_subsequence(ids_list, candidates)
    sample = {"image_path": [""], "prompt": [question], "target": [""]}
    image_tokens = int(model.llava_model.encode_images(canonical.image).shape[1])
    expanded_q = tuple(expanded_positions(q_positions, image_positions[0], image_tokens))
    expanded_length = len(ids_list) - 1 + image_tokens
    answer_count = int(canonical.target_ids.numel()) if include_answer else 0
    boundary = expanded_length - answer_count - 1
    visual_start = image_positions[0]
    visual = tuple(range(visual_start, visual_start + image_tokens))
    answer = tuple(range(expanded_length - answer_count, expanded_length)) if answer_count else ()
    pre = tuple(range(visual_start))
    if not expanded_q or max(expanded_q) >= boundary or (answer and min(answer) <= boundary):
        raise RuntimeError("LIVEEDIT_MED_AMBIGUOUS_TOKEN_SPAN")
    return ExpandedSpans(pre, visual, expanded_q, boundary, answer, expanded_length)


class Layer21ResidualHook:
    def __init__(self, block: torch.nn.Module, residual_fn: Callable[[torch.Tensor], torch.Tensor],
                 assistant_only: bool = False):
        self.block = block; self.residual_fn = residual_fn; self.assistant_only = assistant_only
        self.handle = None; self.enabled = False; self.prompt_boundary: int | None = None

    def set_prompt_boundary(self, boundary: int | None): self.prompt_boundary = boundary

    def _hook(self, _module, _args, output):
        if not self.enabled: return output
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        residual = self.residual_fn(hidden)
        if self.assistant_only:
            mask = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
            if hidden.shape[1] == 1: mask[:, 0] = 1
            elif self.prompt_boundary is not None: mask[:, self.prompt_boundary:] = 1
            residual = residual * mask.unsqueeze(-1)
        if isinstance(output, tuple): return (output[0] + residual, *output[1:])
        if isinstance(output, list): return [output[0] + residual, *output[1:]]
        return output + residual

    def install(self):
        if self.handle is not None: raise RuntimeError("hook already installed")
        self.handle = self.block.register_forward_hook(self._hook); return self

    def remove(self):
        if self.handle is not None: self.handle.remove(); self.handle = None

    def __enter__(self): self.install(); self.enabled = True; return self
    def __exit__(self, *_): self.enabled = False; self.remove()
