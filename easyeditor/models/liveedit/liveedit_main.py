from copy import deepcopy
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from ...trainer import LiveEdit
from ...trainer.algs.liveedit import sanitize_liveedit_metadata
from .liveedit_hparams import LiveEditMultimodalHparams


class LiveEditMultimodalRewriteExecutor:
    def __init__(self):
        self.is_init = False
        self.alg = None
        self.model = None
        self.tokenizer = None

    def init_model(self, model, tok, params: LiveEditMultimodalHparams):
        self.model = model
        self.tokenizer = tok
        self.alg = LiveEdit(self.model, params, lambda: deepcopy(self.model))
        if params.archive is not None:
            archive = torch.load(params.archive, map_location="cpu")
            state = archive["model"] if isinstance(archive, dict) and "model" in archive else archive
            self.alg.load_state_dict(state, strict=False)
        device = f"cuda:{params.device}" if not str(params.device).startswith("cuda") else params.device
        self.alg.to(torch.device(device))
        self.is_init = True

    def _requests_to_batch(self, requests: List[Dict[str, Any]], tok: AutoTokenizer, hparams: LiveEditMultimodalHparams, device):
        src = [request["prompt"] for request in requests]
        trg = [(" " if request["target"][0] != " " else "") + request["target"] for request in requests]
        image = torch.stack([request["image"] for request in requests], dim=0).to(device)
        text_input = [s + t for s, t in zip(src, trg)]
        if hparams.model_name in {"minigpt4", "blip2"}:
            prompts_len = [len(tok.encode(s, add_special_tokens=False)) for s in src]
            labels = tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        else:
            raise NotImplementedError("LiveEdit singleton editing currently supports BLIP2 and MiniGPT-4 only.")
        return {
            "image": image,
            "text_input": text_input,
            "labels": labels,
            "prompts_len": prompts_len,
        }

    def apply_to_model(
        self,
        model,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LiveEditMultimodalHparams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
    ):
        if not self.is_init:
            self.init_model(deepcopy(model) if copy else model, tok, hparams)
        device = next(self.alg.parameters()).device
        batch = self._requests_to_batch(requests, tok, hparams, device)
        self.alg.append_from_batch(batch, metadata=sanitize_liveedit_metadata(requests))
        if hparams.liveedit_repository_path:
            self.alg.repository.save(hparams.liveedit_repository_path)
        return self.alg, {}
