from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from ...trainer.algs.time_edit import TIMEEdit
from .time_edit_hparams import TIMEEditMultimodalHparams


class TIMEEditMultimodalRewriteExecutor:
    def __init__(self) -> None:
        self.is_init = False
        self.alg: TIMEEdit | None = None
        self.model = None
        self.tokenizer = None

    def init_model(self, model, tok, params: TIMEEditMultimodalHparams):
        self.model = model
        self.tokenizer = tok
        self.alg = TIMEEdit(self.model, params, lambda: deepcopy(self.model))
        device = params.device if str(params.device).startswith(("cuda", "cpu", "mps")) else f"cuda:{params.device}"
        self.alg.to(torch.device(device))
        archive = params.archive
        if archive is not None:
            loaded = torch.load(archive, map_location="cpu")
            state = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
            self.alg.load_state_dict(state, strict=False)
        self.is_init = True

    def _requests_to_batch(
        self,
        requests: List[Dict[str, Any]],
        tok: AutoTokenizer,
        hparams: TIMEEditMultimodalHparams,
        device: torch.device,
    ) -> Dict[str, Any]:
        prompts = [str(request["prompt"]) for request in requests]
        targets = []
        for request in requests:
            target = str(request.get("target", request.get("target_new", "")))
            targets.append((" " if target and not target.startswith(" ") else "") + target)
        images = [request.get("image") for request in requests]
        if all(torch.is_tensor(image) for image in images):
            image = torch.stack([image.to(device) for image in images], dim=0)
        else:
            image = images
        text_input = [prompt + target for prompt, target in zip(prompts, targets)]
        labels = tok(targets, add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"].to(device)
        return {
            "image": image,
            "prompt": prompts,
            "target": targets,
            "text_input": text_input,
            "labels": labels,
            "prompts_len": [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts],
        }

    def apply_to_model(
        self,
        model,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: TIMEEditMultimodalHparams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
    ) -> Tuple[TIMEEdit, Dict[str, Any]]:
        if not self.is_init:
            self.init_model(deepcopy(model) if copy else model, tok, hparams)
        if self.alg is None:
            raise RuntimeError("TIME executor was not initialized.")
        device = next(self.alg.parameters()).device
        batch = self._requests_to_batch(requests, tok, hparams, device)
        train_batch = {
            "edit_inner": batch,
            "edit_outer": batch,
            "loc": batch,
            "loc_image": batch,
            "record_id": requests[0].get("id", requests[0].get("record_id", "request_0")) if requests else "request_0",
        }
        self.alg.add_expert(train_batch["record_id"])
        optimizer = torch.optim.AdamW(self.alg.outer_parameters(), lr=float(hparams.lr))
        self.alg.train()
        for _step in range(max(0, int(hparams.time_edit_iters))):
            optimizer.zero_grad(set_to_none=True)
            self.alg.edit_step(train_batch, training=True, optimizer=optimizer)
            torch.nn.utils.clip_grad_norm_(self.alg.outer_parameters(), float(hparams.grad_clip), error_if_nonfinite=True)
            optimizer.step()
        if hparams.time_save_state:
            state_dir = Path(hparams.time_state_dir or Path(hparams.results_dir) / "time")
            self.alg.save_time_state(state_dir)
        if hparams.time_repository_path:
            self.alg.repository.save(hparams.time_repository_path)
        return self.alg, {}

