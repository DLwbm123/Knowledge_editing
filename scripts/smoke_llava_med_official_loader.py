#!/usr/bin/env python3
"""Smoke test the vendored official LLaVA-Med loader on local assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _torch_dtype(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def _first_image(image_root: Path) -> str | None:
    if not image_root.exists():
        return None
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            return str(path)
    return None


def _peak_cuda_mb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower-path", required=True)
    parser.add_argument("--model-name", default="llava-med-v1.5-mistral-7b")
    parser.add_argument("--source-root", default="third_party/LLaVA-Med")
    parser.add_argument("--image-root", default="datasets/MedMKEB/images")
    parser.add_argument("--image-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "import_ok": False,
        "load_ok": False,
        "generation_ok": False,
        "model_class": None,
        "tokenizer_class": None,
        "image_processor_class": None,
        "config_model_type": None,
        "context_len": None,
        "device": args.device,
        "dtype": args.dtype,
        "cuda_memory_peak_mb": None,
        "generated_text": None,
        "image_path": None,
        "skip_generation": bool(args.skip_generation),
        "mm_projector_param_count": 0,
        "vision_tower_loaded": False,
        "official_loader_source": str(Path(args.source_root).resolve()),
        "errors": [],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from PIL import Image
    except Exception:
        payload["errors"].append("base imports failed:\n" + traceback.format_exc())
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2

    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["LLAVA_MED_VISION_TOWER_PATH"] = str(Path(args.vision_tower_path).expanduser())

    source_root = Path(args.source_root).expanduser().resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    try:
        from llava.constants import (
            DEFAULT_IMAGE_PATCH_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        payload["import_ok"] = True
    except Exception:
        payload["errors"].append("official loader import failed:\n" + traceback.format_exc())
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2

    try:
        dtype = _torch_dtype(args.dtype)
        disable_torch_init()
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path,
            None,
            args.model_name,
            device=args.device,
        )
        model.eval()
        payload.update(
            {
                "load_ok": True,
                "model_class": model.__class__.__name__,
                "tokenizer_class": tokenizer.__class__.__name__,
                "image_processor_class": image_processor.__class__.__name__ if image_processor is not None else None,
                "config_model_type": getattr(model.config, "model_type", None),
                "context_len": context_len,
            }
        )
        projector = getattr(getattr(model, "model", None), "mm_projector", None)
        if projector is not None:
            payload["mm_projector_param_count"] = sum(p.numel() for p in projector.parameters())
        vision_tower = model.get_vision_tower() if hasattr(model, "get_vision_tower") else None
        payload["vision_tower_loaded"] = bool(getattr(vision_tower, "is_loaded", False))
    except RuntimeError as exc:
        payload["cuda_memory_peak_mb"] = _peak_cuda_mb()
        if "out of memory" in str(exc).lower():
            payload["errors"].append("official loader model load OOM: " + repr(exc))
        else:
            payload["errors"].append("official loader model load failed:\n" + traceback.format_exc())
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2
    except Exception:
        payload["cuda_memory_peak_mb"] = _peak_cuda_mb()
        payload["errors"].append("official loader model load failed:\n" + traceback.format_exc())
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2

    image_path = args.image_path or _first_image(Path(args.image_root))
    payload["image_path"] = image_path
    if args.skip_generation:
        payload["cuda_memory_peak_mb"] = _peak_cuda_mb()
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0 if payload["load_ok"] else 2

    if not image_path:
        payload["errors"].append(f"no image found under {args.image_root}")
        payload["cuda_memory_peak_mb"] = _peak_cuda_mb()
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2

    try:
        question = "Describe this medical image briefly."
        if getattr(model.config, "mm_use_im_start_end", False):
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(model.device)

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(model.device, dtype=dtype) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=dtype)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        payload["generated_text"] = text
        payload["generation_ok"] = bool(text)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            payload["errors"].append("generation OOM: " + repr(exc))
        else:
            payload["errors"].append("generation failed:\n" + traceback.format_exc())
    except Exception:
        payload["errors"].append("generation failed:\n" + traceback.format_exc())

    payload["cuda_memory_peak_mb"] = _peak_cuda_mb()
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["load_ok"] and (payload["generation_ok"] or args.skip_generation) else 2


if __name__ == "__main__":
    raise SystemExit(main())
