#!/usr/bin/env python3
"""Run the frozen M3Bench runtime canary through project or official LLaVA-Med."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def image_dtype(value) -> str:
    tensor = value[0] if isinstance(value, list) else value
    return str(tensor.dtype)


def swap_map(rows: list[dict], count: int) -> dict[str, str]:
    chosen = sorted(rows, key=lambda row: hashlib.sha256(row["query_id"].encode()).hexdigest())[:count]
    if len(chosen) < 2:
        return {}
    return {row["query_id"]: chosen[(index + 1) % len(chosen)]["image_path"] for index, row in enumerate(chosen)}


def load_runtime(args: argparse.Namespace):
    import torch

    if args.runtime == "project":
        sys.path[:0] = [str(args.project_root / "third_party/LLaVA-Med"), str(args.project_root)]
        from m3bench_repro.inference.llava_med import LlavaMedAdapter, decode_generation_sequence

        adapter = LlavaMedAdapter(args.model_path, args.vision_path, device="cuda:0")
        adapter.load()

        def generate(image_path: str, question: str) -> tuple[str, int, str]:
            batch = adapter.prepare_inputs(image_path, question)
            with torch.inference_mode():
                output = adapter.model.generate(
                    batch["input_ids"], attention_mask=batch["attention_mask"], images=batch["images"],
                    do_sample=False, num_beams=1, max_new_tokens=1024, use_cache=True,
                    pad_token_id=adapter.tokenizer.eos_token_id,
                )
            result = decode_generation_sequence(
                adapter.tokenizer, output[0].detach().cpu().tolist(),
                contract=adapter.generation_sequence_contract,
                prompt_token_count=int(batch["input_ids"].shape[1]),
            )
            return result.decoded_text, len(result.raw_token_ids), image_dtype(batch["images"])

        model, tokenizer = adapter.model, adapter.tokenizer
    else:
        if args.llava_root is None:
            raise ValueError("--llava-root is required for the official runtime")
        sys.path.insert(0, str(args.llava_root))
        from PIL import Image
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        tokenizer, model, processor, _ = load_pretrained_model(
            str(args.model_path), None, get_model_name_from_path(str(args.model_path))
        )

        def generate(image_path: str, question: str) -> tuple[str, int, str]:
            marker = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
            conv = conv_templates["mistral_instruct"].copy()
            conv.append_message(conv.roles[0], marker + "\n" + question)
            conv.append_message(conv.roles[1], None)
            ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            images = process_images([Image.open(image_path).convert("RGB")], processor, model.config)
            images = images.to(model.device, dtype=torch.float16) if not isinstance(images, list) else [value.to(model.device, dtype=torch.float16) for value in images]
            with torch.inference_mode():
                output = model.generate(
                    ids, images=images, do_sample=False, temperature=0.0, num_beams=1,
                    max_new_tokens=1024, use_cache=True,
                )
            tokens = output[0].detach().cpu().tolist()
            return tokenizer.decode(tokens, skip_special_tokens=True).strip(), len(tokens), image_dtype(images)

    tower = model.get_vision_tower()
    info = {
        "runtime": args.runtime,
        "python": sys.version.split()[0],
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "vision_tower_dtype": str(next(tower.parameters()).dtype),
        "projector_dtype": str(next(model.model.mm_projector.parameters()).dtype),
        "tokenizer_padding_side": tokenizer.padding_side,
        "generation": {"do_sample": False, "temperature": 0, "num_beams": 1, "max_new_tokens": 1024, "use_cache": True, "batch_size": 1},
    }
    return generate, info


def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.project_root))
    from scripts.m3bench_base_correctness_v3 import normalize

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_RUNTIME_EXECUTION":
        raise RuntimeError("runtime canary was not frozen before execution")
    rows = manifest["records"]
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    swaps = swap_map(rows, args.swap_count)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    done = read_jsonl(partial) if partial.exists() else []
    if [row["query_id"] for row in done] != [row["query_id"] for row in rows[:len(done)]]:
        raise RuntimeError("canary partial output is not an exact manifest prefix")
    generate, info = load_runtime(args)
    if args.runtime_info.exists():
        if json.loads(args.runtime_info.read_text(encoding="utf-8")) != info:
            raise RuntimeError("runtime identity changed during resume")
    else:
        atomic(args.runtime_info, json.dumps(info, indent=2, sort_keys=True) + "\n")
    with partial.open("a", encoding="utf-8") as handle:
        for row in rows[len(done):]:
            result = {"query_id": row["query_id"], "stratum": row["stratum"], "error": None}
            try:
                text, token_count, tensor_dtype = generate(row["image_path"], row["question"])
                result.update({
                    "raw_answer": text, "normalized_answer": normalize(text), "empty": not bool(text.strip()),
                    "generated_token_count": token_count, "hit_1024_token_limit": token_count >= 1024,
                    "image_tensor_dtype": tensor_dtype,
                })
                if row["query_id"] in swaps:
                    swapped, _, _ = generate(swaps[row["query_id"]], row["question"])
                    result["image_swap_normalized_answer"] = normalize(swapped)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            done.append(result)
            if len(done) % 10 == 0 or len(done) == len(rows):
                atomic(args.progress, json.dumps({"completed": len(done), "total": len(rows)}) + "\n")
            if len(done) % 50 == 0:
                os.fsync(handle.fileno())
    if len(done) != len(rows):
        raise RuntimeError("runtime canary coverage mismatch")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    os.replace(partial, args.output)


def main() -> None:
    if len(sys.argv) == 1 and os.environ.get("M3BENCH_PRIVATE_ARGV"):
        sys.argv.extend(json.loads(os.environ["M3BENCH_PRIVATE_ARGV"]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("project", "official"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--runtime-info", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-path", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--llava-root", type=Path)
    parser.add_argument("--swap-count", type=int, default=16)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
