#!/usr/bin/env python3
"""Run the frozen semantic Judge with the server's existing vLLM AWQ runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_semantic_judge_v3 import PACKET_FIELDS, atomic_new, lock_payload, parse_boolean, render  # noqa: E402


CHOICES = ['{"is_correct": true}', '{"is_correct": false}']
MAX_MODEL_LEN = 1024


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tokenizer_files(model_path: Path) -> dict[str, str]:
    names = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json")
    return {name: sha256_bytes((model_path / name).read_bytes()) for name in names if (model_path / name).is_file()}


def authorized_device() -> dict[str, str]:
    physical = os.environ.get("M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES", "")
    expected_uuid = os.environ.get("M3BENCH_FORMAL_EXPECTED_GPU_UUID", "")
    allowed = {value.strip() for value in os.environ.get("M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES", "2,3").split(",")}
    if physical not in allowed or os.environ.get("CUDA_VISIBLE_DEVICES") != physical or not expected_uuid:
        raise RuntimeError("Judge GPU authorization environment is incomplete or invalid")
    name, actual_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader", "-i", physical], text=True,
    ).strip().rsplit(", ", 1)
    if actual_uuid != expected_uuid:
        raise RuntimeError("Judge GPU UUID mismatch")
    return {"physical_gpu": physical, "gpu_name": name, "gpu_uuid": actual_uuid}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.execution_lock.exists():
        raise FileExistsError(args.output if args.output.exists() else args.execution_lock)
    gpu = authorized_device()
    lock = json.loads(args.lock.read_text())
    expected = lock_payload(args.model_path.resolve(), constrained_boolean=True)
    for key in ("judge_model", "judge_snapshot_sha", "prompt", "temperature", "schema", "generation"):
        if lock.get(key) != expected[key]:
            raise RuntimeError(f"Judge lock mismatch: {key}")
    rows = [json.loads(line) for line in args.packet.read_text().splitlines() if line.strip()]
    if any(set(row) != PACKET_FIELDS for row in rows):
        raise RuntimeError("Judge packet exposes forbidden or missing fields")
    keys = [(row["opaque_query_id"], row["adjudication_pass"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate Judge packet key")

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    llm = LLM(
        model=str(args.model_path),
        quantization="awq",
        dtype="half",
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        generation_config="vllm",
    )
    tokenizer = llm.get_tokenizer()
    prompts = [render(tokenizer, row) for row in rows]
    prompt_token_ids = [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]
    if any(len(ids) > MAX_MODEL_LEN for ids in prompt_token_ids):
        raise RuntimeError("Judge input exceeds max_model_len; truncation is forbidden")
    params = SamplingParams(
        temperature=0,
        max_tokens=24,
        seed=0,
        guided_decoding=GuidedDecodingParams(choice=CHOICES, backend="xgrammar"),
    )
    results = llm.generate(prompts, params, use_tqdm=False)
    output = []
    for row, expected_ids, result in zip(rows, prompt_token_ids, results, strict=True):
        actual_ids = list(result.prompt_token_ids)
        if actual_ids != expected_ids:
            raise RuntimeError("vLLM prompt tokenization differed from the locked sequence")
        text = result.outputs[0].text.strip()
        output.append({
            "opaque_query_id": row["opaque_query_id"],
            "adjudication_pass": row["adjudication_pass"],
            "is_correct": parse_boolean(text),
            "parse_valid": True,
            "judge_output": text,
            "judge_model": lock["judge_model"],
            "judge_snapshot_sha": lock["judge_snapshot_sha"],
            "judge_prompt_sha256": lock["prompt_sha256"],
            "legacy_semantic_protocol_sha256": lock["config_sha256"],
            "output_token_ids": list(result.outputs[0].token_ids),
        })
    if len(output) != len(rows):
        raise RuntimeError("Judge coverage mismatch")

    config = llm.llm_engine.vllm_config
    engine_class = f"{llm.llm_engine.__class__.__module__}.{llm.llm_engine.__class__.__name__}"
    engine_mode = "V1" if ".v1." in engine_class else "V0"
    quant_config = getattr(config, "quant_config", None)
    model_config = getattr(config, "model_config", None)
    torch = __import__("torch")
    execution = {
        "schema_version": "medtrace-judge-vllm-execution-lock-v1",
        "model": {"name": lock["judge_model"], "snapshot": lock["judge_snapshot_sha"], "path": str(args.model_path.resolve())},
        "tokenizer": {
            "snapshot": lock["judge_snapshot_sha"],
            "file_sha256": tokenizer_files(args.model_path.resolve()),
            "chat_template_sha256": sha256_bytes((tokenizer.chat_template or "").encode()),
            "render_function_sha256": sha256_bytes(inspect.getsource(render).encode()),
            "add_special_tokens": True,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "runtime": {
            "vllm": package_version("vllm"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "cuda": torch.version.cuda,
            **gpu,
            "engine_mode": engine_mode,
            "engine_class": engine_class,
            "quantization_requested": "awq",
            "quantization_config_class": None if quant_config is None else type(quant_config).__name__,
            "quantization_config": None if quant_config is None else repr(quant_config),
            "dtype": str(getattr(model_config, "dtype", "unknown")),
            "guided_decoding_backend": "xgrammar",
            "guided_decoding_backend_version": package_version("xgrammar"),
        },
        "generation": {
            "candidate_strings": CHOICES,
            "seed": 0,
            "temperature": 0,
            "max_tokens": 24,
            "max_model_len": MAX_MODEL_LEN,
            "overlength_policy": "hard_error_before_generate; no truncation",
            "model_generation_config": "disabled via generation_config=vllm",
            "resolved_sampling_params": repr(params),
            "resolved_engine_config": repr(config),
        },
        "packet": {
            "sha256": sha256_bytes(args.packet.read_bytes()),
            "row_count": len(rows),
            "unique_key": ["opaque_query_id", "adjudication_pass"],
            "schema": sorted(PACKET_FIELDS),
            "coverage": len(output),
        },
        "requests": [
            {
                "key": list(key),
                "prompt_token_ids": ids,
                "prompt_token_ids_sha256": sha256_json(ids),
                "prompt_token_count": len(ids),
                "output_token_ids": result["output_token_ids"],
            }
            for key, ids, result in zip(keys, prompt_token_ids, output, strict=True)
        ],
        "legacy_semantic_protocol_sha256": lock["config_sha256"],
    }
    execution["backend_execution_sha256"] = sha256_json(execution)
    atomic_new(args.output, "".join(json.dumps(row, sort_keys=True) + "\n" for row in output))
    atomic_new(args.execution_lock, json.dumps(execution, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
