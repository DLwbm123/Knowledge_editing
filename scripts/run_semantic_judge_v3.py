#!/usr/bin/env python3
"""Run a fixed local, method-blind semantic correctness Judge over JSONL packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


PROMPT = (
    "Decide whether the raw base answer is semantically correct for the medical visual question "
    "relative to the gold answer. Be tolerant of equivalent wording and a correct more-specific "
    "answer. Be strict about facts, polarity and negation, numbers, anatomy, pathology, modality, "
    "and whether the answer actually addresses the question. Return exactly one JSON object with "
    "one boolean field: {\"is_correct\": true} or {\"is_correct\": false}."
)
PACKET_FIELDS = {"opaque_query_id", "question", "gold_answer", "raw_base_answer", "adjudication_pass"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def parse_boolean(text: str) -> bool:
    match = re.fullmatch(r"\s*\{\s*\"is_correct\"\s*:\s*(true|false)\s*\}\s*", text, re.IGNORECASE)
    if not match:
        raise ValueError("Judge output is not strict JSON boolean schema")
    return match.group(1).casefold() == "true"


def lock_payload(model_path: Path) -> dict:
    snapshot_sha = model_path.name
    if not re.fullmatch(r"[0-9a-f]{40}", snapshot_sha):
        raise RuntimeError("Judge model path must resolve to an immutable 40-character snapshot")
    generation = {
        "do_sample": False, "temperature": 0, "num_beams": 1,
        "max_new_tokens": 24, "use_cache": True, "enable_thinking": False,
    }
    return {
        "judge_model": "Qwen/Qwen3-32B-AWQ",
        "judge_snapshot_sha": snapshot_sha,
        "prompt": PROMPT,
        "prompt_sha256": digest(PROMPT),
        "temperature": 0,
        "schema": "strict_json_boolean",
        "generation": generation,
        "method_blind": True,
        "allowed_input_fields": sorted(PACKET_FIELDS - {"adjudication_pass"}),
        "retry_policy": "none",
    }


def lock_command(args: argparse.Namespace) -> None:
    payload = lock_payload(args.model_path.resolve())
    payload["config_sha256"] = digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    atomic_new(args.output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("judge_model", "judge_snapshot_sha", "prompt_sha256", "config_sha256")}, sort_keys=True))


def render(tokenizer, row: dict) -> str:
    content = json.dumps({
        "opaque_query_id": row["opaque_query_id"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "raw_base_answer": row["raw_base_answer"],
    }, ensure_ascii=False, sort_keys=True)
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def run_command(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    expected = lock_payload(args.model_path.resolve())
    for key in ("judge_model", "judge_snapshot_sha", "prompt", "temperature", "schema", "generation"):
        if lock.get(key) != expected[key]:
            raise RuntimeError(f"Judge lock mismatch: {key}")
    rows = read_jsonl(args.packet)
    if any(set(row) != PACKET_FIELDS for row in rows):
        raise RuntimeError("Judge packet exposes forbidden or missing fields")
    keys = [(row["opaque_query_id"], row["adjudication_pass"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate Judge packet key")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, device_map={"": "cuda:0"}, torch_dtype="auto", low_cpu_mem_usage=True,
    ).eval()
    outputs = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        prompts = [render(tokenizer, row) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_tokens).to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(
                **encoded, do_sample=False, temperature=0.0, num_beams=1,
                max_new_tokens=24, use_cache=True, pad_token_id=tokenizer.pad_token_id,
            )
        continuations = generated[:, encoded.input_ids.shape[1]:]
        for row, text in zip(batch, tokenizer.batch_decode(continuations, skip_special_tokens=True)):
            outputs.append({
                "opaque_query_id": row["opaque_query_id"],
                "adjudication_pass": row["adjudication_pass"],
                "is_correct": parse_boolean(text),
                "judge_output": text.strip(),
                "judge_model": lock["judge_model"],
                "judge_snapshot_sha": lock["judge_snapshot_sha"],
                "judge_prompt_sha256": lock["prompt_sha256"],
                "judge_config_sha256": lock["config_sha256"],
            })
        if args.progress:
            args.progress.write_text(json.dumps({"completed": len(outputs), "total": len(rows)}) + "\n", encoding="utf-8")
    if len(outputs) != len(rows):
        raise RuntimeError("Judge coverage mismatch")
    atomic_new(args.output, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in outputs))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock")
    lock.add_argument("--model-path", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    lock.set_defaults(func=lock_command)
    run = sub.add_parser("run")
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--packet", type=Path, required=True)
    run.add_argument("--lock", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--progress", type=Path)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--max-input-tokens", type=int, default=768)
    run.set_defaults(func=run_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
