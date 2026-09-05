#!/usr/bin/env python3
"""Run the frozen semantic Judge with the server's existing vLLM AWQ runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_semantic_judge_v3 import PACKET_FIELDS, lock_payload, parse_boolean, render  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lock = json.loads(args.lock.read_text())
    expected = lock_payload(args.model_path.resolve(), constrained_boolean=True)
    for key in ("judge_model", "judge_snapshot_sha", "prompt", "temperature", "schema", "generation"):
        if lock.get(key) != expected[key]:
            raise RuntimeError(f"Judge lock mismatch: {key}")
    rows = [json.loads(line) for line in args.packet.read_text().splitlines() if line.strip()]
    if any(set(row) != PACKET_FIELDS for row in rows):
        raise RuntimeError("Judge packet exposes forbidden or missing fields")

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    llm = LLM(
        model=str(args.model_path),
        quantization="awq",
        dtype="half",
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [render(tokenizer, row) for row in rows]
    params = SamplingParams(
        temperature=0,
        max_tokens=24,
        seed=0,
        guided_decoding=GuidedDecodingParams(choice=['{"is_correct": true}', '{"is_correct": false}']),
    )
    results = llm.generate(prompts, params, use_tqdm=False)
    output = []
    for row, result in zip(rows, results, strict=True):
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
            "judge_config_sha256": lock["config_sha256"],
            "runtime": "vllm-0.9.2-existing-environment",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output))


if __name__ == "__main__":
    main()
