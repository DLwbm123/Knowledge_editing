#!/usr/bin/env python3
"""Run one fixed method-blind semantic Judge over JSONL from stdin."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


MODEL = "gpt-5.6-sol"
PROMPT = """You are the fixed semantic correctness judge for method-blind medical QA.
Return only the required JSON. Set correct=true only when the raw answer semantically entails the
gold/reference for the question. Accept faithful concise paraphrases and standard synonyms. Reject
contradictions, wrong anatomy/laterality/polarity/number/modality, critical omissions, empty answers,
and visibly incomplete answers. Judge each item independently. Do not infer method or expected results.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-bin", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    items = [json.loads(line) for line in sys.stdin if line.strip()]
    required = {"opaque_event_id", "question", "gold_or_reference", "raw_model_answer", "task_metadata"}
    if not items or any(set(item) != required for item in items):
        raise RuntimeError("invalid blind Judge packet")
    if len({item["opaque_event_id"] for item in items}) != len(items):
        raise RuntimeError("duplicate opaque Judge ID")
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "properties": {"opaque_event_id": {"type": "string"}, "correct": {"type": "boolean"}},
            "required": ["opaque_event_id", "correct"], "additionalProperties": False,
        }}},
        "required": ["items"], "additionalProperties": False,
    }
    verdicts = []
    with tempfile.TemporaryDirectory(prefix="audit-judge-") as root_text:
        root = Path(root_text)
        schema_path = root / "schema.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        for index in range(0, len(items), args.batch_size):
            batch = items[index : index + args.batch_size]
            output = root / f"output_{index:04d}.json"
            command = [
                str(args.codex_bin), "-a", "never", "exec", "--ignore-user-config", "--ignore-rules",
                "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-m", MODEL,
                "-c", 'model_reasoning_effort="low"', "--output-schema", str(schema_path),
                "--output-last-message", str(output), "-",
            ]
            payload = PROMPT + "\nFixed model: " + MODEL + "\nItems:\n" + json.dumps(batch, ensure_ascii=False)
            completed = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=600)
            if completed.returncode:
                raise RuntimeError("semantic Judge process failed: " + completed.stderr[-800:])
            rows = json.loads(output.read_text(encoding="utf-8")).get("items", [])
            expected = {item["opaque_event_id"] for item in batch}
            if {row.get("opaque_event_id") for row in rows} != expected:
                raise RuntimeError("semantic Judge coverage mismatch")
            verdicts.extend(rows)
    for row in verdicts:
        print(json.dumps({"opaque_event_id": row["opaque_event_id"], "correct": row["correct"]}, sort_keys=True))


if __name__ == "__main__":
    main()
