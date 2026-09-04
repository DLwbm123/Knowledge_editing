#!/usr/bin/env python3
"""Run one resumable canonical base-inference shard and merge fixed shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m3bench_base_correctness_v3 import normalize
from scripts.m3bench_runtime_canary_v3 import load_runtime


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, text: str, *, new: bool = False) -> None:
    if new and path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def shard_rows(rows: list[dict], index: int, count: int) -> list[dict]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard index/count")
    return rows[index::count]


def merged_rows(inventory: list[dict], shards: list[list[dict]]) -> list[dict]:
    rows = [row for shard in shards for row in shard]
    by_id = {row["query_id"]: row for row in rows}
    expected = [row["query_id"] for row in inventory]
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise RuntimeError("canonical shard union has duplicate or missing queries")
    return [by_id[query_id] for query_id in expected]


def run_command(args: argparse.Namespace) -> None:
    if args.output.exists() or args.manifest_output.exists():
        raise RuntimeError("refusing to reuse canonical shard output")
    inventory = read_jsonl(args.inventory)
    rows = shard_rows(inventory, args.shard_index, args.shard_count)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    done = read_jsonl(partial) if partial.exists() else []
    if [row["query_id"] for row in done] != [row["query_id"] for row in rows[:len(done)]]:
        raise RuntimeError("canonical shard partial is not an exact shard prefix")

    generate, info = load_runtime(args)
    expected_info = json.loads(args.runtime_info.read_text(encoding="utf-8"))
    if info != expected_info:
        raise RuntimeError("canonical inference runtime differs from frozen runtime B")
    with partial.open("a", encoding="utf-8") as handle:
        for row in rows[len(done):]:
            result = {"query_id": row["query_id"], "error": None}
            try:
                answer, tokens, tensor_dtype = generate(row["image_path"], row["question"])
                result.update({
                    "model_answer_raw": answer,
                    "normalized_answer": normalize(answer),
                    "empty": not bool(answer.strip()),
                    "generated_token_count": tokens,
                    "hit_1024_token_limit": tokens >= 1024,
                    "image_tensor_dtype": tensor_dtype,
                })
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            done.append(result)
            if len(done) % 100 == 0 or len(done) == len(rows):
                os.fsync(handle.fileno())
                atomic(args.progress, json.dumps({"completed": len(done), "total": len(rows)}) + "\n")
    errors = sum(bool(row.get("error")) for row in done)
    if len(done) != len(rows) or errors:
        raise RuntimeError(f"canonical shard incomplete or errored: {len(done)}/{len(rows)}, errors={errors}")
    os.replace(partial, args.output)
    manifest = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "query_count": len(done),
        "empty_count": sum(bool(row.get("empty")) for row in done),
        "token_limit_count": sum(bool(row.get("hit_1024_token_limit")) for row in done),
        "output_sha256": sha256(args.output),
        "runtime": "runtime_b_official_native",
    }
    atomic(args.manifest_output, json.dumps(manifest, indent=2, sort_keys=True) + "\n", new=True)


def merge_command(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise RuntimeError("refusing to reuse canonical merge directory")
    inventory = read_jsonl(args.inventory)
    rows = merged_rows(inventory, [read_jsonl(path) for path in args.shards])
    if any(row.get("error") or row.get("empty") for row in rows):
        raise RuntimeError("canonical raw contains errors or empty answers")
    args.output_dir.mkdir(parents=True)
    shard_dir = args.output_dir / "SHARD_MANIFESTS"
    shard_dir.mkdir()
    predictions = args.output_dir / "BASE_PREDICTIONS_CANONICAL.jsonl"
    atomic(predictions, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), new=True)
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in args.shard_manifests]
    for manifest in manifests:
        atomic(
            shard_dir / f"shard_{manifest['shard_index']}.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            new=True,
        )
    runtime = json.loads(args.runtime_info.read_text(encoding="utf-8"))
    checkpoint = json.loads(args.checkpoint_lock.read_text(encoding="utf-8"))
    aggregate = {
        "query_count": len(rows),
        "prediction_sha256": sha256(predictions),
        "query_order": "STATIC_QUERY_INVENTORY",
        "shards": manifests,
        "missing": 0,
        "duplicate": 0,
        "empty": 0,
        "errors": 0,
    }
    atomic(args.output_dir / "BASE_INFERENCE_MANIFEST.json", json.dumps(aggregate, indent=2, sort_keys=True) + "\n", new=True)
    atomic(args.output_dir / "BASE_RUNTIME_PROVENANCE.json", json.dumps({
        "runtime": runtime,
        "checkpoint": checkpoint,
        "editing_methods_rerun": False,
    }, indent=2, sort_keys=True) + "\n", new=True)
    names = ["BASE_PREDICTIONS_CANONICAL.jsonl", "BASE_INFERENCE_MANIFEST.json", "BASE_RUNTIME_PROVENANCE.json"]
    atomic(args.output_dir / "SHA256SUMS.txt", "".join(f"{sha256(args.output_dir / name)}  {name}\n" for name in names), new=True)
    print(json.dumps(aggregate, sort_keys=True))


def main() -> None:
    if len(sys.argv) == 1 and os.environ.get("M3BENCH_PRIVATE_ARGV"):
        sys.argv.extend(json.loads(os.environ["M3BENCH_PRIVATE_ARGV"]))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--runtime", choices=("official",), default="official")
    run.add_argument("--inventory", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--progress", type=Path, required=True)
    run.add_argument("--manifest-output", type=Path, required=True)
    run.add_argument("--runtime-info", type=Path, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--vision-path", type=Path, required=True)
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--llava-root", type=Path, required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--shard-count", type=int, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--inventory", type=Path, required=True)
    merge.add_argument("--shards", type=Path, nargs="+", required=True)
    merge.add_argument("--shard-manifests", type=Path, nargs="+", required=True)
    merge.add_argument("--runtime-info", type=Path, required=True)
    merge.add_argument("--checkpoint-lock", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (run_command if args.command == "run" else merge_command)(args)


if __name__ == "__main__":
    main()
