#!/usr/bin/env python3
"""Freeze and build the operator-approved current-stack M3Bench V4 base."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from scripts.m3bench_base_correctness_v3 import normalize, text_sha256


EXPECTED_QUERY_COUNT = 11_088
OFFICIAL_LLVAMED_COMMIT = "30697ca50b5c29a8e955c99330b259776aef27b9"
PACKAGE_NAMES = ("torch", "transformers", "peft", "accelerate", "tokenizers", "pillow")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: object, *, jsonl: bool = False, text: bool = False) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        str(value)
        if text
        else "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in value)
        if jsonl
        else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    os.chmod(path, 0o444)


def unique(rows: list[dict], key: str) -> dict[str, dict]:
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate {key}")
    return result


def check_source(path: Path) -> None:
    command = lambda *args: subprocess.check_output(  # noqa: E731 - compact immutable probe
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()
    if command("rev-parse", "HEAD") != OFFICIAL_LLVAMED_COMMIT or command("status", "--porcelain"):
        raise RuntimeError("official source checkout is not the frozen clean commit")


def prepare(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise RuntimeError("refusing to reuse V4 output root")
    inventory = read_jsonl(args.inventory)
    old_base = read_jsonl(args.old_base)
    old_verdicts = read_jsonl(args.old_verdicts)
    ids = [row["query_id"] for row in inventory]
    if not (
        len(inventory) == len(set(ids)) == len(old_base) == len(old_verdicts) == EXPECTED_QUERY_COUNT
        and set(ids) == set(unique(old_base, "query_id")) == set(unique(old_verdicts, "query_id"))
    ):
        raise RuntimeError("M3BENCH_V4_BLOCKED__FROZEN_INPUT_COVERAGE")
    check_source(args.official_source)

    temporary = args.output_root.with_name(args.output_root.name + ".tmp")
    temporary.mkdir(parents=True)
    try:
        requirements = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        write_new(temporary / "locks/CURRENT_STACK_REQUIREMENTS_FREEZE.txt", requirements, text=True)
        packages = {}
        for name in PACKAGE_NAMES:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        environment = {
            "schema_version": "m3bench-current-stack-canonical-environment-v4",
            "status": "FROZEN_BEFORE_BASE_V4_OUTPUT",
            "operator_decision": "ADOPT_CURRENT_STACK_AS_FINAL_CANONICAL_AND_SKIP_HISTORICAL_STACK_RECONSTRUCTION",
            "python": sys.version,
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "packages": packages,
            "official_llavamed_source_commit": OFFICIAL_LLVAMED_COMMIT,
            "official_llavamed_source_clean": True,
            "generation_lock_sha256": sha256(args.generation_lock),
            "inventory_sha256": sha256(args.inventory),
            "requirements_freeze_sha256": sha256(temporary / "locks/CURRENT_STACK_REQUIREMENTS_FREEZE.txt"),
            "historical_stack_reconstruction_permitted": False,
        }
        write_new(temporary / "locks/CURRENT_STACK_CANONICAL_ENVIRONMENT.json", environment)
        write_new(temporary / "OPERATOR_DECISION.json", {
            "decision": environment["operator_decision"],
            "parent_commit": args.parent_commit,
            "scoring_code_commit": args.code_commit,
            "old_raw_modified": False,
            "method_output_started": False,
        })

        shards = {2: [], 3: []}
        for index, row in enumerate(inventory):
            shards[2 + index % 2].append({
                "query_id": row["query_id"],
                "question": row["question"],
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
            })
        shard_manifest = {}
        for gpu, rows in shards.items():
            path = temporary / f"private/BASE_V4_GPU{gpu}_INPUTS.jsonl"
            write_new(path, rows, jsonl=True)
            shard_manifest[str(gpu)] = {
                "physical_gpu": gpu,
                "count": len(rows),
                "input_sha256": sha256(path),
                "expected_gpu_uuid": args.gpu_uuid[gpu],
            }
        if sum(row["count"] for row in shard_manifest.values()) != EXPECTED_QUERY_COUNT:
            raise RuntimeError("V4 shard union failure")
        write_new(temporary / "BASE_V4_INFERENCE_MANIFEST.json", {
            "schema_version": "m3bench-base-v4-inference-manifest-v1",
            "status": "FROZEN_BEFORE_BASE_V4_OUTPUT",
            "query_count": EXPECTED_QUERY_COUNT,
            "shard_policy": "inventory_order_even_odd_disjoint",
            "shards": shard_manifest,
            "inventory_sha256": environment["inventory_sha256"],
            "generation_lock_sha256": environment["generation_lock_sha256"],
            "required_per_record_fields": [
                "query_id", "prompt_token_ids", "raw_generated_token_ids", "model_answer_raw",
                "normalized_answer", "image_sha256", "empty", "error",
            ],
            "method_outputs_used": False,
        })
        superseded = []
        for name in ("QUAL8_MANIFEST.json", "LORA_DEV16_MANIFEST.json", "LORA_QUAL16_MANIFEST.json", "LORA_SEQ16_MANIFEST.json"):
            path = args.old_qualification_root / name
            superseded.append({"name": name, "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None})
        write_new(temporary / "SUPERSEDED_V1_SELECTIONS.json", {
            "status": "SUPERSEDED_BY_CURRENT_STACK_BASE_VERDICTS_V4",
            "artifacts": superseded,
            "files_modified_or_deleted": False,
        })
        shutil.copyfile(args.judge_lock, temporary / "private/JUDGE_LOCK_V4.json")
        os.chmod(temporary / "private/JUDGE_LOCK_V4.json", 0o444)
        os.replace(temporary, args.output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": "READY__BASE_V4_TWO_GPU_INFERENCE", "query_count": EXPECTED_QUERY_COUNT, "shards": shard_manifest}, sort_keys=True))


def load_adapter(args: argparse.Namespace):
    from scripts.editor_paperspec_formal import assert_authorized_device, assert_official_llavamed_source

    source = assert_official_llavamed_source()
    assert_authorized_device()
    sys.path.insert(0, str(source))
    if args.runtime == "formal":
        from scripts.editor_paperspec_formal import load_runtime

        runtime = load_runtime(args.cpu_gate, "cuda:0")
        return runtime.adapter, runtime.generation_config
    from m3bench_repro.inference.llava_med import LlavaMedAdapter

    lock = read_json(args.cpu_gate / "locks/FORMAL_MODEL_AND_GENERATION_LOCK.json")
    generation = read_json(args.cpu_gate / "inputs/frozen/llava_med_generation_frozen.json")
    adapter = LlavaMedAdapter(
        lock["generation_config"]["model_path"],
        lock["generation_config"]["vision_tower_path"],
        device="cuda:0",
        load_mode="official_native",
    )
    adapter.load()
    return adapter, generation


def infer(args: argparse.Namespace) -> None:
    inputs = read_jsonl(args.inputs)
    manifest = read_json(args.manifest)
    physical_gpu = int(os.environ["M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES"])
    shard = manifest["shards"].get(str(physical_gpu))
    if args.mode == "base" and (not shard or shard["count"] != len(inputs) or shard["input_sha256"] != sha256(args.inputs)):
        raise RuntimeError("base V4 shard lock mismatch")
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if args.output.exists():
        done = read_jsonl(args.output)
        if len(done) != len(inputs):
            raise RuntimeError("completed output has wrong coverage")
        print(json.dumps({"status": "ALREADY_COMPLETE", "completed": len(done)}))
        return
    done = read_jsonl(partial) if partial.exists() else []
    if [row["query_id"] for row in done] != [row["query_id"] for row in inputs[:len(done)]]:
        raise RuntimeError("inference partial is not an exact input prefix")
    adapter, generation = load_adapter(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a", encoding="utf-8", buffering=1) as handle:
        for ordinal, row in enumerate(inputs[len(done):], len(done) + 1):
            record = {
                "query_id": row["query_id"], "runtime": args.runtime, "prompt_token_ids": [],
                "raw_generated_token_ids": [], "model_answer_raw": "", "normalized_answer": "",
                "image_sha256": "", "generated_token_count": 0, "hit_1024_token_limit": False,
                "empty": True, "error": None,
            }
            try:
                batch = adapter.prepare_inputs(row["image_path"], row["question"])
                if batch["image_sha256"] != row["image_sha256"]:
                    raise RuntimeError("M3BENCH_V4_BLOCKED__STATIC_ASSET_CORRUPTION")
                result = adapter.generate_prepared_with_result(batch, generation)
                prompt_ids = [int(value) for value in batch["input_ids"][0].detach().cpu().tolist()]
                raw_ids = list(result.raw_token_ids)
                record.update({
                    "prompt_token_ids": prompt_ids,
                    "raw_generated_token_ids": raw_ids,
                    "model_answer_raw": result.decoded_text,
                    "normalized_answer": normalize(result.decoded_text),
                    "image_sha256": batch["image_sha256"],
                    "generated_token_count": len(raw_ids),
                    "hit_1024_token_limit": len(raw_ids) >= 1024,
                    "empty": not bool(result.decoded_text.strip()),
                })
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if ordinal % 25 == 0 or record["error"]:
                handle.flush(); os.fsync(handle.fileno())
            progress = args.output.with_suffix(".progress.json")
            temporary = progress.with_suffix(".tmp")
            temporary.write_text(json.dumps({"completed": ordinal, "total": len(inputs)}) + "\n", encoding="utf-8")
            os.replace(temporary, progress)
            if record["error"]:
                raise RuntimeError(record["error"])
        handle.flush(); os.fsync(handle.fileno())
    os.replace(partial, args.output)
    os.chmod(args.output, 0o444)
    print(json.dumps({"status": "PASS", "completed": len(inputs), "runtime": args.runtime}, sort_keys=True))


def merge(args: argparse.Namespace) -> None:
    inventory_rows = read_jsonl(args.inventory)
    inventory = unique(inventory_rows, "query_id")
    old_base = unique(read_jsonl(args.old_base), "query_id")
    old_verdicts = unique(read_jsonl(args.old_verdicts), "query_id")
    generated_rows = [row for path in args.shards for row in read_jsonl(path)]
    generated = unique(generated_rows, "query_id")
    if not (
        len(inventory) == len(generated) == EXPECTED_QUERY_COUNT
        and set(inventory) == set(generated)
        and all(not row["error"] and not row["empty"] for row in generated_rows)
        and all(generated[key]["image_sha256"] == inventory[key]["image_sha256"] for key in inventory)
    ):
        raise RuntimeError("M3BENCH_V4_BLOCKED__11088_INFERENCE_COVERAGE_INCOMPLETE")
    ordered = [generated[row["query_id"]] for row in inventory_rows]
    prediction_path = args.output_root / "private/BASE_PREDICTIONS_V4.jsonl"
    write_new(prediction_path, ordered, jsonl=True)
    reused, packet = [], []
    for row in inventory_rows:
        query_id = row["query_id"]
        if generated[query_id]["model_answer_raw"] == old_base[query_id]["model_answer_raw"]:
            reused.append({"query_id": query_id, "is_correct": old_verdicts[query_id]["is_correct"]})
        else:
            packet.append({
                "opaque_query_id": query_id,
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "raw_base_answer": generated[query_id]["model_answer_raw"],
                "adjudication_pass": 1,
            })
    write_new(args.output_root / "private/BASE_VERDICTS_V4_REUSED.jsonl", reused, jsonl=True)
    write_new(args.output_root / "private/BASE_JUDGE_PACKET_V4.jsonl", packet, jsonl=True)
    write_new(args.output_root / "BASE_V4_MERGE_REPORT.json", {
        "status": "READY_FOR_CHANGED_RAW_JUDGE" if packet else "READY_FOR_BASE_VERDICT_V4_FREEZE",
        "query_count": len(ordered), "raw_exact_reuse_count": len(reused),
        "raw_changed_rejudge_count": len(packet), "empty_count": 0, "error_count": 0,
        "missing_count": 0, "duplicate_count": 0, "image_sha_exact_count": len(ordered),
        "prediction_sha256": sha256(prediction_path), "method_outputs_started": False,
    })
    print(json.dumps({"status": "PASS", "reused": len(reused), "rejudge": len(packet)}, sort_keys=True))


def finalize_verdicts(args: argparse.Namespace) -> None:
    inventory_rows = read_jsonl(args.inventory)
    inventory = unique(inventory_rows, "query_id")
    predictions = unique(read_jsonl(args.predictions), "query_id")
    old_base = unique(read_jsonl(args.old_base), "query_id")
    old_verdicts = unique(read_jsonl(args.old_verdicts), "query_id")
    judge_rows = read_jsonl(args.judge_output) if args.judge_output.is_file() else []
    judge = unique(judge_rows, "opaque_query_id")
    changed = {key for key in predictions if predictions[key]["model_answer_raw"] != old_base[key]["model_answer_raw"]}
    if set(judge) != changed or any(type(row.get("is_correct")) is not bool or not row.get("parse_valid") for row in judge_rows):
        raise RuntimeError("M3BENCH_V4_BLOCKED__SEMANTIC_JUDGE_INCOMPLETE")
    rows = []
    for source in inventory_rows:
        query_id = source["query_id"]
        prediction = predictions[query_id]
        if query_id in changed:
            value = judge[query_id]
            correct, route, votes, reused = value["is_correct"], "semantic_judge_v4_changed_raw", [value["is_correct"]], False
            judge_model = value["judge_model"]
            judge_prompt = value["judge_prompt_sha256"]
            judge_config = value["judge_config_sha256"]
        else:
            value = old_verdicts[query_id]
            correct, route, votes, reused = value["is_correct"], value["authoritative_route"], value["semantic_judge_votes"], True
            judge_model = value["judge_model"]
            judge_prompt = value["judge_prompt_sha256"]
            judge_config = value["judge_config_sha256"]
        rows.append({
            "query_id": query_id, "is_correct": correct, "authoritative_route": route,
            "v3_verdict_reused_by_raw_exact_match": reused, "semantic_judge_used_v4": not reused,
            "semantic_judge_votes": votes, "prediction_sha256": text_sha256(prediction["model_answer_raw"]),
            "question_sha256": text_sha256(source["question"]),
            "gold_sha256": source["gold_sha256"], "image_sha256": source["image_sha256"],
            "judge_model": judge_model, "judge_prompt_sha256": judge_prompt,
            "judge_config_sha256": judge_config,
        })
    verdict_path = args.output_root / "private/BASE_VERDICTS_V4.jsonl"
    write_new(verdict_path, rows, jsonl=True)
    write_new(args.output_root / "BASE_VERDICTS_V4_MANIFEST.json", {
        "schema_version": "m3bench-base-verdict-v4-manifest-v1",
        "status": "BASE_VERDICTS_V4_FROZEN", "query_count": len(rows),
        "correct_count": sum(row["is_correct"] for row in rows),
        "incorrect_count": sum(not row["is_correct"] for row in rows),
        "raw_exact_verdict_reuse_count": len(rows) - len(changed),
        "changed_raw_rejudged_count": len(changed), "missing": 0, "duplicate": 0,
        "prediction_sha256": sha256(args.predictions), "verdict_sha256": sha256(verdict_path),
        "method_outputs_used": False,
    })
    print(json.dumps({"status": "BASE_VERDICTS_V4_FROZEN", "correct": sum(row["is_correct"] for row in rows), "rejudged": len(changed)}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="action", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--inventory", type=Path, required=True)
    command.add_argument("--old-base", type=Path, required=True)
    command.add_argument("--old-verdicts", type=Path, required=True)
    command.add_argument("--generation-lock", type=Path, required=True)
    command.add_argument("--judge-lock", type=Path, required=True)
    command.add_argument("--official-source", type=Path, required=True)
    command.add_argument("--old-qualification-root", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--parent-commit", required=True)
    command.add_argument("--code-commit", required=True)
    command.add_argument("--gpu-uuid", action="append", required=True)
    command.set_defaults(func=prepare)
    command = sub.add_parser("infer")
    command.add_argument("--cpu-gate", type=Path, required=True)
    command.add_argument("--inputs", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--runtime", choices=("official", "formal"), default="official")
    command.add_argument("--mode", choices=("base", "g1r"), default="base")
    command.set_defaults(func=infer)
    command = sub.add_parser("merge")
    command.add_argument("--inventory", type=Path, required=True)
    command.add_argument("--old-base", type=Path, required=True)
    command.add_argument("--old-verdicts", type=Path, required=True)
    command.add_argument("--shards", type=Path, nargs=2, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.set_defaults(func=merge)
    command = sub.add_parser("finalize-verdicts")
    command.add_argument("--inventory", type=Path, required=True)
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--old-base", type=Path, required=True)
    command.add_argument("--old-verdicts", type=Path, required=True)
    command.add_argument("--judge-output", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.set_defaults(func=finalize_verdicts)
    return result


def main() -> None:
    if len(sys.argv) == 1 and os.environ.get("M3BENCH_PRIVATE_ARGV"):
        sys.argv.extend(json.loads(os.environ["M3BENCH_PRIVATE_ARGV"]))
    args = parser().parse_args()
    if hasattr(args, "gpu_uuid"):
        args.gpu_uuid = {int(value.split("=", 1)[0]): value.split("=", 1)[1] for value in args.gpu_uuid}
    args.func(args)


if __name__ == "__main__":
    main()
