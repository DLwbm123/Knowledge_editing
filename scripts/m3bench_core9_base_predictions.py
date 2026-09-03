#!/usr/bin/env python3
"""Replay-gate, fill, and freeze Core-9 base predictions without applying edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[^\w\s/-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    for source, target in {
        "x ray": "xray", "x-ray": "xray", "magnetic resonance imaging": "mri",
        "computed tomography": "ct", "ultrasound": "us", "colour": "color",
    }.items():
        value = re.sub(rf"\b{re.escape(source)}\b", target, value)
    numbers = {name: str(index) for index, name in enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))}
    return " ".join(numbers.get(token, token) for token in value.split() if token not in {"a", "an", "the"})


def token_f1(prediction: str, gold: str) -> float:
    p, g = Counter(normalize(prediction).split()), Counter(normalize(gold).split())
    overlap = sum((p & g).values())
    precision = overlap / sum(p.values()) if p else 0.0
    recall = overlap / sum(g.values()) if g else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def exact_fuzzy(prediction: str, gold: str) -> tuple[bool, str]:
    p, g = normalize(prediction), normalize(gold)
    if not p or not g:
        return False, "empty"
    if p == g:
        return True, "normalized_exact"
    return token_f1(p, g) >= 0.80, "normalized_token_f1_ge_0.80"


def semantic_required(gold: str) -> bool:
    normalized = normalize(gold)
    return normalized not in {"yes", "no", "present", "absent", "true", "false", "none"} and len(normalized.split()) > 3


def replay_record_pass(record: dict) -> bool:
    return bool(record["token_ids_equal"] and record["decoded_equal"] and record["normalized_equal"])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    atomic(path, text)


def indexes(source_path: Path, derived_path: Path) -> tuple[dict, dict]:
    source_rows = read_jsonl(source_path)
    derived_rows = read_jsonl(derived_path)
    source = {row["record_id"]: row for row in source_rows}
    derived = {row["derived_probe_id"]: row for row in derived_rows}
    if len(source) != len(source_rows) or len(derived) != len(derived_rows):
        raise RuntimeError("duplicate legacy prediction IDs")
    return source, derived


def reuse_for(row: dict, source: dict, derived: dict) -> tuple[str, dict] | None:
    source_matches = [source[value] for value in row["legacy_source_record_ids"] if value in source]
    derived_matches = [derived[value] for value in row["legacy_derived_probe_ids"] if value in derived]
    matches = [("source", value) for value in source_matches] + [("derived", value) for value in derived_matches]
    if not matches:
        return None
    answers = {value["model_answer_raw"] for _, value in matches}
    if len(answers) != 1:
        raise RuntimeError(f"conflicting exact-key legacy predictions for {row['query_id']}")
    return matches[0]


def adapter(args: argparse.Namespace):
    sys.path.insert(0, str(args.adapter_root))
    from m3bench_repro.inference.llava_med import LlavaMedAdapter

    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    instance = LlavaMedAdapter(runtime["model_path"], runtime["vision_tower_path"], device="cuda:0")
    instance.load()
    generation = {name: runtime.get(name) for name in ("do_sample", "num_beams", "max_new_tokens", "use_cache")}
    return instance, runtime, generation


def prepare(args: argparse.Namespace) -> None:
    inventory = read_jsonl(args.inventory)
    source, derived = indexes(args.source_predictions, args.derived_predictions)
    reusable, missing = [], []
    for row in inventory:
        match = reuse_for(row, source, derived)
        (reusable if match else missing).append(row["query_id"])
    source_pool = sorted(row["query_id"] for row in inventory if row["legacy_source_record_ids"] and reuse_for(row, source, derived))
    derived_pool = sorted(row["query_id"] for row in inventory if row["legacy_derived_probe_ids"] and reuse_for(row, source, derived))
    sample = source_pool[:8] + derived_pool[:8]
    if len(sample) != 16:
        sample = sorted(reusable)[:16]
    report = {
        "status": "READY__BASE_REPLAY_16",
        "inventory_sha256": sha256(args.inventory),
        "runtime_config_sha256": sha256(args.runtime_config),
        "reusable_count": len(reusable),
        "missing_count": len(missing),
        "sample_query_ids": sample,
        "sample_policy": "first 8 stable source IDs plus first 8 stable derived IDs",
        "token_comparison_policy": "explicit legacy raw_token_ids where present; frozen-tokenizer reconstruction from legacy decoded continuation otherwise",
    }
    write_new(args.output_dir / "BASE_REUSE_PLAN.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_new(args.output_dir / "BASE_MISSING_QUERY_IDS.jsonl", "".join(json.dumps({"query_id": value}) + "\n" for value in missing))
    print(json.dumps(report, sort_keys=True))


def replay(args: argparse.Namespace) -> None:
    plan = json.loads((args.output_dir / "BASE_REUSE_PLAN.json").read_text(encoding="utf-8"))
    if plan["inventory_sha256"] != sha256(args.inventory) or plan["runtime_config_sha256"] != sha256(args.runtime_config):
        raise RuntimeError("replay lock mismatch")
    inventory = {row["query_id"]: row for row in read_jsonl(args.inventory)}
    source, derived = indexes(args.source_predictions, args.derived_predictions)
    model, runtime, generation = adapter(args)
    records = []
    for query_id in plan["sample_query_ids"]:
        row = inventory[query_id]
        kind, legacy = reuse_for(row, source, derived) or (None, None)
        if legacy is None:
            raise RuntimeError(f"sample is not reusable: {query_id}")
        result = model.generate_with_result(row["image_path"], row["question"], generation)
        legacy_text = legacy["model_answer_raw"].strip()
        legacy_ids = tuple(int(value) for value in legacy.get("raw_token_ids", []))
        reconstructed = False
        if not legacy_ids:
            legacy_ids = tuple(int(value) for value in model.tokenizer.encode(legacy_text, add_special_tokens=False))
            reconstructed = True
        token_ids_equal = tuple(result.raw_token_ids) == legacy_ids
        decoded_equal = result.decoded_text == legacy_text
        normalized_equal = normalize(result.decoded_text) == normalize(legacy_text)
        records.append({
            "query_id": query_id, "legacy_kind": kind, "token_ids_equal": token_ids_equal,
            "legacy_token_ids_reconstructed": reconstructed, "decoded_equal": decoded_equal,
            "normalized_equal": normalized_equal,
        })
    passed = all(replay_record_pass(row) for row in records)
    report = {
        "status": "PASS__BASE_REPLAY_16_OF_16" if passed else "M3BENCH_CORE9_BASE_REPLAY_MISMATCH",
        "sample_count": len(records),
        "passed_count": sum(replay_record_pass(row) for row in records),
        "runtime_config_sha256": sha256(args.runtime_config),
        "runtime_config_declared_sha256": runtime["config_sha256"],
        "records": records,
    }
    write_new(args.output_dir / "BASE_REPLAY_REPORT.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise SystemExit(3)


def infer(args: argparse.Namespace) -> None:
    replay_report = json.loads((args.output_dir / "BASE_REPLAY_REPORT.json").read_text(encoding="utf-8"))
    if replay_report["status"] != "PASS__BASE_REPLAY_16_OF_16":
        raise RuntimeError("base replay gate did not pass")
    inventory = read_jsonl(args.inventory)
    source, derived = indexes(args.source_predictions, args.derived_predictions)
    missing = [row for row in inventory if reuse_for(row, source, derived) is None]
    lock = {
        "inventory_sha256": sha256(args.inventory),
        "runtime_config_sha256": sha256(args.runtime_config),
        "code_sha256": sha256(Path(__file__)),
        "missing_query_count": len(missing),
    }
    lock_path = args.output_dir / "BASE_INFERENCE_LOCK.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise RuntimeError("inference resume lock mismatch")
    if not lock_path.exists():
        atomic(lock_path, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    output = args.output_dir / "BASE_NEW_PREDICTIONS.jsonl"
    done = {row["query_id"] for row in read_jsonl(output)} if output.exists() else set()
    model, runtime, generation = adapter(args)
    with output.open("a", encoding="utf-8") as handle:
        for ordinal, row in enumerate(missing, 1):
            if row["query_id"] in done:
                continue
            result = model.generate_with_result(row["image_path"], row["question"], generation)
            record = {
                "query_id": row["query_id"], "status": "success", "model_answer_raw": result.decoded_text,
                "raw_token_ids": list(result.raw_token_ids), "normalized_answer": normalize(result.decoded_text),
                "generation_config_sha256": runtime["config_sha256"], "sequence_contract": result.contract.value,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(row["query_id"])
            if ordinal % 100 == 0 or len(done) == len(missing):
                atomic(args.output_dir / "BASE_INFERENCE_PROGRESS.json", json.dumps({"completed": len(done), "total": len(missing)}, indent=2) + "\n")
    if done != {row["query_id"] for row in missing}:
        raise RuntimeError("new base inference incomplete")
    print(json.dumps({"status": "PASS__BASE_NEW_INFERENCE_COMPLETE", "completed": len(done)}, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    inventory = read_jsonl(args.inventory)
    source, derived = indexes(args.source_predictions, args.derived_predictions)
    canonical = {row["record_id"]: row for row in read_jsonl(args.canonical_verdicts)}
    new_rows = read_jsonl(args.output_dir / "BASE_NEW_PREDICTIONS.jsonl") if (args.output_dir / "BASE_NEW_PREDICTIONS.jsonl").exists() else []
    new = {row["query_id"]: row for row in new_rows}
    if len(new) != len(new_rows):
        raise RuntimeError("duplicate new prediction query IDs")
    predictions, verdicts, judge_packet = [], [], []
    route_counts = Counter()
    for row in inventory:
        match = reuse_for(row, source, derived)
        if match:
            kind, prediction = match
            answer = prediction["model_answer_raw"]
            raw_ids = prediction.get("raw_token_ids")
            if kind == "derived":
                is_correct, route = bool(prediction["is_correct"]), prediction["judge_route"]
            else:
                source_id = row["legacy_source_record_ids"][0]
                if source_id in canonical:
                    is_correct, route = bool(canonical[source_id]["is_correct"]), canonical[source_id]["judge_route"]
                else:
                    is_correct, route = exact_fuzzy(answer, row["gold_answer"])[0], "frozen_exact_fuzzy"
        else:
            prediction = new.get(row["query_id"])
            if prediction is None:
                raise RuntimeError(f"missing base prediction for {row['query_id']}")
            kind, answer, raw_ids = "new", prediction["model_answer_raw"], prediction["raw_token_ids"]
            if semantic_required(row["gold_answer"]):
                judge_packet.append({"opaque_query_id": row["query_id"], "question": row["question"], "gold_answer": row["gold_answer"], "model_answer": answer})
                continue
            is_correct, route = exact_fuzzy(answer, row["gold_answer"])[0], "frozen_exact_fuzzy"
        route_counts[route] += 1
        predictions.append({
            "query_id": row["query_id"], "status": "success", "model_answer_raw": answer,
            "raw_token_ids": raw_ids, "source": kind,
        })
        verdicts.append({
            "query_id": row["query_id"], "is_correct": is_correct, "scorer_route": route,
            "generation_config_sha256": json.loads(args.runtime_config.read_text(encoding="utf-8"))["config_sha256"],
        })
    if judge_packet:
        write_new(args.output_dir / "BASE_SEMANTIC_JUDGE_PACKET.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in judge_packet))
        raise RuntimeError(f"M3BENCH_CORE9_DATA_BLOCKED__BASE_SEMANTIC_JUDGE_PENDING__{len(judge_packet)}")
    if len(predictions) != len(inventory) or len(verdicts) != len(inventory):
        raise RuntimeError("base freeze coverage mismatch")
    prediction_path = args.output_dir / "BASE_PREDICTIONS_CORE9.jsonl"
    verdict_path = args.output_dir / "BASE_VERDICTS_CORE9.jsonl"
    write_new(prediction_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
    write_new(verdict_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in verdicts))
    manifest = {
        "status": "PASS__BASE_PREDICTIONS_AND_VERDICTS_FROZEN",
        "query_count": len(inventory), "prediction_count": len(predictions), "verdict_count": len(verdicts),
        "reused_source_count": sum(row["source"] == "source" for row in predictions),
        "reused_derived_count": sum(row["source"] == "derived" for row in predictions),
        "new_inference_count": sum(row["source"] == "new" for row in predictions),
        "correct_count": sum(row["is_correct"] for row in verdicts),
        "incorrect_count": sum(not row["is_correct"] for row in verdicts),
        "route_counts": dict(sorted(route_counts.items())), "semantic_judge_pending": 0,
        "inventory_sha256": sha256(args.inventory), "runtime_config_sha256": sha256(args.runtime_config),
        "predictions_sha256": sha256(prediction_path), "verdicts_sha256": sha256(verdict_path),
    }
    manifest_path = args.output_dir / "BASE_PREDICTION_MANIFEST.json"
    write_new(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_new(args.output_dir / "BASE_PREDICTION_SHA256SUMS.txt", "".join(f"{sha256(path)}  {path.name}\n" for path in (prediction_path, verdict_path, manifest_path)))
    print(json.dumps(manifest, sort_keys=True))


def main() -> None:
    if len(sys.argv) == 1 and os.environ.get("M3BENCH_PRIVATE_ARGV"):
        sys.argv.extend(json.loads(os.environ["M3BENCH_PRIVATE_ARGV"]))
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "replay", "infer", "finalize"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-predictions", type=Path, required=True)
    parser.add_argument("--derived-predictions", type=Path, required=True)
    parser.add_argument("--canonical-verdicts", type=Path)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare": prepare(args)
    elif args.action == "replay": replay(args)
    elif args.action == "infer": infer(args)
    else:
        if args.canonical_verdicts is None:
            parser.error("--canonical-verdicts is required for finalize")
        finalize(args)


if __name__ == "__main__":
    main()
