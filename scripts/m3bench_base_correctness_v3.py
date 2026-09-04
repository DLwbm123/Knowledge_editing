#!/usr/bin/env python3
"""Score frozen base answers with legacy, public-fuzzy, and semantic routes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_OLD_RAW_SHA256 = "25006913f849d7fedfe0fc100a789badad2ef093c09e9614fa511d4ed73251dc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def normalize(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[^\w\s/-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    aliases = {
        "x ray": "xray", "x-ray": "xray", "magnetic resonance imaging": "mri",
        "computed tomography": "ct", "ultrasound": "us", "colour": "color",
    }
    for source, target in aliases.items():
        value = re.sub(rf"\b{re.escape(source)}\b", target, value)
    numbers = {name: str(index) for index, name in enumerate(
        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    )}
    return " ".join(numbers.get(token, token) for token in value.split() if token not in {"a", "an", "the"})


def token_f1(prediction: object, gold: object) -> float:
    p, g = Counter(normalize(prediction).split()), Counter(normalize(gold).split())
    overlap = sum((p & g).values())
    precision = overlap / sum(p.values()) if p else 0.0
    recall = overlap / sum(g.values()) if g else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def legacy_correct(prediction: object, gold: object) -> bool:
    p, g = normalize(prediction), normalize(gold)
    return bool(p and g and (p == g or token_f1(p, g) >= 0.80))


def public_fuzzy_correct(prediction: object, gold: object) -> bool:
    p, g = normalize(prediction), normalize(gold)
    return bool(p and g and (p == g or g in p or p in g))


def exact_correct(prediction: object, gold: object) -> bool:
    p, g = normalize(prediction), normalize(gold)
    return bool(p and g and p == g)


def majority(votes: list[bool], gate_critical: bool) -> bool:
    if gate_critical:
        if len(votes) not in {2, 3}:
            raise ValueError("gate-critical query requires two votes, or three after disagreement")
        if len(votes) == 2 and votes[0] != votes[1]:
            raise ValueError("third blind adjudication required")
    elif len(votes) != 1:
        raise ValueError("non-gate-critical query requires exactly one vote")
    return sum(votes) > len(votes) / 2


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_new(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def load_unique(path: Path, key: str) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate {key} in {path}")
    return result


def verify_old_raw(path: Path) -> None:
    rows = read_jsonl(path)
    if len(rows) != 11088 or sha256(path) != EXPECTED_OLD_RAW_SHA256:
        raise RuntimeError("M3BENCH_DATA_FINALIZATION_BLOCKED__BASE_RAW_HASH_MISMATCH")


def make_packet(inventory: list[dict], predictions: dict[str, dict], gate_ids: set[str], pass_number: int) -> list[dict]:
    if pass_number not in {1, 2}:
        raise ValueError("initial packet pass must be 1 or 2")
    packet = []
    for row in inventory:
        query_id = row["query_id"]
        answer = predictions[query_id]["model_answer_raw"]
        if exact_correct(answer, row["gold_answer"]):
            continue
        if pass_number == 2 and query_id not in gate_ids:
            continue
        packet.append({
            "opaque_query_id": query_id,
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "raw_base_answer": answer,
            "adjudication_pass": pass_number,
        })
    return packet


def valid_vote(row: dict) -> bool:
    return type(row.get("is_correct")) is bool


def replacement_ids(first: dict[str, dict], second: dict[str, dict]) -> list[str]:
    if not set(second) <= set(first):
        raise RuntimeError("second-pass query is absent from first pass")
    ids = {query_id for query_id, row in first.items() if not valid_vote(row)}
    ids.update(
        query_id for query_id, row in second.items()
        if not valid_vote(row) or not valid_vote(first[query_id])
        or first[query_id]["is_correct"] != row["is_correct"]
    )
    return sorted(ids)


def third_pass_packet(first: dict[str, dict], second: dict[str, dict], packet: dict[str, dict]) -> list[dict]:
    ids = replacement_ids(first, second)
    return [{**packet[query_id], "adjudication_pass": 3} for query_id in ids]


def vote_index(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {str(row["opaque_query_id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate Judge query in {path}")
    return result


def merge_valid_votes(paths: list[Path]) -> list[dict]:
    output, keys = [], set()
    for path in paths:
        for row in read_jsonl(path):
            if not valid_vote(row):
                continue
            key = (str(row["opaque_query_id"]), int(row["adjudication_pass"]))
            if key in keys:
                raise RuntimeError("duplicate valid Judge vote")
            keys.add(key); output.append(row)
    return sorted(output, key=lambda row: (str(row["opaque_query_id"]), int(row["adjudication_pass"])))


def validate_judge_lock(lock: dict) -> tuple[str, str, str]:
    required = {"judge_model", "prompt", "temperature", "schema"}
    if not required <= lock.keys() or lock["temperature"] != 0 or lock["schema"] != "strict_json_boolean":
        raise RuntimeError("invalid semantic Judge lock")
    prompt_hash = text_sha256(lock["prompt"])
    config_hash = text_sha256(json.dumps(lock, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return str(lock["judge_model"]), prompt_hash, config_hash


def build_verdicts(
    inventory: list[dict], predictions: dict[str, dict], gate_ids: set[str],
    vote_rows: list[dict], checkpoint_sha: str, runtime_config_sha: str, judge_lock: dict,
) -> list[dict]:
    judge_model, prompt_hash, judge_config_hash = validate_judge_lock(judge_lock)
    votes: dict[str, dict[int, bool]] = defaultdict(dict)
    for row in vote_rows:
        query_id, pass_number, verdict = str(row["opaque_query_id"]), int(row["adjudication_pass"]), row["is_correct"]
        if type(verdict) is not bool or pass_number not in {1, 2, 3} or pass_number in votes[query_id]:
            raise RuntimeError("invalid or duplicate semantic Judge vote")
        votes[query_id][pass_number] = verdict
    output = []
    for row in inventory:
        query_id = row["query_id"]
        answer = predictions[query_id]["model_answer_raw"]
        exact = exact_correct(answer, row["gold_answer"])
        query_votes = [value for _, value in sorted(votes.get(query_id, {}).items())]
        if exact:
            final, route = True, "normalized_exact"
            if query_votes:
                raise RuntimeError("exact answer must bypass semantic Judge")
        else:
            final = majority(query_votes, query_id in gate_ids)
            route = "semantic_judge_majority" if query_id in gate_ids else "semantic_judge_single"
        output.append({
            "query_id": query_id,
            "is_correct": final,
            "authoritative_route": route,
            "legacy_token_f1_verdict": legacy_correct(answer, row["gold_answer"]),
            "public_release_fuzzy_verdict": public_fuzzy_correct(answer, row["gold_answer"]),
            "semantic_judge_used": not exact,
            "semantic_judge_votes": query_votes,
            "prediction_sha256": text_sha256(answer),
            "question_sha256": text_sha256(row["question"]),
            "gold_sha256": text_sha256(row["gold_answer"]),
            "image_sha256": row["image_sha256"],
            "checkpoint_snapshot_sha": checkpoint_sha,
            "runtime_config_sha256": runtime_config_sha,
            "judge_model": judge_model,
            "judge_prompt_sha256": prompt_hash,
            "judge_config_sha256": judge_config_hash,
        })
    if len(output) != len(inventory):
        raise RuntimeError("semantic verdict coverage mismatch")
    return output


def answer_type(gold: object) -> str:
    value = normalize(gold)
    if value in {"yes", "no", "true", "false", "present", "absent", "none"}:
        return "binary"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        return "numeric"
    return "short_entity" if len(value.split()) <= 3 else "free_text"


def disagreement_rows(inventory: list[dict], verdicts: dict[str, dict], field: str) -> list[dict]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in inventory:
        result = verdicts[row["query_id"]]
        group = row.get(field) if field != "answer_type" else answer_type(row["gold_answer"])
        group = str(group or "UNKNOWN")
        counts[group]["total"] += 1
        counts[group]["legacy_true_semantic_false"] += bool(result["legacy_token_f1_verdict"] and not result["is_correct"])
        counts[group]["legacy_false_semantic_true"] += bool(not result["legacy_token_f1_verdict"] and result["is_correct"])
        counts[group]["public_true_semantic_false"] += bool(result["public_release_fuzzy_verdict"] and not result["is_correct"])
        counts[group]["public_false_semantic_true"] += bool(not result["public_release_fuzzy_verdict"] and result["is_correct"])
    return [dict(group=group, **dict(value)) for group, value in sorted(counts.items())]


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["group"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    packet = sub.add_parser("packet")
    final = sub.add_parser("finalize")
    third = sub.add_parser("third")
    third.add_argument("--pass1", type=Path, required=True); third.add_argument("--pass2", type=Path, required=True)
    third.add_argument("--packet1", type=Path, required=True); third.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True); merge.add_argument("--output", type=Path, required=True)
    for child in (packet, final):
        child.add_argument("--inventory", type=Path, required=True)
        child.add_argument("--predictions", type=Path, required=True)
        child.add_argument("--gate-ids", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--require-old-raw", action="store_true")
    packet.add_argument("--pass-number", type=int, choices=(1, 2), required=True)
    final.add_argument("--judge-votes", type=Path, required=True)
    final.add_argument("--judge-lock", type=Path, required=True)
    final.add_argument("--checkpoint-sha", required=True)
    final.add_argument("--runtime-config-sha", required=True)
    args = parser.parse_args()
    if args.command == "third":
        packet1 = vote_index(args.packet1)
        write_jsonl(args.output, third_pass_packet(vote_index(args.pass1), vote_index(args.pass2), packet1))
        return
    if args.command == "merge":
        write_jsonl(args.output, merge_valid_votes(args.inputs))
        return
    if args.require_old_raw:
        verify_old_raw(args.predictions)
    inventory = read_jsonl(args.inventory)
    predictions = load_unique(args.predictions, "query_id")
    gate_ids = set(json.loads(args.gate_ids.read_text(encoding="utf-8"))["query_ids"])
    if set(predictions) != {row["query_id"] for row in inventory}:
        raise RuntimeError("inventory/prediction coverage mismatch")
    if args.command == "packet":
        name = f"BASE_JUDGE_PACKET_V3_PASS{args.pass_number}.jsonl"
        write_jsonl(args.output_dir / name, make_packet(inventory, predictions, gate_ids, args.pass_number))
        return
    verdict_rows = build_verdicts(
        inventory, predictions, gate_ids, read_jsonl(args.judge_votes),
        args.checkpoint_sha, args.runtime_config_sha,
        json.loads(args.judge_lock.read_text(encoding="utf-8")),
    )
    write_jsonl(args.output_dir / "BASE_VERDICTS_V3_EXISTING_RAW.jsonl", verdict_rows)
    by_id = {row["query_id"]: row for row in verdict_rows}
    write_csv(args.output_dir / "SCORER_DISAGREEMENT_BY_TASK.csv", disagreement_rows(inventory, by_id, "source_task"))
    write_csv(args.output_dir / "SCORER_DISAGREEMENT_BY_ANSWER_TYPE.csv", disagreement_rows(inventory, by_id, "answer_type"))


if __name__ == "__main__":
    main()
