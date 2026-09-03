#!/usr/bin/env python3
"""Build and freeze the public task-specific M3Bench T4L cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


T4L_SHA256 = "9232a38cc3c378df8ccdeb40f2573a0f41aa807dbcca447527aebaf5da3c59ba"
T4L_ROWS = 257
SCALAR_FIELDS = ("image_id", "lesion_a", "lesion_b", "question_a", "question_b", "answer_a", "answer_b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", "".join(ch if ch.isalnum() else " " for ch in str(value).casefold())).strip()


def phrase_in(needle: object, haystack: object) -> bool:
    n = normalize(needle).split()
    h = normalize(haystack).split()
    return bool(n) and any(h[i : i + len(n)] == n for i in range(len(h) - len(n) + 1))


def malformed_scalar(value: object) -> bool:
    text = str(value or "").strip()
    return bool(re.search(r"[\u3400-\u9fff]", text) or re.search(r"[\[\]{}]", text))


def query_key(image_id: str, image_hash: str, question: str, gold: str) -> tuple[str, ...]:
    return ("SLAKE", image_id, image_hash, normalize(question), normalize(gold))


def query_id(key: tuple[str, ...]) -> str:
    return "core9q-" + canonical_hash(key)[:24]


def candidate_id(row: dict) -> str:
    return "t4l-" + canonical_hash({key: row[key] for key in SCALAR_FIELDS})[:24]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    write_new(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    write_new(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def dedupe_queries(rows: list[dict]) -> list[dict]:
    merged: dict[tuple[str, ...], dict] = {}
    for row in rows:
        key = query_key(row["image_id"], row["image_sha256"], row["question"], row["gold_answer"])
        if key not in merged:
            merged[key] = {**row, "query_id": query_id(key), "lineage": []}
        for lineage in row.get("lineage", []):
            if lineage not in merged[key]["lineage"]:
                merged[key]["lineage"].append(lineage)
    return sorted(merged.values(), key=lambda row: row["query_id"])


def build_candidates(csv_path: Path, image_root: Path, expected_sha256: str = T4L_SHA256) -> tuple[list[dict], list[dict], list[dict]]:
    actual_sha = sha256(csv_path)
    if actual_sha != expected_sha256:
        raise RuntimeError(f"T4L source SHA mismatch: {actual_sha}")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != T4L_ROWS:
        raise RuntimeError(f"expected {T4L_ROWS} T4L rows, found {len(rows)}")

    candidates: list[dict] = []
    queries: list[dict] = []
    rejections: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    image_hashes: dict[Path, str] = {}
    for index, source in enumerate(rows, 1):
        row = {key: str(source.get(key, "")).strip() for key in SCALAR_FIELDS}
        reasons = []
        if any(not row[key] for key in SCALAR_FIELDS):
            reasons.append("empty_required_field")
        if any(malformed_scalar(row[key]) for key in SCALAR_FIELDS):
            reasons.append("malformed_list_or_cjk_scalar")
        if normalize(row["lesion_a"]) == normalize(row["lesion_b"]):
            reasons.append("same_lesion")
        if normalize(row["question_a"]) == normalize(row["question_b"]):
            reasons.append("same_question")
        if not phrase_in(row["lesion_a"], row["question_a"]):
            reasons.append("question_a_lesion_a_role_mismatch")
        if not phrase_in(row["lesion_b"], row["question_b"]):
            reasons.append("question_b_lesion_b_role_mismatch")
        image_path = image_root / row["image_id"] / "source.jpg"
        if not image_path.is_file():
            reasons.append("missing_image")
        exact = tuple(normalize(row[key]) for key in SCALAR_FIELDS)
        if exact in seen:
            reasons.append("exact_duplicate")
        seen.add(exact)
        if reasons:
            rejections.append({"source_metadata_row": index, "reasons": sorted(set(reasons))})
            continue

        if image_path not in image_hashes:
            image_hashes[image_path] = sha256(image_path)
        image_hash = image_hashes[image_path]
        lineage_base = {"source_task": "T4L", "source_metadata_file": csv_path.name, "source_metadata_row": index}
        key_a = query_key(row["image_id"], image_hash, row["question_a"], row["answer_a"])
        key_b = query_key(row["image_id"], image_hash, row["question_b"], row["answer_b"])
        cid = candidate_id(row)
        candidates.append({
            "candidate_id": cid,
            "dataset": "SLAKE",
            "image_id": row["image_id"],
            "relative_image_path": f"{row['image_id']}/source.jpg",
            "image_sha256": image_hash,
            "lesion_a": row["lesion_a"],
            "question_a": row["question_a"],
            "answer_a": row["answer_a"],
            "edit_query_id": query_id(key_a),
            "lesion_b": row["lesion_b"],
            "question_b": row["question_b"],
            "answer_b": row["answer_b"],
            "probe_query_id": query_id(key_b),
            "source_metadata_row": index,
            "review_flags": [],
            "amended189_used_as_t4l_anchor": False,
        })
        for role, question, gold in (("edit_target_a", row["question_a"], row["answer_a"]), ("locality_probe_b", row["question_b"], row["answer_b"])):
            queries.append({
                "dataset": "SLAKE",
                "image_id": row["image_id"],
                "relative_image_path": f"{row['image_id']}/source.jpg",
                "image_sha256": image_hash,
                "question": question,
                "normalized_question": normalize(question),
                "gold_answer": gold,
                "normalized_gold": normalize(gold),
                "source_task": "T4L",
                "source_metadata_file": csv_path.name,
                "source_metadata_row": index,
                "relation_id": cid,
                "role": role,
                "lineage": [{**lineage_base, "relation_id": cid, "role": role}],
            })
    return candidates, dedupe_queries(queries), rejections


def checksum_file(root: Path, names: list[str]) -> None:
    lines = [f"{sha256(root / name)}  {name}\n" for name in names]
    write_new(root / "SHA256SUMS.txt", "".join(lines))


def build_command(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise RuntimeError("refusing to reuse T4L build directory")
    args.output_dir.mkdir(parents=True)
    candidates, queries, rejections = build_candidates(args.csv, args.slake_image_root, args.expected_sha256)
    write_jsonl(args.output_dir / "T4L_CANDIDATES.jsonl", candidates)
    write_jsonl(args.output_dir / "T4L_BASE_QUERY_INVENTORY.jsonl", queries)
    write_jsonl(args.output_dir / "T4L_STRUCTURAL_REJECTIONS.jsonl", rejections)
    reasons = Counter(reason for row in rejections for reason in row["reasons"])
    manifest = {
        "schema_version": "m3bench-t4l-task-specific-v1",
        "public_commit": args.public_commit,
        "source_file": args.csv.name,
        "source_sha256": sha256(args.csv),
        "source_row_count": T4L_ROWS,
        "candidate_count": len(candidates),
        "unique_image_count": len({row["image_id"] for row in candidates}),
        "unique_base_query_count": len(queries),
        "rejection_row_count": len(rejections),
        "rejection_reason_counts": dict(sorted(reasons.items())),
        "amended189_used_as_t4l_anchor": False,
    }
    write_json(args.output_dir / "T4L_CANDIDATE_MANIFEST.json", manifest)
    write_new(args.output_dir / "T4L_CANDIDATE_REPORT.md", "\n".join([
        "# T4L task-specific candidate report", "",
        f"- public source rows: {T4L_ROWS}",
        f"- structurally valid candidates: {len(candidates)}",
        f"- unique images: {manifest['unique_image_count']}",
        f"- unique base queries: {len(queries)}",
        f"- rejected rows: {len(rejections)}",
        "- question A is the edit target; question B is the locality probe.",
        "- amended-189 membership is not used.", "",
    ]))
    checksum_file(args.output_dir, ["T4L_CANDIDATES.jsonl", "T4L_BASE_QUERY_INVENTORY.jsonl", "T4L_STRUCTURAL_REJECTIONS.jsonl", "T4L_CANDIDATE_MANIFEST.json", "T4L_CANDIDATE_REPORT.md"])
    print(json.dumps(manifest, sort_keys=True))


def freeze_command(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise RuntimeError("refusing to reuse T4L freeze directory")
    args.output_dir.mkdir(parents=True)
    candidates = jsonl(args.candidates)
    verdict_rows = jsonl(args.base_verdicts)
    verdicts = {row["query_id"]: row for row in verdict_rows}
    audit, formal, queue = [], [], []
    missing = set()
    for row in candidates:
        a = verdicts.get(row["edit_query_id"])
        b = verdicts.get(row["probe_query_id"])
        if a is None:
            missing.add(row["edit_query_id"])
        if b is None:
            missing.add(row["probe_query_id"])
        if a is None or b is None:
            continue
        a_correct, b_correct = bool(a["is_correct"]), bool(b["is_correct"])
        eligible = not a_correct and b_correct
        reason = "eligible" if eligible else ("question_a_base_correct" if a_correct else "question_b_base_wrong")
        audit.append({"candidate_id": row["candidate_id"], "edit_query_id": row["edit_query_id"], "probe_query_id": row["probe_query_id"], "question_a_base_correct": a_correct, "question_b_base_correct": b_correct, "eligible": eligible, "reason": reason})
        queue.append({key: row[key] for key in ("candidate_id", "image_id", "relative_image_path", "image_sha256", "lesion_a", "question_a", "lesion_b", "question_b", "review_flags")})
        if eligible:
            config_hash = a.get("generation_config_sha256", a.get("config_hash"))
            if not config_hash:
                raise RuntimeError("T4L base verdict is missing generation config binding")
            formal.append({**row, "task": "T4L", "event_key": f"T4L:{row['candidate_id']}", "expected_behavior": "no_change", "base_config_hash": config_hash})
    if missing:
        raise RuntimeError(f"missing {len(missing)} T4L base verdicts")
    if len({row["event_key"] for row in formal}) != len(formal):
        raise RuntimeError("duplicate T4L event key")

    write_jsonl(args.output_dir / "T4L_FORMAL_RECORDS.jsonl", formal)
    write_jsonl(args.output_dir / "T4L_BASE_ELIGIBILITY_AUDIT.jsonl", audit)
    write_jsonl(args.output_dir / "T4L_IMAGE_REVIEW_QUEUE.jsonl", queue)
    reasons = Counter(row["reason"] for row in audit)
    manifest = {
        "schema_version": "m3bench-t4l-task-specific-formal-v1",
        "candidate_count": len(candidates),
        "eligible_edit_count": len(formal),
        "eligible_probe_count": len(formal),
        "unique_image_count": len({row["image_id"] for row in formal}),
        "eligibility_reason_counts": dict(sorted(reasons.items())),
        "authority": "public metadata + deterministic structural filter + frozen base eligibility",
        "question_a_base_correct_required": False,
        "question_b_base_correct_required": True,
        "amended189_used_as_t4l_anchor": False,
        "status": "PASS" if formal else "M3BENCH_T4L_BLOCKED__NO_BASE_ELIGIBLE_EDIT_INSTANCES",
    }
    write_json(args.output_dir / "T4L_FORMAL_MANIFEST.json", manifest)
    write_new(args.output_dir / "T4L_FORMAL_COHORT_REPORT.md", "\n".join([
        "# T4L formal cohort", "", f"- candidates: {len(candidates)}", f"- eligible qA-wrong/qB-correct edits: {len(formal)}", f"- unique images: {manifest['unique_image_count']}", f"- status: {manifest['status']}", "",
    ]))
    checksum_file(args.output_dir, ["T4L_FORMAL_RECORDS.jsonl", "T4L_BASE_ELIGIBILITY_AUDIT.jsonl", "T4L_IMAGE_REVIEW_QUEUE.jsonl", "T4L_FORMAL_MANIFEST.json", "T4L_FORMAL_COHORT_REPORT.md"])
    print(json.dumps(manifest, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--csv", type=Path, required=True)
    build.add_argument("--slake-image-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--expected-sha256", default=T4L_SHA256)
    build.add_argument("--public-commit", default="03c6fda3813301dab3be5831fdc94b493c10afc9")
    build.set_defaults(func=build_command)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--candidates", type=Path, required=True)
    freeze.add_argument("--base-verdicts", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.set_defaults(func=freeze_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
