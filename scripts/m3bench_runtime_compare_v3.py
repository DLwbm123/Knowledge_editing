#!/usr/bin/env python3
"""Blindly score and compare the two frozen LLaVA-Med runtime canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m3bench_base_correctness_v3 import exact_correct, normalize
from scripts.m3bench_static_catalog_v3 import select_runtime


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path: Path, value: object, *, jsonl: bool = False) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in value)
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def opaque_id(runtime: str, query_id: str) -> str:
    return "rtq-" + hashlib.sha256(f"{runtime}\0{query_id}".encode()).hexdigest()[:24]


def aligned(manifest: dict, path: Path) -> list[dict]:
    rows = read_jsonl(path)
    expected = [row["query_id"] for row in manifest["records"]]
    if [row.get("query_id") for row in rows] != expected:
        raise RuntimeError(f"runtime output is not manifest-aligned: {path}")
    return rows


def packet_command(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    inventory = {row["query_id"]: row for row in manifest["records"]}
    packets, mapping = [], []
    for runtime, path in (("a", args.runtime_a), ("b", args.runtime_b)):
        for row in aligned(manifest, path):
            query_id = row["query_id"]
            if row.get("error"):
                raise RuntimeError(f"runtime {runtime} contains generation errors")
            answer = row.get("raw_answer", "")
            if normalize(answer) != row.get("normalized_answer"):
                raise RuntimeError("stored normalized answer does not match scorer-v3")
            exact = exact_correct(answer, inventory[query_id]["gold_answer"])
            oid = opaque_id(runtime, query_id)
            mapping.append({"opaque_query_id": oid, "runtime": runtime, "query_id": query_id, "exact": exact})
            if not exact:
                packets.append({
                    "opaque_query_id": oid,
                    "question": inventory[query_id]["question"],
                    "gold_answer": inventory[query_id]["gold_answer"],
                    "raw_base_answer": answer,
                    "adjudication_pass": 1,
                })
    if len({row["opaque_query_id"] for row in mapping}) != len(mapping):
        raise RuntimeError("opaque runtime query collision")
    write_new(args.packet, packets, jsonl=True)
    write_new(args.mapping, mapping, jsonl=True)
    print(json.dumps({"runtime_outputs": len(mapping), "judge_packet": len(packets)}, sort_keys=True))


def runtime_stats(rows: list[dict], semantic: dict[str, bool], strata: dict[str, str]) -> dict:
    counts = Counter()
    by_stratum: dict[str, Counter] = {}
    for row in rows:
        query_id = row["query_id"]
        counts["total"] += 1
        counts["nonempty"] += not row.get("empty", True)
        counts["errors"] += bool(row.get("error"))
        counts["hit_1024_token_limit"] += bool(row.get("hit_1024_token_limit"))
        if "image_swap_normalized_answer" in row:
            counts["image_swap_checked"] += 1
            counts["image_swap_changed"] += row["image_swap_normalized_answer"] != row["normalized_answer"]
        group = strata[query_id]
        bucket = by_stratum.setdefault(group, Counter())
        bucket["total"] += 1
        bucket["correct"] += semantic[query_id]
    source = sum((value for key, value in by_stratum.items() if key in {"SLAKE_ORIGINAL", "VQARAD_ORIGINAL"}), Counter())
    return {
        **dict(counts),
        "nonempty_rate": counts["nonempty"] / counts["total"],
        "source_semantic_accuracy": source["correct"] / source["total"] if source["total"] else None,
        "strata": {key: dict(value) for key, value in sorted(by_stratum.items())},
    }


def finalize_command(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    a, b = aligned(manifest, args.runtime_a), aligned(manifest, args.runtime_b)
    mapping = read_jsonl(args.mapping)
    votes = {row["opaque_query_id"]: row for row in read_jsonl(args.judge_verdicts)}
    semantic = {"a": {}, "b": {}}
    for row in mapping:
        if row["exact"]:
            value = True
        else:
            vote = votes.get(row["opaque_query_id"])
            if vote is None or type(vote.get("is_correct")) is not bool:
                raise RuntimeError("runtime semantic Judge coverage is incomplete")
            value = vote["is_correct"]
        semantic[row["runtime"]][row["query_id"]] = value
    if set(votes) != {row["opaque_query_id"] for row in mapping if not row["exact"]}:
        raise RuntimeError("runtime semantic Judge has extra or missing rows")

    total = len(a)
    decoded = sum(left["raw_answer"] == right["raw_answer"] for left, right in zip(a, b))
    normalized = sum(left["normalized_answer"] == right["normalized_answer"] for left, right in zip(a, b))
    semantic_equal = sum(semantic["a"][row["query_id"]] == semantic["b"][row["query_id"]] for row in a)
    strata = {row["query_id"]: row["stratum"] for row in manifest["records"]}
    stats_a, stats_b = runtime_stats(a, semantic["a"], strata), runtime_stats(b, semantic["b"], strata)
    checkpoint = json.loads(args.checkpoint_lock.read_text(encoding="utf-8"))
    info_a = json.loads(args.runtime_a_info.read_text(encoding="utf-8"))
    info_b = json.loads(args.runtime_b_info.read_text(encoding="utf-8"))
    generation = info_b.get("generation", {})
    audit = {
        "checkpoint_identity_verified": checkpoint.get("local_checkpoint_matches_official_snapshot") is True,
        "native_runtime_stable": stats_b["errors"] == 0 and stats_b["nonempty"] == total,
        "official_prompt_image_generation": info_b.get("runtime") == "official" and generation == {
            "batch_size": 1, "do_sample": False, "max_new_tokens": 1024,
            "num_beams": 1, "temperature": 0, "use_cache": True,
        },
        "no_runtime_errors": stats_a["errors"] == stats_b["errors"] == 0,
        "decoded_parity": decoded / total,
        "normalized_parity": normalized / total,
        "semantic_parity": semantic_equal / total,
        "source_accuracy": stats_b["source_semantic_accuracy"],
    }
    selected = select_runtime(audit)
    result = {
        "canary_count": total,
        "selected_runtime": selected,
        "audit": audit,
        "runtime_a": stats_a,
        "runtime_b": stats_b,
        "runtime_a_info": info_a,
        "runtime_b_info": info_b,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_new(args.output_dir / "RUNTIME_AB_AUDIT.json", result)
    write_new(args.output_dir / "CANONICAL_LLVAMED_RUNTIME_LOCK.json", {
        "selected_runtime": selected,
        "model_snapshot_sha": checkpoint["model_snapshot_sha"],
        "vision_snapshot_sha": checkpoint["vision_snapshot_sha"],
        "llava_med_code_commit": checkpoint["llava_med_code_commit"],
        "generation": generation,
        "parity": {key: audit[key] for key in ("decoded_parity", "normalized_parity", "semantic_parity")},
    })
    write_new(args.output_dir / "EDITOR_RUNTIME_HANDOFF.json", {
        "required_runtime": selected,
        "editing_methods_started": False,
        "require_official_environment_for_future_editing": selected == "runtime_b_official_native",
    })
    report = [
        "# Runtime A/B report", "",
        f"- Canary: {total}/{total}",
        f"- Decoded parity: {decoded}/{total} ({audit['decoded_parity']:.6f})",
        f"- Normalized parity: {normalized}/{total} ({audit['normalized_parity']:.6f})",
        f"- Semantic verdict parity: {semantic_equal}/{total} ({audit['semantic_parity']:.6f})",
        f"- Runtime A non-empty/errors: {stats_a['nonempty']}/{stats_a['errors']}",
        f"- Runtime B non-empty/errors: {stats_b['nonempty']}/{stats_b['errors']}",
        f"- Runtime A image-swap changed: {stats_a.get('image_swap_changed', 0)}/{stats_a.get('image_swap_checked', 0)}",
        f"- Runtime B image-swap changed: {stats_b.get('image_swap_changed', 0)}/{stats_b.get('image_swap_checked', 0)}",
        f"- Selected runtime: `{selected}`", "",
    ]
    report_path = args.output_dir / "RUNTIME_AB_REPORT.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(report), encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({"selected_runtime": selected, **audit}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    packet = sub.add_parser("packet")
    final = sub.add_parser("finalize")
    for child in (packet, final):
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--runtime-a", type=Path, required=True)
        child.add_argument("--runtime-b", type=Path, required=True)
    packet.add_argument("--packet", type=Path, required=True)
    packet.add_argument("--mapping", type=Path, required=True)
    final.add_argument("--mapping", type=Path, required=True)
    final.add_argument("--judge-verdicts", type=Path, required=True)
    final.add_argument("--runtime-a-info", type=Path, required=True)
    final.add_argument("--runtime-b-info", type=Path, required=True)
    final.add_argument("--checkpoint-lock", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (packet_command if args.command == "packet" else finalize_command)(args)


if __name__ == "__main__":
    main()
