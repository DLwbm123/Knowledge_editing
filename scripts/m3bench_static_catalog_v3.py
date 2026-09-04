#!/usr/bin/env python3
"""Freeze the base-independent public-release-aligned M3Bench T0-T4 catalog."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from scripts.m3bench_core9_task_specific_t4l import canonical_hash, normalize


T5_STATUS = "M3BENCH_T5_SEPARATE_EXTENSION_BLOCKED__PADCHEST_GR_ASSETS_UNAVAILABLE"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def hash_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    atomic_new(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_new(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def public_query(row: dict) -> dict:
    source_ids = row.get("legacy_source_record_ids", [])
    return {
        "query_id": row["query_id"],
        "dataset": row["dataset"],
        "image_id": row["image_id"],
        "image_sha256": row["image_sha256"],
        "question": row["question"],
        "normalized_question": normalize(row["question"]),
        "question_id": source_ids[0] if source_ids else row["query_id"],
        "question_id_kind": "public_or_frozen_id" if source_ids else "stable_synthetic_id",
        "gold_answer": row["gold_answer"],
        "gold_sha256": hash_text(row["gold_answer"]),
        "source_file": row["source_metadata_file"],
        "source_row": row["source_metadata_row"],
        "relation_task": row["source_task"],
        "role": row["role"],
        "image_path": row["image_path"],
        "lineage": row["lineage"],
        "method_outputs_used": False,
    }


def lookup(rows: list[dict]) -> tuple[dict[tuple[str, str], list[dict]], dict[tuple[str, str, str, str], dict]]:
    by_image: dict[tuple[str, str], list[dict]] = defaultdict(list)
    exact = {}
    for row in rows:
        by_image[(row["dataset"], row["image_id"])].append(row)
        exact[(row["dataset"], str(row["image_path"]), normalize(row["question"]), normalize(row["gold_answer"]))] = row
    return by_image, exact


def dataset_for(image_id: str) -> str:
    return "VQA-RAD" if image_id.endswith(".jpg") else "SLAKE"


def member(row: dict, role: str) -> dict:
    result = public_query(row)
    result["role"] = role
    return result


def relation(source_file: str, source_row: int, task: str, members: list[dict], extra: dict | None = None) -> dict:
    return {
        "relation_id": task.casefold() + "-" + canonical_hash((source_file, source_row, task))[:24],
        "task": task,
        "source_file": source_file,
        "source_row": source_row,
        "members": members,
        "method_outputs_used": False,
        **(extra or {}),
    }


def t0_candidates(amended: list[dict], exact: dict) -> list[dict]:
    output = []
    for item in amended:
        key = (item["dataset"], str(item["image_path"]), normalize(item["question"]), normalize(item["gold_answer"]))
        row = exact.get(key)
        if row is None:
            raise RuntimeError(f"unresolved amended target at position {item['amended_position']}")
        output.append({
            **member(row, "target_validity_approved_candidate"),
            "amended_position": item["amended_position"],
            "original_position": item["original_position"],
            "target_validity_approved": True,
        })
    return output


def formal_relations(catalog: list[dict], exact: dict, tasks: set[str], label: str) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in catalog:
        if item["task"] not in tasks:
            continue
        key = (item["dataset"], str(item["image_path"]), normalize(item["question"]), normalize(item["reference"]))
        row = exact.get(key)
        if row is None:
            raise RuntimeError(f"unresolved formal probe {item['probe_id']}")
        grouped[(item["task"], item["edit_id"])].append(member(row, item.get("variant_type") or "probe"))
    return [relation("FORMAL_PROBE_CATALOG.jsonl", n, task, members, {"legacy_edit_id": edit_id})
            for n, ((task, edit_id), members) in enumerate(sorted(grouped.items()), 1)]


def metadata_relations(metadata_root: Path, name: str, task: str, by_image: dict, image_fields: tuple[str, ...]) -> list[dict]:
    output = []
    for row_number, source in enumerate(read_csv(metadata_root / name), 1):
        members = []
        image_ids = []
        for field in image_fields:
            value = source.get(field, "")
            if field == "same_disease_images_in_other_modalities":
                # The paired IDs are already represented by the public inventory lineage; keep source row exact here.
                continue
            if value:
                image_ids.append(value)
        if task == "T3":
            import ast
            for paired in ast.literal_eval(source["same_disease_images_in_other_modalities"] or "[]"):
                if isinstance(paired, dict) and paired.get("image_id"):
                    image_ids.append(str(paired["image_id"]))
        for image_id in dict.fromkeys(image_ids):
            for query in by_image.get((dataset_for(image_id), image_id), []):
                members.append(member(query, "candidate_query"))
        output.append(relation(name, row_number, task, members, {"source_relation": source}))
    return output


def t4l_relations(candidates: list[dict], rejections: list[dict], inventory: list[dict]) -> list[dict]:
    by_id = {row["query_id"]: row for row in inventory}
    output = []
    for row in candidates:
        output.append(relation("t4l_compositional_locality.csv", row["source_metadata_row"], "T4L", [
            member(by_id[row["edit_query_id"]], "edit_target_qA"),
            member(by_id[row["probe_query_id"]], "locality_probe_qB"),
        ], {"structural_status": "retained", "candidate_id": row["candidate_id"]}))
    for row in rejections:
        output.append(relation("t4l_compositional_locality.csv", row["source_metadata_row"], "T4L", [], {
            "structural_status": "rejected", "reasons": row["reasons"],
        }))
    return sorted(output, key=lambda row: row["source_row"])


def select_runtime(audit: dict) -> str:
    required = (
        audit.get("checkpoint_identity_verified"), audit.get("native_runtime_stable"),
        audit.get("official_prompt_image_generation"), audit.get("no_runtime_errors"),
        audit.get("normalized_parity", 0) >= 0.995, audit.get("semantic_parity", 0) >= 0.995,
    )
    return "runtime_a_official_parity" if all(required) else "runtime_b_official_native"


def t0_filter(candidates: list[dict], verdict: dict[str, bool]) -> list[dict]:
    return [row for row in candidates if not verdict[row["query_id"]]]


def t3_partition(probe_ids: list[str], verdict: dict[str, bool]) -> tuple[set[str], set[str]]:
    locality = {query_id for query_id in probe_ids if verdict[query_id]}
    generality = set(probe_ids) - locality
    return locality, generality


def t4l_eligible(q_a: bool, q_b: bool) -> bool:
    return not q_a and q_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--amended189", type=Path, required=True)
    parser.add_argument("--formal-catalog", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--t4l-candidates", type=Path, required=True)
    parser.add_argument("--t4l-rejections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("refusing to reuse static catalog output directory")
    source = read_jsonl(args.inventory)
    if len(source) != 11088 or len({row["query_id"] for row in source}) != len(source):
        raise RuntimeError("static query count/uniqueness mismatch")
    if any(not normalize(row["gold_answer"]) for row in source):
        raise RuntimeError("empty gold answer")
    if any(not Path(row["image_path"]).is_file() for row in source):
        raise RuntimeError("unresolved image")
    inventory = [public_query(row) for row in source]
    by_image, exact = lookup(source)
    outputs = {
        "STATIC_QUERY_INVENTORY.jsonl": inventory,
        "STATIC_T0_CANDIDATES.jsonl": t0_candidates(read_jsonl(args.amended189), exact),
        "STATIC_T1_RELATIONS.jsonl": formal_relations(read_jsonl(args.formal_catalog), exact, {"T1L", "T1G"}, "T1"),
        "STATIC_T2L_RELATIONS.jsonl": metadata_relations(args.metadata_root, "t2l_cross_image_pairs.csv", "T2L", by_image, ("image_id_1", "image_id_2")),
        "STATIC_T2G_RELATIONS.jsonl": formal_relations(read_jsonl(args.formal_catalog), exact, {"T2G"}, "T2G"),
        "STATIC_T3_RELATIONS.jsonl": metadata_relations(args.metadata_root, "t3_cross_modality_pairs.csv", "T3", by_image, ("image_A", "same_disease_images_in_other_modalities")),
        "STATIC_T4L_RELATIONS.jsonl": t4l_relations(read_jsonl(args.t4l_candidates), read_jsonl(args.t4l_rejections), source),
        "STATIC_T4G_RELATIONS.jsonl": metadata_relations(args.metadata_root, "t4g_compositional_generality.csv", "T4G", by_image, ("single_image", "multi_image")),
    }
    args.output_dir.mkdir(parents=True)
    for name, rows in outputs.items():
        write_jsonl(args.output_dir / name, rows)
    gate_ids = sorted({member["query_id"] for name, rows in outputs.items() if name != "STATIC_QUERY_INVENTORY.jsonl"
                       for row in rows for member in row.get("members", [row]) if "query_id" in member})
    write_json(args.output_dir / "GATE_CRITICAL_QUERY_IDS.json", {"query_ids": gate_ids, "count": len(gate_ids)})
    manifest = {
        "status": "M3BENCH_STATIC_DATA_RELATIONS_FROZEN",
        "schema_version": "m3bench-static-catalog-v3",
        "public_source_commit": "03c6fda3813301dab3be5831fdc94b493c10afc9",
        "query_count": len(inventory),
        "unique_query_count": len({row["query_id"] for row in inventory}),
        "resolved_image_count": len(inventory),
        "nonempty_gold_count": len(inventory),
        "relation_counts": {name: len(rows) for name, rows in outputs.items() if name != "STATIC_QUERY_INVENTORY.jsonl"},
        "t4l_public_rows": len(outputs["STATIC_T4L_RELATIONS.jsonl"]),
        "t4l_structurally_retained": sum(row["structural_status"] == "retained" for row in outputs["STATIC_T4L_RELATIONS.jsonl"]),
        "t4l_structurally_rejected": sum(row["structural_status"] == "rejected" for row in outputs["STATIC_T4L_RELATIONS.jsonl"]),
        "gate_critical_query_count": len(gate_ids),
        "method_outputs_used": False,
        "positional_fallback_used": False,
    }
    write_json(args.output_dir / "STATIC_CATALOG_MANIFEST.json", manifest)
    names = sorted([*outputs, "GATE_CRITICAL_QUERY_IDS.json", "STATIC_CATALOG_MANIFEST.json"])
    atomic_new(args.output_dir / "SHA256SUMS.txt", "".join(f"{hash_file(args.output_dir / name)}  {name}\n" for name in names))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
