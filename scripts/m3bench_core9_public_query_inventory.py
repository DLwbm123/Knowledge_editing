#!/usr/bin/env python3
"""Build the method-blind public-release-aligned Core-9 base query inventory."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.m3bench_core9_task_specific_t4l import canonical_hash, normalize, sha256


FORBIDDEN_SELECTION_KEYS = ("method", "editor", "post_edit", "model_answer")


def parse_list(value: object) -> list:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            pass
    return []


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def has_cjk(value: object) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(value)))


def key(row: dict) -> tuple[str, ...]:
    return (
        row["dataset"], row["image_id"], row["image_sha256"],
        normalize(row["question"]), normalize(row["gold_answer"]),
    )


def stable_query_id(query_key: tuple[str, ...]) -> str:
    return "core9q-" + canonical_hash(query_key)[:24]


def assert_no_method_fields(rows: list[dict]) -> None:
    for row in rows:
        for name in row:
            lowered = name.casefold()
            if any(token in lowered for token in FORBIDDEN_SELECTION_KEYS):
                raise ValueError(f"method/output field is forbidden in selection inventory: {name}")


def image_identity(dataset: str, image_id: str, source: dict[tuple[str, str], dict]) -> dict | None:
    return source.get((dataset, image_id))


def add(rows: dict[tuple[str, ...], dict], item: dict, lineage: dict) -> None:
    query_key = key(item)
    if query_key not in rows:
        rows[query_key] = {
            **item,
            "query_id": stable_query_id(query_key),
            "normalized_question": normalize(item["question"]),
            "normalized_gold": normalize(item["gold_answer"]),
            "lineage": [],
            "legacy_source_record_ids": [],
            "legacy_derived_probe_ids": [],
        }
    target = rows[query_key]
    if lineage not in target["lineage"]:
        target["lineage"].append(lineage)
    source_id = lineage.get("legacy_source_record_id")
    derived_id = lineage.get("legacy_derived_probe_id")
    if source_id and source_id not in target["legacy_source_record_ids"]:
        target["legacy_source_record_ids"].append(source_id)
    if derived_id and derived_id not in target["legacy_derived_probe_ids"]:
        target["legacy_derived_probe_ids"].append(derived_id)


def build_inventory(
    evaluation: list[dict], metadata_root: Path, t4l_queries: list[dict], formal_catalog: list[dict]
) -> tuple[list[dict], list[dict]]:
    rows: dict[tuple[str, ...], dict] = {}
    rejected: list[dict] = []
    source_images: dict[tuple[str, str], dict] = {}

    for source in evaluation:
        dataset = source["dataset"]
        if not source.get("eligible_for_inference") or not source.get("eligible_for_evaluation"):
            rejected.append({"source": "evaluation_manifest", "record_id": source["record_id"], "reason": "source_ineligible"})
            continue
        if dataset == "SLAKE" and has_cjk(source["question_raw"]):
            rejected.append({"source": "evaluation_manifest", "record_id": source["record_id"], "reason": "slake_non_english"})
            continue
        image_path = Path(source["relative_image_path"])
        source_images[(dataset, source["image_id"])] = {
            "dataset": dataset,
            "image_id": source["image_id"],
            "relative_image_path": source["relative_image_path"],
            "image_sha256": source["image_sha256"],
        }
        item = {
            **source_images[(dataset, source["image_id"])],
            "image_path": None,
            "question": source["question_raw"],
            "gold_answer": source["gold_answer_raw_or_null"] or "",
            "source_task": "PUBLIC_SOURCE_QA",
            "source_metadata_file": "slake_metadata.csv" if dataset == "SLAKE" else "vqarad_metadata.csv",
            "source_metadata_row": source["metadata_row_index"],
            "relation_id": source["record_id"],
            "role": "source_qa",
        }
        if not normalize(item["gold_answer"]):
            rejected.append({"source": "evaluation_manifest", "record_id": source["record_id"], "reason": "missing_gold"})
            continue
        add(rows, item, {
            "source_task": "PUBLIC_SOURCE_QA",
            "source_metadata_file": item["source_metadata_file"],
            "source_metadata_row": item["source_metadata_row"],
            "relation_id": item["relation_id"],
            "role": item["role"],
            "legacy_source_record_id": source["record_id"],
            "global_index": source["global_index"],
        })

    # Synthetic T2L questions are declared for both images in each public pair.
    for number, relation in enumerate(read_csv(metadata_root / "t2l_cross_image_pairs.csv"), 1):
        relation_id = "t2l-public-" + canonical_hash(relation)[:20]
        for image_id in (relation["image_id_1"], relation["image_id_2"]):
            identity = next((v for (d, i), v in source_images.items() if i == image_id), None)
            if identity is None:
                rejected.append({"source": "t2l_cross_image_pairs.csv", "source_metadata_row": number, "image_id": image_id, "reason": "unresolved_image"})
                continue
            for qa in parse_list(relation.get("qa_list")):
                if not isinstance(qa, dict) or not normalize(qa.get("question")) or not normalize(qa.get("answer")):
                    rejected.append({"source": "t2l_cross_image_pairs.csv", "source_metadata_row": number, "image_id": image_id, "reason": "malformed_qa"})
                    continue
                item = {
                    **identity,
                    "image_path": None,
                    "question": qa["question"],
                    "gold_answer": qa["answer"],
                    "source_task": "T2L",
                    "source_metadata_file": "t2l_cross_image_pairs.csv",
                    "source_metadata_row": number,
                    "relation_id": relation_id,
                    "role": "same_disease_locality_probe",
                }
                add(rows, item, {k: item[k] for k in ("source_task", "source_metadata_file", "source_metadata_row", "relation_id", "role")})

    for query in t4l_queries:
        item = {**query, "image_path": None}
        for lineage in query["lineage"]:
            add(rows, item, lineage)

    # Frozen T0/T1/T2G queries add lineage and the derived image/query cases absent from source metadata.
    source_by_record = {
        lineage["legacy_source_record_id"]: row
        for row in rows.values() for lineage in row["lineage"] if lineage.get("legacy_source_record_id")
    }
    for catalog in formal_catalog:
        source = source_by_record.get(catalog["probe_id"])
        image_path = Path(catalog["image_path"])
        if source:
            image_id, image_hash, relative = source["image_id"], source["image_sha256"], source["relative_image_path"]
        else:
            if not image_path.is_file():
                rejected.append({"source": "FORMAL_PROBE_CATALOG", "probe_id": catalog["probe_id"], "reason": "missing_image"})
                continue
            image_id = f"derived:{catalog['probe_id']}"
            image_hash = sha256(image_path)
            relative = str(image_path)
        item = {
            "dataset": catalog["dataset"],
            "image_id": image_id,
            "relative_image_path": relative,
            "image_path": str(image_path),
            "image_sha256": image_hash,
            "question": catalog["question"],
            "gold_answer": catalog["reference"],
            "source_task": catalog["task"],
            "source_metadata_file": "FORMAL_PROBE_CATALOG.jsonl",
            "source_metadata_row": catalog["probe_index"],
            "relation_id": catalog["edit_id"],
            "role": "frozen_formal_probe",
        }
        lineage = {k: item[k] for k in ("source_task", "source_metadata_file", "source_metadata_row", "relation_id", "role")}
        if "::T1G::" in catalog["probe_id"] or "::T2G::" in catalog["probe_id"]:
            lineage["legacy_derived_probe_id"] = catalog["probe_id"]
        elif catalog["probe_id"] in source_by_record:
            lineage["legacy_source_record_id"] = catalog["probe_id"]
        add(rows, item, lineage)

    inventory = sorted(rows.values(), key=lambda row: row["query_id"])
    assert_no_method_fields(inventory)
    if len({row["query_id"] for row in inventory}) != len(inventory):
        raise RuntimeError("duplicate stable query IDs")
    return inventory, rejected


def resolve_paths(inventory: list[dict], slake_root: Path, vqarad_root: Path) -> None:
    for row in inventory:
        if row["image_path"]:
            path = Path(row["image_path"])
        elif row["dataset"] == "SLAKE":
            path = slake_root / row["image_id"] / "source.jpg"
        else:
            path = vqarad_root / row["image_id"]
        if not path.is_file():
            raise RuntimeError(f"unresolved image for {row['query_id']}")
        row["image_path"] = str(path)


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--t4l-queries", type=Path, required=True)
    parser.add_argument("--formal-catalog", type=Path, required=True)
    parser.add_argument("--slake-image-root", type=Path, required=True)
    parser.add_argument("--vqarad-image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("refusing to reuse inventory output directory")

    inventory, rejected = build_inventory(
        read_jsonl(args.evaluation_manifest), args.metadata_root,
        read_jsonl(args.t4l_queries), read_jsonl(args.formal_catalog),
    )
    resolve_paths(inventory, args.slake_image_root, args.vqarad_image_root)
    args.output_dir.mkdir(parents=True)
    paths = {
        "inventory": args.output_dir / "BASE_QUERY_INVENTORY_CORE9.jsonl",
        "lineage": args.output_dir / "BASE_QUERY_LINEAGE_CORE9.jsonl",
        "rejections": args.output_dir / "BASE_QUERY_REJECTIONS_CORE9.jsonl",
    }
    write_new(paths["inventory"], "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in inventory))
    write_new(paths["lineage"], "".join(json.dumps({"query_id": row["query_id"], "lineage": row["lineage"]}, ensure_ascii=False, sort_keys=True) + "\n" for row in inventory))
    write_new(paths["rejections"], "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rejected))
    manifest = {
        "schema_version": "m3bench-core9-base-query-inventory-v1",
        "query_count": len(inventory),
        "unique_image_count": len({(row["dataset"], row["image_id"], row["image_sha256"]) for row in inventory}),
        "source_reuse_candidate_count": sum(bool(row["legacy_source_record_ids"]) for row in inventory),
        "derived_reuse_candidate_count": sum(bool(row["legacy_derived_probe_ids"]) for row in inventory),
        "new_inference_candidate_count": sum(not row["legacy_source_record_ids"] and not row["legacy_derived_probe_ids"] for row in inventory),
        "lineage_count": sum(len(row["lineage"]) for row in inventory),
        "rejection_count": len(rejected),
        "rejection_reason_counts": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        "method_outputs_used": False,
        "files": {name: {"sha256": sha256(path), "rows": sum(1 for line in path.open() if line.strip())} for name, path in paths.items()},
    }
    write_new(args.output_dir / "BASE_QUERY_INVENTORY_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
