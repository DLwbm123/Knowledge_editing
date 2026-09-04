#!/usr/bin/env python3
"""Freeze official checkpoint identity and a score-independent runtime canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


MODEL_REPO = "microsoft/llava-med-v1.5-mistral-7b"
MODEL_SHA = "91bb16c122001ddc9cf1fd36ce1dae09448943a2"
VISION_REPO = "openai/clip-vit-large-patch14-336"
VISION_SHA = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
LLAVA_MED_COMMIT = "30697ca50b5c29a8e955c99330b259776aef27b9"

MODEL_FILES = {
    "config.json": (1407, "f6ae889c5488ef86895e78f641339062962dd6b434666019fa119ab09d2bd8b3"),
    "generation_config.json": (111, "741acba7f5e235dac0e6865ecc212bbadb1ab1d6d853de7d759268cb62aaf2b4"),
    "model.safetensors.index.json": (73152, "d5ecec60dba218c6621cfa524d739de942b4552bcbb25efe63848c18a731b2f6"),
    "model-00001-of-00004.safetensors": (4943162336, "ef2190dc6c2a940e60f03f5fdb4dddb2320eb87801aeca5c40b0a28ce8aa420e"),
    "model-00002-of-00004.safetensors": (4999819336, "2b229607fecd98b8111320178e5bf3e2c527b05a942c85d65b5b507c76c1ed00"),
    "model-00003-of-00004.safetensors": (4927408360, "12b18ecdf8924d5fe28ada797fe6697fa60e62cba630759fbeb52975b261c4e2"),
    "model-00004-of-00004.safetensors": (262144128, "1d2063fcd429d3f0f0a8a091b0522f0e02f2d85fe0e5b0eeb4ae168183a603bc"),
    "tokenizer.model": (493443, "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055"),
    "tokenizer_config.json": (1463, "5b219f9212f7263269898c799cc9d9be2326e853bf1e497f1c412f3a274d0597"),
    "special_tokens_map.json": (438, "719833ff26ac897a3ec8ed946028a135de2a351470af59b4008744ab1f0ee9b7"),
}
VISION_FILES = {
    "config.json": (4757, "51b1c14aabcdf639c4a0370eeda1010b773bbe1df78319c7d0f5882c22ac0ac0"),
    "preprocessor_config.json": (316, "d253881f65322dc546df59cf925a408e5538b8ecb5a1b496cdd36af9992686d4"),
    "pytorch_model.bin": (1711974081, "c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    atomic_new(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def inspect(root: Path, expected: dict[str, tuple[int, str]]) -> list[dict]:
    records = []
    for name, (size, digest) in expected.items():
        path = root / name
        actual_size = path.stat().st_size if path.is_file() else None
        actual_digest = sha256(path) if actual_size == size else None
        records.append({
            "name": name, "size": actual_size, "sha256": actual_digest,
            "expected_size": size, "expected_sha256": digest,
            "match": actual_size == size and actual_digest == digest,
        })
    return records


def checkpoint(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise RuntimeError("refusing to reuse checkpoint audit directory")
    args.output_dir.mkdir(parents=True)
    model = inspect(args.model_path, MODEL_FILES)
    vision = inspect(args.vision_path, VISION_FILES)
    tokenizer_names = {"tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"}
    write_json(args.output_dir / "LOCAL_MODEL_FILE_MANIFEST.json", {"files": [row for row in model if row["name"] not in tokenizer_names]})
    write_json(args.output_dir / "LOCAL_TOKENIZER_MANIFEST.json", {"files": [row for row in model if row["name"] in tokenizer_names]})
    write_json(args.output_dir / "LOCAL_VISION_TOWER_MANIFEST.json", {"files": vision})
    all_rows = [("model/" + row["name"], row) for row in model] + [("vision/" + row["name"], row) for row in vision]
    atomic_new(args.output_dir / "LOCAL_MODEL_SHA256SUMS.txt", "".join(
        f"{row['sha256'] or 'MISSING'}  {name}\n" for name, row in all_rows
    ))
    config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    identity = all(row["match"] for _, row in all_rows) and config.get("architectures") == ["LlavaMistralForCausalLM"] and config.get("mm_vision_tower") == VISION_REPO
    lock = {
        "model_repo": MODEL_REPO, "model_snapshot_sha": MODEL_SHA,
        "vision_repo": VISION_REPO, "vision_snapshot_sha": VISION_SHA,
        "llava_med_code_commit": LLAVA_MED_COMMIT,
        "local_checkpoint_matches_official_snapshot": identity,
        "architecture": config.get("architectures"),
        "conversation_template": "mistral_instruct",
        "resolved_via": "Hugging Face immutable API metadata and official LFS/file hashes",
    }
    write_json(args.output_dir / "OFFICIAL_SNAPSHOT_LOCK.json", lock)
    atomic_new(args.output_dir / "CHECKPOINT_IDENTITY_REPORT.md", "\n".join([
        "# Checkpoint identity report", "",
        f"- Official model snapshot: `{MODEL_SHA}`", f"- Official vision snapshot: `{VISION_SHA}`",
        f"- Official LLaVA-Med code commit: `{LLAVA_MED_COMMIT}`",
        f"- Architecture: `{config.get('architectures')}`", f"- Vision tower: `{config.get('mm_vision_tower')}`",
        f"- All required local files match official size and SHA-256: **{'PASS' if identity else 'FAIL'}**", "",
    ]))
    print(json.dumps(lock, sort_keys=True))


def query_ids(relations: list[dict]) -> set[str]:
    return {member["query_id"] for row in relations for member in row.get("members", [])}


def choose(rows: list[dict], count: int) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(row["query_id"].encode()).hexdigest())[:count]


def canary(args: argparse.Namespace) -> None:
    inventory = read_jsonl(args.static_dir / "STATIC_QUERY_INVENTORY.jsonl")
    source = [row for row in inventory if any(item.get("source_task") == "PUBLIC_SOURCE_QA" for item in row["lineage"])]
    strata = [
        ("SLAKE_ORIGINAL", choose([row for row in source if row["dataset"] == "SLAKE"], 128)),
        ("VQARAD_ORIGINAL", choose([row for row in source if row["dataset"] == "VQA-RAD"], 128)),
    ]
    t23_ids = query_ids(read_jsonl(args.static_dir / "STATIC_T2L_RELATIONS.jsonl")) | query_ids(read_jsonl(args.static_dir / "STATIC_T3_RELATIONS.jsonl"))
    t4_ids = query_ids(read_jsonl(args.static_dir / "STATIC_T4L_RELATIONS.jsonl"))
    by_id = {row["query_id"]: row for row in inventory}
    strata.extend([
        ("T2L_T3_GATE", choose([by_id[value] for value in t23_ids], 128)),
        ("T4L_QA_QB", choose([by_id[value] for value in t4_ids], 128)),
    ])
    used, records = set(), []
    for stratum, rows in strata:
        for row in rows:
            if row["query_id"] in used:
                continue
            used.add(row["query_id"])
            records.append({"stratum": stratum, **row})
    write_json(args.output, {
        "status": "FROZEN_BEFORE_RUNTIME_EXECUTION", "selection": "ascending_sha256(query_id)",
        "count": len(records), "stratum_counts": {name: sum(row["stratum"] == name for row in records) for name, _ in strata},
        "records": records,
    })
    print(json.dumps({"count": len(records), "strata": {name: sum(row["stratum"] == name for row in records) for name, _ in strata}}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("checkpoint")
    c.add_argument("--model-path", type=Path, required=True); c.add_argument("--vision-path", type=Path, required=True)
    c.add_argument("--output-dir", type=Path, required=True); c.set_defaults(func=checkpoint)
    a = sub.add_parser("canary")
    a.add_argument("--static-dir", type=Path, required=True); a.add_argument("--output", type=Path, required=True); a.set_defaults(func=canary)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
