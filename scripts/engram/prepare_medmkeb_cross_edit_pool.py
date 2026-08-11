#!/usr/bin/env python3
"""Stage only the ranked MedMKEB records/images needed by the cross-edit router."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.engram.modality_aware_router_utils import normalize_question, sha256_file, source_sort_key

EXCLUDED_IDS = {"953", "1293", "1592", "2174", "1628", "942", "1382", "1333", "671", "1343"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--candidate-count", type=int, default=128)
    args = parser.parse_args()
    if args.stage_dir.exists() and any(args.stage_dir.iterdir()):
        raise FileExistsError(f"stage directory is not empty: {args.stage_dir}")
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    image_out = args.stage_dir / "images"
    image_out.mkdir()
    raw = json.loads(args.source_json.read_text())
    native_groups: dict[tuple[str, str], list[dict]] = {}
    audited = []
    for index, record in enumerate(raw):
        rid = str(record.get("id", index))
        native = args.image_root / str(record.get("image", ""))
        alternate = args.image_root / str(record.get("image_rephrase", ""))
        questions = {"native": record.get("src"), "textual": record.get("rephrase")}
        valid = all((questions["native"], questions["textual"])) and native.is_file() and alternate.is_file()
        if not valid or rid in EXCLUDED_IDS:
            continue
        native_sha = sha256_file(native)
        alt_sha = sha256_file(alternate)
        row = {**record, "record_id": rid, "source_row_index": index, "native_image_sha256": native_sha, "alternate_image_sha256": alt_sha}
        row["selection_hash"] = source_sort_key(rid, native_sha, str(record["src"]))
        row["source_native_image"] = str(record["image"])
        row["source_alternate_image"] = str(record["image_rephrase"])
        native_groups.setdefault((native_sha, normalize_question(str(record["src"]))), []).append(row)
        audited.append(row)
    conflicts = set()
    for rows in native_groups.values():
        if len({normalize_question(str(row.get("alt", ""))) for row in rows}) > 1:
            conflicts.update(str(row["record_id"]) for row in rows)
    eligible = sorted((row for row in audited if row["record_id"] not in conflicts), key=lambda row: row["selection_hash"])
    selected = eligible[: args.candidate_count]
    staged = []
    for rank, row in enumerate(selected):
        item = dict(row)
        rid = item["record_id"]
        native_suffix = Path(item["source_native_image"]).suffix.lower() or ".img"
        alt_suffix = Path(item["source_alternate_image"]).suffix.lower() or ".img"
        native_name = f"{rank:03d}_{rid}_native{native_suffix}"
        alt_name = f"{rank:03d}_{rid}_alternate{alt_suffix}"
        shutil.copy2(args.image_root / item["source_native_image"], image_out / native_name)
        shutil.copy2(args.image_root / item["source_alternate_image"], image_out / alt_name)
        item["image"] = f"images/{native_name}"
        item["image_rephrase"] = f"images/{alt_name}"
        staged.append(item)
    payload = {
        "source_json": str(args.source_json),
        "source_record_count": len(raw),
        "raw_eligible_count": len(audited),
        "native_conflict_record_ids": sorted(conflicts),
        "eligible_count_after_raw_collision_filter": len(eligible),
        "staged_candidate_count": len(staged),
        "records": staged,
    }
    (args.stage_dir / "source_pool.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
