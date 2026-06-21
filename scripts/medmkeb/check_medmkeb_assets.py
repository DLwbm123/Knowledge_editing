#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medmkeb_editing.asset_resolver import (  # noqa: E402
    audit_assets,
    create_smoke_dataset,
    write_missing_tsv,
)
from medmkeb_editing.paths import ensure_layout, get_paths, raw_json_files, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify MedMKEB image references and create a synthetic smoke dataset.")
    parser.add_argument("--root", default="/Volumes/DataP/knowledge_editing")
    parser.add_argument("--json-file", action="append", help="Optional raw JSON file to check. Can be repeated.")
    parser.add_argument("--no-smoke", action="store_true", help="Do not create the synthetic smoke dataset.")
    args = parser.parse_args()

    paths = get_paths(args.root)
    ensure_layout(paths)
    files = [Path(p) for p in args.json_file] if args.json_file else raw_json_files(paths)
    if not files:
        raise SystemExit(f"No MedMKEB JSON files found in {paths.raw}.")

    report, missing_rows = audit_assets(paths, files)
    write_json(paths.reports / "asset_report.json", report)
    write_missing_tsv(paths.reports / "missing_images.tsv", missing_rows)

    smoke_report = None
    if not args.no_smoke:
        smoke_report = create_smoke_dataset(paths, source_file=files[0])
        write_json(paths.reports / "smoke_dataset_report.json", smoke_report)

    print(f"Checked {report['total_image_references']} image references")
    print(f"Resolved: {report['resolved_image_references']}")
    print(f"Missing: {report['missing_image_references']}")
    print(f"Wrote {paths.reports / 'asset_report.json'}")
    print(f"Wrote {paths.reports / 'missing_images.tsv'}")
    if smoke_report:
        print(f"Created synthetic smoke dataset: {smoke_report['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
