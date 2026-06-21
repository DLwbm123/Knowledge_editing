#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

with redirect_stdout(sys.stderr):
    from easyeditor.models.engram.overlap import compute_bank_overlap, write_overlap_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute pairwise ENGRAM overlap for a bank.")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--heatmap", action="store_true")
    args = parser.parse_args()

    report = compute_bank_overlap(args.bank, threshold=args.threshold)
    paths = write_overlap_report(report, args.out, heatmap=args.heatmap)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
