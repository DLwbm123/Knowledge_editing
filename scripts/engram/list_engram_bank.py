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
    from easyeditor.models.engram.bank import EngramBank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="List edits in an ENGRAM bank.")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--export-csv")
    args = parser.parse_args()

    bank = EngramBank(args.bank)
    edits = bank.list_edits()
    print(json.dumps(edits, indent=2, sort_keys=True))
    if args.export_csv:
        print(bank.export_summary_csv(args.export_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
