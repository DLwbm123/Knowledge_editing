#!/usr/bin/env python3
"""Run Stage A or emit the protocol-defined assets-missing outcome."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.posthoc_validation import file_sha256


STATUS_MISSING = "END_TO_END_UPSTREAM_PORT_PARITY_NOT_RUN_ASSETS_MISSING"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--official-model", type=Path)
    parser.add_argument("--official-sample", type=Path)
    parser.add_argument("--upstream-root", type=Path, default=Path("third_party/liveedit_official_3615a37"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    expected = {
        "official_backbone": "llava-hf/llava-1.5-7b-hf",
        "valid_sample_families": ["E-VQA", "E-IC", "VLKEB"],
        "pinned_upstream_commit": "3615a37b05294509f411df045621940f276a5e6b",
    }
    missing = []
    if args.official_model is None or not args.official_model.exists():
        missing.append("official_llava_v1_5_7b_hf_model")
    if args.official_sample is None or not args.official_sample.is_file():
        missing.append("one_source_valid_E-VQA_or_E-IC_or_VLKEB_sample")
    if missing:
        result = {
            "stage": "A", "status": STATUS_MISSING, "missing_assets": missing, "expected": expected,
            "continued_to_medical_validation": True, "downloads_attempted": False,
            "edited_checkpoint_loaded": False, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    else:
        # The full comparison is deliberately guarded: accepting a merely
        # similar backbone/sample would invalidate the claimed source parity.
        result = {
            "stage": "A", "status": "END_TO_END_UPSTREAM_PORT_PARITY_ASSETS_PRESENT_REQUIRES_EXPLICIT_SOURCE_ADAPTER",
            "expected": expected, "official_model": str(args.official_model.resolve()),
            "official_sample": str(args.official_sample.resolve()),
            "official_sample_sha256": file_sha256(args.official_sample), "edited_checkpoint_loaded": False,
        }
    (args.out_dir / "trace_parity_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = f"# Stage A: upstream end-to-end trace parity\n\nStatus: `{result['status']}`\n\n"
    if missing:
        report += "Missing assets:\n\n" + "".join(f"- `{item}`\n" for item in missing)
        report += "\nNo download or substitute-backbone comparison was attempted. Medical-domain validation may continue under the supplied protocol.\n"
    (args.out_dir / "END_TO_END_TRACE_PARITY.md").write_text(report)


if __name__ == "__main__":
    main()
