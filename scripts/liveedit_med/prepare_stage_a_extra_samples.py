#!/usr/bin/env python3
"""Freeze two non-gating VLKEB robustness samples after the Stage-A gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.eval_json.read_text())
    args.out_root.mkdir(parents=True, exist_ok=False)
    summary = []
    for index in (1, 2):
        row = rows[index]
        destination = args.out_root / f"sample_{index}"
        destination.mkdir()
        sample_path = destination / "sample.json"
        write_new(sample_path, [row])
        shutil.copyfile(args.license, destination / "LICENSE.txt")
        images = []
        for field, role in (("image", "native"), ("image_rephrase", "image_rephrase"), ("m_loc", "image_locality")):
            path = args.image_root / row[field]
            if not path.is_file():
                raise FileNotFoundError(path)
            images.append({"record_field": field, "role": role, "relative_path": row[field],
                           "path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)})
        manifest = {"status": "COMPLETE", "dataset_family": "VLKEB", "repo_id": "HymanH/VLKEB-data",
                    "revision": "cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a", "source_file": "eval.json",
                    "source_file_sha256": sha256(args.eval_json), "source_record_index": index,
                    "sample_path": str(sample_path.resolve()), "sample_record_count": 1,
                    "sample_sha256": sha256(sample_path), "image_root": str(args.image_root.resolve()),
                    "images": images, "license_path": str((destination / "LICENSE.txt").resolve()),
                    "license_sha256": sha256(destination / "LICENSE.txt")}
        write_new(destination / "manifest.json", manifest)
        summary.append({"index": index, "sample": str(sample_path), "sample_sha256": manifest["sample_sha256"]})
    write_new(args.out_root / "extra_samples_manifest.json", {"samples": summary, "non_gating": True})
    print(json.dumps({"status": "STAGE_A_EXTRA_SAMPLES_FROZEN", "samples": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
