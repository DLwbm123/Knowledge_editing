#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected YAML mapping in {path}.")
    return data


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _resolve_one(
    *,
    source: Path,
    output: Path,
    model_path: str,
    vision_tower_path: str,
    image_root: str,
    bank_dir: str | None,
    alpha: float | None,
) -> Dict[str, Any]:
    data = _load_yaml(source)
    data["name"] = model_path
    data["tokenizer_name"] = model_path
    data["llava_med_vision_tower"] = vision_tower_path
    data["coco_image"] = image_root
    data["rephrase_image"] = image_root
    if bank_dir is not None:
        data["bank_dir"] = bank_dir
        data["engram_bank_path"] = bank_dir
    if alpha is not None:
        data["alpha"] = alpha
    _write_yaml(output, data)
    return {
        "source": str(source),
        "output": str(output),
        "alpha": data.get("alpha"),
        "bank_dir": data.get("bank_dir"),
        "model_path": data.get("name"),
        "vision_tower_path": data.get("llava_med_vision_tower"),
        "image_root": data.get("coco_image"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve LLaVA-Med ENGRAM hparams placeholders for a runtime host.")
    parser.add_argument("--source", action="append", required=True, help="Template YAML path. May repeat.")
    parser.add_argument("--output", action="append", required=True, help="Resolved YAML path. Must match --source count.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--bank-dir", action="append", help="Optional bank dir per output.")
    parser.add_argument("--alpha", action="append", type=float, help="Optional alpha per output.")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    sources = [Path(value) for value in args.source]
    outputs = [Path(value) for value in args.output]
    if len(sources) != len(outputs):
        raise RuntimeError("--source and --output must have the same count.")
    if args.bank_dir is not None and len(args.bank_dir) != len(outputs):
        raise RuntimeError("--bank-dir count must match --output count when provided.")
    if args.alpha is not None and len(args.alpha) != len(outputs):
        raise RuntimeError("--alpha count must match --output count when provided.")

    resolved = []
    for idx, (source, output) in enumerate(zip(sources, outputs)):
        bank_dir = args.bank_dir[idx] if args.bank_dir is not None else None
        alpha = args.alpha[idx] if args.alpha is not None else None
        resolved.append(
            _resolve_one(
                source=source,
                output=output,
                model_path=args.model_path,
                vision_tower_path=args.vision_tower_path,
                image_root=args.image_root,
                bank_dir=bank_dir,
                alpha=alpha,
            )
        )

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_path": args.model_path,
        "vision_tower_path": args.vision_tower_path,
        "image_root": args.image_root,
        "resolved_hparams": resolved,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
