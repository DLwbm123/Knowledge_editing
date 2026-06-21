from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image

from .paths import MEDMKEB_JSON_NAMES, MedMKEBPaths, read_json, write_json
from .schema import as_records


IMAGE_FIELDS = ("image", "image_rephrase", "m_loc")


@dataclass(frozen=True)
class ImageResolution:
    original: str
    resolved: Optional[Path]
    checked_paths: Tuple[Path, ...]

    @property
    def exists(self) -> bool:
        return self.resolved is not None and self.resolved.exists()


def default_candidate_roots(paths: MedMKEBPaths) -> List[Path]:
    return [
        paths.data,
        paths.images,
        paths.sources,
        paths.medmkeb_repo,
    ]


def iter_image_refs(value: Any) -> Iterator[str]:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from iter_image_refs(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in {"image", "image_path", "path", "file", "filename", "m_loc"}:
                yield from iter_image_refs(item)
            elif isinstance(item, (dict, list, tuple, set)):
                yield from iter_image_refs(item)


def resolve_image(reference: Any, roots: Sequence[Path]) -> ImageResolution:
    values = list(iter_image_refs(reference))
    if not values:
        return ImageResolution(original="", resolved=None, checked_paths=tuple())

    ref = values[0].replace("\\", "/")
    ref = ref[2:] if ref.startswith("./") else ref
    path = Path(ref).expanduser()
    checked: List[Path] = [path] if path.is_absolute() else []
    if not path.is_absolute():
        for root in roots:
            checked.append(root / ref)
            if ref.startswith("images/"):
                checked.append(root / "data" / ref)

    for candidate in checked:
        if candidate.exists() and candidate.is_file():
            return ImageResolution(original=ref, resolved=candidate.resolve(), checked_paths=tuple(checked))
    return ImageResolution(original=ref, resolved=None, checked_paths=tuple(checked))


def audit_assets(paths: MedMKEBPaths, json_files: Optional[Sequence[Path]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    roots = default_candidate_roots(paths)
    if json_files:
        files = list(json_files)
    else:
        files = [paths.raw / name for name in MEDMKEB_JSON_NAMES if (paths.raw / name).exists()]
        files.extend(p for p in sorted(paths.raw.glob("*.json")) if p not in files and not p.name.startswith("._"))

    total_records = 0
    total_refs = 0
    resolved_refs = 0
    missing_rows: List[Dict[str, Any]] = []
    missing_grouped: Counter[Tuple[str, str]] = Counter()
    resolved_grouped: Counter[Tuple[str, str]] = Counter()

    for file_path in files:
        records = as_records(read_json(file_path))
        total_records += len(records)
        for idx, record in enumerate(records):
            for field in IMAGE_FIELDS:
                for ref in iter_image_refs(record.get(field)):
                    total_refs += 1
                    resolution = resolve_image(ref, roots)
                    key = (file_path.name, field)
                    if resolution.exists:
                        resolved_refs += 1
                        resolved_grouped[key] += 1
                    else:
                        missing_grouped[key] += 1
                        missing_rows.append(
                            {
                                "json_file": file_path.name,
                                "record_index": idx,
                                "record_id": record.get("id"),
                                "field": field,
                                "path": ref,
                            }
                        )

    def grouped(counter: Counter[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
        result: Dict[str, Dict[str, int]] = defaultdict(dict)
        for (json_file, field), count in counter.items():
            result[json_file][field] = count
        return dict(result)

    return {
        "total_records": total_records,
        "total_image_references": total_refs,
        "resolved_image_references": resolved_refs,
        "missing_image_references": total_refs - resolved_refs,
        "missing_count_grouped_by_json_file_and_field": grouped(missing_grouped),
        "resolved_count_grouped_by_json_file_and_field": grouped(resolved_grouped),
        "first_50_missing_paths": missing_rows[:50],
        "candidate_roots_checked": [str(root) for root in roots],
        "json_files": [str(path) for path in files],
    }, missing_rows


def write_missing_tsv(path: Path, missing_rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["json_file", "record_index", "record_id", "field", "path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in missing_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def create_smoke_dataset(paths: MedMKEBPaths, source_file: Optional[Path] = None, count: int = 2) -> Dict[str, Any]:
    source = source_file or next((paths.raw / name for name in MEDMKEB_JSON_NAMES if (paths.raw / name).exists()), None)
    if source is None or not source.exists():
        raise FileNotFoundError(f"No MedMKEB JSON file is available under {paths.raw}")

    smoke_dir = paths.processed / "smoke"
    image_dir = smoke_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    smoke_records: List[Dict[str, Any]] = []
    for idx, record in enumerate(as_records(read_json(source))[:count]):
        copied = dict(record)
        for field in IMAGE_FIELDS:
            image_path = image_dir / f"smoke_{idx}_{field}.png"
            if not image_path.exists():
                Image.new("RGB", (224, 224), color=(245, 245, 245)).save(image_path)
            copied[field] = str(image_path)
        copied.setdefault("id", f"smoke-{idx}")
        copied["smoke_synthetic_image"] = True
        smoke_records.append(copied)

    smoke_json = smoke_dir / "smoke_data.json"
    write_json(smoke_json, smoke_records)
    report = {
        "path": str(smoke_json),
        "records": len(smoke_records),
        "image_dir": str(image_dir),
        "source_file": str(source),
        "valid_benchmark": False,
        "note": "Synthetic blank images are only for code-path tests, not benchmark results.",
    }
    write_json(smoke_dir / "smoke_report.json", report)
    return report


def ensure_placeholder_image(paths: MedMKEBPaths, name: str = "blank_rgb_224.png") -> Path:
    placeholder_dir = paths.processed / "placeholders"
    placeholder_dir.mkdir(parents=True, exist_ok=True)
    image_path = placeholder_dir / name
    if not image_path.exists():
        Image.new("RGB", (224, 224), color=(245, 245, 245)).save(image_path)
    return image_path
