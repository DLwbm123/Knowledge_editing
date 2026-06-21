from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .paths import read_json


SUMMARY_FIELDS = ("modality", "department", "clinical_VQA_task", "perceptual_granularity")


def as_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "examples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
        return [payload]
    raise ValueError(f"Unsupported JSON payload type: {type(payload).__name__}")


def load_records(path: Path) -> List[Dict[str, Any]]:
    return as_records(read_json(path))


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.strip().split())
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " | ".join(normalize_answer(v) for v in value if normalize_answer(v))
    if isinstance(value, dict):
        return " ".join(f"{k}: {normalize_answer(v)}" for k, v in value.items())
    return str(value).strip()


def summarize_json_files(files: Sequence[Path]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"files": {}, "total_records": 0}
    global_keys: Counter[str] = Counter()
    all_keys: set[str] = set()

    for path in files:
        records = load_records(path)
        report["total_records"] += len(records)
        key_counts: Counter[str] = Counter()
        value_counts = {field: Counter() for field in SUMMARY_FIELDS}
        image_examples: List[Dict[str, Any]] = []
        port_examples: List[Any] = []

        for record in records:
            keys = set(record.keys())
            all_keys.update(keys)
            key_counts.update(keys)
            global_keys.update(keys)
            for field in SUMMARY_FIELDS:
                if field in record:
                    value_counts[field][normalize_answer(record.get(field))] += 1
            if len(image_examples) < 10:
                example = {
                    field: record.get(field)
                    for field in ("image", "image_rephrase", "m_loc")
                    if field in record
                }
                if example:
                    image_examples.append({"id": record.get("id"), **example})
            if "port_new" in record and len(port_examples) < 5:
                port_examples.append(record["port_new"])

        missing_counts = {key: len(records) - key_counts.get(key, 0) for key in sorted(all_keys)}
        report["files"][path.name] = {
            "path": str(path),
            "records": len(records),
            "keys": sorted(key_counts.keys()),
            "key_counts": dict(sorted(key_counts.items())),
            "missing_key_counts": missing_counts,
            "top_values": {
                field: value_counts[field].most_common(20)
                for field in SUMMARY_FIELDS
                if value_counts[field]
            },
            "image_path_examples": image_examples,
            "port_new_examples": port_examples,
        }

    for file_report in report["files"].values():
        records = int(file_report["records"])
        key_counts = file_report["key_counts"]
        file_report["missing_key_counts"] = {
            key: records - key_counts.get(key, 0) for key in sorted(all_keys)
        }

    report["all_keys"] = sorted(all_keys)
    report["global_key_counts"] = dict(sorted(global_keys.items()))
    return report


def extract_portability_qa(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = record.get("port_new") or []
    if isinstance(entries, dict):
        entries = [entries]
    output: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        qa = entry.get("Q&A") or entry.get("qa") or {}
        if isinstance(qa, dict):
            output.append(
                {
                    "port_type": entry.get("port_type"),
                    "question": qa.get("Question") or qa.get("question"),
                    "answer": qa.get("Answer") or qa.get("answer"),
                    "raw": entry,
                }
            )
    return output


def missing_required_fields(records: Iterable[Dict[str, Any]], required: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        for field in required:
            if field not in record or record.get(field) in (None, ""):
                counts[field] += 1
    return dict(counts)
