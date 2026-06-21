#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/Volumes/DataP/knowledge_editing/data/medmkeb")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "medmkeb_engram_projected_lora"
DEFAULT_HPARAMS = PROJECT_ROOT / "hparams" / "ENGRAM" / "llava_med_5edit_cure_tiny_lora.yaml"

METHOD_A = "A_no_edit"
METHOD_B = "B_tiny_lora_replacement"
METHOD_C = "C_engram_projected_tiny_lora"

EXPECTED_MODULES = [
    "llava_model.model.layers.0.mlp.gate_proj",
    "llava_model.model.layers.8.mlp.gate_proj",
    "llava_model.model.layers.16.mlp.gate_proj",
    "llava_model.model.layers.24.mlp.gate_proj",
    "llava_model.model.layers.16.self_attn.q_proj",
    "llava_model.model.layers.24.self_attn.q_proj",
    "llava_model.model.layers.16.self_attn.k_proj",
    "llava_model.model.layers.24.self_attn.k_proj",
]

TEXT_FIELDS = ("prompt", "question", "src", "query")
OLD_FIELDS = ("old_answer", "target_true", "answer", "ground_truth", "true_answer", "pred")
NEW_FIELDS = ("new_answer", "target_new", "alt", "edited_answer")
IMAGE_FIELDS = ("image", "image_path", "image_id", "image_file", "img")
REPHRASE_FIELDS = ("rephrase", "rephrase_prompt")
LOCALITY_PROMPT_FIELDS = ("m_loc_q", "locality_prompt", "locality_q")
LOCALITY_ANSWER_FIELDS = ("m_loc_a", "locality_answer", "locality_a")
LOCALITY_IMAGE_FIELDS = ("m_loc", "locality_image", "locality_image_path")
PRIVATE_KEY_FRAGMENTS = ("patient_name", "patient_id", "mrn", "medical_record", "phone", "email", "address")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _run_capture(command: List[str], cwd: Path = PROJECT_ROOT) -> str:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return proc.stdout
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}\n"


def _ensure_layout(out_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": out_dir,
        "audit": out_dir / "audit",
        "modelknown": out_dir / "modelknown_20",
        "generation": out_dir / "generation_diagnostics",
        "tests": out_dir / "test_logs",
        "nonseq": out_dir / "modelknown_20" / "nonseq",
        "sequential": out_dir / "modelknown_20" / "sequential",
        "plots": out_dir / "modelknown_20" / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_git_outputs(out_dir: Path) -> None:
    (out_dir / "git_status.txt").write_text(_run_capture(["git", "status"], PROJECT_ROOT), encoding="utf-8")
    (out_dir / "git_diff.patch").write_text(_run_capture(["git", "diff"], PROJECT_ROOT), encoding="utf-8")


def _write_env_report(out_dir: Path) -> Dict[str, Any]:
    python_report = _run_capture(
        [
            sys.executable,
            "-c",
            (
                "import sys, platform; print('executable', sys.executable); "
                "print('version', sys.version); print('platform', platform.platform()); "
                "mods=['torch','transformers','peft','PIL']; "
                "\nfor m in mods:\n"
                "    try:\n"
                "        mod=__import__(m); print(m, getattr(mod, '__version__', 'ok'))\n"
                "    except Exception as e: print(m, type(e).__name__ + ': ' + str(e))\n"
                "try:\n"
                "    import torch; print('cuda_available', torch.cuda.is_available()); "
                "print('cuda_device_count', torch.cuda.device_count()); "
                "print('gpu0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\n"
                "except Exception as e: print('torch_cuda_probe', type(e).__name__ + ': ' + str(e))\n"
            ),
        ]
    )
    nvidia = _run_capture(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"])
    payload = {
        "cwd": str(PROJECT_ROOT),
        "python": sys.executable,
        "python_report": python_report,
        "nvidia_smi": nvidia,
    }
    (out_dir / "env_report.txt").write_text(
        "\n".join([f"cwd={PROJECT_ROOT}", "python_report:", python_report, "nvidia-smi:", nvidia]) + "\n",
        encoding="utf-8",
    )
    return payload


def _load_json_like(path: Path) -> List[Dict[str, Any]]:
    if path.name.startswith("._"):
        return []
    if path.suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        return records
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("records", "data", "examples", "annotations"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [value]
    return []


def _candidate_data_files(data_root: Path) -> List[Path]:
    preferred = [
        data_root / "raw" / "eval_data_threehop_final.json",
        data_root / "raw" / "eval_data_attack.json",
        data_root / "raw" / "train_data.json",
        data_root / "sources" / "VLKEB" / "eval.json",
        data_root / "sources" / "VLKEB" / "eval_edit_onehop.json",
        data_root / "processed" / "medmkeb_easyedit_manifest.jsonl",
        data_root / "processed" / "medmkeb_vlkeb_manifest.json",
    ]
    files: List[Path] = []
    for path in preferred:
        if path.exists() and not path.name.startswith("._"):
            files.append(path)
    for root in [data_root / "raw", data_root / "processed", data_root / "sources" / "VLKEB"]:
        if root.exists():
            for path in sorted(root.glob("*")):
                if path.suffix in {".json", ".jsonl"} and not path.name.startswith("._") and path not in files:
                    files.append(path)
    return files


def _first_text(record: Dict[str, Any], fields: Iterable[str]) -> Tuple[str, Optional[str]]:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip(), field
    return "", None


def _private_flags(record: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    for key in record:
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in PRIVATE_KEY_FRAGMENTS):
            flags.append(str(key))
    return flags


def _candidate_roots(data_root: Path) -> List[Path]:
    return [
        data_root,
        data_root / "images",
        data_root / "sources",
        data_root / "repos" / "MedMKEB",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "datasets",
    ]


def _resolve_ref(reference: Any, roots: Sequence[Path]) -> Tuple[Optional[Path], List[str]]:
    if reference is None:
        return None, []
    text = str(reference).strip().replace("\\", "/")
    if not text:
        return None, []
    if text.startswith("./"):
        text = text[2:]
    path = Path(text).expanduser()
    checked: List[Path] = []
    if path.is_absolute():
        checked.append(path)
    else:
        for root in roots:
            checked.append(root / text)
            if text.startswith("images/"):
                checked.append(root / "data" / text)
    for candidate in checked:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve(), [str(item) for item in checked[:12]]
    return None, [str(item) for item in checked[:12]]


def _extract_portability(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    raw = record.get("port_new") or record.get("portability") or record.get("portability_qa")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            qa = item.get("Q&A") or item.get("qa") or item
            question = qa.get("Question") or qa.get("question") if isinstance(qa, dict) else None
            answer = qa.get("Answer") or qa.get("answer") if isinstance(qa, dict) else None
            if question and answer:
                output.append({"prompt": str(question), "answer": str(answer), "raw": item})
    return output


def _stable_record_id(source_file: Path, source_index: int, record: Dict[str, Any]) -> Tuple[str, str]:
    for key in ("record_id", "id", "uid", "qid", "question_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return f"medmkeb-{source_file.stem}-{value}", key
    return f"medmkeb-{source_file.stem}-{source_index}", "source_index"


def _normalize_record(
    record: Dict[str, Any],
    *,
    source_file: Path,
    source_index: int,
    data_root: Path,
    image_policy: str = "absolute",
) -> Dict[str, Any]:
    roots = _candidate_roots(data_root)
    prompt, prompt_field = _first_text(record, TEXT_FIELDS)
    old_answer, old_field = _first_text(record, OLD_FIELDS)
    new_answer, new_field = _first_text(record, NEW_FIELDS)
    image_ref, image_field = _first_text(record, IMAGE_FIELDS)
    rephrase, rephrase_field = _first_text(record, REPHRASE_FIELDS)
    loc_prompt, loc_prompt_field = _first_text(record, LOCALITY_PROMPT_FIELDS)
    loc_answer, loc_answer_field = _first_text(record, LOCALITY_ANSWER_FIELDS)
    loc_image_ref, loc_image_field = _first_text(record, LOCALITY_IMAGE_FIELDS)
    image_path, image_checked = _resolve_ref(image_ref, roots)
    loc_image_path, loc_checked = _resolve_ref(loc_image_ref, roots)
    rephrase_image_ref = record.get("image_rephrase") or record.get("rephrase_image") or image_ref
    rephrase_image_path, rephrase_checked = _resolve_ref(rephrase_image_ref, roots)
    record_id, id_field = _stable_record_id(source_file, source_index, record)

    def out_path(path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        return str(path) if image_policy == "absolute" else path.name

    metadata = {
        "id_field": id_field,
        "prompt_field": prompt_field,
        "old_answer_field": old_field,
        "new_answer_field": new_field,
        "image_field": image_field,
        "rephrase_field": rephrase_field,
        "locality_prompt_field": loc_prompt_field,
        "locality_answer_field": loc_answer_field,
        "locality_image_field": loc_image_field,
        "source_file": str(source_file),
        "source_index": source_index,
        "source_file_stem": source_file.stem,
        "private_patient_flags": _private_flags(record),
        "image_checked_paths": image_checked,
        "locality_checked_paths": loc_checked,
        "rephrase_checked_paths": rephrase_checked,
    }
    return {
        "id": record_id,
        "record_id": record_id,
        "src": prompt,
        "prompt": prompt,
        "pred": old_answer,
        "old_answer": old_answer,
        "erase_answer": old_answer,
        "alt": new_answer,
        "new_answer": new_answer,
        "replacement_answer": new_answer,
        "image": out_path(image_path),
        "image_original_ref": image_ref,
        "image_rephrase": out_path(rephrase_image_path or image_path),
        "image_rephrase_original_ref": str(rephrase_image_ref) if rephrase_image_ref is not None else None,
        "rephrase": rephrase or prompt,
        "loc": str(record.get("loc") or ""),
        "loc_ans": str(record.get("loc_ans") or ""),
        "m_loc": out_path(loc_image_path),
        "m_loc_original_ref": loc_image_ref,
        "m_loc_q": loc_prompt,
        "m_loc_a": loc_answer,
        "locality_prompts": [loc_prompt] if loc_prompt else [],
        "locality_answers": [loc_answer] if loc_answer else [],
        "rephrase_prompts": [rephrase] if rephrase else [],
        "portability_prompts": _extract_portability(record),
        "source_file": str(source_file),
        "source_index": source_index,
        "raw_metadata": metadata,
        "raw_record_subset": {key: record.get(key) for key in sorted(record) if key in {
            "id",
            "clinical_VQA_task",
            "department",
            "perceptual_granularity",
            "modality",
            "original_task",
        }},
        "public_non_phi_source_assumption": "public MedMKEB/VLKEB-style data; no private key flags detected"
        if not metadata["private_patient_flags"]
        else "private key flags detected by schema heuristic",
    }


def _field_presence(records: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, int]:
    return {field: sum(1 for record in records if record.get(field) not in (None, "")) for field in fields}


def _audit_data(data_root: Path, out_dir: Path) -> Dict[str, Any]:
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    files = _candidate_data_files(data_root)
    file_rows: List[Dict[str, Any]] = []
    for path in files:
        row: Dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix,
            "appledouble": path.name.startswith("._"),
        }
        try:
            records = _load_json_like(path)
            sample = records[0] if records else {}
            normalized = [
                _normalize_record(record, source_file=path, source_index=idx, data_root=data_root)
                for idx, record in enumerate(records[:200])
            ]
            image_resolved = sum(1 for item in normalized if item.get("image"))
            locality_resolved = sum(1 for item in normalized if item.get("m_loc"))
            row.update(
                {
                    "load_status": "ok",
                    "record_count": len(records),
                    "sample_keys": sorted(sample.keys()) if isinstance(sample, dict) else [],
                    "prompt_fields": _field_presence(records, TEXT_FIELDS),
                    "old_answer_fields": _field_presence(records, OLD_FIELDS),
                    "new_answer_fields": _field_presence(records, NEW_FIELDS),
                    "image_fields": _field_presence(records, IMAGE_FIELDS),
                    "rephrase_fields": _field_presence(records, REPHRASE_FIELDS),
                    "locality_prompt_fields": _field_presence(records, LOCALITY_PROMPT_FIELDS),
                    "locality_answer_fields": _field_presence(records, LOCALITY_ANSWER_FIELDS),
                    "locality_image_fields": _field_presence(records, LOCALITY_IMAGE_FIELDS),
                    "portability_field_count": sum(1 for record in records if record.get("port_new") or record.get("portability_qa")),
                    "first_200_image_resolved": image_resolved,
                    "first_200_locality_images_resolved": locality_resolved,
                    "private_patient_flag_rows_first_200": sum(
                        1 for item in normalized if item.get("raw_metadata", {}).get("private_patient_flags")
                    ),
                    "usable_for_modelknown": bool(
                        records
                        and row.get("appledouble") is False
                        and any(record.get("src") for record in records[:20])
                        and any(record.get("pred") for record in records[:20])
                        and any(record.get("alt") for record in records[:20])
                    ),
                }
            )
        except Exception as exc:
            row.update({"load_status": "error", "error": f"{type(exc).__name__}: {exc}", "record_count": 0})
        file_rows.append(row)

    payload = {
        "status": "pass" if any(row.get("usable_for_modelknown") for row in file_rows) else "fail",
        "data_root": str(data_root),
        "candidate_file_count": len(file_rows),
        "files": file_rows,
        "no_download_performed": True,
        "positional_matching_used": False,
    }
    _json_dump(audit_dir / "medmkeb_file_index.json", payload)

    lines = [
        "# MedMKEB Data Audit",
        "",
        f"- Data root: `{data_root}`",
        f"- Status: `{payload['status']}`",
        f"- Candidate files: `{len(file_rows)}`",
        "- Download performed: `False`",
        "- Positional matching used: `False`",
        "",
        "## Files",
        "",
        "| path | records | usable | image fields | old/new fields | locality fields | notes |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for row in file_rows:
        notes = row.get("error") or row.get("load_status")
        lines.append(
            "| {path} | {count} | {usable} | {img} | {oldnew} | {loc} | {notes} |".format(
                path=row.get("path"),
                count=row.get("record_count"),
                usable=row.get("usable_for_modelknown"),
                img=json.dumps(row.get("image_fields", {}), sort_keys=True),
                oldnew=json.dumps(
                    {"old": row.get("old_answer_fields", {}), "new": row.get("new_answer_fields", {})},
                    sort_keys=True,
                ),
                loc=json.dumps(
                    {
                        "prompt": row.get("locality_prompt_fields", {}),
                        "answer": row.get("locality_answer_fields", {}),
                        "image": row.get("locality_image_fields", {}),
                    },
                    sort_keys=True,
                ),
                notes=notes,
            )
        )
    lines.extend(
        [
            "",
            "## Mapping Decision",
            "",
            "- Preferred MedMKEB mapping is `src -> prompt`, `pred -> old_answer`, `alt -> new_answer`, `image -> image_path`, `m_loc_q/m_loc_a/m_loc -> locality`.",
            "- Explicit `id` is namespaced by source file stem to avoid cross-file collisions.",
            "- Records with unresolved edit images, empty prompt/old/new answers, non-finite model metrics, or private-key flags are filtered before editing.",
        ]
    )
    (audit_dir / "MEDMKEB_DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _schema_adapter_report(out_dir: Path) -> None:
    lines = [
        "# MedMKEB Schema Adapter Report",
        "",
        "## Normalized Schema",
        "",
        "- `record_id`: stable namespaced id, e.g. `medmkeb-eval_data_threehop_final-403`.",
        "- `image_path`: stored in the runner as the VLKEB-compatible `image` field.",
        "- `prompt`: copied from `src` when available.",
        "- `old_answer`: copied from `pred` when available.",
        "- `new_answer`: copied from `alt` when available.",
        "- `locality_prompts/locality_answers`: copied from `m_loc_q/m_loc_a` when available.",
        "- `rephrase_prompts`: copied from `rephrase` when available.",
        "- `portability_prompts`: parsed from `port_new[*]['Q&A']` when available.",
        "- `source_file/source_index/raw_metadata`: preserved for provenance.",
        "",
        "## Safety Rules",
        "",
        "- Positional matching is not used.",
        "- If explicit ids collide across files, the source file stem remains part of `record_id`.",
        "- Multiple possible mappings are recorded through `raw_metadata.*_field`; the MedMKEB-native `src/pred/alt/image` mapping is preferred because it is the local VLKEB `CaptionDataset` contract.",
    ]
    (out_dir / "audit" / "MEDMKEB_SCHEMA_ADAPTER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_image_to_bundle(path: Path, bundle_root: Path, source_data_root: Path) -> str:
    image_dir = bundle_root / "images"
    try:
        relative = path.resolve().relative_to((source_data_root / "images").resolve())
        dest = image_dir / relative
    except Exception:
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
        dest = image_dir / "copied" / f"{digest}_{path.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(path, dest)
    return str(Path("images") / dest.relative_to(image_dir))


def _build_candidate_bundle(
    *,
    data_root: Path,
    out_dir: Path,
    bundle_root: Path,
    source_file: Optional[Path],
    candidate_pool_size: int,
    seed: int,
) -> Dict[str, Any]:
    files = [source_file] if source_file else [data_root / "raw" / "eval_data_threehop_final.json"]
    files = [path for path in files if path and path.exists()]
    if not files:
        files = [path for path in _candidate_data_files(data_root) if path.name.startswith("eval_data")]
    if not files:
        raise RuntimeError(f"No usable MedMKEB source file under {data_root}")

    rng = random.Random(seed)
    normalized_all: List[Dict[str, Any]] = []
    filter_rows: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for file_path in files:
        records = _load_json_like(file_path)
        indexes = list(range(len(records)))
        rng.shuffle(indexes)
        for source_index in indexes:
            record = records[source_index]
            item = _normalize_record(record, source_file=file_path, source_index=source_index, data_root=data_root)
            reasons: List[str] = []
            if item["record_id"] in seen_ids:
                reasons.append("duplicate_record_id")
            if not item.get("src"):
                reasons.append("empty_prompt")
            if not item.get("old_answer"):
                reasons.append("empty_old_answer")
            if not item.get("new_answer"):
                reasons.append("empty_new_answer")
            if not item.get("image"):
                reasons.append("unresolved_image")
            if item.get("raw_metadata", {}).get("private_patient_flags"):
                reasons.append("private_patient_key_flag")
            if reasons:
                filter_rows.append({"record_id": item["record_id"], "source_file": str(file_path), "source_index": source_index, "keep": False, "reasons": reasons})
                continue
            seen_ids.add(item["record_id"])
            item["candidate_pool_rank"] = len(normalized_all)
            normalized_all.append(item)
            filter_rows.append({"record_id": item["record_id"], "source_file": str(file_path), "source_index": source_index, "keep": True, "reasons": []})
            if len(normalized_all) >= candidate_pool_size:
                break
        if len(normalized_all) >= candidate_pool_size:
            break

    if len(normalized_all) < 20:
        raise RuntimeError(f"Fewer than 20 locally valid MedMKEB candidates: {len(normalized_all)}")

    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundled_records: List[Dict[str, Any]] = []
    copied_images: Dict[str, str] = {}
    for item in normalized_all:
        copied = dict(item)
        for field in ("image", "image_rephrase", "m_loc"):
            value = copied.get(field)
            if not value:
                continue
            path = Path(value)
            if not path.exists():
                continue
            key = str(path.resolve())
            if key not in copied_images:
                copied_images[key] = _copy_image_to_bundle(path, bundle_root, data_root)
            copied[field] = copied_images[key]
        bundled_records.append(copied)

    modelknown_dir = out_dir / "modelknown_20"
    _json_dump(modelknown_dir / "medmkeb_candidates_normalized.json", bundled_records)
    _json_dump(
        modelknown_dir / "medmkeb_candidate_filter_report.json",
        {
            "stage": "local_schema_path_preselection",
            "source_files": [str(path) for path in files],
            "candidate_pool_size_requested": candidate_pool_size,
            "candidate_pool_size": len(bundled_records),
            "selected_for_remote_nll_filter": len(bundled_records),
            "rows": filter_rows,
            "positional_matching_used": False,
        },
    )
    records_path = bundle_root / "records.json"
    _json_dump(records_path, bundled_records)
    manifest = {
        "status": "pass",
        "bundle_root": str(bundle_root),
        "records_path": str(records_path),
        "record_count": len(bundled_records),
        "unique_record_count": len({row["record_id"] for row in bundled_records}),
        "copied_image_count": len(copied_images),
        "source_data_root": str(data_root),
        "source_files": [str(path) for path in files],
        "candidate_pool_size": len(bundled_records),
        "no_download_performed": True,
        "positional_matching_used": False,
    }
    _json_dump(bundle_root / "bundle_manifest.json", manifest)
    _json_dump(modelknown_dir / "local_bundle_manifest.json", manifest)
    return manifest


def _metric_value(raw: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not raw or not raw.get("available") or raw.get(key) is None:
        return None
    return float(raw[key])


def _finite(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den in (None, 0.0):
        return None
    return float(num) / float(den)


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _heavy_imports() -> Dict[str, Any]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import torch

    from easyeditor.editors.multimodal_editor import MultimodalEditor
    from easyeditor.models.engram import EngramMultimodalHparams
    from easyeditor.models.engram.bank import EngramBank
    from easyeditor.models.engram.engram_main import select_linear_layers
    from scripts.engram.run_localized_replacement_5edit import (
        EvalLoraPatch,
        _configure_hparams,
        _evaluate_current,
        _extract_projector_bank,
        _make_eval_row,
        _max_snapshot_diff,
        _project_factors,
        _restore_modules,
        _snapshot_modules,
        _train_tiny_lora,
    )

    return {
        "torch": torch,
        "MultimodalEditor": MultimodalEditor,
        "EngramMultimodalHparams": EngramMultimodalHparams,
        "EngramBank": EngramBank,
        "select_linear_layers": select_linear_layers,
        "EvalLoraPatch": EvalLoraPatch,
        "_configure_hparams": _configure_hparams,
        "_evaluate_current": _evaluate_current,
        "_extract_projector_bank": _extract_projector_bank,
        "_make_eval_row": _make_eval_row,
        "_max_snapshot_diff": _max_snapshot_diff,
        "_project_factors": _project_factors,
        "_restore_modules": _restore_modules,
        "_snapshot_modules": _snapshot_modules,
        "_train_tiny_lora": _train_tiny_lora,
    }


def _run_pytest(log_path: Path, args: List[str]) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return {"command": " ".join([sys.executable, "-m", "pytest", *args]), "returncode": proc.returncode, "log": str(log_path)}


def _write_tests(out_dir: Path, run_tests: bool) -> Dict[str, Any]:
    test_dir = out_dir / "test_logs"
    test_dir.mkdir(parents=True, exist_ok=True)
    if not run_tests:
        payload = {"status": "skipped", "reason": "--skip-tests"}
        _json_dump(test_dir / "test_status.json", payload)
        return payload
    engram_tests = sorted(str(path) for path in PROJECT_ROOT.glob("tests/test_engram_*.py"))
    runs = [
        _run_pytest(test_dir / "test_cure_crisp_projection.log", ["tests/test_cure_crisp_projection.py", "-q"]),
        _run_pytest(test_dir / "test_cure_kfac_collector_tiny_mllm.log", ["tests/test_cure_kfac_collector_tiny_mllm.py", "-q"]),
        _run_pytest(test_dir / "test_engram_all.log", [*engram_tests, "-q"]),
    ]
    cure_pass = all(item["returncode"] == 0 for item in runs[:2])
    engram_pass = runs[2]["returncode"] == 0
    payload = {
        "status": "pass" if cure_pass and engram_pass else ("engram_pass_cure_fail" if engram_pass else "fail"),
        "cure_tests_pass": cure_pass,
        "engram_tests_pass": engram_pass,
        "runs": runs,
    }
    _json_dump(test_dir / "test_status.json", payload)
    return payload


def _yaml_value(path: Path, key: str) -> Optional[str]:
    if not path.exists():
        return None
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip().strip("\"'")
    return None


def _write_preflight(out_dir: Path, *, hparams_path: Path, input_records: Path, image_root: Path, test_status: Dict[str, Any]) -> Dict[str, Any]:
    model_path = _yaml_value(hparams_path, "name") or _yaml_value(hparams_path, "model_path") or _yaml_value(hparams_path, "model_name")
    vision_path = (
        _yaml_value(hparams_path, "llava_med_vision_tower")
        or _yaml_value(hparams_path, "clip_vision_path")
        or _yaml_value(hparams_path, "vision_tower")
    )
    checks = {
        "python_path_present": bool(sys.executable),
        "path_exists_hparams": hparams_path.exists(),
        "path_exists_model": bool(model_path and Path(model_path).exists()),
        "path_exists_vision_tower": bool(vision_path and Path(vision_path).exists()),
        "input_records_exists": input_records.exists(),
        "image_root_exists": image_root.exists(),
        "output_dir_writable": out_dir.exists() and out_dir.is_dir(),
        "engram_tests_pass": test_status.get("engram_tests_pass") is True or test_status.get("status") == "skipped",
    }
    try:
        import torch

        checks["cuda_available"] = bool(torch.cuda.is_available())
        checks["cuda_device_count_positive"] = int(torch.cuda.device_count()) > 0
    except Exception as exc:
        checks["cuda_available"] = f"{type(exc).__name__}: {exc}"
        checks["cuda_device_count_positive"] = False
    payload = {
        "status": "pass" if all(value is True for value in checks.values()) else "fail",
        "checks": checks,
        "paths": {
            "hparams": str(hparams_path),
            "model_path": model_path,
            "vision_tower": vision_path,
            "input_records": str(input_records),
            "image_root": str(image_root),
            "output_dir": str(out_dir),
        },
        "test_status": test_status,
    }
    lines = ["# MedMKEB ENGRAM-Projected LoRA Preflight", "", f"- Status: `{payload['status']}`", f"- Python: `{sys.executable}`", ""]
    lines.extend(["## Checks", ""])
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Paths", ""])
    for key, value in payload["paths"].items():
        lines.append(f"- {key}: `{value}`")
    (out_dir / "PREFLIGHT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json_dump(out_dir / "preflight_status.json", payload)
    return payload


def _load_bundle_records(path: Path) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Expected non-empty records list: {path}")
    return records


def _record_reference_available(record: Dict[str, Any]) -> bool:
    return bool(record.get("m_loc_q") and record.get("m_loc_a") and record.get("m_loc"))


def _baseline_row(record: Dict[str, Any], baseline: Dict[str, Any], image_root: Path) -> Dict[str, Any]:
    old_raw = baseline.get("old_raw") or {}
    new_raw = baseline.get("new_raw") or {}
    ref_raw = baseline.get("reference_raw") or {}
    return {
        "record_id": str(record["id"]),
        "source_file": record.get("source_file"),
        "source_index": record.get("source_index"),
        "old_answer_nll": _metric_value(old_raw, "nll"),
        "old_answer_logprob": _metric_value(old_raw, "logprob"),
        "old_answer_token_count": old_raw.get("num_tokens") or old_raw.get("answer_token_count"),
        "new_answer_nll": _metric_value(new_raw, "nll"),
        "new_answer_token_count": new_raw.get("num_tokens") or new_raw.get("answer_token_count"),
        "reference_nll": _metric_value(ref_raw, "nll") if ref_raw else None,
        "reference_token_count": (ref_raw or {}).get("num_tokens") or (ref_raw or {}).get("answer_token_count"),
        "image_path_resolved": bool((image_root / str(record.get("image", ""))).exists() or Path(str(record.get("image", ""))).exists()),
        "x_minus_nonempty": _record_reference_available(record),
        "prompt_token_count": len(str(record.get("src") or "").split()),
        "answer_token_count": len(str(record.get("new_answer") or record.get("alt") or "").split()),
        "nan_inf_detected": not _finite({"old": old_raw, "new": new_raw, "reference": ref_raw}),
        "private_patient_flags": record.get("raw_metadata", {}).get("private_patient_flags", []),
        "old_raw": old_raw,
        "new_raw": new_raw,
        "reference_raw": ref_raw,
    }


def _select_modelknown_records(
    *,
    records: List[Dict[str, Any]],
    baselines: Dict[str, Dict[str, Any]],
    image_root: Path,
    out_dir: Path,
    count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    valid: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen: set[str] = set()
    for record in records:
        row = _baseline_row(record, baselines[str(record["id"])], image_root)
        reasons: List[str] = []
        if row["record_id"] in seen:
            reasons.append("duplicate_record_id")
        if not record.get("src"):
            reasons.append("empty_prompt")
        if not record.get("old_answer") and not record.get("pred"):
            reasons.append("empty_old_answer")
        if not record.get("new_answer") and not record.get("alt"):
            reasons.append("empty_new_answer")
        if not row["image_path_resolved"]:
            reasons.append("unresolved_image")
        if row["old_answer_token_count"] in (None, 0):
            reasons.append("old_answer_zero_tokens")
        if row["new_answer_token_count"] in (None, 0):
            reasons.append("new_answer_zero_tokens")
        if row["old_answer_nll"] is None or not math.isfinite(float(row["old_answer_nll"])):
            reasons.append("old_answer_nll_nonfinite")
        if row["new_answer_nll"] is None or not math.isfinite(float(row["new_answer_nll"])):
            reasons.append("new_answer_nll_nonfinite")
        if row["reference_nll"] is not None and not math.isfinite(float(row["reference_nll"])):
            reasons.append("reference_nll_nonfinite")
        if row["private_patient_flags"]:
            reasons.append("private_patient_key_flag")
        keep = not reasons
        row["keep_after_model_metric_filter"] = keep
        row["filter_reasons"] = reasons
        rows.append(row)
        seen.add(row["record_id"])
        if keep:
            valid.append((record, row))

    if len(valid) < count:
        summary = {
            "status": "blocked",
            "reason": f"valid candidates {len(valid)} < required {count}",
            "valid_candidates": len(valid),
            "required": count,
            "positional_matching_used": False,
        }
        _json_dump(out_dir / "modelknown_20" / "medmkeb_modelknown_20_summary.json", summary)
        raise RuntimeError(summary["reason"])

    old_values = sorted(float(row["old_answer_nll"]) for _, row in valid)
    median_old = old_values[len(old_values) // 2]
    valid.sort(
        key=lambda pair: (
            0 if float(pair[1]["old_answer_nll"]) <= median_old else 1,
            0 if pair[1]["x_minus_nonempty"] else 1,
            0 if pair[0].get("rephrase") else 1,
            float(pair[1]["old_answer_nll"]),
            pair[1]["record_id"],
        )
    )
    selected = [record for record, _ in valid[:count]]
    selected_ids = {record["id"] for record in selected}
    selected_rows = [row for row in rows if row["record_id"] in selected_ids]
    summary = {
        "status": "pass",
        "selected": len(selected),
        "unique_selected": len(selected_ids),
        "candidate_pool_size": len(records),
        "valid_candidates": len(valid),
        "median_old_answer_nll_valid": median_old,
        "selection_policy": "prefer low old-answer NLL below candidate median, locality prompts, rephrase prompts",
        "record_id_match_rate": 1.0 if len(selected_ids) == len(selected) else 0.0,
        "images_resolved": sum(1 for row in selected_rows if row["image_path_resolved"]),
        "answers_nonempty": sum(1 for record in selected if record.get("old_answer") and record.get("new_answer")),
        "finite_base_metrics": sum(1 for row in selected_rows if not row["nan_inf_detected"]),
        "positional_matching_used": False,
    }
    model_dir = out_dir / "modelknown_20"
    _json_dump(model_dir / "medmkeb_candidates_normalized.json", rows)
    _json_dump(model_dir / "medmkeb_candidate_filter_report.json", {"rows": rows, "summary": summary})
    _json_dump(model_dir / "medmkeb_modelknown_20.json", selected)
    _json_dump(model_dir / "medmkeb_modelknown_20_summary.json", summary)
    return selected, rows, summary


def _configure_hparams_for_run(hparams: Any, *, image_root: Path, bank_dir: Path, device: str, beta: float) -> None:
    heavy = _heavy_imports()
    heavy["_configure_hparams"](hparams, image_root=image_root, bank_dir=bank_dir, device=device, edit_mode="erase")
    hparams.replacement_mode = "lora_projected"
    hparams.candidate_delta_source = "tiny_lora"
    hparams.project_delta_with_engram = True
    hparams.replacement_beta = float(beta)
    hparams.replacement_lambda_ref = 0.0
    hparams.lora_rank = int(getattr(hparams, "lora_rank", 4) or 4)
    hparams.lora_steps = int(getattr(hparams, "lora_steps", 20) or 20)
    hparams.lora_lr = float(getattr(hparams, "lora_lr", 1.0e-4) or 1.0e-4)
    hparams.token_scope = "all"
    hparams.module_patterns = [f"^{name.replace('.', r'\\.')}$$".replace("$$", "$") for name in EXPECTED_MODULES]
    hparams.exclude_module_patterns = [r"lm_head$", r"down_proj$", r"mm_projector"]
    hparams.prioritize_module_selection = True
    hparams.module_priority_patterns = [r"gate_proj$", r"q_proj$", r"k_proj$"]
    hparams.engram_layers = [0, 8, 16, 24]
    hparams.engram_max_modules = None


def _aggregate_nonseq(rows: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    metric_rows = [row for row in rows if row.get("method") == method]
    new_decreases = [row.get("new_answer_nll_decrease") for row in metric_rows if row.get("new_answer_nll_decrease") is not None]
    old_increases = [row.get("old_answer_nll_increase") for row in metric_rows if row.get("old_answer_nll_increase") is not None]
    ref_abs = [row.get("reference_delta_abs") for row in metric_rows if row.get("reference_delta_abs") is not None]
    mean_new = _mean(new_decreases)
    mean_ref = _mean(ref_abs)
    return {
        "method": method,
        "record_count": len(metric_rows),
        "mean_new_answer_nll_decrease": mean_new,
        "mean_old_answer_nll_increase": _mean(old_increases),
        "mean_locality_reference_delta_abs": mean_ref,
        "mean_ref_delta": mean_ref,
        "positive_new_answer_edits": sum(1 for value in new_decreases if value is not None and float(value) > 0.0),
        "positive_old_answer_erasure_edits": sum(1 for value in old_increases if value is not None and float(value) > 0.0),
        "locality_damage_edits": sum(1 for row in metric_rows if row.get("locality_damage")),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in metric_rows]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in metric_rows]),
        "nan_inf_count": sum(1 for row in metric_rows if row.get("nan_inf_detected")),
        "target_to_reference_ratio": _safe_div(mean_new, mean_ref),
    }


def _nonseq_acceptance(aggregates: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method = {row["method"]: row for row in aggregates}
    b = by_method.get(METHOD_B, {})
    c = by_method.get(METHOD_C, {})
    checks = {
        "positive_new_answer_edits_at_least_16": int(c.get("positive_new_answer_edits") or 0) >= 16,
        "mean_new_answer_nll_decrease_positive": c.get("mean_new_answer_nll_decrease") is not None and float(c["mean_new_answer_nll_decrease"]) > 0.0,
        "locality_less_than_new_signal": (
            c.get("mean_locality_reference_delta_abs") is not None
            and c.get("mean_new_answer_nll_decrease") is not None
            and float(c["mean_locality_reference_delta_abs"]) < float(c["mean_new_answer_nll_decrease"])
        ),
        "rollback_pass_rate_is_1": float(c.get("rollback_pass_rate") or 0.0) == 1.0,
        "record_id_match_rate_is_1": float(c.get("record_id_match_rate") or 0.0) == 1.0,
        "nan_inf_count_is_0": int(c.get("nan_inf_count") or 0) == 0,
        "locality_damage_edits_lte_B": int(c.get("locality_damage_edits") or 0) <= int(b.get("locality_damage_edits") or 0),
        "mean_locality_reference_delta_abs_lte_B": (
            c.get("mean_locality_reference_delta_abs") is not None
            and b.get("mean_locality_reference_delta_abs") is not None
            and float(c["mean_locality_reference_delta_abs"]) <= float(b["mean_locality_reference_delta_abs"])
        ),
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks, "B": b, "C": c}


def _run_nonseq(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    module_names: List[str],
    hparams: Any,
    out_dir: Path,
    beta: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    skip_generation: bool,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    heavy = _heavy_imports()
    EvalLoraPatch = heavy["EvalLoraPatch"]
    EngramBank = heavy["EngramBank"]
    _make_eval_row = heavy["_make_eval_row"]
    _max_snapshot_diff = heavy["_max_snapshot_diff"]
    _project_factors = heavy["_project_factors"]
    _restore_modules = heavy["_restore_modules"]
    _snapshot_modules = heavy["_snapshot_modules"]
    _train_tiny_lora = heavy["_train_tiny_lora"]
    _evaluate_current = heavy["_evaluate_current"]

    bank = EngramBank(projector_bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    snapshots = _snapshot_modules(model, module_names)
    scale = float(hparams.lora_scale if getattr(hparams, "lora_scale", None) is not None else 1.0)
    rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Dict[str, Any]] = {}
    final_factors: Dict[str, Dict[str, Any]] = {"B": {}, "C": {}}
    try:
        for idx, (record, edit_id) in enumerate(zip(records, edit_ids)):
            _restore_modules(model, snapshots)
            rows.append(
                _make_eval_row(
                    method=METHOD_A,
                    record=record,
                    case_index=idx,
                    before=baselines[str(record["id"])],
                    after=baselines[str(record["id"])],
                    rollback_diff=0.0,
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    record_id_match_rate=1.0,
                    edit_id=None,
                    beta=0.0,
                    extra={"positional_matching_used": False},
                )
            )
            factors, train_summary = _train_tiny_lora(
                model,
                record,
                image_root,
                module_names,
                rank=int(hparams.lora_rank),
                steps=int(hparams.lora_steps),
                lr=float(hparams.lora_lr),
                scale=scale,
                lambda_ref=float(hparams.replacement_lambda_ref),
            )
            safe_factors, projection_summary = _project_factors(factors, bank.load_edit(edit_id))
            metadata[str(record["id"])] = {
                "record_id": str(record["id"]),
                "edit_id": edit_id,
                "lora_train": train_summary,
                "engram_projection": projection_summary,
                "selected_modules": module_names,
                "positional_matching_used": False,
            }
            for method, patch_factors, extra in [
                (METHOD_B, factors, {"project_delta_with_engram": False, "lora_train": train_summary}),
                (METHOD_C, safe_factors, {"project_delta_with_engram": True, "engram_projection": projection_summary}),
            ]:
                patch = EvalLoraPatch(model, patch_factors, beta=beta)
                patch.install()
                try:
                    after = _evaluate_current(
                        model,
                        record,
                        image_root,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=None,
                        skip_generation=skip_generation,
                    )
                finally:
                    patch.remove()
                rows.append(
                    _make_eval_row(
                        method=method,
                        record=record,
                        case_index=idx,
                        before=baselines[str(record["id"])],
                        after=after,
                        rollback_diff=_max_snapshot_diff(model, snapshots),
                        rollback_tolerance=rollback_tolerance,
                        locality_threshold=locality_threshold,
                        record_id_match_rate=1.0 if matching.get("mode") == "record_id" else 0.0,
                        edit_id=edit_id,
                        beta=beta,
                        extra={**extra, "positional_matching_used": False},
                    )
                )
            final_factors["B"][str(record["id"])] = factors
            final_factors["C"][str(record["id"])] = safe_factors
            _restore_modules(model, snapshots)
    finally:
        _restore_modules(model, snapshots)

    aggregates = [_aggregate_nonseq(rows, method) for method in [METHOD_A, METHOD_B, METHOD_C]]
    acceptance = _nonseq_acceptance(aggregates)
    payload = {
        "status": "complete",
        "methods": [METHOD_A, METHOD_B, METHOD_C],
        "beta": beta,
        "record_id_matching": matching,
        "per_record": rows,
        "aggregate_rows": aggregates,
        "acceptance": acceptance,
        "optional_E_status": "skipped_primary_C_gate_only",
    }
    nonseq_dir = out_dir / "modelknown_20" / "nonseq"
    _json_dump(nonseq_dir / "nonseq_results.json", payload)
    _write_csv(nonseq_dir / "nonseq_results.csv", rows)
    _write_csv(nonseq_dir / "nonseq_aggregates.csv", aggregates)
    _json_dump(out_dir / "modelknown_20" / "replacement_bank_metadata.json", metadata)
    return payload, final_factors


def _step_rows(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    snapshots: Dict[str, Any],
    method: str,
    step: int,
    applied_record_ids: List[str],
    beta: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    skip_generation: bool,
    record_id_match_rate: float,
) -> List[Dict[str, Any]]:
    heavy = _heavy_imports()
    _evaluate_current = heavy["_evaluate_current"]
    _make_eval_row = heavy["_make_eval_row"]
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        after = _evaluate_current(
            model,
            record,
            image_root,
            max_new_tokens=max_new_tokens,
            min_new_tokens=None,
            skip_generation=skip_generation,
        )
        rows.append(
            _make_eval_row(
                method=method,
                record=record,
                case_index=idx,
                before=baselines[str(record["id"])],
                after=after,
                rollback_diff=0.0,
                rollback_tolerance=rollback_tolerance,
                locality_threshold=locality_threshold,
                record_id_match_rate=record_id_match_rate,
                edit_id=None,
                beta=beta,
                extra={
                    "step": int(step),
                    "applied_record_ids": list(applied_record_ids),
                    "is_edited_record_at_step": idx < step,
                    "is_previous_edit_record": idx < max(step - 1, 0),
                    "is_current_edit_record": idx == step - 1,
                    "is_future_record": idx >= step,
                    "positional_matching_used": False,
                },
            )
        )
    return rows


def _aggregate_seq_step(rows: List[Dict[str, Any]], method: str, step: int, total: int) -> Dict[str, Any]:
    step_rows = [row for row in rows if row.get("method") == method and int(row.get("step") or 0) == step]
    edited = [row for row in step_rows if row.get("is_edited_record_at_step")]
    previous = [row for row in step_rows if row.get("is_previous_edit_record")]
    future = [row for row in step_rows if row.get("is_future_record")]
    all_ref = [row.get("reference_delta_abs") for row in step_rows if row.get("reference_delta_abs") is not None]
    edited_new = [row.get("new_answer_nll_decrease") for row in edited if row.get("new_answer_nll_decrease") is not None]
    return {
        "method": method,
        "step": step,
        "record_count": len(step_rows),
        "edited_record_count": len(edited),
        "mean_new_answer_nll_decrease_edited_records": _mean(edited_new),
        "mean_new_answer_nll_decrease_all_records": _mean([row.get("new_answer_nll_decrease") for row in step_rows]),
        "mean_locality_reference_delta_abs": _mean(all_ref),
        "mean_reference_delta_abs_all_records": _mean(all_ref),
        "positive_new_answer_edits": sum(1 for row in edited if (row.get("new_answer_nll_decrease") or 0.0) > 0.0),
        "locality_damage_records": sum(1 for row in step_rows if row.get("locality_damage")),
        "mean_previous_edit_retention": _mean([row.get("new_answer_nll_decrease") for row in previous]),
        "mean_future_record_drift": _mean([abs(float(row.get("new_answer_nll_decrease") or 0.0)) for row in future]),
        "rollback_pass_rate": _mean([1.0 if row.get("rollback_pass") else 0.0 for row in step_rows]),
        "record_id_match_rate": _mean([float(row.get("record_id_match_rate") or 0.0) for row in step_rows]),
        "nan_inf_count": sum(1 for row in step_rows if row.get("nan_inf_detected")),
        "final_step": step == total,
    }


def _run_one_seq_method(
    *,
    model: Any,
    method: str,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    module_names: List[str],
    hparams: Any,
    beta: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    skip_generation: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Tuple[str, Dict[str, Any]]]]:
    heavy = _heavy_imports()
    EngramBank = heavy["EngramBank"]
    EvalLoraPatch = heavy["EvalLoraPatch"]
    _project_factors = heavy["_project_factors"]
    _snapshot_modules = heavy["_snapshot_modules"]
    _restore_modules = heavy["_restore_modules"]
    _max_snapshot_diff = heavy["_max_snapshot_diff"]
    _train_tiny_lora = heavy["_train_tiny_lora"]

    bank = EngramBank(projector_bank_dir)
    edit_ids, matching = bank.match_edit_ids_to_records(records, allow_positional_matching=False)
    snapshots = _snapshot_modules(model, module_names)
    scale = float(hparams.lora_scale if getattr(hparams, "lora_scale", None) is not None else 1.0)
    rows: List[Dict[str, Any]] = []
    active_patches: List[Any] = []
    patch_specs: List[Tuple[str, Dict[str, Any]]] = []
    applied_ids: List[str] = []
    rows.extend(
        _step_rows(
            model=model,
            records=records,
            image_root=image_root,
            baselines=baselines,
            snapshots=snapshots,
            method=method,
            step=0,
            applied_record_ids=[],
            beta=beta,
            rollback_tolerance=rollback_tolerance,
            locality_threshold=locality_threshold,
            max_new_tokens=max_new_tokens,
            skip_generation=skip_generation,
            record_id_match_rate=1.0 if matching.get("mode") == "record_id" else 0.0,
        )
    )
    rollback_diff = 0.0
    try:
        for step, (record, edit_id) in enumerate(zip(records, edit_ids), start=1):
            factors, _ = _train_tiny_lora(
                model,
                record,
                image_root,
                module_names,
                rank=int(hparams.lora_rank),
                steps=int(hparams.lora_steps),
                lr=float(hparams.lora_lr),
                scale=scale,
                lambda_ref=float(hparams.replacement_lambda_ref),
            )
            patch_factors = factors
            if method == METHOD_C:
                patch_factors, _ = _project_factors(factors, bank.load_edit(edit_id))
            patch = EvalLoraPatch(model, patch_factors, beta=beta)
            patch.install()
            active_patches.append(patch)
            patch_specs.append((str(record["id"]), patch_factors))
            applied_ids.append(str(record["id"]))
            rows.extend(
                _step_rows(
                    model=model,
                    records=records,
                    image_root=image_root,
                    baselines=baselines,
                    snapshots=snapshots,
                    method=method,
                    step=step,
                    applied_record_ids=applied_ids,
                    beta=beta,
                    rollback_tolerance=rollback_tolerance,
                    locality_threshold=locality_threshold,
                    max_new_tokens=max_new_tokens,
                    skip_generation=skip_generation,
                    record_id_match_rate=1.0 if matching.get("mode") == "record_id" else 0.0,
                )
            )
    finally:
        for patch in reversed(active_patches):
            patch.remove()
        rollback_diff = _max_snapshot_diff(model, snapshots)
        _restore_modules(model, snapshots)

    for row in rows:
        if int(row.get("step") or 0) == len(records):
            row["rollback_max_abs_diff"] = rollback_diff
            row["rollback_pass"] = rollback_diff <= rollback_tolerance
    return rows, {"method": method, "rollback_max_abs_diff": rollback_diff, "rollback_pass": rollback_diff <= rollback_tolerance}, patch_specs


def _sequential_acceptance(summary_rows: List[Dict[str, Any]], rollback: Dict[str, Any]) -> Dict[str, Any]:
    final = [row for row in summary_rows if row.get("final_step")]
    by_method = {row["method"]: row for row in final}
    b = by_method.get(METHOD_B, {})
    c = by_method.get(METHOD_C, {})
    c_new = c.get("mean_new_answer_nll_decrease_edited_records")
    b_new = b.get("mean_new_answer_nll_decrease_edited_records")
    c_ref = c.get("mean_reference_delta_abs_all_records")
    b_ref = b.get("mean_reference_delta_abs_all_records")
    checks = {
        "positive_new_answer_edits_at_least_16": int(c.get("positive_new_answer_edits") or 0) >= 16,
        "mean_new_answer_nll_decrease_positive": c_new is not None and float(c_new) > 0.0,
        "locality_less_than_new_signal": c_ref is not None and c_new is not None and float(c_ref) < float(c_new),
        "locality_damage_records_lte_B": int(c.get("locality_damage_records") or 0) <= int(b.get("locality_damage_records") or 0),
        "rollback_pass": rollback.get("status") == "pass",
        "record_id_match_rate_is_1": float(c.get("record_id_match_rate") or 0.0) == 1.0,
        "nan_inf_count_is_0": int(c.get("nan_inf_count") or 0) == 0,
        "C_new_answer_ratio_vs_B_at_least_0_90": (_safe_div(c_new, b_new) is not None and float(_safe_div(c_new, b_new)) >= 0.90),
        "C_reference_ratio_vs_B_at_most_0_50_if_possible": (
            _safe_div(c_ref, b_ref) is None or float(_safe_div(c_ref, b_ref)) <= 0.50
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else ("partial" if all(checks[k] for k in list(checks)[:7]) else "fail"),
        "checks": checks,
        "B_final": b,
        "C_final": c,
        "C_new_answer_ratio_vs_B": _safe_div(c_new, b_new),
        "C_reference_ratio_vs_B": _safe_div(c_ref, b_ref),
    }


def _run_sequential(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    baselines: Dict[str, Dict[str, Any]],
    projector_bank_dir: Path,
    module_names: List[str],
    hparams: Any,
    out_dir: Path,
    beta: float,
    rollback_tolerance: float,
    locality_threshold: float,
    max_new_tokens: int,
    skip_generation: bool,
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[str, Dict[str, Any]]]]]:
    seq_dir = out_dir / "modelknown_20" / "sequential"
    all_rows: List[Dict[str, Any]] = []
    rollbacks: List[Dict[str, Any]] = []
    final_specs: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for method in [METHOD_B, METHOD_C]:
        rows, rollback, specs = _run_one_seq_method(
            model=model,
            method=method,
            records=records,
            image_root=image_root,
            baselines=baselines,
            projector_bank_dir=projector_bank_dir,
            module_names=module_names,
            hparams=hparams,
            beta=beta,
            rollback_tolerance=rollback_tolerance,
            locality_threshold=locality_threshold,
            max_new_tokens=max_new_tokens,
            skip_generation=skip_generation,
        )
        all_rows.extend(rows)
        rollbacks.append(rollback)
        final_specs[method] = specs
    summary_rows = [
        _aggregate_seq_step(all_rows, method, step, len(records))
        for method in [METHOD_B, METHOD_C]
        for step in range(0, len(records) + 1)
    ]
    rollback_payload = {
        "status": "pass" if all(item.get("rollback_pass") for item in rollbacks) else "fail",
        "methods": rollbacks,
        "rollback_tolerance": rollback_tolerance,
    }
    acceptance = _sequential_acceptance(summary_rows, rollback_payload)
    payload = {
        "status": "complete",
        "methods": [METHOD_B, METHOD_C],
        "beta": beta,
        "per_record_step_rows": all_rows,
        "summary_rows": summary_rows,
        "final_rollback_check": rollback_payload,
        "acceptance": acceptance,
    }
    _json_dump(seq_dir / "sequential_step_matrix.json", all_rows)
    _write_csv(seq_dir / "sequential_step_matrix.csv", all_rows)
    _json_dump(seq_dir / "sequential_summary.json", payload)
    _write_csv(seq_dir / "sequential_summary.csv", summary_rows)
    _json_dump(seq_dir / "final_rollback_check.json", rollback_payload)
    return payload, final_specs


def _generation_diagnostics(
    *,
    model: Any,
    records: List[Dict[str, Any]],
    image_root: Path,
    final_specs: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    out_dir: Path,
    beta: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    heavy = _heavy_imports()
    EvalLoraPatch = heavy["EvalLoraPatch"]
    _evaluate_current = heavy["_evaluate_current"]
    selected = records[:5]
    rows: List[Dict[str, Any]] = []
    for method in ["baseline", METHOD_B, METHOD_C]:
        patches: List[Any] = []
        if method in final_specs:
            for _, factors in final_specs[method]:
                patch = EvalLoraPatch(model, factors, beta=beta)
                patch.install()
                patches.append(patch)
        try:
            for idx, record in enumerate(selected):
                result = _evaluate_current(
                    model,
                    record,
                    image_root,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=None,
                    skip_generation=False,
                )
                rows.append(
                    {
                        "method": method,
                        "record_id": record["id"],
                        "case_index": idx,
                        "prompt": record.get("src"),
                        "old_answer": record.get("old_answer"),
                        "new_answer": record.get("new_answer"),
                        "generation": result.get("generation"),
                        "new_raw": result.get("new_raw"),
                        "reference_raw": result.get("reference_raw"),
                    }
                )
        finally:
            for patch in reversed(patches):
                patch.remove()
    payload = {"status": "complete", "record_count": len(selected), "rows": rows, "primary_gate": False}
    diag_dir = out_dir / "generation_diagnostics"
    _json_dump(diag_dir / "generation_diagnostics_5records.json", payload)
    _write_csv(diag_dir / "generation_diagnostics_5records.csv", rows)
    return payload


def _plot_optional(out_dir: Path) -> Dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    plot_dir = out_dir / "modelknown_20" / "plots"
    nonseq_path = out_dir / "modelknown_20" / "nonseq" / "nonseq_aggregates.csv"
    seq_path = out_dir / "modelknown_20" / "sequential" / "sequential_summary.csv"
    made: List[str] = []
    try:
        if nonseq_path.exists():
            rows = list(csv.DictReader(nonseq_path.open(encoding="utf-8")))
            labels = [row["method"] for row in rows]
            new = [float(row.get("mean_new_answer_nll_decrease") or 0.0) for row in rows]
            ref = [float(row.get("mean_locality_reference_delta_abs") or 0.0) for row in rows]
            plt.figure(figsize=(8, 4))
            x = range(len(labels))
            plt.bar([i - 0.2 for i in x], new, width=0.4, label="new decrease")
            plt.bar([i + 0.2 for i in x], ref, width=0.4, label="reference abs")
            plt.xticks(list(x), labels, rotation=20, ha="right")
            plt.legend()
            plt.tight_layout()
            path = plot_dir / "nonseq_new_vs_reference.png"
            plt.savefig(path)
            plt.close()
            made.append(str(path))
        if seq_path.exists():
            rows = list(csv.DictReader(seq_path.open(encoding="utf-8")))
            for metric, filename in [
                ("mean_new_answer_nll_decrease_edited_records", "sequential_new_curve.png"),
                ("mean_reference_delta_abs_all_records", "sequential_reference_curve.png"),
                ("mean_previous_edit_retention", "sequential_retention_curve.png"),
            ]:
                plt.figure(figsize=(7, 4))
                for method in [METHOD_B, METHOD_C]:
                    method_rows = [row for row in rows if row["method"] == method]
                    xs = [int(row["step"]) for row in method_rows]
                    ys = [float(row.get(metric) or 0.0) for row in method_rows]
                    plt.plot(xs, ys, marker="o", label=method)
                plt.xlabel("step")
                plt.ylabel(metric)
                plt.legend()
                plt.tight_layout()
                path = plot_dir / filename
                plt.savefig(path)
                plt.close()
                made.append(str(path))
        return {"status": "complete", "files": made}
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "files": made}


def _write_final_report(
    *,
    out_dir: Path,
    audit_status: Optional[Dict[str, Any]],
    selected_summary: Dict[str, Any],
    nonseq: Optional[Dict[str, Any]],
    sequential: Optional[Dict[str, Any]],
    generation: Optional[Dict[str, Any]],
    plots: Optional[Dict[str, Any]],
) -> None:
    nonseq_acceptance = (nonseq or {}).get("acceptance", {})
    seq_acceptance = (sequential or {}).get("acceptance", {})
    decision = "C. C fails MedMKEB. Do not scale; inspect data schema, prompt template, model-known filtering, and target/reference construction."
    if nonseq_acceptance.get("status") == "pass" and seq_acceptance.get("status") == "pass":
        decision = "A. C passes MedMKEB nonseq and sequential gates. Next: expand to 50 MedMKEB edits or add another public Med-VQA dataset."
    elif nonseq_acceptance.get("status") == "pass":
        decision = "B. C passes nonseq but sequential is partial or unavailable. Analyze retention/locality before scaling."
    lines = [
        "# Final MedMKEB Model-Known 20 Report",
        "",
        "## Scope",
        "",
        "- Dataset: bounded public MedMKEB/VLKEB-style subset.",
        "- Primary method: `C_engram_projected_tiny_lora`.",
        "- Baselines: `A_no_edit`, `B_tiny_lora_replacement`.",
        "- Generation diagnostics are secondary; pass/fail is NLL/logprob based.",
        "- No medical or clinical efficacy claim is made.",
        "",
        "## Data Audit",
        "",
        f"- Audit status: `{(audit_status or {}).get('status', 'bundle_only')}`",
        f"- Selected records: `{selected_summary.get('selected')}`",
        f"- Unique selected records: `{selected_summary.get('unique_selected')}`",
        f"- Images resolved: `{selected_summary.get('images_resolved')}`",
        f"- Finite base metrics: `{selected_summary.get('finite_base_metrics')}`",
        f"- Positional matching used: `{selected_summary.get('positional_matching_used')}`",
        "",
        "## Non-Sequential",
        "",
        f"- Status: `{(nonseq or {}).get('status')}`",
        f"- C acceptance: `{nonseq_acceptance.get('status')}`",
    ]
    for row in (nonseq or {}).get("aggregate_rows", []):
        lines.append(
            "- {method}: mean_new={new}, mean_ref_abs={ref}, positive_new={pos}, locality_damage={loc}, rollback={roll}, match={match}, nan={nan}".format(
                method=row.get("method"),
                new=_format(row.get("mean_new_answer_nll_decrease")),
                ref=_format(row.get("mean_locality_reference_delta_abs")),
                pos=row.get("positive_new_answer_edits"),
                loc=row.get("locality_damage_edits"),
                roll=_format(row.get("rollback_pass_rate")),
                match=_format(row.get("record_id_match_rate")),
                nan=row.get("nan_inf_count"),
            )
        )
    lines.extend(["", "## Sequential", ""])
    if sequential:
        lines.append(f"- Status: `{sequential.get('status')}`")
        lines.append(f"- C acceptance: `{seq_acceptance.get('status')}`")
        final_rows = [row for row in sequential.get("summary_rows", []) if row.get("final_step")]
        for row in final_rows:
            lines.append(
                "- {method}: final_mean_new={new}, final_ref_abs={ref}, positive_new={pos}, locality_damage={loc}".format(
                    method=row.get("method"),
                    new=_format(row.get("mean_new_answer_nll_decrease_edited_records")),
                    ref=_format(row.get("mean_reference_delta_abs_all_records")),
                    pos=row.get("positive_new_answer_edits"),
                    loc=row.get("locality_damage_records"),
                )
            )
    else:
        lines.append("- Status: `skipped`")
        lines.append("- Reason: non-sequential C gate did not pass.")
    lines.extend(
        [
            "",
            "## Optional E / CURE",
            "",
            "- `E_rescued_cure_dual_projected_tiny_lora` was skipped in this run because the requested primary gate is A/B/C and E is optional.",
            "- CURE tests are reported in `test_logs/test_status.json`; C is not blocked by CURE-only test failure.",
            "",
            "## Generation Diagnostics",
            "",
            f"- Status: `{(generation or {}).get('status', 'skipped')}`",
            "- These diagnostics are not used as primary pass/fail evidence.",
            "",
            "## Plots",
            "",
            f"- Status: `{(plots or {}).get('status', 'skipped')}`",
            "",
            "## Limitations",
            "",
            "- Bounded 20-edit model-known subset only.",
            "- No full MedMKEB benchmark yet.",
            "- No clinical or medical efficacy claim.",
            "- Candidate selection is model-NLL based and depends on the exact LLaVA-Med checkpoint and prompt path.",
            "",
            "## Decision",
            "",
            decision,
            "",
        ]
    )
    (out_dir / "modelknown_20" / "FINAL_MEDMKEB_MODELKNOWN_20_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _package_hygiene(out_dir: Path, *, remove_runtime_bank: bool = True) -> Dict[str, Any]:
    if remove_runtime_bank:
        for path in [out_dir / "modelknown_20" / "projector_bank", out_dir / "projector_bank"]:
            if path.exists():
                shutil.rmtree(path)
    for path in list(out_dir.rglob("._*")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    banned_suffixes = {".pt", ".pth", ".bin", ".pyc"}
    banned_names = {"__pycache__", ".DS_Store"}
    banned_prefixes = ("._",)
    hits: List[str] = []
    for path in out_dir.rglob("*"):
        name = path.name
        if name in banned_names or name.startswith(banned_prefixes) or path.suffix in banned_suffixes:
            hits.append(str(path))
    payload = {
        "status": "pass" if not hits else "fail",
        "checked_root": str(out_dir),
        "banned_hits": hits,
        "runtime_projector_bank_removed": remove_runtime_bank,
        "sync_exclude_policy": [".pt", ".pth", ".bin", "projector_bank tensor files", "HF cache", "CUDA cache", "__pycache__", ".pyc", ".DS_Store", "._*"],
    }
    _json_dump(out_dir / "PACKAGE_HYGIENE_REPORT.md.json", payload)
    lines = ["# Package Hygiene Report", "", f"- Status: `{payload['status']}`", f"- Checked root: `{out_dir}`", f"- Runtime projector bank removed: `{remove_runtime_bank}`", ""]
    if hits:
        lines.extend(["## Banned Hits", ""])
        lines.extend(f"- `{item}`" for item in hits)
    else:
        lines.append("No banned files were found under the output root after runtime cleanup.")
    lines.append("")
    (out_dir / "PACKAGE_HYGIENE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    for path in list(out_dir.rglob("._*")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return payload


def _run_remote_gpu_mode(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    test_status = _write_tests(out_dir, run_tests=not args.skip_tests)
    input_records = Path(args.input_records)
    image_root = Path(args.image_root)
    preflight = _write_preflight(out_dir, hparams_path=Path(args.hparams), input_records=input_records, image_root=image_root, test_status=test_status)
    if test_status.get("engram_tests_pass") is False:
        raise RuntimeError("ENGRAM tests failed; stopping before editing.")
    if preflight.get("status") != "pass":
        raise RuntimeError(f"Preflight failed: {preflight}")

    heavy = _heavy_imports()
    torch = heavy["torch"]
    MultimodalEditor = heavy["MultimodalEditor"]
    EngramMultimodalHparams = heavy["EngramMultimodalHparams"]
    select_linear_layers = heavy["select_linear_layers"]
    _configure_hparams = heavy["_configure_hparams"]
    _evaluate_current = heavy["_evaluate_current"]
    _extract_projector_bank = heavy["_extract_projector_bank"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    candidate_records = _load_bundle_records(input_records)
    hparams = EngramMultimodalHparams.from_hparams(str(args.hparams))
    bank_dir = out_dir / "modelknown_20" / "projector_bank"
    _configure_hparams(
        hparams,
        image_root=image_root,
        bank_dir=bank_dir,
        device=str(args.device),
        edit_mode="erase",
    )
    hparams.replacement_beta = float(args.beta)
    hparams.replacement_lambda_ref = 0.0
    hparams.lora_rank = int(args.lora_rank)
    hparams.lora_steps = int(args.lora_steps)
    hparams.lora_lr = float(args.lora_lr)
    hparams.token_scope = "all"
    _json_dump(
        out_dir / "modelknown_20" / "effective_config.json",
        {
            "beta": args.beta,
            "token_scope": "all",
            "selected_modules": EXPECTED_MODULES,
            "lora_rank": args.lora_rank,
            "lora_steps": args.lora_steps,
            "lora_lr": args.lora_lr,
            "skip_generation": args.skip_generation,
        },
    )
    editor = MultimodalEditor.from_hparams(hparams)
    selected_modules = [layer.name for layer in select_linear_layers(editor.model, hparams)]
    selected_status = {
        "status": "pass" if set(selected_modules) == set(EXPECTED_MODULES) and len(selected_modules) == len(EXPECTED_MODULES) else "fail",
        "selected_module_names": selected_modules,
        "expected_module_names": EXPECTED_MODULES,
    }
    _json_dump(out_dir / "modelknown_20" / "selected_modules_preflight.json", selected_status)
    if selected_status["status"] != "pass":
        raise RuntimeError(f"Selected modules do not match expected q/k/gate set: {selected_status}")

    baselines: Dict[str, Dict[str, Any]] = {}
    for record in candidate_records:
        baselines[str(record["id"])] = _evaluate_current(
            editor.model,
            record,
            image_root,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=None,
            skip_generation=True,
        )
    _json_dump(out_dir / "modelknown_20" / "baseline_metrics_all_candidates.json", baselines)
    selected_records, candidate_rows, selected_summary = _select_modelknown_records(
        records=candidate_records,
        baselines=baselines,
        image_root=image_root,
        out_dir=out_dir,
        count=20,
    )
    selected_data_path = out_dir / "modelknown_20" / "medmkeb_modelknown_20_vlkeb.json"
    _json_dump(selected_data_path, selected_records)
    selected_baselines = {str(record["id"]): baselines[str(record["id"])] for record in selected_records}
    _json_dump(out_dir / "modelknown_20" / "baseline_metrics.json", selected_baselines)

    projector_extract = _extract_projector_bank(editor, hparams, selected_data_path, selected_records, bank_dir)
    _json_dump(out_dir / "modelknown_20" / "projector_extraction_summary.json", projector_extract)
    match_rate = 1.0 if (projector_extract.get("edit_record_matching") or {}).get("mode") == "record_id" else 0.0
    record_id_preflight = {
        "status": "pass" if match_rate == 1.0 and len(selected_records) == 20 else "fail",
        "record_id_match_rate": match_rate,
        "selected_record_count": len(selected_records),
        "unique_record_count": len({record["id"] for record in selected_records}),
        "positional_matching_used": False,
        "positional_matching_refused_by_default": True,
        "edit_record_matching": projector_extract.get("edit_record_matching"),
    }
    _json_dump(out_dir / "modelknown_20" / "record_id_preflight.json", record_id_preflight)
    if record_id_preflight["status"] != "pass":
        raise RuntimeError(f"Record-id preflight failed: {record_id_preflight}")

    nonseq, final_nonseq_factors = _run_nonseq(
        model=editor.model,
        records=selected_records,
        image_root=image_root,
        baselines=selected_baselines,
        projector_bank_dir=bank_dir,
        module_names=EXPECTED_MODULES,
        hparams=hparams,
        out_dir=out_dir,
        beta=float(args.beta),
        rollback_tolerance=float(args.rollback_tolerance),
        locality_threshold=float(args.locality_damage_threshold),
        max_new_tokens=int(args.max_new_tokens),
        skip_generation=True,
    )

    sequential: Optional[Dict[str, Any]] = None
    final_specs: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    generation: Optional[Dict[str, Any]] = None
    if nonseq.get("acceptance", {}).get("status") == "pass":
        sequential, final_specs = _run_sequential(
            model=editor.model,
            records=selected_records,
            image_root=image_root,
            baselines=selected_baselines,
            projector_bank_dir=bank_dir,
            module_names=EXPECTED_MODULES,
            hparams=hparams,
            out_dir=out_dir,
            beta=float(args.beta),
            rollback_tolerance=float(args.rollback_tolerance),
            locality_threshold=float(args.locality_damage_threshold),
            max_new_tokens=int(args.max_new_tokens),
            skip_generation=True,
        )
        if not args.skip_generation_diagnostics:
            generation = _generation_diagnostics(
                model=editor.model,
                records=selected_records,
                image_root=image_root,
                final_specs=final_specs,
                out_dir=out_dir,
                beta=float(args.beta),
                max_new_tokens=int(args.max_new_tokens),
            )
    else:
        _json_dump(
            out_dir / "modelknown_20" / "sequential" / "sequential_skipped.json",
            {"status": "skipped", "reason": "C failed nonseq acceptance", "nonseq_acceptance": nonseq.get("acceptance")},
        )

    plots = _plot_optional(out_dir)
    _write_final_report(
        out_dir=out_dir,
        audit_status=None,
        selected_summary=selected_summary,
        nonseq=nonseq,
        sequential=sequential,
        generation=generation,
        plots=plots,
    )
    _package_hygiene(out_dir)
    return 0


def _run_prepare_bundle(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    _ensure_layout(out_dir)
    _write_git_outputs(out_dir)
    _write_env_report(out_dir)
    data_root = Path(args.data_root)
    audit = _audit_data(data_root, out_dir)
    _schema_adapter_report(out_dir)
    source_file = Path(args.source_file) if args.source_file else None
    manifest = _build_candidate_bundle(
        data_root=data_root,
        out_dir=out_dir,
        bundle_root=Path(args.bundle_output),
        source_file=source_file,
        candidate_pool_size=int(args.candidate_pool_size),
        seed=int(args.seed),
    )
    status = {"audit": audit, "bundle": manifest}
    _json_dump(out_dir / "prepare_bundle_status.json", status)
    _package_hygiene(out_dir, remove_runtime_bank=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded MedMKEB model-known ENGRAM-projected tiny-LoRA validation.")
    parser.add_argument("--mode", choices=["prepare-bundle", "run-gpu"], default="prepare-bundle")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--source-file")
    parser.add_argument("--bundle-output", default="/tmp/medmkeb_engram_projected_lora_bundle")
    parser.add_argument("--candidate-pool-size", type=int, default=60)
    parser.add_argument("--input-records", help="Bundle records.json for --mode run-gpu.")
    parser.add_argument("--image-root", help="Image root/bundle root for --mode run-gpu.")
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-steps", type=int, default=20)
    parser.add_argument("--lora-lr", type=float, default=1.0e-4)
    parser.add_argument("--rollback-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--locality-damage-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-generation", action="store_true", default=True)
    parser.add_argument("--skip-generation-diagnostics", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = time.time()
    if args.mode == "prepare-bundle":
        code = _run_prepare_bundle(args)
    else:
        if not args.input_records or not args.image_root:
            parser.error("--mode run-gpu requires --input-records and --image-root")
        code = _run_remote_gpu_mode(args)
    runtime_path = Path(args.output_dir) / "runtime.json"
    _json_dump(runtime_path, {"runtime_sec": time.time() - start, "mode": args.mode})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
