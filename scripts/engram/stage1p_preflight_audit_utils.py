"""Pure helpers for the fixed-ten ENGRAM V2 data/model-known audit."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer


CLASS_MODEL_KNOWN = "PAIRING_VALID_MODEL_KNOWN"
CLASS_MODEL_UNKNOWN = "PAIRING_VALID_MODEL_UNKNOWN"
CLASS_PAIRING_MISMATCH = "DATA_FIELD_PAIRING_MISMATCH"
CLASS_AMBIGUOUS_MAPPING = "AMBIGUOUS_ANSWER_MAPPING"


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def raw_row_for_id(rows: Sequence[Mapping[str, Any]], record_id: str) -> tuple[int, Mapping[str, Any]]:
    matches = [(index, row) for index, row in enumerate(rows) if str(row.get("id")) == str(record_id)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one raw row for {record_id}, found {len(matches)}")
    return matches[0]


def verify_raw_processed_pairing(raw: Mapping[str, Any], processed: Mapping[str, Any]) -> dict[str, Any]:
    scalar_fields = (
        "id", "src", "pred", "alt", "rephrase", "loc", "loc_ans", "m_loc_q", "m_loc_a",
        "clinical_VQA_task", "department", "perceptual_granularity", "modality", "original_task", "port_new",
    )
    checks = {field: raw.get(field) == processed.get(field) for field in scalar_fields}
    for field in ("image", "image_rephrase", "m_loc"):
        raw_name = Path(str(raw.get(field, ""))).name
        processed_name = Path(str(processed.get(field, ""))).name
        checks[f"{field}_source_basename_propagated"] = bool(raw_name and processed_name.endswith(raw_name))
    return {"checks": checks, "passed": all(checks.values()), "mismatches": [key for key, value in checks.items() if not value]}


def verify_image_hash_propagation(raw_sha256: str, processed_sha256: str) -> bool:
    return bool(raw_sha256 and raw_sha256 == processed_sha256)


def resolve_answer_mapping(record: Mapping[str, Any]) -> dict[str, Any]:
    old = str(record.get("pred") or "").strip()
    target = str(record.get("alt") or "").strip()
    options = record.get("answer_options") or record.get("options") or []
    answer_index = record.get("answer_index")
    if not old or not target:
        return {"resolved": False, "ambiguous": True, "reason": "missing canonical old or edited target", "options": options}
    if options and answer_index is not None:
        try:
            mapped = str(options[int(answer_index)]).strip()
        except (IndexError, TypeError, ValueError):
            return {"resolved": False, "ambiguous": True, "reason": "invalid answer-option index", "options": options}
        return {
            "resolved": mapped == old,
            "ambiguous": mapped != old,
            "reason": "indexed option equals canonical old" if mapped == old else "indexed option conflicts with canonical old",
            "options": list(options),
            "mapped_option": mapped,
        }
    return {"resolved": True, "ambiguous": False, "reason": "canonical old answer stored directly; no options in source row", "options": list(options)}


def token_boundary_contains(output: str, answer: str) -> bool:
    out_tokens = normalize_medical_answer(output).split()
    answer_tokens = normalize_medical_answer(answer).split()
    if not answer_tokens:
        return False
    width = len(answer_tokens)
    return any(out_tokens[index : index + width] == answer_tokens for index in range(max(0, len(out_tokens) - width + 1)))


def preregistered_aliases(record: Mapping[str, Any]) -> list[str]:
    values = record.get("old_answer_aliases") or record.get("accepted_old_answers") or []
    if isinstance(values, str):
        values = [values]
    return [str(item).strip() for item in values if str(item).strip()]


def answer_match_report(output: str, old_answer: str, aliases: Sequence[str]) -> dict[str, Any]:
    raw_output = str(output).strip()
    raw_old = str(old_answer).strip()
    normalized_output = normalize_medical_answer(raw_output)
    normalized_old = normalize_medical_answer(raw_old)
    alias_values = [str(item).strip() for item in aliases if str(item).strip()]
    return {
        "raw_exact_match": bool(raw_output and raw_output == raw_old),
        "normalized_exact_match": bool(normalized_output and normalized_output == normalized_old),
        "preregistered_alias_match": any(
            normalized_output == normalize_medical_answer(alias) or token_boundary_contains(raw_output, alias)
            for alias in alias_values
        ),
        "token_boundary_contains": token_boundary_contains(raw_output, raw_old),
        "normalized_output": normalized_output,
        "normalized_old_answer": normalized_old,
        "aliases": alias_values,
    }


def classify_record(*, pairing_valid: bool, answer_mapping_ambiguous: bool, view_matches: Sequence[Mapping[str, Any]]) -> str:
    if not pairing_valid:
        return CLASS_PAIRING_MISMATCH
    if answer_mapping_ambiguous:
        return CLASS_AMBIGUOUS_MAPPING
    known = any(
        bool(row.get(field))
        for row in view_matches
        for field in ("raw_exact_match", "normalized_exact_match", "preregistered_alias_match", "token_boundary_contains")
    )
    return CLASS_MODEL_KNOWN if known else CLASS_MODEL_UNKNOWN


def select_first_model_known(order: Sequence[str], classifications: Mapping[str, str]) -> str | None:
    return next((str(record_id) for record_id in order if classifications.get(str(record_id)) == CLASS_MODEL_KNOWN), None)


def assert_bank_hash_unchanged(before: str, after: str) -> None:
    if not before or before != after:
        raise RuntimeError("Frozen ENGRAM V2 bank hash changed during Stage-1P audit")


def target_absent_from_prompt(prompt: str, target: str) -> bool:
    normalized_prompt = normalize_medical_answer(prompt)
    normalized_target = normalize_medical_answer(target)
    if not normalized_target:
        return False
    pattern = r"(?<!\w)" + re.escape(normalized_target) + r"(?!\w)"
    return re.search(pattern, normalized_prompt) is None
