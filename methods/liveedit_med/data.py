"""MedMKEB-to-LiveEdit data adaptation and deterministic isolation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.engram.modality_aware_router_utils import normalize_question, sha256_file


EXTERNAL_RECORD_IDS = {"953", "1293", "1592", "2174", "1628", "942", "1382", "1333", "671", "1343"}


def stable_edit_hash(record: Mapping[str, Any], image_root: Path) -> str:
    rid = str(record["id"]); image = image_root / str(record["image"])
    target_hash = hashlib.sha256(str(record["alt"]).encode()).hexdigest()
    payload = "||".join((rid, sha256_file(image), normalize_question(str(record["src"])), target_hash))
    return hashlib.sha256(payload.encode()).hexdigest()


def adapt_record(record: Mapping[str, Any], image_root: Path) -> dict[str, Any]:
    native = str((image_root / str(record["image"])).resolve())
    alternate = str((image_root / str(record["image_rephrase"])).resolve())
    locality = str((image_root / str(record["m_loc"])).resolve())
    paired = []
    for item in record.get("port_new") or []:
        qa = item.get("Q&A") or {}
        if qa.get("Question"):
            paired.append({"image": alternate, "prompt": str(qa["Question"]), "target": str(record["alt"])})
    if not paired:
        paired.append({"image": alternate, "prompt": str(record["rephrase"]), "target": str(record["alt"])})
    return {
        "record_id": str(record["id"]),
        "requests": [{"image": native, "prompt": str(record["src"]), "target_new": str(record["alt"])}],
        "generality": {
            "textual": [{"image": native, "prompt": str(record["rephrase"]), "target": str(record["alt"])}],
            "visual": [{"image": alternate, "prompt": str(record["src"]), "target": str(record["alt"])}],
            "paired": paired,
        },
        "locality": {
            "image_or_paired": [{"image": locality, "prompt": str(record["m_loc_q"]), "target": str(record["m_loc_a"])}],
            "text_only": [{"image": None, "prompt": str(record["loc"]), "target": str(record["loc_ans"])}],
        },
    }


def deterministic_split(records: Sequence[Mapping[str, Any]], image_root: Path):
    eligible = []
    for record in records:
        rid = str(record.get("id"))
        paths = [image_root / str(record.get(name, "")) for name in ("image", "image_rephrase", "m_loc")]
        fields = all(str(record.get(name, "")).strip() for name in ("src", "alt", "rephrase", "m_loc_q", "m_loc_a", "loc", "loc_ans"))
        if rid not in EXTERNAL_RECORD_IDS and fields and all(path.is_file() for path in paths):
            eligible.append((stable_edit_hash(record, image_root), record))
    eligible.sort(key=lambda item: item[0])
    n = len(eligible)
    if n >= 640: counts = (512, 64, 64)
    elif n >= 320: counts = (256, 32, 32)
    elif n >= 160: counts = (128, 16, 16)
    elif n >= 96: counts = (64, 16, 16)
    else: raise RuntimeError("LIVEEDIT_MED_INSUFFICIENT_DATA")
    selected = eligible[:sum(counts)]; a, b, c = counts
    parts = {"train": selected[:a], "validation": selected[a:a+b], "heldout": selected[a+b:a+b+c]}
    result = {name: [{"selection_hash": h, **adapt_record(record, image_root)} for h, record in rows] for name, rows in parts.items()}
    ids = [{row["record_id"] for row in values} for values in result.values()]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2] or "953" in set.union(*ids):
        raise RuntimeError("LIVEEDIT_MED_SPLIT_LEAKAGE")
    return result
