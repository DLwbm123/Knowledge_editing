#!/usr/bin/env python3
"""Fixed-ten data-integrity and model-known audit for ENGRAM V2 Stage-1P."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts.engram.stage0abc_diagnostic_utils import create_new_output_dir
from scripts.engram.stage1p_preflight_audit_utils import (
    CLASS_AMBIGUOUS_MAPPING,
    CLASS_MODEL_KNOWN,
    CLASS_MODEL_UNKNOWN,
    CLASS_PAIRING_MISMATCH,
    answer_match_report,
    assert_bank_hash_unchanged,
    canonical_json_hash,
    classify_record,
    preregistered_aliases,
    raw_row_for_id,
    resolve_answer_mapping,
    select_first_model_known,
    target_absent_from_prompt,
    verify_image_hash_propagation,
    verify_raw_processed_pairing,
)


ORDER = ["953", "1293", "1592", "2174", "1628", "942", "1382", "1333", "671", "1343"]
STARTING_COMMIT = "7ff7a63d3f370d53ed5ebba063b3dac29778954f"
PROTOCOL = "ENGRAM_V2_STAGE1P_DATA_AND_MODEL_KNOWN_AUDIT_V1"
SHORT_INSTRUCTION = "Answer with only the final medical answer. Do not provide an explanation."
CAP = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("raw-ledger", "audit"))
    parser.add_argument("--raw-source", type=Path)
    parser.add_argument("--raw-image-root", type=Path)
    parser.add_argument("--out-ledger", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--raw-ledger", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--physical-gpu", default=2, type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(value.rstrip() + "\n")


def raw_ledger(args: argparse.Namespace) -> None:
    if not all((args.raw_source, args.raw_image_root, args.out_ledger, args.contact_sheet)):
        raise ValueError("raw-ledger requires --raw-source, --raw-image-root, --out-ledger, and --contact-sheet")
    rows = json.loads(args.raw_source.read_text())
    ledger = {
        "source_dataset_name": "MedMKEB",
        "source_split_file": args.raw_source.name,
        "source_file_sha256": sha256_file(args.raw_source),
        "source_row_count": len(rows),
        "fixed_order": ORDER,
        "records": [],
    }
    previews = []
    for order_index, record_id in enumerate(ORDER):
        source_index, row = raw_row_for_id(rows, record_id)
        image_path = args.raw_image_root / str(row["image"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            dimensions = [int(image.width), int(image.height)]
            preview = image.convert("RGB")
            preview.thumbnail((320, 240))
            previews.append((record_id, str(row["pred"]), preview.copy()))
        aliases = preregistered_aliases(row)
        ledger["records"].append({
            "fixed_order_index": order_index,
            "record_id": record_id,
            "raw_source_index": source_index,
            "raw_source_row_hash": canonical_json_hash(row),
            "raw_row": row,
            "raw_image_path": str(image_path),
            "raw_image_sha256": sha256_file(image_path),
            "raw_image_dimensions": dimensions,
            "original_answer_options": row.get("answer_options") or row.get("options") or [],
            "preregistered_old_answer_aliases": aliases,
        })
    write_json(args.out_ledger, ledger)

    if args.contact_sheet.exists():
        raise FileExistsError(args.contact_sheet)
    canvas = Image.new("RGB", (700, 5 * 290), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (record_id, old_answer, image) in enumerate(previews):
        column, row = index % 2, index // 2
        x, y = column * 350, row * 290
        canvas.paste(image, (x + 10, y + 25))
        draw.text((x + 10, y + 5), f"{index + 1}. record {record_id}", fill="black", font=font)
        label = old_answer if len(old_answer) <= 52 else old_answer[:49] + "..."
        draw.text((x + 10, y + 265), label, fill="black", font=font)
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.contact_sheet, format="PNG")
    print(json.dumps({"records": len(ledger["records"]), "ledger": str(args.out_ledger), "contact_sheet": str(args.contact_sheet)}, indent=2))


def stage0_expected(matrix_dir: Path, record_id: str, state_id: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    records = [json.loads(line) for line in (matrix_dir / "records.jsonl").read_text().splitlines() if line.strip()]
    generations = [json.loads(line) for line in (matrix_dir / "generation_outputs.jsonl").read_text().splitlines() if line.strip()]
    record = next(row for row in records if row["record_id"] == record_id and row["state_id"] == state_id and row["view"] == "target")
    generation = next(row for row in generations if row["record_id"] == record_id and row["state_id"] == state_id and row["view"] == "target")
    return record, generation


def cell_generation(model: Any, canonical: Any, eos: list[int]) -> Dict[str, Any]:
    from scripts.engram.run_engram_v2_stage0abc_diagnostics import hf_cached_greedy_trace
    from scripts.engram.stage0_generation_audit_utils import manual_greedy_trace

    manual = manual_greedy_trace(model, canonical, CAP, eos, top_k=5)
    hf = hf_cached_greedy_trace(model, canonical, CAP)
    if manual["token_ids"] != hf["token_ids"]:
        raise RuntimeError("Manual/HF deterministic trajectories differ in Stage-1P")
    if manual["cap_hit"] or hf["cap_hit"]:
        raise RuntimeError("Stage-1P generation hit the uniform 128-token cap")
    return {
        "token_ids": manual["token_ids"],
        "output": manual["raw_output"],
        "stop_reason": manual["stop_reason"],
        "manual_hf_equal": True,
        "cap_hit": False,
    }


def csv_write(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> None:
    if not all((args.raw_ledger, args.contact_sheet, args.out_dir)):
        raise ValueError("audit requires --raw-ledger, --contact-sheet, and --out-dir")
    out_dir = create_new_output_dir(args.out_dir)
    write_text(out_dir / "exact_command_log.txt", " ".join(sys.argv))
    contact_target = out_dir / "fixed_ten_contact_sheet.png"
    if contact_target.exists():
        raise FileExistsError(contact_target)
    shutil.copyfile(args.contact_sheet, contact_target)

    from scripts.engram.run_engram_v2_stage0_generation_audit import (
        DATASET_PATH,
        MODULE_NAME,
        apply_prefix,
        bank_manifest,
        clone_sample_with_target,
        eos_ids,
        load_model_views_bank,
        state_weight_hash,
    )
    from scripts.engram.run_engram_v2_stage0abc_diagnostics import short_answer_sample
    from scripts.engram.run_engram_continual_v2 import build_unrelated
    from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, tensor_sha256

    stage0abc = ROOT / "outputs/engram_v2_stage0_generation_audit/20260810_stage0abc_margin_feasibility_v1"
    matrix_dir = stage0abc / "fixed_matrix_uniform_128"
    previous_stage1 = ROOT / "outputs/engram_v2_stage1_behavioral_margin_one_edit/20260810_record953_v1"
    raw = json.loads(args.raw_ledger.read_text())
    raw_by_id = {row["record_id"]: row for row in raw["records"]}
    processed_rows = json.loads(DATASET_PATH.read_text())
    processed_by_id = {str(row["id"]): row for row in processed_rows}
    bank_before = bank_manifest()
    model, views, bank, records = load_model_views_bank(args.physical_gpu)
    image_root = DATASET_PATH.parent

    integrity_rows: list[Dict[str, Any]] = []
    status_rows: list[Dict[str, Any]] = []
    combined_ledger: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "raw_source": {key: raw[key] for key in ("source_dataset_name", "source_split_file", "source_file_sha256", "source_row_count", "fixed_order")},
        "records": [],
    }
    for order_index, record_id in enumerate(ORDER):
        prefix = order_index
        state_id = f"S{prefix}"
        apply_prefix(model, bank, prefix)
        record = records[record_id]
        processed = processed_by_id[record_id]
        raw_entry = raw_by_id[record_id]
        raw_row = raw_entry["raw_row"]
        pairing = verify_raw_processed_pairing(raw_row, processed)
        answer_mapping = resolve_answer_mapping(raw_row)

        unrestricted_sample = build_unrelated(model, record, image_root)
        unrestricted = build_canonical_inputs(model, unrestricted_sample)
        short_target_sample = short_answer_sample(model, views[record_id]["target"], record)
        short_sample = clone_sample_with_target(short_target_sample, str(record["pred"]), model)
        short = build_canonical_inputs(model, short_sample)
        if SHORT_INSTRUCTION not in short.prompt_text:
            raise RuntimeError("Fixed short-answer instruction changed")
        if not target_absent_from_prompt(unrestricted.prompt_text, str(record["alt"])) or not target_absent_from_prompt(short.prompt_text, str(record["alt"])):
            raise RuntimeError(f"Edited-target leakage in record {record_id} generation prompt")

        image_path = Path(unrestricted_sample["image_path"][0])
        processed_sha = sha256_file(image_path)
        with Image.open(image_path) as image:
            processed_dimensions = [int(image.width), int(image.height)]
        image_hash_ok = verify_image_hash_propagation(raw_entry["raw_image_sha256"], processed_sha)
        dimensions_ok = processed_dimensions == raw_entry["raw_image_dimensions"]
        expected_record, expected_generation = stage0_expected(matrix_dir, record_id, state_id)
        stage0_fields_ok = bool(
            unrestricted.prompt_hash == expected_record["prompt_hash"]
            and unrestricted.pixel_hash == expected_record["pixel_hash"]
        )

        unrestricted_generation = cell_generation(model, unrestricted, eos_ids(model))
        short_generation = cell_generation(model, short, eos_ids(model))
        stage0_generation_equal = unrestricted_generation["token_ids"] == expected_generation["manual"]["token_ids"]
        aliases = raw_entry["preregistered_old_answer_aliases"]
        unrestricted_match = answer_match_report(unrestricted_generation["output"], str(record["pred"]), aliases)
        short_match = answer_match_report(short_generation["output"], str(record["pred"]), aliases)
        pairing_valid = bool(pairing["passed"] and image_hash_ok and dimensions_ok and stage0_fields_ok and stage0_generation_equal)
        classification = classify_record(
            pairing_valid=pairing_valid,
            answer_mapping_ambiguous=bool(answer_mapping["ambiguous"]),
            view_matches=[unrestricted_match, short_match],
        )
        stage1_fields = None
        if record_id == "953":
            prior_manifest = json.loads((previous_stage1 / "run_manifest.json").read_text())
            stage1_fields = {
                "record_id": prior_manifest["preflight"]["s0_target"]["record_id"],
                "state_id": prior_manifest["preflight"]["s0_target"]["state_id"],
                "prompt_hash": prior_manifest["preflight"]["s0_target"]["prompt_hash"],
                "pixel_hash": prior_manifest["preflight"]["s0_target"]["pixel_hash"],
                "output": prior_manifest["preflight"]["s0_target"]["no_cache"]["raw_output"],
            }

        integrity_rows.append({
            "fixed_order_index": order_index,
            "record_id": record_id,
            "raw_source_dataset": raw["source_dataset_name"],
            "raw_source_file": raw["source_split_file"],
            "raw_source_index": raw_entry["raw_source_index"],
            "raw_source_row_hash": raw_entry["raw_source_row_hash"],
            "raw_image_path": raw_entry["raw_image_path"],
            "processed_image_path": str(image_path),
            "raw_image_sha256": raw_entry["raw_image_sha256"],
            "processed_image_sha256": processed_sha,
            "image_hash_propagated": image_hash_ok,
            "raw_dimensions": "x".join(map(str, raw_entry["raw_image_dimensions"])),
            "processed_dimensions": "x".join(map(str, processed_dimensions)),
            "processor_tensor_hash": tensor_sha256(unrestricted.image),
            "question": record["src"],
            "answer_options": json.dumps(raw_entry["original_answer_options"], ensure_ascii=False),
            "old_answer": record["pred"],
            "edited_target": record["alt"],
            "textual_generality_question": record["rephrase"],
            "textual_generality_image": record["image"],
            "visual_generality_image": record["image_rephrase"],
            "paired_text_locality_question": record["loc"],
            "paired_text_locality_answer": record["loc_ans"],
            "paired_visual_locality_image": record["m_loc"],
            "paired_visual_locality_question": record["m_loc_q"],
            "paired_visual_locality_answer": record["m_loc_a"],
            "pairing_valid": pairing_valid,
            "answer_mapping_status": answer_mapping["reason"],
            "classification": classification,
        })
        status_rows.append({
            "fixed_order_index": order_index,
            "record_id": record_id,
            "pre_edit_state": state_id,
            "classification": classification,
            "old_answer": record["pred"],
            "preregistered_aliases": json.dumps(aliases, ensure_ascii=False),
            "unrestricted_output": unrestricted_generation["output"],
            "unrestricted_raw_exact": unrestricted_match["raw_exact_match"],
            "unrestricted_normalized_exact": unrestricted_match["normalized_exact_match"],
            "unrestricted_alias_match": unrestricted_match["preregistered_alias_match"],
            "unrestricted_token_boundary_contains": unrestricted_match["token_boundary_contains"],
            "short_answer_output": short_generation["output"],
            "short_raw_exact": short_match["raw_exact_match"],
            "short_normalized_exact": short_match["normalized_exact_match"],
            "short_alias_match": short_match["preregistered_alias_match"],
            "short_token_boundary_contains": short_match["token_boundary_contains"],
            "unrestricted_stop_reason": unrestricted_generation["stop_reason"],
            "short_stop_reason": short_generation["stop_reason"],
            "manual_hf_parity": unrestricted_generation["manual_hf_equal"] and short_generation["manual_hf_equal"],
            "cap_hit": False,
        })
        combined_ledger["records"].append({
            "record_id": record_id,
            "fixed_order_index": order_index,
            "raw_source_index": raw_entry["raw_source_index"],
            "raw_source_row_hash": raw_entry["raw_source_row_hash"],
            "raw_row": raw_row,
            "processed_row": processed,
            "raw_image": {"path": raw_entry["raw_image_path"], "sha256": raw_entry["raw_image_sha256"], "dimensions": raw_entry["raw_image_dimensions"]},
            "processed_image": {"path": str(image_path), "sha256": processed_sha, "dimensions": processed_dimensions, "processor_tensor_hash": tensor_sha256(unrestricted.image)},
            "pairing": {**pairing, "image_hash_propagated": image_hash_ok, "dimensions_equal": dimensions_ok, "stage0_fields_equal": stage0_fields_ok, "stage0_generation_equal": stage0_generation_equal},
            "answer_mapping": answer_mapping,
            "runner_loaded_fields": {
                "image_path": unrestricted_sample["image_path"],
                "prompt": unrestricted_sample["prompt"],
                "target_old_answer": unrestricted_sample["target"],
                "edited_target_from_target_view": views[record_id]["target"]["target"],
            },
            "stage0_used_fields": {"record_id": record_id, "state_id": state_id, "prompt_hash": expected_record["prompt_hash"], "pixel_hash": expected_record["pixel_hash"], "output": expected_record["greedy_output"]},
            "stage1_used_fields": stage1_fields,
            "unrestricted": {"generation": unrestricted_generation, "match": unrestricted_match, "prompt_hash": unrestricted.prompt_hash, "pixel_hash": unrestricted.pixel_hash},
            "short_answer": {"generation": short_generation, "match": short_match, "prompt_hash": short.prompt_hash, "pixel_hash": short.pixel_hash},
            "classification": classification,
        })

    apply_prefix(model, bank, 0)
    bank_after = bank_manifest()
    assert_bank_hash_unchanged(bank_before["sha256"], bank_after["sha256"])
    classifications = {row["record_id"]: row["classification"] for row in status_rows}
    first_model_known = select_first_model_known(ORDER, classifications)
    known_count = sum(row["classification"] == CLASS_MODEL_KNOWN for row in status_rows)
    model_unknown_count = sum(row["classification"] == CLASS_MODEL_UNKNOWN for row in status_rows)
    invalid = [row for row in status_rows if row["classification"] in (CLASS_PAIRING_MISMATCH, CLASS_AMBIGUOUS_MAPPING)]
    fixed_sequence_valid = not invalid
    record953_class = classifications["953"]
    artifacts_to_regenerate: list[str] = []
    if invalid:
        first_invalid_index = min(int(row["fixed_order_index"]) for row in invalid)
        artifacts_to_regenerate = [
            f"Original ENGRAM V2 bank states S{first_invalid_index + 1} and later",
            "Stage-0 fixed-ten matrix and Stage-0ABC reports",
            "Stage-1 record-953 report if record 953 is at or after the invalid index",
        ]
    natural_branch_permitted = bool(fixed_sequence_valid and first_model_known is not None)
    model_unknown_short_branch_permitted = bool(fixed_sequence_valid and record953_class == CLASS_MODEL_UNKNOWN)
    summary = {
        "protocol": PROTOCOL,
        "starting_commit": STARTING_COMMIT,
        "fixed_order": ORDER,
        "record_953_classification": record953_class,
        "fixed_ten_sequence_semantically_valid": fixed_sequence_valid,
        "model_known_count": known_count,
        "model_unknown_count": model_unknown_count,
        "first_eligible_model_known_record": first_model_known,
        "natural_span_stage1_branch_permitted": natural_branch_permitted,
        "natural_span_branch_label": None if first_model_known else "NATURAL_SPAN_BRANCH_NO_GO_FOR_FIXED_TEN",
        "model_unknown_short_answer_stage1_branch_permitted": model_unknown_short_branch_permitted,
        "artifacts_to_invalidate_and_rerun": artifacts_to_regenerate,
        "optimizer_started": False,
        "candidate_bank_created": False,
        "stage2_started": False,
        "bank_unchanged": bank_before["sha256"] == bank_after["sha256"],
    }
    audit_payload = {"summary": summary, "integrity_rows": integrity_rows, "model_known_rows": status_rows}
    write_json(out_dir / "stage1p_audit.json", audit_payload)
    write_json(out_dir / "source_and_processed_hash_ledger.json", combined_ledger)
    csv_write(out_dir / "fixed_ten_record_integrity.csv", integrity_rows, list(integrity_rows[0]))
    csv_write(out_dir / "fixed_ten_model_known_status.csv", status_rows, list(status_rows[0]))

    def md(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    table = ["| # | ID | Question | Dataset old answer | Edited target | Base unrestricted | Base short-answer | Status |", "|---:|---:|---|---|---|---|---|---|"]
    for integrity, status in zip(integrity_rows, status_rows):
        table.append("| " + " | ".join(md(value) for value in (
            int(integrity["fixed_order_index"]) + 1,
            integrity["record_id"], integrity["question"], integrity["old_answer"], integrity["edited_target"],
            status["unrestricted_output"], status["short_answer_output"], status["classification"],
        )) + " |")
    report = "\n".join([
        "# Stage-1P Data Integrity and Model-Known Audit", "",
        f"- Record 953: `{record953_class}`", f"- Fixed-ten sequence semantically valid: `{fixed_sequence_valid}`",
        f"- Model-known records: `{known_count}/10`", f"- First eligible model-known record: `{first_model_known}`",
        f"- Natural-span Stage-1 branch permitted: `{natural_branch_permitted}`",
        f"- Model-unknown short-answer Stage-1 branch permitted: `{model_unknown_short_branch_permitted}`",
        f"- Frozen bank unchanged: `{summary['bank_unchanged']}`", "", *table,
    ])
    write_text(out_dir / "STAGE1P_DATA_AND_MODEL_KNOWN_AUDIT.md", report)
    decision = "\n".join([
        "# Stage-1P Next Decision", "", "## Verified facts", "",
        f"- Record 953 is `{record953_class}`.",
        f"- The fixed-ten sequence remains semantically valid: `{fixed_sequence_valid}`.",
        f"- Model-known records: `{known_count}/10`.",
        f"- First eligible model-known record: `{first_model_known}`.",
        f"- Original bank hash is unchanged: `{bank_before['sha256']}`.",
        "- No optimizer, candidate bank, or Stage-2 run was started.", "", "## Diagnostic decisions", "",
        f"- Natural-span Stage-1 branch permitted: `{natural_branch_permitted}`.",
        f"- Model-unknown short-answer Stage-1 branch permitted: `{model_unknown_short_branch_permitted}`.",
        f"- Natural-span no-go label: `{summary['natural_span_branch_label']}`.", "", "## Artifact validity", "",
        *( [f"- Regenerate: {item}" for item in artifacts_to_regenerate] if artifacts_to_regenerate else ["- No existing Stage-0 or Stage-1 engineering artifacts are invalidated by data pairing."] ), "",
        "The first eligible natural-span record was selected automatically from the original fixed order; no margin, answer length, or apparent difficulty was used.",
    ])
    write_text(out_dir / "STAGE1P_NEXT_DECISION.md", decision)
    manifest = {
        "protocol": PROTOCOL,
        "starting_commit": STARTING_COMMIT,
        "command": " ".join(sys.argv),
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "uniform_max_new_tokens": CAP,
        "generation": {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0, "target_dependent_stopping": False},
        "summary": summary,
        "bank_manifest_before": bank_before,
        "bank_manifest_after": bank_after,
        "model_s0_hash_after": state_weight_hash(model),
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "size": path.stat().st_size}
            for path in (
                Path(__file__).resolve(),
                ROOT / "scripts/engram/stage1p_preflight_audit_utils.py",
                ROOT / "tests/test_engram_v2_stage1p_preflight_audit.py",
                ROOT / "scripts/engram/run_engram_v2_stage0_generation_audit.py",
                ROOT / "scripts/engram/run_engram_v2_stage0abc_diagnostics.py",
            )
        ],
    }
    manifest["outputs"] = [
        {"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
        for path in sorted(out_dir.iterdir()) if path.is_file() and path.name != "run_manifest.json"
    ]
    write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "raw-ledger":
        raw_ledger(args)
    else:
        audit(args)


if __name__ == "__main__":
    main()
