#!/usr/bin/env python3
"""Freeze and baseline a future-blind medical edit set without edited state."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.blind_set_builder import canonical_hash, collect_observed_ids, freeze_candidates
from methods.liveedit_med.data import EXTERNAL_RECORD_IDS, adapt_record, stable_edit_hash
from methods.liveedit_med.posthoc_validation import file_sha256, sample_to_model_row
from scripts.engram.equivalence_aware_router_utils import router_input_equivalence_key
from scripts.engram.modality_aware_router_utils import normalize_question
from scripts.engram.run_engram_v2_stage0_generation_audit import eos_ids
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, ids_sha256, manual_cached_greedy_trace, tensor_sha256
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import (
    MAX_NEW_TOKENS, capture_prompt, compact_trace, load_clean_model, text_only_canonical,
)
from methods.liveedit_med.llavamed_adapter import resolve_layer21_block


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


@torch.inference_mode()
def eqkey_audit(model: Any, image: str, prompt: str) -> dict[str, Any]:
    rendered = model._conversation_prompt(prompt, None)
    ids = model.tokenizer_image_token(rendered, model.llava_tokenizer, model.IMAGE_TOKEN_INDEX,
                                      return_tensors="pt").unsqueeze(0).to(model.lm_device)
    sample = {"image_path": [image], "prompt": [prompt], "target": [""]}
    pixels = model._image_for_row(sample, 0)
    from PIL import Image
    with Image.open(image) as opened:
        original = [int(opened.width), int(opened.height)]
    image_sizes = [*original, *map(int, pixels.shape)]
    mask = torch.ones_like(ids)
    return {
        "router_input_equivalence_key": router_input_equivalence_key(
            tensor_sha256(pixels), image_sizes, ids[0].tolist(), mask[0].tolist()),
        "raw_image_sha256": file_sha256(Path(image)), "processed_pixel_tensor_sha256": tensor_sha256(pixels),
        "input_ids_sha256": ids_sha256(ids), "normalized_question": normalize_question(prompt),
        "original_image_size_wh": original, "processed_pixel_tensor_shape": list(map(int, pixels.shape)),
    }


def raw_candidates(raw: list[Mapping[str, Any]], image_root: Path, observed: set[str]):
    rows = []
    for record in raw:
        rid = str(record.get("id"))
        required = [image_root / str(record.get(name, "")) for name in ("image", "image_rephrase", "m_loc")]
        fields = all(str(record.get(name, "")).strip() for name in
                     ("src", "alt", "rephrase", "m_loc_q", "m_loc_a", "loc", "loc_ans"))
        if rid not in observed and fields and all(path.is_file() for path in required):
            rows.append((stable_edit_hash(record, image_root), record))
    return sorted(rows, key=lambda item: (item[0], str(item[1]["id"])))


@torch.inference_mode()
def select(args) -> None:
    source = json.loads(args.source_records.read_text())
    raw = json.loads(args.raw_source.read_text())
    observed = collect_observed_ids(source, EXTERNAL_RECORD_IDS)
    candidates = raw_candidates(raw, args.image_root, observed)
    if len(candidates) < args.count:
        raise RuntimeError(f"LIVEEDIT_MED_INSUFFICIENT_UNUSED_ASSETS:{len(candidates)}<{args.count}")
    model, _bank = load_clean_model(args.physical_gpu)
    _name, block = resolve_layer21_block(model)
    audited = []
    raw_by_id = {}
    for selection_hash, raw_row in candidates:
        adapted = adapt_record(raw_row, args.image_root)
        native = adapted["requests"][0]
        audit = eqkey_audit(model, native["image"], native["prompt"])
        audited.append({"record_id": str(raw_row["id"]), "selection_hash": selection_hash,
                        **audit, "source_row_index": raw.index(raw_row)})
        raw_by_id[str(raw_row["id"])] = raw_row
        # Audit a small reserve because processed EqKey collisions are possible.
        if len(audited) >= args.count + 8:
            break
    frozen = freeze_candidates(audited, excluded_ids=observed, count=args.count)
    selected_ids = [row["record_id"] for row in frozen["selected"]]
    adapted = {rid: adapt_record(raw_by_id[rid], args.image_root) for rid in selected_ids}

    # Clean-S0 representation nearest neighbours are selected without any
    # LiveEdit module or checkpoint. The representation itself is not stored.
    reps: dict[str, torch.Tensor] = {}
    rep_hashes = {}
    for rid in selected_ids:
        req = adapted[rid]["requests"][0]
        sample = {"image": req["image"], "prompt": req["prompt"], "target": req["target_new"]}
        canonical = build_canonical_inputs(model, sample_to_model_row(sample))
        _hidden, vision, question = capture_prompt(model, block, canonical)
        rep = torch.cat([vision.float().mean(1), question.float().mean(1)], 1)
        rep = torch.nn.functional.normalize(rep, dim=1)[0].cpu()
        reps[rid] = rep; rep_hashes[rid] = tensor_sha256(rep)

    inputs = []
    for index, rid in enumerate(selected_ids):
        row = adapted[rid]; nxt = adapted[selected_ids[(index + 1) % len(selected_ids)]]
        native = row["requests"][0]
        views = {
            "native": {"image": native["image"], "prompt": native["prompt"], "target": native["target_new"]},
            **{name: row["generality"][name][0] for name in ("textual", "visual", "paired")},
            "same_image_different_question": {"image": native["image"], "prompt": nxt["requests"][0]["prompt"], "target": native["target_new"]},
            "same_question_different_image": {"image": nxt["requests"][0]["image"], "prompt": native["prompt"], "target": native["target_new"]},
            "ordinary_image_locality": row["locality"]["image_or_paired"][0],
            "ordinary_text_locality": row["locality"]["text_only"][0],
        }
        nearest = sorted(((float(torch.dot(reps[rid], reps[other]).item()), other)
                          for other in selected_ids if other != rid), reverse=True)[:5]
        for rank, (similarity, other) in enumerate(nearest, 1):
            other_req = adapted[other]["requests"][0]
            views[f"clean_s0_near_miss_{rank}"] = {"image": other_req["image"], "prompt": other_req["prompt"],
                                                   "target": other_req["target_new"], "near_miss_record_id": other,
                                                   "clean_s0_cosine": similarity}
        for category, entry in views.items():
            inputs.append({"input_id": f"blind:{rid}:{category}", "edit_record_id": rid,
                           "category": category, **entry})
    payload = {
        **frozen, "status": "FROZEN_SELECTION__S0_BASELINE_PENDING", "source_records_sha256": file_sha256(args.source_records),
        "raw_source_sha256": file_sha256(args.raw_source), "image_root": str(args.image_root.resolve()),
        "edited_checkpoint_loaded": False, "clean_s0_only": True, "selection_model_visibility": "CLEAN_S0_ONLY",
        "clean_s0_representation_hashes": rep_hashes, "inputs": inputs,
        "input_count": len(inputs), "generation_config": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128},
    }
    payload["selection_manifest_hash"] = canonical_hash(payload)
    write_new(args.out_dir / "selection_manifest.json", payload)


@torch.inference_mode()
def baseline(args) -> None:
    manifest = json.loads(args.selection_manifest.read_text())
    rows = [row for index, row in enumerate(manifest["inputs"]) if index % args.worker_count == args.worker_index]
    model, _bank = load_clean_model(args.physical_gpu)
    outputs = []
    for row in rows:
        sample = {"image": row.get("image"), "prompt": row["prompt"], "target": row["target"]}
        canonical = (text_only_canonical(model, sample) if sample["image"] is None
                     else build_canonical_inputs(model, sample_to_model_row(sample)))
        trace = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=1)
        compact = compact_trace(trace)
        compact["token_ids_sha256"] = hashlib.sha256(json.dumps(compact["token_ids"], separators=(",", ":")).encode()).hexdigest()
        outputs.append({"input_id": row["input_id"], "clean_s0": compact})
    write_new(args.out, {"worker_index": args.worker_index, "worker_count": args.worker_count,
                         "edited_checkpoint_loaded": False, "clean_s0_only": True, "outputs": outputs})


def finalize(args) -> None:
    selection = json.loads(args.selection_manifest.read_text())
    baseline_by_id = {}
    shards = []
    for path in args.shard:
        value = json.loads(path.read_text()); shards.append({"path": str(path), "sha256": file_sha256(path)})
        if value.get("edited_checkpoint_loaded") is not False:
            raise RuntimeError("LIVEEDIT_MED_BLIND_CHECKPOINT_LEAKAGE")
        for row in value["outputs"]:
            if row["input_id"] in baseline_by_id:
                raise RuntimeError("LIVEEDIT_MED_DUPLICATE_BLIND_BASELINE")
            baseline_by_id[row["input_id"]] = row["clean_s0"]
    expected = [row["input_id"] for row in selection["inputs"]]
    if set(expected) != set(baseline_by_id):
        raise RuntimeError("LIVEEDIT_MED_INCOMPLETE_BLIND_BASELINE")
    final = {**selection, "status": "FUTURE_BLIND_MEDICAL_SET_FROZEN",
             "clean_s0_baselines": [{"input_id": key, **baseline_by_id[key]} for key in expected],
             "baseline_shards": shards, "edited_checkpoint_loaded": False}
    final["manifest_hash"] = canonical_hash(final)
    write_new(args.out, final)


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("select")
    p.add_argument("--source-records", type=Path, required=True); p.add_argument("--raw-source", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, required=True); p.add_argument("--count", type=int, default=16)
    p = sub.add_parser("baseline")
    p.add_argument("--selection-manifest", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, required=True); p.add_argument("--worker-index", type=int, required=True)
    p.add_argument("--worker-count", type=int, required=True)
    p = sub.add_parser("finalize")
    p.add_argument("--selection-manifest", type=Path, required=True); p.add_argument("--shard", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); {"select": select, "baseline": baseline, "finalize": finalize}[args.mode](args)


if __name__ == "__main__":
    main()
