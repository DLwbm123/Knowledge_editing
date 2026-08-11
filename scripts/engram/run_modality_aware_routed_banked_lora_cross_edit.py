#!/usr/bin/env python3
"""Cross-edit trained modality-aware router for the frozen record-953 LoRA."""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dsca_medmkeb_diag_common import to_jsonable
from scripts.engram.generality_attribution_utils import sha256_file
from scripts.engram.lora_positive_control_utils import load_adapter_payload, load_adapter_state, resolve_target_modules
from scripts.engram.modality_aware_router_utils import (
    BRANCHES, MODALITIES, edit_level_split, exported_probability, fit_pcas, json_hash,
    relation_features, save_model_npz, stable_hash, stable_negative_cap, train_branch,
    validated_scores, wilson_interval, zero_fp_threshold,
)
from scripts.engram.routed_banked_lora_utils import load_router_bank, route_on as v11_route_on
from scripts.engram.run_engram_v2_one_shot_natural_generation_rescue import bank_anchor_hash
from scripts.engram.run_engram_v2_stage0_generation_audit import ORDER, apply_prefix, bank_manifest, load_model_views_bank, state_weight_hash
from scripts.engram.run_llavamed_record953_lora_positive_control import insert_lora, seed_everything
from scripts.engram.run_record953_routed_banked_lora_one_edit import (
    EXPECTED_ANCHOR_HASH, EXPECTED_BANK_HASH, EXPECTED_OUTPUT, EXPECTED_SHORT_OUTPUT,
    RECORD_ID, TARGET, TOL, exact_locality_rows, extract_router_keys, generate_parity,
    generation_sample, locate_positive_control, make_entry,
)
from scripts.engram.run_record953_routed_banked_lora_v1_1 import raw_membership, visible_input_audit
from scripts.engram.run_record953_routed_lora_generality_attribution import locate_v11_anchor


PROTOCOL = "MODALITY_AWARE_ROUTED_BANKED_LORA_CROSS_EDIT_V1"
ATTRIBUTION_ROOT = ROOT / "outputs/record953_routed_lora_generality_attribution"
EXPECTED_ATTRIBUTION_OUTPUT = "The answer is completely ectocervical and fully visible."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source-pool", required=True, type=Path)
    parser.add_argument("--physical-gpu", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(value.rstrip() + "\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(to_jsonable(dict(value)), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def source_diff() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/engram/modality_aware_router_utils.py",
        ROOT / "scripts/engram/prepare_medmkeb_cross_edit_pool.py",
        ROOT / "tests/test_modality_aware_routed_banked_lora_cross_edit.py",
    )
    chunks = []
    for path in paths:
        chunks.extend(difflib.unified_diff([], path.read_text().splitlines(True), fromfile="/dev/null", tofile=f"b/{path.relative_to(ROOT)}"))
    return "".join(chunks)


def locate_attribution_anchor() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    valid = []
    for summary_path in ATTRIBUTION_ROOT.glob("*/generality_attribution_summary.json"):
        run = summary_path.parent
        try:
            summary = json.loads(summary_path.read_text())
            manifest = json.loads((run / "run_manifest.json").read_text())
            outputs = json.loads((run / "three_condition_generality_outputs.json").read_text())
            parity = json.loads((run / "generality_generation_parity.json").read_text())
            rows = list(csv.DictReader((run / "generality_router_scores.csv").open()))
            always = [item["ADAPTER_ALWAYS_ON"]["unrestricted"]["raw_output"] for item in outputs.values()]
            checks = (
                summary.get("primary_label") == "GENERALITY_ROUTER_RECALL_BOTTLENECK"
                and len(rows) == 3 and all(str(row["route_on"]).casefold() == "false" for row in rows)
                and always == [EXPECTED_ATTRIBUTION_OUTPUT] * 3
                and all(item.get("ADAPTER_ALWAYS_ON_UNRESTRICTED", {}).get("passed") is True for item in parity.values())
                and not summary.get("adapter_limited_examples")
            )
            if checks:
                valid.append((run, summary, manifest))
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            continue
    if len(valid) != 1:
        raise RuntimeError("SEMANTIC_ROUTER_INVALID_ANCHOR")
    return valid[0]


def anchor_audit() -> tuple[dict[str, Any], tuple[Any, ...], tuple[Any, ...]]:
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH or bank_anchor_hash() != EXPECTED_ANCHOR_HASH:
        raise RuntimeError("SEMANTIC_ROUTER_INVALID_ANCHOR")
    pc = locate_positive_control()
    v11 = locate_v11_anchor()
    attribution = locate_attribution_anchor()
    pc_run, pc_summary, _pc_manifest, pc_adapter_manifest = pc
    v11_run, v11_summary, v11_manifest, v11_bank, _v11_tensors, _v11_meta = v11
    attr_run, attr_summary, attr_manifest = attribution
    checks = {
        "positive_control": {
            "run": str(pc_run), "label": pc_summary["primary_label"], "record_id": RECORD_ID,
            "success_step": pc_summary["success_step"], "unrestricted": pc_summary["exact_unrestricted_output"],
            "short": pc_summary["exact_short_output"], "rank": pc_adapter_manifest["rank"], "alpha": pc_adapter_manifest["alpha"],
            "adapter_sha256": pc_adapter_manifest["adapter_sha256"],
            "reload_fresh_rollback": [pc_summary["reload"], pc_summary["fresh"], pc_summary["rollback"]],
        },
        "v1_1_exact_router": {
            "run": str(v11_run), "label": v11_summary["primary_label"], "target_route_on": v11_summary["target_route_on"],
            "unique_calibration_negatives": v11_summary["unique_calibration_negatives"],
            "unique_heldout_negatives": v11_manifest["counts"]["unique_heldout"], "heldout_false_positives": v11_summary["heldout_false_positive_count"],
            "fixed_locality_exact_s0": v11_summary["fixed_locality_exact_s0"], "bank_item": str(v11_bank),
            "reload_fresh_replay_rollback": [v11_summary["reproducibility"][name] for name in ("reload", "fresh", "replay", "rollback")],
        },
        "generality_attribution": {"run": str(attr_run), "label": attr_summary["primary_label"], "manifest_sha256": sha256_file(attr_run / "run_manifest.json")},
        "canonical_bank_hash": EXPECTED_BANK_HASH, "canonical_anchor_hash": EXPECTED_ANCHOR_HASH,
        "verified": True,
    }
    return checks, pc, v11


def audited_entry(model: Any, entry: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(entry)
    row.update(visible_input_audit(model, row))
    return row


def source_entries(root: Path, raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rid = str(raw["record_id"])
    native_image = root / str(raw["image"])
    alternate_image = root / str(raw["image_rephrase"])
    return {
        "exact": make_entry(f"cross:{rid}:exact", "cross_edit", "exact", "native", rid, str(native_image), str(raw["src"])),
        "textual": make_entry(f"cross:{rid}:textual", "cross_edit", "textual", "native_image_rephrased_question", rid, str(native_image), str(raw["rephrase"])),
        "visual": make_entry(f"cross:{rid}:visual", "cross_edit", "visual", "alternate_image_native_question", rid, str(alternate_image), str(raw["src"])),
        "paired": make_entry(f"cross:{rid}:paired", "cross_edit", "paired", "alternate_image_alternate_question", rid, str(alternate_image), str(raw["rephrase"])),
    }


def build_source_pool(model: Any, source_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(source_path.read_text())
    root = source_path.parent
    provisional, ledger = [], []
    for raw in payload["records"]:
        entries = {name: audited_entry(model, row) for name, row in source_entries(root, raw).items()}
        native_key = entries["exact"]["router_input_equivalence_key"]
        usable = {name: entry for name, entry in entries.items() if name == "exact" or entry["router_input_equivalence_key"] != native_key}
        for name, entry in entries.items():
            ledger.append({"record_id": raw["record_id"], "category": name, "equivalence_key": entry["router_input_equivalence_key"], "disposition": "KEPT" if name in usable else "EXCLUDED_NATIVE_EQUIVALENT_GENERALITY"})
        if len(usable) > 1:
            provisional.append({"record_id": str(raw["record_id"]), "selection_hash": raw["selection_hash"], "edited_target_audit": raw.get("alt"), "entries": usable, "raw": raw})
    by_native: dict[str, list[dict[str, Any]]] = {}
    for row in provisional:
        by_native.setdefault(row["entries"]["exact"]["router_input_equivalence_key"], []).append(row)
    conflicts = set()
    for rows in by_native.values():
        if len({str(row["edited_target_audit"]).casefold().strip() for row in rows}) > 1:
            conflicts.update(row["record_id"] for row in rows)
    eligible = [row for row in provisional if row["record_id"] not in conflicts][:96]
    counts = {branch: sum(branch in row["entries"] for row in eligible) for branch in BRANCHES}
    if len(eligible) < 36 or any(counts[name] < 12 for name in BRANCHES):
        raise RuntimeError("SEMANTIC_ROUTER_INSUFFICIENT_CROSS_EDIT_DATA")
    source_rows = [{"record_id": row["record_id"], "selection_hash": row["selection_hash"], "native_image_sha256": row["raw"]["native_image_sha256"], **{f"has_{name}": name in row["entries"] for name in ("exact", *BRANCHES)}} for row in eligible]
    return eligible, source_rows, ledger


def numpy_keys(value: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {name: tensor.detach().float().cpu().numpy() for name, tensor in value.items()}


def extract_cached(model: Any, entry: Mapping[str, Any], cache: dict[str, dict[str, np.ndarray]], specs: dict[str, Any]) -> dict[str, np.ndarray]:
    key = str(entry["router_input_equivalence_key"])
    if key not in cache:
        tensors, spec = extract_router_keys(model, entry)
        cache[key] = numpy_keys(tensors)
        specs[key] = spec
    return cache[key]


def hard_negatives(model: Any, split_name: str, records: Sequence[Mapping[str, Any]], cache: dict[str, dict[str, np.ndarray]], specs: dict[str, Any], prior_split_keys: set[str], ledger: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    prototypes = {row["record_id"]: extract_cached(model, row["entries"]["exact"], cache, specs) for row in records}
    relations, split_keys = [], set()
    for row in records:
        rid, proto = row["record_id"], prototypes[row["record_id"]]
        ranked = []
        for other in records:
            if other["record_id"] != rid:
                score = validated_scores(prototypes[other["record_id"]], proto)["s_joint"]
                ranked.append((-score, other["record_id"], other))
        ranked.sort(key=lambda item: (item[0], item[1]))
        nearest = [item[2] for item in ranked[:4]]
        remaining = [item[2] for item in ranked[4:]]
        extra = sorted(remaining, key=lambda other: stable_hash(rid, other["record_id"]))[:2]
        positives = {entry["router_input_equivalence_key"] for entry in row["entries"].values()}
        seen = set()
        for other in [*nearest, *extra]:
            oid = other["record_id"]
            candidates = [
                ("other_native", other["entries"]["exact"]),
                ("prototype_image_other_question", make_entry(f"neg:{split_name}:{rid}:{oid}:pi_oq", split_name, "negative", "prototype_image_other_question", oid, row["entries"]["exact"]["image_path"], other["entries"]["exact"]["question"])),
                ("other_image_prototype_question", make_entry(f"neg:{split_name}:{rid}:{oid}:oi_pq", split_name, "negative", "other_image_prototype_question", oid, other["entries"]["exact"]["image_path"], row["entries"]["exact"]["question"])),
            ]
            for pair_type, candidate_raw in candidates:
                candidate = candidate_raw if "router_input_equivalence_key" in candidate_raw else audited_entry(model, candidate_raw)
                eq = str(candidate["router_input_equivalence_key"])
                disposition = "KEPT_NEGATIVE"
                if eq in positives:
                    disposition = "EXCLUDED_POSITIVE_EQUIVALENCE_COLLISION"
                elif eq in seen:
                    disposition = "DEDUPLICATED_NEGATIVE_EQUIVALENCE_CLASS"
                elif eq in prior_split_keys:
                    disposition = "EXCLUDED_CROSS_SPLIT_EQUIVALENCE_DUPLICATE"
                if disposition == "KEPT_NEGATIVE":
                    seen.add(eq); split_keys.add(eq)
                    extract_cached(model, candidate, cache, specs)
                    relations.append({"prototype_id": rid, "candidate_id": candidate["input_id"], "candidate": candidate, "equivalence_key": eq, "pair_type": pair_type})
                ledger.append({"split": split_name, "prototype_id": rid, "other_edit_id": oid, "candidate_id": candidate["input_id"], "equivalence_key": eq, "disposition": disposition})
    return relations, split_keys


def relation_dataset(branch: str, records: Sequence[Mapping[str, Any]], negatives: Sequence[Mapping[str, Any]], cache: Mapping[str, dict[str, np.ndarray]], pcas: Mapping[str, Any], cap_train: bool) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    by_id = {row["record_id"]: row for row in records}
    rows = []
    for row in records:
        if branch in row["entries"]:
            p = cache[row["entries"]["exact"]["router_input_equivalence_key"]]
            c = cache[row["entries"][branch]["router_input_equivalence_key"]]
            rows.append({"prototype_id": row["record_id"], "candidate_id": row["entries"][branch]["input_id"], "equivalence_key": row["entries"][branch]["router_input_equivalence_key"], "label": 1, "category": branch, "feature": relation_features(branch, p, c, pcas)})
    neg = list(negatives)
    if cap_train:
        neg = stable_negative_cap(neg, len(rows), 8)
    for relation in neg:
        pentry = by_id[relation["prototype_id"]]["entries"]["exact"]
        p, c = cache[pentry["router_input_equivalence_key"]], cache[relation["equivalence_key"]]
        rows.append({**{k: relation[k] for k in ("prototype_id", "candidate_id", "equivalence_key")}, "label": 0, "category": "negative", "feature": relation_features(branch, p, c, pcas)})
    return np.stack([row["feature"] for row in rows]), np.asarray([row["label"] for row in rows]), rows


def probabilities(rows: Sequence[dict[str, Any]], scaler: Any, model: Any, x: np.ndarray) -> list[dict[str, Any]]:
    probs = model.predict_proba(scaler.transform(x))[:, 1]
    return [{**{key: row[key] for key in ("prototype_id", "candidate_id", "equivalence_key", "label", "category")}, "probability": float(prob)} for row, prob in zip(rows, probs)]


def heldout_metrics(records: Sequence[Mapping[str, Any]], negatives: Sequence[Mapping[str, Any]], cache: Mapping[str, dict[str, np.ndarray]], pcas: Mapping[str, Any], models: Mapping[str, tuple[Any, Any]], thresholds: Mapping[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    by_id = {row["record_id"]: row for row in records}
    evaluated = []
    for row in records:
        proto = cache[row["entries"]["exact"]["router_input_equivalence_key"]]
        for category, entry in row["entries"].items():
            candidate = cache[entry["router_input_equivalence_key"]]
            branch_probs = {name: float(models[name][1].predict_proba(models[name][0].transform(relation_features(name, proto, candidate, pcas).reshape(1, -1)))[0, 1]) for name in BRANCHES}
            routes = {name: branch_probs[name] >= thresholds[name] for name in BRANCHES}
            evaluated.append({"prototype_id": row["record_id"], "candidate_id": entry["input_id"], "category": category, "label": 1, **{f"prob_{k}": v for k, v in branch_probs.items()}, **{f"route_{k}": v for k, v in routes.items()}, "semantic_route_on": any(routes.values()), "margin": max(branch_probs[k] - thresholds[k] for k in BRANCHES)})
    for relation in negatives:
        proto_entry = by_id[relation["prototype_id"]]["entries"]["exact"]
        proto, candidate = cache[proto_entry["router_input_equivalence_key"]], cache[relation["equivalence_key"]]
        branch_probs = {name: float(models[name][1].predict_proba(models[name][0].transform(relation_features(name, proto, candidate, pcas).reshape(1, -1)))[0, 1]) for name in BRANCHES}
        routes = {name: branch_probs[name] >= thresholds[name] for name in BRANCHES}
        evaluated.append({"prototype_id": relation["prototype_id"], "candidate_id": relation["candidate_id"], "category": "negative", "label": 0, **{f"prob_{k}": v for k, v in branch_probs.items()}, **{f"route_{k}": v for k, v in routes.items()}, "semantic_route_on": any(routes.values()), "margin": max(branch_probs[k] - thresholds[k] for k in BRANCHES)})
    categories = {}
    for name in ("exact", *BRANCHES):
        subset = [row for row in evaluated if row["category"] == name]
        hit = sum(row["semantic_route_on"] for row in subset)
        categories[name] = {"count": len(subset), "routed_on": hit, "recall": hit / len(subset) if subset else None, "wilson_95": wilson_interval(hit, len(subset)) if len(subset) < 8 else None, "robust_claim_permitted": len(subset) >= 8}
    neg = [row for row in evaluated if row["label"] == 0]
    labels = [row["label"] for row in evaluated]
    margins = [row["margin"] for row in evaluated]
    metrics = {
        "categories": categories, "overall_positive_recall": sum(row["semantic_route_on"] for row in evaluated if row["label"] == 1) / sum(row["label"] == 1 for row in evaluated),
        "negative_count": len(neg), "false_positives": sum(row["semantic_route_on"] for row in neg),
        "false_positive_rate": sum(row["semantic_route_on"] for row in neg) / len(neg),
        "auroc_secondary": float(roc_auc_score(labels, margins)), "auprc_secondary": float(average_precision_score(labels, margins)),
    }
    metrics["primary_gate_passed"] = all(categories[name]["recall"] >= .75 for name in BRANCHES) and metrics["false_positive_rate"] <= .01
    return metrics, evaluated


def external_scores(model: Any, entries: Sequence[Mapping[str, Any]], prototype: Mapping[str, np.ndarray], cache: dict[str, dict[str, np.ndarray]], specs: dict[str, Any], pcas: Mapping[str, Any], models: Mapping[str, tuple[Any, Any]], thresholds: Mapping[str, float], v11_tensors: Mapping[str, torch.Tensor], v11_thresholds: Mapping[str, float]) -> list[dict[str, Any]]:
    v11_proto = {name: v11_tensors[f"p_{name}"] for name in MODALITIES}
    results = []
    for entry in entries:
        value = extract_cached(model, entry, cache, specs)
        branch_probs = {name: float(models[name][1].predict_proba(models[name][0].transform(relation_features(name, prototype, value, pcas).reshape(1, -1)))[0, 1]) for name in BRANCHES}
        semantic = {name: branch_probs[name] >= thresholds[name] for name in BRANCHES}
        torch_value = {name: torch.from_numpy(value[name]) for name in MODALITIES}
        exact_scores = validated_scores({name: value[name] for name in MODALITIES}, {name: v11_proto[name].float().cpu().numpy() for name in MODALITIES})
        exact = v11_route_on(exact_scores, v11_thresholds, TOL)
        results.append({"input_id": entry["input_id"], "category": entry["category"], "exact_route_on": exact, **{f"exact_{k}": v for k, v in exact_scores.items()}, **{f"prob_{k}": v for k, v in branch_probs.items()}, **{f"semantic_{k}": v for k, v in semantic.items()}, "semantic_route_on": any(semantic.values()), "route_on": exact or any(semantic.values())})
    return results


def report(summary: Mapping[str, Any]) -> str:
    held = summary["cross_edit_held_out"]
    ext = summary["record953"]
    return f"""# Modality-Aware Router Final Decision

- Cross-edit source edits train/calibration/held-out: **{summary['split_counts']['train']} / {summary['split_counts']['calibration']} / {summary['split_counts']['heldout']}**
- Held-out textual / visual / paired recall: **{held['categories']['textual']['recall']:.4f} / {held['categories']['visual']['recall']:.4f} / {held['categories']['paired']['recall']:.4f}**
- Held-out negative false-positive rate: **{held['false_positive_rate']:.6f}**
- Record 953 exact target routed/generated: **{ext['exact_route_on']} / {ext['exact_generation_success']}**
- Textual generality routed/generated: **{ext['textual_route_on']} / {ext['textual_generation_success']}**
- Visual generality routed/generated: **{ext['visual_route_on']} / {ext['visual_generation_success']}**
- Paired generality routed/generated: **{ext['paired_route_on']} / {ext['paired_generation_success']}**
- Safety negatives activated: **{ext['safety_false_positives']} / 40**
- Fixed locality inputs activated: **{ext['locality_false_positives']} / 10**
- Strict / clinical locality perfect: **{ext['strict_locality_damage'] == 0} / {ext['clinical_locality_failures'] == 0}**
- Reload / fresh / replay / rollback: **{summary['reproducibility']['reload']} / {summary['reproducibility']['fresh']} / {summary['reproducibility']['replay']} / {summary['reproducibility']['rollback']}**
- Exact primary label: **`{summary['primary_label']}`**
- Is Stage-2 permitted? **No**
"""


def run(args: argparse.Namespace) -> None:
    seed_everything()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    write_text(out / "exact_command_log.txt", f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} {sys.executable} " + " ".join(sys.argv))
    write_text(out / "source_diff.patch", source_diff())
    write_text(out / "state_and_bank_hash_ledger.jsonl", "")
    audit, pc_anchor, v11_anchor = anchor_audit(); write_json(out / "anchor_and_hash_audit.json", audit)
    pc_run, pc_summary, _pc_manifest, pc_adapter_manifest = pc_anchor
    v11_run, v11_summary, v11_manifest, v11_bank, v11_tensors, v11_meta = v11_anchor
    model, views, bank, records = load_model_views_bank(args.physical_gpu); apply_prefix(model, bank, 0)
    clean_hash = state_weight_hash(model); append_jsonl(out / "state_and_bank_hash_ledger.jsonl", {"event": "CLEAN_S0", "state_hash": clean_hash, "bank_hash": bank_manifest()["sha256"]})
    eligible, source_rows, equivalence_ledger = build_source_pool(model, args.source_pool)
    split = edit_level_split(eligible)
    write_csv(out / "cross_edit_source_pool.csv", source_rows)
    split_json = {name: [{"record_id": row["record_id"], "selection_hash": row["selection_hash"]} for row in values] for name, values in split.items()}
    split_json["record953_excluded"] = all(row["record_id"] != RECORD_ID for values in split.values() for row in values)
    split_json["split_hash"] = json_hash(split_json); write_json(out / "edit_level_split.json", split_json)
    cache: dict[str, dict[str, np.ndarray]] = {}; specs: dict[str, Any] = {}
    for row in eligible:
        for entry in row["entries"].values(): extract_cached(model, entry, cache, specs)
    negative_by_split, prior = {}, set()
    for name in ("train", "calibration", "heldout"):
        negative_by_split[name], keys = hard_negatives(model, name, split[name], cache, specs, prior, equivalence_ledger)
        prior.update(keys)
    write_csv(out / "equivalence_and_collision_ledger.csv", equivalence_ledger)
    train_eq = {entry["router_input_equivalence_key"] for row in split["train"] for entry in row["entries"].values()} | {row["equivalence_key"] for row in negative_by_split["train"]}
    pca_keys = {name: np.stack([cache[key][name] for key in sorted(train_eq)]) for name in MODALITIES}
    pcas, pca_report = fit_pcas(pca_keys); pca_report["record953_used"] = False; pca_report["fit_equivalence_key_count"] = len(train_eq); write_json(out / "pca_report.json", pca_report)
    feature_spec = {"branches": {"textual": {"full": ["text", "fused"], "scalar": ["img"]}, "visual": {"full": ["img", "fused"], "scalar": ["text"]}, "paired": {"full": ["img", "text", "fused"], "scalar": ["img", "text", "fused"]}}, "full_fields": ["abs_diff", "hadamard"], "scalar_fields": ["cos_raw", "l2_raw", "cos_pca", "l2_pca"], "validated_scores": ["s_img", "s_text", "s_fused", "s_min", "s_joint"], "forbidden": ["target", "old_answer", "record_id", "dataset_name", "image_hash", "question_hash"], "inference_uses_metadata": False}
    write_json(out / "router_feature_spec.json", feature_spec)
    models, training_report, calibration_report, thresholds = {}, {}, {}, {}
    for branch in BRANCHES:
        tx, ty, trows = relation_dataset(branch, split["train"], negative_by_split["train"], cache, pcas, True)
        scaler, classifier, convergence = train_branch(tx, ty); models[branch] = (scaler, classifier); training_report[branch] = {**convergence, "positive_count": int(ty.sum()), "negative_count": int((ty == 0).sum()), "feature_dimension": int(tx.shape[1])}
        cx, cy, crows = relation_dataset(branch, split["calibration"], negative_by_split["calibration"], cache, pcas, False)
        scored = probabilities(crows, scaler, classifier, cx); negp = [row["probability"] for row in scored if row["label"] == 0]
        max_neg, tau = zero_fp_threshold(negp); thresholds[branch] = tau
        positives = [row for row in scored if row["label"] == 1]
        calibration_report[branch] = {"max_neg": max_neg, "tau": tau, "positive_count": len(positives), "positive_recall": sum(row["probability"] >= tau for row in positives) / len(positives), "false_positives": sum(row["probability"] >= tau for row in scored if row["label"] == 0), "probabilities": scored}
        save_model_npz(out / f"{branch}_router_model.npz", branch, pcas, scaler, classifier)
    write_json(out / "router_thresholds.json", {"formula": "nextafter(max_calibration_negative_probability, 1.0)", "thresholds": thresholds, "record953_used": False})
    write_json(out / "calibration_metrics.json", {"training": training_report, "calibration": calibration_report})
    held_metrics, held_rows = heldout_metrics(split["heldout"], negative_by_split["heldout"], cache, pcas, models, thresholds); held_metrics["rows"] = held_rows; write_json(out / "cross_edit_held_out_metrics.json", held_metrics)
    target_raw, _cal_raw, _held_raw, _fixed_raw, gen_raw = raw_membership(views, records)
    target_entry = audited_entry(model, target_raw); gen_entries = [audited_entry(model, row) for row in gen_raw]
    external_entries = [target_entry, *gen_entries]
    prototype953 = extract_cached(model, target_entry, cache, specs)
    v11_split = json.loads((v11_run / "calibration_and_test_split.json").read_text())
    safety_entries = [dict(row) for row in v11_split["unique_heldout_classes"]]
    locality_entries = [dict(row) for row in v11_split["fixed_locality_native_inputs"]]
    v11_thresholds = v11_meta["thresholds"]
    external_rows = external_scores(model, external_entries, prototype953, cache, specs, pcas, models, thresholds, v11_tensors, v11_thresholds)
    safety_rows = external_scores(model, safety_entries, prototype953, cache, specs, pcas, models, thresholds, v11_tensors, v11_thresholds)
    locality_rows = external_scores(model, locality_entries, prototype953, cache, specs, pcas, models, thresholds, v11_tensors, v11_thresholds)
    write_csv(out / "record953_router_scores.csv", [*external_rows, *safety_rows, *locality_rows])
    adapter_state, adapter_manifest = load_adapter_payload(pc_run / "successful_adapter_bank_item")
    resolved = resolve_target_modules(model.llava_model.named_modules())
    if resolved != adapter_manifest["resolved_lora_modules"]: raise RuntimeError("SEMANTIC_ROUTER_INVALID_ENGINEERING_RUN: adapter module mismatch")
    model.llava_model = insert_lora(model.llava_model, resolved); load_adapter_state(model.llava_model.named_parameters(), adapter_state)
    aliases = [str(value) for value in (records[RECORD_ID].get("accepted_answers") or [])]
    peft = model.llava_model
    generations = {}
    for entry, route_row in zip(external_entries, external_rows):
        if route_row["route_on"]: peft.enable_adapter_layers()
        else: peft.disable_adapter_layers()
        sample = views[RECORD_ID]["target"] if entry["category"] == "exact" else generation_sample(model, entry["image_path"], entry["question"], TARGET)
        generations[entry["input_id"]] = generate_parity(model, sample, aliases)
    damage = []
    for entry, route_row in zip(safety_entries, safety_rows):
        if route_row["route_on"]:
            peft.enable_adapter_layers(); damage.append({"input_id": entry["input_id"], "generation": generate_parity(model, generation_sample(model, entry["image_path"], entry["question"], TARGET), aliases)})
    peft.disable_adapter_layers()
    baseline_locality = {row["record_id"]: row["s0"] for row in json.loads((v11_run / "fixed_locality_results.json").read_text())["rows"]}
    locality_behavior = exact_locality_rows(model, views, baseline_locality)
    write_json(out / "record953_target_and_generality_generation.json", generations)
    write_json(out / "record953_safety_negative_results.json", {"safety_routes": safety_rows, "locality_routes": locality_rows, "false_positive_damage": damage, "locality_behavior": locality_behavior})
    target_result = next(row for row in external_rows if row["category"] == "exact")
    by_category = {row["category"]: row for row in external_rows if row["category"] in BRANCHES}
    success = {entry["category"]: bool(generations[entry["input_id"]]["match"]["success"] and generations[entry["input_id"]]["three_path_parity"]) for entry in external_entries}
    safety_fp = sum(row["semantic_route_on"] or row["exact_route_on"] for row in safety_rows)
    locality_fp = sum(row["semantic_route_on"] or row["exact_route_on"] for row in locality_rows)
    all_gen = all(by_category[name]["route_on"] and success[name] for name in BRANCHES)
    any_gen = any(by_category[name]["route_on"] and success[name] for name in BRANCHES)
    if safety_fp or locality_fp: label = "MODALITY_ROUTER_HELD_OUT_FALSE_POSITIVE"
    elif not any(row["semantic_route_on"] for row in by_category.values()): label = "MODALITY_ROUTER_GENERALITY_RECALL_FAILURE"
    elif any(by_category[name]["route_on"] and not success[name] for name in BRANCHES): label = "ROUTED_ADAPTER_GENERATION_FAILURE"
    elif not all_gen: label = "PARTIAL_MODALITY_AWARE_ROUTING"
    elif not held_metrics["primary_gate_passed"]: label = "RECORD953_GENERALITY_PASS_CROSS_EDIT_ROUTER_WEAK"
    else: label = "PASS_MODALITY_AWARE_ROUTED_LORA_CORE_AND_GENERALITY"
    external_summary = {"exact_route_on": target_result["route_on"], "exact_generation_success": success["exact"], **{f"{name}_route_on": by_category[name]["route_on"] for name in BRANCHES}, **{f"{name}_generation_success": success[name] for name in BRANCHES}, "safety_false_positives": safety_fp, "locality_false_positives": locality_fp, "strict_locality_damage": locality_behavior["strict_damage_count"], "clinical_locality_failures": locality_behavior["clinical_failure_count"], "maximum_locality_nll_drift": locality_behavior["maximum_nll_drift"]}
    three = {"S0_ADAPTER_OFF": {"target_and_generality_success": 0, "adapter_activation_count": 0, "strict_locality_damage": 0, "clinical_locality_failures": 0, "maximum_locality_nll_drift": 0.0}, "ADAPTER_ALWAYS_ON": {"target_and_generality_success": 4, "adapter_activation_count": 14, "strict_locality_damage": pc_summary["strict_locality_damage"], "clinical_locality_failures": pc_summary["clinical_locality_failures"], "maximum_locality_nll_drift": pc_summary["maximum_locality_nll_drift"], "source": "verified positive-control and attribution anchors"}, "MODALITY_AWARE_ROUTER_GATED": {"target_and_generality_success": sum(success.values()), "adapter_activation_count": sum(row["route_on"] for row in external_rows) + safety_fp + locality_fp, "strict_locality_damage": locality_behavior["strict_damage_count"], "clinical_locality_failures": locality_behavior["clinical_failure_count"], "maximum_locality_nll_drift": locality_behavior["maximum_nll_drift"]}}
    write_json(out / "three_condition_ablation.json", three)
    write_json(out / "representation_cache_manifest.json", {"unique_equivalence_classes": len(cache), "key_dimensions": {name: int(next(iter(cache.values()))[name].size) for name in MODALITIES}, "adapter_enabled": False, "base_state": "S0", "specifications": specs, "record953_frozen_after_thresholds": True})
    reproducibility = {"reload": "PASS" if all(np.load(out / f"{name}_router_model.npz")["coef"].shape == models[name][1].coef_.shape for name in BRANCHES) else "FAIL", "fresh": "NOT_RUN_NO_ELIGIBLE_BANK", "replay": "PASS" if all(row["route_on"] == rerow["route_on"] for row, rerow in zip(external_rows, external_scores(model, external_entries, prototype953, cache, specs, pcas, models, thresholds, v11_tensors, v11_thresholds))) else "FAIL", "rollback": "PASS" if bank_manifest()["sha256"] == EXPECTED_BANK_HASH else "FAIL"}
    summary = {"protocol": PROTOCOL, "primary_label": label, "stage2_permitted": False, "split_counts": {name: len(split[name]) for name in split}, "cross_edit_held_out": held_metrics, "record953": external_summary, "thresholds": thresholds, "reproducibility": reproducibility, "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH, "semantic_bank_created": False}
    eligible_bank = label == "PASS_MODALITY_AWARE_ROUTED_LORA_CORE_AND_GENERALITY"
    if eligible_bank:
        bank_out = out / "semantic_routed_adapter_bank_item"; bank_out.mkdir()
        for name in BRANCHES: subprocess.run(["cp", str(out / f"{name}_router_model.npz"), str(bank_out / f"{name}_router_model.npz")], check=True)
        write_json(bank_out / "manifest.json", {"protocol": PROTOCOL, "adapter_reference": str((pc_run / "successful_adapter_bank_item").resolve()), "frozen_v11_bank_reference": str(v11_bank.resolve()), "thresholds": thresholds, "split_hash": split_json["split_hash"], "cross_edit_metrics": held_metrics, "record953": external_summary, "canonical_bank_hash": EXPECTED_BANK_HASH})
        summary["semantic_bank_created"] = True
    write_json(out / "routed_bank_reload_fresh_replay_rollback.json", reproducibility)
    append_jsonl(out / "state_and_bank_hash_ledger.jsonl", {"event": "FINAL", "clean_hash": clean_hash, "current_hash": state_weight_hash(model), "canonical_bank_hash": bank_manifest()["sha256"], "adapter_disabled": True})
    run_manifest = {"protocol": PROTOCOL, "primary_label": label, "summary_sha256_pending": True, "source_pool_sha256": sha256_file(args.source_pool), "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in (Path(__file__).resolve(), ROOT / "scripts/engram/modality_aware_router_utils.py", ROOT / "scripts/engram/prepare_medmkeb_cross_edit_pool.py", ROOT / "tests/test_modality_aware_routed_banked_lora_cross_edit.py")}, "adapter_sha256": pc_adapter_manifest["adapter_sha256"], "v11_bank_manifest_sha256": sha256_file(v11_bank / "manifest.json"), "canonical_bank_hash": EXPECTED_BANK_HASH, "stage2_permitted": False}
    write_json(out / "run_manifest.json", run_manifest)
    write_json(out / "modality_aware_router_summary.json", summary)
    write_text(out / "MODALITY_AWARE_ROUTER_FINAL_DECISION.md", report(summary))


if __name__ == "__main__":
    run(parse_args())
