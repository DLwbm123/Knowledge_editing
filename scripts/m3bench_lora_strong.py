#!/usr/bin/env python3
"""Freeze LoRA-Strong cohorts, run bounded training, and apply its gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


GPU3_UUID = "GPU-43e3d478-7979-ea29-8130-64a467b48a5c"
SELECTOR_SALT = "m3bench-lora-strong-v1-cohorts-v1"
GRID = tuple(
    {"max_steps": steps, "learning_rate": learning_rate}
    for steps in (10, 20, 50, 100)
    for learning_rate in (5e-5, 1e-4, 2e-4, 5e-4)
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value) -> None:
    write_new(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    write_new(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_key(record_id: str) -> str:
    return hashlib.sha256(f"{SELECTOR_SALT}\0{record_id}".encode()).hexdigest()


def config_id(config: dict) -> str:
    return f"steps_{int(config['max_steps']):03d}__lr_{float(config['learning_rate']):.0e}"


def normalize_answer(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = text.replace("x-ray", "xray").replace("x ray", "xray")
    text = re.sub(r"[^\w\u3400-\u9fff.+-]+", " ", text, flags=re.UNICODE)
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


def token_f1(left: str, right: str) -> float:
    pattern = r"[\u3400-\u9fff]|[a-z0-9]+(?:\.[0-9]+)?"
    a = Counter(re.findall(pattern, normalize_answer(left)))
    b = Counter(re.findall(pattern, normalize_answer(right)))
    overlap = sum((a & b).values())
    if not a or not b or not overlap:
        return 0.0
    precision, recall = overlap / sum(a.values()), overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall)


def exact_or_fuzzy(answer: str, gold: str) -> bool:
    return normalize_answer(answer) == normalize_answer(gold) or token_f1(answer, gold) >= 0.80


def generation_equal(left: dict, right: dict) -> bool:
    return all(left[key] == right[key] for key in ("decoded_text", "raw_token_ids", "sequence_contract"))


def adaptive_should_stop(checkpoint_passes: list[bool]) -> bool:
    return len(checkpoint_passes) >= 2 and checkpoint_passes[-2:] == [True, True]


def rank_config_summaries(summaries: list[dict]) -> list[dict]:
    return sorted(
        summaries,
        key=lambda row: (
            -row["semantic_t0_correct"],
            -row["nll_decrease_count"],
            -row["exact_fuzzy_correct"],
            row["median_post_nll"],
            row["config"]["max_steps"],
            row["config"]["learning_rate"],
        ),
    )


def freeze(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise RuntimeError("cohort output root already exists")
    records = read_jsonl(args.records)
    if len(records) != 189 or len({row["record_id"] for row in records}) != 189:
        raise RuntimeError("amended-189 record lock failed")
    if [int(row["formal_sequence_position"]) for row in records] != list(range(1, 190)):
        raise RuntimeError("amended-189 order drift")
    smoke = read_json(args.smoke8)
    smoke_ids = {row["record_id"] for row in smoke["rows"]}
    catalog = read_jsonl(args.catalog)
    locality = defaultdict(list)
    for row in catalog:
        if row.get("task") == "T1L" and row.get("pre_is_correct") is True:
            locality[row["edit_id"]].append(row)
    by_id = {row["record_id"]: row for row in records}
    heldout_pool = sorted((set(by_id) - smoke_ids) & set(locality), key=selection_key)
    if len(heldout_pool) < 16:
        raise RuntimeError("fewer than 16 non-smoke records have an eligible locality probe")
    heldout_ids = heldout_pool[:16]
    remaining = sorted(set(by_id) - smoke_ids - set(heldout_ids), key=selection_key)
    sizes = (("LORA_OVERFIT_4", 4), ("LORA_DEV_8", 8), ("LORA_DEV_16", 16), ("LORA_SEQ_16", 16))
    cohorts, offset = {}, 0
    for name, size in sizes:
        cohorts[name] = [by_id[record_id] for record_id in remaining[offset : offset + size]]
        offset += size
    cohorts["LORA_HELDOUT_16"] = [by_id[record_id] for record_id in heldout_ids]
    all_sets = [set(row["record_id"] for row in rows) for rows in cohorts.values()]
    if any(left & right for index, left in enumerate(all_sets) for right in all_sets[index + 1 :]):
        raise RuntimeError("LoRA cohort overlap")
    if any(ids & smoke_ids for ids in all_sets):
        raise RuntimeError("LoRA cohort leaked prior smoke records")
    if any(not Path(row["image_path"]).is_file() for rows in cohorts.values() for row in rows):
        raise RuntimeError("cohort image missing")
    locality_rows = []
    for record_id in heldout_ids:
        candidates = sorted(
            locality[record_id],
            key=lambda row: hashlib.sha256(
                f"{SELECTOR_SALT}\0locality\0{row['probe_id']}".encode()
            ).hexdigest(),
        )
        locality_rows.append(candidates[0])
    if len(locality_rows) != 16 or any(not Path(row["image_path"]).is_file() for row in locality_rows):
        raise RuntimeError("held-out locality binding failed")
    args.output_root.mkdir(parents=True)
    hashes = {}
    for name, rows in cohorts.items():
        path = args.output_root / "cohorts" / f"{name}.jsonl"
        write_jsonl(path, rows)
        hashes[path.name] = sha256(path)
    locality_path = args.output_root / "cohorts/LORA_HELDOUT_LOCALITY_16.jsonl"
    write_jsonl(locality_path, locality_rows)
    hashes[locality_path.name] = sha256(locality_path)
    manifest = {
        "schema_version": "m3bench-lora-strong-cohorts-v1",
        "selector": f"sha256({SELECTOR_SALT} + NUL + record_id)",
        "counts": {name: len(rows) for name, rows in cohorts.items()},
        "prior_smoke_excluded": len(smoke_ids),
        "pairwise_disjoint": True,
        "target_leakage": False,
        "all_images_present": True,
        "heldout_locality_count": len(locality_rows),
        "file_sha256": hashes,
    }
    write_json(args.output_root / "cohorts/COHORT_MANIFEST.json", manifest)
    write_json(args.output_root / "governance/LORA_STRONG_PROTOCOL_AMENDMENT.json", {
        "status": "LORA_STRONG_MUST_PASS",
        "paper_spec_5": {
            "rank": 16, "alpha": 16, "dropout": 0.0, "learning_rate": 5e-5,
            "steps": 5, "scope": "all LM MLP", "modified": False,
        },
        "strong_v1": {
            "rank": 16, "alpha": 16, "dropout": 0.0, "scope": "all LM MLP",
            "optimizer": "AdamW", "gradient_clip": 1.0,
            "grid": list(GRID),
            "adaptive": {"min_steps": 10, "check_interval": 5, "max_steps": 100,
                         "required_consecutive_checkpoints": 2, "target_nll_max": 0.1,
                         "first_target_token_rank": 1, "generation": "normalized exact/fuzzy correct"},
        },
        "selection_rule_frozen_before_training": True,
    })
    write_new(args.output_root / "LORA_STRONG_COHORTS_FROZEN", "PASS\n")
    print(json.dumps({"status": "PASS", **manifest}, sort_keys=True))


def assert_gpu3() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("LoRA-Strong requires physical GPU3 only")
    observed = subprocess.check_output(
        ["nvidia-smi", "-i", "3", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if observed != GPU3_UUID:
        raise RuntimeError(f"GPU3 UUID mismatch: {observed}")


def evaluate_batch(editor, batch) -> tuple[float, int]:
    import torch

    with torch.no_grad():
        output = editor.runtime.model(**batch.forward_kwargs())
    positions = torch.where(batch.labels[0] != -100)[0]
    first = int(positions[0].item())
    target_id = int(batch.labels[0, first].item())
    logits = output.logits[0, first - 1].float()
    rank = int((logits > logits[target_id]).sum().item()) + 1
    return float(output.loss.detach().cpu().item()), rank


def train_one(editor, record, config: dict, *, adaptive: bool) -> dict:
    import torch
    from m3bench_repro.editors.llava_runtime import seed_everything
    from m3bench_repro.editors.methods import finite_gradients, record_seed

    seed_everything(record_seed(record.record_id, "lora-strong-v1"))
    editor._set_enabled(True)
    batch = editor.runtime.build_edit_batch(record)
    parameters = editor.trainable()
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]))
    pre_nll, _ = evaluate_batch(editor, batch)
    first_loss = last_loss = min_loss = None
    finite = []
    checkpoints, checkpoint_passes = [], []
    generation = None
    steps_run = 0
    for step in range(1, int(config["max_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = editor.runtime.compute_loss(batch)
        loss.backward()
        finite.append(finite_gradients(parameters))
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        value = float(loss.detach().cpu().item())
        first_loss = value if first_loss is None else first_loss
        last_loss, min_loss, steps_run = value, value if min_loss is None else min(min_loss, value), step
        if adaptive and step >= 10 and step % 5 == 0:
            nll, rank = evaluate_batch(editor, batch)
            correct = False
            if nll <= 0.1 and rank == 1:
                generation = editor.generate(record, use_cache=True)["generation"]
                correct = exact_or_fuzzy(generation["decoded_text"], record.target)
            checkpoint_passes.append(nll <= 0.1 and rank == 1 and correct)
            checkpoints.append({"step": step, "target_nll": nll, "first_target_token_rank": rank,
                                "exact_fuzzy_correct": correct})
            if adaptive_should_stop(checkpoint_passes):
                break
    post_nll, rank = evaluate_batch(editor, batch)
    generation = editor.generate(record, use_cache=True)["generation"]
    return {
        "seed": record_seed(record.record_id, "lora-strong-v1"),
        "steps_run": steps_run,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "minimum_loss": min_loss,
        "pre_target_nll": pre_nll,
        "post_target_nll": post_nll,
        "target_nll_decreased": post_nll < pre_nll,
        "first_target_token_rank": rank,
        "exact_fuzzy_correct": exact_or_fuzzy(generation["decoded_text"], record.target),
        "empty": not bool(generation["decoded_text"].strip()),
        "finite_gradients": all(finite),
        "target_mask": batch.mask_report(),
        "adaptive_checkpoints": checkpoints,
        "adaptive_stopped": adaptive and steps_run < int(config["max_steps"]),
        "generation": generation,
    }


def load_records(path: Path):
    from m3bench_repro.editors.llava_runtime import EditorRecord

    return [EditorRecord.from_dict(row) for row in read_jsonl(path)]


def load_configs(args: argparse.Namespace) -> list[dict]:
    if args.grid:
        return [dict(config) for config in GRID]
    if args.config_file:
        value = read_json(args.config_file)
        return list(value.get("selected_configs", value))
    return [{"max_steps": args.max_steps, "learning_rate": args.learning_rate}]


def blind_packet(rows: list[dict], output: Path) -> None:
    payloads, seen, mapping = [], {}, []
    for row in rows:
        values = [("t0", row["question"], row["gold"], row["training"]["generation"]["decoded_text"], "T0")]
        if row.get("locality"):
            values.append(("locality", row["locality"]["question"], row["locality"]["gold"],
                           row["locality"]["generation"]["decoded_text"], "T1L"))
        for kind, question, gold, answer, task in values:
            key = (question, gold, answer, task, row["dataset"])
            opaque = seen.get(key)
            if opaque is None:
                opaque = f"judge_{len(seen) + 1:06d}"
                seen[key] = opaque
                payloads.append({"opaque_event_id": opaque, "question": question,
                                 "gold_or_reference": gold, "raw_model_answer": answer,
                                 "task_metadata": {"task_id": task, "dataset": row["dataset"]}})
            mapping.append({"opaque_event_id": opaque, "kind": kind, "config_id": row["config_id"],
                            "opaque_record_index": row["opaque_record_index"]})
    write_jsonl(output / "JUDGE_PACKET.jsonl", payloads)
    write_jsonl(output / "JUDGE_MAP.jsonl", mapping)


def run(args: argparse.Namespace) -> None:
    import torch
    from m3bench_repro.editors.methods import LoraPaperSpecEditor
    from scripts.editor_paperspec_formal import load_runtime, probe_record

    assert_gpu3()
    if args.output.exists():
        raise RuntimeError("run output already exists")
    records = load_records(args.records)
    configs = load_configs(args)
    if not records or any(config["max_steps"] not in {10, 20, 50, 100} for config in configs):
        raise RuntimeError("invalid LoRA-Strong records/config")
    if any(config["learning_rate"] not in {5e-5, 1e-4, 2e-4, 5e-4} for config in configs):
        raise RuntimeError("learning rate outside bounded calibration grid")
    locality_by_edit = {}
    if args.locality:
        locality_by_edit = {row["edit_id"]: row for row in read_jsonl(args.locality)}
        if set(locality_by_edit) != {record.record_id for record in records}:
            raise RuntimeError("locality manifest does not bind every record exactly once")
    args.output.mkdir(parents=True)
    runtime = load_runtime(args.parent_run, "cuda:0")
    editor = LoraPaperSpecEditor(runtime)
    rows = []
    for config in configs:
        cid = config_id(config)
        for index, record in enumerate(records, 1):
            editor.reset_editor_state()
            training = train_one(editor, record, config, adaptive=args.adaptive)
            locality = None
            if record.record_id in locality_by_edit:
                probe = probe_record(locality_by_edit[record.record_id])
                generated = editor.generate(probe, use_cache=True)["generation"]
                locality = {"question": probe.question, "gold": probe.target, "generation": generated}
            parity = None
            state = None
            if args.save_reload:
                state_path = args.output / "states" / cid / f"record_{index:02d}"
                state = editor.save_editor_state(state_path)
                before = training["generation"]
                before_nll = training["post_target_nll"]
                editor.reset_editor_state()
                editor.load_editor_state(state_path)
                reloaded = editor.generate(record, use_cache=True)["generation"]
                reloaded_nll = editor.score_target_nll(record)["nll"]
                parity = generation_equal(before, reloaded) and before_nll == reloaded_nll
            base = editor.base_integrity()
            rows.append({
                "config": config, "config_id": cid, "opaque_record_index": index,
                "record_id": record.record_id, "dataset": record.dataset,
                "question": record.question, "gold": record.target,
                "training": training, "locality": locality, "state": state,
                "save_reload_parity": parity,
                "base_unchanged": base["unchanged"] and not base["base_parameters_requiring_grad"],
            })
    manifest = read_json(args.cohort_manifest)
    report = {
        "schema_version": "m3bench-lora-strong-run-v1",
        "stage": args.stage,
        "records": len(records),
        "configs": configs,
        "adaptive": args.adaptive,
        "save_reload": args.save_reload,
        "cohort_manifest": manifest,
        "rows": rows,
        "gpu": {"physical_index": 3, "uuid": GPU3_UUID, "visible_device": "cuda:0"},
        "paper_spec_5_modified": False,
        "strong_deviations": {"learning_rate": sorted({c["learning_rate"] for c in configs}),
                              "max_steps": sorted({c["max_steps"] for c in configs}),
                              "adaptive": args.adaptive},
    }
    write_json(args.output / "RUN_PRIVATE.json", report)
    blind_packet(rows, args.output)
    write_json(args.output / "RUN_SUMMARY_PREJUDGE.json", {
        "stage": args.stage, "records": len(records), "configs": len(configs),
        "empty_output_count": sum(row["training"]["empty"] for row in rows),
        "nll_decrease_count": sum(row["training"]["target_nll_decreased"] for row in rows),
        "first_token_rank1_count": sum(row["training"]["first_target_token_rank"] == 1 for row in rows),
        "exact_fuzzy_correct_count": sum(row["training"]["exact_fuzzy_correct"] for row in rows),
        "base_unchanged_count": sum(row["base_unchanged"] for row in rows),
        "save_reload_parity_count": sum(row["save_reload_parity"] is True for row in rows),
    })
    print(json.dumps(read_json(args.output / "RUN_SUMMARY_PREJUDGE.json"), sort_keys=True))
    del editor, runtime
    torch.cuda.empty_cache()


def verdicts_from(path: str) -> list[dict]:
    lines = sys.stdin.read().splitlines() if path == "-" else Path(path).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if any(set(row) != {"opaque_event_id", "correct"} or not isinstance(row["correct"], bool) for row in rows):
        raise RuntimeError("invalid semantic Judge verdicts")
    if len({row["opaque_event_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate semantic Judge verdict")
    return rows


def finalize(args: argparse.Namespace) -> None:
    report = read_json(args.run_dir / "RUN_PRIVATE.json")
    packet = read_jsonl(args.run_dir / "JUDGE_PACKET.jsonl")
    mapping = read_jsonl(args.run_dir / "JUDGE_MAP.jsonl")
    verdict_rows = verdicts_from(args.verdicts)
    verdicts = {row["opaque_event_id"]: row["correct"] for row in verdict_rows}
    expected = {row["opaque_event_id"] for row in packet}
    if set(verdicts) != expected:
        raise RuntimeError("semantic Judge coverage mismatch")
    write_jsonl(args.run_dir / "JUDGE_VERDICTS.jsonl", verdict_rows)
    mapped = {(row["config_id"], row["opaque_record_index"], row["kind"]): verdicts[row["opaque_event_id"]]
              for row in mapping}
    summaries = []
    by_config = defaultdict(list)
    for row in report["rows"]:
        by_config[row["config_id"]].append(row)
    for cid, rows in by_config.items():
        locality_values = [mapped[(cid, row["opaque_record_index"], "locality")] for row in rows if row.get("locality")]
        summaries.append({
            "config_id": cid,
            "config": rows[0]["config"],
            "records": len(rows),
            "semantic_t0_correct": sum(mapped[(cid, row["opaque_record_index"], "t0")] for row in rows),
            "exact_fuzzy_correct": sum(row["training"]["exact_fuzzy_correct"] for row in rows),
            "nll_decrease_count": sum(row["training"]["target_nll_decreased"] for row in rows),
            "nll_under_0_5_count": sum(row["training"]["post_target_nll"] < 0.5 for row in rows),
            "first_token_rank1_count": sum(row["training"]["first_target_token_rank"] == 1 for row in rows),
            "empty_output_count": sum(row["training"]["empty"] for row in rows),
            "base_unchanged_count": sum(row["base_unchanged"] for row in rows),
            "save_reload_parity_count": sum(row["save_reload_parity"] is True for row in rows),
            "finite_gradient_count": sum(row["training"]["finite_gradients"] for row in rows),
            "valid_target_mask_count": sum(
                row["training"]["target_mask"]["target_token_count"] > 0
                and row["training"]["target_mask"]["prompt_and_image_positions_masked"]
                and row["training"]["target_mask"]["target_positions_unmasked"]
                for row in rows
            ),
            "locality_correct": sum(locality_values),
            "locality_denominator": len(locality_values),
            "median_post_nll": statistics.median(row["training"]["post_target_nll"] for row in rows),
        })
    ranked = rank_config_summaries(summaries)
    selected = [row["config"] for row in ranked[: args.top]]
    status = "PASS"
    if args.stage == "overfit":
        row = ranked[0]
        status = "PASS" if (
            row["records"] == 4 and row["semantic_t0_correct"] == 4
            and row["nll_under_0_5_count"] == 4 and row["first_token_rank1_count"] == 4
            and row["empty_output_count"] == 0 and row["base_unchanged_count"] == 4
            and row["save_reload_parity_count"] == 4 and row["finite_gradient_count"] == 4
            and row["valid_target_mask_count"] == 4
        ) else "FAIL"
    elif args.stage == "heldout":
        row = ranked[0]
        locality = row["locality_correct"] / row["locality_denominator"] if row["locality_denominator"] else 0.0
        status = "PASS" if (
            row["records"] == 16 and row["semantic_t0_correct"] >= 12
            and row["nll_decrease_count"] >= 14 and row["empty_output_count"] == 0
            and row["base_unchanged_count"] == 16 and row["save_reload_parity_count"] == 16
            and row["finite_gradient_count"] == 16 and report["cohort_manifest"]["target_leakage"] is False
            and row["valid_target_mask_count"] == 16
            and locality >= 0.70
        ) else "FAIL"
    final = {"stage": args.stage, "status": status, "ranking": ranked,
             "selected_configs": selected, "judge_coverage": len(verdicts)}
    write_json(args.run_dir / "FINAL_SUMMARY.json", final)
    write_json(args.run_dir / "SELECTED_CONFIGS.json", {"selected_configs": selected})
    write_new(args.run_dir / (f"LORA_STRONG_{args.stage.upper()}_{status}"), status + "\n")
    print(json.dumps(final, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--records", type=Path, required=True)
    freeze_parser.add_argument("--catalog", type=Path, required=True)
    freeze_parser.add_argument("--smoke8", type=Path, required=True)
    freeze_parser.add_argument("--output-root", type=Path, required=True)
    freeze_parser.set_defaults(func=freeze)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--stage", choices=("overfit", "dev8", "dev16", "heldout"), required=True)
    run_parser.add_argument("--records", type=Path, required=True)
    run_parser.add_argument("--locality", type=Path)
    run_parser.add_argument("--cohort-manifest", type=Path, required=True)
    run_parser.add_argument("--parent-run", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--grid", action="store_true")
    run_parser.add_argument("--config-file", type=Path)
    run_parser.add_argument("--max-steps", type=int, default=100)
    run_parser.add_argument("--learning-rate", type=float, default=2e-4)
    run_parser.add_argument("--adaptive", action="store_true")
    run_parser.add_argument("--save-reload", action="store_true")
    run_parser.set_defaults(func=run)
    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--stage", choices=("overfit", "dev8", "dev16", "heldout"), required=True)
    finalize_parser.add_argument("--run-dir", type=Path, required=True)
    finalize_parser.add_argument("--verdicts", required=True)
    finalize_parser.add_argument("--top", type=int, default=1)
    finalize_parser.set_defaults(func=finalize)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
