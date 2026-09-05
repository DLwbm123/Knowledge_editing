#!/usr/bin/env python3
"""Build fixed Judge input and select one LoRA-Perf DEV configuration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.editor_paperspec_formal import write_frozen_json, write_frozen_text


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def profile_id(profile: dict) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return "profile_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def build_packet(args: argparse.Namespace) -> None:
    inputs = {row["event_id"]: row for row in read_jsonl(args.inputs)}
    packet, mapping = [], []
    for root in args.runs:
        lock = read_json(root / "RUN_LOCK.json")
        pid = profile_id(lock["profile"])
        for event_dir in sorted(root.glob("event_*")):
            if not (event_dir / "raw_event.json").is_file():
                continue
            raw = read_json(event_dir / "raw_event.json")
            source = inputs[raw["event_id"]]
            probes = {row["probe_id"]: row for row in source["probes"]}
            for checkpoint in raw["trajectory"]:
                step = checkpoint["step"]
                items = [(
                    "target", "T0", None, raw["edit_record"]["question"],
                    raw["edit_record"]["gold_answer"], checkpoint["target_generation"]["generation"]["decoded_text"],
                )]
                for generated in checkpoint["probes"]:
                    probe = probes[generated["probe_id"]]
                    items.append(("probe", probe["task"], probe["probe_id"], probe["question"], probe["reference"], generated["raw_text"]))
                    if probe["question"] == raw["edit_record"]["question"] and probe["image_path"] != raw["edit_record"]["image_path"]:
                        items.append(("copy_target", probe["task"], probe["probe_id"], probe["question"], raw["edit_record"]["gold_answer"], generated["raw_text"]))
                for purpose, task, probe_id, question, gold, answer in items:
                    key = f"{pid}|{raw['event_id']}|{step}|{purpose}|{probe_id or 'target'}"
                    opaque = hashlib.sha256(key.encode()).hexdigest()[:32]
                    packet.append({"opaque_query_id": opaque, "question": question, "gold_answer": gold, "raw_base_answer": answer, "adjudication_pass": 1})
                    mapping.append({
                        "opaque_query_id": opaque, "profile_id": pid, "profile": lock["profile"],
                        "event_id": raw["event_id"], "checkpoint": step, "purpose": purpose,
                        "task": task, "probe_id": probe_id,
                    })
    if not packet or len({row["opaque_query_id"] for row in packet}) != len(packet):
        raise RuntimeError("LoRA-Perf Judge packet is empty or has duplicate IDs")
    write_frozen_text(args.packet, "".join(json.dumps(row, sort_keys=True) + "\n" for row in packet))
    write_frozen_text(args.mapping, "".join(json.dumps(row, sort_keys=True) + "\n" for row in mapping))
    print(json.dumps({"status": "LORA_PERF_DEV_JUDGE_PACKET_FROZEN", "count": len(packet)}))


def macro(values: dict[str, list[bool]]) -> float | None:
    per_edit = [sum(rows) / len(rows) for rows in values.values() if rows]
    return sum(per_edit) / len(per_edit) if per_edit else None


def summarize(args: argparse.Namespace) -> None:
    verdicts = {row["opaque_query_id"]: row for row in read_jsonl(args.verdicts)}
    mapping = read_jsonl(args.mapping)
    if set(verdicts) != {row["opaque_query_id"] for row in mapping} or any(type(row.get("is_correct")) is not bool for row in verdicts.values()):
        raise RuntimeError("LoRA-Perf Judge coverage or schema failure")
    judged: dict[tuple[str, int], list[dict]] = defaultdict(list)
    profiles = {}
    for row in mapping:
        row = {**row, "is_correct": verdicts[row["opaque_query_id"]]["is_correct"]}
        judged[(row["profile_id"], row["checkpoint"])].append(row)
        profiles[row["profile_id"]] = row["profile"]
    diagnostics = {}
    for root in args.runs:
        pid = profile_id(read_json(root / "RUN_LOCK.json")["profile"])
        for event_dir in sorted(root.glob("event_*")):
            path = event_dir / "raw_event.json"
            if not path.is_file():
                continue
            raw = read_json(path)
            for checkpoint in raw["trajectory"]:
                diagnostics[(pid, checkpoint["step"], raw["event_id"])] = {
                    "nll_decreased": checkpoint["target_score"]["nll"] < raw["pre_target"]["nll"],
                    "empty": not checkpoint["target_generation"]["generation"]["decoded_text"].strip(),
                    "parameters": raw["trainable_parameter_count"],
                    "runtime_seconds": raw["runtime_seconds"],
                }
    rows = []
    for (pid, step), decisions in sorted(judged.items()):
        by_task: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
        copy_values = []
        for item in decisions:
            if item["purpose"] == "copy_target":
                copy_values.append(item["is_correct"])
            elif item["purpose"] in {"target", "probe"}:
                by_task[item["task"]][item["event_id"]].append(item["is_correct"])
        event_diag = [value for (p, s, _), value in diagnostics.items() if p == pid and s == step]
        metric = lambda task: macro(by_task.get(task, {}))
        t0 = sum(value for values in by_task.get("T0", {}).values() for value in values)
        row = {
            "profile_id": pid, "checkpoint": step, "t0_correct": t0, "t0_total": len(by_task.get("T0", {})),
            "t1l_macro": metric("T1L"), "t1g": metric("T1G"), "t2g": metric("T2G"),
            "edit_target_copy_rate_same_question_other_image": sum(copy_values) / len(copy_values) if copy_values else None,
            "target_nll_decrease": sum(item["nll_decreased"] for item in event_diag),
            "empty_or_error": sum(item["empty"] for item in event_diag),
            "trainable_parameters": event_diag[0]["parameters"] if event_diag else None,
            "runtime_seconds": sum(item["runtime_seconds"] for item in event_diag),
        }
        required = row["t1l_macro"] is not None and row["t1g"] is not None and row["t2g"] is not None
        row["readiness"] = bool(required and row["t0_correct"] >= 14 and row["t1l_macro"] >= 0.25 and row["t1g"] >= 0.85 and row["t2g"] >= 0.60 and row["target_nll_decrease"] >= 14 and row["empty_or_error"] == 0)
        rows.append(row)
    ready = [row for row in rows if row["readiness"]]
    selected = max(ready, key=lambda row: (row["t0_correct"], row["t1l_macro"], row["t1g"], row["t2g"], -row["checkpoint"], -row["trainable_parameters"], -row["runtime_seconds"])) if ready else None
    report = {"schema_version": "lora-perf-v1-dev-summary-v1", "status": "DEV_UNIQUE_PROFILE_SELECTED" if selected else "DEV_READINESS_NOT_MET", "candidates": rows, "selected": selected}
    write_frozen_json(args.output, report)
    with args.csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if selected:
        write_frozen_json(args.selection_lock, {
            "schema_version": "lora-perf-v1-dev-selection-lock-v1", "status": "DEV_UNIQUE_PROFILE_SELECTED",
            "profile_id": selected["profile_id"], "profile": profiles[selected["profile_id"]], "checkpoint": selected["checkpoint"],
        })
    print(json.dumps({"status": report["status"], "candidate_count": len(rows), "selected": selected}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="action", required=True)
    packet = sub.add_parser("packet")
    packet.add_argument("--inputs", type=Path, required=True); packet.add_argument("--runs", type=Path, nargs=3, required=True)
    packet.add_argument("--packet", type=Path, required=True); packet.add_argument("--mapping", type=Path, required=True); packet.set_defaults(func=build_packet)
    summary = sub.add_parser("summarize")
    summary.add_argument("--runs", type=Path, nargs=3, required=True); summary.add_argument("--mapping", type=Path, required=True); summary.add_argument("--verdicts", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True); summary.add_argument("--csv", type=Path, required=True); summary.add_argument("--selection-lock", type=Path, required=True); summary.set_defaults(func=summarize)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args(); arguments.func(arguments)
