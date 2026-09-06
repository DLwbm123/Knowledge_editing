#!/usr/bin/env python3
"""Bounded two-GPU MedTRACE route-representation development campaign."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from m3bench_repro.editors.llava_runtime import EditorRecord, seed_everything  # noqa: E402
from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook, ScopeCalibration  # noqa: E402
from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer, tensor_sha256  # noqa: E402
from scripts.medtrace.build_scope_census import image_identity, normalize, scan_candidates, source_key  # noqa: E402
from scripts.medtrace.run_dev16 import (  # noqa: E402
    atomic_json,
    derive_seed,
    evaluation_rows,
    generate,
    read_jsonl,
    run_event,
    save_checkpoint,
    sha256_file,
    sha256_json,
    validate_dev_rows,
)
from scripts.medtrace.run_generality_ablation import CONDITIONS as A_CONTINUATIONS, train_condition  # noqa: E402
from scripts.medtrace.run_realmodel_core import LAYER, load_real_runtime  # noqa: E402
from scripts.medtrace.run_scope_pilot import generate as scope_generate  # noqa: E402


SEEDS = (20260906, 20260907, 20260908)
REPRESENTATIONS = ("R0_MAGNITUDE_LAST_PROMPT", "R1_SIGNED_CP_RESPONSE", "R2_VISUAL_CONDITIONED_CP_RESPONSE")
BUDGETS = (200, 800)
OPERATING_POINTS = ("SAFETY_FIRST", "COVERAGE_CONSTRAINED")
GPU_UUIDS = {
    "2": "GPU-35be76e9-8ca5-1877-ddfe-27eb08f6721b",
    "3": "GPU-43e3d478-7979-ea29-8130-64a467b48a5c",
}
STOP_REQUESTED = False
SCORE_DEFINITION = "medtrace-frozen-fit-prototype-score-v1"
SCORE_DEFINITION_SHA256 = hashlib.sha256(SCORE_DEFINITION.encode()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.with_suffix(path.suffix + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with path.open("a") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def state_hash(state: dict[str, torch.Tensor]) -> str:
    return sha256_json({name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for name, value in state.items()})


def normalize_rows(value: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return value / (value.norm(dim=-1, keepdim=True) + epsilon)


def response_parts(
    expert: AsymmetricCPExpert,
    prompt: torch.Tensor,
    visual: list[torch.Tensor],
    representation: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    prompt_response = expert.component_contraction(prompt)
    if representation != REPRESENTATIONS[2]:
        return prompt_response, None
    pooled = []
    for prompt_row, visual_rows in zip(prompt_response, visual, strict=True):
        visual_response = expert.component_contraction(visual_rows)
        weights = F.softmax((normalize_rows(visual_response) @ normalize_rows(prompt_row[None])[0]) / 0.1, dim=0)
        pooled.append((weights[:, None] * visual_response).sum(dim=0))
    return prompt_response, torch.stack(pooled)


def build_fit_prototypes(
    expert: AsymmetricCPExpert,
    fit_prompt: torch.Tensor,
    fit_visual: list[torch.Tensor],
    representation: str,
) -> dict[str, torch.Tensor]:
    if representation == REPRESENTATIONS[0]:
        return {}
    if not len(fit_prompt):
        raise ValueError("fit prototypes require at least one fit-positive input")
    prompt_response, visual_response = response_parts(expert, fit_prompt, fit_visual, representation)
    prompt_prototype = normalize_rows(normalize_rows(prompt_response).mean(dim=0, keepdim=True))[0].detach()
    metadata = {"prompt_prototype": prompt_prototype}
    if representation == REPRESENTATIONS[2]:
        if visual_response is None:
            raise RuntimeError("visual-conditioned prototype is missing visual responses")
        metadata["visual_prototype"] = normalize_rows(normalize_rows(visual_response).mean(dim=0, keepdim=True))[0].detach()
    return metadata


def score_with_frozen_prototypes(
    expert: AsymmetricCPExpert,
    prompt: torch.Tensor,
    visual: list[torch.Tensor],
    representation: str,
    prototypes: dict[str, torch.Tensor],
) -> torch.Tensor:
    prompt_response, visual_response = response_parts(expert, prompt, visual, representation)
    if representation == REPRESENTATIONS[0]:
        return prompt_response.abs().mean(dim=-1)
    prompt_prototype = prototypes["prompt_prototype"].to(prompt_response.device)
    score = normalize_rows(prompt_response) @ prompt_prototype
    if representation == REPRESENTATIONS[2]:
        if visual_response is None:
            raise RuntimeError("visual-conditioned score is missing visual responses")
        visual_prototype = prototypes["visual_prototype"].to(visual_response.device)
        score = 0.5 * score + 0.5 * (normalize_rows(visual_response) @ visual_prototype)
    return score


def calibrate_operating_points(
    positive: list[float], hard: list[float], broad: list[float], *, hard_evaluable: bool,
) -> dict[str, dict[str, Any]]:
    if not positive or not broad or (hard_evaluable and not hard):
        raise ValueError("calibration requires positives, broad negatives and declared hard support")
    negatives = hard + broad
    candidates = sorted(set(positive + negatives))
    candidates = [math.nextafter(min(candidates), -math.inf), *candidates, math.nextafter(max(candidates), math.inf)]

    def metrics(threshold: float) -> dict[str, Any]:
        rate = lambda rows: sum(value > threshold for value in rows) / len(rows) if rows else None
        return {"threshold": threshold, "positive_tpr": rate(positive), "hard_fpr": rate(hard), "broad_fpr": rate(broad)}

    rows = [metrics(value) for value in candidates]
    safety = [row for row in rows if row["broad_fpr"] == 0 and (not hard_evaluable or row["hard_fpr"] == 0)]
    safety_first = max(safety, key=lambda row: (row["positive_tpr"], row["threshold"]))
    covered = [row for row in rows if row["positive_tpr"] >= 0.75]
    coverage = min(covered, key=lambda row: (
        row["hard_fpr"] if hard_evaluable else 0,
        row["broad_fpr"], -row["positive_tpr"], -row["threshold"],
    ))
    return {"SAFETY_FIRST": safety_first, "COVERAGE_CONSTRAINED": coverage}


def reviewed_wrappers(question: str) -> dict[str, list[dict[str, str]]]:
    chinese = any("\u4e00" <= char <= "\u9fff" for char in question)
    prefixes = {
        "calibration": (
            ("cal_image_basis", "请根据这幅图像回答：" if chinese else "Based on this image, answer: "),
            ("cal_visual_evidence", "依据图中所示信息，请回答：" if chinese else "Using the visual evidence shown, answer: "),
            ("cal_review", "请查看图像并回答：" if chinese else "Review the displayed image and answer: "),
            ("cal_reference", "参照这幅图像，请回答：" if chinese else "With reference to this image, answer: "),
        ),
        "evaluation": (
            ("eval_examine", "请仔细观察图像并回答：" if chinese else "Examine the image carefully and answer: "),
            ("eval_visual_only", "仅根据图像信息，请回答：" if chinese else "Using only the visual information, answer: "),
            ("eval_consider", "结合图中证据，请回答：" if chinese else "Considering the evidence in the image, answer: "),
            ("eval_look", "观察所示图像后，请回答：" if chinese else "After looking at the displayed image, answer: "),
        ),
    }
    return {
        role: [{"family": family, "question": prefix + question, "review_status": "APPROVED_EQUIVALENT_SOURCE_QUESTION_ONLY"} for family, prefix in values]
        for role, values in prefixes.items()
    }


def stable_rows(record_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{record_id}\0{row.get('source_dataset')}\0{image_identity(row['image_name'])}\0{row.get('source_qid')}".encode()
    ).hexdigest())


def allocate_hard(record_id: str, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {role: [] for role in ("fit", "calibration", "evaluation")}
    capacities = {"fit": 4, "calibration": 4, "evaluation": 5}
    cycle = ("fit", "calibration", "evaluation")
    role_index = 0
    for row in stable_rows(record_id, rows):
        for _ in cycle:
            role = cycle[role_index % len(cycle)]
            role_index += 1
            if len(result[role]) < capacities[role]:
                result[role].append(row)
                break
        if all(len(result[role]) == capacities[role] for role in cycle):
            break
    return result


def prepare_data(args: argparse.Namespace) -> None:
    if args.private_out.exists() or args.public_counts.exists() or args.tasks.exists():
        raise FileExistsError("campaign data/task freeze already exists")
    dev, qual = read_jsonl(args.dev_inputs), read_jsonl(args.qual_inputs)
    validate_dev_rows(dev)
    previous = json.loads(args.previous_data.read_text())
    slake = json.loads(args.slake_train.read_text())
    excluded = set()
    for event in qual:
        edit = event["edit_record"]
        excluded.add(source_key(edit["dataset"], edit["relative_image_path"], edit["question"], edit["gold_answer"]))
        excluded.update(source_key(row["dataset"], row["image_path"], row["question"], row["reference"]) for row in event["probes"])
    excluded.update(source_key(row["dataset"], row["relative_image_path"], row["question"], row["gold_answer"]) for row in read_jsonl(args.formal_records))
    excluded.update(source_key(row["dataset"], row["image_path"], row["question"], row["reference"]) for row in read_jsonl(args.formal_probes))
    old_scope = previous["scope"]
    scopes, public_edits = {}, []
    for index, event in enumerate(dev, 1):
        edit = dict(event["edit_record"])
        record_id = edit["record_id"]
        if record_id == old_scope["primary"]["record_id"]:
            scope = {
                "status": "HARD_EVALUABLE", "primary": old_scope["primary"],
                "positives": old_scope["positives"],
                "negative_roles": {
                    "fit": old_scope["negative_roles"]["mixed_fit"],
                    "calibration": old_scope["negative_roles"]["calibration"],
                    "evaluation": old_scope["negative_roles"]["evaluation"],
                    "same_image_challenge": old_scope["negative_roles"]["same_image_challenge"],
                },
                "preserved_primary_roles": True,
            }
        elif edit["dataset"].upper() != "SLAKE":
            scope = {"status": "FIT_ONLY_INSUFFICIENT", "reason": "VQA-RAD remained audit-only in the authorized source contract"}
        else:
            hits = [row for row in slake if image_identity(str(row.get("img_name") or "")) == image_identity(edit["relative_image_path"]) and normalize(row.get("question")) == normalize(edit["question"]) and normalize(row.get("answer")) == normalize(edit["gold_answer"])]
            if len(hits) != 1:
                scope = {"status": "FIT_ONLY_INSUFFICIENT", "reason": f"SLAKE train source QA unique-match count was {len(hits)}"}
            else:
                edit["source_triple"], edit["source_base_type"] = hits[0].get("triple"), hits[0].get("base_type")
                candidates = scan_candidates(edit, slake, excluded, args.slake_train.parent / "imgs")
                def one_per_image(relation: str) -> list[dict[str, Any]]:
                    selected: dict[str, dict[str, Any]] = {}
                    for row in stable_rows(record_id, [value for value in candidates if value["fact_relation"] == relation]):
                        selected.setdefault(image_identity(row["image_name"]), row)
                    return list(selected.values())
                hard = one_per_image("same_question_different_image_conflicting_source_answer")
                broad = one_per_image("broad_unrelated_source_qa")
                challenge = stable_rows(record_id, [row for row in candidates if row["fact_relation"] == "same_image_other_source_fact"])[:12]
                hard_roles = allocate_hard(record_id, hard)
                used_groups = {image_identity(row["image_name"]) for rows in hard_roles.values() for row in rows}
                broad = [row for row in stable_rows(record_id, broad) if image_identity(row["image_name"]) not in used_groups]
                negative_roles, offset = {}, 0
                for role in ("fit", "calibration", "evaluation"):
                    need = 20 - len(hard_roles[role])
                    negative_roles[role] = hard_roles[role] + broad[offset:offset + need]
                    offset += need
                wrappers = reviewed_wrappers(edit["question"])
                fit = previous["generality_paraphrases"][record_id]
                enough_broad = all(len(negative_roles[role]) == 20 for role in negative_roles)
                hard_evaluable = enough_broad and all(hard_roles[role] for role in hard_roles)
                status = "HARD_EVALUABLE" if hard_evaluable else "BROAD_ONLY" if enough_broad else "FIT_ONLY_INSUFFICIENT"
                scope = {
                    "status": status,
                    "primary": {"record_id": record_id, "dataset": edit["dataset"], "image_name": image_identity(edit["relative_image_path"]), "image_path": edit["image_path"], "question": edit["question"], "target": edit["gold_answer"]},
                    "positives": {"fit": fit, "calibration": wrappers["calibration"], "evaluation": wrappers["evaluation"]},
                    "negative_roles": {**negative_roles, "same_image_challenge": stable_rows(record_id, challenge)},
                    "hard_counts": {role: len(hard_roles[role]) for role in hard_roles},
                    "preserved_primary_roles": False,
                }
        scopes[record_id] = scope
        public_edits.append({
            "event_index": index, "opaque_edit": hashlib.sha256(record_id.encode()).hexdigest()[:16],
            "dataset": edit["dataset"], "status": scope["status"],
            "hard_counts": scope.get("hard_counts", {
                role: sum(row["fact_relation"] == "same_question_different_image_conflicting_source_answer" for row in scope.get("negative_roles", {}).get(role, []))
                for role in ("fit", "calibration", "evaluation")
            }),
            "negative_counts": {role: len(scope.get("negative_roles", {}).get(role, [])) for role in ("fit", "calibration", "evaluation", "same_image_challenge")},
            "reason": scope.get("reason"),
        })
    private = {
        "schema_version": "medtrace-route-representation-campaign-private-v1",
        "status": "FROZEN_BEFORE_MODEL_SCORING__EQKEY_PENDING", "dev": dev,
        "generality_paraphrases": previous["generality_paraphrases"], "scopes": scopes,
    }
    atomic_json(args.private_out, private)
    atomic_json(args.public_counts, {
        "schema_version": "medtrace-route-representation-coverage-public-v1", "status": "FROZEN_BEFORE_MODEL_SCORING__EQKEY_PENDING",
        "edit_counts": dict(Counter(row["status"] for row in public_edits)), "edits": public_edits,
        "patient_disjoint_claimed": False, "independent_clinical_review": False,
        "vqarad_scope_supervision_used": False, "private_artifacts_withheld": True,
    })
    tasks = []
    for seed_index, seed in enumerate(SEEDS):
        for event_index, event in enumerate(dev, 1):
            record_id = event["edit_record"]["record_id"]
            a_id = f"A_s{seed}_e{event_index:02d}"
            tasks.append({"task_id": a_id, "kind": "A", "seed": seed, "event_index": event_index, "record_id": record_id, "priority": seed_index * 100 + event_index * 2, "status": "PENDING", "attempts": 0})
            if scopes[record_id]["status"] in {"HARD_EVALUABLE", "BROAD_ONLY"}:
                b_id = f"B_s{seed}_e{event_index:02d}"
                tasks.append({"task_id": b_id, "kind": "B", "seed": seed, "event_index": event_index, "record_id": record_id, "priority": seed_index * 100 + event_index * 2 + 1, "depends_on": a_id, "status": "PENDING", "attempts": 0})
                tasks.append({"task_id": f"E2E_s{seed}_e{event_index:02d}", "kind": "E2E", "seed": seed, "event_index": event_index, "record_id": record_id, "priority": 1000 + seed_index * 100 + event_index, "depends_on": b_id, "status": "PENDING", "attempts": 0})
    atomic_json(args.tasks, {"schema_version": "medtrace-campaign-task-queue-v1", "tasks": tasks})


class TaskQueue:
    def __init__(self, path: Path, run_root: Path):
        self.path, self.run_root, self.lock_path = path, run_root, path.with_suffix(".lock")

    def _locked(self, operation: Callable[[dict[str, Any]], Any]) -> Any:
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text())
            value = operation(data)
            atomic_json(self.path, data)
            return value

    def snapshot(self) -> dict[str, Any]:
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            return json.loads(self.path.read_text())

    def claim(self, worker: str) -> dict[str, Any] | None:
        def operation(data: dict[str, Any]) -> dict[str, Any] | None:
            tasks = data["tasks"]
            by_id = {row["task_id"]: row for row in tasks}
            selected_exists = (self.run_root / "private/SELECTED_PROFILE.json").exists()
            for row in sorted(tasks, key=lambda item: (item["priority"], item["task_id"])):
                if row["status"] != "PENDING":
                    continue
                dependency = row.get("depends_on")
                if dependency and by_id[dependency]["status"] in {"FAILED", "CANCELLED_BY_BUDGET"}:
                    status = "CANCELLED_BY_BUDGET" if by_id[dependency]["status"] == "CANCELLED_BY_BUDGET" else "FAILED"
                    row.update(status=status, last_error=f"dependency {dependency} did not complete")
                    continue
                if dependency and by_id[dependency]["status"] not in {"COMPLETE", "RAW_READY", "JUDGED"}:
                    continue
                if row["kind"] == "E2E" and not selected_exists:
                    continue
                row.update(status="RUNNING", worker=worker, pid=os.getpid(), started_at=time.time(), attempts=row["attempts"] + 1)
                return dict(row)
            return None
        return self._locked(operation)

    def update(self, task_id: str, status: str, **fields: Any) -> None:
        def operation(data: dict[str, Any]) -> None:
            row = next(item for item in data["tasks"] if item["task_id"] == task_id)
            row.update(status=status, **fields)
        self._locked(operation)

    def recover(self) -> None:
        def operation(data: dict[str, Any]) -> None:
            for row in data["tasks"]:
                if row["status"] == "RUNNING":
                    row.update(status="PENDING", recovered_from_pid=row.get("pid"))
        self._locked(operation)

    def cancel_pending(self) -> None:
        def operation(data: dict[str, Any]) -> None:
            for row in data["tasks"]:
                if row["status"] == "PENDING":
                    row["status"] = "CANCELLED_BY_BUDGET"
        self._locked(operation)


class Telemetry:
    def __init__(self, path: Path, physical_gpu: str, worker: str):
        self.path, self.physical_gpu, self.worker = path, physical_gpu, worker
        self.task_id: str | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(0 if not self.path.exists() else 60):
            query = subprocess.check_output([
                "nvidia-smi", "-i", self.physical_gpu,
                "--query-gpu=utilization.gpu,memory.used,memory.free", "--format=csv,noheader,nounits",
            ], text=True).strip().split(", ")
            append_jsonl(self.path, {
                "at": time.time(), "worker": self.worker, "physical_gpu": int(self.physical_gpu), "task_id": self.task_id,
                "gpu_utilization_percent": int(query[0]), "nvidia_memory_used_mib": int(query[1]), "nvidia_memory_free_mib": int(query[2]),
                "torch_allocated_bytes": int(torch.cuda.memory_allocated()), "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
            })

    def __enter__(self) -> "Telemetry":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def verify_gpu() -> tuple[str, str]:
    physical = os.environ.get("M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES", "")
    expected = os.environ.get("M3BENCH_FORMAL_EXPECTED_GPU_UUID", "")
    if physical not in GPU_UUIDS or os.environ.get("CUDA_VISIBLE_DEVICES") != physical or expected != GPU_UUIDS[physical]:
        raise RuntimeError("campaign GPU authorization environment is invalid")
    actual = subprocess.check_output(["nvidia-smi", "-i", physical, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if actual != expected:
        raise RuntimeError("campaign GPU UUID mismatch")
    return physical, expected


def scope_rows(scope: dict[str, Any], event: dict[str, Any]) -> list[dict[str, Any]]:
    primary = scope["primary"]
    rows = []
    for role in ("fit", "calibration", "evaluation"):
        for index, row in enumerate(scope["positives"][role], 1):
            rows.append({
                "logical_id": f"positive-{role}-{index}", "role": role, "label": "positive",
                "question": row["question"], "reference": primary["target"], "image_path": primary["image_path"],
                "fact_relation": "reviewed_same_fact_text_augmentation",
            })
        for row in scope["negative_roles"][role]:
            rows.append({
                "logical_id": f"negative-{role}-{row['source_qid']}", "role": role, "label": "negative",
                "question": row["question"], "reference": row["source_answer"], "image_path": row["image_path"],
                "fact_relation": row["fact_relation"],
            })
    for row in scope["negative_roles"].get("same_image_challenge", []):
        rows.append({
            "logical_id": f"negative-challenge-{row['source_qid']}", "role": "challenge", "label": "negative",
            "question": row["question"], "reference": row["source_answer"], "image_path": row["image_path"],
            "fact_relation": row["fact_relation"],
        })
    edit = event["edit_record"]
    rows.append({"logical_id": "native", "role": "native", "label": "positive", "question": edit["question"], "reference": edit["gold_answer"], "image_path": edit["image_path"], "fact_relation": "native"})
    rows.extend({
        "logical_id": f"formal-{row['query_id']}", "role": "formal_development", "label": "positive",
        "question": row["question"], "reference": row["reference"], "image_path": row["image_path"],
        "fact_relation": row["task"], "task": row["task"],
    } for row in evaluation_rows(event)[1:])
    if len({row["logical_id"] for row in rows}) != len(rows):
        raise RuntimeError("scope rows contain duplicate logical IDs")
    return rows


def feature_cache(runtime: Any, event: dict[str, Any], scope: dict[str, Any], path: Path, locks: dict[str, str]) -> dict[str, Any]:
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["locks"] != locks:
            raise RuntimeError("feature cache lock mismatch")
        return cached
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False)
        record = EditorRecord.from_dict(event["edit_record"])
        values, eqkeys = {}, []
        for row in scope_rows(scope, event):
            batch = runtime.build_question_batch(record, question=row["question"], image_path=Path(row["image_path"]))
            activation = runtime.extract_layer_input_features(batch, module_path=LAYER).cpu()
            prompt = activation[batch.key_token_index]
            visual = activation[batch.image_token_start:batch.image_token_end]
            if not len(visual):
                raise RuntimeError("empty realized visual-token span")
            attention = batch.attention_mask if batch.attention_mask is not None else torch.ones(batch.inputs_embeds.shape[:2], dtype=torch.long)
            eq_payload = {
                "image_tensor_sha256": batch.image_sha256,
                "routing_input_ids": batch.raw_input_ids[0].detach().cpu().tolist(),
                "attention_mask": attention[0].detach().cpu().tolist(),
                "assistant_boundary_index": batch.key_token_index,
                "image_token_span": [batch.image_token_start, batch.image_token_end],
                **locks,
            }
            eqkey = sha256_json(eq_payload)
            eqkeys.append({"logical_id": row["logical_id"], "role": row["role"], "label": row["label"], "eqkey": eqkey})
            values[row["logical_id"]] = {"prompt": prompt, "visual": visual, "row": row, "eqkey": eqkey}
        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eqkeys:
            by_key[row["eqkey"]].append(row)
        for rows in by_key.values():
            if len({row["role"] for row in rows}) > 1 or len({row["label"] for row in rows}) > 1:
                raise RuntimeError("EqKey crosses roles or labels")
        cached = {"schema_version": "medtrace-route-feature-cache-private-v1", "locks": locks, "values": values, "eqkeys": eqkeys}
        temporary = path.with_suffix(".tmp")
        torch.save(cached, temporary)
        os.replace(temporary, path)
        return cached


def ordered_features(cache: dict[str, Any], *, roles: tuple[str, ...]) -> tuple[list[dict[str, Any]], torch.Tensor, list[torch.Tensor]]:
    rows = [value for value in cache["values"].values() if value["row"]["role"] in roles]
    role_order = {role: index for index, role in enumerate(roles)}
    rows.sort(key=lambda value: (role_order[value["row"]["role"]], value["row"]["label"] != "positive", value["row"]["logical_id"]))
    return rows, torch.stack([value["prompt"] for value in rows]).to("cuda:0"), [value["visual"].to("cuda:0") for value in rows]


def train_router_candidate(
    initial_state: dict[str, torch.Tensor],
    prompt: torch.Tensor,
    visual: list[torch.Tensor],
    fit_count: int,
    representation: str,
    out: Path,
) -> dict[str, Any]:
    expert = AsymmetricCPExpert(prompt.shape[-1], 4096, 4).to("cuda:0")
    expert.load_state_dict(initial_state)
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    expert.u_in.requires_grad_(True)
    expert.v_in.requires_grad_(True)
    optimizer = torch.optim.AdamW([expert.u_in, expert.v_in], lr=1e-3, weight_decay=0)
    trajectory, saved = [], {}
    started = time.monotonic()
    for step in range(1, 801):
        optimizer.zero_grad(set_to_none=True)
        prototypes = build_fit_prototypes(expert, prompt[:fit_count], visual[:fit_count], representation)
        scores = score_with_frozen_prototypes(expert, prompt, visual, representation, prototypes)
        positive, negative = scores[:fit_count], scores[fit_count:]
        logits = torch.cat((positive[:, None], negative.expand(len(positive), -1)), dim=1) / 0.1
        infonce = F.cross_entropy(logits, torch.zeros(len(positive), dtype=torch.long, device=logits.device))
        hinge = F.relu(0.1 + negative.max() - positive.min())
        q = expert.input_basis().float()
        orthogonality = (q.T @ q - torch.eye(expert.rank, device=q.device)).square().mean()
        loss = infonce + hinge + 0.01 * orthogonality
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([expert.u_in, expert.v_in], 1.0).item())
        if not math.isfinite(float(loss.item())) or not math.isfinite(grad_norm):
            raise FloatingPointError("non-finite route-representation optimization")
        optimizer.step()
        expert.normalize_input_factors_()
        if step % 100 == 0:
            trajectory.append({"step": step, "loss": float(loss.item()), "infonce": float(infonce.item()), "hinge": float(hinge.item()), "orthogonality": float(orthogonality.item()), "gradient_norm": grad_norm})
        if step in BUDGETS:
            prototypes = build_fit_prototypes(expert, prompt[:fit_count], visual[:fit_count], representation)
            checkpoint = out / f"{representation}__step{step}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_sha = save_checkpoint(checkpoint, {
                "representation": representation, "step": step, "rank": 4,
                "expert": expert.state_dict(), "prototypes": {key: value.cpu() for key, value in prototypes.items()},
                "q_sha256": tensor_sha256(expert.input_basis()), "prototype_sha256": state_hash(prototypes),
                "score_definition_sha256": SCORE_DEFINITION_SHA256,
            })
            saved[str(step)] = {"checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_sha, "q_sha256": tensor_sha256(expert.input_basis()), "prototype_sha256": state_hash(prototypes), "score_definition_sha256": SCORE_DEFINITION_SHA256, "prototype_bytes": sum(value.numel() * value.element_size() for value in prototypes.values())}
    return {"representation": representation, "trajectory": trajectory, "saved": saved, "elapsed_seconds": time.monotonic() - started}


def score_checkpoint(
    checkpoint: dict[str, Any], rows: list[dict[str, Any]], prompt: torch.Tensor, visual: list[torch.Tensor],
) -> list[dict[str, Any]]:
    expert = AsymmetricCPExpert(prompt.shape[-1], 4096, 4).to("cuda:0")
    expert.load_state_dict(checkpoint["expert"])
    scores = score_with_frozen_prototypes(expert, prompt, visual, checkpoint["representation"], checkpoint["prototypes"])
    return [{"logical_id": value["row"]["logical_id"], "role": value["row"]["role"], "label": value["row"]["label"], "fact_relation": value["row"]["fact_relation"], "score": float(score.item())} for value, score in zip(rows, scores, strict=True)]


def execute_a(runtime: Any, frozen: dict[str, Any], base: dict[str, dict[str, Any]], task: dict[str, Any], out: Path, locks: dict[str, str]) -> dict[str, Any]:
    event = frozen["dev"][task["event_index"] - 1]
    record = EditorRecord.from_dict(event["edit_record"])
    out.mkdir(parents=True, exist_ok=True)
    a0_dir = out / "A0"
    a0_result_path = a0_dir / "result.json"
    if a0_result_path.exists():
        a0 = json.loads(a0_result_path.read_text())
        if a0.get("seed_base") != task["seed"] or sha256_file(a0_dir / "expert.pt") != a0["checkpoint"]["sha256"]:
            raise RuntimeError("A0 resume lock mismatch")
    else:
        a0_dir.mkdir()
        a0 = run_event(runtime, event, base, a0_dir, seed_base=task["seed"])
        a0.update(seed_base=task["seed"], code_commit=locks["code_commit"], config_sha256=locks["config_sha256"], data_sha256=locks["data_sha256"])
        atomic_json(a0_result_path, a0)
    checkpoint = torch.load(a0_dir / "expert.pt", map_location="cuda:0", weights_only=True)
    stages = {"CP_NATIVE_ORIGINAL": {"result": a0, "predictions": a0["evaluation"]}}
    for condition in A_CONTINUATIONS:
        stage_dir = out / condition
        result_path = stage_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("seed_base") != task["seed"] or sha256_file(stage_dir / "expert.pt") != result["checkpoint_sha256"]:
                raise RuntimeError("continuation resume lock mismatch")
            stages[condition] = {"result": result, "predictions": result["predictions"]}
            continue
        stage_dir.mkdir()
        expert, result = train_condition(runtime, record, [row["question"] for row in frozen["generality_paraphrases"][record.record_id]], checkpoint, condition, seed_base=task["seed"])
        hook = MedTraceLayerHook(runtime.get_module(LAYER), expert)
        hook.attach()
        try:
            predictions = []
            for row in evaluation_rows(event):
                generated = generate(runtime, row, hook, int(runtime.generation_config["max_new_tokens"]))
                predictions.append({**row, **generated, "condition": condition, "edit_id": record.record_id, "exact": normalize_medical_answer(generated["decoded_text"]) == normalize_medical_answer(row["reference"]), "truncated_without_eos": generated["cap_hit"] and not (generated["generated_token_ids"] and generated["generated_token_ids"][-1] == runtime.adapter.tokenizer.eos_token_id)})
        finally:
            hook.detach()
        checkpoint_sha = save_checkpoint(stage_dir / "expert.pt", {"rank": 4, "step": 80, "condition": condition, "record_id": record.record_id, "seed": result["seed"], "expert": expert.state_dict()})
        result.update(seed_base=task["seed"], code_commit=locks["code_commit"], config_sha256=locks["config_sha256"], data_sha256=locks["data_sha256"], checkpoint_sha256=checkpoint_sha, predictions=predictions)
        atomic_json(result_path, result)
        stages[condition] = {"result": result, "predictions": predictions}
        del hook, expert
        torch.cuda.empty_cache()
    if stages[A_CONTINUATIONS[0]]["result"]["start_state_sha256"] != stages[A_CONTINUATIONS[1]]["result"]["start_state_sha256"]:
        raise RuntimeError("A1/A2 did not share the same A0 state")
    result = {"schema_version": "medtrace-a-multiseed-task-private-v1", "status": "RAW_READY", "task": task, "locks": locks, "stages": stages}
    atomic_json(out / "result_private.json", result)
    return result


def execute_b(runtime: Any, frozen: dict[str, Any], task: dict[str, Any], out: Path, run_root: Path, locks: dict[str, str]) -> dict[str, Any]:
    event = frozen["dev"][task["event_index"] - 1]
    scope = frozen["scopes"][task["record_id"]]
    cache = feature_cache(runtime, event, scope, run_root / f"private/features/e{task['event_index']:02d}.pt", locks)
    fit_rows, fit_prompt, fit_visual = ordered_features(cache, roles=("fit",))
    fit_count = sum(value["row"]["label"] == "positive" for value in fit_rows)
    if fit_count != 4 or len(fit_rows) <= fit_count:
        raise RuntimeError("router fit data coverage failure")
    a_dir = run_root / "private/tasks" / task["depends_on"] / "CP_NATIVE_PLUS_PARAPHRASE_80"
    a_checkpoint = torch.load(a_dir / "expert.pt", map_location="cuda:0", weights_only=True)
    initial_state = a_checkpoint["expert"]
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    for representation in REPRESENTATIONS:
        marker = out / f"{representation}.json"
        if marker.exists():
            candidates.append(json.loads(marker.read_text()))
            continue
        seed_everything(derive_seed(task["record_id"], base=task["seed"]) + REPRESENTATIONS.index(representation))
        value = train_router_candidate(initial_state, fit_prompt, fit_visual, fit_count, representation, out)
        atomic_json(marker, value)
        candidates.append(value)
    cal_rows, cal_prompt, cal_visual = ordered_features(cache, roles=("calibration",))
    calibrations = {}
    for representation in REPRESENTATIONS:
        for budget in BUDGETS:
            checkpoint = torch.load(out / f"{representation}__step{budget}.pt", map_location="cuda:0", weights_only=True)
            scored = score_checkpoint(checkpoint, cal_rows, cal_prompt, cal_visual)
            positive = [row["score"] for row in scored if row["label"] == "positive"]
            hard = [row["score"] for row in scored if row["fact_relation"] == "same_question_different_image_conflicting_source_answer"]
            broad = [row["score"] for row in scored if row["fact_relation"] == "broad_unrelated_source_qa"]
            points = calibrate_operating_points(positive, hard, broad, hard_evaluable=scope["status"] == "HARD_EVALUABLE")
            calibrations[f"{representation}__step{budget}"] = {"scores": scored, "operating_points": points, "hard_evaluable": scope["status"] == "HARD_EVALUABLE"}
    result = {"schema_version": "medtrace-router-candidates-private-v1", "status": "COMPLETE", "task": task, "locks": locks, "scope_status": scope["status"], "initial_a2_state_sha256": state_hash(initial_state), "candidates": candidates, "calibrations": calibrations}
    atomic_json(out / "result_private.json", result)
    return result


def select_profiles(run_root: Path, queue: TaskQueue, public_dir: Path) -> bool:
    lock_path = run_root / "private/select.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        private_path = run_root / "private/SELECTED_PROFILE.json"
        if private_path.exists():
            return True
        tasks = [row for row in queue.snapshot()["tasks"] if row["kind"] == "B"]
        if not tasks or any(row["status"] not in {"COMPLETE", "FAILED", "CANCELLED_BY_BUDGET"} for row in tasks):
            return False
        results = [json.loads((run_root / "private/tasks" / row["task_id"] / "result_private.json").read_text()) for row in tasks if row["status"] == "COMPLETE"]
        if not results:
            return False

        def aggregate(representation: str, budget: int, operating_point: str) -> dict[str, Any]:
            values = []
            for result in results:
                cal = result["calibrations"][f"{representation}__step{budget}"]
                point = cal["operating_points"][operating_point]
                values.append({**point, "hard_evaluable": cal["hard_evaluable"]})
            hard = [row["hard_fpr"] for row in values if row["hard_evaluable"]]
            return {
                "representation": representation, "budget": budget, "operating_point": operating_point,
                "positive_tpr": mean(row["positive_tpr"] for row in values),
                "hard_fpr": mean(hard) if hard else None,
                "broad_fpr": mean(row["broad_fpr"] for row in values), "task_count": len(values), "hard_task_count": len(hard),
            }

        profiles = [aggregate(rep, budget, point) for rep in REPRESENTATIONS for budget in BUDGETS for point in OPERATING_POINTS]

        def pick(candidates: list[dict[str, Any]]) -> dict[str, Any]:
            covered = [row for row in candidates if row["positive_tpr"] >= 0.75]
            pool = covered or candidates
            return min(pool, key=lambda row: (
                row["hard_fpr"] if row["hard_fpr"] is not None else 1.0,
                row["broad_fpr"], -row["positive_tpr"], row["budget"], row["representation"], row["operating_point"],
            ))

        selected = {
            "schema_version": "medtrace-selected-route-profile-v1", "status": "FROZEN_BEFORE_EVALUATION_SCORING",
            "selection_source": "calibration_only", "r0": pick([row for row in profiles if row["representation"] == REPRESENTATIONS[0]]),
            "alternative": pick([row for row in profiles if row["representation"] in REPRESENTATIONS[1:]]), "all_calibration_profiles": profiles,
        }
        atomic_json(private_path, selected)
        public = {**selected, "private_thresholds_withheld": True}
        atomic_json(public_dir / "SELECTED_PROFILE.json", public)
        return True


def route_score_one(expert: AsymmetricCPExpert, checkpoint: dict[str, Any], prompt: torch.Tensor, visual: torch.Tensor) -> float:
    score = score_with_frozen_prototypes(expert, prompt[None], [visual], checkpoint["representation"], checkpoint["prototypes"])
    return float(score[0].item())


def output_fit(runtime: Any, expert: AsymmetricCPExpert, record: EditorRecord, questions: list[str]) -> dict[str, Any]:
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    for parameter in (expert.u_out, expert.v_out, expert.rho):
        parameter.requires_grad_(True)
    q_hash = tensor_sha256(expert.input_basis())
    optimizer = torch.optim.AdamW([expert.u_out, expert.v_out, expert.rho], lr=1e-3, weight_decay=0)
    native = runtime.build_edit_batch(record)
    paraphrases = [runtime.build_edit_batch(replace(record, question=value)) for value in questions]
    hook = MedTraceLayerHook(runtime.get_module(LAYER), expert)
    hook.attach()
    started = time.monotonic()
    try:
        for step in range(1, 81):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for batch in (native, paraphrases[(step - 1) % len(paraphrases)]):
                hook.set_teacher_routing(batch.labels)
                loss = runtime.compute_loss(batch)
                (0.5 * loss).backward()
                losses.append(float(loss.item()))
            grad_norm = float(torch.nn.utils.clip_grad_norm_([expert.u_out, expert.v_out, expert.rho], 1.0).item())
            if not math.isfinite(grad_norm) or any(not math.isfinite(value) for value in losses):
                raise FloatingPointError("non-finite selected-profile output fit")
            optimizer.step()
            expert.normalize_output_factors_(verify_dense=step in (1, 40, 80))
            if tensor_sha256(expert.input_basis()) != q_hash:
                raise RuntimeError("selected-profile output fit changed frozen Q")
    finally:
        hook.detach()
    return {"steps": 80, "micro_forwards": 160, "elapsed_seconds": time.monotonic() - started, "q_sha256_before_and_after": q_hash}


def lifecycle(output: dict[str, Any]) -> dict[str, Any]:
    trace = output["request_lifecycle"]
    return {"hook_executed": bool(trace), "max_active_residual_norm": max((row.get("active_residual_norm", 0.0) for row in trace), default=0.0)}


def execute_e2e(runtime: Any, frozen: dict[str, Any], task: dict[str, Any], out: Path, run_root: Path, locks: dict[str, str]) -> dict[str, Any]:
    event = frozen["dev"][task["event_index"] - 1]
    scope = frozen["scopes"][task["record_id"]]
    cache = feature_cache(runtime, event, scope, run_root / f"private/features/e{task['event_index']:02d}.pt", locks)
    selection = json.loads((run_root / "private/SELECTED_PROFILE.json").read_text())
    b_dir = run_root / "private/tasks" / task["depends_on"]
    a_task = next(row for row in TaskQueue(run_root / "private/TASK_QUEUE.json", run_root).snapshot()["tasks"] if row["task_id"] == task["depends_on"])["depends_on"]
    a_dir = run_root / "private/tasks" / a_task / "CP_NATIVE_PLUS_PARAPHRASE_80"
    a2 = torch.load(a_dir / "expert.pt", map_location="cuda:0", weights_only=True)
    out.mkdir(parents=True, exist_ok=True)
    profiles = {"R0": selection["r0"], "ALT": selection["alternative"]}
    experts, profile_meta = {}, {}
    for label, profile in profiles.items():
        checkpoint_path = b_dir / f"{profile['representation']}__step{profile['budget']}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0", weights_only=True)
        expert = AsymmetricCPExpert(14336, 4096, 4).to("cuda:0")
        expert.load_state_dict(a2["expert"])
        expert.u_in.data.copy_(checkpoint["expert"]["u_in"])
        expert.v_in.data.copy_(checkpoint["expert"]["v_in"])
        cal = json.loads((b_dir / "result_private.json").read_text())["calibrations"][f"{profile['representation']}__step{profile['budget']}"]["operating_points"][profile["operating_point"]]
        fit = output_fit(runtime, expert, EditorRecord.from_dict(event["edit_record"]), [row["question"] for row in frozen["generality_paraphrases"][task["record_id"]]])
        checkpoint["expert"] = expert.state_dict()
        checkpoint_sha = save_checkpoint(out / f"{label}.pt", checkpoint)
        experts[label], profile_meta[label] = (expert, checkpoint), {"profile": profile, "calibration": cal, "output_fit": fit, "checkpoint_sha256": checkpoint_sha, "prototype_bytes": sum(value.numel() * value.element_size() for value in checkpoint["prototypes"].values())}

    evaluation_values = [value for value in cache["values"].values() if value["row"]["role"] in {"evaluation", "challenge", "native", "formal_development"}]
    evaluation_values.sort(key=lambda value: value["row"]["logical_id"])
    candidate_scores = []
    for candidate_file in sorted(b_dir.glob("R[012]_*.pt")):
        checkpoint = torch.load(candidate_file, map_location="cuda:0", weights_only=True)
        expert = AsymmetricCPExpert(14336, 4096, 4).to("cuda:0")
        expert.load_state_dict(checkpoint["expert"])
        candidate_scores.extend({"candidate": candidate_file.stem, "logical_id": value["row"]["logical_id"], "score": route_score_one(expert, checkpoint, value["prompt"].to("cuda:0"), value["visual"].to("cuda:0"))} for value in evaluation_values)
    atomic_json(out / "candidate_evaluation_scores_private.json", candidate_scores)

    outputs, off_validated = [], {label: False for label in profiles}
    for value in evaluation_values:
        row = value["row"]
        base = scope_generate(runtime, row, None)
        paths = {"base": base}
        decisions = {}
        for label, (expert, checkpoint) in experts.items():
            threshold = profile_meta[label]["calibration"]["threshold"]
            score = route_score_one(expert, checkpoint, value["prompt"].to("cuda:0"), value["visual"].to("cuda:0"))
            on = score > threshold
            hook = MedTraceLayerHook(runtime.get_module(LAYER), expert)
            hook.attach()
            try:
                forced = scope_generate(runtime, row, hook)
                if on:
                    gated = scope_generate(runtime, row, hook)
                    reuse = False
                else:
                    if not off_validated[label]:
                        replay = scope_generate(runtime, row, None)
                        if replay["raw_token_ids"] != base["raw_token_ids"] or replay["raw_answer"] != base["raw_answer"]:
                            raise RuntimeError("real OFF-path base replay mismatch")
                        off_validated[label] = True
                    gated, reuse = dict(base), True
            finally:
                hook.detach()
            paths[f"{label}__forced"] = forced
            paths[f"{label}__gated"] = gated
            decisions[label] = {"score": score, "threshold": threshold, "margin": score - threshold, "on": on, "forced": lifecycle(forced), "gated": lifecycle(gated), "gated_base_reuse": reuse, "off_token_parity": on or (gated["raw_token_ids"] == base["raw_token_ids"] and gated["raw_answer"] == base["raw_answer"])}
        outputs.append({"row": row, "outputs": paths, "decisions": decisions})
    if not all(off_validated.values()):
        raise RuntimeError("selected profile had no real OFF-path validation input")
    guard = runtime.base_guard.verify() if runtime.base_guard else None
    if not guard or not guard["unchanged"]:
        raise RuntimeError("end-to-end base guard failed")
    result = {"schema_version": "medtrace-e2e-private-v1", "status": "RAW_READY", "task": task, "locks": locks, "scope_status": scope["status"], "profiles": profile_meta, "outputs": outputs, "base_guard": guard}
    atomic_json(out / "result_private.json", result)
    return result


def budget_exhausted(run_root: Path, queue: TaskQueue) -> bool:
    start = json.loads((run_root / "private/CAMPAIGN_START.json").read_text())["epoch"]
    wall = time.time() - start
    used = sum(row.get("elapsed_seconds", 0) for row in queue.snapshot()["tasks"] if row["status"] in {"COMPLETE", "RAW_READY", "FAILED"})
    return wall >= 20 * 3600 or used >= 48 * 3600 or (run_root / "STOP").exists() or STOP_REQUESTED


def worker(args: argparse.Namespace) -> None:
    physical, _ = verify_gpu()
    actual_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != args.expected_code_commit:
        raise RuntimeError("worker code commit mismatch")
    run_root = args.run_root
    queue = TaskQueue(run_root / "private/TASK_QUEUE.json", run_root)
    frozen = json.loads((run_root / "private/frozen_data.json").read_text())
    config = json.loads((run_root / "private/CAMPAIGN_RUNTIME_CONFIG.json").read_text())
    locks = {"code_commit": actual_commit, "config_sha256": sha256_file(run_root / "private/CAMPAIGN_RUNTIME_CONFIG.json"), "data_sha256": sha256_file(run_root / "private/frozen_data.json"), "runtime_lock_sha256": sha256_file(Path(config["runtime_lock"]))}
    base = {row["query_id"]: row for row in read_jsonl(Path(config["base_predictions"]))}
    runtime = load_real_runtime(argparse.Namespace(cpu_gate=Path(config["cpu_gate"])))
    worker_id = f"gpu{physical}"
    completed_here = 0
    try:
        with Telemetry(run_root / "private/GPU_TELEMETRY.jsonl", physical, worker_id) as telemetry:
            while True:
                select_profiles(run_root, queue, Path(config["public_dir"]))
                if budget_exhausted(run_root, queue):
                    queue.cancel_pending()
                    break
                task = queue.claim(worker_id)
                if task is None:
                    snapshot = queue.snapshot()["tasks"]
                    if all(row["status"] in {"COMPLETE", "RAW_READY", "FAILED", "CANCELLED_BY_BUDGET"} for row in snapshot):
                        break
                    time.sleep(5)
                    continue
                telemetry.task_id = task["task_id"]
                started = time.monotonic()
                task_dir = run_root / "private/tasks" / task["task_id"]
                try:
                    if task["kind"] == "A":
                        execute_a(runtime, frozen, base, task, task_dir, locks)
                        status = "RAW_READY"
                    elif task["kind"] == "B":
                        execute_b(runtime, frozen, task, task_dir, run_root, locks)
                        status = "COMPLETE"
                    else:
                        execute_e2e(runtime, frozen, task, task_dir, run_root, locks)
                        status = "RAW_READY"
                    queue.update(task["task_id"], status, elapsed_seconds=time.monotonic() - started, finished_at=time.time())
                    completed_here += 1
                except torch.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    if task["attempts"] < 2:
                        queue.update(task["task_id"], "PENDING", last_error=f"OOM retry: {error}", elapsed_seconds=time.monotonic() - started)
                    else:
                        queue.update(task["task_id"], "FAILED", last_error=f"OOM after retry: {error}", elapsed_seconds=time.monotonic() - started)
                except Exception as error:
                    queue.update(task["task_id"], "FAILED", last_error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
                    append_jsonl(run_root / "private/WORKER_ERRORS.jsonl", {"task_id": task["task_id"], "worker": worker_id, "error_type": type(error).__name__, "error": str(error), "at": time.time()})
                finally:
                    telemetry.task_id = None
                    torch.cuda.empty_cache()
                if args.max_tasks and completed_here >= args.max_tasks:
                    break
    finally:
        del runtime
        torch.cuda.empty_cache()


def recover(args: argparse.Namespace) -> None:
    TaskQueue(args.run_root / "private/TASK_QUEUE.json", args.run_root).recover()


def prepare_judge(args: argparse.Namespace) -> None:
    if args.packet.exists() or args.sidecar.exists():
        raise FileExistsError("campaign Judge packet already exists")
    queue = TaskQueue(args.run_root / "private/TASK_QUEUE.json", args.run_root).snapshot()["tasks"]
    packet_by_id, sidecar = {}, []

    def add(question: str, reference: str, raw: str, metadata: dict[str, Any], *, exact: bool, cap_hit: bool) -> None:
        opaque = sha256_json((question, reference, raw, "medtrace-campaign-semantic-v1"))
        packet_by_id.setdefault(opaque, {"opaque_query_id": opaque, "question": question, "gold_answer": reference, "raw_base_answer": raw, "adjudication_pass": 1})
        sidecar.append({"opaque_query_id": opaque, "exact": exact, "cap_hit": cap_hit, **metadata})

    for task in queue:
        if task["status"] != "RAW_READY":
            continue
        result = json.loads((args.run_root / "private/tasks" / task["task_id"] / "result_private.json").read_text())
        if task["kind"] == "A":
            for condition, stage in result["stages"].items():
                for row in stage["predictions"]:
                    raw = row.get("decoded_text", row.get("raw_answer"))
                    exact = row.get("exact", row.get("exact_normalized_reference_match", False))
                    add(row["question"], row["reference"], raw, {"kind": "A", "task_id": task["task_id"], "seed": task["seed"], "event_index": task["event_index"], "edit_id": task["record_id"], "condition": condition, "task": row["task"], "query_id": row["query_id"]}, exact=bool(exact), cap_hit=bool(row.get("cap_hit", False)))
        elif task["kind"] == "E2E":
            for item in result["outputs"]:
                row = item["row"]
                for path, output in item["outputs"].items():
                    raw = output["raw_answer"]
                    add(row["question"], row["reference"], raw, {"kind": "E2E", "task_id": task["task_id"], "seed": task["seed"], "event_index": task["event_index"], "edit_id": task["record_id"], "scope_status": json.loads((args.run_root / "private/frozen_data.json").read_text())["scopes"][task["record_id"]]["status"], "logical_id": row["logical_id"], "role": row["role"], "label": row["label"], "fact_relation": row["fact_relation"], "path": path}, exact=normalize_medical_answer(raw) == normalize_medical_answer(row["reference"]), cap_hit=bool(output["reached_length_limit"] and not output["ended_with_eos"]))
    if not packet_by_id:
        raise RuntimeError("no raw campaign outputs are ready for Judge")
    args.packet.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(args.packet, "".join(json.dumps(row, sort_keys=True) + "\n" for row in packet_by_id.values()))
    atomic_json(args.sidecar, sidecar)


def bootstrap_mean(values: list[float], *, draws: int = 10000) -> tuple[float, float]:
    import random
    rng = random.Random(20260906)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return samples[int(0.025 * draws)], samples[int(0.975 * draws)]


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1%}"


def finalize(args: argparse.Namespace) -> None:
    public = args.public_dir
    public.mkdir(parents=True, exist_ok=True)
    queue_object = TaskQueue(args.run_root / "private/TASK_QUEUE.json", args.run_root)
    queue = queue_object.snapshot()["tasks"]
    verdict_rows = read_jsonl(args.judge_output)
    verdicts = {row["opaque_query_id"]: row for row in verdict_rows}
    sidecar = json.loads(args.sidecar.read_text())
    expected = {row["opaque_query_id"] for row in sidecar}
    if set(verdicts) != expected or any(not verdicts[key]["parse_valid"] for key in expected):
        raise RuntimeError("campaign Judge coverage or parsing failure")
    rows = [{**row, "semantic": bool(verdicts[row["opaque_query_id"]]["is_correct"])} for row in sidecar]

    a_rows = [row for row in rows if row["kind"] == "A"]
    a_csv = public / "A2_MULTISEED_RESULTS.csv"
    with a_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "event_index", "opaque_edit", "condition", "task", "n", "exact", "semantic", "cap_without_eos"])
        grouped = defaultdict(list)
        for row in a_rows:
            grouped[(row["seed"], row["event_index"], hashlib.sha256(row["edit_id"].encode()).hexdigest()[:16], row["condition"], row["task"])].append(row)
        for key, values in sorted(grouped.items()):
            writer.writerow([*key, len(values), sum(row["exact"] for row in values), sum(row["semantic"] for row in values), sum(row["cap_hit"] for row in values)])
    seed_deltas, edit_deltas = [], defaultdict(list)
    for seed in SEEDS:
        by_condition = {}
        for condition in (A_CONTINUATIONS[0], A_CONTINUATIONS[1]):
            per_edit = defaultdict(list)
            for row in a_rows:
                if row["seed"] == seed and row["condition"] == condition and row["task"] == "T2G":
                    per_edit[row["edit_id"]].append(row["semantic"])
            by_condition[condition] = {edit: mean(values) for edit, values in per_edit.items()}
        common = set(by_condition[A_CONTINUATIONS[0]]) & set(by_condition[A_CONTINUATIONS[1]])
        deltas = {edit: by_condition[A_CONTINUATIONS[1]][edit] - by_condition[A_CONTINUATIONS[0]][edit] for edit in common}
        seed_deltas.append(mean(deltas.values()))
        for edit, value in deltas.items():
            edit_deltas[edit].append(value)
    edit_means = [mean(values) for values in edit_deltas.values()]
    ci = bootstrap_mean(edit_means) if edit_means else (math.nan, math.nan)
    atomic_text(public / "A2_MULTISEED_PAIRED_REPORT.md", f"""# A2 multiseed paired development report

Status: `A2_MULTISEED_EVALUATION_COMPLETE`

- Seeds: {', '.join(map(str, SEEDS))}; all DEV16 edits remain paired within each completed seed.
- A2 minus equal-budget A1 T2G macro by seed: {', '.join(f'{value:+.1%}' for value in seed_deltas)}.
- Mean/std across seeds: {mean(seed_deltas):+.1%} / {pstdev(seed_deltas):.1%}.
- Edit-cluster paired bootstrap 95% interval: [{ci[0]:+.1%}, {ci[1]:+.1%}] using edit as the resampling unit.

This repeats an already observed development panel; seeds and probes are not independent patients or a blind confirmation set.
""")

    selection = json.loads((args.run_root / "private/SELECTED_PROFILE.json").read_text()) if (args.run_root / "private/SELECTED_PROFILE.json").exists() else None
    b_tasks = [row for row in queue if row["kind"] == "B" and row["status"] == "COMPLETE"]
    with (public / "ROUTER_CALIBRATION_RESULTS.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "event_index", "opaque_edit", "scope_status", "representation", "budget", "operating_point", "positive_tpr", "hard_fpr", "broad_fpr"])
        for task in b_tasks:
            result = json.loads((args.run_root / "private/tasks" / task["task_id"] / "result_private.json").read_text())
            for profile, value in result["calibrations"].items():
                representation, step = profile.rsplit("__step", 1)
                for point, metrics in value["operating_points"].items():
                    writer.writerow([task["seed"], task["event_index"], hashlib.sha256(task["record_id"].encode()).hexdigest()[:16], result["scope_status"], representation, step, point, metrics["positive_tpr"], metrics["hard_fpr"], metrics["broad_fpr"]])
    atomic_text(public / "ROUTER_REPRESENTATION_REPORT.md", f"""# Router representation development report

Status: `{'ROUTER_CALIBRATION_AND_SELECTION_COMPLETE' if selection else 'ROUTER_SELECTION_UNAVAILABLE'}`

The preregistered candidates were magnitude-only R0, signed-response R1, and visual-conditioned signed-response R2 at steps 200 and 800, each with SAFETY_FIRST and COVERAGE_CONSTRAINED thresholds. Selection used calibration aggregates only; evaluation scores were opened afterward.

- Selected R0: `{selection['r0']['representation']}@{selection['r0']['budget']}/{selection['r0']['operating_point']}` (calibration positive TPR {pct(selection['r0']['positive_tpr'])}, hard FPR {pct(selection['r0']['hard_fpr'])}, broad FPR {pct(selection['r0']['broad_fpr'])}).
- Selected alternative: `{selection['alternative']['representation']}@{selection['alternative']['budget']}/{selection['alternative']['operating_point']}` (calibration positive TPR {pct(selection['alternative']['positive_tpr'])}, hard FPR {pct(selection['alternative']['hard_fpr'])}, broad FPR {pct(selection['alternative']['broad_fpr'])}).

R1/R2 store fit-response prototypes and therefore add routing metadata; they are development extensions, not exact V0.2/TIME reproduction.
""" if selection else "# Router representation development report\n\nSelection was unavailable because no complete calibration block existed.\n")

    e_rows = [row for row in rows if row["kind"] == "E2E"]
    by_task = {row["task_id"]: json.loads((args.run_root / "private/tasks" / row["task_id"] / "result_private.json").read_text()) for row in queue if row["kind"] == "E2E" and row["status"] == "RAW_READY"}
    with (public / "END_TO_END_BY_EDIT_SEED.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "event_index", "opaque_edit", "scope_status", "profile", "subset", "n", "on", "base_correct", "forced_correct", "gated_correct", "forced_damage_base_correct", "gated_damage_base_correct", "off_count", "off_token_parity"])
        e2e_summary = []
        for task_id, result in sorted(by_task.items()):
            task = result["task"]
            judged = {(row["logical_id"], row["path"]): row["semantic"] for row in e_rows if row["task_id"] == task_id}
            for label in ("R0", "ALT"):
                subsets = defaultdict(list)
                for item in result["outputs"]:
                    row = item["row"]
                    subset = "positive" if row["label"] == "positive" and row["role"] in {"evaluation", "native"} else row["fact_relation"] if row["label"] == "negative" else row["role"]
                    subsets[subset].append(item)
                for subset, values in subsets.items():
                    base_correct = [item for item in values if judged[(item["row"]["logical_id"], "base")]]
                    decisions = [item["decisions"][label] for item in values]
                    metrics = [task["seed"], task["event_index"], hashlib.sha256(task["record_id"].encode()).hexdigest()[:16], result["profiles"][label]["profile"].get("scope_status", "DECLARED_IN_COVERAGE_REPORT"), label, subset, len(values), sum(row["on"] for row in decisions), len(base_correct), sum(judged[(item["row"]["logical_id"], f"{label}__forced")] for item in values), sum(judged[(item["row"]["logical_id"], f"{label}__gated")] for item in values), sum(not judged[(item["row"]["logical_id"], f"{label}__forced")] for item in base_correct), sum(not judged[(item["row"]["logical_id"], f"{label}__gated")] for item in base_correct), sum(not row["on"] for row in decisions), sum((not row["on"]) and row["off_token_parity"] for row in decisions)]
                    writer.writerow(metrics)
                    e2e_summary.append(metrics)
    atomic_text(public / "END_TO_END_FORCED_GATED_REPORT.md", "# End-to-end forced/gated development report\n\nStatus: `END_TO_END_EVALUATION_COMPLETE`\n\nPer edit, seed, profile and negative type, activation, semantic correctness, observed base-correct damage and OFF parity are in `END_TO_END_BY_EDIT_SEED.csv`. Gating protection is interpreted only where the matched forced-on expert caused observed damage. Broad, same-question/different-image, same-image/other-fact and formal development probes remain separate.\n")

    disagreements = [row for row in rows if row["exact"] != row["semantic"]]
    atomic_text(public / "JUDGE_EXECUTION_AND_DISAGREEMENT_REPORT.md", f"""# Judge execution and disagreement report

- Unique Judge payloads: {len(verdicts)}; mapped method outputs: {len(rows)}; parse-valid coverage: {sum(value['parse_valid'] for value in verdicts.values())}/{len(verdicts)}.
- Conservative normalized-exact versus semantic disagreements: {len(disagreements)}.
- Generation cap without EOS: {sum(row['cap_hit'] for row in rows)}.
- Judge model/prompt/runtime are bound by the private execution lock; repeated identical logical payloads were judged once.

The historical six exact/Judge disagreements remain unresolved and were not rewritten.
""")

    telemetry = read_jsonl(args.run_root / "private/GPU_TELEMETRY.jsonl") if (args.run_root / "private/GPU_TELEMETRY.jsonl").exists() else []
    start = json.loads((args.run_root / "private/CAMPAIGN_START.json").read_text())
    wall = time.time() - start["epoch"]
    gpu_seconds = sum(row.get("elapsed_seconds", 0) for row in queue if row["status"] in {"COMPLETE", "RAW_READY", "FAILED"})
    atomic_text(public / "GPU_THROUGHPUT_AND_BUDGET_REPORT.md", f"""# GPU throughput and bounded-budget report

- Wall time through finalization: {wall / 3600:.3f} hours; summed task GPU time: {gpu_seconds / 3600:.3f} GPU-hours.
- Telemetry samples: {len(telemetry)} at the configured 60-second interval.
- Mean sampled GPU active utilization: {mean(row['gpu_utilization_percent'] for row in telemetry):.1f}%.
- Peak sampled allocated/reserved: {max(row['torch_allocated_bytes'] for row in telemetry) / 2**30:.2f}/{max(row['torch_reserved_bytes'] for row in telemetry) / 2**30:.2f} GiB.
- Sampled zero-utilization share (includes CPU/IO/queue waits): {mean(row['gpu_utilization_percent'] == 0 for row in telemetry):.1%}.

These are observed samples, not reconstructed utilization claims. GPU1 was forbidden and unused.
""" if telemetry else "# GPU throughput and bounded-budget report\n\nNo telemetry samples were available; utilization was not invented.\n")

    for row in queue:
        if row["status"] == "RAW_READY":
            queue_object.update(row["task_id"], "COMPLETE", judged_at=time.time())
    final_queue = queue_object.snapshot()["tasks"]
    public_ledger = [{key: row.get(key) for key in ("task_id", "kind", "seed", "event_index", "priority", "status", "attempts", "worker", "elapsed_seconds", "last_error") if key in row} for row in final_queue]
    atomic_text(public / "TASK_LEDGER.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in public_ledger))
    artifacts = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in public.iterdir() if path.is_file() and path.name not in {"RAW_CLOSURE_MANIFEST.json", "CAMPAIGN_COMPLETION.json", "GPT_PRO_REVIEW.md"}}
    atomic_json(public / "RAW_CLOSURE_MANIFEST.json", {"schema_version": "medtrace-campaign-raw-closure-public-v1", "unique_judge_payloads": len(verdicts), "mapped_outputs": len(rows), "task_status_counts": dict(Counter(row["status"] for row in final_queue)), "public_artifacts": artifacts, "private_raw_withheld": True})
    partial = any(row["status"] in {"FAILED", "CANCELLED_BY_BUDGET", "PENDING", "RUNNING"} for row in final_queue)
    status = "CAMPAIGN_BUDGET_EXHAUSTED__PARTIAL_RESULTS_PUBLISHED" if partial else "MEDTRACE_ROUTE_REPRESENTATION_CAMPAIGN_COMPLETE"
    atomic_json(public / "CAMPAIGN_COMPLETION.json", {"schema_version": "medtrace-route-representation-campaign-completion-v1", "status": status, "generation": "COMPLETE_FOR_REPORTED_TASKS", "judge": "COMPLETE_FOR_REPORTED_TASKS", "old_artifacts_modified": False, "gpu1_used": False, "private_artifacts_withheld": True, "task_status_counts": dict(Counter(row["status"] for row in final_queue))})
    alt = selection["alternative"] if selection else None
    atomic_text(public / "GPT_PRO_REVIEW.md", f"""# GPT Pro review: MedTRACE bounded route-representation campaign

This is a development campaign, not full TIME, V0.2 qualification, clinical validation, or a blind benchmark. Historical LoRA qualification remains FAIL.

1. Multiseed A2 evidence: see `A2_MULTISEED_PAIRED_REPORT.md` and `A2_MULTISEED_RESULTS.csv`.
2. Selected route alternative: `{alt['representation'] if alt else 'UNAVAILABLE'}`; selection evidence is in `SELECTED_PROFILE.json` and `ROUTER_CALIBRATION_RESULTS.csv`.
3. Actual activation, hard/broad FPR, forced/gated correctness and damage are in `END_TO_END_BY_EDIT_SEED.csv`.
4. Remaining concrete question: does the selected representation reduce same-question/different-image activation at preserved positive joint edit success across hard-evaluable edits?

Status: `{status}`. Low scores, all-OFF profiles and failed tasks remain in the ledger rather than being discarded.
""")


def signal_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-data")
    for name in ("dev-inputs", "qual-inputs", "formal-records", "formal-probes", "slake-train", "previous-data", "private-out", "public-counts", "tasks"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.set_defaults(func=prepare_data)
    recover_parser = sub.add_parser("recover")
    recover_parser.add_argument("--run-root", type=Path, required=True)
    recover_parser.set_defaults(func=recover)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--run-root", type=Path, required=True)
    worker_parser.add_argument("--expected-code-commit", required=True)
    worker_parser.add_argument("--max-tasks", type=int, default=0)
    worker_parser.set_defaults(func=worker)
    judge = sub.add_parser("prepare-judge")
    judge.add_argument("--run-root", type=Path, required=True)
    judge.add_argument("--packet", type=Path, required=True)
    judge.add_argument("--sidecar", type=Path, required=True)
    judge.set_defaults(func=prepare_judge)
    final = sub.add_parser("finalize")
    final.add_argument("--run-root", type=Path, required=True)
    final.add_argument("--judge-output", type=Path, required=True)
    final.add_argument("--sidecar", type=Path, required=True)
    final.add_argument("--public-dir", type=Path, required=True)
    final.set_defaults(func=finalize)
    return value


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_stop)
    options = parser().parse_args()
    options.func(options)
