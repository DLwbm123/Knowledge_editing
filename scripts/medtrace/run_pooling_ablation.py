#!/usr/bin/env python3
"""CP-independent pooling; reuse the frozen-verifier queue, runtime and Judge closure."""
import argparse
import csv
import json
from pathlib import Path
import signal
import sys
import time
from collections import defaultdict
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace import run_frozen_expert_visual_verifier as vf
from methods.medtrace.visual_pooling import GROUPS, pooled_feature, fit_image_head
from methods.medtrace.frozen_verifier import l2_normalize


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty report: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def point_for(result, panel, condition, point):
    if panel == "original" and condition == vf.CONDITIONS[0]:
        point = "CONTINUITY_SAFETY_FIRST" if point == vf.POINTS[0] else "CONTINUITY_COVERAGE_CONSTRAINED"
    return result["calibration"][panel][condition][point]


def rejection_analysis(args):
    """Existing scores and outputs only: no model loading and no Judge request."""
    results = vf._task_results(args.verifier_run)
    if len(results) != 21:
        raise RuntimeError("historical verifier closure is incomplete")
    public = args.public_dir
    public.mkdir(parents=True, exist_ok=True)
    counts, margins, transitions = [], [], []
    for result in results:
        task = result["task"]
        for point in vf.POINTS:
            qth, ith = result["calibration"]["matched"][vf.CONDITIONS[3]][point]["thresholds"]
            count = dict(event_index=task["event_index"], seed=task["seed"], operating_point=point,
                         positive_n=0, on=0, question_only=0, image_only=0, both=0, question_only_upper_on=0,
                         question_threshold=qth, image_threshold=ith)
            for lid, item in result["outputs"]["matched"].items():
                row = item["row"]
                q, v = result["scores"]["matched"][lid][vf.CONDITIONS[3]]
                q_on, v_on = q > qth, v > ith
                if row["label"] == "positive":
                    count["positive_n"] += 1
                    count["on"] += q_on and v_on
                    count["question_only"] += not q_on and v_on
                    count["image_only"] += q_on and not v_on
                    count["both"] += not q_on and not v_on
                    count["question_only_upper_on"] += q_on
                elif q_on and v_on:
                    margins.append(dict(event_index=task["event_index"], seed=task["seed"], operating_point=point,
                                        source_group=row["source_group"], logical_id=lid, question_score=q, image_score=v,
                                        question_threshold=qth, image_threshold=ith, question_margin=q-qth, image_margin=v-ith))
            counts.append(count)
            for panel, outputs in result["outputs"].items():
                by_category = defaultdict(lambda: dict(n=0, gain=0, loss=0, same_on=0, same_off=0))
                for lid, item in outputs.items():
                    row = item["row"]
                    decisions = [vf._decision(result["scores"][panel][lid][c], point_for(result, panel, c, point)) for c in (vf.CONDITIONS[0], vf.CONDITIONS[3])]
                    before, after = decisions
                    cell = by_category[vf._category(panel, row)]
                    cell["n"] += 1; cell["same_on"] += before and after; cell["same_off"] += not before and not after
                    positive = row["label"] == "positive"
                    cell["gain"] += (not before and after) if positive else (before and not after)
                    cell["loss"] += (before and not after) if positive else (not before and after)
                for category, cell in by_category.items():
                    transitions.append(dict(event_index=task["event_index"], seed=task["seed"], operating_point=point, panel=panel, category=category, **cell))
    write_csv(public / "REJECTION_BY_EDIT_SEED.csv", counts)
    write_csv(public / "HISTORICAL_HARD_FP_MARGINS.csv", margins)
    write_csv(public / "HISTORICAL_M0_M3_PAIRED_TRANSITIONS.csv", transitions)
    lines = ["# Historical M3 rejection decomposition", "", "Read-only diagnostic from the completed recovery run. No new model or Judge call. Neither operating point replaces the historical PRIMARY result.", "",
             "|Point|Positive N|ON|Question-only reject|Image-only reject|Both reject|Image-always-ON diagnostic upper bound|", "|---|---:|---:|---:|---:|---:|---:|"]
    for point in vf.POINTS:
        rows = [r for r in counts if r["operating_point"] == point]
        values = [sum(r[k] for r in rows) for k in ("positive_n", "on", "question_only", "image_only", "both", "question_only_upper_on")]
        lines.append("|" + "|".join(map(str, [point, *values])) + "|")
    lines += ["", "Per-edit/seed counts and both thresholds: REJECTION_BY_EDIT_SEED.csv. Hard false-positive head scores and margins: HISTORICAL_HARD_FP_MARGINS.csv. Paired ON/OFF gains and losses (positive activation / negative rejection, not semantic accuracy): HISTORICAL_M0_M3_PAIRED_TRANSITIONS.csv.",
              "The upper bound is a question-gate diagnostic at historical frozen thresholds, not a method score. New two-head calibration may change both thresholds even though question scores are identical. Historical scientific_gain=false remains unchanged."]
    vf.atomic_text(public / "REJECTION_BRANCH_DECOMPOSITION.md", "\n".join(lines) + "\n")


def prepare(args):
    vf.prepare(args)
    config_path = args.run_root / "private/CAMPAIGN_CONFIG.json"
    config = read(config_path)
    config.update(campaign_kind="pooling", verifier_run=str(args.verifier_run.resolve()), fit_variants=list(GROUPS),
                  conditions=list(GROUPS), closure_reserve_seconds=7200,
                  pooling_source_public_commit="3ce3d95e393eba8a6eda6dff0d6e7ad8ef9a9f1a")
    vf.atomic_json(config_path, config)
    tasks = read(args.run_root / "private/TASK_QUEUE.json")["tasks"]
    vf.atomic_json(args.run_root / "private/INPUT_MANIFEST.json", vf.campaign_inputs(config, tasks))
    public_config = read(args.public_dir / "EXPERIMENT_CONFIG.json")
    public_config.update(conditions=list(GROUPS), fits=63, frozen_question_head="original M3 per edit/seed", trainable="image head only",
                         pooling={"G0": "CP-guided matched refit", "G1": "uniform RMS visual-token mean", "G2": "pre-CP cosine query-conditioned softmax tau=0.1"},
                         source_public_commit=config["pooling_source_public_commit"], closure_reserve_seconds=7200,
                         selection_point="PRIMARY_SAFETY_FIRST", historical_scientific_gain=False,
                         thresholds="same frozen calibrate_decisions; both thresholds independently recalibrated per condition/panel/point",
                         confirmation="only after full development retention; at most 16 authorized unseen edits, complete C1 recipe, one fixed seed")
    vf.atomic_json(args.public_dir / "EXPERIMENT_CONFIG.json", public_config)
    vf.atomic_json(args.public_dir / "POOLING_ABLATION_CONFIG.json", public_config)
    vf.atomic_text(args.public_dir / "CAMPAIGN_PROTOCOL.md", "# CP-independent visual pooling\n\nNew development experiment, 7 frozen hard-evaluable facts, 3 seeds, 21 packages, 63 image-only fits. C1 and original M3 question heads frozen. G0 is a matched image-only refit, not old joint M3. No historical files modified. No evaluation checkpoint selection. PRIMARY selects once using the original full M0 retention gate; SECONDARY is diagnostic only. Preserve at least two hours for closure.\n")
    rejection_analysis(args)


def process_task(runtime, args, task):
    started = time.monotonic()
    config = read(args.run_root / "private/CAMPAIGN_CONFIG.json")
    source = Path(config["verifier_run"])
    prior_dir = source / "private/tasks" / task["task_id"]
    prior = read(prior_dir / "result_private.json")
    if vf.task_identity(prior["task"]) != vf.task_identity(task) or not prior["base_guard"]["unchanged"]:
        raise RuntimeError("prior task binding mismatch")
    c1 = args.execution_run / "private/tasks" / f"P0_s{task['seed']}_e{task['event_index']:02d}" / "C1_R2_FIXED_Q_LONG_RECOVERY/step0800.pt"
    if vf.sha256_file(c1) != prior["executor_lock"]["checkpoint_sha256"]:
        raise RuntimeError("C1 checkpoint mismatch")
    checkpoint = torch.load(c1, map_location="cuda:0", weights_only=True)
    expert = vf.AsymmetricCPExpert(14336, 4096, 4).to("cuda:0")
    expert.load_state_dict(checkpoint["expert"]); expert.eval(); expert.requires_grad_(False)
    expert_hash = vf.state_hash(expert.state_dict())
    if expert_hash != prior["executor_lock"]["state_sha256"]:
        raise RuntimeError("C1 state mismatch")
    saved = torch.load(prior_dir / "M3_heads.pt", map_location="cpu", weights_only=False)
    question = {k.removeprefix("question."): v for k, v in saved["state"].items() if k.startswith("question.")}
    question_hash = vf.state_hash(question)
    caches = {
        "original": torch.load(args.old_run / f"private/features/e{task['event_index']:02d}.pt", map_location="cpu", weights_only=False),
        "matched": torch.load(source / f"private/features/e{task['event_index']:02d}.pt", map_location="cpu", weights_only=False),
    }
    if caches["matched"]["locks"] != prior["matched_feature_locks"] or any(caches["matched"]["locks"].get(k) != v for k, v in caches["original"]["locks"].items()):
        raise RuntimeError("raw activation cache binding mismatch")
    features, pooling_cost = {}, {}
    question_check = torch.nn.Linear(14336, 1).to("cuda:0")
    question_check.load_state_dict(question); question_check.requires_grad_(False)
    shape_stats = set()
    with torch.no_grad():
        for panel, cache in caches.items():
            features[panel] = {}
            for lid, value in cache["values"].items():
                prompt, visual = value["prompt"].to("cuda:0"), value["visual"].to("cuda:0")
                if prompt.shape != (14336,) or visual.ndim != 2 or visual.shape[1] != 14336:
                    raise RuntimeError("pre-CP activation shape mismatch")
                shape_stats.add((str(prompt.dtype), str(visual.dtype), len(visual)))
                qscore = question_check(l2_normalize(expert.normalize_activation(prompt))).item()
                if abs(qscore - prior["scores"][panel][lid][vf.CONDITIONS[3]][0]) > 1e-5:
                    raise RuntimeError("frozen question score mismatch")
                features[panel][lid] = {}
                for kind in GROUPS:
                    torch.cuda.synchronize(); begin = time.monotonic()
                    features[panel][lid][kind] = pooled_feature(expert, prompt, visual, kind)
                    torch.cuda.synchronize()
                    pooling_cost.setdefault(kind, []).append(time.monotonic() - begin)
    feature_seconds = time.monotonic() - started
    task_dir = args.run_root / "private/tasks" / task["task_id"]
    fit_rows = sorted((v["row"] for v in caches["matched"]["values"].values() if v["row"]["role"] == "fit"), key=lambda r: r["logical_id"])
    verifiers, training = {}, {}
    began = time.monotonic()
    for kind in GROUPS:
        x = torch.stack([features["matched"][r["logical_id"]][kind] for r in fit_rows])
        verifier, curve = fit_image_head(question, x, fit_rows)
        if vf.state_hash(verifier.question.state_dict()) != question_hash:
            raise RuntimeError("changed frozen question hash")
        verifiers[kind] = verifier
        training[kind] = vf._head_report(verifier, curve, task_dir / f"{kind}_heads.pt") | {
            "fit_source": "NEW_IMAGE_ONLY_800_STEP_FIT", "input_width": 14336,
            "trainable_parameter_count": 14337, "trainable_parameter_bytes": 57348,
            "frozen_question_sha256": question_hash, "image_examples": len(fit_rows),
            "image_positive": sum(r["label"] == "positive" for r in fit_rows),
            "fit_order_sha256": vf.sha256_json(fit_rows),
        }
    training_seconds = time.monotonic() - began
    if vf.state_hash(expert.state_dict()) != expert_hash:
        raise RuntimeError("changed frozen C1 Q/P/rho")
    scores = {}
    with torch.no_grad():
        for panel, values in features.items():
            scores[panel] = {lid: {kind: [prior["scores"][panel][lid][vf.CONDITIONS[3]][0], verifiers[kind].image(f[kind]).item()] for kind in GROUPS} for lid, f in values.items()}
    calibration = {panel: vf._calibrate_panel(cache, scores[panel], caches["original"], scores["original"], GROUPS) for panel, cache in caches.items()}
    rows = sorted((item["row"] for item in prior["outputs"]["matched"].values()), key=lambda r: r["logical_id"])
    def generate(row, desired):
        if not desired:
            return vf.scope_generate(runtime, row, None)
        hook = vf.MedTraceLayerHook(runtime.get_module(vf.LAYER), expert); hook.attach()
        try:
            return vf.scope_generate(runtime, row, hook)
        finally:
            hook.detach()
    replay_started = time.monotonic()
    replays = vf.natural_replays(rows, scores["matched"], calibration["matched"], prior["outputs"]["matched"], generate, GROUPS)
    # A forced branch fixture is not a natural request and does not alter thresholds.
    for desired in (True, False):
        if any(r["status"] == f"NO_NATURAL_{'ON' if desired else 'OFF'}_IN_THIS_TASK" for r in replays):
            actual = generate(rows[0], desired)
            expected = prior["outputs"]["matched"][rows[0]["logical_id"]]["forced" if desired else "base"]
            if any(actual[k] != expected[k] for k in ("raw_answer", "raw_token_ids")):
                raise RuntimeError("independent branch fixture replay differs")
            replays.append(dict(condition="INDEPENDENT_FIXTURE", decision="ON" if desired else "OFF", status="VERIFIED", exact_replay=True))
    guard = runtime.base_guard.verify()
    if not guard["unchanged"]:
        raise RuntimeError("base guard failed")
    result = dict(schema_version="medtrace-pooling-task-private-v1", status="COMPLETE", task=task,
                  attempt_id=args.run_root.name, execution_code_commit=args.expected_code_commit,
                  executor_lock=prior["executor_lock"], matched_feature_locks=prior["matched_feature_locks"],
                  cache_binding={"source_result_sha256": vf.sha256_file(prior_dir / "result_private.json"),
                                 "source_head_sha256": vf.sha256_file(prior_dir / "M3_heads.pt"),
                                 "question_state_sha256": question_hash, "activation_shapes_dtype_tokens": sorted(shape_stats),
                                 "feature_definition": "expert-OFF target-free layer21 down_proj raw activations; original exact tokens/preprocess/runtime locks"},
                  training=training, calibration=calibration, scores=scores, outputs=prior["outputs"],
                  row_metadata={panel: {lid: v["row"] for lid, v in cache["values"].items()} for panel, cache in caches.items()},
                  replays=replays, base_guard=guard,
                  timing={"feature_seconds": feature_seconds, "training_seconds": training_seconds, "replay_seconds": time.monotonic()-replay_started,
                          "pooling_mean_seconds_per_row": {k: mean(v) for k, v in pooling_cost.items()}, "total_seconds": time.monotonic()-started})
    vf.atomic_json(task_dir / "result_private.json", result)
    return result


def worker(args):
    vf.worker(args, process_task)


def finalize(args):
    from scripts.medtrace.summarize_pooling_ablation import finalize as summarize
    summarize(args)


if __name__ == "__main__":
    parser = vf.parser()
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices
    subparsers["prepare"].add_argument("--verifier-run", type=Path, required=True)
    subparsers["prepare"].set_defaults(func=prepare)
    subparsers["worker"].set_defaults(func=worker)
    subparsers["finalize"].set_defaults(func=finalize)
    signal.signal(signal.SIGTERM, vf.signal_stop)
    args = parser.parse_args(); args.func(args)
