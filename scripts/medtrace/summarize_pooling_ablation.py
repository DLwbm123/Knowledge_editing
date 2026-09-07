"""Read-only evaluation of completed pooling fits and exact reused outputs."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import time

from scripts.medtrace import run_frozen_expert_visual_verifier as vf
from scripts.medtrace.run_pooling_ablation import read, write_csv, point_for
from methods.medtrace.visual_pooling import GROUPS


def finalize(args):
    results = vf._task_results(args.run_root)
    if len(results) != 21:
        raise RuntimeError("pooling requires all 21 packages / 63 fits")
    config = read(args.run_root / "private/CAMPAIGN_CONFIG.json")
    prior = {r["task"]["task_id"]: r for r in vf._task_results(Path(config["verifier_run"]))}
    verdicts = vf._verdicts(args.sidecar, args.judge_output)
    if read(args.sidecar)["new_count"]:
        raise RuntimeError("expected exact Judge reuse without new raw tuples")
    detailed, reject, timings = [], [], []
    for result in results:
        task = result["task"]
        old = prior[task["task_id"]]
        if result["outputs"] != old["outputs"]:
            raise RuntimeError("derived output source mismatch")
        for kind in GROUPS:
            timings.append(dict(event_index=task["event_index"], seed=task["seed"], condition=kind,
                                pooling_mean_seconds_per_row=result["timing"]["pooling_mean_seconds_per_row"][kind],
                                trainable_parameters=14337, total_parameters=28674, parameter_bytes=114696))
        for panel, outputs in result["outputs"].items():
            for lid, item in outputs.items():
                row = item["row"]
                base, forced = item["base"]["raw_answer"], item["forced"]["raw_answer"]
                be = vf.normalize_medical_answer(base) == vf.normalize_medical_answer(row["reference"])
                fe = vf.normalize_medical_answer(forced) == vf.normalize_medical_answer(row["reference"])
                bs, fs = vf._semantic(verdicts, row, base), vf._semantic(verdicts, row, forced)
                for kind in (*GROUPS, vf.CONDITIONS[0], vf.CONDITIONS[3]):
                    source = result if kind in GROUPS else old
                    for point in vf.POINTS:
                        scores = source["scores"][panel][lid][kind]
                        thresholds = point_for(source, panel, kind, point)["thresholds"]
                        branches = [s > t for s, t in zip(scores, thresholds, strict=True)]
                        on = all(branches)
                        detailed.append(dict(event_index=task["event_index"], seed=task["seed"], condition=kind, panel=panel,
                            operating_point=point, logical_id=lid, category=vf._category(panel, row),
                            source_group=row.get("source_group") or vf.sha256_json(row["image_path"])[:16],
                            on=on, base_exact=be, forced_exact=fe, gated_exact=fe if on else be,
                            base_semantic=bs, forced_semantic=fs, gated_semantic=fs if on else bs,
                            joint_on_semantic=on and fs, gated_raw_source="forced" if on else "base",
                            raw_changed=on and (base != forced or item["base"]["raw_token_ids"] != item["forced"]["raw_token_ids"]),
                            selected_generation_seconds=item["forced" if on else "base"].get("elapsed_seconds"),
                            question_only_reject=len(branches)==2 and not branches[0] and branches[1],
                            image_only_reject=len(branches)==2 and branches[0] and not branches[1],
                            both_reject=len(branches)==2 and not any(branches),
                            question_upper_on=branches[0], question_upper_joint=branches[0] and fs,
                            question_threshold=thresholds[0], image_threshold=thresholds[1] if len(thresholds)==2 else None))
        for panel in ("matched", "original"):
            for kind in GROUPS:
                for point in vf.POINTS:
                    rows = [r for r in detailed if r["event_index"]==task["event_index"] and r["seed"]==task["seed"] and r["condition"]==kind and r["panel"]==panel and r["operating_point"]==point]
                    for category in sorted({r["category"] for r in rows}):
                        values = [r for r in rows if r["category"]==category]
                        reject.append(dict(event_index=task["event_index"], seed=task["seed"], condition=kind, panel=panel, operating_point=point, category=category,
                            n=len(values), question_threshold=values[0]["question_threshold"], image_threshold=values[0]["image_threshold"],
                            **{k: sum(r[k] for r in values) for k in ("on", "question_only_reject", "image_only_reject", "both_reject", "question_upper_on", "question_upper_joint")}))
    vf.atomic_json(args.run_root / "private/FINAL_DETAILED_RESULTS.json", detailed)
    write_csv(args.public_dir / "POOLING_REJECTIONS_AND_THRESHOLDS.csv", reject)
    write_csv(args.public_dir / "POOLING_COST_BY_EDIT_SEED.csv", timings)
    grouped = defaultdict(list)
    for row in detailed:
        grouped[tuple(row[k] for k in ("condition", "panel", "operating_point", "category"))].append(row)
    aggregate = []
    for key, rows in sorted(grouped.items()):
        aggregate.append(dict(zip(("condition", "panel", "operating_point", "category"), key)) | {
            "micro": vf._metric(rows), "macro": {k: vf._macro(rows, k) for k in ("on", "base_semantic", "forced_semantic", "gated_semantic", "joint_on_semantic")},
            "raw_changed_num": sum(r["raw_changed"] for r in rows), "independent_edits": len({r["event_index"] for r in rows}),
            "source_image_groups": len({(r["event_index"],r["source_group"]) for r in rows})})
    vf.atomic_json(args.public_dir / "RESULTS_MACRO_MICRO.json", aggregate)
    negative_categories = {"matched_hard_image", vf.HARD_RELATION, vf.BROAD_RELATION, "same_image_other_source_fact"}
    summaries, paired = {}, []
    for point in vf.POINTS:
        summaries[point] = {}
        for kind in (*GROUPS, vf.CONDITIONS[0], vf.CONDITIONS[3]):
            rows = [r for r in detailed if r["condition"]==kind and r["operating_point"]==point]
            pools = {"hard": [r for r in rows if r["category"]=="matched_hard_image"],
                     "positive": [r for r in rows if r["category"]=="matched_evaluation_positive"],
                     "t1g": [r for r in rows if r["category"]=="formal_T1G"],
                     "t2g": [r for r in rows if r["category"]=="formal_T2G"]}
            damage = [r for r in rows if r["category"] in negative_categories and r["base_semantic"]]
            cell = {f"{name}_by_edit": vf._hierarchical_by_edit(values, "on" if name=="hard" else "joint_on_semantic") for name, values in pools.items()}
            cell.update({f"{name}_macro": mean(values.values()) if values else None for name, values in [(n,cell[f"{n}_by_edit"]) for n in pools]})
            cell.update(damage_num=sum(not r["gated_semantic"] for r in damage), damage_den=len(damage),
                        damage_raw_change_num=sum(r["raw_changed"] for r in damage), positive_activation=vf._macro(pools["positive"],"on"),
                        question_only_upper_activation=vf._macro(pools["positive"],"question_upper_on"),
                        question_only_upper_joint=vf._macro(pools["positive"],"question_upper_joint"))
            summaries[point][kind] = cell
        control = summaries[point][vf.CONDITIONS[0]]
        for kind in GROUPS:
            cell = summaries[point][kind]
            cell["full_retention_checks"] = {
                "hard_drop_ge_0_10": cell["hard_macro"] <= control["hard_macro"]-0.10,
                "positive_loss_le_0_02": cell["positive_macro"] >= control["positive_macro"]-0.02,
                "t1g_loss_le_0_02": cell["t1g_macro"] >= control["t1g_macro"]-0.02,
                "t2g_loss_le_0_02": cell["t2g_macro"] >= control["t2g_macro"]-0.02,
                "damage_not_increased": cell["damage_den"]==control["damage_den"] and cell["damage_num"]<=control["damage_num"],
            }
            for reference in ("G0", vf.CONDITIONS[0]):
                if kind == reference: continue
                other = summaries[point][reference]
                for metric in ("hard", "positive", "t1g", "t2g"):
                    low, high = vf._paired_ci(cell[f"{metric}_by_edit"], other[f"{metric}_by_edit"])
                    for edit in sorted(cell[f"{metric}_by_edit"]):
                        paired.append(dict(operating_point=point, condition=kind, reference=reference, metric=metric, event_index=edit,
                            candidate=cell[f"{metric}_by_edit"][edit], control=other[f"{metric}_by_edit"][edit],
                            paired_delta=cell[f"{metric}_by_edit"][edit]-other[f"{metric}_by_edit"][edit], edit_cluster_delta_ci_low=low, edit_cluster_delta_ci_high=high))
    write_csv(args.public_dir / "POOLING_PAIRED_RESULTS.csv", paired)
    primary = summaries[vf.POINTS[0]]
    passing = [g for g in GROUPS if all(primary[g]["full_retention_checks"].values())]
    selected = min(passing, key=lambda g: (primary[g]["hard_macro"], -primary[g]["positive_macro"], 28674, {"G1":0,"G2":1,"G0":2}[g])) if passing else None
    vf.atomic_json(args.public_dir / "POOLING_SIGNAL_SUMMARY.json", summaries)
    selection = dict(selected_condition=selected, selected_once=True, point=vf.POINTS[0], passing=passing,
                     historical_scientific_gain=False, source_code_commit=config["code_commit"], algorithm_config_sha256=vf.sha256_file(args.public_dir / "POOLING_ABLATION_CONFIG.json"))
    vf.atomic_json(args.public_dir / "ALGORITHM_SELECTION.json", selection)
    vf.atomic_json(args.public_dir / "NEW_EDIT_CONFIRMATION.json", {
        "status": "PENDING_ELIGIBILITY_AND_BUDGET" if selected else "NOT_RUN", "selected_condition": selected,
        "reason": "Primary development retention passed; authorize only frozen one-shot confirmation within remaining budget" if selected else "No condition passed all original M0 development retention requirements; conditional confirmation gate is closed", "n": 0})
    replay = [{"task_id":r["task"]["task_id"], **{k:x[k] for k in ("condition","decision","status","exact_replay")}} for r in results for x in r["replays"]]
    vf.atomic_json(args.public_dir / "JUDGE_AND_RAW_CLOSURE.json", dict(status="REUSED_COMPLETE", unique_tuples=len(verdicts), new_raw_tuples=0, new_judge_calls=0,
        all_parse_valid=True, generated_conditions="DERIVED_FROM_FROZEN_TWO_PATH_OUTPUTS", actual_replays=sum(r["exact_replay"] is True for r in replay), replay_coverage=replay,
        reuse_validation=read(args.sidecar)["reuse_validation"]))
    start = vf.validate_start(args.run_root)
    life = vf.read_jsonl(args.run_root / "private/WORKER_LIFECYCLE.jsonl")
    gpu_hours = sum(r["resident_seconds"] for r in life if r["event"]=="exit")/3600
    vf.atomic_json(args.public_dir / "GPU_USAGE_AND_COMPLETION.json", dict(task_complete=21,task_expected=21,fits=63,selected_simplest_condition=selected,
        gpu_hours=gpu_hours,wall_seconds=time.time()-start["epoch"],gpu1_used=False,PUBLICATION="RETRY_REQUIRED"))
    lines = ["# Pooling generality and safety", "", "New development experiment; historical scientific_gain=false and QUAL_VALIDATION_FAIL are unchanged.", "",
             "|Point|Condition|Hard macro FPR|Positive joint|T1G joint|T2G joint|Base-correct negative damage|Full retention|", "|---|---|---:|---:|---:|---:|---:|---|"]
    for point in vf.POINTS:
        for kind, cell in summaries[point].items():
            lines.append(f"|{point}|{kind}|{cell['hard_macro']:.6f}|{cell['positive_macro']:.6f}|{cell['t1g_macro']:.6f}|{cell['t2g_macro']:.6f}|{cell['damage_num']}/{cell['damage_den']}|{all(cell['full_retention_checks'].values()) if 'full_retention_checks' in cell else 'historical reference'}|")
    lines += ["", "## A. Mechanism comparison", "", "Compare G1/G2 with G0 using POOLING_PAIRED_RESULTS.csv (candidate minus reference). Image-group averages precede seed and edit averages; paired bootstrap resamples the seven edit clusters, not 21 seeds or individual rewrites. Intervals are descriptive viewed-development evidence, not independent clinical validation. G0 is a fresh image-only refit, not the old jointly trained M3.",
        "", "## B. Full development retention", "", f"Selected once at PRIMARY: {selected}." if selected else "池化替换未形成可保留优势。No condition passes all original M0 PRIMARY requirements. Stop adding routing candidates on these seven facts; conditional new-edit confirmation is NOT_RUN.",
        "", "SECONDARY retains its original diagnostic role and never overrides PRIMARY selection. For the original generality panel, historical M0 uses its stored CONTINUITY_SAFETY_FIRST / CONTINUITY_COVERAGE_CONSTRAINED thresholds; matched M0 uses the original PRIMARY/SECONDARY calibration. These are the unchanged historical operating-point mappings, not recalibrated M0.",
        "", "## Coverage, rejection, safety and cost", "", "RESULTS_MACRO_MICRO.json reports native, T1G, T2G, T1L, broad, same-image other fact and matched hard/positive separately, with actual support counts and shared Base/forced-on capability. Missing categories have no fabricated support. Base-correct damage and raw-change counts are both reported; exact and semantic correctness remain distinct.",
        "POOLING_REJECTIONS_AND_THRESHOLDS.csv reports both recalibrated thresholds, question-only/image-only/both rejections, and question-only activation/joint upper bounds. They are diagnostics, not actual image-always-ON method results. Identical question scores do not imply identical thresholds.",
        "Each condition has 28,674 linear-head parameters / 114,696 FP32 bytes, of which 14,337 / 57,348 bytes are trained this round. Pooling has zero new trainable parameters. C1 is additional and shared. G1/G2 never read Q to compute weights. G2 is pre-CP query-conditioned pooling, not proven lesion localization or grounding.",
        "POOLING_COST_BY_EDIT_SEED.csv reports synchronized feature-pooling timings (cached activations, excluding backbone), which are not new end-to-end latency measurements. Full generated outputs are reused; actual natural replays compare complete tokens and text. No new Judge call is made. GPU usage includes model-resident worker time, not only optimizer time.",
        "All original fit/cal/eval roles and step800 checkpoints are fixed. Same-native-image positive rewrites are not cross-patient positive generalization. C1 and question state hashes are checked unchanged; training uses only sorted matched fit pairs and image-only gradient clipping."]
    vf.atomic_text(args.public_dir / "POOLING_GENERALITY_SAFETY_REPORT.md", "\n".join(lines)+"\n")
    vf.atomic_text(args.public_dir / "GPT_PRO_REVIEW.md", "\n".join(["# GPT Pro review package", "", *lines[2:], "", "Review claim-evidence alignment, paired edit uncertainty, calibration-mediated changes, retained historical failure and the conditional confirmation status. This is an automated review package, not human signoff."])+"\n")
    latency = []
    for kind in GROUPS:
        rows = [r for r in detailed if r["condition"]==kind and r["operating_point"]==vf.POINTS[0]]
        observed = [r["selected_generation_seconds"] for r in rows if isinstance(r["selected_generation_seconds"],(int,float))]
        latency.append(dict(condition=kind, n=len(rows), measured_cached_generation_n=len(observed), selected_path_cached_generation_mean_seconds=mean(observed) if observed else None,
                            status="DERIVED_CACHED_GENERATION_TIME_NOT_FRESH_E2E", new_e2e_latency_measured=False))
    vf.atomic_json(args.public_dir / "LATENCY_PROVENANCE.json", latency)
