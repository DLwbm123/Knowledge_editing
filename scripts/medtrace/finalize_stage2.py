#!/usr/bin/env python3
"""Stage2 exact-tuple Judge reuse and denominator-explicit descriptive reports."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.run_stage2 import read, vf
from scripts.medtrace.finalize_selective_write import csv_write, hierarchical, opaque, stratum


def results(run):
    values = []
    for task in read(run / "private/TASK_QUEUE.json")["tasks"]:
        if task["kind"] == "INIT" or task["status"] not in {"RAW_READY", "JUDGED"}:
            continue
        value = read(run / "private/tasks" / task["task_id"] / "result_private.json")
        if value["status"] != "RAW_READY" or value["step"] != (50 if task["kind"] == "BE" else 320):
            raise ValueError("incomplete endpoint cannot enter Judge")
        if value["task"]["task_id"] != task["task_id"] or not value["base_guard"]["unchanged"]:
            raise ValueError("result identity/Base guard mismatch")
        values.append(value)
    for index in sorted({r["task"]["event_index"] for r in values if r["task"]["kind"] == "CP"}):
        value = read(run / f"private/initial/e{index:02d}/reference_private.json")
        value["task"].update(task_id=f"A2_e{index:03d}", kind="REFERENCE", parameterization="REFERENCE")
        values.append(value)
    return values


def execution_identity(lock):
    # Hardware slot and packet contents differ, numerical execution settings must not.
    return dict(model=lock["model"], tokenizer=lock["tokenizer"],
        protocol=lock["legacy_semantic_protocol_sha256"],
        runtime={k: v for k, v in lock["runtime"].items() if k not in {"physical_gpu", "gpu_uuid"}},
        generation={k: v for k, v in lock["generation"].items() if k != "resolved_engine_config"})


def validated_judge(directory):
    sidecar = read(directory / "JUDGE_SIDECAR_PRIVATE.json")
    execution = read(directory / "JUDGE_EXECUTION_LOCK_PRIVATE.json")
    if execution["packet"]["sha256"] != sidecar["packet_sha256"]:
        raise ValueError("Judge execution packet binding mismatch")
    verdicts = {}
    for row in vf.read_jsonl(directory / "JUDGE_OUTPUT_PRIVATE.jsonl"):
        key = row["opaque_query_id"]
        if key in verdicts or type(row.get("is_correct")) is not bool or not row.get("parse_valid"):
            raise ValueError("invalid Judge verdict")
        if row["legacy_semantic_protocol_sha256"] != sidecar["protocol_sha256"] or row["judge_snapshot_sha"] != sidecar["snapshot"]:
            raise ValueError("foreign Judge verdict")
        verdicts[key] = row["is_correct"]
    if set(verdicts) != set(sidecar["expected"]):
        raise ValueError("incomplete Judge coverage")
    return verdicts, execution, sidecar


def prepare_judge(args, base_only=False):
    run = args.run_root
    config = read(run / "private/CAMPAIGN_CONFIG.json")
    old = Path(config["stage1_run"]) / "private/judge"
    old_verdicts, execution, old_sidecar = validated_judge(old)
    protocol = read(Path(config["runtime"]["cpu_gate"]).parent / "private/JUDGE_LOCK_V4.json")
    if protocol["config_sha256"] != old_sidecar["protocol_sha256"] or protocol["judge_snapshot_sha"] != old_sidecar["snapshot"]:
        raise ValueError("Stage1 Judge protocol differs")
    # Reconstruct every reusable key from its actual full tuple, never query ID alone.
    prior_packets = {r["opaque_query_id"]: r for r in vf.read_jsonl(old / "JUDGE_PACKET_PRIVATE.jsonl")}
    if set(prior_packets) != set(old_verdicts):
        raise ValueError("historical packet coverage mismatch")
    for key, row in prior_packets.items():
        if key != opaque(row["question"], row["gold_answer"], row["raw_base_answer"], protocol["config_sha256"]):
            raise ValueError("historical tuple binding mismatch")
    before_dir = run / 'private/base_judge'
    if not base_only and before_dir.exists():
        before_sidecar = read(before_dir / 'JUDGE_SIDECAR_PRIVATE.json')
        if before_sidecar['new']:
            before_verdicts, before_execution, _ = validated_judge(before_dir)
            if execution_identity(before_execution) != execution_identity(execution):
                raise ValueError('Base-before numerical Judge execution drift')
            for row in vf.read_jsonl(before_dir / 'JUDGE_PACKET_PRIVATE.jsonl'):
                key = opaque(row['question'], row['gold_answer'], row['raw_base_answer'], protocol['config_sha256'])
                if key != row['opaque_query_id']:
                    raise ValueError('Base-before full tuple drift')
            old_verdicts.update(before_verdicts)
    tuples = {}
    values = results(run)
    if base_only:
        values = []
        for episode in read(run / 'private/NEW_EPISODE_MANIFEST_PRIVATE.json')['episodes']:
            i = episode['event_index']
            data = read(run / f'private/edits/e{i:02d}.json')
            outputs = {}
            for row in data['rows']:
                base = read(run / f'private/base_generation/e{i:02d}' / (vf.sha256_json(row)+'.json'))
                outputs[row['logical_id']] = dict(row=row, base=base, forced=base, fixed=base)
            values.append(dict(task=dict(event_index=i), outputs=outputs))
    for result in values:
        data = read(run / f"private/edits/e{result['task']['event_index']:02d}.json")
        target = data["event"]["edit_record"]["gold_answer"]
        for item in result["outputs"].values():
            row = item["row"]
            for ref in {row["reference"], target}:
                for branch in ("base", "forced", "fixed"):
                    raw = item[branch]["raw_answer"]
                    key = opaque(row["question"], ref, raw, protocol["config_sha256"])
                    tuples[key] = dict(opaque_query_id=key, question=row["question"], gold_answer=ref,
                                       raw_base_answer=raw, adjudication_pass=1)
    if not tuples:
        raise RuntimeError("no actual endpoints to Judge")
    directory = run / ('private/base_judge' if base_only else 'private/judge')
    directory.mkdir(exist_ok=False)
    reused = {key: old_verdicts[key] for key in tuples.keys() & old_verdicts.keys()}
    new = {key: value for key, value in tuples.items() if key not in reused}
    packet = directory / "JUDGE_PACKET_PRIVATE.jsonl"
    vf.atomic_text(packet, "".join(json.dumps(row, sort_keys=True)+"\n" for row in new.values()))
    vf.atomic_json(directory / "REUSED_VERDICTS_PRIVATE.json", reused)
    vf.atomic_json(directory / "REUSE_EXECUTION_IDENTITY_PRIVATE.json", execution_identity(execution))
    vf.atomic_json(directory / "JUDGE_SIDECAR_PRIVATE.json", dict(protocol_sha256=protocol["config_sha256"],
        snapshot=protocol["judge_snapshot_sha"], expected=sorted(new), all_expected=sorted(tuples),
        packet_sha256=vf.sha256_file(packet), reused=len(reused), new=len(new), complete_answers=True))


def lock_base_before(args):
    directory = args.run_root / 'private/base_judge'
    sidecar = read(directory / 'JUDGE_SIDECAR_PRIVATE.json')
    verdicts = read(directory / 'REUSED_VERDICTS_PRIVATE.json')
    if sidecar['new']:
        new, execution, _ = validated_judge(directory)
        if execution_identity(execution) != read(directory / 'REUSE_EXECUTION_IDENTITY_PRIVATE.json'):
            raise ValueError('Base-before Judge execution mismatch')
        verdicts.update(new)
    if set(verdicts) != set(sidecar['all_expected']):
        raise ValueError('Base-before incomplete')
    membership, support = {}, []
    for episode in read(args.run_root / 'private/NEW_EPISODE_MANIFEST_PRIVATE.json')['episodes']:
        i = episode['event_index']
        data = read(args.run_root / f'private/edits/e{i:02d}.json')
        membership[str(i)] = {}
        for row in data['rows']:
            base = read(args.run_root / f'private/base_generation/e{i:02d}' / (vf.sha256_json(row)+'.json'))
            key = opaque(row['question'], row['reference'], base['raw_answer'], sidecar['protocol_sha256'])
            correct = verdicts[key]
            membership[str(i)][row['logical_id']] = dict(base_correct=correct, eqkey=row['eqkey'], tuple_key=key,
                H_keep=correct and stratum(row) == 'H', U_keep=correct and stratum(row) == 'U')
        for role, panel in sorted({(row['role'], stratum(row)) for row in data['rows']}):
            rows = [r for r in data['rows'] if (r['role'], stratum(r)) == (role, panel)]
            support.append(dict(edit=i, role=role, panel=panel, inputs=len(rows), source_images=len({r['image_path'] for r in rows}),
                base_correct_inputs=sum(membership[str(i)][r['logical_id']]['base_correct'] for r in rows)))
    vf.atomic_json(args.run_root / 'private/BASE_BEFORE_ROLE_LOCK_PRIVATE.json', dict(membership=membership,
        protocol=sidecar['protocol_sha256'], new_students_started=False, immutable_predicate='locked source correctness of frozen Base full answer'))
    csv_write(args.public_dir / 'NEW_BASE_BEFORE_SUPPORT.csv', support)
    queue = vf.TaskQueue(args.run_root / 'private/TASK_QUEUE.json', args.run_root)
    def activate(data):
        if any(t['event_index'] >= 100 and t['status'] != 'WAITING_BASE_BEFORE' for t in data['tasks']):
            raise ValueError('new student queue changed before Base-before lock')
        for task in data['tasks']:
            if task['status'] == 'WAITING_BASE_BEFORE':
                task['status'] = 'PENDING'
    queue._locked(activate)


METRICS = ("semantic", "target_consistency", "token_parity", "kl", "base_correct", "base_correct_damage",
           "base_correct_preserved", "base_wrong_became_correct", "base_wrong_changed", "on", "joint_on_correct")


def aggregate(details):
    keys = ("track", "edit", "method", "mode", "role", "stratum")
    cells = defaultdict(list)
    for row in details:
        cells[tuple(row[k] for k in keys)].append(row)
    by_edit = []
    for key, rows in sorted(cells.items()):
        row = dict(zip(keys, key))
        row.update(inputs=len(rows), eqkeys=len({r["eqkey"] for r in rows}),
            source_images=len({r.get("source_image_id", r["source_group"]) for r in rows}),
            base_correct_inputs=sum(int(r["base_correct"]) for r in rows),
            base_correct_images=len({r.get("source_image_id", r["source_group"]) for r in rows if r["base_correct"]}))
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            row[metric] = hierarchical(rows, metric)
            row[metric+"_numerator"] = sum(values) if values else None
            row[metric+"_denominator"] = len(values)
        by_edit.append(row)
    cells = defaultdict(list)
    for row in by_edit:
        cells[tuple(row[k] for k in keys if k != "edit")].append(row)
    macros = []
    for key, rows in sorted(cells.items()):
        row = dict(zip((k for k in keys if k != "edit"), key))
        row.update(edits=len(rows), inputs=sum(r["inputs"] for r in rows),
            source_image_edit_pairs=sum(r["source_images"] for r in rows),
            base_correct_inputs=sum(r["base_correct_inputs"] for r in rows),
            base_correct_edits=sum(r["base_correct_inputs"] > 0 for r in rows))
        matching = [r for r in details if all(r[k] == v for k, v in zip((k for k in keys if k != "edit"), key))]
        row["distinct_source_images"] = len({r.get("source_image_id", r["source_group"]) for r in matching})
        row["distinct_base_correct_images"] = len({r.get("source_image_id", r["source_group"]) for r in matching if r["base_correct"]})
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            row[metric] = mean(values) if values else None
        macros.append(row)
    return by_edit, macros


def finalize(args):
    run, public = args.run_root, args.public_dir
    directory = run / "private/judge"
    sidecar = read(directory / "JUDGE_SIDECAR_PRIVATE.json")
    verdicts = read(directory / "REUSED_VERDICTS_PRIVATE.json")
    if sidecar["new"]:
        new, execution, _ = validated_judge(directory)
        if execution_identity(execution) != read(directory / "REUSE_EXECUTION_IDENTITY_PRIVATE.json"):
            raise ValueError("Judge numerical execution drift: cannot reuse old verdicts")
        verdicts.update(new)
    if set(verdicts) != set(sidecar["all_expected"]):
        raise ValueError("new/reused Judge closure incomplete")
    def score(row, ref, raw):
        return float(verdicts[opaque(row["question"], ref, raw, sidecar["protocol_sha256"])])
    details, costs = [], []
    values = results(run)
    gates = {}
    before_path = run / 'private/BASE_BEFORE_ROLE_LOCK_PRIVATE.json'
    base_membership = read(before_path)['membership'] if before_path.exists() else {}
    for result in values:
        task = result["task"]
        data = read(run / f"private/edits/e{task['event_index']:02d}.json")
        target = data["event"]["edit_record"]["gold_answer"]
        method = "BE" if task["kind"] == "BE" else "A2" if task["kind"] == "REFERENCE" else task["parameterization"]+"-"+task["condition"]
        if data["track"] == "NEW_CONFIRMATION":
            decisions = {k: item["fixed_on"] for k, item in result["outputs"].items()}
            if task["event_index"] in gates and decisions != gates[task["event_index"]]:
                raise ValueError("BE gate changed across new writers")
            gates[task["event_index"]] = decisions
        costs.append(dict(task_id=task["task_id"], track=data["track"], method=method,
            parameters=result.get("parameters"), training_seconds=result.get("training_seconds"),
            elapsed_seconds=result.get("elapsed_seconds"), storage_bytes=result.get("storage_bytes"),
            load_seconds=result.get("load_seconds"), peak_vram_bytes=result.get("peak_vram_bytes"),
            forward_count=result.get("forward_count"), backward_count=result.get("backward_count")))
        for item in result["outputs"].values():
            row = item["row"]
            base_correct = score(row, row["reference"], item["base"]["raw_answer"])
            if data['track'] == 'NEW_CONFIRMATION':
                before = base_membership[str(task['event_index'])][row['logical_id']]
                if before['eqkey'] != row['eqkey'] or bool(base_correct) != before['base_correct']:
                    raise ValueError('new Base-correct support changed after student launch')
            for mode, branch in (("BE_FORCED_ON" if method == "BE" else "FORCED_ON", "forced"),
                                 ("BE_NATIVE_ROUTED" if method == "BE" else "BE_ROUTE+"+method, "fixed"), ("BASE", "base")):
                correct = score(row, row["reference"], item[branch]["raw_answer"])
                on = branch == "forced" or branch == "fixed" and item["fixed_on"]
                details.append(dict(track=data["track"], edit=task["event_index"], method=method, mode=mode,
                    role=row["role"], stratum=row.get("confirmation_panel", stratum(row)), eqkey=row["eqkey"], source_group=row["source_group"],
                    source_image_id=vf.sha256_json(str(Path(row["image_path"]).resolve())),
                    semantic=correct, target_consistency=score(row, target, item[branch]["raw_answer"]),
                    token_parity=float(item[branch]["raw_token_ids"] == item["base"]["raw_token_ids"]),
                    kl=item["kl"] if on else 0. if item["kl"] is not None else None, base_correct=base_correct,
                    base_correct_damage=1-correct if base_correct else None,
                    base_correct_preserved=correct if base_correct else None,
                    base_wrong_became_correct=correct if not base_correct else None,
                    base_wrong_changed=float(item[branch]["raw_token_ids"] != item["base"]["raw_token_ids"]) if not base_correct else None,
                    on=float(on), joint_on_correct=float(on)*correct))
    views = []
    for row in details:
        if row['track'] == 'NEW_CONFIRMATION' and row['stratum'] in ('H', 'U'):
            group = row['stratum']
            row['stratum'] = group+'_all'
            views.append(dict(row, stratum=group+('_keep' if row['base_correct'] else '_base_wrong')))
    details.extend(views)
    by_edit, macros = aggregate(details)
    csv_write(public / "STAGE2_BEHAVIOR_BY_EDIT.csv", by_edit)
    csv_write(public / "STAGE2_BEHAVIOR_MACRO.csv", macros)
    csv_write(public / "METHOD_COSTS.csv", costs)
    csv_write(public / "BALANCEDIT_MATCHED_DEV_RESULTS.csv", [r for r in macros if r["track"] == "OLD_DEV16"])
    csv_write(public / "NEW_EDIT_FORCED_ON_RESULTS.csv", [r for r in macros if r["track"] == "NEW_CONFIRMATION" and "FORCED_ON" in r["mode"]])
    csv_write(public / "NEW_EDIT_SYSTEM_RESULTS.csv", [r for r in macros if r["track"] == "NEW_CONFIRMATION" and "FORCED_ON" not in r["mode"]])
    comparisons = []
    new_rows = [r for r in by_edit if r["track"] == "NEW_CONFIRMATION" and r["mode"] == "FORCED_ON"]
    for candidate, control in (("P4-W1_KL_0.1", "P4-W0_TASK_ONLY"), ("P4-W2_GROUP_CONSTRAINED", "P4-W1_KL_0.1"),
                               ("P4-W1_KL_0.1", "A2"), ("P4-W2_GROUP_CONSTRAINED", "A2"), ("L16-W2_GROUP_CONSTRAINED", "A2")):
        for role, panel in sorted({(r["role"], r["stratum"]) for r in new_rows}):
            for metric in ("semantic", "base_correct_damage", "kl"):
                mapping = lambda m: {r["edit"]: r[metric] for r in new_rows if r["method"] == m and r["role"] == role and r["stratum"] == panel and r[metric] is not None}
                a, b = mapping(candidate), mapping(control)
                common = set(a) & set(b)
                low, high = vf._paired_ci(a, b) if len(common) >= 2 else (None, None)
                delta = mean(a[k]-b[k] for k in common) if common else None
                comparisons.append(dict(candidate=candidate, control=control, role=role, stratum=panel, metric=metric,
                    paired_edits=len(common), delta=delta, ci_low=low, ci_high=high,
                    practical_minus_5pp_reference=(delta >= -.05) if delta is not None and metric == "semantic" else None))
    csv_write(public / "NEW_PAIRED_EDIT_EFFECTS.csv", comparisons)
    queue = vf.TaskQueue(run / "private/TASK_QUEUE.json", run)
    for result in values:
        if result["task"]["kind"] != "REFERENCE":
            queue.update(result["task"]["task_id"], "JUDGED")
    tasks = queue.snapshot()["tasks"]
    manifest = read(public / "NEW_EPISODE_MANIFEST_PUBLIC.json")
    n = manifest.get("actual_n", manifest.get("n", 0))
    candidate_n = manifest.get("candidate_episode_count", n)
    old_done = sum(t["kind"] == "BE" and t["event_index"] < 100 and t["status"] == "JUDGED" for t in tasks)
    done = (all(t["status"] in {"JUDGED", "COMPLETE"} for t in tasks) and candidate_n == n
            and sum(t['event_index'] >= 100 and t['kind'] != 'INIT' and t['status'] == 'JUDGED' for t in tasks) == 5*n)
    vf.atomic_json(public / "QUEUE_COMPLETION.json", dict(tasks=[{k: t[k] for k in ("task_id", "kind", "event_index", "status")} for t in tasks], counts=dict(Counter(t["status"] for t in tasks))))
    vf.atomic_json(public / "JUDGE_CLOSURE.json", dict(reused=sidecar["reused"], new=sidecar["new"], total=len(verdicts),
        same_execution_verified=True, full_answer_exact_tuple_only=True))
    status = dict(status="COMPLETE" if done else "PARTIAL_RESULTS", old_dev16_judged=old_done, new_n=n, candidate_n=candidate_n,
                  counts=dict(Counter(t["status"] for t in tasks)), publication="PENDING_PUBLIC_PUSH")
    vf.atomic_json(public / "RUN_COMPLETION.json", status)
    def table(rows):
        lines = ["| Path | Role | Panel | Inputs | Edits | Source correct macro | Base-correct inputs/edits | Damage macro |", "|---|---|---|---:|---:|---:|---:|---:|"]
        f = lambda x: "NA" if x is None else f"{100*x:.2f}%"
        return "\n".join(lines+[f"| {r['mode']} | {r['role']} | {r['stratum']} | {r['inputs']} | {r['edits']} | {f(r['semantic'])} | {r['base_correct_inputs']}/{r['base_correct_edits']} | {f(r['base_correct_damage'])} |" for r in rows])
    vf.atomic_text(public / "BALANCEDIT_MATCHED_DEV_REPORT.md", f"# BalancEdit matched DEV16\n\nActual judged edits: {old_done}/16. Frozen V4 adaptation, independent single-edit 50-step full up-projection transforms; not author sequential reproduction. Both native routed and forced-on outputs actually generated. Low scores never gate evaluation.\n\n"+table([r for r in macros if r["track"] == "OLD_DEV16"])+"\n\nSeven old Stage1 facts use identical native/fit/cal/evaluation/H/U/challenge rows. Other DEV16 edits use their available original T0/T1G/T2G/T1L probes. Compare Stage1 FORCED_ON only with BE_FORCED_ON; routing is a separate system axis. Original Stage1 outputs were not retrained. See Stage1 full behavior addendum and METHOD_COSTS.csv. NA is zero support, not zero damage.\n")
    vf.atomic_text(public / "NEW_EDIT_CONFIRMATION_REPORT.md", f"# New-edit confirmation\n\nSource candidates={candidate_n}; frozen runnable episodes N={n}; task status {status['status']}. See the source/exposure manifest for authorization boundaries and exclusions.\n\n"+("No new-edit performance claim. N=0 runnable does NOT mean the full source scan found no candidates. Remaining readiness/identity boundaries are in the source manifest. The executable initializer is run_stage2.initialize_episode; no old fact has been renamed new.\n" if not n else table([r for r in macros if r["track"] == "NEW_CONFIRMATION"])+"\n\nEqKey to source-image to edit aggregation; paired intervals resample edits, not paraphrases. The minus-five-percentage-point margin is descriptive, not statistical or clinical proof. Native and BOTH evaluation text families must be read together with damage; KL alone cannot rank winners.\n"))
    vf.atomic_text(public / "GPT_PRO_REVIEW.md", f"# Stage2 V2 review\n\nStatus: {status['status']}. BalancEdit old DEV16 {old_done}/16 judged; source candidates={candidate_n}, frozen runnable new episodes N={n}.\n\nStage1 T2G audit found no binding correction: P4 W1(.1) protection comes with observed original-T2G loss; old qualification labels are unchanged. H-eval old Base-correct support is only 4 inputs/1 edit; U has 44/7.\n\nRead BALANCEDIT_MATCHED_DEV_REPORT.md (actual native routing vs forced-on), NEW_EDIT_CONFIRMATION_REPORT.md, paired effects and costs. No extra lambda/rank/layer sweep, LoRA gate, W3, new router, sequential or clinical claim.\n\n"+("New-fact generalization remains untested. Inspect actual source candidates and unresolved readiness/identity boundaries, not an assertion of source exhaustion.\n" if not n else "Judge protection and both cross-family generality jointly; low KL is not sufficient. Missing or failed tasks remain in the queue ledger, not dropped from the eligible cohort.\n"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare-judge", "prepare-base-judge", "lock-base-before", "finalize"))
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--public-dir", required=True, type=Path)
    args = p.parse_args()
    if args.action == 'prepare-base-judge':
        prepare_judge(args, base_only=True)
    else:
        {'prepare-judge': prepare_judge, 'lock-base-before': lock_base_before, 'finalize': finalize}[args.action](args)


if __name__ == "__main__":
    main()
