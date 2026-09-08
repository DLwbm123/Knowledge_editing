"""Stage2 insertion replay: frozen bank, real selected writer, no training/Judge.

run_bank(runtime, args, task) is one indivisible three-method prefix task.
Only Stage3 private/bank is written; the Stage2 manifest/checkpoints stay read-only.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import time

import torch

from scripts.medtrace.run_stage2 import sw, vf, read, same_output
from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
from m3bench_repro.editors.routing import MemoryRouter

PREFIXES = (1, 4, 8, 16)
METHODS = {"S0": "W0_TASK_ONLY", "S1": "W1_KL_0.1", "B": "BALANCEDIT"}


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def fact_key(row):
    # Source lineage is usable only with the explicit same-fact positive review.
    keys = ("dataset", "source_group", "source_qid")
    if all(row.get(k) is not None for k in keys):
        return tuple(str(row[k]) for k in keys)
    return None


def matching_target(row, target):
    return (bool(row.get("eqkey")) and row["eqkey"] == target.get("eqkey")) or (
        fact_key(row) is not None and fact_key(row) == fact_key(target))


def freeze_roles(episodes):
    """Use input/source metadata only; no output or Judge is an argument."""
    targets = [(e["record_id"], r) for e in episodes for r in e["rows"]
               if r["label"] == "positive" and (r["role"] == "native" or
                   (e.get("positive_review", {}).get("approved_equivalent") is True and
                    r.get("fact_relation") == "reviewed_same_fact_text_augmentation"))]
    native_groups = {r.get("source_group") for _, r in targets}
    roles, overlap = {}, []
    for e in episodes:
        for row in e["rows"]:
            matches = [(rid, r) for rid, r in targets if matching_target(row, r)]
            refs = sorted({r["reference"] for _, r in matches})
            legitimate = sorted({rid for rid, _ in matches})
            if len(refs) > 1:
                role, ref = "TARGET_CONFLICT", None
            elif matches:
                role = "EDIT_TARGET" if row["label"] == "positive" else "NOW_EDITED_CONTEXT"
                ref = refs[0]
            elif (row["label"] == "negative" and row.get("negative_group") in ("H", "U")
                  and row.get("relation_evidence") and fact_key(row) is not None
                  and None not in native_groups and row["source_group"] not in native_groups):
                role, ref = "STRICT_BASE", row["reference"]
            else:
                role, ref = "UNKNOWN", None
            roles[(e["record_id"], row["logical_id"])] = dict(
                strict_role=role, effective_reference=ref, legitimate_experts=legitimate,
                target_references=refs, role_evidence="frozen exact input/source fact and reviewed positive metadata",
                strict_metric_eligible=role in ("EDIT_TARGET", "NOW_EDITED_CONTEXT", "STRICT_BASE"))
            for rid, target in matches:
                if rid != e["record_id"] and row["role"] != target["role"]:
                    overlap.append(dict(source_expert=e["record_id"], logical_id=row["logical_id"],
                        source_role=row["role"], matched_expert=rid, matched_role=target["role"],
                        exact_input=row.get("eqkey") == target.get("eqkey")))
    return roles, overlap


def evaluation_rows(episode):
    return [r for r in episode["rows"] if r["role"] == "native" or
            (r["role"] == "evaluation" and (r["label"] == "positive" or
                                             r.get("negative_group") in ("H", "U")))]


def selection_class(selected, source, role):
    if selected is None:
        return "OFF"
    if selected == source:
        return "OWN_EXPERT"
    if role["strict_role"] in ("UNKNOWN", "TARGET_CONFLICT"):
        return "UNKNOWN"
    return "KNOWN_EQUIVALENT_EXPERT" if selected in role["legitimate_experts"] else "OTHER_EXPERT"


def input_batch(runtime, row):
    # Construct a neutral query rather than inheriting source/task/target fields.
    record = vf.EditorRecord.from_dict(dict(record_id="bank-query", dataset="",
        question=row["question"], image_path=row["image_path"], gold_answer="", official_rephrase="",
        relative_image_path="", formal_sequence_position=0, question_type=""))
    batch = runtime.build_question_batch(record)
    eqkey = vf.sha256_json(dict(image=batch.image_sha256, ids=batch.raw_input_ids.tolist(),
        attention=batch.attention_mask.tolist() if batch.attention_mask is not None else None,
        boundary=batch.key_token_index, generation=runtime.generation_config))
    if eqkey != row.get("eqkey"):
        raise ValueError("Stage2 realized input/generation binding differs")
    return batch


def load_manifest(stage2):
    frozen = read(stage2 / "private/NEW_EPISODE_MANIFEST_PRIVATE.json")["episodes"]
    if len(frozen) != 16 or len({e["record_id"] for e in frozen}) != 16:
        raise ValueError("Track B requires the original sixteen independent episodes")
    episodes = []
    for original in frozen:  # Never sort by IDs, outcomes, or filesystem order.
        e = read(stage2 / f"private/edits/e{original['event_index']:02d}.json")
        if e["event"] != original["event"] or e["record_id"] != original["record_id"]:
            raise ValueError("Stage2 manifest/event mismatch")
        if not e.get("source_eligibility_frozen") or e["track"] != "NEW_CONFIRMATION":
            raise ValueError("not a frozen Stage2 new-confirmation episode")
        current = {r["logical_id"]: r for r in e["rows"]}
        if len(current) != len(original["rows"]):
            raise ValueError("Stage2 frozen row support changed")
        for row in original["rows"]:
            actual = current[row["logical_id"]]
            if any(actual.get(k) != v for k, v in row.items()) or not actual.get("eqkey"):
                raise ValueError("Stage2 row/source binding changed")
        episodes.append(e)
    return episodes


def load_artifacts(stage2, episodes, guard):
    """Resolve the completed endpoint, never select among attempts by performance."""
    artifacts, entries = {}, []
    for e in episodes:
        i, rid = e["event_index"], e["record_id"]
        for method, condition in METHODS.items():
            kind = "BE_BE" if method == "B" else "CP_P4"
            directory = stage2 / f"private/tasks/S2_e{i:03d}_{kind}_{condition}"
            result = read(directory / "result_private.json")
            task = result["task"]
            if (result["status"] != "RAW_READY" or task["event_index"] != i or
                    task["condition"] != condition or not result["base_guard"]["unchanged"] or
                    result["base_guard"]["after_sha256"] != guard["after_sha256"]):
                raise ValueError("Stage2 result/Base identity mismatch")
            if method == "B":
                path = directory / "editor_state.pt"
            else:
                candidates = list(directory.glob("attempt_chunk*/step0320.pt"))
                if len(candidates) != 1:
                    raise ValueError(f"ambiguous/missing completed endpoint: {directory}")
                path = candidates[0]
            state = torch.load(path, map_location="cpu", weights_only=True)
            if method == "B":
                if (state["method"] != "balancedit" or state["edit_history"] != [rid] or
                    state["router"]["distance"] != "euclidean" or
                    [v["logical_edit_id"] for v in state["router"]["entries"]] != [rid] or
                    set(state["wrapper"].get("edits", {})) != {rid} or result["step"] != 50):
                    raise ValueError("not this independent native BE state")
                lock = read(directory / "METHOD_CONFIG_LOCK.json")
                if (lock["alpha"], lock["steps_per_edit"], lock["learning_rate"]) != (.2, 50, .01):
                    raise ValueError("native BE method drift")
                entry = state["router"]["entries"][0]
                if not torch.isfinite(entry["key"]).all():
                    raise ValueError("nonfinite frozen key")
                entries.append(dict(entry, label=[]))  # Targets never enter the selector.
                target = state["target"]
                weights = state["wrapper"]["edits"][rid]
                training = read(directory / "TRAINING_PRIVATE.json")
                if (training["record_id"] != rid or training["steps"] != 50 or
                    not training["finite_losses"] or not training["finite_gradients"]):
                    raise ValueError("BE checkpoint lacks completed native training provenance")
                seed = training["seed"]
            else:
                if (result["step"] != 320 or state["step"] != 320 or
                    any(state["task"].get(k) != task.get(k) for k in
                        ("task_id", "event_index", "parameterization", "condition", "seed")) or
                    task["parameterization"] != "P4" or result["a2_sha256"] != e["a2_sha256"]):
                    raise ValueError("P4 checkpoint/task/A2 mismatch")
                training = read(path.parent / "training_private.json")
                if training["curve"][-1]["step"] != 320 or training["diagnostics"][-1]["step"] != 320:
                    raise ValueError("P4 endpoint lacks completed training provenance")
                target, weights = vf.LAYER, state["expert"]
                seed = task["seed"]
            if any(not torch.isfinite(t).all() for t in weights.values()):
                raise ValueError("nonfinite frozen writer")
            artifacts[(method, rid)] = dict(path=path, identity=file_identity(path), result=result,
                target=target, parameters=result["parameters"], seed=seed)
            del state, weights
    return artifacts, entries


def index_legacy(artifacts, episodes):
    """Index FORCED_ON by actual writer and exact realized input, never source alone."""
    cached = {}
    for e in episodes:
        for method in METHODS:
            artifact = artifacts[(method, e["record_id"])]
            values = artifact["result"]["outputs"]
            for row in e["rows"]:
                value = values[row["logical_id"]]
                if value["row"] != row:
                    raise ValueError("Stage2 cached row differs from frozen model input")
                for writer, output in ((None, value["base"]),
                                       ((method, e["record_id"]), value["forced"])):
                    key = (writer, row["eqkey"])
                    if key in cached and not same_output(cached[key][0], output):
                        raise ValueError("conflicting exact-input/checkpoint cached outputs")
                    cached[key] = (output, dict(kind="STAGE2_EXACT_BOUND", derived=True,
                        source_result=str(artifact["path"].parent.parent / "result_private.json")
                        if method != "B" else str(artifact["path"].parent / "result_private.json"),
                        writer=artifact["identity"] if writer else "BASE"))
    return cached


def run_bank(runtime, args, task):
    run = Path(args.run_root)
    config = read(run / "private/CAMPAIGN_CONFIG.json")
    stage2 = Path(config["stage2_run"]).resolve(strict=True)
    prefix = task["prefix"]
    if task["kind"] != "BANK" or prefix not in PREFIXES:
        raise ValueError("BANK task must be prefix 1,4,8,16")
    out = run / f"private/bank/prefix{prefix}"
    if out.resolve().is_relative_to(stage2):
        raise ValueError("refusing any write inside the historical Stage2 run")
    started = time.monotonic()
    old_config = read(stage2 / "private/CAMPAIGN_CONFIG.json")
    runtime_lock = read(Path(old_config["runtime"]["runtime_lock"]))
    generation = read(Path(old_config["runtime"]["cpu_gate"]) / "inputs/frozen/llava_med_generation_frozen.json")
    if (config["runtime"] != old_config["runtime"] or runtime.generation_config != generation or
        runtime_lock["selected_runtime"] != "runtime_b_official_native"):
        raise ValueError("bank runtime differs from Stage2 frozen execution")
    guard = runtime.base_guard.verify()
    if not guard["unchanged"]:
        raise RuntimeError("bank requires unmodified Base")
    episodes = load_manifest(stage2)[:prefix]
    roles, overlaps = freeze_roles(episodes)
    vf.atomic_json(out / "ROLE_LOCK_PRIVATE.json", dict(task=task,
        roles=[dict(source_expert=rid, logical_id=lid, **v) for (rid, lid), v in roles.items()],
        exposure="viewed insertion replay, not online training or blind evaluation", cross_edit_overlap=overlaps))
    artifacts, entries = load_artifacts(stage2, episodes, guard)
    cached = index_legacy(artifacts, episodes)
    routers = {m: MemoryRouter.from_state(dict(distance="euclidean", entries=entries), device=runtime.device)
               for m in METHODS}
    costs = {m: dict(training_seconds=0., optimizer_steps=0, generation_seconds=0., load_seconds=0.,
        route_seconds=0., feature_seconds=0., request_seconds=0., generated=0, derived=0,
        disk_bytes=sum(a["identity"]["size_bytes"] for (method, _), a in artifacts.items() if method == m),
        parameters=sum(a["parameters"] for (method, _), a in artifacts.items() if method == m),
        peak_resident_vram_bytes=0, replay_seconds=0.) for m in METHODS}
    result = dict(status="PARTIAL", task=task, outputs=[], costs=costs, base_guard=guard,
        expected_inputs=sum(len(evaluation_rows(e)) for e in episodes),
        expected_items=3*sum(len(evaluation_rows(e)) for e in episodes),
        roles_frozen_before_outputs=True, prefix=prefix, manifest_order=[e["record_id"] for e in episodes],
        experiment="STAGE2_EXPERT_BANK_INSERTION_REPLAY16", judge_status="PENDING",
        replays={m: {b: {"status": "NOT_OBSERVED"} for b in ("ON", "OFF", "WRONG_SELECTION")} for m in METHODS},
        checkpoint_provenance=[dict(method=m, expert=rid, **a["identity"], seed=a["seed"])
                               for (m, rid), a in artifacts.items()],
        latency_scope="cached request latency and actual replay latency separate; shared Base features charged per system",
        storage_policy="original checkpoints read-only; one selected BE writer resident; P4 loaded on demand")
    editor = None
    target = runtime.target_lock["balancedit"]["targets"][0]
    base_module = runtime.get_module(target)
    loaded = None
    input_cache = {}
    binding = dict(runtime=runtime_lock, generation=generation,
                   mask="Stage2 MedTraceLayerHook.generation_request / BE full linear")

    def generate(method, selected, row):
        nonlocal loaded
        editor.wrapper.set_active(None)
        if method != "B" and loaded is not None:
            editor.wrapper.clear()
            loaded = None
        load_start = time.monotonic()
        expert = None
        if selected is not None:
            a = artifacts[(method, selected)]
            if method == "B":
                if loaded != selected:
                    editor.wrapper.clear()
                    state = torch.load(a["path"], map_location="cpu", weights_only=True)
                    editor.wrapper.load_exported_state(state["wrapper"])
                    loaded = selected
            else:
                state = torch.load(a["path"], map_location="cpu", weights_only=True)
                expert = vf.AsymmetricCPExpert(14336, 4096, 4).to(runtime.device)
                expert.load_state_dict(state["expert"])
                expert.requires_grad_(False)
        costs[method]["load_seconds"] += time.monotonic()-load_start
        gen_start = time.monotonic()
        try:
            if method == "B":
                with editor._activated(selected):
                    value = vf.scope_generate(runtime, row, None)
            else:
                value = sw.generated(runtime, row, expert)
            costs[method]["generated"] += 1
            return value
        finally:
            editor.wrapper.set_active(None)
            costs[method]["generation_seconds"] += time.monotonic()-gen_start
            if runtime.device.type == "cuda":
                costs[method]["peak_resident_vram_bytes"] = max(costs[method]["peak_resident_vram_bytes"],
                                                               torch.cuda.max_memory_allocated(runtime.device))

    def output(method, selected, row):
        key = ((method, selected) if selected is not None else None, row["eqkey"])
        if key in cached:
            costs[method]["derived"] += 1
            return cached[key]
        identity = dict(binding=binding, eqkey=row["eqkey"],
                        writer=artifacts[(method, selected)]["identity"] if selected else "BASE")
        # Prefix groups can run concurrently; their atomic-write temporaries must not collide.
        path = out / "cache" / (vf.sha256_json(identity)+".json")
        if path.exists():
            prior = read(path)
            if prior["identity"] != identity:
                raise ValueError("bank cache binding mismatch")
            value = prior["output"]
            provenance = dict(kind="STAGE3_EXACT_BOUND", derived=True, cache_path=str(path), **identity)
            costs[method]["derived"] += 1
        else:
            value = generate(method, selected, row)
            vf.atomic_json(path, dict(identity=identity, output=value))
            provenance = dict(kind="ACTUAL_GENERATION", derived=False, cache_path=str(path), **identity)
        cached[key] = value, dict(provenance, derived=True)
        return value, provenance

    try:
        editor = BalanceEditPaperSpecEditor(runtime)
        if any(a["target"] != (target if m == "B" else vf.LAYER) for (m, _), a in artifacts.items()):
            raise ValueError("checkpoint target differs from runtime")
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        for position, e in enumerate(episodes, 1):
            source = e["record_id"]
            first = next(k for k in PREFIXES if position <= k)
            for row in evaluation_rows(e):
                request_start = time.monotonic()
                editor.wrapper.set_active(None)
                if row["eqkey"] not in input_cache:
                    feature_start = time.monotonic()
                    batch = input_batch(runtime, row)
                    with torch.no_grad(), editor.disabled():
                        key = runtime.extract_layer_input_key(batch, module_path=target, pooling="mean")
                    if not torch.isfinite(key).all():
                        raise ValueError("nonfinite Base query key")
                    input_cache[row["eqkey"]] = key, time.monotonic()-feature_start
                key, feature_seconds = input_cache[row["eqkey"]]
                decisions = {}
                for method in METHODS:
                    route_start = time.monotonic()
                    decisions[method] = asdict(routers[method].route(key))
                    costs[method]["route_seconds"] += time.monotonic()-route_start
                    costs[method]["feature_seconds"] += feature_seconds
                if decisions["S0"] != decisions["S1"]:
                    raise RuntimeError("shared P4 bank route mismatch")
                role = roles[(source, row["logical_id"])]
                for method in METHODS:
                    method_start = time.monotonic()
                    decision = decisions[method]
                    selected = decision["logical_edit_id"]
                    base, base_provenance = output(method, None, row)
                    own, own_provenance = output(method, source, row)
                    actual, provenance = output(method, selected, row)
                    branch = "OFF" if selected is None else "ON"
                    branches = [branch] + (["WRONG_SELECTION"] if selected not in (None, source) else [])
                    needed = [b for b in branches if result["replays"][method][b]["status"] == "NOT_OBSERVED"]
                    if needed:
                        replay_start = time.monotonic()
                        replay = generate(method, selected, row)
                        parity = same_output(actual, replay)
                        evidence = dict(status="PASSED" if parity else "FAILED", source_expert=source,
                            logical_id=row["logical_id"], selected_expert=selected, route=decision,
                            output=replay, elapsed_seconds=time.monotonic()-replay_start)
                        for b in needed:
                            result["replays"][method][b] = evidence
                        costs[method]["replay_seconds"] += evidence["elapsed_seconds"]
                        if not parity:
                            raise RuntimeError("deterministic bank replay differs from exact-bound output")
                    item = dict(row=row, base=base, actual=actual, own_forced=own, method=method,
                        source_edit_index=e["event_index"], source_expert=source, selected_expert=selected,
                        nearest_expert=decision["nearest_logical_edit_id"], prefix=prefix, on=decision["activated"],
                        route=decision, **role, selection_class=selection_class(selected, source, role),
                        first_evaluable_prefix=first, history_key=source+":"+row["logical_id"],
                        provenance=dict(actual=provenance, base=base_provenance, own_forced=own_provenance),
                        costs=dict(request_seconds=time.monotonic()-method_start, feature_seconds=feature_seconds),
                        error_class="PENDING_JUDGE", exposure="STAGE2_VIEWED_REPLAY")
                    result["outputs"].append(item)
                    costs[method]["request_seconds"] += item["costs"]["request_seconds"]
                if editor.wrapper.active_logical_id is not None:
                    raise RuntimeError("active writer leaked after bank request")
                result["last_group_request_seconds"] = time.monotonic()-request_start
        # Explicit active->OFF check uses one real input and its frozen Base answer.
        native = evaluation_rows(episodes[0])[0]
        disabled = generate("B", None, native)
        if not same_output(disabled, cached[(None, native["eqkey"])][0]):
            raise RuntimeError("bank unload failed Base parity")
        result["disabled_replay"] = dict(status="PASSED", output=disabled)
        result["status"] = "RAW_READY"
    except Exception as exc:
        result["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if editor is not None:
            editor.reset_editor_state()
        runtime.replace_module(target, base_module)
        result["base_guard"] = runtime.base_guard.verify()
        if not result["base_guard"]["unchanged"]:
            result["status"] = "PARTIAL"
            result["error"] = dict(type="BaseMutation", message="frozen Base changed")
        result["elapsed_seconds"] = time.monotonic()-started
        result["support"] = {m: dict(Counter(v["strict_role"] for v in result["outputs"] if v["method"] == m))
                             for m in METHODS}
        vf.atomic_json(out / "result_private.json", result)
        if not result["base_guard"]["unchanged"]:
            raise RuntimeError("frozen Base changed")
    return result
