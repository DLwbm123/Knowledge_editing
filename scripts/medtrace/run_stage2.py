#!/usr/bin/env python3
"""Stage2 fixed single-edit comparison; reuse V4 BalancEdit and V1 CP trainers."""
import argparse
from dataclasses import replace
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.medtrace import run_selective_write as sw
from scripts.medtrace.run_dev16 import evaluation_rows, run_event
from scripts.medtrace.run_generality_ablation import train_condition
from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor

vf, read = sw.vf, sw.read
SEED = 20260910
NEW_METHODS = (("P4", sw.CONDITIONS[0]), ("P4", "W1_KL_0.1"),
               ("P4", sw.CONDITIONS[-1]), ("L16", sw.CONDITIONS[-1]))


def queue_task(edit, kind, priority, dependency=None, parameterization="BE", condition="BALANCEDIT"):
    return dict(task_id=f"S2_e{edit:03d}_{kind}_{parameterization}_{condition}", event_index=edit,
                kind=kind, parameterization=parameterization, condition=condition,
                priority=priority, depends_on=dependency, attempts=0, status="PENDING", seed=SEED)


def prepare(args):
    run, old = args.run_root, args.stage1_run
    if run.exists():
        raise FileExistsError(run)
    config = read(old / "private/CAMPAIGN_CONFIG.json")
    config.update(kind="MEDTRACE_STAGE2_V2", stage1_run=str(old), seed=SEED,
                  code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  train_seconds=20*3600, wall_hours=24, gpu_hours=48, gpu_uuids=sw.GPUS)
    if args.reuse_stage2_run:
        config.update(reuse_stage2_run=str(args.reuse_stage2_run),
                      campaign_epoch=read(args.reuse_stage2_run / "COORDINATOR_START.json")["epoch"])
    run.mkdir(parents=True)
    (run / "private/edits").mkdir(parents=True)
    vf.atomic_json(run / "private/CAMPAIGN_CONFIG.json", config)
    frozen = read(Path(config["old_run"]) / "private/frozen_data.json")
    # QUAL8 states are a different event cohort; record ID match is necessary, not sufficient.
    qual = read(Path(config["runtime"]["cpu_gate"]).parent / "qual8_v2/balancedit/QUAL8_RAW_MANIFEST.json")
    overlap = set(qual["event_ids"]) & {e["event_id"] for e in frozen["dev"]}
    if overlap:
        raise RuntimeError("matching BalancEdit IDs require exact artifact binding review before retraining")
    vf.atomic_json(run / "private/BALANCEDIT_REUSE_AUDIT.json", dict(dev16=16, qual8=8,
        matched_event_ids=0, action="train missing DEV16 independent states; QUAL8 is not reused as DEV16"))
    tasks = []
    for i, event in enumerate(frozen["dev"], 1):
        source = old / f"private/edits/e{i:02d}.json"
        if source.exists():
            data = read(source)
            extension = old / f"private/initial/e{i:02d}/A2_FIT_EXTENSION_PRIVATE.json"
            if extension.exists():
                extra = read(extension)
                data["rows"].extend(extra["rows"])
            data["extra_fit"] = []
        else:
            rows = [dict(logical_id="native" if r["task"] == "T0" else "formal-"+r["query_id"],
                         role="native" if r["task"] == "T0" else "formal_development",
                         label="positive", question=r["question"], reference=r["reference"],
                         image_path=r["image_path"], fact_relation=r["task"], task=r["task"])
                    for r in evaluation_rows(event)]
            data = dict(event=event, rows=rows, extra_fit=[])
        data.update(track="OLD_DEV16", event_index=i)
        vf.atomic_json(run / f"private/edits/e{i:02d}.json", data)
        task = queue_task(i, "BE", i)
        if args.reuse_stage2_run:
            previous = args.reuse_stage2_run / "private/tasks" / task["task_id"]
            if (previous / "editor_state.pt").exists() and (previous / "TRAINING_PRIVATE.json").exists():
                task["reuse_be_directory"] = str(previous)
        tasks.append(task)
    vf.atomic_json(run / "private/TASK_QUEUE.json", dict(tasks=tasks))


def query_record(record, row):
    # Neither reference nor expected expert/task is a routing input.
    return replace(record, record_id=row["logical_id"], question=row["question"],
                   image_path=Path(row["image_path"]), target="", official_rephrase="")


def bind_rows(runtime, data):
    record = vf.EditorRecord.from_dict(data["event"]["edit_record"])
    for row in data["rows"]:
        if "eqkey" not in row:
            batch = runtime.build_question_batch(query_record(record, row))
            row["eqkey"] = vf.sha256_json(dict(image=batch.image_sha256, ids=batch.raw_input_ids.tolist(),
                attention=batch.attention_mask.tolist() if batch.attention_mask is not None else None,
                boundary=batch.key_token_index, generation=runtime.generation_config))
        row.setdefault("source_group", vf.sha256_json((record.dataset, str(Path(row["image_path"]).resolve()))))
    if data['track'] == 'NEW_CONFIRMATION':
        seen = {}
        for row in data['rows']:
            value = (row['role'], row['label'])
            if row['eqkey'] in seen and seen[row['eqkey']] != value:
                raise ValueError('new realized model input crosses roles or labels')
            seen[row['eqkey']] = value


def same_output(a, b):
    return a["raw_answer"] == b["raw_answer"] and a["raw_token_ids"] == b["raw_token_ids"]


def base_for(runtime, run, data, row):
    index = data["event_index"]
    path = run / f"private/base_generation/e{index:02d}" / (vf.sha256_json(row)+".json")
    if path.exists():
        return read(path)
    config = read(run / "private/CAMPAIGN_CONFIG.json")
    historical = Path(config["stage1_run"]) / f"private/base_generation/e{index:02d}" / path.name
    if data["track"] == "OLD_DEV16" and historical.exists():
        value = read(historical)
    else:
        canonical = {r["query_id"]: r for r in evaluation_rows(data["event"])}
        qid = data["event"]["edit_record"]["record_id"] if row["role"] == "native" else row["logical_id"].removeprefix("formal-")
        source = canonical.get(qid)
        known = runtime.stage2_base.get(qid)
        # Augmented positives inherit source lineage, NOT the native model input.
        # Their original source query ID must never substitute a paraphrase Base answer.
        if known is None and row.get('base_query_ids') and (row['role'] == 'native' or row['label'] == 'negative'):
            candidates = [runtime.stage2_base[q] for q in row['base_query_ids'] if q in runtime.stage2_base]
            if candidates and all((c['model_answer_raw'], c['raw_generated_token_ids']) == (candidates[0]['model_answer_raw'], candidates[0]['raw_generated_token_ids']) for c in candidates):
                known = candidates[0]
                source = row  # scanner joined source image/question/reference to the canonical query catalog
        if source and known and all(source[k] == row[k] for k in ("question", "image_path", "reference")):
            prepared = runtime.adapter.prepare_inputs(row['image_path'], row['question'], None)
            bound = (prepared['input_ids'][0].tolist() == known['prompt_token_ids'] and
                     prepared['image_sha256'] == known['image_sha256'] and not known.get('error'))
            value = (dict(raw_answer=known["model_answer_raw"], raw_token_ids=known["raw_generated_token_ids"],
                          provenance="frozen V4 Base exact prompt tokens and image binding") if bound else vf.scope_generate(runtime, row, None))
        else:
            value = vf.scope_generate(runtime, row, None)
    vf.atomic_json(path, value)
    return value


def be_task(runtime, args, task):
    run = args.run_root
    config = read(run / "private/CAMPAIGN_CONFIG.json")
    i = task["event_index"]
    data = read(run / f"private/edits/e{i:02d}.json")
    if data["track"] == "NEW_CONFIRMATION" and not (run / "private/BASE_BEFORE_ROLE_LOCK_PRIVATE.json").exists():
        raise RuntimeError("new student requires frozen same-Judge Base-before membership")
    bind_rows(runtime, data)
    vf.atomic_json(run / f"private/edits/e{i:02d}.json", data)
    record = vf.EditorRecord.from_dict(data["event"]["edit_record"])
    out = run / "private/tasks" / task["task_id"]
    out.mkdir(parents=True, exist_ok=True)
    runtime.run_root = out
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    editor = BalanceEditPaperSpecEditor(runtime)
    base_module = editor.wrapper.base
    target = editor.target
    outputs = {}
    try:
        lock = editor.config_lock()
        if lock["alpha"] != .2 or lock["steps_per_edit"] != 50 or lock["learning_rate"] != .01:
            raise RuntimeError("frozen BalancEdit config drift")
        vf.atomic_json(out / "METHOD_CONFIG_LOCK.json", lock)
        native = next(r for r in data["rows"] if r["role"] == "native")
        with editor.disabled():
            base_native = base_for(runtime, run, data, native)
            if not same_output(base_native, vf.scope_generate(runtime, native, None)):
                raise RuntimeError("target-free no-edit Base parity failure")
        reused = task.get("reuse_be_directory")
        if reused:
            previous = Path(reused)
            training = read(previous / "TRAINING_PRIVATE.json")
            if training["record_id"] != record.record_id or read(previous / "METHOD_CONFIG_LOCK.json") != lock:
                raise ValueError("reused transform identity/config mismatch")
            editor.load_editor_state(previous / "editor_state.pt")
            if editor.router.logical_ids != [record.record_id] or editor.edit_history != [record.record_id]:
                raise ValueError("reuse is not this independent single edit")
            training_seconds = None  # original failed closure did not persist a training timer
        else:
            train_start = time.monotonic()
            training = editor.apply_edit(record)
            training_seconds = time.monotonic()-train_start
        if training["steps"] != 50 or not training["finite_losses"] or not training["finite_gradients"]:
            raise RuntimeError("BalancEdit did not execute fixed 50 steps")
        # Native implementation computes anchors with all transforms disabled before writing.
        vf.atomic_json(out / "TRAINING_PRIVATE.json", training)
        if config.get('kind') == 'MEDTRACE_STAGE5':
            from scripts.medtrace.stage5_single_scope import freeze
            freeze(runtime, editor, data, run)
        hook_off = SimpleNamespace(clear_request_routing=lambda: editor.wrapper.set_active(None))
        teacher = sw.TeacherCache(runtime, run, i, config)
        for row in data["rows"]:
            if sw.active_elapsed(run) >= config["train_seconds"]:
                raise TimeoutError("Stage2 training/generation boundary")
            with editor.disabled():
                base = base_for(runtime, run, data, row)
            with editor._activated(record.record_id):
                forced = vf.scope_generate(runtime, row, None)
            actual = editor.generate(query_record(record, row))
            native_output = dict(raw_answer=actual["generation"]["decoded_text"], raw_token_ids=actual["generation"]["raw_token_ids"])
            on = actual["route"]["activated"]
            expected = forced if on else base
            parity = same_output(native_output, expected)
            if editor.wrapper.active_logical_id is not None:
                raise RuntimeError("BalancEdit request state leaked")
            kl = None
            if row["label"] == "negative":
                # Read-through existing teacher only with its complete original binding.
                prior = Path(config["stage1_run"]) / f"private/teacher/e{i:02d}/score_only" / (vf.sha256_json(row)+".pt")
                if data["track"] == "OLD_DEV16" and prior.exists():
                    cached = torch.load(prior, map_location="cpu", weights_only=True)
                    kwargs, labels, mask, binding = sw.teacher_batch(runtime, row, cached["tokens"])
                    old_config = read(Path(config["stage1_run"]) / "private/CAMPAIGN_CONFIG.json")
                    expected_lock = dict(teacher.lock, code=old_config["code_commit"])
                    binding.update(lock=expected_lock, role=row["role"], cache_scope="score_only")
                    if cached["binding"] != binding:
                        raise ValueError("historical teacher binding mismatch")
                    logp = cached["logp"]
                else:
                    kwargs, labels, mask, logp = teacher.get(row, hook_off, training=False)
                with editor._activated(record.record_id), torch.no_grad():
                    kl = float(sw.full_vocab_kl(runtime.model(**kwargs).logits[mask], logp, chunk=16))
                del kwargs, labels, mask, logp
            outputs[row["logical_id"]] = dict(row=row, base=base, forced=forced, fixed=native_output,
                fixed_on=on, route=actual["route"], kl=kl, route_branch_parity=parity,
                disabled_parity=row['role'] == 'native', disabled_evidence='actual native replay; other inputs use exact-bound frozen Base cache')
        checkpoint = out / "editor_state.pt"
        if reused:
            checkpoint = Path(reused) / "editor_state.pt"
            saved = dict(size_bytes=checkpoint.stat().st_size)
        else:
            with editor.disabled():
                saved = editor.save_editor_state(checkpoint)
        editor.reset_editor_state()
        if not same_output(vf.scope_generate(runtime, native, None), base_native):
            raise RuntimeError("unload did not restore Base")
        load_start = time.monotonic()
        editor.load_editor_state(checkpoint)
        load_seconds = time.monotonic()-load_start
        with editor._activated(record.record_id):
            if not same_output(vf.scope_generate(runtime, native, None), outputs[native['logical_id']]["forced"]):
                raise RuntimeError("BalancEdit saved-state reload mismatch")
        gates = {lid: value["fixed_on"] for lid, value in outputs.items()}
        if data["track"] in {"NEW_CONFIRMATION", "V4_STAGE3"}:
            data.update(frozen_gate=gates, frozen_gate_sha256=vf.sha256_json(gates),
                        router_provenance="BE pre-edit Base anchors; single expert; no eval threshold calibration")
            vf.atomic_json(run / f"private/edits/e{i:02d}.json", data)
        state = editor.state_summary()
    finally:
        editor.reset_editor_state()
        runtime.replace_module(target, base_module)
    guard = runtime.base_guard.verify()
    if not guard["unchanged"]:
        raise RuntimeError("frozen Base changed")
    result = dict(status="RAW_READY", task=task, step=50, outputs=outputs, base_guard=guard,
        training_seconds=training_seconds, elapsed_seconds=time.monotonic()-started,
        parameters=state["edited_parameter_count"], storage_bytes=saved["size_bytes"], load_seconds=load_seconds,
        peak_vram_bytes=torch.cuda.max_memory_allocated(), mode_identity="BalancEdit V4 adaptation",
        system_status="ACTUAL_NATIVE_ROUTING", training=training, reused_completed_transform=bool(reused),
        additional_optimizer_steps=0 if reused else 50, disabled_actual_replay_scope='native')
    vf.atomic_json(out / "result_private.json", result)
    return result


def initialize_episode(runtime, args, task):
    """Runnable native->A2 entry: early stop reads native only, never held-out probes."""
    run, i = args.run_root, task["event_index"]
    data = read(run / f"private/edits/e{i:02d}.json")
    if data["track"] not in {"NEW_CONFIRMATION", "V4_STAGE3"} or not data.get("source_eligibility_frozen"):
        raise ValueError("initializer requires a pre-student authorized source manifest")
    if data["track"] == "NEW_CONFIRMATION" and not (run / "private/BASE_BEFORE_ROLE_LOCK_PRIVATE.json").exists():
        raise RuntimeError("new initializer requires frozen Base-before Judge membership")
    bind_rows(runtime, data)
    event = dict(data["event"], probes=[])
    out = run / f"private/initial/e{i:02d}/native"
    out.mkdir(parents=True, exist_ok=False)
    native = next(r for r in data["rows"] if r["role"] == "native")
    before = base_for(runtime, run, data, native)
    record = vf.EditorRecord.from_dict(event["edit_record"])
    base = {record.record_id: dict(raw_generated_token_ids=before["raw_token_ids"], model_answer_raw=before["raw_answer"])}
    result = run_event(runtime, event, base, out, seed_base=SEED,
                       condition_limit=None if data['track'] == 'V4_STAGE3' else 1e4)
    vf.atomic_json(out / "result.json", result)
    checkpoint = torch.load(out / "expert.pt", map_location=runtime.device, weights_only=True)
    cp, metadata = train_condition(runtime, record, [r["question"] for r in data["fit_paraphrases"]], checkpoint, sw.A2, seed_base=SEED)
    a2 = out.parent / "A2.pt"
    sw.save(a2, dict(rank=4, step=80, condition=sw.A2, record_id=record.record_id, seed=metadata["seed"], expert=cp.state_dict()))
    vf.atomic_json(out.parent / "A2_INITIALIZATION_PRIVATE.json", metadata)
    batch = runtime.build_question_batch(record)
    with torch.no_grad():
        activation = runtime.extract_layer_input_features(batch, module_path=vf.LAYER).cpu()
        fixture = torch.cat([activation[batch.key_token_index][None], activation[batch.image_token_start:batch.image_token_end][:8]])
        cp = cp.cpu()
        low_rank = sw.LowRankExpert(cp, vf.derive_seed(record.record_id, SEED))
        expected, actual = cp.residual(fixture), low_rank.residual(fixture)
        transfer_ok = bool(torch.allclose(expected, actual, rtol=2e-5, atol=2e-5))
    vf.atomic_json(out.parent / 'A2_TRANSFER_PRIVATE.json', dict(transfer_cpu=transfer_ok,
        actual_activation_max_abs=float((expected-actual).abs().max()),
        actual_activation_relative_l2=float((expected-actual).norm()/expected.norm().clamp_min(1e-12)),
        input='pre-intervention Base native prompt and first eight visual tokens', rtol=2e-5, atol=2e-5))
    data.update(a2=str(a2), a2_sha256=vf.sha256_file(a2), transfer_cpu=transfer_ok, extra_fit=[])
    vf.atomic_json(run / f"private/edits/e{i:02d}.json", data)
    return dict(elapsed_seconds=result["timing"]["end_to_end_seconds"]+metadata["training_seconds"])


def base_task(runtime, args, task):
    start = time.monotonic()
    data = read(args.run_root / f"private/edits/e{task['event_index']:02d}.json")
    bind_rows(runtime, data)
    for row in data['rows']:
        base_for(runtime, args.run_root, data, row)
    vf.atomic_json(args.run_root / f"private/edits/e{task['event_index']:02d}.json", data)
    return dict(elapsed_seconds=time.monotonic()-start)


def worker(args):
    run = args.run_root
    config = read(run / "private/CAMPAIGN_CONFIG.json")
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    resource = sw.gpu_check(gpu)
    runtime = vf.load_real_runtime(SimpleNamespace(cpu_gate=Path(config["runtime"]["cpu_gate"])))
    runtime.stage2_base = {r["query_id"]: r for r in vf.read_jsonl(Path(config["runtime"]["base_predictions"]))}
    with (run / "private/start.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not (run / "private/CAMPAIGN_START.json").exists():
            vf.atomic_json(run / "private/CAMPAIGN_START.json", dict(epoch=config.get('campaign_epoch', time.time()), code_commit=config["code_commit"]))
    sw.SEED = SEED
    queue = vf.TaskQueue(run / ('private/BASE_TASK_QUEUE.json' if args.base_only else 'private/TASK_QUEUE.json'), run)
    with vf.Telemetry(run / "private/GPU_TELEMETRY.jsonl", gpu, f"gpu{gpu}") as telemetry:
        while sw.active_elapsed(run) < config["train_seconds"] and not (run / "STOP").exists():
            task = queue.claim(f"gpu{gpu}")
            if task is None:
                break
            telemetry.task_id = task["task_id"]
            print("START", task["task_id"], flush=True)
            try:
                result = (base_task(runtime, args, task) if args.base_only else be_task(runtime, args, task) if task["kind"] == "BE" else
                          initialize_episode(runtime, args, task) if task["kind"] == "INIT" else
                          sw.train_task(runtime, args, task))
                queue.update(task["task_id"], "COMPLETE" if task["kind"] in {"INIT", "BASE"} else "RAW_READY",
                             elapsed_seconds=result["elapsed_seconds"], finished_epoch=time.time())
                print("DONE", task["task_id"], flush=True)
            except Exception as error:
                queue.update(task["task_id"], "FAILED", last_error=f"{type(error).__name__}: {error}")
                # End this process on engineering failure to avoid a contaminated runtime.
                raise
            gc.collect(); torch.cuda.empty_cache()
            if args.first_only:
                break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "worker"))
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--stage1-run", type=Path)
    p.add_argument("--reuse-stage2-run", type=Path)
    p.add_argument("--base-only", action='store_true')
    p.add_argument("--first-only", action="store_true")
    args = p.parse_args()
    {"prepare": prepare, "worker": worker}[args.action](args)


if __name__ == "__main__":
    main()
