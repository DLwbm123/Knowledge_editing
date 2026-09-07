#!/usr/bin/env python3
"""Selective-write V1: isolated fit caches, exact-token teacher prefixes, 70 paired jobs."""
import argparse
import csv
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import replace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from methods.medtrace.selective_write import (CONDITIONS, RELATIONS, LowRankExpert, Protection,
    optimizer_for, predictor_mask, full_vocab_kl, fit_groups, balanced_schedule, group_mean)
from scripts.medtrace import run_frozen_expert_visual_verifier as vf
from scripts.medtrace.run_longrun_campaign import scope_rows

SEED = 20260906
A2 = "CP_NATIVE_PLUS_PARAPHRASE_80"
GPUS = {"0": "GPU-688ed629-7a46-b62c-e3e4-0940e1649b5f", "1": "GPU-5fac6011-5432-229f-bb11-9ce9f104a334"}
STEPS = 320


def read(path):
    return json.loads(Path(path).read_text())


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    for attempt in range(3):
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(.5)


def load_cp(path, device="cpu"):
    value = torch.load(path, map_location=device, weights_only=True)
    if value.get("condition") != A2 or value.get("step") != 80 or value.get("rank") != 4:
        raise ValueError("not an original native+fit-paraphrase A2 checkpoint")
    expert = vf.AsymmetricCPExpert(14336, 4096, 4).to(device)
    expert.load_state_dict(value["expert"])
    return expert, value


def prepare(args):
    run, old, prior = args.run_root, args.old_run, args.verifier_run
    if run.exists():
        raise FileExistsError(run)
    frozen = read(old / "private/frozen_data.json")
    hard = [(i, e) for i, e in enumerate(frozen["dev"], 1)
            if frozen["scopes"][e["edit_record"]["record_id"]]["status"] == "HARD_EVALUABLE"]
    if [i for i, _ in hard] != [1, 2, 3, 7, 13, 15, 16]:
        raise ValueError("frozen hard-evaluable cohort changed")
    config = dict(kind="selective_write_v1", code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  old_run=str(old), verifier_run=str(prior), runtime=read(old / "private/CAMPAIGN_RUNTIME_CONFIG.json"),
                  gpu_uuids=GPUS, authorization="User 2026-09-07: 可以用 gpu0 和 1; supersedes plan GPU1 prohibition; idle-only",
                  wall_hours=24, gpu_hours=48, train_seconds=20*3600, seed=SEED, novel_seed=20260910, novel_n=0,
                  source_locks={"frozen_data": vf.sha256_file(old / "private/frozen_data.json")})
    run.mkdir(parents=True); (run / "private").mkdir()
    audits, inventories = [], []
    for i, event in hard:
        rid = event["edit_record"]["record_id"]
        scope = frozen["scopes"][rid]
        path = old / f"private/tasks/A_s{SEED}_e{i:02d}" / A2 / "expert.pt"
        result = read(path.parent / "result.json")
        cp, checkpoint = load_cp(path)
        if result["seed_base"] != SEED or checkpoint["seed"] != vf.derive_seed(rid, SEED) or checkpoint["record_id"] != rid:
            raise ValueError("A2 source identity mismatch")
        digest = vf.sha256_file(path)
        if digest != result["checkpoint_sha256"]:
            raise ValueError("A2 checkpoint differs from historical manifest")
        lr = LowRankExpert(cp, vf.derive_seed(rid, SEED))
        original = torch.load(old / f"private/features/e{i:02d}.pt", mmap=True, map_location="cpu", weights_only=False)
        matched = torch.load(prior / f"private/features/e{i:02d}.pt", mmap=True, map_location="cpu", weights_only=False)
        history = read(prior / f"private/tasks/VERIFY_s{SEED}_e{i:02d}/result_private.json")
        if not history["base_guard"]["unchanged"] or matched["locks"] != history["matched_feature_locks"]:
            raise ValueError("historical feature/router binding mismatch")
        # Check real, pre-intervention prompt/visual activations without GPU or a dense 4096x14336 map.
        fixture = next(iter(original["values"].values()))
        activations = torch.cat([fixture["prompt"][None], fixture["visual"][:8]], dim=0)
        with torch.no_grad():
            expected, actual = cp.residual(activations), lr.residual(activations)
            error = float((expected-actual).abs().max())
            relative = float((expected-actual).norm()/expected.norm().clamp_min(1e-12))
            transfer_ok = torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)
        audits.append(dict(edit=i, checkpoint_binding="PASS", actual_activation_max_abs=error,
                           actual_activation_relative_l2=relative, transfer_cpu=bool(transfer_ok),
                           p4_parameters=sum(p.numel() for p in cp.parameters()), l16_parameters=sum(p.numel() for p in lr.parameters()),
                           beta=cp.beta, cp_scale=cp.beta/math.sqrt(cp.rank), natural_generation="NOT_RUN"))
        rows, frozen_gate = [], {}
        for panel, cache in (("matched", matched), ("original", original)):
            for lid, value in cache["values"].items():
                row = dict(value["row"])
                # Original hard images repeat matched sources; retain only matched H for the primary hard stratum.
                if panel == "original" and row["fact_relation"] == RELATIONS["H"]:
                    continue
                # Avoid counting two sets of fit/cal/eval positives: approved matched panel is canonical.
                if panel == "original" and row["label"] == "positive" and row["role"] in {"fit", "calibration", "evaluation"}:
                    continue
                row.update(logical_id=f"{panel}:{lid}", eqkey=value["eqkey"], panel=panel,
                           source_group=row.get("source_group") or vf.sha256_json(row["image_path"])[:16])
                if lid not in history["scores"][panel]:
                    raise ValueError("missing frozen M0 score")
                point = "CONTINUITY_SAFETY_FIRST" if panel == "original" else vf.POINTS[0]
                frozen_gate[row["logical_id"]] = vf._decision(history["scores"][panel][lid][vf.CONDITIONS[0]],
                    history["calibration"][panel][vf.CONDITIONS[0]][point])
                rows.append(row)
        for g, relation in RELATIONS.items():
            pool = [r for r in rows if r["role"] == "fit" and r["fact_relation"] == relation]
            groups = fit_groups(pool, g)
            inventories.append(dict(edit=i, group=g, role="fit", rows=len(pool), eqkeys=len({r['eqkey'] for r in pool}),
                                    source_images=len(groups), relation=relation))
        paras = frozen["generality_paraphrases"][rid]
        if not paras or any(p["review_status"] != "APPROVED_EQUIVALENT" for p in paras):
            raise ValueError("unapproved A2 fit paraphrases")
        fit_q = {p["question"] for p in scope["positives"]["fit"]}
        if not {p["question"] for p in paras} <= fit_q:
            raise ValueError("A2 fit paraphrase roles differ")
        negative_roles = {}
        for row in rows:
            if row["label"] == "negative":
                prior_role = negative_roles.setdefault(row["eqkey"], row["role"])
                if prior_role != row["role"]:
                    raise ValueError("negative EqKey crosses roles")
        vf.atomic_json(run / f"private/edits/e{i:02d}.json", dict(event=event, a2=str(path), a2_sha256=digest,
            rows=rows, fit_paraphrases=paras, frozen_gate=frozen_gate, frozen_gate_sha256=vf.sha256_json(frozen_gate),
            router_provenance=dict(source=str(prior), checkpoint=history["executor_lock"], calibration=history["calibration"]),
            transfer_cpu=bool(transfer_ok), cache_locks={"original": original["locks"], "matched": matched["locks"]}))
        del original, matched, cp, lr
        gc.collect()
    # Rotate conditions across all edits, with P4/L16 paired at each edit.
    tasks = []
    for condition in (CONDITIONS[1], CONDITIONS[0], *CONDITIONS[2:]):
        for i, event in hard:
            for parameterization in ("P4", "L16"):
                tasks.append(dict(task_id=f"SW_e{i:02d}_{parameterization}_{condition}", kind="SELECTIVE_WRITE", event_index=i,
                    parameterization=parameterization, condition=condition, seed=SEED, priority=len(tasks), status="PENDING", attempts=0))
    vf.atomic_json(run / "private/CAMPAIGN_CONFIG.json", config)
    vf.atomic_json(run / "private/TASK_QUEUE.json", dict(tasks=tasks))
    vf.atomic_json(run / "private/INPUT_MANIFEST.json", dict(tasks=tasks, edits={str(i): vf.sha256_file(run / f"private/edits/e{i:02d}.json") for i, _ in hard}))
    public = args.public_dir
    public.mkdir(parents=True, exist_ok=False)
    vf.atomic_json(public / "TRANSFER_CPU_RESULTS.json", audits)
    vf.atomic_json(public / "FIT_SOURCE_INVENTORY.json", inventories)
    vf.atomic_json(public / "SELECTIVE_WRITE_PROTOCOL.json", dict(version="V1", research_code=config["code_commit"],
        authorization=config["authorization"], gpu_uuids=GPUS, cohort=[i for i,_ in hard], seed=SEED, novel_seed=20260910,
        tasks=tasks, teacher="V4 Base, every expert OFF; same exact generated prefix", kl="Base||ON, full vocabulary, FP32, T=1, answer predictor mean",
        prefix_cap=128, eos="include if generated within cap", steps=320, diagnostics=[0,80,160,320],
        optimizer=dict(name="Adam", input_lr=1e-4, output_lr=1e-3, betas=[.9,.999], eps=1e-8, weight_decay=0, clip=1),
        positive_loss="0.5 native CE + 0.5 rotating approved fit paraphrase CE", teacher_cache="fit-only; score-only cache physically separate",
        grouping="EqKey question mean -> source-image mean -> edit mean; edit bootstrap, no seed inflation",
        W1=dict(lambdas=[.1,1,10], loss="Lpos+lambda*(KH/sH+KU/sU)/2", selection="global per parameterization; cal positive >= A2-.05 AND W0-.05, cal U KL <= W0; smallest cal H KL then smaller lambda; no eligible -> 1 unqualified"),
        W2=dict(scale="max(initial A2 group KL, 1e-3)", epsilon="max(.5*initial A2 group KL,1e-4)", dual_init=.5, dual_lr=.05, ema=.9, dual_clip=[0,20], tuning="none"),
        evaluation=["FORCED_ON primary", "FIXED_ROUTER original M0 decisions secondary", "DISABLED Base parity"],
        selection_role="calibration only", main_parameterization="L16 predeclared; not capacity matched", old_scientific_gain=False,
        wall_hours=24, gpu_hours=48, train_hours=20, judge_reserve_hours=4, novelty="unproven; not TIME/LiveEdit/M-ORE reproduction"))
    vf.atomic_text(public / "TEACHER_AND_TRANSFER_AUDIT.md", "# Teacher and transfer audit\n\nOriginal pre-scope A2 checkpoint and seed bindings checked against historical manifests for all 7 edits. See TRANSFER_CPU_RESULTS.json for real pre-CP prompt/visual residual errors and parameter counts. No SVD. CP scaling is beta/sqrt(4). L16 extra rows are seeded unit vectors; extra output columns are zero.\n\nTeacher uses exact generated token IDs appended to the target-free multimodal prompt, not detokenized/re-tokenized answers. Shifted answer labels include first-answer and generated EOS predictors and exclude image/padding. FP32 full-vocabulary Base||student has live student gradients. Fit and score-only cache roots are separate. Runtime/model/image/prompt/prefix/mask/dtype are cache-bound. Natural-generation transfer and real-model disabled parity must still pass before any L16 trajectory.\n")
    summarize(args)


def gpu_check(gpu):
    if gpu not in GPUS:
        raise ValueError("only newly authorized GPU0/1 are used by this attempt")
    uuid, used = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    if uuid != GPUS[gpu] or int(used) > 1000:
        raise RuntimeError(f"GPU{gpu} busy or UUID mismatch: {uuid}, {used} MiB")
    return dict(gpu=gpu, uuid=uuid, used_mib=int(used), checked_epoch=time.time())


def teacher_batch(runtime, row, tokens):
    if not tokens or len(tokens) > 128:
        raise ValueError("invalid teacher answer length")
    raw = runtime.adapter.prepare_inputs(row["image_path"], row["question"], None)
    answer = torch.tensor([tokens], device=runtime.device, dtype=raw["input_ids"].dtype)
    ids = torch.cat([raw["input_ids"], answer], dim=1)
    attention = torch.cat([raw["attention_mask"], torch.ones_like(answer)], dim=1)
    labels = torch.cat([torch.full_like(raw["input_ids"], -100), answer], dim=1)
    embeds, attention, positions, labels = runtime._expand_multimodal(raw_input_ids=ids,
        attention_mask=attention, labels=labels, images=raw["images"])
    mask = predictor_mask(labels, attention)
    if int(mask.sum()) != len(tokens) or labels[0, labels[0] != -100].tolist() != tokens:
        raise ValueError("teacher causal alignment changed after image expansion")
    binding = dict(prompt_tokens=raw["input_ids"][0].tolist(), image=raw["image_sha256"], image_shape=list(embeds.shape),
        positions=positions.tolist() if positions is not None else None, attention=attention.tolist() if attention is not None else None,
        labels=labels.tolist(), tokens=tokens, dtype=str(embeds.dtype), predictor_positions=mask.nonzero().tolist(), eqkey=row["eqkey"])
    return dict(inputs_embeds=embeds, attention_mask=attention, position_ids=positions, labels=None, use_cache=False, return_dict=True), labels, mask, binding


class TeacherCache:
    def __init__(self, runtime, run, edit, config):
        self.runtime, self.run, self.edit, self.config = runtime, run, edit, config
        self.root = run / f"private/teacher/e{edit:02d}"
        self.lock = dict(runtime=vf.sha256_file(Path(config["runtime"]["runtime_lock"])),
                         generation=runtime.generation_config, layer=vf.LAYER, temperature=1., cap=128,
                         code=config["code_commit"], teacher="ALL_EXPERTS_OFF")
        self.forward_count = 0

    def get(self, row, hook, *, training):
        if training and (row["role"] != "fit" or row["fact_relation"] not in RELATIONS.values()):
            raise ValueError("non-fit input requested from training teacher cache")
        scope = "fit" if training else "score_only"
        path = self.root / scope / (vf.sha256_json(row)+".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        hook.clear_request_routing()
        with path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                tokens = saved["tokens"]
            else:
                generated = vf.scope_generate(self.runtime, row, None, cap=128)
                tokens = generated["raw_token_ids"]
                saved = None
            kwargs, labels, mask, binding = teacher_batch(self.runtime, row, tokens)
            binding.update(lock=self.lock, role=row["role"], cache_scope=scope)
            if saved is not None:
                if saved["binding"] != binding:
                    raise ValueError("teacher cache binding mismatch")
            else:
                with torch.no_grad():
                    logits = self.runtime.model(**kwargs).logits[mask]
                    logp = logits.float().log_softmax(-1).cpu()
                    self.forward_count += 1
                saved = dict(binding=binding, tokens=tokens, logp=logp, cap_hit=len(tokens) == 128,
                             eos_included=generated["ended_with_eos"])
                save(path, saved)
            if saved["logp"].dtype != torch.float32 or saved["logp"].requires_grad:
                raise ValueError("invalid cached teacher distribution")
        return kwargs, labels, mask, saved["logp"]

    def kl(self, row, hook, *, training, chunk):
        kwargs, labels, mask, logp = self.get(row, hook, training=training)
        hook.set_teacher_routing(labels)
        if not hook.enabled or not torch.equal(hook.token_mask, mask):
            raise RuntimeError("negative expert must be ON at every answer predictor")
        logits = self.runtime.model(**kwargs).logits[mask]
        return full_vocab_kl(logits, logp, chunk=chunk)


def generated(runtime, row, expert):
    if expert is None:
        return vf.scope_generate(runtime, row, None)
    hook = vf.MedTraceLayerHook(runtime.get_module(vf.LAYER), expert)
    hook.attach()
    try:
        return vf.scope_generate(runtime, row, hook)
    finally:
        hook.detach()


def shared_initial(runtime, run, task, data, config):
    directory = run / f"private/initial/e{task['event_index']:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "initial.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cp, _ = load_cp(data["a2"], runtime.device)
        cp.requires_grad_(False)
        path = directory / "initial.json"
        if path.exists():
            result = read(path)
            if result["a2_sha256"] != data["a2_sha256"]:
                raise ValueError("initial drift binding mismatch")
            return result
        record = vf.EditorRecord.from_dict(data["event"]["edit_record"])
        native = next(r for r in data["rows"] if r["role"] == "native")
        fixtures = [native, next(r for r in data["rows"] if r["role"] == "fit" and r["label"] == "positive" and r["question"] != native["question"])]
        fixtures += [next(r for r in data["rows"] if r["role"] == "fit" and r["fact_relation"] == rel) for rel in RELATIONS.values()]
        lr = LowRankExpert(cp, vf.derive_seed(record.record_id, SEED))
        transfer = []
        for row in fixtures:
            before, after = generated(runtime, row, cp), generated(runtime, row, lr)
            transfer.append(dict(logical_id=row["logical_id"], parity=before["raw_token_ids"] == after["raw_token_ids"] and before["raw_answer"] == after["raw_answer"]))
        del lr
        teacher = TeacherCache(runtime, run, task["event_index"], config)
        hook = vf.MedTraceLayerHook(runtime.get_module(vf.LAYER), cp)
        hook.attach()
        initial, rows = {}, data["rows"]
        try:
            with torch.no_grad():
                for g, rel in RELATIONS.items():
                    pool = [r for r in rows if r["role"] == "fit" and r["fact_relation"] == rel]
                    values = {r["logical_id"]: float(teacher.kl(r, hook, training=True, chunk=16)) for r in pool}
                    initial[g] = group_mean(values, pool)
        finally:
            hook.detach()
        result = dict(a2_sha256=data["a2_sha256"], initial=initial, transfer=transfer,
                      l16_eligible=data["transfer_cpu"] and all(r["parity"] for r in transfer), teacher_forwards=teacher.forward_count)
        vf.atomic_json(path, result)
        return result


def endpoint(runtime, run, task, data, expert, teacher, chunk):
    outputs = {}
    base_root = run / f"private/base_generation/e{task['event_index']:02d}"
    base_root.mkdir(parents=True, exist_ok=True)
    hook = vf.MedTraceLayerHook(runtime.get_module(vf.LAYER), expert)
    hook.attach()
    try:
        for row in data["rows"]:
            base_path = base_root / (vf.sha256_json(row)+".json")
            hook.clear_request_routing()
            with base_path.with_suffix(".lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not base_path.exists():
                    vf.atomic_json(base_path, vf.scope_generate(runtime, row, None))
                base = read(base_path)
            forced = vf.scope_generate(runtime, row, hook)
            on = data["frozen_gate"][row["logical_id"]]
            # Actual fixed-gate and disabled requests; not relabelled forced outputs.
            fixed = vf.scope_generate(runtime, row, hook if on else None)
            hook.clear_request_routing()
            disabled = vf.scope_generate(runtime, row, None)
            if disabled["raw_token_ids"] != base["raw_token_ids"] or disabled["raw_answer"] != base["raw_answer"]:
                raise RuntimeError("DISABLED failed Base parity")
            expected = forced if on else base
            if fixed["raw_token_ids"] != expected["raw_token_ids"] or fixed["raw_answer"] != expected["raw_answer"]:
                raise RuntimeError("fixed-router replay differs")
            kl = None
            if row["label"] == "negative":
                with torch.no_grad():
                    kl = float(teacher.kl(row, hook, training=False, chunk=chunk))
            outputs[row["logical_id"]] = dict(row=row, base=base, forced=forced, fixed=fixed,
                disabled_parity=True, fixed_on=on, kl=kl,
                token_parity=forced["raw_token_ids"] == base["raw_token_ids"])
    finally:
        hook.detach()
    return outputs


def train_task(runtime, args, task, chunk=16):
    run = args.run_root
    config = read(run / "private/CAMPAIGN_CONFIG.json")
    data = read(run / f"private/edits/e{task['event_index']:02d}.json")
    if vf.sha256_file(Path(data["a2"])) != data["a2_sha256"]:
        raise ValueError("A2 changed since preparation")
    start = time.monotonic()
    initial = shared_initial(runtime, run, task, data, config)
    if task["parameterization"] == "L16" and not initial["l16_eligible"]:
        raise RuntimeError("L16 step0 transfer mismatch; P4 remains eligible")
    record = vf.EditorRecord.from_dict(data["event"]["edit_record"])
    cp, _ = load_cp(data["a2"], runtime.device)
    expert = cp if task["parameterization"] == "P4" else LowRankExpert(cp, vf.derive_seed(record.record_id, SEED))
    if expert is not cp:
        del cp
    expert.requires_grad_(True)
    optimizer = optimizer_for(expert, runtime.model)
    protect = Protection(initial["initial"], task["condition"])
    rows = data["rows"]
    pools = {g: [r for r in rows if r["role"] == "fit" and r["fact_relation"] == rel] for g,rel in RELATIONS.items()}
    schedules = {g: balanced_schedule(fit_groups(pool, g), STEPS, vf.derive_seed(record.record_id, SEED)+j)
                 for j,(g,pool) in enumerate(pools.items())}
    positive_batches = [runtime.build_edit_batch(record)] + [runtime.build_edit_batch(replace(record, question=p["question"])) for p in data["fit_paraphrases"]]
    teacher = TeacherCache(runtime, run, task["event_index"], config)
    hook = vf.MedTraceLayerHook(runtime.get_module(vf.LAYER), expert)
    hook.attach()
    out = run / "private/tasks" / task["task_id"] / f"attempt_chunk{chunk:02d}"
    out.mkdir(parents=True, exist_ok=True)
    training_start = time.monotonic()
    curve, diagnostics, forwards, backwards = [], [], 0, 0
    try:
        for step in range(STEPS+1):
            if time.time()-read(run / "private/CAMPAIGN_START.json")["epoch"] >= config["train_seconds"]:
                raise TimeoutError("20h training/generation boundary reached")
            if step:
                optimizer.zero_grad(set_to_none=True)
                positive_loss, observed = 0., {}
                for batch in (positive_batches[0], positive_batches[1+(step-1)%(len(positive_batches)-1)]):
                    hook.set_teacher_routing(batch.labels)
                    if not torch.equal(hook.token_mask, predictor_mask(batch.labels, batch.attention_mask)):
                        raise ValueError("positive answer mask mismatch")
                    loss = .5*runtime.compute_loss(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite positive loss")
                    loss.backward(); forwards += 1; backwards += 1
                    positive_loss += float(loss.detach())
                    del loss
                dual_before = dict(protect.dual)
                if task["condition"] != CONDITIONS[0]:
                    for g in ("H", "U"):
                        loss = teacher.kl(schedules[g][step-1], hook, training=True, chunk=chunk)
                        observed[g] = float(loss.detach())
                        (protect.coefficient(g)*loss).backward()
                        forwards += 1; backwards += 1
                        del loss
                norm = float(torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.))
                if not math.isfinite(norm):
                    raise FloatingPointError("nonfinite expert gradient")
                optimizer.step()
                expert.normalize_factors_(verify_dense=False)
                if any(not torch.isfinite(p).all() for p in expert.parameters()):
                    raise FloatingPointError("nonfinite expert parameters")
                if observed:
                    protect.update(observed, role="fit")
                curve.append(dict(step=step, positive_loss=positive_loss, sampled_fit_kl=observed,
                    dual_before=dual_before, dual_after=dict(protect.dual), ema=dict(protect.ema), grad_norm=norm,
                    constraint_constant=sum(dual_before[g]*protect.epsilon[g]/protect.scale[g] for g in ("H","U")) if task["condition"] == CONDITIONS[-1] else 0.,
                    elapsed=time.monotonic()-training_start))
            if step in (0,80,160,320):
                with torch.no_grad():
                    full = {g: group_mean({r["logical_id"]: float(teacher.kl(r, hook, training=True, chunk=chunk)) for r in pool}, pool)
                            for g,pool in pools.items()}
                    forwards += sum(len(p) for p in pools.values())
                diagnostics.append(dict(step=step, full_fit_kl=full,
                    constraint_residual={g: full[g]-protect.epsilon[g] for g in full}, dual=dict(protect.dual),
                    saturation=dict(protect.saturation), scale=protect.scale, epsilon=protect.epsilon))
                save(out / f"step{step:04d}.pt", dict(expert=expert.state_dict(), task=task, step=step, optimizer=optimizer.state_dict(), protection=vars(protect)))
                vf.atomic_json(out / "training_private.json", dict(curve=curve, diagnostics=diagnostics, forward_count=forwards, backward_count=backwards))
    finally:
        hook.detach()
    training_seconds = time.monotonic()-training_start
    expert.requires_grad_(False)
    del optimizer, positive_batches
    reference_path = run / f"private/initial/e{task['event_index']:02d}/reference_private.json"
    with reference_path.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not reference_path.exists():
            reference, _ = load_cp(data["a2"], runtime.device)
            reference.requires_grad_(False)
            vf.atomic_json(reference_path, dict(status="RAW_READY", task=dict(event_index=task["event_index"],condition="A2"),
                outputs=endpoint(runtime, run, task, data, reference, teacher, chunk), a2_sha256=data["a2_sha256"]))
            del reference
    outputs = endpoint(runtime, run, task, data, expert, teacher, chunk)
    if vf.sha256_json(data["frozen_gate"]) != data["frozen_gate_sha256"]:
        raise ValueError("fixed router decisions changed")
    guard = runtime.base_guard.verify()
    if not guard["unchanged"]:
        raise RuntimeError("backbone changed")
    result = dict(status="RAW_READY", task=task, step=320, outputs=outputs, diagnostics=diagnostics,
        parameters=sum(p.numel() for p in expert.parameters()), forward_count=forwards, backward_count=backwards,
        teacher_forward_count=teacher.forward_count, training_seconds=training_seconds, elapsed_seconds=time.monotonic()-start,
        frozen_gate_sha256=data["frozen_gate_sha256"], base_guard=guard, a2_sha256=data["a2_sha256"], scientific_gain="NOT_EVALUATED")
    vf.atomic_json(out.parent / "result_private.json", result)
    return result


def worker(args):
    config = read(args.run_root / "private/CAMPAIGN_CONFIG.json")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_check(gpu)
    if os.environ.get("M3BENCH_FORMAL_EXPECTED_GPU_UUID") != GPUS[gpu]:
        raise ValueError("worker GPU authorization mismatch")
    runtime = vf.load_real_runtime(argparse.Namespace(cpu_gate=Path(config["runtime"]["cpu_gate"])))
    # Start is after the first real model load, not while waiting for a device.
    with (args.run_root / "private/CAMPAIGN_START.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start_path = args.run_root / "private/CAMPAIGN_START.json"
        if not start_path.exists():
            vf.atomic_json(start_path, dict(epoch=time.time(), gpu=gpu, code_commit=config["code_commit"],
                config_sha256=vf.sha256_file(args.run_root / "private/CAMPAIGN_CONFIG.json"), gpu_uuids=GPUS))
    queue = vf.TaskQueue(args.run_root / "private/TASK_QUEUE.json", args.run_root)
    with vf.Telemetry(args.run_root / "private/GPU_TELEMETRY.jsonl", gpu, f"gpu{gpu}") as telemetry:
        while time.time()-read(start_path)["epoch"] < config["train_seconds"] and not (args.run_root / "STOP").exists():
            task = queue.claim(f"gpu{gpu}")
            if task is None:
                break
            telemetry.task_id = task["task_id"]
            for attempt, chunk in enumerate((16,1)):
                try:
                    result = train_task(runtime, args, task, chunk=chunk)
                    queue.update(task["task_id"], "RAW_READY", elapsed_seconds=result["elapsed_seconds"], finished_epoch=time.time())
                    break
                except Exception as error:
                    oom = isinstance(error, torch.cuda.OutOfMemoryError)
                    vf.append_jsonl(args.run_root / "private/FAILURES.jsonl", dict(task_id=task["task_id"], attempt=attempt,
                        error=f"{type(error).__name__}: {error}", oom=oom, chunk=chunk, epoch=time.time()))
                    gc.collect(); torch.cuda.empty_cache()
                    if oom and attempt == 0:
                        continue
                    queue.update(task["task_id"], "FAILED", last_error=f"{type(error).__name__}: {error}")
                    if args.first_only:
                        raise
                    break
            gc.collect(); torch.cuda.empty_cache()
            if args.first_only:
                break


def summarize(args):
    public, run = args.public_dir, args.run_root
    tasks = read(run / "private/TASK_QUEUE.json")["tasks"]
    public.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        row = {k: task[k] for k in ("task_id","event_index","parameterization","condition","status")}
        row.update(steps=0, H_kl=None, U_kl=None, training_seconds=None, semantic_status="NOT_JUDGED")
        path = run / "private/tasks" / task["task_id"] / "result_private.json"
        if path.exists():
            result = read(path)
            row.update(steps=result["step"], H_kl=result["diagnostics"][-1]["full_fit_kl"]["H"],
                       U_kl=result["diagnostics"][-1]["full_fit_kl"]["U"], training_seconds=result["training_seconds"])
        rows.append(row)
    with (public / "SELECTIVE_WRITE_BY_EDIT.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    counts = {s: sum(t["status"] == s for t in tasks) for s in sorted({t['status'] for t in tasks})}
    vf.atomic_json(public / "RUN_COMPLETION.json", dict(status="NOT_RUN" if counts == {"PENDING":70} else "INCOMPLETE",
        counts=counts, expected=70, judge="NOT_RUN", evaluation="NOT_COMPLETE", scientific_gain="NOT_EVALUATED", novel_n=0))
    vf.atomic_text(public / "SELECTIVE_WRITE_REPORT.md", f"# Selective-write status\n\nQueue: {counts}. Main endpoint 320 appended steps. Unrun or failed cells remain missing, never zero scores. No semantic conclusion before fixed Judge and calibration-only lambda selection. Fit KL diagnostics are not clinical safety.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","worker","summarize"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path)
    parser.add_argument("--old-run", type=Path)
    parser.add_argument("--verifier-run", type=Path)
    parser.add_argument("--first-only", action="store_true")
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == "__main__":
    main()
