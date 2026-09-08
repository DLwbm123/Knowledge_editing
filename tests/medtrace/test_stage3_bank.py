"""Focused CPU checks for frozen bank selection, attribution and lifecycle."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.medtrace import run_stage3_bank as bank


def test_nearest_radius_tie_and_no_future_expert():
    router = bank.MemoryRouter("euclidean")
    router.add("first", torch.tensor([0.]), .1)
    router.add("second", torch.tensor([2.]), 99.)
    assert router.route(torch.tensor([1.])).logical_edit_id is None
    assert router.route(torch.tensor([1.])).nearest_logical_edit_id == "first"
    assert router.route(torch.tensor([.05])).logical_edit_id == "first"
    one = bank.MemoryRouter.from_state(dict(distance="euclidean", entries=router.export_state()["entries"][:1]))
    assert one.route(torch.tensor([2.])).logical_edit_id is None


def row(lid, key, role="native", label="positive", group="image", reference="answer"):
    return dict(logical_id=lid, eqkey=key, question=lid, image_path="/"+group,
        role=role, label=label, source_group=group, dataset="SLAKE", source_qid=lid,
        reference=reference, fact_relation="native" if role == "native" else "reviewed_same_fact_text_augmentation",
        negative_group="H" if label == "negative" else None,
        relation_evidence="explicit source exclusion" if label == "negative" else None)


def test_roles_are_metadata_only_and_unknown_is_not_strict_base():
    target = row("a", "x")
    h = row("b", "x", "evaluation", "negative")
    unknown = row("unknown", "z", "evaluation", "negative")
    distinct = row("distinct", "d", "evaluation", "negative", group="different")
    episodes = [dict(record_id="e1", rows=[target, h, unknown, distinct])]
    roles, _ = bank.freeze_roles(episodes)
    assert roles[("e1", "b")]["strict_role"] == "NOW_EDITED_CONTEXT"
    assert roles[("e1", "unknown")]["strict_role"] == "UNKNOWN"
    assert roles[("e1", "distinct")]["strict_role"] == "STRICT_BASE"
    episodes.append(dict(record_id="e2", rows=[dict(target, reference="incompatible")]))
    roles, _ = bank.freeze_roles(episodes)
    assert roles[("e1", "a")]["strict_role"] == "TARGET_CONFLICT"
    assert roles[("e1", "b")]["effective_reference"] is None
    episodes[1]["rows"][0]["reference"] = "answer"
    roles, _ = bank.freeze_roles(episodes)
    assert bank.selection_class("e2", "e1", roles[("e1", "a")]) == "KNOWN_EQUIVALENT_EXPERT"


def test_realized_input_binding_and_neutral_query():
    captured = []
    batch = SimpleNamespace(image_sha256="image", raw_input_ids=torch.tensor([[1, 2]]),
                            attention_mask=torch.ones(1, 2), key_token_index=1)
    runtime = SimpleNamespace(generation_config={"max_new_tokens": 7},
        build_question_batch=lambda record: (captured.append(record), batch)[1])
    r = row("q", bank.vf.sha256_json(dict(image="image", ids=[[1, 2]], attention=[[1., 1.]],
                                          boundary=1, generation=runtime.generation_config)))
    bank.input_batch(runtime, r)
    query = asdict(captured[0])
    assert query["record_id"] == "bank-query" and query["target"] == ""
    assert r["reference"] not in str(query)
    with pytest.raises(ValueError, match="binding"):
        bank.input_batch(runtime, dict(r, eqkey="wrong-input"))


class TinyExpert(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))


def setup_bank(tmp_path):
    stage2, run = tmp_path / "stage2", tmp_path / "stage3"
    runtime_paths = dict(runtime_lock=str(tmp_path / "runtime.json"), cpu_gate=str(tmp_path / "gate"))
    bank.vf.atomic_json(Path(runtime_paths["runtime_lock"]), dict(selected_runtime="runtime_b_official_native"))
    bank.vf.atomic_json(Path(runtime_paths["cpu_gate"])/"inputs/frozen/llava_med_generation_frozen.json", {})
    bank.vf.atomic_json(stage2/"private/CAMPAIGN_CONFIG.json", dict(runtime=runtime_paths))
    bank.vf.atomic_json(run/"private/CAMPAIGN_CONFIG.json", dict(stage2_run=str(stage2), runtime=runtime_paths))
    guard = dict(unchanged=True, after_sha256="base")
    base = torch.nn.Linear(1, 1, bias=False)
    modules = {"up": base}
    runtime = SimpleNamespace(device=torch.device("cpu"), generation_config={},
        target_lock={"balancedit": {"targets": ["up"]}},
        get_module=lambda key: modules[key], replace_module=lambda key, value: modules.__setitem__(key, value),
        base_guard=SimpleNamespace(verify=lambda: dict(guard)), run_root=run)
    episodes, artifacts, entries, inputs = [], {}, [], {}

    def answer(method, rid, r):
        text = f"{method}:{rid}:{r['question']}" if rid else f"base:{r['question']}"
        return dict(raw_answer=text, raw_token_ids=list(text.encode()))

    for i in range(16):
        rid = f"e{i}"
        rows = [row(f"native{i}", f"n{i}", group=f"g{i}"),
                row(f"cross{i}", f"p{i}", "evaluation", group=f"g{i}"),
                row(f"off{i}", f"h{i}", "evaluation", "negative", group="outside")]
        for r, key in zip(rows, (i*10., (i+1)*10., 1000.)):
            inputs[r["eqkey"]] = torch.tensor([key])
        episodes.append(dict(record_id=rid, event_index=101+i, rows=rows,
                             positive_review=dict(approved_equivalent=True)))
        entry = dict(logical_edit_id=rid, key=torch.tensor([i*10.]), radius=1., label=[])
        entries.append(entry)
        for m in bank.METHODS:
            path = stage2/f"{m}-{rid}.pt"
            if m == "B":
                payload = dict(wrapper=dict(storage_mode="resident_float32", edits={rid: {"weight": torch.ones(1, 1)}}))
            else:
                payload = dict(expert=dict(marker=torch.tensor([float(i*10+(m == 'S1'))])))
            torch.save(payload, path)
            outputs = {r["logical_id"]: dict(row=r, base=answer(None, None, r), forced=answer(m, rid, r)) for r in rows}
            artifacts[(m, rid)] = dict(path=path, identity=bank.file_identity(path), target="up" if m == "B" else bank.vf.LAYER,
                                      result=dict(outputs=outputs), seed=123+i, parameters=1)
    runtime.extract_layer_input_key = lambda batch, **kw: inputs[batch.eqkey]

    def scope(runtime, r, hook):
        wrapper = modules["up"]
        selected = wrapper.active_logical_id
        return answer("B", selected, r)

    def cp(runtime, r, expert):
        assert modules["up"].active_logical_id is None
        if expert is None:
            return answer(None, None, r)
        value = round(expert.marker.item())
        return answer("S1" if value % 10 else "S0", f"e{value//10}", r)

    return SimpleNamespace(run=run, runtime=runtime, episodes=episodes, artifacts=artifacts,
        entries=entries, modules=modules, base=base, cp=cp, scope=scope, inputs=inputs)


def run_fixture(fixture, prefix, scope=None):
    def artifacts(stage2, episodes, guard):
        ids = {e["record_id"] for e in episodes}
        return ({k:v for k,v in fixture.artifacts.items() if k[1] in ids}, fixture.entries[:len(episodes)])
    with patch.object(bank, "load_manifest", return_value=fixture.episodes), \
         patch.object(bank, "load_artifacts", side_effect=artifacts), \
         patch.object(bank, "input_batch", side_effect=lambda runtime, row: SimpleNamespace(eqkey=row["eqkey"])), \
         patch.object(bank.vf, "AsymmetricCPExpert", TinyExpert), \
         patch.object(bank.vf, "scope_generate", side_effect=scope or fixture.scope), \
         patch.object(bank.sw, "generated", side_effect=fixture.cp):
        return bank.run_bank(fixture.runtime, SimpleNamespace(run_root=fixture.run),
                             dict(kind="BANK", prefix=prefix, task_id=f"S3_B_k{prefix:02d}"))


def test_group_executes_selected_writer_and_replays_natural_branches(tmp_path):
    f = setup_bank(tmp_path)
    result = run_fixture(f, 4)
    assert result["status"] == "RAW_READY" and len(result["outputs"]) == result["expected_items"] == 36
    assert {x["method"] for x in result["outputs"]} == {"S0", "S1", "B"}
    for m in bank.METHODS:
        selected = next(v for v in result["outputs"] if v["method"] == m and v["row"]["logical_id"] == "cross0")
        assert selected["selected_expert"] == "e1"
        assert selected["actual"]["raw_answer"] == f"{m}:e1:cross0"
        assert selected["own_forced"]["raw_answer"] == f"{m}:e0:cross0"
        assert not selected["provenance"]["actual"]["derived"]
        assert all(v["status"] == "PASSED" for v in result["replays"][m].values())
        assert result["costs"][m]["optimizer_steps"] == 0
    assert f.modules["up"] is f.base
    resumed = run_fixture(f, 4)
    assert all(v["provenance"]["actual"]["derived"] for v in resumed["outputs"])


def test_single_prefix_and_all_off_are_valid_without_manufactured_replay(tmp_path):
    f = setup_bank(tmp_path)
    result = run_fixture(f, 1)
    assert len(result["outputs"]) == 9
    assert all(result["replays"][m]["WRONG_SELECTION"]["status"] == "NOT_OBSERVED" for m in bank.METHODS)
    f.inputs = {k: torch.tensor([1000.]) for k in f.inputs}
    f.runtime.extract_layer_input_key = lambda batch, **kw: f.inputs[batch.eqkey]
    result = run_fixture(f, 1)
    assert not any(v["on"] for v in result["outputs"])
    assert all(result["replays"][m]["ON"]["status"] == "NOT_OBSERVED" for m in bank.METHODS)


def test_exception_restores_base_and_preserves_partial_group(tmp_path):
    f = setup_bank(tmp_path)
    def fail(runtime, row, hook):
        assert f.modules["up"].active_logical_id == "e0"
        raise RuntimeError("injected generation failure")
    with pytest.raises(RuntimeError, match="injected"):
        run_fixture(f, 1, scope=fail)
    assert f.modules["up"] is f.base
    result = bank.read(f.run/"private/bank/prefix1/result_private.json")
    assert result["status"] == "PARTIAL" and result["base_guard"]["unchanged"]
    assert len(result["outputs"]) < result["expected_items"]


def test_legacy_cache_does_not_accept_changed_input(tmp_path):
    f = setup_bank(tmp_path)
    artifact = f.artifacts[("S0", "e0")]
    artifact["result"]["outputs"] = deepcopy(artifact["result"]["outputs"])
    artifact["result"]["outputs"]["native0"]["row"]["eqkey"] = "other-image-or-prompt"
    with pytest.raises(ValueError, match="cached row"):
        bank.index_legacy(f.artifacts, f.episodes)
