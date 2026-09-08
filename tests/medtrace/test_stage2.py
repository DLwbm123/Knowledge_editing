"""Small CPU check for routing boundaries, frozen tasks and Judge reuse identity."""
import tempfile
from unittest.mock import Mock, patch
from types import SimpleNamespace
from pathlib import Path
from dataclasses import asdict

from scripts.medtrace.run_stage2 import NEW_METHODS, queue_task, query_record, same_output, base_for, vf, sw
from scripts.medtrace.finalize_stage2 import execution_identity, aggregate, METRICS
from scripts.medtrace.closeout_stage2_local import check_public


def test_stage2_contract():
    record = vf.EditorRecord.from_dict(dict(record_id="edit", dataset="SLAKE", question="native",
        gold_answer="private target", official_rephrase="fit", image_path="/image.png", relative_image_path="image.png",
        formal_sequence_position=1, question_type="OPEN"))
    row = dict(logical_id="query", question="input", image_path="/other.png", reference="secret", task="T2G")
    query = query_record(record, row)
    assert query.target == "" and query.record_id == "query" and query.question == "input"
    assert "secret" not in str(asdict(query)) and query.official_rephrase == ""
    assert [p for p, _ in NEW_METHODS] == ["P4", "P4", "P4", "L16"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "private").mkdir()
        first = queue_task(1, "BE", 1)
        second = queue_task(101, "INIT", 2, first["task_id"])
        path = root / "private/queue.json"
        vf.atomic_json(path, dict(tasks=[first, second]))
        queue = vf.TaskQueue(path, root)
        assert queue.claim("test")["task_id"] == first["task_id"]
        assert queue.claim("test") is None
        queue.update(first["task_id"], "RAW_READY")
        assert queue.claim("test")["task_id"] == second["task_id"]
        report = root / 'public.json'
        report.write_text('{"patient_id": "private"}')
        try:
            check_public(report)
        except ValueError:
            pass
        else:
            raise AssertionError('private record passed the aggregate publication gate')
    lock = dict(model={}, tokenizer={}, legacy_semantic_protocol_sha256="x", runtime=dict(vllm="1", gpu_uuid="a", physical_gpu="2"), generation=dict(max_model_len=2048, resolved_engine_config="dynamic"))
    other = dict(lock, runtime=dict(lock["runtime"], gpu_uuid="b", physical_gpu="3"))
    assert execution_identity(lock) == execution_identity(other)
    assert execution_identity(lock) != execution_identity(dict(lock, generation=dict(max_model_len=4096)))
    details = dict(track="OLD_DEV16", edit=1, method="BE", mode="BE_FORCED_ON", role="evaluation", stratum="H",
                   eqkey="x", source_group="a", **{k: None for k in METRICS})
    details["base_correct"] = 0.
    edits, macros = aggregate([details])
    assert edits[0]["base_correct_damage"] is None and macros[0]["base_correct_edits"] == 0
    assert same_output(dict(raw_answer="a", raw_token_ids=[1]), dict(raw_answer="a", raw_token_ids=[1]))


def test_system_mismatch_retains_forced_endpoint():
    row = dict(logical_id='query', label='positive')
    base, forced, mismatch = [dict(raw_answer=str(i), raw_token_ids=[i]) for i in range(3)]
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        vf.atomic_json(run / 'private/base_generation/e01' / (vf.sha256_json(row)+'.json'), base)
        with patch.object(vf, 'MedTraceLayerHook', return_value=Mock()), patch.object(vf, 'scope_generate', side_effect=[forced, mismatch, base]):
            output = sw.endpoint(SimpleNamespace(get_module=lambda _: None), run,
                dict(event_index=1, task_id='S2_fixture'), dict(rows=[row], frozen_gate={'query': True}), None, None, 16)
        assert output['query']['forced'] == forced and not output['query']['system_replay_valid']


def test_paraphrase_lineage_is_not_a_native_base_cache_key():
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        vf.atomic_json(run / 'private/CAMPAIGN_CONFIG.json', dict(stage1_run=str(run / 'past')))
        edit = dict(record_id='native', question='Original?', image_path='/image', gold_answer='a')
        row = dict(logical_id='fit-1', role='fit', label='positive', question='Rewritten?', image_path='/image',
                   reference='a', base_query_ids=['legacy-native'])
        runtime = SimpleNamespace(stage2_base={'legacy-native': dict(model_answer_raw='wrong reuse', raw_generated_token_ids=[1])})
        data = dict(event_index=101, track='NEW_CONFIRMATION', event=dict(event_id='e101', edit_record=edit, probes=[]))
        fresh = dict(raw_answer='actual rewritten Base answer', raw_token_ids=[2])
        with patch.object(vf, 'scope_generate', return_value=fresh) as generate:
            assert base_for(runtime, run, data, row) == fresh
            generate.assert_called_once()
