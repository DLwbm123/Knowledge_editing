import pytest

from scripts.medtrace.run_stage3 import method_task, preflight, read, vf, sw
from methods.medtrace.selective_write import Protection


def test_stage3_frozen_group_and_start_contract(tmp_path):
    group = dict(task_id='S3_A_e0001_GROUP', event_index=1)
    s0, s1 = method_task(group, 'S0'), method_task(group, 'S1')
    assert s0['seed'] == s1['seed'] == 20260910
    assert s0['parameterization'] == s1['parameterization'] == 'P4'
    assert s0['task_id'] != s1['task_id']
    assert (s0['condition'], s1['condition']) == ('W0_TASK_ONLY', 'W1_KL_0.1')
    config = dict(kind='MEDTRACE_STAGE3', gpu_uuids=sw.GPUS, code_commit='source')
    vf.atomic_json(tmp_path/'private/CAMPAIGN_CONFIG.json', config)
    vf.atomic_json(tmp_path/'private/TASK_QUEUE.json', dict(tasks=[]))
    with pytest.raises(FileNotFoundError):
        preflight(tmp_path)
    vf.atomic_json(tmp_path/'private/CAMPAIGN_START.json', dict(epoch=123, code_commit='source'))
    assert preflight(tmp_path) == config
    assert read(tmp_path/'private/CAMPAIGN_START.json')['epoch'] == 123


def test_missing_protection_support_is_only_allowed_for_stage3_task_only():
    assert Protection({}, 'W0_TASK_ONLY', allow_missing_task_only=True).scale == {}
    with pytest.raises(ValueError):
        Protection({}, 'W0_TASK_ONLY')
    with pytest.raises(ValueError):
        Protection({}, 'W1_KL_0.1', allow_missing_task_only=True)


def test_resource_expansion_preserves_budget_and_process_identity(tmp_path, monkeypatch):
    from scripts.medtrace import coordinate_stage3 as coordinator
    from scripts.medtrace.run_stage3 import resource_amendment, ORIGINAL_GPUS
    monkeypatch.setattr(sw, 'GPUS', dict(ORIGINAL_GPUS))
    amendment = dict(authorization='USER_ADD_GPU0_GPU1', epoch=3600,
                     gpu_uuids=dict(ORIGINAL_GPUS, **{'0':'GPU-zero', '1':'GPU-one'}))
    vf.atomic_json(tmp_path/'private/GPU_EXPANSION.json', amendment)
    assert resource_amendment(tmp_path) == amendment
    assert set(sw.GPUS) == {'0','1','2','3'}
    assert coordinator.gpu_seconds_bound(0, 7200, amendment) == 6*3600
    assert coordinator.gpu_seconds_bound(0, 10800, amendment, 7200) == 7*3600
    monkeypatch.setattr(coordinator, 'process_identity', lambda pid: 'start')
    worker = coordinator.AdoptedWorker(123, 'start')
    assert worker.poll() is None
    monkeypatch.setattr(coordinator, 'process_identity', lambda pid: 'different-start')
    assert worker.poll() == 0
    amendment['gpu_uuids']['2'] = 'GPU-wrong'
    vf.atomic_json(tmp_path/'private/GPU_EXPANSION.json', amendment)
    with pytest.raises(ValueError):
        resource_amendment(tmp_path)
