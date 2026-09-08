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
