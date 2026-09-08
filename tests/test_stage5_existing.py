import pytest
from scripts.medtrace.stage5_existing import reject
from scripts.medtrace.stage4_scope import calibrate


def test_rejection_keeps_expert_or_exact_base():
    route = dict(activated=True, radius=10., nearest_distance=4., logical_edit_id='e1')
    item = dict(route=route, actual={'raw_answer': 'edited'}, base={'raw_answer': 'base'})
    on = reject(item, .5, True)
    off = reject(item, .7, True)
    assert on['actual'] is item['actual'] and on['selected_expert'] == 'e1'
    assert off['actual'] is item['base'] and off['selected_expert'] is None
    assert on['route'] == off['route'] == route
    with pytest.raises(ValueError, match='evaluation'):
        calibrate([dict(role='evaluation')])
