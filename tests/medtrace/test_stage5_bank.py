from types import SimpleNamespace

from test_stage3_bank import setup_bank, TinyExpert
from scripts.medtrace import stage5_bank as s5


def test_final_bank_uses_selected_writer_and_keeps_missing_calibration_explicit(tmp_path, monkeypatch):
    f = setup_bank(tmp_path)
    episodes = f.episodes[:2]
    ids = {e['record_id'] for e in episodes}
    artifacts = {k:v for k,v in f.artifacts.items() if k[1] in ids}
    monkeypatch.setattr(s5.bank, 'METHODS', dict(s5.CONDITIONS))
    monkeypatch.setattr(s5, 'available', lambda *_: episodes)
    monkeypatch.setattr(s5.bank, 'load_artifacts', lambda *_: (artifacts, f.entries[:2]))
    monkeypatch.setattr(s5.bank, 'input_batch', lambda runtime, row: SimpleNamespace(eqkey=row['eqkey']))
    monkeypatch.setattr(s5.vf, 'AsymmetricCPExpert', TinyExpert)
    monkeypatch.setattr(s5.vf, 'scope_generate', f.scope)
    monkeypatch.setattr(s5.sw, 'generated', f.cp)
    monkeypatch.setattr(s5.sw, 'active_elapsed', lambda _: 0.)
    s5.evaluate(f.runtime, f.run, 'W0')
    result = s5.read(f.run/'private/final_bank/W0/result_private.json')
    assert result['status'] == 'RAW_READY'
    assert result['threshold']['status'] == 'CALIBRATION_UNSUPPORTED'
    assert result['threshold']['kappa'] == 0
    cross = next(r for r in result['outputs'] if r['row']['logical_id'] == 'cross0' and r['route_mode'] == 'RC')
    assert cross['selected_expert'] == 'e1'
    assert cross['actual']['raw_answer'] == 'S0:e1:cross0'
    assert cross['own_forced']['raw_answer'] == 'S0:e0:cross0'
    assert result['replays']['ON']['status'] == result['replays']['OFF']['status'] == 'PASSED'
    assert f.modules['up'] is f.base
