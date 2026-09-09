"""Minimal fixed-transfer contract check; no data or GPU required."""
import json
from pathlib import Path
import tempfile
import hashlib
from scripts.medtrace.stage6 import transfer, threshold, FIXED


def test_fixed_transfer():
    base={'raw_answer':'base'};writer={'raw_answer':'writer'}
    entry=dict(track='B',item=dict(route=dict(activated=True,radius=1.,nearest_distance=.4),
        base=base,actual=writer,on=True,selected_expert='real-selected'))
    kept=transfer(entry,.5);off=transfer(entry,.7)
    assert kept['item']['actual']==writer and kept['item']['selected_expert']=='real-selected'
    assert off['item']['actual']==base and off['item']['selected_expert'] is None
    assert kept['route_mode']==FIXED and entry['item']['on']
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);(root/'private').mkdir();source=root/'old16.json'
        source.write_text('{"kappa": 0.7}')
        (root/'private/TRANSFER_LOCK.json').write_text(json.dumps(dict(source_path=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),kappa=.7)))
        assert threshold(root)==.7  # No H calibration or calibrate() is needed.
        source.write_text('{"kappa": 0.8}')
        try:threshold(root)
        except AssertionError:pass
        else:raise AssertionError('changed immutable source must fail')
