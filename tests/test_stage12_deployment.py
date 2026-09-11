from scripts.medtrace.stage12 import deploy,MODES


def test_mixed_report_fields(tmp_path):
    import csv
    from scripts.medtrace.stage12 import mixed_csv
    path=tmp_path/'report.csv'
    mixed_csv(path,[{'row_kind':'METRIC','semantic':1},{'row_kind':'PAIRED_EFFECT','delta':.5}])
    with path.open() as handle:rows=list(csv.DictReader(handle))
    assert rows[0]['semantic']=='1' and rows[0]['delta']==''
    assert rows[1]['delta']=='0.5' and rows[1]['semantic']==''


def test_fixed_rejection_derivation_preserves_source():
    entry=dict(item=dict(route=dict(activated=True,nearest_distance=.5,radius=1.),forced={'answer':'writer'},base={'answer':'base'}))
    r0=deploy(entry,MODES[0],.7696741135364367)
    rc=deploy(entry,MODES[1],.7696741135364367)
    assert r0['item']['fixed']==entry['item']['forced'] and r0['item']['fixed_on']
    assert rc['item']['fixed']==entry['item']['base'] and not rc['item']['fixed_on']
    assert 'fixed' not in entry['item']
    positive=dict(entry,item=dict(entry['item'],route=dict(activated=True,nearest_distance=0.,radius=1.)))
    assert deploy(positive,MODES[1],.7696741135364367)['item']['fixed_on']
