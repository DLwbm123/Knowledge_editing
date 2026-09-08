import pytest
from scripts.medtrace.stage4_scope import accepted, calibrate
from methods.medtrace.selective_write import Protection


def test_weak_normalized_kl_and_rejection_only_calibration():
    weak=Protection({'H':.02,'U':.004},'W1_KL_0.01')
    strong=Protection({'H':.02,'U':.004},'W1_KL_0.1')
    assert weak.coefficient('H') == pytest.approx(.1*strong.coefficient('H'))
    def row(role,label,d,group=None):
        return dict(edit=1,role=role,label=label,family='f',negative_group=group,
                    strict_role='STRICT_BASE' if group else 'EDIT_TARGET',
                    route=dict(activated=True,nearest_distance=d,radius=1.,logical_edit_id='one'))
    rows=[row('native','positive',0),row('calibration','positive',.1),
          row('calibration','negative',.8,'H'),row('calibration','negative',.9,'U')]
    lock=calibrate(rows)
    assert lock['status']=='CALIBRATED' and lock['rc_objective']==0
    assert accepted(rows[0]['route'],lock['kappa']) and not accepted(rows[-1]['route'],lock['kappa'])
    assert not accepted(dict(rows[0]['route'],activated=False),0.)
    assert calibrate(rows[:2])['status']=='CALIBRATION_UNSUPPORTED'
    with pytest.raises(ValueError): calibrate([dict(rows[0],role='evaluation')])
    assert not accepted(dict(rows[0]['route'],radius=0),0.)
