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


def test_stage4_paired_support_and_bootstrap():
    from scripts.medtrace.finalize_stage4 import paired,bootstrap,OLD
    rows=[]
    for edit in (1,2):
        for method,value in [('W0',0.),('W01',1.)]:
            rows.append(dict(cohort=OLD,track='A',prefix=0,role='formal_development',stratum='T2G',
                strict_role='EDIT_TARGET',is_diagnostic=False,edit=edit,eqkey=str(edit),source_group='shared',
                source_image='shared',method=method,mode='FORCED_ON',semantic=value,v4_primary=value,
                base_correct_damage=None,base_wrong_became_correct=value,base_wrong_changed=value,fpr=None))
    row=next(r for r in paired(rows) if r['candidate']=='W01' and r['control']=='W0' and
             r['mode']=='FORCED_ON' and r['metric']=='v4_primary')
    assert row['delta']==1 and row['ci_low']==row['ci_high']==1
    assert row['paired_inputs']==row['paired_edits']==2 and row['source_images']==1
    assert row['image_cluster_ci_low'] is None
    assert bootstrap([0.,1.])==bootstrap([0.,1.])


def test_bank_waits_for_writer_branch_and_failed_writer_does_not_block_scope(tmp_path):
    from scripts.medtrace.run_stage4 import Queue,vf
    path=tmp_path/'private/TASK_QUEUE.json'
    vf.atomic_json(path,dict(tasks=[dict(task_id='a',kind='WRITER',priority=0,status='RUNNING',attempts=1),
                                  dict(task_id='b',kind='BANK',priority=1,status='PENDING',attempts=0)]))
    q=Queue(path,tmp_path)
    assert not q.ready() and q.claim('test') is None
    q.update('a','FAILED')
    assert q.claim('test')['task_id']=='b'
