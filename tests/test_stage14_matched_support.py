from scripts.medtrace.stage14_sources import select_support,image_key,input_key


def test_matched_source_rules_and_explicit_rank():
    import torch
    from scripts.medtrace.stage11_worker import make_expert
    from methods.medtrace.core import AsymmetricCPExpert
    cp=AsymmetricCPExpert(16,16,4)
    for name,rank in [('J1_FREE_R4',4),('J2_FREE_R16',16),('C_FACT',4),('C_NO_H',4),('C_EXTRA_QA',4)]:
        assert make_expert(cp,name,17).rank==rank
    assert make_expert(cp,'J0_CP_R4',17) is cp
    try:make_expert(cp,'typo',17)
    except ValueError:pass
    else:raise AssertionError('unknown name must not silently use rank16')
    x=torch.randn(3,16)
    assert torch.allclose(cp.residual(x),make_expert(cp,'C_EXTRA_QA',17).residual(x))
    h=dict(dataset='SLAKE',image_path='/data/xmlab1/source.jpg',image_id='xmlab1',question='Does the picture contain liver?',reference='Yes',source_qid=1,logical_id='H',role='fit')
    n=dict(h,image_path='/data/xmlab2/source.jpg',image_id='xmlab2',reference='No',logical_id='native',role='native')
    g=dict(h,question='Does the picture contain kidney?',source_qid=2)
    fallback=dict(g,image_path=n['image_path'],image_id='xmlab2',source_qid=3)
    t=dict(data=dict(rows=[n,h]),native_id='native',h_ids=['H'])
    args=(set(),set(),set(),lambda r:len(r['reference']),17)
    a=select_support(t,[g,fallback],*args);assert a[1]['slots'][0]['tier']=='SAME_H_IMAGE' and len(a[0])==1
    assert select_support(t,[fallback,g],*args)==a
    b=select_support(t,[g,fallback],{image_key(h)},set(),set(),lambda r:1,17)
    assert b[1]['slots'][0]['tier']=='DIFFERENT_TRAIN_IMAGE'
    assert not select_support(t,[g],set(),set(),{input_key(g)},lambda r:1,17)[0]
    assert not select_support(t,[g],set(),{('SLAKE','2')},set(),lambda r:1,17)[0]
    assert not select_support(t,[h],*args)[0]


def test_pair_denominators_and_source_sensitivity():
    from scripts.medtrace.stage14 import pair_tables,leave_one_image_out
    freeze=dict(rows=[dict(cohort='NEW7',edit=i,status='SUPPORTED',tier='SAME_H_IMAGE') for i in (1,2)])
    rows=[];units=[]
    for method,values in [('C_FACT',[(1,1,1),(2,0,2)]),('C_EXTRA_QA',[(1,0,1),(2,0,2)])]:
        for edit,yes,n in values:
            rows.append(dict(cohort='NEW7',method=method,mode='FORCED_ON',edit=edit,role='evaluation',native_correct=int(edit==1),
                pairs=n,pair_correct=yes/n,both_correct=yes,native_only=int(edit==1)-yes,H_only=0,both_wrong=n-int(edit==1)))
            for j in range(n):units.append(dict(cohort='NEW7',method=method,mode='FORCED_ON',edit=edit,role='evaluation',image=('SLAKE',str(edit)),both_correct=int(j<yes)))
    tables,effects=pair_tables(rows,freeze)
    r=next(r for r in tables if r['cohort']=='NEW7' and r['subset']=='COMMON_SUPPORTED' and r['mode']=='FORCED_ON' and r['role']=='evaluation' and r['method']=='C_FACT')
    assert r['native_correct']==r['pair_macro']==.5 and r['pair_micro']==1/3 and r['pair_edit_support']==2
    lo=leave_one_image_out(units,freeze)
    assert {r['delta'] for r in lo if r['subset']=='COMMON_SUPPORTED' and r['mode']=='FORCED_ON' and r['control']=='C_EXTRA_QA'}=={0.,1.}
