from scripts.medtrace.stage13_source_audit import source_support


def test_base_only_exposure_is_not_student_development():
    rows=[dict(dataset='SLAKE',image_path=f'/data/{image}/source.jpg',base_correct=False) for image in ('old','new')]
    groups,used,fresh=source_support(rows,{'Stage2':[{'rows':[rows[0]]}]})
    assert len(groups)==2 and fresh==[rows[1]] and used['SLAKE','old']=={'Stage2'}
    assert source_support(rows,{'Stage2':[{'rows':rows}]})[2]==[]
