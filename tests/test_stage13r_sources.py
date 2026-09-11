from scripts.medtrace.stage13r_sources import canonical, partition, conflict, assemble


def test_source_identity_and_prospective_roles():
    assert canonical('SLAKE', 'imgs/xmlab32/source.jpg') == ('SLAKE', 'xmlab32')
    assert canonical('SLAKE', 'imgs/xmlab32/source_blur.jpg') == ('SLAKE', 'xmlab32')
    assert canonical('VQA-RAD', 'images/synpic1.jpg') == ('VQA-RAD', 'synpic1.jpg')
    groups = {'a', 'b', 'c', 'd'}
    roles = partition(groups, {g: 'body' for g in groups})
    assert roles == partition(groups, {g: 'body' for g in reversed(sorted(groups))})
    assert set(roles.values()) == {'adaptation', 'evaluation'}
    a = dict(source_group='a', question='Does the picture contain liver?', reference='Yes')
    b = dict(a, source_group='b', reference='No')
    assert conflict(a, b)
    assert not conflict(a, dict(b, reference='Unknown'))
    assert not conflict(a, dict(b, source_group='a'))
    # Source-package assembly has no Base correctness field or student output input.
    pool = [dict(a, image_id='a', source_qid=1), dict(b, image_id='b', source_qid=2),
            dict(b, source_group='c', image_id='c', source_qid=3),
            dict(b, source_group='d', image_id='d', source_qid=4, question='What modality is used to take this image?', reference='CT')]
    packages = assemble(pool, {'a':'adaptation', 'b':'adaptation', 'c':'evaluation', 'd':'adaptation'})
    assert len(packages) == 1
    assert packages[0]['H_evaluation'][0]['source_group'] == 'c'
