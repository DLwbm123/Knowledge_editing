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


def test_existing_worker_input_boundary_and_rank():
    import torch
    from scripts.medtrace.stage13r import strict_training
    from scripts.medtrace.stage11_worker import make_expert
    from methods.medtrace.core import AsymmetricCPExpert
    data = dict(event=dict(probes=[]), rows=[dict(role='native'), dict(role='fit', negative_group='H'), dict(role='fit', negative_group='U')])
    strict_training(data)
    data['rows'].append(dict(role='evaluation', reference='must not enter trainer'))
    try: strict_training(data)
    except AssertionError: pass
    else: raise AssertionError('evaluation answer accepted by training adapter')
    cp = AsymmetricCPExpert(16, 16, 4)
    a, b = (make_expert(cp, name, 17) for name in ('C_FACT','C_NO_H'))
    assert a.rank == b.rank == 4 and a is not b
    x = torch.randn(3, 16)
    assert torch.allclose(a.residual(x), b.residual(x))
