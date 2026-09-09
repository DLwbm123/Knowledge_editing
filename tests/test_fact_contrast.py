import torch
from methods.medtrace.fact_contrast import answer_score,pair_half,legal_pair


def test_content_mask_gradients_and_source_boundary():
    logits=torch.zeros(1,6,8,requires_grad=True);labels=torch.tensor([[-100,-100,3,4,2,-100]])
    score=answer_score(logits,labels,torch.ones_like(labels),2,1,0)
    score.backward();assert logits.grad[0,0].abs().sum()==0 and logits.grad[0,3:].abs().sum()==0
    assert logits.grad[0,1,3]>0 and logits.grad[0,2,4]>0
    good=torch.tensor(0.,requires_grad=True);wrong=torch.tensor(0.,requires_grad=True)
    pair_half(good,wrong).backward();assert good.grad<0 and wrong.grad>0
    native=dict(role='native',question='Does the picture contain spleen?',reference='No',source_file='source',image_path='positive')
    h=dict(role='fit',negative_group='H',question=native['question'],reference='Yes',source_file='source',image_path='negative',conflict_verified=True,relation_evidence='source')
    assert legal_pair(native,h)[0]
    assert not legal_pair(native,dict(h,question='Does the picture not contain spleen?'))[0]
    assert not legal_pair(native,dict(h,reference='Maybe'))[0]
    try:legal_pair(native,dict(h,role='evaluation'))
    except ValueError:pass
    else:raise AssertionError('evaluation gold must not enter training')
