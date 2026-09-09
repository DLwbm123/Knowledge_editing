"""Fixed Stage9 answer-content ranking, distinct from EOS-inclusive CE."""
import torch
import torch.nn.functional as F


def answer_score(logits,labels,attention,eos,bos,pad):
    target=labels[:,1:];valid=target.ne(-100)
    if attention is not None:valid=valid & attention[:,1:].bool() & attention[:,:-1].bool()
    for token in (eos,bos,pad):
        if token is not None:valid=valid & target.ne(token)
    if not valid.any():raise ValueError('no answer-content tokens')
    logp=logits[:,:-1][valid].float().log_softmax(-1)
    return logp.gather(1,target[valid].unsqueeze(1)).mean()


def pair_half(correct,swapped):
    return .5*F.relu(.5-correct+swapped)


def legal_pair(native,row):
    if native['role']!='native' or row['role']!='fit' or row.get('negative_group')!='H':raise ValueError('non-fit H supervision')
    if row['question']!=native['question']:return False,'NO_EXACT_QUESTION_OR_APPROVED_EQUIVALENCE'
    if not row.get('source_file') or not native.get('source_file'):return False,'MISSING_DIRECT_SOURCE_BINDING'
    if not row.get('conflict_verified') or not row.get('relation_evidence'):return False,'MISSING_RELATION_EVIDENCE'
    if row['image_path']==native['image_path']:return False,'SAME_IMAGE'
    if any(row.get(k) for k in ('human_review_required','clinical_review_pending','requires_human_review')):return False,'HUMAN_REVIEW_PENDING'
    q=native['question'];a=native['reference'].strip().lower();b=row['reference'].strip().lower()
    # Only existing source-supported single-valued slots or the same yes/no proposition.
    if q in ('Does the picture contain spleen?','Does the picture contain kidney?','Does the picture contain liver?'):
        legal={a,b}=={'yes','no'};reason='SAME_PROPOSITION_YES_NO'
    elif q=='Which part of the body does this image belong to?':
        legal=a!=b and {a,b}<={'chest','abdomen','head'};reason='SOURCE_SINGLE_REGION_SLOT'
    elif q=='What is the mr weighting in this image?':
        legal={a,b}=={'t1','t2'};reason='SOURCE_SINGLE_WEIGHTING_SLOT'
    elif q=='What is the largest organ in the picture?':
        legal=a!=b and {a,b}<={'liver','brain'};reason='SOURCE_UNIQUE_LARGEST_SLOT'
    else:return False,'NO_EXPLICIT_MUTUAL_EXCLUSION_PROOF'
    return legal,reason if legal else 'VALUES_NOT_COVERED_BY_RELATION'
