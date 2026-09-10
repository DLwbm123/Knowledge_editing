"""Official-mask, pre-vision-encoder perturbations with request-local cleanup."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from .fact_contrast import answer_score

OFFICIAL_MAPPING_URL='https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/mask.txt'
CLASSES={'spleen':[230],'kidney':[135,155],'liver':[70]}


def anatomy(row):
    q=row['question']
    for name in CLASSES:
        if q==f'Does the picture contain {name}?':return name
    return None


def geometry(image_path,processor):
    """All supported source images are square: pad is identity; exact CLIP methods."""
    image=Image.open(image_path).convert('RGB')
    if image.width!=image.height:raise ValueError('unsupported non-square pad geometry')
    pixels=processor.preprocess(image,return_tensors='pt')['pixel_values'][0].float()
    return image,pixels


def prepare_roi(row,processor,seed=20260911):
    name=anatomy(row)
    if not name:return None,'NONLOCAL_OR_COMPARATIVE'
    if row['reference'].strip().lower()!='yes':return None,'NO_AFFIRMATIVE_LOCAL_EVIDENCE'
    path=Path(row['image_path']);mp=path.parent/'mask.png'
    if not mp.exists():return None,'MISSING_OFFICIAL_MASK'
    image,pixels=geometry(path,processor);mask=Image.open(mp)
    if mask.size!=image.size:return None,'ANNOTATION_DIMENSION_MISMATCH'
    arr=np.array(mask)
    if arr.ndim==3:
        if not np.array_equal(arr[:,:,0],arr[:,:,1]) or not np.array_equal(arr[:,:,0],arr[:,:,2]):return None,'UNSUPPORTED_NON_GRAYSCALE_PALETTE'
        arr=arr[:,:,0]
    region=np.isin(arr,CLASSES[name])
    if not region.any():return None,'ORGAN_NOT_ANNOTATED_NOT_FACT_ABSENT'
    roiimage=Image.fromarray((region*255).astype('uint8')).convert('RGB')
    mapped=processor.preprocess(roiimage,return_tensors='pt',do_normalize=False,do_rescale=False,resample=Image.Resampling.NEAREST)['pixel_values'][0,0]>127
    assert mapped.shape==pixels.shape[-2:]
    rgb=pixels*torch.tensor(processor.image_std)[:,None,None]+torch.tensor(processor.image_mean)[:,None,None]
    tissue=rgb.mean(0)>.05
    ys,xs=torch.where(mapped);n=int(mapped.sum());height,width=mapped.shape
    choices=[]
    for dy in range(-height+16,height,16):
        for dx in range(-width+16,width,16):
            y=ys+dy;x=xs+dx
            if y.min()<0 or y.max()>=height or x.min()<0 or x.max()>=width:continue
            c=torch.zeros_like(mapped);c[y,x]=True
            if (c & mapped).any() or float(tissue[c].float().mean())<.8:continue
            choices.append((dy,dx,c))
    if not choices:return None,'NO_EQUAL_AREA_NONOVERLAPPING_TISSUE_CONTROL'
    random.Random(seed).shuffle(choices);chosen=choices[:5]
    # Same source images inspected for text overlays before freezing these controls.
    return dict(row=row,anatomy=name,region=mapped,controls=[c for _,_,c in chosen[:4]],
        diagnostic_control=chosen[4][2] if len(chosen)>4 else chosen[0][2],diagnostic_control_new=len(chosen)>4,
        offsets=[(dy,dx) for dy,dx,_ in chosen],pixels=pixels,source_size=image.size,annotation=str(mp),
        pixels_R=n,pixels_C=[int(c.sum()) for _,_,c in chosen[:4]],
        tissue_R=float(tissue[mapped].float().mean()),tissue_C=[float(tissue[c].float().mean()) for _,_,c in chosen[:4]],
        nonpadding_fraction=1.,geometry='square pad identity; actual CLIP resize/center-crop; ROI nearest-neighbor',
        granularity='official organ segmentation; not lesion annotation',text_overlay_check='bounded visual inspection of selected source images'),None


def suppressed(pixels,mask):
    out=pixels.clone();out[:,mask]=pixels.mean((-2,-1))[:,None];return out


@contextmanager
def pixel_view(adapter,row,pixels,expected=None):
    original=adapter.prepare_inputs
    def prepare(image_path,question,answer=None):
        if Path(image_path)!=Path(row['image_path']) or question!=row['question']:raise ValueError('auxiliary view escaped its bound request')
        result=original(image_path,question,answer);base=result['images']
        if not isinstance(base,torch.Tensor) or base.shape!=pixels.unsqueeze(0).shape:raise ValueError('unexpected processor shape')
        if expected is not None:torch.testing.assert_close(base,expected.unsqueeze(0).to(base),rtol=0,atol=0)
        result['images']=pixels.unsqueeze(0).to(device=base.device,dtype=base.dtype)
        return result
    adapter.prepare_inputs=prepare
    try:yield
    finally:adapter.prepare_inputs=original


def evidence_loss(runtime,record,entry,control,hook,sigma):
    row=entry['row']
    if row['role'] not in ('native','fit'):raise ValueError('evaluation ROI cannot supply gradients')
    outputs=[];tokens=0
    for mask in (control,entry['region']):
        pixels=suppressed(entry['pixels'],mask)
        with pixel_view(runtime.adapter,row,pixels,entry['pixels']):
            batch=runtime.build_edit_batch(replace(record,question=row['question'],image_path=Path(row['image_path']),target=row['reference']))
        hook.set_teacher_routing(batch.labels);result=runtime.model(**batch.forward_kwargs())
        score=answer_score(result.logits,batch.labels,batch.attention_mask,runtime.adapter.tokenizer.eos_token_id,runtime.adapter.tokenizer.bos_token_id,runtime.adapter.tokenizer.pad_token_id)
        outputs.append((result.loss,score));tokens+=len(batch.target_token_ids)
    d=outputs[0][1]-outputs[1][1]
    loss=outputs[0][0]+torch.relu(.2-sigma*d)
    return loss,dict(evidence_ce=float(outputs[0][0].detach()),evidence_d=float(d.detach()),sigma=sigma,evidence_tokens=tokens)
