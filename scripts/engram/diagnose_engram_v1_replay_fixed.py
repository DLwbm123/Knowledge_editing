#!/usr/bin/env python3
"""Re-evaluate ENGRAM V1 direct/replay/reload under the fixed deterministic harness."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dsca_medmkeb_diag_common import clone_batch, ensure_offline_env
from easyeditor.models.engram import EngramBank, EngramMultimodalHparams, EngramMultimodalRewriteExecutor
from easyeditor.models.engram.engram_main import select_linear_layers
from easyeditor.trainer.models import get_model
from scripts.engram.engram_eval_utils import full_state_sha256, shifted_teacher_forced_metrics, tensor_sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def set_determinism(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_image(root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return root.parent / path if root.name == "images" and str(value).startswith("images/") else root / path


def prompt(value):
    return f"Question: {str(value or '')} Short answer: "


def make_sample(model, question, answer, image):
    text = prompt(question)
    target = str(answer or "")
    labels = model.llava_tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.lm_device)
    return {"image_path": [str(image)], "prompt": [text], "target": [target], "text_input": [text + target], "labels": labels, "prompts_len": [len(model.llava_tokenizer(text, add_special_tokens=False).input_ids)]}


def evaluate(model, sample):
    batch = clone_batch(sample)
    with torch.inference_mode():
        outputs = model(batch)
    return shifted_teacher_forced_metrics(outputs.logits, outputs.labels, ignore_index=model.IGNORE_INDEX)


def snapshot(layers):
    return {layer.name: {"weight": layer.module.weight.detach().clone(), "bias": layer.module.bias.detach().clone() if layer.module.bias is not None else None} for layer in layers}


def restore(layers, state):
    with torch.no_grad():
        for layer in layers:
            layer.module.weight.copy_(state[layer.name]["weight"])
            if layer.module.bias is not None:
                layer.module.bias.copy_(state[layer.name]["bias"])


def layer_hashes(layers):
    return {layer.name: {"weight": tensor_sha256(layer.module.weight), "bias": tensor_sha256(layer.module.bias) if layer.module.bias is not None else None} for layer in layers}


def layer_diff(layers, state):
    values=[]
    for layer in layers:
        values.append(float((layer.module.weight.detach().float()-state[layer.name]["weight"].float()).abs().max().cpu()))
        if layer.module.bias is not None:
            values.append(float((layer.module.bias.detach().float()-state[layer.name]["bias"].float()).abs().max().cpu()))
    return max(values,default=0.0)


def main():
    args=parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    bank_root=args.out_dir/"bank"
    if bank_root.exists(): shutil.rmtree(bank_root)
    ensure_offline_env(); set_determinism()
    cfg=EngramMultimodalHparams.from_hparams(str(ROOT/'hparams/ENGRAM/llava_med_continual_v1.yaml'))
    cfg.dropout=0.0; cfg.no_grad_layers=None; cfg.device='cuda'; cfg.bank_dir=str(bank_root); cfg.edit_id='edit_01_953'; cfg.concept_id='953'
    model=get_model(cfg).to(torch.device('cuda')).eval()
    layers=select_linear_layers(model,cfg)
    anchor=snapshot(layers)
    anchor_full_hash,anchor_meta=full_state_sha256(model)
    records=json.loads((ROOT/'datasets/MedMKEB/eval.json').read_text()); record=records[0]
    image_root=ROOT/Path(cfg.coco_image)
    sample=make_sample(model,record['src'],record['alt'],resolve_image(image_root,record['image']))
    request={
        'record_id':str(record['id']), 'prompt':prompt(record['src']), 'target':record['alt'], 'image':str(resolve_image(image_root,record['image'])),
        'rephrase_prompt':prompt(record['rephrase']), 'image_rephrase':str(resolve_image(image_root,record['image_rephrase'])),
        'multimodal_locality_prompt':prompt(record['m_loc_q']), 'multimodal_locality_ground_truth':record['m_loc_a'], 'multimodal_locality_image':str(resolve_image(image_root,record['m_loc'])),
    }
    baseline=evaluate(model,sample)
    executor=EngramMultimodalRewriteExecutor()
    executor.apply_to_model(model,model.llava_tokenizer,[request],cfg)
    direct_metric=evaluate(model,sample); direct_layers=layer_hashes(layers); direct_full_hash,direct_meta=full_state_sha256(model)

    bank=EngramBank(bank_root)
    bank.rollback_edit(model,cfg.edit_id)
    arithmetic_rollback_error=layer_diff(layers,anchor)
    restore(layers,anchor)
    exact_rollback_hash,_=full_state_sha256(model)

    bank.apply_edit(model,cfg.edit_id)
    replay_metric=evaluate(model,sample); replay_layers=layer_hashes(layers); replay_full_hash,replay_meta=full_state_sha256(model)
    restore(layers,anchor)

    fresh_bank=EngramBank(bank_root)
    fresh_bank.apply_edit(model,cfg.edit_id)
    fresh_metric=evaluate(model,sample); fresh_layers=layer_hashes(layers); fresh_full_hash,fresh_meta=full_state_sha256(model)

    payload={
        'record_id':'953', 'selected_layers':[layer.name for layer in layers], 'baseline':baseline,
        'direct_metric':direct_metric, 'replay_metric':replay_metric, 'fresh_reload_metric':fresh_metric,
        'direct_replay_nll_abs_diff':abs(direct_metric['target_nll']-replay_metric['target_nll']),
        'direct_fresh_nll_abs_diff':abs(direct_metric['target_nll']-fresh_metric['target_nll']),
        'anchor_full_state_sha256':anchor_full_hash, 'direct_full_state_sha256':direct_full_hash,
        'replay_full_state_sha256':replay_full_hash, 'fresh_full_state_sha256':fresh_full_hash,
        'exact_rollback_full_state_sha256':exact_rollback_hash,
        'direct_replay_full_state_equal':direct_full_hash==replay_full_hash,
        'direct_fresh_full_state_equal':direct_full_hash==fresh_full_hash,
        'exact_rollback_anchor_equal':anchor_full_hash==exact_rollback_hash,
        'direct_layer_hashes':direct_layers, 'replay_layer_hashes':replay_layers, 'fresh_layer_hashes':fresh_layers,
        'arithmetic_fp16_rollback_max_abs_error':arithmetic_rollback_error,
        'full_state_meta':{'anchor':anchor_meta,'direct':direct_meta,'replay':replay_meta,'fresh':fresh_meta},
        'effective_update_norm_ratio':executor.last_report['metadata'].get('effective_update_norm_ratio'),
        'gate':None,
    }
    payload['gate']='PASS' if payload['direct_replay_full_state_equal'] and payload['direct_fresh_full_state_equal'] and payload['exact_rollback_anchor_equal'] and payload['direct_replay_nll_abs_diff']==0.0 and payload['direct_fresh_nll_abs_diff']==0.0 else 'FAIL'
    (args.out_dir/'v1_replay_fixed.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=='__main__': main()
