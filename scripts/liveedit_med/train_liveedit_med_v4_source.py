#!/usr/bin/env python3
"""Stage S: exact source-objective joint LiveEdit training on cached frozen reps."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.cached_suffix import answer_kl, answer_nll, forward_suffix_hidden
from methods.liveedit_med.serialization import save_safe_state
from methods.liveedit_med.source_ops import apply_low_rank_expert_residual, compute_text_soft_weights, source_routing_losses, source_soft_losses
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from dsca_medmkeb_diag_common import ensure_offline_env
from easyeditor.models.engram import EngramMultimodalHparams
from easyeditor.models.engram_v2 import SequentialEngramBankV2
from easyeditor.trainer.models import get_model
from scripts.engram.run_engram_continual_v2 import set_determinism
from scripts.engram.run_engram_v2_stage0_generation_audit import BANK_ROOT, MODEL_CONFIG, MODULE_KEY, MODULE_NAME, apply_prefix, bank_manifest, load_model_views_bank, state_weight_hash
from scripts.engram.run_llavamed_record953_lora_positive_control import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--suffix-physical-gpu", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def module_state(modules: LiveEditMedicalModules):
    return {name: value for name, value in modules.state_dict().items()}


def load_training_model(primary: int, suffix: int | None):
    if suffix is None:
        model,_views,bank,_raw=load_model_views_bank(primary);return model,bank,model.lm_device
    if os.environ.get("CUDA_VISIBLE_DEVICES") != f"{primary},{suffix}" or torch.cuda.device_count()!=2:
        raise RuntimeError("LIVEEDIT_MED_TWO_GPU_VISIBILITY_MISMATCH")
    ensure_offline_env();set_determinism(42);config=EngramMultimodalHparams.from_hparams(str(MODEL_CONFIG));config.dropout,config.no_grad_layers,config.device=0.0,None,"cuda:0"
    model=get_model(config).to(torch.device("cuda:0")).eval();bank=SequentialEngramBankV2(BANK_ROOT);module=dict(model.named_modules()).get(MODULE_NAME)
    expected=bank.anchor_state()[MODULE_KEY].to(dtype=module.weight.dtype)
    if not torch.equal(module.weight.detach().cpu(),expected):raise RuntimeError("LIVEEDIT_MED_S0_ANCHOR_MISMATCH")
    suffix_device=torch.device("cuda:1");core=model.llava_model.model
    for layer in core.layers[22:core.config.num_hidden_layers]:layer.to(suffix_device)
    core.norm.to(suffix_device);core.rotary_emb.to(suffix_device);model.llava_model.lm_head.to(suffix_device)
    return model,bank,suffix_device


def split_variant(tensors: dict[str, torch.Tensor], key: str, device: torch.device):
    prefix = key + "__"
    return {name[len(prefix):]: value.to(device) for name, value in tensors.items() if name.startswith(prefix)}


def spans(row: dict[str, torch.Tensor]):
    hidden = row["hidden"].float().unsqueeze(0)
    return hidden[:, row["vision"]], hidden[:, row["prompt"]], hidden[:, row["answer"]]


def pad_rows(rows: list[dict[str, torch.Tensor]], values: list[torch.Tensor], device: torch.device):
    maximum = max(int(value.shape[1]) for value in values); batch = len(values); dim = int(values[0].shape[-1])
    hidden = torch.zeros(batch, maximum, dim, device=device, dtype=values[0].dtype)
    attention = torch.zeros(batch, maximum, device=device, dtype=torch.long)
    labels = torch.full((batch, maximum), -100, device=device, dtype=torch.long)
    for index, (row, value) in enumerate(zip(rows, values)):
        length = value.shape[1]; hidden[index, :length] = value[0]; attention[index, :length] = row["attention"].long(); labels[index, :length] = row["labels"].long()
    return hidden, attention, labels


def edited_values(modules, model, rows, moe_cs, moe_rs, masks, eqrs):
    edited = []
    for row, mask in zip(rows, masks):
        vision, question, _answer = spans(row)
        input_key = modules.input_extractor.extract_query(question)
        selected = int(mask.sum())
        if selected == 0:
            residual = torch.zeros_like(row["hidden"].float().unsqueeze(0))
        else:
            weights = compute_text_soft_weights(input_key, eqrs[mask])
            residual = apply_low_rank_expert_residual(row["hidden"].float().unsqueeze(0), moe_cs[mask], moe_rs[mask], weights, modules.instant_reps_norm)
        edited.append((row["hidden"].float().unsqueeze(0) + residual).to(model.llava_model.dtype))
    return edited


def grouped_residual_losses(modules, model, groups, moe_cs, moe_rs, eqrs, suffix_device):
    all_rows=[];all_values=[];ranges=[]
    for name,rows,masks,locality in groups:
        begin=len(all_rows);all_rows.extend(rows);all_values.extend(edited_values(modules,model,rows,moe_cs,moe_rs,masks,eqrs));ranges.append((name,begin,len(all_rows),locality))
    hidden, attention, labels = pad_rows(all_rows, all_values, suffix_device)
    suffix_hidden=forward_suffix_hidden(model.llava_model,hidden,attention,gradient_checkpointing=True)
    suffix_hidden_logits=[]
    for index,row in enumerate(all_rows):
        answer_positions=torch.where(row["answer"])[0];predictors=answer_positions-1
        if bool((predictors<0).any()):raise RuntimeError("LIVEEDIT_MED_INVALID_PREDICTOR_POSITION")
        suffix_hidden_logits.append(model.llava_model.lm_head(suffix_hidden[index,predictors.to(suffix_hidden.device)]))
    result={}
    for name,begin,end,locality in ranges:
        losses=[]
        for index in range(begin,end):
            row=all_rows[index]
            if locality:losses.append(answer_kl(suffix_hidden_logits[index],row["base_answer_logits"].to(suffix_hidden_logits[index].device)))
            else:
                target=row["labels"][row["answer"]].long().to(suffix_hidden_logits[index].device);losses.append(torch.nn.functional.cross_entropy(suffix_hidden_logits[index].float(),target))
        result[name]=torch.stack(losses).mean().to(model.lm_device)
    return result


def routing_pairs(batch, rng_data):
    names=("textual","visual","paired"); neighbors=[[],[]]; prototypes=[[],[]]
    for record in batch:
        rel=record["variants"]["native_0"]; gen={name:record["variants"][f"gen_{name}_0"] for name in names}; loc=record["variants"]["loc_image_or_paired_0"]
        first=int(rng_data.integers(0,3)); gn=names[int(rng_data.integers(0,3))]; choices=(rel,gen[gn],loc); neighbors[0].append(spans(choices[first])[:2])
        second=int(rng_data.integers(0,2)) if first != 2 else 2; gn=names[int(rng_data.integers(0,3))]; choices=(rel,gen[gn],loc); neighbors[1].append(spans(choices[second])[:2])
        kind=int(rng_data.integers(0,2)); gn=names[int(rng_data.integers(0,3))]; positive=(rel,gen[gn])[int(rng_data.integers(0,2))]; prototypes[0].append(spans((positive,loc)[kind])[:2])
        gn=names[int(rng_data.integers(0,3))]; positive=(rel,gen[gn])[int(rng_data.integers(0,2))]; prototypes[1].append(spans((positive,loc)[1-kind])[:2])
    return neighbors,prototypes


def main():
    args=parse_args(); seed_everything();
    expected_visible=str(args.physical_gpu) if args.suffix_physical_gpu is None else f"{args.physical_gpu},{args.suffix_physical_gpu}"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible: raise RuntimeError("CUDA_VISIBLE_DEVICES mismatch")
    if args.batch_size != 8 or args.epochs not in (1,50): raise RuntimeError("LIVEEDIT_MED_SOURCE_CONFIG_DRIFT")
    if args.max_steps is not None and args.max_steps < 1: raise RuntimeError("LIVEEDIT_MED_INVALID_MAX_STEPS")
    manifest=json.loads((args.cache_dir/"manifest.json").read_text()); entries=manifest["records"]
    if args.epochs==50 and len(entries)!=512: raise RuntimeError(f"LIVEEDIT_MED_CACHE_COUNT:{len(entries)}")
    if args.epochs==1 and len(entries)!=8: raise RuntimeError(f"LIVEEDIT_MED_SMOKE_CACHE_COUNT:{len(entries)}")
    model,bank,suffix_device=load_training_model(args.physical_gpu,args.suffix_physical_gpu); apply_prefix(model,bank,0); clean_hash=state_weight_hash(model)
    for parameter in model.llava_model.parameters(): parameter.requires_grad_(False)
    modules=LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float(); modules.assert_trainable_boundary(model.llava_model); modules.train()
    optimizer,scheduler=modules.optimizer(); rng_data=np.random.default_rng(42); rng_train=np.random.default_rng(43); rng_order=np.random.default_rng(42);step=0
    trajectory=args.run_dir/"training/source_training_trajectory.jsonl";trajectory.write_text("")
    for epoch in range(1,args.epochs+1):
        order=rng_order.permutation(len(entries))
        for begin in range(0,len(entries),args.batch_size):
            batch=[]
            for index in order[begin:begin+args.batch_size]:
                entry=entries[int(index)]; tensors=load_file(str(args.cache_dir/entry["file"]),device="cpu"); keys=[v["key"] for v in entry["variants"]]
                batch.append({"record_id":entry["record_id"],"variants":{key:split_variant(tensors,key,model.lm_device) for key in keys}})
            step+=1; optimizer.zero_grad(set_to_none=True)
            edits=[]
            for record in batch:
                vision,question,answer=spans(record["variants"]["native_0"]); edits.append(modules.generated_edit(vision,question,answer))
            eqrs=torch.cat([x[0] for x in edits]); moe_cs=torch.cat([x[2] for x in edits]); moe_rs=torch.cat([x[3] for x in edits]); count=len(batch)
            rel_mask=torch.eye(count,device=model.lm_device,dtype=torch.bool); gen_mask=rel_mask.clone(); loc_mask=torch.zeros_like(rel_mask)
            prefixes=[]
            for index in range(count):
                ns=rng_train.integers(0,count+1,3); prefixes.append(ns.tolist()); rel_mask[index,:ns[0]]=True; gen_mask[index,:ns[1]]=True; loc_mask[index,:ns[2]]=True
            rel_rows=[r["variants"]["native_0"] for r in batch];loc_rows=[r["variants"]["loc_image_or_paired_0"] for r in batch]
            groups=[("rel",rel_rows,rel_mask,False)]+[(f"gen_{name}",[r["variants"][f"gen_{name}_0"] for r in batch],gen_mask,False) for name in ("textual","visual","paired")]+[("loc",loc_rows,loc_mask,True)]
            task_losses=grouped_residual_losses(modules,model,groups,moe_cs,moe_rs,eqrs,suffix_device);rel=task_losses["rel"];generalities=[task_losses[f"gen_{name}"] for name in ("textual","visual","paired")];loc=task_losses["loc"]
            neighbors,prototypes=routing_pairs(batch,rng_data)
            input_keys=torch.cat([modules.input_extractor.extract_query(pair[1]) for pair in neighbors[0]])
            edit_keys=torch.cat([modules.edit_extractor.extract_query(pair[1]) for pair in neighbors[1]])
            soft_rel,soft_abs=source_soft_losses(input_keys,edit_keys)
            hard_neighbor,hard_prototype=source_routing_losses(modules.input_extractor,modules.edit_extractor,neighbors[0],neighbors[1],prototypes[0],prototypes[1])
            total=rel+sum(generalities)+loc+soft_rel+soft_abs+hard_neighbor+hard_prototype
            if not torch.isfinite(total): raise RuntimeError("LIVEEDIT_MED_NONFINITE_SOURCE_LOSS")
            total.backward(); grad=torch.nn.utils.clip_grad_norm_(modules.parameters(),float("inf"))
            if not torch.isfinite(grad): raise RuntimeError("LIVEEDIT_MED_NONFINITE_SOURCE_GRADIENT")
            optimizer.step(); scheduler.step()
            row={"epoch":epoch,"step":step,"record_ids":[r["record_id"] for r in batch],"loss":float(total.detach()),"reliability":float(rel.detach()),"generality":[float(x.detach()) for x in generalities],"locality":float(loc.detach()),"soft_relative":float(soft_rel.detach()),"soft_absolute":float(soft_abs.detach()),"hard_neighbor":float(hard_neighbor.detach()),"hard_prototype":float(hard_prototype.detach()),"lr":scheduler.get_last_lr()[0],"grad_norm":float(grad),"prefixes":prefixes}
            with trajectory.open("a") as handle: handle.write(json.dumps(row,sort_keys=True)+"\n")
            if step%10==0: print(json.dumps({k:row[k] for k in ("epoch","step","loss","reliability","locality")}),flush=True)
            if step%500==0: save_safe_state(args.run_dir/f"training/checkpoint_{step:04d}",module_state(modules),{"stage":"S","step":step,"epoch":epoch,"source_objective":True})
            if args.max_steps is not None and step>=args.max_steps:
                save_safe_state(args.run_dir/"training"/f"checkpoint_smoke_{step}",module_state(modules),{"stage":"S","step":step,"epoch":epoch,"source_objective":True,"smoke":True});print(json.dumps({"status":"STAGE_S_MAX_STEPS_COMPLETE","steps":step}));return
    final_name="checkpoint_3200" if args.epochs==50 else "checkpoint_smoke"
    save_safe_state(args.run_dir/"training"/final_name,module_state(modules),{"stage":"S","step":step,"epoch":args.epochs,"source_objective":True,"smoke":args.epochs==1})
    if state_weight_hash(model)!=clean_hash or bank_manifest()["sha256"]!="35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db": raise RuntimeError("LIVEEDIT_MED_BASE_OR_BANK_MUTATION")
    print(json.dumps({"status":"STAGE_S_TRAINING_COMPLETE","steps":step}))


if __name__=="__main__": main()
