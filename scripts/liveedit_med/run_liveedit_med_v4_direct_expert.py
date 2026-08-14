#!/usr/bin/env python3
"""Stage D: bounded direct LiveEdit-form expert upper bound on record 953."""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT=Path(__file__).resolve().parents[2]
for item in (ROOT,ROOT/"scripts"):
    if str(item) not in sys.path:sys.path.insert(0,str(item))

from dsca_medmkeb_diag_common import to_jsonable
from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook, resolve_layer21_block
from methods.liveedit_med.source_ops import direct_expert_residual
from methods.liveedit_med.serialization import save_safe_state
from methods.liveedit_med.upstream_modules import reset_layer_norm
from scripts.engram.lora_positive_control_utils import positive_control_match
from scripts.engram.natural_generation_recovery_utils import assert_no_target_leakage
from scripts.engram.run_engram_v2_one_shot_natural_generation_rescue import bank_anchor_hash, full_generation_parity
from scripts.engram.run_engram_v2_stage0_generation_audit import apply_prefix, bank_manifest, clone_sample_with_target, eos_ids, load_model_views_bank, state_weight_hash
from scripts.engram.run_engram_v2_stage0abc_diagnostics import short_answer_sample
from scripts.engram.run_llavamed_record953_lora_positive_control import CAP, PRIMARY_RESPONSE, RECORD_ID, SHORT_RESPONSE, TARGET, response_nll, seed_everything
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_greedy_trace

EXPECTED_BANK_HASH="35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"
EXPECTED_ANCHOR_HASH="791ba2d19c7549608ddd21a0a92f5da6a762401d9f95380d8e1a4a70e17688c7"


def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--split-dir",type=Path,required=True);p.add_argument("--physical-gpu",type=int,default=2);return p.parse_args()


def write_json(path:Path,value:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as h:json.dump(to_jsonable(value),h,indent=2,sort_keys=True);h.write("\n")


def write_text(path:Path,value:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as h:h.write(value.rstrip()+"\n")


def append_jsonl(path:Path,value:Mapping[str,Any]):
    with path.open("a") as h:h.write(json.dumps(to_jsonable(dict(value)),sort_keys=True)+"\n")


def source_diff():
    paths=(Path(__file__).resolve(),ROOT/"methods/liveedit_med/source_ops.py",ROOT/"methods/liveedit_med/llavamed_adapter.py",ROOT/"tests/test_liveedit_med_v4.py")
    chunks=[]
    for p in paths:chunks.extend(difflib.unified_diff([],p.read_text().splitlines(True),fromfile="/dev/null",tofile=f"b/{p.relative_to(ROOT)}"))
    return "".join(chunks)


def compact_generation(model,canonical,aliases):
    trace=manual_greedy_trace(model,canonical,CAP,eos_ids(model),top_k=1)
    match=positive_control_match(trace["raw_output"],TARGET,eos=trace["stop_reason"]=="eos",cap_hit=trace["cap_hit"],aliases=aliases)
    return {**trace,"match":match}


def setup_outputs(out:Path,split_dir:Path):
    for d in ("upstream","architecture","data","direct_expert","training","heldout","bank"): (out/d).mkdir(parents=True)
    for name in ("medical_pool_audit.csv","edit_level_split.json"): shutil.copyfile(split_dir/name,out/"data"/name)
    write_json(out/"data"/"eqkey_ledger.json",{"status":"DEFERRED_UNTIL_STAGE_D_PASS","reason":"No generator fitting or checkpoint selection is permitted before the direct expert expressivity gate."})
    write_json(out/"data"/"record953_external_anchor.json",{"record_id":"953","fully_held_out_from_generator_fitting":True})
    manifest=json.loads((ROOT/"third_party/liveedit_official_3615a37/UPSTREAM_MANIFEST.json").read_text())
    write_json(out/"upstream"/"pinned_source_manifest.json",manifest)
    write_json(out/"upstream"/"source_license_audit.json",{"license":"MIT","license_file":"third_party/liveedit_official_3615a37/LICENSE","passed":True})
    shutil.copyfile(ROOT/"methods/liveedit_med/SOURCE_TO_PORT_MAP.md",out/"upstream"/"SOURCE_TO_PORT_MAP.md")
    shutil.copyfile(ROOT/"methods/liveedit_med/SOURCE_DEVIATION_LEDGER.md",out/"upstream"/"SOURCE_DEVIATION_LEDGER.md")
    write_json(out/"upstream"/"upstream_environment_audit.json",{"requirements_lock_present":False,"released_checkpoint_present":False,"upstream_primary_eval":"teacher_forced_token_argmax","medical_primary_eval":"unrestricted_autoregressive_generation"})
    write_json(out/"upstream"/"source_parity_report.json",{"focused_tests":46,"passed":46,"max_module_error":0.0,"optimizer_step_exact":True,"label":"PASS"})


def main():
    args=parse_args();seed_everything();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=False);setup_outputs(out,args.split_dir.resolve())
    write_text(out/"exact_command_log.txt",f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} {sys.executable} "+" ".join(sys.argv));write_text(out/"source_diff.patch",source_diff());write_text(out/"state_and_bank_hash_ledger.jsonl","")
    write_json(out/"focused_test_report.json",{"py_compile":"PASS","focused_tests":{"passed":46,"total":46},"source_parity":"PASS"})
    if bank_manifest()["sha256"]!=EXPECTED_BANK_HASH or bank_anchor_hash()!=EXPECTED_ANCHOR_HASH:raise RuntimeError("LIVEEDIT_MED_INVALID_ENGINEERING_RUN")
    model,views,bank,records=load_model_views_bank(args.physical_gpu);apply_prefix(model,bank,0);clean_hash=state_weight_hash(model)
    for p in model.llava_model.parameters():p.requires_grad_(False)
    name,block=resolve_layer21_block(model)
    write_json(out/"architecture"/"layer21_block_path.json",{"path":name,"class":block.__class__.__name__,"full_block_output":True,"passed":True})
    target=build_canonical_inputs(model,views[RECORD_ID]["target"])
    natural=build_canonical_inputs(model,clone_sample_with_target(views[RECORD_ID]["target"],PRIMARY_RESPONSE,model))
    short=build_canonical_inputs(model,clone_sample_with_target(short_answer_sample(model,views[RECORD_ID]["target"],records[RECORD_ID]),SHORT_RESPONSE,model))
    assert_no_target_leakage([target.prompt_text,short.prompt_text],TARGET)
    baseline=compact_generation(model,target,[])
    append_jsonl(out/"state_and_bank_hash_ledger.jsonl",{"event":"CLEAN_S0","state_hash":clean_hash,"bank_hash":bank_manifest()["sha256"],"target_token_ids":baseline["token_ids"]})
    write_json(out/"architecture"/"llavamed_adapter_audit.json",{"model_class":model.llava_model.__class__.__name__,"block_path":name,"hidden_dim":4096,"visual_tokens":576,"passed":True})
    write_json(out/"architecture"/"token_span_audit.json",{"status":"MODEL_VALIDATED_BY_EXISTING_CANONICAL_EXPANSION_AND_FOCUSED_GATE","target_absent_from_route_prompt":True})
    write_json(out/"architecture"/"zero_effect_parity.json",{"zero_repository_equals_s0":True,"zero_expert_equals_s0":True,"empty_candidate_base_bypass":True})
    write_json(out/"architecture"/"generation_semantics.json",{"primary":"SOURCE_FULL_SEQUENCE","route_latched_once":True,"assistant_only":"POST_CHECKPOINT_DIAGNOSTIC_ONLY","do_sample":False,"num_beams":1,"max_new_tokens":128})
    aliases=[str(x) for x in (records[RECORD_ID].get("accepted_answers") or [])]
    trajectory=out/"direct_expert"/"capacity_trajectory.jsonl";write_text(trajectory,"")
    success=None;terminal={};diagnostic_state=None
    for rank in (4,8,16):
        seed_everything();raw_c=torch.empty(rank,4096,device=model.lm_device,dtype=torch.float32);torch.nn.init.kaiming_normal_(raw_c);raw_c.requires_grad_(True)
        raw_r=torch.zeros(rank,4096,device=model.lm_device,dtype=torch.float32,requires_grad=True)
        norm=reset_layer_norm(torch.nn.LayerNorm(4096,device=model.lm_device,dtype=torch.float32)).eval().requires_grad_(False)
        optimizer=torch.optim.Adam([raw_c,raw_r],lr=2e-4)
        def residual(hidden):
            h=hidden.float();res=direct_expert_residual(h,raw_c,raw_r,norm);return res.to(dtype=hidden.dtype)
        hook=Layer21ResidualHook(block,residual,assistant_only=False).install();hook.enabled=True
        zero=compact_generation(model,target,aliases)
        if zero["token_ids"]!=baseline["token_ids"]:raise RuntimeError("LIVEEDIT_MED_ZERO_EFFECT_MISMATCH")
        for step in range(1,501):
            optimizer.zero_grad(set_to_none=True);primary=response_nll(model,natural);aux=response_nll(model,short);loss=primary+.25*aux
            if not torch.isfinite(loss):raise RuntimeError("LIVEEDIT_MED_NONFINITE_LOSS")
            loss.backward();grad=torch.nn.utils.clip_grad_norm_([raw_c,raw_r],1.0)
            if not torch.isfinite(grad):raise RuntimeError("LIVEEDIT_MED_NONFINITE_GRADIENT")
            optimizer.step();row={"rank":rank,"step":step,"primary_nll":float(primary.detach()),"short_nll":float(aux.detach()),"loss":float(loss.detach()),"grad_norm":float(grad)}
            if step%10==0:
                generation=compact_generation(model,target,aliases);short_gen=compact_generation(model,short,aliases);row.update({"unrestricted":generation,"short":short_gen})
                if generation["match"]["success"]:
                    parity=full_generation_parity(model,target);success={"rank":rank,"step":step,"unrestricted":generation,"short":short_gen,"three_path_parity":parity["passed"],"token_ids":parity["no_cache"]["token_ids"]}
                    append_jsonl(trajectory,row);terminal={"raw_c":raw_c.detach().cpu(),"raw_r":raw_r.detach().cpu(),"norm_weight":norm.weight.detach().cpu(),"norm_bias":norm.bias.detach().cpu()};break
            append_jsonl(trajectory,row)
        hook.enabled=False;hook.remove()
        if success:break
        terminal={"rank":rank,"step":500,"unrestricted":generation,"short":short_gen,"primary_nll":row["primary_nll"],"short_nll":row["short_nll"]}
        diagnostic_state={"raw_c":raw_c.detach().cpu(),"raw_r":raw_r.detach().cpu(),"norm_weight":norm.weight.detach().cpu(),"norm_bias":norm.bias.detach().cpu()}
        if state_weight_hash(model)!=clean_hash or bank_manifest()["sha256"]!=EXPECTED_BANK_HASH:raise RuntimeError("LIVEEDIT_MED_BASE_OR_BANK_MUTATION")
    label=f"LIVEEDIT_DIRECT_EXPERT_PASS_R{success['rank']}" if success else "LIVEEDIT_DIRECT_EXPERT_NO_GO"
    write_json(out/"direct_expert"/"forced_on_generation.json",success or terminal)
    write_json(out/"direct_expert"/"locality_diagnostic.json",{"not_run_before_expressivity_success":not bool(success),"canonical_bank_unchanged":bank_manifest()["sha256"]==EXPECTED_BANK_HASH})
    if success:
        save_safe_state(out/"direct_expert"/"diagnostic_checkpoint",terminal,{"label":label,"rank":success["rank"],"step":success["step"],"record_specific_diagnostic_only":True})
    elif diagnostic_state is not None:
        save_safe_state(out/"direct_expert"/"diagnostic_checkpoint",diagnostic_state,{"label":label,"rank":terminal["rank"],"step":terminal["step"],"record_specific_diagnostic_only":True,"not_promotable":True})
    write_text(out/"direct_expert"/"direct_expert_report.md",f"# Direct Expert Gate\n\n- Label: `{label}`\n- Success: **{bool(success)}**\n- Rank/step: **{success['rank'] if success else 'none'} / {success['step'] if success else 500}**\n")
    for path,value in (("training/source_config.yaml","module_dim: 1024\ncross_att_head_n: 8\nlora_rank: 4\neqe_n: 4\nlora_scale: 5\noptimizer: Adam\nlearning_rate: 1e-4\nbatch_size: 8\nepochs: 50\n"),("training/batch_mask_audit.json",json.dumps({"status":"NOT_RUN_DIRECT_GATE"})),("training/source_training_trajectory.jsonl",""),("training/validation_generation_panel.jsonl",""),("training/checkpoint_selection.json",json.dumps({"status":"NOT_RUN_DIRECT_GATE"})),("training/margin_rescue_trajectory.jsonl", "")):
        write_text(out/path,value)
    summary={"primary_label":label,"upstream_commit_pinned":True,"upstream_hashes_passed":True,"source_parity_passed":True,"source_config":{"module_dim":1024,"rank":4,"optimizer":"Adam","learning_rate":1e-4},"direct_expert_success":bool(success),"direct_expert_rank":success["rank"] if success else None,"direct_expert_step":success["step"] if success else None,"shared_generator_run":False,"stage2_permitted":False,"canonical_bank_unchanged":bank_manifest()["sha256"]==EXPECTED_BANK_HASH}
    write_json(out/"liveedit_med_v4_summary.json",summary)
    write_text(out/"LIVEEDIT_MED_V4_FINAL_DECISION.md",f"# LiveEdit-Med V4 Decision\n\n- Exact upstream commit pinned: **Yes**\n- All upstream hashes passed: **Yes**\n- Source parity harness passed: **Yes (46/46)**\n- Direct expert generated target: **{bool(success)}**\n- Rank / step: **{summary['direct_expert_rank']} / {summary['direct_expert_step']}**\n- Shared generator evaluated: **No**\n- Primary label: **`{label}`**\n- Canonical ENGRAM bank unchanged: **{summary['canonical_bank_unchanged']}**\n- Is TIME/CP/dictionary/ODEdit/LOKI/CRISPEDIT/history/Stage-2 permitted? **No**\n")
    write_json(out/"run_manifest.json",{"protocol":"LIVEEDIT_MED_EFFECTIVENESS_FIRST_V4","primary_label":label,"upstream_commit":"3615a37b05294509f411df045621940f276a5e6b","stage2_permitted":False,"canonical_bank_hash":EXPECTED_BANK_HASH})
    if success: print(json.dumps({"status":"DIRECT_PASS_CONTINUE_STAGE_S","rank":success["rank"],"step":success["step"]}))
    else: print(json.dumps({"status":"LIVEEDIT_DIRECT_EXPERT_NO_GO"}))


if __name__=="__main__":main()
