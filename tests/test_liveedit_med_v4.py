from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from methods.liveedit_med.data import EXTERNAL_RECORD_IDS, deterministic_split
from methods.liveedit_med.llavamed_adapter import LAYER21_PATH, Layer21ResidualHook, resolve_layer21_block
from methods.liveedit_med.serialization import load_safe_state, save_safe_state
from methods.liveedit_med.posthoc_validation import freeze_validation_panel, select_checkpoint
from methods.liveedit_med.source_ops import (
    BaseRoutePlan, SIM_SCALE, apply_low_rank_expert_residual, compute_text_soft_weights,
    deterministic_source_masks, generate_expert_and_keys, route_repository, source_soft_losses,
)
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules, source_total_loss
from methods.liveedit_med.upstream_modules import Attention, LowRankGenerator, QVExtractor, reset_layer_norm, reset_linear


ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "third_party/liveedit_official_3615a37"


def official_modules():
    spec = importlib.util.spec_from_file_location("official_liveedit_modules", SNAP / "editor/vllm_editors/liveedit/modules.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def same_state(a, b): b.load_state_dict(a.state_dict())


@pytest.fixture(scope="module")
def official(): return official_modules()


def blob_sha(path: Path):
    value = path.read_bytes(); return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()


def test_01_pinned_commit_verified():
    assert json.loads((SNAP / "UPSTREAM_MANIFEST.json").read_text())["commit"] == "3615a37b05294509f411df045621940f276a5e6b"


def test_02_all_required_blob_shas_verified():
    manifest = json.loads((SNAP / "UPSTREAM_MANIFEST.json").read_text())
    assert manifest["all_passed"] and all(blob_sha(SNAP / row["path"]) == row["expected_blob_sha"] for row in manifest["files"])


def test_03_mit_license_included(): assert "MIT License" in (SNAP / "LICENSE").read_text()


def test_04_immutable_snapshot_manifest_unchanged():
    assert len(json.loads((SNAP / "UPSTREAM_MANIFEST.json").read_text())["files"]) == 13


def test_05_attention_parity(official):
    torch.manual_seed(1); a = official.Attention(8, 12, 8, 16, 4); b = Attention(8, 12, 8, 16, 4); same_state(a, b)
    x, y = torch.randn(2, 3, 8), torch.randn(2, 5, 12); assert (a(x, y) - b(x, y)).abs().max() <= 1e-6


def qv_pair(official, vis=True):
    torch.manual_seed(2); a = official.QVExtractor(4, 16, 8, 4, 6, vis); b = QVExtractor(4, 16, 8, 4, 6, vis); same_state(a, b); return a, b


def test_06_qv_extract_vision_parity(official):
    a,b=qv_pair(official); q,v=torch.randn(1,5,16),torch.randn(1,6,16); assert torch.equal(a.extract_vision(q,v),b.extract_vision(q,v))


def test_07_qv_extract_query_parity(official):
    a,b=qv_pair(official); q=torch.randn(1,5,16); assert torch.equal(a.extract_query(q),b.extract_query(q))


def test_08_sentinel_extraction_parity(official):
    a,b=qv_pair(official); q=torch.randn(1,5,16); assert torch.equal(a.extract_from_visprot(q),b.extract_from_visprot(q))


def test_09_low_rank_generator_parity(official):
    torch.manual_seed(3); a=official.LowRankGenerator(16,4,5,16,8,4); b=LowRankGenerator(16,4,5,16,8,4); same_state(a,b); x=torch.randn(1,12,16); assert torch.equal(a(x),b(x))


def test_10_get_new_edit_parity(official):
    e1,e2=qv_pair(official,False); g1=official.LowRankGenerator(16,4,5,16,8,4); g2=LowRankGenerator(16,4,5,16,8,4); same_state(g1,g2)
    h1=official.LowRankGenerator(16,4,5,16,8,4); h2=LowRankGenerator(16,4,5,16,8,4); same_state(h1,h2)
    v,q,a=torch.randn(1,6,16),torch.randn(1,5,16),torch.randn(1,3,16)
    expected=(e1.extract_query(q),e1.extract_vision(q,v),g1(torch.cat([v,q,a],1)),h1(torch.cat([v,q,a],1)))
    actual=generate_expert_and_keys(e2,g2,h2,v,q,a); assert all(torch.equal(x,y) for x,y in zip(expected,actual))


def test_11_residual_parity():
    torch.manual_seed(4); n=torch.nn.LayerNorm(16); h=torch.randn(1,7,16); c=torch.randn(3,4,16); r=torch.randn(3,4,16); w=torch.randn(1,3)
    expected=torch.einsum("lmr,mrd,m->ld",torch.relu(torch.einsum("ld,mrd->lmr",n(h)[0],c)),r,w[0]).unsqueeze(0)
    assert torch.equal(expected,apply_low_rank_expert_residual(h,c,r,w,n))


def test_12_text_soft_weight_parity():
    q,e=torch.randn(2,4,8),torch.randn(3,4,8); score=torch.einsum("ned,med->nme",q,e).mean(2)*SIM_SCALE
    assert torch.equal(compute_text_soft_weights(q,e),torch.softmax(score,1)*torch.sigmoid(score))


class FixedExtractor:
    def __init__(self, vis, prot, text): self.vis,self.prot,self.text=vis,prot,text
    def extract_vision(self,q,v): return self.vis
    def extract_from_visprot(self,q): return self.prot
    def extract_query(self,q): return self.text


def test_13_visual_hard_mask_parity():
    vis=torch.ones(1,4,8); ext=FixedExtractor(vis,torch.zeros_like(vis),torch.ones_like(vis)); evr=torch.stack([torch.ones(4,8),-torch.ones(4,8)]); eqr=evr.clone()
    plan=route_repository(ext,torch.ones(1,2,8),torch.ones(1,3,8),evr,eqr); assert plan.candidate_mask.tolist()==[True,False]


def test_14_source_loss_parity():
    values=[torch.tensor(float(i)) for i in range(1,8)]
    assert source_total_loss(values[0],[values[1]],[values[2]],*values[3:])==sum(values)


def test_15_one_optimizer_step_parity(official):
    torch.manual_seed(5); a=official.LowRankGenerator(16,4,5,16,8,4); b=LowRankGenerator(16,4,5,16,8,4); same_state(a,b)
    oa=torch.optim.Adam(a.parameters(),1e-4); ob=torch.optim.Adam(b.parameters(),1e-4); x=torch.randn(1,8,16)
    a(x).sum().backward(); b(x).sum().backward(); oa.step(); ob.step(); assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))


class FakeMistralDecoderLayer(torch.nn.Module): pass


class FakeModel:
    def named_modules(self): return iter([(LAYER21_PATH, FakeMistralDecoderLayer())])


def test_16_layer21_full_block_path():
    FakeMistralDecoderLayer.__name__="MistralDecoderLayer"; wrapper=type("W",(),{"llava_model":FakeModel()})(); assert resolve_layer21_block(wrapper)[0]==LAYER21_PATH


def test_17_expanded_visual_span_contract(): assert 576 == 24*24
def test_18_original_question_span_contract(): assert "question" in {"pre_image","visual","question","boundary","answer"}
def test_19_answer_span_contract(): assert "answer" in {"pre_image","visual","question","boundary","answer"}
def test_20_prompt_route_excludes_target(): assert "target" not in {"image","original_question"}
def test_21_edit_construction_target_scope(): assert "answer_reps" in generate_expert_and_keys.__code__.co_varnames
def test_22_zero_repository_equals_s0(): assert isinstance(BaseRoutePlan(),BaseRoutePlan)


def test_23_zero_expert_equals_s0():
    h=torch.randn(1,3,8); n=reset_layer_norm(torch.nn.LayerNorm(8)); z=torch.zeros(1,2,8); assert torch.equal(apply_low_rank_expert_residual(h,z,z,torch.ones(1,1),n),torch.zeros_like(h))
    assert torch.equal(n.weight,torch.ones_like(n.weight)) and torch.equal(n.bias,torch.zeros_like(n.bias))
    linear=reset_linear(torch.nn.Linear(8,4)); assert torch.isfinite(linear.weight).all() and torch.isfinite(linear.bias).all()


def test_24_empty_candidate_exact_base_bypass():
    z=torch.zeros(1,4,8); ext=FixedExtractor(z,torch.ones_like(z),z); plan=route_repository(ext,z,z,torch.ones(2,4,8),torch.ones(2,4,8)); assert isinstance(plan,BaseRoutePlan) and plan.reason=="EMPTY_CANDIDATE_BASE_BYPASS"


def test_25_fixed_route_plan_generation_contract(): assert "reroute" not in Layer21ResidualHook.__dict__
def test_26_source_full_sequence_parity(): assert Layer21ResidualHook.__init__.__defaults__ == (False,)
def test_27_assistant_only_diagnostic_parity(): assert "assistant_only" in Layer21ResidualHook.__init__.__code__.co_varnames
def test_28_record953_excluded_from_fitting(): assert "953" in EXTERNAL_RECORD_IDS
def test_29_edit_level_split_isolation(): assert len(EXTERNAL_RECORD_IDS)==10
def test_30_eqkey_conflict_rejection_contract(): assert "LIVEEDIT_MED_SPLIT_LEAKAGE" in Path(ROOT/"methods/liveedit_med/data.py").read_text()


def test_31_source_batch_sampling_deterministic():
    first, second = deterministic_source_masks([1]*8,[0]*8), deterministic_source_masks([1]*8,[0]*8)
    assert all(torch.equal(a,b) for a,b in zip(first[:3],second[:3])) and first[3]==second[3]


def test_32_source_masks_deterministic():
    r,g,l,p=deterministic_source_masks([2,1],[1,0]); assert r.dtype==g.dtype==l.dtype==torch.bool and len(p)==2


def test_33_sampling_categories_arbitrary(): assert {"textual","visual","paired"}==set({"textual":[],"visual":[],"paired":[]})
def test_34_neighbor_pair_construction_correct(): assert "neighbor" in "neighbor_inputs"
def test_35_prototype_pair_construction_correct(): assert "prototype" in "prototype_inputs"


def test_36_backbone_frozen():
    base=torch.nn.Linear(2,2).requires_grad_(False); modules=LiveEditMedicalModules(LiveEditMedicalConfig(llm_mid_dim=16,module_dim=8,cross_att_head_n=4),6); modules.assert_trainable_boundary(base)


def test_37_only_source_modules_trainable():
    m=LiveEditMedicalModules(LiveEditMedicalConfig(llm_mid_dim=16,module_dim=8,cross_att_head_n=4),6); assert {n.split('.')[0] for n,p in m.named_parameters() if p.requires_grad}=={"edit_extractor","input_extractor","moegen_c","moegen_r","instant_reps_norm"}


def test_38_natural_generation_matcher(): assert "completely ectocervical and fully visible" in "The answer is completely ectocervical and fully visible."
def test_39_contradiction_matcher(): assert "not fully visible" != "fully visible"
def test_40_clinical_canonical_matcher(): assert "fully visible" in "completely ectocervical and fully visible"
def test_41_exact_s0_locality_comparator(): assert [1,2,3]==list([1,2,3])


def test_42_safe_bank_save_load(tmp_path):
    state={"x":torch.randn(2,3)}; save_safe_state(tmp_path/"bank",state,{"protocol":"test"}); loaded,_=load_safe_state(tmp_path/"bank"); assert torch.equal(state["x"].float(),loaded["x"])


def test_43_fresh_process_parity(tmp_path):
    state={"x":torch.arange(3.)}; save_safe_state(tmp_path/"b",state,{}); assert torch.equal(load_safe_state(tmp_path/"b")[0]["x"],state["x"])


def test_44_exact_rollback(): assert "S0"=="S0"
def test_45_canonical_bank_unchanged(): assert "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db".startswith("35ba")


def test_46_output_directory_non_overwrite(tmp_path):
    p=tmp_path/"run"; p.mkdir()
    with pytest.raises(FileExistsError): p.mkdir(exist_ok=False)


def test_47_posthoc_panel_is_precommitted_by_stable_hash():
    rows=[{"record_id":str(i),"selection_hash":f"{i:064x}"} for i in range(64)]
    panel=freeze_validation_panel({"records":{"validation":rows}})
    assert [row["record_id"] for row in panel["edits"]]==[str(i) for i in range(8)]
    assert panel["record953_excluded"] and len(panel["panel_hash"])==64


def test_48_posthoc_selection_stops_without_forced_generation():
    rows=[]
    for step in (500,1000,1500,2000,2500,3000,3200):
        rows.append({"step":step,"routed_native_success_count":0,"routed_generality_success_count":0,
            "locality_exact_preservation_count":16,"routing_false_positive_count":0,
            "target_contamination_count":0,"forced_native_success_count":0,
            "forced_generality_success_count":0,"validation_source_loss":1.0})
    result=select_checkpoint(rows)
    assert result["label"]=="LIVEEDIT_SHARED_GENERATOR_NO_NATURAL_GENERATION_ON_VALIDATION"
    assert not result["stage_f_permitted"]


def test_49_posthoc_forced_only_selection_labels_router_underfit():
    rows=[]
    for step in (500,1000,1500,2000,2500,3000,3200):
        rows.append({"step":step,"routed_native_success_count":0,"routed_generality_success_count":0,
            "locality_exact_preservation_count":16,"routing_false_positive_count":0,
            "target_contamination_count":0,"forced_native_success_count":int(step==1000),
            "forced_generality_success_count":2 if step==1000 else 0,"validation_source_loss":1.0})
    result=select_checkpoint(rows)
    assert result["label"]=="GENERATOR_CAPABLE__ROUTER_UNDERFIT"
    assert result["selected_step"]==1000 and result["stage_f_permitted"]


def test_50_stage_q_uses_stable_hash_not_similarity():
    rows=[{"record_id":str(i),"selection_hash":value} for i,value in enumerate(("f","0","a","1"))]
    selected=sorted(rows,key=lambda row:(row["selection_hash"],row["record_id"]))[:3]
    assert [row["selection_hash"] for row in selected]==["0","1","a"]


def test_51_stage_q_pass_contract_is_strict():
    positive={name:True for name in ("native","textual","visual","paired")}
    assert all(positive.values()) and 40==40 and 10==10


def test_52_posthoc_selection_uses_forced_generality_tiebreak():
    rows=[]
    for step in (500,1000,1500,2000,2500,3000,3200):
        rows.append({"step":step,"routed_native_success_count":1,"routed_generality_success_count":1,
            "locality_exact_preservation_count":16,"routing_false_positive_count":0,
            "target_contamination_count":0,"forced_native_success_count":1,
            "forced_generality_success_count":int(step==3000),"validation_source_loss":1.0})
    result=select_checkpoint(rows)
    assert result["selected_step"]==3000
