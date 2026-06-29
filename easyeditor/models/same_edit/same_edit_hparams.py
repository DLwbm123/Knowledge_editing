from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import yaml

from ...util.hparams import HyperParams


@dataclass
class SAMEEditMultimodalHparams(HyperParams):
    device: Any = 0
    alg: str = "SAME_EDIT"
    alg_name: str = "SAME_EDIT"
    name: str = ""
    model_name: str = "llava-med"
    model_class: str = ""
    tokenizer_class: str = ""
    tokenizer_name: str = ""
    inner_params: List[str] = field(default_factory=list)
    archive: Any = None
    results_dir: str = "./results"
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    eval_only: bool = True
    half: bool = False
    save: bool = False
    verbose: bool = True
    debug: bool = False
    silent: bool = False
    train_base: bool = False
    dropout: Optional[float] = 0.0
    no_grad_layers: Any = None
    model_save_pt: int = 5000
    max_iters: Optional[int] = 0
    max_epochs: Optional[int] = None
    log_interval: int = 1
    eval_log_interval: int = 1
    final_eval: bool = True
    val_interval: int = 5000
    early_stop_patience: int = 20000
    early_stop_key: str = "loss/same_edit_total_val"
    val_batch_size: int = 1
    val_steps: int = 1
    accumulate_bs: int = 1
    opt: str = "Adam"
    lr: float = 1.0e-4
    edit_lr: float = 1.0e-4
    lr_lr: float = 1.0e-4
    lr_scale: float = 1.0
    seed: int = 42
    cedit: float = 1.0
    iedit: float = 0.0
    cloc: float = 0.0
    cbase: float = 0.0
    grad_clip: float = 100.0

    qformer_checkpoint: Optional[str] = None
    qformer_name_or_path: Optional[str] = None
    state_dict_file: Optional[str] = None
    freeze_qformer: bool = True
    pretrained_ckpt: Optional[str] = None
    coco_image: str = ""
    rephrase_image: str = ""
    llava_med_vision_tower: Optional[str] = None
    llava_med_model_name: str = "llava-med-v1.5-mistral-7b"
    llava_med_loader_source: str = "third_party/LLaVA-Med"
    llava_med_conversation_template: str = "mistral_instruct"
    llava_med_dtype: str = "float16"

    # PRISM/SAME-style MoE-LoRA knobs.
    same_edit_lora_r: int = 8
    same_edit_lora_alpha: float = 16.0
    same_edit_lora_dropout: float = 0.0
    same_edit_expert_num: int = 4
    same_edit_top_k: int = 1
    same_edit_current_edit: int = 0
    same_edit_oracle_edit_routing: bool = True
    same_edit_eval_oracle_edit_routing: bool = False
    same_edit_learned_hidden_routing: bool = True
    same_edit_adaptive_activation: bool = False
    same_edit_tau_score: float = 0.1
    same_edit_curvature_mode: str = "off"
    same_edit_curvature_mu: float = 0.9
    same_edit_curvature_max_grad_ratio: float = 10.0
    same_edit_allow_missing_covariance: bool = True
    same_edit_spectral_router: bool = False
    same_edit_router_start_step: int = 10
    same_edit_router_scaling_factor: float = 0.2
    same_edit_window_size: int = 3
    same_edit_max_components: int = 64
    same_edit_cumulative_energy_ratio: float = 0.9
    same_edit_target_modules: str = "last4_down_proj"
    same_edit_target_module_patterns: List[str] = field(default_factory=list)
    same_edit_target_last_n_layers: int = 4
    same_edit_exclude_vision_tower: bool = True
    same_edit_exclude_module_path_segments: List[str] = field(default_factory=list)
    same_edit_update_covariance: bool = True
    same_edit_adapter_dtype: str = "float32"

    # Loss/state policy.
    same_edit_loss: str = "edit_nll"
    same_edit_use_rephrase_loss: bool = False
    same_edit_use_locality_kl: bool = False
    same_edit_route_loss_weight: float = 0.0
    same_edit_reference_nll_preservation: bool = False
    same_edit_routing_contrastive_loss: bool = False
    same_edit_num_steps: int = 20
    same_edit_save_state: bool = True
    same_edit_state_dir: Optional[str] = None
    same_edit_sequential_edit: bool = False

    def resolved_alg(self) -> str:
        return str(self.alg_name or self.alg).upper().replace("-", "_")

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream) or {}
            config = super().construct_float_from_scientific_notation(config)
        alg = str(config.get("alg_name", config.get("alg", ""))).upper().replace("-", "_")
        if alg not in {"SAME_EDIT", "SAMEEDIT", "MOE_SAME_EDIT", "STABILIZED_MOE_EDIT"}:
            raise ValueError(f"SAMEEditMultimodalHparams expected alg_name=SAME_EDIT, got {config.get('alg_name')!r}")
        config.setdefault("alg", "SAME_EDIT")
        config.setdefault("alg_name", "SAME_EDIT")
        config.setdefault("inner_params", [])
        return cls(**config)
