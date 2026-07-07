from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import yaml

from ...util.hparams import HyperParams


@dataclass
class TIMEEditMultimodalHparams(HyperParams):
    device: Any = 0
    alg: str = "TIME"
    alg_name: str = "TIME"
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
    early_stop_key: str = "loss/time_total_val"
    val_batch_size: int = 1
    val_steps: int = 1
    accumulate_bs: int = 1
    opt: str = "Adam"
    lr: float = 1.0e-5
    edit_lr: float = 1.0e-5
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

    time_target_layer: int = 21
    time_layer_module: Optional[str] = None
    time_rank: int = 4
    time_alpha: float = 0.1
    time_gamma: float = 0.5
    time_tau: float = 1.0
    time_scale_mode: str = "lora_like"
    time_activation: str = "gelu"
    time_token_scope: str = "all"
    time_init_std: float = 1.0e-3
    time_freeze_vlm: bool = True
    time_repository_path: Optional[str] = None
    time_lambda_rel: float = 1.0
    time_lambda_gen: float = 1.0
    time_lambda_loc: float = 1.0
    time_lambda_align: float = 0.5
    time_negative_experts: int = 8
    time_disable_selection: bool = False
    time_disable_score_mixing: bool = False
    time_disable_align_loss: bool = False
    time_topk: int = 0
    time_routing_mode: str = "threshold"
    time_score_norm: str = "none"
    time_align_score_norm: str = "none"
    time_relative_threshold: Optional[float] = None
    time_mixing_mode: str = "softmax"
    time_max_selected_experts: Optional[int] = None
    time_calibration_mode: str = "none"
    time_calibration_beta: float = 0.0
    time_score_pool: str = "token"
    time_anti_collapse_loss: bool = False
    time_lambda_anti_collapse: float = 0.0
    time_anti_collapse_margin: float = 0.05
    time_anti_collapse_score_norm: str = "factor_z"
    time_routing_margin_loss: bool = False
    time_lambda_routing_margin: float = 0.0
    time_routing_margin_current: float = 0.10
    time_routing_margin_prev: float = 0.15
    time_routing_margin_score_norm: str = "factor_z"
    time_lambda_factor_norm_reg: float = 0.0
    time_enable_adaptive_rank_margin_rescue: bool = False
    time_adaptive_rank_margin: float = 0.02
    time_adaptive_rank_margin_use_rank3: bool = True
    time_adaptive_rank_margin_debug: bool = False
    time_force_current_during_training: bool = True
    time_residual_sign: str = "plus"
    time_expert_gain: float = 1.0
    time_reliability_only: bool = False
    time_edit_iters: int = 30
    time_save_state: bool = True
    time_state_dir: Optional[str] = None
    time_generation_mode: str = "normal"
    time_debug: bool = False

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
        if alg not in {"TIME", "TIME_EDIT"}:
            raise ValueError(f"TIMEEditMultimodalHparams expected alg_name=TIME, got {config.get('alg_name')!r}")
        config.setdefault("alg", "TIME")
        config.setdefault("alg_name", "TIME")
        config.setdefault("inner_params", [])
        return cls(**config)
