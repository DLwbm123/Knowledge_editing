from dataclasses import dataclass
from typing import Any, List, Optional

import yaml

from ...util.hparams import HyperParams


@dataclass
class DSCAMultimodalTrainingHparams(HyperParams):
    device: Any = 0
    name: str = ""
    model_name: str = "blip2"
    model_class: str = ""
    tokenizer_class: str = ""
    tokenizer_name: str = ""
    inner_params: List[str] = None
    archive: Any = None

    alg: str = "DSCA"
    alg_name: str = "DSCA"
    lr: float = 1.0e-4
    edit_lr: float = 1.0e-4
    lr_lr: float = 1.0e-4
    seed: int = 42
    debug: bool = False
    cedit: float = 1.0
    iedit: float = 1.0
    cloc: float = 1.0
    cbase: float = 0.0
    dropout: Optional[float] = 0.0
    train_base: bool = False
    no_grad_layers: Any = None
    results_dir: str = "./results"
    batch_size: int = 1
    model_save_pt: int = 5000
    silent: bool = False
    log_interval: int = 100
    eval_log_interval: int = 1000
    final_eval: bool = True
    val_interval: int = 5000
    early_stop_patience: int = 20000
    early_stop_key: str = "loss/dsca_total_val"
    eval_only: bool = False
    half: bool = False
    save: bool = False
    verbose: bool = True
    val_batch_size: int = 1
    accumulate_bs: int = 1
    val_steps: int = 500
    opt: str = "Adam"
    grad_clip: float = 100.0
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None
    model_parallel: bool = False

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

    dsca_layer: int = 21
    dsca_layer_module: Optional[str] = None
    dsca_freeze_vlm: bool = True
    dsca_rank: int = 128
    dsca_gate_bottleneck: int = 64
    dsca_min_samples: int = 32
    dsca_refine_interval: int = 500
    dsca_cluster_alpha: float = 2.0
    dsca_proto_ema: float = 0.95
    dsca_tau_visual: float = 0.0
    dsca_route_temperature: float = 0.07
    dsca_distill_temperature: float = 0.07
    dsca_lambda_align: float = 0.5
    dsca_lambda_distill: float = 1.0
    dsca_lambda_sparse: float = 1.0e-2
    dsca_task_weight: float = 1.0
    dsca_update_clusters_during_training: bool = True
    dsca_update_clusters_during_inference: bool = False
    dsca_freeze_repository_at_eval: bool = True
    dsca_require_masks: bool = True
    dsca_candidate_topk: Optional[int] = None
    dsca_residual_scale: float = 1.0
    dsca_residual_apply_mask: str = "attention"
    dsca_generation_mode: str = "normal"
    dsca_generation_residual_apply_mask: str = "attention"
    dsca_generation_reuse_prefill_route: bool = False
    dsca_generation_update_repository: bool = False
    dsca_repository_path: Optional[str] = None
    dsca_dsam_init_std: float = 0.02
    dsca_max_buffer_size: Optional[int] = None
    dsca_debug: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        alg = str(config.get("alg", config.get("alg_name", ""))).upper()
        assert alg == "DSCA" or print(f"DSCAMultimodalTrainingHparams can not load {hparams_name_or_path}")
        config.setdefault("alg", "DSCA")
        config.setdefault("alg_name", "DSCA")
        config.setdefault("inner_params", [])
        return cls(**config)
