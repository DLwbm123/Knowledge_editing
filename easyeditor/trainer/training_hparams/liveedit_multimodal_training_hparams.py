from dataclasses import dataclass
from typing import Any, List, Optional

import yaml

from ...util.hparams import HyperParams


@dataclass
class LiveEditMultimodalTrainingHparams(HyperParams):
    # Model
    device: Any = 0
    name: str = ""
    model_name: str = "blip2"
    model_class: str = ""
    tokenizer_class: str = ""
    tokenizer_name: str = ""
    inner_params: List[str] = None
    archive: Any = None

    # Training framework
    alg: str = "LiveEdit"
    alg_name: str = "LiveEdit"
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
    early_stop_key: str = "loss/liveedit_total_val"
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

    # Multimodal model assets
    qformer_checkpoint: Optional[str] = None
    qformer_name_or_path: Optional[str] = None
    state_dict_file: Optional[str] = None
    freeze_qformer: bool = True
    pretrained_ckpt: Optional[str] = None
    coco_image: str = ""
    rephrase_image: str = ""

    # LiveEdit
    liveedit_layer: int = 21
    liveedit_module_dim: int = 1024
    liveedit_feature_k: int = 4
    liveedit_rank: int = 4
    liveedit_lora_scale: float = 5.0
    liveedit_cross_att_heads: int = 8
    liveedit_sentinel_tokens: int = 32
    liveedit_similarity: str = "inner_product"
    liveedit_hard_topk: Optional[int] = None
    liveedit_force_topk_when_empty: bool = False
    liveedit_repository_path: Optional[str] = None
    liveedit_freeze_vllm: bool = True
    liveedit_debug: bool = False
    liveedit_rel_weight: float = 1.0
    liveedit_gen_weight: float = 1.0
    liveedit_loc_weight: float = 1.0
    liveedit_route_weight: float = 1.0
    liveedit_hr_weight: float = 1.0
    liveedit_sr1_weight: float = 1.0
    liveedit_sr2_weight: float = 1.0

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        alg = str(config.get("alg", config.get("alg_name", ""))).upper()
        assert alg == "LIVEEDIT" or print(
            f"LiveEditMultimodalTrainingHparams can not load from {hparams_name_or_path}, alg is {config.get('alg')}"
        )
        config.setdefault("alg_name", "LiveEdit")
        config.setdefault("inner_params", [])
        return cls(**config)
