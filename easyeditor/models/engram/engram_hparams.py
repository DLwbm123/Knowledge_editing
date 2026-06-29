from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import yaml

from ...util.hparams import HyperParams


@dataclass
class EngramMultimodalHparams(HyperParams):
    """Config for forward-only ENGRAM editing in multimodal EasyEdit flows."""

    # Runtime/model fields expected by MultimodalEditor.
    device: Any = 0
    alg: str = "ENGRAM"
    alg_name: str = "ENGRAM"
    name: str = ""
    model_name: str = "blip2"
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

    # Multimodal loader fields.
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

    # ENGRAM extraction and application.
    edit_mode: str = "erase"  # erase | replacement
    alpha: Optional[float] = None
    beta: float = 1.0
    engram_alpha: float = 0.6
    engram_update_direction: str = "subtract"  # subtract | add | auto_nll
    behavior_objective: Optional[str] = None
    auto_sign_probe_alpha: float = 0.01
    auto_sign_lambda_ref: float = 1.0
    auto_sign_min_target_gain: float = 0.0
    engram_layer_scale: str = "uniform"
    engram_token_scope: str = "answer"
    engram_mask_fallback: str = "all"
    token_scope: Optional[str] = None
    target_variants: List[str] = field(default_factory=lambda: ["edit", "rephrase", "image_rephrase"])
    reference_variants: List[str] = field(default_factory=lambda: ["locality_text", "locality_multimodal"])
    engram_target_variants: Optional[List[str]] = None
    engram_reference_variants: Optional[List[str]] = None
    retain_pool_path: Optional[str] = None
    min_reference_examples: int = 0
    skip_if_insufficient_reference: bool = False

    # Conservative defaults: q/k, gate, and multimodal projectors only.
    module_patterns: List[str] = field(
        default_factory=lambda: [
            r"(q_proj|k_proj|gate_proj)$",
            r"(mm_projector|llama_proj|opt_proj)(\.|$)",
        ]
    )
    exclude_module_patterns: List[str] = field(default_factory=lambda: [r"lm_head$", r"down_proj$"])
    engram_module_patterns: Optional[List[str]] = None
    engram_exclude_module_patterns: Optional[List[str]] = None
    module_priority_patterns: List[str] = field(default_factory=list)
    prioritize_module_selection: bool = False
    engram_layers: Optional[List[int]] = None
    engram_max_modules: Optional[int] = None

    # OOM controls.
    covariance_device: str = "cpu"
    solve_device: str = "cpu"
    covariance_dtype: str = "float32"
    max_cov_dim: Optional[int] = None
    skip_if_dim_larger_than: Optional[int] = None
    solve_per_module: bool = True
    clear_cuda_cache: bool = False
    engram_storage_device: Optional[str] = None
    engram_solve_device: Optional[str] = None
    engram_cov_dtype: Optional[str] = None
    engram_max_input_dim: Optional[int] = None

    # Numerics.
    absorb_bias: bool = True
    engram_absorb_bias: Optional[bool] = None
    normalize_covariance: bool = False
    engram_normalize_covariance: Optional[bool] = None
    jitter: float = 0.0
    engram_jitter: Optional[float] = None
    solver: str = "pinv"  # pinv | svd
    rcond: float = 1.0e-6
    engram_rcond: Optional[float] = None
    svd_rank: Optional[int] = None
    energy_threshold: Optional[float] = None
    store_projector: bool = True
    norm_ratio_warn_threshold: float = 0.25
    skip_if_norm_ratio_larger_than: Optional[float] = None

    # Bank/checkpointing.
    bank_dir: Optional[str] = None
    engram_bank_path: Optional[str] = None
    edit_id: Optional[str] = None
    concept_id: Optional[str] = None
    modality: str = "mixed"
    engram_edit_id: Optional[str] = None
    engram_save_stats: bool = True
    sequential_edit: bool = False

    # Replacement mode. This is intentionally conservative and experimental.
    replacement_mode: str = "none"  # none | lora_projected
    project_delta_with_engram: bool = False
    replacement_beta: float = 1.0
    replacement_lambda_ref: float = 0.0
    candidate_delta_source: str = "none"  # none | state_dict_pair | lora_adapter
    base_state_dict_path: Optional[str] = None
    candidate_state_dict_path: Optional[str] = None
    lora_adapter_path: Optional[str] = None
    lora_rank: int = 4
    lora_steps: int = 20
    lora_lr: float = 1.0e-4
    lora_scale: Optional[float] = None
    projector_for_delta: str = "new_target"  # old_target | new_target | reference_aware

    # CURE-MedEdit / CrispEdit-style low-curvature projection.
    use_crisp_projection: bool = False
    crisp_energy_threshold: float = 0.9
    crisp_cache_dataset: str = "reference"  # reference | previous | mixed
    crisp_cache_device: str = "cpu"
    crisp_cache_dtype: str = "float32"
    crisp_cache_update_policy: str = "streaming_average"  # static | streaming_average | ema
    crisp_ema_decay: float = 0.9
    crisp_recompute_policy: str = "never"  # never | on_weight_change | every_edit
    crisp_recalculate_weight_threshold: float = 0.25
    crisp_max_dim: int = 4097

    # Script/dry-run helpers.
    engram_dry_run: bool = False
    dry_run: bool = False
    overlap_threshold: float = 0.35

    def resolved_alpha(self) -> float:
        return float(self.engram_alpha if self.alpha is None else self.alpha)

    def resolved_update_direction(self) -> str:
        from .solver import normalize_update_direction

        return normalize_update_direction(self.engram_update_direction)

    def resolved_direction_sign(self) -> int:
        from .solver import direction_sign_for_update

        return direction_sign_for_update(self.resolved_update_direction())

    def resolved_token_scope(self) -> str:
        return str(self.token_scope or self.engram_token_scope or "all")

    def resolved_target_variants(self) -> List[str]:
        return list(self.engram_target_variants or self.target_variants)

    def resolved_reference_variants(self) -> List[str]:
        return list(self.engram_reference_variants or self.reference_variants)

    def resolved_module_patterns(self) -> List[str]:
        return list(self.engram_module_patterns or self.module_patterns)

    def resolved_exclude_patterns(self) -> List[str]:
        return list(self.engram_exclude_module_patterns or self.exclude_module_patterns)

    def resolved_covariance_device(self) -> str:
        return str(self.engram_storage_device or self.covariance_device)

    def resolved_solve_device(self) -> str:
        return str(self.engram_solve_device or self.solve_device)

    def resolved_covariance_dtype(self) -> str:
        return str(self.engram_cov_dtype or self.covariance_dtype)

    def resolved_absorb_bias(self) -> bool:
        return bool(self.absorb_bias if self.engram_absorb_bias is None else self.engram_absorb_bias)

    def resolved_normalize_covariance(self) -> bool:
        return bool(
            self.normalize_covariance
            if self.engram_normalize_covariance is None
            else self.engram_normalize_covariance
        )

    def resolved_jitter(self) -> float:
        return float(self.jitter if self.engram_jitter is None else self.engram_jitter)

    def resolved_rcond(self) -> float:
        return float(self.rcond if self.engram_rcond is None else self.engram_rcond)

    def resolved_skip_dim(self) -> Optional[int]:
        values = [v for v in [self.skip_if_dim_larger_than, self.max_cov_dim, self.engram_max_input_dim] if v is not None]
        return int(min(values)) if values else None

    def resolved_bank_dir(self) -> Optional[str]:
        return self.bank_dir or self.engram_bank_path

    def resolved_edit_id(self) -> Optional[str]:
        return self.edit_id or self.engram_edit_id

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream) or {}
            config = super().construct_float_from_scientific_notation(config)

        alg = str(config.get("alg_name", config.get("alg", ""))).upper()
        if alg not in {"ENGRAM", "AI_ENGRAM", "AI-ENGRAM"}:
            raise ValueError(f"ENGRAM hparams expected alg_name=ENGRAM, got {config.get('alg_name')!r}")
        config.setdefault("alg", "ENGRAM")
        config.setdefault("alg_name", "ENGRAM")
        config.setdefault("inner_params", [])
        return cls(**config)
