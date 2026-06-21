# DSCA Stage 1 Audit

## Scope

This Stage 1 implementation adds a framework-level DSCA editor for BLIP2-OPT and MiniGPT-4 in the EasyEdit multimodal training/editing flow. It is not a paper-table reproduction and does not implement the full continual evaluation protocol from "DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing".

Unsupported backbones, including LLaVA and PaliGemma, raise `NotImplementedError` until this repository exposes verified full-sequence region masks for them.

## Files Added Or Modified

Added:

- `easyeditor/trainer/algs/dsca_utils.py`
- `easyeditor/trainer/algs/dsca.py`
- `easyeditor/trainer/training_hparams/dsca_multimodal_training_hparams.py`
- `easyeditor/models/dsca/`
- `hparams/TRAINING/DSCA/{blip2,minigpt4}.yaml`
- `hparams/TRAINING/DSCA/blip2_stage1_smoke.yaml`
- `hparams/TRAINING/DSCA/blip2_20edit_pilot.yaml`
- `hparams/DSCA/{blip2,minigpt4}.yaml`
- `hparams/DSCA/blip2_20edit_pilot.yaml`
- `tests/test_dsca.py`
- `tests/test_dsca_real_smoke.py`
- `scripts/smoke_dsca_stage1.py`
- `scripts/run_dsca_20edit_pilot.py`
- `docs/DSCA_PHASE1_AUDIT.md`

Modified:

- `easyeditor/trainer/algs/__init__.py`
- `easyeditor/trainer/training_hparams/__init__.py`
- `easyeditor/models/__init__.py`
- `easyeditor/util/alg_train_dict.py`
- `easyeditor/util/alg_dict.py`
- `easyeditor/trainer/MultimodalTrainer.py`
- `docs/NEW_METHODS_FROM_PAPERS.md`

## Paper Component Mapping

- Online semantic partitioning: `DSCAConceptRepository.assign_or_create` creates or updates concept clusters using nearest fused prototype distance and a dynamic `mu + alpha * sigma` novelty threshold.
- Concept prototypes: each cluster stores non-gradient fused and visual prototypes `p_f` and `p_v`.
- Dynamic threshold: distance mean and variance are maintained with Welford-style stats and used for cluster novelty.
- DSAM: each concept cluster owns one `DSAModule` with trainable coordinate target parameters and a gated residual branch.
- PCA / residualized PCA: `initialize_basis_if_ready` computes a torch SVD basis from the cluster buffer after `dsca_min_samples`; later clusters residualize against earlier active subspaces first.
- Two-stage routing: `dsca_route` filters by visual cosine threshold, then softmaxes fused cosine scores over candidates only.
- Gated residual intervention: the layer hook adds `sum_k w_k * Psi_k(H_layer)` and zeroes padding positions according to `dsca_residual_apply_mask`.
- `L_task`: answer-token causal LM CE through the existing EasyEdit edit loss.
- `L_align`: edit-sample fused representation aligned to the detached base/text anchor.
- `L_cdistill`: replay InfoNCE between edited replay fused features and detached base replay fused features.
- `L_sparse`: L1 routing-weight penalty on replay routing weights.

## Masks And Backbones

DSCA reuses the LiveEdit-style full-sequence masks already exposed by BLIP2 and MiniGPT-4:

- `h_v`: masked mean over `vision_mask`.
- `h_t`: masked mean over `prompt_mask`.
- `h_f`: masked mean over `vision_mask | prompt_mask`.

`answer_mask` is used for task loss when available, but answer tokens and padding tokens are excluded from routing features, cluster buffers, and PCA features. During inference, `answer_mask` may be absent because routing only requires `vision_mask`, `prompt_mask`, and `attention_mask`.

Hook paths:

- BLIP2-OPT: `opt_model.model.decoder.layers.{dsca_layer}`
- MiniGPT-4: `llama_model.model.layers.{dsca_layer}`
- Explicit override: `dsca_layer_module`

If the layer path cannot be resolved, DSCA raises `ValueError`.

## Identity And Freezing Behavior

- Empty repository is an exact no-op.
- No visual candidates is an exact no-op.
- Inactive DSAM clusters are an exact no-op.
- Base VLM parameters are frozen by default with `dsca_freeze_vlm: true`.
- Cluster prototypes and subspace bases are buffers, not trainable parameters.
- DSAM `W`, `b`, and gate parameters are trainable once the cluster exists.

## Approximations And Stage 2 TODOs

- Stage 1 uses buffered residualized PCA, not true streaming Incremental PCA.
- Stage 1 applies sequence-level hidden-state residuals from vector-level concept routing.
- LLaVA-1.5 and PaliGemma adapters are deferred.
- CoIN continual learning protocol, 1000 sequential-edit evaluation, and BWT/FWT/ACC/At metrics are deferred.
- Hyperparameters are defaults for framework integration, not tuned paper reproduction values.

## Verification

Commands:

```bash
python3 -m py_compile \
  easyeditor/trainer/algs/dsca_utils.py \
  easyeditor/trainer/algs/dsca.py \
  easyeditor/trainer/blip2_models/blip2_opt.py \
  easyeditor/trainer/blip2_models/mini_gpt4.py \
  easyeditor/trainer/MultimodalTrainer.py

python3 -m pytest -q tests/test_dsca.py tests/test_liveedit.py tests/test_asam.py
```

Result:

```text
py_compile passed
67 passed, 3 warnings
dsca registration smoke ok
```

The warnings are existing `timm` deprecation warnings from package imports.

## Server Real-Backbone Validation

Date: 2026-06-16

Remote target: `/remote-home/wangbomin/Knowledge_editing` on `my-gpu`.

Final log directory:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_server_validation/20260616_024018
```

Setup:

- Synced the local checkout to `/remote-home/wangbomin/Knowledge_editing` with `tar | ssh`, excluding `hugging_cache/`, `.pytest_cache/`, `__pycache__/`, `outputs/`, `results/`, and `.DS_Store`.
- Created `hugging_cache -> /remote-home/wangbomin/hugging_cache`.
- Used `/root/anaconda3/bin/python`.
- Exported `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONPATH=$PWD:$PYTHONPATH`, and `CUDA_VISIBLE_DEVICES=1`.
- Installed missing Python runtime packages on the remote environment: `higher`, `iopath`, `fairscale`, `icecream`, `sentence-transformers`, `openai`, and `peft`.
- Did not download or modify model weights.

Commands captured in the log directory:

```bash
python -m py_compile \
  easyeditor/trainer/algs/dsca_utils.py \
  easyeditor/trainer/algs/dsca.py \
  easyeditor/trainer/algs/liveedit_utils.py \
  easyeditor/trainer/algs/liveedit.py \
  easyeditor/trainer/algs/asam_utils.py \
  easyeditor/trainer/blip2_models/blip2_opt.py \
  easyeditor/trainer/blip2_models/mini_gpt4.py \
  easyeditor/trainer/MultimodalTrainer.py \
  scripts/smoke_dsca_stage1.py \
  tests/test_dsca_real_smoke.py

python -m pytest -q tests/test_dsca.py tests/test_liveedit.py tests/test_asam.py

RUN_DSCA_REAL_SMOKE=1 python -m pytest -q tests/test_dsca_real_smoke.py -k blip2

python scripts/smoke_dsca_stage1.py \
  --model blip2 \
  --hparams hparams/TRAINING/DSCA/blip2_stage1_smoke.yaml \
  --steps 3 \
  --num_edit_samples 3 \
  --num_replay_samples 1 \
  --min_samples 2 \
  --rank 4 \
  --device cuda \
  --log_dir /remote-home/wangbomin/Knowledge_editing/outputs/dsca_server_validation/20260616_024018
```

Status:

```text
nvidia_smi_start        passed
model_cache_check       passed
dependency_check        passed; sentencepiece missing
py_compile              passed
unit_tests              passed: 67 passed, 6 warnings
dsca_registration_smoke passed
validate_dsca_stage1    skipped: scripts/validate_dsca_stage1.py not present
blip2_real_smoke_pytest passed: 1 passed, 1 deselected
blip2_dsca_3step_smoke  passed
minigpt4_dependency     skipped by pytest because sentencepiece is unavailable
blip2_20edit_pilot      skipped: no standard 20-edit pilot command in this checkout
```

BLIP2 3-step smoke metrics:

```text
step 1: clusters=1, active_dsams=0, finite loss=4.2320, base_changed=False
step 2: clusters=1, active_dsams=1, DSAM grad mean=6.5485, R_k_requires_grad=False, repository_round_trip_ok=True
step 3: clusters=1, active_dsams=1, DSAM grad mean=2.5488, base_changed=False, duplicate_optimizer_param_groups=False
```

Artifacts:

- `env.txt`
- `nvidia_smi_start.txt`
- `model_cache_check.txt`
- `dependency_check.txt`
- `py_compile.log`
- `unit_tests.log`
- `dsca_registration_smoke.log`
- `validate_dsca_stage1.log`
- `blip2_real_smoke_pytest.log`
- `blip2_dsca_3step_smoke.log`
- `blip2_dsca_3step_smoke_metrics.csv`
- `minigpt4_dependency_smoke.log`
- `summary.txt`

Next recommended experiment: run a BLIP2 20-edit pilot with a real evaluation dataset and an explicit pilot script/config, reusing the same offline cache and `CUDA_VISIBLE_DEVICES=1`. MiniGPT-4 should remain skipped until `sentencepiece` is available and the local Vicuna tokenizer can be loaded cleanly.

## Medical Backbone Transition After BLIP2 Decoded-Output Failure

Date: 2026-06-16

Remote target: `/remote-home/wangbomin/Knowledge_editing` on `my-gpu`.

BLIP2 MedMKEB status:

- The MedMKEB 20-edit pilot completed and DSCA engineering safety checks passed: base VLM unchanged, `R_k` non-gradient, no duplicate optimizer params, repository save/load OK, and finite losses.
- Decoded editing efficacy failed: Rel., T-Gen., and M-Gen. were all `0.0`; edited free predictions contained the target in `0/20` edit samples.
- Failure triage showed label/mask integrity passed, mean target NLL improved from `10.7915` to `10.4442`, and target NLL improved on `16/20` samples.
- One-edit task-only forced-route diagnostics reached near-zero teacher-forced target NLL on `3/3` samples, but decoded predictions still did not contain targets.
- Generation-path diagnostics showed DSCA generation hook is active, generation residual is nonzero, answer-residual dominance is `0.0`, prompt-only first-token rank improves on only `6/20` samples, and force-route improves `0/20`.

Conclusion: this is not a generation-hook bypass, not answer-token leakage, and not a pure routing failure. The current BLIP2-OPT bottleneck is weak decoded logit shift and likely medical-domain mismatch, so BLIP2 hyperparameter tuning remains secondary until medical-backbone feasibility is resolved.

Medical VLM asset check:

- LLaVA code exists in the checkout at `easyeditor/trainer/llava`.
- Local cache contains Vicuna base weights at `/remote-home/wangbomin/hugging_cache/vicuna-7b`.
- No local LLaVA-Med checkpoint was found.
- No local generic LLaVA checkpoint such as `/remote-home/wangbomin/hugging_cache/llava-v1.5-7b` was found.
- No local CLIP vision tower such as `/remote-home/wangbomin/hugging_cache/clip-vit-large-patch14-336` was found.
- No `mm_projector.bin`, `non_lora_trainables.bin`, or full checkpoint containing projector weights was found.
- `sentencepiece` is missing from `/root/anaconda3/bin/python`, so Vicuna/LLaMA tokenizers cannot currently load.

Asset probe artifacts:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/asset_search_remote.txt
/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/asset_search_cache.txt
/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/dependency_check.txt
/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/medical_vlm_asset_report.json
/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/medical_vlm_missing_assets_report.md
```

LLaVA-Med adapter status:

- Not implemented in this stage because runnable LLaVA-Med or generic LLaVA assets are absent.
- No smoke test or 20-edit run was executed; running a fake medical VLM experiment was intentionally avoided.

Required upload checklist before continuing:

- A LLaVA-Med checkpoint compatible with the repo LLaVA loader, including config/tokenizer files.
- CLIP vision tower weights/config used by the LLaVA-Med checkpoint.
- Multimodal projector weights if the checkpoint is not fully merged.
- A Python 3.12 Linux x86_64 `sentencepiece` wheel, for example `sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` or a newer compatible cp312 manylinux wheel.

Exact upload commands:

```bash
rsync -avP /local/path/llava-med/ my-gpu:/remote-home/wangbomin/hugging_cache/llava-med/
rsync -avP /local/path/clip-vit-large-patch14-336/ my-gpu:/remote-home/wangbomin/hugging_cache/clip-vit-large-patch14-336/
ssh my-gpu "mkdir -p /remote-home/wangbomin/wheels"
rsync -avP /local/path/sentencepiece-*-cp312-*-linux*.whl my-gpu:/remote-home/wangbomin/wheels/
ssh my-gpu "/root/anaconda3/bin/python -m pip install --no-index /remote-home/wangbomin/wheels/sentencepiece-*-cp312-*-linux*.whl"
```

Next command after upload:

```bash
cd /remote-home/wangbomin/Knowledge_editing
export HF_HOME=/remote-home/wangbomin/hugging_cache
export TRANSFORMERS_CACHE=/remote-home/wangbomin/hugging_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

/root/anaconda3/bin/python scripts/smoke_dsca_medical_vlm_stage1.py \
  --model llava-med \
  --dataset MEDMKEB \
  --image-root /remote-home/wangbomin/Knowledge_editing/datasets/MedMKEB/images \
  --hparams hparams/DSCA/llava_med_20edit.yaml \
  --training-hparams hparams/TRAINING/DSCA/llava_med_20edit.yaml \
  --num-edit-samples 4 \
  --num-replay-samples 4 \
  --steps 3 \
  --rank 4 \
  --min-samples 2 \
  --device cuda \
  --output-dir outputs/dsca_medmkeb_llava_med_smoke/$(date +%Y%m%d_%H%M%S)
```

## BLIP2 20-Edit Pilot

Date: 2026-06-16

Remote target: `/remote-home/wangbomin/Knowledge_editing` on `my-gpu`.

Output directory:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_20edit_pilot/20260616_030426
```

Added reusable runner and hparams:

- `scripts/run_dsca_20edit_pilot.py`
- `hparams/TRAINING/DSCA/blip2_20edit_pilot.yaml`
- `hparams/DSCA/blip2_20edit_pilot.yaml`

Command:

```bash
python scripts/run_dsca_20edit_pilot.py \
  --model blip2 \
  --dataset E-VQA \
  --hparams hparams/DSCA/blip2_20edit_pilot.yaml \
  --training-hparams hparams/TRAINING/DSCA/blip2_20edit_pilot.yaml \
  --num-edits 20 \
  --batch-size 1 \
  --rank 8 \
  --min-samples 4 \
  --refine-interval 10 \
  --seed 42 \
  --device cuda \
  --output-dir outputs/dsca_20edit_pilot/20260616_030426
```

Validation before the pilot:

```text
py_compile passed
tests/test_dsca.py tests/test_liveedit.py tests/test_asam.py: 67 passed, 6 warnings
```

Pilot status: blocked before model loading because the real E-VQA/VLKEB data is absent. No synthetic data was used and no final editing metrics were claimed.

Missing paths reported by the runner:

```text
/remote-home/wangbomin/Knowledge_editing/datasets/eval.json
/remote-home/wangbomin/Knowledge_editing/datasets/eval_multihop.json
/remote-home/wangbomin/Knowledge_editing/datasets/E-VQA/eval.json
/remote-home/wangbomin/Knowledge_editing/datasets/EVQA/eval.json
/remote-home/wangbomin/Knowledge_editing/datasets/VLKEB/eval.json
/remote-home/wangbomin/Knowledge_editing/data/E-VQA/eval.json
/remote-home/wangbomin/Knowledge_editing/data/EVQA/eval.json
/remote-home/wangbomin/Knowledge_editing/data/VLKEB/eval.json
```

Required image root also missing:

```text
/remote-home/wangbomin/Knowledge_editing/datasets/VLKEB_images/mmkb_images
```

Artifacts written:

- `env.txt`
- `py_compile.log`
- `unit_tests.log`
- `blip2_dsca_20edit_pilot.log`
- `config_resolved.yaml`
- `dataset_discovery.json`
- `dataset_error.txt`
- `final_summary.json`
- `status.txt`

Final metrics: not available because `pilot_ran=false`.

Repository stats and DSCA diagnostics: not available because no real edit samples were loaded.

Safety checks: compile/unit gates passed; the pilot stopped before BLIP2 loading and before DSCA updates because dataset validation failed.

Next recommended command after placing VLKEB/E-VQA data under `datasets/`:

```bash
cd /remote-home/wangbomin/Knowledge_editing
export HF_HOME=/remote-home/wangbomin/hugging_cache
export TRANSFORMERS_CACHE=/remote-home/wangbomin/hugging_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PWD:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1
RUN_ID=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=outputs/dsca_20edit_pilot/${RUN_ID}
mkdir -p "$OUT_DIR"
/root/anaconda3/bin/python scripts/run_dsca_20edit_pilot.py \
  --model blip2 \
  --dataset E-VQA \
  --hparams hparams/DSCA/blip2_20edit_pilot.yaml \
  --training-hparams hparams/TRAINING/DSCA/blip2_20edit_pilot.yaml \
  --num-edits 20 \
  --batch-size 1 \
  --rank 8 \
  --min-samples 4 \
  --refine-interval 10 \
  --seed 42 \
  --device cuda \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/blip2_dsca_20edit_pilot.log"
```

## Local MedMKEB Data Reuse

Date: 2026-06-16

A local medical editing dataset was found before any dataset download was needed:

```text
/Volumes/DataP/knowledge_editing/data/medmkeb/raw/eval_data_attack.json
/Volumes/DataP/knowledge_editing/data/medmkeb/raw/eval_data_threehop_final.json
/Volumes/DataP/knowledge_editing/data/medmkeb/raw/train_data.json
/Volumes/DataP/knowledge_editing/data/medmkeb/images/
```

Existing local asset report for `eval_data_attack.json`:

```text
records: 721
image references: 2163
resolved image references: 2163
missing image references: 0
```

A real 20-edit MedMKEB subset was prepared locally without synthetic images:

```text
outputs/dsca_medmkeb_20_dataset/MedMKEB/eval.json
outputs/dsca_medmkeb_20_dataset/MedMKEB/images/
records: 20
image files: 60
local package size: about 155 MB
```

The subset was uploaded to:

```text
/remote-home/wangbomin/Knowledge_editing/datasets/MedMKEB/eval.json
/remote-home/wangbomin/Knowledge_editing/datasets/MedMKEB/images/
```

Runner changes:

- `scripts/run_dsca_20edit_pilot.py` now accepts `--dataset MEDMKEB`.
- The runner now accepts `--image-root` and `--rephrase-image-root` so hparams do not need to be patched for prepared datasets.
- `--base-signature-check final|every-step|off` was added because per-step full BLIP2 base-parameter scans are too expensive for pilot timing.
- `--phase-timing` was added to print per-step phase diagnostics.

Remote dry-run command:

```bash
cd /remote-home/wangbomin/Knowledge_editing
HF_HOME=/remote-home/wangbomin/hugging_cache \
TRANSFORMERS_CACHE=/remote-home/wangbomin/hugging_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=$PWD:$PYTHONPATH \
/root/anaconda3/bin/python scripts/run_dsca_20edit_pilot.py \
  --dataset MEDMKEB \
  --dataset-path datasets/MedMKEB/eval.json \
  --image-root datasets/MedMKEB \
  --num-edits 20 \
  --output-dir outputs/dsca_server_validation/medmkeb_dryrun_20260616_032449 \
  --dry-run
```

Dry-run result:

```text
Dry run passed.
dataset_path=/remote-home/wangbomin/Knowledge_editing/datasets/MedMKEB/eval.json
```

Remote artifact directory:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_server_validation/medmkeb_dryrun_20260616_032449
```

BLIP2 real-pilot status:

```text
model loaded: yes
DSCA hook: opt_model.model.decoder.layers.21
empty repository identity before first edit: passed
repository_step_000.pt: written
repository_step_010.pt: written and load-checked
base_param_delta_norm through completed steps: 0.0
R_k_requires_grad_any through completed diagnostics: false
duplicate optimizer param groups through completed diagnostics: false
```

The full 20-edit run was not completed. It was stopped after repeated long-tail behavior in `DSCA.edit_step`.

Evidence from phase timing:

```text
output dir:
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_server_validation/medmkeb_phase_timing_13_20260616_041052

step 10 edit_step: 1.7417 sec
step 10 repository_save_load: 0.0941 sec
step 10 evaluation phases: about 0.09-0.30 sec each

step 11 sample_batch: 0.0064 sec
step 11 validate_masks: 0.1486 sec
step 11 edit_step: 439.9297 sec
step 11 optimizer_step: 0.0036 sec
step 11 evaluation phases: about 0.09-0.31 sec each

step 12 edit_step: 0.8214 sec
step 13 sample_batch: 0.0077 sec
step 13 validate_masks: 0.1441 sec
step 13 stopped while inside edit_step
```

Partial run artifacts:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_server_validation/medmkeb_20edit_20260616_032603
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_server_validation/medmkeb_20edit_full_finalcheck_20260616_035356
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_server_validation/medmkeb_phase_timing_13_20260616_041052
```

Current conclusion:

- The local medical editing data exists and is usable; no VLKEB/E-VQA dataset download is needed for this MedMKEB pilot.
- The remote dataset package is valid and answer/image fields load under the existing BLIP2 `VQADataset`.
- The BLIP2 backbone and DSCA hook load from the offline cache.
- The 20-edit acceptance gate remains blocked by an intermittent but repeatable `DSCA.edit_step` long-tail after active DSAMs are present.

Next recommended experiment:

Instrument `DSCA.edit_step` internally around `_update_clusters_from_batch`, teacher capture, replay capture, edited forward, replay forward, loss construction, and `safe_backward`. Run the same MedMKEB 13-edit command until step 11 to identify the exact internal subphase before attempting the full 20-edit gate again.

## MedMKEB Edit-Step Profiling And Patch

Date: 2026-06-16

Remote repo:

```text
/remote-home/wangbomin/Knowledge_editing
```

Data source:

```text
/remote-home/wangbomin/Knowledge_editing/datasets/MedMKEB/eval.json
/remote-home/wangbomin/Knowledge_editing/datasets/MedMKEB/images/...
```

No data or model weights were downloaded. The run used the existing offline model cache at `/remote-home/wangbomin/hugging_cache`.

Instrumentation added:

- `DSCAPhaseTimer` in `easyeditor/trainer/algs/dsca_utils.py`, enabled only by profile flags.
- Internal `DSCA.edit_step` phase timing for cluster update, PCA/refine, teacher forwards, edited forwards, routing/residual injection, losses, backward, and diagnostics.
- Runner flags in `scripts/run_dsca_20edit_pilot.py` for `--profile-edit-step`, profile step ranges, timeout traceback dumps, base-signature modes, and diagnostic ablations.
- `scripts/summarize_dsca_profile.py` for JSONL profile summaries.

Pre-patch root cause:

```text
profile dir:
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_medmkeb_profile/20260616_071636_full_signature

step 11 residualized_pca: 214.4390 sec
traceback:
  dsca_utils.py: orthonormalize_rows
  dsca_utils.py: pca_basis
  dsca_utils.py: residualized_pca_basis
  dsca_utils.py: initialize_basis_if_ready
  dsca_utils.py: assign_batch
  dsca.py: _update_clusters_from_batch
  dsca.py: edit_step
```

The first bug was repeated basis initialization for already-active clusters during ordinary `assign_batch` updates. Those updates should append to the PCA buffer but should not reinitialize an active DSAM unless the scheduled refine path explicitly uses `force=True`.

Patch 1:

- `DSCAConceptRepository.initialize_basis_if_ready()` now returns early for `already_active and not force`.
- Scheduled `refine_subspaces(force=True)` remains enabled.

Patch 1 validation:

```text
profile dir:
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_medmkeb_profile/20260616_072229_patched_14step

step 11 edit_step: 0.8705 sec
profile bottleneck: edit_step_total
residualized_pca total: 0.0115 sec
EXIT_STATUS: 0
```

A subsequent 20-edit attempt exposed the same CUDA small-matrix PCA/orthonormalization slow path during scheduled step-20 refine:

```text
profile dir:
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_medmkeb_profile/20260616_072823_patched_step20_refine

step 20 refine_subspaces: triggered
step 20 residualized_pca started with buffer_shape=[15, 2560]
timeout traceback again pointed at the basis/PCA path
```

Patch 2:

- `pca_basis()` now computes PCA on CPU float32 and returns the basis to the original device/dtype.
- `residualized_pca_basis()` moves small feature/basis tensors to CPU float32 for the residualized PCA solve and returns to the original device/dtype.
- `orthonormalize_rows()` now performs Gram-Schmidt on CPU float32, returns to the original device/dtype, and skips identity fallback construction when rank is already filled.

This keeps the same DSCA basis semantics but avoids CUDA launch overhead and pathological slowdowns for small `N x 2560` PCA buffers.

Verification:

```bash
python3 -m py_compile \
  easyeditor/trainer/algs/dsca_utils.py \
  easyeditor/trainer/algs/dsca.py \
  scripts/run_dsca_20edit_pilot.py \
  scripts/summarize_dsca_profile.py

python3 -m pytest -q tests/test_dsca.py
```

Local result:

```text
22 passed, 3 warnings
```

Remote compile:

```bash
cd /remote-home/wangbomin/Knowledge_editing
/root/anaconda3/bin/python -m py_compile \
  easyeditor/trainer/algs/dsca_utils.py \
  easyeditor/trainer/algs/dsca.py \
  scripts/run_dsca_20edit_pilot.py \
  scripts/summarize_dsca_profile.py
```

Final 14-step validation:

```text
profile dir:
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_medmkeb_profile/20260616_073217_cpu_pca_14step

EXIT_STATUS: 0
completed_edits: 14
step 11 edit_step: 0.8728 sec
step 13 edit_step: 0.9011 sec
profile max edit_step_total: 1.2724 sec
residualized_pca: 0.0138 sec
set_basis: 0.0019 sec
base_vlm_params_changed: false
R_k_requires_grad_any: false
duplicate_optimizer_param_groups: false
loss_finite: true
repository_save_load: true
```

Final 20-edit validation:

```text
output dir:
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_medmkeb_20edit/20260616_073357_cpu_pca_final

EXIT_STATUS: 0
completed_edits: 20
total_runtime_sec: 48.2509
time_per_edit_sec: 2.4125
step 20 edit_step: 1.3218 sec
profile max edit_step_total: 1.3131 sec
step 20 refine_subspaces: 0.0537 sec
step 20 residualized_pca: 0.0233 sec
step 20 set_basis: 0.0071 sec
final_repository_size: 3
final_active_dsam_count: 1
final_base_param_delta_norm: 0.0
base_vlm_params_changed: false
R_k_requires_grad_any: false
duplicate_optimizer_param_groups: false
loss_finite: true
repository_save_load: true
peak_gpu_memory_mb: 9536.5849
```

Final metrics for this 20-edit smoke:

```text
rel: 0.0
t_gen: 0.0
m_gen: 0.0
t_loc: NaN
m_loc: 1.0
avg: 0.25
```

Current conclusion:

- The MedMKEB BLIP2 20-edit pilot now completes on the remote GPU server.
- The previous step-11/13 slowdown was caused by repeated active-cluster basis recomputation on the CUDA PCA/orthonormalization path.
- The later step-20 scheduled-refine slowdown was caused by the same CUDA small-matrix basis path, now moved to CPU float32.
- DSCA base-VLM freezing, non-gradient `R_k`, optimizer param-group uniqueness, finite losses, and repository save/load checks passed.

Next recommended experiment:

Run the same 20-edit command without internal profiling for a clean timing baseline, then run a larger MedMKEB pilot only if the no-profile timing remains stable and the desired metric behavior is acceptable.

## Medical VLM Asset Preparation

Date: 2026-06-16

Remote target: `/remote-home/wangbomin/Knowledge_editing` on `my-gpu`.

Boundary:

- No unverified LLaVA-Med adapter was implemented.
- DSCA core, ASAM, and LiveEdit were not changed.
- No model weights were downloaded.
- No fake medical VLM experiment was run.

SentencePiece:

- Remote Python: `/root/anaconda3/bin/python`
- Python version: `3.12.7`
- Compatible wheel tag: `cp312-cp312-manylinux2014_x86_64` and newer manylinux x86_64 tags.
- Installed package: `sentencepiece 0.2.1`
- Install source: the remote environment's configured PyPI mirror.
- Verification artifact: `/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/sentencepiece_install_report.md`

MiniGPT-4 generic control smoke:

- Vicuna tokenizer loaded from `/remote-home/wangbomin/hugging_cache/vicuna-7b` with `local_files_only=True`.
- Vicuna config loaded from the same local cache path.
- Local MiniGPT-4 assets were found:
  - `/remote-home/wangbomin/hugging_cache/pretrained_minigpt4_7b.pth`
  - `/remote-home/wangbomin/hugging_cache/eva_vit_g.pth`
  - `/remote-home/wangbomin/hugging_cache/blip2_pretrained_flant5xxl.pth`
  - `/remote-home/wangbomin/hugging_cache/bert-base-uncased`
- The repository `MiniGPT4` class imported successfully.
- Full MiniGPT-4 instantiation and generation were skipped because loading Vicuna-7B plus EVA/QFormer is not a cheap smoke.
- Result: generic VLM control available for tokenizer/config and local asset preflight, but it is not a medical VLM.
- Verification artifact: `/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/minigpt4_control_smoke_report.md`

Medical LLaVA-Med asset status:

- Vicuna base is present, but Vicuna alone is not a multimodal model.
- No verified LLaVA-Med checkpoint directory is present.
- No verified generic LLaVA checkpoint directory is present.
- No matching CLIP/EVA/SigLIP vision tower directory for LLaVA-Med is present.
- No `mm_projector`, `non_lora_trainables`, adapter weights, or merged checkpoint projector evidence is present.

Added verifier:

```bash
python3 -m py_compile scripts/verify_medical_vlm_assets.py
```

Remote verification command:

```bash
cd /remote-home/wangbomin/Knowledge_editing
HF_HOME=/remote-home/wangbomin/hugging_cache \
TRANSFORMERS_CACHE=/remote-home/wangbomin/hugging_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=$PWD:$PYTHONPATH \
/root/anaconda3/bin/python scripts/verify_medical_vlm_assets.py \
  --model-root /remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_7b \
  --vision-tower-root /remote-home/wangbomin/hugging_cache/medical_vlms/vision_towers/clip-vit-large-patch14-336 \
  --model-type llava-med \
  --output outputs/medical_vlm_backbone_feasibility/llava_med_asset_verification.json
```

Verifier result with current assets:

```text
runnable_asset_set: false
model root missing: /remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_7b
vision tower root missing: /remote-home/wangbomin/hugging_cache/medical_vlms/vision_towers/clip-vit-large-patch14-336
```

Upload checklist and missing-asset report:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/medical_vlm_asset_upload_checklist.md
/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/medical_vlm_missing_assets_report.md
```

Recommended upload layout:

```text
/remote-home/wangbomin/hugging_cache/medical_vlms/
  llava_med_7b/
  vision_towers/
    clip-vit-large-patch14-336/
```

Next gate:

Implement LLaVA-Med DSCA wrapper, region masks, layer hook, hparams, adapter tests, and real smoke runner only after `scripts/verify_medical_vlm_assets.py` reports `runnable_asset_set: true` for the uploaded model and vision tower.

## LLaVA-Med v1.5 Mistral Asset Preparation

Date: 2026-06-16

Remote target: `/remote-home/wangbomin/Knowledge_editing` on `my-gpu`.

Boundary:

- No DSCA core, ASAM, or LiveEdit code was changed.
- No LLaVA-Med DSCA adapter was implemented.
- No fake medical VLM experiment was run.

Remote Hugging Face access:

- Direct remote `snapshot_download` failed because `huggingface.co:443` timed out.
- The model and vision tower were downloaded locally and uploaded to the remote server with `tar | ssh tar` because remote `rsync` was unavailable.

Uploaded assets:

```text
/remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b
/remote-home/wangbomin/hugging_cache/openai/clip-vit-large-patch14-336
```

Model asset summary:

```text
repo: microsoft/llava-med-v1.5-mistral-7b
model file count: 13
logical file-size sum: 14.09 GiB
files: config/tokenizer/generation config plus 4 safetensors shards
```

Vision tower summary:

```text
repo: openai/clip-vit-large-patch14-336
vision file count: 11
logical file-size sum: 3.19 GiB
files: config, preprocessor_config, pytorch_model.bin, tokenizer files, tf_model.h5
```

Verifier result:

```text
command:
/root/anaconda3/bin/python scripts/verify_medical_vlm_assets.py \
  --model-root /remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b \
  --model-type llava-med \
  --output outputs/medical_vlm_backbone_feasibility/llava_med_v1_5_asset_verification.json

runnable_asset_set: false
tokenizer load: true
vision config load: true
model config JSON parse: true
mm_vision_tower: openai/clip-vit-large-patch14-336
mm_projector_type: mlp2x_gelu
projector keys in safetensors index: 4
model AutoConfig load: false
```

Current blocker:

```text
The remote `transformers==4.51.3` install does not recognize model_type `llava_mistral`
through AutoConfig.from_pretrained, even though it exposes generic LlavaConfig,
LlavaForConditionalGeneration, and MistralConfig classes.
```

Artifacts:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/llava_med_v1_5_asset_preparation_report.md
/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/llava_med_v1_5_asset_verification.json
/remote-home/wangbomin/Knowledge_editing/outputs/medical_vlm_backbone_feasibility/llava_med_v1_5_asset_verification.log
```

Decision: stop before adapter work. The asset files are present, but the loading stack must support `llava_mistral` before a DSCA adapter can be tested honestly.

## LLaVA-Med Official Loader And DSCA Adapter Validation

Date: 2026-06-16

Remote target: `/remote-home/wangbomin/Knowledge_editing` on `my-gpu`.

This section supersedes the previous LLaVA-Med stop point. The plain Transformers AutoConfig path still does not support `model_type: llava_mistral` in the remote environment, but the vendored official LLaVA-Med loader can import, load, and generate from the uploaded assets offline.

Assets:

```text
/remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b
/remote-home/wangbomin/hugging_cache/openai/clip-vit-large-patch14-336
```

Code and config changes:

- Added source-only official LLaVA-Med code under `third_party/LLaVA-Med`.
- Patched the vendored official vision-tower builder to honor `LLAVA_MED_VISION_TOWER_PATH` for offline local CLIP loading.
- Patched vendored `llava_mistral.py` for the installed Transformers generation API by accepting `cache_position` and `logits_to_keep`.
- Added `easyeditor/trainer/llava_med_models/llava_med.py`, a LLaVA-Med EasyEdit wrapper that uses the official loader and constructs full-sequence DSCA masks after official image-token expansion.
- Added `llava-med` support to the DSCA hook resolver through `llava_model.model.layers.{dsca_layer}`.
- Added LLaVA-Med DSCA hparams in `hparams/DSCA/llava_med.yaml`, `hparams/TRAINING/DSCA/llava_med.yaml`, and `hparams/TRAINING/DSCA/llava_med_stage1_smoke.yaml`.
- Extended `scripts/smoke_dsca_stage1.py` with `--model llava-med`.
- Tightened `scripts/verify_medical_vlm_assets.py` so `runnable_asset_set=true` requires a successful official-loader generation smoke when AutoConfig is unsupported.
- Fixed a concrete DSCA dtype bug exposed by LLaVA-Med fp16 hidden states: `DSAModule.forward` now computes the gate branch in the gate layer parameter dtype and casts the gate back to hidden dtype.

Official-loader smoke:

```bash
CUDA_VISIBLE_DEVICES=1 /root/anaconda3/bin/python scripts/smoke_llava_med_official_loader.py \
  --model-path /remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b \
  --vision-tower-path /remote-home/wangbomin/hugging_cache/openai/clip-vit-large-patch14-336 \
  --model-name llava-med-v1.5-mistral-7b \
  --source-root third_party/LLaVA-Med \
  --image-root /remote-home/wangbomin/Knowledge_editing/datasets/MedMKEB/images \
  --device cuda \
  --dtype float16 \
  --output outputs/medical_vlm_backbone_feasibility/llava_med_official_loader_smoke.json
```

Result:

```text
import_ok=true
load_ok=true
generation_ok=true
model_class=LlavaMistralForCausalLM
tokenizer_class=LlamaTokenizerFast
image_processor_class=CLIPImageProcessor
config_model_type=llava_mistral
context_len=2048
mm_projector_param_count=20979712
vision_tower_loaded=true
cuda_memory_peak_mb=14593.19
generated_text="This"
```

Asset verifier:

```bash
/root/anaconda3/bin/python scripts/verify_medical_vlm_assets.py \
  --model-root /remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b \
  --vision-tower-root /remote-home/wangbomin/hugging_cache/openai/clip-vit-large-patch14-336 \
  --model-type llava-med \
  --official-loader-source third_party/LLaVA-Med \
  --official-loader-smoke-json outputs/medical_vlm_backbone_feasibility/llava_med_official_loader_smoke.json \
  --output outputs/medical_vlm_backbone_feasibility/llava_med_v1_5_asset_verification.json
```

Result:

```text
files_complete=true
auto_config_supported=false
auto_config_error="known unsupported AutoConfig path for llava_mistral"
official_loader_supported=true
runnable_asset_set=true
mm_vision_tower=openai/clip-vit-large-patch14-336
mm_projector_type=mlp2x_gelu
projector keys found in model.safetensors.index.json
tokenizer_load.ok=true
vision_config_load.ok=true
missing=[]
```

LLaVA-Med DSCA 3-step smoke:

```bash
CUDA_VISIBLE_DEVICES=1 /root/anaconda3/bin/python scripts/smoke_dsca_stage1.py \
  --model llava-med \
  --hparams hparams/TRAINING/DSCA/llava_med_stage1_smoke.yaml \
  --device cuda \
  --image-root datasets/MedMKEB/images \
  --steps 3 \
  --min_samples 2 \
  --rank 4 \
  --log_dir outputs/medical_vlm_backbone_feasibility
```

Result:

```text
step 1: clusters=1, active_dsams=0, finite loss=7.0919, base_changed=False
step 2: clusters=1, active_dsams=1, DSAM grad mean=0.1444, residual_norm=2.7656, base_changed=False
step 3: clusters=1, active_dsams=1, DSAM grad mean=0.2169, residual_norm=9.4062, base_changed=False
R_k_requires_grad_any=False
optimizer_duplicate_dsam_param_groups=False
repository_round_trip_ok=True
```

Artifacts:

```text
outputs/medical_vlm_backbone_feasibility/llava_med_official_loader_smoke.json
outputs/medical_vlm_backbone_feasibility/llava_med_v1_5_asset_verification.json
outputs/medical_vlm_backbone_feasibility/llava_med_v1_5_asset_verification_rerun.log
outputs/medical_vlm_backbone_feasibility/llava_med_dsca_3step_smoke.log
outputs/medical_vlm_backbone_feasibility/llava_med_dsca_3step_smoke_metrics.csv
```

Validation status:

```text
Official loader generation smoke passed.
Asset verifier passed with official_loader_supported=true and runnable_asset_set=true.
LLaVA-Med DSCA teacher-forced 3-step smoke passed.
No LLaVA-Med 20-edit pilot was run in this stage.
```

Next recommended experiment: run a small LLaVA-Med MedMKEB decoded-generation smoke that exercises the DSCA hook during generation, then run a 20-edit pilot only if the generation smoke shows nonzero DSCA residuals and sane decoded outputs.

## LLaVA-Med Decoded-Generation Gate

Date: 2026-06-16

Remote output directory:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_medmkeb_llava_med_generation_gate/20260616_120130
```

Environment and lightweight checks:

```text
CUDA_VISIBLE_DEVICES=1
cuda_available=True
cuda_device=NVIDIA A100-PCIE-40GB
torch=2.6.0+cu124
py_compile=passed
unit_tests=30 passed, 8 warnings
tests/test_dsca_llava_med_adapter.py=skipped; file absent
```

Official LLaVA-Med loader smoke:

```text
import_ok=true
load_ok=true
generation_ok=true
generated_text="This"
cuda_memory_peak_mb=14593.19
vision_tower_loaded=true
mm_projector_param_count=20979712
```

DSCA decoded-generation smoke result:

```text
empty_repository_identity=true
inactive_dsam_identity=true
residual_scale0_identity=true
masks_align=true
repository_unchanged_during_generation=true
active_routed_dsam_residual_nonzero=false
active_hidden_delta_nonzero=false
active_logits_delta_nonzero=false
generation_hook_active=false
```

Per-sample active-DSAM evidence from `dsca_generation_smoke_per_sample.csv`:

```text
sample 0: candidate_count=1, active_dsams=1, generation_residual_norm=0.046408, active_logits_delta_norm=3176.0833
sample 1: candidate_count=1, active_dsams=1, generation_residual_norm=0.0, active_logits_delta_norm=0.0
sample 2: candidate_count=1, active_dsams=1, generation_residual_norm=0.0, active_logits_delta_norm=0.0
```

Interpretation: ordinary LLaVA-Med generation works offline, and DSCA no-op invariants are now stable for empty repository, inactive DSAM, and residual scale zero. However, the active decoded-generation smoke does not pass because only 1 of 3 routed samples produces a nonzero active residual/logit delta under the current strict acceptance. The 5-sample generation-path diagnostic and 20-edit pilot were not run.

Patches made during this gate:

```text
Added scripts/smoke_llava_med_dsca_generation.py.
Extended scripts/diagnose_dsca_generation_path.py with llava-med dispatch but did not run it because the smoke failed.
Patched DSCA generation hook to no-op cached one-token decode steps whose masks cannot align with full prefill masks.
Patched DSAModule residual math to compute coordinates in fp32, sanitize non-finite hook states, clamp before returning to fp16/bf16, and make residual_scale=0 an exact zero residual.
Added DSAM fp16/non-finite regression coverage in tests/test_dsca.py.
Adjusted the smoke's synthetic active DSAM fixture to use a valid active basis and small trainable bias.
```

No model/data downloads were performed. DSCA hyperparameters were not changed. ASAM and LiveEdit were not modified.

Next debug command:

```bash
cd /remote-home/wangbomin/Knowledge_editing
export HF_HOME=/remote-home/wangbomin/hugging_cache
export TRANSFORMERS_CACHE=/remote-home/wangbomin/hugging_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PWD:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1
OUT_DIR=outputs/dsca_medmkeb_llava_med_generation_gate/20260616_120130
/root/anaconda3/bin/python scripts/smoke_llava_med_dsca_generation.py \
  --model llava-med \
  --dataset MEDMKEB \
  --image-root /remote-home/wangbomin/Knowledge_editing/datasets/MedMKEB/images \
  --hparams hparams/DSCA/llava_med.yaml \
  --training-hparams hparams/TRAINING/DSCA/llava_med_stage1_smoke.yaml \
  --num-samples 3 \
  --device cuda \
  --output-dir "$OUT_DIR/dsca_generation_smoke_debug_rerun" \
  2>&1 | tee "$OUT_DIR/dsca_generation_smoke_debug_rerun.log"
```

Next recommended work: instrument per-step generation hook calls for samples 1 and 2 to determine why selected active DSAMs produce zero residuals during decoded generation before running the 5-sample diagnostic or any 20-edit pilot.

## LLaVA-Med Generation Instrumentation Follow-Up

Date: 2026-06-16

Remote output directory:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_medmkeb_llava_med_generation_gate/20260616_120130
```

Code changes made for this diagnostic:

```text
scripts/smoke_llava_med_dsca_generation.py
- Added route-aware acceptance based only on samples where an active DSAM is actually selected.
- Added --require-all-samples-active-residual as a strict diagnostic gate.
- Added generation diagnostics for use_cache, generation mode, force routing, residual mask mode, and per-hook JSONL events.
- Wrote generation_hook_events.jsonl with prefill/cached_decode hook fields, route ids, residual norms, masks, skip/error reasons, and cached-route reuse state.

easyeditor/trainer/algs/dsca.py
- Added diagnostic-only cache_reuse_route support: prefill caches route weights/selection, and one-token cached decode can reuse that route without recomputing vision/prompt representations.
- Added diagnostic residual mask modes, including current_token.
- Kept empty repository, inactive DSAM, and zero selected experts as exact no-ops.

easyeditor/trainer/algs/dsca_utils.py
- Added DSCAContext diagnostic fields used by the generation smoke.
```

Verification after patches:

```text
local py_compile: passed
local tests/test_dsca.py tests/test_dsca_generation_path.py: 30 passed, 3 warnings
remote py_compile: passed
remote tests/test_dsca.py tests/test_dsca_generation_path.py: 30 passed, 7 warnings
```

Normal cached generation diagnostic:

```text
output: dsca_generation_smoke_cached_instrumented
generation_hook_active=true
all_identity_checks_pass=true
active_route_case_count=3
active_route_nonzero_residual_count=1
active_route_nonzero_residual_rate=0.3333333333333333
active_routed_dsam_residual_nonzero=false
masks_align=true
repository_unchanged_during_generation=true
```

Per-sample active path:

```text
sample 0: active_ids=[0], residual=0.0253401082, hidden_delta=0.0255672522, logits_delta=8.39026356
sample 1: active_ids=[0], residual=0.0, hidden_delta=0.0, logits_delta=0.0
sample 2: active_ids=[0], residual=0.0, hidden_delta=0.0, logits_delta=0.0
```

Hook-event evidence:

```text
active_dsam phases: prefill=3, cached_decode=21
cached_decode errors: DSCA `vision_mask` shape (1, 611/610) does not match hidden (1, 1)
cached_decode residuals: none in normal mode because full-sequence masks cannot align with one-token hidden states
```

No-cache generation diagnostic:

```text
output: dsca_generation_smoke_nocache_instrumented
generation_hook_active=true
all_identity_checks_pass=true
active_route_case_count=3
active_route_nonzero_residual_count=1
active_route_nonzero_residual_rate=0.3333333333333333
active_routed_dsam_residual_nonzero=false
masks_align=true
repository_unchanged_during_generation=true
```

Per-sample active path:

```text
sample 0: active_ids=[0], residual=0.0, hidden_delta=0.0, logits_delta=3176.0659
sample 1: active_ids=[0], residual=0.0, hidden_delta=0.0, logits_delta=0.0
sample 2: active_ids=[0], residual=0.00960758794, hidden_delta=0.0096687302, logits_delta=3720.04199
```

Hook-event evidence: this LLaVA-Med generation path still produces one-token cached_decode hook calls even with `use_cache=false`, and those calls hit the same full-sequence-mask mismatch as cached generation.

Cache-reuse/current-token forced-route diagnostic:

```text
output: dsca_generation_smoke_cache_reuse_current_token_force_rerun
command flags: --dsca-generation-mode cache_reuse_route --residual-apply-mask current_token --force-route-assigned-cluster
exit_code=0
generation_hook_active=true
all_identity_checks_pass=true
active_route_case_count=3
active_route_nonzero_residual_count=3
active_route_nonzero_residual_rate=1.0
active_routed_dsam_residual_nonzero=true
masks_align=true
repository_unchanged_during_generation=true
```

Per-sample active path:

```text
sample 0: active_ids=[0], cached_decode_route_reused=true, residual=0.0018309365, hidden_delta=0.0456632301, logits_delta=8.65203476
sample 1: active_ids=[0], cached_decode_route_reused=true, residual=0.0012738740, hidden_delta=0.0317702256, logits_delta=1906.15540
sample 2: active_ids=[0], cached_decode_route_reused=true, residual=0.0003848870, hidden_delta=0.0098309815, logits_delta=8.26051617
```

Hook-event evidence:

```text
active_dsam phases: prefill=3, cached_decode=21
cached_decode_route_reused=true for cached decode events
cached_decode apply_mask_sum=1 for current-token residual application
cached_decode residual_norm is nonzero for all 21 cached decode hook events
hook errors: none
```

Interpretation:

- The previous `generation_hook_active=false` result was a logging/acceptance bug; hook entry is now confirmed in all generation diagnostics.
- Normal routing is not the immediate blocker for these three samples: active DSAM id `0` is selected for every sample in normal cached and no-cache diagnostics.
- The normal LLaVA-Med generation path applies DSCA only during prefill, then one-token cached decode calls cannot reuse full-sequence masks and become diagnostic no-ops.
- The cache-reuse diagnostic shows residual application itself works on cached decode when the prefill route is reused and the residual mask targets the current token.
- The remaining implementation decision is whether to promote a safe cached-decode route-reuse policy for LLaVA-Med generation. No 20-edit pilot was run, and no DSCA hyperparameters were tuned.

Recommended next step: implement the minimal production path for LLaVA-Med generation route reuse only after deciding the intended residual mask semantics for cached decode (`current_token` vs full prefill-only intervention), then rerun this 3-sample smoke before any 20-edit pilot.

## LLaVA-Med Generation Route-Reuse Promotion

Date: 2026-06-16

Remote output directory:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_medmkeb_llava_med_generation_gate/20260616_132720
```

Implemented behavior:

```text
dsca_generation_mode: cache_reuse_route
dsca_generation_residual_apply_mask: current_token
dsca_generation_reuse_prefill_route: true
dsca_generation_update_repository: false
```

Files changed:

```text
easyeditor/trainer/algs/dsca.py
easyeditor/trainer/algs/dsca_utils.py
easyeditor/trainer/training_hparams/dsca_multimodal_training_hparams.py
easyeditor/models/dsca/dsca_hparams.py
hparams/DSCA/llava_med.yaml
hparams/TRAINING/DSCA/llava_med.yaml
hparams/TRAINING/DSCA/llava_med_stage1_smoke.yaml
scripts/smoke_llava_med_dsca_generation.py
scripts/diagnose_dsca_generation_path.py
tests/test_dsca_llava_med_generation_hook.py
```

Implementation notes:

- Normal DSCA routing is still computed during full-sequence prefill; no force-route is used in normal evaluation.
- Cached decode reuses the prefill route and applies the DSAM residual only to the one-token hidden state via `current_token`.
- Empty repository, inactive DSAM, no-candidate, and `residual_scale=0` remain exact no-ops.
- Generation contexts do not update clusters, prototypes, or PCA buffers.
- Teacher-forced training/edit paths continue to use the existing non-generation context.
- The smoke's synthetic active-DSAM fixture now fixes its gate open so route/mask tests are not dominated by random gate saturation; this is test scaffolding only, not a DSCA formula or hyperparameter change.

Verification:

```text
local py_compile: passed
local tests/test_dsca_llava_med_generation_hook.py tests/test_dsca.py tests/test_dsca_generation_path.py: 37 passed, 3 warnings
remote py_compile: passed
remote tests/test_dsca_llava_med_generation_hook.py tests/test_dsca.py tests/test_dsca_generation_path.py: 37 passed, 8 warnings
remote fixture rerun tests after smoke-gate fixture patch: 37 passed, 7 warnings
```

Three-sample generation gate without force-route:

```text
output: dsca_generation_smoke_fixture_gate
exit_code=0
empty_repository_identity=true
inactive_dsam_identity=true
residual_scale0_identity=true
masks_align=true
repository_unchanged_during_generation=true
generation_hook_active=true
active_route_case_count=3
active_route_nonzero_residual_count=3
active_route_nonzero_residual_rate=1.0
no_hook_errors=true
hook_error_count=0
all_generation_values_finite=true
```

Per-sample generation residuals:

```text
sample 0: active_ids=[0], cached_decode_route_reused=true, residual=0.0017614422
sample 1: active_ids=[0], cached_decode_route_reused=true, residual=0.0017615891
sample 2: active_ids=[0], cached_decode_route_reused=true, residual=0.0017615887
```

Five-sample generation-path diagnostic:

```text
output: generation_path_5sample
exit_code=0
generation_path_uses_dsca_hook=true
generation_residual_zero_rate=0.0
hook_error_count=0
mask_alignment_failures=0
nonfinite_event_residual_count=0
answer_residual_dominance_rate=0.0
assigned_cluster_routed_rate=1.0
active_dsam_available_rate=1.0
teacher_forced_nll_improved_count=4
prompt_only_first_token_rank_improved_count=3
edited_free_generation_contains_target_count=0
edited_equals_base_rate=1.0
mean_base_target_nll=14.568314743041991
mean_edited_target_nll=14.435478401184081
```

Per-sample diagnostic evidence:

```text
sample 0: residual=0.79500699, target_nll 12.204705 -> 12.047191, first-rank 13459 -> 12784
sample 1: residual=1.06041276, target_nll 16.909302 -> 16.697943, first-rank 10616 -> 10361
sample 2: residual=0.81552792, target_nll 12.851246 -> 12.865570, first-rank 4207 -> 4254
sample 3: residual=0.76622266, target_nll 14.421390 -> 14.239046, first-rank 4197 -> 4117
sample 4: residual=0.97520059, target_nll 16.454931 -> 16.327642, first-rank 2815 -> 2841
```

Interpretation:

- The LLaVA-Med decoded-generation path now uses DSCA hooks during generation without force routing.
- Cached decode no longer hits full-sequence-mask versus one-token-hidden mismatch; route reuse works and hook errors are zero.
- The generation intervention is nonzero on all 5 diagnostic samples and shows target-NLL improvement on 4/5 plus first-token rank improvement on 3/5.
- Free decoded text still equals base on 5/5 and contains the target on 0/5, so this validates the code path but not task-level editing success.
- No 20-edit pilot was run in this stage.

Recommended next step: inspect decoded text and target-token rank movement from the 5-sample diagnostic before deciding whether a 20-edit pilot is useful.

## LLaVA-Med 5-Sample Decoded Diagnostic Analysis

Date: 2026-06-17

Input diagnostic directory:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/dsca_medmkeb_llava_med_generation_gate/20260616_132720/generation_path_5sample
```

Analysis output directory:

```text
/remote-home/wangbomin/Knowledge_editing/outputs/llava_med_generation_diagnostic_analysis/20260617_022554
```

Added offline analysis script:

```text
scripts/analyze_llava_med_generation_diagnostic.py
```

The script reads the existing diagnostic files only:

```text
generation_path_summary.json
generation_path_per_sample.csv
generation_path_debug.jsonl
generation_path_report.md
generation_hook_events.jsonl
datasets/MedMKEB/eval.json
```

No model inference was run. DSCA core, ASAM, LiveEdit, and hyperparameters were not changed.

Generated analysis artifacts:

```text
sample_rank_movement.csv
decoded_text_comparison.csv
target_token_rank_table.csv
top_token_debug.md
generation_failure_report.md
generation_failure_summary.json
```

Key table:

| sample | target | base -> edited NLL | first-token rank base -> edited | edited text |
| --- | --- | --- | --- | --- |
| 0 | completely ectocervical and fully visible | 12.2047 -> 12.0472 | 13459 -> 12784 | The most likely abnormality shown in |
| 1 | fascia covered kidney parenchyma | 16.9093 -> 16.6979 | 10616 -> 10361 | The marked area in the image displays the |
| 2 | benign melanocyte | 12.8512 -> 12.8656 | 4207 -> 4254 | The most likely abnormality shown in |
| 3 | blood artifacts | 14.4214 -> 14.2390 | 4197 -> 4117 | The artifact that stands out in the |
| 4 | pulmonary infiltrate | 16.4549 -> 16.3276 | 2815 -> 2841 | The most likely abnormality depicted in |

Aggregate result:

```text
edited_equals_base_rate: 1.0
target_contains_count: 0
mean_base_nll: 14.568314743041991
mean_edited_nll: 14.435478401184081
mean_delta_nll: -0.13283634185791016
teacher_forced_nll_improved_count: 4/5
first_token_rank_improved_count: 3/5
first_token_top10_count_before_after: 0/0
first_token_top50_count_before_after: 0/0
first_token_top100_count_before_after: 0/0
generated_length_words: min=6, max=8, mean=6.6
early_stop_count: 0
generic_prefix_count: 4/5
target_length_words: min=2, max=5, mean=3.0
```

Root cause classification:

```text
weak_logit_shift
```

Rationale:

- The DSCA generation hook is active and residuals/logit deltas are nonzero, so this is not a hook bypass.
- Teacher-forced target NLL improves on 4/5 samples, but the first target token remains far outside top-100 for all samples.
- Free decoded text remains identical to base on 5/5 samples and target appears in 0/5 samples.
- Generated outputs are prompt-like generic prefixes, but there is no evidence of a 1-2 token stop-token issue because decoded lengths are 6-8 words and cached decode events are present.
- Existing diagnostic files did not store top-10 token identities or per-token logprobs, so the analysis reports top-k membership from saved first-target-token ranks only.

Decision:

```text
20-edit is not approved.
```

Next recommended command:

```bash
cd /remote-home/wangbomin/Knowledge_editing
CUDA_VISIBLE_DEVICES=1 /root/anaconda3/bin/python scripts/overfit_dsca_one_medmkeb_edit.py \
  --model llava-med \
  --hparams hparams/DSCA/llava_med.yaml \
  --training-hparams hparams/TRAINING/DSCA/llava_med_stage1_smoke.yaml \
  --dataset-path datasets/MedMKEB/eval.json \
  --image-root datasets/MedMKEB/images \
  --device cuda \
  --output-dir outputs/dsca_medmkeb_llava_med_one_edit_overfit/$(date +%Y%m%d_%H%M%S)
```

If this one-edit decoded overfit still leaves target tokens far from top-k, do not run LLaVA-Med 20-edit. In parallel, run a separate short-answer/template diagnostic only to check whether the prompt-like decoded prefixes come from conversation-template or answer-format mismatch.

## LLaVA-Med generate-vs-forward mismatch diagnostic

Run date: 2026-06-17

Inputs:

```text
force-route overfit:
outputs/dsca_medmkeb_llava_med_one_edit_overfit/sample0_force_route_20260617_024735

force-route residual_scale=2 overfit:
outputs/dsca_medmkeb_llava_med_one_edit_overfit/sample0_force_route_scale2_20260617_025523
```

Diagnostic outputs:

```text
outputs/dsca_llava_med_generate_vs_forward/sample0_force_route_20260617_031851
outputs/dsca_llava_med_generate_vs_forward/sample0_scale2_20260617_031738
```

Required artifacts were produced in both output directories:

```text
generate_vs_forward_summary.json
first_step_logits_comparison.csv
top20_tokens_direct_vs_generate.md
template_generate_vs_forward.csv
manual_greedy_trace.csv
generated_ids_debug.json
generation_hook_events.jsonl
diagnosis_report.md
```

One-edit overfit context:

| run | free decoded target | final target NLL | teacher-forced first target rank | teacher-forced argmax |
| --- | --- | ---: | ---: | --- |
| force-route | no | 0.3332 | 1 | target exact match |
| force-route scale2 | no | 0.000171 | 1 | target exact match |

Generate-vs-forward result:

| run | direct target rank | generate scores target rank | direct argmax | generate argmax | allclose | max abs diff | generated suffix contains target |
| --- | ---: | ---: | --- | --- | --- | ---: | --- |
| force-route | 10300 | 10330 | The | The | false | 0.115234 | false |
| force-route scale2 | 10442 | 10449 | The | The | false | 0.0078125 | false |

Slicing/template findings:

- `generated_suffix_text` and `full_decoded_text` are identical in these runs, so this diagnostic did not find a decode slicing bug for the current LLaVA-Med `generate()` return shape.
- The shared `generate_text` helper was still patched to decode `generated_ids` when sequences include the prompt and to retain `full_text` for auditability.
- Current pilot, official conversation, short-answer instruction, target-prefix, and question-only prompts all generated prompt-like text and none contained the target.
- `use_cache=True` and `use_cache=False` had the same first-token argmax/rank behavior.
- Manual greedy did not contain the target.

Root cause:

```text
direct rank diagnostic used different context
```

Rationale:

- The teacher-forced one-edit overfit can make the target rank 1 under the full answer-conditioned loss context.
- In the actual prompt-only generation context, both direct forward and `generate().scores[0]` keep the target first token around rank 10k and choose `The`.
- Therefore `generate()` is not clearly bypassing DSCA; the stronger evidence is that the earlier rank-1 diagnostic measured a different context from the first free-generation token.

Patch applied:

- Added `scripts/diagnose_llava_med_generate_vs_forward.py`.
- Patched `scripts/smoke_llava_med_dsca_generation.py` so future `generate_text` decodes generated suffix IDs when prompt IDs are included.
- Patched `scripts/overfit_dsca_one_medmkeb_edit.py` so future overfit runs also export `final_repository.pt`, `final_dsca_state.pt`, and `final_config_resolved.yaml`.

Decision:

```text
20-edit is not approved.
```

Next recommended command:

```bash
cd /remote-home/wangbomin/Knowledge_editing
export HF_HOME=/remote-home/wangbomin/hugging_cache
export TRANSFORMERS_CACHE=/remote-home/wangbomin/hugging_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PWD:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1

# Add/run a generation-aligned prompt-only next-token overfit diagnostic first.
# Do not run 20-edit until prompt-only first-token rank can be moved near top-k.
```
