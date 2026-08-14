# Minimal MedMKEB Editing Runbook

This repo is now kept as a small local VLKEB/EasyEdit editing workspace.
Large data, images, models, caches, and outputs stay outside the code repo:

```text
/Volumes/DataP/knowledge_editing/
  data/medmkeb/raw/          # MedMKEB JSON files
  data/medmkeb/images/       # resolved benchmark images
  data/medmkeb/processed/    # smoke data and generated VLKEB input
  outputs/medmkeb/           # run outputs
```

## What To Read

```text
scripts/medmkeb/run_medmkeb_editing.py    # main entry point
scripts/medmkeb/check_medmkeb_assets.py   # image sanity check
src/medmkeb_editing/adapter.py            # MedMKEB -> VLKEB/EasyEdit records
src/medmkeb_editing/asset_resolver.py     # image path resolution
easyeditor/trainer/algs/ASAM.py           # ASAM_FT / ASAM_MEND
easyeditor/trainer/algs/asam_utils.py     # LAR delta generation and RCSL loss
easyeditor/trainer/blip2_models/blip2_opt.py
easyeditor/trainer/blip2_models/mini_gpt4.py
```

## Required Data On Server

You do not need the original source datasets on the server. Copy only the JSON
files and resolved images needed for the split you will run.

```text
data/medmkeb/raw/
data/medmkeb/images/
```

Approximate split-specific image sizes from the local prepared copy:

```text
eval_data_attack.json         721 records   ~1.95 GB images
eval_data_threehop_final.json 2497 records  ~5.12 GB images
train_data.json               4490 records  ~7.55 GB images
```

## Check Assets

```bash
python3 scripts/medmkeb/check_medmkeb_assets.py \
  --root /Volumes/DataP/knowledge_editing
```

This writes:

```text
/Volumes/DataP/knowledge_editing/data/medmkeb/reports/asset_report.json
/Volumes/DataP/knowledge_editing/data/medmkeb/reports/missing_images.tsv
```

## Dry Run

```bash
python3 scripts/medmkeb/run_medmkeb_editing.py \
  --root /Volumes/DataP/knowledge_editing \
  --method ASAM_FT \
  --max-edits 2 \
  --dry-run
```

## Mock Run

This checks adapter/output plumbing without loading a model.

```bash
python3 scripts/medmkeb/run_medmkeb_editing.py \
  --root /Volumes/DataP/knowledge_editing \
  --method ASAM_FT \
  --max-edits 2 \
  --mock-edit
```

## Real Single-Edit Run

Set the model/checkpoint paths in the hparams file first.

```bash
python3 scripts/medmkeb/run_medmkeb_editing.py \
  --root /Volumes/DataP/knowledge_editing \
  --method ASAM_FT \
  --hparams hparams/ASAM_FT/blip2.yaml \
  --data-file /Volumes/DataP/knowledge_editing/data/medmkeb/raw/eval_data_attack.json \
  --max-edits 10 \
  --device 0
```

## Sequential Editing

Use the same runner and set `--sequential-edit true`.

```bash
python3 scripts/medmkeb/run_medmkeb_editing.py \
  --root /Volumes/DataP/knowledge_editing \
  --method MEND \
  --data-file /Volumes/DataP/knowledge_editing/data/medmkeb/raw/eval_data_attack.json \
  --max-edits 20 \
  --sequential-edit true \
  --device 0
```

## ASAM Notes

`ASAM_FT` can run as an FT-style editor. `ASAM_MEND` is a MEND-compatible
training objective; for real evaluation, train an ASAM_MEND checkpoint first and
put its path in `hparams/ASAM_MEND/blip2.yaml` under `archive`. This repository
does not currently define WISE, so ASAM support is limited to `ASAM_FT` and
`ASAM_MEND`.

The ASAM implementation now follows the paper's LAR + RCSL structure:

```text
asam_use_lar: true
asam_epsilon: 1.0e-3
asam_num_variants: 4
asam_lar_step_size: 1.0e-3
asam_tau: 4.0
asam_beta: 10.0
asam_use_dataset_variants: false
asam_capture_module: null
asam_allow_attention_pooling_fallback: false
asam_debug_grad_check: false
asam_debug_require_all_inner_grads: false
```

LAR injects an additive delta at the joint vision-language embedding tensor:

```text
Blip2OPT:  after torch.cat([inputs_opt, text token embeds], dim=1)
MiniGPT4:  after torch.cat([img_embeds, text token embeds], dim=1)
```

The delta is masked before application and projection. Image/query tokens and
prompt/input tokens are perturbable; answer target tokens and padding positions
are not. The local BLIP2 and MiniGPT-4 wrappers build this mask from
`labels == -100` combined with `attention_mask`, aligned to `inputs_embeds`.

ASAM internal LAR/RCSL calls bypass logits-only `EditableModel.forward` wrappers
and call the raw wrapped backbone so the full output object is preserved. RCSL
pooling therefore sees the full-sequence `asam_perturb_mask`, `labels`, and
`attention_mask` produced by BLIP2/MiniGPT-4.

Dataset `rephrase` / `image_rephrase` examples are not used as adversarial
variants. They can only be enabled as additional supervised variants with
`asam_use_dataset_variants: true` and `asam_variant_ce_weight > 0`.

RCSL captures the edit-layer module output, pools one representation per batch
example, detaches the original representation as the semantic anchor, and runs
SVD directly on the normalized hidden-state matrix `H_s`.

When `asam_use_lar: true`, RCSL pooling refuses to average full sequences unless
it has a reliable prompt/image mask aligned to the captured hidden-state length.
The attention-mask-only fallback is disabled by default; it requires explicitly
setting `asam_allow_attention_pooling_fallback: true` for diagnostic use.

By default `asam_capture_module: null` infers a block-level capture point:

```text
Same block params:
  block.fc1.weight, block.fc2.weight -> block

OPT decoder params:
  opt_model.model.decoder.layers.29.*
  opt_model.model.decoder.layers.30.*
  opt_model.model.decoder.layers.31.*
  -> opt_model.model.decoder.layers.31
```

If this inference is not safe for a new backbone, set `asam_capture_module`
explicitly. Use `asam_alignment_params` to restrict the debug gradient check to
the parameters that should be covered by alignment.

Known limitation: LAR currently supports local BLIP2 and MiniGPT-4 wrappers.
Other MLLM wrappers must add the same `maybe_apply_asam_delta(...)` call at
their joint embedding construction point; otherwise `asam_use_lar=true` raises
`NotImplementedError` instead of silently falling back to rephrases.
