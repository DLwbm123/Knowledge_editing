# ENGRAM for Medical MLLM Editing

ENGRAM is a forward-only editing prototype for medical multimodal language models. It estimates a target subspace from edit activations and a reference subspace from locality/retain activations, then edits selected Linear layers without gradient training.

For a selected Linear layer with weight `W`:

```text
Sigma_plus  = sum x x^T over target activations
Sigma_minus = sum x x^T over reference activations
P           = Sigma_plus @ pinv(Sigma_plus + Sigma_minus)
E           = W @ P
W_new       = W - alpha * E
```

If bias absorption is enabled, activations are augmented as `[x; 1]`, `W` is augmented as `[W, b]`, and the resulting update is split back into weight and bias deltas.

## Continual Engram Editing

Each edit can be saved into an ENGRAM bank. The bank stores JSON metadata and tensor deltas, not full model weights. Multiple edits can be composed as:

```text
W_new = W - sum_i alpha_i * E_i
```

Rollback is supported by adding the stored delta back to the edited model. In normal evaluation, set `sequential_edit: false` or pass the existing EasyEdit flag that restores original weights after each case. For continual editing experiments, set `sequential_edit: true` and save edits into a bank.

## Erasure vs Replacement

Erasure mode removes the target-associated response subspace:

```text
W_new = W - alpha * E_old
```

Replacement mode is experimental. It uses an external candidate delta and applies:

```text
W_new = W - alpha * E_old + beta * Delta_safe
```

`Delta_safe` can come from a pair of matching state dicts. LoRA support is intentionally not silently merged in this prototype; merge LoRA into explicit state dicts first.

## Medical X_plus and X_minus

Use `target_variants` for `X_plus`:

- `edit`: edit prompt plus target
- `rephrase`: rephrase prompt plus target
- `image_rephrase`: image rephrase plus target

Use `reference_variants` for `X_minus`:

- `locality_text`: text locality prompt and ground truth
- `locality_multimodal`: multimodal locality prompt/image and ground truth
- `retain_pool`: optional JSONL retain pool from `retain_pool_path`

If `X_minus` is empty, ENGRAM logs a warning because the edit degenerates to target-only erasure. Use `min_reference_examples` and `skip_if_insufficient_reference` to enforce stricter runs.

## Dry-Run and Mock Commands

These commands require PyTorch but no model weights or private medical data:

```bash
python scripts/engram/run_engram_alpha_sweep.py --mock --out <OUTPUT_DIR>/alpha_sweep
python scripts/engram/run_engram_layer_ablation.py --mock --out <OUTPUT_DIR>/layer_ablation
```

MedMKEB adapter dry-run without model loading:

```bash
python scripts/medmkeb/run_medmkeb_editing.py \
  --root <DATA_ROOT> \
  --method ENGRAM \
  --hparams hparams/ENGRAM/blip2.yaml \
  --dry-run \
  --max-edits 2 \
  --output-dir <OUTPUT_DIR>/medmkeb_dry_run
```

## Real Model Run

Use placeholders for private paths:

```bash
python scripts/medmkeb/run_medmkeb_editing.py \
  --root <DATA_ROOT> \
  --data-file <DATA_FILE> \
  --image-root <IMAGE_ROOT> \
  --method ENGRAM \
  --hparams hparams/ENGRAM/llava_med.yaml \
  --model-name-or-path <MODEL_PATH> \
  --max-edits 5 \
  --output-dir <OUTPUT_DIR>/engram_real
```

To save a bank, set in YAML or override in code:

```yaml
bank_dir: <BANK_DIR>
edit_id: null
sequential_edit: false
```

## Bank Utilities

```bash
python scripts/engram/list_engram_bank.py --bank <BANK_DIR> --export-csv <OUTPUT_DIR>/engram_bank.csv

python scripts/engram/apply_engram_bank.py \
  --bank <BANK_DIR> \
  --edit-id <EDIT_ID> \
  --dry-run

python scripts/engram/rollback_engram.py \
  --bank <BANK_DIR> \
  --edit-id <EDIT_ID> \
  --dry-run
```

Programmatic application:

```python
from easyeditor.models.engram import EngramBank

bank = EngramBank("<BANK_DIR>")
bank.apply_edit(model, "<EDIT_ID>")
bank.rollback_edit(model, "<EDIT_ID>")
```

## Overlap and Conflict Detection

```bash
python scripts/engram/compute_engram_overlap.py \
  --bank <BANK_DIR> \
  --out <OUTPUT_DIR>/engram_overlap_report \
  --threshold 0.35 \
  --heatmap
```

Overlap uses stored projectors when available:

```text
overlap(P_i, P_j) = ||P_i P_j||_F / (||P_i||_F ||P_j||_F + eps)
```

If projectors are not stored, it falls back to normalized update tensors.

## Alpha Sweep and Layer Ablation

```bash
python scripts/engram/run_engram_alpha_sweep.py \
  --mock \
  --alphas 0.05,0.1,0.2,0.4,0.6,0.8,1.0 \
  --out <OUTPUT_DIR>/alpha_sweep

python scripts/engram/run_engram_layer_ablation.py \
  --mock \
  --groups qk_only,qk_gate,projector_only,qk_gate_projector,all_configured,no_qk_gate \
  --out <OUTPUT_DIR>/layer_ablation
```

Real-model alpha/layer sweeps should wrap `scripts/medmkeb/run_medmkeb_editing.py` and restore weights between settings unless `sequential_edit` is explicitly intended.

## Known Limitations

- Retrospective-only: ENGRAM edits from explicit target/reference sets.
- Requires careful construction of `X_plus` and `X_minus`.
- Covariance storage is `O(d^2)` per selected layer.
- Replacement mode is experimental and requires external candidate deltas.
- Not validated for clinical deployment.
- No experimental result should be claimed from mock or dry-run commands.

