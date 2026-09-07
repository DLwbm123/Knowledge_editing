# Baseline and novelty ledger

| Item | Verified scope | Current limitation |
|---|---|---|
| A2 | Original pre-scope CP-R4 native+paraphrase checkpoint for each of seven edits, historical seed/checkpoint binding | Viewed DEV reference, not a new fact or C1 |
| W0 | Equal 320 appended positive optimizer steps | No negative backward; actual compute differs |
| W1 | Three ordinary full-vocabulary Base-to-student KL weights; one global calibration-only choice per parameterization | A control, not complete TIME or a novelty claim |
| W2 | Standard sampled, normalized two-group primal-dual objective and detached EMA | No feasibility/convergence/clinical safety guarantee |
| L16 | Exact-algebra A2 transfer and real-activation numerical checks | 294,912 parameters vs P4's 1,476; capacity/parameterization/optimization are confounded |
| LiveEdit generator | Existing full source-objective checkpoint at step 3000 and 412,548,448-byte safetensors file present; manifest contains both low-rank generators, edit/input extractors, sentinel and residual normalization | Current V4 preprocessing/runtime compatibility and new-episode exposure have not been established; not run as a stage-1 or new-edit baseline |
| TIME / M-ORE / ScopeEdit | Related-work review items identified by the user-provided plan | No new paper reproduction or novelty clearance performed in this execution |

LiveEdit audit sources checked read-only: `methods/liveedit_med/SOURCE_DEVIATION_LEDGER.md`; the source-to-port descriptions in `docs/LIVEEDIT_MED_REPRODUCTION_SUMMARY_FOR_GPT_PRO.md`; remote original `liveedit_med_effectiveness_first_v4/20260812T090628_direct_v4_valid/training/checkpoint_3000/manifest.json`; EqKey-clean `20260815T122907Z/checkpoint_reselection/checkpoint_selection.json` and eligibility metadata. The pinned port uses full layer-21 decoder output, masked generator features, source attention/generator/router equations and safetensors. This is not the CP/L16 down_proj expert used here.

The historical EqKey-clean selection still identifies step 3000 without using its held-out, sealed-blind or forbidden external record. Its selected validation summary includes forced-on native/textual/visual/paired counts 19/17/18/18 and routed 32-expert counts 18/16/11/10. These are historical validation counts under that protocol, **not current V4 results or paper-matched rates**. The older EasyEdit-only audit describing LLaVA as unsupported is not a current verdict on the later `methods/liveedit_med` port.

The presence of a complete generator does not prove that it is compatible with the current canonical V4 source/preprocessing/generation contract. The original shared-generator training exposure and new confirmation candidates must be reconciled before a fair current-baseline comparison. No held-out training was authorized, no hand-trained CP/L16 expert is labelled LiveEdit, and no legacy routed-system failure has been rewritten.

Ordinary KL, low-rank residuals, CP factorization and primal-dual optimization are existing tools. This run tests selective-write behavior in the frozen current fact-control setting. It does not implement M-ORE's shared update geometry, recursive update, MMD, HSIC, curvature projection, or a new router. W2 needs incremental evidence over calibrated W1 before it is worth retaining. No novelty, SOTA, formal M3Bench or clinical-validity claim is made.
