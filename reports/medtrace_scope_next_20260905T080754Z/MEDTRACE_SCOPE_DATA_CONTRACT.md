# MedTRACE single-edit scope data contract

Status: `MEDTRACE_INTRINSIC_SCOPE_NOT_READY__NOT_RUN`

This sidecar extends only the MedTRACE development/calibration layer. It does not change any V4 query, answer, cohort, order, or verdict.

## Bound specification

- V0.1 SHA-256: `f6c07541345c7033d68c20f4dcd4ab21f928c08ea09c9cdde815986ae3bfecbd`.
- V0.2 SHA-256: `0ba02155eb658201e9c0d140c6bcd811587b3a44af4923e2b72eebf3583d179c`; it overrides conflicting V0.1 requirements.
- V0.2 requires scope-train, threshold-calibration, and held-out evaluation roles to be frozen before scoring. Its calibration minimum is 4 positive and 20 negative records, with no model-visible EqKey overlap.

## Role-pair isolation matrix

| Left role | Right role | Group isolation required | Current overlap/use |
|---|---|---|---|
| Expert fitting | Scope fitting | Not universally fact-disjoint; inputs and use must be declared | Scope fitting is empty |
| Expert fitting | Threshold calibration | Calibration must not train expert parameters | Calibration is empty |
| Scope fitting | Threshold calibration | No model-visible EqKey overlap; calibration does not enter the optimizer | Both are empty |
| Scope fitting | Evaluation | Evaluation-only records cannot be reused for fitting | Seven formal probes remain evaluation-only |
| Threshold calibration | Evaluation | No model-visible EqKey overlap; evaluation cannot choose threshold | Calibration is empty |
| Native in-sample replay | Expert fitting | Explicit overlap is permitted and is not unseen generalization | The primary edit is in both roles |
| Native in-sample replay | Evaluation | Report separately from held-out evaluation | One replay; seven exposed formal probes |

Source-image, fact, and semantic-equivalence overlap are recorded independently. Distinct query IDs or file hashes alone are not independence evidence.

## Verified inventory

- One SLAKE source QA is the already-used native expert-fitting record and native in-sample replay.
- Four T1G records are blur/crop/contrast/noise descendants of the same source image and edit fact. They are on-scope positives, never negatives.
- Three T2G records are frozen question variants linked to the same edit. They are on-scope evaluation records, not calibration material.
- All seven formal probes were already exposed by the zero-effect run and remain evaluation-only; none is called unseen.
- The source rows, original/derived image lineage, image hashes, edit linkage, and frozen formal relation IDs were verified from the V4 static inventory and formal event records. Processed tensor hashes and V0.2 model-visible EqKeys are not present.
- The complete record-level sidecar is private at `/remote-home/wangbomin/medtrace_runs/20260905T080754Z/MEDTRACE_SCOPE_SIDECAR_PRIVATE.json` (SHA-256 `4a9d362a955dac9e972576825ce6dd8f3a53a1d5a38943d701f80eac607ec59e`); the public manifest contains only aggregate counts, missing fields, and this lock hash.

## Exact blocker and minimum supplementation

No intrinsic-scope score, fit, or threshold was run. The available roles are expert fitting 1, scope fitting 0, threshold calibration 0, evaluation 7, and native replay 1.

Before scope can start, the single edit still needs:

1. Pre-registered scope-fit textual, visual, and paired positives plus source-supported factual-shift negatives that are known not to accept the edit.
2. A separate calibration set with at least 4 positives and 20 negatives, as required by V0.2.
3. A pre-registered independent off-scope evaluation-negative panel; V0.2 gives no numeric minimum for this panel, so this report does not invent one.
4. Processed-image hashes, routing-input hashes, attention-mask hashes, image sizes, and assistant-boundary indices needed to construct and de-duplicate V0.2 EqKeys.
5. Frozen role manifests and hashes before any activation scoring, fitting, threshold selection, or checkpoint continuation.

String inequality, different files, base-model errors, and a text-only semantic Judge are not acceptable negative labels. LoRA QUAL material is excluded throughout.

## Scientific boundary

The preserved result is a `TIME-inspired asymmetric CP single-edit core` forced-on expressivity result. Intrinsic routing remains unexecuted; no TPR, FPR, threshold, off-scope parity, or scope qualification value exists for this run.
