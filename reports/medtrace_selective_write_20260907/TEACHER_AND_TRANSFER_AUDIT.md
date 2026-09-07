# Teacher and transfer audit

Original pre-scope A2 checkpoint and seed bindings checked against historical manifests for all 7 edits. See TRANSFER_CPU_RESULTS.json for real pre-CP prompt/visual residual errors and parameter counts. No SVD. CP scaling is beta/sqrt(4). L16 extra rows are seeded unit vectors; extra output columns are zero.

Teacher uses exact generated token IDs appended to the target-free multimodal prompt, not detokenized/re-tokenized answers. Shifted answer labels include first-answer and generated EOS predictors and exclude image/padding. FP32 full-vocabulary Base||student has live student gradients. Fit and score-only cache roots are separate. Runtime/model/image/prompt/prefix/mask/dtype are cache-bound. Natural-generation transfer and real-model disabled parity must still pass before any L16 trajectory.
