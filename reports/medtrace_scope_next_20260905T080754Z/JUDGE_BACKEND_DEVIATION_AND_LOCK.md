# MedTRACE vLLM Judge backend deviation and lock

Status: `VLLM_EXECUTION_LOCKED__BRIDGE_NOT_RUN`

## Preserved semantic identity

The one-row CP development packet used the existing Qwen3-32B-AWQ Judge model and the legacy two-candidate semantic contract. The legacy semantic protocol SHA-256 is `77260a1a994f72d72155a43367d3fc4fe5779502735b1e24a0b96684d081d4bd`.

The vLLM run parsed successfully and preserved the prior `is_correct=true` result. This is auxiliary evidence only; the CP result remains supported by normalized literal target matching, NLL, first-token rank, three generation paths, reload, and rollback checks.

## Separate execution identity

The new backend execution SHA-256 is `133a9636e872b5e6ce2aa52071deb447d3acc490807dd088451690ecb39800fe`. It is intentionally separate from the legacy semantic protocol hash.

The actual run used vLLM 0.9.2, PyTorch 2.7.0, Transformers 4.53.2, CUDA 12.6, the V1 engine, `torch.float16`, resolved `AWQConfig`, and xgrammar 0.1.19 on physical GPU3 (`GPU-43e3d478-7979-ea29-8130-64a467b48a5c`). Model and tokenizer snapshot were both `0499c3ac83fdef8810b907a23894ba91e95eddd8`.

The CLI now locks the live engine/runtime configuration, tokenizer files and rendering code, structured-output candidates, resolved sampling settings, packet content, and actual prompt/output token sequences. It rejects duplicate packet keys, refuses output overwrite, atomically publishes output and the private lock, verifies the GPU UUID, and rejects inputs above 1,024 tokens before generation.

The public lock contains the corresponding counts and hashes. The complete token sequences, QA payload, opaque key value, raw Judge output, and full engine representation remain in the private server lock, whose SHA-256 is `4b02963afbcf17c2e1d5ffa4867fc0649f5a79cb8117cbc2476d2d18b3a4342d`.

## Difference from the legacy Transformers path

The legacy runner defaults to a 768-token input limit with truncation. This vLLM lane uses `max_model_len=1024` and hard-fails before generation rather than silently truncating. The observed rendered request had 190 prompt tokens; vLLM returned the same sequence that was pre-tokenized and locked.

Backend bridge status is `NOT_RUN`. No set of at most 32 historical Judge inputs was verified as authorized for new development reuse without consuming LoRA QUAL or formal blind-test material. Therefore cross-backend agreement and token-level equivalence remain unproven, and the vLLM result must not be described as Transformers-equivalent.
