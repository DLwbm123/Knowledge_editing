# MedTRACE focused test report

Status: `PASS`

Command: `python scripts/medtrace/verify_core.py`

Result: 4 tests passed.

Covered checks:

- factorized and materialized-dense residual output parity;
- factorized and materialized-dense gradient parity;
- factor normalization and serialization invariance;
- exact zero effect and disabled bypass;
- assistant predictor mask for teacher forcing and generation;
- nonzero `rho` gradient from zero initialization;
- CP-only optimizer membership and frozen-base exclusion;
- fresh hook state contains no previous request mask.
