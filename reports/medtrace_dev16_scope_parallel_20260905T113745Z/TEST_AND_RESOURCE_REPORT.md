# MedTRACE DEV16/scope prelaunch checks

Status: `PASS__READY_TO_LAUNCH`

- `python scripts/medtrace/verify_core.py`: 7 tests passed.
- `python -m py_compile scripts/medtrace/run_dev16.py scripts/medtrace/build_scope_census.py`: passed.
- `git diff --check`: passed.
- DEV16: 16 unique records in frozen order; private input SHA-256 matches the V4 manifest.
- Model/runtime: frozen official-native LLaVA-Med contract; expected layer size 14,336 to 4,096 is checked again after load.
- GPU2 and GPU3 each showed 1 MiB used and 40,445 MiB free before launch; no MedTRACE process was present.
- GPU1 is excluded. No other process will be terminated, and no cron, heartbeat, or monitor is created.
