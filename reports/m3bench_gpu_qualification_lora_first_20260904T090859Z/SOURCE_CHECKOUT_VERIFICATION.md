# Source checkout verification

Status: `PASS`

- Repository: `microsoft/LLaVA-Med`
- Commit: `30697ca50b5c29a8e955c99330b259776aef27b9`
- Checkout: detached, clean, and imported from the verified checkout
- GPU initialization during source lock: false
- Qualification interpreter: Python 3.12.7, PyTorch 2.6.0+cu124, Transformers 4.51.3, PEFT 0.19.1
- Official source files were not modified. The project adapter temporarily resolves the checkpoint's remote vision-tower identifier to the already frozen local snapshot while loading offline.

The imported-file checksums and host paths remain in the private run evidence. This public report deliberately omits local absolute paths.
