# GPU usage report

- GPU2 UUID: `GPU-35be76e9-8ca5-1877-ddfe-27eb08f6721b`; used only for the fresh official no-edit canary.
- GPU3 UUID: `GPU-43e3d478-7979-ea29-8130-64a467b48a5c`; used only for the formal-runtime no-edit canary.
- Observed peak process allocation during the canary was approximately 15.2 GiB on each allowed GPU.
- GPU1 was excluded from every process environment and was not used.
- Both allowed GPUs returned to 1 MiB idle allocation after the canary.
- No editor was installed, trained, or evaluated.
