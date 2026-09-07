# CPU regression for the published implementation

Verified research HEAD: `f19ecb7b7127eaf838448b65abe293794d638593`.

Command in the existing server Python 3.12 / Torch 2.6.0 environment, with CUDA hidden:

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 /root/anaconda3/bin/python3.12 -m unittest discover -s tests/medtrace -p 'test_v*.py' -v
```

Result: exit 0, **7 tests passed**, 3.311 seconds in the publication-turn check.

- prepare/start/preflight fixture without loading a model;
- ALL_OFF / ALL_ON and mismatching replay rejection;
- concurrent start and stable resume epoch;
- closure requires queue, exit codes and complete Judge tuples;
- missing/corrupt/misbound start rejected before loader;
- two workers claim exclusively from a fixed 21-item fixture queue;
- pooling shape/normalization, G0 correspondence, G1/G2 no Q access, fit-only supervision, unchanged C1/question state, linear scoring order invariance and G0/G1/G2 branch fixtures.

These are synthetic CPU checks, including a tiny synthetic image-head fit. They are not the 63 scientific fits. Real 14336-dimensional data, live runtime/cache consistency, natural token replay, GPU memory/concurrency and scientific aggregation have not yet been validated by a completed first task.
