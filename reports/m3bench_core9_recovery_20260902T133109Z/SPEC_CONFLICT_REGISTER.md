# Core-9 specification conflict register

Operational status: `PUBLIC_RELEASE_ALIGNED`, not paper-exact.

| Topic | Paper-facing specification | Public release at `03c6fda3813301dab3be5831fdc94b493c10afc9` | Operational decision |
|---|---|---|---|
| Scope | Ten tasks including temporality | T0-T4 builders are runnable; the T5 README says the reference experiments did not run T5 | Use Core-9 only; keep T5 as a separate blocked extension |
| T5 metric/assets | Temporal evaluation is reported | Public definition uses `delta_ACC`; required model-specific baseline/post-edit files and source assets are not released | Do not create a substitute T5 task or metric |
| T1G image semantics | Same-semantic matched images | Public builder records four perturbation variants of the same image | Follow the reproducible public builder and label the result public-release-aligned |
| Edit pools | Task-specific pools are described for T0, T2L, T3, T4L, T4G, and T5 | Public metadata has separate task tables | Do not assume the T0-derived amended-189 sequence intersects every task pool |
| Builder semantics | Task roles require correct image, question, gold, and edit/probe direction | T2L uses synthetic metadata questions; T3 applies the anchor question to paired-modality images; T4L requires exact `question_a` matching | Audit both public-builder parity and semantic role validity |
| Missing release material | Results depend on model-specific predictions and third-party datasets | Several model-specific outputs and third-party source assets are intentionally absent | No reconstruction claim beyond available public data |

Authority used here:

- data construction: public release commit plus its official metadata;
- primary metric, if a catalog becomes legal: macro average over eligible edit requests;
- current scope: T0-T4 Core-9;
- paper-exact claim: not permitted.

