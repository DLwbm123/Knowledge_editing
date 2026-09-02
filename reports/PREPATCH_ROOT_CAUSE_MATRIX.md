# Pre-patch root-cause matrix

The same authoritative-retained smoke record was traced in four fresh GPU2
processes. Private raw tokens, text, image paths, and record binding remain in
the run root and are not published here.

| Editor | Pre NLL | Post NLL | Gradient norm | State delta | Self route | Raw equals base | Evidence classification |
|---|---:|---:|---:|---:|---|---|---|
| LoRA | 6.629343 | 2.671752 | 1.000000 | 0.994039 | NA | yes | `GENERATION_PATH_BYPASSES_EDITOR` rule triggered; adapter was nevertheless active at the traced target in prefill and all decode calls |
| GRACE | 6.629343 | 6.629343 | 0.000000 | 0.000000 | yes | yes | `OPTIMIZER_OR_TRAINABILITY_FAILURE` + `EDIT_STATE_NOOP`; replacement mask excludes positions contributing target loss |
| BalanceEdit | 6.629343 | 0.000521 | 0.000234 | 450.714849 | yes | no | no failure on this record; retain explicit empty-generation regression because historical batch output contained empties |
| BELoRA | 6.629343 | 6.375094 | 1.000000 | 0.177014 | yes | yes | `GENERATION_PATH_BYPASSES_EDITOR` rule triggered; routed adapter was active in prefill and all decode calls |

All four traces verified one image token, multimodal expansion, non-empty target
labels, and unchanged frozen base parameters. The repair therefore targets the
shared lifecycle/observability contracts and the proven GRACE replacement bug;
it does not replace the Foundation input path or alter the locked scientific
hyperparameters.
