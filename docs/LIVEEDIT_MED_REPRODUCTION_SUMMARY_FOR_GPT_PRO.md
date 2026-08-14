# LiveEdit-Med 复现阶段总结（供 GPT Pro 审阅）

更新时间：2026-08-14  
项目：`Knowledge_editing`  
远端权威目录：`my-gpu:/remote-home/wangbomin/Knowledge_editing`  
主要运行目录：

```text
outputs/liveedit_med_effectiveness_first_v4/
20260812T090628_direct_v4_valid/
```

## 给 GPT Pro 的审阅任务

请把下面内容视为一份待审计的实验记录，而不是预设结论。希望你重点判断：

1. 当前证据是否足以说明 LiveEdit 的核心 expert-generation 机制已经复现；
2. 完整 repository 路由失败应被解释为“官方复现失败”“医疗域迁移失败”，还是“额外严格压力测试失败”；
3. 当前实现与官方 LiveEdit 的差异是否可能实质影响 visual、paired 和 safety 结果；
4. 下一步最小、最忠于官方实现、且不发生测试泄漏的验证应是什么；
5. 是否应该先补跑官方同口径指标，再讨论对 LiveEdit router 的改进。

---

## 1. 一句话结论

当前最准确的结论是：

> **LiveEdit 的源码模块、核心张量方程和 expert generation 能力已得到正向验证；但是在一个由 32 个 experts 构成的医疗多模态 repository 中，端到端路由没有通过我们预先定义的严格 Stage Q gate。主要问题是 visual/paired 召回失败和 safety false positives，而不是 generator 完全没有能力。**

因此，不应简单写成“LiveEdit 全部复现成功”，也不应简单写成“LiveEdit 完全复现失败”。当前状态应拆成三层：

| 层级 | 当前状态 | 说明 |
|---|---:|---|
| 官方源码模块与核心方程 port | PASS | pinned commit、hash、模块输出和 optimizer-step parity 均通过 |
| 单 expert / forced-on generation | PASS（带条件） | Direct Expert unrestricted 通过；后续 Stage F 五种输入全部通过 |
| 完整 32-expert repository routing | FAIL | native/textual 通过，visual/paired 漏路由；safety 仅 27/40 |

---

## 2. 复现目标与边界

本次任务复现的是 **LiveEdit 在医疗视觉语言模型上的行为**，不是复现 ENGRAM。

代码中出现的 canonical ENGRAM bank 只承担以下基础设施作用：

- 恢复项目已有的 clean S0 权重锚点；
- 验证评估前后 backbone 和 canonical bank 没有被修改；
- 提供已有的 LLaVA-Med 加载环境。

本次训练和 repository 中的 experts 来自 LiveEdit port 的：

- `QVExtractor`；
- `LowRankGenerator`；
- visual-hard retrieval；
- textual sigmoid × softmax fusion；
- low-rank residual intervention。

因此，方法身份应始终表述为 **LiveEdit-Med reproduction**。ENGRAM 不是本轮被评估的方法。

---

## 3. 官方源码固定与实现一致性

### 3.1 固定的官方版本

```text
qizhou000/LiveEdit
commit: 3615a37b05294509f411df045621940f276a5e6b
```

本地保存了不可变 upstream snapshot：

```text
third_party/liveedit_official_3615a37/
```

### 3.2 已通过的一致性检查

- upstream commit pinned：PASS；
- upstream blob hashes：PASS；
- MIT license：已保留；
- 原始 source parity harness：46/46；
- 最大模块数值误差：0.0；
- optimizer step exact parity：PASS；
- 后续加入 post-hoc validation/Stage Q 测试后：51/51 tests passed；
- `py_compile`：PASS。

核心对应关系包括：

| LiveEdit 官方逻辑 | 当前 port |
|---|---|
| `Attention` | `methods/liveedit_med/upstream_modules.py::Attention` |
| `QVExtractor` | `methods/liveedit_med/upstream_modules.py::QVExtractor` |
| `LowRankGenerator` | `methods/liveedit_med/upstream_modules.py::LowRankGenerator` |
| `get_new_edit` | `methods/liveedit_med/source_ops.py::generate_expert_and_keys` |
| `get_edit_residual` | `apply_low_rank_expert_residual` |
| `get_moe_fuse_coe` | `compute_text_soft_weights` |
| `retrieve_moes` | `route_repository` |
| `train_a_batch` | `trainer.py` 与 `source_ops.py` |

### 3.3 明确记录的有意差异

1. backbone 改为项目中的 LLaVA-Med Mistral 7B；
2. residual 注入位置是完整 `model.layers.21` decoder-block output；
3. natural generation 不向输入泄漏 target，并额外要求 manual no-cache / cached / HF greedy token parity；
4. visual candidate 为空时精确 bypass 到 S0，避免对空集合计算 softmax；
5. checkpoint/repository 使用 safetensors + JSON，而不是 pickle；
6. 医疗数据使用 deterministic edit-level split 和 equivalence-key isolation；
7. LayerNorm 显式恢复 PyTorch 默认初始化，因为项目的 LLaVA utility 会使其 `reset_parameters` 失效。

账本声明：source attention、extractor、generator、residual、routing、loss、optimizer 和 scheduler 方程没有被主动修改。

需要 GPT Pro 判断的是：上述“方程等价”是否足以支持“实验设置等价”，尤其是 backbone、注入层、数据映射和 generation protocol 的变化。

---

## 4. 模型、数据和训练设置

### 4.1 模型和训练配置

| 项目 | 设置 |
|---|---|
| Backbone | LLaVA-Med Mistral 7B |
| Intervention layer | full decoder block output at layer 21 |
| LiveEdit rank | 4 |
| Module dimension | 1024 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Shared-source training | 50 epochs / 3200 optimizer steps |
| Checkpoints | 500, 1000, 1500, 2000, 2500, 3000, 3200 |
| Generation | `do_sample=False`, `num_beams=1`, `max_new_tokens=128` |

### 4.2 数据隔离

医疗 source records 的 deterministic split 为：

| Split | Edit 数量 |
|---|---:|
| Train | 512 |
| Validation | 64 |
| Held-out | 64 |

record 953 以及其所有输入被视为 external target，未进入训练或 checkpoint selection。

### 4.3 训练状态

- 完成 3200/3200 optimizer steps；
- 完成 50/50 epochs；
- 七个 checkpoint 全部存在并通过 tensor/hash 审计；
- 无 NaN、OOM、traceback 或 runtime failure；
- 训练结束时原有 `validation_generation_panel.jsonl` 为空；
- 因此训练结束时的严格状态是：

```text
LIVEEDIT_SOURCE_TRAINING_CONVERGED__BEHAVIORAL_VALIDATION_PENDING
```

这意味着训练完成，但当时尚未完成自然生成行为驱动的 checkpoint selection。

---

## 5. Direct Expert 正向能力检查

Direct Expert 是一个 expressivity positive control，不是最终 shared generator 的完整评估。

结果：

```text
label: LIVEEDIT_DIRECT_EXPERT_PASS_R4
rank: 4
success step: 230
manual no-cache / cached / HF parity: PASS
```

Canonical target：

```text
completely ectocervical and fully visible
```

Direct Expert unrestricted output：

```text
The most likely abnormality shown in the image is a completely ectocervical
and fully visible and palpable cervix.
```

该输出包含 canonical target span，正常 EOS，无显式 contradiction，因此 unrestricted gate 通过。

但是，Direct Expert 当时的 short-answer 输出为：

```text
The most likely abnormality shown in the picture is a perforated ectocervical
and fully visible and ectocervically located cervical tumor.
```

其 canonical target span matcher 未通过。因此不能把 Direct Expert 的结果描述成“五种自然生成全部通过”。Direct Expert 只证明 rank-4 residual 有产生目标答案的表达能力。

---

## 6. Post-hoc validation recovery：无 record-953 泄漏地选择 checkpoint

由于原训练没有生成 validation panel，后续执行了只读恢复：

```text
POSTHOC_VALIDATION_RECOVERY__NO_TEST_LEAKAGE
```

约束：

- 不重新训练；
- 不重跑 Direct Expert；
- checkpoint selection 完成前不加载 record 953；
- 不修改 canonical bank；
- 每个 checkpoint 都从 fresh clean S0 加载；
- 使用 frozen validation split 中 stable-hash 排序的前 8 个 edits；
- 每个 edit 评估 native、textual、visual、paired、image locality 和 text-only locality。

Validation panel IDs：

```text
1883, 1625, 549, 866, 2036, 1829, 1453, 1064
```

Panel hash：

```text
b811ff63f17cd579adcbd46ca53427b5d824699dd93e21b7f96dbffcb331ccc8
```

### 6.1 七个 checkpoint 的 validation 结果

| Step | Routed native /8 | Routed gen. /24 | Exact locality /16 | Routing FP | Contam. | Forced native /8 | Forced gen. /24 | Validation source loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 4 | 11 | 15 | 14 | 1 | 4 | 12 | 1.167118 |
| 1000 | 5 | 13 | 16 | 7 | 0 | 5 | 14 | 0.462760 |
| 1500 | 7 | 16 | 15 | 24 | 1 | 7 | 20 | 0.307832 |
| 2000 | 6 | 17 | 16 | 68 | 0 | 6 | 17 | 0.248881 |
| 2500 | 6 | 17 | 15 | 33 | 1 | 6 | 17 | 0.230653 |
| **3000** | **7** | **17** | **15** | **26** | **1** | **7** | **19** | **0.243293** |
| 3200 | 7 | 17 | 15 | 30 | 1 | 7 | 17 | 0.202435 |

按预先固定的 lexicographic rule，step 3000 被选中。step 3000 与 step 3200 的 routed native、routed generality 和 locality 相同，但 step 3000 的 routing false positives 更少（26 vs. 30）。

Selection label：

```text
LIVEEDIT_POSTHOC_VALIDATION_CHECKPOINT_SELECTED
```

关键审计：

```text
record953_used_for_selection: false
checkpoint set hash:
5d8c98c51a320ff96b78b13dd95a1291dda7ee86d147e2558f8eb9d56494d7a8
selection hash:
1ee3ece2b0cbf3779d1e54fe8697ecdfae825b67b1ff69ea63ecefe83a9a30b0
```

训练目录在恢复前后的 tree hash 完全一致：

```text
701815074dfe956ead3cb6f7319a78787188f3df090eae27cdeb1ea98ab447d9
```

---

## 7. Stage F：选定 shared generator 的 record-953 forced-on 检查

只有在 validation-only 选出 step 3000 后，才首次执行 record-953 Stage F。

约束：

- 不进行 record-specific optimization；
- 使用 step-3000 shared generator 直接生成 record-953 expert；
- 强制 expert ON，weight = 1；
- 只运行一次；
- 要求 manual no-cache / cached / HF parity。

结果：

| View | Exact unrestricted output | Success |
|---|---|---:|
| Native | `completely ectocervical and fully visible` | PASS |
| Short answer | `completely ectocervical and fully visible` | PASS |
| Textual | `completely ectocervical and fully visible` | PASS |
| Visual | `completely ectocervical and fully visible` | PASS |
| Paired | `completely ectocervical and fully visible` | PASS |

其他结果：

```text
label: LIVEEDIT_GENERATOR_FORCED_ON_PASS_SOURCE_OBJECTIVE
native three-path parity: PASS
canonical bank unchanged: true
base S0 state unchanged: true
stage_q_permitted: true
```

这一阶段是当前最强的 generator 正向证据：当正确 expert 被强制使用时，shared generator 可以在五种输入上自然生成目标答案。

---

## 8. Stage Q：完整 32-expert repository routing gate

### 8.1 Repository

构造：

```text
1 record-953 generated expert
+ 31 held-out distractor experts selected by stable hash
= 32 experts
```

未使用 similarity 选择 distractors，也没有对 record 953 进行额外优化。

```text
repository hash:
d4a7c165e7a1295b48f402beb705b378340da4de3e4fa79f2efc87d569d073d2

input manifest hash:
40fd78dfd8348b0a738382062fbcb9c3c06f8dff42e49a88a889b0650366dfcb
```

### 8.2 Positive routing

| View | Route | Target rank | Target weight | Output | Success |
|---|---|---:|---:|---|---:|
| Native | candidate `953` | 1 | 0.999991 | `completely ectocervical and fully visible` | PASS |
| Textual | candidates `953, 893, 2299` | 1 | 0.999558 | `completely ectocervical and fully visible` | PASS |
| Visual | `EMPTY_CANDIDATE_BASE_BYPASS` | — | 0 | base output: choroidal melanoma | FAIL |
| Paired | `EMPTY_CANDIDATE_BASE_BYPASS` | — | 0 | base output: gallstone | FAIL |

四个 positives 的 manual no-cache / cached / HF parity 全部通过。visual 和 paired 的失败不是生成路径不一致，而是 visual-hard routing 阶段没有产生任何候选，因此根本没有应用 record-953 expert。

### 8.3 Safety 与 locality

| Gate | 结果 |
|---|---:|
| Safety exact S0 | 27/40 |
| Safety changed behavior | 13/40 |
| Safety routed inputs | 20/40 |
| Clinical/canonical failures | 10 |
| Stop-reason mismatches | 2 |
| Record-953 target contamination | 1 |
| Maximum reported safety NLL drift | 6.511905 |
| Fixed locality exact S0 | 10/10 |

发生 target contamination 的输入：

```text
heldout_fixed_ten:2174:prototype_image
```

错误输出：

```text
completely ectocervical and fully visible
```

13 个 safety failures 主要来自两类：

1. MedMKEB held-out negatives 被 repository 中的 record-953 或 distractor experts 错误激活；
2. 使用 prototype image 配无关问题时，visual-hard 仍可能通过，而 text-soft/final weight 不足以保证精确回退 S0。

单独的 10 个 fixed locality native probes 全部走 base bypass，因此达到 10/10 exact S0。

### 8.4 Stage Q 最终判定

```text
LIVEEDIT_FULL_REPOSITORY_ROUTING_GATE_FAILURE
```

```text
positive_success:
  native: true
  textual: true
  visual: false
  paired: false

safety: 27/40
locality: 10/10
required parity: true
canonical bank unchanged: true
base state unchanged: true
```

Stage Q summary SHA-256：

```text
aa4927b50ca15e37b179fa8c71fda32a8b9b3105fe80bca8f00dd875931e64d7
```

---

## 9. 当前最可能的技术解释

### 9.1 Generator 不是唯一或首要瓶颈

如果 generator 本身完全失败，那么 forced-on Stage F 不应在 native、short、textual、visual 和 paired 上全部生成完全正确的目标。Stage F 的五项全通过说明：

- step-3000 shared generator 能为 record 953 生成有效 expert；
- residual intervention 能稳定地产生 canonical target；
- visual/paired Stage Q 失败更接近 retrieval/routing failure。

### 9.2 Visual-hard routing 存在双向问题

1. **召回不足**：alternate image 下的 visual/paired 没有任何候选；
2. **安全性不足**：相同或 prototype image 配无关问题时，visual-hard 可能仍然触发 expert。

即：换图时可能漏掉正确 expert，同图但问题不相关时又可能错误激活 expert。

### 9.3 不能根据 Stage Q 直接调一个标量阈值

观察到的 13 个 safety failures 中，最大单-expert final weight 均不超过约 0.869；native/textual positive 的 target weights 接近 1。表面上看，设置 `final_weight > 0.9` 似乎能修复当前 safety。

但 validation 中已经存在成功的正例：

```text
record 1883 paired success: max weight ≈ 0.757
record 1883 visual success: max weight ≈ 0.787
```

因此固定 0.9 阈值会删除 validation 中真实成功的 visual/paired positives。更重要的是，阈值若根据已查看的 Stage Q 结果选择，会构成测试调参。

---

## 10. 为什么当前不能直接宣称“官方 LiveEdit 复现失败”

Stage Q 是本项目在 source-objective 训练之后增加的严格 natural-generation repository gate，包含：

- 32-expert repository；
- unrestricted natural generation；
- exact token-level S0 safety；
- 40 个 equivalence-key safety negatives；
- 10 个 fixed locality probes；
- manual no-cache / cached / HF parity；
- zero target contamination。

这些要求对医疗知识编辑非常有意义，但它们是否与官方 LiveEdit 论文的主要报告协议完全同口径，尚未被证明。

尤其需要区分：

1. 官方 paper 可能主要报告 teacher-forced reliability/generality/locality；
2. 当前项目把 target-free unrestricted natural generation 设为主要成功标准；
3. 当前 safety 要求 exact token IDs，与一般 top-1 或语义 locality 指标相比更严格；
4. 当前只深入评估了 record 953，不能替代完整 benchmark aggregate；
5. backbone 和 layer-21 full-block intervention 是医疗适配，不是原论文完全相同的模型环境。

因此，当前 Stage Q failure 最直接支持的结论是：

> **在当前 LLaVA-Med 适配、当前 repository 构造和当前严格自然生成/安全 gate 下，LiveEdit 的 full-repository routing 没有通过。**

它尚不能单独证明官方代码在官方数据和官方指标下不可复现。

---

## 11. 尚未完成或不应被误写为已完成的内容

- 尚未用官方论文完全同口径的 aggregate metrics 对齐论文表格；
- 尚未证明当前 Stage Q 与官方 evaluation protocol 完全等价；
- 尚未运行多个 external target records 的完整自然生成 repository evaluation；
- 尚未运行 source-full-sequence 结果之后的 assistant-only diagnostic；
- 尚未对 router 进行任何基于 Stage Q 的修复或调参；
- 尚未重训 step-3000 generator，也没有修改 canonical bank；
- Stage Q 已被观察，后续若按其结果调参，原 40/10 集合只能作为 development/regression set，不能继续声称是全新 blind test。

---

## 12. 希望 GPT Pro 给出的具体判断

请基于上述证据回答以下问题，并明确区分“事实”“推断”和“建议”：

### A. 复现结论

1. 是否同意以下三层表述？
   - source/module reproduction：成功；
   - generator expressivity / forced-on transfer：成功；
   - full repository routing：失败。
2. 当前最严谨的论文式结论应该怎么写？

### B. 官方一致性

1. LLaVA-Med Mistral、full layer-21 block intervention、empty-candidate bypass 和 natural-generation gate 中，哪些最可能改变官方 LiveEdit 行为？
2. 当前 46/46 source parity 是否足以证明 method fidelity，还是只能证明局部张量方程 fidelity？
3. Stage Q 是否明显比官方 evaluation 更严格？如果是，应如何补跑官方同口径结果？

### C. 失败归因

1. visual/paired 的 empty-candidate failure 更像：
   - 数据 adaptation 问题；
   - visual key construction 问题；
   - sentinel calibration 问题；
   - 医疗域迁移问题；
   - repository scaling 问题？
2. safety false positives 是官方 router 的预期弱点，还是我们构建 negatives/repository 的方式可能偏离官方？

### D. 下一步实验

请给出一个最小、分阶段、无测试泄漏的执行顺序。优先考虑：

1. 先补官方同口径 evaluation；
2. 对 official implementation 与当前 port 做 end-to-end trace parity，而不仅是模块 parity；
3. 使用 validation-only 诊断 visual score、sentinel、text score 和 route decision；
4. 若确认是方法局限，再进入“改进 LiveEdit”而不是继续称为纯复现；
5. 为后续改进预留新的 blind safety/locality set。

---

## 13. 关键远端产物

```text
# 原始 run
outputs/liveedit_med_effectiveness_first_v4/
20260812T090628_direct_v4_valid/

# Upstream audit
upstream/pinned_source_manifest.json
upstream/source_parity_report.json
upstream/SOURCE_DEVIATION_LEDGER.md
upstream/SOURCE_TO_PORT_MAP.md

# Direct Expert
direct_expert/forced_on_generation.json
direct_expert/direct_expert_report.md

# Training
training/source_training_trajectory.jsonl
training/checkpoint_0500 ... checkpoint_3200

# No-leakage post-hoc validation
posthoc_validation_recovery_gpu23_20260813T023200Z/
validation_panel_manifest.json
checkpoint_hash_audit.json
checkpoint_selection.json
original_training_immutability.json

# Stage F
posthoc_validation_recovery_gpu23_20260813T023200Z/
stage_f/record953_forced_on_selected_checkpoint.json

# Stage Q
posthoc_validation_recovery_gpu23_20260813T023200Z/stage_q/
repository_audit.json
input_manifest.json
worker_0.json
worker_1.json
stage_q_summary.json
STAGE_Q_FINAL_DECISION.md
```

## 14. 建议 GPT Pro 使用的最终输出格式

请按以下结构回复：

1. **Overall verdict**：一句话判断当前是成功、部分成功还是失败；
2. **What is actually reproduced**；
3. **What is not yet reproduced**；
4. **Most likely root causes ranked**；
5. **Official-fidelity audit checklist**；
6. **Next experiment plan with promotion/stop gates**；
7. **Claims that are safe for a paper**；
8. **Claims that must not yet be made**。
