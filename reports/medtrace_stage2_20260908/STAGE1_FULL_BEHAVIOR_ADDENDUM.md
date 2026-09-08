# Stage1 完整行为补充说明

日期：2026-09-08。分析性质：**事后描述，不重新选型**。

本说明使用已完成的 Stage1 `20260907_selective_write_v1_r01`：7 个编辑、70 条训练轨迹、7 个 A2 参考。没有重训、重判、搜索 checkpoint 或修改旧报告。既有 V1 主分析、P4 W1 λ=.1 qualified 标记及 L16 W1 技术回退性质均保留。

## 文件与来源

- [全角色汇总 CSV](STAGE1_FULL_BEHAVIOR.csv)：495 行，11 条件 × 3 模式 × 15 个 role/stratum 面板。完整保留原宏平均指标，追加支持数。
- [逐编辑支持数 CSV](STAGE1_FULL_BEHAVIOR_SUPPORT_BY_EDIT.csv)：输入、来源组、编辑及 Base-correct 分母，不包含 query ID、EqKey、图像路径或 QA。
- [审计证据 JSON](STAGE1_FULL_BEHAVIOR_AUDIT.json)：源文件名称、字段定义、绑定核对计数、T2G micro 保持/丢失数及全局支持数。
- [Stage1 公开锚点](https://github.com/DLwbm123/Knowledge_editing-public/tree/230454602e2c5441a3ecd87ff7eb130b32e46a7e/reports/medtrace_selective_write_20260907)：原始公开分析不覆盖。

宏平均来源是原 `SELECTIVE_WRITE_BEHAVIOR_MACRO.csv`；支持数来自 A2 参考输出内保存的输入行与已有 Judge 映射。由于跨条件 Base 与输入一致，支持数适用于全部条件和模式。这里只导出汇总，不公开私有输入、答案、生成文本、tokens、映射键或权重。

## 主要结论

**未发现原 26 条 T2G 的问题、图像、参考答案或 Judge 串绑。P4 W1 的保护收益伴随明显的原 T2G 泛化下降，应作为已有输出中的真实行为取舍保留。**

以下为 FORCED_ON，正确率和破坏率单位 %；KL 无量纲。完整 native、fit、calibration、evaluation、T1G、T2G、T1L、H、U、challenge 以及 FIXED_ROUTER/DISABLED 结果均在配套 CSV。

| 条件 | eval 正例 | 原 T2G | 原 T1L | H eval KL | U eval KL | U Base-correct 破坏率 |
|---|---:|---:|---:|---:|---:|---:|
| A2 | 100.00 | 76.19 | 13.33 | 0.376737 | 0.028378 | 25.77 |
| P4 W0 | 100.00 | 80.95 | 13.33 | 0.537489 | 0.032869 | 31.46 |
| P4 W1 λ=0.1 | 100.00 | 25.00 | 13.33 | 0.106724 | 0.005395 | 10.63 |
| P4 W1 λ=1 | 82.14 | 14.29 | 13.33 | 0.047850 | 0.003320 | 6.80 |
| P4 W1 λ=10 | 25.00 | 4.76 | 13.33 | 0.016365 | 0.001751 | 2.38 |
| P4 W2 | 100.00 | 61.90 | 13.33 | 0.226265 | 0.009379 | 10.63 |
| L16 W0 | 100.00 | 84.52 | 13.33 | 1.983764 | 0.428426 | 77.72 |
| L16 W1 λ=0.1 | 78.57 | 38.10 | 40.00 | 0.087837 | 0.004129 | 8.25 |
| L16 W1 λ=1 | 57.14 | 17.86 | 40.00 | 0.079907 | 0.004571 | 10.29 |
| L16 W1 λ=10 | 46.43 | 11.90 | 46.67 | 0.026916 | 0.004171 | 12.67 |
| L16 W2 | 100.00 | 69.05 | 13.33 | 0.249876 | 0.005614 | 10.63 |

- P4 W1 λ=.1 的 eval 正例为 100%，但 T2G 宏平均只有 25%。A2 答对 20/26；P4 W1 λ=.1 答对 7/26，丢失 13 条 A2 成功、没有新增成功。7/26 是 micro，不能与 25% 宏平均混用。
- P4 W2 和 L16 W2 分别保留 16/26、18/26，分别丢失 4、2 条 A2 成功。不能只凭 KL 较低就宣布某方案完整行为最佳。
- 固定路由下 A2/P4 W0/P4 W1 λ=.1/P4 W2/L16 W2 的 T2G 宏平均为 50%/50%/14.29%/46.43%/50%；路由没有消除全部差异。
- 原 T1L 面板的 Base 正确率为 60%，上述主要候选均为 13.33%。此面板与新构造 H 不同，不能用 H 的保持结果替代。
- 高 λ 常降低负例 KL，同时损害正例与 T2G；原校准结论只对原规则成立。不得将本次已可见 T2G 改称盲测、转入 fit 或用来重选 λ/checkpoint。
- Stage1 仍仅有 7 个既有开发编辑，不是完整 M3Bench 复现或新编辑确认。

## 全角色分母

下表每个面板只计一次，不随 11 条件/3 模式重复膨胀。来源图像按已存 image_path 去重，不等同患者数，也没有做图像内容去重。

| role / stratum | 输入行 | 去重图像-问题 | 来源图像 | 编辑×原来源组 | 编辑 | Base-correct 行 | Base-correct 去重输入 | Base-correct 编辑 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration / H | 96 | 92 | 23 | 24 | 7 | 12 | 10 | 3 |
| calibration / U | 116 | 115 | 98 | 116 | 7 | 44 | 44 | 7 |
| calibration / positive | 28 | 28 | 7 | 7 | 7 | 2 | 2 | 1 |
| challenge / same_image_other_fact_challenge | 26 | 26 | 3 | 3 | 3 | 9 | 9 | 2 |
| evaluation / H | 112 | 112 | 27 | 28 | 7 | 4 | 4 | 1 |
| evaluation / U | 112 | 109 | 90 | 112 | 7 | 44 | 43 | 7 |
| evaluation / positive | 28 | 28 | 7 | 7 | 7 | 0 | 0 | 0 |
| fit / H | 120 | 120 | 24 | 24 | 7 | 13 | 13 | 4 |
| fit / U | 116 | 114 | 104 | 116 | 7 | 42 | 42 | 7 |
| fit / positive | 35 | 35 | 7 | 8 | 7 | 0 | 0 | 0 |
| fit / scope_fit_positive_not_used_for_positive_CE | 4 | 4 | 1 | 1 | 1 | 0 | 0 | 0 |
| formal_development / T1G | 26 | 26 | 26 | 26 | 7 | 0 | 0 | 0 |
| formal_development / T1L | 15 | 5 | 5 | 15 | 3 | 9 | 3 | 3 |
| formal_development / T2G | 26 | 26 | 7 | 7 | 7 | 0 | 0 | 0 |
| native / positive | 7 | 7 | 7 | 7 | 7 | 0 | 0 | 0 |

重要解释：

1. **H-eval**：112 行，Base-correct 仅 4 行，位于 2 张已存来源图像、1 个编辑 e01。除了 P4 W1 λ=10 的 damage 为 66.67%，其余条件均为 0；这个很窄的支持集不能证明广泛保持。
2. **U-eval**：112 个事件-查询行，对应 109 个去重图像-问题输入、90 张来源图像。Base-correct 为 44 行，但跨编辑去重后为 43 个输入、42 张图像。各编辑 Base-correct 行数为 e01/e02/e03/e07/e13/e15/e16 = 12/7/4/8/4/6/3。
3. **T1L**：15 行实际来自 5 个去重输入，在 3 个编辑下重复评估。Base-correct 为 9 个事件-查询行、3 个去重输入，不是 15 个独立来源问题。
4. **fit 正例**：35 行之外，还有 e01 的 4 条 `scope_fit_positive_not_used_for_positive_CE`，必须独立列出。保存的 fit-positive 来源组为 8 个编辑×来源组而图像只有 7 张，反映既有来源组命名空间，不应在本补充说明中悄悄重新分组。
5. 破坏率只对 Base-correct 输入定义。宏平均先按保存的 EqKey/来源组/编辑计算，缺少 Base-correct 的编辑不按零计入。不能将宏平均百分比乘 44 推导错误题数。
6. CSV 空数值表示该指标不可用，不是零。原 T1L 虽被 legacy 行标注为 positive，现有汇总按 task 正确分层；它没有负例 KL。正例 CE 仅使用 native 与获准 fit paraphrases，没有 T1L/T2G 训练泄漏。

## 原 26 条 T2G 与 28 条 evaluation-positive 绑定

核对链：V4 task-specific 单事件目录 → 获准 DEV16 输入 → frozen_data → Stage1 per-edit 输入 → raw 输出 → 原 Judge packet/verdict。

- 7/7 个编辑事件与 frozen_data、获准 DEV16 输入完全一致。
- 26/26 条 T2G 的 query、问题、图像、参考、task、edit ID 对应正确；完整 probe 列表与各自 T2G 单事件一致。
- 26/26 使用本编辑原图、参考等于编辑目标；问题均不同于 native。与同编辑 fit（含获准 A2 paraphrases）及 evaluation-positive 没有精确问题重叠。此处只证明精确不重叠，不宣称语义独立。
- 28/28 条 evaluation-positive 都有唯一冻结问题和批准状态，图像与目标正确。e01 的 4 条对应此前冻结数据及原审核候选；其余 24 条精确等于 6 个编辑的原问题加四种固定 evaluation 前缀。
- 这两组测试问法来源不同：不能将包装式正例 100% 外推成所有来源/表达形式泛化 100%。批准状态指既有记录，不新增独立临床等价性背书。

### 已解释的位置命名空间差异

| 编辑 | T2G 探针 | eval 正例 | T0 位置 | T2G 所属事件位置 |
|---|---:|---:|---:|---:|
| e01 | 3 | 4 | 27 | 72 |
| e02 | 4 | 4 | 35 | 113 |
| e03 | 3 | 4 | 48 | 172 |
| e07 | 4 | 4 | 66 | 153 |
| e13 | 4 | 4 | 143 | 17 |
| e15 | 4 | 4 | 168 | 74 |
| e16 | 4 | 4 | 172 | 32 |

`calibration_selection_v2()` 按 record_id 把 T1G/T2G/T1L 的 probes 合入 T0 父事件，并保留 task-specific 位置。因此 T2G 的 sequence_position 与 T0/顺序目录不同，但 26/26 均匹配自己的 T2G 单事件。应按 task、edit_id、probe_id 连接，不能强制不同任务的位置相等。**这是已解释的元数据差异，不是需要修分的错误。**

## raw、Judge 与汇总一致性

| 检查 | 观察结果 |
|---|---|
| 任务完成 | 70/70 JUDGED |
| 结果文件与准确输出键覆盖 | 77/77，含 7 个 A2 参考 |
| raw row 与冻结输入/获准 fit 扩展 | 9,537/9,537 完全一致 |
| 保存的 disabled parity | 9,537/9,537 为真；没有重新运行模型 |
| fixed 实际输出与预期分支 | 9,537/9,537 一致 |
| 跨条件 Base 或 fixed-on 漂移 | 均为 0 |
| Judge packet/唯一 verdict/预期集合 | 均为 5,435，键集合相等 |
| Judge 布尔输出、协议及 snapshot | 5,435/5,435 一致有效 |
| 既有 Judge tuple 键重建 | 5,435/5,435 一致 |
| raw 分支与完整 question/reference/answer tuple 连接 | 66,759 次通过 |
| 独立重算 | 495 个单元 × 六项指标，差异为 0，绝对容差 1e-12 |

六项独立重算指标为 semantic、base_correct、target_consistency、KL、base_correct_damage、fixed_on。其余原有指标也完整复制到 CSV，但不宣称本轮逐项独立重算。没有重判，也没有重新计算文件级哈希；Judge 键重建只是原协议映射验证。

审计不支持“所有潜在模型实现问题都被排除”的强断言；它支持没有发现上述数据绑定、既有 Judge 映射和六项汇总计算错误。无需修改既有分数，主任务可直接带入这一完整行为描述。
