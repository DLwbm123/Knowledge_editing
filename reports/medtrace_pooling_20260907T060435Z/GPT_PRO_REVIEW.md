# GPT Pro 审阅：CP-independent pooling，尚未启动的开发实验

## 当前状态与需要决定的问题

本次交付是**代码和启动前诊断审阅包，不是完成的实验结果**。21 个任务包、63 个 image-head fits 均未开始；没有新模型结果、新 Judge 调用或新科学结论。历史 `scientific_gain=false` 和 LoRA `QUAL_VALIDATION_FAIL` 不变。

用户要求先推送供 GPT Pro 决定。本次发布不会启动 GPU 作业、修改共享策略、停止其他任务或创建后台监控。

需要审阅的决定：是否继续保留当前空闲 GPU 保护并等待 GPU2/3 空闲，或提出明确的共享运行方案交用户确认。代码实现的保护阈值是单卡已用显存不超过 1000 MiB，**不是显存不足的证明，也不是本轮研究方案明文要求独占**。共享会改变并发、时延和 OOM 风险，不能把当前保护失败称为科学失败。

最后占用观测为 2026-09-07 06:30:56 UTC（不是本文件发布时重新测量）：

|GPU|总显存 MiB|已用 MiB|利用率|UUID|
|---|---:|---:|---:|---|
|2|40960|9783|100%|GPU-35be76e9-8ca5-1877-ddfe-27eb08f6721b|
|3|40960|9233|100%|GPU-43e3d478-7979-ea29-8130-64a467b48a5c|

容器内 `nvidia-smi` 的进程列表未显示占用者，不能据此声称 GPU 空闲，也不能确定占用任务的归属或剩余时间。UUID 与授权列表一致，实际触发的是显存占用保护。GPU1 禁止使用。

启动检查发生在模型加载前：没有 `CAMPAIGN_START.json`，21 项均 `PENDING`，尝试次数总和 0，没有本轮 worker PID。原始 `RUN_COMPLETION.json` 保留协调者写出的 initialization=FAIL、scientific_gain=not_evaluated；这指启动保护，不指训练失败。其 publication=RETRY_REQUIRED 是当时快照；本次审阅包发布状态另见 `PUBLICATION_STATUS.json`。

## 研究来源和代码

- 研究起点：`2f142769a26971b12f6e089653e4065d697fbdf6`。
- 当前实验实现：`f19ecb7b7127eaf838448b65abe293794d638593`。
- 独立研究分支：`medtrace-cp-independent-pooling-20260907T060435Z`。
- 历史公开交付：`3ce3d95e393eba8a6eda6dff0d6e7ad8ef9a9f1a`，`reports/medtrace_visual_verifier_recovery_20260907T045800Z/`。
- 公开源码快照：[`experiments/medtrace_pooling_20260907T060435Z`](https://github.com/DLwbm123/Knowledge_editing-public/tree/d4005f02008d945dcf913f90a1448b3f3ef3fe36/experiments/medtrace_pooling_20260907T060435Z)。其源码来自研究工作树，不复制研究 Git 历史。

主要文件：`methods/medtrace/visual_pooling.py`、`scripts/medtrace/run_pooling_ablation.py`、`scripts/medtrace/summarize_pooling_ablation.py`。沿用并小幅扩展原 coordinator、start binding、atomic queue、worker 和自然回放函数；旧调用默认行为不变。

## 已完成的无新模型诊断

只读取历史分数、阈值和绑定的输出，没有模型前向或 Judge 请求：

|工作点|正例 N|ON|仅 question 拒绝|仅 image 拒绝|两者拒绝|image 全 ON 时诊断覆盖上界|
|---|---:|---:|---:|---:|---:|---:|
|PRIMARY_SAFETY_FIRST|84|65|16|1|2|66|
|SECONDARY_COVERAGE90|84|65|16|1|2|66|

这些是旧 M3 的拒绝分解，不是 G0/G1/G2 成绩。上界固定旧阈值；本轮两头重新校准可能改变 question threshold，因此不能把 66/84 当作新实验不可突破的全局上界。

逐 edit/seed 分解、hard FP 分数和阈值余量、M0→M3 正负例开关得失分别见 `REJECTION_BY_EDIT_SEED.csv`、`HISTORICAL_HARD_FP_MARGINS.csv`、`HISTORICAL_M0_M3_PAIRED_TRANSITIONS.csv`。最后一项是正例激活／负例拒绝变化，不是语义正确率变化。

## 已实现与尚待实测

G0 使用旧 CP-guided 权重，G1 为均匀池化，G2 为投影前 query-conditioned 池化。三者均读取 14336 维 pre-CP 特征，复制冻结原 M3 question head，只将零初始化 image head 训练 800 步。G0 是匹配的 image-only refit，不要求等于历史联合双头 M3。

训练沿用 fit matched pairs、类别及 source-group 平衡、0.5 pair margin、Adam lr=0.01、L2=0.001；clip=1 仅用于 image head。G1/G2 池化不读取 Q。两个阈值按现有 `calibrate_decisions()` 独立校准，PRIMARY 只选择一次，SECONDARY 仅报告。

实现包含 C1/question-head hash、原缓存绑定、现场 target-free scores/decision 校验、完整 tokens/text 自然 ON/OFF 回放及端到端时延记录。**这些真实模型路径尚未通过首任务集成验证**；CPU fixture 通过不能替代它们。现场特征前向仅用于少量回放校验和时延，不重新生成全量 Base。

统计实现按 source-image-group → seed → edit 聚合，以 7 个事实作为配对 bootstrap 单位；同时保留旧 M0/M3、generality、damage 和 raw changes。完整保留仍要求对旧 M0：hard FPR 至少降 0.10、positive joint 与 T1G/T2G 各最多降 0.02、Base-correct damage 不增。

注意：当前主实验代码尚未实测；若开发门槛通过，finalizer 只会写出 `PENDING_ELIGIBILITY_AND_BUDGET`。**新 edit 的完整 C1 建模和有限确认自动执行路径尚未接通**，不能将条件确认说成已经可自动完成。确认必须遵守新 edit 来源／排除范围、全套 A0/A2/R2 Q/recovery 配方、单一冻结算法和剩余预算，不能复制其他事实 expert。

主实验预算仍为启动后最多 12 小时墙钟 / 24 GPU-hours，至少预留 2 小时评价与交付。不存在用于续跑的虚构开始时间。若恢复执行，先重查资源；不能仅依据本文件的旧 GPU 观测启动。

## 审阅边界

这是自动生成的 GPT Pro 审阅材料，不是人工签字，也不是医学 grounding、跨患者或临床验证。公开内容仅包括允许的源码、配置、CPU 检查说明、匿名分组分数／统计和阻塞状态。图像、QA/reference/raw answers、完整 tokens、features、模型权重、私有 Judge 映射、完整运行日志均未发布。尚不存在新 pooling paired results 或 generality/safety 结果，未用占位数值冒充。
