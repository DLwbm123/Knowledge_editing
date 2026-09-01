# C0R 本地图像感知审阅说明

## 审阅前提

审阅者必须满足以下任一条件：

1. 已通过相应数据集官方来源独立取得数据并接受其条款；或
2. 在已获授权的 my-gpu 数据环境内进行原位审阅。

不得把图像上传到第三方模型、云盘、聊天工具、公开网页或 GitHub。不得截图、下载缩略图或建立新的图像副本。

## 审阅对象

本轮审阅的是 200 个 formal edit targets，而不是方法输出。每项只显示：

```text
opaque review_id
source image
question
target/reference
解释答案所必需的最小 task metadata
```

不会显示：

```text
formal position
record ID
方法或匿名组
模型原始回答
上一轮 audit 判定
是否属于异常项
replacement 建议
```

## 核心问题

> 在给定图像和问题的条件下，该 target/reference 是否构成有效答案，并达到问题要求的语义和视觉特异性？

## 判定顺序

1. 明确问题询问的属性：实体、位置、左右侧、数量、模态、状态或其他属性。
2. 判断 target 是直接答案、可接受视觉指代、可接受同义表达、依赖上下文但合理，还是不匹配。
3. 检查 anatomy、laterality、count、modality、state 和 specificity。
4. 不因语法生硬自动判错；数据集答案可能是短语或视觉指代。
5. 非答案、错误实体、错误位置、错误字段或与图像明显冲突时判 invalid。
6. 无法可靠判断时选择 uncertain/manual_review，不猜测。
7. 不参考其他样本、模型输出或方法性能。

## 输出字段

```json
{
  "review_id": "target_review_0001",
  "valid": true,
  "confidence": "high",
  "relation": "direct_answer",
  "issue_type": "none",
  "recommended_action": "retain",
  "reason": "The answer directly identifies the structure requested by the question."
}
```

允许枚举：

```text
confidence:
  high | medium | low

relation:
  direct_answer
  acceptable_visual_deixis
  acceptable_synonym
  context_dependent
  ambiguous
  mismatch

issue_type:
  none
  question_reference_mismatch
  wrong_source_field
  wrong_entity
  wrong_location
  wrong_count
  wrong_modality
  under_specific
  over_specific
  ambiguous_question
  ambiguous_reference
  duplicate_annotation
  other

recommended_action:
  retain | repair | exclude | manual_review
```

## 审阅质量控制

- Reviewer A 审阅全部 200 项。
- Reviewer B 复核隐藏的三个重点项、Reviewer A 的所有非高置信有效项和 20 个随机控制项。
- 两位审阅者不得交流具体 item 结论，直到各自输出被冻结。
- 分歧项交给 adjudicator。
- 审阅工具在显示前必须验证图像 SHA-256；hash 不一致时不得继续该 item。

## 禁止行为

```text
上传图像到任何外部 VLM/API
把图像放入 review ZIP
截图或复制图像
查看 formal position 或方法身份
修改 source dataset
根据模型输出推断 gold
跳过 uncertain 项并强制二选一
```
