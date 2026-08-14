import torch

from methods.liveedit_med.official_metric_compat import locality_preservation, teacher_forced_accuracy


def test_teacher_forced_causal_shift_and_mask():
    labels = torch.tensor([[-100, 4, 5, -100]])
    logits = torch.zeros(1, 4, 8)
    logits[0, 0, 4] = 2
    logits[0, 1, 5] = 2
    result = teacher_forced_accuracy(logits, labels)
    assert result["exact"] and result["correct_tokens"] == result["total_tokens"] == 2
    assert result["predicted_token_ids"] == [4, 5]


def test_locality_compares_post_to_pre_not_reference():
    labels = torch.tensor([[-100, 4, 5]])
    pre = torch.zeros(1, 3, 8); post = torch.zeros_like(pre)
    pre[0, 0, 7] = post[0, 0, 7] = 2
    pre[0, 1, 6] = post[0, 1, 6] = 2
    result = locality_preservation(pre, post, labels)
    assert result["exact"] and result["pre_token_ids"] == [7, 6]
