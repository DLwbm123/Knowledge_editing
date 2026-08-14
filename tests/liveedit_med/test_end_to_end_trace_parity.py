import torch

from methods.liveedit_med.trace_parity import compare_tensor, summarize_trace


def test_trace_summary_requires_every_named_boundary():
    row = compare_tensor("hidden", torch.ones(2), torch.ones(2), atol=0, rtol=0)
    assert summarize_trace([row], required_names=["hidden"])["all_passed"]
    assert not summarize_trace([row], required_names=["hidden", "logits"])["all_passed"]
