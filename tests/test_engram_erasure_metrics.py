import torch

from easyeditor.models.engram.erasure_metrics import erasure_delta_metrics, sequence_nll_and_logprob


def test_sequence_nll_and_logprob_tracks_target_likelihood():
    labels = torch.tensor([[-100, 1, 2]])
    good = torch.zeros(1, 3, 4)
    bad = torch.zeros(1, 3, 4)
    good[0, 0, 1] = 5.0
    good[0, 1, 2] = 5.0
    bad[0, 0, 1] = -1.0
    bad[0, 1, 2] = -1.0
    before = sequence_nll_and_logprob(good, labels)
    after = sequence_nll_and_logprob(bad, labels)
    metrics = erasure_delta_metrics(target_before=before, target_after=after)
    assert metrics["erase_target_nll_after"] > metrics["erase_target_nll_before"]
    assert metrics["erase_success_logprob_drop"] > 0
    assert metrics["erase_logprob_metrics_available"] is True


def test_erasure_metrics_unavailable_is_explicit():
    metrics = erasure_delta_metrics(unavailable_reason="wrapper did not return logits")
    assert metrics["erase_logprob_metrics_available"] is False
    assert "wrapper" in metrics["erase_logprob_unavailable_reason"]
