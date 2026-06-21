import torch

from easyeditor.models.engram.erasure_metrics import safe_model_answer_nll_and_logprob


class MockCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.seen_images = []

    def forward(self, sample):
        labels = sample["labels"]
        self.seen_images.append(sample.get("image"))
        logits = torch.zeros(labels.shape[0], labels.shape[1], self.vocab_size)
        shifted = labels[:, 1:].masked_fill(labels[:, 1:].eq(-100), 0)
        for batch_idx in range(labels.shape[0]):
            for pos in range(labels.shape[1] - 1):
                logits[batch_idx, pos, int(shifted[batch_idx, pos])] = 4.0
        return {"logits": logits, "labels": labels}


class MissingLogitsModel(torch.nn.Module):
    def forward(self, sample):
        return {"labels": sample["labels"]}


def test_mock_causal_lm_metrics_work_for_text_image_target_sample():
    model = MockCausalLM()
    sample = {
        "prompt": "Question: What condition is shown?",
        "image": torch.zeros(1, 3, 4, 4),
        "labels": torch.tensor([[-100, 3, 4, -100]]),
    }
    metrics = safe_model_answer_nll_and_logprob(model, sample)
    assert metrics["available"] is True
    assert metrics["num_tokens"] == 2
    assert metrics["nll"] >= 0
    assert metrics["logprob"] <= 0
    assert model.seen_images[-1] is sample["image"]


def test_mock_causal_lm_metrics_work_for_multimodal_locality_sample():
    model = MockCausalLM()
    sample = {
        "prompt": "Question: What organ is visible?",
        "image": torch.ones(1, 3, 4, 4),
        "labels": torch.tensor([[-100, 2, -100]]),
    }
    metrics = safe_model_answer_nll_and_logprob(model, sample)
    assert metrics["available"] is True
    assert metrics["num_tokens"] == 1
    assert model.seen_images[-1] is sample["image"]


def test_missing_logits_reports_unavailable():
    metrics = safe_model_answer_nll_and_logprob(
        MissingLogitsModel(),
        {"labels": torch.tensor([[-100, 1]])},
    )
    assert metrics["available"] is False
    assert "logits" in metrics["unavailable_reason"]
