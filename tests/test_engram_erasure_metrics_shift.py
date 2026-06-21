import math

import torch

from easyeditor.models.engram.erasure_metrics import sequence_nll_and_logprob


class TinyTokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return ["<eos>" if int(token_id) == 2 else f"T{int(token_id)}" for token_id in token_ids]


def test_shifted_logits_score_answer_tokens_and_ignore_prompt_tokens():
    labels = torch.tensor([[-100, -100, 5, 6, 2]])
    logits = torch.zeros(1, 5, 8)
    logits[0, 0, 7] = 10.0  # ignored prompt position would dominate if -100 masking failed.
    logits[0, 1, 5] = 10.0
    logits[0, 2, 6] = 10.0
    logits[0, 3, 2] = 10.0
    logits[0, 4, 1] = 10.0  # raw EOS label position must not be used to score EOS.

    metrics = sequence_nll_and_logprob(logits, labels, tokenizer=TinyTokenizer())

    assert metrics["shift_applied"] is True
    assert metrics["num_tokens"] == 3
    assert metrics["answer_token_count"] == 3
    assert metrics["answer_token_ids"] == [5, 6, 2]
    assert metrics["answer_tokens"] == ["T5", "T6", "<eos>"]
    assert metrics["ignored_token_count"] == 1
    assert metrics["nll"] < 1.0e-3
    assert metrics["logprob"] < 0.0


def test_logprob_and_nll_signs_follow_answer_probability():
    labels = torch.tensor([[-100, -100, 3]])
    high_probability = torch.zeros(1, 3, 6)
    low_probability = torch.zeros(1, 3, 6)
    high_probability[0, 1, 3] = 8.0
    low_probability[0, 1, 3] = -8.0
    low_probability[0, 1, 4] = 8.0

    before = sequence_nll_and_logprob(high_probability, labels)
    after = sequence_nll_and_logprob(low_probability, labels)

    assert after["nll"] > before["nll"]
    assert after["logprob"] < before["logprob"]
    assert before["logprob"] - after["logprob"] > 0.0


def test_multi_token_sum_and_average_are_consistent_with_eos_counted():
    labels = torch.tensor([[-100, -100, 1, 2]])
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 9.0],
            ]
        ]
    )

    metrics = sequence_nll_and_logprob(logits, labels, tokenizer=TinyTokenizer())

    assert metrics["answer_token_ids"] == [1, 2]
    assert metrics["answer_tokens"] == ["T1", "<eos>"]
    assert metrics["answer_token_count"] == 2
    assert math.isclose(metrics["nll"], -metrics["logprob"] / metrics["answer_token_count"], rel_tol=1.0e-6)
