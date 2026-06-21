import torch

from easyeditor.models.engram.covariance import (
    loss_predictor_mask_from_labels,
    token_scope_mask_from_batch,
)


def test_loss_predictor_selects_previous_positions_for_causal_loss():
    labels = torch.tensor([[-100, -100, 11, 12, 2, -100]])
    attention_mask = torch.ones_like(labels, dtype=torch.bool)

    mask, diag = loss_predictor_mask_from_labels(labels, attention_mask=attention_mask)

    assert mask.tolist() == [[False, True, True, True, False, False]]
    assert diag["raw_label_answer_positions"] == [[2, 3, 4]]
    assert diag["shifted_loss_positions"] == [[1, 2, 3]]
    assert diag["num_selected_tokens"] == 3
    assert diag["fallback_used"] is False


def test_token_scope_loss_predictor_uses_labels_when_available():
    labels = torch.tensor([[-100, -100, 5, 6]])
    batch = {
        "labels": labels,
        "attention_mask": torch.ones_like(labels, dtype=torch.bool),
        "answer_mask": torch.tensor([[False, False, True, True]]),
    }

    mask, diag = token_scope_mask_from_batch(batch, "loss_predictor")

    assert mask.tolist() == [[False, True, True, False]]
    assert diag["raw_label_answer_positions"] == [[2, 3]]
    assert diag["shifted_loss_positions"] == [[1, 2]]
    assert diag["num_selected_tokens"] == 2
    assert diag["fallback_used"] is False


def test_token_scope_loss_predictor_warnable_fallback_shifts_answer_mask():
    answer_mask = torch.tensor([[False, False, True, True, False]])
    batch = {
        "answer_mask": answer_mask,
        "attention_mask": torch.ones_like(answer_mask, dtype=torch.bool),
    }

    mask, diag = token_scope_mask_from_batch(batch, "loss_predictor", mask_fallback="all")

    assert mask.tolist() == [[False, True, True, False, False]]
    assert diag["raw_label_answer_positions"] == [[2, 3]]
    assert diag["shifted_loss_positions"] == [[1, 2]]
    assert diag["num_selected_tokens"] == 2
    assert diag["fallback_used"] is True
    assert "labels unavailable" in diag["fallback_reason"]


def test_prompt_last_selects_final_prompt_token_before_answer():
    batch = {
        "prompt_mask": torch.tensor([[True, True, False, False]]),
        "answer_mask": torch.tensor([[False, False, True, True]]),
        "attention_mask": torch.tensor([[True, True, True, True]]),
    }

    mask, diag = token_scope_mask_from_batch(batch, "prompt_last")

    assert mask.tolist() == [[False, True, False, False]]
    assert diag["num_selected_tokens"] == 1
