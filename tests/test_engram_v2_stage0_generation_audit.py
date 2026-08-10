from types import SimpleNamespace

import torch

from scripts.engram.stage0_generation_audit_utils import (
    CanonicalInputs,
    assert_no_gold_leakage,
    classify_termination,
    first_supervised_position,
    incremental_mean_nll,
    manual_greedy_trace,
    medical_answer_match,
    score_target_incrementally,
)


class ToyTokenizer:
    eos_token_id = 4
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        table = {0: "<pad>", 1: "a", 2: "b", 3: "c", 4: "</s>"}
        values = [table[int(item)] for item in ids]
        if skip_special_tokens:
            values = [item for item in values if item not in {"<pad>", "</s>"}]
        return "".join(values)


class ToyLM(torch.nn.Module):
    def forward(self, input_ids, images=None, attention_mask=None, return_dict=True, use_cache=False):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 5), -5.0, device=input_ids.device)
        for row in range(batch):
            for index in range(length):
                current = int(input_ids[row, index])
                next_id = {0: 1, 1: 2, 2: 4, 3: 4, 4: 4}.get(current, 1)
                logits[row, index, next_id] = 5.0
        return SimpleNamespace(logits=logits)


class ToyWrapper:
    def __init__(self):
        self.llava_model = ToyLM()
        self.llava_tokenizer = ToyTokenizer()


def canonical(prompt=(0,), target=(1, 2, 4)):
    prompt_ids = torch.tensor([prompt], dtype=torch.long)
    full_ids = torch.tensor([prompt + target], dtype=torch.long)
    return CanonicalInputs(
        prompt_text="prompt",
        full_text="prompt answer",
        prompt_ids=prompt_ids,
        full_ids=full_ids,
        image=torch.zeros(1, 3, 2, 2),
        answer_start=len(prompt),
        target_ids=torch.tensor(target, dtype=torch.long),
        prompt_hash="prompt",
        full_hash="full",
        pixel_hash="pixel",
    )


def test_prompt_prefix_contract():
    item = canonical()
    assert torch.equal(item.full_ids[:, : item.answer_start], item.prompt_ids)


def test_no_gold_answer_leakage_guard():
    item = canonical()
    assert_no_gold_leakage(item.prompt_ids, item)
    leaked = torch.cat([item.prompt_ids, item.target_ids[:1].view(1, 1)], dim=1)
    try:
        assert_no_gold_leakage(leaked, item)
    except AssertionError:
        pass
    else:
        raise AssertionError("gold suffix was not rejected")


def test_first_target_token_is_first_supervised():
    labels = torch.tensor([[-100, -100, 7, 8]])
    assert first_supervised_position(labels, -100) == 2


def test_incremental_nll_matches_shifted_causal_nll():
    model = ToyWrapper()
    item = canonical()
    rows = score_target_incrementally(model, item, [4])
    logits = model.llava_model(item.full_ids).logits[:, :-1]
    labels = item.full_ids[:, 1:]
    mask = torch.arange(labels.shape[1]) >= item.answer_start - 1
    direct = torch.nn.functional.cross_entropy(logits[0, mask], labels[0, mask])
    assert abs(incremental_mean_nll(rows) - float(direct)) < 1e-7


def test_raw_manual_and_greedy_first_token_agree():
    model = ToyWrapper()
    item = canonical()
    raw = int(model.llava_model(item.prompt_ids).logits[0, -1].argmax())
    manual = manual_greedy_trace(model, item, 8, [4])
    assert raw == manual["token_ids"][0] == 1


def test_cap_hit_and_early_eos_and_termination_classification():
    assert classify_termination([1, 1, 1], [1, 2], [4], 3)["cap_hit"]
    assert classify_termination([4], [1, 4], [4], 3)["early_eos_failure"]
    assert classify_termination([1, 2], [1, 2], [4], 3)["termination_failure"]


def test_negated_output_does_not_match_positive_target():
    result = medical_answer_match("no pneumothorax", "pneumothorax")
    assert not result["normalized_exact_match"]
    assert not result["clinical_constraint_match"]


def test_laterality_and_modifier_checks():
    wrong = medical_answer_match("right pleural effusion", "left pleural effusion", required_laterality="left")
    right = medical_answer_match("left pleural effusion", "left pleural effusion", required_laterality="left")
    assert not wrong["clinical_constraint_match"]
    assert right["clinical_constraint_match"]


def test_reconstructed_tokens_and_metrics_are_identical():
    model = ToyWrapper()
    item = canonical()
    first = score_target_incrementally(model, item, [4])
    reconstructed = score_target_incrementally(model, item, [4])
    assert first == reconstructed


def test_rollback_restores_prior_output():
    model = ToyWrapper()
    item = canonical()
    prior = manual_greedy_trace(model, item, 8, [4])
    snapshot = {name: value.clone() for name, value in model.llava_model.state_dict().items()}
    model.llava_model.load_state_dict(snapshot)
    restored = manual_greedy_trace(model, item, 8, [4])
    assert prior["token_ids"] == restored["token_ids"]
    assert prior["trajectory"] == restored["trajectory"]
