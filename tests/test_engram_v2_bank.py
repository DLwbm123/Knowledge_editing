import json

import pytest
import torch

from easyeditor.models.engram_v2 import METHOD_VERSION, SequentialEngramBankV2


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)


def _save(bank, edit_id, delta, parent, factor):
    return bank.save_edit(
        edit_id=edit_id,
        module_deltas={"linear.weight": delta},
        target_factors={"linear.weight": factor},
        parent_state_hash=parent,
        source_example_ids=[edit_id],
        target_representation_metadata={"source": "oracle_negative_gradient"},
        solver_parameters={"beta_ref": 1.0, "beta_old": 1.0},
        solver_stats={"delta_norm": float(delta.norm())},
        code_hash="code",
        config_hash="config",
    )


def test_v2_bank_serialization_replay_reload_and_anchor_rollback(tmp_path):
    model = Tiny()
    anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    model.linear.weight.data.copy_(anchor)
    bank = SequentialEngramBankV2(tmp_path / "bank")
    anchor_hash = bank.initialize_anchor({"linear.weight": anchor}, metadata={"model": "tiny"})
    delta_a = torch.tensor([[0.1, 0.0], [0.0, 0.0]])
    meta_a = _save(bank, "A", delta_a, anchor_hash, torch.tensor([[1.0], [0.0]]))
    delta_b = torch.tensor([[0.0, 0.0], [0.0, -0.2]])
    meta_b = _save(bank, "B", delta_b, meta_a["resulting_state_hash"], torch.tensor([[0.0], [1.0]]))

    assembled = bank.assemble_state()
    expected = anchor + delta_a + delta_b
    assert torch.equal(assembled["linear.weight"], expected)
    assert meta_b["method_version"] == METHOD_VERSION
    bank.assemble_state_into_model(model)
    assert torch.equal(model.linear.weight, expected)

    fresh = SequentialEngramBankV2(tmp_path / "bank")
    fresh.assemble_state_into_model(model)
    assert torch.equal(model.linear.weight, expected)
    fresh.rollback_to_prefix(model, 1)
    assert torch.equal(model.linear.weight, anchor + delta_a)
    fresh.rollback_to_prefix(model, 0)
    assert torch.equal(model.linear.weight, anchor)

    history = fresh.history_factors("linear.weight")
    assert history.shape == (2, 2)
    assert json.loads((tmp_path / "bank" / "index.json").read_text())["bank_semantics"] == "sequential_anchor_reconstruction"


def test_v2_bank_rejects_reorder_and_wrong_parent(tmp_path):
    anchor = torch.eye(2)
    bank = SequentialEngramBankV2(tmp_path / "bank")
    anchor_hash = bank.initialize_anchor({"linear.weight": anchor})
    meta = _save(bank, "A", torch.zeros_like(anchor), anchor_hash, torch.ones(2, 1))
    _save(bank, "B", torch.zeros_like(anchor), meta["resulting_state_hash"], torch.ones(2, 1))
    with pytest.raises(RuntimeError, match="ordered prefix"):
        bank.assemble_state(["B", "A"])

    wrong = SequentialEngramBankV2(tmp_path / "wrong")
    wrong.initialize_anchor({"linear.weight": anchor})
    with pytest.raises(RuntimeError, match="Parent-state hash mismatch"):
        _save(wrong, "X", torch.zeros_like(anchor), "not-the-anchor", torch.ones(2, 1))


def test_v2_bank_detects_delta_corruption(tmp_path):
    anchor = torch.eye(2)
    bank = SequentialEngramBankV2(tmp_path / "bank")
    anchor_hash = bank.initialize_anchor({"linear.weight": anchor})
    meta = _save(bank, "A", torch.ones_like(anchor), anchor_hash, torch.ones(2, 1))
    payload = torch.load(meta["tensor_path"], map_location="cpu")
    payload["deltas"]["linear.weight"][0, 0] += 1.0
    torch.save(payload, meta["tensor_path"])
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        bank.load_edit("A")
