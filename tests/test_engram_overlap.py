from easyeditor.models.engram.bank import EngramBank
from easyeditor.models.engram.overlap import compute_bank_overlap, tensor_overlap
from easyeditor.models.engram.solver import EngramLayerUpdate

import torch


def test_tensor_overlap_identifies_aligned_updates():
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[2.0, 0.0]])
    c = torch.tensor([[0.0, 1.0]])
    assert tensor_overlap(a, b) > 0.99
    assert abs(tensor_overlap(a, c)) < 1.0e-6


def test_compute_bank_overlap(tmp_path):
    bank = EngramBank(tmp_path / "bank")
    update_a = EngramLayerUpdate("q_proj", weight=torch.tensor([[1.0, 0.0]]), projector=torch.eye(2), alpha=1.0)
    update_b = EngramLayerUpdate("q_proj", weight=torch.tensor([[0.0, 1.0]]), projector=torch.eye(2), alpha=1.0)
    bank.save_edit(edit_id="a", metadata={"alpha": 1.0}, updates={"q_proj": update_a})
    bank.save_edit(edit_id="b", metadata={"alpha": 1.0}, updates={"q_proj": update_b})
    report = compute_bank_overlap(tmp_path / "bank", threshold=0.1)
    assert report["pairs"]
    assert report["pairs"][0]["aggregate_overlap"] >= 0.0

