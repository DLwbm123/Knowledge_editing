import torch

from easyeditor.models.engram.bank import EngramBank
from easyeditor.models.engram.solver import EngramLayerUpdate


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 1, bias=True)


def _update():
    return EngramLayerUpdate(
        module_name="q_proj",
        weight=torch.tensor([[0.25, 0.0]]),
        bias=torch.tensor([0.1]),
        projector=torch.eye(2),
        alpha=0.5,
        stats={
            "module_name": "q_proj",
            "in_dim": 2,
            "out_dim": 1,
            "num_target_vectors": 2,
            "num_reference_vectors": 2,
            "rank_plus": 1,
            "rank_total": 2,
            "norm_W": 1.0,
            "norm_E": 0.25,
            "norm_ratio": 0.25,
        },
    )


def test_engram_bank_save_list_load_delete_export(tmp_path):
    bank = EngramBank(tmp_path / "bank")
    bank.save_edit(edit_id="edit_a", metadata={"concept_id": "c1", "alpha": 0.5}, updates={"q_proj": _update()})
    edits = bank.list_edits()
    assert edits[0]["edit_id"] == "edit_a"
    loaded = bank.load_edit("edit_a")
    assert "q_proj" in loaded["updates"]
    csv_path = bank.export_summary_csv(tmp_path / "summary.csv")
    assert csv_path.exists()
    bank.delete_edit("edit_a")
    assert bank.list_edits() == []


def test_engram_bank_apply_and_rollback(tmp_path):
    bank = EngramBank(tmp_path / "bank")
    bank.save_edit(edit_id="edit_a", metadata={"concept_id": "c1", "alpha": 0.5}, updates={"q_proj": _update()})
    model = TinyModel()
    model.q_proj.weight.data[:] = torch.tensor([[1.0, 1.0]])
    model.q_proj.bias.data[:] = torch.tensor([0.3])
    original_weight = model.q_proj.weight.detach().clone()
    original_bias = model.q_proj.bias.detach().clone()
    bank.apply_edit(model, "edit_a")
    assert not torch.allclose(model.q_proj.weight, original_weight)
    bank.rollback_edit(model, "edit_a")
    assert torch.allclose(model.q_proj.weight, original_weight)
    assert torch.allclose(model.q_proj.bias, original_bias)


def test_engram_bank_compose_updates(tmp_path):
    bank = EngramBank(tmp_path / "bank")
    bank.save_edit(edit_id="edit_a", metadata={"concept_id": "c1", "alpha": 0.5}, updates={"q_proj": _update()})
    composed = bank.compose_updates()
    assert "q_proj" in composed
    assert torch.allclose(composed["q_proj"]["weight"], -0.5 * _update().weight)
