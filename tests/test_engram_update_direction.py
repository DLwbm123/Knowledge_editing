import torch

from easyeditor.models.engram.bank import EngramBank
from easyeditor.models.engram.solver import EngramLayerUpdate, apply_update_to_module


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 1, bias=True)


def _set_known_params(layer: torch.nn.Linear) -> None:
    layer.weight.data[:] = torch.tensor([[1.0, -2.0]])
    layer.bias.data[:] = torch.tensor([0.5])


def _update(*, alpha: float = 0.5, direction: str = "subtract", sign: int = -1) -> EngramLayerUpdate:
    return EngramLayerUpdate(
        module_name="q_proj",
        weight=torch.tensor([[0.25, -0.5]]),
        bias=torch.tensor([0.125]),
        projector=torch.eye(2),
        alpha=alpha,
        beta=0.0,
        engram_update_direction=direction,
        direction_sign=sign,
        paper_direction_equivalent="paper_style_W_minus_alpha_E"
        if sign < 0
        else "equivalent_to_paper_subtract_with_signed_alpha_negative",
        stats={
            "module_name": "q_proj",
            "norm_ratio": 0.25,
            "effective_norm_ratio": abs(alpha) * 0.25,
            "effective_update_norm_ratio": abs(alpha) * 0.25,
            "engram_update_direction": direction,
            "direction_sign": sign,
        },
    )


def test_subtract_direction_matches_old_w_minus_alpha_e():
    layer = torch.nn.Linear(2, 1, bias=True)
    _set_known_params(layer)
    apply_update_to_module(layer, _update(alpha=0.5, direction="subtract", sign=-1), direction=-1)
    assert torch.equal(layer.weight, torch.tensor([[0.875, -1.75]]))
    assert torch.equal(layer.bias, torch.tensor([0.4375]))


def test_add_direction_matches_w_plus_alpha_e():
    layer = torch.nn.Linear(2, 1, bias=True)
    _set_known_params(layer)
    apply_update_to_module(layer, _update(alpha=0.5, direction="add", sign=1), direction=-1)
    assert torch.equal(layer.weight, torch.tensor([[1.125, -2.25]]))
    assert torch.equal(layer.bias, torch.tensor([0.5625]))


def test_add_alpha_matches_old_negative_signed_alpha():
    layer_add = torch.nn.Linear(2, 1, bias=True)
    layer_old_signed = torch.nn.Linear(2, 1, bias=True)
    _set_known_params(layer_add)
    _set_known_params(layer_old_signed)
    apply_update_to_module(layer_add, _update(alpha=0.5, direction="add", sign=1), direction=-1)
    apply_update_to_module(layer_old_signed, _update(alpha=-0.5, direction="subtract", sign=-1), direction=-1)
    assert torch.equal(layer_add.weight, layer_old_signed.weight)
    assert torch.equal(layer_add.bias, layer_old_signed.bias)


def test_bank_metadata_saves_and_loads_direction_sign(tmp_path):
    bank = EngramBank(tmp_path / "bank")
    bank.save_edit(
        edit_id="edit_add",
        metadata={"concept_id": "c1"},
        updates={"q_proj": _update(alpha=0.5, direction="add", sign=1)},
    )
    loaded = bank.load_edit("edit_add")
    assert loaded["metadata"]["engram_update_direction"] == "add"
    assert loaded["metadata"]["direction_sign"] == 1
    assert loaded["updates"]["q_proj"]["engram_update_direction"] == "add"
    assert loaded["updates"]["q_proj"]["direction_sign"] == 1


def test_bank_compose_respects_per_edit_direction_sign(tmp_path):
    bank = EngramBank(tmp_path / "bank")
    subtract = _update(alpha=0.5, direction="subtract", sign=-1)
    add = _update(alpha=0.25, direction="add", sign=1)
    bank.save_edit(edit_id="edit_subtract", metadata={"concept_id": "c1"}, updates={"q_proj": subtract})
    bank.save_edit(edit_id="edit_add", metadata={"concept_id": "c2"}, updates={"q_proj": add})

    composed = bank.compose_updates(["edit_subtract", "edit_add"])
    expected_weight = -0.5 * subtract.weight + 0.25 * add.weight
    expected_bias = -0.5 * subtract.bias + 0.25 * add.bias
    assert torch.equal(composed["q_proj"]["weight"], expected_weight)
    assert torch.equal(composed["q_proj"]["bias"], expected_bias)

    model = TinyModel()
    _set_known_params(model.q_proj)
    original_weight = model.q_proj.weight.detach().clone()
    original_bias = model.q_proj.bias.detach().clone()
    bank.apply_edit(model, "edit_subtract")
    bank.apply_edit(model, "edit_add")
    assert torch.equal(model.q_proj.weight, original_weight + expected_weight)
    assert torch.equal(model.q_proj.bias, original_bias + expected_bias)


def test_bank_rollback_restores_exact_weights(tmp_path):
    bank = EngramBank(tmp_path / "bank")
    bank.save_edit(
        edit_id="edit_add",
        metadata={"concept_id": "c1"},
        updates={"q_proj": _update(alpha=0.5, direction="add", sign=1)},
    )
    model = TinyModel()
    _set_known_params(model.q_proj)
    original_weight = model.q_proj.weight.detach().clone()
    original_bias = model.q_proj.bias.detach().clone()
    bank.apply_edit(model, "edit_add")
    bank.rollback_edit(model, "edit_add")
    assert torch.equal(model.q_proj.weight, original_weight)
    assert torch.equal(model.q_proj.bias, original_bias)
