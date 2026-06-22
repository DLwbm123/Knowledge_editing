import torch

from scripts.engram.run_medmkeb_routed_edit_bank import RoutedLoraPatch, route_edit_ids


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edit = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.edit.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def forward(self, x):
        return self.edit(x)


def _entry(edit_id, record_id, proto):
    return {
        "edit_id": edit_id,
        "record_id": record_id,
        "target_prototype": torch.tensor(proto, dtype=torch.float32),
        "reference_prototype": torch.tensor(proto, dtype=torch.float32),
        "contrastive_prototype": torch.tensor(proto, dtype=torch.float32),
    }


def test_routed_lora_patch_rolls_back_forward_and_weights():
    model = TinyModel()
    sample = torch.tensor([[2.0, 3.0, 5.0]])
    before_weight = model.edit.weight.detach().clone()
    before_output = model(sample).detach().clone()
    entry = {
        "edit_id": "edit-a",
        "record_id": "rid-a",
        "beta": 1.0,
        "factors": {
            "edit": {
                "A": torch.tensor([[0.0, 0.0, 1.0]]),
                "B": torch.tensor([[1.0], [0.0]]),
                "scale": 0.5,
            }
        },
    }

    patch = RoutedLoraPatch(model, [entry], [1.0])
    patch.install()
    patched_output = model(sample).detach().clone()
    patch.remove()
    after_output = model(sample).detach().clone()

    assert torch.equal(model.edit.weight, before_weight)
    assert torch.equal(after_output, before_output)
    assert not torch.equal(patched_output, before_output)
    assert torch.allclose(patched_output, before_output + torch.tensor([[2.5, 0.0]]))


def test_top1_routing_selects_nearest_record_id():
    entries = [
        _entry("e0", "rid-0", [1.0, 0.0, 0.0]),
        _entry("e1", "rid-1", [0.0, 1.0, 0.0]),
        _entry("e2", "rid-2", [0.0, 0.0, 1.0]),
    ]
    routed = route_edit_ids(
        query=torch.tensor([0.0, 0.9, 0.1]),
        entries=entries,
        query_record_id="rid-1",
        query_kind="new",
        route_policy="top1_no_threshold",
        prototype_type="target_prototype",
        threshold=None,
        max_active_edits=1,
    )

    assert routed["active_edit_ids"] == ["e1"]
    assert routed["active_record_ids"] == ["rid-1"]
    assert routed["top1_record_id"] == "rid-1"


def test_threshold_routing_can_select_no_edit():
    entries = [_entry("e0", "rid-0", [1.0, 0.0, 0.0])]
    routed = route_edit_ids(
        query=torch.tensor([0.0, 1.0, 0.0]),
        entries=entries,
        query_record_id="rid-x",
        query_kind="new",
        route_policy="top1_threshold",
        prototype_type="target_prototype",
        threshold=0.5,
        max_active_edits=1,
    )

    assert routed["active_edit_ids"] == []
    assert routed["active_edit_count"] == 0
    assert routed["self_edit_active"] is False


def test_oracle_uses_record_id_not_position():
    entries = [
        _entry("first-position", "rid-other", [1.0, 0.0]),
        _entry("second-position", "rid-target", [0.0, 1.0]),
    ]
    routed = route_edit_ids(
        query=torch.tensor([1.0, 0.0]),
        entries=entries,
        query_record_id="rid-target",
        query_kind="new",
        route_policy="oracle_self",
        prototype_type="target_prototype",
        threshold=None,
        max_active_edits=1,
    )

    assert routed["active_edit_ids"] == ["second-position"]
    assert routed["active_record_ids"] == ["rid-target"]
    assert routed["self_edit_active"] is True


def test_oracle_does_not_activate_for_reference_queries():
    entries = [_entry("e0", "rid-0", [1.0, 0.0])]
    routed = route_edit_ids(
        query=torch.tensor([1.0, 0.0]),
        entries=entries,
        query_record_id="rid-0",
        query_kind="reference",
        route_policy="oracle_self",
        prototype_type="target_prototype",
        threshold=None,
        max_active_edits=1,
    )

    assert routed["active_edit_ids"] == []
    assert routed["active_record_ids"] == []
    assert routed["self_edit_active"] is False
