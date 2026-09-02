from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from m3bench_repro.editors.llava_runtime import build_target_only_labels
from m3bench_repro.editors.routed_layers import (
    GraceValueLinear,
    RoutedFullLinear,
    RoutedLoRALinear,
    safe_slot,
)
from m3bench_repro.editors.routing import (
    canonical_float32,
    route_dict_equal,
    GraceCodebook,
    MemoryRouter,
    balanced_radius,
    cosine_distances,
    euclidean_distances,
)


class RoutingTests(unittest.TestCase):
    def test_cosine_numerical(self):
        keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        query = torch.tensor([1.0, 0.0])
        self.assertTrue(torch.allclose(cosine_distances(keys, query), torch.tensor([0.0, 1.0])))

    def test_euclidean_numerical(self):
        keys = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        query = torch.tensor([0.0, 0.0])
        self.assertTrue(torch.allclose(euclidean_distances(keys, query), torch.tensor([0.0, 5.0])))

    def test_radius_below_equal_above(self):
        router = MemoryRouter("euclidean")
        router.add("edit", torch.tensor([0.0]), 1.0)
        self.assertTrue(router.route(torch.tensor([0.999])).activated)
        self.assertTrue(router.route(torch.tensor([1.0])).activated)
        self.assertFalse(router.route(torch.tensor([1.001])).activated)

    def test_balance_radius_regression(self):
        radius = balanced_radius(
            torch.tensor([0.0]),
            torch.tensor([0.2]),
            torch.tensor([1.0]),
            alpha=0.2,
            distance="euclidean",
        )
        self.assertAlmostEqual(float(radius), 0.84, places=6)

    def test_router_roundtrip(self):
        router = MemoryRouter("cosine")
        router.add("a", torch.tensor([1.0, 0.0]), 0.2, [7, 8])
        restored = MemoryRouter.from_state(router.export_state())
        self.assertEqual(restored.logical_ids, ["a"])
        self.assertTrue(restored.route(torch.tensor([1.0, 0.0])).activated)


class RouteComparisonTests(unittest.TestCase):
    def test_float32_radius_canonicalization_is_exact_without_tolerance(self):
        before = {
            "activated": True,
            "distance": "cosine",
            "logical_edit_id": "edit",
            "nearest_logical_edit_id": "edit",
            "nearest_distance": 0.0,
            "radius": 0.16454942392349242,
        }
        after = dict(before, radius=0.16454942524433136)
        self.assertFalse(route_dict_equal(before, after, radius_mode="exact"))
        self.assertTrue(route_dict_equal(before, after, radius_mode="float32"))
        self.assertEqual(canonical_float32(before["radius"]), canonical_float32(after["radius"]))

    def test_float32_mode_keeps_non_radius_fields_exact(self):
        before = {"activated": True, "radius": 0.09970249103546143}
        after = {"activated": False, "radius": 0.09970249235630035}
        self.assertFalse(route_dict_equal(before, after, radius_mode="float32"))

    def test_float32_mode_rejects_missing_or_none_radius_mismatch(self):
        self.assertFalse(route_dict_equal({"activated": True}, {"activated": True, "radius": 1.0}, radius_mode="float32"))
        self.assertFalse(route_dict_equal({"radius": None}, {"radius": 1.0}, radius_mode="float32"))


class GraceCodebookTests(unittest.TestCase):
    def test_insert_reuse_and_collision(self):
        codebook = GraceCodebook(distance="cosine", eps_init=1.0)
        first = codebook.insert_with_source_semantics("a", torch.tensor([1.0, 0.0]), [10])
        self.assertEqual(first.action, "insert")
        reused = codebook.insert_with_source_semantics("a2", torch.tensor([0.99, 0.01]), [10])
        self.assertEqual(reused.action, "same_label_reuse")
        self.assertEqual(reused.effective_logical_edit_id, "a")
        collision = codebook.insert_with_source_semantics("b", torch.tensor([0.0, 1.0]), [20])
        self.assertEqual(collision.action, "collision_split")
        self.assertEqual(len(codebook), 2)

    def test_euclidean_far_insert_diagnostic(self):
        codebook = GraceCodebook(distance="euclidean", eps_init=1.0)
        codebook.insert_with_source_semantics("a", torch.tensor([0.0]), [1])
        result = codebook.insert_with_source_semantics("b", torch.tensor([3.0]), [2])
        self.assertEqual(result.action, "insert_far")

    def test_source_label_match(self):
        self.assertTrue(GraceCodebook.source_label_match([1, 3], [1, 3]))
        self.assertFalse(GraceCodebook.source_label_match([1, 3], [2, 2]))
        self.assertFalse(GraceCodebook.source_label_match([1, 3], [2, 3]))


class TargetMaskTests(unittest.TestCase):
    def test_prompt_image_target_padding_contract(self):
        prefix = torch.tensor([[1, -200, 2, 3]])
        full = torch.tensor([[1, -200, 2, 3, 7, 8, 0]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
        labels, targets = build_target_only_labels(
            full, prefix, attention, image_token_index=-200
        )
        self.assertEqual(targets, (7, 8))
        self.assertTrue(torch.equal(labels, torch.tensor([[-100, -100, -100, -100, 7, 8, -100]])))
        labels2, targets2 = build_target_only_labels(
            full.clone(), prefix.clone(), attention.clone(), image_token_index=-200
        )
        self.assertTrue(torch.equal(labels, labels2))
        self.assertEqual(targets, targets2)

    def test_prefix_mismatch_fails_closed(self):
        with self.assertRaises(RuntimeError):
            build_target_only_labels(
                torch.tensor([[1, 2, 4]]),
                torch.tensor([[1, 3]]),
                torch.ones((1, 3), dtype=torch.long),
                image_token_index=-200,
            )


class RoutedLayerTests(unittest.TestCase):
    def test_grace_value_hit_miss_and_roundtrip(self):
        base = nn.Linear(2, 3, bias=False)
        nn.init.zeros_(base.weight)
        wrapper = GraceValueLinear(base, replacement="replace_prompt")
        value = wrapper.add_cold_value("edit.one", seed=9)
        value.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
        inputs = torch.ones((1, 4, 2))
        miss = wrapper(inputs)
        self.assertTrue(torch.equal(miss, torch.zeros_like(miss)))
        wrapper.set_active("edit.one", token_index=3)
        hit = wrapper(inputs)
        self.assertTrue(torch.equal(hit, torch.tensor([[[1.0, 2.0, 3.0]]]).expand(1, 4, -1)))
        state = wrapper.export_state()
        restored = GraceValueLinear(copy.deepcopy(base), replacement="replace_prompt")
        restored.load_exported_state(state)
        restored.set_active("edit.one", token_index=3)
        self.assertTrue(torch.equal(hit, restored(inputs)))

    def test_balancedit_full_copy_hit_miss_isolation(self):
        base = nn.Linear(2, 2, bias=False)
        nn.init.eye_(base.weight)
        wrapper = RoutedFullLinear(base)
        original = wrapper(torch.tensor([[1.0, 2.0]]))
        edit = wrapper.add_edit("a")
        nn.init.zeros_(edit.weight)
        wrapper.set_active("a")
        self.assertTrue(torch.equal(wrapper(torch.tensor([[1.0, 2.0]])), torch.zeros((1, 2))))
        wrapper.set_active(None)
        self.assertTrue(torch.equal(wrapper(torch.tensor([[1.0, 2.0]])), original))
        self.assertTrue(torch.equal(base.weight, torch.eye(2)))

    def test_belora_logical_id_coordinates_full_adapter_set(self):
        base1 = nn.Linear(2, 2, bias=False)
        base2 = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(base1.weight)
        nn.init.zeros_(base2.weight)
        wrappers = [
            RoutedLoRALinear(base1, rank=1, alpha=1, dropout=0.0),
            RoutedLoRALinear(base2, rank=1, alpha=1, dropout=0.0),
        ]
        for index, wrapper in enumerate(wrappers):
            a, b = wrapper.add_adapter("edit-a", seed=10 + index)
            a.data.fill_(1.0)
            b.data.fill_(1.0 + index)
            wrapper.set_active("edit-a")
        x = torch.tensor([[1.0, 2.0]])
        outputs = [wrapper(x) for wrapper in wrappers]
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        for wrapper in wrappers:
            wrapper.set_active(None)
            self.assertTrue(torch.equal(wrapper(x), torch.zeros_like(x)))

    def test_belora_two_edit_parameter_isolation_and_roundtrip(self):
        base = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(base.weight)
        wrapper = RoutedLoRALinear(base, rank=1, alpha=1, dropout=0.0)
        a1, b1 = wrapper.add_adapter("one", seed=1)
        a2, b2 = wrapper.add_adapter("two", seed=2)
        a1.data.fill_(1.0)
        b1.data.fill_(1.0)
        a2.data.fill_(2.0)
        b2.data.fill_(2.0)
        x = torch.tensor([[1.0, 1.0]])
        wrapper.set_active("one")
        out1 = wrapper(x)
        wrapper.set_active("two")
        out2 = wrapper(x)
        self.assertFalse(torch.equal(out1, out2))
        state = wrapper.export_state()
        restored = RoutedLoRALinear(copy.deepcopy(base), rank=1, alpha=1, dropout=0.0)
        restored.load_exported_state(state)
        restored.set_active("one")
        self.assertTrue(torch.equal(out1, restored(x)))
        self.assertTrue(torch.equal(base.weight, torch.zeros_like(base.weight)))


class DiskBackedRoutedLayerTests(unittest.TestCase):
    def test_disk_backing_preserves_exact_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            torch.manual_seed(7)
            base = nn.Linear(5, 3, bias=False)
            wrapper = RoutedFullLinear(base, inactive_store_dir=Path(temporary) / "states")
            edited = wrapper.add_edit("record-a")
            with torch.no_grad():
                edited.weight.add_(0.125)
            wrapper.train_only("record-a")
            wrapper.set_active("record-a")
            inputs = torch.randn(2, 5)
            expected = wrapper(inputs).detach().clone()
            wrapper.train_only(None)
            wrapper.set_active(None)
            self.assertEqual(len(wrapper.edited), 1)
            self.assertEqual(len(wrapper.archived), 1)
            wrapper.set_active("record-a")
            actual = wrapper(inputs).detach()
            self.assertTrue(torch.equal(expected, actual))
            wrapper.set_active(None)
            self.assertEqual(len(wrapper.edited), 1)

    def test_disk_state_reload_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            torch.manual_seed(11)
            base = nn.Linear(4, 2, bias=False)
            state_root = Path(temporary) / "states"
            wrapper = RoutedFullLinear(base, inactive_store_dir=state_root)
            edited = wrapper.add_edit("record-b")
            with torch.no_grad():
                edited.weight.mul_(1.5)
            wrapper.train_only("record-b")
            wrapper.set_active("record-b")
            inputs = torch.randn(3, 4)
            expected = wrapper(inputs).detach().clone()
            wrapper.train_only(None)
            wrapper.set_active(None)
            state = wrapper.export_state()
            restored = RoutedFullLinear(base, inactive_store_dir=state_root)
            restored.load_exported_state(state)
            restored.set_active("record-b")
            actual = restored(inputs).detach()
            self.assertTrue(torch.equal(expected, actual))
            stats = restored.parameter_statistics()
            self.assertEqual(stats["entry_count"], 1)
            self.assertEqual(stats["archived_entry_count"], 1)

    def test_existing_orphan_is_preserved_on_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            torch.manual_seed(19)
            state_root = Path(temporary) / "states"
            state_root.mkdir()
            logical_id = "record-c"
            orphan = state_root / f"{safe_slot(logical_id)}.pt"
            orphan.write_bytes(b"incomplete-preserved-state")
            original = orphan.read_bytes()
            base = nn.Linear(4, 2, bias=False)
            wrapper = RoutedFullLinear(base, inactive_store_dir=state_root)
            edited = wrapper.add_edit(logical_id)
            with torch.no_grad():
                edited.weight.add_(0.25)
            wrapper.train_only(logical_id)
            wrapper.set_active(logical_id)
            expected = wrapper(torch.ones(1, 4)).detach().clone()
            wrapper.train_only(None)
            wrapper.set_active(None)
            self.assertEqual(orphan.read_bytes(), original)
            archive = Path(str(wrapper.archived[logical_id]["path"]))
            self.assertNotEqual(archive, orphan)
            self.assertTrue(archive.name.endswith(".recovery_001.pt"))
            wrapper.set_active(logical_id)
            self.assertTrue(torch.equal(expected, wrapper(torch.ones(1, 4))))


if __name__ == "__main__":
    unittest.main()
