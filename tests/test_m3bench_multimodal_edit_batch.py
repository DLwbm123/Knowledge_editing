import unittest
import torch

from m3bench_repro.editors.llava_runtime import (
    IGNORE_INDEX,
    LlavaMedEditorRuntime,
    build_target_only_labels,
)


class _Core:
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, _positions, attention, _cache, labels, images, image_sizes=None
    ):
        expanded = torch.zeros((1, input_ids.shape[1] + 2, 4))
        expanded_labels = torch.cat(
            (labels[:, :1], torch.full((1, 2), IGNORE_INDEX), labels[:, 1:]), dim=1
        )
        return None, None, torch.ones(expanded.shape[:2]), None, expanded, expanded_labels


def _runtime():
    runtime = object.__new__(LlavaMedEditorRuntime)
    runtime.llava_model = lambda: _Core()
    return runtime


class MultimodalEditBatchTests(unittest.TestCase):
    def test_image_tensor_is_required_and_expansion_is_verified(self):
        runtime = _runtime()
        ids = torch.tensor([[-200, 4, 7]])
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 7]])
        with self.assertRaisesRegex(RuntimeError, "image tensor"):
            runtime._expand_multimodal(
                raw_input_ids=ids,
                attention_mask=torch.ones_like(ids),
                labels=labels,
                images=[],
            )
        embeds, _, _, expanded_labels = runtime._expand_multimodal(
            raw_input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=labels,
            images=torch.ones((1, 3, 2, 2)),
        )
        self.assertGreater(embeds.shape[1], ids.shape[1])
        self.assertEqual(expanded_labels.shape[:2], embeds.shape[:2])


    def test_target_only_mask_and_zero_target_failure(self):
        labels, target = build_target_only_labels(
            torch.tensor([[1, -200, 2, 8]]),
            torch.tensor([[1, -200, 2]]),
            torch.ones((1, 4), dtype=torch.long),
            image_token_index=-200,
        )
        self.assertEqual(target, (8,))
        self.assertEqual(labels.tolist(), [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 8]])
        with self.assertRaisesRegex(RuntimeError, "no tokens"):
            build_target_only_labels(
                torch.tensor([[1, -200, 2]]),
                torch.tensor([[1, -200, 2]]),
                torch.ones((1, 3), dtype=torch.long),
                image_token_index=-200,
            )


if __name__ == "__main__":
    unittest.main()
