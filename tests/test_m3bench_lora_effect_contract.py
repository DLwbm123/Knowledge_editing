import unittest

from m3bench_repro.editors.methods import LoraPaperSpecEditor, LoraRuntimeConfig


class _Peft:
    def __init__(self):
        self.active_adapter = None
        self.enabled = False

    def set_adapter(self, name):
        self.active_adapter = name

    def enable_adapter_layers(self):
        self.enabled = True

    def disable_adapter_layers(self):
        self.enabled = False


class LoraEffectContractTests(unittest.TestCase):
    def _editor(self, config=None):
        editor = object.__new__(LoraPaperSpecEditor)
        editor.targets = ["model.layers.0.mlp.up_proj"]
        editor.runtime_config = config or LoraRuntimeConfig()
        return editor

    def test_paper_spec_lora_default_unchanged(self):
        self.assertEqual(self._editor().config_lock(), {
            "schema_version": "m3bench-editor-method-config-v2",
            "method": "LoRA",
            "classification": "M3BENCH_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V2_EFFECT_REPAIRED",
            "scope": "all language-model MLP blocks",
            "targets": ["model.layers.0.mlp.up_proj"],
            "rank": 16,
            "lora_alpha": 16,
            "dropout": 0.0,
            "optimizer": "AdamW",
            "learning_rate": 5e-5,
            "batch_size": 1,
            "gradient_clip": 1.0,
            "epochs_per_edit": 5,
            "projector": "excluded",
            "vision_encoder": "excluded",
            "source": "PEFT 0.19.1 @ ba6a19060d6ab54a87538a6e77e3e4d5a907375b",
        })

    def test_calibrated_lora_profile_isolated(self):
        paper = self._editor().config_lock()
        calibrated = self._editor(LoraRuntimeConfig(
            profile_name="LoRA-M3Bench-Calibrated-v1",
            learning_rate=2e-4,
            steps_per_edit=20,
        )).config_lock()
        self.assertEqual(paper["learning_rate"], 5e-5)
        self.assertNotIn("paper_spec_deviation", paper)
        self.assertTrue(calibrated["paper_spec_deviation"])
        self.assertEqual(calibrated["epochs_per_edit"], 20)

    def test_lora_explicit_adapter_lifecycle(self):
        editor = object.__new__(LoraPaperSpecEditor)
        editor.peft_model = _Peft()
        editor.adapter_name = "default"
        editor._set_enabled(True)
        self.assertEqual(editor.peft_model.active_adapter, "default")
        self.assertTrue(editor.peft_model.enabled)
        with editor.disabled():
            self.assertFalse(editor.peft_model.enabled)
        self.assertTrue(editor.peft_model.enabled)
        self.assertEqual(editor.peft_model.active_adapter, "default")


if __name__ == "__main__":
    unittest.main()
