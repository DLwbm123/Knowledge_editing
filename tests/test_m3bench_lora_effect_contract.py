import unittest

from m3bench_repro.editors.methods import LoraPaperSpecEditor


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
