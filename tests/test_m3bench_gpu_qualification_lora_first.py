import inspect
import unittest

from m3bench_repro.editors.methods import LoraPaperSpecEditor, _BalanceRoutingMixin
from scripts import editor_paperspec_formal
from scripts.m3bench_gpu_qualification_lora_first import no_edit_selection


class GpuQualificationLoraFirstTests(unittest.TestCase):
    def test_no_edit_raw_token_parity_contract_selection_deduplicates(self):
        events = []
        tasks = ("T0", "T1L", "T3L", "T3G", "T4L", "T1G", "T2L", "T2G", "T4G")
        for task in tasks:
            count = 65 if task in {"T1G", "T2L", "T2G", "T4G"} else 2
            events.append({
                "task": task,
                "event_id": f"{task}:event",
                "probes": [{"probe_id": f"{task}:{index}"} for index in range(count)] + [{"probe_id": "shared"}],
            })
        selected, report = no_edit_selection(events)
        ids = [row["query_id"] for row in selected]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(report["source_selection"]["T1G"]["occurrences"], 64)
        self.assertIn("shared", ids)

    def test_target_diagnostic_not_counted_as_official_probe(self):
        source = inspect.getsource(editor_paperspec_formal.command_single_events)
        self.assertIn('"raw_output_count": len(outputs)', source)
        self.assertIn('"target_diagnostic_counted_as_official_probe": False', source)

    def test_identity_fallback_not_used_by_lora(self):
        self.assertNotIn("official_rephrase", inspect.getsource(LoraPaperSpecEditor))

    def test_identity_fallback_used_by_balance_routing(self):
        self.assertIn("record.official_rephrase", inspect.getsource(_BalanceRoutingMixin))


if __name__ == "__main__":
    unittest.main()
