import unittest

from scripts.medtrace.run_dev16 import derive_seed, evaluation_rows, validate_dev_rows


class Dev16RunnerTests(unittest.TestCase):
    def test_seed_and_frozen_coverage(self):
        rows = [{
            "event_id": f"event-{i}", "event_position": i,
            "edit_record": {"record_id": f"record-{i}", "question": "q", "image_path": "i", "gold_answer": "a"},
            "probes": [],
        } for i in range(1, 17)]
        validate_dev_rows(rows)
        seeds = [derive_seed(row["edit_record"]["record_id"]) for row in rows]
        self.assertEqual(seeds, [derive_seed(row["edit_record"]["record_id"]) for row in rows])
        self.assertEqual(len(set(seeds)), 16)
        rows[0]["probes"] = [{
            "probe_id": "probe", "task": "T1G", "question": "p", "image_path": "j",
            "reference": "a", "edit_id": "record-1", "probe_index": 0,
            "variant_type": "visual", "sequence_position": 1,
        }]
        selected = evaluation_rows(rows[0])
        self.assertEqual([item["task"] for item in selected], ["T0", "T1G"])


if __name__ == "__main__":
    unittest.main()
