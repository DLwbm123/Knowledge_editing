import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.editor_paperspec_formal import load_records, normalize_sequence_args
from scripts.prepare_m3bench_amended189 import expected_counts


class VariableSequenceTests(unittest.TestCase):
    def test_synthetic_n3_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "record_id": f"r{position}",
                    "dataset": "synthetic",
                    "question": "q",
                    "gold_answer": "a",
                    "official_rephrase": "q2",
                    "image_path": "/not-opened.png",
                    "relative_image_path": "not-opened.png",
                    "formal_sequence_position": position,
                    "question_type": "synthetic",
                }
                for position in range(1, 4)
            ]
            (root / "records.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            records = load_records(root, "records.jsonl", 3)
            self.assertEqual([record.formal_sequence_position for record in records], [1, 2, 3])
            args = normalize_sequence_args(
                argparse.Namespace(
                    prefixes="1,3",
                    expected_record_count=3,
                    final_prefix=3,
                    sequence_label="SYNTHETIC_N3",
                )
            )
            self.assertEqual(args.prefix_values, (1, 3))
            counts = expected_counts(
                [
                    {"task": "T0", "sequence_position": 1},
                    {"task": "T1L", "sequence_position": 3},
                ],
                3,
            )
            self.assertEqual(counts["sequential"]["prefixes"]["3"]["raw_outputs_per_method"], 2)

    def test_rejects_n3_position_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "record_id": f"r{position}",
                    "dataset": "synthetic",
                    "question": "q",
                    "gold_answer": "a",
                    "official_rephrase": "q2",
                    "image_path": "/not-opened.png",
                    "relative_image_path": "not-opened.png",
                    "formal_sequence_position": position,
                    "question_type": "synthetic",
                }
                for position in (1, 2, 4)
            ]
            (root / "records.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "order drift"):
                load_records(root, "records.jsonl", 3)


if __name__ == "__main__":
    unittest.main()
