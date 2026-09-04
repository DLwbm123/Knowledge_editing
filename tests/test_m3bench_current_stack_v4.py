import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import m3bench_current_stack_v4 as v4


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CurrentStackV4Tests(unittest.TestCase):
    def test_merge_reuses_only_exact_raw_and_rejudges_every_change(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(v4, "EXPECTED_QUERY_COUNT", 2):
            root = Path(directory)
            inventory = [
                {"query_id": "a", "question": "qa", "gold_answer": "ga", "gold_sha256": "g1", "image_sha256": "i1"},
                {"query_id": "b", "question": "qb", "gold_answer": "gb", "gold_sha256": "g2", "image_sha256": "i2"},
            ]
            old_base = [{"query_id": "a", "model_answer_raw": "same"}, {"query_id": "b", "model_answer_raw": "old"}]
            old_verdicts = [
                {"query_id": "a", "is_correct": True, "authoritative_route": "old", "semantic_judge_votes": [True], "judge_model": "j", "judge_prompt_sha256": "p", "judge_config_sha256": "c"},
                {"query_id": "b", "is_correct": False, "authoritative_route": "old", "semantic_judge_votes": [False], "judge_model": "j", "judge_prompt_sha256": "p", "judge_config_sha256": "c"},
            ]
            generated = [
                {"query_id": "a", "model_answer_raw": "same", "image_sha256": "i1", "empty": False, "error": None},
                {"query_id": "b", "model_answer_raw": "new", "image_sha256": "i2", "empty": False, "error": None},
            ]
            for name, rows in (("inventory", inventory), ("old_base", old_base), ("old_verdicts", old_verdicts), ("s0", generated[:1]), ("s1", generated[1:])):
                write_jsonl(root / name, rows)
            v4.merge(argparse.Namespace(
                inventory=root / "inventory", old_base=root / "old_base", old_verdicts=root / "old_verdicts",
                shards=[root / "s0", root / "s1"], output_root=root / "out",
            ))
            packet = v4.read_jsonl(root / "out/private/BASE_JUDGE_PACKET_V4.jsonl")
            self.assertEqual([row["opaque_query_id"] for row in packet], ["b"])

            write_jsonl(root / "judge", [{
                "opaque_query_id": "b", "is_correct": True, "parse_valid": True,
                "judge_model": "j", "judge_prompt_sha256": "p", "judge_config_sha256": "c",
            }])
            v4.finalize_verdicts(argparse.Namespace(
                inventory=root / "inventory", predictions=root / "out/private/BASE_PREDICTIONS_V4.jsonl",
                old_base=root / "old_base", old_verdicts=root / "old_verdicts", judge_output=root / "judge",
                output_root=root / "out",
            ))
            verdicts = v4.read_jsonl(root / "out/private/BASE_VERDICTS_V4.jsonl")
            self.assertTrue(verdicts[0]["v3_verdict_reused_by_raw_exact_match"])
            self.assertFalse(verdicts[1]["v3_verdict_reused_by_raw_exact_match"])
            self.assertTrue(verdicts[1]["is_correct"])


if __name__ == "__main__":
    unittest.main()
