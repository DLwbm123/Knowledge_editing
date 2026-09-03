import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.m3bench_core9_task_specific_t4l import build_candidates, dedupe_queries, freeze_command


FIELDS = ("image_id", "lesion_a", "lesion_b", "question_a", "question_b", "answer_a", "answer_b")


def valid_row(image="img1"):
    return {"image_id": image, "lesion_a": "Mass", "lesion_b": "Edema", "question_a": "Does this show mass?", "question_b": "Does this show edema?", "answer_a": "Yes", "answer_b": "Yes"}


class Args:
    pass


class TaskSpecificT4LTest(unittest.TestCase):
    def build(self, changes=None, duplicate=False):
        root = Path(self.temp.name)
        image = root / "images/img1/source.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        rows = [valid_row() for _ in range(257)]
        for i, row in enumerate(rows):
            row["image_id"] = f"img{i + 1}"
            p = root / f"images/img{i + 1}/source.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"image-{i + 1}".encode())
        if changes:
            rows[0].update(changes)
        if duplicate:
            rows[1] = dict(rows[0])
        csv_path = root / "t4l.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(rows)
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        return build_candidates(csv_path, root / "images", digest)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def test_independent_of_amended189(self):
        candidates, _, _ = self.build()
        self.assertEqual(len(candidates), 257)
        self.assertTrue(all(row["amended189_used_as_t4l_anchor"] is False for row in candidates))

    def test_role_mismatch_filtered(self):
        candidates, _, rejected = self.build({"question_a": "Is edema present?"})
        self.assertEqual(len(candidates), 256)
        self.assertIn("question_a_lesion_a_role_mismatch", rejected[0]["reasons"])

    def test_malformed_cjk_list_filtered(self):
        candidates, _, rejected = self.build({"lesion_b": "['水肿']"})
        self.assertEqual(len(candidates), 256)
        self.assertIn("malformed_list_or_cjk_scalar", rejected[0]["reasons"])

    def test_duplicate_filtered(self):
        candidates, _, rejected = self.build(duplicate=True)
        self.assertEqual(len(candidates), 256)
        self.assertIn("exact_duplicate", rejected[0]["reasons"])

    def test_query_dedup_keeps_lineage(self):
        row = {"image_id": "i", "image_sha256": "h", "question": "Q?", "gold_answer": "Yes", "lineage": [{"row": 1}]}
        merged = dedupe_queries([row, {**row, "lineage": [{"row": 2}]}])
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["lineage"]), 2)

    def test_only_a_wrong_b_correct_is_eligible(self):
        candidates, _, _ = self.build()
        root = Path(self.temp.name)
        candidate_path = root / "candidates.jsonl"
        candidate_path.write_text(json.dumps(candidates[0]) + "\n")
        verdict_path = root / "verdicts.jsonl"
        verdict_path.write_text("\n".join([
            json.dumps({"query_id": candidates[0]["edit_query_id"], "is_correct": False, "config_hash": "c"}),
            json.dumps({"query_id": candidates[0]["probe_query_id"], "is_correct": True, "config_hash": "c"}),
        ]) + "\n")
        args = Args(); args.candidates = candidate_path; args.base_verdicts = verdict_path; args.output_dir = root / "freeze"
        freeze_command(args)
        rows = [json.loads(x) for x in (args.output_dir / "T4L_FORMAL_RECORDS.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)

    def test_a_correct_and_b_wrong_are_excluded(self):
        candidates, _, _ = self.build()
        root = Path(self.temp.name)
        candidate_path = root / "candidates.jsonl"; candidate_path.write_text(json.dumps(candidates[0]) + "\n")
        for a_correct, b_correct in ((True, True), (False, False)):
            verdict_path = root / f"verdicts-{a_correct}-{b_correct}.jsonl"
            verdict_path.write_text("\n".join([
                json.dumps({"query_id": candidates[0]["edit_query_id"], "is_correct": a_correct, "config_hash": "c"}),
                json.dumps({"query_id": candidates[0]["probe_query_id"], "is_correct": b_correct, "config_hash": "c"}),
            ]) + "\n")
            args = Args(); args.candidates = candidate_path; args.base_verdicts = verdict_path; args.output_dir = root / f"freeze-{a_correct}-{b_correct}"
            freeze_command(args)
            self.assertEqual((args.output_dir / "T4L_FORMAL_RECORDS.jsonl").read_text(), "")

    def test_source_sha_mismatch_hard_fails(self):
        with self.assertRaisesRegex(RuntimeError, "source SHA mismatch"):
            root = Path(self.temp.name); csv_path = root / "bad.csv"; csv_path.write_text("x\n")
            build_candidates(csv_path, root, "0" * 64)


if __name__ == "__main__":
    unittest.main()
