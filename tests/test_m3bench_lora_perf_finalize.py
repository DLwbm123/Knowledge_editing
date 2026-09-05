import unittest

from scripts.m3bench_lora_perf_finalize import macro, parser, profile_id


class LoraPerfFinalizeTests(unittest.TestCase):
    def test_macro_is_per_edit_and_profile_id_is_stable(self):
        self.assertEqual(macro({"a": [True, False], "b": [True]}), 0.75)
        profile = {"learning_rate": 0.0001, "rank": 16}
        self.assertEqual(profile_id(profile), profile_id(dict(reversed(list(profile.items())))))

    def test_qual_accepts_one_run_without_selection_output(self):
        args = parser().parse_args(["summarize", "--runs", "qual", "--mapping", "map", "--verdicts", "votes", "--output", "summary", "--csv", "table"])
        self.assertEqual([path.name for path in args.runs], ["qual"])
        self.assertIsNone(args.selection_lock)


if __name__ == "__main__":
    unittest.main()
