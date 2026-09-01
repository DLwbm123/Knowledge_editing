import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/c0r_review"
sys.path.insert(0, str(SCRIPTS))

import launch_local_review_fast as fast
import resolve_authorized_images as resolver
import reviewctl


def form(**values):
    return {key: [str(value)] for key, value in values.items()}


def context():
    return json.loads(Path(os.environ["C0R_FAST_TEST_CONTEXT"]).read_text(encoding="utf-8"))


class FastReviewTests(unittest.TestCase):
    def sample_page(self):
        return fast.page({"review_id": "target_review_0001", "question": "Q", "target_reference": "A"}, 0, 200, 0, "token")

    def test_fast_ui_has_no_default_verdict(self):
        page = self.sample_page()
        self.assertIn('<option value="" selected>Choose…</option>', page)
        self.assertIn('id="submit" type="submit" disabled', page)
        self.assertNotIn(" checked", page)

    def test_preset_json_is_valid_inline_javascript(self):
        page = self.sample_page()
        self.assertIn('const PRESETS={"1":', page)
        self.assertNotIn("&quot;", page)

    def test_freeze_keeps_verdict_fields_enabled_for_post(self):
        page = self.sample_page()
        self.assertIn("querySelectorAll('button')", page)
        self.assertNotIn("button,select,textarea", page)

    def test_freeze_uses_unshadowed_native_form_submit(self):
        page = self.sample_page()
        self.assertIn("HTMLFormElement.prototype.submit.call(form)", page)
        self.assertNotIn("form.submit()", page)

    def test_enter_without_selection_does_not_submit(self):
        self.assertIn("if(!preset.value)return false", self.sample_page())

    def test_preset_1_mapping(self):
        self.assertEqual(fast.PRESETS["1"]["relation"], "direct_answer")
        self.assertEqual(fast.PRESETS["1"]["recommended_action"], "retain")

    def test_preset_2_mapping(self):
        self.assertEqual(fast.PRESETS["2"]["relation"], "acceptable_visual_deixis")

    def test_preset_3_mapping(self):
        self.assertEqual(fast.PRESETS["3"]["confidence"], "medium")
        self.assertEqual(fast.PRESETS["3"]["relation"], "context_dependent")

    def test_invalid_requires_issue_and_reason(self):
        values = form(preset=4, valid="false", confidence="high", relation="mismatch", issue_type="none", recommended_action="exclude", reason="short", confirmed="false")
        with self.assertRaises(ValueError):
            fast.validate_verdict(values, "target_review_0001")

    def test_uncertain_normalizes_to_unresolved(self):
        self.assertEqual(resolver.outcome({"valid": False, "confidence": "low", "relation": "ambiguous", "recommended_action": "manual_review"}), "UNRESOLVED")

    def test_textarea_focus_blocks_global_hotkeys(self):
        page = self.sample_page()
        self.assertIn("!editing&&/^[1-6]$/.test", page)
        self.assertIn("!editing&&e.key.toLowerCase()==='f'", page)

    def test_flag_does_not_create_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events.jsonl"; output = Path(directory) / "output.jsonl"
            resolver.append_verdict(events, {"event": "flagged_for_later", "review_id": "a"})
            self.assertFalse(output.exists())
            self.assertEqual(resolver.flag_counts(events), {"a": 1})

    def test_flag_queue_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events.jsonl"
            resolver.append_verdict(events, {"event": "flagged_for_later", "review_id": "c"})
            resolver.append_verdict(events, {"event": "flagged_for_later", "review_id": "b"})
            self.assertEqual(resolver.remaining_order(["a", "b", "c"], set(), events), ["a", "c", "b"])

    def test_flag_limit(self):
        self.assertTrue(fast.flag_allowed(0)); self.assertTrue(fast.flag_allowed(1)); self.assertFalse(fast.flag_allowed(2))

    def test_duplicate_post_rejected(self):
        with self.assertRaises(FileExistsError):
            fast.accept_review_id("a", {"a"}, {"a"})

    def test_resume_from_hash_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.jsonl"
            resolver.append_verdict(path, {"review_id": "a"}); resolver.append_verdict(path, {"review_id": "b"})
            self.assertTrue(resolver.verify_chain(path))

    def test_reviewctl_start_is_idempotent(self):
        with mock.patch.object(reviewctl, "status", return_value={"server_running": True, "pid": 7}), mock.patch.object(reviewctl, "token_url", return_value="local"):
            self.assertTrue(reviewctl.start({}, 8765)["already_running"])

    def test_reviewctl_stop_is_pid_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {"state": Path(directory), "session": Path(directory) / "session.json", "resolver": Path(directory)}
            with self.assertRaises(RuntimeError):
                reviewctl.stop(paths)

    def test_freeze_requires_200_unique_ids(self):
        with self.assertRaises(RuntimeError):
            reviewctl.validate_freeze_rows([])

    def test_reviewer_b_queue_hides_selection_reason(self):
        queue = resolver.build_reviewer_b_queue(["focus"], [{"review_id": "u", "valid": False, "confidence": "low", "relation": "ambiguous", "recommended_action": "manual_review"}], ["control"], 3)
        self.assertEqual(set(queue), {"focus", "u", "control"})
        self.assertTrue(all(isinstance(value, str) for value in queue))

    def test_package_input_order_hashes_unchanged(self):
        self.assertTrue(context()["package_hash_unchanged"] and context()["input_hash_unchanged"] and context()["order_hash_unchanged"])

    def test_server_binds_loopback_only(self):
        source = (SCRIPTS / "launch_local_review_fast.py").read_text()
        self.assertIn('BIND_HOST = "127.0.0.1"', source); self.assertNotIn('BIND_HOST = "0.0.0.0"', source)

    def test_no_external_assets(self):
        page = self.sample_page().lower()
        self.assertNotIn("<link", page); self.assertNotIn("src=\"http", page); self.assertNotIn("https://", page)

    def test_zero_image_export(self):
        source = (SCRIPTS / "launch_local_review_fast.py").read_text().lower()
        self.assertTrue(context()["zero_image_export"]); self.assertNotIn("canvas", source); self.assertNotIn("download=", source); self.assertNotIn("base64", source)

    def test_security_headers_present(self):
        source = (SCRIPTS / "launch_local_review_fast.py").read_text()
        for header in ("X-Frame-Options", "Cross-Origin-Resource-Policy", "Permissions-Policy", "Content-Disposition"):
            self.assertIn(header, source)


if __name__ == "__main__":
    unittest.main()
