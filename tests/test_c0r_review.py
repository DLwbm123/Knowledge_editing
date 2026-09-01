import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/c0r_review/resolve_authorized_images.py"
spec = importlib.util.spec_from_file_location("resolver", SCRIPT)
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


def context():
    return json.loads(Path(os.environ["C0R_TEST_CONTEXT"]).read_text(encoding="utf-8"))


class C0RReviewTests(unittest.TestCase):
    def test_blocked_c0_unique_binding(self): self.assertEqual(context()["blocked_c0_candidate_count"], 1)
    def test_blocked_c0_read_only_preservation(self): self.assertTrue(context()["blocked_c0_preserved"])
    def test_uncommitted_patch_provenance(self): self.assertTrue(context()["uncommitted_code_provenance_pass"])
    def test_c0_completed_evidence_reverification(self): self.assertEqual(context()["evidence_status"], "PASS")
    def test_formal_selection_reconstruction_exact_order(self): self.assertEqual(context()["selection_status"], "EXACT_200_OF_200")
    def test_formal_target_structural_preflight(self): self.assertEqual(context()["structural_passed"], 200)
    def test_review_input_is_metadata_only(self): self.assertEqual(context()["review_images_embedded"], 0)
    def test_review_zip_contains_zero_image_magic_bytes(self): self.assertEqual(context()["zip_image_magic_entries"], 0)
    def test_no_thumbnail_crop_base64_or_embedding_export(self): self.assertEqual(context()["static_export_violations"], 0)

    def test_dataset_relative_path_containment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "x.jpg").write_bytes(b"x")
            item = {"review_id":"x", "source_dataset_identifier":"D", "dataset_relative_image_id_or_path":"x.jpg", "image_sha256":resolver.sha256_file(root / "x.jpg")}
            self.assertEqual(resolver.resolve_item(item, {"D":root.resolve()}), (root / "x.jpg").resolve())
            item["dataset_relative_image_id_or_path"] = "../x.jpg"
            with self.assertRaises(ValueError): resolver.resolve_item(item, {"D":root.resolve()})

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside:
            root=Path(d); target=Path(outside)/"x.jpg"; target.write_bytes(b"x"); (root/"x.jpg").symlink_to(target)
            item={"review_id":"x","source_dataset_identifier":"D","dataset_relative_image_id_or_path":"x.jpg","image_sha256":resolver.sha256_file(target)}
            with self.assertRaises(ValueError): resolver.resolve_item(item,{"D":root.resolve()})

    def test_image_hash_verified_before_display(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/"x.jpg").write_bytes(b"x")
            item={"review_id":"x","source_dataset_identifier":"D","dataset_relative_image_id_or_path":"x.jpg","image_sha256":"0"*64}
            with self.assertRaises(ValueError): resolver.resolve_item(item,{"D":root.resolve()})

    def test_local_server_binds_loopback_only(self):
        source=(ROOT/"scripts/c0r_review/launch_local_review.py").read_text(); self.assertIn('BIND_HOST = "127.0.0.1"',source); self.assertNotIn('"0.0.0.0"',source)
    def test_no_external_network_dependency(self): self.assertEqual(context()["external_network_dependencies"], 0)
    def test_review_console_has_no_download_endpoint(self): self.assertNotIn("download", (ROOT/"scripts/c0r_review/launch_local_review.py").read_text().lower())
    def test_public_private_review_join(self): self.assertTrue(context()["public_private_review_join"])
    def test_review_input_no_position_method_raw_answer_or_verdict_leakage(self): self.assertEqual(context()["review_input_forbidden_fields"], 0)

    def test_append_only_review_hash_chain(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"out.jsonl"; resolver.append_verdict(p,{"review_id":"a"}); resolver.append_verdict(p,{"review_id":"b"}); self.assertTrue(resolver.verify_chain(p))

    def test_reviewer_b_queue_blinding(self):
        queue=resolver.build_reviewer_b_queue(["a"],[{"review_id":"b","valid":False,"confidence":"low"}],["c"],7); self.assertEqual(set(queue),{"a","b","c"})
    def test_future_blind_reserve_not_auto_used(self): self.assertFalse(context()["future_blind_reserve_auto_used"])
    def test_state_snapshot_inventory_requires_all_components(self): self.assertFalse(resolver.snapshot_complete({"clean_base":True}))
    def test_parent_route_b_and_c0_runs_unchanged(self): self.assertTrue(context()["source_runs_preserved"])

    def test_atomic_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x"; resolver.atomic_new(p,b"x")
            with self.assertRaises(FileExistsError): resolver.atomic_new(p,b"y")


if __name__ == "__main__": unittest.main()
