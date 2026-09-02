import csv
import tempfile
import unittest
from pathlib import Path

from scripts.m3bench_core9_anchor_audit import audit_records, normalize, phrase_in


class Core9AnchorAuditTest(unittest.TestCase):
    def test_normalization_and_phrase_matching(self):
        self.assertEqual(normalize("Brain-edema?"), "brain edema")
        self.assertTrue(phrase_in("brain edema", "Is brain-edema present?"))
        self.assertFalse(phrase_in("edema", "edematous change"))

    def test_minimal_valid_relations(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            metadata = root / "metadata"
            slake = root / "slake"
            vqarad = root / "vqarad"
            metadata.mkdir()
            for image_id in ("imgA", "imgB"):
                path = slake / image_id / "source.jpg"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"image")

            def write_csv(name, fields, rows):
                with (metadata / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)

            qas = "[{'qid':'q1','question':'Does this show edema?','answer':'yes'}, {'qid':'q2','question':'Where is the lesion?','answer':'left'}]"
            write_csv("slake_metadata.csv", ["image_id", "qa_list"], [{"image_id": "imgA", "qa_list": qas}, {"image_id": "imgB", "qa_list": qas}])
            write_csv("vqarad_metadata.csv", ["image_id", "qa_list"], [])
            write_csv("t2l_cross_image_pairs.csv", ["image_id_1", "image_id_2", "qa_list"], [{"image_id_1": "imgA", "image_id_2": "imgB", "qa_list": qas}])
            write_csv("t3_cross_modality_pairs.csv", ["image_A", "modality_A", "diseases", "same_disease_images_in_other_modalities"], [{"image_A": "imgA", "modality_A": "xray", "diseases": "['edema']", "same_disease_images_in_other_modalities": "[{'image_id':'imgB','modality':'ct','disease':'edema'}]"}])
            write_csv("t4l_compositional_locality.csv", ["image_id", "lesion_a", "lesion_b", "question_a", "question_b", "answer_a", "answer_b"], [{"image_id": "imgA", "lesion_a": "edema", "lesion_b": "mass", "question_a": "Does this show edema?", "question_b": "Where is the lesion?", "answer_a": "yes", "answer_b": "left"}])
            write_csv("t4g_compositional_generality.csv", ["single_image", "single_lesion", "multi_image", "multi_lesions"], [{"single_image": "imgA", "single_lesion": "edema", "multi_image": "imgB", "multi_lesions": "edema,mass"}])
            records = [{"record_id": "r1", "original_position": 1, "amended_position": 1, "dataset": "SLAKE", "relative_image_path": "imgA/source.jpg", "question": "Does this show edema?", "gold_answer": "yes"}]
            _, census = audit_records(records, metadata, slake, vqarad)
            self.assertEqual(census["status"], "PASS")
            self.assertGreater(census["tasks"]["T2L"]["candidate_probe_relations"], 0)
            self.assertGreater(census["tasks"]["T3L"]["candidate_probe_relations"], 0)
            self.assertGreater(census["tasks"]["T4L"]["candidate_probe_relations"], 0)
            self.assertGreater(census["tasks"]["T4G"]["candidate_probe_relations"], 0)


if __name__ == "__main__":
    unittest.main()
