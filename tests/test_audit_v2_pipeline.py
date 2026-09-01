import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "audit_v2_pipeline.py"
if not SCRIPT.exists():
    SCRIPT = Path(__file__).with_name("audit_v2_pipeline.py")
SPEC = importlib.util.spec_from_file_location("audit_v2_pipeline", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class PipelineTests(unittest.TestCase):
    def test_result_census_and_schema(self):
        rows = []
        for i in range(1, 201):
            rows.append({
                "audit_id": f"audit_{i:04d}",
                "verdict": i <= 64,
                "confidence": "low" if i in {18, 40, 54} else "medium" if i in {9, 152, 168, 191} else "high",
                "reason": "test",
                "issue_type": "ambiguous_reference" if i in {18, 40, 54} else "none",
                **({"dataset_issue": True} if i in {18, 40, 54} else {}),
            })
        census = MOD.validate_results(rows)
        self.assertEqual(census["records"], 200)
        self.assertEqual(census["dataset_issue_ids"], MOD.ISSUE_IDS)

    def test_safe_zip_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escape", "x")
            with self.assertRaises(RuntimeError):
                MOD.safe_extract(archive, Path(d) / "out")

    def test_atomic_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            MOD.atomic_json(path, {"ok": True})
            with self.assertRaises(RuntimeError):
                MOD.atomic_json(path, {"ok": False})

    def test_tool_can_import_sibling_module(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "helper.py").write_text("VALUE = 7\n")
            (root / "tool.py").write_text("from helper import VALUE\nassert VALUE == 7\n")
            result = MOD.invoke_tool(root / "tool.py", [])
            self.assertEqual(result["exit_code"], 0)

    def test_source_path_is_resolved_beneath_parent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            self.assertEqual(MOD.resolve_beneath(root, "a/b.json"), root / "a/b.json")
            with self.assertRaises(RuntimeError):
                MOD.resolve_beneath(root, "../escape.json")


if __name__ == "__main__":
    unittest.main()
