import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

from qc_notebook import Workflow


class NotebookWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "translation_qc").mkdir()
        (self.root / "translation_qc/__main__.py").write_text("# fixture")
        (self.root / "qc_notebook.py").write_text("# fixture")
        self.flow = Workflow(self.root, self.root / "main-input", self.root / "pilot-input",
                             self.root / "pilot-reference", self.root / "venv/bin/python", self.root / "qc")

    def test_disabled_paid_cell_never_prompts_or_runs_even_with_key(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}), \
                patch.object(self.flow, "run", side_effect=AssertionError("Must not run")), \
                patch("qc_notebook.getpass.getpass", side_effect=AssertionError("Must not prompt")):
            self.assertIsNone(self.flow.live_judge(self.flow.main))

    def test_direct_live_command_requires_explicit_key(self):
        with self.assertRaisesRegex(ValueError, "live_judge"):
            self.flow.run("judge", "--live-jev")

    def test_offline_subprocess_does_not_receive_key_and_records_provenance(self):
        completed = subprocess.CompletedProcess([], 0, '{"status":"passed"}', "")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}), \
                patch("qc_notebook.subprocess.run", return_value=completed) as child:
            self.flow.run("validate", "--input", self.flow.main)
        self.assertNotIn("OPENROUTER_API_KEY", child.call_args.kwargs["env"])
        records = list((self.flow.output_root / "notebook-invocations").glob("*.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text())
        self.assertFalse(record["live_requested"])
        self.assertIn("qc_notebook.py", record["interface_sha256"])
        self.assertNotIn("test-secret", records[0].read_text())

    def test_enabled_cell_preserves_explicit_limits_and_no_retries(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}), \
                patch.object(self.flow, "run", return_value={"ok": 3}) as run:
            self.flow.live_judge(self.flow.pilot, enabled=True, limit=3, max_requests=3, max_cost_usd=0.01)
        self.assertEqual(run.call_count, 2)
        args = run.call_args.args
        self.assertIn("--live-jev", args)
        self.assertEqual(args[args.index("--max-requests") + 1], 3)
        self.assertEqual(args[args.index("--retries") + 1], 0)
        self.assertEqual(run.call_args.kwargs["api_key"], "test-secret")

    def test_key_prompt_is_only_used_for_enabled_cell_and_failure_is_redacted(self):
        failed = subprocess.CompletedProcess([], 1, "", "fixture failure: test-secret")
        with patch("qc_notebook.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, r"fixture failure: \[REDACTED\]"):
                self.flow.run("judge", "--live-jev", api_key="test-secret")
        for path in (self.flow.output_root / "notebook-invocations").glob("*.json"):
            self.assertNotIn("test-secret", path.read_text())

    def test_unrecognized_existing_directory_is_preserved(self):
        self.flow.main.mkdir(parents=True)
        marker = self.flow.main / "my-notes.md"
        marker.write_text("Keep these notes.")
        with self.assertRaisesRegex(FileExistsError, "nothing was overwritten"):
            self.flow.prepare_main()
        self.assertEqual(marker.read_text(), "Keep these notes.")

    def test_misconfigured_output_cannot_write_inside_main_inputs(self):
        with self.assertRaisesRegex(ValueError, "outside the experiment inputs"):
            Workflow(self.root, self.root / "main-input", self.root / "pilot-input",
                     self.root / "pilot-reference", self.root / "python", self.root / "main-input/qc")


if __name__ == "__main__":
    unittest.main()
