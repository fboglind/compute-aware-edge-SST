import tempfile
import unittest
from pathlib import Path

from translation_qc.common import write_jsonl
from translation_qc.human import confusion
from translation_qc.inputs import select_review
from translation_qc.jev import parse_response
from translation_qc.reports import configuration_summary, export_review, usage_summary
from translation_qc.tests.test_inputs import synthetic_windows
from translation_qc.tests.test_jev import response


class ReportTests(unittest.TestCase):
    def test_denominators_exclude_errors_and_unjudged(self):
        windows, checks = synthetic_windows(4, 100)
        select_review(windows)
        decisions = {
            windows[0]["window_id"]: {"status": "ok", "probabilities": parse_response(response({"meaningful_omission": 0.8}))},
            windows[1]["window_id"]: {"status": "ok", "probabilities": parse_response(response())},
            windows[2]["window_id"]: {"status": "transport_error"},
            windows[3]["window_id"]: {"status": "skipped_budget"},
        }
        result = configuration_summary(windows, decisions, (0.5, 0.9))
        self.assertEqual(result["successful"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["not_judged"], 1)
        self.assertEqual(result["skipped_budget"], 1)
        self.assertEqual(result["thresholds"]["0.5"]["candidate_review_rate"], {"numerator": 1, "denominator": 2, "value": 0.5})
        self.assertEqual(result["thresholds"]["0.9"]["candidate_review_rate"]["value"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            rows = export_review(Path(temporary), windows, checks, decisions)
            self.assertEqual(rows[0]["semantic_status"], "candidate")
            self.assertIn("transport_error", [row["semantic_status"] for row in rows])
            self.assertNotIn("correct", [row["semantic_status"] for row in rows])

    def test_zero_judgments_have_null_rate(self):
        windows, checks = synthetic_windows()
        select_review(windows)
        result = configuration_summary(windows, {}, (0.5,))
        self.assertIsNone(result["thresholds"]["0.5"]["candidate_review_rate"]["value"])

    def test_human_confusion_excludes_uncertain_unlabelled_and_failed(self):
        labels, decisions = [], {}
        cases = [("yes", 0.9), ("no", 0.9), ("yes", 0.1), ("no", 0.1), ("uncertain", 0.1), ("", 0.1), ("yes", None)]
        for index, (truth, score) in enumerate(cases):
            identifier = str(index)
            labels.append({"window_id": identifier, "issue": truth, "omission": truth})
            decisions[identifier] = {"status": "ok" if score is not None else "http_error",
                                     "probabilities": {"meaningful_omission": score}}
        result = confusion(labels, decisions, "omission", 0.5)
        self.assertEqual(result["confusion_matrix"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        self.assertEqual(result["evaluated"], 4)
        self.assertEqual(result["excluded"], {"unlabelled": 1, "uncertain": 1, "no_successful_judgment": 1})
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1"], 0.5)

    def test_no_human_labels_cannot_produce_accuracy(self):
        result = confusion([], {}, "omission", 0.5)
        self.assertEqual(result["evaluated"], 0)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["f1"])

    def test_usage_includes_failed_attempts_and_preserves_unknowns(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_jsonl(directory / "jev_attempts.jsonl", [
                {"attempt_id": "first", "status": "parse_error", "usage": {"cost": 0.01, "input_tokens": 100}},
                {"attempt_id": "second", "status": "transport_error"},
                {"attempt_id": "third", "status": "ok", "usage": {"cost": 0.02, "input_tokens": 200}},
            ])
            result = usage_summary(directory)
            self.assertAlmostEqual(result["fields"]["cost"]["known_sum"], 0.03)
            self.assertEqual(result["fields"]["cost"]["unknown_attempts"], 1)
            self.assertIsNone(result["fields"]["output_tokens"]["known_sum"])


if __name__ == "__main__":
    unittest.main()
