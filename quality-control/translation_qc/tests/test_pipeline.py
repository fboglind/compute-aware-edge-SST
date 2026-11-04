import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from translation_qc.common import digest, read_json, read_jsonl, write_csv, write_jsonl
from translation_qc.explanations import export_prompts, import_annotations, validate_annotation
from translation_qc.fixtures import CASES
from translation_qc.human import evaluate_labels
from translation_qc.inputs import LABEL_FIELDS, load_run, prepare, validate
from translation_qc.jev import judge
from translation_qc.mock_run import run_mock
from translation_qc.reports import report
from translation_qc.tests.helpers import prepare_synthetic
from translation_qc.tests.test_jev import response, transport_response


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        network = patch("socket.socket", side_effect=AssertionError("All tests must remain offline"))
        network.start()
        self.addCleanup(network.stop)
        self.out, self.runs, self.references = prepare_synthetic(self.root)

    def mock_judge(self, limit=3):
        raw = response({"meaningful_omission": 0.8})
        transport = Mock(side_effect=[transport_response(raw=raw), transport_response(), transport_response()])
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-test-key"}), \
                patch("translation_qc.jev.http_post", transport):
            result = judge(self.out, self.out / "windows.jsonl", live=True, limit=limit)
        return result, transport

    def test_prepare_and_dry_run_preserve_every_input(self):
        before = {path: digest(path) for path in self.runs.rglob("*") if path.is_file()}
        windows, checks, manifest = validate(self.out)
        self.assertEqual(len(windows), 12)
        self.assertEqual(manifest["selected_count"], 12)
        plan = judge(self.out, self.out / "windows.jsonl", limit=3)
        self.assertEqual(plan["network_requests"], 0)
        summary = report(self.out)
        self.assertEqual(summary["counts"]["not_judged"], 12)
        self.assertEqual(before, {path: digest(path) for path in before})
        self.assertEqual(read_jsonl(self.out / "jev_decisions.jsonl"), [])
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(self.runs, self.references, self.out)

    def test_export_mismatch_fails_and_source_mutation_is_detected(self):
        run = next(self.runs.iterdir())
        metadata = read_json(run / "metadata.json")
        (run / "transcript.en.txt").write_text("Modified transcript", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "concatenation mismatch"):
            load_run(run, metadata["args"]["asr"], metadata["args"]["chunk_s"], read_json(self.references / "manifest.json"))
        with self.assertRaisesRegex(ValueError, "Input changed"):
            validate(self.out)

    def test_window_tampering_detected_before_mock_http(self):
        path = self.out / "windows.jsonl"
        path.write_text(path.read_text().replace("Result 0", "Replacement"), encoding="utf-8")
        with patch("translation_qc.jev.http_post", side_effect=AssertionError("Should not call HTTP")):
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                judge(self.out, path, live=True)

    def test_mocked_judge_resume_and_rethreshold_no_new_requests(self):
        result, transport = self.mock_judge()
        self.assertEqual(result["ok"], 3)
        self.assertEqual(transport.call_count, 3)
        result, transport = self.mock_judge()
        self.assertEqual(result["http_attempts"], 0)
        self.assertEqual(result["cached"], 3)
        transport.assert_not_called()
        summary = report(self.out, (0.5, 0.9))
        self.assertEqual(summary["counts"]["successful"], 3)
        self.assertEqual(summary["counts"]["not_judged"], 9)
        self.assertEqual(summary["provider_usage"]["http_attempts"], 3)
        self.assertEqual(summary["counts"]["thresholds"]["0.5"]["candidate_review_rate"]["numerator"], 1)
        self.assertEqual(summary["counts"]["thresholds"]["0.9"]["candidate_review_rate"]["numerator"], 0)

    def test_human_templates_preserved_and_metrics_only_labelled(self):
        self.mock_judge()
        windows = read_jsonl(self.out / "windows.jsonl")
        labels = [{"window_id": window["window_id"], "issue": truth, "omission": truth, "reviewer": "reviewer-A"}
                  for window, truth in zip(windows[:3], ("yes", "uncertain", "no"))]
        write_csv(self.out / "human_labels.csv", LABEL_FIELDS, labels)
        before = digest(self.out / "human_labels.csv")
        report(self.out)
        self.assertEqual(before, digest(self.out / "human_labels.csv"))
        result = evaluate_labels(self.out, self.out / "human_labels.csv")
        self.assertEqual(result["annotated_instances"], 3)
        self.assertEqual(result["omission_detection"]["evaluated"], 2)
        self.assertEqual(result["omission_detection"]["excluded"]["uncertain"], 1)
        self.assertEqual(result["omission_detection"]["f1"], 1)
        labels[1]["reviewer"] = "reviewer-B"
        write_csv(self.out / "human_labels.csv", LABEL_FIELDS, labels)
        with self.assertRaisesRegex(ValueError, "Multiple reviewers"):
            evaluate_labels(self.out, self.out / "human_labels.csv")

    def test_mqm_export_includes_unflagged_controls_and_preserves_provenance(self):
        self.mock_judge()
        result = export_prompts(self.out, "synthetic-generative-model", include_reference=True)
        self.assertEqual(result["requests"], 3)
        self.assertEqual(result["unflagged_sampled"], 2)
        self.assertEqual(result["network_requests"], 0)
        requests = read_jsonl(self.out / "llm_requests.jsonl")
        self.assertIn("whole_talk_unaligned", requests[0]["request"]["messages"][1]["content"])
        responses = [{"window_id": row["window_id"], "request_fingerprint": row["request_fingerprint"],
                      "provider": row["provider"], "model": row["model"], "annotation": {"issues": []}}
                     for row in requests]
        path = self.root / "llm-responses.jsonl"
        write_jsonl(path, responses)
        imported = import_annotations(self.out, path)
        self.assertEqual(imported["label_source"], "llm")
        self.assertEqual(report(self.out)["llm_annotated_windows"], 3)
        self.assertEqual(evaluate_labels(self.out, self.out / "human_labels.csv")["annotated_instances"], 0)
        with self.assertRaisesRegex(ValueError, "Preserve existing LLM"):
            export_prompts(self.out, "another-model")

    def test_mqm_span_and_attribution_validation(self):
        window = {"source_en": CASES[1]["source_en"], "target_de": CASES[1]["target_de"]}
        annotation = {"issues": [{"category": "omission", "severity": "major",
                                  "source_span": "only two produced usable results", "target_span": None,
                                  "explanation": "The results clause is missing.", "boundary_ambiguous": False,
                                  "possible_origin": "uncertain", "origin_evidence": None}]}
        self.assertEqual(validate_annotation(annotation, window), annotation)
        for field, value in (("source_span", "invented evidence"), ("source_span", None),
                             ("possible_origin", "asr"), ("severity", "severe"), ("boundary_ambiguous", "yes")):
            changed = copy.deepcopy(annotation)
            changed["issues"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_annotation(changed, window)

    def test_injected_error_mock_is_separate_from_pilot_performance(self):
        destination = self.root / "synthetic-mock"
        result = run_mock(destination)
        self.assertEqual(result["network_requests"], 0)
        self.assertEqual(result["mode"], "synthetic_mock")
        self.assertEqual(len(read_jsonl(destination / "synthetic_windows.jsonl")), 4)
        self.assertFalse((destination / "jev_decisions.jsonl").exists())
        self.assertIn("SCRIPTED ANSWERS", (destination / "review.md").read_text())


if __name__ == "__main__":
    unittest.main()
