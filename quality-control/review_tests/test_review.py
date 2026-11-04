"""Offline tests only. All socket creation is forbidden; responses are scripted."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import qc_review as r
from translation_qc.tests.helpers import prepare_synthetic
from translation_qc.tests.test_jev import response


def annotation():
    return {k: {"verdict": "absent", "explanation": "Synthetic transport test, not a translation assessment.",
                "source_unclear": False, "boundary_unclear": False, "evidence": []} for k in r.CATEGORIES}


def raw_reply(ann=None, **changes):
    result = {"model": "anthropic/claude-sonnet-4.5", "provider": "Anthropic",
              "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(ann or annotation())}}],
              "usage": {"prompt_tokens": 100, "completion_tokens": 200, "cost": 0.0033}}
    result.update(changes)
    return 200, json.dumps(result)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        guard = patch("socket.socket", side_effect=AssertionError("Offline test: network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.main, _, _ = prepare_synthetic(self.root)
        self.windows = r.read_jsonl(self.main / "windows.jsonl")
        decisions = []
        for i, w in enumerate(self.windows):
            req = r.build_request(w)
            scores = {k: (0.81 if i % 3 == 0 else 0.49 if i % 3 == 1 else 0.1) for k in r.CATEGORIES}
            decisions.append({"window_id": w["window_id"], "mode": "live", "status": "ok",
                              "request": req, "request_fingerprint": req["request_fingerprint"],
                              "probabilities": scores, "raw_response": response(scores)})
        r.write_jsonl(self.main / "jev_decisions.jsonl", decisions)
        self.decisions = {d["window_id"]: d for d in decisions}
        self.out = self.main / "api-review-test"
        self.settings = r.Settings(high_count=3, borderline_count=3, unflagged_count=2)
        self.plan = r.prepare(self.main, self.out, self.settings)
        self.requests = r.load(self.out)[1]

    def test_selection_is_disjoint_order_independent_and_includes_controls(self):
        a = r.select(self.windows, self.decisions, self.settings)
        b = r.select(list(reversed(self.windows)), self.decisions, self.settings)
        self.assertEqual(a, b)
        self.assertEqual(len(a), len({x["window_id"] for x in a}))
        self.assertEqual(r.Counter(x["group"] for x in a), {"high": 3, "borderline": 3, "unflagged": 2})
        for x in a:
            p = x["jev_probabilities"]
            if x["group"] == "unflagged": self.assertLess(max(p.values()), .5)

    def test_diagnostics_are_retained_once_and_shortfalls_are_explicit(self):
        windows = copy.deepcopy(self.windows)
        windows[0]["diagnostic_selected"] = True
        selected = r.select(windows, self.decisions, replace(self.settings, high_count=99))
        self.assertEqual(sum(x["window_id"] == windows[0]["window_id"] for x in selected), 1)
        self.assertEqual(next(x for x in selected if x["window_id"] == windows[0]["window_id"])["group"], "diagnostic")
        self.assertLess(sum(x["group"] == "high" for x in selected), 99)

    def test_payload_is_blinded_and_schema_is_strict(self):
        payload = self.requests[0]["payload"]
        state = json.loads(payload["messages"][1]["content"])
        for key in ("config", "device", "window_id", "jev_probabilities", "group", "provenance"):
            self.assertNotIn(key, state)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertFalse(payload["provider"]["allow_fallbacks"])

    def test_prepare_resume_preserves_sources_and_rejects_changed_method(self):
        before = r.digest(self.main / "jev_decisions.jsonl")
        r.prepare(self.main, self.out, self.settings)
        self.assertEqual(before, r.digest(self.main / "jev_decisions.jsonl"))
        with self.assertRaisesRegex(ValueError, "settings changed"):
            r.prepare(self.main, self.out, replace(self.settings, seed=42))

    def test_incomplete_first_stage_is_rejected(self):
        p = self.main / "jev_decisions.jsonl"
        r.write_jsonl(p, r.read_jsonl(p)[:-1])
        with self.assertRaisesRegex(ValueError, "Finish first-stage"):
            r.prepare(self.main, self.main / "new-review", self.settings)

    def test_disabled_run_never_prompts_or_reads_files(self):
        with patch("qc_review.getpass.getpass") as key, patch("qc_review.send") as send:
            self.assertIsNone(r.run(self.root / "missing", enabled=False))
            key.assert_not_called()
            send.assert_not_called()

    def test_invalid_evidence_and_missing_categories_rejected(self):
        payload = self.requests[0]["payload"]
        ann = annotation()
        ann["meaningful_omission"].update(verdict="present", evidence=[{
            "scope": "focal", "chunk_id": 0, "source_quote": "invented string", "target_quote": None}])
        with self.assertRaises(ValueError): r.validate_review(ann, payload)
        ann["meaningful_omission"]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "focal evidence"): r.validate_review(ann, payload)
        del ann["meaningful_omission"]
        with self.assertRaisesRegex(ValueError, "all four"): r.validate_review(ann, payload)

    def test_exact_evidence_passes_and_uncertain_remains_distinct(self):
        payload = self.requests[0]["payload"]
        c = json.loads(payload["messages"][1]["content"])["chunks"][0]
        ann = annotation()
        ann["meaningful_omission"].update(verdict="present", evidence=[{
            "scope": "focal", "chunk_id": c["chunk_id"], "source_quote": c["source_en"], "target_quote": None}])
        self.assertEqual(r.validate_review(ann, payload), ann)
        ann = annotation()
        ann["meaningful_omission"].update(verdict="uncertain", source_unclear=True)
        with patch("qc_review.send", return_value=raw_reply(ann)):
            r.run(self.out, enabled=True, max_requests=1, api_key="fake-key")
        summary, rows = r.report(self.out)
        self.assertTrue(any(x["comparison"] == "uncertain" for x in rows))
        self.assertTrue(any(x["comparison"] == "not_reviewed" for x in rows))

    def test_request_cap_cache_and_full_mock_batch(self):
        before = r.digest(self.main / "jev_decisions.jsonl")
        with patch("qc_review.send", return_value=raw_reply()) as send:
            result = r.run(self.out, enabled=True, max_requests=1, api_key="fake-key")
            self.assertEqual(result["new_attempts"], 1)
            r.run(self.out, enabled=True, max_requests=100, max_cost_usd=5, api_key="fake-key")
            calls = send.call_count
            result = r.run(self.out, enabled=True, max_requests=100, max_cost_usd=5, api_key="fake-key")
            self.assertEqual(result["new_attempts"], 0)
            self.assertEqual(send.call_count, calls)
        summary, rows = r.report(self.out)
        self.assertEqual(summary["successful_windows"], summary["selected_windows"])
        self.assertEqual(len(rows), len(self.requests) * 4)
        self.assertEqual(before, r.digest(self.main / "jev_decisions.jsonl"))

    def test_budget_reserves_unknown_cost_and_no_automatic_failure_retry(self):
        with patch("qc_review.send", side_effect=TimeoutError("fake-key")) as send:
            result = r.run(self.out, enabled=True, max_requests=3, api_key="fake-key")
            self.assertEqual(send.call_count, 1)
        self.assertGreater(result["accounted_cost_usd"], 0)
        self.assertNotIn("fake-key", (self.out / "review_attempts.jsonl").read_text())
        with patch("qc_review.send") as send:
            r.run(self.out, enabled=True, max_requests=3, max_cost_usd=0.001, api_key="fake-key")
            send.assert_not_called()
        first_fp = r.attempts(self.out)[0]["request_fingerprint"]
        with patch("qc_review.send", return_value=raw_reply()):
            r.run(self.out, enabled=True, max_requests=1, max_cost_usd=5, api_key="fake-key")
        self.assertNotEqual(first_fp, r.attempts(self.out)[1]["request_fingerprint"])

    def test_http_failure_retained_and_secret_redacted(self):
        with patch("qc_review.send", return_value=(401, '{"error":"fake-key"}')) as send:
            r.run(self.out, enabled=True, max_requests=3, api_key="fake-key")
            self.assertEqual(send.call_count, 1)
        row = r.attempts(self.out)[0]
        self.assertEqual(row["http_status"], 401)
        self.assertNotIn("fake-key", r.canonical(row))

    def test_truncation_fails_and_model_drift_stops(self):
        _, text = raw_reply()
        raw = json.loads(text)
        raw["choices"][0]["finish_reason"] = "length"
        with patch("qc_review.send", return_value=(200, json.dumps(raw))):
            r.run(self.out, enabled=True, api_key="fake-key")
        self.assertEqual(r.attempts(self.out)[0]["status"], "failed")
        with patch("qc_review.send", return_value=raw_reply(model="anthropic/claude-sonnet-4.5-new")):
            result = r.run(self.out, enabled=True, api_key="fake-key", retry_failed=True)
        self.assertEqual(result["new_attempts"], 1)
        self.assertIn("changed", r.attempts(self.out)[-1]["error"])

    def test_source_and_manifest_tampering_fail_before_send(self):
        p = self.out / "review_manifest.json"
        m = r.read_json(p)
        m["settings"]["max_tokens"] = 8192
        r.write_json(p, m)
        with patch("qc_review.send") as send:
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                r.run(self.out, enabled=True, api_key="fake-key")
            send.assert_not_called()

    def test_concurrent_run_rejected(self):
        with r.locked(self.out):
            with self.assertRaisesRegex(RuntimeError, "Another review"):
                with r.locked(self.out): pass

    def test_redirect_refused(self):
        with self.assertRaisesRegex(ValueError, "redirect refused"):
            r.NoRedirect().redirect_request(None)


if __name__ == "__main__":
    unittest.main()
