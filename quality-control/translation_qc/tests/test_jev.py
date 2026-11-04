import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from translation_qc.common import fingerprint
from translation_qc.jev import (Budget, MODEL, QUESTION_TEXT, build_request, evaluate_request,
                                judge, parse_response, plan_requests, select_windows)
from translation_qc.inputs import select_review
from translation_qc.tests.test_inputs import synthetic_windows


def response(scores=None):
    scores = scores or {}
    return {"model": MODEL + "-20260917", "provider": "mock-provider",
            "answers": {name: {"type": "noul", "noul": scores.get(name, 0.1)} for name in QUESTION_TEXT},
            "usage": {"input_tokens": 100, "output_tokens": 20, "cost": 0.0000042}}


def transport_response(status=200, raw=None, headers=None):
    return status, json.dumps(response() if raw is None else raw), headers or {}


class JevTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name) / "cache"
        self.window = synthetic_windows()[0][0]
        self.request = build_request(self.window)

    def evaluate(self, transport, **settings):
        return evaluate_request(self.request, self.cache, settings.pop("budget", Budget(3)), "test-secret",
                                transport=transport, sleep=settings.pop("sleep", Mock()), **settings)

    def test_request_uses_actual_pair_and_no_paths_or_telemetry(self):
        state = self.request["payload"]["state"]
        self.assertEqual(state["source_en"], self.window["source_en"])
        self.assertEqual(len(state["chunks"]), 5)
        self.assertNotIn("provenance", state)
        self.assertNotIn("emitted_s", json.dumps(state))
        self.assertEqual(set(self.request["payload"]["questions"]), set(QUESTION_TEXT))

    def test_response_valid_and_all_malformed_probabilities_rejected(self):
        self.assertEqual(parse_response(response())["meaningful_omission"], 0.1)
        for bad in (None, "0.9", True, float("nan"), float("inf"), -0.1, 1.1, {}):
            with self.subTest(bad=bad):
                raw = response()
                raw["answers"]["meaningful_omission"]["noul"] = bad
                with self.assertRaisesRegex(ValueError, "probability"):
                    parse_response(raw)

    def test_missing_question_model_wrong_type_and_error_object(self):
        for change in ("question", "model", "type", "error"):
            raw = response()
            if change == "question":
                del raw["answers"]["meaningful_omission"]
            elif change == "model":
                raw["model"] = "typesafe/jev-latest"
            elif change == "type":
                raw["answers"]["meaningful_omission"]["type"] = "choice"
            else:
                raw["error"] = {"message": "provider failed"}
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_response(raw)

    def test_429_and_503_retry_then_success_with_attempt_records(self):
        transport = Mock(side_effect=[transport_response(429, {"error": "rate"}, {"Retry-After": "2"}),
                                      transport_response(503, {"error": "unavailable"}), transport_response()])
        sleeper, recorder = Mock(), Mock()
        result = self.evaluate(transport, sleep=sleeper, on_attempt=recorder)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(recorder.call_count, 3)
        self.assertEqual(sleeper.call_count, 2)
        self.assertEqual(result["model"], MODEL + "-20260917")

    def test_timeout_does_not_retry_or_become_success(self):
        transport = Mock(side_effect=TimeoutError("test-secret"))
        result = self.evaluate(transport)
        self.assertEqual(result["status"], "transport_error")
        self.assertEqual(result["probabilities"], {})
        self.assertEqual(transport.call_count, 1)
        self.assertNotIn("test-secret", json.dumps(result))
        self.assertTrue(result["billing_uncertain"])

    def test_successful_retry_does_not_inherit_error_or_usage(self):
        failure = {"error": "unavailable", "usage": {"cost": 0.0001}, "provider": "failed-provider"}
        success = response()
        del success["usage"]
        del success["provider"]
        result = self.evaluate(Mock(side_effect=[transport_response(503, failure), transport_response(raw=success)]))
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("error", result)
        self.assertNotIn("usage", result)
        self.assertNotIn("provider", result)
        self.assertEqual(result["attempts"][0]["usage"]["cost"], 0.0001)

    def test_credential_not_persisted_even_if_provider_echoes_it(self):
        result = self.evaluate(Mock(return_value=transport_response(401, {"error": "test-secret"})))
        self.assertNotIn("test-secret", json.dumps(result))

    def test_http_cap_counts_retries_and_stops_other_windows(self):
        budget = Budget(2)
        transport = Mock(return_value=transport_response(429, {"error": "rate"}))
        result = self.evaluate(transport, budget=budget)
        self.assertEqual(result["status"], "http_error")
        skipped = self.evaluate(transport, budget=budget)
        self.assertEqual(skipped["status"], "skipped_budget")
        self.assertEqual(transport.call_count, 2)

    def test_invalid_json_is_parse_error_without_retry(self):
        transport = Mock(return_value=(200, "not json", {}))
        result = self.evaluate(transport)
        self.assertEqual(result["status"], "parse_error")
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result["raw_response_text"], "not json")

    def test_cache_resume_skips_http_and_versions_change_key(self):
        transport = Mock(return_value=transport_response())
        first = self.evaluate(transport)
        second = self.evaluate(transport)
        self.assertEqual(transport.call_count, 1)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        changed = build_request(self.window, question_version="mt-fidelity-v2")
        self.assertNotEqual(changed["request_fingerprint"], self.request["request_fingerprint"])
        evaluate_request(changed, self.cache, Budget(), "test-secret", transport=transport)
        self.assertEqual(transport.call_count, 2)
        modified = copy.deepcopy(self.window)
        modified["target_de"] += " Neu."
        self.assertNotEqual(build_request(modified)["request_fingerprint"], self.request["request_fingerprint"])

    def test_corrupt_cache_fails_without_http(self):
        transport = Mock(return_value=transport_response())
        self.evaluate(transport)
        cache_file = next(self.cache.glob("*.json"))
        record = json.loads(cache_file.read_text())
        record["probabilities"]["meaningful_omission"] = 0.9
        cache_file.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "cached response"):
            self.evaluate(transport)
        self.assertEqual(transport.call_count, 1)

    def test_dry_run_never_connects_even_with_key(self):
        windows, checks = synthetic_windows()
        select_review(windows)
        with patch("socket.socket", side_effect=AssertionError("Network forbidden")), \
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "exists"}), \
                patch("translation_qc.jev.validate", return_value=(windows, checks, {})), \
                patch("translation_qc.jev.http_post", side_effect=AssertionError("HTTP forbidden")):
            directory = Path(self.temporary.name)
            result = judge(directory, directory / "windows.jsonl")
            self.assertEqual(result["network_requests"], 0)
            self.assertEqual(result["new_requests"], 3)
            self.assertFalse((directory / "jev_decisions.jsonl").exists())
            self.assertGreater(result["rough_cost_usd"], 0)

    def test_caps_and_missing_key(self):
        for value in (0, -1, 101, True):
            with self.assertRaises(ValueError):
                Budget(value)
        result = self.evaluate(Mock(side_effect=AssertionError("Budget must block HTTP")), budget=Budget(3, 1e-12))
        self.assertEqual(result["status"], "skipped_budget")
        windows, checks = synthetic_windows()
        select_review(windows)
        with patch.dict("os.environ", {}, clear=True), patch("translation_qc.jev.validate", return_value=(windows, checks, {})):
            directory = Path(self.temporary.name)
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                judge(directory, directory / "windows.jsonl", live=True)

    def test_unknown_usage_is_not_fabricated(self):
        raw = response()
        del raw["usage"]
        result = self.evaluate(Mock(return_value=transport_response(raw=raw)))
        self.assertNotIn("usage", result)

    def test_oversize_request_and_alias_rejected(self):
        window = copy.deepcopy(self.window)
        window["source_en"] = "word " * 10000
        with self.assertRaisesRegex(ValueError, "32,000"):
            build_request(window)
        with self.assertRaisesRegex(ValueError, "pinned"):
            build_request(self.window, "typesafe/jev-latest")


if __name__ == "__main__":
    unittest.main()
