import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from translation_qc.common import canonical, digest, read_json, read_jsonl, write_json, write_jsonl
from translation_qc.inputs import build_windows, validate
from translation_qc.jev import (Budget, build_request, evaluate_request, judge, judge_lock,
                                plan_requests, select_windows)
from translation_qc.main_experiment import MAIN_AUDIO, prepare_main, select_paired
from translation_qc.reports import report
from translation_qc.tests.helpers import prepare_synthetic
from translation_qc.tests.test_jev import response, transport_response


def make_main_fixture(root):
    _, pilot_runs, _ = prepare_synthetic(root / "pilot")
    runs, evaluation = root / "main-runs", root / "evaluation"
    data = {"talks": [], "reference_segments": 266, "files_sha256": {}}
    scores = {"complete_grid": True, "manifest": data, "runs": []}
    for (talk, audio), count in zip(sorted(MAIN_AUDIO.items()), (48, 46, 32, 43, 52, 45)):
        ref = evaluation / "references" / talk
        ref.mkdir(parents=True)
        for language in ("en", "de"):
            (ref / f"{language}.txt").write_text("\n".join(f"Synthetic reference {i}." for i in range(count)) + "\n")
        (ref / "audio-segments.yaml").write_text("[]\n")
        for path in ref.iterdir():
            data["files_sha256"][f"{talk}/{path.name}"] = digest(path)
        data["talks"].append({"id": talk, "audio_sha256": audio, "duration_s": 50.5,
                              "reference_dir": talk, "reference_segments": count})
        for device in ("pi", "laptop"):
            for label in ("tiny", "base"):
                for chunk_s in (4, 8):
                    for rep in (1, 2):
                        job = {"id": f"{talk}-{label}-{chunk_s}s-r{rep}", "talk": talk,
                               "device": device, "asr": label + ".en", "chunk_s": chunk_s, "repetition": rep}
                        job_dir = runs / device / "jobs" / job["id"]
                        attempt = job_dir / "attempt-001"
                        src = next(pilot_runs.glob(f"*-{label}-{chunk_s}s-*-r1"))
                        shutil.copytree(src, attempt)
                        metadata = read_json(attempt / "metadata.json")
                        metadata["audio_sha256"] = audio
                        write_json(attempt / "metadata.json", metadata)
                        hypotheses = read_jsonl(attempt / "hypothesis.jsonl")
                        hypotheses[0]["source"] = talk + ".wav"
                        write_jsonl(attempt / "hypothesis.jsonl", hypotheses)
                        hashes = {p.name: digest(p) for p in attempt.iterdir()}
                        write_json(job_dir / "complete.json", {"job": job, "attempt": "attempt-001", "output_sha256": hashes})
                        segments_file = f"results/{device}/{job['id']}/segments.json"
                        segpath = evaluation / segments_file
                        segpath.parent.mkdir(parents=True)
                        write_json(segpath, [{"src": s} for s in (ref / "en.txt").read_text().splitlines()])
                        scores["runs"].append({**job, "key": device + "/" + job["id"], "input_sha256": hashes,
                                               "segments_file": segments_file, "segments_sha256": digest(segpath)})
    write_json(evaluation / "scores.json", scores)
    return runs, evaluation


class MainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "pilot").mkdir()
        cls.runs, cls.evaluation = make_main_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.output_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.output_temp.cleanup)
        self.out = Path(self.output_temp.name) / "qc"
        guard = patch("socket.socket", side_effect=AssertionError("Network forbidden in offline tests"))
        guard.start()
        self.addCleanup(guard.stop)
        self.windows, self.checks = prepare_main(self.runs, self.evaluation, self.out)

    def test_all_runs_audited_paired_sample_tail_and_context_preserved(self):
        windows, _, manifest = validate(self.out)
        self.assertEqual(manifest["audited_runs"], 96)
        self.assertEqual(manifest["retained_runs"], 48)
        self.assertEqual(len(windows), 144)
        self.assertEqual(sum(w["review_selected"] for w in windows), 96)
        first_run = windows[:3]
        self.assertEqual([(w["source_start_s"], w["source_end_s"]) for w in first_run], [(0, 24), (24, 48), (48, 50.5)])
        self.assertEqual(first_run[0]["context_before"], [])
        self.assertEqual(first_run[-1]["context_after"], [])
        self.assertEqual(first_run[1]["context_before"][0]["source_start_s"], 16)
        for passage in {w["paired_passage_id"] for w in windows}:
            rows = [w for w in windows if w["paired_passage_id"] == passage]
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({w["review_selected"] for w in rows}), 1)

    def test_sample_ignores_output_content_and_configuration_order(self):
        other = copy.deepcopy(self.windows)[::-1]
        for w in other:
            w["source_en"] = "changed contents"
            w["target_de"] = "changed contents"
        select_paired(other)
        self.assertEqual({w["window_id"] for w in self.windows if w["review_selected"]},
                         {w["window_id"] for w in other if w["review_selected"]})

    def test_requests_hide_device_scores_paths_and_include_context(self):
        request = build_request(self.windows[1])
        payload = canonical(request["payload"])
        for forbidden in ("audio_sha256", "emitted_s", "input_sha256", "wer_pct", str(self.root), "laptop"):
            self.assertNotIn(forbidden, payload)
        self.assertEqual(len(request["payload"]["state"]["context_before"]), 2)
        self.assertIn("context_after", request["payload"]["state"])

    def test_same_api_payload_shares_cache_preserving_each_mapping(self):
        first = self.windows[0]
        second = copy.deepcopy(first)
        second["window_id"] += "-other"
        second["content_fingerprint"] = "other-provenance"
        a, b = build_request(first), build_request(second)
        self.assertNotEqual(a["request_fingerprint"], b["request_fingerprint"])
        self.assertEqual(a["cache_fingerprint"], b["cache_fingerprint"])
        transport = Mock(return_value=transport_response())
        cache = self.out / "cache"
        evaluate_request(a, cache, Budget(), "synthetic-key", transport=transport)
        result = evaluate_request(b, cache, Budget(), "synthetic-key", transport=transport)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(result["window_id"], second["window_id"])
        self.assertEqual(result["cache_source_window_id"], first["window_id"])
        self.assertEqual(transport.call_count, 1)

    def test_dry_runs_archive_each_plan_and_use_no_network(self):
        a = judge(self.out, self.out / "windows.jsonl", limit=3)
        b = judge(self.out, self.out / "windows.jsonl", limit=6)
        self.assertEqual(a["network_requests"], b["network_requests"], 0)
        self.assertEqual(len(list((self.out / "invocations").glob("*/requests.jsonl"))), 2)
        self.assertEqual(read_jsonl(self.out / "jev_decisions.jsonl"), [])
        self.assertEqual(report(self.out)["review_subset"]["not_judged"], 96)

    def test_live_mock_supports_main_guard_and_locks_returned_model(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-key"}), \
                patch("translation_qc.jev.http_post", return_value=transport_response()) as transport:
            result = judge(self.out, self.out / "windows.jsonl", live=True, limit=3, retries=0)
        self.assertEqual(result["ok"], 3)
        self.assertLessEqual(transport.call_count, 3)
        self.assertEqual(read_json(self.out / "resolved_model.json")["model"], response()["model"])

    def test_changed_code_blocks_live_before_http(self):
        manifest = read_json(self.out / "manifest.json")
        manifest["qc_code_sha256"]["jev.py"] = "changed"
        write_json(self.out / "manifest.json", manifest)
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-key"}), \
                patch("translation_qc.jev.http_post", side_effect=AssertionError("Must not reach network")):
            with self.assertRaisesRegex(ValueError, "code changed"):
                judge(self.out, self.out / "windows.jsonl", live=True)

    def test_revision_drift_is_a_failed_judgment_not_a_cached_success(self):
        raw = response()
        raw["model"] = "typesafe/jev-1.13-new-revision"
        result = evaluate_request(build_request(self.windows[0]), self.out / "cache", Budget(), "synthetic-key",
                                  transport=Mock(return_value=transport_response(raw=raw)), expected_resolved_model=response()["model"])
        self.assertEqual(result["status"], "parse_error")
        self.assertFalse((self.out / "cache").exists())
        self.assertEqual(result["probabilities"], {})

    def test_diagnostic_selector_separate_from_fixed_sample(self):
        windows = copy.deepcopy(self.windows)
        windows[0]["diagnostic_selected"] = True
        self.assertEqual([w["window_id"] for w in select_windows(windows, selection="diagnostic", limit=None)],
                         [windows[0]["window_id"]])

    def test_concurrent_judge_rejected_before_network(self):
        with judge_lock(self.out):
            with self.assertRaisesRegex(ValueError, "Another judge"):
                judge(self.out, self.out / "windows.jsonl")


if __name__ == "__main__":
    unittest.main()
