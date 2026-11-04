"""Regression and integrity tests. No models, network, or real-time replay needed."""
import copy
import argparse
import contextlib
import io
import json
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from common import (ROOT, GRID, audit_run, completed_attempt, fingerprint,
                    jobs_for, plan_for, read_json, require, sha, validate_plan, write_json)
from evaluate_main import clock_control, latency, score_one
from experiment import finish_job, prepare, run_batch
from score_comet_main import build_samples

FIXTURE = ROOT / "tests/fixtures/pilot"


class ProtocolTests(unittest.TestCase):
    def test_pooled_metrics_use_their_correct_denominators(self):
        from summarize_main import summarize
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = []
            for ident, seconds, processing, refs, errors, count, lag in [
                ("a", 10, 2, 10, 2, 1, 1), ("b", 30, 30, 90, 9, 3, 3)]:
                pairs = [{"src": "Hello", "prediction": "Hallo", "reference": "Hallo"}] * count
                write_json(root / (ident + ".json"), pairs)
                runs.append({"key": ident, "talk": ident, "device": "pi", "asr": "tiny.en", "chunk_s": 4, "repetition": 1,
                             "segments_file": ident + ".json", "segments_sha256": sha(root / (ident + ".json")),
                             "audio_duration_s": seconds, "total_processing_s": processing,
                             "total_asr_s": processing * .75, "total_mt_s": processing * .25,
                             "wer_errors": errors, "wer_reference_words": refs,
                             "first_output_s": 5, "final_tail_s": 1, "queue_wait_max_s": 0,
                             "sampled_peak_replay_rss_mib": 400, "process_peak_rss_mib_including_load_warmup": 500,
                             "empty_asr_chunks": 0, "mt_limit_chunks": 0, "resource_sampling_error": None,
                             "latency": {c: {"segment_scores_s": [lag] * count, "words_at_or_after_wav_end": 0} for c in ("cu", "ca")}})
            write_json(root / "scores.json", {"manifest": {"talks": [{"id": "a"}, {"id": "b"}]}, "runs": runs, "complete_grid": False})
            write_json(root / "comet-scores.json", {"scores_json_sha256": sha(root / "scores.json"),
                       "segments_sha256": {r["key"]: r["segments_sha256"] for r in runs},
                       "runs": {"a": {"segment_scores": [.2]}, "b": {"segment_scores": [.6, .6, .6]}}})
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(root)
            score = read_json(root / "summary.json")["passes"][0]
            self.assertAlmostEqual(score["rtf_processing"], .8)
            self.assertAlmostEqual(score["wer_pct"], 11)
            self.assertAlmostEqual(score["longyaal_ca_s"], 2.5)
            self.assertAlmostEqual(score["comet"], .5)
            self.assertAlmostEqual(score["chrf2"], 100)

    def test_preparation_freezes_audio_and_references(self):
        # Synthetic silence tests packaging ONLY. It never leaves the temporary
        # directory and is never presented as experiment audio or a measured run.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "mcif"
            (source / "ref").mkdir(parents=True)
            (source / "audio").mkdir()
            official = ROOT / "tests/fixtures/official-reference-inputs"
            shutil.copyfile(official / "audio-segments.yaml", source / "audio-segments.yaml")
            for lang in ("en", "de"):
                shutil.copyfile(official / (lang + ".txt"), source / "ref" / (lang + ".txt"))
            for talk in read_json(ROOT / "selection.json")["talks"]:
                with wave.open(str(source / "audio" / (talk["id"] + ".wav")), "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    wav.writeframes(bytes(round((talk["last_reference_end_s"] + 1) * 16000) * 2))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(argparse.Namespace(dataset=source, out=root / "prepared"))
            from common import dataset
            m = dataset(root / "prepared")
            self.assertEqual(len(m["talks"]), 6)
            self.assertEqual(m["reference_segments"], 266)
            self.assertTrue(1800 <= m["duration_s"] <= 2400)
            with self.assertRaises(ValueError):
                prepare(argparse.Namespace(dataset=source, out=root / "prepared"))
            wav_path = root / "prepared" / m["talks"][0]["audio"]
            with wav_path.open("ab") as f:
                f.write(b"changed")
            with self.assertRaises(ValueError):
                dataset(root / "prepared")

    def test_selection_and_balanced_run_order(self):
        selection = read_json(ROOT / "selection.json")
        talks = selection["talks"]
        self.assertEqual(len(talks), 6)
        self.assertEqual(sum(t["reference_segments"] for t in talks), 266)
        self.assertNotIn(selection["excluded_pilot"], [t["id"] for t in talks])
        for device in ("pi", "laptop"):
            jobs = jobs_for({"talks": talks}, device)
            self.assertEqual(len(jobs), 48)
            self.assertEqual(len({j["id"] for j in jobs}), 48)
            first = [(j["talk"], j["asr"], j["chunk_s"]) for j in jobs[:24]]
            second = [(j["talk"], j["asr"], j["chunk_s"]) for j in jobs[24:]]
            self.assertEqual(first, second[::-1])

    def test_plan_rejects_config_or_manifest_drift(self):
        m = {"manifest_id": "fixture", "talks": read_json(ROOT / "selection.json")["talks"]}
        plan = plan_for(m, "pi")
        validate_plan(plan, m)
        changed = copy.deepcopy(plan)
        changed["jobs"][0]["chunk_s"] = 2
        with self.assertRaises(ValueError):
            validate_plan(changed, m)
        with self.assertRaises(ValueError):
            validate_plan(plan, dict(m, manifest_id="different"))

    def test_known_clock_and_full_wav_cutoff(self):
        self.assertEqual(clock_control()["status"], "passed")
        rows = [{"prediction": "eins zwei", "reference": "eins zwei", "source_length": 4000,
                 "time_to_recording_end": 4000, "emission_cu": [3900, 4100], "emission_ca": [4500, 4700]}]
        value = latency(rows, [{"offset": 0}], 5)
        self.assertEqual(value["ca"]["eligible_segments"], 1)
        self.assertEqual(value["ca"]["upstream_eligible_segments"], 0)
        self.assertEqual(value["cu"]["longyaal_s"], 3.0)
        self.assertEqual(value["ca"]["longyaal_s"], 3.6)

    def test_empty_latency_is_missing_not_zero(self):
        rows = [{"prediction": "", "reference": "Hallo", "source_length": 1000,
                 "time_to_recording_end": 2000, "emission_cu": [], "emission_ca": []}]
        value = latency(rows, [{"offset": 0}], 2)
        self.assertIsNone(value["ca"]["longyaal_s"])
        self.assertEqual(value["ca"]["eligible_segments"], 0)

    def test_exact_wav_end_is_excluded_with_fractional_offset(self):
        duration, offset = 357.540875, 356.438
        end_relative = duration * 1000 - offset * 1000
        rows = [{"prediction": "Danke", "reference": "Danke", "source_length": 860,
                 "time_to_recording_end": 860, "emission_cu": [end_relative], "emission_ca": [end_relative + 100]}]
        value = latency(rows, [{"offset": offset}], duration)
        self.assertEqual(value["cu"]["eligible_segments"], 0)

    def test_comet_deduplicates_without_losing_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pairs = [{"src": "hello", "prediction": "hallo", "reference": "Hallo"}] * 2
            write_json(root / "segments.json", pairs)
            report = {"runs": [{"key": "pi/a", "segments_file": "segments.json", "quality_segments": 2,
                                "segments_sha256": sha(root / "segments.json")},
                               {"key": "laptop/a", "segments_file": "segments.json", "quality_segments": 2,
                                "segments_sha256": sha(root / "segments.json")}]}
            samples, spans, hashes = build_samples(report, root)
            self.assertEqual(len(samples), 1)
            self.assertEqual(spans, {"pi/a": [0, 0], "laptop/a": [0, 0]})


class PilotRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = read_json(FIXTURE / "expected-scores.json")["runs"]
        cls.results = []

    def test_four_pilot_runs_reproduce(self):
        with tempfile.TemporaryDirectory() as temp:
            for expected in self.expected:
                run = FIXTURE / "runs" / expected["run"]
                meta = read_json(run / "metadata.json")
                talk = {"id": "JhbtCwcsWY", "audio": "JhbtCwcsWY.wav", "duration_s": meta["audio_duration_s"],
                        "audio_sha256": meta["audio_sha256"]}
                job = {"id": expected["run"], "talk": talk["id"], "asr": expected["asr"],
                       "chunk_s": expected["chunk_s"], "device": "pi", "repetition": 1, "order": 1}
                result, rows = score_one(run, talk, job, FIXTURE / "references", Path(temp) / expected["run"])
                for got, old in [("chrf2", "resegmented_chrf2"), ("longyaal_cu_s", "longyaal_cu_s"),
                                 ("longyaal_ca_s", "longyaal_ca_s"), ("wer_pct", "wer_pct")]:
                    self.assertAlmostEqual(result[got], expected[old], places=9, msg=f"{expected['run']} {got}")
                self.assertEqual(len(rows), 39)
                self.assertEqual(result["latency"]["ca"]["eligible_segments"], 38)
                self.results.append({"run": expected["run"], "chrf2": result["chrf2"],
                                     "longyaal_cu_s": result["longyaal_cu_s"], "longyaal_ca_s": result["longyaal_ca_s"],
                                     "wer_pct": result["wer_pct"], "timestamp_roundtrip": "passed"})

    def test_corrupt_time_export_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            src = FIXTURE / "runs" / self.expected[0]["run"]
            dest = Path(temp) / "attempt-001"
            shutil.copytree(src, dest)
            meta = read_json(dest / "metadata.json")
            talk = {"audio": "JhbtCwcsWY.wav", "duration_s": meta["audio_duration_s"], "audio_sha256": meta["audio_sha256"]}
            job = {"asr": meta["args"]["asr"], "chunk_s": meta["args"]["chunk_s"]}
            hyp = read_json(dest / "hypothesis.jsonl")
            hyp["elapsed"][0] /= 1000
            (dest / "hypothesis.jsonl").write_text(json.dumps(hyp) + "\n")
            with self.assertRaisesRegex(ValueError, "Word timestamps"):
                audit_run(dest, talk, job)

    def test_completed_runs_skip_only_with_intact_hashes(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            src = FIXTURE / "runs" / self.expected[0]["run"]
            run = folder / "attempt-001"
            shutil.copytree(src, run)
            meta = read_json(run / "metadata.json")
            talk = {"audio": "JhbtCwcsWY.wav", "duration_s": meta["audio_duration_s"], "audio_sha256": meta["audio_sha256"]}
            job = {"asr": meta["args"]["asr"], "chunk_s": meta["args"]["chunk_s"]}
            plan = {"plan_id": "fixture"}
            self.assertIsNone(completed_attempt(folder, plan, job, talk))
            finish_job(folder, run, plan, job, "fixture")
            self.assertEqual(completed_attempt(folder, plan, job, talk), run)
            with (run / "translation.de.txt").open("a") as stream:
                stream.write("Changed")
            with self.assertRaises(ValueError):
                completed_attempt(folder, plan, job, talk)

    def test_batch_launch_resume_and_preserved_incomplete_attempt(self):
        meta = read_json(FIXTURE / "runs/mcif-JhbtCwcsWY-tiny-4s-mtbeam4-r1/metadata.json")
        talk = {"id": "JhbtCwcsWY", "audio": "JhbtCwcsWY.wav", "duration_s": meta["audio_duration_s"], "audio_sha256": meta["audio_sha256"]}
        m = {"talks": [talk], "manifest_id": "test-fixture-only", "duration_s": talk["duration_s"]}
        launched = []
        class FakeChild:
            def __init__(self, command, **kwargs):
                out = Path(command[command.index("--out")+1])
                self.target = out
                label = command[command.index("--asr")+1].removesuffix(".en")
                seconds = command[command.index("--chunk-s")+1]
                fixture = FIXTURE / "runs" / f"mcif-JhbtCwcsWY-{label}-{seconds}s-mtbeam4-r1"
                shutil.copytree(fixture, out)
                launched.append(out)
            def wait(self, **kwargs):
                return 0
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(data=root / "data", out=root / "runs", models=root / "models", device="pi",
                                      dry_run=False, preflight_only=False, notes="synthetic harness test", max_jobs=1, retry_incomplete=False)
            with patch('experiment.dataset', return_value=m), patch('experiment.hardware', return_value={}), \
                 patch('experiment.check_runtime', return_value={}), patch('experiment.subprocess.Popen', FakeChild), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_batch(args)
                self.assertEqual(len(launched), 1)
                first = args.out / "jobs/JhbtCwcsWY-tiny-4s-r1"
                self.assertTrue((first / "complete.json").exists())
                # Simulate parent death after successful inference but before
                # writing complete.json. Resume must adopt, not repeat it.
                (first / "complete.json").unlink()
                run_batch(args)
                self.assertEqual(len(launched), 2)
                self.assertTrue(read_json(first / "complete.json")["recovered"])
                # Next config has a failed partial attempt; default must stop.
                plan = read_json(args.out / "plan.json")
                job = plan["jobs"][2]
                partial = args.out / "jobs" / job["id"] / "attempt-001"
                partial.mkdir(parents=True)
                write_json(partial / "attempt.json", {"plan_id": plan["plan_id"], "job": job})
                (partial / "console.log").write_text("interrupted fixture")
                with self.assertRaises(ValueError):
                    run_batch(args)
                args.retry_incomplete = True
                run_batch(args)
                self.assertEqual((partial / "console.log").read_text(), "interrupted fixture")
                self.assertTrue((partial.parent / "attempt-002/summary.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
