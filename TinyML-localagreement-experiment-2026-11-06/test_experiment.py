"""Code-only regressions: synthetic logs and clocks; no model loading or replay."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
from common import ROOT, audit_run, jobs_for, plan_for, read_json, sha, write_json
from timestamp_validation import normalize_empty_timestamps


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"manifest_id": "synthetic", "talks": read_json(ROOT / "selection.json")["talks"]}

    def test_grid_and_balancing(self):
        for device in ("pi", "laptop"):
            jobs = jobs_for(self.manifest, device)
            self.assertEqual(len(jobs), 48)
            self.assertEqual(len({j["id"] for j in jobs}), 48)
            self.assertEqual([(j["talk"], j["asr"], j["min_chunk_s"]) for j in jobs[:24]],
                             list(reversed([(j["talk"], j["asr"], j["min_chunk_s"]) for j in jobs[24:]])))
        self.assertEqual(jobs_for(self.manifest, "pi")[0]["asr"], "tiny.en")
        self.assertEqual(jobs_for(self.manifest, "laptop")[0]["asr"], "base.en")

    def test_plan_freezes_policy_and_shared_source(self):
        plan = plan_for(self.manifest, "pi")
        common.validate_plan(plan, self.manifest)
        self.assertTrue(any(k.endswith("runner/run_pilot.py") for k in plan["source_sha256"]))
        plan["settings"]["buffer_trim_s"] = 30
        with self.assertRaises(ValueError):
            common.validate_plan(plan, self.manifest)

    def test_dry_run_cannot_preflight_or_launch(self):
        import argparse
        import experiment
        plan = {"jobs": [{"order": 1, "id": "synthetic"}]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"XDG_RUNTIME_DIR": tmp}), \
                patch.object(experiment, "make_plan", return_value=({}, plan)), \
                patch.object(experiment, "check_runtime", side_effect=AssertionError("inference")), \
                patch.object(experiment.subprocess, "Popen", side_effect=AssertionError("launch")):
            experiment.run_batch(argparse.Namespace(dry_run=True))


class AuditTests(unittest.TestCase):
    def fixture(self, root, empty=False):
        word = {"start": 0, "end": .8, "text": "Hello"}
        rows = []
        for i in range(3):
            eof = i == 2
            en = "Hello" if i == 1 and not empty else ""
            row = dict(update_index=i, event="eof_flush" if eof else "update",
                       source_start_s=min(i, 2), source_end_s=min(i+1, 2),
                       processing_start_s=[1, 2, 2.2][i], emitted_s=[1.2, 2.2, 2.3][i],
                       process_s=.1 if eof else .2, asr_s=0 if eof else .1,
                       mt_s=.05, policy_s=.01, english=en, german="Hallo" if en else "",
                       commit_reason="eof_unconfirmed" if eof else "agreement",
                       confirmed_words=[word] if en else [], pending_words=[],
                       mt_limit_reached=False, update_ready_s=min(i+1, 2), update_wait_s=0)
            if eof:
                row["flushed_words"] = []
            else:
                row.update(asr_hypothesis=[] if empty else [word], buffer_start_s=0,
                           decoded_audio_s=i+1, trim_at_s=None)
                row["pending_words"] = [word] if i == 0 and not empty else []
            rows.append(row)
        words = [] if empty else [dict(word_index=0, word="Hallo", update_index=1,
                                      source_consumed_s=2, emitted_s=2.2)]
        expected = read_json(ROOT / "expected-models.json")
        shared = ROOT.parent / "TinyML-main-experiment-2026-10-18" / "runner"
        paths = [ROOT / "run_localagreement.py", ROOT / "local_agreement.py",
                 shared / "run_pilot.py", shared / "stream_timing.py"]
        meta = dict(audio_sha256="synthetic", audio_duration_s=2, audio_total_duration_s=2,
                    source_offset_s=0, args=dict(asr="tiny.en", min_chunk_s=1, buffer_trim_s=15,
                                                threads=4, max_audio_s=None),
                    script_sha256={str(p.relative_to(ROOT.parent)): sha(p) for p in paths},
                    effective_compute_type={"asr": "int8_float32", "mt": "int8_float32"},
                    models={"asr": expected["asr-tiny.en"], "mt": expected["mt-en-de-int8"]},
                    decoding=dict(asr_beam=1, mt_beam=4, temperature=0, vad=False,
                                  condition_on_previous_text=False, word_timestamps=True,
                                  without_timestamps=False, prompt_max_words=200, asr_language="en",
                                  asr_task="transcribe", mt_max_output_tokens=256, mt_max_input_tokens=512))
        summary = dict(complete=True, total_processing_s=.5, rtf_processing=.25,
                       total_asr_s=.2, total_mt_s=.15, total_policy_s=.03,
                       final_tail_s=.3, agreement_words=0 if empty else 1, eof_unconfirmed_words=0,
                       updates=2, output_words=len(words), agreement_commits=0 if empty else 1,
                       buffer_trims=0, mt_limit_updates=0, update_wait_max_s=0,
                       max_decoded_buffer_s=2, decoded_audio_total_s=3)
        write_json(root / "metadata.json", meta)
        write_json(root / "summary.json", summary)
        for name, values in [("updates.jsonl", rows), ("words.jsonl", words)]:
            (root / name).write_text("".join(json.dumps(v)+"\n" for v in values))
        write_json(root / "hypothesis.jsonl", dict(prediction="" if empty else "Hallo",
                   delays=[] if empty else [2000], elapsed=[] if empty else [2200], source_length=2000))
        (root / "transcript.en.txt").write_text("" if empty else "Hello\n")
        (root / "translation.de.txt").write_text("" if empty else "Hallo\n")
        return {"audio_sha256": "synthetic", "duration_s": 2}, {"asr": "tiny.en", "min_chunk_s": 1}

    def test_valid_and_empty_outputs_are_accepted(self):
        for empty in (False, True):
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                talk, job = self.fixture(root, empty)
                self.assertEqual(audit_run(root, talk, job)[3], "" if empty else "Hallo")

    def test_tampering_is_rejected(self):
        cases = [("metadata.json", "audio_sha256", "wrong"),
                 ("metadata.json", "audio_total_duration_s", 4),
                 ("metadata.json", "effective_compute_type", {}),
                 ("metadata.json", "script_sha256", {}),
                 ("metadata.json", "decoding", {}),
                 ("summary.json", "total_policy_s", 4),
                 ("summary.json", "agreement_words", 99),
                 ("summary.json", "decoded_audio_total_s", 99),
                 ("hypothesis.jsonl", "elapsed", [2.2])]
        for name, field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); talk, job = self.fixture(root)
                doc = read_json(root / name); doc[field] = value; write_json(root / name, doc)
                with self.assertRaises(ValueError):
                    audit_run(root, talk, job)

    def test_causality_agreement_and_eof_tampering_rejected(self):
        cases = [(0, "processing_start_s", .5), (0, "asr_hypothesis", []),
                 (1, "update_ready_s", 1), (2, "commit_reason", "agreement")]
        for index, key, value in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); talk, job = self.fixture(root)
                rows = common.read_jsonl(root / "updates.jsonl"); rows[index][key] = value
                (root / "updates.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
                with self.assertRaises(ValueError):
                    audit_run(root, talk, job)

    def test_completed_attempt_checks_hashes_before_skip(self):
        import experiment
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); run = folder / "attempt-001"; run.mkdir()
            talk, job = self.fixture(run); plan = {"plan_id": "synthetic"}
            experiment.finish_job(folder, run, plan, job, "test")
            self.assertEqual(common.completed_attempt(folder, plan, job, talk), run)
            (run / "translation.de.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                common.completed_attempt(folder, plan, job, talk)


class TimestampTests(unittest.TestCase):
    def test_only_empty_missing_timestamps_are_repaired(self):
        rows, changes = normalize_empty_timestamps([{"prediction": "", "emission_ca": None}])
        self.assertEqual(rows[0]["emission_cu"], [])
        self.assertEqual(len(changes), 1)
        for row in [{"prediction": "Hallo"},
                    {"prediction": "Hallo", "emission_cu": [float("nan")], "emission_ca": [1]}]:
            with self.assertRaises(ValueError):
                normalize_empty_timestamps([row])


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("omnisteval", "jiwer", "sacrebleu")),
                     "Pinned scoring environment is not installed")
class ScoringTests(unittest.TestCase):
    def test_clock_control_and_exact_end_exclusion(self):
        from evaluate_localagreement import clock_control, latency
        self.assertEqual(clock_control()["status"], "passed")
        rows = [dict(prediction="eins", reference="eins", source_length=1000,
                     time_to_recording_end=1000, emission_cu=[1000], emission_ca=[1200])]
        result = latency(rows, [{"offset": 0}], 1)
        self.assertIsNone(result["cu"]["longyaal_s"])
        self.assertIsNone(result["ca"]["longyaal_s"])


if __name__ == "__main__":
    unittest.main()
