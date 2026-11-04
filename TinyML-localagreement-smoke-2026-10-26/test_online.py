"""Integration checks for buffering, prompts, MT routing, and saved evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from check_smoke import validate
from local_agreement import Word
from run_localagreement import OnlineCascade


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.translations = []

    def transcribe_words(self, audio, prompt):
        self.calls.append((len(audio), prompt))
        if len(self.calls) <= 2:
            text = ["Alpha", "Beta", "Gamma"] + (["Delta"] if len(self.calls) == 2 else [])
            return [Word(i, i + 0.8, t) for i, t in enumerate(text)], [2, 4, 8]
        return [Word(i, i + 0.8, t) for i, t in enumerate(["Gamma", "Delta", "Epsilon"])], [2, 7]

    def translate(self, english):
        self.translations.append(english)
        return english.upper(), len(english.split()), False


class OnlineTests(unittest.TestCase):
    def test_trim_and_prompt_preserve_pending_audio_and_route_only_new_commits(self):
        backend = FakeBackend()
        online = OnlineCascade(backend, trim_s=5)
        first = online.update(b"\0\0" * 64000, 4)
        second = online.update(b"\0\0" * 64000, 8)
        third = online.update(b"\0\0" * 16000, 9)
        tail = online.finish()
        self.assertEqual([first["english"], second["english"], third["english"], tail["english"]],
                         ["", "Alpha Beta Gamma", "Delta", "Epsilon"])
        self.assertEqual(second["trim_at_s"], 2)
        self.assertEqual(backend.calls, [(64000, ""), (128000, ""), (112000, "Alpha Beta")])
        self.assertEqual(third["buffer_start_s"], 2)
        self.assertEqual(tail["commit_reason"], "eof_unconfirmed")
        self.assertEqual(tail["asr_s"], 0)
        self.assertEqual(backend.translations, ["", "Alpha Beta Gamma", "Delta", "Epsilon"])

    def test_noncontiguous_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "samples"):
            OnlineCascade(FakeBackend()).update(b"\0\0" * 16000, 2)


class EvidenceTests(unittest.TestCase):
    def fixture(self, root):
        event = {"chunk_index": 0, "source_start_s": 0, "source_end_s": 1,
                 "processing_start_s": 1, "emitted_s": 1.2, "process_s": 0.2,
                 "asr_s": 0.1, "mt_s": 0.1, "english": "Hello", "german": "Hallo"}
        word = {"word_index": 0, "word": "Hallo", "chunk_index": 0,
                "source_consumed_s": 1, "emitted_s": 1.2}
        documents = {
            "metadata.json": {"audio_duration_s": 1},
            "summary.json": {"complete": True, "total_processing_s": 0.2, "rtf_processing": 0.2},
            "chunks.jsonl": event, "words.jsonl": word,
            "hypothesis.jsonl": {"prediction": "Hallo", "delays": [1000], "elapsed": [1200], "source_length": 1000},
        }
        for name, value in documents.items():
            (root / name).write_text(json.dumps(value) + "\n")
        (root / "translation.de.txt").write_text("Hallo\n")
        (root / "transcript.en.txt").write_text("Hello\n")

    def test_saved_timestamp_roundtrip_passes_and_wrong_units_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            self.assertTrue(validate(root)[2]["passed"])
            path = root / "hypothesis.jsonl"
            value = json.loads(path.read_text())
            value["elapsed"] = [1.2]
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "unit mismatch"):
                validate(root)

    def test_future_audio_and_lost_word_are_detected(self):
        for field, value, error in [("processing_start_s", 0.5, "before arrival"),
                                    ("german", "Hallo Welt", "length mismatch")]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.fixture(root)
                path = root / "chunks.jsonl"
                event = json.loads(path.read_text())
                event[field] = value
                path.write_text(json.dumps(event))
                with self.assertRaisesRegex(ValueError, error):
                    validate(root)


if __name__ == "__main__":
    unittest.main()
