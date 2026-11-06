"""Behavioral tests: revision, overlap, EOF, trimming, and causal scheduling."""
import io
import unittest
import wave

from local_agreement import LocalAgreement2, Word, confirmed_segment_cut, replay


def words(text, start=0):
    return [Word(start + i, start + i + 0.8, word) for i, word in enumerate(text.split())]


class AgreementTests(unittest.TestCase):
    def test_requires_successive_hypotheses_and_stops_at_first_revision(self):
        la = LocalAgreement2()
        self.assertEqual(la.update(words("We want tea")), [])
        self.assertEqual([w.text for w in la.update(words("We want coffee today"))], ["We", "want"])
        self.assertEqual([w.text for w in la.pending], ["coffee", "today"])
        self.assertEqual([w.text for w in la.update(words("We want coffee tomorrow"))], ["coffee"])
        self.assertEqual([w.text for w in la.committed], ["We", "want", "coffee"])

    def test_changed_confirmed_text_cannot_retract_emitted_words(self):
        la = LocalAgreement2()
        la.update(words("We agree"))
        la.update(words("We agree"))
        self.assertEqual(la.update(words("They disagree now")), [])
        self.assertEqual([w.text for w in la.committed], ["We", "agree"])
        self.assertEqual([w.text for w in la.update(words("They disagree now"))], ["now"])

    def test_jittered_committed_overlap_is_not_emitted_twice(self):
        la = LocalAgreement2()
        la.update(words("Hello world"))
        la.update(words("Hello world"))
        jittered = [Word(1.75, 1.9, "world"), Word(2, 2.8, "again")]
        self.assertEqual(la.update(jittered), [])
        self.assertEqual([w.text for w in la.pending], ["again"])
        self.assertEqual([w.text for w in la.update(jittered)], ["again"])

    def test_empty_update_breaks_agreement_chain(self):
        la = LocalAgreement2()
        la.update(words("Hello"))
        la.update([])
        self.assertEqual(la.update(words("Hello")), [])
        self.assertEqual([w.text for w in la.update(words("Hello"))], ["Hello"])

    def test_eof_flush_is_idempotent_and_does_not_claim_confirmation(self):
        la = LocalAgreement2()
        la.update(words("latest tail"))
        self.assertEqual([w.text for w in la.finish()], ["latest", "tail"])
        self.assertEqual(la.committed, [])
        self.assertEqual(la.finish(), [])
        with self.assertRaises(ValueError):
            la.update([])

    def test_punctuation_and_case_are_part_of_agreement(self):
        la = LocalAgreement2()
        la.update(words("hello world"))
        self.assertEqual(la.update(words("Hello world")), [])
        self.assertEqual([w.text for w in la.update(words("Hello world."))], ["Hello"])

    def test_invalid_timestamps_rejected(self):
        for word in [Word(float("nan"), 1, "bad"), Word(2, 1, "bad"), Word(0, 1, "")]:
            with self.subTest(word=word), self.assertRaises(ValueError):
                LocalAgreement2().update([word])

    def test_trim_retains_last_segment_and_unconfirmed_audio(self):
        self.assertEqual(confirmed_segment_cut([5, 10, 19], 12, 0, 20, 15), 10)
        self.assertEqual(confirmed_segment_cut([5, 10, 19], 6, 0, 20, 15), 5)
        self.assertIsNone(confirmed_segment_cut([19], 19, 0, 20, 15))
        self.assertIsNone(confirmed_segment_cut([5, 19], None, 0, 20, 15))
        self.assertIsNone(confirmed_segment_cut([5, 10], 10, 0, 12, 15))


class FakeClock:
    def __init__(self):
        self.now = 100  # nonzero origin detects accidental absolute timestamps

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def wav_bytes(frames, rate=10):
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\0\0" * frames)
    data.seek(0)
    return wave.open(data, "rb")


class ReplayTests(unittest.TestCase):
    def run_replay(self, compute, frames=35, min_chunk=1, max_frames=None):
        clock = FakeClock()
        rows, sizes = [], []
        origin = clock()

        def infer(raw, end):
            self.assertLessEqual(end, clock() - origin + 1e-9)
            sizes.append(len(raw) // 2)
            clock.sleep(compute)
            return {"english": "test", "german": "Test"}

        def finish():
            clock.sleep(0.2)
            return {"english": "tail", "german": "Ende"}

        with wav_bytes(frames) as wav:
            wall = replay(wav, min_chunk, infer, finish, rows.append,
                          clock=clock, sleep=clock.sleep, max_frames=max_frames)
        return rows, sizes, wall

    def test_fast_inference_waits_on_one_clock_and_reads_partial_final_chunk(self):
        rows, sizes, wall = self.run_replay(0.25)
        self.assertEqual(sizes, [10, 10, 10, 5])
        self.assertEqual([r["source_end_s"] for r in rows], [1, 2, 3, 3.5, 3.5])
        self.assertAlmostEqual(wall, 3.95)
        self.assertEqual(rows[-1]["event"], "eof_flush")
        self.assertAlmostEqual(sum(r["process_s"] for r in rows), 1.2)

    def test_slow_inference_coalesces_arrived_audio_without_skipping_samples(self):
        rows, sizes, _ = self.run_replay(2.5, frames=60)
        self.assertEqual(sizes, [10, 25, 25])
        self.assertEqual(sum(sizes), 60)
        self.assertEqual([r["source_end_s"] for r in rows[:-1]], [1, 3.5, 6])
        self.assertAlmostEqual(rows[1]["update_wait_s"], 1.5)
        self.assertAlmostEqual(rows[-1]["emitted_s"], 8.7)

    def test_short_audio_and_prefix_limit(self):
        rows, sizes, _ = self.run_replay(0.1, frames=20, min_chunk=1, max_frames=5)
        self.assertEqual(sizes, [5])
        self.assertEqual(rows[-1]["source_end_s"], 0.5)

    def test_invalid_interval_and_empty_audio(self):
        for interval in [0, -1, float("nan"), float("inf"), 31]:
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.run_replay(0, min_chunk=interval)
        with self.assertRaises(ValueError):
            self.run_replay(0, frames=0)


if __name__ == "__main__":
    unittest.main()
