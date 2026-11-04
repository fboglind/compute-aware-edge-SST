"""Protect the measurement semantics without needing models or real-time waits."""
import io
import unittest
import wave
from stream_timing import replay


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def wav_bytes(frames):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(10)
        out.writeframes(bytes(frames * 2))
    buf.seek(0)
    return buf


class TimingTests(unittest.TestCase):
    def scenario(self, processing_s, frames=60, empty=False):
        clock = Clock()
        rows = []
        lengths = []
        def infer(raw):
            lengths.append(len(raw) // 2)
            self.assertGreaterEqual(clock(), sum(lengths) / 10)
            clock.sleep(processing_s)
            return {"english": "hello", "german": "" if empty else "Hallo Welt"}
        with wave.open(wav_bytes(frames), "rb") as wav:
            replay(wav, 2, infer, rows.append, clock=clock, sleep=clock.sleep)
        return rows, lengths

    def test_fast_processing_keeps_original_arrival_clock(self):
        rows, _ = self.scenario(.5)
        self.assertEqual([r["processing_start_s"] for r in rows], [2, 4, 6])
        self.assertEqual([r["emitted_s"] for r in rows], [2.5, 4.5, 6.5])

    def test_slow_processing_accumulates_queue_without_dropping_audio(self):
        rows, lengths = self.scenario(3)
        self.assertEqual([r["processing_start_s"] for r in rows], [2, 5, 8])
        self.assertEqual([r["queue_wait_s"] for r in rows], [0, 1, 2])
        self.assertEqual([r["emitted_s"] for r in rows], [5, 8, 11])
        self.assertEqual(sum(lengths), 60)

    def test_final_partial_chunk_uses_its_actual_end(self):
        rows, lengths = self.scenario(.5, frames=45)
        self.assertEqual(lengths, [20, 20, 5])
        self.assertEqual(rows[-1]["source_end_s"], 4.5)
        self.assertEqual(rows[-1]["emitted_s"], 5)

    def test_empty_translation_keeps_chunk_timing(self):
        rows, _ = self.scenario(.5, empty=True)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["words"] == [] for r in rows))


if __name__ == "__main__":
    unittest.main()
