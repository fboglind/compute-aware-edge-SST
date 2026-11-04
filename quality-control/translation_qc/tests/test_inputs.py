import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from translation_qc.common import read_jsonl
from translation_qc.inputs import audit_chunks, build_windows, select_review
from translation_qc.reports import export_review


def synthetic_chunks(chunk_s=4, duration=45.5):
    chunks = []
    index = 0
    while index * chunk_s < duration:
        start, end = index * chunk_s, min((index + 1) * chunk_s, duration)
        german = f"Ergebnis {index}."
        chunks.append({"chunk_index": index, "source_start_s": start, "source_end_s": end,
                       "chunk_duration_s": end - start, "processing_start_s": end + 0.01,
                       "emitted_s": end + 0.2, "english": f"Result {index}.", "german": german,
                       "words": german.split()})
        index += 1
    return chunks


def synthetic_windows(chunk_s=4, duration=45.5):
    return build_windows(synthetic_chunks(chunk_s, duration), f"synthetic-{chunk_s}", "tiny.en", chunk_s, {})


class InputTests(unittest.TestCase):
    def test_four_second_windows_tail_and_stable_ids(self):
        chunks = synthetic_chunks()
        audit_chunks(chunks, 4, 45.5)
        windows, checks = synthetic_windows()
        self.assertEqual([window["audio_span_s"] for window in windows], [20, 20, 5.5])
        self.assertEqual(windows[-1]["chunks"][-1]["chunk_duration_s"], 1.5)
        self.assertEqual(windows, synthetic_windows()[0])
        self.assertEqual([identifier for window in windows for identifier in window["chunk_ids"]], list(range(12)))

    def test_eight_second_windows_tail(self):
        audit_chunks(synthetic_chunks(8, 50.5), 8, 50.5)
        windows, checks = synthetic_windows(8, 50.5)
        self.assertEqual([window["audio_span_s"] for window in windows], [24, 24, 2.5])
        self.assertEqual(windows[0]["chunk_boundaries_s"], [8, 16])

    def test_empty_text_keeps_order_and_timestamps(self):
        chunks = synthetic_chunks()
        chunks[0]["english"] = " "
        chunks[1]["german"] = ""
        chunks[1]["words"] = []
        windows, checks = build_windows(chunks, "synthetic", "tiny.en", 4, {})
        self.assertEqual(windows[0]["source_en"], " ".join(chunk["english"] for chunk in chunks[:5]))
        self.assertEqual(windows[0]["chunks"][1]["source_end_s"], 8)
        self.assertEqual({warning["code"] for warning in checks[0]["warnings"]},
                         {"empty_en", "empty_de", "nonempty_en_zero_de"})

    def test_split_idea_is_not_missing_chunk(self):
        chunks = synthetic_chunks(4, 8)
        chunks[0].update(english="The model did", german="Das Modell hat")
        chunks[1].update(english="not fail.", german="nicht versagt.")
        for chunk in chunks:
            chunk["words"] = chunk["german"].split()
        audit_chunks(chunks, 4, 8)
        windows, checks = build_windows(chunks, "split", "tiny.en", 4, {})
        self.assertEqual(windows[0]["source_en"], "The model did not fail.")
        self.assertEqual(checks[0]["warnings"], [])
        self.assertEqual(checks[0]["semantic_status"], "not_judged")

    def test_missing_duplicate_out_of_order_ids(self):
        for bad_id in (None, 0, 3, True):
            with self.subTest(bad_id=bad_id):
                chunks = synthetic_chunks()
                chunks[1]["chunk_index"] = bad_id
                with self.assertRaisesRegex(ValueError, "chunk ID"):
                    audit_chunks(chunks, 4, 45.5)
        with self.assertRaisesRegex(ValueError, "chunk count"):
            audit_chunks(synthetic_chunks()[:-1], 4, 45.5)

    def test_gaps_missing_fields_and_export_mismatch(self):
        for change in ("gap", "missing", "words", "clock", "nan"):
            with self.subTest(change=change):
                chunks = synthetic_chunks()
                if change == "gap":
                    chunks[1]["source_start_s"] = 5
                elif change == "missing":
                    del chunks[1]["english"]
                elif change == "words":
                    chunks[1]["words"] = []
                elif change == "clock":
                    chunks[1]["emitted_s"] = 0
                else:
                    chunks[1]["source_end_s"] = float("nan")
                with self.assertRaises(ValueError):
                    audit_chunks(chunks, 4, 45.5)

    def test_jsonl_reports_bad_line_and_duplicate_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chunks.jsonl"
            for bad in ("broken", "", '{"id":1,"id":2}', '{"score":NaN}', "[]"):
                path.write_text('{}\n' + bad + '\n', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "chunks.jsonl:2"):
                    read_jsonl(path)

    def test_repetition_across_windows_is_candidate(self):
        chunks = synthetic_chunks()
        chunks[5]["german"] = chunks[0]["german"]
        chunks[5]["words"] = chunks[0]["words"]
        windows, checks = build_windows(chunks, "repeat", "tiny.en", 4, {})
        self.assertEqual(checks[1]["warnings"][0]["previous_chunk_id"], 0)
        self.assertEqual(checks[1]["warnings"][0]["kind"], "candidate")

    def test_selection_reproducible_and_spread(self):
        windows, checks = synthetic_windows(4, 357.540875)
        original = copy.deepcopy(windows)
        select_review(windows)
        select_review(original)
        self.assertEqual(windows, original)
        self.assertEqual(sum(window["review_selected"] for window in windows), 10)
        self.assertTrue(any(window["review_selected"] for window in windows[:2]))
        self.assertTrue(any(window["review_selected"] for window in windows[-2:]))

    def test_reference_is_never_exactly_aligned_and_exports_offline(self):
        windows, checks = synthetic_windows()
        select_review(windows)
        with tempfile.TemporaryDirectory() as temporary, patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            export_review(Path(temporary), windows, checks)
            review = (Path(temporary) / "review.md").read_text()
            self.assertIn("no sentence/window alignment", review)
            self.assertEqual(windows[0]["reference_evidence"]["alignment"], "whole_talk_unaligned")
            self.assertIn("not_judged", review)


if __name__ == "__main__":
    unittest.main()
