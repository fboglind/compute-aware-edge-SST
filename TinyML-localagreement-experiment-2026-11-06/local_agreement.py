"""LocalAgreement-2 and causal, adaptive replay. No inference dependencies.

Algorithm reference: Machacek et al. (2023), section 3, and the authors'
ufal/whisper_streaming implementation (HypothesisBuffer/OnlineASRProcessor).
This is a project-specific implementation, not a vendored upstream runtime.
"""
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


class LocalAgreement2:
    """Commit exact, case/punctuation-sensitive prefixes of successive tails.

    Timestamps are absolute audio times, not output emission times. Previously
    committed text stays fixed. Timestamp filtering plus a 1--5 word overlap
    check within one second tolerates small alignment shifts, as in the paper.
    """
    def __init__(self):
        self.pending = []
        self.committed = []
        self.last_end = None
        self.finished = False

    def update(self, words):
        if self.finished:
            raise ValueError("Cannot update a finished agreement stream")
        words = list(words)
        for word in words:
            if not (math.isfinite(word.start) and math.isfinite(word.end)
                    and 0 <= word.start <= word.end and word.text.strip()):
                raise ValueError(f"Invalid ASR word: {word!r}")
        if self.last_end is not None:
            words = [w for w in words if w.start > self.last_end - 0.1]
            if words and abs(words[0].start - self.last_end) < 1:
                # Prefer the longest matching suffix to handle repeated n-grams.
                for size in range(min(5, len(words), len(self.committed)), 0, -1):
                    if ([w.text for w in words[:size]] ==
                            [w.text for w in self.committed[-size:]]):
                        words = words[size:]
                        break
        count = 0
        for old, new in zip(self.pending, words):
            if old.text != new.text:
                break
            count += 1
        confirmed = words[:count]
        self.pending = words[count:]
        self.committed.extend(confirmed)
        if confirmed:
            self.last_end = max(self.last_end or 0, *(w.end for w in confirmed))
        return confirmed

    def finish(self):
        """Release the latest tail once at EOF; this is NOT LA confirmation."""
        if self.finished:
            return []
        self.finished = True
        tail, self.pending = self.pending, []
        return tail


def confirmed_segment_cut(segment_ends, confirmed_end, buffer_start, source_end,
                          trim_s):
    """Trim at a confirmed Whisper segment, retaining the final segment.

    trim_s is a trigger, not a hard cap: never discard unconfirmed audio just
    to satisfy the threshold. This matches upstream's segment trimming option.
    """
    if confirmed_end is None or source_end - buffer_start <= trim_s:
        return None
    candidates = [end for end in segment_ends[:-1]
                  if buffer_start < end <= confirmed_end and end < source_end]
    return max(candidates, default=None)


def replay(wav, min_chunk_s, infer, finish, emit, *, max_frames=None,
           clock=time.perf_counter, sleep=time.sleep):
    """Read only arrived samples; coalesce audio accumulated during inference.

    A single sequential worker includes ASR, agreement, and MT. The next call
    waits until min_chunk_s of new audio has arrived, or EOF. A slow call makes
    the next input larger, rather than discarding audio or resetting the clock.
    infer receives newly arrived PCM and its absolute endpoint; it owns the
    retained audio buffer. finish flushes once without another ASR decode.
    """
    if not math.isfinite(min_chunk_s) or not 0 < min_chunk_s <= 30:
        raise ValueError("min_chunk_s must be finite and in (0, 30]")
    rate = wav.getframerate()
    step = max(1, round(rate * min_chunk_s))
    total = wav.getnframes() if max_frames is None else min(max_frames, wav.getnframes())
    if total <= 0:
        raise ValueError("Replay requires nonempty audio")
    origin = clock()
    pos = 0
    index = 0
    while pos < total:
        minimum_end = min(pos + step, total)
        ready_s = minimum_end / rate
        while (wait := ready_s - (clock() - origin)) > 0:
            sleep(wait)
        started = clock()
        available = math.floor((started - origin) * rate)
        end = min(total, max(minimum_end, available))
        raw = wav.readframes(end - pos)
        if len(raw) != (end - pos) * wav.getsampwidth() * wav.getnchannels():
            raise ValueError("WAV ended before its declared frame count")
        result = infer(raw, end / rate)
        completed = clock()
        emit(dict(result, update_index=index, event="update",
                  source_start_s=pos / rate, source_end_s=end / rate,
                  new_audio_s=(end - pos) / rate,
                  update_ready_s=ready_s,
                  processing_start_s=started - origin,
                  emitted_s=completed - origin, process_s=completed - started,
                  update_wait_s=max(0, started - origin - ready_s),
                  lag_after_source_end_s=completed - origin - end / rate))
        pos = end
        index += 1
    started = clock()
    result = finish()
    completed = clock()
    emit(dict(result, update_index=index, event="eof_flush",
              source_start_s=total / rate, source_end_s=total / rate,
              new_audio_s=0, update_ready_s=total / rate,
              processing_start_s=started - origin,
              emitted_s=completed - origin, process_s=completed - started,
              update_wait_s=0, lag_after_source_end_s=completed - origin - total / rate))
    return clock() - origin
