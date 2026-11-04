"""Fixed chunks on one continuous clock. Standard library only."""
import math
import time


def replay(wav, chunk_s, infer, emit, *, clock=time.perf_counter, sleep=time.sleep):
    """Read a chunk only after its last sample arrives; never skip overdue chunks.

    infer accepts PCM bytes and returns a dict containing english/german strings.
    emit is called once per completed chunk, including chunks with no output.
    The consumer receives an entire chunk translation atomically.
    """
    if not math.isfinite(chunk_s) or not 0 < chunk_s <= 30:
        raise ValueError("chunk_s must be finite and in (0, 30]")
    rate = wav.getframerate()
    chunk_frames = max(1, round(chunk_s * rate))
    total_frames = wav.getnframes()
    origin = clock()
    pos = 0
    index = 0
    while pos < total_frames:
        end = min(pos + chunk_frames, total_frames)
        ready_s = end / rate
        # A single clock is essential: sleep(chunk_s) after each inference is wrong.
        while (wait_s := ready_s - (clock() - origin)) > 0:
            sleep(wait_s)
        started = clock()
        raw = wav.readframes(end - pos)
        expected = (end - pos) * wav.getsampwidth() * wav.getnchannels()
        if len(raw) != expected:
            raise ValueError("WAV ended before its declared frame count")
        result = infer(raw)
        result["words"] = result["german"].split()
        committed = clock()
        record = dict(result)
        record.update(
            chunk_index=index,
            source_start_s=pos / rate,
            source_end_s=ready_s,
            chunk_duration_s=(end - pos) / rate,
            processing_start_s=started - origin,
            emitted_s=committed - origin,
            process_s=committed - started,
            queue_wait_s=max(0.0, started - origin - ready_s),
            lag_after_chunk_end_s=committed - origin - ready_s,
        )
        emit(record)
        pos = end
        index += 1
    return clock() - origin
