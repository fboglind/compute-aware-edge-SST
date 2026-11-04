"""Small binding to the pinned, published Whistle C API.

No auto-downloads, Python Needle package, streaming agreement, or cloud calls.
The native model lives in the replay process, so RSS includes ASR and MT.
"""
import ctypes
import hashlib
import json
import mmap
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_pins():
    return json.loads((ROOT / 'whistle-runtime.json').read_text())


def architecture():
    machine = platform.machine()
    machine = {'arm64': 'aarch64', 'AMD64': 'x86_64'}.get(machine, machine)
    if platform.system() != 'Linux' or machine not in ('aarch64', 'x86_64'):
        raise ValueError('The pinned Whistle assets target Linux aarch64 and x86_64')
    return machine


def verified_assets(folder, *, all_platforms=False):
    folder = Path(folder)
    pins = expected_pins()
    manifest = folder / 'asset-manifest.json'
    if not manifest.is_file() or json.loads(manifest.read_text()) != pins:
        raise ValueError(f'Missing or different pinned assets: {folder}. Run prepare_whistle.py first.')
    files = {'whistle.cact': pins['weights']['sha256']}
    arches = pins['runtime']['platforms'] if all_platforms else [architecture()]
    for arch in arches:
        files[f'{arch}/libneedle3.so'] = pins['runtime']['platforms'][arch]['library_sha256']
    for name, expected in files.items():
        if sha(folder / name) != expected:
            raise ValueError(f'Whistle asset checksum differs: {folder / name}')
    return pins


def parse_response(payload, tokens):
    """Validate the documented response without fabricating empty successes."""
    if tokens < 0:
        raise ValueError('Native transcription returned an error')
    result = json.loads(payload)
    if not isinstance(result, dict) or not isinstance(result.get('text'), str):
        raise ValueError('Whistle response has no text string')
    if result.get('error'):
        raise ValueError(f'Whistle response reports an error: {result["error"]}')
    if result['text'].strip() and result.get('language') != 'en':
        raise ValueError('Whistle did not report English despite forced language=en')
    result['returned_tokens'] = int(tokens)
    return result


class Whistle:
    """One process-global, non-thread-safe native speech model per replay job."""
    def __init__(self, folder):
        folder = Path(folder)
        self.pins = verified_assets(folder)
        self.library = ctypes.CDLL(str((folder / architecture() / 'libneedle3.so').resolve()))
        lib = self.library
        lib.needle_load.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulonglong]
        lib.needle_load.restype = ctypes.c_int
        lib.needle_models.argtypes = []
        lib.needle_models.restype = ctypes.c_int
        lib.needle_last_error.argtypes = []
        lib.needle_last_error.restype = ctypes.c_char_p
        lib.needle_transcribe.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_char), ctypes.c_int]
        lib.needle_transcribe.restype = ctypes.c_int
        # Keep the mapping alive for the entire job; the C API may retain pointers.
        # ACCESS_COPY supplies a writable buffer view without eagerly copying it.
        with (folder / 'whistle.cact').open('rb') as stream:
            self.mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
        self.weights = (ctypes.c_ubyte * len(self.mapping)).from_buffer(self.mapping)
        code = lib.needle_load(self.weights, len(self.mapping))
        if code < 0:
            self._raise('loading')
        if not lib.needle_models() & 2:
            raise ValueError('Native library did not load a speech model')
        self.output = ctypes.create_string_buffer(1024 * 1024)

    def _raise(self, stage):
        error = self.library.needle_last_error()
        detail = error.decode('utf-8', errors='replace') if error else 'no native error message'
        raise ValueError(f'Whistle {stage} failed: {detail}')

    def transcribe(self, audio):
        import numpy as np
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not 0 < audio.size <= 30 * 16000:
            raise ValueError('Expected one nonempty mono chunk of at most 30 seconds at 16 kHz')
        if not np.isfinite(audio).all() or (np.abs(audio) > 1).any():
            raise ValueError('Expected finite float PCM in [-1, 1]')
        self.output[0] = b'\0'
        tokens = self.library.needle_transcribe(
            audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(audio.size),
            b'en', None, 0, self.output, len(self.output))
        if tokens < 0:
            self._raise('transcription')
        return parse_response(self.output.value.decode('utf-8'), tokens)
