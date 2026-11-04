"""Offline contract tests: synthetic clocks/responses only; no model execution."""
import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runner'))
import common
import experiment
from stream_timing import replay
from test_timing import TimingTests, Clock
from timestamp_validation import normalize_empty_timestamps
from whistle_backend import Whistle, parse_response


class ProtocolTests(unittest.TestCase):
    def manifest(self):
        return {'manifest_id':'synthetic-test', 'talks':common.read_json(ROOT/'selection.json')['talks']}

    def test_complete_balanced_grid(self):
        m = self.manifest()
        for device, first in [('pi', 4), ('laptop', 8)]:
            jobs = common.jobs_for(m, device)
            self.assertEqual(len(jobs), 24)
            self.assertEqual(len({j['id'] for j in jobs}), 24)
            self.assertEqual([j['order'] for j in jobs], list(range(1, 25)))
            self.assertEqual(jobs[0]['chunk_s'], first)
            self.assertEqual(jobs[12]['chunk_s'], 12 - first)
            self.assertEqual(jobs[12]['talk'], m['talks'][-1]['id'])
            for rep in (1, 2):
                for seconds in (4, 8):
                    self.assertEqual(len([j for j in jobs if j['repetition']==rep and j['chunk_s']==seconds]), 6)

    def test_changed_plan_is_rejected(self):
        m = self.manifest()
        p = common.plan_for(m, 'pi')
        common.validate_plan(p, m)
        p['settings']['mt_threads'] = 8
        with self.assertRaises(ValueError):
            common.validate_plan(p, m)

    def test_empty_segment_fix_never_invents_nonempty_timestamps(self):
        rows, changes = normalize_empty_timestamps([{'prediction':'', 'emission_cu':None}])
        self.assertEqual(rows[0]['emission_cu'], [])
        self.assertEqual(rows[0]['emission_ca'], [])
        self.assertEqual(len(changes), 1)
        for row in [{'prediction':'Hallo'},
                    {'prediction':'Hallo', 'emission_cu':[], 'emission_ca':[]},
                    {'prediction':'Hallo', 'emission_cu':[float('nan')], 'emission_ca':[1]}]:
            with self.assertRaises(ValueError):
                normalize_empty_timestamps([row])

    def test_native_response_checks_and_silence(self):
        self.assertEqual(parse_response('{"text":"", "language":""}', 0)['text'], '')
        self.assertEqual(parse_response('{"text":"hello", "language":"en"}', 1)['returned_tokens'], 1)
        for payload in ['{}', '{"text":null}', '{"text":"bonjour", "language":"fr"}',
                        '{"text":"", "error":"failed"}']:
            with self.assertRaises(ValueError):
                parse_response(payload, 1)

    def test_native_call_forces_english_and_disables_keywords_and_word_times(self):
        # Exercise the C-call boundary with a Python stand-in; no native library.
        import ctypes
        import numpy as np
        class FakeLibrary:
            def needle_transcribe(inner, pcm, count, language, keywords, times, output, capacity):
                self.assertEqual(count, 160)
                self.assertEqual(language, b'en')
                self.assertIsNone(keywords)
                self.assertEqual(times, 0)
                output.value = b'{"text":"test", "language":"en"}'
                return 1
        backend = Whistle.__new__(Whistle)
        backend.library = FakeLibrary()
        backend.output = ctypes.create_string_buffer(2048)
        self.assertEqual(backend.transcribe(np.zeros(160))['text'], 'test')

    def test_causal_exports_audit_and_resume_tamper_detection(self):
        # A 4.5-second synthetic stream has one full and one short final chunk.
        # Model identities are fixtures, not assertions that inference occurred.
        clock = Clock()
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
            out.writeframes(bytes(72000 * 2))
        buffer.seek(0)
        rows = []
        def infer(pcm):
            clock.sleep(.3)
            return {'english':'hello', 'german':'Hallo', 'asr_s':.2, 'mt_s':.1,
                    'mt_tokens':1, 'mt_limit_reached':False}
        with wave.open(buffer, 'rb') as wav:
            replay(wav, 4, infer, rows.append, clock=clock, sleep=clock.sleep)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/'job'; run = folder/'attempt-001'; run.mkdir(parents=True)
            job = {'id':'fixture-whistle-4s-r1', 'asr':'whistle', 'chunk_s':4}
            talk = {'audio':'fixture.wav', 'audio_sha256':'synthetic', 'duration_s':4.5}
            words = [dict(word_index=i, word='Hallo', chunk_index=i,
                          source_consumed_s=r['source_end_s'], emitted_s=r['emitted_s'])
                     for i,r in enumerate(rows)]
            hypothesis = {'source':'fixture.wav', 'prediction':'Hallo Hallo', 'source_length':4500,
                          'delays':[w['source_consumed_s']*1000 for w in words],
                          'elapsed':[w['emitted_s']*1000 for w in words]}
            metadata = {'audio_sha256':'synthetic', 'args':{'asr':'whistle','chunk_s':4,'threads':4},
                'script_sha256':{n:common.sha(ROOT/'runner'/n) for n in
                                ('run_whistle.py','stream_timing.py','whistle_backend.py','needle.h')},
                'effective_compute_type':{'asr':'cactus-quants-shipped','mt':'int8_float32'},
                'models':{'asr':common.read_json(ROOT/'whistle-runtime.json'),
                          'mt':common.read_json(ROOT/'expected-models.json')['mt-en-de-int8']},
                'decoding':{'mt_beam':4,'asr_beam':'engine_default','asr_threads':'engine_default_not_exposed',
                            'asr_language':'en','vad':False,'condition_on_previous_text':False,
                            'whistle_policy':common.read_json(ROOT/'whistle-runtime.json')['inference_policy']}}
            summary = {'complete':True,'asr_model':'whistle','chunk_s':4,'audio_duration_s':4.5,
                       'chunks':2,'output_words':2,'total_processing_s':.6,'rtf_processing':.6/4.5,
                       'total_asr_s':.4,'total_mt_s':.2,'final_tail_s':.3}
            for name,value in [('metadata',metadata),('summary',summary)]: common.write_json(run/(name+'.json'),value)
            for name,value in [('chunks',rows),('words',words),('hypothesis',[hypothesis])]:
                (run/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in value))
            (run/'translation.de.txt').write_text('Hallo Hallo\n')
            (run/'transcript.en.txt').write_text('hello hello\n')
            common.audit_run(run,talk,job)
            plan = {'plan_id':'synthetic-plan'}
            experiment.finish_job(folder,run,plan,job,'test-session')
            self.assertEqual(common.completed_attempt(folder,plan,job,talk),run)
            (run/'translation.de.txt').write_text('Changed\n')
            with self.assertRaises(ValueError):
                common.completed_attempt(folder,plan,job,talk)


if __name__ == '__main__':
    unittest.main()
