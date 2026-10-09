"""CPU-only tests of prompt separation and non-destructive run setup."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from e4_pulse_prompt_v2 import build_pulse_prompt
from e4_review_pulse_v2 import prepare

ROOT = Path(__file__).parent


class PromptTests(unittest.TestCase):
    def setUp(self):
        # Execute only the existing text helpers, without importing the GPU stack.
        import re
        module = types.ModuleType('e4_pulse_generation')
        module.__dict__['re'] = re
        tree = ast.parse((ROOT/'e4_pulse_generation.py').read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in ('_excerpt', '_lab_excerpt')]
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<text_helpers>', 'exec'), module.__dict__)
        self.mock = patch.dict(sys.modules, {'e4_pulse_generation': module})
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_zero_shot(self):
        text = build_pulse_prompt({'condition': 'zero_shot', 'evidence': []})
        for term in ('admission', 'other patient', 'percentile', 'laboratory', 'examples', 'json'):
            self.assertNotIn(term, text.lower())
        self.assertIn('plain text', text)

    def test_same_modality(self):
        text = build_pulse_prompt({'condition': 'trimodal_same', 'evidence': [
            {'anchor_id': 'a', 'ecg': {'machine_diagnosis': 'Sinus rhythm.'}}]})
        self.assertIn('OTHER patients', text)
        self.assertNotIn('percentiles', text)
        self.assertIn('Sinus rhythm.', text)

    def test_full(self):
        text = build_pulse_prompt({'condition': 'trimodal_full', 'evidence': [
            {'anchor_id': 'a', 'ecg': {'machine_diagnosis': 'Sinus rhythm.'},
             'labs': {'text': 'Test p90'}, 'cxr': {'report': 'Impression: Clear lungs.', 'labels': []}}]})
        self.assertIn('Laboratory percentiles', text)
        self.assertIn('Clear lungs.', text)

    def test_zero_shot_rejects_evidence(self):
        with self.assertRaises(ValueError):
            build_pulse_prompt({'condition': 'zero_shot', 'evidence': [{'anchor_id': 'a'}]})

    def test_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'review';source.mkdir()
            prior = source/'generations/pulse7b_ecg';prior.mkdir(parents=True)
            proto = {'settings': {'model_key': 'pulse7b_ecg'}, 'resolved_revision': 'a'*40}
            (prior/'protocol.json').write_text(json.dumps(proto))
            (prior/'reports.jsonl').write_text('{"status":"error"}\n')
            for name in ('protocol.json','contexts.jsonl','image_audit.jsonl'):
                (source/name).write_text('{}\n')
            qwen=source/'generations/qwen35_4b';qwen.mkdir()
            (qwen/'reports.jsonl').write_text('original qwen\n')
            dest=prepare(source,ROOT,activate=False)
            self.assertEqual((prior/'reports.jsonl').read_text(),'{"status":"error"}\n')
            self.assertFalse((dest/'generations/pulse7b_ecg').exists())
            self.assertTrue((dest/'generations/qwen35_4b').is_symlink())
            self.assertEqual((dest/'generations/qwen35_4b/reports.jsonl').read_text(),'original qwen\n')
            self.assertIn('from e4_pulse_prompt_v2 import', (dest/'code_snapshot/e4_pulse_generation.py').read_text())
            self.assertIn(repr('a'*40), (dest/'code_snapshot/e4_review_controls.py').read_text())
            self.assertEqual(prepare(source,ROOT,activate=False),dest)
            (source/'contexts.jsonl').write_text('changed')
            with self.assertRaises(ValueError):prepare(source,ROOT,activate=False)

    def test_notebooks_compile(self):
        for number in ('20','21'):
            path=next(ROOT.glob(number+'_*.ipynb'))
            for cell in json.loads(path.read_text())['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),str(path),'exec')


if __name__ == '__main__':
    unittest.main()
