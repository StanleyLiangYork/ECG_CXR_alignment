"""Synthetic-only tests. No protected data, network, tokens or model weights."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import types
from pathlib import Path
import numpy as np
import e4_review_controls as rc


class ControlsTest(unittest.TestCase):
    def fixture(self, root):
        source=root/'source';dest=root/'new';rng=np.random.default_rng(3)
        train=[dict(sample_id=f'a{i}',subject_id=100+i,hadm_id=200+i,ecg_study_id=300+i,
                    cxr_study_id=400+i,ecg_machine_diagnosis='Sinus bradycardia.',
                    cxr_report=f'Report {i}',cxr_labels=[],lab_summary={'text':f'Lab {i}'}) for i in range(8)]
        external=[dict(sample_id=f'q{i}',subject_id=500+i,waveform_path=f'/wave/q{i}',
                       machine_diagnosis='Sinus bradycardia.') for i in range(4)]
        rc.write_json(source/'partition/global_partition_audit.json',{'status':'complete','patient_disjoint_verified':True})
        rc.write_rows(source/'bridge/prepared/train/manifest.jsonl',train)
        np.save(source/'bridge/prepared/train/ecg.npy',rng.normal(size=(8,4)))
        rc.write_rows(source/'external/ecg/test/manifest.jsonl',external)
        np.save(source/'external/ecg/test/foundation.npy',rng.normal(size=(4,4)))
        br=source/'evidence/banks/clip/genuine/symile';rc.write_rows(br/'manifest.jsonl',train)
        for m in ['ecg','cxr','labs']:np.save(br/f'{m}.npy',rc.normalized(rng.normal(size=(8,4))))
        qr=source/'evidence/queries/clip/genuine/ecg/test';rc.write_rows(qr/'manifest.jsonl',external)
        np.save(qr/'ecg.npy',rc.normalized(rng.normal(size=(4,4))))
        rc.write_rows(source/'contexts/bidirectional_contexts.jsonl',[
            dict(direction='ecg',method='clip',condition='full_trimodal',sample_id=q['sample_id'],subject_id=q['subject_id'])
            for q in external])
        return source,dest

    def test_prepare_frozen_and_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            source,dest=self.fixture(Path(td))
            p=rc.prepare_controls(source,dest,max_queries=4,render_subset=2)
            self.assertEqual(p['rows'],28)
            rows=rc.verify_contexts(dest)
            self.assertEqual(len(rc.generation_tasks(dest,'qwen35_4b')),28)
            self.assertEqual(len(rc.generation_tasks(dest,'pulse7b_ecg')),12)
            by={(r['sample_id'],r['condition']):r for r in rows}
            for sid in {r['sample_id'] for r in rows}:
                f=by[sid,'trimodal_full'];b=by[sid,'trimodal_broken']
                self.assertEqual([x['ecg'] for x in f['evidence']],[x['ecg'] for x in b['evidence']])
                self.assertTrue(all(str(x['subject_id'])!=str(x['component_donor']['subject_id']) for x in b['evidence']))
            self.assertEqual(rc.prepare_controls(source,dest,max_queries=4,render_subset=2),p)
            with self.assertRaises(ValueError):rc.prepare_controls(source,dest,max_queries=3,render_subset=2)
            ext=rc.read_rows(source/'external/ecg/test/manifest.jsonl');ext[0]['subject_id']=100
            rc.write_rows(source/'external/ecg/test/manifest.jsonl',ext)
            with self.assertRaises(ValueError):rc.prepare_controls(source,Path(td)/'leak',max_queries=4,render_subset=2)

    def test_permutation_preserves_multiset(self):
        a=[dict(anchor_id=str(i),subject_id=i,hadm_id=i,ecg={'machine_diagnosis':str(i)},
                cxr={'report':'x'*i},labs={'text':str(i)}) for i in range(8)]
        b,_=rc.break_linkage(a)
        self.assertEqual(sorted(rc.sig([r['cxr'],r['labs']]) for r in a),
                         sorted(rc.sig([r['cxr'],r['labs']]) for r in b))
        same=copy.deepcopy(a)
        for r in same:r['subject_id']=1
        with self.assertRaises(ValueError):rc.break_linkage(same)

    def test_rank_tie_and_patient_exclusion(self):
        top,_=rc.rank(np.array([1.,0.]),np.array([[1.,0.]]*3),[{'subject_id':1},{'subject_id':2},{'subject_id':3}],1)
        self.assertEqual(top,[1,2])

    def test_torn_append_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'reports.jsonl';p.write_bytes(b'{"a":1}\n{"a":')
            self.assertEqual(rc.read_log(p,repair=True),[{'a':1}])
            self.assertTrue(list(Path(td).glob('*.torn-*')))
            self.assertEqual(p.read_bytes(),b'{"a":1}\n')
            p.write_bytes(b'broken\n{"a":1}\n')
            with self.assertRaises(ValueError):rc.read_log(p,repair=True)

    def test_paired_bootstrap(self):
        a=[dict(sample_id=str(i),subject_id=i//2,y_true=[1,0],y_pred=[0,0]) for i in range(6)]
        b=[dict(r,y_pred=[1,0]) for r in a]
        out=rc.paired_bootstrap(a,b,20)
        self.assertEqual(out[0]['delta_right_minus_left'],1)
        self.assertEqual(out[0]['patients'],3)
        self.assertEqual(out[0]['ci_low'],1)

    def test_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            fields={'fs':100,'units':['uV']*12,'sig_name':['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']}
            x=np.sin(np.arange(1000)[:,None]/10)*np.ones((1,12))*1000
            out=rc.calibrated_ecg(x,fields,Path(td)/'image.png',dpi=60)
            self.assertAlmostEqual(out['pixels_per_mv'],10*60/25.4)
            self.assertEqual(out['duration_seconds'],10)
            self.assertTrue((Path(td)/'image.png').is_file())
            from PIL import Image
            with Image.open(Path(td)/'image.png') as image:
                self.assertLessEqual(abs(image.width-282*60/25.4),1)
            fields['units']=['unknown']*12
            with self.assertRaises(ValueError):rc.calibrated_ecg(x,fields,Path(td)/'bad.png')

    def test_pilot_is_balanced(self):
        with tempfile.TemporaryDirectory() as td:
            source,dest=self.fixture(Path(td));rc.prepare_controls(source,dest,max_queries=4,render_subset=2)
            qwen=rc.balanced_tasks(rc.generation_tasks(dest,'qwen35_4b'))
            self.assertEqual(len({r['condition'] for r in qwen[:7]}),7)
            pulse=rc.balanced_tasks(rc.generation_tasks(dest,'pulse7b_ecg'))
            self.assertEqual(len({(r['condition'],r['rendering']) for r in pulse[:6]}),6)

    def test_notebook_code_compiles(self):
        for p in Path(__file__).parent.glob('*.ipynb'):
            if p.name[:2] not in {'17','18','19','20','21'}:continue
            nb=json.loads(p.read_text())
            for i,c in enumerate(nb['cells']):
                if c['cell_type']=='code':compile(''.join(c['source']),f'{p.name}:{i}','exec')

    def test_evaluation_partial_and_protocol(self):
        # Mock only clinical extraction to test evaluation mechanics without GPU
        # imports. Production evaluation imports the unchanged versioned extractor.
        fake=types.ModuleType('e4_evaluation')
        fake.ECG_LABELS=['bradycardia'];fake.ECG_RULES={}
        fake.present=lambda text,rules: {'bradycardia'} if 'bradycardia' in text else set()
        with tempfile.TemporaryDirectory() as td,patch.dict('sys.modules',{'e4_evaluation':fake}):
            source,dest=self.fixture(Path(td));rc.prepare_controls(source,dest,max_queries=4,render_subset=2)
            status=rc.evaluate(dest,bootstraps=10,allow_partial=True)
            self.assertEqual(status['status'],'partial_do_not_use_as_final')
            self.assertTrue((dest/'evaluation/summary.csv').is_file())
            with self.assertRaises(RuntimeError):rc.evaluate(dest,bootstraps=10)
            for model in ['qwen35_4b','pulse7b_ecg']:
                tasks=rc.generation_tasks(dest,model);out=dest/'generations'/model
                protocol={'settings':{'contexts':rc.sig(tasks)}};rc.write_json(out/'protocol.json',protocol)
                rows=[dict(r,task_key=rc.task_key(r),model_key=model,protocol_signature=rc.sig(protocol),
                           status='ok',generated_report='Sinus bradycardia.') for r in tasks]
                rc.write_rows(out/'reports.jsonl',rows)
            self.assertEqual(rc.evaluate(dest,bootstraps=10)['status'],'complete')


if __name__=='__main__':unittest.main()
