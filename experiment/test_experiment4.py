"""Small synthetic smoke test; never uses protected clinical data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows: stream.write(json.dumps(row) + "\n")


def build_fixture(artifact, symile):
    rng = np.random.default_rng(7); dims = 8
    split_subjects = {"train": list(range(10000100, 10000108)), "val": list(range(10000200, 10000204)),
                      "test": list(range(10000300, 10000303))}
    all_ecg, all_cxr = [], []
    for split, subjects in split_subjects.items():
        rows = []
        for i, subject in enumerate(subjects):
            e, c, h = 40000000 + subject, 50000000 + subject, 20000000 + subject
            rows.append({"npy_index": i, "subject_id": subject, "hadm_id": h,
                         "ecg_study_id": e, "cxr_study_id": c,
                         "ecg_path": f"files/p{subject}/s{e}/{e}", "cxr_path": f"files/p{subject}/s{c}/x.jpg"})
            all_ecg.append({"sample_id": f"ecg_{e}", "subject_id": subject, "study_id": e,
                            "waveform_path": f"/tmp/{e}", "machine_diagnosis": "Sinus rhythm; normal ECG"})
            all_cxr.append({"sample_id": f"cxr_{c}", "subject_id": subject, "study_id": c,
                            "image_path": f"/tmp/{c}.jpg", "json_path": f"/tmp/{c}.json",
                            "report_text": "No acute cardiopulmonary abnormality.", "labels": ["No Finding"]})
        jsonl(artifact / "manifests" / f"{split}.jsonl", rows)
        n = len(rows); pdir = symile / "data_npy" / split; pdir.mkdir(parents=True, exist_ok=True)
        np.save(pdir / f"labs_percentiles_{split}.npy", rng.random((n, 50), dtype=np.float32))
        np.save(pdir / f"labs_missingness_{split}.npy", np.ones((n, 50), dtype=np.float32))
        np.save(pdir / f"hadm_id_{split}.npy", np.array([r['hadm_id'] for r in rows]))
        from e4_evidence import LAB_IDS
        frame = pd.DataFrame(rows)
        values = np.load(pdir / f"labs_percentiles_{split}.npy")
        for j, item in enumerate(LAB_IDS): frame[item + '_percentile'] = values[:, j]
        frame.to_csv(symile / f'{split}.csv', index=False)
        fdir = artifact / "features" / split; fdir.mkdir(parents=True, exist_ok=True)
        base = rng.normal(size=(n, dims)).astype("float32")
        np.save(fdir / "ecg_hubert.npy", base)
        np.save(fdir / "cxr_raddino.npy", base + rng.normal(scale=.01, size=base.shape).astype("float32"))
    # Add many non-Symile subjects so every global external split is populated.
    for subject in range(10001000, 10001120):
        e, c = 41000000 + subject, 51000000 + subject
        all_ecg.append({"sample_id": f"ecg_{e}", "subject_id": subject, "study_id": e,
                        "waveform_path": f"/tmp/{e}", "machine_diagnosis": "Atrial fibrillation"})
        all_cxr.append({"sample_id": f"cxr_{c}", "subject_id": subject, "study_id": c,
                        "image_path": f"/tmp/{c}.jpg", "json_path": f"/tmp/{c}.json",
                        "report_text": "Small pleural effusion.", "labels": ["Pleural Effusion"]})
    for row in all_ecg:
        row['waveform_path'] = f"/native/p{row['subject_id']}/s{row['study_id']}/record"
    for row in all_cxr:
        row['image_path'] = f"/native/p{row['subject_id']}/s{row['study_id']}/image.jpg"
        path = artifact / 'fixture_json' / f"{row['sample_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'report': row['report_text'], 'labels': row['labels']}))
        row['json_path'] = str(path)
    jsonl(artifact / "external" / "ecg" / "manifest.jsonl", all_ecg)
    jsonl(artifact / "external" / "cxr" / "manifest.jsonl", all_cxr)
    for modality, rows, name in [("ecg", all_ecg, "ecg_foundation.npy"),
                                 ("cxr", all_cxr, "cxr_foundation.npy")]:
        root = artifact / "external" / modality; root.mkdir(parents=True, exist_ok=True)
        np.save(root / name, rng.normal(size=(len(rows), dims)).astype("float32"))
        np.save(root / "encoding_status.npy", np.ones(len(rows), dtype="int8"))
    (artifact / "config.json").write_text("{}")


def test_pipeline(tmp_path, monkeypatch):
    # This historical fixture writes legacy-format generation records.
    monkeypatch.setenv('E4_GENERATION_RUN', 'legacy')
    # Environment variables must point e4_common at this fixture before module import.
    import os
    artifact, symile, out = tmp_path / "artifact", tmp_path / "symile", tmp_path / "artifact" / "experiment4"
    os.environ.update(ECG_CXR_BRIDGE_ARTIFACT_ROOT=str(artifact), SYMILE_ROOT=str(symile),
                      EXPERIMENT4_ROOT=str(out), HF_HOME=str(tmp_path / "hf"))
    build_fixture(artifact, symile)
    from e4_partition import build_global_partition
    from e4_evidence import prepare_external_features, build_banks, build_contexts
    from e4_bridge import prepare_symile, train, evaluate_runs, calibrate_confidence_gates
    assert build_global_partition()["patient_disjoint_verified"]
    prepare_external_features(); prepare_symile()
    for seed in [43]:
        train(method="clip", pairing="genuine", labs_mode="values", seed=seed,
              epochs=1, batch_size=4, patience=1, device="cpu")
        train(method="clip", pairing="shuffled", labs_mode="values", seed=seed,
              epochs=1, batch_size=4, patience=1, device="cpu")
        train(method="clip", pairing="genuine", labs_mode="mask_only", seed=seed,
              epochs=1, batch_size=4, patience=1, device="cpu")
    for method in ['codebind', 'combined']:
        for pairing, labs_mode, no_code_match in [('genuine','values',False), ('shuffled','values',False),
                                                 ('genuine','mask_only',False), ('genuine','values',True)]:
            train(method=method, pairing=pairing, labs_mode=labs_mode, no_code_match=no_code_match,
                  epochs=1, batch_size=4, patience=1, device='cpu')
    frame, selected = evaluate_runs()
    build_banks(device='cpu'); gates = calibrate_confidence_gates()
    assert len(frame) and "clip" in selected
    assert build_contexts(k=2, max_queries=2)['rows'] > 0
    from e4_common import read_jsonl
    from e4_evaluation import evaluate
    contexts = read_jsonl(out / 'contexts' / 'bidirectional_contexts.jsonl')
    generated = []
    for i, row in enumerate(contexts):
        generated.append({**{k:row[k] for k in ['direction','sample_id','method','condition','context_signature']},
            'generation_key': str(i), 'model_key':'qwen35_4b', 'status':'ok',
            'input_scope':'raw_sensor_plus_retrieval', 'processor_input_tokens':100,
            'generated_report':'Atrial fibrillation.' if row['direction']=='ecg' else 'Small pleural effusion.',
            'evidence_trace':[], 'generation_protocol_signature':'synthetic-v1'})
    jsonl(out / 'generations' / 'qwen35_4b.jsonl', generated)
    summary = evaluate(bootstraps=10)
    assert not summary.empty
    complete = pd.read_csv(out / 'evaluation' / 'generation_completeness.csv').set_index('model_key')
    assert complete.loc['gpt_oss_20b', 'successful'] == 0
    assert complete.loc['qwen35_4b', 'missing'] == 0
    paired = pd.read_csv(out / 'evaluation' / 'paired_bootstrap.csv')
    assert set(paired.resampling_unit) == {'patient'}
    from e4_evaluation import extractor_self_test, cxr_reference, build_review_packets
    from e4_generation import _extract_clinical_output, _extract_json, _trim_prompt
    from e4_evidence import _rank
    assert extractor_self_test()['tests'] >= 6
    assert cxr_reference({'labels': {'Edema': -1, 'Atelectasis': 1, 'Pneumonia': 0}}) == {'Atelectasis'}
    assert _extract_json('{"report":"x"}') is None
    valid = '{"report":"x","evidence_trace":[],"uncertainties":[]}'
    assert _extract_json('<|channel|>analysis<|message|>{"report":"wrong"}<|channel|>final<|message|>' + valid)['report'] == 'x'
    assert _extract_json('<|channel|>analysis<|message|>' + valid) is None
    parsed, stage, compliant, visible = _extract_clinical_output(
        'Sinus rhythm. No acute ST-segment change.', set())
    assert parsed['report'].startswith('Sinus rhythm') and stage == 'visible_free_text'
    assert not compliant and visible == parsed['report']
    hidden = _extract_clinical_output('<|channel|>analysis<|message|>private reasoning', set())
    assert hidden[0] is None and hidden[1] == 'empty_or_hidden'
    with pytest.raises(ValueError): _trim_prompt('abc', lambda *a, **k: {'input_ids': [1,2,3]}, 2)
    from e4_format_rescue import parse_rescue_result
    normalized, stage = parse_rescue_result(
        '```json\n{"final_report":"Normal ECG.","uncertainty":"None","evidence":[]}\n```', set())
    assert stage == 'normalized' and normalized['report'] == 'Normal ECG.'
    invalid_source, _ = parse_rescue_result(
        '{"report":"x","uncertainties":[],"evidence_trace":'
        '[{"claim":"y","source_ids":["not-allowed"]}]}', {'allowed'})
    assert invalid_source is None
    bank = np.array([[1.,0.],[.9,0.],[.8,0.]])
    top, _, _ = _rank(np.array([1.,0.]), bank, 1, '1', [{'subject_id':i} for i in [1,2,3]])
    assert top == [1]
    root = out / 'clinical_review'; root.mkdir()
    (root / 'reviewer_1.csv').write_text('preserve ratings')
    with pytest.raises(FileExistsError): build_review_packets()
    assert (root / 'reviewer_1.csv').read_text() == 'preserve ratings'
    from e4_partition import _canonicalize, _dedupe
    verified, rejected = _canonicalize([{'sample_id':'x', 'subject_id':123, 'study_id':456,
                                        'image_path':'/unknown/image.jpg'}], 'cxr', {})
    assert not verified and len(rejected) == 1
    # Flat multi_kg pairs are valid when native IDs in JSON agree with the
    # manifest and the image/JSON filename pair identifies that study.
    flat = tmp_path / '50000456.json'
    flat.write_text(json.dumps({'subject_id': 123, 'study_id': 50000456,
                                'Caption': 'No acute cardiopulmonary abnormality.'}))
    verified, rejected = _canonicalize([{
        'sample_id': 'cxr_50000456', 'subject_id': 123, 'study_id': 50000456,
        'image_path': str(flat.with_suffix('.jpg')), 'json_path': str(flat)}], 'cxr', {})
    assert len(verified) == 1 and not rejected
    assert verified[0]['identity_provenance'].startswith('native MIMIC')
    with pytest.raises(ValueError):
        _dedupe([{'study_id':1,'subject_id':2}, {'study_id':1,'subject_id':3}], ['study_id'])
    from e4_bridge import _observed_labs
    with pytest.raises(ValueError): _observed_labs(np.zeros((2,50)), np.full((2,50), 2))
    from e4_evaluation import summarize_clinical_review
    fields = ['sensor_quality_adequate','reference_correct','report_correctness_1_to_5',
              'unsupported_or_harmful_claim','important_omission','evidence_trace_supported']
    ratings = pd.DataFrame([{'review_id':'r1', **{f: ('5' if f.endswith('1_to_5') else 'Y') for f in fields}}])
    for name in ['reviewer_1.csv','reviewer_2.csv']: ratings.to_csv(root / name, index=False)
    pd.DataFrame([{'review_id':'r1', 'method':'clip'}]).to_csv(root / 'private_key.csv', index=False)
    assert summarize_clinical_review()['status'] == 'complete'
    assert len(pd.read_csv(root / 'adjudicated_results_private.csv')) == 1


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test_pipeline(Path(directory))
    print("Experiment 4 synthetic smoke test passed")
