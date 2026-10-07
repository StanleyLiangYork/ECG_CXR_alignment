"""Verify lab semantics without protected data or large model weights."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def decoder(tmp_path, monkeypatch):
    for name in ('EXPERIMENT4_ROOT', 'ECG_CXR_BRIDGE_ARTIFACT_ROOT', 'SYMILE_ROOT', 'HF_HOME'):
        monkeypatch.setenv(name, str(tmp_path / name))
    import e4_evidence
    return e4_evidence


def test_all_50_column_positions(decoder):
    vector = np.concatenate([np.arange(50) / 50, np.ones(50)])
    result = decoder.lab_summary(vector)
    assert len(result['observed_labs']) == 50
    for i, entry in enumerate(result['observed_labs']):
        assert entry['itemid'] == decoder.LAB_IDS[i]
        assert entry['training_ecdf_percentile'] == i / 50
        assert entry['raw_value'] is None and entry['unit'] is None
        assert entry['clinical_abnormality'] == 'not_determinable_from_percentile_alone'
    assert result['salient_count_before_limit'] >= result['salient_count']


def test_missing_is_not_normal_or_measured(decoder):
    result = decoder.lab_summary(np.concatenate([np.ones(50), np.zeros(50)]))
    assert result['observations'] == result['observed_labs'] == []
    assert len(result['missing_itemids']) == 50
    assert 'No laboratory tests were observed' in result['text']


@pytest.mark.parametrize('value', [np.nan, np.inf, -.1, 1.1])
def test_invalid_observed_value_fails(decoder, value):
    vector = np.concatenate([np.full(50,.5), np.ones(50)])
    vector[0] = value
    with pytest.raises(ValueError): decoder.lab_summary(vector)


@pytest.mark.parametrize('value', [np.nan, 2, -1])
def test_invalid_mask_fails(decoder, value):
    vector = np.concatenate([np.full(50,.5), np.ones(50)])
    vector[50] = value
    with pytest.raises(ValueError): decoder.lab_summary(vector)


def test_custom_threshold_text_and_opaque_labels(decoder):
    vector = np.concatenate([np.full(50,.5), np.ones(50)])
    assert '20th' in decoder.lab_summary(vector, lower=.2, upper=.8)['text']
    vector[decoder.LAB_IDS.index('50934')] = .99
    result = decoder.lab_summary(vector)
    assert 'H [unexpanded source label]' in result['text']
    assert result['observations'][0]['source_name'] == 'H'


def test_float32_threshold_boundaries(decoder):
    vector = np.concatenate([np.full(50,.5), np.ones(50)]).astype(np.float32)
    vector[0], vector[1] = .1, .9
    result = decoder.lab_summary(vector)
    assert {r['direction'] for r in result['observations']} == {'low_distributional_percentile', 'high_distributional_percentile'}


def test_table_provenance_written(decoder, tmp_path):
    report = decoder.load_lab_code_table(tmp_path)
    assert report['upstream_mapping']['source_commit'] == 'd12e2c8b0a528cd536ccc5175f910f38f07bb6cb'
    assert report['source_names']['50878'] == 'Asparate Aminotransferase (AST)'
    assert report['names']['50878'] == 'Aspartate Aminotransferase (AST)'
    assert len(report['upstream_mapping']['columns']) == 50


def test_cloud_array_audit_rejects_inverted_mask(decoder, tmp_path):
    import pandas as pd
    symile = tmp_path / 'symile'; symile.mkdir()
    keys = [item + '_percentile' for item in decoder.LAB_IDS]
    for split in ('train', 'val', 'test'):
        columns = {'hadm_id': [1,2]}
        for item in decoder.LAB_IDS:
            columns[item] = [3., np.nan]
            columns[item + '_percentile'] = [1., np.nan]
        frame = pd.DataFrame(columns)
        frame.to_csv(symile / f'{split}.csv', index=False)
        root = symile / 'data_npy' / split; root.mkdir(parents=True)
        np.save(root / f'hadm_id_{split}.npy', [1,2])
        np.save(root / f'labs_percentiles_{split}.npy', np.ones((2,50)))
        np.save(root / f'labs_missingness_{split}.npy', np.array([[1]*50,[0]*50]))
    (symile / 'labs_means.json').write_text(json.dumps({key:1. for key in keys}))
    out = tmp_path / 'out'
    assert decoder.audit_lab_encoding(symile, out)['status'] == 'complete'
    np.save(symile / 'data_npy' / 'val' / 'labs_missingness_val.npy', np.array([[0]*50,[1]*50]))
    with pytest.raises(ValueError, match='observation mask'):
        decoder.audit_lab_encoding(symile, out)
    assert json.loads((out / 'evidence' / 'lab_encoding_audit.json').read_text())['status'] == 'failed'


def test_matches_official_mapping_and_encoder(decoder, tmp_path):
    """Optional upstream-source conformance test, run during the repository audit."""
    source = os.environ.get('SYMILE_SOURCE_REPO')
    if not source: pytest.skip('Set SYMILE_SOURCE_REPO to a checkout of the pinned official repository')
    root = Path(source) / 'experiments' / 'data_processing' / 'symile_mimic'
    constants = root / 'constants.py'
    assert hashlib.sha256(constants.read_bytes()).hexdigest() == decoder.LAB_TABLE['source_sha256']
    tree = ast.parse(constants.read_text())
    labs = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'LABS' for t in n.targets))
    assert labs == decoder.LAB_SOURCE_NAMES
    import pandas as pd
    import torch
    tree = ast.parse((root / 'process_and_save_tensors.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_labs')
    namespace = dict(json=json, pd=pd, torch=torch)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'official_get_labs', 'exec'), namespace)
    means = {f'{item}_percentile': .73 for item in reversed(decoder.LAB_IDS)}
    (tmp_path / 'means.json').write_text(json.dumps(means))
    row = {}
    for i, item in enumerate(decoder.LAB_IDS):
        row[item] = np.nan if i % 3 == 0 else 100+i
        row[item + '_percentile'] = np.nan if i % 3 == 0 else i/50
    percentiles, mask = namespace['get_labs'](SimpleNamespace(data_dir=tmp_path, labs_means='means.json'), row)
    decoded = decoder.lab_summary(np.concatenate([percentiles.numpy(), mask.numpy()]))
    assert decoded['observed_count'] == sum(i % 3 != 0 for i in range(50))
    for entry in decoded['observed_labs']:
        i = decoder.LAB_IDS.index(entry['itemid'])
        assert i % 3 != 0
        assert entry['training_ecdf_percentile'] == pytest.approx(i/50)
    from e4_bridge import _observed_labs
    prepared = _observed_labs(percentiles.numpy()[None,:], mask.numpy()[None,:])[0]
    assert decoder.lab_summary(prepared) == decoded
