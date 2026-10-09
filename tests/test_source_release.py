"""Validate public working-tree candidates without reading ignored clinical data."""
import ast
import json
from pathlib import Path
import re
import subprocess

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def candidates():
    result = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT,
    ).decode()
    return sorted({ROOT / name for name in result.split('\0') if name
                   and (ROOT / name).is_file()})


def test_no_clinical_or_binary_artifacts():
    forbidden = {'.csv', '.tsv', '.jsonl', '.npy', '.npz', '.parquet', '.hea',
                 '.dat', '.jpg', '.jpeg', '.png', '.dcm', '.xlsx', '.pdf',
                 '.docx', '.pptx', '.zip', '.gz', '.pt', '.pth', '.safetensors'}
    protected = {'analysis', 'results', 'clinical_review', 'query_images',
                 'checkpoints', 'generations', 'manuscript'}
    for path in candidates():
        relative = path.relative_to(ROOT)
        assert path.suffix.lower() not in forbidden, str(relative)
        assert not protected.intersection(relative.parts), str(relative)
        assert not re.search(r'(?:hf_)?token.*\.txt$', path.name, re.I), str(relative)


def test_no_embedded_secrets_or_personal_home_paths():
    patterns = [r'hf_' + r'[A-Za-z0-9]{25,}', r'github_pat_' + r'[A-Za-z0-9_]{30,}',
                r'gh[pousr]_' + r'[A-Za-z0-9]{30,}',
                r'/(?:Users|home|vf/users)/' + r'(?!path\b|user\b)[A-Za-z0-9_.-]+',
                r'/data/' + r'(?:liang' + r'z2|zhaohui' + r'liang)',
                r'/Volumes/' + r'Extreme SSD']
    for path in candidates():
        content = path.read_text(encoding='utf-8')
        for pattern in patterns:
            # Report filenames only, never the matched potential secret.
            assert not re.search(pattern, content), str(path.relative_to(ROOT))


def test_all_notebooks_are_clean_valid_and_compile():
    paths = list(ROOT.glob('notebooks/*.ipynb')) + list(ROOT.glob('experiment/*.ipynb'))
    assert len(list(ROOT.glob('experiment/*.ipynb'))) == 29
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for index, cell in enumerate(nb.cells):
            assert not cell.get('attachments'), path.name
            if cell.cell_type == 'code':
                assert cell.get('outputs') == [], path.name
                assert cell.get('execution_count') is None, path.name
                compile(cell.source, f'{path.name}:cell{index}', 'exec')


def test_all_python_sources_compile():
    for path in candidates():
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))


def test_five_model_analysis_bundle():
    directory = ROOT / 'experiment'
    config = json.loads((directory / 'analysis_model_runs.json').read_text())
    assert len(config['model_runs']) == 5
    assert config['model_runs']['pulse7b_ecg'] == 'pulse_compact_v1'
    for prefix in ('13E', '14', '15', '16'):
        [path] = list(directory.glob(prefix + '_*.ipynb'))
        assert 'e4_jbhi_analysis' in path.read_text(), path.name


def test_extension_release_is_complete():
    directory = ROOT / 'experiment'
    for prefix in ('17', '18', '19', '20', '21'):
        assert len(list(directory.glob(prefix + '_*.ipynb'))) == 1
    for name in ('e4_review_controls.py', 'e4_pulse_prompt_v2.py',
                 'e4_review_pulse_v2.py', 'test_review_controls.py',
                 'test_pulse_prompt_v2.py'):
        assert (directory / name).is_file()
    assert (ROOT / 'docs/extension_workflow.md').is_file()
    [pulse] = directory.glob('20_*.ipynb')
    assert 'IMAGES_APPROVED = False' in pulse.read_text()
    [evaluation] = directory.glob('21_*.ipynb')
    assert 'pulse_prompt_v2' in evaluation.read_text()
