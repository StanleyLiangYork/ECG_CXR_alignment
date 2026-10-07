"""Offline tests for isolated runs, input budgets, and visible-output diagnostics."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv('E4_GENERATION_RUN', 'test_v3')
    for key in ('EXPERIMENT4_ROOT', 'ECG_CXR_BRIDGE_ARTIFACT_ROOT', 'SYMILE_ROOT', 'HF_HOME'):
        monkeypatch.setenv(key, str(tmp_path / key))
    g = importlib.import_module('e4_generation')
    r = importlib.import_module('e4_generation_runtime')
    monkeypatch.setattr(importlib.import_module('e4_evidence'), 'validate_contexts', lambda *a: None)
    monkeypatch.setattr(g, 'version', lambda *a: 'test')
    image = tmp_path / 'image.png'
    Image.new('RGB', (8, 8)).save(image)
    monkeypatch.setattr(g, 'query_image', lambda *a: image)
    context = dict(direction='cxr', sample_id='case', method='clip', condition='full_trimodal',
                   context_signature='current', evidence=[])
    path = tmp_path / 'contexts' / 'bidirectional_contexts.jsonl'
    path.parent.mkdir()
    path.write_text(json.dumps(context) + '\n')
    state = SimpleNamespace(input_tokens=8000, output_ids=[4, 5, 2], calls=0,
                            text='{"report":"No acute finding.","evidence_trace":[],"uncertainties":[]}')

    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0

        def __call__(self, text, **kwargs):
            return {'input_ids': [1, 2, 3]}

        def decode(self, ids, **kwargs):
            return state.text

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs['enable_thinking'] is False
            return {'input_ids': torch.ones((1, state.input_tokens), dtype=torch.long)}

    def generate(**kwargs):
        state.calls += 1
        return torch.cat([kwargs['input_ids'], torch.tensor([state.output_ids])], dim=1)

    model = SimpleNamespace(config=SimpleNamespace(_commit_hash='pinned-test',
                            text_config=SimpleNamespace(max_position_embeddings=32768)),
                            generation_config=SimpleNamespace(eos_token_id=[2]),
                            get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1)),
                            generate=generate)
    monkeypatch.setattr(g, '_load_model', lambda *a: (model, Processor(), Tokenizer(), torch.float32))
    return SimpleNamespace(g=g, r=r, state=state, out=tmp_path, context=context,
                           result=r.generation_root(tmp_path) / 'qwen35_4b.jsonl')


def read_result(runtime):
    return json.loads(runtime.result.read_text().splitlines()[-1])


def test_expanded_8000_tokens_succeed_and_legacy_untouched(runtime):
    legacy = runtime.out / 'generations' / 'qwen35_4b.jsonl'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('legacy audit log\n')
    result = runtime.g.run_model('qwen35_4b', out=runtime.out)
    row = read_result(runtime)
    assert row['status'] == 'ok' and row['processor_input_tokens'] == 8000
    assert row['generated_token_count'] == 3 and row['finish_reason'] == 'eos'
    assert result['progress']['successful'] == 1
    assert legacy.read_text() == 'legacy audit log\n'
    assert runtime.g.run_model('qwen35_4b', out=runtime.out)['status'] == 'already_complete'
    assert runtime.state.calls == 1


def test_overflow_precedes_generation_and_is_not_retried(runtime):
    runtime.state.input_tokens = 13000
    runtime.g.run_model('qwen35_4b', out=runtime.out)
    row = read_result(runtime)
    assert row['error_category'] == 'input_budget'
    assert row['processor_input_tokens'] == 13000 and runtime.state.calls == 0
    result = runtime.g.run_model('qwen35_4b', out=runtime.out)
    assert result['failed'] == 1 and result['pending'] == 0
    assert len(runtime.result.read_text().splitlines()) == 1


def test_malformed_visible_response_preserved_without_hidden_text(runtime):
    runtime.state.text = '<think>SECRET_REASONING_SENTINEL</think>{"report":'
    runtime.g.run_model('qwen35_4b', out=runtime.out)
    row = read_result(runtime)
    assert row['visible_response'] == '{"report":'
    assert row['output_parse_stage'] == 'malformed_structured_output'
    assert row['error_category'] == 'output_format'
    assert 'SECRET_REASONING_SENTINEL' not in runtime.result.read_text()


def test_token_limited_report_excluded(runtime):
    runtime.state.output_ids = [7] * 1024
    runtime.g.run_model('qwen35_4b', out=runtime.out)
    row = read_result(runtime)
    assert row['output_token_limit_reached'] and row['finish_reason'] == 'length'
    assert row['status'] == 'error' and not row['generated_report']
    assert row['error_category'] == 'output_budget' and row['visible_response']


def test_changed_settings_require_new_run(runtime):
    runtime.g.run_model('qwen35_4b', out=runtime.out)
    with pytest.raises(ValueError, match='NEW E4_GENERATION_RUN'):
        runtime.g.run_model('qwen35_4b', out=runtime.out, max_new_tokens=2048)


def test_protocol_tampering_rejected(runtime):
    runtime.g.run_model('qwen35_4b', out=runtime.out)
    row = read_result(runtime)
    row['generation_protocol_signature'] = 'wrong'
    runtime.result.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='mixed generation protocols'):
        runtime.g.run_model('qwen35_4b', out=runtime.out)


def test_model_context_reserve_checked(runtime):
    with pytest.raises(ValueError, match='output reserve'):
        runtime.r.check_input_budget(8000, 12288, 1024,
            SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=8192)))


def test_run_lock_prevents_second_writer(runtime):
    @runtime.r.generation_lock
    def recurse(model_key, out):
        return runtime.g.run_model(model_key, out=out)
    with pytest.raises(RuntimeError, match='active writer'):
        recurse('qwen35_4b', runtime.out)


def test_evaluation_does_not_mix_legacy(runtime):
    runtime.g.run_model('qwen35_4b', out=runtime.out)
    from e4_evaluation import _successful_generations
    legacy = runtime.out / 'generations' / 'qwen35_4b.jsonl'
    row = read_result(runtime)
    row['generated_report'] = 'Legacy report.'
    legacy.write_text(json.dumps(row) + '\n')
    rows = _successful_generations(runtime.out, {'current': runtime.context})
    assert len(rows) == 1 and rows[0]['generated_report'] == 'No acute finding.'
    assert runtime.r.analysis_root(runtime.out) == runtime.result.parent


def test_stratified_pilot_and_path_validation(runtime, monkeypatch):
    rows = [{**runtime.context, 'direction': d, 'condition': c, 'sample_id': str(i)}
            for d in ('ecg', 'cxr') for c in ('zero_shot', 'full_trimodal') for i in range(3)]
    pilot = runtime.r.select_pilot(rows, 4)
    assert len({(r['direction'], r['condition']) for r in pilot}) == 4
    monkeypatch.setenv('E4_GENERATION_RUN', '../legacy')
    with pytest.raises(ValueError):
        runtime.r.generation_root(runtime.out)


def test_legacy_rescue_disabled(runtime):
    from e4_format_rescue import run_format_rescue
    with pytest.raises(RuntimeError, match='legacy-only'):
        run_format_rescue('qwen35_4b', out=runtime.out)


@pytest.mark.parametrize('model_key', ['pulse7b_ecg', 'ecg_instruct_llama32_11b'])
def test_specialist_failure_diagnostics_and_restart(runtime, monkeypatch, model_key):
    import sys
    import e4_ecg_specialists as specialist
    context = {**runtime.context, 'direction': 'ecg'}
    (runtime.out / 'contexts' / 'bidirectional_contexts.jsonl').write_text(json.dumps(context) + '\n')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=lambda: SimpleNamespace(
        model_info=lambda *a, **kw: SimpleNamespace(sha='pinned-test'))))
    monkeypatch.setattr(specialist, 'validate_specialist_environment', lambda *a: {'test': True})
    monkeypatch.setattr(specialist, 'query_image', lambda *a: runtime.out / 'image.png')
    fake = {'tokenizer': lambda *a, **kw: {'input_ids': [1, 2, 3]}}
    monkeypatch.setattr(specialist, '_load_pulse', lambda *a: fake)
    monkeypatch.setattr(specialist, '_load_mllama_transformers', lambda *a: fake)

    def generate(model_runtime, *args):
        model_runtime['diagnostics'] = {'processor_input_tokens': 3000, 'generated_token_count': 3,
                                        'finish_reason': 'eos', 'output_token_limit_reached': False}
        return '<think>SECRET_REASONING_SENTINEL</think>{"report":', 3000

    monkeypatch.setattr(specialist, '_pulse_generate', generate)
    monkeypatch.setattr(specialist, '_mllama_generate', generate)
    specialist.run_ecg_specialist(model_key, out=runtime.out)
    path = runtime.r.generation_root(runtime.out) / f'{model_key}.jsonl'
    result = json.loads(path.read_text())
    assert result['error_category'] == 'output_format' and result['processor_input_tokens'] == 3000
    assert result['generated_token_count'] == 3 and result['visible_response'] == '{"report":'
    assert 'SECRET_REASONING_SENTINEL' not in path.read_text()
    assert specialist.run_ecg_specialist(model_key, out=runtime.out)['failed'] == 1
    assert len(path.read_text().splitlines()) == 1
