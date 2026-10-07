"""Offline regression tests for the direction-specific generator amendment."""
import importlib
import json
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(tmp_path, monkeypatch):
    monkeypatch.setenv('E4_GENERATION_RUN', 'legacy')
    for name in ('EXPERIMENT4_ROOT', 'ECG_CXR_BRIDGE_ARTIFACT_ROOT', 'SYMILE_ROOT', 'HF_HOME'):
        monkeypatch.setenv(name, str(tmp_path / name))
    return SimpleNamespace(g=importlib.import_module('e4_generation'),
                           s=importlib.import_module('e4_ecg_specialists'),
                           e=importlib.import_module('e4_evaluation'))


def row(direction='ecg', context='current', model='qwen35_4b'):
    return dict(direction=direction, sample_id='sample', method='clip', condition='full_trimodal',
                context_signature=context, model_key=model, status='ok', generated_report='Normal ECG.',
                generation_protocol_signature='protocol', evidence=[])


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


@pytest.mark.parametrize('text', [
    '<think>unfinished private reasoning',
    '<|channel|>analysis<|message|>{"report":"not final"}',
    '<|eot_id|>', '</s>', '<eos>',
])
def test_hidden_and_empty_outputs_rejected(modules, text):
    assert modules.g._extract_clinical_output(text)[0] is None
    assert modules.g._visible_response(text) == ''


def test_special_tokens_and_invalid_citations(modules):
    data = {'report': 'Normal ECG.', 'uncertainties': [],
            'evidence_trace': [{'claim': 'Normal', 'source_ids': ['invented']}]}
    parsed, stage, schema, visible = modules.g._extract_clinical_output(json.dumps(data) + '<|eot_id|>')
    assert parsed['evidence_trace'][0]['source_ids'] == ['invented']
    assert schema and stage == 'strict_json' and '<|eot_id|>' not in visible
    assert modules.g._visible_response('<think>private</think>Normal ECG.</s>') == 'Normal ECG.'


def test_rescue_does_not_extract_hidden_json(modules):
    from e4_format_rescue import parse_rescue_result
    result, _ = parse_rescue_result('<|channel|>analysis<|message|>{"final_report":"Normal ECG."}', set())
    assert result is None


def test_stale_contexts_are_not_completed(modules, tmp_path):
    save(tmp_path / 'qwen35_4b.jsonl', [row(context='old')])
    assert not modules.g._completed_logical_keys('qwen35_4b', tmp_path, [row()])
    save(tmp_path / 'qwen35_4b_format_rescue.jsonl', [row()])
    assert len(modules.g._completed_logical_keys('qwen35_4b', tmp_path, [row()])) == 1


def test_complete_run_does_not_load_model(modules, tmp_path, monkeypatch):
    from pathlib import Path
    from e4_generation_runtime import prepare_run, register_protocol
    from e4_common import signature
    monkeypatch.setenv('E4_GENERATION_RUN', 'test_complete')
    save(tmp_path / 'contexts' / 'bidirectional_contexts.jsonl', [row()])
    gfile = Path(modules.g.__file__)
    settings = {'max_new_tokens': 1024, 'max_input_tokens': 12288,
                'spec': modules.g.MODELS['qwen35_4b'], 'evidence_component_character_cap': 1800}
    root, _, _ = prepare_run('qwen35_4b', tmp_path, [row()], settings,
                            [gfile, gfile.with_name('e4_model_registry.py')])
    protocol = {'resolved_revision': 'test'}
    register_protocol(root, 'qwen35_4b', protocol)
    item = row(); item['generation_protocol_signature'] = signature(protocol)
    save(root / 'qwen35_4b.jsonl', [item])
    monkeypatch.setattr(importlib.import_module('e4_evidence'), 'validate_contexts', lambda *a: None)
    monkeypatch.setattr(modules.g, '_load_model', lambda *a: pytest.fail('must not load'))
    assert modules.g.run_model('qwen35_4b', out=tmp_path)['status'] == 'already_complete'
    settings = {'max_new_tokens': 512, 'max_input_tokens': 3584, 'evidence_component_character_cap': 240,
                'load_in_4bit': False, 'spec': modules.s.SPECIALIST_MODELS['pulse7b_ecg']}
    root, _, _ = prepare_run('pulse7b_ecg', tmp_path, [row()], settings,
        [Path(modules.s.__file__), gfile, gfile.with_name('e4_model_registry.py')])
    register_protocol(root, 'pulse7b_ecg', protocol)
    item = row(model='pulse7b_ecg'); item['generation_protocol_signature'] = signature(protocol)
    save(root / 'pulse7b_ecg.jsonl', [item])
    monkeypatch.setattr(modules.s, 'validate_specialist_environment', lambda *a: pytest.fail('no CUDA check needed'))
    assert modules.s.run_ecg_specialist('pulse7b_ecg', out=tmp_path)['status'] == 'already_complete'


def test_resume_rejects_changed_settings(modules, tmp_path):
    from e4_common import signature
    protocol = {'max_new_tokens': 512}
    item = row(); item['generation_protocol_signature'] = signature(protocol)
    save(tmp_path / 'qwen35_4b.jsonl', [item])
    (tmp_path / 'qwen35_4b_protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='max_new_tokens'):
        modules.g._check_resume_settings('qwen35_4b', tmp_path, [row()], {'max_new_tokens': 1024})


def test_registry_scope_and_rescue_precedence(modules, tmp_path):
    base = row(); base['generated_report'] = 'Original.'
    rescue = row(); rescue['generated_report'] = 'Rescued.'
    save(tmp_path / 'generations' / 'qwen35_4b.jsonl', [base])
    save(tmp_path / 'generations' / 'qwen35_4b_format_rescue.jsonl', [rescue])
    save(tmp_path / 'generations' / 'medgemma15_4b.jsonl', [row(model='medgemma15_4b')])
    output = modules.e._successful_generations(tmp_path, {'current': row()})
    assert len(output) == 1 and output[0]['generated_report'] == 'Original.'


def test_bootstrap_does_not_pair_different_protocols(modules, tmp_path):
    import pandas as pd
    a, b = row(), row()
    b.update(condition='same_only', generation_protocol_signature='different')
    for r in (a,b): r['input_scope'] = 'raw_sensor_plus_retrieval'
    (tmp_path / 'evaluation').mkdir()
    modules.e._paired_bootstrap(pd.DataFrame([a,b]), tmp_path, bootstraps=5)
    assert pd.read_csv(tmp_path / 'evaluation' / 'paired_bootstrap.csv').empty


def test_pulse_image_budget_and_pad_zero(modules, monkeypatch, tmp_path):
    import torch
    from PIL import Image
    path = tmp_path / 'ecg.png'; Image.new('RGB', (100, 100)).save(path)
    constants = SimpleNamespace(DEFAULT_IMAGE_TOKEN='<image>', DEFAULT_IM_END_TOKEN='',
                                DEFAULT_IM_START_TOKEN='', IMAGE_TOKEN_INDEX=-200)
    conversation = SimpleNamespace(roles=['user', 'assistant'], append_message=lambda *a: None,
                                   get_prompt=lambda: 'chat')
    conversation.copy = lambda: conversation
    input_length = [200]
    monkeypatch.setitem(sys.modules, 'llava.constants', constants)
    monkeypatch.setitem(sys.modules, 'llava.conversation', SimpleNamespace(conv_templates={'llava_v1': conversation}))
    monkeypatch.setitem(sys.modules, 'llava.mm_utils', SimpleNamespace(
        process_images=lambda *a: torch.zeros((1,5,3,2,2)),
        tokenizer_image_token=lambda *a, **kw: torch.arange(input_length[0])))
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return torch.tensor([[10,11]])
    model = SimpleNamespace(config=SimpleNamespace(mm_patch_merge_type='flat'),
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1)),
        get_vision_tower=lambda: SimpleNamespace(num_patches=576), generate=generate)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
        batch_decode=lambda output, **kw: ['Normal ECG.</s>'])
    runtime = dict(model=model, tokenizer=tokenizer, image_processor=None, context_length=4096)
    text, tokens = modules.s._pulse_generate(runtime, path, 'prompt', 512, 6000)
    assert tokens == 3079 and calls[0]['pad_token_id'] == 0
    assert text.endswith('</s>')
    input_length[0] = 1200
    with pytest.raises(ValueError, match='PULSE expanded input'):
        modules.s._pulse_generate(runtime, path, 'prompt', 512, 6000)
    assert len(calls) == 1


def test_mllama_template_and_completion_only(modules, monkeypatch, tmp_path):
    import torch
    from PIL import Image
    image = tmp_path / 'ecg.png'; Image.new('RGB', (8,8)).save(image)
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['tokenize'] is False
            return 'chat'
        def __call__(self, **kwargs):
            assert kwargs['add_special_tokens'] is False
            return {'input_ids': torch.tensor([[1,2,3]])}
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
        decode=lambda tokens, **kw: 'Normal ECG.' if tokens.tolist() == [4,5] else 'PROMPT LEAK')
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=100)),
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1)),
        generate=lambda **kw: torch.tensor([[1,2,3,4,5]]))
    text, tokens = modules.s._mllama_generate(dict(model=model, processor=Processor(), tokenizer=tokenizer,
                                                 dtype=torch.float32),
                                             image, 'prompt', 10, 50)
    assert text == 'Normal ECG.' and tokens == 3
