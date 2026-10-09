"""Prepare an isolated PULSE prompt revision without altering Qwen's hashed sources."""
from pathlib import Path
import hashlib
import json
import os
import sys
import fcntl
import shutil
import importlib


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, code_root, activate=True):
    source, code_root = Path(source).resolve(), Path(code_root).resolve()
    dest = source / 'pulse_prompt_v2'
    dest.mkdir(exist_ok=True)
    with (dest / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = source / 'generations/pulse7b_ecg/protocol.json'
        prior_protocol = json.loads(prior.read_text())
        if prior_protocol['settings']['model_key'] != 'pulse7b_ecg':
            raise ValueError('Expected the original PULSE pilot protocol')
        revision = prior_protocol['resolved_revision']
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('Expected a pinned Hugging Face commit hash')
        inputs = ['protocol.json', 'contexts.jsonl', 'image_audit.jsonl']
        modules = sorted(code_root.glob('e4_*.py'))
        manifest = {
            'prompt_version': 'pulse_conditional_plain_report_v2',
            'source': str(source), 'resolved_revision': revision,
            'inputs': {n: _sha(source/n) for n in inputs},
            'source_modules': {p.name: _sha(p) for p in modules},
        }
        marker = dest / 'prompt_revision.json'
        if marker.exists() and json.loads(marker.read_text()) != manifest:
            raise ValueError('Frozen v2 inputs or source code changed; do not mix protocols')
        snapshot = dest / 'code_snapshot'
        snapshot.mkdir(exist_ok=True)
        for p in modules:
            content = p.read_text()
            if p.name == 'e4_pulse_generation.py':
                content += '\nfrom e4_pulse_prompt_v2 import build_pulse_prompt, PROMPT_VERSION\n'
            if p.name == 'e4_review_controls.py':
                old = "revision=old['resolved_revision'] if old else HfApi().model_info(spec['model_id']).sha"
                if content.count(old) != 1:
                    raise ValueError('Runner changed; inspect the model revision binding before proceeding')
                content = content.replace(old, "revision=old['resolved_revision'] if old else " + repr(revision))
            target = snapshot / p.name
            if target.exists():
                if target.read_text() != content:
                    raise ValueError(f'Frozen code snapshot changed: {target}')
            else:
                with target.open('x') as stream:
                    stream.write(content)
        for name in inputs:
            target = dest / name
            if target.exists():
                if _sha(target) != manifest['inputs'][name]:
                    raise ValueError(f'Frozen input changed: {target}')
            else:
                shutil.copy2(source/name, target)
        # Evaluation reads the original Qwen log without copying successes or changing its protocol.
        gen = dest / 'generations'
        gen.mkdir(exist_ok=True)
        qwen = gen / 'qwen35_4b'
        original_qwen = source / 'generations/qwen35_4b'
        if qwen.is_symlink():
            if qwen.resolve() != original_qwen.resolve():
                raise ValueError('Unexpected Qwen evaluation link')
        elif qwen.exists():
            raise ValueError('Refusing to replace an existing Qwen directory')
        else:
            qwen.symlink_to(original_qwen, target_is_directory=True)
        if not marker.exists():
            with marker.open('x') as stream:
                json.dump(manifest, stream, indent=2)
    if activate:
        # Reject mixed imports rather than silently reloading live generation modules.
        for name in ('e4_review_controls', 'e4_pulse_generation', 'e4_ecg_specialists'):
            loaded = sys.modules.get(name)
            if loaded and Path(loaded.__file__).resolve().parent != snapshot:
                raise RuntimeError('Restart the kernel and run Notebook 20 from the top; legacy helpers are already imported')
        sys.path.insert(0, str(snapshot))
        importlib.invalidate_caches()
    print('PULSE v2 output:', dest / 'generations/pulse7b_ecg')
    print('Pinned model revision:', revision)
    return dest
