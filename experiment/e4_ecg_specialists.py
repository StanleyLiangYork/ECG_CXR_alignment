"""ECG-image specialist generation for Experiment 4.

The two specialists use different upstream runtimes, so imports are deliberately
lazy.  Both consume the same ECG PNG rendered from the query WFDB waveform and
the same frozen retrieval contexts used by the generalist generators.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image

from e4_common import (EXPERIMENT4_ROOT, HF_CACHE_ROOT, read_jsonl, sha256,
                       signature, write_json)
from e4_generation import (_check_resume_settings, _completed_logical_keys, _extract_clinical_output, _logical_key, _trim_prompt,
                           build_prompt, query_image)
from e4_model_registry import SPECIALIST_MODELS
from e4_generation import _visible_response
from e4_generation_runtime import (generation_lock, generation_root, prepare_run, run_progress,
    select_pilot, pinned_revision, register_protocol, token_diagnostics, error_category, advance_progress)


def _pulse_repo_root():
    return Path(os.environ.get(
        "PULSE_REPO_ROOT", str(Path(__file__).resolve().parent.parent / "external" / "PULSE")
    )).resolve()


def validate_specialist_environment(model_key):
    if model_key not in SPECIALIST_MODELS:
        raise ValueError(f"Unknown ECG specialist: {model_key}")
    spec = SPECIALIST_MODELS[model_key]
    report = {"model_key": model_key, "model_id": spec["model_id"],
              "loader": spec["loader"], "cuda": torch.cuda.is_available(),
              "python": sys.version, "torch": torch.__version__}
    if not torch.cuda.is_available():
        raise RuntimeError("ECG specialist generation requires a CUDA GPU node")
    if spec["loader"] == "pulse_llava":
        root = _pulse_repo_root()
        package = root / "LLaVA" / "llava"
        report.update(pulse_repo_root=str(root), pulse_package_exists=package.is_dir())
        if not package.is_dir():
            raise FileNotFoundError(
                f"PULSE source not found at {package}. Install the official AIMedLab/PULSE "
                "repository in its separate Python 3.10 environment and set PULSE_REPO_ROOT."
            )
        try:
            report['pulse_git_commit'] = subprocess.check_output(
                ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            report['pulse_git_commit'] = 'unavailable'
    elif spec["loader"] == "mllama_transformers":
        try:
            from transformers import AutoProcessor, MllamaForConditionalGeneration  # noqa: F401
            report["ecg_instruct_backend"] = "transformers_mllama"
            report["transformers_version"] = version("transformers")
        except Exception as exc:
            raise RuntimeError(
                "ECG-Instruct requires a Transformers build with "
                "MllamaForConditionalGeneration (Transformers >= 4.46)."
            ) from exc
        if os.environ.get("E4_ECG_INSTRUCT_4BIT", "1").lower() in {"1", "true", "yes"}:
            try:
                import bitsandbytes  # noqa: F401
                report["bitsandbytes_version"] = version("bitsandbytes")
            except Exception as exc:
                raise RuntimeError(
                    "Four-bit ECG-Instruct loading requires bitsandbytes. Install it in the "
                    "active kernel (`python -m pip install bitsandbytes`) or set "
                    "E4_ECG_INSTRUCT_4BIT=0 on a sufficiently large GPU."
                ) from exc
    else:
        raise ValueError(f"Unsupported specialist loader: {spec['loader']}")
    for name in ('transformers', 'huggingface_hub', 'accelerate', 'bitsandbytes'):
        try: report[name] = version(name)
        except Exception: report[name] = 'not installed'
    return report


def _load_pulse(spec):
    root = _pulse_repo_root()
    llava_root = root / "LLaVA"
    if str(llava_root) not in sys.path:
        sys.path.insert(0, str(llava_root))
    try:
        import llava
        if not Path(llava.__file__).resolve().is_relative_to(llava_root):
            raise RuntimeError('A different LLaVA package is already imported; restart in the PULSE kernel')
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
    except Exception as exc:
        raise RuntimeError(
            f"Could not import the official PULSE LLaVA runtime from {llava_root}. "
            "Use the dedicated Python 3.10 PULSE kernel described in the notebook."
        ) from exc
    disable_torch_init()
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(
        repo_id=spec["model_id"], revision=spec["revision"],
        cache_dir=str(HF_CACHE_ROOT),
    )
    tokenizer, model, image_processor, context_length = load_pretrained_model(
        snapshot, None, "pulse-7b", device_map="auto"
    )
    model.eval()
    # The upstream builder defaults to 2048 when max_sequence_length is absent,
    # but the published PULSE configuration explicitly supports 4096.
    limits = [getattr(model.config, k, None) for k in ('max_position_embeddings', 'tokenizer_model_max_length')]
    limits = [int(v) for v in limits if isinstance(v, int) and 0 < v < 10**7]
    return {"tokenizer": tokenizer, "model": model, "image_processor": image_processor,
            "context_length": min(limits) if limits else int(context_length)}


def _pulse_generate(runtime, image_path, prompt, max_new_tokens, max_input_tokens):
    from llava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                                 DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    tokenizer, model = runtime["tokenizer"], runtime["model"]
    image_processor = runtime["image_processor"]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    image_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                   if getattr(model.config, "mm_use_im_start_end", False)
                   else DEFAULT_IMAGE_TOKEN)
    conversation = conv_templates["llava_v1"].copy()
    conversation.append_message(conversation.roles[0], image_token + "\n" + prompt)
    conversation.append_message(conversation.roles[1], None)
    model_prompt = conversation.get_prompt()
    input_ids = tokenizer_image_token(
        model_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    input_tokens = int(input_ids.shape[-1])
    runtime['diagnostics'] = {'processor_input_tokens': input_tokens}
    if input_tokens > max_input_tokens:
        raise ValueError(f"PULSE prompt has {input_tokens} tokens, exceeding {max_input_tokens}")
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    images = process_images([image], image_processor, model.config)
    images = ([value.to(device=device, dtype=torch.float16) for value in images]
              if isinstance(images, list) else images.to(device=device, dtype=torch.float16))
    if getattr(model.config, 'mm_patch_merge_type', 'flat') != 'flat':
        raise ValueError('PULSE context accounting currently requires the published flat patch merge')
    tensor = images[0] if isinstance(images, list) or images.ndim == 5 else images
    crops = int(tensor.shape[0])
    image_tokens = crops * int(model.get_vision_tower().num_patches)
    expanded_tokens = input_tokens - 1 + image_tokens
    runtime['diagnostics']['processor_input_tokens'] = expanded_tokens
    if expanded_tokens > max_input_tokens or expanded_tokens + max_new_tokens > runtime['context_length']:
        raise ValueError(f'PULSE expanded input={expanded_tokens}, output reserve={max_new_tokens}, '
                         f'context={runtime["context_length"]}; lower E4_PULSE_COMPONENT_CAP for the whole run')
    with torch.inference_mode():
        output = model.generate(
            input_ids, images=images, image_sizes=[image.size], do_sample=False,
            max_new_tokens=max_new_tokens, use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    # Official PULSE generate uses inputs_embeds and returns only newly generated IDs.
    runtime['diagnostics'].update(token_diagnostics(output[0], max_new_tokens,
        getattr(getattr(model, 'generation_config', None), 'eos_token_id', tokenizer.eos_token_id)))
    return tokenizer.batch_decode(output, skip_special_tokens=False)[0].strip(), expanded_tokens


def _load_mllama_transformers(spec, max_input_tokens, max_new_tokens):
    """Load the merged ECG-Instruct Mllama checkpoint without requiring Unsloth."""
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              MllamaForConditionalGeneration)
    load_in_4bit = os.environ.get("E4_ECG_INSTRUCT_4BIT", "1").lower() in {
        "1", "true", "yes"
    }
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    common = {"revision": spec["revision"], "cache_dir": str(HF_CACHE_ROOT)}
    processor = AutoProcessor.from_pretrained(spec["model_id"], **common)
    model_kwargs = {**common, "device_map": "auto", "low_cpu_mem_usage": True, "dtype": dtype}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = MllamaForConditionalGeneration.from_pretrained(
        spec["model_id"], **model_kwargs
    )
    model.eval()
    return {"model": model, "processor": processor,
            "tokenizer": processor.tokenizer, "load_in_4bit": load_in_4bit,
            "dtype": dtype, "backend": "transformers_mllama"}


def _mllama_generate(runtime, image_path, prompt, max_new_tokens, max_input_tokens):
    model, processor, tokenizer = runtime["model"], runtime["processor"], runtime["tokenizer"]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}
    ]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat, images=image, return_tensors="pt", add_special_tokens=False)
    input_tokens = int(inputs["input_ids"].shape[-1])
    runtime['diagnostics'] = {'processor_input_tokens': input_tokens}
    if input_tokens > max_input_tokens:
        raise ValueError(
            f"ECG-Instruct prompt has {input_tokens} tokens, exceeding {max_input_tokens}"
        )
    context_length = getattr(model.config.text_config, 'max_position_embeddings', max_input_tokens + max_new_tokens)
    if input_tokens + max_new_tokens > context_length: raise ValueError('ECG-Instruct context limit exceeded')
    device = model.get_input_embeddings().weight.device
    inputs = {key: (value.to(device=device, dtype=runtime["dtype"])
                    if value.is_floating_point() else value.to(device))
              for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            use_cache=True, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[-1]:]
    runtime['diagnostics'].update(token_diagnostics(generated, max_new_tokens,
        getattr(getattr(model, 'generation_config', None), 'eos_token_id', tokenizer.eos_token_id)))
    return tokenizer.decode(generated, skip_special_tokens=False).strip(), input_tokens


def _existing_successes(path, contexts):
    if not path.is_file():
        return set()
    return _completed_logical_keys(Path(path).stem, Path(path).parent, contexts)


@generation_lock
def run_ecg_specialist(model_key, out=EXPERIMENT4_ROOT, max_new_tokens=None,
                       max_input_tokens=None, max_rows=0):
    """Run one ECG-image specialist; append-only and resumable by logical cell."""
    if model_key not in SPECIALIST_MODELS: raise ValueError(f'Unknown specialist: {model_key}')
    if max_new_tokens is None: max_new_tokens = 512 if model_key == 'pulse7b_ecg' else 1024
    if max_input_tokens is None: max_input_tokens = 3584 if model_key == 'pulse7b_ecg' else 6000
    if max_rows < 0 or min(max_new_tokens, max_input_tokens) < 1: raise ValueError('Invalid generation limits')
    spec = SPECIALIST_MODELS[model_key]
    out = Path(out)
    contexts_path = out / "contexts" / "bidirectional_contexts.jsonl"
    all_contexts = read_jsonl(contexts_path)
    from e4_evidence import validate_contexts
    validate_contexts(all_contexts, out)
    contexts = [row for row in all_contexts if row["direction"] == "ecg"]

    destination = generation_root(out) / f"{model_key}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    component_cap = int(os.environ.get('E4_PULSE_COMPONENT_CAP', '240')) if spec['loader'] == 'pulse_llava' else 1800
    if component_cap < 1: raise ValueError('Evidence component cap must be positive')
    quantized = (os.environ.get('E4_ECG_INSTRUCT_4BIT', '1').lower() in {'1','true','yes'}
                 if spec['loader'] == 'mllama_transformers' else False)
    all_selected_contexts = contexts
    dest, pending, progress = prepare_run(model_key, out, contexts,
        {'max_new_tokens': max_new_tokens, 'max_input_tokens': max_input_tokens,
         'evidence_component_character_cap': component_cap, 'load_in_4bit': quantized, 'spec': spec},
        [Path(__file__), Path(__file__).with_name('e4_generation.py'), Path(__file__).with_name('e4_model_registry.py')])
    print(progress, flush=True)
    contexts = select_pilot(pending, max_rows)
    if not contexts:
        return {**progress, 'status': 'already_complete' if not progress['failed'] else 'failed_cells_require_new_protocol'}
    environment = validate_specialist_environment(model_key)
    from huggingface_hub import HfApi
    revision = HfApi().model_info(spec["model_id"], revision=pinned_revision(dest, model_key)).sha
    protocol = {
        "protocol_type": "ecg_image_specialist_v3",
        "model_key": model_key, **spec, "directions": ["ecg"],
        "resolved_revision": revision,
        "specialist_code_sha256": sha256(Path(__file__)),
        "generation_code_sha256": sha256(Path(__file__).with_name("e4_generation.py")),
        "contexts_sha256": sha256(contexts_path),
        "max_new_tokens": int(max_new_tokens), "max_input_tokens": int(max_input_tokens),
        "evidence_component_character_cap": component_cap,
        "load_in_4bit": os.environ.get('E4_ECG_INSTRUCT_4BIT', '1').lower() in {'1','true','yes'}
                        if spec['loader'] == 'mllama_transformers' else False,
        "clinical_output_policy": (
            "strict JSON preferred; visible final free text accepted; schema compliance is separate"
        ),
        "raw_sensor_input": "12-lead ECG PNG deterministically rendered from the query WFDB waveform",
        "hidden_reasoning_storage": "not requested or stored",
        "environment": environment,
    }
    protocol['attempts_per_cell'] = 1
    protocol['run_manifest_sha256'] = sha256(dest / f'{model_key}_run_manifest.json')
    register_protocol(dest, model_key, protocol)

    pinned_spec = {**spec, "revision": revision}
    runtime = (_load_pulse(pinned_spec) if spec["loader"] == "pulse_llava"
               else _load_mllama_transformers(pinned_spec, max_input_tokens, max_new_tokens))
    generator = _pulse_generate if spec["loader"] == "pulse_llava" else _mllama_generate
    for position, row in enumerate(contexts, 1):
        prompt = build_prompt(row, "raw_sensor_plus_retrieval", component_cap)
        image_path, image_error, image_hash = None, None, None
        try:
            image_path = query_image(row, out)
            image_hash = sha256(image_path)
        except Exception as exc:
            image_error = repr(exc)
        result = {
            "generation_key": signature([model_key, revision, row["context_signature"],
                                         prompt, max_new_tokens, max_input_tokens, signature(protocol),
                                         image_hash]),
            "model_key": model_key, "model_id": spec["model_id"],
            "resolved_revision": revision, "direction": "ecg",
            "sample_id": row["sample_id"], "method": row["method"],
            "condition": row["condition"], "context_signature": row["context_signature"],
            "input_scope": "raw_sensor_plus_retrieval", "query_image": str(image_path) if image_path else None,
            "query_image_sha256": image_hash,
            "gate_passed": row.get("gate_passed"), "prompt": prompt,
            "generation_protocol_signature": signature(protocol),
            "evidence_component_character_cap": component_cap,
            "model_role": spec["role"], "training_overlap_warning": spec["training_overlap"],
        }
        result.update(visible_response=None, output_parse_stage=None, response_sha256=None,
                      generated_token_count=None, output_token_limit_reached=None, finish_reason=None,
                      error_category=None, retryable=False)
        runtime['diagnostics'] = {}
        try:
            if image_error:
                raise ValueError(image_error)
            _trim_prompt(prompt, runtime["tokenizer"], max_input_tokens)
            text, input_tokens = generator(
                runtime, image_path, prompt, max_new_tokens, max_input_tokens
            )
            result.update(runtime['diagnostics'])
            visible = _visible_response(text)
            result.update(visible_response=visible, response_sha256=hashlib.sha256(visible.encode()).hexdigest())
            allowed = {anchor["anchor_id"] for anchor in row["evidence"]}
            parsed, stage, compliant, visible = _extract_clinical_output(text, allowed)
            result['output_parse_stage'] = stage
            if result['finish_reason'] == 'length':
                raise ValueError('Output token limit reached without EOS; partial report excluded from primary analysis')
            if parsed is None:
                raise ValueError(f"No usable visible clinical report ({stage})")
            result.update(
                status="ok", processor_input_tokens=input_tokens,
                generated_report=parsed["report"], evidence_trace=parsed.get("evidence_trace", []),
                uncertainties=parsed.get("uncertainties", []), output_parse_stage=stage,
                structured_schema_compliant=bool(compliant), visible_response=visible,
                response_sha256=hashlib.sha256(visible.encode()).hexdigest(), error=None,
            )
        except Exception as exc:
            result.update(runtime['diagnostics'])
            result.update(
                status="error", generated_report="", evidence_trace=[], uncertainties=[],
                structured_schema_compliant=False, error=repr(exc),
                error_category=error_category(exc, result.get('output_parse_stage')),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with destination.open("a") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        advance_progress(dest, progress, result)
        print(model_key, position, "/", len(contexts), row["sample_id"],
              row["method"], row["condition"], result["status"], flush=True)
        if result.get('error_category') == 'out_of_memory':
            raise RuntimeError('GPU out of memory; run stopped. Review resources before choosing a new run.')
    del runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    progress, _ = run_progress(model_key, out, all_selected_contexts)
    return {'protocol': protocol, 'progress': progress}


def specialist_coverage(out=EXPERIMENT4_ROOT):
    out = Path(out)
    contexts = [row for row in read_jsonl(out / "contexts" / "bidirectional_contexts.jsonl")
                if row["direction"] == "ecg"]
    expected = {_logical_key(row) for row in contexts}
    rows = []
    for key, spec in SPECIALIST_MODELS.items():
        path = generation_root(out) / f"{key}.jsonl"
        attempts = read_jsonl(path)
        successful = _existing_successes(path, contexts)
        rows.append({"model_key": key, "expected_ecg_cells": len(expected),
                     "attempt_rows": len(attempts), "successful_cells": len(successful & expected),
                     "missing_cells": len(expected - successful),
                     "training_overlap_warning": spec["training_overlap"]})
    return rows
