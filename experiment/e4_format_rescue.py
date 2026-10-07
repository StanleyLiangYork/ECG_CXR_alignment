"""Formatting-only rescue for Experiment 4 report generation.

This module deliberately does not modify e4_generation.py.  Original successful
rows remain the primary result.  A rescue row is produced only when no original
successful row exists for the same logical experimental cell.
"""
from __future__ import annotations

import collections
import fcntl
import hashlib
import json
import os
import re
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image

from e4_common import EXPERIMENT4_ROOT, read_jsonl, sha256, signature, write_json
from e4_generation import (MODELS, _extract_json, _load_model, _trim_prompt, _visible_response,
                           build_prompt, query_image)


def logical_key(row):
    return (str(row["direction"]), str(row["sample_id"]),
            str(row["method"]), str(row["condition"]))


def _expected_contexts(model_key, out):
    rows = read_jsonl(Path(out) / "contexts" / "bidirectional_contexts.jsonl")
    allowed = set(MODELS[model_key].get("directions", ("ecg", "cxr")))
    rows = [row for row in rows if row["direction"] in allowed]
    if model_key == "gpt_oss_20b":
        rows = [row for row in rows if row["condition"] != "zero_shot"]
    keys = [logical_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate logical keys in current Experiment 4 contexts")
    return rows


def _read_attempts(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open() as stream:
        lines = stream.readlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSONL at {path}:{line_number}. Preserve the file and repair only "
                "the incomplete final line before resuming."
            ) from exc
    return rows


def _base_state(model_key, out):
    root = Path(out) / "generations"
    base_path = root / f"{model_key}.jsonl"
    protocol_path = root / f"{model_key}_protocol.json"
    if not base_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(
            f"Formatting rescue requires the original output and protocol: {base_path}, {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    rows = _read_attempts(base_path)
    successes = {logical_key(row) for row in rows if row.get("status") == "ok"}
    latest = {}
    for row in rows:
        latest[logical_key(row)] = row
    return base_path, protocol_path, protocol, rows, successes, latest


def _extract_objects(text):
    """Yield JSON objects without retaining or writing the raw response."""
    value = _visible_response(text)
    if "<|channel|>final<|message|>" in value:
        value = value.rsplit("<|channel|>final<|message|>", 1)[1]
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.S)
    value = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", value.strip(), flags=re.I | re.S)
    candidates = [value]
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", value):
        try:
            obj, _ = decoder.raw_decode(value[match.start():])
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            pass


def _first(mapping, names):
    lowered = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in mapping.items()}
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in lowered:
            return lowered[key]
    return None


def _normalize_result(obj, allowed_source_ids):
    """Normalize harmless schema variations; never invent clinical content."""
    if not isinstance(obj, dict):
        return None
    nested = _first(obj, ("result", "response", "output"))
    if isinstance(nested, dict):
        obj = nested
    report = _first(obj, ("report", "final_report", "generated_report",
                          "ecg_report", "cxr_report", "impression"))
    if not isinstance(report, str) or not report.strip():
        return None

    uncertainties = _first(obj, ("uncertainties", "uncertainty", "limitations"))
    if uncertainties is None:
        uncertainties = []
    elif isinstance(uncertainties, str):
        uncertainties = [uncertainties] if uncertainties.strip() else []
    elif not isinstance(uncertainties, list) or not all(isinstance(x, str) for x in uncertainties):
        return None

    trace = _first(obj, ("evidence_trace", "evidence", "citations"))
    if trace is None:
        trace = []
    if not isinstance(trace, list):
        return None
    normalized_trace = []
    for item in trace:
        if not isinstance(item, dict):
            return None
        claim = _first(item, ("claim", "statement", "finding"))
        sources = _first(item, ("source_ids", "sources", "evidence_ids", "anchor_ids"))
        if isinstance(sources, str):
            sources = [sources]
        if not isinstance(claim, str) or not isinstance(sources, list):
            return None
        sources = [str(source) for source in sources]
        if not all(source in allowed_source_ids for source in sources):
            return None
        normalized_trace.append({"claim": claim, "source_ids": sources})
    return {"report": report.strip(), "evidence_trace": normalized_trace,
            "uncertainties": [x.strip() for x in uncertainties if x.strip()]}


def parse_rescue_result(text, allowed_source_ids):
    text = _visible_response(text)
    strict = _extract_json(text)
    if strict is not None:
        if all(str(source) in allowed_source_ids for item in strict["evidence_trace"]
               for source in item["source_ids"]):
            return strict, "strict"
    for obj in _extract_objects(text):
        result = _normalize_result(obj, allowed_source_ids)
        if result is not None:
            return result, "normalized"
    return None, None


def _move_inputs(inputs, device, dtype):
    return {key: (value.to(device=device, dtype=dtype) if value.is_floating_point()
                  else value.to(device)) for key, value in inputs.items()}


def _generate_original(model, processor, tokenizer, dtype, spec, row, prompt,
                       image_path, max_new_tokens, max_input_tokens):
    _trim_prompt(prompt, tokenizer, max_input_tokens)
    model_id = spec["model_id"]
    device = model.get_input_embeddings().weight.device
    if spec["mode"] == "multimodal":
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if model_id.startswith("Qwen/Qwen3.5"):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False)
        else:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=chat, images=[image], return_tensors="pt")
    else:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "gpt-oss" in model_id.lower():
            kwargs["reasoning_effort"] = "low"
        chat = tokenizer.apply_chat_template(messages, **kwargs)
        inputs = tokenizer(chat, return_tensors="pt")
    input_tokens = int(inputs["input_ids"].shape[-1])
    inputs = _move_inputs(inputs, device, dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(output[0, input_tokens:], skip_special_tokens=False).strip()
    return text, input_tokens


def _repair_prompt(raw_text, allowed_source_ids):
    raw_text = _visible_response(raw_text)
    if not raw_text: raise ValueError('No visible final response to reformat')
    example_sources = [sorted(allowed_source_ids)[0]] if allowed_source_ids else []
    example = {"evidence_trace": [{"claim": "claim copied from the response",
                                    "source_ids": example_sources}] if example_sources else [],
               "uncertainties": [], "report": "clinical report copied from the response"}
    return (
        "Reformat the RESPONSE below as exactly one valid JSON object and nothing else. "
        "Do not add, remove, reinterpret, or correct any clinical claim. Do not expose hidden reasoning. "
        "Use exactly the keys evidence_trace, uncertainties, and report. evidence_trace must be a list "
        "of objects with claim and source_ids. uncertainties must be a list of strings. Use source_ids "
        f"only from this list: {sorted(allowed_source_ids)}. If no source supports a claim, omit that "
        "trace item. Example structure:\n" + json.dumps(example, ensure_ascii=False) +
        "\n\nRESPONSE TO REFORMAT:\n" + raw_text
    )


def _generate_text_only(model, processor, tokenizer, dtype, spec, prompt, max_new_tokens):
    device = model.get_input_embeddings().weight.device
    model_id = spec["model_id"]
    if spec["mode"] == "multimodal":
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if model_id.startswith("Qwen/Qwen3.5"):
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False)
        else:
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=chat, return_tensors="pt")
    else:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "gpt-oss" in model_id.lower():
            kwargs["reasoning_effort"] = "low"
        chat = tokenizer.apply_chat_template(messages, **kwargs)
        inputs = tokenizer(chat, return_tensors="pt")
    input_tokens = int(inputs["input_ids"].shape[-1])
    inputs = _move_inputs(inputs, device, dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0, input_tokens:], skip_special_tokens=False).strip(), input_tokens


def _stratified_limit(rows, maximum):
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["direction"], row["method"], row["condition"])].append(row)
    result = []
    names = sorted(groups)
    while len(result) < maximum and names:
        remaining = []
        for name in names:
            if groups[name] and len(result) < maximum:
                result.append(groups[name].pop(0))
            if groups[name]:
                remaining.append(name)
        names = remaining
    return result


def audit_rescue(model_key, out=EXPERIMENT4_ROOT):
    if model_key not in MODELS:
        raise ValueError(model_key)
    contexts = _expected_contexts(model_key, out)
    _, _, base_protocol, base_rows, base_success, _ = _base_state(model_key, out)
    missing = [row for row in contexts if logical_key(row) not in base_success]
    return {"model_key": model_key, "expected": len(contexts),
            "original_attempts": len(base_rows), "original_successful": len(base_success),
            "requires_rescue": len(missing),
            "base_transformers_version": base_protocol.get("transformers_version"),
            "base_model_revision": base_protocol.get("resolved_revision")}


def run_format_rescue(model_key, out=EXPERIMENT4_ROOT, max_rows=50,
                      repair_attempts=2, retry_terminal=False):
    """Rescue missing logical cells without touching the original JSONL."""
    raise RuntimeError('Notebooks 11B/12B/13B are legacy-only and disabled in generation v3. Use 11/12/13 in a new versioned run; do not mix rescue outputs with the primary comparison.')
    if model_key not in MODELS:
        raise ValueError(model_key)
    out = Path(out)
    generation_root = out / "generations"
    generation_root.mkdir(parents=True, exist_ok=True)
    base_path, base_protocol_path, base_protocol, _, base_success, base_latest = _base_state(model_key, out)
    current_transformers = version("transformers")
    if current_transformers != base_protocol.get("transformers_version"):
        raise RuntimeError(
            f"Formatting rescue must reproduce the original environment: Transformers "
            f"{base_protocol.get('transformers_version')} required, found {current_transformers}")
    generation_code = Path(__file__).with_name("e4_generation.py")
    if sha256(generation_code) != base_protocol.get("code_sha256"):
        raise RuntimeError("e4_generation.py differs from the original protocol; restore it before rescue")

    spec = {**MODELS[model_key], "revision": base_protocol["resolved_revision"]}
    protocol = {
        "protocol_type": "formatting_only_rescue_v1",
        "model_key": model_key, "model_id": spec["model_id"],
        "resolved_revision": spec["revision"],
        "base_protocol_sha256": sha256(base_protocol_path),
        "base_output": str(base_path), "base_output_sha256_at_start": sha256(base_path),
        "base_generation_code_sha256": sha256(generation_code),
        "rescue_code_sha256": sha256(Path(__file__)),
        "transformers_version": current_transformers, "torch_version": torch.__version__,
        "original_max_new_tokens": int(base_protocol["max_new_tokens"]),
        "original_max_input_tokens": int(base_protocol["max_input_tokens"]),
        "repair_max_new_tokens": max(768, int(base_protocol["max_new_tokens"])),
        "repair_attempts": int(repair_attempts),
        "policy": "reproduce original deterministic response; normalize harmless JSON variants; otherwise reformat in memory without changing clinical content",
        "raw_response_storage": "SHA256 only; raw response and hidden reasoning are not persisted",
    }
    protocol_signature = signature(protocol)
    protocol_path = generation_root / f"{model_key}_format_rescue_protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if signature(old) != protocol_signature:
            raise ValueError(
                f"Rescue protocol changed. Preserve existing rescue outputs and use a new versioned filename: {protocol_path}")
    else:
        write_json(protocol_path, protocol)

    rescue_path = generation_root / f"{model_key}_format_rescue.jsonl"
    existing = _read_attempts(rescue_path)
    terminal = {logical_key(row) for row in existing
                if row.get("format_rescue_protocol_signature") == protocol_signature
                and row.get("status") in {"ok", "terminal_error"}}
    successful = {logical_key(row) for row in existing
                  if row.get("format_rescue_protocol_signature") == protocol_signature
                  and row.get("status") == "ok"}
    contexts = _expected_contexts(model_key, out)
    pending_all = [row for row in contexts if logical_key(row) not in base_success
                   and (retry_terminal or logical_key(row) not in terminal)]
    selected = _stratified_limit(pending_all, int(max_rows))

    lock_path = generation_root / f"{model_key}_format_rescue.lock"
    lock_stream = lock_path.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise RuntimeError(f"Another rescue writer is active for {model_key}") from exc

    model = processor = tokenizer = None
    try:
        if selected:
            model, processor, tokenizer, dtype = _load_model(spec)
        for number, row in enumerate(selected, 1):
            key = logical_key(row)
            base_row = base_latest.get(key)
            input_scope = "raw_sensor_plus_retrieval" if spec["mode"] == "multimodal" else "retrieval_text_only"
            prompt = build_prompt(row, input_scope)
            if base_row is not None and base_row.get("prompt") != prompt:
                raise ValueError(f"Original prompt mismatch for {key}; contexts or prompt code changed")
            allowed_sources = {str(anchor["anchor_id"]) for anchor in row["evidence"]}
            result = {
                "generation_key": signature(["format_rescue_v1", model_key, row["context_signature"], protocol_signature]),
                "model_key": model_key, "model_id": spec["model_id"],
                "resolved_revision": spec["revision"], "direction": row["direction"],
                "sample_id": row["sample_id"], "method": row["method"],
                "condition": row["condition"], "context_signature": row["context_signature"],
                "input_scope": input_scope, "prompt": prompt,
                "result_source": "format_rescue", "base_attempt_status": (
                    base_row.get("status") if base_row is not None else "not_attempted"),
                "base_generation_key": base_row.get("generation_key") if base_row else None,
                "base_generation_protocol_signature": base_row.get("generation_protocol_signature") if base_row else None,
                "generation_protocol_signature": protocol_signature,
                "format_rescue_protocol_signature": protocol_signature,
            }
            raw_hashes = []
            try:
                image_path = query_image(row, out) if spec["mode"] == "multimodal" else None
                image_hash = sha256(image_path) if image_path else None
                if base_row is not None and base_row.get("query_image_sha256") != image_hash:
                    raise ValueError(f"Query image hash differs from original attempt for {key}")
                raw, processor_tokens = _generate_original(
                    model, processor, tokenizer, dtype, spec, row, prompt, image_path,
                    int(base_protocol["max_new_tokens"]), int(base_protocol["max_input_tokens"]))
                raw_hashes.append(hashlib.sha256(raw.encode()).hexdigest())
                parsed, stage = parse_rescue_result(raw, allowed_sources)
                repair_tokens = []
                for attempt in range(int(repair_attempts)):
                    if parsed is not None:
                        break
                    repair = _repair_prompt(raw, allowed_sources)
                    raw, token_count = _generate_text_only(
                        model, processor, tokenizer, dtype, spec, repair,
                        int(protocol["repair_max_new_tokens"]))
                    repair_tokens.append(token_count)
                    raw_hashes.append(hashlib.sha256(raw.encode()).hexdigest())
                    parsed, parse_mode = parse_rescue_result(raw, allowed_sources)
                    if parsed is not None:
                        stage = f"repair_{attempt + 1}_{parse_mode}"
                if parsed is None:
                    raise ValueError("Formatting rescue exhausted without a valid schema")
                result.update(status="ok", generated_report=parsed["report"],
                              evidence_trace=parsed["evidence_trace"],
                              uncertainties=parsed["uncertainties"],
                              formatting_stage=stage,
                              processor_input_tokens=processor_tokens,
                              repair_input_tokens=repair_tokens,
                              query_image=str(image_path) if image_path else None,
                              query_image_sha256=image_hash,
                              response_sha256=raw_hashes[-1], error=None)
            except Exception as exc:
                result.update(status="terminal_error", generated_report="", evidence_trace=[],
                              uncertainties=[], formatting_stage=None,
                              response_sha256=raw_hashes[-1] if raw_hashes else None,
                              error=repr(exc))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            result["response_attempt_sha256"] = raw_hashes
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
            with rescue_path.open("a") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            print(model_key, number, "/", len(selected), *key, result["status"], flush=True)
    finally:
        if model is not None:
            del model, processor, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()

    all_rescue = _read_attempts(rescue_path)
    same_protocol = [row for row in all_rescue
                     if row.get("format_rescue_protocol_signature") == protocol_signature]
    rescue_success = {logical_key(row) for row in same_protocol if row.get("status") == "ok"}
    rescue_terminal = {logical_key(row) for row in same_protocol
                       if row.get("status") == "terminal_error" and logical_key(row) not in rescue_success}
    expected = {logical_key(row) for row in contexts}
    report = {
        "model_key": model_key, "expected": len(expected),
        "original_successful": len(base_success & expected),
        "rescue_successful": len(rescue_success - base_success),
        "combined_successful": len((base_success | rescue_success) & expected),
        "terminal_rescue_failures": len(rescue_terminal - base_success),
        "remaining_without_success": len(expected - base_success - rescue_success),
        "selected_this_run": len(selected), "pending_before_this_run": len(pending_all),
        "format_rescue_protocol_signature": protocol_signature,
    }
    write_json(generation_root / f"{model_key}_format_rescue_progress.json", report)
    return report
