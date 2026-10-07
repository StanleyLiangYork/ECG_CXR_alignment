"""Versioned generation storage and diagnostics; no model imports required."""
from collections import Counter
from functools import wraps
import fcntl
import inspect
import json
import os
from pathlib import Path
import re

from e4_common import EXPERIMENT4_ROOT, read_jsonl, sha256, signature, write_json


def generation_root(out=EXPERIMENT4_ROOT):
    run = os.environ.get("E4_GENERATION_RUN", "generation_v3")
    if run == "legacy":
        return Path(out) / "generations"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run):
        raise ValueError("E4_GENERATION_RUN must be a simple directory name")
    return Path(out) / "generations" / "runs" / run


def logical_key(row):
    return tuple(str(row[k]) for k in ("direction", "sample_id", "method", "condition"))


def analysis_root(out=EXPERIMENT4_ROOT):
    return Path(out) if os.environ.get('E4_GENERATION_RUN', 'generation_v3') == 'legacy' else generation_root(out)


def generation_lock(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        if os.environ.get("E4_GENERATION_RUN", "generation_v3") == "legacy":
            raise ValueError("Legacy outputs are read-only under the revised runner; choose a new run name")
        root = generation_root(bound.arguments["out"])
        root.mkdir(parents=True, exist_ok=True)
        # flock is released on process exit, including interrupted notebook kernels.
        with (root / (bound.arguments["model_key"] + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This model/run already has an active writer") from exc
            return function(*args, **kwargs)
    return wrapped


def prepare_run(model_key, out, contexts, settings, code_files):
    root = generation_root(out)
    root.mkdir(parents=True, exist_ok=True)
    contract = {"settings": settings, "contexts_sha256": sha256(Path(out) / "contexts" / "bidirectional_contexts.jsonl"),
                "code_sha256": {Path(p).name: sha256(Path(p)) for p in
                                [*code_files, Path(__file__)]}}
    manifest = root / f"{model_key}_run_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != json.loads(json.dumps(contract)):
            attempts_path = root / f"{model_key}.jsonl"
            protocol_path = root / f"{model_key}_protocol.json"
            # A validation failure can leave only a preflight manifest. With no
            # model attempt and no registered protocol, replacing that manifest
            # cannot mix experimental results.
            if (not attempts_path.exists() or not read_jsonl(attempts_path)) and not protocol_path.exists():
                write_json(manifest, contract)
            else:
                raise ValueError("Generation settings, code, or contexts changed. Choose a NEW E4_GENERATION_RUN; existing results were preserved.")
    else:
        if (root / f"{model_key}.jsonl").exists():
            raise ValueError("Unversioned results found in selected run; choose a new run name")
        write_json(manifest, contract)
    attempts = read_jsonl(root / f"{model_key}.jsonl")
    if attempts:
        protocol_path = root / f"{model_key}_protocol.json"
        if not protocol_path.exists():
            raise ValueError("Run has attempts but no protocol; refusing unsafe resume")
        expected = signature(json.loads(protocol_path.read_text()))
        if any(r.get('generation_protocol_signature') != expected for r in attempts):
            raise ValueError("Run contains mixed generation protocols; refusing unsafe resume")
    progress, pending = run_progress(model_key, out, contexts)
    return root, pending, progress


def run_progress(model_key, out, contexts):
    root = generation_root(out)
    current = {logical_key(r): r["context_signature"] for r in contexts}
    attempts = read_jsonl(root / f"{model_key}.jsonl")
    counts, successful, latest = Counter(), set(), {}
    for row in attempts:
        key = logical_key(row)
        if row.get("model_key") != model_key or row.get("context_signature") != current.get(key):
            continue
        counts[key] += 1
        latest[key] = row
        if row.get("status") == "ok" and str(row.get("generated_report", "")).strip():
            successful.add(key)
    # One deterministic attempt per cell. A changed protocol belongs in a new run.
    pending = [r for r in contexts if logical_key(r) not in counts]
    failed = set(counts) - successful
    report = {"model_key": model_key, "generation_root": str(root), "planned": len(current),
              "attempt_rows": len(attempts), "successful": len(successful), "failed": len(failed),
              "pending": len(pending), "duplicate_attempts": sum(n - 1 for n in counts.values()),
              "failure_categories": dict(Counter(latest[k].get("error_category", "unknown") for k in failed))}
    write_json(root / f"{model_key}_progress.json", report)
    return report, pending


def select_pilot(contexts, max_rows):
    """Round-robin cells by direction/condition/method instead of a prefix pilot."""
    if not max_rows:
        return contexts
    from collections import defaultdict, deque
    groups = defaultdict(deque)
    for row in contexts:
        groups[(row["direction"], row["condition"], row["method"])].append(row)
    selected = []
    while groups and len(selected) < max_rows:
        for key in list(sorted(groups)):
            selected.append(groups[key].popleft())
            if not groups[key]:
                del groups[key]
            if len(selected) == max_rows:
                break
    return selected


def advance_progress(root, progress, result):
    progress["attempt_rows"] += 1
    progress["pending"] -= 1
    progress["successful" if result["status"] == "ok" else "failed"] += 1
    if result["status"] != "ok":
        category = result.get("error_category", "unknown")
        progress["failure_categories"][category] = progress["failure_categories"].get(category, 0) + 1
    write_json(root / f"{result['model_key']}_progress.json", progress)


def pinned_revision(root, model_key):
    path = root / f"{model_key}_protocol.json"
    return json.loads(path.read_text())["resolved_revision"] if path.exists() else "main"


def register_protocol(root, model_key, protocol):
    path = root / f"{model_key}_protocol.json"
    if path.exists() and signature(json.loads(path.read_text())) != signature(protocol):
        raise ValueError("Model/runtime protocol changed; use a NEW E4_GENERATION_RUN")
    write_json(path, protocol)
    write_json(root / "protocols" / f"{signature(protocol)}.json", protocol)


def check_input_budget(input_tokens, max_input_tokens, max_new_tokens, config):
    limits = []
    for cfg in (config, getattr(config, "text_config", None)):
        value = getattr(cfg, "max_position_embeddings", None)
        if isinstance(value, int) and 0 < value < 10**9:
            limits.append(value)
    if input_tokens > max_input_tokens:
        raise ValueError(f"Expanded input {input_tokens} exceeds max_input_tokens={max_input_tokens}; no truncation performed")
    if limits and input_tokens + max_new_tokens > min(limits):
        raise ValueError(f"Input plus output reserve exceeds model context limit={min(limits)}")


def token_diagnostics(generated, max_new_tokens, eos_token_id):
    ids = generated.tolist() if hasattr(generated, "tolist") else list(generated)
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    eos = bool(ids and ids[-1] in eos_ids)
    reached = len(ids) >= max_new_tokens
    return {"generated_token_count": len(ids), "output_token_limit_reached": reached,
            "ended_with_eos": eos, "finish_reason": "eos" if eos else ("length" if reached else "other")}


def error_category(exc, stage=None):
    message = str(exc).lower()
    if "out of memory" in message:
        return "out_of_memory"
    if "output token limit" in message:
        return "output_budget"
    if "exceed" in message or "expanded input=" in message:
        return "input_budget"
    if stage:
        return "output_format"
    if isinstance(exc, (FileNotFoundError, OSError)):
        return "input_io"
    return "inference_or_preprocessing"
