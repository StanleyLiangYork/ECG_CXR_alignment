"""Shared configuration and provenance-safe I/O for Experiment 4."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = Path(os.environ.get(
    "ECG_CXR_BRIDGE_ARTIFACT_ROOT",
    str(REPOSITORY_ROOT / "artifacts"),
)).expanduser()
EXPERIMENT4_ROOT = Path(os.environ.get(
    "EXPERIMENT4_ROOT", str(ARTIFACT_ROOT / "experiment4")
)).expanduser()
SYMILE_ROOT = Path(os.environ.get(
    "SYMILE_ROOT",
    str(REPOSITORY_ROOT / "data" / "symile-mimic" / "1.0.0"),
)).expanduser()
HF_CACHE_ROOT = Path(os.environ.get(
    "HF_HOME", str(REPOSITORY_ROOT / ".cache" / "huggingface")
)).expanduser()

for name, path in {
    "ECG_CXR_BRIDGE_ARTIFACT_ROOT": ARTIFACT_ROOT,
    "EXPERIMENT4_ROOT": EXPERIMENT4_ROOT,
    "SYMILE_ROOT": SYMILE_ROOT,
    "HF_HOME": HF_CACHE_ROOT,
}.items():
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute, got {path}")
if EXPERIMENT4_ROOT == ARTIFACT_ROOT:
    raise ValueError("EXPERIMENT4_ROOT must be a dedicated child directory")

EXPERIMENT4_ROOT.mkdir(parents=True, exist_ok=True)
HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE_ROOT / "xet"))
os.environ.setdefault("MPLCONFIGDIR", str(EXPERIMENT4_ROOT / "cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

VERSION = "experiment4_v2_reviewed"
SPLITS = ("train", "val", "test")
MODALITIES = ("ecg", "cxr", "labs")


def require(path, purpose="input"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
    return path


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def uid(value):
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}:
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip() or None


def positive_chexpert(value):
    """Only asserted positive labels; -1 is uncertain, 0 negative, null unknown."""
    if isinstance(value, str):
        try: value = json.loads(value)
        except json.JSONDecodeError: value = [s.strip() for s in value.split(',') if s.strip()]
    if isinstance(value, dict):
        return sorted(str(k) for k, v in value.items() if str(v).lower() in {'1', '1.0', 'true', 'positive'})
    return sorted(set(map(str, value or [])))


def patient_split(subject_id, seed="experiment4-global-patient-split"):
    value = uid(subject_id)
    if value is None:
        return "quarantine"
    bucket = int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:12], 16) % 100
    return "train" if bucket < 80 else ("val" if bucket < 90 else "test")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def atomic_numpy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)


def environment_report():
    report = {
        "version": VERSION,
        "artifact_root": str(ARTIFACT_ROOT),
        "experiment4_root": str(EXPERIMENT4_ROOT),
        "symile_root": str(SYMILE_ROOT),
        "hf_cache": str(HF_CACHE_ROOT),
        "inputs": {},
    }
    required = {
        "config": ARTIFACT_ROOT / "config.json",
        "external_ecg_manifest": ARTIFACT_ROOT / "external" / "ecg" / "manifest.jsonl",
        "external_cxr_manifest": ARTIFACT_ROOT / "external" / "cxr" / "manifest.jsonl",
        "symile_train_manifest": ARTIFACT_ROOT / "manifests" / "train.jsonl",
        "symile_train_ecg_features": ARTIFACT_ROOT / "features" / "train" / "ecg_hubert.npy",
        "symile_train_cxr_features": ARTIFACT_ROOT / "features" / "train" / "cxr_raddino.npy",
    }
    for key, path in required.items():
        require(path, key)
        report["inputs"][key] = {"path": str(path), "sha256": sha256(path)}
    write_json(EXPERIMENT4_ROOT / "environment_report.json", report)
    return report
