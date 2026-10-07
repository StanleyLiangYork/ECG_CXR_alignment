"""Global patient partition with Symile priority and linked-evidence enrichment."""
from __future__ import annotations

import collections
import csv
import json
import os
import re
from pathlib import Path

from e4_common import (ARTIFACT_ROOT, EXPERIMENT4_ROOT, SPLITS, patient_split,
                       read_jsonl, signature, uid, write_json, write_jsonl, sha256, positive_chexpert)


def _json_identifier(meta, aliases):
    """Read an identifier from JSON without depending on key capitalization."""
    normalized = {re.sub(r'[^a-z0-9]', '', str(k).lower()): v for k, v in meta.items()}
    return next((uid(normalized.get(re.sub(r'[^a-z0-9]', '', key.lower())))
                 for key in aliases
                 if uid(normalized.get(re.sub(r'[^a-z0-9]', '', key.lower()))) is not None), None)


def _flat_cxr_json_identity(row):
    """Verify the identity of a flat image/JSON CXR pair.

    The multi_kg export does not retain native p########/s######## directories,
    but its JSON contains the native MIMIC subject and study identifiers.  We
    accept that representation only when JSON and manifest identifiers agree
    and the image/JSON filenames form the expected pair.
    """
    json_value, image_value = row.get('json_path'), row.get('image_path')
    if not json_value or not image_value:
        return None
    json_path, image_path = Path(str(json_value)), Path(str(image_value))
    if not json_path.is_file():
        return None
    try:
        meta = json.loads(json_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    subject = _json_identifier(meta, ('subject_id', 'patient_id', 'patientId'))
    study = _json_identifier(meta, ('study_id', 'dicom_study_id', 'studyId'))
    if subject is None or study is None:
        return None
    manifest_subject, manifest_study = uid(row.get('subject_id')), uid(row.get('study_id'))
    if manifest_subject is not None and manifest_subject != subject:
        raise ValueError(f'CXR manifest/JSON subject conflict: {row.get("sample_id")}')
    if manifest_study is not None and manifest_study != study:
        raise ValueError(f'CXR manifest/JSON study conflict: {row.get("sample_id")}')
    if image_path.stem != json_path.stem:
        raise ValueError(f'CXR image/JSON filename mismatch: {row.get("sample_id")}')
    # The current multi_kg image+JSON export names each pair by study ID.  If a
    # future export uses DICOM IDs instead, require an explicit crosswalk rather
    # than silently treating the namespace as equivalent.
    if uid(json_path.stem) != study:
        raise ValueError(f'CXR filename/JSON study conflict: {row.get("sample_id")}')
    return subject, study, 'native MIMIC subject/study IDs verified in paired CXR JSON'


def _canonicalize(rows, modality, crosswalk):
    valid, rejected = [], []
    for row in rows:
        key = (modality, str(row.get('sample_id')))
        mapping = crosswalk.get(key)
        path = str(row.get('waveform_path') if modality == 'ecg' else row.get('image_path'))
        subjects = re.findall(r'(?:^|/)p(\d{8})(?=/|$)', path)
        studies = re.findall(r'(?:^|/)s(\d+)(?=/|$)', path)
        if mapping:
            subject, study = uid(mapping.get('subject_id')), uid(mapping.get('study_id'))
            proof = mapping.get('provenance', '').strip()
        elif len(subjects) == len(studies) == 1:
            subject, study, proof = uid(subjects[0]), uid(studies[0]), 'native MIMIC patient/study path'
            if any(uid(row.get(k)) not in {None, v} for k, v in [('subject_id', subject), ('study_id', study)]):
                raise ValueError(f'{modality} manifest IDs contradict native path: {row.get("sample_id")}')
        elif modality == 'cxr' and (json_identity := _flat_cxr_json_identity(row)):
            subject, study, proof = json_identity
        else:
            rejected.append({**row, 'global_split': 'quarantine',
                             'partition_reason': 'unverified_patient_namespace',
                             'modality': modality})
            continue
        if not subject or not study or not proof:
            raise ValueError(f'Incomplete identity crosswalk for {key}')
        valid.append({**row, 'original_subject_id': row.get('subject_id'),
                      'original_study_id': row.get('study_id'), 'subject_id': int(subject),
                      'study_id': int(study), 'namespace': 'mimic_native', 'identity_provenance': proof})
    return valid, rejected


def _cxr_payload(row):
    path = Path(row['json_path'])
    meta = {str(k).lower(): v for k, v in json.loads(path.read_text()).items()}
    def text(v):
        if isinstance(v, str): return ' '.join(v.split())
        if isinstance(v, list): return ' '.join(filter(None, map(text, v)))
        if isinstance(v, dict): return ' '.join(text(v[k]) for k in v if k.lower() in {'findings','impression','text','notes'})
        return ''
    report = next((text(meta[k]) for k in ['caption','report_text','report','notes','clinical_notes','text']
                   if k in meta and text(meta[k])), '')
    if not report: report = ' '.join(filter(None, [text(meta.get('findings')),text(meta.get('impression'))]))
    raw_labels = meta.get('labels', meta.get('chexpert_labels'))
    return {**row, 'report_text': report, 'labels': positive_chexpert(raw_labels), 'raw_labels': raw_labels,
            'labels_available': raw_labels is not None, 'json_sha256': sha256(path)}


def _dedupe(rows, keys):
    kept, rejected, seen = [], [], set()
    owners = {}
    # Prefer complete frontal CXR records when a study has several images.
    rows = sorted(rows, key=lambda r: (not bool(r.get('report_text') or r.get('machine_diagnosis')),
                  str(r.get('view_position', '')).upper() not in {'AP','PA'}, str(r.get('sample_id'))))
    for row in rows:
        identity = next((f"{key}:{uid(row.get(key))}" for key in keys if uid(row.get(key))), None)
        if identity is None:
            identity = "row:" + signature(row)
        if identity in seen:
            if owners[identity] != uid(row.get('subject_id')):
                raise ValueError(f'Study has conflicting canonical patient identities: {identity}')
            rejected.append({"identity": identity, "row": row, "reason": "duplicate_record"})
        else:
            seen.add(identity)
            owners[identity] = uid(row.get('subject_id'))
            kept.append(row)
    return kept, rejected


def build_global_partition(artifact=ARTIFACT_ROOT, out=EXPERIMENT4_ROOT):
    out = Path(out)
    partition_root = out / "partition"
    write_json(partition_root / 'global_partition_audit.json', {'status': 'preparing'})
    symile = {split: read_jsonl(artifact / "manifests" / f"{split}.jsonl") for split in SPLITS}
    if any(not rows for rows in symile.values()):
        raise ValueError('All three Symile manifests are required; an empty manifest is not a valid partition.')
    if any(uid(r.get('subject_id')) is None for rows in symile.values() for r in rows):
        raise ValueError('Symile contains a missing patient ID.')
    symile_subjects = {split: {uid(r.get("subject_id")) for r in rows} - {None}
                       for split, rows in symile.items()}
    for left in SPLITS:
        for right in SPLITS:
            if left < right and symile_subjects[left] & symile_subjects[right]:
                raise ValueError(f"Symile {left}/{right} patient overlap")
    protected = set().union(*symile_subjects.values())

    ecg_raw = [{**row, "source_index": i} for i, row in enumerate(
        read_jsonl(artifact / "external" / "ecg" / "manifest.jsonl"))]
    cxr_raw = [{**row, "source_index": i} for i, row in enumerate(
        read_jsonl(artifact / "external" / "cxr" / "manifest.jsonl"))]
    if not ecg_raw or not cxr_raw: raise ValueError('Both external source manifests are required.')
    crosswalk_path = os.environ.get('E4_IDENTITY_CROSSWALK')
    crosswalk = {}
    if crosswalk_path:
        with open(crosswalk_path, newline='') as stream:
            for row in csv.DictReader(stream):
                key = (row['modality'], row['sample_id'])
                if key in crosswalk: raise ValueError(f'Duplicate identity crosswalk key: {key}')
                crosswalk[key] = row
    ecg_verified, ecg_unverified = _canonicalize(ecg_raw, 'ecg', crosswalk)
    cxr_verified, cxr_unverified = _canonicalize(cxr_raw, 'cxr', crosswalk)
    cxr_verified = [_cxr_payload(r) for r in cxr_verified]
    ecg, ecg_dup = _dedupe(ecg_verified, ["study_id", "sample_id", "waveform_path"])
    cxr, cxr_dup = _dedupe(cxr_verified, ["study_id", "sample_id", "image_path"])

    ecg_by_study = {uid(r.get("study_id")): r for r in ecg if uid(r.get("study_id"))}
    cxr_by_study = {uid(r.get("study_id")): r for r in cxr if uid(r.get("study_id"))}
    linked, coverage = [], collections.Counter()
    for split, rows in symile.items():
        for row in rows:
            erow = ecg_by_study.get(uid(row.get("ecg_study_id")))
            crow = cxr_by_study.get(uid(row.get("cxr_study_id")))
            for candidate in [erow, crow]:
                if candidate and uid(candidate['subject_id']) != uid(row['subject_id']):
                    raise ValueError(f'Symile study join conflicts with canonical patient for admission {row["hadm_id"]}')
            if erow: coverage[f"{split}_ecg_diagnosis"] += 1
            if crow: coverage[f"{split}_cxr_report"] += 1
            if erow and str(erow.get('machine_diagnosis', '')).strip():
                coverage[f"{split}_ecg_text"] += 1
            if crow and str(crow.get('report_text', '')).strip():
                coverage[f"{split}_cxr_text"] += 1
            if (erow and crow and str(erow.get('machine_diagnosis', '')).strip()
                    and str(crow.get('report_text', '')).strip()):
                coverage[f"{split}_joint_text"] += 1
            linked.append({
                **row,
                "sample_id": f"symile:{int(row['hadm_id'])}",
                "split": split,
                "namespace": "mimic_native",
                "ecg_machine_diagnosis": (erow or {}).get("machine_diagnosis", ""),
                "ecg_source_sample_id": (erow or {}).get("sample_id"),
                "cxr_report": (crow or {}).get("report_text", ""),
                "cxr_labels": (crow or {}).get("labels", []),
                "cxr_source_sample_id": (crow or {}).get("sample_id"),
                "cxr_json_path": (crow or {}).get("json_path"),
                "cxr_json_sha256": (crow or {}).get("json_sha256"),
                "linkage": "canonical_patient_and_study_id_within_symile_encounter",
            })

    def partition_external(rows, modality):
        result = {split: [] for split in (*SPLITS, "quarantine")}
        removed = []
        for row in rows:
            subject = uid(row.get("subject_id"))
            if subject is None:
                result["quarantine"].append({**row, "global_split": "quarantine",
                                             "partition_reason": "missing_subject_id"})
            elif subject in protected:
                removed.append({"sample_id": row.get("sample_id"), "subject_id": subject,
                                "reason": "reserved_for_symile", "modality": modality})
            else:
                split = patient_split(subject)
                result[split].append({**row, "global_split": split,
                                      "original_split": row.get("split"),
                                      "partition_reason": "global_subject_hash"})
        return result, removed

    ecg_parts, ecg_removed = partition_external(ecg, "ecg")
    cxr_parts, cxr_removed = partition_external(cxr, "cxr")
    ecg_parts['quarantine'].extend(ecg_unverified)
    cxr_parts['quarantine'].extend(cxr_unverified)
    for modality, parts in [("ecg", ecg_parts), ("cxr", cxr_parts)]:
        for split, rows in parts.items():
            write_jsonl(partition_root / "external" / modality / f"{split}.jsonl", rows)
    for split in SPLITS:
        write_jsonl(partition_root / "symile" / f"{split}.jsonl",
                    [r for r in linked if r["split"] == split])
    write_jsonl(partition_root / "removed_external_overlap.jsonl", ecg_removed + cxr_removed)
    write_jsonl(partition_root / "duplicate_records.jsonl", ecg_dup + cxr_dup)

    audit = {
        "status": "complete",
        "policy": "Symile priority; exact study-linked text retained only inside Symile; all Symile patients removed from external cohorts; remaining external modalities share one subject-hash split.",
        "symile_patients": {k: len(v) for k, v in symile_subjects.items()},
        "external_input_rows": {"ecg": len(ecg_raw), "cxr": len(cxr_raw)},
        "duplicates_removed": {"ecg": len(ecg_dup), "cxr": len(cxr_dup)},
        "symile_overlap_removed": {"ecg": len(ecg_removed), "cxr": len(cxr_removed)},
        "external_rows": {
            m: {s: len(v) for s, v in parts.items()}
            for m, parts in [("ecg", ecg_parts), ("cxr", cxr_parts)]
        },
        "linked_symile_text_coverage": dict(coverage),
        "unverified_namespace_quarantined": {'ecg': len(ecg_unverified), 'cxr': len(cxr_unverified)},
        "identity_crosswalk_sha256": sha256(crosswalk_path) if crosswalk_path else None,
        "input_hashes": {str(p): sha256(p) for p in [
            *[artifact / 'manifests' / f'{s}.jsonl' for s in SPLITS],
            artifact / 'external' / 'ecg' / 'manifest.jsonl', artifact / 'external' / 'cxr' / 'manifest.jsonl']},
    }
    for split in SPLITS:
        left = {uid(r.get("subject_id")) for r in ecg_parts[split]}
        right = {uid(r.get("subject_id")) for r in cxr_parts[split]}
        audit.setdefault("cross_modal_subjects_by_split", {})[split] = len((left & right) - {None})
    global_sets = {}
    for split in SPLITS:
        global_sets[split] = ({uid(r.get("subject_id")) for r in ecg_parts[split]} |
                              {uid(r.get("subject_id")) for r in cxr_parts[split]}) - {None}
        if global_sets[split] & protected:
            raise AssertionError(f"External {split} still overlaps protected Symile subjects")
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if global_sets[left] & global_sets[right]:
                raise AssertionError(f"Global external patient leakage: {left}/{right}")
    audit["patient_disjoint_verified"] = True
    audit['verification_scope'] = 'Retained records with canonical identities only; quarantined records excluded.'
    audit['retained_manifests_signature'] = signature([linked, ecg_parts, cxr_parts])
    audit["partition_signature"] = signature(audit)
    write_json(partition_root / "global_partition_audit.json", audit)
    return audit
