"""Build patient-disjoint evidence banks and bidirectional ablation contexts."""
from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch

from e4_bridge import encode, load_model, load_split
from e4_common import (ARTIFACT_ROOT, EXPERIMENT4_ROOT, SPLITS, SYMILE_ROOT, atomic_numpy,
                       read_jsonl, sha256, signature, uid, write_json, write_jsonl)

LAB_TABLE_PATH = Path(__file__).with_name('symile_lab_code_table.json')
LAB_TABLE = json.loads(LAB_TABLE_PATH.read_text())
LAB_IDS = [r['itemid'] for r in LAB_TABLE['columns']]
if len(LAB_IDS) != 50 or len(set(LAB_IDS)) != 50 or LAB_IDS != sorted(LAB_IDS):
    raise ValueError('Lab decoding table must contain 50 unique sorted Symile item IDs')
for j, item in enumerate(LAB_TABLE['columns']):
    if item['percentile_index'] != j or item['observed_mask_index'] != 50+j or item['percentile_key'] != item['itemid']+'_percentile':
        raise ValueError('Invalid lab decoding table column indices')
LAB_NAMES = {r['itemid']: r['display_name'] for r in LAB_TABLE['columns']}
LAB_SOURCE_NAMES = {r['itemid']: r['source_name'] for r in LAB_TABLE['columns']}
OPAQUE_LAB_IDS = {r['itemid'] for r in LAB_TABLE['columns'] if r['opaque_source_label']}


def load_lab_code_table(out=EXPERIMENT4_ROOT):
    """Prefer the official MIMIC code table; retain a versioned fallback for portability."""
    path_value = os.environ.get("MIMIC_D_LABITEMS", "")
    provenance = f"{LAB_TABLE['source_repository']}@{LAB_TABLE['source_commit']}:{LAB_TABLE['source_path']}"
    table_hash = None
    names = dict(LAB_NAMES)
    if path_value:
        path = Path(path_value)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_id = {uid(r.get("itemid")): (r.get("label") or r.get("name") or "").strip()
                 for r in rows}
        for item in LAB_IDS:
            if by_id.get(item): names[item] = by_id[item]
        provenance = str(path)
        table_hash = sha256(path)
    report = {"itemids": LAB_IDS, "names": names, "provenance": provenance,
              "upstream_mapping": LAB_TABLE, "source_names": LAB_SOURCE_NAMES,
              "decoding_table_sha256": sha256(LAB_TABLE_PATH), "d_labitems_sha256": table_hash,
              "semantics": "Symile training-ECDF percentiles; not raw values or clinical reference ranges"}
    write_json(out / "evidence" / "lab_code_table.json", report)
    return report


def lab_summary(vector, names=None, lower=.10, upper=.90, maximum=12):
    names = names or LAB_NAMES
    x = np.asarray(vector)
    if x.shape != (100,): raise ValueError(f"Expected a 100-value lab vector, got {x.shape}")
    if not np.isin(x[50:], [0, 1]).all(): raise ValueError('Lab mask must be binary: 1 observed, 0 missing')
    observed_values = x[:50][x[50:].astype(bool)]
    if not np.isfinite(observed_values).all() or np.any((observed_values < 0) | (observed_values > 1)):
        raise ValueError('Observed lab percentiles must be finite in [0,1]')
    if not 0 <= lower < upper <= 1 or not isinstance(maximum, int) or maximum < 1:
        raise ValueError('Invalid salience thresholds or maximum')
    rows, observed_labs, missing_itemids = [], [], []
    for j, item in enumerate(LAB_IDS):
        observed = bool(x[50 + j])
        percentile = float(x[j]) if observed else None
        if not observed:
            missing_itemids.append(item)
            continue
        is_low = percentile <= lower or np.isclose(percentile, lower, rtol=0, atol=1e-7)
        is_high = percentile >= upper or np.isclose(percentile, upper, rtol=0, atol=1e-7)
        direction = ('low_distributional_percentile' if is_low else
                     'high_distributional_percentile' if is_high else 'within_distributional_band')
        entry = {"itemid": item, "name": names[item], "source_name": LAB_SOURCE_NAMES[item],
                 "training_ecdf_percentile": percentile, "observed": True,
                 "direction": direction, "raw_value": None, "unit": None,
                 "clinical_abnormality": "not_determinable_from_percentile_alone",
                 "opaque_source_label": item in OPAQUE_LAB_IDS}
        observed_labs.append(entry)
        if direction != 'within_distributional_band': rows.append(entry)
    rows.sort(key=lambda r: abs(r["training_ecdf_percentile"] - .5), reverse=True)
    rows = rows[:maximum]
    if rows:
        text = "; ".join(
            f"{r['name']}" + (' [unexpanded source label]' if r['opaque_source_label'] and r['name'] == r['source_name'] else '') +
            f" ({r['itemid']}): {100*r['training_ecdf_percentile']:.1f}th "
            f"training-distribution percentile" for r in rows)
    else:
        text = (f"No observed laboratory result met the prespecified <= {100*lower:g}th or >= {100*upper:g}th "
                "training-percentile thresholds." if observed_labs else
                "No laboratory tests were observed; imputed slots are not patient measurements.")
    prefix = ("Other-patient laboratory evidence. Percentiles are relative to the Symile training ECDF, "
              "not clinical abnormality thresholds or raw concentrations. ")
    return {"observations": rows, "text": prefix + text,
            "observed_labs": observed_labs, "missing_itemids": missing_itemids,
            "observed_count": len(observed_labs), "salient_count": len(rows),
            "salient_count_before_limit": sum(r['direction'] != 'within_distributional_band' for r in observed_labs),
            "thresholds": {"lower": lower, "upper": upper, "maximum": maximum},
            "source_commit": LAB_TABLE['source_commit']}


def audit_lab_encoding(symile=SYMILE_ROOT, out=EXPERIMENT4_ROOT):
    """Read-only verification of raw Symile CSV/NPY lab encoding on Biowulf."""
    import pandas as pd
    symile, out = Path(symile), Path(out)
    report = {'status': 'checking', 'source_commit': LAB_TABLE['source_commit'], 'splits': {}}
    destination = out / 'evidence' / 'lab_encoding_audit.json'
    write_json(destination, report)
    keys = [item + '_percentile' for item in LAB_IDS]
    try:
        train = pd.read_csv(symile / 'train.csv')
        if not set(keys + LAB_IDS) <= set(train):
            raise ValueError('Full official split CSVs with 50 raw lab and 50 percentile columns are required for verification')
        train_means = train[keys].mean().to_numpy(dtype=float)
        if not np.isfinite(train_means).all(): raise ValueError('A training lab has no finite mean percentile')
        means_path = symile / 'labs_means.json'
        if means_path.is_file():
            means = json.loads(means_path.read_text())
            if sorted(means) != keys: raise ValueError('labs_means.json does not match the pinned 50-column schema')
            if not np.allclose([means[k] for k in keys], train_means, rtol=1e-5, atol=1e-6):
                raise ValueError('Stored training mean percentiles differ from train.csv')
            report['labs_means_sha256'] = sha256(means_path)
        for split in SPLITS:
            csv_path = symile / f'{split}.csv'
            frame = train if split == 'train' else pd.read_csv(csv_path)
            if not set(keys + LAB_IDS + ['hadm_id']) <= set(frame):
                raise ValueError(f'{split}: missing source CSV columns; encoded arrays alone cannot establish their provenance')
            root = symile / 'data_npy' / split
            pp, mp, hp = [root / f'{prefix}_{split}.npy' for prefix in ('labs_percentiles', 'labs_missingness', 'hadm_id')]
            p, m, hadm = np.load(pp), np.load(mp), np.load(hp).reshape(-1)
            if p.shape != (len(frame), 50) or m.shape != p.shape:
                raise ValueError(f'{split}: lab tensor shape mismatch')
            if not np.array_equal(hadm, frame.hadm_id.to_numpy()):
                raise ValueError(f'{split}: admission identity/order mismatch')
            observed = frame[LAB_IDS].notna().to_numpy()
            if not np.isin(m, [0,1]).all() or not np.array_equal(m, observed.astype(int)):
                raise ValueError(f'{split}: observation mask is inverted, reordered, or inconsistent with CSV')
            expected = np.where(observed, frame[keys].to_numpy(dtype=float), train_means[None,:])
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError(f'{split}: invalid percentile values')
            if not np.allclose(p, expected, rtol=1e-5, atol=1e-6):
                raise ValueError(f'{split}: values, column order, or missing-value imputation disagree with official preprocessing')
            report['splits'][split] = {'rows': len(frame), 'observed_values': int(observed.sum()),
                'missing_values': int((~observed).sum()), 'input_hashes': {str(f): sha256(f) for f in (csv_path,pp,mp,hp)}}
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        write_json(destination, report)
        raise
    write_json(destination, report)
    return report


def prepare_external_features(artifact=ARTIFACT_ROOT, out=EXPERIMENT4_ROOT):
    audit = {"status": "preparing", "modalities": {}}
    for modality, feature_name in [("ecg", "ecg_foundation.npy"), ("cxr", "cxr_foundation.npy")]:
        source = artifact / "external" / modality
        features = np.load(source / feature_name, mmap_mode="r")
        status = np.load(source / "encoding_status.npy", mmap_mode="r")
        source_manifest = read_jsonl(source / "manifest.jsonl")
        if len(source_manifest) != len(features) or len(status) != len(features):
            raise ValueError(f"{modality}: source manifest, foundation features, and status are misaligned")
        audit["modalities"][modality] = {'input_hashes': {str(p): sha256(p) for p in
            [source / 'manifest.jsonl', source / feature_name, source / 'encoding_status.npy']}}
        for split in SPLITS:
            rows = read_jsonl(out / "partition" / "external" / modality / f"{split}.jsonl")
            kept, indexes, rejected = [], [], []
            for row in rows:
                i = int(row["source_index"])
                if i < 0 or i >= len(features) or i >= len(status):
                    raise IndexError(f"{modality} source_index outside foundation array: {i}")
                if uid(source_manifest[i].get("sample_id")) != uid(row.get("sample_id")):
                    raise ValueError(f"{modality}: foundation manifest order has changed")
                if int(status[i]) != 1 or not np.isfinite(features[i]).all():
                    rejected.append({"sample_id": row.get("sample_id"), "source_index": i,
                                     "reason": "foundation_encoding_unavailable"})
                    continue
                kept.append(row); indexes.append(i)
            dest = out / "external" / modality / split
            write_jsonl(dest / "manifest.jsonl", kept)
            write_jsonl(dest / "rejected.jsonl", rejected)
            atomic_numpy(dest / "foundation.npy", np.asarray(features[indexes], dtype=np.float32))
            audit["modalities"][modality][split] = {"partition_rows": len(rows),
                                                     "encoded_rows": len(kept),
                                                     "rejected_rows": len(rejected)}
    audit["status"] = "complete"
    audit["signature"] = signature(audit)
    write_json(out / "external" / "feature_audit.json", audit)
    return audit


def _save_bank(root, records, vectors):
    write_jsonl(root / "manifest.jsonl", records)
    for modality, values in vectors.items(): atomic_numpy(root / f"{modality}.npy", values)


def build_banks(out=EXPERIMENT4_ROOT, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    selected = json.loads((out / "bridge" / "selected_bridges.json").read_text())
    lab_codes = load_lab_code_table(out)
    train_records, train_arrays0 = load_split(out, "train")
    train_arrays = {m: np.asarray(train_arrays0[m]) for m in ("ecg", "cxr", "labs")}

    # Bridge training uses only frozen ECG/CXR/lab arrays.  Linked report text is
    # metadata added by the global partition step.  Read the current enriched
    # partition here so a corrected text join does not require retraining an
    # otherwise identical sensor bridge.  Exact encounter identity is checked
    # before any fields are overlaid.
    linked_path = out / "partition" / "symile" / "train.jsonl"
    linked_records = read_jsonl(linked_path)
    linked_by_hadm = {uid(r.get("hadm_id")): r for r in linked_records}
    if len(linked_by_hadm) != len(linked_records):
        raise ValueError("Duplicate admission IDs in the enriched Symile training partition")
    enriched_records = []
    identity_fields = ("subject_id", "hadm_id", "ecg_study_id", "cxr_study_id")
    evidence_fields = ("ecg_machine_diagnosis", "ecg_source_sample_id", "cxr_report",
                       "cxr_labels", "cxr_source_sample_id", "cxr_json_path",
                       "cxr_json_sha256", "linkage")
    for record in train_records:
        linked = linked_by_hadm.get(uid(record.get("hadm_id")))
        if linked is None:
            raise ValueError(f'Missing enriched Symile admission: {record.get("hadm_id")}')
        if any(uid(record.get(key)) != uid(linked.get(key)) for key in identity_fields):
            raise ValueError(f'Prepared/enriched Symile identity mismatch: {record.get("hadm_id")}')
        enriched_records.append({**record, **{key: linked.get(key) for key in evidence_fields}})
    train_records = enriched_records

    eligible = [i for i, r in enumerate(train_records)
                if str(r.get("ecg_machine_diagnosis", "")).strip()
                and str(r.get("cxr_report", "")).strip()]
    if not eligible:
        ecg_count = sum(bool(str(r.get("ecg_machine_diagnosis", "")).strip()) for r in train_records)
        cxr_count = sum(bool(str(r.get("cxr_report", "")).strip()) for r in train_records)
        raise ValueError(
            "No Symile training encounter has both linked ECG diagnosis and CXR evidence. "
            f"Coverage after the exact-study join: ECG={ecg_count}, CXR={cxr_count}, joint=0. "
            "Rerun updated notebooks 01 and 02, then restart and rerun notebook 08."
        )
    evidence_records = []
    for i in eligible:
        summary = lab_summary(train_arrays["labs"][i], lab_codes["names"])
        evidence_records.append({**train_records[i], "lab_summary": summary,
                                 "evidence_linkage": "same Symile admission; exact ECG/CXR study IDs"})
    controls = {}
    for method, genuine in selected.items():
        controls[method] = {"genuine": genuine, **genuine.get("controls", {})}
    audit = {"status": "preparing", "eligible_symile_encounters": len(eligible),
             "linked_text_coverage": {
                 "ecg": sum(bool(str(r.get("ecg_machine_diagnosis", "")).strip()) for r in train_records),
                 "cxr": sum(bool(str(r.get("cxr_report", "")).strip()) for r in train_records),
                 "joint": len(eligible),
             },
             "enrichment_source": str(linked_path),
             "enrichment_sha256": sha256(linked_path),
             "identity_join_fields": list(identity_fields),
             "banks": {}}
    for method, variants in controls.items():
        audit["banks"][method] = {}
        for variant, run in variants.items():
            model, protocol = load_model(run["checkpoint"], device=device)
            arrays = {m: np.array(train_arrays[m], copy=True) for m in train_arrays}
            if protocol["labs_mode"] == "mask_only": arrays["labs"][:, :50] = .5
            z = encode(model, arrays, device=device)
            bank_root = out / "evidence" / "banks" / method / variant / "symile"
            _save_bank(bank_root, evidence_records,
                       {m: np.asarray(z[m][eligible], dtype=np.float32) for m in z})
            audit["banks"][method][variant] = {"checkpoint": run["checkpoint"],
                                                "records": len(evidence_records)}
            for query_modality in ("ecg", "cxr"):
                for split in ("val", "test"):
                    root = out / "external" / query_modality / split
                    records = read_jsonl(root / "manifest.jsonl")
                    features = np.load(root / "foundation.npy", mmap_mode="r")
                    qz = encode(model, {query_modality: features}, device=device)[query_modality]
                    qroot = out / "evidence" / "queries" / method / variant / query_modality / split
                    _save_bank(qroot, records, {query_modality: qz.astype(np.float32)})
    audit["status"] = "complete"; audit["signature"] = signature(audit)
    write_json(out / "evidence" / "bank_audit.json", audit)
    return audit


def _trimodal_bank(root):
    # Equal-weight cosine scores against all three linked admission representations.
    return sum(np.load(root / f"{m}.npy", mmap_mode="r") for m in ("ecg", "cxr", "labs")) / 3


def _rank(query, bank, k, forbidden_subject=None, records=None):
    values = bank @ query
    order = np.argsort(-values, kind="stable")
    if forbidden_subject is not None and records is None:
        raise ValueError("Patient exclusion requires candidate records")
    eligible = [i for i in order if records is None or uid(records[i]["subject_id"]) != uid(forbidden_subject)]
    if len(eligible) < max(k, 2): raise ValueError("Insufficient evidence candidates")
    top = eligible[:k]
    margin = float(values[eligible[0]] - values[eligible[1]])
    return top, [float(values[i]) for i in top], margin


def calibrate_gates(out=EXPERIMENT4_ROOT):
    from e4_evaluation import ECG_RULES, present, cxr_reference
    selected = json.loads((out / "bridge" / "selected_bridges.json").read_text())
    gates = {}
    for method in selected:
        root = out / "evidence" / "banks" / method / "genuine" / "symile"
        records, bank = read_jsonl(root / "manifest.jsonl"), _trimodal_bank(root)
        gates[method] = {}
        for direction in ("ecg", "cxr"):
            qroot = out / "evidence" / "queries" / method / "genuine" / direction / "val"
            queries = read_jsonl(qroot / "manifest.jsonl")
            vectors = np.load(qroot / f"{direction}.npy")
            margins, truth = [], []
            for query, vector in zip(queries, vectors):
                qlabels = (present(query.get("machine_diagnosis", ""), ECG_RULES)
                           if direction == "ecg" else cxr_reference(query))
                if not qlabels: continue
                top, _, margin = _rank(vector, bank, 1, query["subject_id"], records)
                anchor = records[top[0]]
                labels = (present(anchor.get("ecg_machine_diagnosis", ""), ECG_RULES)
                          if direction == "ecg" else set(anchor.get("cxr_labels", [])))
                margins.append(margin); truth.append(bool(qlabels & labels))
            best = None
            if len(truth) >= 20 and any(truth) and not all(truth):
                margins, truth = np.asarray(margins), np.asarray(truth)
                for threshold in np.unique(np.quantile(margins, np.linspace(0, 1, 101))):
                    predicted = margins >= threshold
                    tp = int(np.sum(predicted & truth))
                    f1 = 2 * tp / max(int(predicted.sum() + truth.sum()), 1)
                    item = {"threshold": float(threshold), "f1": f1,
                            "coverage": float(predicted.mean()), "n_validation": len(truth),
                            "calibration_task": "external_val_to_symile_train_mean_three_cosines",
                            "target": "any shared positive machine label; NOT clinical confidence"}
                    if item["coverage"] >= .1 and (best is None or f1 > best["f1"]): best = item
            gates[method][direction] = best
    write_json(out / "bridge" / "confidence_gates.json", gates)
    return gates


def _anchor(record, score, rank, query_modality):
    return {"anchor_id": record["sample_id"], "subject_id": record["subject_id"],
            "hadm_id": record["hadm_id"], "rank": rank, "score": score,
            "linkage": record["evidence_linkage"],
            "ecg": {"machine_diagnosis": record.get("ecg_machine_diagnosis", ""),
                    "study_id": record.get("ecg_study_id")},
            "cxr": {"report": record.get("cxr_report", ""), "labels": record.get("cxr_labels", []),
                    "study_id": record.get("cxr_study_id")},
            "labs": record["lab_summary"], "query_modality": query_modality}


def _record_text_length(record):
    return (len(str(record.get("ecg_machine_diagnosis", ""))) +
            len(str(record.get("cxr_report", ""))) +
            len(json.dumps(record.get("cxr_labels", []))) +
            len(str(record.get("lab_summary", {}).get("text", ""))))


def _length_matched_random(records, retrieved_indexes, k, rng):
    """Greedy other-anchor match on complete evidence length; matching is query-local."""
    blocked = set(map(int, retrieved_indexes))
    pool = [i for i in range(len(records)) if i not in blocked]
    rng.shuffle(pool)
    # A random subsample keeps the match stochastic without scanning a huge bank per query.
    pool = pool[:min(len(pool), 5000)]
    chosen = []
    for target_index in retrieved_indexes:
        target = _record_text_length(records[int(target_index)])
        best = min((i for i in pool if i not in chosen),
                   key=lambda i: abs(_record_text_length(records[i]) - target))
        chosen.append(best)
    return np.asarray(chosen, dtype=int)


def _condition_anchors(base, direction, condition):
    keep = []
    for row in base:
        item = {k: row[k] for k in ["anchor_id", "subject_id", "hadm_id", "rank", "score", "linkage"]}
        if direction == "ecg":
            if condition in {"same_only", "same_labs", "same_cross", "full_trimodal", "full_trimodal_gated"}:
                item["ecg"] = row["ecg"]
            if condition in {"same_labs", "full_trimodal", "full_trimodal_gated"}: item["labs"] = row["labs"]
            if condition in {"same_cross", "full_trimodal", "full_trimodal_gated"}: item["cxr"] = row["cxr"]
        else:
            if condition in {"same_only", "same_labs", "same_cross", "full_trimodal", "full_trimodal_gated"}:
                item["cxr"] = row["cxr"]
            if condition in {"same_labs", "full_trimodal", "full_trimodal_gated"}: item["labs"] = row["labs"]
            if condition in {"same_cross", "full_trimodal", "full_trimodal_gated"}: item["ecg"] = row["ecg"]
        keep.append(item)
    return keep


def build_contexts(out=EXPERIMENT4_ROOT, k=2, max_queries=250, seed=4407):
    selected = json.loads((out / "bridge" / "selected_bridges.json").read_text())
    gates = json.loads((out / "bridge" / "confidence_gates.json").read_text())
    rng = np.random.default_rng(seed)
    contexts, query_audit = [], []
    for direction in ("ecg", "cxr"):
        base_manifest = read_jsonl(out / "external" / direction / "test" / "manifest.jsonl")
        usable = [i for i, r in enumerate(base_manifest)
                  if (str(r.get("machine_diagnosis", "")).strip() if direction == "ecg"
                      else str(r.get("report_text", "")).strip() and r.get('labels_available'))]
        chosen = sorted(rng.choice(usable, size=min(max_queries, len(usable)), replace=False).tolist())
        for method, genuine in selected.items():
            records = read_jsonl(out / "evidence" / "banks" / method / "genuine" / "symile" / "manifest.jsonl")
            bank = _trimodal_bank(out / "evidence" / "banks" / method / "genuine" / "symile")
            qroot = out / "evidence" / "queries" / method / "genuine" / direction / "test"
            qrecords = read_jsonl(qroot / "manifest.jsonl")
            qvectors = np.load(qroot / f"{direction}.npy", mmap_mode="r")
            if [r["sample_id"] for r in qrecords] != [r["sample_id"] for r in base_manifest]:
                raise ValueError("Query bank manifest order mismatch")
            control_cache = {}
            for control in ["shuffled", "mask_only", "no_code_match"]:
                root = out / "evidence" / "banks" / method / control / "symile"
                qpath = out / "evidence" / "queries" / method / control / direction / "test" / f"{direction}.npy"
                if root.exists() and qpath.exists():
                    control_cache[control] = (read_jsonl(root / "manifest.jsonl"),
                        _trimodal_bank(root), np.load(qpath, mmap_mode="r"))
            for qi in chosen:
                query = qrecords[qi]
                top, scores, margin = _rank(qvectors[qi], bank, k, query.get("subject_id"), records)
                anchors = [_anchor(records[idx], score, rank + 1, direction)
                           for rank, (idx, score) in enumerate(zip(top, scores))]
                random_indexes = _length_matched_random(records, top, k, rng)
                random_anchors = [_anchor(records[idx], None, rank + 1, direction)
                                  for rank, idx in enumerate(random_indexes)]
                gate = gates[method][direction]
                passed = bool(gate and margin >= gate["threshold"])
                definitions = {
                    "zero_shot": [], "same_only": _condition_anchors(anchors, direction, "same_only"),
                    "same_labs": _condition_anchors(anchors, direction, "same_labs"),
                    "same_cross": _condition_anchors(anchors, direction, "same_cross"),
                    "full_trimodal": _condition_anchors(anchors, direction, "full_trimodal"),
                    "full_random": _condition_anchors(random_anchors, direction, "full_trimodal"),
                    "full_trimodal_gated": _condition_anchors(
                        anchors, direction, "full_trimodal" if passed else "same_only"),
                }
                for control, (crecords, cbank, cq) in control_cache.items():
                    ctop, cscores, _ = _rank(cq[qi], cbank, k, query.get("subject_id"), crecords)
                    canchors = [_anchor(crecords[idx], score, rank + 1, direction)
                                for rank, (idx, score) in enumerate(zip(ctop, cscores))]
                    definitions[f"full_{control}"] = _condition_anchors(canchors, direction, "full_trimodal")
                for condition, evidence in definitions.items():
                    if condition == "zero_shot" and method != sorted(selected)[0]:
                        continue
                    row = {"direction": direction, "sample_id": query["sample_id"],
                           "subject_id": query.get("subject_id"), "query": query,
                           "method": method, "condition": condition, "evidence": evidence,
                           "k": k, "retrieval_margin": margin, "gate_threshold": gate["threshold"] if gate else None,
                           "gate_passed": passed if condition == "full_trimodal_gated" else None,
                           "global_patient_disjoint": True}
                    row["context_signature"] = signature(row)
                    contexts.append(row)
                query_audit.append({"direction": direction, "method": method,
                                    "sample_id": query["sample_id"], "margin": margin,
                                    "gate_passed": passed})
    validate_contexts(contexts, out)
    write_jsonl(out / "contexts" / "bidirectional_contexts.jsonl", contexts)
    import pandas as pd
    pd.DataFrame(query_audit).to_csv(out / "contexts" / "query_and_gate_audit.csv", index=False)
    protocol = {"status": "complete", "directions": ["ecg", "cxr"], "k": k,
                "max_queries_per_direction": max_queries, "rows": len(contexts),
                "conditions": sorted({r["condition"] for r in contexts}),
                "methods": sorted({r["method"] for r in contexts}),
                "reasoning_policy": "record concise evidence trace and uncertainty; do not request hidden chain-of-thought",
                "source_signature": signature(contexts)}
    write_json(out / "contexts" / "context_protocol.json", protocol)
    return protocol


def validate_contexts(rows, out=EXPERIMENT4_ROOT):
    audit = json.loads((out / "partition" / "global_partition_audit.json").read_text())
    if audit.get("status") != "complete" or not audit.get("patient_disjoint_verified"):
        raise ValueError("Global partition audit is not complete")
    if not rows or {r["direction"] for r in rows} != {"ecg", "cxr"}:
        raise ValueError("Both query directions must contain usable cases")
    keys = [(r["direction"], r["sample_id"], r["method"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)): raise ValueError("Duplicate context keys")
    bank_patients = {r['sample_id']: uid(r['subject_id']) for r in
                     read_jsonl(out / 'bridge' / 'prepared' / 'train' / 'manifest.jsonl')}
    query_patients = {(direction, r['sample_id']): uid(r['subject_id']) for direction in ('ecg', 'cxr')
                      for r in read_jsonl(out / 'external' / direction / 'test' / 'manifest.jsonl')}
    groups = {}
    for row in rows:
        groups.setdefault((row['direction'], row['method'], row['condition']), set()).add(row['sample_id'])
    for direction in ('ecg', 'cxr'):
        cohorts = [v for (d, _, _), v in groups.items() if d == direction]
        if any(v != cohorts[0] for v in cohorts): raise ValueError('Unbalanced query cohort across ablations')
    for row in rows:
        if query_patients.get((row['direction'], row['sample_id'])) != uid(row['subject_id']):
            raise ValueError('Query does not match current test partition')
        for anchor in row["evidence"]:
            if bank_patients.get(anchor['anchor_id']) != uid(anchor['subject_id']):
                raise ValueError('Evidence does not match current training partition')
            if uid(anchor["subject_id"]) == uid(row["subject_id"]):
                raise ValueError("Query patient leaked into retrieval context")
            if not anchor.get("hadm_id") or not anchor["linkage"].startswith("same Symile admission"):
                raise ValueError("Evidence lacks admission linkage")
    return {"contexts": len(rows), "validated": True}
