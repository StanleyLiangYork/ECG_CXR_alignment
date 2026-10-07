"""Canonical bidirectional clinical-content, hallucination, and review evaluation."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from e4_common import EXPERIMENT4_ROOT, positive_chexpert, read_jsonl, signature, write_json, write_jsonl
from e4_generation_runtime import generation_root, analysis_root

ECG_RULE_VERSION = "ecg_assertion_rules_v5_reviewed"
ECG_RULES = {
    "normal_ecg": [r"\bnormal(?:\s+12[- ]lead)?\s+(?:ecg|electrocardiogram)\b",
                   r"\b(?:ecg|electrocardiogram)\s+(?:is\s+)?within normal limits\b"],
    "atrial_fibrillation": [r"\batrial fibrillation\b", r"\ba[- ]?fib(?:rillation)?\b"],
    "atrial_flutter": [r"\batrial flutter\b"],
    "tachycardia": [r"\btachycard(?:ia|ic)\b", r"\b(?:svt|v[- ]?tach)\b"],
    "bradycardia": [r"\bbradycard(?:ia|ic)\b"],
    "conduction_abnormality": [r"\bbundle branch block\b", r"\b(?:rbbb|lbbb|ivcd)\b",
        r"\b(?:av|atrioventricular|heart) block\b", r"\bintraventricular conduction (?:delay|defect)\b"],
    "ventricular_hypertrophy": [r"\b(?:(?:left|right)\s+|bi)ventricular hypertrophy\b", r"\b(?:lvh|rvh)\b"],
    "ischemia": [r"\bischemi\w*\b", r"\bst[- ]t abnormalit\w*\b",
                 r"\bst(?:[- ]segment)?\s+(?:depression|elevation)\b"],
    "infarction": [r"\bmyocardial infarction\b", r"\binfarct(?:ion|ed)?\b", r"\b(?:stemi|nstemi)\b"],
    "paced_rhythm": [r"\b(?:atrial|ventricular|dual[- ]chamber)?\s*paced rhythm\b",
                     r"\bpacing spikes?\b", r"\bpacemaker rhythm\b"],
    "low_voltage": [r"\blow[- ]voltage\b"],
    "qt_abnormality": [r"\b(?:prolonged|long|short)\s+qt(?:c)?(?: interval)?\b",
                       r"\bqt(?:c)? prolongation\b"],
}
ECG_LABELS = sorted(ECG_RULES)

CXR_RULE_VERSION = "chexpert_assertion_rules_v2_reviewed"
CXR_RULES = {
    "Atelectasis": [r"\batelecta\w*\b"], "Cardiomegaly": [r"\bcardiomegal\w*\b", r"\benlarged cardiac silhouette\b"],
    "Consolidation": [r"\bconsolidat\w*\b"], "Edema": [r"\b(?:pulmonary )?edema\b", r"\bvascular congestion\b"],
    "Enlarged Cardiomediastinum": [r"\bwidened mediastinum\b", r"\benlarged cardiomediastin\w*\b"],
    "Fracture": [r"\bfracture\b"], "Lung Lesion": [r"\b(?:lung|pulmonary) (?:nodule|mass|lesion)\b"],
    "Lung Opacity": [r"\b(?:lung|pulmonary|airspace) opacit\w*\b", r"\binfiltrate\b"],
    "No Finding": [r"\bno acute cardiopulmonary (?:abnormality|process|finding)\b", r"\bnormal chest\b"],
    "Pleural Effusion": [r"\bpleural effusion\b"], "Pleural Other": [r"\bpleural thickening\b", r"\bpleural plaque\b"],
    "Pneumonia": [r"\bpneumonia\b"], "Pneumothorax": [r"\bpneumothorax\b"],
    "Support Devices": [r"\b(?:endotracheal|enteric|feeding|ng) tube\b", r"\b(?:central venous|picc) (?:line|catheter)\b",
                        r"\bpacemaker\b", r"\bdefibrillator\b"],
}
CXR_LABELS = sorted(CXR_RULES)

NEGATED_BEFORE = re.compile(r"(?:\bno\b|\bwithout\b|\bnegative for\b|\babsence of\b|\bfree of\b)(?:\W+\w+){0,8}\W*$", re.I)
NEGATED_AFTER = re.compile(r"^\W*(?:is |are |was |were )?(?:not|absent)(?:\W|$)", re.I)
UNCERTAIN_BEFORE = re.compile(r"(?:\bpossible\b|\bpossibly\b|\bprobable\b|\blikely\b|\bmay\b|\bmight\b|\bcannot exclude\b|\bcannot rule out\b|\bsuspicious for\b)(?:\W+\w+){0,8}\W*$", re.I)


def _assertion(text, match):
    before = text[max(0, match.start() - 120):match.start()]
    before = re.split(r"[.;\n]|\b(?:but|however|although|yet)\b", before, flags=re.I)[-1]
    after = text[match.end():match.end() + 50]
    if UNCERTAIN_BEFORE.search(before) or re.match(r"\W*(?:is |are )?(?:not excluded|cannot be excluded)", after): return "uncertain"
    if NEGATED_BEFORE.search(before) or NEGATED_AFTER.search(after): return "absent"
    return "present"


def assertions(text, rules):
    value = " ".join(str(text).split()).lower()
    result = {"present": set(), "absent": set(), "uncertain": set()}
    for concept, patterns in rules.items():
        for pattern in patterns:
            for match in re.finditer(pattern, value): result[_assertion(value, match)].add(concept)
    return result


def present(text, rules):
    result = set(assertions(text, rules)["present"])
    normal = "normal_ecg" if rules is ECG_RULES else "No Finding"
    if normal in result and len(result) > 1: result.remove(normal)
    return result


def cxr_reference(row):
    values = positive_chexpert(row.get("labels", []))
    if isinstance(values, str):
        try: values = json.loads(values)
        except json.JSONDecodeError: values = re.split(r"[,;|]", values)
    labels = {str(x).strip() for x in values if str(x).strip() in CXR_LABELS}
    if "No Finding" in labels and len(labels) > 1: labels.remove("No Finding")
    return labels


def _rouge_l(reference, candidate):
    a, b = str(reference).lower().split(), str(candidate).lower().split()
    if not a or not b: return 0.0
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]; precision = lcs / len(b); recall = lcs / len(a)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _metrics(group, label_names):
    yt = np.asarray(group["y_true"].tolist(), dtype=int)
    yp = np.asarray(group["y_pred"].tolist(), dtype=int)
    known = yt >= 0
    positives = int((yt == 1).sum())
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    micro_precision = tp / max(tp + fp, 1)
    micro_recall = tp / max(tp + fn, 1)
    micro_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    per_label = []
    for j in range(yt.shape[1]):
        ltp = int(np.sum((yt[:, j] == 1) & (yp[:, j] == 1)))
        lfp = int(np.sum((yt[:, j] == 0) & (yp[:, j] == 1)))
        lfn = int(np.sum((yt[:, j] == 1) & (yp[:, j] == 0)))
        per_label.append(2 * ltp / max(2 * ltp + lfp + lfn, 1))
    return {"n": len(group), "empty_reference_n": int(((yt == 1).sum(1) == 0).sum()),
            "micro_f1": micro_f1, "macro_f1": float(np.mean(per_label)),
            "micro_precision": micro_precision, "micro_recall": micro_recall,
            "exact_match": float(np.mean(np.all((yt == yp) | ~known, axis=1))),
            "hamming_accuracy": float(np.sum((yt == yp) & known) / max(known.sum(), 1)),
            "all_negative_hamming_baseline": 1 - positives / max(int(known.sum()), 1),
            "reference_discordant_positive_rate": float(group["reference_discordant_positive_rate"].mean()),
            "omission_rate": float(group["omission_rate"].mean()),
            "context_supported_prediction_fraction": float(group["context_supported_prediction_fraction"].mean()),
            "context_attributable_discordance_fraction": float(group["context_attributable_discordance_fraction"].mean()),
            "rougeL_f1": float(group["rougeL_f1"].mean()),
            "valid_evidence_citation_fraction": float(group["valid_evidence_citation_fraction"].mean()),
            "structured_schema_compliance_rate": float(group["structured_schema_compliant"].mean()),
            "input_tokens_mean": float(group["processor_input_tokens"].mean())}


def _model_registry():
    from e4_model_registry import ALL_MODELS
    return ALL_MODELS


def _write_model_scope_audit(out):
    rows = []
    for key, spec in _model_registry().items():
        rows.append({"model_key": key, "model_id": spec["model_id"],
                     "model_role": spec.get("role", "unspecified"),
                     "allowed_directions": "|".join(spec.get("directions", ("ecg", "cxr"))),
                     "input_mode": spec["mode"], "license": spec.get("license", "see model card"),
                     "training_overlap_warning": spec.get(
                         "training_overlap", "foundation-model patient overlap unverified")})
    pd.DataFrame(rows).to_csv(analysis_root(out) / "evaluation" / "model_scope_and_overlap_audit.csv", index=False)


def _successful_generations(out, context_lookup):
    """Merge original, rescue, and specialist files with deterministic provenance."""
    registry = _model_registry()
    latest = {}
    sources = {}
    root = generation_root(out)
    files = sorted(root.glob("*.jsonl"))
    protocols = {p.name.removesuffix('_protocol.json'): signature(json.loads(p.read_text()))
                 for p in root.glob('*_protocol.json')}
    # A successful formatting rescue supersedes an absent original result.  It
    # never replaces an existing successful base row by construction.
    files.sort(key=lambda path: ("format_rescue" in path.name, path.name))
    for path in files:
        for row_number, row in enumerate(read_jsonl(path), 1):
            model = row.get("model_key")
            if model not in registry or row.get("status") != "ok":
                continue
            if path.name not in {f'{model}.jsonl', f'{model}_format_rescue.jsonl'}: continue
            if root != Path(out) / 'generations':
                if model not in protocols or row.get('generation_protocol_signature') != protocols[model]:
                    raise ValueError(f'Unmatched generation protocol in {path.name}:{row_number}')
            if row.get("context_signature") not in context_lookup:
                continue
            if row.get("direction") not in registry[model].get("directions", ("ecg", "cxr")):
                continue
            if not str(row.get("generated_report", "")).strip():
                continue
            from e4_generation import _logical_key
            if _logical_key(row) != _logical_key(context_lookup[row['context_signature']]):
                raise ValueError(f'Generation/context identity mismatch in {path.name}:{row_number}')
            if registry[model]['mode'] == 'text_only' and row['condition'] == 'zero_shot': continue
            key = (model, row["direction"], row["sample_id"], row["method"], row["condition"])
            if 'format_rescue' in path.name and key in latest: continue
            enriched = dict(row)
            enriched["generation_source_file"] = path.name
            enriched["generation_source_row"] = row_number
            # All original successes were strict JSON. Rescue rows report their
            # own formatting stage; new runs explicitly store compliance.
            if "structured_schema_compliant" not in enriched:
                enriched["structured_schema_compliant"] = not bool(
                    enriched.get("result_source") == "format_rescue" and
                    str(enriched.get("formatting_stage", "")) != "strict"
                )
            enriched.setdefault("output_parse_stage", enriched.get("formatting_stage", "legacy_strict_json"))
            latest[key] = enriched
            sources[key] = path.name
    return list(latest.values())


def evaluate(out=EXPERIMENT4_ROOT, bootstraps=1000):
    out = Path(out)
    (analysis_root(out) / 'evaluation').mkdir(parents=True, exist_ok=True)
    contexts = read_jsonl(out / "contexts" / "bidirectional_contexts.jsonl")
    lookup = {r["context_signature"]: r for r in contexts}
    generations = _successful_generations(out, lookup)
    if not generations: raise RuntimeError("No successful Experiment 4 generations found")
    details = []
    for row in generations:
        context = lookup.get(row["context_signature"])
        if context is None: continue
        query = context["query"]
        if row["direction"] == "ecg":
            names, rules = ECG_LABELS, ECG_RULES
            reference_text = query.get("machine_diagnosis", "")
            truth = present(reference_text, rules)
        else:
            names, rules = CXR_LABELS, CXR_RULES
            reference_text = query.get("report_text", "")
            truth = cxr_reference(query)
        predicted = present(row["generated_report"], rules)
        known = set(names)
        raw_labels = query.get('raw_labels')
        if row['direction'] == 'cxr' and isinstance(raw_labels, dict):
            known = {x for x in names if str(raw_labels.get(x)).lower() in {'0','0.0','1','1.0','true','false','positive','negative'}}
        if not known: raise ValueError(f'No evaluable reference labels for {row["sample_id"]}')
        fp, fn = (predicted - truth) & known, truth - predicted
        context_labels = set()
        component_cap = int(row.get('evidence_component_character_cap', 1800))
        for anchor in context["evidence"]:
            if row["direction"] == "ecg" and "ecg" in anchor:
                context_labels |= present(str(anchor["ecg"].get("machine_diagnosis", ""))[:component_cap], ECG_RULES)
            if row["direction"] == "cxr" and "cxr" in anchor:
                context_labels |= present(str(anchor["cxr"].get("report", ""))[:component_cap], CXR_RULES)
                context_labels |= {x for x in anchor["cxr"].get("labels", []) if x in CXR_LABELS}
        y_true = [int(x in truth) if x in known else -1 for x in names]; y_pred = [int(x in predicted) for x in names]
        valid_sources = {a["anchor_id"] for a in context["evidence"]}
        trace = row.get("evidence_trace") or []
        cited = [str(s) for item in trace if isinstance(item, dict) for s in item.get("source_ids", [])]
        details.append({**row, "subject_id": context["subject_id"], "reference_text": reference_text,
            "reference_labels": sorted(truth), "predicted_labels": sorted(predicted),
            "reference_discordant_labels": sorted(fp), "omitted_labels": sorted(fn),
            "y_true": y_true, "y_pred": y_pred,
            "unknown_reference_labels": sorted(set(names) - known),
            "reference_discordant_positive_rate": len(fp) / max(len(predicted & known), 1),
            "omission_rate": len(fn) / max(len(truth), 1),
            "context_supported_prediction_fraction": len(predicted & context_labels) / max(len(predicted), 1),
            "context_attributable_discordance_fraction": len(fp & context_labels) / max(len(fp), 1),
            "rougeL_f1": _rouge_l(reference_text, row["generated_report"]),
            "valid_evidence_citation_fraction": (sum(s in valid_sources for s in cited) / len(cited)
                                                   if cited else (1.0 if not valid_sources else 0.0)),
            "evidence_trace_claims": len(trace),
            "structured_schema_compliant": bool(row.get("structured_schema_compliant", False)),
            "model_role": _model_registry()[row["model_key"]].get("role", "unspecified"),
            "training_overlap_warning": _model_registry()[row["model_key"]].get(
                "training_overlap", "foundation-model patient overlap unverified"),
            "evaluation_rule_version": ECG_RULE_VERSION if row["direction"] == "ecg" else CXR_RULE_VERSION})
    if not details: raise RuntimeError("No successful generations match the current contexts")
    protocol_rows = []
    for model in sorted({r['model_key'] for r in details}):
        protocols = {r.get('generation_protocol_signature') for r in details
                     if r['model_key'] == model}
        protocol_rows.append({"model_key": model, "successful_protocol_count": len(protocols),
                              "protocol_homogeneous": len(protocols) == 1 and None not in protocols,
                              "note": "mixed protocols/rescue outputs are descriptive only; paired tests require matching signatures"})
    pd.DataFrame(protocol_rows).to_csv(analysis_root(out) / "evaluation" / "generation_protocol_audit.csv", index=False)
    write_jsonl(analysis_root(out) / "evaluation" / "detailed.jsonl", details)
    frame = pd.DataFrame(details)
    frame.to_csv(analysis_root(out) / "evaluation" / "detailed.csv", index=False)
    summary = []
    keys = ["model_key", "direction", "method", "condition", "input_scope"]
    for values, group in frame.groupby(keys, dropna=False):
        names = ECG_LABELS if values[1] == "ecg" else CXR_LABELS
        spec = _model_registry()[values[0]]
        summary.append({**dict(zip(keys, values)),
                        "model_role": spec.get("role", "unspecified"),
                        "training_overlap_warning": spec.get(
                            "training_overlap", "foundation-model patient overlap unverified"),
                        **_metrics(group, names)})
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(analysis_root(out) / "evaluation" / "metrics_summary.csv", index=False)
    _write_model_scope_audit(out)
    (frame.groupby(["model_key", "direction", "output_parse_stage"], dropna=False)
          .size().rename("n").reset_index()
          .to_csv(analysis_root(out) / "evaluation" / "output_format_audit.csv", index=False))
    _paired_bootstrap(frame, out, bootstraps=bootstraps)
    _completeness(contexts, generations, out)
    _evidence_integration_decision(summary_frame, out)
    return summary_frame


def _paired_bootstrap(frame, out, bootstraps=1000, seed=4408):
    rng = np.random.default_rng(seed)
    comparisons = {
        "incremental_labs": ("same_labs", "same_only"),
        "incremental_cross": ("same_cross", "same_only"),
        "incremental_full_trimodal": ("full_trimodal", "same_only"),
        "retrieval_specificity": ("full_trimodal", "full_random"),
        "shuffled_bridge_attribution": ("full_trimodal", "full_shuffled"),
        "lab_value_attribution": ("full_trimodal", "full_mask_only"),
        "confidence_gate": ("full_trimodal_gated", "full_trimodal"),
        "code_match": ("full_trimodal", "full_no_code_match"),
    }
    rows = []
    group_keys = ["model_key", "direction", "method", "input_scope"]
    for values, group in frame.groupby(group_keys, dropna=False):
        for label, (a, b) in comparisons.items():
            left, right = group[group.condition == a], group[group.condition == b]
            common = sorted(set(left.sample_id) & set(right.sample_id))
            if not common: continue
            left = left.set_index("sample_id").loc[common]; right = right.set_index("sample_id").loc[common]
            matched = (left.generation_protocol_signature.notna() &
                       left.generation_protocol_signature.eq(right.generation_protocol_signature))
            common = left.index[matched].tolist()
            if not common: continue
            left, right = left.loc[common], right.loc[common]
            names = ECG_LABELS if values[1] == "ecg" else CXR_LABELS
            metrics = {}
            for condition, part in [(a, left), (b, right)]: metrics[condition] = _metrics(part.reset_index(), names)
            patients = left.subject_id.astype(str).to_numpy()
            unique = np.unique(patients)
            patient_rows = [np.flatnonzero(patients == p) for p in unique]
            resamples = [np.concatenate([patient_rows[i] for i in rng.integers(0, len(unique), len(unique))])
                         for _ in range(bootstraps)]
            def sufficient(part):
                yt, yp = np.asarray(part.y_true.tolist()), np.asarray(part.y_pred.tolist())
                return np.column_stack([((yt == 1) & (yp == 1)).sum(1),
                    ((yt == 0) & (yp == 1)).sum(1), ((yt == 1) & (yp == 0)).sum(1),
                    ((yt == yp) & (yt >= 0)).sum(1), (yt >= 0).sum(1),
                    part.reference_discordant_positive_rate.to_numpy(), part.omission_rate.to_numpy()])
            def metric_values(stats):
                tp, fp, fn, correct, known, discordance, omission = stats.sum(0)
                return {'micro_f1': 2*tp/max(2*tp+fp+fn, 1), 'hamming_accuracy': correct/max(known,1),
                        'reference_discordant_positive_rate': discordance/len(stats), 'omission_rate': omission/len(stats)}
            left_stats, right_stats = sufficient(left), sufficient(right)
            draws = [(metric_values(left_stats[ix]), metric_values(right_stats[ix])) for ix in resamples]
            for metric in ["micro_f1", "reference_discordant_positive_rate", "omission_rate", "hamming_accuracy"]:
                observed = metrics[a][metric] - metrics[b][metric]
                if metric in {"reference_discordant_positive_rate", "omission_rate"}: observed *= -1
                samples = []
                for draw_a, draw_b in draws:
                    ma, mb = draw_a[metric], draw_b[metric]
                    value = ma - mb
                    samples.append(-value if metric in {"reference_discordant_positive_rate", "omission_rate"} else value)
                rows.append({**dict(zip(group_keys, values)), "comparison": label,
                             "condition_a": a, "condition_b": b, "metric": metric,
                             "n_paired": len(common), "n_patients": left.subject_id.nunique(),
                             "resampling_unit": "patient", "improvement_a_over_b": observed,
                             "ci_low": float(np.quantile(samples, .025)),
                             "ci_high": float(np.quantile(samples, .975))})
    pd.DataFrame(rows, columns=group_keys + ['comparison', 'condition_a', 'condition_b', 'metric',
        'n_paired', 'n_patients', 'resampling_unit', 'improvement_a_over_b', 'ci_low', 'ci_high']).to_csv(
            analysis_root(out) / "evaluation" / "paired_bootstrap.csv", index=False)


def _evidence_integration_decision(summary, out, safety_margin=.02):
    """Prespecified claim gate for evidence-integration utility, not clinical effectiveness."""
    paired = pd.read_csv(analysis_root(out) / "evaluation" / "paired_bootstrap.csv")
    rows = []
    keys = ["model_key", "direction", "method", "input_scope"]
    for values, group in paired.groupby(keys, dropna=False):
        def cell(comparison, metric):
            part = group[(group.comparison == comparison) & (group.metric == metric)]
            return None if part.empty else part.iloc[0]
        incremental = cell("incremental_full_trimodal", "micro_f1")
        specificity = cell("retrieval_specificity", "micro_f1")
        discordance = cell("incremental_full_trimodal", "reference_discordant_positive_rate")
        omission = cell("incremental_full_trimodal", "omission_rate")
        selected = summary
        for key, value in zip(keys, values): selected = selected[selected[key] == value]
        selected = selected[selected.condition == "full_trimodal"]
        citation = None if selected.empty else float(selected.iloc[0].valid_evidence_citation_fraction)
        criteria = {
            "generation_complete": bool(pd.read_csv(analysis_root(out) / 'evaluation' / 'generation_completeness.csv').set_index('model_key').loc[values[0], 'missing'] == 0),
            "protocol_homogeneous": bool(pd.read_csv(analysis_root(out) / 'evaluation' / 'generation_protocol_audit.csv').set_index('model_key').loc[values[0], 'protocol_homogeneous']),
            "improves_over_same_sensor": bool(incremental is not None and incremental.ci_low > 0),
            "outperforms_random_evidence": bool(specificity is not None and specificity.ci_low > 0),
            "discordance_noninferior": bool(discordance is not None and discordance.ci_low >= -safety_margin),
            "omission_noninferior": bool(omission is not None and omission.ci_low >= -safety_margin),
            "provenance_adherence": bool(citation is not None and citation >= .95),
        }
        rows.append({**dict(zip(keys, values)), **criteria, "citation_fraction": citation,
                     "safety_margin": safety_margin,
                     "evidence_integration_supported": all(criteria.values())})
    pd.DataFrame(rows).to_csv(analysis_root(out) / "evaluation" / "evidence_integration_decision.csv", index=False)


def _completeness(contexts, generations, out):
    expected = {(r["direction"], r["sample_id"], r["method"], r["condition"]): r for r in contexts}
    rows = []
    for model, spec in _model_registry().items():
        mode = 'retrieval_text_only' if spec['mode'] == 'text_only' else 'raw_sensor_plus_retrieval'
        directions = set(spec.get("directions", ("ecg", "cxr")))
        expected_keys = {k for k in expected if k[0] in directions and
                         not (mode == "retrieval_text_only" and k[3] == "zero_shot")}
        actual = {(r["direction"], r["sample_id"], r["method"], r["condition"])
                  for r in generations if r["model_key"] == model}
        rows.append({"model_key": model, "expected": len(expected_keys), "successful": len(actual & expected_keys),
                     "missing": len(expected_keys - actual), "unexpected": len(actual - expected_keys)})
    pd.DataFrame(rows).to_csv(analysis_root(out) / "evaluation" / "generation_completeness.csv", index=False)


def generation_coverage(out=EXPERIMENT4_ROOT):
    """Audit direction-aware generator coverage without computing report metrics."""
    out = Path(out)
    contexts = read_jsonl(out / "contexts" / "bidirectional_contexts.jsonl")
    lookup = {row["context_signature"]: row for row in contexts}
    generations = _successful_generations(out, lookup)
    (analysis_root(out) / "evaluation").mkdir(parents=True, exist_ok=True)
    _completeness(contexts, generations, out)
    _write_model_scope_audit(out)
    return pd.read_csv(analysis_root(out) / "evaluation" / "generation_completeness.csv")


def build_review_packets(out=EXPERIMENT4_ROOT, per_stratum=5, seed=4409):
    review_root = analysis_root(out) / 'clinical_review'
    if any((review_root / name).exists() for name in ['reviewer_1.csv', 'reviewer_2.csv', 'private_key.csv']):
        raise FileExistsError('Clinical review already exists; reviewer work will not be overwritten')
    rows = read_jsonl(analysis_root(out) / "evaluation" / "detailed.jsonl")
    if not rows: raise ValueError('No evaluated reports to review')
    contexts = {r['context_signature']: r for r in read_jsonl(out / 'contexts' / 'bidirectional_contexts.jsonl')}
    from e4_generation import query_image, _evidence_text
    rng = np.random.default_rng(seed)
    strata = {}
    for row in rows:
        key = (row["direction"], row["model_key"], row["method"], row["condition"])
        strata.setdefault(key, []).append(row)
    selected = []
    for values in strata.values():
        selected.extend(values[i] for i in rng.choice(len(values), min(per_stratum, len(values)), replace=False))
    order = rng.permutation(len(selected)); packet, key_rows = [], []
    for n, i in enumerate(order, 1):
        row = selected[i]; review_id = f"E4R{n:05d}"
        context = contexts[row['context_signature']]
        image_path = row.get('query_image') or str(query_image(context, out))
        packet.append({"review_id": review_id, "direction": row["direction"],
                       "query_image": image_path, "reference_text": row["reference_text"],
                       "retrieved_evidence": row.get('prompt') or _evidence_text(context),
                       "generated_report": row["generated_report"], "evidence_trace": json.dumps(row.get("evidence_trace", [])),
                       "sensor_quality_adequate": "", "reference_correct": "", "report_correctness_1_to_5": "",
                       "unsupported_or_harmful_claim": "", "important_omission": "", "evidence_trace_supported": "",
                       "reviewer_confidence_1_to_5": "", "reviewer_id": "", "notes": ""})
        key_rows.append({"review_id": review_id, "model_key": row["model_key"], "method": row["method"],
                         "condition": row["condition"], "sample_id": row["sample_id"],
                         "context_signature": row["context_signature"]})
    review_root = analysis_root(out) / "clinical_review"
    review_root.mkdir(parents=True, exist_ok=True)
    packet_frame = pd.DataFrame(packet)
    packet_frame.to_csv(review_root / "blinded_review.csv", index=False)
    packet_frame.to_csv(review_root / "reviewer_1.csv", index=False)
    packet_frame.to_csv(review_root / "reviewer_2.csv", index=False)
    pd.DataFrame(key_rows).to_csv(review_root / "private_key.csv", index=False)
    write_json(review_root / "protocol.json", {"reviewers": 2, "independent": True,
        "adjudication": "third reviewer resolves disagreements", "rows": len(packet),
        "blinding": "model, bridge method, and condition hidden from reviewers",
        "terminology": "automated positives are reference-discordant until clinical adjudication"})
    return len(packet)


def summarize_clinical_review(out=EXPERIMENT4_ROOT):
    root = analysis_root(out) / "clinical_review"
    one, two = pd.read_csv(root / "reviewer_1.csv", dtype=str), pd.read_csv(root / "reviewer_2.csv", dtype=str)
    fields = ["sensor_quality_adequate", "reference_correct", "report_correctness_1_to_5",
              "unsupported_or_harmful_claim", "important_omission", "evidence_trace_supported"]
    key = pd.read_csv(root / 'private_key.csv', dtype=str)
    if not len(key) or set(one.review_id) != set(key.review_id):
        raise ValueError('Reviewer IDs do not match the frozen private key')
    if set(one.review_id) != set(two.review_id): raise ValueError("Reviewer packets contain different review IDs")
    merged = one.merge(two, on="review_id", suffixes=("_r1", "_r2"), validate="one_to_one")
    incomplete = []
    for field in fields:
        incomplete.extend(merged.loc[merged[f"{field}_r1"].fillna("").eq("") |
                                     merged[f"{field}_r2"].fillna("").eq(""), "review_id"].tolist())
    if incomplete:
        raise RuntimeError(f"Clinical review is incomplete for {len(set(incomplete))} cases")
    def validate_ratings(frame, suffix=''):
        for field in fields:
            allowed = {'1','2','3','4','5'} if field.endswith('1_to_5') else {'Y','N','NI'}
            if not frame[field + suffix].isin(allowed).all():
                raise ValueError(f'Invalid ratings in {field}; allowed: {sorted(allowed)}')
    validate_ratings(one); validate_ratings(two)
    disagreements = []
    for _, row in merged.iterrows():
        differing = [field for field in fields if row[f"{field}_r1"] != row[f"{field}_r2"]]
        if differing:
            disagreements.append({"review_id": row.review_id, "disagreeing_fields": "|".join(differing),
                                  **{f"{field}_r1": row[f"{field}_r1"] for field in fields},
                                  **{f"{field}_r2": row[f"{field}_r2"] for field in fields},
                                  **{f"{field}_adjudicated": "" for field in fields},
                                  "adjudicator_id": "", "adjudication_notes": ""})
    path = root / "adjudication.csv"
    if disagreements and not path.exists(): pd.DataFrame(disagreements).to_csv(path, index=False)
    agreement = {field: float((merged[f"{field}_r1"] == merged[f"{field}_r2"]).mean()) for field in fields}
    adjudication_complete = not disagreements
    if disagreements and path.exists():
        adjudicated = pd.read_csv(path, dtype=str)
        required = [f"{field}_adjudicated" for field in fields]
        if adjudicated.review_id.duplicated().any() or set(adjudicated.review_id) != {r['review_id'] for r in disagreements}:
            raise ValueError('Adjudication IDs must exactly match current disagreements')
        adjudication_complete = all(column in adjudicated for column in required) and bool(
            np.all(adjudicated[required].fillna("").ne("")))
        if adjudication_complete:
            validate_ratings(adjudicated, '_adjudicated')
            if 'adjudicator_id' not in adjudicated or adjudicated.adjudicator_id.fillna('').eq('').any():
                raise ValueError('Completed adjudications require an adjudicator ID')
    if adjudication_complete:
        final = one[['review_id'] + fields].copy().set_index('review_id')
        if disagreements:
            for _, row in adjudicated.iterrows():
                for field in fields: final.loc[row.review_id, field] = row[field + '_adjudicated']
        final.reset_index().merge(key, on='review_id', validate='one_to_one').to_csv(
            root / 'adjudicated_results_private.csv', index=False)
    report = {"review_rows": len(merged), "disagreement_rows": len(disagreements),
              "raw_agreement": agreement, "adjudication_required": bool(disagreements),
              "status": "complete" if adjudication_complete else "awaiting_adjudication"}
    write_json(root / "review_summary.json", report)
    return report


def extractor_self_test():
    tests = [("No evidence of atrial fibrillation.", ECG_RULES, set()),
             ("Left ventricular hypertrophy.", ECG_RULES, {"ventricular_hypertrophy"}),
             ("Possible left ventricular hypertrophy.", ECG_RULES, set()),
             ("Ventricular paced rhythm.", ECG_RULES, {"paced_rhythm"}),
             ("No pleural effusion or pneumothorax.", CXR_RULES, set()),
             ("Mild bibasilar atelectasis.", CXR_RULES, {"Atelectasis"})]
    for text, rules, expected in tests:
        observed = present(text, rules)
        if observed != expected: raise AssertionError((text, observed, expected))
    return {"tests": len(tests), "ecg_version": ECG_RULE_VERSION, "cxr_version": CXR_RULE_VERSION}


def reference_audit(out=EXPERIMENT4_ROOT):
    contexts = read_jsonl(out / "contexts" / "bidirectional_contexts.jsonl")
    queries = {}
    for row in contexts: queries[(row["direction"], row["sample_id"])] = row["query"]
    report = {"extractors": {"ecg": ECG_RULE_VERSION, "cxr": CXR_RULE_VERSION}, "directions": {}}
    prevalence = []
    for direction in ["ecg", "cxr"]:
        rows = [q for (d, _), q in queries.items() if d == direction]
        names = ECG_LABELS if direction == "ecg" else CXR_LABELS
        labels = []
        for row in rows:
            labels.append(present(row.get("machine_diagnosis", ""), ECG_RULES) if direction == "ecg"
                          else cxr_reference(row))
        positives = sum(len(x) for x in labels)
        report["directions"][direction] = {"queries": len(rows),
            "empty_reference_count": sum(not x for x in labels),
            "total_positive_labels": positives,
            "all_negative_hamming_baseline": 1 - positives / max(len(rows) * len(names), 1),
            "reference_status": "machine-derived; clinician adjudication pending"}
        for name in names:
            prevalence.append({"direction": direction, "label": name,
                               "positive_count": sum(name in x for x in labels),
                               "prevalence": sum(name in x for x in labels) / max(len(labels), 1)})
    write_json(out / "contexts" / "reference_label_audit.json", report)
    pd.DataFrame(prevalence).to_csv(out / "contexts" / "reference_label_prevalence.csv", index=False)
    return report
