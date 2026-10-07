"""Explicit five-pipeline selection, patient-paired analysis, and blinded review.

Generation files are read-only. Analysis artifacts have an independent frozen ID.
No model loading, generation-module edits, or merging of historical runs is required.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from e4_common import EXPERIMENT4_ROOT, sha256, signature
from e4_model_registry import ALL_MODELS
from e4_evaluation import (ECG_LABELS, CXR_LABELS, ECG_RULES, CXR_RULES,
    ECG_RULE_VERSION, CXR_RULE_VERSION, present, cxr_reference, _rouge_l, _metrics)

WITHIN_CONTRASTS = {
    'primary_full_vs_same_sensor': ('full_trimodal', 'same_only'),
    'primary_full_vs_random': ('full_trimodal', 'full_random'),
    'incremental_labs': ('same_labs', 'same_only'),
    'incremental_cross': ('same_cross', 'same_only'),
    'shuffled_training_bridge': ('full_trimodal', 'full_shuffled'),
    'lab_mask_only_bridge': ('full_trimodal', 'full_mask_only'),
    'compatibility_gate': ('full_trimodal_gated', 'full_trimodal'),
    'code_match': ('full_trimodal', 'full_no_code_match'),
}
GROUPS = ['model_key', 'direction', 'method', 'condition']
REVIEW_FIELDS = ['sensor_quality_adequate', 'reference_correct', 'report_correctness_1_to_5',
                 'unsupported_or_harmful_claim', 'important_omission', 'evidence_trace_supported',
                 'reviewer_confidence_1_to_5']


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def _write_rows(path, values):
    with Path(path).open('w', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + '\n')


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError(f'Unsafe run/analysis name: {value!r}')
    return value


def configuration(path=None):
    path = Path(path or os.environ.get('E4_ANALYSIS_CONFIG', Path(__file__).with_name('analysis_model_runs.json')))
    value = _json(path)
    if set(value['model_runs']) != set(ALL_MODELS):
        raise ValueError('Explicitly select one run for each of the five models')
    for run in value['model_runs'].values():
        _name(run)
    _name(value['analysis_id'])
    if int(value['bootstrap_replicates']) < 1:
        raise ValueError('Bootstrap replicate count must be positive')
    return value


def analysis_path(out=EXPERIMENT4_ROOT, config=None):
    config = config or configuration()
    return Path(out) / 'analysis' / _name(os.environ.get('E4_ANALYSIS_ID', config['analysis_id']))


def _key(row):
    return tuple(str(row[k]) for k in ('direction', 'sample_id', 'method', 'condition'))


def _eligible(row, model):
    spec = ALL_MODELS[model]
    return row['direction'] in spec['directions'] and not (
        spec['mode'] == 'text_only' and row['condition'] == 'zero_shot')


def _non_report(row):
    text = str(row.get('generated_report', '')).strip()
    if not text:
        return 'empty_report'
    text = re.sub(r'</?s>', '', text)
    words = set(re.sub(r'[^a-z]+', ' ', text.lower()).split())
    if words <= {'report','ecg','query','evidence','trace','uncertainties','uncertainty',
                 'findings','impression','source','ids'}:
        return 'schema_echo'
    if row.get('finish_reason') == 'length':
        return 'output_budget'
    return None


def _snapshot(path):
    path = Path(path)
    if not path.exists():
        return None, None
    before = path.stat()
    values = _rows(path) if path.suffix == '.jsonl' else _json(path)
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Input is being modified: {path}. Wait for generation to stop.')
    return values, {'path': str(path), 'sha256': digest, 'bytes': after.st_size}


def collect(out=EXPERIMENT4_ROOT, config=None):
    """Validate exact files/protocols and deduplicate cells without pooling runs."""
    out, config = Path(out), config or configuration()
    context_path = out / 'contexts' / 'bidirectional_contexts.jsonl'
    contexts, context_inventory = _snapshot(context_path)
    if not contexts:
        raise ValueError('Current contexts are missing or empty')
    lookup = {_key(c): c for c in contexts}
    if len(lookup) != len(contexts) or len({c['context_signature'] for c in contexts}) != len(contexts):
        raise ValueError('Duplicate context identities/signatures')
    if any(c.get('subject_id') is None for c in contexts):
        raise ValueError('Patient IDs are required for clustered comparisons')
    queries = {}
    for c in contexts:
        key = (c['direction'],c['sample_id'])
        identity = signature([str(c['subject_id']),c['query']])
        if key in queries and queries[key] != identity:
            raise ValueError('Query/reference identity changes across evidence conditions')
        queries[key] = identity
    inventory, exclusions, cells, successes = {}, [], [], []
    for model, run in config['model_runs'].items():
        root = out / 'generations' / 'runs' / run
        attempts, file_info = _snapshot(root / f'{model}.jsonl')
        protocol, protocol_info = _snapshot(root / f'{model}_protocol.json')
        manifest, manifest_info = _snapshot(root / f'{model}_run_manifest.json')
        inventory[model] = {'run': run, 'results': file_info, 'protocol': protocol_info, 'manifest': manifest_info}
        attempts = attempts or []
        if attempts and (not protocol or not manifest):
            raise ValueError(f'{model}: missing protocol or run manifest')
        if attempts:
            if manifest.get('contexts_sha256') != context_inventory['sha256']:
                raise ValueError(f'{model}: run was not generated from the current context file')
            if protocol.get('run_manifest_sha256') != manifest_info['sha256']:
                raise ValueError(f'{model}: protocol/manifest hash mismatch')
        grouped = defaultdict(list)
        for line, row in enumerate(attempts, 1):
            if row.get('model_key') != model or row.get('generation_protocol_signature') != signature(protocol):
                raise ValueError(f'{model}:{line}: unexpected model or generation protocol')
            if row.get('model_id') != ALL_MODELS[model]['model_id'] or row.get('resolved_revision') != protocol.get('resolved_revision'):
                raise ValueError(f'{model}:{line}: model identity/revision mismatch')
            key = _key(row)
            if key not in lookup or row.get('context_signature') != lookup[key]['context_signature'] or not _eligible(row, model):
                exclusions.append({'model_key':model, 'source_line':line, 'reason':'stale_unexpected_or_out_of_scope',
                                   'sample_id':row.get('sample_id')})
                continue
            expected_scope = 'retrieval_text_only' if ALL_MODELS[model]['mode'] == 'text_only' else 'raw_sensor_plus_retrieval'
            if row.get('input_scope') != expected_scope:
                raise ValueError(f'{model}:{line}: unexpected input scope')
            grouped[key].append((line, row))
        for key, context in lookup.items():
            if not _eligible(context, model):
                continue
            values = grouped.get(key, [])
            good = [(line,r) for line,r in values if r.get('status') == 'ok' and not _non_report(r)]
            if len({r.get('generated_report') for _,r in good}) > 1:
                raise ValueError(f'{model}/{key}: conflicting successful duplicates; resolve explicitly')
            selected = good[0] if good else (values[-1] if values else None)
            status = 'successful' if good else ('failed' if values else 'pending')
            plain = bool(protocol and protocol.get('protocol_type') == 'pulse_compact_plain_report_v1')
            reason = None
            if selected and not good:
                row = selected[1]
                reason = _non_report(row) if row.get('status') == 'ok' else row.get('error_category', 'unknown_error')
            cell = {**{k:context[k] for k in ('direction','sample_id','method','condition','context_signature','subject_id')},
                    'model_key':model, 'selected_run':run, 'status':status, 'attempts':len(values),
                    'duplicate_attempts':max(len(values)-1,0), 'failure_category':reason,
                    'structured_output_applicable':not plain,
                    'generation_protocol_signature':signature(protocol) if protocol else None}
            cells.append(cell)
            if good:
                line, row = selected
                successes.append({**row, 'subject_id':context['subject_id'], 'selected_run':run,
                    'generation_source_file':file_info['path'], 'generation_source_row':line,
                    'structured_output_applicable':not plain})
    fingerprint = {'configuration':config, 'contexts':context_inventory, 'sources':inventory,
                   'analysis_code_sha256':sha256(Path(__file__)),
                   'metric_code_sha256':sha256(Path(__file__).with_name('e4_evaluation.py'))}
    return {'contexts':contexts, 'successes':successes, 'cells':cells, 'inventory':fingerprint, 'exclusions':exclusions}


def _check_freeze(root, inventory):
    path = root / 'analysis_manifest.json'
    if path.exists() and _json(path)['inventory'] != inventory:
        raise ValueError('Selected outputs, contexts, settings, or analysis code changed. Use a NEW E4_ANALYSIS_ID; frozen analysis/reviewer files preserved.')


def _coverage(bundle):
    frame = pd.DataFrame(bundle['cells'])
    rows = []
    for values, group in frame.groupby(GROUPS, sort=True):
        counts = group.status.value_counts()
        rows.append({**dict(zip(GROUPS,values)), 'selected_run':group.selected_run.iloc[0],
            'expected':len(group), 'patients':group.subject_id.nunique(),
            'successful':int(counts.get('successful',0)), 'failed':int(counts.get('failed',0)),
            'pending':int(counts.get('pending',0)), 'attempt_rows':int(group.attempts.sum()),
            'duplicate_attempts':int(group.duplicate_attempts.sum()),
            'availability_rate':float((group.status == 'successful').mean())})
    return pd.DataFrame(rows)


def audit_generation(out=EXPERIMENT4_ROOT):
    config = configuration(); root = analysis_path(out,config)
    bundle = collect(out,config)
    _check_freeze(root,bundle['inventory'])
    root.mkdir(parents=True,exist_ok=True)
    coverage = _coverage(bundle)
    coverage.to_csv(root/'generation_coverage.csv',index=False)
    pd.DataFrame(bundle['cells']).to_csv(root/'generation_cells.csv',index=False)
    pd.DataFrame(bundle['exclusions'],columns=['model_key','source_line','reason','sample_id']).to_csv(root/'excluded_attempts.csv',index=False)
    failures = pd.DataFrame(bundle['cells'])
    failures[failures.status.eq('failed')].groupby(GROUPS+['failure_category'],dropna=False).size().rename('cells').reset_index().to_csv(root/'failure_categories.csv',index=False)
    _write(root/'selected_run_inventory.json',bundle['inventory'])
    print('Analysis output:',root)
    return coverage


def _reference(context):
    query = context['query']; direction = context['direction']
    names = ECG_LABELS if direction == 'ecg' else CXR_LABELS
    text = query.get('machine_diagnosis','') if direction == 'ecg' else query.get('report_text','')
    if direction == 'ecg':
        truth = present(text,ECG_RULES)
        # An empty interpretation is not a confirmed all-negative reference.
        known = set(names) if str(text).strip() else set()
    else:
        raw = query.get('raw_labels',query.get('labels'))
        if isinstance(raw,dict):
            known = {k for k in names if str(raw.get(k)).lower() in {'0','0.0','1','1.0','true','false','positive','negative'}}
            truth = cxr_reference({'labels':raw})
        else:
            # Positive-only lists do not establish negative/unknown label states.
            truth = cxr_reference(query)
            known = set(truth)
    return names, text, [int(k in truth) if k in known else -1 for k in names]


def _score(row,context):
    names,text,yt = _reference(context)
    if not any(v >= 0 for v in yt):
        return None
    rules = ECG_RULES if row['direction'] == 'ecg' else CXR_RULES
    predictions = present(row['generated_report'],rules)
    yp = [int(k in predictions) for k in names]
    truth = {k for k,v in zip(names,yt) if v == 1}; known = {k for k,v in zip(names,yt) if v >= 0}
    fp,fn = (predictions-truth)&known,truth-predictions
    # Use actual submitted prompt excerpts, not the full evidence bank.
    prompt = row.get('prompt','')
    evidence = '\n'.join(line for line in prompt.splitlines() if re.match(
        r'^(?:ECG(?: machine interpretation)?|CXR(?: CheXpert labels)?|CXR labels):',line))
    context_labels = present(evidence,rules)
    sources = {a['anchor_id'] for a in context['evidence']}
    cited = [str(s) for item in row.get('evidence_trace',[]) if isinstance(item,dict) for s in item.get('source_ids',[])]
    applicable = row['structured_output_applicable']
    citation = ((sum(s in sources for s in cited)/len(cited)) if cited else (1.0 if not sources else 0.0)) if applicable else None
    return {**row,'reference_text':text,'reference_labels':sorted(truth),'predicted_labels':sorted(predictions),
        'y_true':yt,'y_pred':yp,'unknown_reference_labels':sorted(set(names)-known),
        'reference_discordant_labels':sorted(fp),'omitted_labels':sorted(fn),
        'reference_discordant_positive_rate':len(fp)/max(len(predictions&known),1),
        'omission_rate':len(fn)/max(len(truth),1),
        'context_supported_prediction_fraction':len(predictions&context_labels)/max(len(predictions),1),
        'context_attributable_discordance_fraction':len(fp&context_labels)/max(len(fp),1),
        'rougeL_f1':_rouge_l(text,row['generated_report']), 'valid_evidence_citation_fraction':citation,
        'structured_schema_compliant':bool(row.get('structured_schema_compliant')) if applicable else None,
        'processor_input_tokens':row.get('processor_input_tokens'),
        'evaluation_rule_version':ECG_RULE_VERSION if row['direction']=='ecg' else CXR_RULE_VERSION}


def _stats(frame):
    yt,yp = np.asarray(frame.y_true.tolist()),np.asarray(frame.y_pred.tolist())
    return np.column_stack([((yt==1)&(yp==1)).sum(1),((yt==0)&(yp==1)).sum(1),
        ((yt==1)&(yp==0)).sum(1),frame.reference_discordant_positive_rate.to_numpy(),frame.omission_rate.to_numpy()])


def _values(stats):
    tp,fp,fn,discordance,omission = stats.sum(0)
    return np.array([2*tp/max(2*tp+fp+fn,1),discordance/len(stats),omission/len(stats)])


def patient_bootstrap(left,right,replicates,seed):
    if len(left) != len(right) or not len(left):
        raise ValueError('Nonempty aligned pairs required')
    if left.subject_id.astype(str).tolist() != right.subject_id.astype(str).tolist():
        raise ValueError('Patient identities differ between paired rows')
    subjects = left.subject_id.astype(str).to_numpy(); unique = np.unique(subjects)
    clusters = [np.flatnonzero(subjects == p) for p in unique]
    ls,rs = _stats(left),_stats(right); observed = _values(ls)-_values(rs)
    if len(unique) < 2:
        return observed, [None]*3, [None]*3
    rng = np.random.default_rng(seed); draws=[]
    for _ in range(replicates):
        # Concatenation preserves repeated draws of the same patient cluster.
        ix = np.concatenate([clusters[i] for i in rng.integers(0,len(unique),len(unique))])
        draws.append(_values(ls[ix])-_values(rs[ix]))
    return observed,np.quantile(draws,.025,axis=0),np.quantile(draws,.975,axis=0)


def paired_comparisons(details,cells,config):
    frame = pd.DataFrame(details); planned = pd.DataFrame(cells); results=[]
    if frame.empty:
        return pd.DataFrame(columns=['comparison_type','comparison','model_a','model_b',
            'direction','method','condition_a','condition_b','planned_common_cases',
            'paired_evaluable_successes','metric','difference_a_minus_b','ci_low','ci_high'])

    def compare(left,right,a_cells,b_cells,metadata,within):
        expected = sorted(set(a_cells.sample_id)&set(b_cells.sample_id))
        common = sorted(set(left.sample_id)&set(right.sample_id)&set(expected))
        contract_excluded=0
        if common:
            left = left.set_index('sample_id').loc[common]; right = right.set_index('sample_id').loc[common]
            valid = left.subject_id.astype(str).eq(right.subject_id.astype(str))
            if within:
                valid &= left.generation_protocol_signature.eq(right.generation_protocol_signature)
            else:
                valid &= left.context_signature.eq(right.context_signature)
            for sample in common:
                if left.loc[sample,'input_scope'] == right.loc[sample,'input_scope'] == 'raw_sensor_plus_retrieval':
                    valid.loc[sample] &= bool(left.loc[sample].get('query_image_sha256')) and left.loc[sample].get('query_image_sha256') == right.loc[sample].get('query_image_sha256')
            contract_excluded = int((~valid).sum())
            left,right = left.loc[valid],right.loc[valid]
        else:
            left,right = left.iloc[:0],right.iloc[:0]
        base = {**metadata,'planned_common_cases':len(expected),'paired_evaluable_successes':len(left),
            'excluded_from_paired_quality':len(expected)-len(left),'contract_excluded':contract_excluded,
            'n_patients':int(left.subject_id.nunique()) if len(left) else 0,
            'successful_a':int(a_cells.status.eq('successful').sum()),'successful_b':int(b_cells.status.eq('successful').sum()),
            'analysis_status':'exploratory; no multiplicity adjustment','resampling_unit':'patient',
            'contract':'same generator protocol; evidence condition varies' if within else
                       'same context/query; configured pipelines differ in model, input capability, prompt and evidence budget'}
        observed,lo,hi = patient_bootstrap(left,right,int(config['bootstrap_replicates']),int(config['bootstrap_seed'])) if len(left) else ([None]*3,[None]*3,[None]*3)
        for j,metric in enumerate(('micro_f1','reference_discordant_positive_rate','omission_rate')):
            results.append({**base,'metric':metric,'difference_a_minus_b':observed[j],
                            'ci_low':lo[j],'ci_high':hi[j], 'better_direction':'higher' if j==0 else 'lower'})

    for (model,direction,method), group in planned.groupby(['model_key','direction','method']):
        scores = frame[(frame.model_key==model)&(frame.direction==direction)&(frame.method==method)]
        for label,(a,b) in WITHIN_CONTRASTS.items():
            ac,bc = group[group.condition==a],group[group.condition==b]
            if ac.empty or bc.empty:
                continue
            compare(scores[scores.condition==a],scores[scores.condition==b],ac,bc,
                    dict(comparison_type='within_model',comparison=label,model_a=model,model_b=model,
                         direction=direction,method=method,condition_a=a,condition_b=b),True)
    for (direction,method,condition), group in planned.groupby(['direction','method','condition']):
        for a,b in combinations(sorted(group.model_key.unique()),2):
            scores = frame[(frame.direction==direction)&(frame.method==method)&(frame.condition==condition)]
            compare(scores[scores.model_key==a],scores[scores.model_key==b],group[group.model_key==a],group[group.model_key==b],
                    dict(comparison_type='cross_pipeline',comparison='configured_pipeline_comparison',model_a=a,model_b=b,
                         direction=direction,method=method,condition_a=condition,condition_b=condition),False)
    return pd.DataFrame(results)


def evaluate(out=EXPERIMENT4_ROOT,allow_incomplete=False):
    config=configuration(); root=analysis_path(out,config); bundle=collect(out,config)
    _check_freeze(root,bundle['inventory'])
    coverage=_coverage(bundle)
    if coverage.pending.sum() and not allow_incomplete:
        raise RuntimeError('Unattempted cells remain. Finish generation or explicitly allow preliminary analysis.')
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'analysis_manifest.json').exists():
        _write(root/'analysis_manifest.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'inventory':bundle['inventory'],'interpretation':'Exploratory analysis freeze, not retrospective prespecification'})
    lookup={c['context_signature']:c for c in bundle['contexts']}; details=[]; excluded=[]
    for row in bundle['successes']:
        scored=_score(row,lookup[row['context_signature']])
        if scored is None:
            excluded.append({**{k:row[k] for k in GROUPS},'sample_id':row['sample_id'],'reason':'no_known_reference_labels'})
        else:
            details.append(scored)
    _write_rows(root/'detailed.jsonl',details)
    pd.DataFrame(details).to_csv(root/'detailed.csv',index=False)
    coverage.to_csv(root/'generation_coverage.csv',index=False)
    pd.DataFrame(bundle['cells']).to_csv(root/'generation_cells.csv',index=False)
    pd.DataFrame(excluded,columns=GROUPS+['sample_id','reason']).to_csv(root/'reference_exclusions.csv',index=False)
    detail_lookup={(r['model_key'],r['context_signature']):r for r in details}
    frame=pd.DataFrame(details); summaries=[]; per_labels=[]
    for values,group in pd.DataFrame(bundle['cells']).groupby(GROUPS):
        model,direction,method,condition=values
        selected=[detail_lookup[(model,r.context_signature)] for r in group.itertuples() if (model,r.context_signature) in detail_lookup]
        eligible=[r for r in group.itertuples() if any(v>=0 for v in _reference(lookup[r.context_signature])[2])]
        exact=sum(all(t<0 or t==p for t,p in zip(r['y_true'],r['y_pred'])) for r in selected)
        summary={**dict(zip(GROUPS,values)),'selected_run':config['model_runs'][model],
            'application_priority':'primary' if direction=='ecg' else 'secondary',
            'model_role':ALL_MODELS[model]['role'],'input_scope':'retrieval_text_only' if ALL_MODELS[model]['mode']=='text_only' else 'raw_sensor_plus_retrieval',
            'training_overlap_warning':ALL_MODELS[model].get('training_overlap','foundation-model overlap unverified'),
            'planned':len(group),'successful':int(group.status.eq('successful').sum()),'failed':int(group.status.eq('failed').sum()),
            'pending':int(group.status.eq('pending').sum()),'evaluable_successes':len(selected),
            'availability_rate':float(group.status.eq('successful').mean()), 'quality_denominator':'evaluable successful reports only',
            'reference_eligible_planned':len(eligible),
            'usable_exact_reference_agreement_yield':exact/len(eligible) if eligible else None,
            'yield_sensitivity':'failure/pending is not a usable exact report, not an incorrect diagnosis; exploratory',
            'structured_output_applicable':bool(group.structured_output_applicable.iloc[0])}
        if selected:
            part=pd.DataFrame(selected); names=ECG_LABELS if direction=='ecg' else CXR_LABELS
            summary.update(_metrics(part,names))
            known_label_f1=[]
            if not summary['structured_output_applicable']:
                summary.update(valid_evidence_citation_fraction=None,structured_schema_compliance_rate=None)
            for j,label in enumerate(names):
                yt=np.array([r['y_true'][j] for r in selected]); yp=np.array([r['y_pred'][j] for r in selected])
                tp=int(((yt==1)&(yp==1)).sum()); fp=int(((yt==0)&(yp==1)).sum()); fn=int(((yt==1)&(yp==0)).sum())
                known=int((yt>=0).sum())
                if known:
                    known_label_f1.append(2*tp/max(2*tp+fp+fn,1))
                per_labels.append({**dict(zip(GROUPS,values)),'label':label,'known_reference_n':known,
                    'positive_reference_n':int((yt==1).sum()),'tp':tp,'fp':fp,'fn':fn,
                    'precision':tp/max(tp+fp,1) if known else None,'recall':tp/max(tp+fn,1) if known else None,
                    'f1':2*tp/max(2*tp+fp+fn,1) if known else None})
            summary['macro_f1']=float(np.mean(known_label_f1)) if known_label_f1 else None
        summaries.append(summary)
    summary=pd.DataFrame(summaries); summary.to_csv(root/'metrics_summary.csv',index=False)
    summary[summary.direction.eq('ecg')].to_csv(root/'primary_ecg_results.csv',index=False)
    summary[summary.direction.eq('cxr')].to_csv(root/'secondary_cxr_results.csv',index=False)
    pd.DataFrame(per_labels).to_csv(root/'per_label_results.csv',index=False)
    paired=paired_comparisons(details,bundle['cells'],config)
    paired.to_csv(root/'paired_comparisons.csv',index=False)
    _readiness(root,out,bundle,details)
    return summary


def _readiness(root,out,bundle,details):
    items=[('IV.A cohort flow','partition/global_partition_audit.json'),
           ('IV.A lab decoding','evidence/lab_encoding_audit.json'),
           ('IV.B bridge retrieval','bridge/retrieval_evaluation.csv'),
           ('IV.B compatibility gates','bridge/confidence_gates.json')]
    readiness=[]
    for section,path in items:
        source=Path(out)/path
        readiness.append({'section':section,'status':'artifact_available_not_independently_verified' if source.exists() else 'missing',
                          'source':str(source),'sha256':sha256(source) if source.exists() else None})
    for section,note in [('IV.D broken-linkage control','Shuffled training is not an inference-time broken-linkage experiment.'),
        ('IV.D ECG-only retriever','Same-only evidence does not establish ECG-only retrieval.'),
        ('IV.F efficiency','Token counts are descriptive costs; controlled latency/GPU memory/energy are not inferred.'),
        ('IV.F missing-evidence robustness','Same-only/same-labs are prompt component ablations; mask-only does not remove lab text.'),
        ('IV.E clinician assessment','Pending two-reviewer assessment and adjudication; no automated hallucination claim.')]:
        readiness.append({'section':section,'status':'not_established_by_this_analysis','source':note,'sha256':None})
    pd.DataFrame(readiness).to_csv(root/'manuscript_readiness.csv',index=False)
    tokens=[]
    for values,group in pd.DataFrame(bundle['cells']).groupby(GROUPS):
        rows=[r for r in details if tuple(r[k] for k in GROUPS)==values]
        for field in ('processor_input_tokens','generated_token_count'):
            observed=[r[field] for r in rows if isinstance(r.get(field),(int,float))]
            tokens.append({**dict(zip(GROUPS,values)),'measurement':field,'available_n':len(observed),
                           'median':float(np.median(observed)) if observed else None,
                           'scope':'evaluable successful reports; excludes failure/retry cost'})
    pd.DataFrame(tokens).to_csv(root/'token_cost_descriptive.csv',index=False)
    _write(root/'interpretation.json',{'primary_endpoint':'diagnostic micro-F1 agreement with imperfect references',
        'primary_application':'ECG','secondary_application':'CXR','comparisons':'within-model first; cross-pipeline secondary',
        'clinical_safety':'not established by automated discordance or computational thresholds',
        'causality':'admission linkage establishes co-occurrence, not causation',
        'pulse':'plain-report prompt; structured schema and attribution metrics not applicable',
        'missing_references':'unknown CXR states masked; positive-only lists do not establish negatives',
        'multiplicity':'exploratory contrasts, no multiplicity adjustment',
        'failures':'reported separately; conditional quality and exploratory failure-inclusive exact-agreement yield'})


def _frozen(out):
    config=configuration();root=analysis_path(out,config)
    if not (root/'analysis_manifest.json').exists():
        raise RuntimeError('Run Notebook 14 before clinical review')
    _check_freeze(root,collect(out,config)['inventory'])
    return config,root


def prepare_review(out=EXPERIMENT4_ROOT,per_stratum=5):
    if per_stratum<1:
        raise ValueError('Review sample size must be positive')
    config,root=_frozen(out); review=root/'clinical_review'
    if review.exists():
        raise FileExistsError('Clinical review directory already exists; reviewer work will not be overwritten')
    details=_rows(root/'detailed.jsonl'); frame=pd.DataFrame(details)
    if frame.empty:
        raise ValueError('No evaluable reports for clinical review')
    a,b=config['review_contrast']; rng=np.random.default_rng(config['review_seed']); selected=[]
    for values,group in frame.groupby(['model_key','direction','method']):
        left,right=group[group.condition==a],group[group.condition==b]
        common=sorted(set(left.sample_id)&set(right.sample_id))
        if not common:
            continue
        chosen=rng.choice(common,min(per_stratum,len(common)),replace=False)
        selected.extend(group[group.sample_id.isin(chosen)&group.condition.isin([a,b])].to_dict('records'))
    if not selected:
        raise ValueError('No matched successful pairs for the configured review contrast')
    from e4_generation import query_image
    context_lookup={c['context_signature']:c for c in _rows(Path(out)/'contexts/bidirectional_contexts.jsonl')}
    review.mkdir(parents=True); packet=[]; keys=[]
    for index in rng.permutation(len(selected)):
        row=selected[index]; review_id=f'JBHIR{len(packet)+1:05d}'
        sensor = row.get('query_image') or str(query_image(context_lookup[row['context_signature']],Path(out)))
        packet.append({'review_id':review_id,'direction':row['direction'],'query_image':sensor,
            'reference_text':row['reference_text'],'retrieved_evidence':row.get('prompt',''),
            'generated_report':row['generated_report'],'evidence_trace':json.dumps(row.get('evidence_trace',[])),
            'evidence_trace_applicable':row['structured_output_applicable'],
            **{f:('NA' if f=='evidence_trace_supported' and not row['structured_output_applicable'] else '') for f in REVIEW_FIELDS},
            'reviewer_id':'','notes':''})
        keys.append({'review_id':review_id,**{k:row[k] for k in GROUPS},'sample_id':row['sample_id'],
                     'subject_id':row['subject_id'],'selected_run':row['selected_run'],
                     'context_signature':row['context_signature'],'generation_protocol_signature':row['generation_protocol_signature']})
    pd.DataFrame(packet).to_csv(review/'blinded_review.csv',index=False)
    for n in (1,2):
        pd.DataFrame(packet).to_csv(review/f'reviewer_{n}.csv',index=False)
    pd.DataFrame(keys).to_csv(review/'private_key.csv',index=False)
    _write(review/'review_protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'contrast':[a,b],
        'sampling':'matched examinations within model/direction/method; randomized individual report presentation',
        'per_stratum_pairs':per_stratum,'seed':config['review_seed'],'rows':len(packet),
        'packet_sha256':sha256(review/'blinded_review.csv'),'key_sha256':sha256(review/'private_key.csv'),
        'blinding_limit':'model/method/condition labels hidden; prompt content and applicability may reveal pipeline',
        'analysis':'exploratory; clinically unsupported findings require qualified reviewers and adjudication'})
    return len(packet)


def summarize_review(out=EXPERIMENT4_ROOT):
    _,root=_frozen(out); review=root/'clinical_review'; protocol=_json(review/'review_protocol.json')
    final_manifest=review/'final_review_manifest.json'
    if final_manifest.exists():
        for filename,digest in _json(final_manifest).items():
            if not (review/filename).exists() or sha256(review/filename)!=digest:
                raise ValueError('Finalized reviewer inputs changed; use a new analysis ID')
    if sha256(review/'blinded_review.csv')!=protocol['packet_sha256'] or sha256(review/'private_key.csv')!=protocol['key_sha256']:
        raise ValueError('Frozen review packet/key changed')
    packet=pd.read_csv(review/'blinded_review.csv',dtype=str,keep_default_na=False).set_index('review_id')
    key=pd.read_csv(review/'private_key.csv',dtype=str,keep_default_na=False)
    reviewers=[]
    for n in (1,2):
        frame=pd.read_csv(review/f'reviewer_{n}.csv',dtype=str,keep_default_na=False)
        if frame.review_id.duplicated().any() or set(frame.review_id)!=set(packet.index):
            raise ValueError('Reviewer IDs do not match frozen packet')
        frame=frame.set_index('review_id').loc[packet.index]
        if frame.reviewer_id.eq('').any():
            raise ValueError('Reviewer identities are required')
        for col in packet.columns:
            if col not in REVIEW_FIELDS+['reviewer_id','notes'] and not frame[col].eq(packet[col]).all():
                raise ValueError(f'Reviewer changed frozen field {col}')
        for field in REVIEW_FIELDS:
            for review_id,value in frame[field].items():
                na=field=='evidence_trace_supported' and packet.loc[review_id,'evidence_trace_applicable'].lower()=='false'
                allowed={'NA'} if na else ({'1','2','3','4','5'} if field.endswith('1_to_5') else {'Y','N','NI'})
                if value not in allowed:
                    raise ValueError(f'Incomplete/invalid {field} for {review_id}: allowed {sorted(allowed)}')
        reviewers.append(frame)
    one,two=reviewers
    if one.reviewer_id.eq(two.reviewer_id).any():
        raise ValueError('Two distinct reviewer identities are required per report')
    disagreements=[];agreement=[]
    for field in REVIEW_FIELDS:
        use=one[field].ne('NA')
        agreement.append({'field':field,'applicable_n':int(use.sum()),
                          'raw_agreement':float(one.loc[use,field].eq(two.loc[use,field]).mean()) if use.any() else None})
    for rid in packet.index:
        differing=[f for f in REVIEW_FIELDS if one.loc[rid,f]!=two.loc[rid,f]]
        if differing:
            disagreements.append({'review_id':rid,'disagreeing_fields':'|'.join(differing),
                **{f'{f}_r1':one.loc[rid,f] for f in REVIEW_FIELDS},**{f'{f}_r2':two.loc[rid,f] for f in REVIEW_FIELDS},
                **{f'{f}_adjudicated':('' if f in differing else one.loc[rid,f]) for f in REVIEW_FIELDS},
                'adjudicator_id':'','adjudication_notes':''})
    path=review/'adjudication.csv'; final=one[REVIEW_FIELDS].copy()
    pd.DataFrame(agreement).to_csv(review/'reviewer_agreement.csv',index=False)
    if disagreements and not path.exists():
        pd.DataFrame(disagreements).to_csv(path,index=False)
    complete=not disagreements
    if disagreements:
        adj=pd.read_csv(path,dtype=str,keep_default_na=False)
        if adj.review_id.duplicated().any() or set(adj.review_id)!={r['review_id'] for r in disagreements}:
            raise ValueError('Adjudication IDs differ from disagreements')
        complete=True
        for row in adj.to_dict('records'):
            rid=row['review_id']
            for field in REVIEW_FIELDS:
                value=row.get(f'{field}_adjudicated','')
                if not value:
                    complete=False;continue
                na=field=='evidence_trace_supported' and packet.loc[rid,'evidence_trace_applicable'].lower()=='false'
                allowed={'NA'} if na else ({'1','2','3','4','5'} if field.endswith('1_to_5') else {'Y','N','NI'})
                if value not in allowed:
                    raise ValueError('Invalid adjudicated rating')
                if one.loc[rid,field]==two.loc[rid,field] and value!=one.loc[rid,field]:
                    raise ValueError('Adjudication must preserve agreed ratings')
                final.loc[rid,field]=value
            if not row.get('adjudicator_id'):
                complete=False
            elif row['adjudicator_id'] in {one.loc[rid,'reviewer_id'],two.loc[rid,'reviewer_id']}:
                raise ValueError('Adjudication requires a third reviewer')
    if complete:
        joined=final.reset_index().merge(key,on='review_id',validate='one_to_one')
        joined.to_csv(review/'adjudicated_results_private.csv',index=False)
        summary=[]
        for values,group in joined.groupby(GROUPS):
            for field in REVIEW_FIELDS:
                use=group[group[field]!='NA']
                for rating,count in use[field].value_counts().items():
                    summary.append({**dict(zip(GROUPS,values)),'field':field,'rating':rating,'count':int(count),
                                    'applicable_n':len(use),'proportion':count/len(use)})
        pd.DataFrame(summary).to_csv(review/'clinical_review_summary.csv',index=False)
        inputs=['reviewer_1.csv','reviewer_2.csv']+(['adjudication.csv'] if disagreements else [])
        _write(final_manifest,{name:sha256(review/name) for name in inputs})
    report={'rows':len(packet),'disagreements':len(disagreements),
            'status':'complete' if complete else 'awaiting_adjudication'}
    _write(review/'review_status.json',report)
    return report
