# Five-model JBHI analysis update

## Upload and run

Upload the four notebooks 13E, 14, 15 and 16, `e4_jbhi_analysis.py`, and
`analysis_model_runs.json` into your existing Biowulf experiment code directory.
The analysis-only ZIP contains these six required files and this guide.
Existing `e4_common.py`, `e4_model_registry.py`, `e4_evaluation.py` and
`e4_generation.py` are reused, not replaced. Restart the analysis kernels.
No regeneration is required by this analysis update. Running generation jobs
are unaffected because their modules and notebooks are unchanged.

The artifact root remains:
`/path/to/ecg_cxr_bridge_artifacts/experiment4`

The configuration selects exactly these files relative to that root:

| Model | Selected JSONL |
|---|---|
| Qwen3.5-4B | generations/runs/generation_v3/qwen35_4b.jsonl |
| MedGemma-1.5-4B | generations/runs/generation_v3/medgemma15_4b.jsonl |
| GPT-OSS-20B | generations/runs/generation_v3/gpt_oss_20b.jsonl |
| ECG-Instruct-Llama-3.2-11B | generations/runs/generation_v3/ecg_instruct_llama32_11b.jsonl |
| PULSE-7B | generations/runs/pulse_compact_v1/pulse7b_ecg.jsonl |

Each file must retain its neighboring `<model>_protocol.json` and
`<model>_run_manifest.json`. No legacy, rescue, or alternative-run discovery is
performed. If the actual selected run name differs, edit the configuration before
analysis; do not concatenate JSONLs or move them away from their manifests.

1. Run 13E to audit coverage and selected protocols. A JSONL line is an attempt,
   not necessarily a unique usable report. Duplicate cells, failures, pending
   cells and excluded records are reported separately.
2. Once generation has stopped, run 14 to freeze input hashes and evaluate.
   Terminal failures are allowed and remain in availability denominators.
   Pending cells block final analysis. `E4_ALLOW_INCOMPLETE=1` explicitly allows
   preliminary analysis; use a separate `E4_ANALYSIS_ID` for that snapshot.
3. Run 15 once to create matched full-evidence versus same-sensor review pairs,
   blinded presentation, two reviewer forms and a private linking key. Set
   `E4_REVIEW_PER_STRATUM` in the notebook for the intended sample size.
4. Complete both reviewer forms, then run 16. Disagreements create an adjudication
   form for a third reviewer. Complete it and rerun 16. Existing review packets
   are not overwritten, and finalized reviewer inputs are hash-locked.

Default output directory:
`/path/to/ecg_cxr_bridge_artifacts/experiment4/analysis/jbhi_five_models_v1/`

Use the same `E4_ANALYSIS_ID` in all four notebooks. If selected files, contexts,
analysis code or settings change after freezing, choose a new analysis ID.
`E4_GENERATION_RUN` does not select inputs to these notebooks. The audit can be
repeated before freezing, but actively changing files may be rejected; a final
analysis must use stationary inputs.

## Comparisons and interpretation

- ECG primary: Qwen, GPT-OSS, ECG-Instruct and PULSE. CXR secondary: Qwen,
  MedGemma and GPT-OSS. GPT-OSS is a retrieval-text-only pipeline, not a sensor
  reader. There is no interchangeable five-model ECG leaderboard.
- Within-model full linked evidence versus same-sensor and random evidence
  comparisons are primary. Other available conditions are exploratory.
- Cross-pipeline comparisons pair common query examinations but allow the
  declared model/prompt/evidence-budget differences. They do not isolate model
  architecture effects. Both sensor pipelines must agree on query-image hashes.
- Confidence intervals resample patients, retaining all selected examinations
  and repeated cluster draws. Default: 1,000 replicates. Intervals are unavailable
  with fewer than two patients. No multiplicity adjustment is applied.
- Automated quality is conditional on evaluable successful reports. Availability
  and an explicitly exploratory failure-inclusive usable exact-agreement yield
  are separate. Failures are not assigned diagnoses or counted as hallucinations.
- PULSE compact prompts request plain reports. Structured-schema and evidence-
  attribution metrics are not applicable to them, including in review forms.
- Unknown CXR labels are masked. A positive-label-only list does not establish
  negative findings. ECG machine text and CXR labels are imperfect references;
  automated reference discordance is not an adjudicated hallucination.
- Human review is exploratory and paired within each model/direction/method.
  Hidden labels do not guarantee complete blinding: prompt style and applicability
  fields may reveal the pipeline. Qualified reviewers must inspect the sensor.
- Admission linkage establishes co-occurrence, not causality. Cohort/lab/bridge
  audit artifacts are indexed, not independently revalidated by this module.
- Broken-linkage controls, genuine ECG-only retrieval, controlled efficiency and
  missing-evidence robustness are not inferred from different experiments.
  `manuscript_readiness.csv` explicitly lists these limitations. Token counts do
  not substitute for measured latency, memory, energy or total retry costs.
- Outputs were already inspected during development: this is an exploratory
  analysis freeze, not a retrospective claim of prespecification.

## Outputs and validation

Download the entire selected analysis directory plus both selected generation
directories with protocols/manifests. `list_of_results.txt` lists the tables.
Patient-level records, sensor paths and private review keys must remain in
approved access-controlled storage.

Local tests exercise synthetic generation/protocol files, duplicate handling,
missing runs, terminal failures, reference masking, patient bootstraps, frozen
inputs and adjudication. Full Biowulf outputs and model execution have not been
validated locally by this update. Run 13E first to detect deployment differences.

Validation on September 30, 2026: 67 tests passed and one upstream-repository
test was skipped. The four notebooks also passed nbformat and Python 3.10
syntax validation.
