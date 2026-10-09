# Experiment 4: ECG–CXR–laboratory retrieval and report generation

This is the current trimodal workflow. Run notebooks from this directory, or set
`EXPERIMENT4_CODE_ROOT` to its absolute path. The older root `notebooks/14`–`20`
are not replacements for the current analysis notebooks here.

The source folder is named `experiment`. Existing `EXPERIMENT4_*` configuration
names and the default output subdirectory `experiment4` are retained for
compatibility with prior runs; this source-folder rename does not move results
or change patient-split seeds.

## Setup

On a Linux GPU node, create a main Python environment (Python 3.12 was used in
the source workflow), then install:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel ipywidgets
python -m ipykernel install --user --name ecg-trimodal --display-name "ECG trimodal"
```

Requirements preserve source compatibility bounds, not an exact original lockfile.
Validate model classes and GPU compatibility before full runs. Do not upgrade
a running generation environment in place.

From the repository root, copy `config/experiment.environment.example.sh` to
`config/experiment.environment.sh`, replace placeholders, and source it before
launching Jupyter. The private copy is ignored by Git.

```bash
source config/experiment.environment.sh
cd experiment
jupyter lab
```

| Variable | Meaning |
|---|---|
| `EXPERIMENT4_CODE_ROOT` | This source directory |
| `ECG_CXR_BRIDGE_ARTIFACT_ROOT` | Existing upstream manifests and frozen features |
| `EXPERIMENT4_ROOT` | Dedicated output directory, defaulting to the upstream root's `experiment4` child |
| `SYMILE_ROOT` | Extracted Symile-MIMIC release |
| `HF_HOME` | Hugging Face cache on project storage |
| `HF_TOKEN_FILE` | Optional private one-token text file outside Git |
| `PULSE_REPO_ROOT` | Official PULSE checkout, for 13C |
| `E4_IDENTITY_CROSSWALK` | Optional verified private CSV crosswalk; never distribute |

Unset core paths resolve to ignored repository `artifacts/`, `data/`, and `.cache/`
directories. These defaults do not supply data. All configured core paths must be
absolute. Select your own Biowulf allocation and storage; no account-specific
paths or scheduler resources are embedded. Private manifest paths must also be
valid on the execution machine.

Authentication reads an explicitly configured token file, then environment or
cached tokens, then a hidden prompt. Alternatively use `hf auth login`. Accept
gated-model access terms first. Never paste tokens into notebook source.

### Specialist environments

- PULSE (13C): install the official AIMedLab/PULSE LLaVA runtime in a separate
  Python 3.10 environment and use `requirements-pulse-extra.txt`. Set
  `PULSE_REPO_ROOT`; do not install main Transformers requirements there.
  This generation implementation feeds PULSE rendered ECG images.
- ECG-Instruct (13D): use `requirements-ecg-instruct.txt` separately. Its current
  loader is Transformers `MllamaForConditionalGeneration`, not Unsloth.
  Four-bit loading requires bitsandbytes and a supported GPU.
- Qwen/MedGemma/GPT-OSS use the main compatible runtime. GPT-OSS consumes
  retrieved text, not query images. Keep all model protocols and revisions.

## Inputs

The original `../notebooks/` workflow prepares upstream artifacts. Notebooks
00–04 prepare Symile manifests and ECG/CXR features; 09–13 prepare external
manifests and foundation features. Notebook 13's projection stages may require
the original bridge checkpoints: follow its dependencies, not just these ranges.
Existing compatible frozen artifacts can be reused without rerunning old analyses.

Under `ECG_CXR_BRIDGE_ARTIFACT_ROOT`, provide:

```text
config.json
manifests/{train,val,test}.jsonl
features/{train,val,test}/ecg_hubert.npy
features/{train,val,test}/cxr_raddino.npy
external/ecg/manifest.jsonl
external/ecg/ecg_foundation.npy
external/ecg/encoding_status.npy
external/cxr/manifest.jsonl
external/cxr/cxr_foundation.npy
external/cxr/encoding_status.npy
```

Manifests must preserve array row order and verified subject/admission/study
identities. External manifests contain ECG waveform paths and machine diagnoses,
or CXR image/JSON paths, reports and labels. Referenced raw files must be accessible.

Under `SYMILE_ROOT`, provide `train.csv`, `val.csv`, `test.csv`, plus
`data_npy/{split}/labs_percentiles_{split}.npy`,
`labs_missingness_{split}.npy`, and `hadm_id_{split}.npy`. Lab ordering is checked
against `labs_means.json` or all 50 percentile columns. The shipped lab-code JSON
is public metadata, not patient measurements.

CXR preprocessing supports `Caption` reports and `labels` in image–JSON pairs;
local bounding boxes are not report text. Numeric filenames are not sufficient
proof of patient identity. Notebook 01 prioritizes retaining Symile patients,
links source evidence before excluding overlapping external queries, and
quarantines unverifiable identities. Do not suppress leakage checks.

## Run order

| Order | Notebook(s) | Purpose |
|---|---|---|
| 1 | 00 | Validate roots, inputs and protocol |
| 2 | 01 | Global patient partition and verified evidence links |
| 3 | 02 | Select disjoint external frozen features |
| 4 | 03, 03B | Prepare Symile trimodal data; audit lab decoding |
| 5 | 04, 05, 06 | Train Continuous, VQ and Hybrid bridges and controls |
| 6 | 07 | Evaluate retrieval, select checkpoints, calibrate gates |
| 7 | 08 | Build linked training-admission evidence banks |
| 8 | 09, 10 | Construct bidirectional contexts and validate prompts |
| 9 | 11, 12, 13, 13C, 13D | Generate each model's supported directions in its correct kernel |
| 10 | 13E | Audit selected runs, duplicate cells, failures and pending cases |
| 11 | 14 | Freeze stationary inputs and evaluate the five-model workflow |
| 12 | 15 | Create blinded independent review forms and private linking key |
| 13 | 16 | Summarize reviews; adjudicate disagreements and rerun to finalize |
| 14 | 17 | Freeze extension cohort; compare raw ECG, bridge ECG-only, trimodal, broken-linkage and copying controls |
| 15 | 18 | Prepare and visually inspect paired legacy/calibrated ECG images |
| 16 | 19 | Qwen extension pilot, then complete unchanged run |
| 17 | 20 | Revised PULSE prompt and rendering comparison in an isolated snapshot |
| 18 | 21 | Evaluate revised PULSE and original Qwen extension runs together |

**Before notebook 20:** its provenance-preserving preparation requires the original
PULSE pilot protocol. Fresh installations must create that pilot first; follow
[the extension instructions](../docs/extension_workflow.md) for the exact setup.
Do not reuse old private output protocols after changing packaged source code:
source hashes are intentionally checked on resume.

04–06 can run in separate allocations after preparation. Models can run
separately after 10, but never run concurrent writers for the same model/run.
**11B/12B/13B are historical strict-JSON rescue notebooks, disabled for generation-v3
and not part of this run order.** They are retained only for provenance.

## Comparisons

| Description | Code key | Role |
|---|---|---|
| Continuous contrastive bridge | `clip` | Continuous trimodal alignment |
| Vector-quantized bridge | `codebind` | CodeBind-inspired compositional-codebook ablation, not an exact reproduction |
| Hybrid bridge | `combined` | Continuous and quantized representations |

All bridges use ECG, CXR and labs. Controls include shuffled pairing, mask-only
labs and no-code-match variants where applicable. Notebook 09 records each
evidence condition and signature; there is no assumed codebook advantage.

| Generator | ECG | CXR | Role |
|---|---|---|---|
| Qwen3.5-4B | Yes | Yes | General-purpose vision-language model |
| MedGemma-1.5-4B | No | Yes | CXR-focused medical model |
| GPT-OSS-20B | Yes | Yes | Retrieved-text-only model |
| PULSE-7B | Yes | No | ECG-image specialist, compact report prompt |
| ECG-Instruct-Llama-3.2-11B | Yes | No | ECG-image specialist |

This is not an interchangeable five-model ECG leaderboard. Downstream
patient-disjointness does not establish foundation-model pretraining-disjointness.
Laboratory percentiles are not abnormality thresholds. Retrieved admissions are
other patients' context, never authoritative query findings.

## Pilots and interruption recovery

Generation defaults to 50 new pilot attempts per call. Inspect output quality,
format, token budgets and image loading before setting `E4_MAX_GENERATION_ROWS=0`
(general models) or `E4_MAX_SPECIALIST_ROWS=0` (specialists). Zero removes the
per-call limit; the total depends on eligible queries, directions and conditions.
A JSONL line is an attempt, not necessarily a unique successful report.

Runs append attempts and retain protocol/progress files. Unchanged runs skip
both recorded successes and recorded failures on restart. Failures are audited,
not treated as successes or retried indefinitely. Changed contexts, code,
prompts, model revisions or settings require a new run name.

Portable-source edits change hashes compared with private source copies. Use a
fresh run for new generation with this release. To analyze existing runs, preserve
their original JSONLs and adjacent protocols/manifests and use a new analysis ID
if analysis code or frozen settings changed.

Final analysis selects inputs using `analysis_model_runs.json`, **not**
`E4_GENERATION_RUN`: `generation_v3` for four models and `pulse_compact_v1` for
PULSE by default. 13C explicitly sets the compact run name. Update the map if
your run names differ. See [analysis details](JBHI_ANALYSIS_UPDATE.md).

## Outputs and review

Outputs are beneath `EXPERIMENT4_ROOT`:

```text
partition/                             splits and exclusions
bridge/                                prepared arrays and retrieval experiments
evidence/                              banks and laboratory audits
contexts/                              bidirectional retrieval contexts
generations/runs/<run>/                 JSONLs, protocols, manifests, progress
query_images/                          protected query images
analysis/<analysis_id>/                 audits, metrics, comparisons
analysis/<analysis_id>/clinical_review/ reviewer forms, adjudication, private key
```

Preserve whole run directories. Never publish these outputs or overwrite filled
review packets. Final analysis requires no pending cases; terminal failures stay
in availability denominators. Use `E4_ALLOW_INCOMPLETE=1` and a separate analysis
ID only for explicitly preliminary snapshots. Stop generation before freezing
inputs in 14. Complete both independent forms from 15, run 16, fill the requested
third-reviewer adjudication, and rerun 16.

Report diagnostic agreement, omissions, reference discordance, availability,
paired patient-bootstrap comparisons and human review together. Reference
discordance is not clinically adjudicated hallucination. The supplied analysis
is exploratory, with 1,000 default bootstrap replicates and no multiplicity
adjustment. See the root README for synthetic-test commands and licensing.
