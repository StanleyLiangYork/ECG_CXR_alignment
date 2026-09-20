# Cross-Sensor Multimodal RAG Integrating ECG and CXR Evidence for ECG Report Generation

**Zhaohui Liang, Niccolo Marini, Sivaramakrishnan Rajaraman, Zhiyun Xue, Sameer Antani**

National Library of Medicine, National Institutes of Health, Bethesda, Maryland, USA

## Overview

This repository implements a non-text ECG-CXR alignment and retrieval-augmented generation workflow. The central question is whether chest radiograph patterns that co-occur with an ECG during the same patient admission can provide useful context for ECG assessment and report generation.

The workflow has three evidence levels:

1. **Relational learnability:** train an ECG-CXR bridge on verified same-patient, same-admission pairs from Symile-MIMIC and test retrieval of the documented encounter partner.
2. **Semantic portability:** freeze the bridge and evaluate it on independent MIMIC-IV-ECG waveform-diagnosis and MIMIC-CXR image-report collections.
3. **Generative utility:** compare zero-shot generation with ECG-only, retrieved-CXR, ECG+CXR, random-CXR, and combined retrieval contexts for ECG report generation.

Continuous contrastive alignment is the reference bridge. Vector-quantized shared and shared-specific CodeBind-inspired variants are architectural ablations, not presumed improvements. A cross-patient shuffled-pair model is the negative control.

## Scientific scope

The Symile bridge is trained from ECG waveforms and chest radiographs only. It does not use ECG diagnoses, radiology reports, CheXpert labels, laboratory results, or other text targets during bridge training. Although Symile-MIMIC includes laboratory data, the present experiments use only its ECG-CXR relation.

A positive Symile pair establishes that the ECG and CXR came from the same patient admission; it does not imply that both sensors express the same diagnosis. The ECG is selected within 24 hours of admission and the CXR within 24 to 72 hours after admission. The bridge therefore learns clinical concurrence and encounter-level relation rather than global semantic equivalence.

The external ECG and CXR repositories are independent and do not provide a certified cross-repository patient identity link. External analyses measure within-modality retrieval and prespecified concept compatibility. They must not be interpreted as external same-patient ECG-CXR retrieval. Retrieved CXR examples used for generation come from other training patients and are non-authoritative context.

## Repository layout

```text
.
├── config/
│   └── environment.example.sh
├── notebooks/
│   ├── 00_environment_and_configuration.ipynb
│   ├── ...
│   └── 20_evaluate_rag_report_generation.ipynb
├── tests/
│   └── test_notebook_release.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements-biowulf.txt
```

The notebooks are the executable research record. Their outputs and execution counts are cleared in the repository. Patient data, embeddings, generated reports, checkpoints, model weights, and evaluation artifacts are intentionally excluded.

## Data access

The experiments require the following datasets:

- [Symile-MIMIC v1.0.0](https://physionet.org/content/symile-mimic/1.0.0/) for same-patient, same-admission ECG-CXR training and retrieval evaluation.
- [MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/) for external ECG waveforms and machine diagnoses.
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) or an authorized derivative of image-report pairs for external CXR evaluation. MIMIC-CXR is credentialed-access data; comply with its data-use agreement.

This repository does not distribute any dataset records. Users are responsible for obtaining access and complying with the applicable licenses and data-use agreements.

### Expected Symile layout

Set `SYMILE_ROOT` to the extracted Symile-MIMIC version directory. The notebooks expect:

```text
$SYMILE_ROOT/
├── train.csv
├── val.csv
├── val_retrieval.csv
├── test.csv
└── data_npy/
    ├── train/
    │   ├── ecg_train.npy
    │   ├── cxr_train.npy
    │   └── hadm_id_train.npy
    ├── val/
    ├── val_retrieval/
    └── test/
```

Notebook 01 verifies array completeness, CSV-to-array row identity, subject identity encoded in ECG and CXR paths, ten-candidate retrieval groups, and patient-disjoint training, validation, and test sets.

### Expected MIMIC-IV-ECG layout

Set `MIMIC_ECG_ROOT` to the release directory:

```text
$MIMIC_ECG_ROOT/
├── record_list.csv
├── machine_measurements.csv
└── files/
    └── p1000/
        └── p10000032/
            └── s40689238/
                ├── 40689238.hea
                └── 40689238.dat
```

Notebook 10 reconstructs canonical WFDB paths from subject and study identifiers, concatenates nonempty `report_0` through `report_17` fields as the machine diagnosis, removes unresolved records, and creates deterministic patient-disjoint splits.

### Expected external CXR layout

Set `MIMIC_CXR_TRAIN_ROOT` and `MIMIC_CXR_TEST_ROOT` to recursively searchable directories containing same-stem image and JSON pairs:

```text
$MIMIC_CXR_TRAIN_ROOT/
├── 50017442.jpg
└── 50017442.json
```

The JSON object should store the radiology report in `Caption` and CheXpert findings in `labels`. Key matching is case-insensitive, compatible report fallbacks are supported, and local bounding-box annotations are ignored. For the release used in this study, notebook 11 should discover **43,700 training images** before report filtering. If a subject occurs in both supplied folders, the test partition takes precedence for every record from that subject.

## Models

The notebooks download or load the following model families:

- [HuBERT-ECG](https://huggingface.co/Edoardo-BS/hubert-ecg-base) for ECG waveform features.
- [RAD-DINO](https://huggingface.co/microsoft/rad-dino) for CXR image features.
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) for external diagnosis and report text features.
- [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) and [Qwen3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B) for ECG report generation.

MedGemma access may require accepting its Hugging Face terms and authenticating before notebook 19. Model licenses remain separate from this repository's MIT License.

## Environment setup

The workflow was designed for a GPU-enabled HPC environment such as NIH Biowulf. Python 3.12 was used in the final run. Create and activate a virtual environment, then install the provided environment:

```bash
python3 -m venv /path/to/venvs/ecg-cxr
source /path/to/venvs/ecg-cxr/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-biowulf.txt
python -m ipykernel install --user --name ecg-cxr --display-name "ECG CXR alignment"
```

Install the CUDA-compatible PyTorch build recommended for the target cluster if the generic PyPI installation does not match the available driver.

Copy and edit the environment template:

```bash
cp config/environment.example.sh config/environment.sh
chmod 600 config/environment.sh
source config/environment.sh
```

`config/environment.sh` is ignored by Git. Do not commit access tokens, credentials, protected data paths, or generated patient-level artifacts. Authenticate to Hugging Face through its supported CLI or cluster secret mechanism rather than adding a token to the repository.

Start Jupyter from the repository root so notebook-relative behavior is consistent:

```bash
jupyter lab
```

Select the `ECG CXR alignment` kernel.

## Recommended execution order

### Stage 1: Symile pair verification and baseline bridge

| Order | Notebook | Purpose |
|---:|---|---|
| 00 | `00_environment_and_configuration.ipynb` | Freeze dataset, cache, model, split, and artifact configuration. |
| 01 | `01_verify_symile_ecg_cxr_pairs.ipynb` | Verify same-patient pairing, retrieval labels, row order, and patient separation. |
| 02 | `02_build_ecg_cxr_pair_manifests.ipynb` | Build row-aligned, non-text ECG-CXR manifests. |
| 03 | `03_encode_ecg_hubert.ipynb` | Encode Symile ECG arrays with frozen HuBERT-ECG. |
| 04 | `04_encode_cxr_raddino.ipynb` | Encode Symile CXR arrays with frozen RAD-DINO. |
| 05 | `05_define_ecg_cxr_codebind.ipynb` | Materialize and validate the reusable bridge module. |
| 06 | `06_train_ecg_cxr_bridge.ipynb` | Train the initial shared/specific bridge. |
| 07 | `07_evaluate_ecg_cxr_bridge.ipynb` | Evaluate bidirectional encounter retrieval. |

Notebooks 03 and 04 can run independently after notebook 02. Notebook 08 documents the external adapter contract and is informative; it is not required for the main run.

### Stage 2: Bridge and semantic ablations

| Order | Notebook | Purpose |
|---:|---|---|
| 09 | `09_external_ablation_design_and_configuration.ipynb` | Freeze external roots, variants, seeds, and text model. |
| 10 | `10_prepare_external_ecg_mimic.ipynb` | Build the external ECG waveform-machine-diagnosis manifest. |
| 11 | `11_prepare_external_cxr_mimic.ipynb` | Build leakage-controlled CXR image-report manifests. |
| 12 | `12_train_symile_bridge_ablations.ipynb` | Train five bridge variants over three seeds. |
| 13 | `13_encode_external_modalities_with_shared_codebook.ipynb` | Encode external ECG and CXR records through frozen bridges. |
| 14 | `14_encode_external_text_and_concepts.ipynb` | Encode raw text and assertion-aware clinical concepts. |
| 15 | `15_train_external_semantic_adapters.ipynb` | Train resumable semantic adapters while freezing bridge codebooks. |
| 16 | `16_evaluate_external_retrieval_ablations.ipynb` | Evaluate within-modality retrieval and cross-sensor concept compatibility. |

Notebooks 10 and 11 can run independently after notebook 09. Notebook 12 uses only Symile outputs from notebooks 02 to 04. Notebook 13 requires notebooks 10 to 12; notebook 14 requires notebooks 10 and 11. Notebook 15 validates completion markers and skips finished configurations unless `SEMANTIC_FORCE_RETRAIN=1`.

### Stage 3: ECG-focused multimodal RAG

| Order | Notebook | Purpose |
|---:|---|---|
| 17 | `17_configure_rag_report_generation.ipynb` | Select validation-performing continuous and VQ bridges and freeze RAG conditions. |
| 18 | `18_build_rag_retrieval_contexts.ipynb` | Build ECG-only, CXR-only, ECG+CXR, combined, and random-CXR contexts. |
| 19 | `19_generate_rag_reports.ipynb` | Generate resumable MedGemma and Qwen ECG reports. |
| 20 | `20_evaluate_rag_report_generation.ipynb` | Compute diagnostic, safety, text, context-quality, and paired-bootstrap metrics. |

The RAG branch consumes outputs through notebook 14 and can run while notebook 15 is still training. Notebooks 15 and 16 are required for the semantic-adapter analysis, not for notebooks 17 to 20.

## Experimental variants

### Bridge variants

- `clip_continuous`: continuous contrastive reference without vector quantization.
- `shared_vq`: one vector-quantized shared codebook.
- `shared_specific_vq`: shared and modality-specific vector-quantized codebooks with reconstruction and orthogonality losses.
- `shared_specific_no_code_match`: shared-specific VQ without explicit cross-modal code-distribution matching.
- `shuffled_pairs_control`: full shared-specific architecture trained after cross-patient CXR reassignment.

### Semantic variants

- `frozen_zero_shot`: frozen bridge without an external semantic adapter.
- `shared_raw_text`: one text projector shared by ECG diagnoses and CXR reports.
- `separate_raw_text`: modality-specific raw-text projectors.
- `shared_concepts`: one projector over normalized assertion-aware concepts.

### Generation conditions

Notebook 17 constructs zero-shot, ECG-only, CXR-only, and ECG+CXR conditions for continuous, VQ, and combined retrieval, plus a random-CXR control. The primary safety requirement is that retrieved CXR context outperform random CXR context without an unacceptable increase in fabrication or omission.

## Smoke test

Before a full multi-seed run, restrict the workload:

```bash
export BRIDGE_FEATURE_SPLITS=train,val
export ABLATION_VARIANTS=clip_continuous,shuffled_pairs_control
export ABLATION_SEEDS=43
export ABLATION_EPOCHS=2
export EXTERNAL_MAX_ROWS=128
export SEMANTIC_BRIDGE_VARIANTS=clip_continuous
export SEMANTIC_VARIANTS=shared_raw_text
export SEMANTIC_ADAPTER_SEEDS=43
export SEMANTIC_ADAPTER_EPOCHS=2
export RAG_MAX_TEST_ECG=10
export RAG_MODELS=qwen35_4b
export RAG_GENERATION_MAX_ROWS=10
export RAG_EVAL_BOOTSTRAPS=20
```

Use a separate smoke-test artifact root so partial files cannot be mistaken for the full experiment:

```bash
export ECG_CXR_BRIDGE_ARTIFACT_ROOT=/path/to/ecg_cxr_bridge_smoke_artifacts
```

## Outputs

All generated content is stored below `ECG_CXR_BRIDGE_ARTIFACT_ROOT`. Important locations include:

```text
$ECG_CXR_BRIDGE_ARTIFACT_ROOT/
├── config.json
├── manifests/
├── features/
├── models/
├── ablation/
├── external/
│   ├── ecg/
│   ├── cxr/
│   ├── text/
│   └── adapters/
├── reports/
└── rag_report_generation_ecg/
    ├── contexts/
    ├── query_images/
    ├── generations/
    └── reports/
```

Notebook 20 evaluates fabrication rate, omission rate, macro and micro F1, Hamming accuracy, per-label performance, ROUGE, BLEU, METEOR, BERTScore F1, CIDEr, context-attributable fabrication, retrieval-context quality, and paired bootstrap comparisons.

## Reproducibility and validity controls

- Symile training, validation, and test partitions are patient-disjoint.
- The official ten-candidate retrieval groups contain one documented same-admission positive.
- Same-patient observations are excluded from the contrastive negative set where identifiers are available.
- Bridge checkpoints are selected using validation retrieval only; the test split is not used for model selection.
- External ECG and CXR splits are patient-disjoint within each repository.
- RAG queries use the external ECG test split, while retrieval candidates come from training splits.
- Query identifiers and same-subject candidates are excluded when identifiers are available.
- Random-CXR context is retained as a specificity control.
- Completed generation rows and semantic-adapter runs are resumable and validated before being skipped.
- External ECG-CXR identity evaluation is intentionally unavailable because the independent repositories lack a certified shared patient link.

## Troubleshooting

### Scientific Python binary mismatch

If `wfdb` import fails with a pandas datetime C-API error after upgrading Transformers, reinstall the compiled stack in the same environment and restart the kernel:

```bash
python -m pip install --no-cache-dir --force-reinstall \
  "numpy==1.26.4" "pandas==2.2.3" "scipy==1.15.3" "wfdb==4.3.0"
python -m pip check
```

### MedGemma architecture not recognized

Notebook 19 uses `AutoProcessor` and `AutoModelForImageTextToText`. Upgrade Transformers to a release that supports the configured MedGemma architecture, restart the kernel, and confirm that the model terms have been accepted:

```bash
python -m pip install --upgrade "transformers>=5.14.1" accelerate
```

### Home-directory quota

Keep `HF_HOME`, `HF_HUB_CACHE`, the NLTK cache, Matplotlib cache, and experiment artifacts on project or scratch storage. Do not place them under a quota-limited home directory.

### BERTScore memory or tokenizer failures

Reduce `RAG_BERTSCORE_BATCH_SIZE` and, if needed, `RAG_BERTSCORE_MAX_LENGTH`. Notebook 20 uses a bounded tokenizer path to avoid oversized tokenizer truncation values.

## Release checks

Run the notebook-structure tests from the repository root:

```bash
pytest -q
```

The tests verify the ordered release set, valid notebook JSON, cleared outputs, cleared execution counts, and the presence of the required configuration variables.

## External software and attribution

The bridge variants are inspired by the compositional-codebook design in [CodeBind](https://github.com/Visual-AI/CodeBind) but are adapted for the ECG-CXR experiment. Upstream datasets, models, and software retain their own licenses and terms. Review those terms before use or redistribution.

## License

Repository code is released under the [MIT License](LICENSE). The license does not apply to third-party datasets, model checkpoints, generated patient-level artifacts, or external software.
