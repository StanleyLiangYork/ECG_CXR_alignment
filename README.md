# Admission-Linked Multimodal Evidence Integration for ECG Report Generation

**Zhaohui Liang, Niccolo Marini, Sivaramakrishnan Rajaraman, Zhiyun Xue, Sameer Antani**

National Library of Medicine, National Institutes of Health, Bethesda, Maryland, USA

Research code for testing whether ECG, chest radiograph (CXR), and laboratory
patterns from other patients' linked admissions can support report generation.
The current workflow is **[Experiment 4](experiment/README.md)**: globally
patient-disjoint partitioning, non-text trimodal bridge training, evidence
retrieval, five direction-specific generators, automated evaluation, and blinded
clinical review. Earlier two-modal notebooks remain for preprocessing and
historical comparisons.

## Scientific scope

Frozen sensor features and laboratory measurements are aligned using Continuous
(`clip`), vector-quantized (`codebind`, CodeBind-inspired), and Hybrid
(`combined`) bridges. Report text is not a bridge-training target.
Same-admission concurrence does **not** imply diagnostic equivalence between
modalities. Retrieved cases are other patients' evidence, not facts about the
query. Laboratory distributional percentiles are not abnormality thresholds.

Experiments compare no retrieval, same-sensor evidence, cross-sensor and lab
additions, full trimodal evidence, and random/shuffled/gated controls. They do
not presume vector quantization is superior. Agreement against machine
interpretations and CXR labels does not establish clinical correctness.
This is research software, not a clinical decision system.

## Repository layout

| Directory/file | Purpose |
|---|---|
| [experiment/](experiment/README.md) | Current trimodal workflow, modules, analysis configuration and synthetic tests |
| [notebooks/](notebooks) | Original 00–20 ECG–CXR workflow, including upstream frozen-feature preparation |
| [config/](config) | Environment-variable templates without personal server paths or tokens |
| [docs/legacy_workflow.md](docs/legacy_workflow.md) | Historical two-modal instructions |
| [docs/release_notes.md](docs/release_notes.md) | Packaging decisions and limitations |
| [tests/](tests) | Notebook and source-only release checks |
| [LICENSE](LICENSE) | MIT license |

No clinical CSVs, JSONL records, images, review responses, weights, generated
reports, embeddings, or notebook outputs are distributed. The laboratory
code-table JSON is static public metadata; its attribution is in
[THIRD_PARTY_NOTICES.md](experiment/THIRD_PARTY_NOTICES.md).

## Replicate the experiments

1. Obtain authorized access to [Symile-MIMIC](https://physionet.org/content/symile-mimic/1.0.0/),
   [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/), and
   [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/).
   Accept applicable data-use agreements and generator-model licenses.
2. Prepare frozen arrays and row-aligned manifests using the original
   [notebooks](notebooks). Read the [Experiment 4 input contract](experiment/README.md#inputs).
   Existing compatible artifacts can be reused; Experiment 4 is not a raw-data downloader.
3. Create a suitable runtime and edit a private copy of
   [config/experiment.environment.example.sh](config/experiment.environment.example.sh).
   Use project storage for outputs and caches. Keep secrets outside Git.
4. Follow the [ordered notebook workflow](experiment/README.md#run-order).
   Test small generation pilots before full GPU jobs. PULSE and ECG-Instruct
   use separate environments; one environment is not assumed to support every model.
5. Audit coverage, freeze stationary inputs, evaluate, and complete independent
   review/adjudication. Preserve reports together with protocols and run manifests.

Paths inside your **private input manifests** must also resolve on the execution
machine. The source contains no embedded cluster account or scheduler allocation.

## Tests

Synthetic tests use mocked generators, not protected records or model downloads.
From the repository root, with Python 3.12:

```bash
python -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q tests experiment
```

GPU loading and clinical-data reproduction require separate cluster validation.
Requirements are compatibility bounds, not a complete lockfile of the original
experiments. Record installed versions and resolved model revisions.

## License and data governance

Code is MIT-licensed; dataset, model, and third-party licenses remain separate.
Do not upload patient-level inputs/outputs, review forms, private identity
crosswalks, or access tokens. Ignore rules are a safeguard, not a substitute
for inspecting Git status and the staged diff before publication.
