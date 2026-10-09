# Retrieval, linkage, copying, and rendering controls

Notebooks 17–21 extend the completed base experiment without overwriting it.
Use the environment template and launch notebooks from `experiment/`. Freeze
source code, cohort, bridge selection and budgets before generation. These
instructions describe reproduction, not a migration of existing private runs.
Public packaging changes source hashes; do not point new code at a populated
run and disable its protocol checks.

## Run order

1. Finish base preparation and analysis (00–16). In notebook 17, freeze the
   original ECG cohort and candidate admissions; generate retrieval controls
   and nearest-report copying scores. Default: Continuous (`clip`), up to
   250 examinations, and an outcome-blind 100-examination rendering subset.
2. Run notebook 18 to render legacy and calibrated ECGs. Inspect traces, units,
   lead names, duration, calibration and clipping against the source waveforms.
   Only then explicitly set `E4_REVIEW_IMAGES_APPROVED=1`.
3. Run notebook 19 in the Qwen environment: 50 pilot tasks first. Inspect actual
   reports and errors, not just `ok` status. Set `E4_REVIEW_MAX_ROWS=0` to finish
   all remaining tasks with the same settings (default total: 1,750).
4. Preserve the original PULSE pilot protocol before running revised notebook
   20. If starting from scratch, run the following once in a **separate, fresh
   PULSE kernel**, with the configured environment and image approval:

   ```python
   import os
   from e4_review_controls import roots, run_generation
   _, original_review_root = roots()
   assert os.environ.get("E4_REVIEW_IMAGES_APPROVED") == "1"
   run_generation("pulse7b_ecg", dest=original_review_root, max_rows=50,
                  component_cap=120, max_new_tokens=512, max_input_tokens=3584)
   ```

   This deliberately reproduces the original pilot, including format failures;
   it is not the final PULSE result. It records the resolved model revision.
   Do not invent the protocol or discard failed attempts.
5. Restart the PULSE kernel, then run notebook 20 from the top. It requires the
   preceding pilot protocol, pins the same model revision, and creates a frozen
   code snapshot under `pulse_prompt_v2`. Set `IMAGES_APPROVED=True` only after
   inspection. The revised prompt gives query-only instructions for no retrieval
   and adds evidence instructions only when evidence exists. All PULSE arms run
   afresh; original successes and failures are not reused. Inspect 50 pilot
   attempts, then set `E4_REVIEW_MAX_ROWS=0` (default total: 600).
6. In a fresh general/CPU kernel, run notebook 21. It explicitly evaluates the
   revised PULSE root together with the unchanged Qwen logs. For pilot-only
   inspection set `E4_REVIEW_ALLOW_PARTIAL=1`; final analysis requires all planned
   attempts. Failed attempts remain explicit outcomes.

Use the dedicated official PULSE/LLaVA environment, not the main Qwen
Transformers environment. Source modules, image hashes, inputs and settings are
frozen on resume. One writer per model/run is allowed. Recorded successes **and
errors** are skipped; rerunning does not silently retry unfavorable outputs.

## Controls

| Arm | Purpose |
|---|---|
| `zero_shot` | No retrieved evidence |
| `raw_ecg_same` | Frozen ECG cosine retrieval, ECG text only; no trimodal training |
| `raw_ecg_full` | Same raw ECG retriever, all linked evidence text |
| `bridge_ecg_same` | ECG-only scoring in the trimodal-trained representation |
| `trimodal_same` | Trimodal scoring, ECG text only |
| `trimodal_full` | Trimodal scoring, linked ECG, laboratory and CXR text |
| `trimodal_broken` | Same ECG anchors but permuted CXR/laboratory bundles |

Nearest-report copying transfers the nearest admission's machine interpretation
without generation. Broken linkage is a research manipulation, not valid clinical
evidence. The design preserves bundle multisets, not exact per-query token counts.
Agreement with machine interpretations does not establish clinical correctness;
completion and common-successful-case estimates must be reported together.

## Output locations (private; never commit)

`EXPERIMENT4_ROOT` contains base experiment artifacts. `E4_REVIEW_ROOT` defaults
to its `review_extension_v1` child and contains:

- `protocol.json`, `cohort.jsonl`, `contexts.jsonl`, `retrieval.csv`, and the
  restricted broken-linkage donor audit;
- `query_images/legacy`, `query_images/calibrated`, and `image_audit.jsonl`;
- `generations/qwen35_4b/` and the original `generations/pulse7b_ecg/` pilot;
- `pulse_prompt_v2/code_snapshot`, `prompt_revision.json`, and revised
  `generations/pulse7b_ecg/`;
- `pulse_prompt_v2/evaluation/`: coverage, summary, paired comparisons,
  resources, failure-as-abstention sensitivity, detailed records and status.

The revised root links to the original Qwen generation directory for evaluation.
Download both actual directories: a symlink alone does not contain those reports.
Never run Qwen through the revised PULSE snapshot. No protected reports, images,
review forms, patient identifiers, or credentials are included in this repository.
