# Symile-MIMIC lab decoder verification

Audit date: 28 September 2026. Upstream commit:
`d12e2c8b0a528cd536ccc5175f910f38f07bb6cb`.

## Verified interpretation

| Question | Verified implementation |
| --- | --- |
| Which test belongs to each value? | The 50 item IDs in `constants.py:LABS`, sorted by the corresponding `itemid_percentile` keys, not dictionary insertion order. |
| What is in the 100-value vector? | First 50: training-set ECDF percentiles. Last 50: corresponding observation indicators. |
| What does mask 1 mean? | Observed. Mask 0 means missing. The `labs_missingness` filename and README wording must not be interpreted as 1=missing. |
| What happens to missing values upstream? | The mean training percentile for that lab is stored, while its mask is 0. |
| What happens in Experiment 4? | Prepared bridge arrays use 0.5 in unobserved slots. This is an existing modeling choice, not exact reproduction of upstream mean imputation. The decoder ignores those slots either way. |
| Can a high percentile establish clinical abnormality? | No. The population distribution is not a clinical reference interval. |
| Can encoded percentiles recover raw values or units? | Not from this code-name table and vector alone. The decoder leaves those fields null. |
| Do H, L, I mean high, low, or diagnoses? | The upstream table contains only these short labels for 50934, 51678, 50947. We preserve them without expanding or assigning clinical meaning. |

The original item-ID order and observation-mask convention were correct. The original display names differed
from the upstream table only in `INR(PT)` spacing and correction of its `Asparate` spelling to `Aspartate`.
The new table retains the exact source label and a separate display label, so these differences are explicit.

## Changes

- Added `symile_lab_code_table.json`, including all 50 names, both vector indices, source commit, and source-file
  SHA256. This file must accompany the Python modules when uploaded.
- Replaced the unversioned inline mapping with the pinned table. Optional `MIMIC_D_LABITEMS` overrides remain
  supported and their file hash is recorded separately.
- Enforced binary masks and finite [0,1] observed percentiles; missing/imputed values never appear as observations.
- Kept every observed lab in structured output, while retaining the limited salience-selected prompt summary.
  Salience is explicitly distributional, not a clinical positive/negative classification.
- Distinguished all-missing labs from observed labs without extreme percentiles. Threshold text now follows
  configured thresholds, and float32 threshold boundaries are handled consistently.
- Added upstream-source conformance and regression tests. The conformance test executes only the extracted
  upstream `get_labs` function against synthetic inputs and checks our decoder's output.
- Added `03B_audit_lab_decoding.ipynb` to check real CSV/NPY column alignment, mask direction, percentile values,
  mean imputation, and admission order without changing prepared arrays or bridge checkpoints.

## Biowulf sequence

1. Upload the updated modules, notebooks, and `symile_lab_code_table.json`. Biowulf paths remain unchanged.
2. Run notebook 03B with the full official Symile split CSVs and encoded arrays available. It writes
   `evidence/lab_encoding_audit.json`. A failed audit must be resolved before trusting lab evidence.
3. Rebuild evidence with 08, contexts with 09, and validate with 10. Do not rerun bridge training merely for
   this decoder update: neither its numeric prepared arrays nor the training implementation were changed.
4. Regenerate reports for changed contexts and rerun evaluation. Existing reports are preserved, but are not
   interchangeable with the revised contexts. The current resume logic checks the context signature.

This review verified source semantics and synthetic behavior locally. It did not access the protected Biowulf
arrays. Actual-array conformance remains conditional on notebook 03B passing. A code-name mapping by itself
cannot prove the provenance or feature order of an arbitrary encoded file.

Verification: 29 tests passed, including direct comparison with the pinned upstream mapping and synthetic
execution of its get_labs function. All 24 notebooks and 62 code cells passed schema/syntax checks. The existing
bridge training source and Biowulf path defaults were unchanged.

## Sources

- [Official item-ID/name mapping](https://github.com/rajesh-lab/symile/blob/d12e2c8b0a528cd536ccc5175f910f38f07bb6cb/experiments/data_processing/symile_mimic/constants.py)
- [Official tensor construction and observation masks](https://github.com/rajesh-lab/symile/blob/d12e2c8b0a528cd536ccc5175f910f38f07bb6cb/experiments/data_processing/symile_mimic/process_and_save_tensors.py)
- [Training ECDF and saved mean percentiles](https://github.com/rajesh-lab/symile/blob/d12e2c8b0a528cd536ccc5175f910f38f07bb6cb/experiments/data_processing/symile_mimic/create_dataset_splits.py)
- [Concatenation of percentiles and masks for the lab encoder](https://github.com/rajesh-lab/symile/blob/d12e2c8b0a528cd536ccc5175f910f38f07bb6cb/experiments/models/symile_mimic_model.py)
