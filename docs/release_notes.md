# Source-only Experiment 4 release

Includes 24 current Experiment 4 notebooks, helper modules, four synthetic test
modules, and the public lab-code mapping with its third-party license. Notebooks
13E–16, their analysis module and configuration come from the newer
`jbhi_analysis_update_20260930` source package. The original repository's MIT
license and 21 two-modal notebooks are retained.

Excluded: clinical CSVs/JSONLs, analysis/results directories, review responses,
images, manuscripts, arrays, checkpoints, archives, caches, credentials and all
notebook outputs. The obsolete `_old` notebook and stale notebook generator are
omitted; the latter could overwrite the newer analysis notebooks.

Personal cluster, home, and storage paths were removed from working-tree source files.
Core defaults resolve beneath the repository, with environment overrides.
Notebook module lookup defaults to its working directory. Hugging Face token
files are optional explicit private paths, not embedded account-specific paths.

Scientific objectives were not changed and experimental results were not
recomputed. Requirements are compatibility bounds rather than a historical
lockfile. Validate actual GPU environments and private manifest paths on the
cluster. Preserve protocol hashes; changed generation code needs a new run and
changed frozen analysis settings need a new analysis ID.

Three stale specialist test cases were updated to mock the current Transformers
Mllama loader instead of the removed Unsloth backend. Production generation logic
was not changed by that test maintenance.

The clone retains remote Git history; no history rewrite was performed. Current
working-tree checks do not certify every historical Git object. The original
README is retained in `docs/legacy_workflow.md`; its historical two-modal scope
does not describe the new trimodal Experiment 4 workflow.
