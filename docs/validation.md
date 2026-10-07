# Local release validation

Source-only packaging was tested on macOS, Python 3.12, using a separate
temporary test environment; original experiment source files were not modified.

- 49 tests passed; one optional upstream-repository comparison was skipped
  because `SYMILE_SOURCE_REPO` was not configured.
- 45 notebooks (24 current, 21 historical) passed notebook-schema validation,
  code-cell compilation, and checks for cleared outputs and execution counts.
- All published Python sources parsed successfully.
- Working-tree release candidates passed checks for experimental CSVs and other
  protected file types, saved outputs, common credential patterns and personal
  home/cluster paths. Ignored private directories were not copied.
- Synthetic tests exercised bridge training/retrieval, lab semantics, generation
  output handling, and restart safeguards without clinical data or model weights.

Full GPU generation and protected-data reproduction were not run. These checks
do not certify clinical validity, historical Git objects, arbitrary secret
formats, or all future dependency combinations. Review changes before publishing.

Reproduce using `python -m pytest -q tests experiment` from the repository root
after installing `requirements-dev.txt`. The optional mapping check additionally
requires the official Symile source checkout specified by the test.
