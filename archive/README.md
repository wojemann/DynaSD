This `archive/` directory stores historical development artifacts that are not part of the
active package surface, test suite, or release workflow.

Contents:
- `legacy-models/`: superseded or experimental model implementations no longer
  imported by the package (`GIN_old`, `NDD_old`, `NDD_fixed`, `LiRNDDA`,
  `LiRNDDA_backup`, `MINDD`, `ONDD`, `absolute_slope`, and the redundant
  `models.py` re-export shim). `LiRNDDA` and `MINDD` were experimental
  benchmark variants that did not appear in the published paper and lacked
  pretrained decision boundaries, so they were dropped from the public API
  rather than maintained. Preserved for historical reference; the original
  `wo_dev` tip before release-prep cleanup is also tagged as `submission`.
- `legacy-tests/`: ad hoc and exploratory test scripts kept for historical reference.
- `notes/`: optimization notes retained for context from prior development iterations.

These files are intentionally excluded from the primary package layout and CI-oriented test flow.
