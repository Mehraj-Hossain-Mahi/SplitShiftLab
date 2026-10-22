# Changelog

All notable changes to SplitShiftLab are documented here.

## [0.1.1] - 2026-10-22

### Added

- Reproducibility documentation covering installation, dataset preparation, experiment execution, and expected results.
- Independent exhaustive feasibility verification for UCI HAR and MHEALTH.
- Archived machine-readable scientific evidence and dataset provenance metadata.
- Previous-versus-optimized solver benchmark reports from the original and fresh Python environments.
- Reproducible scikit-learn GroupShuffleSplit and StratifiedGroupKFold baseline experiments.
- Documentation-based functional comparison with DataSAIL.
- Research environment metadata and dataset source attribution.

### Improved

- Installation and reproducibility instructions in the main README.
- Worked examples and research experiment documentation.
- Portable benchmark reports using repository-relative source paths.
- Verification scripts referencing archived reproducibility evidence.

### Validation

- 52 automated tests passed in the tested environments.
- Independent UCI HAR audit confirmed maximum feasible minimum class support of 322 across 54,264 group allocations.
- Independent MHEALTH audit confirmed maximum feasible minimum class support of 3,174 across 120 group allocations.
- Original and fresh-environment paired benchmarks confirmed solver-result agreement.
- Research evidence integrity and consistency checks passed.

### Notes

- This release focuses on reproducibility and research documentation.
- No general superiority over other splitting software is claimed.
- Raw benchmark datasets are not bundled with the software.
- Experimental environment records retain the software version used during the original evaluations.


## [0.1.0] - 2026-10-22

### Added

- Python API and command-line interface.
- Random, group, temporal, and group-temporal splitting.
- Partition auditing and constraint validation.
- Group-disjoint feasibility preflight with certificates.
- Minimum row-count and per-class support constraints.
- Feasible, infeasible, and undetermined preflight outcomes.
- Development-set distribution-shift diagnostics.
- Reproducible JSON reports and dataset fingerprints.
- UCI HAR and MHEALTH research examples.
- Automated tests and GitHub Actions CI.
- Python wheel and source distribution packaging.
- Software citation metadata using CITATION.cff.

### Validation

- 52 automated tests passed on GitHub Actions.
- Independent feasibility boundary checks on UCI HAR.
- Additional feasibility checks on MHEALTH.
- Wheel installation verified in a separate environment.
- Package metadata and source distribution contents checked.

### Limitations

- Research prototype, not a production-ready platform.
- Feasibility preflight currently supports group mode.
- Shift-stress diagnostics are exploratory and development-only.
- No comprehensive semantic leakage detection.
- No guaranteed improvement in predictive performance.
- No graphical user interface in this release.
- Additional datasets and independent evaluations are needed.
