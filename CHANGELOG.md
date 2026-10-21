# Changelog

All notable changes to SplitShiftLab are documented here.

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
