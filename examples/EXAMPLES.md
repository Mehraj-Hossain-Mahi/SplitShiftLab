# SplitShiftLab ? Worked Examples

This directory contains examples, development experiments, independent verification scripts, and benchmark reproduction utilities for SplitShiftLab v0.1.0.

For complete instructions, see:

- [Reproducibility Runbook](../reproducibility/RUNBOOK.md)
- [Dataset Sources and Licenses](../reproducibility/DATASETS.md)
- [External Software Comparison](../reproducibility/EXTERNAL_COMPARISON.md)

## 1. Synthetic Demonstration

Generate demonstration data:

    python -m splitshiftlab demo

Generate a distribution-shift stress demonstration:

    python -m splitshiftlab demo --stress

These demonstrations use synthetic data and are not real-world performance benchmarks.

The generated manifest includes partition indices, audit information, and dataset fingerprint metadata.

Group-temporal mode is also supported through the CLI. Consult the CLI help and root README for the applicable group and timestamp arguments.

## 2. UCI HAR Dataset Preparation

Download the official UCI Human Activity Recognition Using Smartphones dataset.

Place the extracted dataset under:

    data/uci_har/UCI HAR Dataset/

Prepare the official partitions:

    python examples/prepare_uci_har.py

Development experiments use:

    out/uci_har/uci_har_official_train.csv

The official UCI HAR test partition is kept separate and excluded from the documented development feasibility experiments.

Expected development data:

- 7,352 records
- 21 subjects
- 6 activity classes
- 561 feature columns

## 3. Independent UCI HAR Boundary Verification

Run:

    python examples/verify_uci_har_class_support_boundary.py

Expected results:

- 54,264 possible six-group allocations
- 54,264 allocations satisfying the tested row minima
- Maximum feasible minimum class support: 322
- Exactly one feasible allocation at support 322
- No feasible allocation at support 323

Archived evidence:

    reproducibility/evidence/uci_har_class_support_boundary_audit.json

This verification independently enumerates group allocations without importing the SplitShiftLab solver.

## 4. MHEALTH Dataset Preparation

Download the official MHEALTH dataset and extract its ten subject log files under:

    data/mhealth/

Run:

    python examples/prepare_mhealth.py

Expected prepared data:

- 1,215,745 records
- 10 subjects
- 23 sensor features
- 13 class labels (0 through 12)
- Background activity label 0 retained

Generated files:

    out/mhealth/mhealth_all_subjects.csv
    out/mhealth/mhealth_preparation_report.json

## 5. Independent MHEALTH Verification

Run:

    python examples/verify_mhealth_independent_boundary.py

Expected results:

- 120 whole-group allocations
- Maximum feasible minimum class support: 3174
- Eight feasible allocations at support 3174
- Zero feasible allocations at support 3175
- Archived reference evidence matched

Archived evidence:

    reproducibility/evidence/mhealth_independent_boundary_audit.json

The script does not import the SplitShiftLab feasibility solver.

## 6. Internal Solver Performance Benchmark

The following experiment compares the previous feasibility implementation with the optimized batched implementation.

Run:

    python examples/benchmark_solver_versions_paired.py --runs 10 --warmups 2

Required implementation sources:

    examples/legacy_protocol_compiler_step232.py
    src/splitshiftlab/protocol_compiler.py

The benchmark uses the prepared UCI HAR development CSV.

It records:

- Individual execution times
- Mean and median execution times
- Runtime standard deviations
- Paired timing differences
- Solver correctness agreement
- Dataset and implementation fingerprints

Archived results:

    reproducibility/evidence/benchmark_solver_versions_paired.json
    reproducibility/evidence/benchmark_solver_versions_paired.csv

A second benchmark run performed in a fresh Python virtual environment is archived as:

    reproducibility/evidence/benchmark_solver_versions_fresh_env.json
    reproducibility/evidence/benchmark_solver_versions_fresh_env.csv

The reported timings are environment-dependent.

This is an internal implementation benchmark, not a runtime comparison against external software.

## 7. scikit-learn GroupShuffleSplit Baseline

Run:

    python examples/verify_sklearn_group_shuffle_baseline.py

Evaluated setting:

- scikit-learn 1.9.1
- UCI HAR development dataset
- 1,000 randomized group-disjoint splits
- Six validation groups

Observed results:

- 990 unique group allocations
- No sampled allocation satisfying minimum class support 322
- Best observed minimum class support: 317

Archived evidence:

    reproducibility/evidence/sklearn_group_shuffle_baseline.json

## 8. scikit-learn StratifiedGroupKFold Baseline

Run:

    python examples/verify_sklearn_stratified_group_baseline.py

Observed results:

- 1,000 generated folds
- 750 five-group validation folds
- 250 six-group validation folds
- One unique six-group allocation
- No eligible fold satisfying minimum class support 322
- Best observed eligible support: 269

Archived evidence:

    reproducibility/evidence/sklearn_stratified_group_baseline.json

Only six-group folds are compared against the corresponding exact-six-group feasibility protocol.

Repeated folds are not independent unique allocation samples.

These baseline experiments do not establish general superiority over scikit-learn.

## 9. DataSAIL Functional Comparison

DataSAIL is discussed through its published research and official documentation.

See:

    reproducibility/EXTERNAL_COMPARISON.md

No direct DataSAIL runtime or predictive-performance experiment was conducted.

## 10. Automated Tests

Install the project and test dependencies:

    python -m pip install -e ".[test]"

Run:

    python -m pytest -q

SplitShiftLab v0.1.0 passed 52 automated tests in the documented testing environments.

## 11. Reproducibility Evidence

Archived scientific evidence is maintained in:

    reproducibility/evidence/

Dataset provenance:

    reproducibility/dataset_provenance.json

Environment metadata:

    reproducibility/environment.json

Complete execution instructions:

    reproducibility/RUNBOOK.md

Raw datasets are not bundled with the research repository. Obtain them from the official sources and follow the documented preparation instructions.

## 12. Scientific Limitations

The reported feasibility boundaries apply only to the stated datasets, group cardinalities, and constraints.

The experiments do not establish:

- Universal solver correctness
- Comprehensive semantic leakage prevention
- Predictive-performance improvements
- Universal superiority over alternative tools
- Cross-machine runtime equivalence

The software's official v0.1.0 release predates some of the additional reproducibility materials developed for journal submission. Those materials must be committed and archived in a corresponding reproducibility release before submission.
