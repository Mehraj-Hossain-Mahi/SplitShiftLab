# SplitShiftLab Reproducibility Runbook

## Scope

This guide documents how to reproduce selected scientific results
reported for SplitShiftLab v0.1.0.

The evaluations cover:
- UCI HAR independent group-feasibility boundary verification.
- MHEALTH independent group-feasibility boundary verification.
- Paired previous-versus-optimized preflight benchmarking.
- Automated software tests.

These evaluations do not establish universal solver correctness,
superior predictive performance, or complete leakage prevention.

## 1. Software Environment

Source repository:
https://github.com/Mehraj-Hossain-Mahi/SplitShiftLab

Research software release:
https://github.com/Mehraj-Hossain-Mahi/SplitShiftLab/releases/tag/v0.1.0

The repository is currently private.

The recorded experimental environment is available in:

reproducibility/environment.json

Install from the repository root:

    python -m venv .venv

Activate the virtual environment.

Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Linux/macOS:

    source .venv/bin/activate

Install the software and testing dependencies:

    python -m pip install -e ".[test]"

Verify the installation:

    python -m pytest -q

The previously tested version passed 52 tests.

## 2. Dataset Sources

Consult:

reproducibility/DATASETS.md

Download UCI HAR and MHEALTH from the official UCI
Machine Learning Repository.

Raw datasets are not bundled with the current software release.

## 3. UCI HAR Preparation

Place the extracted dataset under:

data/uci_har/UCI HAR Dataset/

Run:

    python examples/prepare_uci_har.py

Expected development output:

out/uci_har/uci_har_official_train.csv

The preparation script also converts the official test
partition into a separate CSV. The development boundary
experiments below use only the official training partition.

Expected development dataset:
- 7,352 rows
- 21 subject groups
- 6 classes

## 4. Independent UCI HAR Verification

Run:

    python examples/verify_uci_har_class_support_boundary.py

Expected results:
- 54,264 total group allocations
- 54,264 eligible allocations
- Maximum feasible minimum class support: 322
- Feasible allocations at support 322: 1
- No feasible allocation at support 323

Generated evidence:

out/uci_har/uci_har_class_support_boundary_audit.json

Archived reference:

reproducibility/evidence/uci_har_class_support_boundary_audit.json

The independent audit does not import the SplitShiftLab solver.

## 5. MHEALTH Preparation

Extract the ten official subject logs into:

data/mhealth/

Subdirectories are supported.

Run:

    python examples/prepare_mhealth.py

Expected prepared dataset:
- 1,215,745 rows
- 10 subjects
- 23 sensor features
- 13 labels, including background label 0

Generated outputs:

out/mhealth/mhealth_all_subjects.csv
out/mhealth/mhealth_preparation_report.json

## 6. Independent MHEALTH Verification

Run:

    python examples/verify_mhealth_independent_boundary.py

Expected results:
- 120 eligible whole-group allocations
- Maximum feasible minimum class support: 3174
- 8 feasible allocations at support 3174
- 0 feasible allocations at support 3175

This script does not import the SplitShiftLab solver.

It compares computed results against:

reproducibility/evidence/mhealth_independent_boundary_audit.json

The original development-stage solver evidence is archived
separately as:

reproducibility/evidence/mhealth_support_boundary_evidence.json

## 7. Paired Solver Performance Benchmark

Required files:

examples/benchmark_solver_versions_paired.py
examples/legacy_protocol_compiler_step232.py
src/splitshiftlab/protocol_compiler.py

The prepared UCI HAR development CSV is also required.

Run:

    python examples/benchmark_solver_versions_paired.py --runs 10 --warmups 2

The benchmark compares previous and batched implementations
in the same Python process.

It validates solver-result agreement and records:
- Individual execution times
- Mean and median runtimes
- Standard deviations
- Paired timing differences
- Dataset and solver fingerprints

Generated results:

out/uci_har/benchmark_solver_versions_paired.json
out/uci_har/benchmark_solver_versions_paired.csv

Archived results:

reproducibility/evidence/benchmark_solver_versions_paired.json
reproducibility/evidence/benchmark_solver_versions_paired.csv

The archived JSON uses repository-relative solver paths.

Runtime results depend on hardware, software versions,
operating-system scheduling, and other environmental factors.

The measurements include hashing, validation, row-feasibility
processing, and class-support checks. They exclude process
startup and CSV loading.

## 8. Evidence Integrity

Dataset source and prepared-data fingerprints:

reproducibility/dataset_provenance.json

Recorded software environment:

reproducibility/environment.json

Scientific evidence:

reproducibility/evidence/

The original development evidence inventory is recorded in:

out/research_evidence_register.json

The out/ directory is locally generated and is not included
in the current repository release.

## 9. Interpretation and Limitations

- The exact feasibility findings apply to the specified
  datasets and group-allocation constraints.
- The support thresholds are correctness-testing boundaries,
  not recommended partition-design parameters.
- The official UCI HAR test partition is excluded from
  the documented development feasibility experiments.
- No classifier-performance improvement is claimed.
- Runtime comparisons are internal implementation benchmarks.
- Independent evidence scripts verify selected mathematical
  properties, not all software behaviors.

## 10. Reproduction Status

UCI HAR and MHEALTH boundary results have been reproduced
in the development environment.

Automated tests and paired benchmark checks have passed.

Independent clean-machine reproduction of this complete
research package remains to be performed before submission.

## 11. External Software Baseline Reproduction

### Scope

The following experiments compare selected scikit-learn
splitting methods against an independently established
UCI HAR development-set feasibility boundary.

These comparisons are specific to the evaluated dataset,
group cardinality, and class-support requirements.

### 11.1 GroupShuffleSplit

Evaluated software:
scikit-learn 1.9.1

Run from the project root:

    python examples/verify_sklearn_group_shuffle_baseline.py

Expected results:
- 1,000 random splits
- 990 unique group allocations
- Group disjointness maintained
- 0 allocations satisfying class support >= 322
- Best observed minimum class support: 317

Archived reference:

reproducibility/evidence/sklearn_group_shuffle_baseline.json

### 11.2 StratifiedGroupKFold

Evaluated software:
scikit-learn 1.9.1

Run:

    python examples/verify_sklearn_stratified_group_baseline.py

Expected results:
- 1,000 generated folds
- 750 five-group validation folds
- 250 six-group validation folds
- 1 unique six-group allocation
- 0 eligible allocations satisfying support >= 322
- Best observed eligible support: 269

Archived reference:

reproducibility/evidence/sklearn_stratified_group_baseline.json

Only six-group validation folds are compared against
the exact-six-group feasibility protocol.

Repeated folds are not independent unique allocations.

### 11.3 Independent Feasibility Reference

Run:

    python examples/verify_uci_har_class_support_boundary.py

Expected results:
- 54,264 group allocations
- 1 feasible allocation at minimum class support 322
- 0 feasible allocations at support 323

### 11.4 Interpretation

These experiments compare distinct splitting objectives.

GroupShuffleSplit samples group-disjoint allocations.

StratifiedGroupKFold constructs group-disjoint folds
while attempting to preserve class distributions.

SplitShiftLab provides a feasibility preflight for
specified group cardinality, row-count minima, and
per-class support requirements.

The results do not demonstrate universal superiority
over scikit-learn.

The results are reproducible under the documented
dataset preparation and tested software environment.

### 11.5 DataSAIL

DataSAIL is discussed separately in:

reproducibility/EXTERNAL_COMPARISON.md

The DataSAIL comparison is documentation-based.

No direct DataSAIL experiment or cross-software
runtime comparison was performed.
