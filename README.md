# SplitShiftLab

**Auditable Data Partitioning, Constraint Feasibility, and Distribution-Shift Diagnostics for Machine Learning**

- **Version:** 0.1.0 (development prototype)
- **Language:** Python
- **License:** MIT
- **Status:** Functional research prototype under validation

## Overview

SplitShiftLab is an open-source Python research-software prototype for constructing, auditing, and validating machine-learning evaluation partitions.

It supports random, group-disjoint, temporal, and group-temporal splitting protocols. It also provides a constraint-feasibility preflight engine for determining whether specified group-partition requirements can be satisfied before model training.

The software distinguishes among three feasibility outcomes:

- **Feasible:** A valid allocation certificate has been identified.
- **Infeasible:** The supported constraints have been proven impossible to satisfy.
- **Undetermined:** The available search budget or supported checks are insufficient to establish either conclusion.

SplitShiftLab additionally includes development-set-only distribution-shift stress diagnostics based on approximate kernel Maximum Mean Discrepancy (MMD).

The software is not a comprehensive data-leakage detector. Its guarantees are limited to explicitly implemented and verified constraints.

## Key Features

### 1. Reproducible Data Partitioning

Four split modes are available:

- `random`: Random train–validation partitioning.
- `group`: Group-disjoint partitioning.
- `temporal`: Chronologically ordered partitioning.
- `group_temporal`: Group-aware, chronologically constrained partitioning, including exclusion of groups crossing relevant temporal boundaries.

### 2. Partition Auditing

Supported audit checks include:

- Training and validation row overlap.
- Partition coverage.
- Minimum partition sizes.
- Group overlap.
- Chronological ordering where applicable.
- Optional per-class support requirements.
- Exact duplicate-feature indicators.

Duplicate-feature indicators are warnings, not proof of data leakage.

### 3. Constraint Feasibility Preflight

The preflight engine currently supports group-mode feasibility analysis.

Supported requirements include:

- Fixed numbers of whole groups in each partition.
- Group-disjoint allocation.
- Minimum training records.
- Minimum validation records.
- Minimum observations per class in both partitions.
- Configurable search limits.

The engine uses row-feasibility dynamic programming and bounded exact group-combination enumeration with batched NumPy class-support checks.

Where supported, the generated JSON report contains an allocation certificate or an explanation of infeasibility or unresolved requirements.

### 4. Distribution-Shift Diagnostics

The optional shift-stress functionality searches among valid group partitions using approximate RBF-kernel MMD², calculated from unlabeled feature values.

This is a development-data diagnostic. It must not be used to select or optimize partitions using a protected final test dataset.

The selected stress score is not an unbiased estimate of model generalization.

### 5. Reproducible Reports

Reports can include:

- Dataset SHA-256 fingerprint.
- Protocol specification.
- Selected group identifiers or row positions.
- Constraint and audit results.
- Feasibility status.
- Search diagnostics.
- Distribution-shift diagnostics, where requested.

## Installation

Clone or obtain the project source and enter the project directory.

```bash
cd splitshiftlab_mvp
python -m pip install -e ".[test]"
```

Verify the installation:

```bash
python -m pytest -q
```

The current development version has passed 52 automated tests in the project's tested environment.

This does not establish correctness for all possible input datasets or evaluation requirements.

## Quick Start

### Generate Synthetic Demo Data

```bash
python -m splitshiftlab demo
```

This creates synthetic example data and demonstration outputs.

### Run a Group-Disjoint Split

```bash
python -m splitshiftlab run --csv "out/demo_data.csv" --target label --mode group --group subject --features feature_1 feature_2 --evaluate --output "out/group_manifest.json"
```

### Run a Feasibility Preflight

```bash
python -m splitshiftlab preflight --csv "out/demo_data.csv" --target label --group subject --test-size 0.25 --output "out/demo_preflight.json"
```

The `test-size` parameter in group preflight refers to a fraction of groups, not necessarily the same fraction of rows.

### Apply a Class-Support Constraint

```bash
python -m splitshiftlab preflight --csv "out/demo_data.csv" --target label --group subject --test-size 0.25 --minimum-class-support 20 --output "out/class_support_report.json"
```

The result depends on the distribution of group identifiers and class labels in the supplied dataset.

### Limit Exact Search

```bash
python -m splitshiftlab preflight --csv "out/demo_data.csv" --target label --group subject --test-size 0.25 --minimum-class-support 20 --max-class-allocations 100 --output "out/limited_search.json"
```

A restricted search budget may result in `undetermined`. Such a result must not be interpreted as proof of infeasibility.

## Python API

```python
import pandas as pd
from splitshiftlab import ProtocolSpec, generate_split

df = pd.read_csv("out/demo_data.csv")

spec = ProtocolSpec(
    target="label",
    mode="group",
    group_col="subject",
    seed=42,
)

result = generate_split(df, spec)

print(result.audit)
```

### Group Feasibility API

```python
import pandas as pd
from splitshiftlab import ProtocolSpec
from splitshiftlab.protocol_compiler import preflight_group_protocol

df = pd.read_csv("out/demo_data.csv")

spec = ProtocolSpec(
    target="label",
    mode="group",
    group_col="subject",
    test_size=0.25,
    min_class_support=20,
)

report = preflight_group_protocol(df, spec)

print(report["status"])
print(report.get("certificate"))
```

## Real-World Dataset Validation

SplitShiftLab has undergone development-stage feasibility testing using two public human-activity-recognition datasets.

### UCI Human Activity Recognition (UCI HAR)

Development dataset:

- 7,352 records.
- 21 subject groups.
- Six activity classes.
- Six validation groups under the evaluated protocol.

An independent exhaustive audit evaluated all 54,264 eligible whole-group allocations.

For the tested class-support protocol:

| Minimum support per class | Result |
|---|---|
| 322 | Feasible |
| 323 | Infeasible |

At support 322, exactly one valid allocation was found by the independent reference analysis.

The official UCI HAR test dataset was not used for these development feasibility experiments.

### UCI MHEALTH

Prepared dataset:

- 1,215,745 records.
- 10 subjects.
- 23 sensor measurements.
- Activity labels 0–12.
- Background activity label 0 retained.

For the evaluated three-subject validation protocol:

| Minimum support per class | Result |
|---|---|
| 3,174 | Feasible |
| 3,175 | Infeasible |

At support 3,175, the solver exhaustively examined all 120 candidate allocations and found no valid solution.

The class-support thresholds are dataset-specific correctness-test boundaries, not generally recommended requirements for model evaluation.

## Performance Evaluation

The class-support feasibility implementation was optimized using batched NumPy operations.

A paired in-process benchmark compared the previous and batched implementations on the UCI HAR development dataset.

| Scenario | Previous mean | Batched mean |
|---|---:|---:|
| Feasible, support 322 | 0.0790 s | 0.0419 s |
| Infeasible, support 323 | 0.5007 s | 0.0817 s |

These measurements correspond to approximately 1.89× and 6.13× speedups based on mean runtime for the two tested scenarios.

Additional deterministic synthetic group-scaling experiments were conducted with 10, 14, 18, and 21 groups.

The benchmarks measure the preflight function, including fingerprinting and row-feasibility processing, while excluding Python process startup and CSV loading.

All reported speedups are workload- and environment-specific. They do not establish universal performance superiority.

## Reproducibility

Dataset preparation and evaluation scripts are maintained under `examples/`.

Relevant development scripts include:

- `prepare_uci_har.py`
- `prepare_mhealth.py`
- `verify_uci_har_class_support.py`
- `verify_uci_har_class_support_boundary.py`
- `benchmark_uci_har_inprocess.py`
- `benchmark_solver_versions_paired.py`
- `benchmark_group_scaling_step243.py`

Some benchmark scripts require a separately retained previous-solver implementation or local dataset files. These dependencies must be included or documented in the reproducibility release.

### UCI HAR Preparation

Obtain the original UCI HAR dataset and follow the dataset preparation script's input-directory requirements.

```bash
python examples/prepare_uci_har.py
```

### MHEALTH Preparation

Download and extract the original MHEALTH subject logs into the expected data directory.

```bash
python examples/prepare_mhealth.py
```

The complete prepared MHEALTH CSV is large and should not be assumed to be included in the source distribution.

Dataset licenses and redistribution conditions must be respected.

### Testing

```bash
python -m pytest -q
```

Generated reports and benchmark artifacts should be retained with the reproducibility materials for a formal software release.

## Scientific and Technical Limitations

1. **Scoped leakage checks:** Valid partitions do not guarantee the absence of semantic leakage, target-derived features, or preprocessing leakage.

2. **Preflight mode support:** The constrained feasibility preflight currently supports group-based allocation. Equivalent exact feasibility certification is not established for all temporal or mixed constraints.

3. **Search complexity:** Exact group-combination enumeration has combinatorial worst-case growth. Larger configurations may require a search budget and can return `undetermined`.

4. **Class-label dependence:** Minimum class-support constraints depend on target labels. They should not be used to retrospectively optimize a protected final-test partition.

5. **Distribution-shift limitations:** Maximizing approximate MMD² does not guarantee maximal prediction error or establish a statistically significant relationship with generalization difficulty.

6. **Group-size imbalance:** A specified group fraction does not guarantee an equivalent row fraction.

7. **Benchmark scope:** Current correctness and runtime evaluations cover selected real-world and synthetic configurations, not arbitrary datasets or hardware environments.

8. **Privacy:** Dataset fingerprints support integrity checking but do not provide anonymization.

9. **Missing features:** The package does not currently offer a general SAT/MILP protocol compiler, nested cross-validation, a graphical dashboard, or comprehensive leakage detection.

## Research Software Status

SplitShiftLab is a functional, actively developed research prototype.

The project has completed initial automated testing, independent feasibility-boundary validation on UCI HAR and MHEALTH, and preliminary paired performance evaluations.

Before a formal archived research-software release, remaining work includes comprehensive documentation review, clean-environment installation testing, continuous integration, dependency verification, citation metadata, and a versioned source archive.

## License

Distributed under the MIT License. See `LICENSE` for the complete terms.

## Citation

Formal citation metadata and a persistent release identifier will be provided after the first archived public software release.

Until then, do not cite an unassigned DOI or unpublished SoftwareX article.