# SplitShiftLab: External Software Comparison

## Purpose

This document records evidence-based distinctions between
SplitShiftLab and related machine-learning splitting tools.

It separates documented capabilities from experimental results.
No universal superiority or predictive-performance advantage
is claimed.

## 1. scikit-learn

Software: scikit-learn
Evaluated version: 1.9.1

Relevant methods:
- GroupShuffleSplit
- StratifiedGroupKFold

GroupShuffleSplit provides randomized group-disjoint partitions
with a requested fraction of groups.

StratifiedGroupKFold attempts class stratification while
maintaining non-overlapping groups.

These tools generate evaluation partitions. The specific
interfaces evaluated here were not designed to provide
constructive feasibility or exhaustive infeasibility
certificates for exact group-cardinality and minimum
per-class count requirements.

### UCI HAR development experiment

Dataset:
- 7,352 development records
- 21 subjects
- 6 classes
- 6 validation groups in the evaluated protocol

GroupShuffleSplit:
- 1,000 generated splits
- 990 unique allocations
- 0 allocations with minimum class support >= 322
- Best observed support: 317

StratifiedGroupKFold:
- 1,000 generated folds
- 250 folds with exactly 6 validation groups
- 1 unique eligible allocation
- 0 eligible folds with minimum class support >= 322
- Best observed eligible support: 269

Independent exhaustive reference:
- 54,264 possible six-group allocations
- Exactly 1 allocation satisfies support >= 322
- No allocation satisfies support >= 323

These results apply to the evaluated dataset and settings only.

Repeated folds must not be treated as independent
unique candidate allocations.

## 2. DataSAIL

Software: DataSAIL

Primary documented capabilities:
- Identity-based splitting
- Similarity-aware splitting
- Clustering-based preparation
- Optimization using integer linear programming
- Class-stratification constraints
- Split-size constraints
- Python API and command-line interface

DataSAIL primarily addresses information leakage arising
from similarity between data assigned to different splits.

Its published formulation includes stratification and
optimization constraints.

SplitShiftLab instead emphasizes auditable partition
protocols and pre-training feasibility analysis for
specific group-allocation constraints.

In the implemented group preflight, SplitShiftLab can:
- Check exact validation-group cardinality
- Check minimum partition row counts
- Check minimum per-class support in both partitions
- Return a verified feasible allocation certificate
- Establish infeasibility by exhaustive supported checks
- Return undetermined when a proof is unavailable

DataSAIL was not experimentally evaluated in this study.

The available documentation reviewed for this comparison
does not establish an equivalent API providing the exact
same three-state feasibility certificates.

This should not be interpreted as proof that DataSAIL
cannot solve related optimization problems.

No DataSAIL runtime or quality comparison is claimed.

## 3. Scope of the Contribution

SplitShiftLab is positioned as a research-software tool for:

1. Auditable ML evaluation partition specifications.
2. Constraint-feasibility preflight before model training.
3. Constructive certificates or scoped impossibility evidence.
4. Dataset fingerprints and machine-readable reporting.
5. Exploratory development-set distribution-shift diagnostics.

It does not claim comprehensive semantic leakage detection,
universally optimal partitioning, or general superiority
over existing splitting software.

## 4. Evidence Files

Independent UCI HAR verification:

reproducibility/evidence/uci_har_class_support_boundary_audit.json

GroupShuffleSplit baseline:

reproducibility/evidence/sklearn_group_shuffle_baseline.json

StratifiedGroupKFold baseline:

reproducibility/evidence/sklearn_stratified_group_baseline.json

Reproduction scripts:

examples/verify_sklearn_group_shuffle_baseline.py
examples/verify_sklearn_stratified_group_baseline.py

## 5. References

scikit-learn GroupShuffleSplit:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html

scikit-learn StratifiedGroupKFold:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

DataSAIL documentation:
https://datasail.readthedocs.io/en/latest/

Joeres et al. (2025), Data splitting to avoid
information leakage with DataSAIL.
Nature Communications.
https://doi.org/10.1038/s41467-025-58606-8

## 6. Limitations of This Comparison

- Only two scikit-learn methods were experimentally evaluated.
- DataSAIL comparison is documentation-based.
- No controlled cross-software runtime benchmark was performed.
- No downstream classifier evaluation was performed.
- General superiority is not established.
- All UCI HAR experiments used the development partition.
