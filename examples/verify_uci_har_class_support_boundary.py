"""Step 2.24: independent exhaustive boundary audit for class-support constraints.

Run from the SplitShiftLab project root:
    python examples/verify_uci_har_class_support_boundary.py

Uses ONLY the official UCI HAR *development/train* CSV. Does not import
SplitShiftLab or access the official UCI HAR test split.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from math import ceil, comb
from pathlib import Path
import json

import numpy as np
import pandas as pd


def audit(csv_path: Path, test_size: float, min_train_rows: int, min_validation_rows: int) -> dict:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing development CSV: {csv_path}")
    if not (0 < test_size < 1):
        raise ValueError("test-size must be between 0 and 1")

    # Read only the metadata required for this independent audit.
    data = pd.read_csv(csv_path, usecols=["subject", "label"], dtype="string")
    if data.empty or data[["subject", "label"]].isna().any().any():
        raise ValueError("Expected non-empty data without missing subject/label identifiers")

    subjects = sorted(data["subject"].unique().tolist())
    labels = sorted(data["label"].unique().tolist())
    n_subjects = len(subjects)
    validation_groups = ceil(test_size * n_subjects)
    if not 0 < validation_groups < n_subjects:
        raise ValueError("Validation group count must be strictly between 0 and total groups")

    group_class = pd.crosstab(data["subject"], data["label"]).reindex(
        index=subjects, columns=labels, fill_value=0
    )
    matrix = group_class.to_numpy(dtype=np.int64)
    totals = matrix.sum(axis=0)
    group_sizes = matrix.sum(axis=1)
    n_rows = int(group_sizes.sum())

    feasible_support = Counter()
    eligible_allocations = 0
    max_support = -1
    witnesses_at_max = 0
    best_witness: tuple[int, ...] | None = None
    max_val_count = 0
    min_val_count = n_rows

    # Exact independent enumeration: no library solver, no model fitting,
    # no feature statistics, no protected final test set.
    for selected in combinations(range(n_subjects), validation_groups):
        idx = np.array(selected, dtype=int)
        val_counts = matrix[idx].sum(axis=0)
        train_counts = totals - val_counts
        n_val = int(group_sizes[idx].sum())
        n_train = n_rows - n_val

        if n_train < min_train_rows or n_val < min_validation_rows:
            continue
        eligible_allocations += 1
        min_val_count = min(min_val_count, n_val)
        max_val_count = max(max_val_count, n_val)
        simultaneous_min = int(min(val_counts.min(), train_counts.min()))
        feasible_support[simultaneous_min] += 1
        if simultaneous_min > max_support:
            max_support = simultaneous_min
            best_witness = selected
            witnesses_at_max = 1
        elif simultaneous_min == max_support:
            witnesses_at_max += 1

    if best_witness is None:
        raise RuntimeError("No allocation satisfies row requirements. Relax row minima first.")
    witness_indices = np.array(best_witness, dtype=int)
    witness_validation = matrix[witness_indices].sum(axis=0)
    witness_training = totals - witness_validation
    witness_val_rows = int(group_sizes[witness_indices].sum())

    count_at_20 = sum(n for support, n in feasible_support.items() if support >= 20)
    count_at_best = sum(n for support, n in feasible_support.items() if support >= max_support)
    count_beyond_best = sum(n for support, n in feasible_support.items() if support >= max_support + 1)

    return {
        "study": "Step 2.24 independent exhaustive class-support boundary audit",
        "development_only": True,
        "official_test_used": False,
        "independent_of_splitshiftlab": True,
        "input_csv": str(csv_path),
        "development_rows": n_rows,
        "subjects": n_subjects,
        "class_labels": labels,
        "validation_subjects_required": validation_groups,
        "total_allocations": comb(n_subjects, validation_groups),
        "allocations_satisfying_row_minima": eligible_allocations,
        "min_train_rows": min_train_rows,
        "min_validation_rows": min_validation_rows,
        "eligible_validation_row_range": [min_val_count, max_val_count],
        "max_feasible_minimum_per_class_both_partitions": max_support,
        "immediately_infeasible_per_class_requirement": max_support + 1,
        "allocations_meeting_minimum_20": count_at_20,
        "allocations_meeting_maximum_feasible_requirement": count_at_best,
        "allocations_meeting_immediately_infeasible_requirement": count_beyond_best,
        "witness_at_maximum": {
            "validation_subject_ids": [subjects[i] for i in best_witness],
            "training_rows": n_rows - witness_val_rows,
            "validation_rows": witness_val_rows,
            "per_class_train_counts": {lab: int(cnt) for lab, cnt in zip(labels, witness_training)},
            "per_class_validation_counts": {lab: int(cnt) for lab, cnt in zip(labels, witness_validation)},
            "subject_overlap": 0,
        },
        "limitations": [
            "This is an exact reference only for whole-group cardinality, row minima and per-class support.",
            "The maximum support threshold is derived post hoc from development labels for correctness testing.",
            "It does not verify timing constraints or establish better predictive performance.",
            "The protected official UCI HAR test set is not read.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path,
        default=Path("out/uci_har/uci_har_official_train.csv"),
        help="Development CSV, never the official test CSV",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--min-train-rows", type=int, default=2)
    parser.add_argument("--min-validation-rows", type=int, default=2)
    parser.add_argument(
        "--output", type=Path,
        default=Path("out/uci_har/uci_har_class_support_boundary_audit.json"),
    )
    args = parser.parse_args()
    if "official_test" in args.csv.name.lower():
        parser.error("Do not use the protected official test CSV for development boundary testing")

    result = audit(args.csv, args.test_size, args.min_train_rows, args.min_validation_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("INDEPENDENT CLASS-SUPPORT BOUNDARY AUDIT: COMPLETED")
    print(f"Development rows: {result['development_rows']:,} | Subjects: {result['subjects']}")
    print(f"Allocations enumerated: {result['total_allocations']:,}")
    print(f"Eligible allocations: {result['allocations_satisfying_row_minima']:,}")
    print(f"Maximum feasible minimum-per-class: {result['max_feasible_minimum_per_class_both_partitions']}")
    print(f"Allocations feasible at maximum: {result['allocations_meeting_maximum_feasible_requirement']:,}")
    print(f"Immediately infeasible minimum-per-class: {result['immediately_infeasible_per_class_requirement']}")
    print(f"Allocations feasible above maximum: {result['allocations_meeting_immediately_infeasible_requirement']}")
    print("Official UCI HAR test set NOT used.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
