"""Step 2.23 — independently audit UCI HAR per-class group-split feasibility.

Place this file in ``splitshiftlab_mvp/examples/`` and run from the project root:

    python examples/verify_uci_har_class_support.py

The implementation uses only pandas/numpy and Python's standard library.
It does NOT import splitshiftlab, train an estimator, or open the official
UCI HAR test partition. All exhaustive calculations use development data.
"""

from __future__ import annotations

from itertools import combinations
from math import ceil, comb
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "out" / "uci_har" / "uci_har_official_train.csv"
PREFLIGHT = ROOT / "out" / "uci_har" / "uci_har_class_support_feasible.json"
OUTPUT = ROOT / "out" / "uci_har" / "uci_har_class_support_independent_audit.json"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise AssertionError("Independent class-support audit FAILED: " + reason)


def main() -> None:
    require(DATA.is_file(), f"Missing development-only input: {DATA}")
    require(PREFLIGHT.is_file(), f"Missing Step 2.22 JSON: {PREFLIGHT}")

    report = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    spec = report["specification"]
    require(report["status"] == "feasible", "report does not claim feasibility")
    require(spec["mode"] == "group", "unexpected split mode")
    require(spec["group_col"] == "subject", "unexpected group column")
    require(spec["target"] == "label", "unexpected target column")
    minimum = int(spec["min_class_support"])
    require(minimum > 0, "class-support minimum must be > 0")

    # Deliberately read only subject IDs and labels. Features and protected
    # official test data are not used or opened.
    data = pd.read_csv(DATA, usecols=["subject", "label"])
    require(not data.isna().any().any(), "missing subject or label values")
    require(len(data) == int(report["input_rows"]), "development record count mismatch")

    subjects = data["subject"].astype(str)
    labels = data["label"].astype(str)
    subject_ids = sorted(subjects.unique().tolist())
    class_ids = sorted(labels.unique().tolist())
    n_groups = len(subject_ids)
    n_records = len(data)
    k = ceil(n_groups * float(spec["test_size"]))
    require(k == int(report["required_validation_groups"]), "validation group count mismatch")
    require(k > 0 and k < n_groups, "both partitions must have groups")
    total_combinations = comb(n_groups, k)
    require(total_combinations <= 500_000, "exhaustive search safety limit exceeded")

    subject_codes = pd.Categorical(subjects, categories=subject_ids).codes
    label_codes = pd.Categorical(labels, categories=class_ids).codes
    counts = np.zeros((n_groups, len(class_ids)), dtype=np.int64)
    np.add.at(counts, (subject_codes, label_codes), 1)
    group_size = counts.sum(axis=1)
    total_class_counts = counts.sum(axis=0)

    witness = report["certificate"]
    require(isinstance(witness, dict), "feasible report has no certificate")
    chosen_subjects = witness["validation_group_ids"]
    require(len(chosen_subjects) == k, "incorrect number of certificate groups")
    require(len(set(chosen_subjects)) == k, "duplicate group ID in certificate")
    require(set(chosen_subjects) <= set(subject_ids), "unknown group ID in certificate")
    selected_indices = [subject_ids.index(x) for x in chosen_subjects]
    witness_valid_counts = counts[selected_indices].sum(axis=0)
    witness_train_counts = total_class_counts - witness_valid_counts
    witness_validation_rows = int(witness_valid_counts.sum())
    witness_training_rows = n_records - witness_validation_rows

    require(witness_validation_rows == int(witness["validation_rows"]),
            "certificate validation row count incorrect")
    require(witness_training_rows == int(witness["training_rows"]),
            "certificate training row count incorrect")
    require(int(witness["validation_groups"]) == k, "validation group count incorrect")
    require(int(witness["training_groups"]) == n_groups - k, "training group count incorrect")
    require(witness_validation_rows >= int(spec["min_test_rows"]),
            "certificate violates validation row minimum")
    require(witness_training_rows >= int(spec["min_train_rows"]),
            "certificate violates training row minimum")
    require(np.all(witness_valid_counts >= minimum), "validation class minimum violated")
    require(np.all(witness_train_counts >= minimum), "training class minimum violated")

    for col, actual in (("per_class_train_counts", witness_train_counts),
                        ("per_class_validation_counts", witness_valid_counts)):
        claimed = {str(c): int(v) for c, v in witness[col].items()}
        independently_computed = dict(zip(class_ids, map(int, actual)))
        require(claimed == independently_computed, f"incorrect {col} in certificate")

    # Exhaustive enumeration independent of subset-sum DP and the package's
    # bounded combination search. This demonstrates both existence and the
    # number of satisfying whole-subject allocations for this small benchmark.
    feasible_count = 0
    row_feasible_count = 0
    examined = 0
    min_rows, max_rows = n_records, 0
    for selection in combinations(range(n_groups), k):
        examined += 1
        selected = counts[list(selection)].sum(axis=0)
        remaining = total_class_counts - selected
        valid_rows = int(selected.sum())
        train_rows = n_records - valid_rows
        min_rows = min(min_rows, valid_rows)
        max_rows = max(max_rows, valid_rows)
        rows_ok = (train_rows >= int(spec["min_train_rows"]) and
                   valid_rows >= int(spec["min_test_rows"]))
        if rows_ok:
            row_feasible_count += 1
            if np.all(selected >= minimum) and np.all(remaining >= minimum):
                feasible_count += 1

    require(examined == total_combinations, "exhaustive count mismatch")
    require(feasible_count > 0, "exhaustive verification found no feasible allocation")

    # A necessary-condition impossible example with a reproducible bound:
    # if both sides require > half of the rarest class, no allocation works.
    rarest_total = int(total_class_counts.min())
    impossible_minimum = rarest_total // 2 + 1
    require(2 * impossible_minimum > rarest_total,
            "impossible-case arithmetic check failed")

    report_out = {
        "study": "Step 2.23 independent exhaustive per-class feasibility verification",
        "official_test_used": False,
        "implementation_independent_of_splitshiftlab": True,
        "source": str(DATA),
        "development_rows": n_records,
        "subject_groups": n_groups,
        "activity_classes": class_ids,
        "required_validation_subjects": k,
        "allocations_enumerated": examined,
        "allocations_satisfying_row_minima": row_feasible_count,
        "allocations_satisfying_row_and_class_minima": feasible_count,
        "attainable_validation_row_range": [min_rows, max_rows],
        "requested_minimum_class_support": minimum,
        "certificate_verified": True,
        "certificate_training_rows": witness_training_rows,
        "certificate_validation_rows": witness_validation_rows,
        "certificate_training_subjects": n_groups - k,
        "certificate_validation_subjects": k,
        "certificate_subject_overlap": 0,
        "independent_train_class_counts": dict(zip(class_ids, map(int, witness_train_counts))),
        "independent_validation_class_counts": dict(zip(class_ids, map(int, witness_valid_counts))),
        "rarest_class_total": rarest_total,
        "mathematically_impossible_minimum_class_support_both_partitions": impossible_minimum,
        "limitations": [
            "Proves correctness of the reported witness and counts feasible allocations for this UCI HAR configuration.",
            "The impossible-support example is based on a necessary global-count bound, not an arbitrary solver test.",
            "Does not prove correctness of all future constraints, datasets, time splits, or search-budget behavior.",
            "No model training, predictive improvement, generalization, or statistical significance is claimed.",
            "Dataset contents are checked by row/class/group counts, not an independently reproduced package SHA-256 fingerprint.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report_out, indent=2) + "\n", encoding="utf-8")
    print("INDEPENDENT CLASS-SUPPORT AUDIT: PASSED")
    print(f"Development dataset: {n_records:,} records, {n_groups} subjects, {len(class_ids)} classes")
    print(f"Combinations enumerated: {examined:,}")
    print(f"Valid whole-group allocations for minimum {minimum}/class/partition: {feasible_count:,}")
    print(f"Certificate verified: train={witness_training_rows:,}, validation={witness_validation_rows:,}")
    print("Official UCI HAR test set NOT used.")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
