"""Independent exhaustive audit of SplitShiftLab's UCI HAR group preflight.

Uses ONLY pandas, the built-in Python standard library, the prepared development
CSV and saved JSON report. Does NOT import the SplitShiftLab package, read the
official UCI HAR test set, or train a classifier.

Run from the splitshiftlab_mvp project root:
    python examples/verify_uci_har_preflight.py
"""

from __future__ import annotations

from itertools import combinations
from collections import Counter
from pathlib import Path
import json
import math

import pandas as pd

DATA = Path("out/uci_har/uci_har_official_train.csv")
PREFLIGHT = Path("out/uci_har/uci_har_preflight_report.json")
OUTPUT = Path("out/uci_har/uci_har_preflight_independent_audit.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError("Independent audit failed: " + message)


def main() -> None:
    require(DATA.exists(), f"Missing development CSV: {DATA}")
    require(PREFLIGHT.exists(), f"Missing preflight report: {PREFLIGHT}")

    # Only the group column is needed; label values and the official test set
    # are never used for deciding feasibility.
    groups = pd.read_csv(DATA, usecols=["subject"])["subject"]
    require(not groups.isna().any(), "subject column contains missing IDs")
    n_rows = len(groups)
    sizes = Counter(groups.astype(str))
    group_ids = sorted(sizes)
    report = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    feasible_case = report["feasible_case"]
    infeasible_case = report["infeasible_case"]

    for case in (feasible_case, infeasible_case):
        require(case["input_rows"] == n_rows, "dataset row count mismatches preflight report")
        require(case["mode"] == "group", "unexpected evaluation mode")
        require(case["specification"]["group_col"] == "subject", "unexpected group column")
        k = math.ceil(len(group_ids) * case["specification"]["test_size"])
        require(k == case["required_validation_groups"], "validation group count mismatches")
        require(len(group_ids) == case["total_groups"], "number of unique subjects mismatches")

    k = feasible_case["required_validation_groups"]
    require(k == infeasible_case["required_validation_groups"], "cases use different validation group counts")
    size_vector = [sizes[g] for g in group_ids]

    # Brute-force enumeration deliberately uses no DP algorithm and no
    # splitshiftlab imports. This is practical for UCI HAR: C(21,6)=54,264.
    all_row_sums: set[int] = set()
    feasible_counts = [0, 0]
    config_count = 0
    for selection in combinations(range(len(size_vector)), k):
        config_count += 1
        val_n = sum(size_vector[i] for i in selection)
        train_n = n_rows - val_n
        all_row_sums.add(val_n)
        for j, case in enumerate((feasible_case, infeasible_case)):
            spec = case["specification"]
            if train_n >= spec["min_train_rows"] and val_n >= spec["min_test_rows"]:
                feasible_counts[j] += 1

    require(config_count == math.comb(len(group_ids), k), "enumeration count is wrong")
    min_rows, max_rows = min(all_row_sums), max(all_row_sums)

    cert = feasible_case["certificate"]
    require(feasible_case["status"] == "feasible", "feasible case status mismatch")
    require(isinstance(cert, dict), "feasible case has no certificate")
    selected = cert["validation_group_ids"]
    require(len(selected) == k, "certificate selects wrong number of groups")
    require(len(set(selected)) == k, "duplicate subject in certificate")
    require(set(selected).issubset(group_ids), "certificate contains unknown subject")
    cert_val_n = sum(sizes[g] for g in selected)
    cert_train_n = n_rows - cert_val_n
    require(cert_val_n == cert["validation_rows"], "certificate validation row count incorrect")
    require(cert_train_n == cert["training_rows"], "certificate training row count incorrect")
    require(cert_val_n >= feasible_case["specification"]["min_test_rows"], "certificate misses minimum validation rows")
    require(cert_train_n >= feasible_case["specification"]["min_train_rows"], "certificate misses minimum training rows")
    require(feasible_counts[0] > 0, "exhaustive search found no feasible allocations")

    nominal = n_rows * feasible_case["specification"]["test_size"]
    eligible_sums = {
        val for val in all_row_sums
        if val >= feasible_case["specification"]["min_test_rows"]
        and n_rows - val >= feasible_case["specification"]["min_train_rows"]
    }
    closest_row_sum = min(eligible_sums, key=lambda val: (abs(val - nominal), val))
    require(cert_val_n == closest_row_sum, "certificate does not achieve closest feasible row count")
    if "feasible_validation_row_count_options" in feasible_case:
        require(feasible_case["feasible_validation_row_count_options"] == len(eligible_sums),
                "number of attainable feasible validation row counts mismatches")

    require(infeasible_case["status"] == "infeasible", "infeasible case status mismatch")
    require(feasible_counts[1] == 0, "brute-force search found a feasible supposedly impossible allocation")
    require(infeasible_case["certificate"] is None, "infeasible case contains a certificate")
    require(infeasible_case["blockers"][0]["code"] == "NO_VALID_GROUP_ALLOCATION",
            "infeasible case gives unexpected blocker")
    require(infeasible_case.get("attainable_validation_row_range") == [min_rows, max_rows],
            "attainable row range mismatches exhaustive search")

    audit = {
        "study": "Step 2.15 independent exhaustive feasibility audit of development subjects",
        "official_test_used": False,
        "implementation_independent_of_splitshiftlab": True,
        "development_records": n_rows,
        "subjects": len(group_ids),
        "validation_subjects_required": k,
        "allocations_enumerated": config_count,
        "attainable_validation_row_range": [min_rows, max_rows],
        "distinct_attainable_validation_row_counts": len(all_row_sums),
        "feasible_case": {
            "independently_validated": True,
            "feasible_allocation_count": feasible_counts[0],
            "validation_rows_in_certificate": cert_val_n,
            "training_rows_in_certificate": cert_train_n,
            "closest_feasible_validation_rows_to_nominal": closest_row_sum,
            "distinct_feasible_validation_row_counts": len(eligible_sums),
        },
        "infeasible_case": {
            "independently_validated": True,
            "feasible_allocation_count": feasible_counts[1],
            "requested_min_validation_rows": infeasible_case["specification"]["min_test_rows"],
        },
        "limitations": [
            "Independent exhaustive verification is limited to exact-k group cardinality and minimum row counts.",
            "It does not establish correctness of arbitrary future class, time, or feature constraints.",
            "No classifier performance or generalization claims are implied.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("INDEPENDENT EXHAUSTIVE PREFLIGHT AUDIT: PASSED")
    print(f"Enumerated {config_count:,} different assignments of {k} from {len(group_ids)} subjects")
    print(f"Attainable validation-row range: {min_rows:,} to {max_rows:,}")
    print(f"Feasible report: {cert_train_n:,} training / {cert_val_n:,} validation rows verified")
    print(f"Feasible group allocations satisfying constraints: {feasible_counts[0]:,}")
    print(f"Infeasible report: {feasible_counts[1]} allocations satisfy the requested constraints")
    print("Official UCI HAR test data was NOT opened.")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
