"""Step 2.6: Independently audit two UCI HAR development manifests.

Run from the project root:
    python examples/compare_uci_har_splits.py

Reads only subject identifiers from the UCI HAR *official training* CSV.
It never accesses the original UCI HAR official test partition.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "out" / "uci_har"
TRAIN_CSV = BENCHMARK_DIR / "uci_har_official_train.csv"
MANIFESTS = {
    "normal": BENCHMARK_DIR / "uci_har_baseline_manifest.json",
    "stress": BENCHMARK_DIR / "uci_har_stress_manifest.json",
}
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")


def check_manifest(name: str, report: dict, subjects: list[str]) -> dict:
    n = len(subjects)
    if int(report["input_rows"]) != n:
        raise ValueError(f"{name}: manifest row count does not match CSV")

    protocol = report["specification"]
    if protocol.get("mode") != "group" or protocol.get("group_col") != "subject":
        raise ValueError(f"{name}: expected a subject-group split")

    split = report["split"]
    train = [int(x) for x in split["train_idx"]]
    valid = [int(x) for x in split["test_idx"]]
    excluded = [int(x) for x in split.get("excluded_idx", [])]
    partitions = train + valid + excluded

    if len(partitions) != n or set(partitions) != set(range(n)):
        raise AssertionError(f"{name}: partition missing/duplicate/out-of-range row indices")
    if not split["audit"].get("valid", False):
        raise AssertionError(f"{name}: manifest reports invalid protocol")
    if split["audit"].get("violations"):
        raise AssertionError(f"{name}: manifest contains reported violations")

    train_groups = {subjects[i] for i in train}
    valid_groups = {subjects[i] for i in valid}
    overlap = train_groups & valid_groups
    if overlap:
        raise AssertionError(f"{name}: subject leakage: {sorted(overlap)}")

    scores = report.get("classifier_demo") or {}
    if any(k not in scores for k in METRICS):
        raise ValueError(f"{name}: classifier demonstration scores absent")

    return {
        "rows_train": len(train),
        "rows_validation": len(valid),
        "rows_excluded": len(excluded),
        "subjects_train": len(train_groups),
        "subjects_validation": len(valid_groups),
        "overlapping_subjects": len(overlap),
        "validation_subject_ids": sorted(valid_groups),
        **{k: float(scores[k]) for k in METRICS},
    }


def main() -> None:
    required = [TRAIN_CSV, *MANIFESTS.values()]
    absent = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if absent:
        raise FileNotFoundError("Missing files: " + ", ".join(absent))

    # Read group identities only; don't use official test data or feature labels.
    subjects = (
        pd.read_csv(TRAIN_CSV, usecols=["subject"])["subject"]
        .astype(str)
        .tolist()
    )
    reports = {name: json.loads(path.read_text(encoding="utf-8"))
               for name, path in MANIFESTS.items()}
    fingerprints = [r["split"]["metadata"].get("dataset_sha256")
                    for r in reports.values()]
    if not fingerprints[0] or len(set(fingerprints)) != 1:
        raise AssertionError("Dataset SHA256 fingerprints differ across experiments")

    checked = {name: check_manifest(name, report, subjects)
               for name, report in reports.items()}

    csv_path = BENCHMARK_DIR / "uci_har_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "normal", "stress", "stress_minus_normal"])
        for metric in METRICS:
            normal = checked["normal"][metric] * 100
            stress = checked["stress"][metric] * 100
            writer.writerow([metric + "_percent", f"{normal:.4f}",
                             f"{stress:.4f}", f"{stress-normal:+.4f}"])

    stress_meta = reports["stress"]["split"]["metadata"]
    summary = {
        "dataset": "UCI HAR official training partition only",
        "dataset_sha256": fingerprints[0],
        "normal": checked["normal"],
        "stress": checked["stress"],
        "stress_diagnostic": {
            "candidate_count": stress_meta.get("candidate_count"),
            "valid_candidate_count": stress_meta.get("valid_candidate_count"),
            "selected_approx_mmd_squared": stress_meta.get("approx_mmd_squared"),
            "mean_random_feasible_approx_mmd_squared": stress_meta.get("random_feasible_mean_mmd_squared"),
        },
        "interpretation_warning": (
            "Single-seed, different-partition development demonstration. "
            "Differences are descriptive, not causal or statistically significant. "
            "The official UCI HAR test set was not read."
        ),
    }
    summary_path = BENCHMARK_DIR / "uci_har_comparison_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("INDEPENDENT SUBJECT-SPLIT AUDIT: PASSED")
    for name, result in checked.items():
        print(f"{name.upper()}: train={result['rows_train']}, "
              f"validation={result['rows_validation']}; "
              f"subjects={result['subjects_train']} train / "
              f"{result['subjects_validation']} validation; "
              f"group overlap={result['overlapping_subjects']}")
    print("\nPERFORMANCE (development partitions only):")
    for metric in METRICS:
        a = checked["normal"][metric] * 100
        b = checked["stress"][metric] * 100
        print(f"{metric}: normal={a:.2f}%, stress={b:.2f}%, "
              f"difference={b-a:+.2f} percentage points")
    print(f"\nSaved: {csv_path.relative_to(ROOT)}")
    print(f"Saved: {summary_path.relative_to(ROOT)}")
    print("CAUTION: No significance or superiority claim follows from this comparison.")


if __name__ == "__main__":
    main()
