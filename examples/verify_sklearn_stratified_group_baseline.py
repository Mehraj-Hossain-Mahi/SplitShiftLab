"""Reproduce StratifiedGroupKFold baseline on UCI HAR development data."""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedGroupKFold


def main():
    df = pd.read_csv(
        "out/uci_har/uci_har_official_train.csv",
        usecols=["subject", "label"],
    )

    groups = df["subject"].astype(str).to_numpy()
    labels, y = np.unique(
        df["label"].to_numpy(),
        return_inverse=True,
    )

    assert len(df) == 7352
    assert len(np.unique(groups)) == 21
    assert len(labels) == 6

    X = np.zeros((len(df), 1), dtype=np.uint8)

    group_counts = Counter()
    supports = []
    allocations = set()
    total_folds = 0

    for seed in range(250):
        splitter = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=seed,
        )

        for train, valid in splitter.split(X, y, groups):
            total_folds += 1

            train_groups = set(groups[train])
            valid_groups = set(groups[valid])

            assert not (train_groups & valid_groups)

            n_valid = len(valid_groups)
            group_counts[n_valid] += 1

            if n_valid != 6:
                continue

            allocations.add(tuple(sorted(valid_groups)))

            train_counts = np.bincount(
                y[train], minlength=len(labels)
            )

            valid_counts = np.bincount(
                y[valid], minlength=len(labels)
            )

            supports.append(
                int(min(train_counts.min(), valid_counts.min()))
            )

    observed = {
        "version": sklearn_version,
        "total_folds": total_folds,
        "validation_group_count_distribution": {
            str(k): v for k, v in sorted(group_counts.items())
        },
        "exact_six_group_folds": len(supports),
        "unique_six_group_allocations": len(allocations),
        "support_322_successes": sum(x >= 322 for x in supports),
        "support_323_successes": sum(x >= 323 for x in supports),
        "best_observed_support": max(supports) if supports else None,
    }

    reference = json.loads(
        Path(
            "reproducibility/evidence/sklearn_stratified_group_baseline.json"
        ).read_text(encoding="utf-8")
    )

    for key, value in observed.items():
        assert reference[key] == value, (
            f"Mismatch: {key}: observed={value}, "
            f"reference={reference[key]}"
        )

    print("STRATIFIED GROUP REPRODUCTION: PASSED")
    print("scikit-learn:", sklearn_version)
    print("Total folds:", total_folds)
    print("Group distribution:", dict(group_counts))
    print("Exact six-group folds:", len(supports))
    print("Unique allocations:", len(allocations))
    print("Support >= 322:", observed["support_322_successes"])
    print("Support >= 323:", observed["support_323_successes"])
    print("Best support:", observed["best_observed_support"])
    print("Original evidence: MATCHED")


if __name__ == "__main__":
    main()
