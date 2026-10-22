"""Reproduce the UCI HAR scikit-learn GroupShuffleSplit baseline.

Run from the project root:
    python examples/verify_sklearn_group_shuffle_baseline.py

Uses only the official UCI HAR development CSV.
Does not read the official test dataset.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import GroupShuffleSplit


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

    splitter = GroupShuffleSplit(
        n_splits=1000,
        test_size=0.25,
        random_state=42,
    )

    X = np.zeros((len(df), 1), dtype=np.uint8)

    supports = []
    allocations = set()
    group_disjoint = True

    for train, valid in splitter.split(X, y, groups):
        train_groups = set(groups[train])
        valid_groups = set(groups[valid])

        if train_groups & valid_groups:
            group_disjoint = False

        assert len(valid_groups) == 6

        allocations.add(tuple(sorted(valid_groups)))

        train_counts = np.bincount(
            y[train], minlength=len(labels)
        )
        valid_counts = np.bincount(
            y[valid], minlength=len(labels)
        )

        supports.append(int(min(
            train_counts.min(),
            valid_counts.min(),
        )))

    observed = {
        "version": sklearn_version,
        "splits_requested": 1000,
        "unique_allocations_sampled": len(allocations),
        "group_disjoint": group_disjoint,
        "allocations_satisfying_322": sum(
            x >= 322 for x in supports
        ),
        "allocations_satisfying_323": sum(
            x >= 323 for x in supports
        ),
        "best_observed_support": max(supports),
    }

    reference_path = Path(
        "reproducibility/evidence/sklearn_group_shuffle_baseline.json"
    )

    reference = json.loads(
        reference_path.read_text(encoding="utf-8")
    )

    for key, value in observed.items():
        assert reference[key] == value, (
            f"Mismatch for {key}: "
            f"observed={value}, reference={reference[key]}"
        )

    print("SCIKIT-LEARN BASELINE REPRODUCTION: PASSED")
    print("Version:", sklearn_version)
    print("Random splits:", observed["splits_requested"])
    print("Unique allocations:",
          observed["unique_allocations_sampled"])
    print("Support >= 322:",
          observed["allocations_satisfying_322"])
    print("Support >= 323:",
          observed["allocations_satisfying_323"])
    print("Best observed support:",
          observed["best_observed_support"])
    print("Original evidence comparison: MATCHED")


if __name__ == "__main__":
    main()
