"""Independent exhaustive verification of the MHEALTH support boundary.

Run from the project root:
    python examples/verify_mhealth_independent_boundary.py

Does not import SplitShiftLab or run model training.
Compares independently computed results with archived evidence.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("out/mhealth/mhealth_all_subjects.csv")
REFERENCE = Path("reproducibility/evidence/mhealth_independent_boundary_audit.json")


def main():
    df = pd.read_csv(DATA, usecols=["subject", "label"])

    if df.empty or df.isna().any().any():
        raise ValueError("Invalid or missing dataset metadata")

    subjects = sorted(df["subject"].unique())
    labels = sorted(df["label"].unique())

    assert len(df) == 1215745
    assert len(subjects) == 10
    assert labels == list(range(13))

    counts = pd.crosstab(
        df["subject"], df["label"]
    ).reindex(
        index=subjects,
        columns=labels,
        fill_value=0
    ).to_numpy(dtype=np.int64)

    totals = counts.sum(axis=0)
    sizes = counts.sum(axis=1)

    maximum = -1
    eligible = 0
    feasible_3174 = 0
    feasible_3175 = 0
    witness = None

    for selected in combinations(range(10), 3):
        idx = list(selected)

        validation = counts[idx].sum(axis=0)
        training = totals - validation

        validation_rows = int(sizes[idx].sum())
        training_rows = len(df) - validation_rows

        if training_rows < 2 or validation_rows < 2:
            continue

        eligible += 1

        support = int(min(
            training.min(),
            validation.min()
        ))

        if support >= 3174:
            feasible_3174 += 1

        if support >= 3175:
            feasible_3175 += 1

        if support > maximum:
            maximum = support
            witness = [str(subjects[i]) for i in selected]

    observed = {
        "rows": len(df),
        "subjects": len(subjects),
        "classes": len(labels),
        "validation_subjects": 3,
        "total_allocations": 120,
        "eligible_allocations": eligible,
        "maximum_feasible_support": maximum,
        "feasible_allocations_at_3174": feasible_3174,
        "feasible_allocations_at_3175": feasible_3175,
        "witness_validation_subjects": witness,
    }

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))

    for key, value in observed.items():
        assert reference[key] == value, (
            f"Mismatch: {key}: observed={value}, "
            f"archived={reference[key]}"
        )

    assert eligible == 120
    assert maximum == 3174
    assert feasible_3174 == 8
    assert feasible_3175 == 0

    print("MHEALTH INDEPENDENT REPRODUCTION: PASSED")
    print("Records:", len(df))
    print("Allocations:", eligible)
    print("Maximum support:", maximum)
    print("Feasible at 3174:", feasible_3174)
    print("Feasible at 3175:", feasible_3175)
    print("Archived evidence comparison: MATCHED")


if __name__ == "__main__":
    main()
