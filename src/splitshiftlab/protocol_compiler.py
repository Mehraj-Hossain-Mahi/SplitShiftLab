"""Explainable feasibility preflight for group-disjoint split protocols.

Step 2.21: exact class-support feasibility for bounded group assignments.
Does not modify data, train models, or open final test data.
For group mode, proves feasibility of row minima given *exactly* the number
of test groups that scikit-learn GroupShuffleSplit would select.

It is a preflight checker, NOT a comprehensive logical constraint compiler.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import combinations, islice
from math import ceil, comb
from typing import Any

import numpy as np
import pandas as pd

from .core import ProtocolSpec, _validate_frame, dataset_fingerprint


def _violation(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _base(df: pd.DataFrame, spec: ProtocolSpec) -> dict[str, Any]:
    return {
        "software_component": "splitshiftlab.protocol_compiler",
        "stage": "Step 2.21 group and per-class feasibility preflight",
        "status": "undetermined",
        "supported_mode": "group",
        "mode": spec.mode,
        "input_rows": len(df),
        "dataset_sha256": dataset_fingerprint(df),
        "specification": asdict(spec),
        "blockers": [],
        "unresolved_requirements": [],
        "certificate": None,
        "caution": "A feasibility certificate is not an experiment, model evaluation, or a leakage-free guarantee outside checked constraints.",
    }


def _class_support_counts(
    df: pd.DataFrame, spec: ProtocolSpec, inverse_groups: np.ndarray, group_count: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Construct group-by-class counts without losing label identity."""
    class_codes, classes = pd.factorize(df[spec.target], sort=False)
    if (class_codes < 0).any():
        raise ValueError("Class-support evaluation cannot contain missing target labels")
    counts = np.zeros((group_count, len(classes)), dtype=np.int64)
    np.add.at(counts, (inverse_groups, class_codes), 1)
    return [str(value) for value in classes], counts, counts.sum(axis=0)


def _witness_support(
    group_indices: tuple[int, ...], counts: np.ndarray, total: np.ndarray, minimum: int
) -> tuple[bool, np.ndarray, np.ndarray]:
    valid = counts[list(group_indices)].sum(axis=0)
    train = total - valid
    return bool(np.all(valid >= minimum) and np.all(train >= minimum)), train, valid



def _find_class_support_allocation(
    n_groups: int,
    k: int,
    sizes: list[int],
    total_rows: int,
    min_train_rows: int,
    min_validation_rows: int,
    per_group: np.ndarray,
    totals: np.ndarray,
    minimum: int,
    *,
    batch_size: int = 1024,
) -> tuple[tuple[int, ...] | None, np.ndarray | None, np.ndarray | None, int, int]:
    """Find first exact-k feasible allocation, streaming NumPy-checked batches.

    Group combinations remain in the same lexicographic order as the original
    Python enumeration. Only a bounded batch is materialized at a time.
    Returned examined count stops at the first valid allocation, inclusive.
    A complete no-witness search proves infeasibility within the caller's cap.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    # Bound temporary advanced-indexing arrays even on many-class datasets.
    max_batch = max(1, 262_144 // max(1, k * per_group.shape[1]))
    batch_size = min(batch_size, max_batch)
    iterator = combinations(range(n_groups), k)
    checked = 0
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return None, None, None, 0, checked
        indices = np.asarray(batch, dtype=np.intp)
        row_counts = sizes_arr[indices].sum(axis=1)
        row_ok = ((row_counts >= min_validation_rows)
                  & (total_rows - row_counts >= min_train_rows))
        # Skip the more expensive class sums if no allocation in the batch
        # satisfies the basic row constraints.
        if row_ok.any():
            eligible_idx = np.flatnonzero(row_ok)
            valid_counts = per_group[indices[eligible_idx]].sum(axis=1)
            support_ok = np.all(
                (valid_counts >= minimum) & (totals - valid_counts >= minimum),
                axis=1,
            )
            hits = np.flatnonzero(support_ok)
            if hits.size:
                eligible_position = int(hits[0])
                position = int(eligible_idx[eligible_position])
                validation = valid_counts[eligible_position]
                selected = tuple(int(i) for i in indices[position])
                return (
                    selected, totals - validation, validation,
                    int(row_counts[position]), checked + position + 1,
                )
        checked += len(batch)


def preflight_group_protocol(
    df: pd.DataFrame,
    spec: ProtocolSpec,
    *,
    max_groups: int = 60,
    max_rows: int = 200_000,
    max_class_allocations: int = 250_000,
) -> dict[str, Any]:
    """Check whether an exact-k-group train/test split can meet row minima.

    Returns status='feasible' with constructive witness when verified,
    'infeasible' only when impossibility is established, or 'undetermined'
    when resource limits or unverified class-support constraints prevent proof.

    `test_size` is a GROUP fraction, as in sklearn GroupShuffleSplit,
    not a guarantee of an exact fraction of rows.

    Row feasibility is solved using cardinality-constrained subset-sum DP.
    If class support is requested, a row-feasible witness is checked first;
    then bounded exact combination search can prove existence/impossibility.
    When the exhaustive search budget is insufficient and no witness has been
    identified, status is undetermined rather than falsely infeasible.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if max_class_allocations < 1:
        raise ValueError("max_class_allocations must be >= 1")
    result = _base(df, spec)
    if spec.mode != "group":
        result["status"] = "unsupported"
        result["unresolved_requirements"] = ["Only mode='group' is supported by this preflight implementation."]
        return result
    try:
        _validate_frame(df, spec)
    except (ValueError, TypeError) as exc:
        result["status"] = "infeasible"
        result["blockers"] = [_violation("INVALID_INPUT_OR_SPECIFICATION", str(exc))]
        return result

    group_values = df[spec.group_col].to_numpy()
    labels, inv = np.unique(group_values.astype(str), return_inverse=True)
    # Explicitly guard against distinct IDs collapsing after string coercion.
    original_count = df[spec.group_col].nunique(dropna=True)
    if len(labels) != original_count:
        result["status"] = "undetermined"
        result["unresolved_requirements"] = ["Group IDs are not injective under string representation; normalize group IDs before preflight."]
        return result

    sizes = np.bincount(inv).astype(int).tolist()
    n_groups = len(sizes)
    k = ceil(n_groups * spec.test_size)
    result.update({"total_groups": n_groups, "required_validation_groups": k,
                   "required_training_groups": n_groups - k,
                   "group_fraction_not_row_fraction": True,
                   "validated_constraints": ["group cardinality", "group disjointness", "minimum row counts"]})

    if n_groups < 2 or k <= 0 or k >= n_groups:
        result["status"] = "infeasible"
        result["blockers"] = [_violation("INSUFFICIENT_GROUPS", "At least one whole group must remain in both training and validation.")]
        return result
    if n_groups > max_groups or len(df) > max_rows:
        result["status"] = "undetermined"
        result["unresolved_requirements"] = [
            f"Exact preflight capped at {max_groups} groups and {max_rows} rows; dataset exceeds this bound."
        ]
        return result

    # DP state: number of selected groups -> validation row count -> example group indices.
    states: list[dict[int, tuple[int, ...]]] = [{0: ()}] + [{} for _ in range(k)]
    for group_index, group_size in enumerate(sizes):
        for selected_count in range(min(k, group_index + 1), 0, -1):
            for sum_rows, witness in list(states[selected_count - 1].items()):
                new_sum = sum_rows + group_size
                if new_sum not in states[selected_count]:
                    states[selected_count][new_sum] = witness + (group_index,)

    valid_sums = [s for s in states[k]
                  if s >= spec.min_test_rows and len(df) - s >= spec.min_train_rows]
    if not valid_sums:
        result["status"] = "infeasible"
        result["blockers"] = [_violation(
            "NO_VALID_GROUP_ALLOCATION",
            "No allocation of exactly the required number of whole validation groups satisfies both minimum row requirements."
        )]
        result["attainable_validation_row_range"] = (
            [min(states[k]), max(states[k])] if states[k] else None
        )
        return result

    # Best documented witness: group-respecting split closest to nominal ROW proportion.
    target_rows = len(df) * spec.test_size
    best_sum = min(valid_sums, key=lambda s: (abs(s - target_rows), s))
    selected = states[k][best_sum]
    mask = np.isin(inv, selected)
    train_n = int((~mask).sum())
    valid_n = int(mask.sum())
    certificate: dict[str, Any] = {
        "validation_group_ids": labels[list(selected)].tolist(),
        "training_groups": n_groups - k,
        "validation_groups": k,
        "training_rows": train_n,
        "validation_rows": valid_n,
        "subject_overlap": 0,
        "source": "exact cardinality-constrained subset-sum dynamic programming",
        "guarantees": ["group disjointness", "minimum training rows", "minimum validation rows"],
    }
    result["certificate"] = certificate
    result["feasible_validation_row_count_options"] = len(valid_sums)

    if spec.min_class_support:
        minimum = spec.min_class_support
        class_labels, per_group, totals = _class_support_counts(df, spec, inv, n_groups)
        result["requested_minimum_class_support"] = minimum
        result["class_labels"] = class_labels
        result["class_support_search_method"] = "bounded exact combination enumeration"

        # A total of 2*m examples per class is necessary, regardless of split.
        underrepresented = [class_labels[i] for i, total in enumerate(totals)
                            if total < 2 * minimum]
        if underrepresented:
            result["status"] = "infeasible"
            result["certificate"] = None
            result["blockers"] = [_violation(
                "INSUFFICIENT_GLOBAL_CLASS_SUPPORT",
                f"Classes {underrepresented} contain fewer than 2 * {minimum} "
                "examples overall, so both partitions cannot meet the minimum."
            )]
            result["classes_with_insufficient_global_support"] = underrepresented
            return result

        found, train_support, validation_support = _witness_support(
            selected, per_group, totals, minimum
        )
        allocations_examined = 1
        exhaustive = False
        if not found:
            n_allocations = comb(n_groups, k)
            result["possible_group_allocations"] = n_allocations
            if n_allocations > max_class_allocations:
                result["status"] = "undetermined"
                result["certificate"] = None
                result["unresolved_requirements"] = [
                    f"Class-support search requires {n_allocations} group allocations, "
                    f"exceeding the limit of {max_class_allocations}; no complete "
                    "proof of class-support feasibility or infeasibility was made."
                ]
                result["class_support_allocations_examined"] = 1
                return result
            exhaustive = True
            (candidate, tr_counts, val_counts, candidate_size,
             allocations_examined) = _find_class_support_allocation(
                 n_groups, k, sizes, len(df), spec.min_train_rows,
                 spec.min_test_rows, per_group, totals, minimum,
             )
            if candidate is not None:
                selected = candidate
                train_support, validation_support = tr_counts, val_counts
                best_sum = candidate_size
                certificate["source"] = (
                    "exact-k combination search with batched class-support checks "
                    "after row-feasibility dynamic programming"
                )
                found = True
        result["class_support_allocations_examined"] = allocations_examined
        result["class_support_search_exhausted"] = bool(exhaustive and not found)
        if not found:
            result["status"] = "infeasible"
            result["certificate"] = None
            result["blockers"] = [_violation(
                "NO_VALID_CLASS_SUPPORT_ALLOCATION",
                "All exact-k whole-group allocations were checked; none satisfies "
                "minimum row counts AND minimum support for every class in both partitions."
            )]
            return result
        mask = np.isin(inv, selected)
        certificate.update({
            "validation_group_ids": labels[list(selected)].tolist(),
            "training_rows": len(df) - int(mask.sum()),
            "validation_rows": int(mask.sum()),
            "minimum_class_support": minimum,
            "per_class_train_counts": dict(zip(class_labels, map(int, train_support))),
            "per_class_validation_counts": dict(zip(class_labels, map(int, validation_support))),
        })
        certificate["guarantees"].append("minimum class support on this witness")
        result["validated_constraints"].append("minimum per-class support in both partitions")
    result["status"] = "feasible"
    return result
