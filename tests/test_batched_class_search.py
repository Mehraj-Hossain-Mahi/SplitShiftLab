"""Regression tests for bounded, vectorized exact-k class-support search."""
from itertools import combinations
from math import ceil

import numpy as np
import pandas as pd
import pytest

from splitshiftlab.core import ProtocolSpec
from splitshiftlab.protocol_compiler import (
    _find_class_support_allocation, preflight_group_protocol,
)


def reference(n, k, sizes, min_train, min_validation, classes, support):
    total_rows = sum(sizes)
    totals = classes.sum(axis=0)
    for checked, indices in enumerate(combinations(range(n), k), start=1):
        nr = sum(sizes[i] for i in indices)
        if nr < min_validation or total_rows - nr < min_train:
            continue
        val = classes[list(indices)].sum(axis=0)
        if np.all(val >= support) and np.all(totals - val >= support):
            return indices, checked, val
    return None, checked, None


@pytest.mark.parametrize('seed', [3, 4, 5, 6, 7, 8])
def test_vector_search_matches_independent_bruteforce(seed):
    rng = np.random.default_rng(seed)
    n, k, c = 9, 4, 3
    classes = rng.integers(0, 14, (n, c), dtype=np.int64)
    sizes = classes.sum(axis=1).tolist()
    min_train, min_val, support = 15, 12, 17
    expected, expected_checked, expected_values = reference(
        n, k, sizes, min_train, min_val, classes, support
    )
    for batch_size in (1, 5, 10, 128):
        selected, train, val, rows, checked = _find_class_support_allocation(
            n, k, sizes, sum(sizes), min_train, min_val,
            classes, classes.sum(axis=0), support, batch_size=batch_size,
        )
        assert selected == expected
        assert checked == expected_checked
        if expected is None:
            assert train is None and val is None
        else:
            np.testing.assert_array_equal(val, expected_values)
            np.testing.assert_array_equal(train, classes.sum(axis=0) - expected_values)
            assert rows == sum(sizes[i] for i in expected)


def test_no_witness_exhaustive_count_exact():
    classes = np.array([[10, 0], [0, 10], [10, 0], [0, 10]], dtype=np.int64)
    r = _find_class_support_allocation(4, 2, [10]*4, 40, 1, 1,
                                      classes, classes.sum(axis=0), 11, batch_size=2)
    assert r[0] is None
    assert r[4] == 6


def test_only_feasible_candidate_at_batch_boundary():
    # Three classes concentrated in three groups, all must be in validation.
    classes = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10],
                        [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
    # This arrangement is intentionally impossible for support in BOTH parts.
    r = _find_class_support_allocation(6, 3, classes.sum(axis=1).tolist(),
                                      30, 1, 1, classes, classes.sum(axis=0), 1,
                                      batch_size=4)
    assert r[0] is None
    assert r[4] == 20


def test_full_preflight_returns_class_certificate_on_non_dp_witness():
    rows=[]
    for group, labels in enumerate([[0]*5,[0]*5,[1]*5,[1]*5]):
        rows.extend({'subject':f'p{group}','label':label} for label in labels)
    report=preflight_group_protocol(pd.DataFrame(rows), ProtocolSpec(
        target='label', mode='group', group_col='subject',
        test_size=.5, min_class_support=2))
    assert report['status']=='feasible'
    assert report['certificate']['subject_overlap']==0
    assert 'batched class-support checks' in report['certificate']['source']
    assert report['certificate']['per_class_validation_counts']=={'0':5,'1':5}


def test_budget_guard_still_returns_undetermined():
    rows=[]
    for i, labels in enumerate([[0]*5 for _ in range(3)] + [[1]*5 for _ in range(3)]):
        rows.extend({'subject':f'p{i}', 'label':label} for label in labels)
    report=preflight_group_protocol(pd.DataFrame(rows), ProtocolSpec(
        target='label', mode='group', group_col='subject', test_size=.5,
        min_class_support=2),max_class_allocations=1)
    assert report['status']=='undetermined'
    assert report['class_support_allocations_examined']==1
