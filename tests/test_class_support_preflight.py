"""Exact per-class preflight; independent combination oracle for small cases."""
from itertools import combinations
from math import ceil

import pandas as pd

from splitshiftlab.core import ProtocolSpec
from splitshiftlab.protocol_compiler import preflight_group_protocol


def df_from_group_labels(groups):
    """One row per specified class value within each group."""
    return pd.DataFrame([
        {'subject': f'P{i}', 'label': cls, 'x': j}
        for i, labels in enumerate(groups) for j, cls in enumerate(labels)
    ])


def spec(**options):
    values = dict(target='label', mode='group', group_col='subject',
                  test_size=.5, min_class_support=2)
    values.update(options)
    return ProtocolSpec(**values)


def oracle_exists(groups, fraction, min_train, min_validation, support):
    df = df_from_group_labels(groups)
    k = ceil(len(groups)*fraction)
    subjects = sorted(df.subject.unique())
    labels = df.label.unique()
    for chosen in combinations(subjects, k):
        mask = df.subject.isin(chosen)
        tr, va = df.loc[~mask], df.loc[mask]
        if len(tr) < min_train or len(va) < min_validation:
            continue
        if all(tr.label.eq(cls).sum() >= support and va.label.eq(cls).sum() >= support for cls in labels):
            return True
    return False


def test_other_allocation_is_found_when_dp_witness_lacks_class():
    groups = [[0] * 5, [0] * 5, [1] * 5, [1] * 5]
    result = preflight_group_protocol(df_from_group_labels(groups), spec())
    assert result['status'] == 'feasible'
    cert = result['certificate']
    assert cert['per_class_validation_counts'] == {'0': 5, '1': 5}
    assert cert['per_class_train_counts'] == {'0': 5, '1': 5}
    assert 'minimum per-class support in both partitions' in result['validated_constraints']


def test_class_impossible_even_though_global_support_sufficient():
    groups = [[0]*10, [1]*10, [1]*10, [1]*10]
    result = preflight_group_protocol(df_from_group_labels(groups), spec())
    assert result['status'] == 'infeasible'
    assert result['blockers'][0]['code'] == 'NO_VALID_CLASS_SUPPORT_ALLOCATION'
    assert result['certificate'] is None
    assert result['class_support_search_exhausted'] is True


def test_insufficient_total_support_is_immediately_impossible():
    groups = [[0]*2, [1]*5, [1]*5, [1]*5]
    result = preflight_group_protocol(df_from_group_labels(groups), spec())
    assert result['status'] == 'infeasible'
    assert result['blockers'][0]['code'] == 'INSUFFICIENT_GLOBAL_CLASS_SUPPORT'


def test_large_exact_search_is_undetermined_not_false_impossibility():
    groups = [[0]*5 for _ in range(3)] + [[1]*5 for _ in range(3)]
    result = preflight_group_protocol(df_from_group_labels(groups), spec(), max_class_allocations=2)
    assert result['status'] == 'undetermined'
    assert result['certificate'] is None
    assert result['blockers'] == []


def test_group_support_can_be_certified_above_low_budget():
    groups = [[0, 0, 1, 1]]*6
    result = preflight_group_protocol(df_from_group_labels(groups), spec(), max_class_allocations=1)
    assert result['status'] == 'feasible'
    assert result['certificate']['per_class_train_counts'] == {'0': 6, '1': 6}


def test_exact_oracle_multiple_small_cases():
    cases = [
      ([[0]*4,[0]*4,[1]*4,[1]*4],.5,1,1,2),
      ([[0]*7,[1]*7,[1]*7,[1]*7],.5,1,1,1),
      ([[0,1,0,1], [1,1,0,0], [0,0,1,1]],.34,4,4,2),
      ([[0]*3,[1]*3,[0]*3,[1]*3,[0]*3],.4,3,3,2),
    ]
    for groups, fraction, min_tr, min_val, m in cases:
        result = preflight_group_protocol(df_from_group_labels(groups),
            spec(test_size=fraction, min_train_rows=min_tr,
                 min_test_rows=min_val, min_class_support=m))
        expected = oracle_exists(groups, fraction, min_tr, min_val, m)
        assert result['status'] == ('feasible' if expected else 'infeasible')
        if expected:
            cert = result['certificate']
            assert min(cert['per_class_train_counts'].values()) >= m
            assert min(cert['per_class_validation_counts'].values()) >= m


def test_cli_class_support_result(tmp_path):
    import json
    from splitshiftlab.cli import main
    groups = [[0]*5,[0]*5,[1]*5,[1]*5]
    csv = tmp_path / 'development.csv'
    df_from_group_labels(groups).to_csv(csv, index=False)
    outfile = tmp_path / 'report.json'
    main(['preflight','--csv',str(csv),'--target','label','--group','subject',
          '--test-size','0.5','--minimum-class-support','2','--output',str(outfile)])
    saved = json.loads(outfile.read_text())
    assert saved['status'] == 'feasible'
    assert saved['certificate']['minimum_class_support'] == 2


def test_cli_respects_class_search_budget(tmp_path):
    import json
    from splitshiftlab.cli import main
    groups = [[0]*5 for _ in range(3)] + [[1]*5 for _ in range(3)]
    csv = tmp_path / 'development.csv'
    df_from_group_labels(groups).to_csv(csv, index=False)
    outfile = tmp_path / 'capped.json'
    main(['preflight','--csv',str(csv),'--target','label','--group','subject',
          '--test-size','0.5','--minimum-class-support','2',
          '--max-class-allocations','1','--output',str(outfile)])
    saved = json.loads(outfile.read_text())
    assert saved['status'] == 'undetermined'
    assert saved['certificate'] is None
