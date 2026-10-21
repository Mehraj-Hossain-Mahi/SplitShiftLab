import pandas as pd

from splitshiftlab.core import ProtocolSpec
from splitshiftlab.protocol_compiler import preflight_group_protocol


def make_data(sizes, targets=None):
    data = []
    for i, n in enumerate(sizes):
        for j in range(n):
            data.append({"subject": f"P{i}", "label": (j % 2) if targets is None else targets[i], "x": j})
    return pd.DataFrame(data)


def spec(**kwargs):
    options = {"target": "label", "mode": "group", "group_col": "subject", "test_size": .5}
    options.update(kwargs)
    return ProtocolSpec(**options)


def test_feasible_constructive_witness():
    df = make_data([9, 3, 6])
    r = preflight_group_protocol(df, spec(test_size=0.34, min_train_rows=9, min_test_rows=7))
    assert r["status"] == "feasible"
    c = r["certificate"]
    assert c["validation_groups"] == 2
    assert c["subject_overlap"] == 0
    assert c["training_rows"] >= 9 and c["validation_rows"] >= 7
    assert c["training_rows"] + c["validation_rows"] == len(df)


def test_infeasible_even_when_total_rows_large_enough():
    df = make_data([10, 10, 10, 10])
    r = preflight_group_protocol(df, spec(min_train_rows=10, min_test_rows=25))
    assert r["status"] == "infeasible"
    assert r["blockers"][0]["code"] == "NO_VALID_GROUP_ALLOCATION"


def test_too_few_groups_infeasible():
    df = make_data([20])
    r = preflight_group_protocol(df, spec())
    assert r["status"] == "infeasible"


def test_unsupported_mode_is_not_falsely_feasible():
    df = make_data([7, 7])
    r = preflight_group_protocol(df, ProtocolSpec(target='label', mode='random'))
    assert r["status"] == "unsupported"


def test_resource_limit_unknown_not_infeasible():
    df = make_data([7, 7, 7])
    r = preflight_group_protocol(df, spec(), max_groups=2)
    assert r["status"] == "undetermined"


def test_class_support_not_incorrectly_certified():
    df = make_data([6, 6, 6, 6], targets=[0, 0, 1, 1])
    r = preflight_group_protocol(df, spec(min_class_support=2))
    assert r["status"] in {"undetermined", "feasible"}
    if r["status"] == "feasible":
        assert "minimum class support on this witness" in r["certificate"]["guarantees"]


def test_deterministic_certificate_and_fingerprint():
    df = make_data([10, 11, 12, 13, 14])
    r1 = preflight_group_protocol(df, spec())
    r2 = preflight_group_protocol(df, spec())
    assert r1 == r2


def test_missing_column_reports_reason():
    df = make_data([7, 7]).drop(columns='subject')
    r = preflight_group_protocol(df, spec())
    assert r['status'] == 'infeasible'
    assert r['blockers'][0]['code'] == 'INVALID_INPUT_OR_SPECIFICATION'
