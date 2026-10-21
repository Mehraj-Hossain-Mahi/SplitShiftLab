import numpy as np
import pandas as pd
import pytest

from splitshiftlab import (ProtocolSpec, audit_split, dataset_fingerprint,
                           evaluate_classifier, generate_split, shift_stress_split)
from splitshiftlab.cli import demo_data
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def df():
    return demo_data(20, 10)


def test_group_disjoint(df):
    spec = ProtocolSpec(target="label", group_col="subject", features=("feature_1", "feature_2"))
    s = generate_split(df, spec)
    assert s.audit["valid"]
    assert set(df.iloc[s.train_idx].subject).isdisjoint(set(df.iloc[s.test_idx].subject))
    assert sorted(s.train_idx + s.test_idx) == list(range(len(df)))


def test_reproducible_split_and_fingerprint(df):
    spec = ProtocolSpec(target="label", mode="group", group_col="subject")
    a, b = generate_split(df, spec), generate_split(df, spec)
    assert a.train_idx == b.train_idx
    assert a.metadata["dataset_sha256"] == b.metadata["dataset_sha256"]
    assert dataset_fingerprint(df) != dataset_fingerprint(df.iloc[::-1])


def test_random_overlap_warned(df):
    spec = ProtocolSpec(target="label", mode="random", group_col="subject")
    s = generate_split(df, spec)
    assert s.audit["valid"]
    assert any("shares" in warning for warning in s.audit["warnings"])


def test_temporal_time_ordered(df):
    spec = ProtocolSpec(target="label", mode="temporal", time_col="event_time")
    s = generate_split(df, spec)
    assert s.audit["valid"]
    assert pd.to_datetime(df.iloc[s.train_idx].event_time).max() < pd.to_datetime(df.iloc[s.test_idx].event_time).min()


def test_group_temporal_time_and_group(df):
    spec = ProtocolSpec(target="label", mode="group_temporal", group_col="subject", time_col="event_time")
    s = generate_split(df, spec)
    assert s.audit["valid"]
    assert set(df.iloc[s.train_idx].subject).isdisjoint(set(df.iloc[s.test_idx].subject))
    assert pd.to_datetime(df.iloc[s.train_idx].event_time).max() < pd.to_datetime(df.iloc[s.test_idx].event_time).min()


def test_group_temporal_purges_crossing_groups():
    df = pd.DataFrame({"group": ["A", "A", "B", "B", "C", "C"],
                       "time": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-10", "2025-01-06", "2025-01-07"],
                       "x": [0, 1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 0, 1]})
    spec = ProtocolSpec(target="y", mode="group_temporal", group_col="group", time_col="time",
                        min_train_rows=2, min_test_rows=2)
    split = generate_split(df, spec)
    assert split.audit["valid"]
    assert set(split.excluded_idx) == {2, 3}


def test_invalid_missing_group(df):
    spec = ProtocolSpec(target="label", mode="group", group_col="wrong")
    with pytest.raises(ValueError, match="Required columns"):
        generate_split(df, spec)


def test_invalid_impossible_groups(df):
    df["subject"] = "only-group"
    spec = ProtocolSpec(target="label", mode="group", group_col="subject")
    with pytest.raises(ValueError, match="two distinct groups"):
        generate_split(df, spec)


def test_stress_reproducible_valid_and_uses_numeric_features(df):
    spec = ProtocolSpec(target="label", mode="group", group_col="subject",
                        features=("feature_1", "feature_2"))
    a = shift_stress_split(df, spec, candidates=30, components=64)
    b = shift_stress_split(df, spec, candidates=30, components=64)
    assert a.train_idx == b.train_idx
    assert a.audit["valid"]
    assert a.metadata["approx_mmd_squared"] >= a.metadata["random_feasible_mean_mmd_squared"]
    assert a.metadata["candidate_count"] == 30


def test_classification_demo(df):
    spec = ProtocolSpec(target="label", mode="group", group_col="subject",
                        features=("feature_1", "feature_2"))
    s = generate_split(df, spec)
    results = evaluate_classifier(df, spec, s, make_pipeline(StandardScaler(), LogisticRegression()))
    assert all(0 <= v <= 1 for v in results.values())


def test_audit_detects_manual_overlap(df):
    spec = ProtocolSpec(target="label", group_col="subject")
    split = generate_split(df, spec)
    split.test_idx.append(split.train_idx[0])
    assert not audit_split(df, spec, split)["valid"]


def test_temporal_ties_not_crossed():
    df = pd.DataFrame({"x": range(12), "y": [0, 1] * 6,
                       "t": ["2026-01-01"] * 6 + ["2026-01-02"] * 6})
    s = generate_split(df, ProtocolSpec(target="y", mode="temporal", time_col="t"))
    assert len(s.test_idx) == 6
    assert s.audit["valid"]


def test_unsupported_stress_protocol(df):
    with pytest.raises(ValueError, match="only group mode"):
        shift_stress_split(df, ProtocolSpec(target="label", mode="random"))


def test_explicit_target_leakage_rejected(df):
    spec = ProtocolSpec(target="label", mode="group", group_col="subject", features=("label", "feature_1"))
    with pytest.raises(ValueError, match="target or protected"):
        generate_split(df, spec)


def test_out_of_range_assignments_rejected(df):
    spec = ProtocolSpec(target="label", group_col="subject")
    split = generate_split(df, spec)
    split.test_idx[0] = len(df) + 100
    assert not audit_split(df, spec, split)["valid"]
