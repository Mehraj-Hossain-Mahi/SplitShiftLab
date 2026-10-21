"""Property checks for experimental robust shift selection."""
import pytest
from dataclasses import replace

from splitshiftlab.cli import demo_data
from splitshiftlab.core import ProtocolSpec
from splitshiftlab.robust_shift import robust_shift_split


@pytest.fixture
def example():
    return demo_data(n_groups=15, rows_per_group=8)


def test_reproducibility_and_group_disjoint(example):
    spec = ProtocolSpec(target="label", mode="group", group_col="subject")
    a = robust_shift_split(example, spec, candidates=10, components=16, repeats=2)
    b = robust_shift_split(example, spec, candidates=10, components=16, repeats=2)
    assert a.selected.train_idx == b.selected.train_idx
    assert a.selected.test_idx == b.selected.test_idx
    assert a.selected.audit["valid"]
    assert set(example.iloc[a.selected.train_idx].subject).isdisjoint(
        set(example.iloc[a.selected.test_idx].subject))
    assert len(a.candidates) == 10
    assert a.candidates["is_selected"].sum() == 1


def test_label_permutation_does_not_change_selection(example):
    spec = ProtocolSpec(target="label", mode="group", group_col="subject")
    a = robust_shift_split(example, spec, candidates=9, components=16)
    changed = example.copy()
    changed["label"] = changed["label"].sample(frac=1, random_state=88).to_numpy()
    b = robust_shift_split(changed, spec, candidates=9, components=16)
    assert a.selected.train_idx == b.selected.train_idx
    assert a.metadata["score"] == b.metadata["score"]


def test_label_constraints_disallowed(example):
    spec = ProtocolSpec(target="label", mode="group", group_col="subject", min_class_support=1)
    with pytest.raises(ValueError, match="min_class_support"):
        robust_shift_split(example, spec)


def test_bad_bandwidth_rejected(example):
    spec = ProtocolSpec(target="label", mode="group", group_col="subject")
    with pytest.raises(ValueError, match="Bandwidths"):
        robust_shift_split(example, spec, bandwidths=(-1,))


def test_random_mode_rejected(example):
    spec = ProtocolSpec(target="label", mode="random")
    with pytest.raises(ValueError, match="group-disjoint"):
        robust_shift_split(example, spec)
