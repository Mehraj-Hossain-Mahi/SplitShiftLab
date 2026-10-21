"""Auditable split protocols and development-set distribution-shift diagnostics.

Guarantees are limited to the explicit constraints implemented and checked here.
In particular, this module cannot discover all semantic or pipeline leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Literal

import numpy as np
import pandas as pd


Mode = Literal["random", "group", "temporal", "group_temporal"]


@dataclass(frozen=True)
class ProtocolSpec:
    target: str
    mode: Mode = "group"
    group_col: str | None = None
    time_col: str | None = None
    features: tuple[str, ...] | None = None
    test_size: float = 0.25
    seed: int = 42
    min_train_rows: int = 2
    min_test_rows: int = 2
    min_class_support: int = 0  # only use for classification, requires target labels
    stratify_random: bool = False

    def __post_init__(self):
        if self.mode not in {"random", "group", "temporal", "group_temporal"}:
            raise ValueError(f"Unknown split mode: {self.mode}")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.min_train_rows < 1 or self.min_test_rows < 1 or self.min_class_support < 0:
            raise ValueError("Row and class support minima must be nonnegative (rows >= 1)")
        if self.mode in {"group", "group_temporal"} and not self.group_col:
            raise ValueError(f"{self.mode} mode requires group_col")
        if self.mode in {"temporal", "group_temporal"} and not self.time_col:
            raise ValueError(f"{self.mode} mode requires time_col")


@dataclass
class SplitResult:
    train_idx: list[int]
    test_idx: list[int]
    excluded_idx: list[int] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_frame(df: pd.DataFrame, spec: ProtocolSpec) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    required = [spec.target]
    if spec.group_col:
        required.append(spec.group_col)
    if spec.time_col:
        required.append(spec.time_col)
    if spec.features:
        required += list(spec.features)
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Required columns not found: {missing}")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported")
    if spec.features is not None:
        forbidden = {spec.target, spec.group_col, spec.time_col}.intersection(spec.features)
        if forbidden:
            raise ValueError(f"Feature list includes target or protected metadata columns: {sorted(forbidden)}")
        if len(set(spec.features)) != len(spec.features):
            raise ValueError("Feature list must not contain duplicates")
    if len(df) < spec.min_train_rows + spec.min_test_rows:
        raise ValueError("Not enough rows to satisfy minimum split sizes")
    if spec.group_col and df[spec.group_col].isna().any():
        raise ValueError("Missing group IDs; clean these before splitting")
    if spec.time_col:
        _timestamps(df, spec.time_col)
    if spec.min_class_support and df[spec.target].isna().any():
        raise ValueError("Missing target labels when min_class_support is enabled")


def _timestamps(df: pd.DataFrame, column: str) -> pd.Series:
    ts = pd.to_datetime(df[column], errors="coerce", utc=True)
    if ts.isna().any():
        raise ValueError(f"Column {column!r} contains missing/invalid timestamps")
    return ts.reset_index(drop=True)


def _selected_features(df: pd.DataFrame, spec: ProtocolSpec) -> list[str]:
    if spec.features is not None:
        return list(spec.features)
    excluded = {spec.target, spec.group_col, spec.time_col}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """SHA256 for dataset order, column names, dtypes and pandas row hashes."""
    header = json.dumps({"columns": list(map(str, df.columns)),
                         "dtypes": list(map(str, df.dtypes)),
                         "shape": list(df.shape)}, sort_keys=True).encode()
    hashes = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(header + hashes.tobytes()).hexdigest()


def _make_result(train: np.ndarray, test: np.ndarray, n: int, method: str) -> SplitResult:
    train = np.asarray(train, dtype=int)
    test = np.asarray(test, dtype=int)
    used = set(train.tolist()) | set(test.tolist())
    return SplitResult(sorted(map(int, train)), sorted(map(int, test)),
                       sorted(set(range(n)) - used), metadata={"method": method})


def _split_random(df: pd.DataFrame, spec: ProtocolSpec, attempt: int) -> SplitResult:
    from sklearn.model_selection import train_test_split

    ix = np.arange(len(df))
    tr, te = train_test_split(ix, test_size=spec.test_size, random_state=spec.seed + attempt,
                              stratify=df[spec.target] if spec.stratify_random else None)
    return _make_result(tr, te, len(df), "random")


def _split_group(df: pd.DataFrame, spec: ProtocolSpec, attempt: int) -> SplitResult:
    from sklearn.model_selection import GroupShuffleSplit

    groups = df[spec.group_col].to_numpy()
    if pd.Series(groups).nunique() < 2:
        raise ValueError("At least two distinct groups are required")
    splitter = GroupShuffleSplit(n_splits=1, test_size=spec.test_size,
                                 random_state=spec.seed + attempt)
    tr, te = next(splitter.split(np.zeros(len(df)), groups=groups))
    return _make_result(tr, te, len(df), "group")


def _split_temporal(df: pd.DataFrame, spec: ProtocolSpec) -> SplitResult:
    ts = _timestamps(df, spec.time_col)
    # Never split equal timestamps across train and test.
    candidates = sorted(ts.unique())[:-1]
    valid = []
    for cutoff in candidates:
        train = np.flatnonzero((ts <= cutoff).to_numpy())
        test = np.flatnonzero((ts > cutoff).to_numpy())
        if len(train) >= spec.min_train_rows and len(test) >= spec.min_test_rows:
            valid.append((abs(len(test)/len(df) - spec.test_size), cutoff, train, test))
    if not valid:
        raise ValueError("No feasible chronological split (including tied timestamps)")
    _, _, train, test = min(valid, key=lambda x: x[0])
    return _make_result(train, test, len(df), "temporal")


def _split_group_temporal(df: pd.DataFrame, spec: ProtocolSpec) -> SplitResult:
    """Chronologically ordered, entity-disjoint splits; purge crossing groups."""
    ts = _timestamps(df, spec.time_col)
    groups = pd.Series(df[spec.group_col].to_numpy())
    info = pd.DataFrame({"group": groups, "ts": ts}).groupby("group", dropna=False)["ts"].agg(["min", "max"])
    if len(info) < 2:
        raise ValueError("Group-temporal split requires at least two groups")
    eligible_cutoffs = sorted(info["max"].unique())[:-1]
    valid = []
    for cutoff in eligible_cutoffs:
        training_groups = set(info.index[info["max"] <= cutoff])
        testing_groups = set(info.index[info["min"] > cutoff])
        train = np.flatnonzero(groups.isin(training_groups).to_numpy())
        test = np.flatnonzero(groups.isin(testing_groups).to_numpy())
        if len(train) < spec.min_train_rows or len(test) < spec.min_test_rows:
            continue
        excluded = len(df) - len(train) - len(test)
        # Choose feasible cutoffs with fewer purged rows, then closer test fraction.
        objective = (excluded / len(df) +
                     abs(len(test) / (len(train) + len(test)) - spec.test_size))
        valid.append((objective, excluded, str(cutoff), train, test))
    if not valid:
        raise ValueError("Group and chronological constraints are incompatible; no valid split")
    _, _, _, train, test = min(valid, key=lambda x: (x[0], x[1], x[2]))
    return _make_result(train, test, len(df), "group_temporal")


def audit_split(df: pd.DataFrame, spec: ProtocolSpec, result: SplitResult) -> dict[str, Any]:
    """Validate explicit hard constraints; report limited duplicate-feature warnings."""
    n = len(df)
    tr, te, ex = set(result.train_idx), set(result.test_idx), set(result.excluded_idx)
    issues: list[str] = []
    warnings: list[str] = []
    if any(len(values) != len(set(values)) for values in (result.train_idx, result.test_idx, result.excluded_idx)):
        issues.append("Duplicate row assignment within a partition")
    if any(ix < 0 or ix >= n for ix in (tr | te | ex)):
        issues.append("Out-of-range row indices")
        return {"valid": False, "violations": issues, "warnings": [], "n_train": len(tr), "n_test": len(te), "n_excluded": len(ex), "checked_constraints": []}
    if (tr & te) or (tr & ex) or (te & ex):
        issues.append("A row is assigned to multiple partitions")
    if (tr | te | ex) != set(range(n)):
        issues.append("Partitions do not cover input row positions exactly once")
    if len(tr) < spec.min_train_rows or len(te) < spec.min_test_rows:
        issues.append("Minimum partition size violated")
    if spec.mode != "group_temporal" and ex:
        issues.append("Unexpected exclusion of rows for this protocol")
    if spec.mode in {"group", "group_temporal"} and tr and te:
        train_groups = set(df.iloc[sorted(tr)][spec.group_col])
        test_groups = set(df.iloc[sorted(te)][spec.group_col])
        if train_groups & test_groups:
            issues.append("Group overlap across train/test")
    if spec.mode in {"temporal", "group_temporal"} and tr and te:
        ts = _timestamps(df, spec.time_col)
        if ts.iloc[sorted(tr)].max() >= ts.iloc[sorted(te)].min():
            issues.append("Temporal ordering is not strict")
    if spec.min_class_support and tr and te:
        for ix, label in [(tr, "training"), (te, "test")]:
            counts = df.iloc[sorted(ix)][spec.target].value_counts(dropna=False)
            all_classes = set(df[spec.target].dropna().unique())
            if any(counts.get(label_value, 0) < spec.min_class_support for label_value in all_classes):
                issues.append(f"Minimum class support violated in {label}")
    feat = _selected_features(df, spec)
    dup_count = 0
    if feat and tr and te:
        # Exact feature duplicates are warnings: repeated feature vectors can be legitimate.
        train_hashes = set(pd.util.hash_pandas_object(df.iloc[sorted(tr)][feat], index=False).tolist())
        test_hashes = pd.util.hash_pandas_object(df.iloc[sorted(te)][feat], index=False)
        dup_count = int(test_hashes.isin(train_hashes).sum())
        if dup_count:
            warnings.append(f"{dup_count} test rows have exact feature matches in training; inspect for leakage")
    if spec.mode == "random" and spec.group_col and tr and te:
        overlapped = set(df.iloc[sorted(tr)][spec.group_col]) & set(df.iloc[sorted(te)][spec.group_col])
        if overlapped:
            warnings.append(f"Random split shares {len(overlapped)} group IDs between train and test")
    return {"valid": not issues, "violations": issues, "warnings": warnings,
            "n_train": len(tr), "n_test": len(te), "n_excluded": len(ex),
            "exact_feature_duplicate_test_rows": dup_count,
            "checked_constraints": ["unique_partition", "partition_coverage", "minimum_size"] +
                (["group_disjoint"] if spec.mode in {"group", "group_temporal"} else []) +
                (["strict_chronology"] if spec.mode in {"temporal", "group_temporal"} else []) +
                (["minimum_class_support"] if spec.min_class_support else [])}


def generate_split(df: pd.DataFrame, spec: ProtocolSpec, max_attempts: int = 100) -> SplitResult:
    _validate_frame(df, spec)
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    for attempt in range(max_attempts):
        if spec.mode == "random":
            result = _split_random(df, spec, attempt)
        elif spec.mode == "group":
            result = _split_group(df, spec, attempt)
        elif spec.mode == "temporal":
            result = _split_temporal(df, spec)
        else:
            result = _split_group_temporal(df, spec)
        result.audit = audit_split(df, spec, result)
        if result.audit["valid"]:
            result.metadata.update({"attempt": attempt + 1, "seed": spec.seed,
                                    "dataset_sha256": dataset_fingerprint(df),
                                    "protocol": asdict(spec)})
            return result
        if spec.mode in {"temporal", "group_temporal"}:
            break
    raise ValueError("Could not satisfy protocol constraints: " + "; ".join(result.audit["violations"]))


def _rff_embeddings(df: pd.DataFrame, features: list[str], seed: int,
                    components: int = 256) -> np.ndarray:
    X = df[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    if X.isna().all().any():
        raise ValueError("Stress-test feature columns cannot be entirely missing")
    X = X.fillna(X.median()).to_numpy(dtype=float)
    center = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale < 1e-12] = 1
    X = (X - center) / scale
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1.0 / np.sqrt(max(1, X.shape[1])), (X.shape[1], components))
    b = rng.uniform(0, 2 * np.pi, components)
    return np.sqrt(2 / components) * np.cos(X @ w + b)


def shift_stress_split(df: pd.DataFrame, spec: ProtocolSpec,
                       candidates: int = 80, components: int = 256) -> SplitResult:
    """Maximize approximate RBF MMD among feasible group splits of DEVELOPMENT data.

    Only input features enter optimization. This is a diagnostic stress test,
    NOT a statistically representative estimate of deployment performance.
    Never pass the reserved final test data to this function.
    """
    if spec.mode != "group":
        raise ValueError("The v0.1 stress optimizer currently supports only group mode")
    if candidates < 2 or components < 8:
        raise ValueError("Use at least 2 candidate splits and 8 RFF components")
    _validate_frame(df, spec)
    features = _selected_features(df, spec)
    if not features:
        raise ValueError("No numeric features available for shift scoring")
    if any(not pd.api.types.is_numeric_dtype(df[c]) for c in features):
        raise ValueError("Shift scoring requires numeric feature columns")
    z = _rff_embeddings(df, features, spec.seed, components)
    scored = []
    for attempt in range(candidates):
        result = _split_group(df, spec, attempt)
        result.audit = audit_split(df, spec, result)
        if not result.audit["valid"]:
            continue
        m1, m2 = z[result.train_idx].mean(axis=0), z[result.test_idx].mean(axis=0)
        score = float(np.sum((m1 - m2) ** 2))
        scored.append((score, attempt, result))
    if not scored:
        raise ValueError("No valid stress-test partition found under the declared constraints")
    best_score, best_attempt, result = max(scored, key=lambda item: (item[0], -item[1]))
    scores = np.array([item[0] for item in scored])
    result.metadata.update({"method": "development_only_rff_mmd_stress",
                            "candidate_count": candidates,
                            "valid_candidate_count": len(scored),
                            "rff_components": components,
                            "approx_mmd_squared": best_score,
                            "random_feasible_mean_mmd_squared": float(scores.mean()),
                            "selected_attempt": best_attempt + 1,
                            "seed": spec.seed, "features": features,
                            "dataset_sha256": dataset_fingerprint(df),
                            "protocol": asdict(spec),
                            "warning": "Diagnostic stress partition selected using unlabeled feature shift. Do not treat as unbiased final-test performance."})
    return result


def evaluate_classifier(df: pd.DataFrame, spec: ProtocolSpec,
                        split: SplitResult, estimator: Any) -> dict[str, float]:
    """Evaluate a scikit-learn estimator; preprocessing should live in its Pipeline.

    The provided estimator is cloned and fitted ONLY on training data.
    Model selection must be done by the caller on development data.
    """
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    if not audit_split(df, spec, split)["valid"]:
        raise ValueError("Refusing to evaluate an invalid split")
    features = _selected_features(df, spec)
    if not features:
        raise ValueError("No features supplied or inferred")
    Xtr = df.iloc[split.train_idx][features]
    Xte = df.iloc[split.test_idx][features]
    ytr = df.iloc[split.train_idx][spec.target]
    yte = df.iloc[split.test_idx][spec.target]
    model = clone(estimator)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    return {"accuracy": float(accuracy_score(yte, yp)),
            "macro_f1": float(f1_score(yte, yp, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, yp))}
